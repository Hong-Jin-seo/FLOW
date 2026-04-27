"""
================================================================================
02_chain.py - RAG 체인 구성 (CoT + Structured Output + Prompt Compression)
================================================================================

[목적]
검색된 문서를 바탕으로 성분 안전성을 분석하는 LLM 체인 구성.
5가지 고급 RAG 기술 적용:
1. CoT (Chain of Thought): 단계적 추론
2. Structured Output: 정형화된 응답 (ewg_grade, safety_label 등)
3. Prompt Compression: 300자 압축
4. GPT 기반 성분명 추출: sources 필터링
5. Cohere Rerank: 문서 재정렬

[수업 자료 참고]
- CoT: 07_advanced_rag/02_generation_optimization/01_rag_cot.ipynb
- Struct: 07_advanced_rag/02_generation_optimization/04_rag_structured_output.ipynb
- Compress: 07_advanced_rag/02_generation_optimization/03_rag_prompt_compression.ipynb
- Rerank: 07_advanced_rag/01_retrieval_optimization/05_cohere_rerank.ipynb

================================================================================
"""

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from retriever import build_retriever, rerank_docs

load_dotenv()

# ── 경로 설정 ──────────────────────────────────────────────
BASE_DIR = os.environ.get(
    "FAISS_BASE_DIR",
    os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..",
        "00_data", "02_processed"
    ))
)

def get_faiss_path(preset_id: int = 2) -> str:
    """
    [함수 설명]
    FAISS 프리셋 경로 반환 (preset1~4)

    [preset 종류]
    - preset1: 균등 (0.33/0.33/0.33) → general
    - preset2: EWG 중심 (0.50/0.35/0.15) → ingredient
    - preset3: 기본정보 중심 (0.40/0.45/0.15) → (미사용)
    - preset4: 균형 (0.45/0.45/0.10) → recommend
    """
    return os.path.join(BASE_DIR, f"faiss_index_preset{preset_id}")


# ============================================================================
# [Structured Output 스키마 1] 성분명 추출
# ============================================================================
class QueryIntent(BaseModel):
    """
    [목적]
    사용자 질문에서 화장품 성분명만 추출.
    성분명으로 sources 필터링할 때 사용.

    [예시]
    질문: "나이아신아마이드와 레티놀 중 뭐가 더 안전해?"
    추출: ["나이아신아마이드", "레티놀"]
    """
    ingredient_names: list[str] = Field(
        description="질문에서 추출한 화장품 성분명 목록. 성분명이 없으면 빈 리스트."
    )


# ============================================================================
# [Structured Output 스키마 2] 답변 구조화
# ============================================================================
class IngredientAnalysis(BaseModel):
    """
    [목적]
    성분 안전성 분석 결과를 정형화된 형식으로 반환.

    [필드]
    - ewg_grade: EWG 안전 등급 (1~10)
    - safety_label: 최종 안전성 판단 (안전/주의/위험/등급없음)
    - sources: 참고한 출처들 (coos, 화해, Paula's Choice 등)
    - skin_types: 적합한 피부 타입 (건성/지성/민감성 등)
    - summary: 2~3문장 요약
    """
    ewg_grade:    int       = Field(description="EWG 안전 등급 (1~10, 숫자만, 모르면 0)")
    safety_label: str       = Field(description="안전/주의/위험 중 하나")
    sources:      list[str] = Field(description="출처 목록 (coos, 화해, Paula's Choice 등)")
    skin_types:   list[str] = Field(description="적합한 피부 타입 목록, 없으면 빈 리스트")
    summary:      str       = Field(description="성분 안전성 요약 2~3문장")


# ============================================================================
# [1단계] FAISS 로드
# ============================================================================
def load_vectorstore(faiss_path: str):
    """
    [함수 설명]
    FAISS 벡터 스토어를 메모리에 로드.

    [프리셋별 로드 경로]
    - ingredient: faiss_index_preset2 (EWG 비중 0.50)
    - recommend: faiss_index_preset4 (균형 0.45/0.45)
    - general: faiss_index_preset1 (균등 0.33)
    """
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local(
        faiss_path, embedding, allow_dangerous_deserialization=True
    )


# ============================================================================
# [2단계] 성분명 추출 (GPT)
# ============================================================================
def extract_ingredients(query: str) -> list[str]:
    """
    [함수 설명]
    GPT로 질문에서 성분명을 자동 추출.
    sources 필터링에 사용.

    [예시]
    질문: "나이아신아마이드 안전해?"
    추출: ["나이아신아마이드"]
    ↓
    후보 20개 중 "나이아신아마이드" 포함된 것만 필터

    [효과]
    - 엉뚱한 성분 섞임 제거
    - 정확도 향상
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(QueryIntent)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "사용자 질문에서 화장품 성분명만 추출하세요. 성분명이 없으면 빈 리스트를 반환하세요."),
        ("human", "{query}")
    ])
    result = (prompt | llm).invoke({"query": query})
    return result.ingredient_names


# ============================================================================
# [3단계] CoT 시스템 프롬프트
# ============================================================================
SYSTEM_PROMPT = """
당신은 화장품 성분 안전 전문가입니다.
아래 [검색된 성분 정보]를 바탕으로 [질문]에 단계적으로 추론하며 답변하세요.

[답변 절차]
1. **성분 기본 정보 확인**: 검색된 문서에서 성분명과 EWG 안전 등급(1~10)을 찾아 명시하세요.
2. **데이터 품질 평가**: 각 출처(coos, 화해, Paula's Choice)별로 데이터 등급과 신뢰도를 확인하세요.
3. **최종 안전성 판단**: 출처-데이터를 종합해 안전/주의/위험 중 하나로 판단하세요.
4. **피부 타입 적합성**: 피부 타입(건성/지성/민감성 등)별 적합 여부를 알려주세요.

[주의]
- 여러 성분이 언급된 경우 각 성분별로 개별 EWG 등급을 제시하세요.
- 성분별로 등급이 다를 수 있으므로 하나의 등급으로 뭉뚱그리지 마세요.
- EWG 등급 데이터가 없는 성분은 ewg_grade를 0으로, safety_label을 '등급없음'으로 설정하세요.

[검색된 성분 정보]
{context}
"""


# ============================================================================
# [4단계] Prompt Compression
# ============================================================================
def compress_docs(docs: list, query: str) -> str:
    """
    [함수 설명]
    검색된 5개 문서를 300자 이내로 압축.
    컨텍스트 길이 초과 방지 + 비용 절감.

    [효과]
    - 20,000자 → 300자 (98% 압축)
    - 성능 손실 최소화 (핵심 정보 유지)
    - GPT 토큰 비용 감소

    [수업 자료]
    07_advanced_rag/02_generation_optimization/03_rag_prompt_compression.ipynb
    """
    llm_compress = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate.from_template("""
아래 문서를 300자 이내로 압축하세요. 성분명, EWG 등급, 출처, 효능만 남기세요.

{docs}
""")
    summarizer = prompt | llm_compress | StrOutputParser()
    context = "\n\n".join(
        f"[{doc.metadata.get('source', '?')}]\n{doc.page_content}"
        for doc in docs
    )
    return summarizer.invoke({"docs": context})


# ============================================================================
# [5단계] RAG 체인 구성
# ============================================================================
def build_chain(search_type: str = "hyde", faiss_path: str = None):
    """
    [함수 설명]
    CoT + Structured Output + Compression을 포함한 RAG 체인 구성.

    [처리 흐름]
    1. retriever로 후보 20개 검색
    2. Compression으로 300자 압축
    3. CoT 프롬프트로 단계적 추론
    4. Structured Output으로 정형화

    [매개변수]
    - search_type: "bm25" | "dense" | "rrf" | "hyde"
    - faiss_path: FAISS 프리셋 경로
    """
    vs = load_vectorstore(faiss_path or get_faiss_path(2))
    retriever = build_retriever(vs, search_type=search_type, k=20)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(IngredientAnalysis)

    chain = (
        {
            "context":  RunnableLambda(lambda q: compress_docs(retriever.invoke(q), q)),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return chain, retriever


# ============================================================================
# [메인 함수] get_answer
# ============================================================================
def get_answer(
    query: str,
    search_type: str = "hyde",
    history: list = None,
    preset_id: int = 2
) -> dict:
    """
    [함수 설명]
    사용자 질문에 대한 성분 안전성 분석 답변 생성.

    [처리 흐름]
    1단계: 히스토리 기반 컨텍스트 쿼리 생성
    2단계: 후보 20개 검색
    3단계: GPT로 성분명 추출
    4단계: 성분명으로 필터링
    5단계: Cohere Rerank (20개 → 5개)
    6단계: GPT 답변 생성
    7단계: 출처 정보 추출

    [매개변수]
    - query: 사용자 질문
    - search_type: "bm25" | "dense" | "rrf" | "hyde"
    - history: 대화 히스토리 (최근 4개)
    - preset_id: FAISS 프리셋 (1~4)

    [반환값]
    {
        "answer": "EWG 등급: 1 (안전)...",
        "sources": [{"ingredient": "나이아신아마이드", ...}, ...],
        "ewg_grade": 1,
        "safety_label": "안전"
    }
    """
    history = history or []
    faiss_path = get_faiss_path(preset_id)
    chain, retriever = build_chain(search_type=search_type, faiss_path=faiss_path)

    # [1단계] 히스토리 포함 컨텍스트 생성
    context_query = query
    if history:
        recent = history[-4:]
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
        context_query = f"[이전 대화]\n{history_text}\n\n[현재 질문]\n{query}"

    # [2단계] 후보 20개 검색
    raw_docs_candidates = retriever.invoke(context_query)

    # [3단계] GPT로 성분명 추출
    ingredient_names = extract_ingredients(query)

    # [4,5단계] 성분명 필터링 후 Rerank
    if ingredient_names:
        filtered_docs = [
            doc for doc in raw_docs_candidates
            if any(name in doc.page_content for name in ingredient_names)
        ]
        final_docs = rerank_docs(context_query, filtered_docs, top_k=3) if filtered_docs else rerank_docs(context_query, raw_docs_candidates, top_k=3)
    else:
        final_docs = rerank_docs(context_query, raw_docs_candidates, top_k=3)

    # [6단계] GPT 답변 생성
    analysis: IngredientAnalysis = chain.invoke(context_query)

    # [7단계] 출처 정보 추출
    answer = f"""**EWG 등급**: {analysis.ewg_grade} ({analysis.safety_label})
**출처**: {', '.join(analysis.sources) if analysis.sources else '정보 없음'}
**적합 피부 타입**: {', '.join(analysis.skin_types) if analysis.skin_types else '정보 없음'}

{analysis.summary}"""

    sources = [
        {
            "ingredient": doc.metadata.get("ingredient", ""),
            "ewg_score":  doc.metadata.get("ewg_score", "?"),
            "source":     doc.metadata.get("source", "?"),
            "chunk_type": doc.metadata.get("chunk_type", "?"),
            "content":    doc.page_content[:200]
        }
        for doc in final_docs
    ]

    return {
        "answer":       answer,
        "sources":      sources,
        "ewg_grade":    analysis.ewg_grade,
        "safety_label": analysis.safety_label
    }