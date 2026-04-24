"""
jihye_rag_chain/chain.py
수업 자료:
  - CoT      : 07_advanced_rag/02_generation_optimization/01_rag_cot.ipynb
  - Struct   : 07_advanced_rag/02_generation_optimization/04_rag_structured_output.ipynb
  - Compress : 07_advanced_rag/02_generation_optimization/03_rag_prompt_compression.ipynb
  - Rerank   : 07_advanced_rag/01_retrieval_optimization/05_cohere_rerank.ipynb
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
        "..", "..", "..", "..",
        "00_data", "02_processed"
    ))
)
def get_faiss_path(preset_id: int = 2) -> str:
    return os.path.join(BASE_DIR, f"faiss_index_preset{preset_id}")


# ── Structured Output 스키마 1: 성분명 추출 ────────────────
class QueryIntent(BaseModel):
    ingredient_names: list[str] = Field(
        description="질문에서 추출한 화장품 성분명 목록. 성분명이 없으면 빈 리스트."
    )


# ── Structured Output 스키마 2: 답변 구조화 ───────────────
class IngredientAnalysis(BaseModel):
    ewg_grade:    int       = Field(description="EWG 안전 등급 (1~10, 숫자만, 모르면 0)")
    safety_label: str       = Field(description="안전/주의/위험 중 하나")
    sources:      list[str] = Field(description="출처 목록 (coos, 화해, Paula's Choice 등)")
    skin_types:   list[str] = Field(description="적합한 피부 타입 목록, 없으면 빈 리스트")
    summary:      str       = Field(description="성분 안전성 요약 2~3문장")


# ── FAISS 로드 ─────────────────────────────────────────────
def load_vectorstore(faiss_path: str):
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local(
        faiss_path, embedding, allow_dangerous_deserialization=True
    )


# ── 성분명 추출 ────────────────────────────────────────────
def extract_ingredients(query: str) -> list[str]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(QueryIntent)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "사용자 질문에서 화장품 성분명만 추출하세요. 성분명이 없으면 빈 리스트를 반환하세요."),
        ("human", "{query}")
    ])
    result = (prompt | llm).invoke({"query": query})
    return result.ingredient_names


# ── CoT 시스템 프롬프트 ────────────────────────────────────
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


# ── Prompt Compression ─────────────────────────────────────
def compress_docs(docs: list, query: str) -> str:
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


# ── RAG 체인 구성 ──────────────────────────────────────────
def build_chain(search_type: str = "hyde", faiss_path: str = None):
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


# ── 메인 함수 ──────────────────────────────────────────────
def get_answer(
    query: str,
    search_type: str = "hyde",
    history: list = None,
    preset_id: int = 2
) -> dict:
    history = history or []
    faiss_path = get_faiss_path(preset_id)
    chain, retriever = build_chain(search_type=search_type, faiss_path=faiss_path)

    context_query = query
    if history:
        recent = history[-4:]
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
        context_query = f"[이전 대화]\n{history_text}\n\n[현재 질문]\n{query}"

    # 1단계: 후보 20개 검색
    raw_docs_candidates = retriever.invoke(context_query)

    # 2단계: GPT로 성분명 추출
    ingredient_names = extract_ingredients(query)

    # 3단계: 성분명으로 필터링
    if ingredient_names:
        filtered_docs = [
            doc for doc in raw_docs_candidates
            if any(name in doc.page_content for name in ingredient_names)
        ]
        final_docs = rerank_docs(context_query, filtered_docs, top_k=3) if filtered_docs else rerank_docs(context_query, raw_docs_candidates, top_k=3)
    else:
        final_docs = rerank_docs(context_query, raw_docs_candidates, top_k=3)

    # 4단계: GPT 답변 생성
    analysis: IngredientAnalysis = chain.invoke(context_query)

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