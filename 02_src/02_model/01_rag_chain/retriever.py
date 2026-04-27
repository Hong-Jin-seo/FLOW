"""
================================================================================
01_retriever.py - 검색 방식 구현 (4가지: BM25, Dense, RRF, HyDE)
================================================================================

[목적]
사용자 질문에 최적의 문서를 검색하는 4가지 방식 구현.
평가 결과에 따라 질문 유형별 최적 검색 방식:
- ingredient (성분 안전성): Dense (NDCG 1.444)
- recommend (피부 고민): BM25 (NDCG 1.822)
- general (일반 정보): BM25 (NDCG 0.926)

[수업 자료 참고]
- 07_advanced_rag/01_retrieval_optimization/02_bm25_dense_comparison.ipynb
- 07_advanced_rag/01_retrieval_optimization/03_rrf.ipynb
- 07_advanced_rag/01_retrieval_optimization/04_hyde.ipynb
- 07_advanced_rag/01_retrieval_optimization/05_cohere_rerank.ipynb

================================================================================
"""

import os
import cohere
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda


def build_retriever(vs: FAISS, search_type: str = "hyde", k: int = 20):
    """
    [함수 설명]
    4가지 검색 방식 중 하나를 선택해서 retriever 생성

    [매개변수]
    - vs: FAISS 벡터 스토어
    - search_type: "bm25" | "dense" | "rrf" | "hyde" 중 선택
    - k: 검색할 문서 수 (기본 20개 → rerank 후 5개로 줄임)

    [반환값]
    선택한 방식의 retriever 객체
    """
    if search_type == "dense":
        return _dense_retriever(vs, k)
    elif search_type == "bm25":
        return _bm25_retriever(vs, k)
    elif search_type == "rrf":
        return _rrf_retriever(vs, k)
    elif search_type == "hyde":
        return _hyde_retriever(vs, k)
    else:
        raise ValueError(f"지원하지 않는 search_type: {search_type}")


# ============================================================================
# [1단계] Dense (의미 기반 검색)
# ============================================================================
def _dense_retriever(vs: FAISS, k: int):
    """
    [검색 방식 설명]
    - 질문과 문서를 "의미"로 변환해서 유사도 계산
    - 의미 추론이 필요한 질문에 강함
    - 예: "보습 효과 좋은 성분?" → "보습", "수분", "보혈" 등 의미 검색

    [평가 결과]
    - ingredient: 1.444 (최고 성능) ✅

    [특징]
    - 속도: 중간
    - 정확도: 중간~높음 (의미 기반)
    """
    return vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


# ============================================================================
# [2단계] BM25 (키워드 정확 검색)
# ============================================================================
def _bm25_retriever(vs: FAISS, k: int):
    """
    [검색 방식 설명]
    - Ctrl+F처럼 "정확한 단어"를 문서에서 찾음
    - 키워드 매칭에 강함
    - 예: "나이아신아마이드" → 문서에서 "나이아신아마이드" 포함된 것만 검색

    [평가 결과]
    - recommend: 1.822 (최고 성능) ✅
    - general: 0.926 (최고 성능) ✅

    [특징]
    - 속도: 매우 빠름
    - 정확도: 높음 (키워드 정확도)
    """
    all_docs = list(vs.docstore._dict.values())
    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = k
    return bm25


# ============================================================================
# [3단계] RRF (BM25 + Dense 결합)
# ============================================================================
def _rrf_retriever(vs: FAISS, k: int):
    """
    [검색 방식 설명]
    RRF = Reciprocal Rank Fusion
    - BM25 검색 결과 + Dense 검색 결과를 합치기
    - 각 방식의 상위 문서들을 점수 매겨서 통합

    [평가 결과]
    - ingredient: 1.356 (2등)
    - recommend: 1.333 (3등)
    - general: 0.519 (3등)

    [특징]
    - 속도: 느림 (2가지 방식 모두 실행)
    - 정확도: 가장 높음 (정확성 + 의미성)
    """
    dense = _dense_retriever(vs, k)
    bm25  = _bm25_retriever(vs, k)

    def rrf_search(query: str):
        # 1단계: Dense와 BM25로 각각 검색
        dense_docs = dense.invoke(query)
        bm25_docs  = bm25.invoke(query)

        rrf_k = 60
        scores = {}

        # 2단계: Dense 검색 결과에 점수 부여
        for rank, doc in enumerate(dense_docs):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + 1 / (rrf_k + rank + 1)
            scores[key + "__doc__"] = doc

        # 3단계: BM25 검색 결과에 점수 부여 (누적)
        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + 1 / (rrf_k + rank + 1)
            scores[key + "__doc__"] = doc

        # 4단계: 점수 높은 순서로 정렬
        sorted_keys = sorted(
            [k for k in scores if not k.endswith("__doc__")],
            key=lambda x: scores[x],
            reverse=True
        )
        return [scores[key + "__doc__"] for key in sorted_keys[:k]]

    return RunnableLambda(rrf_search)


# ============================================================================
# [4단계] HyDE (가상 답변 기반 검색)
# ============================================================================
def _hyde_retriever(vs: FAISS, k: int):
    """
    [검색 방식 설명]
    HyDE = Hypothetical Document Embeddings
    - GPT가 "가짜 답변"을 먼저 생성
    - 그 가짜 답변으로 Dense 검색 실행

    [예시]
    질문: "나이아신아마이드 EWG 등급?"
    ↓
    GPT 생성: "나이아신아마이드는 EWG 1등급으로 매우 안전합니다"
    ↓
    그 문장으로 의미 기반 검색

    [평가 결과]
    - ingredient: 0.700 (4등 - 가장 낮음)
    - recommend: 1.067 (4등)
    - general: 0.185 (4등 - 가장 낮음)

    [특징]
    - 속도: 느림 (GPT 호출 추가)
    - 정확도: 중간 (가상 답변 품질에 의존)
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    hyde_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 화장품 성분 전문가입니다. 아래 질문에 대해 성분 정보 문서처럼 짧게 답변하세요."),
        ("human", "{question}")
    ])

    hyde_chain = hyde_prompt | llm | StrOutputParser()
    dense = _dense_retriever(vs, k)

    def hyde_search(query: str):
        # 1단계: GPT로 가상 답변 생성
        hypothetical_doc = hyde_chain.invoke({"question": query})
        # 2단계: 가상 답변으로 Dense 검색
        return dense.invoke(hypothetical_doc)

    return RunnableLambda(hyde_search)


# ============================================================================
# [5단계] Cohere Rerank - 문서 재정렬
# ============================================================================
def rerank_docs(query: str, docs: list, top_k: int = 5) -> list:
    """
    [함수 설명]
    BM25/Dense로 뽑은 후보 문서들을 Cohere AI로 "질문과의 실제 관련성"을 재평가해서 재정렬

    [효과]
    - 검색 점수 vs 실제 적합도 괴리 해결
    - 예: "나이아신아마이드" 질문에 "아스코빌팔미테이트" 같은 엉뚱한 문서 제거

    [처리 흐름]
    후보 20개 검색 → Cohere Rerank → 상위 5개 선택

    [Cohere Rerank 모델]
    - rerank-multilingual-v3.0: 한국어 포함 다국어 지원
    - 점수 범위: 0~1 (높을수록 관련성 높음)

    [수업 자료]
    07_advanced_rag/01_retrieval_optimization/05_cohere_rerank.ipynb
    """
    if not docs:
        return docs

    co = cohere.Client(os.environ.get("COHERE_API_KEY"))
    texts = [doc.page_content for doc in docs]

    response = co.rerank(
        model="rerank-multilingual-v3.0",
        query=query,
        documents=texts
    )

    reranked = sorted(
        response.results,
        key=lambda x: x.relevance_score,
        reverse=True
    )
    return [docs[r.index] for r in reranked[:top_k]]