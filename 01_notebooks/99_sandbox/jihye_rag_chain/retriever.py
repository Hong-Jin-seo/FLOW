"""
jihye_rag_chain/retriever.py
수업 자료 참고: 07_advanced_rag/01_retrieval_optimization/
"""

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda


def build_retriever(vs: FAISS, search_type: str = "dense", k: int = 5):
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


def _dense_retriever(vs: FAISS, k: int):
    return vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


def _bm25_retriever(vs: FAISS, k: int):
    all_docs = list(vs.docstore._dict.values())
    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = k
    return bm25


def _rrf_retriever(vs: FAISS, k: int):
    """
    EnsembleRetriever 없이 RRF 직접 구현
    BM25 + Dense 각각 검색 후 순위 점수 합산
    """
    dense = _dense_retriever(vs, k)
    bm25  = _bm25_retriever(vs, k)

    def rrf_search(query: str):
        dense_docs = dense.invoke(query)
        bm25_docs  = bm25.invoke(query)

        # RRF 점수 계산 (k=60 기본값)
        rrf_k = 60
        scores = {}

        for rank, doc in enumerate(dense_docs):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + 1 / (rrf_k + rank + 1)
            scores[key + "__doc__"] = doc

        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + 1 / (rrf_k + rank + 1)
            scores[key + "__doc__"] = doc

        # 점수 높은 순으로 정렬
        sorted_keys = sorted(
            [k for k in scores if not k.endswith("__doc__")],
            key=lambda x: scores[x],
            reverse=True
        )

        return [scores[key + "__doc__"] for key in sorted_keys[:k]]

    return RunnableLambda(rrf_search)


def _hyde_retriever(vs: FAISS, k: int):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    hyde_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 화장품 성분 전문가입니다. 아래 질문에 대해 성분 정보 문서처럼 짧게 답변하세요."),
        ("human", "{question}")
    ])

    hyde_chain = hyde_prompt | llm | StrOutputParser()
    dense = _dense_retriever(vs, k)

    def hyde_search(query: str):
        hypothetical_doc = hyde_chain.invoke({"question": query})
        return dense.invoke(hypothetical_doc)

    return RunnableLambda(hyde_search)