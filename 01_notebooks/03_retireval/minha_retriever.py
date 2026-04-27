"""
stage5_retriever.py
===================
변경 이력
---------
- 임베딩 모델 : jhgan/ko-sroberta-multitask (768d)
               → text-embedding-3-small (1536d)
- Dense 점수  : 1/(1+L2) 근사 → similarity_search_with_relevance_scores() 정확한 cosine
- HyDE        : BM25 전용 → BM25 + Dense RRF 병합으로 개선
- load_faiss  : 768d 분기 제거, 1536d 기본 / 3072d 선택
"""

from __future__ import annotations

import os
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from rank_bm25 import BM25Okapi

from dotenv import load_dotenv
load_dotenv()


# ──────────────────────────────────────────────
# 임베딩 모델 설정
# ──────────────────────────────────────────────

def get_embeddings(model: str = "text-embedding-3-small") -> OpenAIEmbeddings:
    """
    OpenAI 임베딩 모델 반환.

    model 선택지:
      - "text-embedding-3-small" : 1536d, 빠름, 저렴 ($0.02/1M tokens)  ← 기본값으로 채택
      - "text-embedding-3-large" : 3072d, 최고 품질, 5배 비쌈
    """
    return OpenAIEmbeddings(
        model=model,
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )


# ──────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────

@dataclass
class SearchResult:
    rank: int
    score: float
    document: Document
    method: str


@dataclass
class SearchResponse:
    method: str
    query: str
    results: List[SearchResult]
    latency_ms: float
    extra: Dict = field(default_factory=dict)


# ──────────────────────────────────────────────
# 5-1. BM25 인덱스 (변경 없음)
# ──────────────────────────────────────────────

class BM25Index:
    """rank_bm25 기반 키워드 검색 + max-min 정규화"""

    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.corpus = self._build_corpus(documents)
        self.bm25 = BM25Okapi(self.corpus)

    def _build_corpus(self, documents: List[Document]) -> List[List[str]]:
        tokenized = []
        for doc in documents:
            text = " ".join([
                str(doc.metadata.get("ingredient_ko", "")),
                str(doc.metadata.get("ingredient_en", "")),
                doc.page_content,
            ])
            tokenized.append(text.split())
        return tokenized

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, Document]]:
        """Returns: List of (rank, normalized_score 0~1, document)"""
        tokens = query.split()
        raw_scores = self.bm25.get_scores(tokens)

        max_s, min_s = max(raw_scores), min(raw_scores)
        if max_s - min_s > 0:
            norm_scores = [(s - min_s) / (max_s - min_s) for s in raw_scores]
        else:
            norm_scores = [0.0] * len(raw_scores)

        ranked = sorted(enumerate(norm_scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            (rank + 1, norm_scores[idx], self.documents[idx])
            for rank, (idx, _) in enumerate(ranked)
        ]


# ──────────────────────────────────────────────
# 5-1 ~ 5-4. 통합 Retriever
# ──────────────────────────────────────────────

class CosmeticRetriever:
    """
    BM25 / Dense / RRF / HyDE 4가지 검색 통합 모듈

    Parameters
    ----------
    faiss_index : FAISS
        text-embedding-3-small 1536d 기반 FAISS VectorStore
    documents : List[Document]
        전체 청크 (BM25 인덱스 구성용)
    top_k : int
        검색 결과 수
    rrf_k : int
        RRF 상수 (기본 60)
    llm : ChatOpenAI
        HyDE 성분명 생성용 LLM
    """

    def __init__(
        self,
        faiss_index: FAISS,
        documents: List[Document],
        top_k: int = 5,
        rrf_k: int = 60,
        llm: Optional[ChatOpenAI] = None,
    ):
        self.faiss_index = faiss_index
        self.documents = documents
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.bm25_index = BM25Index(documents)
        self.llm = llm or ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0,
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        )

    # ── 5-1. BM25 ─────────────────────────────

    def search_bm25(self, query: str) -> SearchResponse:
        t0 = time.time()
        raw = self.bm25_index.search(query, top_k=self.top_k)
        latency = (time.time() - t0) * 1000

        results = [
            SearchResult(rank=rank, score=score, document=doc, method="bm25")
            for rank, score, doc in raw
        ]
        return SearchResponse(method="bm25", query=query, results=results, latency_ms=latency)

    # ── 5-2. Dense ────────────────────────────
    # [변경] 1/(1+L2) 근사 → similarity_search_with_relevance_scores() 정확한 cosine 유사도

    def search_dense(self, query: str) -> SearchResponse:
        """
        cosine 유사도 직접 반환.
        FAISS가 내부적으로 정규화 후 내적(= cosine)을 계산하므로 별도 변환 불필요.
        """
        t0 = time.time()

        # relevance_score_fn이 cosine 유사도(0~1)를 그대로 반환
        raw = self.faiss_index.similarity_search_with_relevance_scores(
            query, k=self.top_k
        )
        latency = (time.time() - t0) * 1000

        results = [
            SearchResult(
                rank=rank + 1,
                score=float(score),   # 이미 cosine 유사도 (0~1)
                document=doc,
                method="dense",
            )
            for rank, (doc, score) in enumerate(raw)
        ]
        return SearchResponse(method="dense", query=query, results=results, latency_ms=latency)

    # ── 5-3. RRF ──────────────────────────────

    def search_rrf(self, query: str) -> SearchResponse:
        """BM25 + Dense 순위 기반 결합 score = Σ 1 / (k + rank)"""
        t0 = time.time()

        bm25_raw  = self.bm25_index.search(query, top_k=self.top_k * 2)
        dense_raw = self.faiss_index.similarity_search_with_relevance_scores(
            query, k=self.top_k * 2
        )

        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, score, doc in bm25_raw:
            key = self._doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
            doc_map[key] = doc

        for rank, (doc, _) in enumerate(dense_raw, start=1):
            key = self._doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
            doc_map[key] = doc

        top = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[: self.top_k]
        results = [
            SearchResult(rank=rank + 1, score=score, document=doc_map[key], method="rrf")
            for rank, (key, score) in enumerate(top)
        ]
        latency = (time.time() - t0) * 1000
        return SearchResponse(method="rrf", query=query, results=results, latency_ms=latency)

    # ── 5-4. HyDE ─────────────────────────────
    # [변경] BM25 전용 → BM25 + Dense 모두 활용 후 RRF 병합

    def search_hyde(self, query: str) -> SearchResponse:
        """
        HyDE 개선판:
          1) GPT로 관련 성분명 5개 생성
          2) 성분명별 BM25 검색  ← 기존
          3) 성분명별 Dense 검색 ← 신규 추가 (text-embedding-3-small 활용)
          4) BM25 + Dense 결과를 RRF로 통합
        """
        t0 = time.time()

        # 1) GPT 성분명 생성
        prompt = (
            "화장품 성분 전문가로서 아래 질문과 관련된 성분명을 한국어로 5개 나열하세요.\n"
            "반드시 성분명만 쉼표로 구분해서 출력하세요. 설명 금지.\n"
            "예시: 나이아신아마이드, 레티놀, 히알루론산, 세라마이드, 판테놀\n\n"
            f"질문: {query}\n\n성분명:"
        )
        generated = self.llm.invoke(prompt).content.strip()
        candidate_names = [
            n.strip() for n in generated.replace("、", ",").split(",") if n.strip()
        ]
        print(f"\n  [HyDE 생성 성분명] {candidate_names}")

        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for sub_query in candidate_names[:5]:

            # 2) BM25 검색
            bm25_results = self.bm25_index.search(sub_query, top_k=10)
            bm25_results = [
                (rank, score, doc)
                for rank, score, doc in bm25_results
                if score > 0  # 0점 제외
            ]
            for rank, score, doc in bm25_results:
                key = self._doc_key(doc)
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
                doc_map[key] = doc

            # 3) Dense 검색 (text-embedding-3-small 의미론적 검색)
            dense_results = self.faiss_index.similarity_search_with_relevance_scores(
                sub_query, k=10
            )
            for rank, (doc, score) in enumerate(dense_results, start=1):
                if score < 0.3:   # 낮은 cosine 유사도 제외
                    continue
                key = self._doc_key(doc)
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
                doc_map[key] = doc

        top = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[: self.top_k]

        print("  [HyDE 검색결과]")
        results = []
        for rank, (key, score) in enumerate(top):
            doc = doc_map[key]
            print(f"    {rank + 1}. {doc.metadata.get('ingredient_ko')} (RRF={score:.4f})")
            results.append(
                SearchResult(rank=rank + 1, score=score, document=doc, method="hyde")
            )

        return SearchResponse(
            method="hyde",
            query=query,
            results=results,
            latency_ms=(time.time() - t0) * 1000,
            extra={"generated_names": candidate_names},
        )

    # ── 전체 병렬 실행 ─────────────────────────

    def search_all(self, query: str) -> Dict[str, SearchResponse]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        method_map = {
            "bm25":  self.search_bm25,
            "dense": self.search_dense,
            "rrf":   self.search_rrf,
            "hyde":  self.search_hyde,
        }
        responses = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(fn, query): name for name, fn in method_map.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    responses[name] = future.result()
                except Exception as e:
                    print(f"[WARNING] {name} 실패: {e}")
        return responses

    # ── 유틸 ──────────────────────────────────

    @staticmethod
    def _doc_key(doc: Document) -> str:
        return doc.metadata.get(
            "doc_id",
            str(hash(doc.metadata.get("ingredient_ko", "") + doc.page_content[:50]))
        )


# ──────────────────────────────────────────────
# FAISS 로더
# [변경] 768d(ko-sroberta) 분기 제거 → 1536d 기본 / 3072d 선택
# ──────────────────────────────────────────────

def load_faiss_auto(faiss_path: str) -> Tuple[FAISS, List[Document]]:
    """
    FAISS 인덱스 차원을 확인 후 적절한 OpenAI 임베딩 모델로 로드.

    지원 차원:
      1536d → text-embedding-3-small  (기본, 권장)
      3072d → text-embedding-3-large  (최고 품질)
      기타  → text-embedding-3-small 로 시도 후 오류 시 확인 요청

    ※ 기존 768d(ko-sroberta) 인덱스는 아래 rebuild_faiss_index()로 재빌드 필요.
    """
    # 차원 확인용 임시 로드
    _probe_emb = get_embeddings("text-embedding-3-small")
    _probe_idx = FAISS.load_local(
        faiss_path, _probe_emb, allow_dangerous_deserialization=True
    )
    dim = _probe_idx.index.d
    print(f"[INFO] FAISS 차원: {dim}d")

    if dim == 3072:
        embeddings  = get_embeddings("text-embedding-3-large")
        faiss_index = FAISS.load_local(
            faiss_path, embeddings, allow_dangerous_deserialization=True
        )
    elif dim == 1536:
        faiss_index = _probe_idx   # 이미 올바른 모델로 로드됨
    else:
        raise ValueError(
            f"[ERROR] 지원하지 않는 FAISS 차원: {dim}d\n"
            "기존 768d 인덱스라면 rebuild_faiss_index()로 재빌드하세요."
        )

    documents = list(faiss_index.docstore._dict.values())
    print(f"[INFO] Document {len(documents)}개 로드 완료")
    return faiss_index, documents


# ──────────────────────────────────────────────
# FAISS 인덱스 재빌드 (768d → 1536d 마이그레이션)
# ──────────────────────────────────────────────

def rebuild_faiss_index(
    documents: List[Document],
    save_path: str,
    embedding_model: str = "text-embedding-3-small",
) -> FAISS:
    """
    기존 ko-sroberta 768d 인덱스를 text-embedding-3-small 1536d로 재빌드.

    Args:
        documents:       기존 FAISS에서 꺼낸 Document 리스트
        save_path:       새 인덱스 저장 경로
        embedding_model: 사용할 OpenAI 임베딩 모델

    Returns:
        새로 빌드된 FAISS 인덱스

    사용 예시:
        # 기존 768d 인덱스에서 문서만 추출
        old_emb = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask", ...)
        old_idx = FAISS.load_local(old_path, old_emb, allow_dangerous_deserialization=True)
        docs = list(old_idx.docstore._dict.values())

        # 새 1536d 인덱스로 재빌드
        new_idx = rebuild_faiss_index(docs, save_path="faiss_index_1536d")
    """
    print(f"[INFO] {len(documents)}개 문서를 {embedding_model}({_dim_of(embedding_model)}d)로 재임베딩 중…")
    embeddings  = get_embeddings(embedding_model)
    faiss_index = FAISS.from_documents(documents, embeddings)
    faiss_index.save_local(save_path)
    print(f"[INFO] 새 FAISS 인덱스 저장 완료: {save_path}  ({_dim_of(embedding_model)}d)")
    return faiss_index


def _dim_of(model: str) -> int:
    return {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}.get(model, -1)


# ──────────────────────────────────────────────
# CLI 실행
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--query",   default="나이아신아마이드 안전한가요?")
    parser.add_argument("--method",  default="all",
                        choices=["bm25", "dense", "rrf", "hyde", "all"])
    parser.add_argument("--top_k",   type=int, default=5)
    parser.add_argument("--rebuild", action="store_true",
                        help="기존 768d 인덱스를 1536d로 재빌드 후 실행")
    parser.add_argument("--old_faiss", default=None,
                        help="--rebuild 시 기존 768d 인덱스 경로")
    args, _ = parser.parse_known_args()

    # 필요 시 재빌드
    if args.rebuild:
        if not args.old_faiss:
            raise ValueError("--rebuild 사용 시 --old_faiss 경로를 지정하세요.")

        from langchain_huggingface import HuggingFaceEmbeddings
        old_emb = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={"device": "cpu"},
        )
        old_idx = FAISS.load_local(
            args.old_faiss, old_emb, allow_dangerous_deserialization=True
        )
        old_docs = list(old_idx.docstore._dict.values())
        rebuild_faiss_index(old_docs, save_path=args.faiss)

    targets = ["bm25", "dense", "rrf", "hyde"] if args.method == "all" else [args.method]

    for preset_num in range(1, 5):  # preset 1 ~ 4 반복
        faiss_path = f"../../00_data/02_processed/faiss_index_preset{preset_num}"

        print("\n" + "=" * 80)
        print(f"  📦 PRESET {preset_num}  |  {faiss_path}")
        print("=" * 80)

        faiss_index, documents = load_faiss_auto(faiss_path)
        retriever = CosmeticRetriever(
            faiss_index=faiss_index, documents=documents, top_k=args.top_k
        )

        fn_map = {
            "bm25":  retriever.search_bm25,
            "dense": retriever.search_dense,
            "rrf":   retriever.search_rrf,
            "hyde":  retriever.search_hyde,
        }

        for m in targets:
            resp = fn_map[m](args.query)
            print(f"\n▶ [{m.upper()}] {resp.latency_ms:.0f}ms")
            if resp.extra.get("generated_names"):
                print(f"  [생성 성분명] {resp.extra['generated_names']}")
            print(f"  {'순위':<4} {'성분명':<25} {'EWG':<6} {'COOS':<6} {'PC':<6} {'점수':>8}  청크유형")
            print("  " + "-" * 72)
            for r in resp.results:
                meta  = r.document.metadata
                name  = meta.get("ingredient_ko", "?")
                ewg   = str(meta.get("hw_ewg") or "-")
                coos  = str(meta.get("coos_score", "-"))
                pc    = str(meta.get("pc_rating", "-"))
                chunk = meta.get("chunk_type", "?")
                print(f"  {r.rank:<4} {name:<25} {ewg:<6} {coos:<6} {pc:<6} {r.score:>8.4f}  {chunk}")