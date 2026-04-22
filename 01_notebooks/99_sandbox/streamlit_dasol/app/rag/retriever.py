"""
FAISS 벡터 검색

- 최초 호출 시 인덱스 + 청크를 메모리에 로드 (싱글턴)
- 질문을 임베딩 → top-k 유사 청크 반환
"""

import pickle
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR    = Path(__file__).resolve().parent.parent.parent
INDEX_PATH  = BASE_DIR / "vectorstore" / "index.faiss"
CHUNKS_PATH = BASE_DIR / "vectorstore" / "chunks.pkl"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def _load_resources():
    """인덱스, 청크, 임베딩 모델을 한 번만 로드."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            "벡터 인덱스가 없습니다. 먼저 indexer.py를 실행하세요.\n"
            "  python -m app.rag.indexer"
        )
    index  = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    model  = SentenceTransformer(EMBED_MODEL)
    return index, chunks, model


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    질문과 가장 유사한 청크 top_k개 반환.
    각 청크: {"type", "product_name", "text", "score"}
    """
    index, chunks, model = _load_resources()

    emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = chunks[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)

    return results
