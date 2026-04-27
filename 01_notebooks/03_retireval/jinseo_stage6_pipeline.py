"""
6단계 전체 파이프라인 통합

실행 흐름:
  [minha_retriever.py SearchResponse]
       ↓ convert_to_stage6_input()
  [6-1/6-2] 이중 가중치 재정렬 (chunk_weight × domain_weight)
       ↓
  [6-3] Contextual Compression → 상위 3개 선별
       ↓
  [6-4] GPT 프롬프트 빌드 → 최종 답변
       ↓
  [Stage6Result]
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from jinseo_stage6_rerank import RankedChunk, rerank, print_rerank_table
from jinseo_stage6_compressor import (
    CompressedChunk,
    contextual_compress,
    build_prompt,
    call_final_gpt,
    print_compressed_table,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. 5단계 → 6단계 형식 변환
# ──────────────────────────────────────────────

def convert_to_stage6_input(response: Any) -> list[dict[str, Any]]:
    """
    minha_retriever.SearchResponse → stage6 입력 형식 변환.

    Parameters
    ----------
    response : SearchResponse (bm25 / dense / rrf / hyde 결과)

    Returns
    -------
    [{"content": str, "metadata": dict, "score": float}, ...]
    """
    return [
        {
            "content":  r.document.page_content,
            "metadata": r.document.metadata,
            "score":    r.score,
        }
        for r in response.results
    ]


# ──────────────────────────────────────────────
# 2. 파이프라인 설정
# ──────────────────────────────────────────────

@dataclass
class Stage6Config:
    """6단계 파이프라인 동작 설정."""

    # 재정렬
    rerank_top_k: int = 5
    deduplicate: bool = True
    similarity_threshold: float = 0.85
    custom_chunk_weights: dict[str, float] | None = None

    # 압축
    compress_top_k: int = 5
    final_top_k: int = 3
    compression_model: str = field(
        default_factory=lambda: os.getenv("COMPRESSION_MODEL", "gpt-4o-mini")
    )
    min_compressed_len: int = 30

    # 최종 답변
    final_model: str = field(
        default_factory=lambda: os.getenv("FINAL_ANSWER_MODEL", "gpt-4o")
    )
    temperature: float = 0.2
    max_tokens: int = 1024


# ──────────────────────────────────────────────
# 3. 결과 컨테이너
# ──────────────────────────────────────────────

@dataclass
class Stage6Result:
    """6단계 파이프라인 최종 결과."""
    query:             str
    search_method:     str                       # bm25 / dense / rrf / hyde
    reranked_chunks:   list[RankedChunk]
    compressed_chunks: list[CompressedChunk]
    final_prompt:      list[dict[str, str]]
    answer:            str
    elapsed_sec:       float

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"  6단계 파이프라인 결과  [{self.search_method.upper()}]",
            "=" * 60,
            f"  질문   : {self.query}",
            f"  재정렬 : {len(self.reranked_chunks)}개",
            f"  압축   : {len(self.compressed_chunks)}개",
            f"  소요   : {self.elapsed_sec:.2f}초",
            "-" * 60,
            "  [최종 답변]",
            self.answer,
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query":             self.query,
            "search_method":     self.search_method,
            "reranked_chunks":   [c.to_dict() for c in self.reranked_chunks],
            "compressed_chunks": [c.to_dict() for c in self.compressed_chunks],
            "final_prompt":      self.final_prompt,
            "answer":            self.answer,
            "elapsed_sec":       self.elapsed_sec,
        }


# ──────────────────────────────────────────────
# 4. 메인 파이프라인
# ──────────────────────────────────────────────

def run_stage6(
    query: str,
    search_results: list[dict[str, Any]],
    search_method: str = "rrf",
    client: OpenAI | None = None,
    config: Stage6Config | None = None,
    verbose: bool = False,
) -> Stage6Result:
    """
    6단계 전체 파이프라인 실행.

    Parameters
    ----------
    query          : 사용자 질문
    search_results : convert_to_stage6_input() 결과
    search_method  : 사용된 검색 방법 레이블 (기록용)
    client         : OpenAI 클라이언트 (None이면 자동 생성)
    config         : Stage6Config (None이면 기본값)
    verbose        : True이면 각 단계 중간 결과 출력
    """
    cfg    = config or Stage6Config()
    client = client or OpenAI()
    t0     = time.perf_counter()

    # ── 6-1 / 6-2: 이중 가중치 재정렬 ──────────
    logger.info("[6-1/6-2] 재정렬 시작 (입력 %d개)", len(search_results))
    reranked = rerank(
        search_results=search_results,
        top_k=cfg.rerank_top_k,
        deduplicate=cfg.deduplicate,
        similarity_threshold=cfg.similarity_threshold,
        custom_chunk_weights=cfg.custom_chunk_weights,
    )

    if verbose:
        print("\n[6-1/6-2] 재정렬 결과")
        print_rerank_table(reranked)

    if not reranked:
        logger.warning("재정렬 결과 없음. 파이프라인 중단.")
        return Stage6Result(
            query=query, search_method=search_method,
            reranked_chunks=[], compressed_chunks=[],
            final_prompt=[], answer="검색 결과가 없어 답변을 생성할 수 없습니다.",
            elapsed_sec=time.perf_counter() - t0,
        )

    # ── 6-3: Contextual Compression ─────────────
    logger.info("[6-3] Contextual Compression 시작 (대상 %d개)", len(reranked))
    compressed = contextual_compress(
        client=client,
        query=query,
        ranked_chunks=reranked,
        top_k_compress=cfg.compress_top_k,
        top_k_final=cfg.final_top_k,
        model=cfg.compression_model,
        min_len=cfg.min_compressed_len,
    )

    if verbose:
        print_compressed_table(compressed)

    # 압축 실패 시 원문으로 폴백
    if not compressed:
        logger.warning("압축 결과 없음. 원문으로 대체.")
        compressed = [
            CompressedChunk(
                original_content=rc.content,
                compressed_content=rc.content,
                metadata=rc.metadata,
                final_score=rc.final_score,
            )
            for rc in reranked[:cfg.final_top_k]
        ]

    # ── 6-4: 프롬프트 빌드 & GPT 호출 ──────────
    logger.info("[6-4] GPT 최종 답변 생성")
    messages = build_prompt(query=query, compressed_chunks=compressed)

    if verbose:
        print("\n[6-4] 프롬프트 미리보기")
        for m in messages:
            print(f"\n--- {m['role'].upper()} ---")
            print(m["content"][:400] + ("…" if len(m["content"]) > 400 else ""))

    answer = call_final_gpt(
        client=client,
        messages=messages,
        model=cfg.final_model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )

    elapsed = time.perf_counter() - t0
    logger.info("6단계 완료 (%.2f초)", elapsed)

    result = Stage6Result(
        query=query,
        search_method=search_method,
        reranked_chunks=reranked,
        compressed_chunks=compressed,
        final_prompt=messages,
        answer=answer,
        elapsed_sec=elapsed,
    )

    return result