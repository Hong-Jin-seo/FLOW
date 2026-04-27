"""
5단계 + 6단계 통합 실행 스크립트

실행 예시:
    # preset 1만 실행 (기본)
    python run_pipeline.py

    # preset 4개 전부 실행
    python run_pipeline.py --all_presets

    # 특정 preset만 실행
    python jinseo_stage6_run_pipeline.py --preset 2

    # 옵션 조합
    python jinseo_stage6_run_pipeline.py --all_presets --query "레티놀 부작용 알려줘" --method rrf --verbose --save
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from minha_retriever import CosmeticRetriever, load_faiss_auto, SearchResponse
from jinseo_stage6_pipeline import convert_to_stage6_input, run_stage6, Stage6Config
from jinseo_stage6_rerank import PRESET_CHUNK_WEIGHTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 단일 preset 실행
# ──────────────────────────────────────────────

def run_preset(
    preset_num: int,
    query: str,
    method: str,
    top_k: int,
    client: OpenAI,
    config: Stage6Config,
    verbose: bool,
    save: bool,
) -> dict:
    """
    단일 preset에 대해 5~6단계 파이프라인 실행.
    결과 dict 반환.
    """
    print(f"\n{'='*60}")
    print(f"  PRESET {preset_num}  |  검색방법: {method.upper()}")
    print(f"{'='*60}")

    faiss_path = f"../../00_data/02_processed/faiss_index_preset{preset_num}_v2"
    logger.info("[Preset %d] FAISS 로드: %s", preset_num, faiss_path)

    # ── 5단계 ──────────────────────────────────
    try:
        faiss_index, documents = load_faiss_auto(faiss_path)
    except Exception as e:
        logger.error("[Preset %d] FAISS 로드 실패: %s", preset_num, e)
        return {"preset": preset_num, "error": str(e)}

    retriever = CosmeticRetriever(
        faiss_index=faiss_index,
        documents=documents,
        top_k=top_k,
    )

    fn_map = {
        "bm25":  retriever.search_bm25,
        "dense": retriever.search_dense,
        "rrf":   retriever.search_rrf,
        "hyde":  retriever.search_hyde,
    }

    logger.info("[Preset %d] %s 검색 시작: %s", preset_num, method.upper(), query)
    stage5_response: SearchResponse = fn_map[method](query)
    logger.info("[Preset %d] 검색 완료 → %d개 결과 (%.0fms)",
                preset_num, len(stage5_response.results), stage5_response.latency_ms)

    # 5단계 결과 출력 (verbose)
    if verbose:
        print(f"\n[5단계 결과] Preset {preset_num} | {method.upper()} | {stage5_response.latency_ms:.0f}ms")
        print(f"  {'순위':<4} {'성분명':<25} {'COOS':<6} {'EWG':<6} {'PC':<8} {'점수':>8}  청크유형")
        print("  " + "-" * 72)
        for r in stage5_response.results:
            meta = r.document.metadata
            print(
                f"  {r.rank:<4} "
                f"{meta.get('ingredient_ko', '?'):<25} "
                f"{str(meta.get('coos_score', '-')):<6} "
                f"{str(meta.get('hw_ewg', '-')):<6} "
                f"{str(meta.get('pc_rating', '-')):<8} "
                f"{r.score:>8.4f}  "
                f"{meta.get('chunk_type', '?')}"
            )

    # ── 5 → 6단계 변환 ─────────────────────────
    search_results = convert_to_stage6_input(stage5_response)

    # ── 6단계 ──────────────────────────────────
    logger.info("[Preset %d] 6단계 파이프라인 시작", preset_num)
    result = run_stage6(
        query=query,
        search_results=search_results,
        search_method=method,
        client=client,
        config=config,
        verbose=verbose,
    )

    print(result.summary())

    # JSON 저장
    if save:
        output_path = f"result_preset{preset_num}_{method}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("[Preset %d] 결과 저장: %s", preset_num, output_path)

    top1_final = result.reranked_chunks[0].final_score if result.reranked_chunks else 0.0
    all_finals = [c.final_score for c in result.reranked_chunks]

    return {
        "preset":         preset_num,
        "answer":         result.answer,
        "elapsed_sec":    result.elapsed_sec,
        "rerank_count":   len(result.reranked_chunks),
        "compress_count": len(result.compressed_chunks),
        "top1_final":     top1_final,
        "all_finals":     all_finals,
    }


# ──────────────────────────────────────────────
# 4개 preset 비교 요약 출력
# ──────────────────────────────────────────────

def print_comparison(results: list[dict], query: str) -> None:
    """4개 preset 결과를 비교 테이블로 출력."""
    print(f"\n{'='*70}")
    print(f"  전체 Preset 비교 요약")
    print(f"  질문: {query}")
    print(f"{'='*70}")
    print(f"  {'Preset':<8} {'소요(초)':>8}  {'재정렬':>6}  {'압축':>4}  {'Top1':>8}  All Finals (1~5)                      답변 미리보기")
    print(f"  {'-'*100}")
    for r in results:
        if "error" in r:
            print(f"  Preset {r['preset']}  ERROR: {r['error']}")
            continue
        preview    = r["answer"][:30].replace("\n", " ")
        finals_str = "  ".join(f"{s:.4f}" for s in r["all_finals"])
        print(
            f"  Preset {r['preset']:<3} "
            f"{r['elapsed_sec']:>8.2f}  "
            f"{r['rerank_count']:>6}  "
            f"{r['compress_count']:>4}  "
            f"{r['top1_final']:>8.4f}  "
            f"{finals_str:<38}  "
            f"{preview}…"
        )
    print(f"{'='*100}\n")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="화장품 성분 RAG 파이프라인 (5~6단계)")
    parser.add_argument("--query",       default="나이아신아마이드 안전한가요?")
    parser.add_argument("--method",      default="rrf",
                        choices=["bm25", "dense", "rrf", "hyde"])
    parser.add_argument("--preset",      type=int, default=1, choices=[1, 2, 3, 4],
                        help="단일 preset 번호 (--all_presets 없을 때 사용)")
    parser.add_argument("--all_presets", action="store_true",
                        help="preset 1~4 전부 순차 실행")
    parser.add_argument("--top_k",       type=int, default=5)
    parser.add_argument("--verbose",     action="store_true")
    parser.add_argument("--save",        action="store_true", help="각 결과를 JSON으로 저장")
    args = parser.parse_args()

    client = OpenAI()
    config_map = {
        p: Stage6Config(
            rerank_top_k=args.top_k,
            final_top_k=3,
            compression_model="gpt-4o-mini",
            final_model="gpt-4o",
            custom_chunk_weights=PRESET_CHUNK_WEIGHTS.get(p),
        )
        for p in [1, 2, 3, 4]
    }

    presets = [1, 2, 3, 4] if args.all_presets else [args.preset]

    t_total = time.perf_counter()
    all_results = []

    for preset_num in presets:
        result = run_preset(
            preset_num=preset_num,
            query=args.query,
            method=args.method,
            top_k=args.top_k,
            client=client,
            config=config_map[preset_num],
            verbose=args.verbose,
            save=args.save,
        )
        all_results.append(result)

    # 4개 전부 돌렸을 때만 비교 요약 출력
    if args.all_presets:
        print_comparison(all_results, args.query)
        logger.info("전체 소요: %.2f초", time.perf_counter() - t_total)


if __name__ == "__main__":
    main()