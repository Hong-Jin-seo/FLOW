from __future__ import annotations

import math
from typing import List, Dict, Set
from dataclasses import dataclass

from minha_retriever import SearchResponse


# ──────────────────────────────────────────────
# 5-5. 성능 평가 지표
# ──────────────────────────────────────────────

@dataclass
class EvalResult:
    method: str
    precision_at_3: float
    recall_at_3: float
    mrr: float
    ndcg_at_3: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "Precision@3": self.precision_at_3,
            "Recall@3":    self.recall_at_3,
            "MRR":         self.mrr,
            "NDCG@3":      self.ndcg_at_3,
        }


class Evaluator:
    """
    검색 결과 성능 평가

    Parameters
    ----------
    relevant_ids : List[str]
        정답 문서 ID 목록 (ingredient_ko 기반)
    """

    def __init__(self, relevant_ids: List[str]):
        self.relevant_set: Set[str] = set(relevant_ids)

    # ── doc_id 추출 ───────────────────────────

    @staticmethod
    def _get_id(result) -> str:
        meta = result.document.metadata
        return meta.get("ingredient_ko", str(hash(result.document.page_content[:50])))

    # ── Precision@3 ───────────────────────────

    def precision_at_k(self, response: SearchResponse, k: int = 3) -> float:
        """상위 k개 중 정답 비율"""
        top_k = response.results[:k]
        hits  = sum(1 for r in top_k if self._get_id(r) in self.relevant_set)
        return round(hits / k, 4) if k > 0 else 0.0

    # ── Recall@3 ──────────────────────────────

    def recall_at_k(self, response: SearchResponse, k: int = 3) -> float:
        """상위 k개에서 전체 정답 중 찾은 비율"""
        if not self.relevant_set:
            return 0.0
        top_k = response.results[:k]
        hits  = sum(1 for r in top_k if self._get_id(r) in self.relevant_set)
        return round(hits / len(self.relevant_set), 4)

    # ── MRR ───────────────────────────────────

    def mrr(self, response: SearchResponse) -> float:
        """첫 번째 정답의 역순위"""
        for r in response.results:
            if self._get_id(r) in self.relevant_set:
                return round(1.0 / r.rank, 4)
        return 0.0

    # ── NDCG@3 ────────────────────────────────

    def ndcg_at_k(self, response: SearchResponse, k: int = 3) -> float:
        """Normalized Discounted Cumulative Gain"""
        top_k = response.results[:k]

        # DCG
        dcg = sum(
            (1 if self._get_id(r) in self.relevant_set else 0) / math.log2(i + 2)
            for i, r in enumerate(top_k)
        )

        # Ideal DCG (정답이 상위에 몰려있는 완벽한 경우)
        ideal_hits = min(len(self.relevant_set), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

        return round(dcg / idcg if idcg > 0 else 0.0, 4)

    # ── 전체 지표 한 번에 ─────────────────────

    def evaluate(self, response: SearchResponse) -> EvalResult:
        return EvalResult(
            method         = response.method,
            precision_at_3 = self.precision_at_k(response, k=3),
            recall_at_3    = self.recall_at_k(response, k=3),
            mrr            = self.mrr(response),
            ndcg_at_3      = self.ndcg_at_k(response, k=3),
        )

    def evaluate_all(self, responses: Dict[str, SearchResponse]) -> Dict[str, EvalResult]:
        """모든 검색 방법에 대해 한 번에 평가"""
        return {method: self.evaluate(resp) for method, resp in responses.items()}

    # ── 결과 출력 ─────────────────────────────

    @staticmethod
    def print_report(eval_results: Dict[str, EvalResult]):
        print("\n" + "=" * 55)
        print(f"  {'방법':<8} {'P@3':>7} {'R@3':>7} {'MRR':>7} {'NDCG@3':>8}")
        print("  " + "-" * 50)
        for method, er in eval_results.items():
            print(
                f"  {method.upper():<8} "
                f"{er.precision_at_3:>7.4f} "
                f"{er.recall_at_3:>7.4f} "
                f"{er.mrr:>7.4f} "
                f"{er.ndcg_at_3:>8.4f}"
            )
        print("=" * 55)
