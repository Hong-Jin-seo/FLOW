"""
6단계: 이중 가중치 재정렬 모듈

실제 메타데이터 키 (minha_retriever.py 기준)
  - ingredient_ko : 성분명 한국어
  - ingredient_en : 성분명 영어
  - chunk_type    : 청크 유형
  - coos_score    : 정수  0=결측, 1=안전, 2=주의, 3=위험
  - hw_ewg        : 정수  0=결측, 1~3=Good, 4~10=Others
  - pc_rating     : 정수  0=결측, 1=훌륭함, 2=좋음, 3=보통, 4=나쁨, 5=매우나쁨

최종점수 공식:
  rerank_score = search_score × chunk_weight × source_weight

  chunk_weight  : preset별 비율 (config.yaml weight_presets 기반)
  source_weight : 데이터 출처 신뢰도 기반 (coos×1.8, hwahae×1.5, paula×1.3)
                  여러 출처가 있으면 평균값 사용. 결측 시 1.0(중립)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. 청크 유형 가중치
# ──────────────────────────────────────────────

# config.yaml weight_presets 기반 preset별 chunk_weight (비율 그대로 사용)
PRESET_CHUNK_WEIGHTS: dict[int, dict[str, float]] = {
    1: {"ewg": 0.33, "basic_info": 0.33, "expert": 0.33},
    2: {"ewg": 0.50, "basic_info": 0.35, "expert": 0.15},
    3: {"ewg": 0.40, "basic_info": 0.45, "expert": 0.15},
    4: {"ewg": 0.45, "basic_info": 0.45, "expert": 0.10},
}


# ──────────────────────────────────────────────
# 2. source_weight 테이블
# ──────────────────────────────────────────────

# 데이터 출처 신뢰도 기반 가중치
SOURCE_WEIGHT_MAP: dict[str, float] = {
    "coos":   1.8,   # coos EWG 기반
    "hwahae": 1.5,   # 화해
    "paula":  1.3,   # Paula's Choice
}
SOURCE_WEIGHT_DEFAULT = 1.0   # 출처 없을 때 중립


def compute_source_weight(used_sources: list[str]) -> float:
    """
    used_sources 리스트 기반 source_weight 계산.
    Q값으로 가중 평균 적용 (결측 출처는 자동 제외).

    Q_COOS=0.3419 / Q_HWAHAE=0.4989 / Q_PAULA=0.5011
    """
    if not used_sources:
        return SOURCE_WEIGHT_DEFAULT

    Q_MAP = {"coos": Q_COOS, "hwahae": Q_HWAHAE, "paula": Q_PAULA}
    weighted_sum = sum(
        SOURCE_WEIGHT_MAP.get(s, SOURCE_WEIGHT_DEFAULT) * Q_MAP.get(s, 1.0)
        for s in used_sources
    )
    total_q = sum(Q_MAP.get(s, 1.0) for s in used_sources)
    return round(weighted_sum / total_q, 4)


# ──────────────────────────────────────────────
# 3. 도메인 점수 테이블 & Q값
# ──────────────────────────────────────────────

# coos_score 정수값 → COOS 수치 매핑
# 0=결측, 1=안전(2.0), 2=주의(-1.0), 3=위험(-3.0)
COOS_SCORE_MAP: dict[int, float] = {
    1:  2.0,   # 안전
    2: -1.0,   # 주의
    3: -3.0,   # 위험
}

# 화해 EWG → WoE
WOE_HWAHAE: dict[str, float] = {
    "Good":   0.3715,
    "Others": -2.5706,
}

# pc_rating 정수값 → WoE 매핑
# 0=결측, 1=훌륭함 ~ 5=매우나쁨
WOE_PAULA: dict[int, float] = {
    1:  0.5081,   # 훌륭함
    2:  0.2313,   # 좋음
    3: -0.5579,   # 보통
    4: -1.0926,   # 나쁨
    5: -1.5810,   # 매우나쁨
}

# Q값 (전체 3개 사이트 기준 / 화해+paula 재정규화)
Q_COOS   = 0.3419
Q_HWAHAE = 0.4989
Q_PAULA  = 0.5011


def _get_hwahae_grade(ewg_val: Any) -> str | None:
    """hw_ewg 정수값 → 'Good' / 'Others' 변환. 0=결측."""
    if ewg_val is None:
        return None
    try:
        val = int(ewg_val)
    except (ValueError, TypeError):
        return None
    if val == 0:
        return None
    return "Good" if val <= 3 else "Others"


def compute_final_score(
    coos_score: Any,
    hw_ewg: Any,
    pc_rating: Any,
) -> tuple[float | None, list[str]]:
    """
    Final Score = Q_coos × coos수치 + Q_hwahae × WoE_hwahae + Q_paula × WoE_paula
    결측 출처는 Q값 재정규화로 처리.

    Parameters
    ----------
    coos_score : 정수  0=결측, 1=안전, 2=주의, 3=위험
    hw_ewg     : 정수  0=결측, 1~3=Good, 4~10=Others
    pc_rating  : 정수  0=결측, 1=훌륭함 ~ 5=매우나쁨

    Returns
    -------
    (final_score, used_sources)
    """
    scores:  list[float] = []
    weights: list[float] = []
    sources: list[str]   = []

    # coos
    try:
        coos_int = int(coos_score) if coos_score is not None else 0
    except (ValueError, TypeError):
        coos_int = 0
    if coos_int in COOS_SCORE_MAP:
        scores.append(COOS_SCORE_MAP[coos_int])
        weights.append(Q_COOS)
        sources.append("coos")

    # 화해
    hw_grade = _get_hwahae_grade(hw_ewg)
    if hw_grade and hw_grade in WOE_HWAHAE:
        scores.append(WOE_HWAHAE[hw_grade])
        weights.append(Q_HWAHAE)
        sources.append("hwahae")

    # paula
    try:
        pc_int = int(pc_rating) if pc_rating is not None else 0
    except (ValueError, TypeError):
        pc_int = 0
    if pc_int in WOE_PAULA:
        scores.append(WOE_PAULA[pc_int])
        weights.append(Q_PAULA)
        sources.append("paula")

    if not scores:
        return None, []

    # 결측 시 Q값 재정규화
    total_w      = sum(weights)
    norm_weights = [w / total_w for w in weights]
    final_score  = sum(s * w for s, w in zip(scores, norm_weights))
    return round(final_score, 4), sources



# ──────────────────────────────────────────────
# 4. 데이터 모델
# ──────────────────────────────────────────────

@dataclass
class RankedChunk:
    """재점수 후 청크 컨테이너."""
    content:        str
    metadata:       dict[str, Any]
    original_score: float
    chunk_weight:   float = 1.0
    source_weight:  float = 1.0
    used_sources:   list[str] = field(default_factory=list)
    final_score:    float = field(init=False)

    def __post_init__(self) -> None:
        self.final_score = (
            self.original_score
            * self.chunk_weight
            * self.source_weight
        )

    def recompute(self) -> None:
        self.final_score = (
            self.original_score
            * self.chunk_weight
            * self.source_weight
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "content":        self.content,
            "metadata":       self.metadata,
            "original_score": self.original_score,
            "chunk_weight":   self.chunk_weight,
            "source_weight":  self.source_weight,
            "used_sources":   self.used_sources,
            "final_score":    self.final_score,
        }


# ──────────────────────────────────────────────
# 5. 중복 제거
# ──────────────────────────────────────────────

def _deduplicate(
    chunks: list[RankedChunk],
    similarity_threshold: float = 0.85,
) -> list[RankedChunk]:
    """Jaccard 유사도 기반 중복 청크 제거."""
    def jaccard(a: str, b: str) -> float:
        sa, sb = set(a.split()), set(b.split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    kept: list[RankedChunk] = []
    for candidate in chunks:
        if not any(jaccard(candidate.content, k.content) >= similarity_threshold for k in kept):
            kept.append(candidate)
    return kept


# ──────────────────────────────────────────────
# 6. 메인 재정렬 함수
# ──────────────────────────────────────────────

def rerank(
    search_results: list[dict[str, Any]],
    top_k: int = 5,
    deduplicate: bool = True,
    similarity_threshold: float = 0.85,
    custom_chunk_weights: dict[str, float] | None = None,
) -> list[RankedChunk]:
    """
    5단계 검색 결과에 3중 가중치를 적용하고 상위 top_k 청크 반환.

    rerank_score = search_score × chunk_weight × source_weight × domain_weight

    Parameters
    ----------
    search_results       : convert_to_stage6_input() 변환 결과
    top_k                : 반환할 상위 청크 수
    deduplicate          : 중복 제거 여부
    similarity_threshold : Jaccard 중복 판단 임계값
    domain_w_min/max     : 도메인 점수 → 가중치 변환 범위
    custom_chunk_weights : preset별 chunk_weight (PRESET_CHUNK_WEIGHTS[n] 전달)
    """
    c_map = custom_chunk_weights or {
        "ewg":        0.33,
        "basic_info": 0.33,
        "expert":     0.33,
    }
    ranked: list[RankedChunk] = []

    for idx, result in enumerate(search_results):
        try:
            content        = result["content"]
            metadata       = result.get("metadata", {})
            original_score = float(result.get("score", 0.0))

            # ── chunk_weight ─────────────────────
            chunk_type = (metadata.get("chunk_type") or "unknown").lower()
            cw = c_map.get(chunk_type, 0.33)

            # ── source_weight ────────────────────
            coos_score = metadata.get("coos_score")
            hw_ewg     = metadata.get("hw_ewg")
            pc_rating  = metadata.get("pc_rating")

            _, used = compute_final_score(coos_score, hw_ewg, pc_rating)
            sw = compute_source_weight(used)

            chunk = RankedChunk(
                content=content,
                metadata=metadata,
                original_score=original_score,
                chunk_weight=cw,
                source_weight=sw,
                used_sources=used,
            )
            ranked.append(chunk)
            logger.debug(
                "[%d] %s | orig=%.4f cw=%.2f sw=%.2f → final=%.4f  출처=%s",
                idx,
                metadata.get("ingredient_ko", "?"),
                original_score, cw, sw, chunk.final_score,
                "+".join(used) if used else "-",
            )

        except (KeyError, TypeError, ValueError) as e:
            logger.warning("결과 [%d] 처리 중 오류, 건너뜀: %s", idx, e)

    ranked.sort(key=lambda c: c.final_score, reverse=True)

    if deduplicate:
        before = len(ranked)
        ranked = _deduplicate(ranked, similarity_threshold)
        logger.info("중복 제거: %d → %d개", before, len(ranked))

    top = ranked[:top_k]
    logger.info("재정렬 완료 → 상위 %d개 반환 (전체 %d개 중)", len(top), len(ranked))
    return top


# ──────────────────────────────────────────────
# 7. 디버그 출력
# ──────────────────────────────────────────────

def print_rerank_table(chunks: list[RankedChunk]) -> None:
    header = (
        f"{'순위':<4} {'성분명':<20} {'orig':>7} {'cw':>5} "
        f"{'sw':>5} {'final':>8}  출처"
    )
    print(header)
    print("─" * len(header))
    for i, c in enumerate(chunks, 1):
        name    = c.metadata.get("ingredient_ko", "?")[:18]
        sources = "+".join(c.used_sources) if c.used_sources else "-"
        print(
            f"{i:<4} {name:<20} {c.original_score:>7.4f} {c.chunk_weight:>5.2f} "
            f"{c.source_weight:>5.2f} {c.final_score:>8.4f}  {sources}"
        )


# ──────────────────────────────────────────────
# 동작 확인
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    dummy_results = [
        {
            "content": "나이아신아마이드(Niacinamide)는 COOS 안전 등급 성분으로 EWG 1등급에 해당합니다. 미백·모공 축소 효과가 있습니다.",
            "metadata": {
                "ingredient_ko": "나이아신아마이드",
                "ingredient_en": "Niacinamide",
                "chunk_type":    "ewg",
                "coos_score":    1,
                "hw_ewg":        1,
                "pc_rating":     1,
            },
            "score": 0.91,
        },
        {
            "content": "나이아신아마이드는 미백·모공 축소·항산화 등 다양한 효능을 가진 성분입니다.",
            "metadata": {
                "ingredient_ko": "나이아신아마이드",
                "ingredient_en": "Niacinamide",
                "chunk_type":    "basic_info",
                "coos_score":    1,
                "hw_ewg":        1,
                "pc_rating":     1,
            },
            "score": 0.84,
        },
        {
            "content": "나이아신아마이드는 최대 20% 농도에서 안전하다고 전문가 패널이 평가했습니다.",
            "metadata": {
                "ingredient_ko": "나이아신아마이드",
                "ingredient_en": "Niacinamide",
                "chunk_type":    "expert",
                "coos_score":    1,
                "hw_ewg":        2,
                "pc_rating":     2,
            },
            "score": 0.78,
        },
        {
            "content": "파라벤 계열 방부제는 EWG 4~6 등급으로 호르몬 교란 가능성이 제기됩니다.",
            "metadata": {
                "ingredient_ko": "파라벤",
                "ingredient_en": "Paraben",
                "chunk_type":    "ewg",
                "coos_score":    2,
                "hw_ewg":        4,
                "pc_rating":     3,
            },
            "score": 0.61,
        },
    ]

    # preset 1 기준으로 테스트
    from jinseo_stage6_rerank import PRESET_CHUNK_WEIGHTS
    results = rerank(
        dummy_results,
        top_k=5,
        custom_chunk_weights=PRESET_CHUNK_WEIGHTS[1],
    )
    print_rerank_table(results)