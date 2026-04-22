"""
02_src/01_data/01_preprocessing/chunker.py
4가지 가중치 프리셋 청크 생성 + 검증

변경사항:
  - select_best_rows() : 여러 행 병합 방식
  - _fill_defaults()   : coos_score/hw_ewg/pc_rating null → 0
                         hw_ewg 범위값(1_2, 2_9 등) → 뒤 숫자로 변환
                         각국 규제 컬럼 null → "없음"
  - build_chunks()     : 각국 규제 컬럼 expert 청크에 포함
                         메타데이터 완전판
"""

import os
import sys

_HERE   = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.join(_HERE, "..", "..", "00_common")
if _COMMON not in sys.path:
    sys.path.insert(0, os.path.normpath(_COMMON))

import numpy as np
from collections import Counter, defaultdict
from logger import get_logger

logger = get_logger(__name__)


# ── 유효값 필터 ───────────────────────────────────────────────
def is_valid(val) -> bool:
    if val is None:
        return False
    s = str(val).strip()
    if s in ("", "nan", "NaN", "None", "없음", "0"):
        return False
    try:
        if np.isnan(float(s)):
            return False
    except (ValueError, TypeError):
        pass
    return True


# ── hw_ewg 범위값 변환 ────────────────────────────────────────
def _parse_hw_ewg(val) -> int:
    """
    hw_ewg 값을 정수로 변환합니다.
    범위값(1_2, 2_9, 3_10 등)은 뒤 숫자를 사용합니다.
    예) "1_2" → 2, "2_9" → 9, "3_10" → 10, "1" → 1
    """
    if val is None:
        return 0
    val_str = str(val).strip()
    if val_str in ("", "nan", "NaN", "None", "0"):
        return 0
    try:
        if "_" in val_str:
            # 범위값 → 뒤 숫자 사용
            return int(val_str.split("_")[-1])
        else:
            return int(float(val_str))
    except (ValueError, TypeError):
        return 0


# ── 병합 후 결측값 채우기 ─────────────────────────────────────
def _fill_defaults(row: dict) -> dict:
    """
    점수 컬럼 : null/빈칸 → 0
    hw_ewg    : 범위값(1_2 등) → 뒤 숫자로 변환
    규제 컬럼 : null/빈칸 → "없음"
    """
    # coos_score, pc_rating → 0
    for col in ("coos_score", "pc_rating"):
        val = row.get(col)
        if val is None or str(val).strip() in ("", "nan", "NaN", "None"):
            row[col] = 0
        else:
            try:
                if np.isnan(float(str(val))):
                    row[col] = 0
            except (ValueError, TypeError):
                pass

    # hw_ewg → 범위값 포함 정수 변환
    row["hw_ewg"] = _parse_hw_ewg(row.get("hw_ewg"))

    # 규제 컬럼 → "없음"
    reg_cols = [
        "coos_kr_restricted",
        "coos_cn_restricted",
        "coos_tw_restricted",
        "coos_jp_restricted",
        "coos_eu_restricted",
        "coos_asean_restricted",
    ]
    for col in reg_cols:
        val = row.get(col)
        if val is None or str(val).strip() in ("", "nan", "NaN", "None"):
            row[col] = "없음"

    return row


# ── 행 병합 ───────────────────────────────────────────────────
def select_best_rows(data: list, priority_cols: list) -> dict:
    """
    같은 ingredient_ko를 가진 행들을 모두 병합합니다.
    각 컬럼에서 유효한 값이 처음 나오면 채우고, 이미 채워졌으면 건너뜁니다.
    병합 후 결측값을 기본값으로 채웁니다.
    """
    groups: dict[str, list] = defaultdict(list)
    for row in data:
        groups[row.get("ingredient_ko", "")].append(row)

    merged = {}
    for ing, rows in groups.items():
        base = {}
        for row in rows:
            for col, val in row.items():
                if not is_valid(base.get(col)) and is_valid(val):
                    base[col] = val
        merged[ing] = _fill_defaults(base)

    dup_count = sum(1 for rows in groups.values() if len(rows) > 1)
    logger.info(
        f"[병합] {len(data)}행 → {len(merged)}개 성분 "
        f"(중복 성분 {dup_count}개 병합 완료)"
    )
    return merged


# ── 청크 생성 ─────────────────────────────────────────────────
def build_chunks(best_rows: dict, weights: dict, score_label_map: dict) -> list:
    chunks = []

    for ing, row in best_rows.items():
        ingredient    = str(row.get("ingredient_ko") or "")
        ingredient_en = str(row.get("ingredient_en") or "")

        # ── 메타데이터 ────────────────────────────────────────
        base_meta = {
            "ingredient_ko":                 ingredient,
            "ingredient_en":                 ingredient_en,
            # EWG 점수 (null→0, 범위값→뒤숫자 처리됨)
            "coos_score":                    row.get("coos_score", 0),
            "coos_data_grade":               row.get("coos_data_grade"),
            "hw_ewg":                        row.get("hw_ewg", 0),
            "hw_ewg_data_availability_text": row.get("hw_ewg_data_availability_text"),
            # 등급 (null→0 처리됨)
            "pc_rating":                     row.get("pc_rating", 0),
            # 규제 (null→"없음" 처리됨)
            "coos_kr_restricted":            row.get("coos_kr_restricted", "없음"),
            "coos_cn_restricted":            row.get("coos_cn_restricted", "없음"),
            "coos_tw_restricted":            row.get("coos_tw_restricted", "없음"),
            "coos_jp_restricted":            row.get("coos_jp_restricted", "없음"),
            "coos_eu_restricted":            row.get("coos_eu_restricted", "없음"),
            "coos_asean_restricted":         row.get("coos_asean_restricted", "없음"),
        }

        # ── EWG 청크 ──────────────────────────────────────────
        ewg_parts = []

        coos_score = row.get("coos_score", 0)
        if coos_score and str(coos_score) != "0":
            label = score_label_map.get(str(coos_score), str(coos_score))
            ewg_parts.append(f"EWG 스코어: {label} ({coos_score}등급)")

        if is_valid(row.get("coos_data_grade")):
            ewg_parts.append(f"데이터 등급: {row['coos_data_grade']}")

        hw_ewg = row.get("hw_ewg", 0)
        if hw_ewg and str(hw_ewg) != "0":
            ewg_parts.append(f"화해 EWG: {hw_ewg}")

        if is_valid(row.get("hw_ewg_data_availability_text")):
            ewg_parts.append(f"화해 데이터 등급: {row['hw_ewg_data_availability_text']}")

        if ewg_parts:
            chunks.append({
                "page_content": f"[{ingredient}] " + " / ".join(ewg_parts),
                "metadata": {
                    **base_meta,
                    "chunk_type":   "ewg",
                    "chunk_weight": weights["ewg"],
                },
            })

        # ── Basic Info 청크 ────────────────────────────────────
        basic_parts = []
        for col, label in [
            ("coos_function",  "기능"),
            ("coos_type",      "종류"),
            ("pc_effect",      "효과"),
            ("pc_category",    "분류"),
            ("hw_purpose",     "목적"),
            ("hw_limitation",  "사용제한"),
            ("hw_forbidden",   "금지여부"),
        ]:
            if is_valid(row.get(col)):
                basic_parts.append(f"{label}: {row[col]}")

        if basic_parts:
            chunks.append({
                "page_content": f"[{ingredient}] " + " / ".join(basic_parts),
                "metadata": {
                    **base_meta,
                    "chunk_type":   "basic_info",
                    "chunk_weight": weights["basic_info"],
                },
            })

        # ── Expert 청크 ────────────────────────────────────────
        expert_parts = []

        if is_valid(row.get("pc_description")):
            expert_parts.append(f"설명: {row['pc_description']}")
        if is_valid(row.get("coos_ai_description")):
            expert_parts.append(f"AI설명: {row['coos_ai_description']}")

        # 각국 규제 (실제 규제 내용이 있을 때만 포함)
        for col, label in [
            ("coos_kr_restricted",    "국내규제"),
            ("coos_cn_restricted",    "중국규제"),
            ("coos_tw_restricted",    "대만규제"),
            ("coos_jp_restricted",    "일본규제"),
            ("coos_eu_restricted",    "유럽규제"),
            ("coos_asean_restricted", "아세안규제"),
        ]:
            val = row.get(col, "없음")
            if val and str(val).strip() not in ("없음", "", "nan", "NaN", "None"):
                expert_parts.append(f"{label}: {val}")

        if is_valid(row.get("hw_category")):
            expert_parts.append(f"카테고리: {row['hw_category']}")

        if expert_parts:
            chunks.append({
                "page_content": f"[{ingredient}] " + " | ".join(expert_parts),
                "metadata": {
                    **base_meta,
                    "chunk_type":   "expert",
                    "chunk_weight": weights["expert"],
                },
            })

    logger.info(f"[청크 생성] {len(best_rows)}개 성분 → {len(chunks)}개 청크")
    return chunks


# ── 청크 검증 ─────────────────────────────────────────────────
def validate_chunks(chunks: list, preset_id: int) -> None:
    type_map: dict[str, list] = defaultdict(list)
    for c in chunks:
        m = c.get("metadata", {})
        type_map[m.get("chunk_type", "unknown")].append(m.get("chunk_weight", 0))

    for t in ("ewg", "basic_info", "expert"):
        ws   = type_map.get(t, [])
        avg  = sum(ws) / len(ws) if ws else 0
        ings = [c["metadata"]["ingredient_ko"]
                for c in chunks if c["metadata"].get("chunk_type") == t]
        dups   = {k: v for k, v in Counter(ings).items() if v > 1}
        status = "✅ 중복 없음" if not dups else f"❌ 중복 {len(dups)}건"
        logger.info(
            f"  프리셋{preset_id} | {t:12}: {len(ws):5}개 "
            f"weight={avg:.2f} | {status}"
        )

    empty = [c for c in chunks if not str(c.get("page_content", "")).strip()]
    if empty:
        logger.warning(f"  프리셋{preset_id} | 빈 청크 {len(empty)}개 발견!")

    # hw_ewg 변환 결과 확인
    ewg_chunks = [c for c in chunks if c["metadata"].get("chunk_type") == "ewg"]
    hw_ewg_vals = [c["metadata"].get("hw_ewg", 0) for c in chunks]
    str_vals = [v for v in hw_ewg_vals if isinstance(v, str) and "_" in str(v)]
    if str_vals:
        logger.warning(f"  프리셋{preset_id} | hw_ewg 범위값 미변환: {len(str_vals)}개")
    else:
        logger.info(f"  프리셋{preset_id} | hw_ewg 범위값 변환 완료 ✅")