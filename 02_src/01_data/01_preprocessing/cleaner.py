"""
02_src/01_data/01_preprocessing/cleaner.py
각 소스 전처리 (드롭/리네임/결측치/스코어 변환)

PaulasChoice, COOS, 화해, EWG 각 데이터 소스에 대해
불필요 컬럼 제거, 컬럼명 정규화(접두어 부여), 결측치 처리,
스코어 매핑을 수행합니다.
"""

import os
import sys
import re

_HERE   = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.join(_HERE, "..", "..", "00_common")
if _COMMON not in sys.path:
    sys.path.insert(0, os.path.normpath(_COMMON))

import pandas as pd
from logger import get_logger

logger   = get_logger(__name__)
KEY_COLS = ["ingredient_ko", "ingredient_en"]


def clean_paulaschoice(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """PaulasChoice 전처리 — 결측 행 제거, 컬럼에 pc_ 접두어 부여"""
    df = df.drop(columns=[c for c in cfg["drop_cols"] if c in df.columns])
    df = df.dropna(how="any")
    df = df.rename(columns=cfg["rename_cols"])
    df = df.rename(columns=lambda c: f"pc_{c}" if c not in KEY_COLS else c)
    logger.info(f"[PaulasChoice] 전처리 완료: {df.shape}")
    return df


def clean_coos(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """COOS 전처리 — 지정 컬럼 결측치 대체 후 coos_ 접두어 부여"""
    df = df.drop(columns=[c for c in cfg["drop_cols"] if c in df.columns])
    for col, val in cfg["fillna_cols"].items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    df = df.dropna(how="any")
    df = df.rename(columns=cfg["rename_cols"])
    df = df.rename(columns=lambda c: f"coos_{c}" if c not in KEY_COLS else c)
    logger.info(f"[COOS] 전처리 완료: {df.shape}")
    return df


def clean_hwahae(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """화해 전처리 — 지정 컬럼 결측치 대체 후 hw_ 접두어 부여"""
    df = df.drop(columns=[c for c in cfg["drop_cols"] if c in df.columns])
    for col, val in cfg["fillna_cols"].items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    df = df.dropna(how="any")
    df = df.rename(columns=cfg["rename_cols"])
    df = df.rename(columns=lambda c: f"hw_{c}" if c not in KEY_COLS else c)
    logger.info(f"[화해] 전처리 완료: {df.shape}")
    return df


def _map_coos_score(val, score_map):
    """COOS 텍스트 등급 → 숫자 코드 변환 (키워드 매칭)"""
    if pd.isna(val) or str(val).strip() == "":
        return 0
    for keyword, code in score_map.items():
        if keyword in str(val):
            return code
    return 0


def _map_pc_rating(val, rating_map):
    """PaulasChoice 등급 문자열 → 숫자 코드 변환 (정확 매칭)"""
    if pd.isna(val) or str(val).strip() == "":
        return 0
    return rating_map.get(str(val).strip(), 0)


def apply_score_mapping(df: pd.DataFrame, pre_cfg: dict) -> pd.DataFrame:
    """coos_score, pc_rating 텍스트를 숫자 코드로 일괄 변환"""
    if "coos_score" in df.columns:
        df["coos_score"] = df["coos_score"].apply(
            lambda v: _map_coos_score(v, pre_cfg["coos_score_map"])
        )
    if "pc_rating" in df.columns:
        df["pc_rating"] = df["pc_rating"].apply(
            lambda v: _map_pc_rating(v, pre_cfg["pc_rating_map"])
        )
    logger.info("[스코어 변환] 완료")
    return df


def parse_ewg_score(raw) -> int:
    """EWG 스코어 파싱 (범위 → 끝값, 없으면 0)"""
    if raw is None:
        return 0
    raw_str = str(raw).strip()
    if raw_str in ("", "nan", "None", "N/A", "-"):
        return 0
    # 숫자와 범위 구분자(-/–)만 남김
    cleaned = re.sub(r"[^\d\-–]", " ", raw_str).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    range_match = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", cleaned)
    if range_match:
        return int(range_match.group(2))
    single_match = re.match(r"^(\d+)$", cleaned)
    if single_match:
        return int(single_match.group(1))
    # 위 패턴에 안 맞으면 마지막 숫자 사용
    numbers = re.findall(r"\d+", cleaned)
    if numbers:
        return int(numbers[-1])
    return 0


def clean_ewg(df: pd.DataFrame, ing_col: str, score_col: str) -> pd.DataFrame:
    """EWG 원본 → score_parsed + ingredient_key 컬럼 생성, 빈 성분명 제거"""
    df = df.copy()
    df["score_parsed"] = df[score_col].apply(parse_ewg_score)
    df = df[
        df[ing_col].notna() &
        (df[ing_col].astype(str).str.strip() != "")
    ].copy()
    # 조인용 키: 소문자 + 공백 제거
    df["ingredient_key"] = df[ing_col].astype(str).str.strip().str.lower()
    logger.info(f"[EWG cleaner] 완료: {df.shape}")
    return df