"""
02_src/01_data/01_preprocessing/merger.py
3소스 outer join 병합 + product_db 추출

병합 후 product_db로 분리된 상품 정보 컬럼은
config.yaml의 post_merge_drop_cols에서 드롭합니다.
"""

import os
import sys

_HERE   = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.join(_HERE, "..", "..", "00_common")
if _COMMON not in sys.path:
    sys.path.insert(0, os.path.normpath(_COMMON))

import pandas as pd
from logger import get_logger

logger   = get_logger(__name__)
KEY_COLS = ["ingredient_ko", "ingredient_en"]


def merge_sources(df_pc: pd.DataFrame,
                  df_coos: pd.DataFrame,
                  df_hwahae: pd.DataFrame,
                  post_drop_cols: list) -> pd.DataFrame:
    # 키 컬럼 공백 제거 (매칭 누락 방지)
    for df in [df_pc, df_coos, df_hwahae]:
        for col in KEY_COLS:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

    df_merged = (
        df_pc
        .merge(df_coos,   on=KEY_COLS, how="outer")
        .merge(df_hwahae, on=KEY_COLS, how="outer")
    )

    # product_db로 분리된 컬럼 + 불필요 컬럼 드롭
    drop_targets = [c for c in post_drop_cols if c in df_merged.columns]
    df_merged = df_merged.drop(columns=drop_targets, errors="ignore")

    logger.info(
        f"[병합] 완료: {df_merged.shape} | "
        f"드롭된 컬럼 {len(drop_targets)}개: {drop_targets}"
    )
    return df_merged


def build_product_db(df_hwahae_raw: pd.DataFrame,
                     product_cfg: dict) -> pd.DataFrame:
    """화해 원본에서 상품 정보 컬럼만 추출합니다."""
    source_cols = [c for c in product_cfg["source_cols"]
                   if c in df_hwahae_raw.columns]
    df = df_hwahae_raw[source_cols].copy()
    df = df.rename(columns=product_cfg["rename_cols"])
    logger.info(f"[product_db] 생성 완료: {df.shape}")
    return df
