"""
02_src/01_data/00_ingestion/loader.py
원본 데이터 로드 + 스키마 검증
"""

import os
import sys

# 00_common 경로 추가 (이 파일이 직접 실행될 때를 위해)
_HERE = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.join(_HERE, "..", "..", "00_common")
if _COMMON not in sys.path:
    sys.path.insert(0, os.path.normpath(_COMMON))

import pandas as pd
from config_loader import load_config, resolve_path
from logger import get_logger

logger = get_logger(__name__)


def _check_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"원본 파일을 찾을 수 없습니다: {path}")


def _validate_schema(df: pd.DataFrame, required_cols: list, name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] 필수 컬럼 누락: {missing}")
    logger.info(f"[{name}] 스키마 검증 통과")


def load_paulaschoice(raw_dir, filename, required_cols):
    path = os.path.join(raw_dir, filename)
    _check_file(path)
    df = pd.read_csv(path, encoding="utf-8")
    _validate_schema(df, required_cols, "PaulasChoice")
    logger.info(f"[PaulasChoice] 로드 완료: {df.shape}")
    return df


def load_coos(raw_dir, filename, required_cols):
    path = os.path.join(raw_dir, filename)
    _check_file(path)
    df = pd.read_csv(path)
    _validate_schema(df, required_cols, "COOS")
    logger.info(f"[COOS] 로드 완료: {df.shape}")
    return df


def load_hwahae(raw_dir, filename, required_cols):
    path = os.path.join(raw_dir, filename)
    _check_file(path)
    df = pd.read_csv(path, encoding="utf-8")
    _validate_schema(df, required_cols, "화해")
    logger.info(f"[화해] 로드 완료: {df.shape}")
    return df


def load_all_raw(config: dict = None):
    if config is None:
        config = load_config()

    raw_dir   = resolve_path(config, "raw_dir")
    raw_files = config["paths"]["raw_files"]
    val_cfg   = config["validation"]

    df_pc = load_paulaschoice(
        raw_dir, raw_files["paulaschoice"],
        val_cfg["paulaschoice"]["required_cols"]
    )
    df_coos = load_coos(
        raw_dir, raw_files["coos"],
        val_cfg["coos"]["required_cols"]
    )
    df_hwahae = load_hwahae(
        raw_dir, raw_files["hwahae"],
        val_cfg["hwahae"]["required_cols"]
    )
    return df_pc, df_coos, df_hwahae
