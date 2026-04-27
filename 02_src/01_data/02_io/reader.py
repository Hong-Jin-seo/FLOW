"""
02_src/01_data/02_io/reader.py
JSON / CSV 읽기 표준화
"""

import os
import sys

_HERE   = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.join(_HERE, "..", "..", "00_common")
if _COMMON not in sys.path:
    sys.path.insert(0, os.path.normpath(_COMMON))

import json
import pandas as pd
from logger import get_logger

logger = get_logger(__name__)


def load_json(path: str) -> list:
    """JSON 파일 로드 — BOM(UTF-8 BOM) 자동 제거 포함"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {path}")
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:3] == b"\xef\xbb\xbf":
        raw = raw[3:]
    # UTF-8 BOM(0xEF 0xBB 0xBF)이 있으면 제거
    data = json.loads(raw.decode("utf-8", errors="ignore"))
    logger.info(f"[로드] JSON: {path} ({len(data)}건)")
    return data


def load_csv(path: str, **kwargs) -> pd.DataFrame:
    """CSV 파일 로드 — kwargs는 pd.read_csv에 그대로 전달"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {path}")
    df = pd.read_csv(path, **kwargs)
    logger.info(f"[로드] CSV: {path} ({df.shape})")
    return df
