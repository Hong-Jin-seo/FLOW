"""
02_src/01_data/02_io/writer.py
JSON / CSV 저장 표준화
"""

import os
import sys

_HERE   = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.join(_HERE, "..", "..", "00_common")
if _COMMON not in sys.path:
    sys.path.insert(0, os.path.normpath(_COMMON))

import json
import numpy as np
import pandas as pd
from logger import get_logger

logger = get_logger(__name__)


class _SafeEncoder(json.JSONEncoder):
    """numpy 타입 및 NaN을 JSON 직렬화 가능한 형태로 변환"""
    def default(self, obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _ensure_dir(path):
    """저장 경로의 상위 디렉토리가 없으면 생성"""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_json(records: list, output_path: str) -> None:
    """리스트를 JSON 파일로 저장"""
    _ensure_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2, cls=_SafeEncoder)
    logger.info(f"[저장] JSON: {output_path} ({len(records)}건)")


def save_df_as_json(df: pd.DataFrame, output_path: str) -> None:
    """DataFrame을 JSON으로 저장 — NaN을 None으로 치환 후 저장"""
    records = df.where(pd.notna(df), None).to_dict(orient="records")
    save_json(records, output_path)


def save_csv(df: pd.DataFrame, output_path: str) -> None:
    """DataFrame을 CSV로 저장 — utf-8-sig(BOM 포함)로 Excel 호환"""
    _ensure_dir(output_path)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"[저장] CSV: {output_path} ({len(df)}행)")
