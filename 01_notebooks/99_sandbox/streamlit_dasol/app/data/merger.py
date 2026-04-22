"""
merger.py
─────────
제품 CSV  (product_id, product_name, ingredient_name, ewg, purpose,
           is_allergy, limitation, forbidden, category)
+
성분 레퍼런스 CSV (성분명, INCI, 한글명, AI설명, 스코어, 데이터 등급, 링크)

→ merged_ingredients.csv  (모든 컬럼 + ref_* 접두사 컬럼)

매칭 전략 (우선순위 순):
  1. ingredient_name (소문자·공백 제거) == 성분명 (정규화)
  2. ingredient_name == INCI (정규화)
  3. ingredient_name == 한글명 (정규화)

실행:
  python -m app.data.merger \
      --products  data/products.csv \
      --reference data/ingredients_ref.csv \
      --output    data/merged_ingredients.csv
"""

import argparse
import re
import unicodedata
import pandas as pd
from pathlib import Path


# ── 정규화 유틸 ──────────────────────────────────────────────
def _normalize(text: str) -> str:
    """소문자, 전각→반각, 공백·특수문자 제거"""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)   # 전각 → 반각
    text = text.lower().strip()
    text = re.sub(r"[\s\-_/·•,.]", "", text)     # 구분자 제거
    return text


def _parse_score(score_str: str) -> dict:
    """
    "[안전] 1 등급" → {"safety_label": "안전", "safety_grade": 1}
    "[주의] 3 등급" → {"safety_label": "주의", "safety_grade": 3}
    파싱 실패 시 None 반환
    """
    if not isinstance(score_str, str):
        return {"safety_label": None, "safety_grade": None}
    m = re.search(r"\[(.+?)\].*?(\d+)", score_str)
    if m:
        return {"safety_label": m.group(1).strip(), "safety_grade": int(m.group(2))}
    return {"safety_label": None, "safety_grade": None}


# ── 메인 로직 ────────────────────────────────────────────────
def merge(products_path: str, reference_path: str, output_path: str) -> pd.DataFrame:
    products_path  = Path(products_path)
    reference_path = Path(reference_path)
    output_path    = Path(output_path)

    print(f"[merge] 제품 CSV 로드: {products_path}")
    prod_df = pd.read_csv(products_path, encoding="utf-8-sig")

    print(f"[merge] 성분 레퍼런스 로드: {reference_path}")
    ref_df = pd.read_csv(reference_path, encoding="utf-8-sig")

    # 레퍼런스 컬럼 정리
    ref_df = ref_df.rename(columns={
        "성분명":   "ref_name",
        "INCI":    "ref_inci",
        "한글명":   "ref_korean",
        "AI설명":   "ref_ai_desc",
        "스코어":   "ref_score_raw",
        "데이터 등급": "ref_data_grade",
        "링크":    "ref_link",
    })

    # 정규화 키 생성
    ref_df["_key_name"]   = ref_df["ref_name"].map(_normalize)
    ref_df["_key_inci"]   = ref_df["ref_inci"].map(_normalize)
    ref_df["_key_korean"] = ref_df["ref_korean"].map(_normalize)

    # 스코어 파싱
    parsed = ref_df["ref_score_raw"].map(_parse_score).apply(pd.Series)
    ref_df = pd.concat([ref_df, parsed], axis=1)

    # ── 3단계 매핑 딕셔너리 구성 ──
    # 나중에 추가된 키가 앞선 키를 덮어쓰지 않도록 순서 보장
    lookup: dict[str, pd.Series] = {}

    for _, row in ref_df.iterrows():
        for key_col in ("_key_korean", "_key_inci", "_key_name"):
            k = row[key_col]
            if k and k not in lookup:
                lookup[k] = row

    print(f"[merge] 레퍼런스 항목 수: {len(ref_df)}  /  룩업 키 수: {len(lookup)}")

    # ── 제품 CSV 성분명 컬럼 자동 감지 ──
    print(f"[merge] 제품 CSV 컬럼: {list(prod_df.columns)}")
    ING_CANDIDATES = ["ingredient_name", "korean", "성분명", "성분", "name", "ingr_name"]
    ing_col = next((c for c in ING_CANDIDATES if c in prod_df.columns), None)
    if ing_col is None:
        raise ValueError(
            f"성분명 컬럼을 찾을 수 없습니다.\n"
            f"제품 CSV 컬럼: {list(prod_df.columns)}\n"
            f"다음 중 하나로 컬럼명을 맞춰주세요: {ING_CANDIDATES}"
        )
    print(f"[merge] 성분명 컬럼 사용: '{ing_col}'")

    # ── 제품 CSV에 매핑 ──
    prod_df["_key"] = prod_df[ing_col].map(_normalize)

    ref_cols = [
        "ref_name", "ref_inci", "ref_korean",
        "ref_ai_desc", "ref_score_raw", "safety_label", "safety_grade",
        "ref_data_grade", "ref_link",
    ]

    def _lookup_row(key: str) -> pd.Series:
        row = lookup.get(key)
        if row is not None:
            return row[ref_cols]
        return pd.Series({c: None for c in ref_cols})

    matched = prod_df["_key"].apply(_lookup_row)
    merged  = pd.concat([prod_df.drop(columns=["_key"]), matched], axis=1)

    # ── 매칭 통계 ──
    total       = len(merged)
    matched_cnt = merged["ref_ai_desc"].notna().sum()
    unmatched   = merged[merged["ref_ai_desc"].isna()][ing_col].unique()

    print(f"\n[merge] 전체 성분 행: {total}")
    print(f"[merge] 매칭 성공:    {matched_cnt} ({matched_cnt/total*100:.1f}%)")
    print(f"[merge] 미매칭 고유 성분: {len(unmatched)}개")

    if len(unmatched) > 0:
        print("\n[merge] 미매칭 성분 목록 (상위 30개):")
        for ing in unmatched[:30]:
            print(f"  - {ing}")
        if len(unmatched) > 30:
            print(f"  ... 외 {len(unmatched)-30}개")

    # ── 저장 ──
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[merge] 저장 완료: {output_path}")

    return merged


# ── CLI ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="제품 CSV + 성분 레퍼런스 매핑")
    parser.add_argument("--products",  default="data/products.csv",
                        help="크롤링한 제품 CSV 경로")
    parser.add_argument("--reference", default="data/ingredients_ref.csv",
                        help="성분 레퍼런스 CSV 경로")
    parser.add_argument("--output",    default="data/merged_ingredients.csv",
                        help="출력 CSV 경로")
    args = parser.parse_args()

    merge(args.products, args.reference, args.output)
