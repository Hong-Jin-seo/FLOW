"""
CSV → 문서 청크 → FAISS 인덱스 + 제품 메타데이터 JSON 생성

우선 순위에 따라 CSV 자동 선택:
  1. data/merged_ingredients.csv  (merger.py 실행 후 생성)
  2. data/hwahae_ingredients.csv  (기본 크롤링 결과)

저장 파일:
  vectorstore/index.faiss         - FAISS 벡터 인덱스
  vectorstore/chunks.pkl          - 청크 리스트
  vectorstore/products_meta.json  - 제품별 메타데이터 (Curator 필터링용)
"""

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR     = Path(__file__).resolve().parent.parent.parent
DATA_DIR     = BASE_DIR / "data"
VS_DIR       = BASE_DIR / "vectorstore"
INDEX_PATH   = VS_DIR / "index.faiss"
CHUNKS_PATH  = VS_DIR / "chunks.pkl"
META_PATH    = VS_DIR / "products_meta.json"

EMBED_MODEL  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 원본 CSV 컬럼 → 표준 이름 매핑
# 크롤링 결과에 따라 컬럼명이 다를 수 있으므로 유연하게 처리
COL_ALIASES = {
    "ingredient_name": ["ingredient_name", "korean", "성분명"],
    "english"        : ["english", "inci", "ref_inci", "INCI"],
    "ewg"            : ["ewg", "ewg_grade"],
    "purpose"        : ["purpose", "기능", "function"],
    "is_allergy"     : ["is_allergy", "allergy", "알레르기"],
    "limitation"     : ["limitation", "사용제한", "limit"],
    "forbidden"      : ["forbidden", "사용금지", "forbidden_use"],
    "category"       : ["category", "카테고리"],
    "product_name"   : ["product_name", "제품명"],
    "brand_name"     : ["brand_name", "브랜드명", "brand"],
    "sub_product_name": ["sub_product_name", "sub_name", "서브제품명"],
}


def _col(df: pd.DataFrame, key: str, default="") -> pd.Series:
    """별칭 목록에서 실제 존재하는 컬럼을 반환"""
    for alias in COL_ALIASES.get(key, [key]):
        if alias in df.columns:
            return df[alias]
    return pd.Series([default] * len(df), index=df.index)


def _val(row: pd.Series, key: str, default="") -> str:
    for alias in COL_ALIASES.get(key, [key]):
        if alias in row.index and pd.notna(row[alias]):
            return str(row[alias])
    return default


# ── EWG 헬퍼 ──────────────────────────────────────────────────
def ewg_label(ewg) -> str:
    if not ewg or pd.isna(ewg):
        return "정보없음"
    ewg_str = str(ewg).replace("_", "~")
    try:
        score = int(str(ewg).split("_")[0])
        if score <= 2:   return f"안전({ewg_str}등급)"
        elif score <= 4: return f"보통({ewg_str}등급)"
        else:            return f"주의({ewg_str}등급)"
    except Exception:
        return f"{ewg_str}등급"


def ewg_min(ewg) -> int:
    try:
        return int(str(ewg).split("_")[0])
    except Exception:
        return 5


# ── 그룹키 보정 ───────────────────────────────────────────────
def _ensure_group_cols(df: pd.DataFrame) -> pd.DataFrame:
    """groupby에 필요한 컬럼이 없으면 기본값으로 채움"""
    if "product_id" not in df.columns:
        df["product_id"] = range(len(df))
    if "sub_product_name" not in df.columns:
        df["sub_product_name"] = _col(df, "product_name").fillna("기본")
    return df


# ── 제품 메타데이터 생성 ───────────────────────────────────────
def build_products_meta(df: pd.DataFrame) -> list[dict]:
    """
    제품 단위로 집계한 메타데이터.
    Curator의 후보 필터링에 사용됨.
    """
    df = _ensure_group_cols(df)
    metas = []

    for (pid, sub), grp in df.groupby(["product_id", "sub_product_name"]):
        ing_names   = _col(grp, "ingredient_name").dropna().tolist()
        ewg_scores  = _col(grp, "ewg").dropna().tolist()
        is_allergy  = _col(grp, "is_allergy").astype(str).str.lower().eq("true").any()
        min_ewg     = min((ewg_min(e) for e in ewg_scores), default=5)

        # 레퍼런스 매핑이 된 경우 safety_grade 활용
        if "safety_grade" in grp.columns:
            safety_grades = grp["safety_grade"].dropna().tolist()
            if safety_grades:
                min_ewg = min(int(g) for g in safety_grades)

        metas.append({
            "product_id"      : int(pid),
            "product_name"    : _val(grp.iloc[0], "product_name"),
            "brand_name"      : _val(grp.iloc[0], "brand_name"),
            "sub_name"        : str(sub),
            "category"        : _val(grp.iloc[0], "category"),
            "ingredients"     : ing_names,
            "has_allergy"     : bool(is_allergy),
            "min_ewg"         : min_ewg,
            "ingredient_count": len(ing_names),
        })

    return metas


# ── 청크 생성 ──────────────────────────────────────────────────
def build_chunks(df: pd.DataFrame) -> list[dict]:
    df = _ensure_group_cols(df)
    chunks = []

    # ① 제품 단위 청크 (제품 조회 Q&A용)
    for (pid, sub), grp in df.groupby(["product_id", "sub_product_name"]):
        product_name = _val(grp.iloc[0], "product_name")
        brand        = _val(grp.iloc[0], "brand_name")
        category     = _val(grp.iloc[0], "category")

        ing_lines = []
        for _, row in grp.iterrows():
            name    = _val(row, "ingredient_name")
            eng     = _val(row, "english")
            ewg     = ewg_label(_val(row, "ewg"))
            purp    = _val(row, "purpose")
            allergy = "⚠️알레르기유발" if _val(row, "is_allergy").lower() == "true" else ""

            # 레퍼런스 매핑 성분 설명 (있을 경우 추가)
            ai_desc = ""
            if "ref_ai_desc" in row.index and pd.notna(row["ref_ai_desc"]):
                ai_desc = f" | 설명: {str(row['ref_ai_desc'])[:80]}..."

            ing_lines.append(
                f"  - {name}"
                + (f" ({eng})" if eng else "")
                + f" | EWG:{ewg} | 기능:{purp} {allergy}{ai_desc}"
            )

        text = (
            f"[제품] {product_name}\n"
            f"브랜드: {brand}\n"
            f"카테고리: {category}\n"
            f"서브제품: {sub}\n"
            f"전성분 ({len(grp)}개):\n" + "\n".join(ing_lines)
        )

        chunks.append({
            "type"        : "product",
            "product_id"  : int(pid),
            "product_name": product_name,
            "sub_name"    : str(sub),
            "category"    : category,
            "text"        : text,
        })

    # ② 성분 단위 청크 (성분 추천 Q&A용)
    seen = set()
    for _, row in df.iterrows():
        name = _val(row, "ingredient_name").strip()
        if not name or name in seen:
            continue
        seen.add(name)

        eng        = _val(row, "english")
        ewg        = ewg_label(_val(row, "ewg"))
        purp       = _val(row, "purpose")
        allergy    = _val(row, "is_allergy")
        limitation = _val(row, "limitation", "해당 없음")
        forbidden  = _val(row, "forbidden",  "해당 없음")

        # ★ 레퍼런스 매핑 필드 (있을 경우에만 추가)
        ai_block = ""
        if "ref_ai_desc" in row.index and pd.notna(row.get("ref_ai_desc")):
            ai_block += f"\nAI 성분 설명: {row['ref_ai_desc']}"
        if "safety_label" in row.index and pd.notna(row.get("safety_label")):
            ai_block += f"\n안전 등급: [{row['safety_label']}] {row.get('safety_grade','')}등급"
        if "ref_data_grade" in row.index and pd.notna(row.get("ref_data_grade")):
            ai_block += f"\n데이터 신뢰도: {row['ref_data_grade']}"

        text = (
            f"[성분] {name}"
            + (f" ({eng})" if eng else "") + "\n"
            f"EWG 안전등급: {ewg}\n"
            f"기능: {purp}\n"
            f"알레르기 유발: {'있음' if allergy.lower() == 'true' else '없음'}\n"
            f"사용 제한: {limitation}\n"
            f"사용 금지: {forbidden}"
            + ai_block
        )

        chunks.append({
            "type"        : "ingredient",
            "product_id"  : None,
            "product_name": name,
            "sub_name"    : "",
            "category"    : "",
            "text"        : text,
        })

    return chunks


# ── CSV 자동 선택 ──────────────────────────────────────────────
def _load_data() -> pd.DataFrame:
    candidates = [
        DATA_DIR / "merged_ingredients.csv",   # 1순위: 매핑 완료 파일
        DATA_DIR / "hwahae_ingredients.csv",   # 2순위: 원본 크롤링 파일
    ]
    for path in candidates:
        if path.exists():
            print(f"📂 데이터 로드: {path.name}")
            df = pd.read_csv(path, encoding="utf-8-sig")
            print(f"   행 수: {len(df)}, 제품 수: {df['product_id'].nunique() if 'product_id' in df.columns else '?'}")

            has_ref = "ref_ai_desc" in df.columns
            matched = df["ref_ai_desc"].notna().sum() if has_ref else 0
            if has_ref:
                print(f"   성분 레퍼런스 매핑: {matched}/{len(df)} 행")
            return df

    raise FileNotFoundError(
        f"CSV 파일을 찾을 수 없습니다.\n"
        f"다음 중 하나를 data/ 폴더에 위치시켜 주세요:\n"
        f"  - merged_ingredients.csv  (권장: merger.py 실행 후)\n"
        f"  - hwahae_ingredients.csv  (기본 크롤링 결과)"
    )


# ── 메인 빌드 ──────────────────────────────────────────────────
def build_index():
    df     = _load_data()
    chunks = build_chunks(df)

    prod_n = sum(1 for c in chunks if c["type"] == "product")
    ing_n  = sum(1 for c in chunks if c["type"] == "ingredient")
    print(f"📄 청크 생성: 총 {len(chunks)}개 (제품:{prod_n}, 성분:{ing_n})")

    metas = build_products_meta(df)
    print(f"📋 제품 메타데이터: {len(metas)}개")

    print(f"🤖 임베딩 모델 로딩: {EMBED_MODEL}")
    model      = SentenceTransformer(EMBED_MODEL)
    texts      = [c["text"] for c in chunks]

    print("🔢 임베딩 생성 중...")
    embeddings = model.encode(
        texts, batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    VS_DIR.mkdir(exist_ok=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

    print(f"\n✅ index.faiss        → {INDEX_PATH}")
    print(f"✅ chunks.pkl         → {CHUNKS_PATH}")
    print(f"✅ products_meta.json → {META_PATH}")


if __name__ == "__main__":
    build_index()
