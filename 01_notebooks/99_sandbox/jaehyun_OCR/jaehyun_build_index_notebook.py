import os
import re
import pickle
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
COOS_FILE   = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "00_data", "00_raw", "coos_성분정보.xlsx"))
HWAHAE_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "00_data", "00_raw", "hwahae_all.csv"))
INDEX_FILE  = os.path.join(BASE_DIR, "cosmetic.index")
CHUNKS_FILE = os.path.join(BASE_DIR, "chunks.pkl")

print(f"coos 경로:   {COOS_FILE}")
print(f"hwahae 경로: {HWAHAE_FILE}")
print(f"coos 존재:   {os.path.exists(COOS_FILE)}")
print(f"hwahae 존재: {os.path.exists(HWAHAE_FILE)}")

# ─────────────────────────────────────────
# 청크 생성 함수
# ─────────────────────────────────────────
def make_coos_chunk(row):
    name  = str(row.get("성분명", "")).strip()
    inci  = str(row.get("INCI",   "")).strip()
    func  = str(row.get("기능",   "")).strip()
    desc  = str(row.get("AI설명", "")).strip()
    score = str(row.get("스코어", "")).strip()
    kind  = str(row.get("종류",   "")).strip()

    inci  = "정보없음" if inci  in ("nan", "", "-") else inci
    func  = "정보없음" if func  in ("nan", "", "-") else func
    desc  = "정보없음" if desc  in ("nan", "", "-", "COOS") else desc
    score = "정보없음" if score in ("nan", "", "-") else score
    kind  = "정보없음" if kind  in ("nan", "", "-") else kind

    if name in ("nan", "", "-") and desc == "정보없음":
        return ""

    return f"""[성분 정보]
성분명: {name}
INCI명: {inci}
종류: {kind}
기능: {func}
안전등급: {score}
설명: {desc}"""


def make_hwahae_chunk(row):
    product  = str(row.get("product_name", "")).strip()
    brand    = str(row.get("brand_name",   "")).strip()
    category = str(row.get("category",     "")).strip()
    korean   = str(row.get("korean",       "")).strip()   # 성분 한글명
    english  = str(row.get("english",      "")).strip()   # 성분 영문명
    ewg      = str(row.get("ewg",          "")).strip()   # EWG 등급
    purpose  = str(row.get("purpose",      "")).strip()   # 성분 용도
    positive = str(row.get("topics_positive", "")).strip()  # 긍정 리뷰 키워드
    negative = str(row.get("topics_negative", "")).strip()  # 부정 리뷰 키워드
    skin_good = str(row.get("skin_remark_good", "")).strip()
    skin_bad  = str(row.get("skin_remark_bad",  "")).strip()
    is_allergy = str(row.get("is_allergy", "")).strip()

    product  = "정보없음" if product  in ("nan", "") else product
    brand    = "정보없음" if brand    in ("nan", "") else brand
    category = "정보없음" if category in ("nan", "") else category
    korean   = "정보없음" if korean   in ("nan", "") else korean
    english  = "정보없음" if english  in ("nan", "") else english
    ewg      = "정보없음" if ewg      in ("nan", "") else ewg
    purpose  = "정보없음" if purpose  in ("nan", "") else purpose
    positive = "정보없음" if positive in ("nan", "") else positive
    negative = "정보없음" if negative in ("nan", "") else negative
    skin_good = "정보없음" if skin_good in ("nan", "") else skin_good
    skin_bad  = "정보없음" if skin_bad  in ("nan", "") else skin_bad
    allergy   = "알레르기 유발 성분" if is_allergy == "True" else "알레르기 없음"

    return f"""[화해 제품 정보]
제품명: {product}
브랜드: {brand}
카테고리: {category}
성분(한글): {korean}
성분(영문): {english}
EWG 등급: {ewg}
성분 용도: {purpose}
알레르기: {allergy}
피부 적합: {skin_good}
피부 비적합: {skin_bad}
긍정 리뷰: {positive}
부정 리뷰: {negative}"""


# ─────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────
if __name__ == "__main__":

    # 1. 데이터 로드
    print("\n" + "=" * 50)
    print("1단계: 데이터 로드")
    print("=" * 50)

    print("coos 데이터 로드 중...")
    df_coos = pd.read_excel(COOS_FILE)
    print(f"  → {len(df_coos)}행 로드 완료")

    print("화해 데이터 로드 중...")
    df_hwahae = pd.read_csv(
        HWAHAE_FILE,
        encoding="utf-8-sig",
        usecols=[
            "product_name", "brand_name", "category",
            "korean", "english", "ewg",
            "purpose", "is_allergy",
            "skin_remark_good", "skin_remark_bad",
            "topics_positive", "topics_negative",
        ]
    )
    print(f"  → {len(df_hwahae)}행 로드 완료")

    # 2. 청크 생성
    print("\n" + "=" * 50)
    print("2단계: 청크 생성")
    print("=" * 50)

    all_chunks = []

    # coos 전체
    for i, (_, row) in enumerate(df_coos.iterrows()):
        text = make_coos_chunk(row)
        if len(text) > 30:
            all_chunks.append(text)
        if (i + 1) % 3000 == 0:
            print(f"  coos 진행중: {i+1}/{len(df_coos)}")
    print(f"  → coos 청크 완료: {len(all_chunks)}개")

    # 화해 5000개 샘플
    df_hwahae_sample = df_hwahae.sample(n=min(5000, len(df_hwahae)), random_state=42)
    before = len(all_chunks)
    for _, row in df_hwahae_sample.iterrows():
        text = make_hwahae_chunk(row)
        if len(text) > 30:
            all_chunks.append(text)
    print(f"  → 화해 청크 완료: {len(all_chunks) - before}개")
    print(f"  → 전체 청크: {len(all_chunks)}개")

    sample_chunks = all_chunks[:2000]
    print(f"  → 임베딩 대상: {len(sample_chunks)}개")

    # 3. 임베딩
    print("\n" + "=" * 50)
    print("3단계: 임베딩 (10~20분 소요)")
    print("=" * 50)

    print("모델 로드 중...")
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    print("모델 로드 완료!")

    print("임베딩 시작...")
    embeddings = model.encode(
        sample_chunks,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype("float32")
    print(f"임베딩 완료! shape: {embeddings.shape}")

    # 4. FAISS 인덱스 저장
    print("\n" + "=" * 50)
    print("4단계: FAISS 인덱스 저장")
    print("=" * 50)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"인덱스 벡터 수: {index.ntotal}개")

    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(sample_chunks, f)

    print(f"\n저장 완료!")
    print(f"  → {INDEX_FILE}")
    print(f"  → {CHUNKS_FILE}")
    print("\n이제 streamlit run으로 메인 파일 실행하면 돼!")