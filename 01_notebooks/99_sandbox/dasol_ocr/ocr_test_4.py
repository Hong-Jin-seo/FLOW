import os

# PaddlePaddle 3.x + Windows oneDNN/PIR 호환성 버그 방지
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_DISABLE_MKLDNN"] = "1"
os.environ["FLAGS_enable_pir_in_executor"] = "0"   # PIR 실행 엔진 비활성화 (oneDNN 충돌 방지)

import re
import numpy as np
import cv2
import pandas as pd
from dotenv import load_dotenv
from paddleocr import PaddleOCR
from rapidfuzz import process, fuzz

load_dotenv()

_HERE = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────
# 1. coos_ewg_cleaned.csv 성분 매칭
# ─────────────────────────────────────────
_CSV_PATH = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "00_data", "02_processed", "coos_ewg_cleaned.csv"))
_df = pd.read_csv(_CSV_PATH, encoding="utf-8-sig")

ko_map   = {
    row["ingredient"]: {"ingredient_ko": row["ingredient"], "coos_score": int(row["coos_score"])}
    for _, row in _df.iterrows()
}
ko_names = list(ko_map.keys())

# ─────────────────────────────────────────
# 2. OCR 초기화 (PaddleOCR) — 지연 초기화
# ─────────────────────────────────────────
_ocr = None

def _get_ocr() -> PaddleOCR:
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang='korean',
        )
    return _ocr

# ─────────────────────────────────────────
# 이미지 전처리
# ─────────────────────────────────────────
def preprocess_image(img: np.ndarray) -> np.ndarray:
    """빛반사 제거 + 대비 향상"""
    # 1. 글레어(빛반사) 마스크 생성 후 인페인팅
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, glare_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(img_bgr, glare_mask, 3, cv2.INPAINT_TELEA)

    # 2. CLAHE로 명암 대비 향상 (어두운 영역 글씨도 잘 읽힘)
    lab = cv2.cvtColor(inpainted, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 3. 선명화 (엣지 강조)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    result = cv2.filter2D(result, -1, kernel)

    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# ─────────────────────────────────────────
# 3. 전성분 섹션 추출
# ─────────────────────────────────────────
def extract_ingredient_section(raw_text: str) -> str:
    # 1차: 전성분 헤더 (한글 복합어 안의 '성분' 오매칭 방지)
    start_match = re.search(r'전성분\s*[:｜|]?', raw_text)

    # 2차 fallback: '·성분:' 또는 '성분:' (앞에 한글이 없는 경우만)
    if not start_match:
        start_match = re.search(r'(?<![가-힣])성분\s*[:｜|]', raw_text)

    # 3차 fallback: [용량] NNNml 다음에 성분이 나오는 경우
    if not start_match:
        start_match = re.search(r'용량[^\d]*\d+\s*m[Ll]', raw_text)

    if not start_match:
        return raw_text

    text = raw_text[start_match.end():]
    end_match = re.search(r'(주의사|사용시의|사용방|용랑|용량|보관|제조|고객)', text)
    if end_match:
        text = text[:end_match.start()]
    return text.strip()


# ─────────────────────────────────────────
# 4. 성분명 파싱
# ─────────────────────────────────────────
NOISE_WORDS = {'추출등', '예렉', '뼈일', '스', '비'}


def parse_ingredients(raw_text: str) -> list:
    # ; / · 를 , 로 통일 (·은 한국 라벨 bullet, . 은 제외 - FL. OZ 등 오분리 방지)
    text = re.sub(r'[;；/·|｜]', ',', raw_text)
    text = re.sub(r'\n', ',', text)
    # 한글-한글 사이 마침표를 콤마로 변환 (FL.OZ 등 영문은 제외)
    text = re.sub(r'(?<=[가-힣])\.(?=[가-힣])', ',', text)

    seen        = set()
    ingredients = []
    for chunk in text.split(','):
        # OCR 개행 아티팩트로 삽입된 공백 제거 → 성분명 복원
        # 예) '귀리커널 추출물' → '귀리커널추출물'
        #     '수퍼옥사 이드디스뮤타아제' → '수퍼옥사이드디스뮤타아제'
        name = re.sub(r'\s+', '', chunk)

        if not name:
            continue
        if ':' in name:                        # 섹션 레이블 제거 (성분:, 용량: 등)
            continue
        if not re.search(r'[가-힣]', name):   # 한글 포함 필수
            continue
        if len(name) < 2 or len(name) > 40:
            continue
        if re.search(r'[\[\]{}\'\"@#]', name):
            continue
        if re.search(r'\d{3,}', name):        # 3자리 이상 숫자만 제거 (세테아레스-20 허용)
            continue
        if name in NOISE_WORDS:
            continue
        if name in seen:
            continue

        seen.add(name)
        ingredients.append(name)

    return ingredients


# ─────────────────────────────────────────
# 5. 성분 매핑
# ─────────────────────────────────────────
def get_ewg_score(item: dict):
    try:
        score = item.get("coos_score")
        if score is None:
            return None
        return int(float(score))
    except (ValueError, TypeError):
        return None


def find_ingredient(ocr_name: str) -> dict | None:
    # ── 1차: coos_ewg_cleaned.csv 정확/퍼지 매칭 ─────────────────
    if ocr_name in ko_map:
        return ko_map[ocr_name]

    hit = process.extractOne(ocr_name, ko_names, scorer=fuzz.token_sort_ratio, score_cutoff=80)
    if hit:
        return ko_map[hit[0]]

    # ── 2차: FAISS 의미 기반 fallback (비활성화) ─────────────────
    # docs = vectorstore.similarity_search(ocr_name, k=6)
    # if not docs:
    #     return None
    # best_ing = docs[0].metadata.get("ingredient_ko")
    # result   = dict(docs[0].metadata)
    # for doc in docs:
    #     if doc.metadata.get("ingredient_ko") != best_ing:
    #         continue
    #     ct = doc.metadata.get("chunk_type")
    #     pc = doc.page_content
    #     if ct == "basic_info" and "coos_function" not in result:
    #         m = re.search(r'기능: ([^/|]+)', pc)
    #         if m:
    #             result["coos_function"] = m.group(1).strip()
    #     elif ct == "expert" and "coos_ai_description" not in result:
    #         m = re.search(r'AI설명: ([^|]+)', pc)
    #         if m:
    #             result["coos_ai_description"] = m.group(1).strip()
    # return result

    return None


# ─────────────────────────────────────────
# 6. PaddleOCR 텍스트 추출
# ─────────────────────────────────────────
def extract_text_with_paddle(image_input) -> str:
    """image_input: 파일 경로(str) 또는 업로드된 이미지 bytes"""
    import io
    import tempfile
    import numpy as np
    from PIL import Image

    MAX_SIDE = 1500

    if isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    else:
        img = Image.fromarray(image_input).convert("RGB")

    w, h = img.size
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # numpy 배열 대신 임시 파일 경로로 전달 (Paddle 내부 메모리 오류 방지)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        img.save(tmp_path, "JPEG")

    try:
        result = _get_ocr().predict(input=tmp_path)
    finally:
        import os
        os.unlink(tmp_path)

    texts = []
    for res in result:
        rec_texts = res.get("rec_texts", [])
        texts.extend(rec_texts)

    return " ".join(texts)


# ─────────────────────────────────────────
# 7. 전체 실행
# ─────────────────────────────────────────
def analyze_image(image_input) -> list:
    """image_input: 파일 경로(str) 또는 업로드된 이미지 bytes"""
    raw_text = extract_text_with_paddle(image_input)
    print("OCR 원본:", raw_text)

    section = extract_ingredient_section(raw_text)
    print("성분 섹션:", section)

    ingredients = parse_ingredients(section)
    print("파싱된 성분:", ingredients)

    results = []
    for ingredient in ingredients:
        item = find_ingredient(ingredient)
        if item is None:          # CSV에 없는 성분은 제외
            continue
        ewg = get_ewg_score(item)
        if not ewg:               # EWG 0 또는 None은 제외
            continue
        results.append({
            "ingredient": ingredient,
            "item":        item,
            "ewg":         ewg,
            "function":    item.get("coos_function")       if item else None,
            "description": item.get("coos_ai_description") if item else None,
        })

    results.sort(key=lambda x: x["ewg"] or 0, reverse=True)
    return results


# ─────────────────────────────────────────
# 실행
# ─────────────────────────────────────────
if __name__ == "__main__":
    results = analyze_image("test_label.jpg")  # ← 이미지 경로

    print("\n=== 분석 결과 ===")
    for r in results:
        ewg  = r["ewg"]
        item = r["item"]

        if ewg:
            print(f"✅ {r['ingredient']} → EWG: {ewg}")
        elif item:
            print(f"⚠️  {r['ingredient']} → 매핑 성공 / EWG 데이터 없음")
        else:
            print(f"❌ {r['ingredient']} → 매핑 실패")
