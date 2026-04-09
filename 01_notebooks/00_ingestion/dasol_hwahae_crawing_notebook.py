from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urlparse
import pandas as pd
import time
import json
import random
import re
import os

BASE_URL = "https://www.hwahae.co.kr"

# =============================================
# 드라이버 설정
# =============================================
def create_driver(headless: bool = False):
    options = Options()
    mobile_emulation = {
        "deviceMetrics": {"width": 390, "height": 844, "pixelRatio": 3.0},
        "userAgent": (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) "
            "Version/16.0 Mobile/15E148 Safari/604.1"
        ),
    }
    options.add_experimental_option("mobileEmulation", mobile_emulation)
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.set_capability("goog:loggingPrefs", {"performance": "ALL"})
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )
    driver.set_window_size(390, 844)
    return driver


# =============================================
# URL 파싱
# =============================================
def parse_hwahae_url(full_url: str):
    parsed = urlparse(full_url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 3 and parts[0] == "goods":
        return parts[1], int(parts[2])
    raise ValueError(f"URL 파싱 실패: {full_url}")


# =============================================
# __NEXT_DATA__ 추출
# =============================================
def extract_next_data(driver) -> dict:
    try:
        script_el = driver.find_element(By.ID, "__NEXT_DATA__")
        return json.loads(script_el.get_attribute("innerHTML"))
    except Exception as e:
        print(f"  ⚠️  __NEXT_DATA__ 추출 실패: {e}")
        return {}


# =============================================
# [NEW] 카테고리 페이지에서 제품 URL 자동 수집
# =============================================
def collect_product_urls(driver, category_url: str, max_products: int = 50) -> list:
    print(f"\n[URL 수집] {category_url}")

    # ✅ 접속 전 랜덤 딜레이
    time.sleep(random.uniform(3, 6))
    driver.get(category_url)

    # ✅ 페이지 로딩 충분히 대기
    time.sleep(5)

    # ✅ 서버 오류 페이지 감지 후 재시도
    for retry in range(3):
        page_text = driver.find_element(By.TAG_NAME, "body").text
        if "서버와 연결하지 못했어요" in page_text or "연결" in page_text:
            wait = random.uniform(10, 20)
            print(f"  ⚠️ 서버 오류 감지 → {wait:.0f}초 대기 후 재시도 ({retry + 1}/3)")
            time.sleep(wait)
            driver.refresh()
            time.sleep(5)
        else:
            break
    else:
        print("  ❌ 3회 재시도 실패, 스킵")
        return []

    collected_urls = set()

    # 기존 __NEXT_DATA__ 추출
    data = extract_next_data(driver)
    if data:
        try:
            page_props = data.get("props", {}).get("pageProps", {})
            ranking_list = (
                    page_props.get("rankingGoodsData", []) or
                    page_props.get("goodsData", []) or
                    page_props.get("rankingData", []) or
                    []
            )
            for item in ranking_list:
                pid = item.get("id") or item.get("goods_id")
                slug = item.get("slug") or item.get("goods_name_slug")
                if pid and slug:
                    url = f"{BASE_URL}/goods/{slug}/{pid}?goods_tab=review_ingredients"
                    collected_urls.add(url)
        except Exception as e:
            print(f"  ⚠️ __NEXT_DATA__ 파싱 실패: {e}")

    # 스크롤 수집
    prev_height = 0
    no_change_count = 0

    for scroll_count in range(30):
        before = len(collected_urls)
        for a in driver.find_elements(By.TAG_NAME, "a"):
            href = a.get_attribute("href")
            if href and "/goods/" in href:
                clean = href.split("?")[0]
                collected_urls.add(clean + "?goods_tab=review_ingredients")

        new_found = len(collected_urls) - before
        print(f"  스크롤 {scroll_count + 1}회: 총 {len(collected_urls)}개 (신규 {new_found}개)")

        if len(collected_urls) >= max_products:
            break

        curr_height = driver.execute_script("return document.body.scrollHeight")
        if curr_height == prev_height:
            no_change_count += 1
            if no_change_count >= 3:
                print("  페이지 끝 도달")
                break
        else:
            no_change_count = 0
        prev_height = curr_height

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # ✅ 스크롤 간 랜덤 딜레이
        time.sleep(random.uniform(2, 4))

    result = list(collected_urls)[:max_products]
    print(f"  ✅ 최종 {len(result)}개 URL 수집 완료")
    return result

# =============================================
# 네트워크 로그에서 리뷰 API 응답 캡처
# =============================================
def capture_reviews_from_network(driver, product_id: int, max_reviews: int = 10) -> list[dict]:
    try:
        logs = driver.get_log("performance")
    except Exception:
        return []

    request_ids = {}
    for entry in logs:
        try:
            msg = json.loads(entry["message"])["message"]
            if msg.get("method") == "Network.requestWillBeSent":
                req_url = msg["params"].get("request", {}).get("url", "")
                req_id  = msg["params"].get("requestId", "")
                if "review" in req_url.lower() and str(product_id) in req_url:
                    request_ids[req_id] = req_url
                    # print(f"  📡 리뷰 API: {req_url}")
        except Exception:
            continue

    results = []
    for req_id in request_ids:
        try:
            body = driver.execute_cdp_cmd("Network.getResponseBody", {"requestId": req_id})
            data = json.loads(body.get("body", "{}"))
            reviews_raw = (
                data.get("data", {}).get("reviews")
                or data.get("data", {}).get("list")
                or data.get("reviews")
                or data.get("list")
                or (data.get("data") if isinstance(data.get("data"), list) else None)
                or []
            )
            for r in reviews_raw[:max_reviews]:
                results.append({
                    "product_id": product_id,
                    "rating"    : r.get("rating") or r.get("star"),
                    "content"   : r.get("content") or r.get("review_content"),
                    "skin_type" : r.get("skin_type") or r.get("skinType"),
                    "created_at": r.get("created_at") or r.get("createdAt"),
                })
        except Exception:
            continue

    return results


# =============================================
# 단일 제품 성분 추출
# =============================================
def get_product_ingredients(driver, product_id: int, slug: str) -> dict | None:
    url = f"{BASE_URL}/goods/{slug}/{product_id}?goods_tab=review_ingredients"
    print(f"\n[ID: {product_id}] 접속 중...")
    driver.get(url)

    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.ID, "__NEXT_DATA__"))
        )
    except Exception:
        print("  ⚠️  로딩 타임아웃 — 10초 쉬고 재시도")
        time.sleep(10)
        try:
            driver.get(url)
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.ID, "__NEXT_DATA__"))
            )
        except Exception:
            print("  ❌ 재시도 실패, 스킵")
            return None
    time.sleep(2)

    data = extract_next_data(driver)
    if not data:
        return None

    page_props = data.get("props", {}).get("pageProps", {})
    ingr_info  = page_props.get("productIngredientInfoData", {})
    # ── 제품 기본 정보 ──────────────────────────────────
    goods_pair = page_props.get("productGoodsPairData", {})
    common_data = goods_pair.get("common", {})
    product_data = goods_pair.get("product", {})
    goods_data = goods_pair.get("goods", {})

    # 제품명 / 브랜드 (기존 폴백 로직 대체)
    product_name = product_data.get("name")
    brand_name = product_data.get("brand", {}).get("name")

    # 평점 / 리뷰수 / 가격
    avg_ratings = common_data.get("avg_ratings")
    review_count = common_data.get("review_count")
    price = goods_data.get("price")
    consumer_price = goods_data.get("consumer_price")
    discount_rate = goods_data.get("discount_rate")
    capacity = goods_data.get("capacity")

    # 피부 속성
    product_attributes = product_data.get("product_attributes", [])
    primary_attr = ""
    sub_attrs = ""
    if product_attributes:
        pa = product_attributes[0]
        primary_attr = pa.get("primary_product_attribute", {}).get("name", "")
        sub_attrs = ", ".join(a.get("name", "") for a in pa.get("product_attributes", []))

    # 리뷰 토픽
    topics_positive = ", ".join(t.get("name", "") for t in page_props.get("productTopicsPositive", []))
    topics_negative = ", ".join(t.get("name", "") for t in page_props.get("productTopicsNegative", []))

    print(f"  📦 {product_name} | ⭐{avg_ratings} | 리뷰{review_count} | {price}원")

    # ── 제품명 추출 (4단계 폴백) ─────────────────────────────────
    # product_name, brand_name = None, None

    # 1단계: relationGoodsData에서 ID 매칭
    # for goods in page_props.get("relationGoodsData", []):
    #     if goods.get("id") == product_id:
    #         product_name = goods.get("name") or goods.get("goods_name")
    #         brand_name   = goods.get("brand_name")
    #         break
    #
    # # 2단계: goodsData 직접 접근
    # if not product_name:
    #     goods_data   = page_props.get("goodsData") or {}
    #     product_name = goods_data.get("name") or goods_data.get("goods_name")
    #     brand_name   = brand_name or goods_data.get("brand_name")
    #
    # # 3단계: productIngredientInfoData 안에서 탐색
    # if not product_name:
    #     product_name = (
    #         ingr_info.get("goods_name") or
    #         ingr_info.get("name") or
    #         ingr_info.get("product_name")
    #     )

    # 4단계: URL slug 디코딩 (최후 수단)
    if not product_name:
        from urllib.parse import unquote
        product_name = unquote(slug).replace("-", " ").strip()
        print(f"  ⚠️  제품명 없음 → slug 사용: {product_name}")

    print(f"  📦 제품명: {product_name}")

    # ── 성분 추출 수정 ────────────────────────────────────
    sub_products = ingr_info.get("sub_product_ingredients")
    sub_results = []

    # def parse_ingredients(ingredients_raw: list) -> list:
    #     result = []
    #     for ing in ingredients_raw:
    #         if ing.get("korean"):
    #             result.append({
    #                 # "korean": ing.get("korean", "").split(",")[0].strip(),
    #                 "korean": re.split(r',\s*(?![0-9])', ing.get("korean", ""))[0].strip(),
    #                 "english": ing.get("english", ""),
    #                 "ewg": ing.get("ewg", ""),
    #                 "is_allergy": ing.get("is_allergy", False),
    #                 "purpose": ing.get("purpose", ""),
    #                 "limitation": ing.get("limitation", ""),
    #                 "forbidden": ing.get("forbidden", ""),
    #             })
    #     return result
    def parse_ingredients(ingredients_raw: list) -> list:
        result = []
        for ing in ingredients_raw:
            if ing.get("korean"):
                result.append({
                    "ingredient_id": ing.get("id"),  # ✅ 추가
                    "korean": re.split(r',\s*(?![0-9])', ing.get("korean", ""))[0].strip(),
                    "english": ing.get("english", ""),
                    "ewg": ing.get("ewg", ""),
                    "ewg_data_availability_text": ing.get("ewg_data_availability_text", ""),  # ✅ 추가
                    "is_allergy": ing.get("is_allergy", False),
                    "skin_type": ing.get("skin_type", ""),  # ✅ 추가
                    "skin_remark_good": ing.get("skin_remark_good", ""),  # ✅ 추가
                    "skin_remark_bad": ing.get("skin_remark_bad", ""),  # ✅ 추가
                    "purpose": ing.get("purpose", ""),
                    "purposes": str(ing.get("purposes", [])),  # ✅ 추가
                    "limitation": ing.get("limitation", ""),
                    "forbidden": ing.get("forbidden", ""),
                    "concentration_info": ing.get("concentration_info", ""),  # ✅ 추가
                })
        return result

    if sub_products:
        for sub in sub_products:
            sub_name = sub.get("sub_product_name", "")
            ingredients_raw = sub.get("ingredients") or []
            parsed = parse_ingredients(ingredients_raw)
            sub_results.append({
                "sub_product_name": sub_name,
                "ingredients_detail": parsed,
                "ingredient_count": len(parsed),
            })
            print(f"  ✅ [{sub_name}] 성분 {len(parsed)}개 추출")
    else:
        ingredients_raw = ingr_info.get("ingredients") or []
        parsed = parse_ingredients(ingredients_raw)
        sub_results.append({
            "sub_product_name": "단일제품",
            "ingredients_detail": parsed,
            "ingredient_count": len(parsed),
        })
        print(f"  ✅ [단일제품] 성분 {len(parsed)}개 추출")

    return {
        "product_id": product_id,
        # "slug": slug,
        "product_name": product_name,
        "brand_name": brand_name,
        "avg_ratings": avg_ratings,  # ✅ 추가
        "review_count": review_count,  # ✅ 추가
        "price": price,  # ✅ 추가
        "consumer_price": consumer_price,  # ✅ 추가
        "discount_rate": discount_rate,  # ✅ 추가
        "capacity": capacity,  # ✅ 추가
        "primary_attr": primary_attr,  # ✅ 추가
        "sub_attrs": sub_attrs,  # ✅ 추가
        "topics_positive": topics_positive,  # ✅ 추가
        "topics_negative": topics_negative,  # ✅ 추가
        "sub_products": sub_results,
    }

# =============================================
# 결과 flat 변환
# =============================================
# def flatten_result(result: dict) -> list:
#     rows = []
#     for sub in result.get("sub_products", []):
#         for ing in sub.get("ingredients_detail", []):
#             rows.append({
#                 "product_id"      : result["product_id"],
#                 "product_name"    : result["product_name"],
#                 "brand_name"      : result["brand_name"],
#                 # "url"             : result["url"],
#                 "sub_product_name": sub["sub_product_name"],
#                 "ingredient_count": sub["ingredient_count"],
#                 "korean"          : ing["korean"],
#                 "english"         : ing["english"],
#                 "ewg"             : ing["ewg"],
#                 "is_allergy"      : ing["is_allergy"],
#                 "purpose"         : ing["purpose"],
#                 "limitation"      : ing["limitation"],
#                 "forbidden"       : ing["forbidden"],
#             })
#     return rows
def flatten_result(result: dict) -> list:
    rows = []
    for sub in result.get("sub_products", []):
        for ing in sub.get("ingredients_detail", []):
            rows.append({
                # 제품 레벨
                "product_id"                : result["product_id"],
                "product_name"              : result["product_name"],
                "brand_name"                : result["brand_name"],
                "avg_ratings"               : result["avg_ratings"],
                "review_count"              : result["review_count"],
                "price"                     : result["price"],
                "consumer_price"            : result["consumer_price"],
                "discount_rate"             : result["discount_rate"],
                "capacity"                  : result["capacity"],
                "primary_attr"              : result["primary_attr"],
                "sub_attrs"                 : result["sub_attrs"],
                "topics_positive"           : result["topics_positive"],
                "topics_negative"           : result["topics_negative"],
                "sub_product_name"          : sub["sub_product_name"],
                "ingredient_count"          : sub["ingredient_count"],
                # 성분 레벨
                "ingredient_id"             : ing["ingredient_id"],
                "korean"                    : ing["korean"],
                "english"                   : ing["english"],
                "ewg"                       : ing["ewg"],
                "ewg_data_availability_text": ing["ewg_data_availability_text"],
                "is_allergy"                : ing["is_allergy"],
                "skin_type"                 : ing["skin_type"],
                "skin_remark_good"          : ing["skin_remark_good"],
                "skin_remark_bad"           : ing["skin_remark_bad"],
                "purpose"                   : ing["purpose"],
                "purposes"                  : ing["purposes"],
                "limitation"                : ing["limitation"],
                "forbidden"                 : ing["forbidden"],
                "concentration_info"        : ing["concentration_info"],
            })
    return rows

# =============================================
# 여러 제품 수집 (URL 리스트 기반)
# =============================================
def crawl_multiple(url_list: list, driver=None, headless: bool = False, delay: float = 2.0) -> list:
    """
    Args:
        url_list: 화해 제품 URL 리스트 (실제 URL이어야 함)
        driver  : 기존 드라이버 전달 가능 (없으면 새로 생성)
        headless: 창 없이 실행 여부
        delay   : 요청 간격(초)
    """
    own_driver = driver is None
    if own_driver:
        driver = create_driver(headless=headless)

    all_rows = []
    try:
        for i, url in enumerate(url_list):
            print(f"\n{'='*50}")
            print(f"[{i+1}/{len(url_list)}] {url[:70]}...")
            try:
                slug, pid = parse_hwahae_url(url)
                result = get_product_ingredients(driver, pid, slug)
                if result:
                    all_rows.extend(flatten_result(result))
            except Exception as e:
                print(f"  ❌ 오류: {e}")
            time.sleep(delay)
    finally:
        if own_driver:
            driver.quit()

    return all_rows

# =============================================
# 카테고리
# =============================================
def collect_all_categories(driver) -> list[dict]:
    driver.get("https://www.hwahae.co.kr/rankings?english_name=category")
    time.sleep(4)

    data       = extract_next_data(driver)
    page_props = data.get("props", {}).get("pageProps", {})
    root       = page_props.get("rankingsCategories", {})

    categories = []

    def traverse(node):
        if isinstance(node, dict):
            # ✅ depth=3 노드에서 depth=4 '전체' 자식 ID를 URL에 사용
            if node.get("depth") == 3:
                name = node.get("name", "").strip()
                if name and name not in {"전체", "카테고리 전체"}:
                    children = node.get("children", [])
                    # 첫 번째 자식 중 '전체' 항목의 id 찾기
                    all_child = next(
                        (c for c in children if c.get("name", "").strip() == "전체"),
                        None
                    )
                    if all_child:
                        theme_id = all_child.get("id")
                        categories.append({
                            "name"    : name,
                            "theme_id": theme_id,
                            "url"     : f"{BASE_URL}/rankings?english_name=category&theme_id={theme_id}"
                        })
            for child in node.get("children", []):
                traverse(child)

    traverse(root)

    print(f"✅ 카테고리 {len(categories)}개 수집")
    for c in categories:
        print(f"  - [{c['theme_id']}] {c['name']}")
    return categories

# =============================================
# 리뷰 추출
# =============================================
def get_product_reviews(driver, product_id: int, slug: str, max_reviews: int = 3):
    url = f"{BASE_URL}/goods/{slug}/{product_id}?goods_tab=reviews"
    try:
        driver.get(url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.ID, "__NEXT_DATA__"))
        )
        time.sleep(3)
    except Exception as e:
        print(f"  ❌ 리뷰 페이지 로딩 실패: {e}")
        return []

    # 스크롤해서 리뷰 카드 렌더링 유도
    driver.execute_script("window.scrollTo(0, 600);")
    time.sleep(2)

    results = []
    try:
        # 리뷰 카드: data-testid, class 등 다양한 패턴 시도
        review_cards = driver.find_elements(By.CSS_SELECTOR, "[data-testid='review-item']")

        # 못 찾으면 article 태그 시도
        if not review_cards:
            review_cards = driver.find_elements(By.CSS_SELECTOR, "article")

        # 그래도 없으면 리뷰 텍스트가 담긴 li 시도
        if not review_cards:
            review_cards = driver.find_elements(By.CSS_SELECTOR, "ul > li")

        for card in review_cards[:max_reviews]:
            text = card.text.strip()
            if not text or len(text) < 10:
                continue

            # 별점: 텍스트에서 숫자 추출 (예: "4.5" or "5")
            rating_match = re.search(r'(\d+\.\d+|\d+)\s*점', text)
            rating = rating_match.group(1) if rating_match else None

            # 피부타입: 건성|지성|복합성|중성|민감성
            skin_match = re.search(r'(건성|지성|복합성|중성|민감성)', text)
            skin_type = skin_match.group(1) if skin_match else None

            results.append({
                "product_id": product_id,
                "rating": rating,
                "content": text[:500],
                "skin_type": skin_type,
                "created_at": None,
            })

        if results:
            print(f"  ✅ 리뷰 {len(results)}개 (DOM)")
        else:
            print(f"  ⚠️ 리뷰 DOM 추출 실패 — 셀렉터 미일치")

    except Exception as e:
        print(f"  ❌ 리뷰 DOM 실패: {e}")

    return results

# =============================================
# 결과 저장
# =============================================
def sanitize_filename(name: str) -> str:
    """파일명에 사용 불가한 문자를 언더스코어로 치환"""
    return re.sub(r'[\\/:*?"<>|]', '_', name)

def save_results(rows: list, filename: str = "hwahae_ingredients.csv") -> pd.DataFrame | None:
    if not rows:
        print("저장할 데이터 없음")
        return None
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"\n✅ 총 {len(df)}행 저장 완료: {filename}")
    print(df[["product_name", "sub_product_name", "ingredient_count"]].to_string())
    return df

# =============================================
# 실행
# =============================================
EXCLUDE_NAMES = {
    # 성분 분석 의미 없음
    "물티슈",
    "기타",
    "기타 식품",
    "네일컬러",  # 성분 데이터 부실할 가능성
    "헤어컬러링",  # 염색약은 성분 특수
    "리빙퍼퓸",  # 화장품 아님
	"젤",
	"속눈썹영양제",
	"남성향수",
	"여성향수",
	"네일리무버",
	"네일케어",
	"트리트먼트/팩",
	"두피 스케일러",
	"스타일링",
	"헤어미스트",
	"헤어에센스/오일",
	"여성청결제",
	"입욕제",
	"데오드란트",
	"헤어",
	"스킨케어",
	"선케어",
	"메이크업",
	"바디",
	"바디워시",
	"바디로션",
	"바디크림/젤",
	"바디오일/에센스",
	"바디스크럽",
	"바디미스트/샤워코롱",
	"핸드크림/밤",
	"풋케어",
	"바디기타",
	"바디케어",
	"핸드워시",
	"핸드케어",
	"샴푸",
	"린스/컨디셔너",
	"밤/멀티밤",

    # 이너뷰티 (먹는 제품 - 화장품 아님)
    "피부 건강",
    "모발/손톱 건강",
    "소화/위장 건강",
    "체지방 관리",
    "근육량 증가",
    "항산화 관리",
    "면역/피로 관리",
    "종합 건강",
    "눈 건강",
    "뼈/관절/치아 건강",
    "혈행 개선",

    # 도구류 (성분 없음)
    "스킨케어 소품",
    "메이크업 소품",
    "헤어 소품",
    "기타 소품",
    "스킨케어 디바이스",
    "클렌징 디바이스",
}

# =============================================
# 실행
# =============================================
if __name__ == "__main__":
    driver = create_driver(headless=False)
    all_rows = []

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CAT_DIR = os.path.join(BASE_DIR, "categories")
    os.makedirs(CAT_DIR, exist_ok=True)

    try:
        # ── 기존 완료된 카테고리 CSV 전부 미리 로드 ──────────────────
        existing_files = [f for f in os.listdir(CAT_DIR)
                          if f.startswith("hwahae_") and f.endswith(".csv") and f != "hwahae_list.csv"]
        for fname in existing_files:
            try:
                df = pd.read_csv(os.path.join(CAT_DIR, fname), encoding="utf-8-sig")
                all_rows.extend(df.to_dict("records"))
            except Exception as e:
                print(f"  ⚠️ 기존 파일 로드 실패 ({fname}): {e}")
        print(f"✅ 기존 완료 카테고리 {len(existing_files)}개 로드 ({len(all_rows)}행)")

        categories = collect_all_categories(driver)
        categories = [c for c in categories if c["name"] not in EXCLUDE_NAMES]
        print(f"\n▶ 크롤링 대상: {len(categories)}개 카테고리")

        for cat_idx, cat in enumerate(categories):
            safe_name = sanitize_filename(cat["name"])
            cat_file = os.path.join(CAT_DIR, f"hwahae_{safe_name}.csv")

            if os.path.exists(cat_file):
                print(f"  ⏭️ 이미 완료된 카테고리 스킵: {cat['name']}")
                continue

            print(f"\n{'='*60}")
            print(f"[{cat_idx+1}/{len(categories)}] 📂 {cat['name']} (id={cat['theme_id']})")

            url_list = collect_product_urls(driver, cat["url"], max_products=50)
            if not url_list:
                print("  ⚠️ 제품 없음, 스킵")
                continue

            cat_rows = []
            review_rows = []
            for i, url in enumerate(url_list):
                print(f"  [{i+1}/{len(url_list)}] {url[:65]}...")
                try:
                    slug, pid = parse_hwahae_url(url)
                    result = get_product_ingredients(driver, pid, slug)
                    if result:
                        rows = flatten_result(result)
                        for row in rows:
                            row["category"] = cat["name"]
                        cat_rows.extend(rows)
                        all_rows.extend(rows)
                        print(f"  → 누적 {len(all_rows)}행")
                        # 성분 페이지 로드 중 발생한 리뷰 API 캡처
                        reviews = capture_reviews_from_network(driver, pid)
                        if reviews:
                            review_rows.extend(reviews)
                            print(f"  💬 리뷰 {len(reviews)}개 캡처")
                except Exception as e:
                    print(f"  ❌ 오류: {e} → 5초 후 재시도")
                    time.sleep(5)
                    try:
                        slug, pid = parse_hwahae_url(url)
                        result = get_product_ingredients(driver, pid, slug)
                        if result:
                            rows = flatten_result(result)
                            for row in rows:
                                row["category"] = cat["name"]
                            cat_rows.extend(rows)
                            all_rows.extend(rows)
                            reviews = capture_reviews_from_network(driver, pid)
                            if reviews:
                                review_rows.extend(reviews)
                    except Exception as e2:
                        print(f"  ❌ 재시도 실패: {e2}")
                time.sleep(3)

            save_results(cat_rows, cat_file)
            save_results(all_rows, os.path.join(BASE_DIR, "hwahae_all.csv"))

            # 리뷰 저장 (누적)
            if review_rows:
                rev_path = os.path.join(BASE_DIR, "reviews.csv")
                if os.path.exists(rev_path):
                    existing_rev = pd.read_csv(rev_path, encoding="utf-8-sig")
                    review_rows = existing_rev.to_dict("records") + review_rows
                pd.DataFrame(review_rows).drop_duplicates().to_csv(
                    rev_path, index=False, encoding="utf-8-sig"
                )
                print(f"  💾 리뷰 저장: {len(review_rows)}개")

            print(f"  ⏳ 다음 카테고리까지 10초 대기...")
            time.sleep(10)

    finally:
        driver.quit()

    save_results(all_rows, os.path.join(BASE_DIR, "hwahae_all.csv"))
    print(f"\n🎉 전체 크롤링 완료! 총 {len(all_rows)}행")


# =============================================
# 임시 테스트 실행 (카테고리 1개, 제품 3개)
# python hwahae_crawing_f_f.py test
# =============================================
# def run_test():
#     import sys
#     driver = create_driver(headless=False)
#     all_rows = []
#
#     try:
#         categories = collect_all_categories(driver)
#         categories = [c for c in categories if c["name"] not in EXCLUDE_NAMES]
#
#         test_cat = random.choice(categories)
#         print(f"\n▶ 테스트 카테고리: {test_cat['name']} (id={test_cat['theme_id']})")
#
#         url_list = collect_product_urls(driver, test_cat["url"], max_products=3)
#         if not url_list:
#             print("  ⚠️ 제품 없음")
#             return
#
#         for i, url in enumerate(url_list):
#             print(f"\n  [{i+1}/{len(url_list)}] {url[:65]}...")
#             try:
#                 slug, pid = parse_hwahae_url(url)
#                 result = get_product_ingredients(driver, pid, slug)
#                 if result:
#                     rows = flatten_result(result)
#                     for row in rows:
#                         row["category"] = test_cat["name"]
#                     all_rows.extend(rows)
#                     reviews = capture_reviews_from_network(driver, pid)
#                     if reviews:
#                         print(f"  💬 리뷰 {len(reviews)}개 캡처")
#             except Exception as e:
#                 print(f"  ❌ 오류: {e}")
#             time.sleep(3)
#
#     finally:
#         driver.quit()
#
#     save_results(all_rows, "hwahae_test.csv")
#     print(f"\n테스트 완료! 총 {len(all_rows)}행 → hwahae_test.csv")
#
#
# if __name__ == "__test__" or (len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "test"):
#     run_test()