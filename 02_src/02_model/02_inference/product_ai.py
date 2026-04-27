"""
product_ai.py — OpenAI Function Calling 기반 제품 추천 AI

처리 흐름:
  사용자 자연어 입력 → GPT가 recommend_products 도구 파라미터 결정
  → product_db.csv 필터링 → 결과 포맷 → GPT 최종 응답 생성

외부 데이터:
  00_data/02_processed/product_db.csv       - 제품별 성분·가격·평점 정보
  00_data/02_processed/ingredient_merged2.json - 성분별 EWG 점수 (coos_score / hw_ewg)
"""

import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_HERE = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────
_DB_PATH   = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "00_data", "02_processed", "product_db.csv"))
_JSON_PATH = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "00_data", "02_processed", "ingredient_merged2.json"))

_df = pd.read_csv(_DB_PATH, encoding="utf-8-sig")

with open(_JSON_PATH, "r", encoding="utf-8") as f:
    _json_data = json.load(f)

_ewg_map: dict[str, int] = {}
for item in _json_data:
    ko = item.get("ingredient_ko")
    if not ko:
        continue
    score = None
    try:
        v = float(item.get("coos_score") or 0)
        if v > 0:
            score = int(v)
    except (ValueError, TypeError):
        pass
    if score is None:
        try:
            # coos_score 없을 때 화해 EWG 점수를 fallback으로 사용
            v = float(item.get("hw_ewg") or 0)
            if v > 0:
                score = int(v)
        except (ValueError, TypeError):
            pass
    if score is not None:
        _ewg_map[ko] = score


def _build_product_table() -> pd.DataFrame:
    """ingredient-level rows → product-level DataFrame"""
    df = _df.copy()
    df["ewg_score"] = df["ingredient_ko"].map(_ewg_map)

    agg = (
        df.groupby("hw_product_id")
        .agg(
            hw_product_name   = ("hw_product_name",   "first"),
            hw_brand_name     = ("hw_brand_name",     "first"),
            hw_primary_attr   = ("hw_primary_attr",   "first"),
            hw_price          = ("hw_price",          "first"),
            hw_consumer_price = ("hw_consumer_price", "first"),
            hw_avg_ratings    = ("hw_avg_ratings",    "first"),
            hw_review_count   = ("hw_review_count",   "first"),
            hw_ingredient_count = ("hw_ingredient_count", "first"),
            hw_topics_positive = ("hw_topics_positive", "first"),
            hw_topics_negative = ("hw_topics_negative", "first"),
            ingredients       = ("ingredient_ko",     list),
            ewg_scores        = ("ewg_score",         list),
        )
        .reset_index()
    )

    agg["danger_count"]  = agg["ewg_scores"].apply(lambda xs: sum(1 for x in xs if x and x >= 7))
    agg["caution_count"] = agg["ewg_scores"].apply(lambda xs: sum(1 for x in xs if x and 3 <= x <= 6))
    valid_ewg            = agg["ewg_scores"].apply(lambda xs: [x for x in xs if x and x > 0])
    agg["avg_ewg"]       = valid_ewg.apply(lambda xs: sum(xs) / len(xs) if xs else 0.0)

    return agg


_products = _build_product_table()


# ─────────────────────────────────────────
# 추천 함수
# ─────────────────────────────────────────
def recommend_products(
    query: str | None        = None,
    category: str | None     = None,
    max_price: float | None  = None,
    min_rating: float        = 0.0,
    exclude_danger: bool     = True,
    top_n: int               = 5,
) -> pd.DataFrame:
    """
    제품 추천.

    Parameters
    ----------
    query          : 제품명/브랜드명 검색 키워드 (부분 일치)
    category       : hw_primary_attr 카테고리 필터 (예: "토너", "클렌징")
    max_price      : 최대 소비자가 (원)
    min_rating     : 최소 평점 (기본 0.0)
    exclude_danger : True 이면 danger_count > 0 제품 제외
    top_n          : 반환 개수

    Returns
    -------
    상위 제품 DataFrame (avg_ewg 오름차순, 평점 내림차순, 리뷰수 내림차순)
    """
    df = _products.copy()

    if exclude_danger:
        df = df[df["danger_count"] == 0]

    if query:
        q = query.lower()
        mask = (
            df["hw_product_name"].str.lower().str.contains(q, na=False) |
            df["hw_brand_name"].str.lower().str.contains(q, na=False)
        )
        df = df[mask]

    if category:
        df = df[df["hw_primary_attr"].str.contains(category, na=False)]

    if max_price is not None:
        df = df[df["hw_consumer_price"] <= max_price]

    if min_rating > 0:
        df = df[df["hw_avg_ratings"] >= min_rating]

    # avg_ewg 오름차순(낮을수록 안전) → 평점·리뷰수 내림차순(인기 우선)
    df = df.sort_values(
        ["avg_ewg", "hw_avg_ratings", "hw_review_count"],
        ascending=[True, False, False],
    ).head(top_n)

    return df[[
        "hw_product_id", "hw_product_name", "hw_brand_name",
        "hw_primary_attr", "hw_consumer_price",
        "hw_avg_ratings", "hw_review_count",
        "avg_ewg", "danger_count", "caution_count",
        "hw_topics_positive", "hw_topics_negative",
    ]]


def recommend_from_ocr(
    ocr_results: list,
    max_price: float | None = None,
    min_rating: float       = 0.0,
    exclude_danger: bool    = True,
    top_n: int              = 10,
) -> pd.DataFrame:
    """
    ocr_test_4.py의 analyze_image() 결과를 받아
    해당 성분들을 포함하지 않는 안전한 대체 제품을 추천.

    ocr_results : [{"ingredient": str, "ewg": int|None, ...}, ...]
    """
    danger_ingredients = [
        r["ingredient"] for r in ocr_results
        if r.get("ewg") and r["ewg"] >= 7
    ]
    caution_ingredients = [
        r["ingredient"] for r in ocr_results
        if r.get("ewg") and 3 <= r["ewg"] <= 6
    ]

    print(f"[OCR 분석] 위험 성분: {danger_ingredients}")
    print(f"[OCR 분석] 주의 성분: {caution_ingredients}")

    df = _products.copy()

    if exclude_danger and danger_ingredients:
        mask = df["ingredients"].apply(
            lambda ings: not any(d in ings for d in danger_ingredients)
        )
        df = df[mask]

    if max_price is not None:
        df = df[df["hw_consumer_price"] <= max_price]

    if min_rating > 0:
        df = df[df["hw_avg_ratings"] >= min_rating]

    df = df.sort_values(
        ["avg_ewg", "hw_avg_ratings", "hw_review_count"],
        ascending=[True, False, False],
    ).head(top_n)

    return df[[
        "hw_product_id", "hw_product_name", "hw_brand_name",
        "hw_primary_attr", "hw_consumer_price",
        "hw_avg_ratings", "hw_review_count",
        "avg_ewg", "danger_count", "caution_count",
        "hw_topics_positive", "hw_topics_negative",
    ]]


# ─────────────────────────────────────────
# LLM 도구 정의 (OpenAI function calling)
# ─────────────────────────────────────────
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recommend_products",
            "description": (
                "화장품 제품을 추천합니다. "
                "사용자가 카테고리, 가격, 평점, 성분 안전성 조건을 말하면 이 도구를 호출하세요."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "제품명·브랜드명·제품 타입 키워드 (부분 일치). "
                            "세럼, 토너, 로션, 스킨, 에센스, 클렌징, 선크림 등 제품 타입을 찾을 때는 "
                            "category가 아닌 이 파라미터를 사용하세요."
                        ),
                    },
                    "category": {
                        "type": "string",
                        "description": (
                            "피부 기능·특성 라벨 필터. 사용 가능한 값: "
                            "수분, 보습, 진정, 모공, 브라이트닝, 안티에이징, 트러블, 각질, 톤업, "
                            "노세범, 오일, 모이스처, 리퀴드, 크림, 새틴, 매트, 워터프루프, 볼륨, 컬링 등. "
                            "세럼·토너·로션 같은 제품 타입에는 사용하지 마세요 — query를 쓰세요."
                        ),
                    },
                    "max_price": {
                        "type": "number",
                        "description": "최대 가격 (원 단위, 예: 30000)",
                    },
                    "min_rating": {
                        "type": "number",
                        "description": "최소 평점 (0.0 ~ 5.0)",
                    },
                    "exclude_danger": {
                        "type": "boolean",
                        "description": "위험 성분(EWG 3등급) 포함 제품 제외 여부 (기본 true)",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "추천 개수 (기본 5)",
                    },
                },
                "required": [],
            },
        },
    }
]

_SYSTEM_PROMPT = """당신은 화장품 성분 안전성 전문 AI 큐레이터입니다.
사용자의 요청을 분석해 recommend_products 도구를 호출하고, 결과를 친절하고 간결하게 설명해주세요.

### 파라미터 선택 규칙 (중요)
- query: 제품 타입(세럼, 토너, 로션, 에센스, 클렌징, 선크림 등) 또는 브랜드명/제품명 검색
- category: 피부 기능 라벨(수분, 보습, 진정, 모공, 브라이트닝, 안티에이징, 트러블, 각질, 톤업 등)
- 세럼·토너·로션 같은 제품 타입 → query 사용, category에 넣지 말 것
- 민감성·수분·진정 같은 피부 특성 → category 사용

### EWG 안전 등급
- 1~2등급(안전): 피부 자극·독성 우려 없음
- 3~6등급(주의): 일부 민감성 피부 주의
- 7~10등급(위험): 독성·알레르기 위험 성분

결과를 설명할 때는 브랜드명, 제품명, 가격, 평점, 위험 성분 여부를 포함해주세요.
위험 성분이 없는 제품(✅)을 우선적으로 추천해주세요."""


def _format_results(df: pd.DataFrame) -> str:
    if df.empty:
        return "조건에 맞는 제품을 찾지 못했습니다."
    lines = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        price = f"{int(row['hw_consumer_price']):,}원" if pd.notna(row['hw_consumer_price']) else "가격 미정"
        safety = "✅ 위험 성분 없음" if int(row['danger_count']) == 0 else f"⚠️ 위험 성분 {int(row['danger_count'])}개"
        caution = f"주의 성분 {int(row['caution_count'])}개" if int(row['caution_count']) > 0 else "주의 성분 없음"
        lines.append(
            f"{i}. [{row['hw_brand_name']}] {row['hw_product_name']}\n"
            f"   카테고리: {row['hw_primary_attr']} | 가격: {price} | 평점: {row['hw_avg_ratings']}\n"
            f"   안전: {safety} | {caution}\n"
            f"   긍정 키워드: {row['hw_topics_positive']}"
        )
    return "\n\n".join(lines)


# ─────────────────────────────────────────
# 질의응답 채팅
# ─────────────────────────────────────────
def chat(user_input: str, history: list | None = None) -> tuple[str, list]:
    """
    단일 턴 질의응답.

    Parameters
    ----------
    user_input : 사용자 자연어 입력
    history    : 이전 대화 메시지 목록 (멀티턴 지원)

    Returns
    -------
    (assistant 응답 텍스트, 업데이트된 history)
    """
    client   = OpenAI()
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=_TOOLS,
        tool_choice="auto",
    )

    msg = response.choices[0].message

    if msg.tool_calls:
        tool_call  = msg.tool_calls[0]
        args       = json.loads(tool_call.function.arguments)
        result_df  = recommend_products(**args)
        result_str = _format_results(result_df)

        messages.append(msg.model_dump())
        messages.append({
            "role":         "tool",
            "tool_call_id": tool_call.id,
            "content":      result_str,
        })

        final = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        answer = final.choices[0].message.content
        messages.append({"role": "assistant", "content": answer})
    else:
        answer = msg.content
        messages.append({"role": "assistant", "content": answer})

    history = [m for m in messages if (m.get("role") if isinstance(m, dict) else m.role) != "system"]
    return answer, history


def run_chat():
    """터미널 대화 루프"""
    print("화장품 성분 안전 큐레이터입니다. 종료하려면 'q'를 입력하세요.\n")
    history = []
    while True:
        user_input = input("사용자: ").strip()
        if not user_input or user_input.lower() == "q":
            break
        answer, history = chat(user_input, history)
        print(f"\nAI: {answer}\n")


# ─────────────────────────────────────────
# 실행
# ─────────────────────────────────────────
if __name__ == "__main__":
    run_chat()
