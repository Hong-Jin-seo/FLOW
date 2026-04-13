"""
Skin Curator — 스무고개형 큐레이션 상태머신

단계(Stage):
  0  ANALYZE  : 고민 분석 → 필요/기피 성분 추출 → 후보 검색 → Q1 생성
  1  Q1       : 제품 타입 선택 (예: 에센스 vs 크림)
  2  Q2       : 추가 선호도 선택 (예: 자연 성분 vs 가성비)
  3  FINAL    : 최종 1개 추천 + 성분 근거 설명

설계 원칙:
- 백엔드는 완전 무상태(stateless) — 세션 dict를 Streamlit이 들고 있다가 매 요청마다 전달
- LLM이 질문/답변을 동적으로 생성 (하드코딩 없음)
- 후보 필터링은 products_meta.json 기반 (벡터 검색과 별도)
"""

import json
import os
from functools import lru_cache
from pathlib import Path

from app.rag.retriever import retrieve
from app.rag.chain import call_llm

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
META_PATH = BASE_DIR / "vectorstore" / "products_meta.json"

# ── 단계 상수 ──────────────────────────────────────────────────
STAGE_ANALYZE = 0
STAGE_Q1      = 1
STAGE_Q2      = 2
STAGE_FINAL   = 3


@lru_cache(maxsize=1)
def _load_meta() -> list[dict]:
    if not META_PATH.exists():
        raise FileNotFoundError("products_meta.json 없음. indexer.py를 먼저 실행하세요.")
    with open(META_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── Stage 0: 고민 분석 ─────────────────────────────────────────
def start_curation(concern: str) -> dict:
    """
    사용자 고민 → LLM 성분 분석 → 후보 검색 → Q1 반환
    """
    # 1. LLM으로 필요/기피 성분 추출
    analysis_prompt = f"""사용자의 피부 고민을 분석해서 아래 JSON 형식으로만 답하세요.

사용자 고민: "{concern}"

{{
  "needed_ingredients": ["필요한 성분1", "성분2"],
  "avoided_ingredients": ["피해야 할 성분1", "성분2"],
  "skin_keywords": ["피부 상태 키워드"],
  "summary": "고민 한줄 요약"
}}

- needed_ingredients: 이 고민에 도움이 되는 핵심 성분 (한국어, 최대 5개)
- avoided_ingredients: 자극이 될 수 있는 성분 (한국어, 최대 3개)
- 반드시 유효한 JSON만 출력하세요."""

    raw = call_llm("당신은 화장품 성분 전문가입니다.", analysis_prompt)

    # JSON 파싱 (마크다운 펜스 제거)
    import re
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        analysis = json.loads(raw)
    except Exception:
        analysis = {
            "needed_ingredients": [],
            "avoided_ingredients": [],
            "skin_keywords": [],
            "summary": concern[:50],
        }

    # 2. RAG로 관련 제품 후보 검색
    query   = concern + " " + " ".join(analysis.get("needed_ingredients", []))
    chunks  = retrieve(query, top_k=15)

    # 제품 청크만 필터링, product_id 중복 제거
    seen_ids, candidates = set(), []
    for c in chunks:
        if c["type"] == "product" and c["product_id"] not in seen_ids:
            seen_ids.add(c["product_id"])
            candidates.append(c)

    # 후보가 너무 적으면 메타에서 직접 보강
    if len(candidates) < 5:
        all_meta = _load_meta()
        needed   = set(analysis.get("needed_ingredients", []))
        for m in all_meta:
            if m["product_id"] in seen_ids:
                continue
            if any(ing in m["ingredients"] for ing in needed):
                candidates.append({
                    "type": "product",
                    "product_id": m["product_id"],
                    "product_name": m["product_name"],
                    "category": m["category"],
                    "text": f"[제품] {m['product_name']}\n카테고리: {m['category']}",
                })
                seen_ids.add(m["product_id"])
            if len(candidates) >= 10:
                break

    # 3. Q1 생성 — 제품 타입 선호도
    categories = list({c.get("category", "") for c in candidates if c.get("category")})
    cat_text   = ", ".join(categories[:6]) if categories else "토너, 세럼, 크림"

    q1_prompt = f"""사용자 고민: "{concern}"
분석 요약: {analysis.get('summary','')}
후보 제품 카테고리: {cat_text}

위 상황에서 사용자의 선호를 좁힐 수 있는 질문 1개와 선택지 2개를 만드세요.
선택지는 카테고리(예: 토너 vs 크림) 또는 제형(예: 가벼운 타입 vs 리치한 타입)으로 구성하세요.

아래 JSON 형식으로만 답하세요:
{{
  "question": "질문 내용",
  "choices": ["선택지A", "선택지B"]
}}"""

    raw_q1 = call_llm("당신은 뷰티 큐레이터입니다.", q1_prompt)
    raw_q1 = re.sub(r"```(?:json)?|```", "", raw_q1).strip()
    try:
        q1_data = json.loads(raw_q1)
    except Exception:
        q1_data = {
            "question": "어떤 제형을 선호하시나요?",
            "choices": ["가벼운 토너/에센스 타입", "촉촉한 크림/로션 타입"],
        }

    # 후보를 직렬화 가능하게 정리
    serializable_candidates = [
        {
            "product_id"  : c.get("product_id"),
            "product_name": c.get("product_name"),
            "category"    : c.get("category", ""),
            "text"        : c.get("text", "")[:500],   # 너무 길면 자름
        }
        for c in candidates
    ]

    session = {
        "stage"      : STAGE_Q1,
        "concern"    : concern,
        "analysis"   : analysis,
        "candidates" : serializable_candidates,
        "q1"         : q1_data,
        "q1_answer"  : None,
        "q2"         : None,
        "q2_answer"  : None,
    }

    intro = (
        f"**피부 고민 분석 완료!** 🔍\n\n"
        f"> {analysis.get('summary', concern)}\n\n"
        f"필요 성분: {', '.join(analysis.get('needed_ingredients', []))}\n"
        f"피해야 할 성분: {', '.join(analysis.get('avoided_ingredients', []))}\n\n"
        f"후보 제품 **{len(serializable_candidates)}개**를 찾았어요. "
        f"몇 가지 여쭤볼게요!\n\n"
        f"**{q1_data['question']}**"
    )

    return {
        "message": intro,
        "choices": q1_data["choices"],
        "session": session,
        "stage"  : STAGE_Q1,
        "is_final": False,
    }


# ── Stage 1: Q1 답변 처리 → Q2 생성 ──────────────────────────
def process_q1(session: dict, answer: str) -> dict:
    import re
    session["q1_answer"] = answer
    candidates = session["candidates"]

    # 답변 키워드로 카테고리 필터링
    lower = answer.lower()
    filtered = [
        c for c in candidates
        if any(kw in (c.get("category","") + c.get("product_name","")).lower()
               for kw in lower.split()) or True   # 못 걸러지면 전부 유지
    ]
    # 실제 필터링: 카테고리 매칭
    cat_filtered = [
        c for c in candidates
        if any(kw in c.get("category","").lower() for kw in lower.split())
    ]
    if len(cat_filtered) >= 2:
        filtered = cat_filtered

    session["candidates"] = filtered[:8]  # 최대 8개 유지

    # Q2 생성 — 성분 안전성 vs 가성비
    q2_prompt = f"""사용자 고민: "{session['concern']}"
Q1 답변: "{answer}"
남은 후보 수: {len(session['candidates'])}개

사용자의 최종 선택을 좁힐 두 번째 질문을 만드세요.
(성분 안전성 vs 가성비, 자연유래 vs 기능성, 무향 vs 향기 등 기준으로)

JSON 형식으로만:
{{
  "question": "질문",
  "choices": ["선택지A", "선택지B"]
}}"""

    raw_q2 = call_llm("당신은 뷰티 큐레이터입니다.", q2_prompt)
    raw_q2 = re.sub(r"```(?:json)?|```", "", raw_q2).strip()
    try:
        q2_data = json.loads(raw_q2)
    except Exception:
        q2_data = {
            "question": "성분 안전성과 가성비 중 어느 쪽이 더 중요하신가요?",
            "choices": ["성분이 안전한 제품 우선", "가성비 좋은 제품 우선"],
        }

    session["q2"]    = q2_data
    session["stage"] = STAGE_Q2

    return {
        "message" : f"좋아요! 한 가지만 더 여쭤볼게요.\n\n**{q2_data['question']}**",
        "choices" : q2_data["choices"],
        "session" : session,
        "stage"   : STAGE_Q2,
        "is_final": False,
    }


# ── Stage 2: Q2 답변 처리 → 최종 추천 ────────────────────────
def process_q2(session: dict, answer: str) -> dict:
    session["q2_answer"] = answer
    candidates = session["candidates"]

    # 최종 후보가 없으면 전체에서 다시
    if not candidates:
        all_meta   = _load_meta()
        candidates = [
            {"product_id": m["product_id"], "product_name": m["product_name"],
             "category": m["category"], "text": ""}
            for m in all_meta[:5]
        ]

    # 최종 추천 LLM
    cand_text = "\n".join(
        f"- {c['product_name']} ({c.get('category','')})"
        for c in candidates[:5]
    )
    needed   = ", ".join(session["analysis"].get("needed_ingredients", []))
    avoided  = ", ".join(session["analysis"].get("avoided_ingredients", []))

    final_prompt = f"""사용자 피부 고민: "{session['concern']}"
필요 성분: {needed}
피해야 할 성분: {avoided}
선호 제형: {session.get('q1_answer','')}
선호 기준: {answer}

후보 제품:
{cand_text}

위 후보 중 가장 적합한 제품 1개를 선택하고, 아래 형식으로 추천 이유를 설명하세요.

**추천 제품: [제품명]**

**왜 이 제품인가요?**
- 고민과의 연관성 설명
- 핵심 성분 2~3개와 각 효능
- 주의할 점 (있다면)

**사용 팁**
- 간단한 사용 팁 1~2개

따뜻하고 전문적인 톤으로 작성하세요."""

    recommendation = call_llm("당신은 친절한 피부 전문가입니다.", final_prompt)

    session["stage"] = STAGE_FINAL

    return {
        "message" : recommendation,
        "choices" : [],
        "session" : session,
        "stage"   : STAGE_FINAL,
        "is_final": True,
    }


# ── 진입점 ─────────────────────────────────────────────────────
def curate(message: str, session: dict) -> dict:
    """
    stage에 따라 적절한 함수 호출.
    session이 비어 있으면 Stage 0(최초 고민 입력)으로 시작.
    """
    stage = session.get("stage", STAGE_ANALYZE)

    if stage == STAGE_ANALYZE:
        return start_curation(message)
    elif stage == STAGE_Q1:
        return process_q1(session, message)
    elif stage == STAGE_Q2:
        return process_q2(session, message)
    else:
        # 이미 FINAL → 새 고민으로 재시작
        return start_curation(message)
