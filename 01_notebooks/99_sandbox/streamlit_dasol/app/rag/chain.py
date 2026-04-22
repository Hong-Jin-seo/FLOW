"""
RAG 체인 — 검색된 청크 + 사용자 질문 → LLM 답변

지원 LLM: OpenAI (gpt-4o / gpt-4o-mini) / Anthropic Claude
LLM_PROVIDER 환경변수로 선택 (기본값: openai)
"""

import os
from app.rag.retriever import retrieve

# ── 프롬프트 템플릿 ────────────────────────────────────────────
SYSTEM_PROMPT = """당신은 화장품 성분 전문가입니다.
아래 [참고 데이터]를 바탕으로 사용자 질문에 친절하고 정확하게 답하세요.

규칙:
- 참고 데이터에 있는 내용만 사용하세요.
- 데이터에 없는 내용은 "데이터에 없는 내용입니다"라고 말하세요.
- 성분 추천 시 EWG 등급과 기능을 함께 설명하세요.
- 알레르기 유발 성분은 반드시 언급하세요.
- 답변은 한국어로 작성하세요.
"""

def build_prompt(question: str, chunks: list[dict], skin_type: str = None) -> str:
    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    skin_info = f"\n사용자 피부 타입: {skin_type}" if skin_type else ""
    return f"""[참고 데이터]
{context}

[질문]{skin_info}
{question}"""


# ── LLM 호출 ──────────────────────────────────────────────────
def call_llm(system: str, user: str) -> str:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return resp.choices[0].message.content

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        msg = client.messages.create(
            model=os.getenv("LLM_MODEL", "claude-opus-4-6"),
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text

    else:
        raise ValueError(f"지원하지 않는 LLM_PROVIDER: {provider}")


# ── 메인 RAG 함수 ─────────────────────────────────────────────
def ask(question: str, skin_type: str = None, top_k: int = 5) -> dict:
    """
    Returns:
        {
          "answer" : str,
          "sources": [{"product_name": str, "content": str}, ...]
        }
    """
    # 1. 검색
    chunks = retrieve(question, top_k=top_k)

    # 2. 프롬프트 구성
    user_prompt = build_prompt(question, chunks, skin_type)

    # 3. LLM 호출
    answer = call_llm(SYSTEM_PROMPT, user_prompt)

    # 4. 출처 정리
    sources = [
        {"product_name": c["product_name"], "content": c["text"][:200] + "..."}
        for c in chunks
    ]

    return {"answer": answer, "sources": sources}
