"""
services/api.py
FastAPI 서버 호출 담당 — /api/chat, /api/curate 엔드포인트 연결
"""
from __future__ import annotations
from typing import Any
import requests

API_BASE = "http://localhost:8000/api"
_TIMEOUT_CHAT   = 60
_TIMEOUT_CURATE = 90


class APIError(Exception):
    """서버 응답 오류 예외"""


def chat(
    question: str,
    skin_type: str | None = None,
    search_type: str = "hyde",
    history: list = None
) -> dict[str, Any]:
    """
    POST /api/chat
    Returns: {"answer": str, "sources": list[{"product_name": str, "content": str}]}
    Raises: APIError
    """
    try:
        resp = requests.post(
            f"{API_BASE}/chat",
            json={
                "question": question,
                "skin_type": skin_type,
                "search_type": search_type,
                "history": history or []
            },
            timeout=_TIMEOUT_CHAT,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        raise APIError(
            "❌ FastAPI 서버에 연결할 수 없습니다. "
            "`uvicorn app.main:app --reload`로 서버를 먼저 실행하세요."
        )
    except Exception as exc:
        raise APIError(f"❌ 오류: {exc}") from exc


def curate(message: str, session: dict) -> dict[str, Any]:
    """
    POST /api/curate
    Returns: {"message": str, "choices": list, "session": dict,
              "stage": int, "is_final": bool, "products": list}
    Raises: APIError
    """
    try:
        resp = requests.post(
            f"{API_BASE}/curate",
            json={"message": message, "session": session},
            timeout=_TIMEOUT_CURATE,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        raise APIError(
            "❌ FastAPI 서버에 연결할 수 없습니다. "
            "`uvicorn app.main:app --reload`로 서버를 먼저 실행하세요."
        )
    except Exception as exc:
        raise APIError(f"❌ 오류: {exc}") from exc