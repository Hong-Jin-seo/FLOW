"""
services/api.py
FastAPI 백엔드 클라이언트 — /api/chat, /api/curate 호출 래퍼
"""
from __future__ import annotations
from typing import Any
import requests

API_BASE = "http://localhost:8000/api"
_TIMEOUT_CHAT   = 60
_TIMEOUT_CURATE = 90


class APIError(Exception):
    """백엔드 연결 또는 응답 오류"""


def chat(question: str, skin_type: str | None = None) -> dict[str, Any]:
    """
    POST /api/chat
    Returns: {"answer": str, "sources": list[{"product_name": str, "content": str}]}
    Raises: APIError
    """
    try:
        resp = requests.post(
            f"{API_BASE}/chat",
            json={"question": question, "skin_type": skin_type},
            timeout=_TIMEOUT_CHAT,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        raise APIError(
            "❌ FastAPI 서버에 연결할 수 없습니다. "
            "`uvicorn app.main:app --reload` 를 먼저 실행해주세요."
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
            "`uvicorn app.main:app --reload` 를 먼저 실행해주세요."
        )
    except Exception as exc:
        raise APIError(f"❌ 오류: {exc}") from exc
