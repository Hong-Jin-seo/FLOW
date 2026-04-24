"""
services/api.py
FastAPI 서버 호출 담당
"""
from __future__ import annotations
from typing import Any
import requests

API_BASE = "http://localhost:8000/api"
_TIMEOUT_CHAT      = 60
_TIMEOUT_CURATE    = 90
_TIMEOUT_SCAN      = 180
_TIMEOUT_RECOMMEND = 60


class APIError(Exception):
    pass


def chat(
    question: str,
    skin_type: str | None = None,
    search_type: str = "hyde",
    history: list = None,
) -> dict[str, Any]:
    try:
        resp = requests.post(
            f"{API_BASE}/chat",
            json={
                "question":    question,
                "skin_type":   skin_type,
                "search_type": search_type,
                "history":     history or [],
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


def scan(image_bytes: bytes, filename: str = "image.jpg") -> dict[str, Any]:
    try:
        resp = requests.post(
            f"{API_BASE}/scan",
            files={"file": (filename, image_bytes, "image/jpeg")},
            timeout=_TIMEOUT_SCAN,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        raise APIError(
            "❌ FastAPI 서버에 연결할 수 없습니다. "
            "`uvicorn app.main:app --reload`를 먼저 실행해주세요."
        )
    except Exception as exc:
        raise APIError(f"❌ 오류: {exc}") from exc


def recommend_chat(message: str, session_id: str = "default") -> dict[str, Any]:
    try:
        resp = requests.post(
            f"{API_BASE}/recommend/chat",
            json={"message": message, "session_id": session_id},
            timeout=_TIMEOUT_RECOMMEND,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        raise APIError(
            "❌ FastAPI 서버에 연결할 수 없습니다. "
            "`uvicorn app.main:app --reload`를 먼저 실행해주세요."
        )
    except Exception as exc:
        raise APIError(f"❌ 오류: {exc}") from exc


def curate(message: str, session: dict) -> dict[str, Any]:
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