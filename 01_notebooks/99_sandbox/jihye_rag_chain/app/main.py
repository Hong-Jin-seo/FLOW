"""
jihye_rag_chain/app/main.py
실행: uvicorn app.main:app --reload --port 8000
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from graph import run_graph

app = FastAPI()


class HistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    skin_type: Optional[str] = None
    history: Optional[list[HistoryItem]] = []


class SourceItem(BaseModel):
    product_name: str
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    question_type: str
    preset_id: int
    search_type: str    # ← 추가


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history = [{"role": h.role, "content": h.content} for h in req.history]

    result = run_graph(
        query=req.question,
        history=history
    )

    sources = [
        SourceItem(
            product_name=s.get("ingredient", s.get("source", "?")),
            content=s.get("content", "")
        )
        for s in result.get("sources", [])
    ]

    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        question_type=result.get("question_type", ""),
        preset_id=result.get("preset_id", 2),
        search_type=result.get("search_type", "dense")    # ← 추가
    )


@app.post("/api/curate")
def curate(body: dict):
    return {
        "message": "curate 기능은 준비 중입니다.",
        "choices": [],
        "session": {},
        "is_final": True
    }


@app.get("/")
def health():
    return {"status": "ok"}