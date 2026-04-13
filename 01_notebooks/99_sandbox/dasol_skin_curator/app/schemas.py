from pydantic import BaseModel
from typing import Optional, List, Any


# ── 일반 Q&A ──────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    skin_type: Optional[str] = None


class SourceChunk(BaseModel):
    product_name: str
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk] = []


# ── Skin Curator ───────────────────────────────────────────────
class CurateRequest(BaseModel):
    message: str                        # 고민 or 선택 답변
    session: dict = {}                  # 이전 세션 상태 (Streamlit이 들고 있음)


class CurateResponse(BaseModel):
    message: str                        # LLM 응답 / 다음 질문
    choices: List[str] = []            # 선택지 (버튼으로 표시)
    session: dict                       # 업데이트된 세션
    stage: int                          # 현재 단계 (0~3)
    is_final: bool = False              # True면 최종 추천 완료
