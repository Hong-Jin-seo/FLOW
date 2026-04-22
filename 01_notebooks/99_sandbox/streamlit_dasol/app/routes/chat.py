from fastapi import APIRouter
from app.schemas import ChatRequest, ChatResponse, SourceChunk
from app.rag.chain import ask

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    result = ask(
        question=req.question,
        skin_type=req.skin_type,
    )
    sources = [SourceChunk(**s) for s in result["sources"]]
    return ChatResponse(answer=result["answer"], sources=sources)
