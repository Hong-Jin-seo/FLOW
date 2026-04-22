from fastapi import APIRouter
from app.schemas import CurateRequest, CurateResponse
from app.rag.curator import curate

router = APIRouter()


@router.post("/curate", response_model=CurateResponse)
async def curate_endpoint(req: CurateRequest):
    result = curate(message=req.message, session=req.session)
    return CurateResponse(**result)
