import sys
import os
import traceback
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fastapi import APIRouter, UploadFile, File, HTTPException
from dasol_ocr.ocr_test_4 import analyze_image
from app.schemas import ScanIngredient, ScanResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/scan", response_model=ScanResponse)
async def scan(file: UploadFile = File(...)):
    image_bytes = await file.read()
    logger.info(f"[scan] 파일명={file.filename}, 크기={len(image_bytes):,} bytes")

    try:
        results = analyze_image(image_bytes)
    except Exception as e:
        logger.error(f"[scan] analyze_image 실패:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    try:
        ingredients = [
            ScanIngredient(
                ingredient  = r["ingredient"],
                ewg         = r.get("ewg"),
                function    = r.get("function"),
                description = r.get("description"),
            )
            for r in results
        ]

        return ScanResponse(
            ingredients   = ingredients,
            total         = len(ingredients),
            danger_count  = sum(1 for i in ingredients if i.ewg and i.ewg >= 7),
            caution_count = sum(1 for i in ingredients if i.ewg and 3 <= i.ewg <= 6),
        )
    except Exception as e:
        logger.error(f"[scan] 응답 직렬화 실패:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
