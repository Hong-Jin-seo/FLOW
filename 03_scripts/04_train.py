"""
03_scripts/04_train.py
청크 JSON → HuggingFace 임베딩 → FAISS 인덱스 구축 & 저장

※ Google Colab GPU 런타임에서 실행하세요.

실행 (Colab 셀):
    %run 03_scripts/04_train.py
    %run 03_scripts/04_train.py --drive_base /content/drive/MyDrive/data
"""

import argparse
import sys
import os

# 02_src 폴더를 Python 경로에 추가
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "02_src")
sys.path.insert(0, SRC_DIR)

# Colab 의존성 자동 설치 (처음 1회)
try:
    from langchain_huggingface import HuggingFaceEmbeddings  # noqa
except ImportError:
    import subprocess
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "sentence-transformers", "langchain-huggingface",
        "langchain-community", "faiss-gpu-cu12", "-q",
    ])

from langchain_core.documents import Document

from common.config_loader import load_config, resolve_output
from common.logger import get_logger
from data.io.reader import load_json
from model.architectures.embedder import build_embedding_model
from model.registry.faiss_registry import build_faiss, save_faiss

logger = get_logger(__name__)


def chunks_to_documents(chunks: list) -> list:
    return [
        Document(page_content=c["page_content"], metadata=c["metadata"])
        for c in chunks
    ]


def main(drive_base: str = "/content/drive/MyDrive/data"):
    logger.info("====== [04] 임베딩 & FAISS 인덱스 구축 시작 ======")
    cfg    = load_config()
    em_cfg = cfg["embedding"]
    ch_cfg = cfg["chunking"]

    # 임베딩 모델 1회 로드
    model = build_embedding_model(em_cfg)

    faiss_prefix = em_cfg["faiss_save_prefix"]

    for preset_id in ch_cfg["weight_presets"].keys():
        chunk_path = resolve_output(cfg, "chunk_prefix", f"{preset_id}.json")

        if not os.path.exists(chunk_path):
            logger.warning(f"청크 파일 없음, 건너뜀: {chunk_path}")
            continue

        logger.info(f"--- 프리셋 {preset_id} ---")
        chunks = load_json(chunk_path)
        docs   = chunks_to_documents(chunks)

        faiss_path = os.path.join(drive_base, f"{faiss_prefix}{preset_id}")
        vs = build_faiss(docs, model)
        save_faiss(vs, faiss_path)

    logger.info("====== [04] 임베딩 완료 ✅ ======")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drive_base",
        type=str,
        default="/content/drive/MyDrive/data",
    )
    args = parser.parse_args()
    main(args.drive_base)
