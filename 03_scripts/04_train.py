"""
03_scripts/04_train.py
청크 JSON → 임베딩 → FAISS 인덱스 구축 & 저장

저장 경로: 00_data/02_processed/faiss_index_preset{1~4}/
           (config.yaml의 processed_dir 기준)

실행:
    python 03_scripts/04_train.py
    python 03_scripts/04_train.py --preset_id 1  # 특정 프리셋만
"""

import argparse
import sys
import os
import shutil

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "02_src", "00_common"))
sys.path.insert(0, os.path.join(ROOT, "02_src", "01_data", "00_ingestion"))
sys.path.insert(0, os.path.join(ROOT, "02_src", "01_data", "01_preprocessing"))
sys.path.insert(0, os.path.join(ROOT, "02_src", "01_data", "02_io"))
sys.path.insert(0, os.path.join(ROOT, "02_src", "02_model", "00_architectures"))
sys.path.insert(0, os.path.join(ROOT, "02_src", "02_model", "03_registry"))

# 의존성 자동 설치
try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
except ImportError:
    import subprocess
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "langchain-openai", "langchain-community",
        "langchain-huggingface", "sentence-transformers",
        "faiss-cpu", "openai", "python-dotenv", "-q",
    ])

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config_loader import load_config, resolve_path, resolve_output
from logger import get_logger
from reader import load_json
from embedder import build_embedding_model

logger = get_logger(__name__)


# ── Document 변환 ─────────────────────────────────────────────
def chunks_to_documents(chunks: list) -> list:
    return [
        Document(page_content=c["page_content"], metadata=c["metadata"])
        for c in chunks
    ]


# ── FAISS 배치 구축 ───────────────────────────────────────────
def build_faiss_batched(docs: list, model, batch_size: int) -> FAISS:
    total = len(docs)
    logger.info(f"[FAISS] 구축 시작 (총 {total:,}개 / 배치 {batch_size})")

    vs = FAISS.from_documents(docs[:batch_size], model)
    logger.info(f"  배치 1/{(total-1)//batch_size+1} 완료: {min(batch_size, total):,}개")

    total_batches = (total - 1) // batch_size + 1
    for i in range(1, total_batches):
        start    = i * batch_size
        end      = min(start + batch_size, total)
        batch_vs = FAISS.from_documents(docs[start:end], model)
        vs.merge_from(batch_vs)
        logger.info(f"  배치 {i+1}/{total_batches} 완료: {end:,}개 누적")

    logger.info(f"[FAISS] 구축 완료 | 벡터 수: {vs.index.ntotal:,}")
    return vs


# ── 저장 + 검증 ───────────────────────────────────────────────
def save_and_verify(vs: FAISS, save_path: str, model,
                    expected_count: int) -> bool:
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        logger.info(f"기존 폴더 삭제: {save_path}")

    os.makedirs(save_path, exist_ok=True)
    vs.save_local(save_path)

    faiss_mb = os.path.getsize(os.path.join(save_path, "index.faiss")) / 1024 / 1024
    pkl_mb   = os.path.getsize(os.path.join(save_path, "index.pkl"))   / 1024 / 1024
    logger.info(f"[저장] {save_path}")
    logger.info(f"  index.faiss : {faiss_mb:.1f} MB")
    logger.info(f"  index.pkl   : {pkl_mb:.1f} MB")

    reload_vs      = FAISS.load_local(
        save_path, model, allow_dangerous_deserialization=True
    )
    vector_count   = reload_vs.index.ntotal
    docstore_count = len(reload_vs.docstore._dict)

    if vector_count == docstore_count == expected_count:
        logger.info(
            f"✅ 검증 통과: 벡터={vector_count:,} / "
            f"docstore={docstore_count:,} / 원본={expected_count:,}"
        )
        results = reload_vs.similarity_search("나이아신아마이드 EWG 등급", k=1)
        if results:
            logger.info(f"  검색 테스트 ✅: {results[0].page_content[:60]}")
        return True
    else:
        logger.error(
            f"❌ 검증 실패: 벡터={vector_count:,} / "
            f"docstore={docstore_count:,} / 원본={expected_count:,}"
        )
        return False


# ── 메인 ─────────────────────────────────────────────────────
def main(preset_id: int = None):
    logger.info("====== [04] 임베딩 & FAISS 인덱스 구축 시작 ======")

    cfg    = load_config()
    em_cfg = cfg["embedding"]
    ch_cfg = cfg["chunking"]

    # FAISS 저장 경로 → 청크와 동일한 processed_dir
    processed_dir = resolve_path(cfg, "processed_dir")
    faiss_prefix  = em_cfg["faiss_save_prefix"]

    logger.info(f"저장 경로: {processed_dir}")

    # 임베딩 모델 로드
    provider = em_cfg.get("provider", "openai")
    logger.info(f"임베딩 provider: {provider}")
    model = build_embedding_model(em_cfg)

    # 배치 크기
    batch_size = (
        em_cfg.get("openai", {}).get("batch_size", 500)
        if provider == "openai"
        else 2000
    )

    # 실행할 프리셋 결정
    preset_ids = (
        [preset_id]
        if preset_id is not None
        else list(ch_cfg["weight_presets"].keys())
    )

    for pid in preset_ids:
        chunk_path = resolve_output(cfg, "chunk_prefix", f"{pid}.json")

        if not os.path.exists(chunk_path):
            logger.warning(f"청크 파일 없음, 건너뜀: {chunk_path}")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"프리셋 {pid} 시작")
        logger.info(f"{'='*50}")

        chunks = load_json(chunk_path)
        docs   = chunks_to_documents(chunks)
        logger.info(f"청크 로드 완료: {len(docs):,}개")

        vs = build_faiss_batched(docs, model, batch_size=batch_size)

        # 청크와 동일한 폴더에 저장
        faiss_path = os.path.join(processed_dir, f"{faiss_prefix}{pid}")
        ok = save_and_verify(vs, faiss_path, model, expected_count=len(docs))

        if ok:
            logger.info(f"✅ 프리셋 {pid} 완료! → {faiss_path}")
        else:
            logger.error(f"❌ 프리셋 {pid} 실패!")

    logger.info("\n====== [04] 전체 임베딩 완료 ✅ ======")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="임베딩 & FAISS 구축")
    parser.add_argument(
        "--preset_id",
        type=int,
        default=None,
        help="특정 프리셋만 실행 (기본: 1~4 전부)",
    )
    args = parser.parse_args()
    main(args.preset_id)
