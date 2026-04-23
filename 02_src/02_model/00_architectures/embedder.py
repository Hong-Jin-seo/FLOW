"""
02_src/02_model/00_architectures/embedder.py
HuggingFaceEmbeddings 모델을 설정값 기반으로 생성하는 팩토리 함수입니다.
"""

from langchain_huggingface import HuggingFaceEmbeddings

from src.common.logger import get_logger

logger = get_logger(__name__)


def build_embedding_model(em_cfg: dict) -> HuggingFaceEmbeddings:
    """
    config["embedding"] 섹션을 받아 임베딩 모델을 반환합니다.

    Parameters
    ----------
    em_cfg : dict
        keys: model_name, device, normalize
    """
    model = HuggingFaceEmbeddings(
        model_name=em_cfg["model_name"],
        model_kwargs={"device": em_cfg["device"]},
        encode_kwargs={"normalize_embeddings": em_cfg["normalize"]},
    )
    test_vec = model.embed_query("나이아신아마이드 EWG 등급")
    logger.info(
        f"[임베딩 모델] {em_cfg['model_name']} | "
        f"device={em_cfg['device']} | dim={len(test_vec)}"
    )
    return model
