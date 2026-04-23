"""
02_src/02_model/00_architectures/embedder.py
임베딩 모델 팩토리

config.yaml의 embedding.provider 값에 따라
HuggingFace 또는 OpenAI 임베딩 모델을 반환합니다.

provider: "huggingface" → jhgan/ko-sroberta-multitask (768d, GPU 필요)
provider: "openai"      → text-embedding-3-small (1536d, API 키 필요)
"""

import os
import sys

_HERE   = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.join(_HERE, "..", "..", "00_common")
if _COMMON not in sys.path:
    sys.path.insert(0, os.path.normpath(_COMMON))

from logger import get_logger

logger = get_logger(__name__)


def build_embedding_model(em_cfg: dict):
    """
    config.yaml의 embedding 섹션을 받아 임베딩 모델을 반환합니다.

    Parameters
    ----------
    em_cfg : dict
        config["embedding"] 섹션 전체

    Returns
    -------
    HuggingFaceEmbeddings 또는 OpenAIEmbeddings 인스턴스
    """
    provider = em_cfg.get("provider", "huggingface")

    if provider == "openai":
        return _build_openai_model(em_cfg["openai"])
    else:
        return _build_huggingface_model(em_cfg["huggingface"])


def _build_huggingface_model(hf_cfg: dict):
    """HuggingFace 임베딩 모델 (GPU 필요)"""
    from langchain_huggingface import HuggingFaceEmbeddings

    model = HuggingFaceEmbeddings(
        model_name=hf_cfg["model_name"],
        model_kwargs={"device": hf_cfg["device"]},
        encode_kwargs={"normalize_embeddings": hf_cfg["normalize"]},
    )
    test_vec = model.embed_query("나이아신아마이드 EWG 등급")
    logger.info(
        f"[임베딩] HuggingFace | "
        f"model={hf_cfg['model_name']} | "
        f"device={hf_cfg['device']} | "
        f"dim={len(test_vec)}"
    )
    return model


def _build_openai_model(openai_cfg: dict):
    """OpenAI 임베딩 모델 (API 키 필요, GPU 불필요)"""
    from langchain_openai import OpenAIEmbeddings

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.\n"
            "Colab: import os; os.environ['OPENAI_API_KEY'] = 'sk-...'\n"
            "PyCharm: $env:OPENAI_API_KEY = 'sk-...'"
        )

    model = OpenAIEmbeddings(
        model=openai_cfg["model_name"],
        openai_api_key=api_key,
    )
    test_vec = model.embed_query("나이아신아마이드 EWG 등급")
    logger.info(
        f"[임베딩] OpenAI | "
        f"model={openai_cfg['model_name']} | "
        f"dim={len(test_vec)}"
    )
    return model
