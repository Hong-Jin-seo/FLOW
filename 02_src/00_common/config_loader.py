"""
02_src/00_common/config_loader.py
config.yaml 로드 + 경로 헬퍼 + .env 자동 로드
"""

import os
import yaml


def get_project_root() -> str:
    """프로젝트 루트(flow/) 반환 — 이 파일 기준 3단계 상위"""
    return os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )


def load_env() -> None:
    """
    프로젝트 루트의 .env 파일을 환경변수로 로드합니다.
    python-dotenv가 없으면 수동으로 파싱합니다.
    """
    env_path = os.path.join(get_project_root(), ".env")
    if not os.path.exists(env_path):
        return

    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        # python-dotenv 없으면 수동 파싱
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())


def load_config(config_path: str = None) -> dict:
    """
    .env 로드 후 config.yaml을 읽어 반환합니다.
    config_path 미지정 시 04_configs/config.yaml을 기본으로 사용합니다.
    """
    # .env 먼저 로드 (API 키 등 환경변수 설정)
    load_env()

    if config_path is None:
        config_path = os.path.join(
            get_project_root(), "04_configs", "config.yaml"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(config: dict, key: str) -> str:
    """paths[key]를 프로젝트 루트 기준 절대경로로 반환"""
    return os.path.join(get_project_root(), config["paths"][key])


def resolve_output(config: dict, key: str, suffix: str = "") -> str:
    """
    output_files[key] + suffix 를 processed_dir 기준 절대경로로 반환
    예) resolve_output(cfg, "merged_json")
        resolve_output(cfg, "chunk_prefix", "2.json")
    """
    root      = get_project_root()
    processed = config["paths"]["processed_dir"]
    filename  = config["paths"]["output_files"][key] + suffix
    return os.path.join(root, processed, filename)
