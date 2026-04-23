"""
02_src/00_common/config_loader.py
config.yaml 로드 + 경로 헬퍼
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


def load_config(config_path: str = None) -> dict:
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
