from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def load_repo_dotenv() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    load_dotenv()


def expand_env_values(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand_env_values(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env_values(item) for key, item in value.items()}
    return value


def configure_runtime_environment() -> dict[str, str]:
    load_repo_dotenv()

    cache_base_dir = Path(os.getenv("CAPTR_CACHE_BASE_DIR", Path.home() / ".cache" / "captr")).expanduser()
    shared_hf_cache = Path(
        os.getenv("CAPTR_SHARED_HF_CACHE", cache_base_dir / "shared_hf_cache")
    ).expanduser()
    local_hf_cache = Path(os.getenv("CAPTR_LOCAL_HF_CACHE", cache_base_dir / "hf_home")).expanduser()

    cache_base_dir.mkdir(parents=True, exist_ok=True)
    shared_hf_cache.mkdir(parents=True, exist_ok=True)
    local_hf_cache.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HOME", str(local_hf_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(shared_hf_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(shared_hf_cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_base_dir / "datasets_cache"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_base_dir / ".torch_compile_cache"))
    os.environ.setdefault("TORCH_COMPILE_CACHE_DIR", str(cache_base_dir / ".torch_compile_cache"))
    os.environ.setdefault("VLLM_CACHE_ROOT", str(cache_base_dir / ".vllm_cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_base_dir))

    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    return {
        "cache_base_dir": str(cache_base_dir),
        "shared_hf_cache": str(shared_hf_cache),
        "local_hf_cache": str(local_hf_cache),
    }