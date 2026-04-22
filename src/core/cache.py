"""Hugging Face cache handling."""

import os
from pathlib import Path


def resolve_hf_cache_dir() -> Path:
    default = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    return Path(os.getenv("HF_CACHE_DIR", str(default)))


def setup_hf_cache_dirs() -> Path:
    cache_dir = resolve_hf_cache_dir()
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_dir / "sentence_transformers"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "transformers").mkdir(parents=True, exist_ok=True)
    (cache_dir / "sentence_transformers").mkdir(parents=True, exist_ok=True)
    return cache_dir
