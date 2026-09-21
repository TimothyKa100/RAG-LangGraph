from __future__ import annotations

import os
import gc
from functools import lru_cache
from pathlib import Path
from typing import Any


def _cache_folder() -> str | None:
    configured = os.getenv("UCA_RAG_MODEL_CACHE")
    return str(Path(configured).expanduser()) if configured else None


@lru_cache(maxsize=None)
def load_causal_lm(model_name: str, cache_folder: str | None = None):
    """Load one tokenizer/model pair once per process and model path."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_kwargs: dict[str, Any] = {"cache_dir": cache_folder} if cache_folder else {}
    model_kwargs: dict[str, Any] = {"cache_dir": cache_folder, "torch_dtype": "auto"}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    return tokenizer, model


def cached_causal_lm(model_name: str):
    return load_causal_lm(model_name, _cache_folder())


def clear_causal_lm_cache() -> None:
    """Release cached model pairs and return memory to the runtime where possible."""
    load_causal_lm.cache_clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass