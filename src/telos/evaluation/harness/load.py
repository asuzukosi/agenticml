"""shared causal lm loading for evaluation harness (cuda:0 + cpu offload)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Union

import torch
from transformers import AutoModelForCausalLM


class AdapterMode(str, Enum):
    MERGED = "merged"
    PEFT = "peft"


def causal_lm_load_kwargs(dtype: torch.dtype) -> dict[str, Any]:
    """device_map/max_memory for inference; avoids multi-gpu auto split."""
    if not torch.cuda.is_available():
        return {"torch_dtype": dtype, "device_map": "cpu"}
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    cap = max(1, int(total_gib - 2))
    return {
        "torch_dtype": dtype,
        "device_map": "auto",
        "max_memory": {0: f"{cap}GiB", "cpu": "100GiB"},
        "offload_buffers": True,
    }


def load_model(
    model_id: str,
    adapter_mode: Union[AdapterMode, str],
    adapter_id: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
):
    mode = AdapterMode(adapter_mode)
    load_kw = causal_lm_load_kwargs(dtype)
    if mode == AdapterMode.MERGED:
        return AutoModelForCausalLM.from_pretrained(model_id, **load_kw)
    if mode == AdapterMode.PEFT:
        if not adapter_id:
            raise ValueError(f"adapter_mode={AdapterMode.PEFT.value!r} requires adapter_id")
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError(
                f"adapter_mode={AdapterMode.PEFT.value!r} requires: pip install peft"
            ) from e
        base = AutoModelForCausalLM.from_pretrained(model_id, **load_kw)
        return PeftModel.from_pretrained(base, adapter_id)
    raise ValueError(f"unsupported adapter_mode: {adapter_mode!r}")


def model_device(model) -> torch.device:
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device
