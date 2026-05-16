from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import torch
from transformers import AutoModelForCausalLM

@dataclass
class HfGenerator:
    model: AutoModelForCausalLM

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, 
                        *, 
                        dtype: Optional[torch.dtype] = torch.bfloat16, 
                        device_map: Optional[str] = "auto",
                        **model_kwargs: dict[str, Any]) -> HfGenerator:
        model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=dtype, torch_dtype=dtype, device_map=device_map, **model_kwargs)
        return cls(model)

    def _first_execution_device(self) -> torch.device:
        hf_map = getattr(self.model, "hf_device_map", None)
        if isinstance(hf_map, dict) and hf_map:
            key = "model.embed_tokens" if "model.embed_tokens" in hf_map else next(iter(hf_map.keys()))
            val = hf_map[key]
            if isinstance(val, int):
                return torch.device(f"cuda:{val}")
            if isinstance(val, str):
                return torch.device(val)
        return next(self.model.parameters()).device

    def generate(self, input_ids: list[int], stop_token_id: int, max_new_tokens: int) -> list[int]:
        device = self._first_execution_device()
        input_tensors = torch.tensor([input_ids], device=device, dtype=torch.long)
        attention_mask = torch.ones_like(input_tensors)
        with torch.inference_mode():
            out = self.model.generate(input_tensors, 
            max_new_tokens=max_new_tokens, 
            attention_mask=attention_mask,
            do_sample=False, 
            pad_token_id=stop_token_id, 
            eos_token_id=stop_token_id)
        return out[0].tolist()[len(input_ids):]

    def __call__(self, input_ids: list[int], stop_token_id: int, max_new_tokens: int) -> list[int]:
        return self.generate(input_ids, stop_token_id, max_new_tokens)
