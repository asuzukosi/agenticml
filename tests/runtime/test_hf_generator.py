# tests/runtime/test_hf_generator.py
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from telos.constants import DEFAULT_BASE_MODEL
from telos.evaluation.harness.load import causal_lm_load_kwargs
from telos.runtime.hf_generator import HfGenerator
from telos.tokenizer import TelosTokenizer


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for hf_generator integration tests",
)


def _pad_id(tt: TelosTokenizer) -> int:
    hf = tt.hf
    return hf.pad_token_id if hf.pad_token_id is not None else hf.eos_token_id


def _stop_ids(tt: TelosTokenizer) -> list[int]:
    ids = [tt.end_id]
    eos = tt.hf.eos_token_id
    if eos is not None and eos not in ids:
        ids.append(eos)
    return ids


@pytest.fixture(scope="module")
def tt() -> TelosTokenizer:
    return TelosTokenizer.from_pretrained(DEFAULT_BASE_MODEL)


@pytest.fixture(scope="module")
def gen() -> HfGenerator:
    return HfGenerator.from_pretrained(DEFAULT_BASE_MODEL, dtype=torch.bfloat16)


def test_hf_generator_returns_suffix_only(tt: TelosTokenizer, gen: HfGenerator):
    prompt_ids = tt.encode("<|goal|>You are concise.<|mission|>Say hi.")
    out = gen.generate(
        prompt_ids,
        pad_token_id=_pad_id(tt),
        eos_token_id=_stop_ids(tt),
        max_new_tokens=24,
    )
    assert isinstance(out, list)
    assert len(out) > 0
    assert out != prompt_ids


def test_hf_generator_respects_max_new_tokens(tt: TelosTokenizer, gen: HfGenerator):
    prompt_ids = tt.encode("<|goal|>test<|mission|>test")
    out = gen.generate(
        prompt_ids,
        pad_token_id=_pad_id(tt),
        eos_token_id=tt.end_id,
        max_new_tokens=8,
    )
    assert len(out) <= 8


def test_hf_generator_emits_attention_mask(monkeypatch, tt: TelosTokenizer, gen: HfGenerator):
    captured = {}

    original_generate = gen.model.generate

    def wrapped_generate(*args, **kwargs):
        captured["attention_mask"] = kwargs.get("attention_mask")
        captured["pad_token_id"] = kwargs.get("pad_token_id")
        captured["eos_token_id"] = kwargs.get("eos_token_id")
        return original_generate(*args, **kwargs)

    monkeypatch.setattr(gen.model, "generate", wrapped_generate)
    pad = _pad_id(tt)
    stops = _stop_ids(tt)
    _ = gen.generate(
        tt.encode("<|goal|>x<|mission|>y"),
        pad_token_id=pad,
        eos_token_id=stops,
        max_new_tokens=4,
    )

    assert captured["attention_mask"] is not None
    assert captured["attention_mask"].dtype == torch.long
    assert captured["pad_token_id"] == pad
    assert captured["eos_token_id"] == stops


def test_from_pretrained_uses_shared_load_kwargs(monkeypatch):
    seen: dict = {}

    def fake_load_model(model_id, adapter_mode, adapter_id=None, dtype=torch.bfloat16):
        seen["model_id"] = model_id
        seen["adapter_mode"] = adapter_mode
        seen["load_kw"] = causal_lm_load_kwargs(dtype)
        return MagicMock()

    monkeypatch.setattr(
        "telos.runtime.hf_generator.load_model",
        fake_load_model,
    )
    HfGenerator.from_pretrained("test-model", dtype=torch.float16)
    assert seen["model_id"] == "test-model"
    assert "max_memory" in seen["load_kw"] or seen["load_kw"].get("device_map") == "cpu"
