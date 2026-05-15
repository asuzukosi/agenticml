# tests/test_hf_generator.py
from __future__ import annotations

import pytest
import torch

from telos.constants import DEFAULT_BASE_MODEL
from telos.runtime.hf_generator import HfGenerator
from telos.tokenizer import TelosTokenizer


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for hf_generator integration tests",
)


@pytest.fixture(scope="module")
def tt() -> TelosTokenizer:
    return TelosTokenizer.from_pretrained(DEFAULT_BASE_MODEL)


@pytest.fixture(scope="module")
def gen() -> HfGenerator:
    return HfGenerator.from_pretrained(
        DEFAULT_BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
    )


def test_hf_generator_returns_suffix_only(tt: TelosTokenizer, gen: HfGenerator):
    prompt_ids = tt.encode("<|goal|>You are concise.<|mission|>Say hi.")
    out = gen(prompt_ids, tt.end_id, max_new_tokens=24)
    assert isinstance(out, list)
    assert len(out) > 0
    assert out != prompt_ids


def test_hf_generator_respects_max_new_tokens(tt: TelosTokenizer, gen: HfGenerator):
    prompt_ids = tt.encode("<|goal|>test<|mission|>test")
    out = gen(prompt_ids, tt.end_id, max_new_tokens=8)
    assert len(out) <= 8


def test_hf_generator_emits_attention_mask(monkeypatch, tt: TelosTokenizer, gen: HfGenerator):
    captured = {}

    original_generate = gen.model.generate

    def wrapped_generate(*args, **kwargs):
        captured["attention_mask"] = kwargs.get("attention_mask")
        return original_generate(*args, **kwargs)

    monkeypatch.setattr(gen.model, "generate", wrapped_generate)
    _ = gen(tt.encode("<|goal|>x<|mission|>y"), tt.end_id, max_new_tokens=4)

    assert captured["attention_mask"] is not None
    assert captured["attention_mask"].dtype == torch.long
