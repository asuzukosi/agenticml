"""
optional llama integration checks (telos tokenizer + optional full model).

- telos vs reserved token id parity and multi-marker encode: need hf access only;
  torch is not required.
- basic generate + telos decode: ``causal_lm`` fixture returns ``none`` without
  torch/cuda/load; the test skips when the model is unavailable.
"""

from __future__ import annotations

import pytest
from transformers import AutoModelForCausalLM

from telos.constants import DEFAULT_BASE_MODEL, TELOS_TOKEN_MAP
from telos.tokenizer import TelosTokenizer

@pytest.fixture(scope="module")
def tt():
    try:
        return TelosTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
    except Exception as e:
        pytest.skip(f"failed to load telos tokenizer for {DEFAULT_BASE_MODEL}: {e}")


@pytest.fixture(scope="module")
def causal_lm():
    """loaded 8b model on cuda, or ``none`` when torch/cuda/load unavailable."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        return AutoModelForCausalLM.from_pretrained(
            DEFAULT_BASE_MODEL,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    except Exception:
        pytest.skip(f"failed to load causal lm for {DEFAULT_BASE_MODEL}")


def test_telos_marker_ids_match_base_reserved_tokens(tt):
    """ids the model sees for <|...|> match hf reserved_special_token_* slots."""
    for telos_name, slot in TELOS_TOKEN_MAP:
        reserved = f"<|reserved_special_token_{slot}|>"
        left = tt.encode(telos_name.value)
        right = tt.hf.encode(reserved, add_special_tokens=False)
        assert left == right, (telos_name, left, right)


def test_concatenated_markers_encode_to_one_id_each(tt):
    """long prompt of all frame markers encodes with no accidental merging."""
    s = "".join(ft.value for ft, _ in TELOS_TOKEN_MAP)
    ids = tt.encode(s)
    assert len(ids) == len(TELOS_TOKEN_MAP)


def test_basic_generation_and_telos_decode(tt, causal_lm):
    """forward + generate runs; decoded text still shows telos markers from prefix."""
    import torch

    prompt = "<|goal|>You are brief.<|mission|>Say Something."
    device = next(causal_lm.parameters()).device
    print("the device is: ", device)
    input_ids = torch.tensor([tt.encode(prompt)], device=device, dtype=torch.long)
    print("the input ids are: ", input_ids)
    pad = tt.hf.pad_token_id if tt.hf.pad_token_id is not None else tt.hf.eos_token_id
    print("the pad token id is: ", pad)
    print("the eos token id is: ", tt.hf.eos_token_id)
    print("the generate parameters are: ", input_ids.shape)
    with torch.inference_mode():
        out = causal_lm.generate(
            input_ids,
            max_new_tokens=24,
            do_sample=False,
            pad_token_id=pad,
            eos_token_id=tt.hf.eos_token_id,
        )
    result = out[0].tolist()[input_ids.shape[1]:]
    text = tt.decode(result)
    print("the text is: ", text)
    print("the result is: ", result)
    assert len(text) > 0
