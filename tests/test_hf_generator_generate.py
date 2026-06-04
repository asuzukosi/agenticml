from __future__ import annotations

from unittest.mock import MagicMock

import torch

from telos.runtime.hf_generator import HfGenerator


def test_generate_separate_pad_and_eos(monkeypatch):
    captured: dict = {}
    param = torch.nn.Parameter(torch.zeros(1))
    model = MagicMock()
    model.parameters.return_value = iter([param])
    monkeypatch.setattr(
        "telos.runtime.hf_generator.model_device",
        lambda _m: torch.device("cpu"),
    )
    def capture_generate(*_a, **k):
        captured.update(k)
        return torch.tensor([[1, 2, 3, 4]])

    model.generate.side_effect = capture_generate

    gen = HfGenerator(model)
    out = gen.generate(
        [10, 20],
        pad_token_id=0,
        eos_token_id=[128001, 128009],
        max_new_tokens=16,
    )

    assert out == [3, 4]
    assert captured["pad_token_id"] == 0
    assert captured["eos_token_id"] == [128001, 128009]
    assert captured["do_sample"] is False


def test_generate_return_full_sequence(monkeypatch):
    param = torch.nn.Parameter(torch.zeros(1))
    model = MagicMock()
    model.parameters.return_value = iter([param])
    model.generate.return_value = torch.tensor([[10, 20, 30, 40]])
    monkeypatch.setattr(
        "telos.runtime.hf_generator.model_device",
        lambda _m: torch.device("cpu"),
    )

    gen = HfGenerator(model)
    suffix = gen.generate(
        [10, 20],
        pad_token_id=0,
        eos_token_id=1,
        max_new_tokens=8,
    )
    full = gen.generate(
        [10, 20],
        pad_token_id=0,
        eos_token_id=1,
        max_new_tokens=8,
        return_full_sequence=True,
    )

    assert suffix == [30, 40]
    assert full == [10, 20, 30, 40]
