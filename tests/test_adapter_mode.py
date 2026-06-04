import pytest

from telos.evaluation.harness.load import AdapterMode, load_model


def test_adapter_mode_str_coercion():
    assert AdapterMode("merged") == AdapterMode.MERGED
    assert AdapterMode.MERGED == "merged"


def test_adapter_mode_invalid():
    with pytest.raises(ValueError):
        AdapterMode("lora")


def test_load_model_invalid_mode():
    with pytest.raises(ValueError):
        load_model("x", "invalid")


def test_load_model_peft_requires_adapter_id():
    with pytest.raises(ValueError, match="adapter_id"):
        load_model("base", AdapterMode.PEFT)
