from __future__ import annotations

import pytest

from telos.constants import END_MARKER
from telos.evaluation.harness.backends import TelosBackend
from telos.runtime import Tool, ToolRegistry
from telos.trajectory import Trajectory


class FakeTokenizer:
    end_id = 999_999

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if i == self.end_id:
                out.append("<|end|>")
            else:
                out.append(chr(i))
        return "".join(out)

    @property
    def hf(self):
        return self


class ScriptedGenerator:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    def __call__(
        self,
        input_ids,
        eos_token_id,
        max_new_tokens,
        *,
        pad_token_id=None,
        return_full_sequence=False,
    ):
        text = self.responses.pop(0)
        ids = [ord(c) for c in text]
        if isinstance(eos_token_id, list):
            ids.append(eos_token_id[0])
        else:
            ids.append(eos_token_id)
        if len(ids) > max_new_tokens:
            ids = ids[:max_new_tokens]
        if return_full_sequence:
            return list(input_ids) + ids
        return ids


@pytest.fixture
def backend() -> TelosBackend:
    return TelosBackend(
        tokenizer=FakeTokenizer(),
        generator=ScriptedGenerator([]),
    )


def test_telos_backend_format(backend: TelosBackend):
    assert backend.format == "telos"


def test_telos_backend_step(backend: TelosBackend):
    backend.generator = ScriptedGenerator(
        ['<|belief|>ok<|action|>{"tool":"answer","text":"42"}']
    )
    trajectory = [
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
    ]
    out = backend.step(trajectory, max_new_tokens=256)
    assert out.step.stopped_on == END_MARKER
    assert out.generated_tokens > 0
    assert out.prompt_tokens > 0
    assert out.inference_sec >= 0.0
    types = [f["type"] for f in out.step.new_frames.to_dict()]
    assert types == ["belief", "action"]


def test_telos_backend_run_terminal_answer(backend: TelosBackend):
    backend.generator = ScriptedGenerator(
        ['<|action|>{"tool":"answer","text":"42"}']
    )
    reg = ToolRegistry()
    initial = Trajectory([
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
    ])
    out = backend.run(initial, reg)
    assert out.run.stopped_on == "terminal_action"
    assert out.run.final_answer == "42"
    assert out.run.iterations == 1
    assert out.generated_tokens > 0
    assert out.total_sec >= 0.0


def test_telos_backend_run_multi_step_tool(backend: TelosBackend):
    backend.generator = ScriptedGenerator([
        '<|action|>{"tool":"echo","value":"a"}',
        '<|action|>{"tool":"answer","text":"done"}',
    ])
    reg = ToolRegistry()
    reg.register(
        Tool(
            "echo",
            lambda value: value,
            {
                "name": "echo",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            },
        )
    )
    initial = Trajectory([
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
    ])
    out = backend.run(initial, reg)
    assert out.run.stopped_on == "terminal_action"
    assert out.run.iterations == 2
    assert out.tool_sec >= 0.0


def test_telos_backend_satisfies_protocol(backend: TelosBackend):
    from telos.evaluation.harness.backend import ModelBackend

    assert isinstance(backend, ModelBackend)
