from __future__ import annotations

import json

import pytest

from telos.evaluation.harness.backend import ModelBackend
from telos.evaluation.harness.backends import ChatMLBackend
from telos.evaluation.harness.backends.chatml_backend import _parse, _to_messages
from telos.runtime import Tool, ToolRegistry
from telos.trajectory import Trajectory


class FakeChatMLTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 2

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **_):
        text = json.dumps(messages) + ("<|assistant|>" if add_generation_prompt else "")
        return [ord(c) for c in text] if tokenize else text

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(i) for i in ids if 32 <= i < 127)

    def convert_tokens_to_ids(self, token: str):
        del token
        return 99


class ScriptedGenerator:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    def __call__(self, input_ids, eos_token_id, max_new_tokens, *, pad_token_id=None, **_):
        del input_ids, eos_token_id, max_new_tokens, pad_token_id
        return [ord(c) for c in self.responses.pop(0)]


@pytest.fixture
def backend() -> ChatMLBackend:
    return ChatMLBackend(FakeChatMLTokenizer(), ScriptedGenerator([]))


def test_to_messages():
    msgs = _to_messages([{"type": "goal", "content": "g"}, {"type": "mission", "content": "m"}])
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"


def test_parse_tool_and_text():
    call, _, stop = _parse('<|python_tag|>{"name": "answer", "arguments": "{}"}<|eom_id|>')
    assert stop == "tool_call"
    assert call["name"] == "answer"
    _, text, stop = _parse("hello<|eot_id|>")
    assert text == "hello"
    assert stop == "assistant_text"


def test_run_terminal(backend: ChatMLBackend):
    backend.generator = ScriptedGenerator([
        '<|python_tag|>{"name": "answer", "arguments": "{\\"text\\": \\"42\\"}"}<|eom_id|>',
    ])
    out = backend.run(
        [{"type": "goal", "content": "g"}, {"type": "mission", "content": "m"}],
        ToolRegistry(),
    )
    assert out.stopped_on == "terminal_action"
    assert out.final_answer == "42"
    assert isinstance(backend, ModelBackend)


def test_run_tool_loop(backend: ChatMLBackend):
    backend.generator = ScriptedGenerator([
        '<|python_tag|>{"name": "echo", "arguments": "{\\"value\\": \\"a\\"}"}<|eom_id|>',
        '<|python_tag|>{"name": "answer", "arguments": "{\\"text\\": \\"done\\"}"}<|eom_id|>',
    ])
    reg = ToolRegistry()
    reg.register(Tool("echo", lambda value: value, {"name": "echo", "parameters": {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}}))
    out = backend.run(Trajectory([{"type": "goal", "content": "g"}, {"type": "mission", "content": "m"}]), reg)
    assert out.final_answer == "done"
    assert out.iterations == 2
