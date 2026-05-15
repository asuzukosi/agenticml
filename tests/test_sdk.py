"""Tests for telos.sdk."""
 
import json
 
import pytest
 
from telos.sdk import StepResult, _render_tool_schema, step
from telos.trajectory import Trajectory

class FakeTokenizer:
    """A trivial char-level tokenizer with <|end|> rendered as marker."""
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

class ScriptedGenerator:
    """a generator that returns a pre-set text response."""
 
    def __init__(self, response_text: str, append_end: bool = True):
        self.response_text = response_text
        self.append_end = append_end
 
    def __call__(self, input_ids, stop_token_id, max_new_tokens):
        ids = [ord(c) for c in self.response_text]
        if self.append_end:
            ids.append(stop_token_id)
        if len(ids) > max_new_tokens:
            ids = ids[:max_new_tokens]
        return ids

def test_render_tool_schema_empty_returns_empty():
    assert _render_tool_schema([]) == ""

def test_render_tool_schema_empty_returns_empty():
    assert _render_tool_schema([]) == ""
 
 
def test_render_tool_schema_emits_namespace_header():
    tools = [{
        "name": "read_file",
        "description": "Read a file's contents.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        }
    }]
    out = _render_tool_schema(tools)
    assert out.startswith("tools:\nnamespace tools {")
    assert out.endswith("}")

def test_render_tool_schema_emits_function_signature():
    tools = [{
        "name": "read_file",
        "description": "Read a file's contents.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        }
    }]
    out = _render_tool_schema(tools)
    assert "// Read a file's contents." in out
    assert "type read_file = (_: {" in out
    assert "path: string," in out
    assert "}) => any;" in out

def test_render_tool_schema_marks_optional_args():
    tools = [{
        "name": "read_file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "encoding": {"type": "string"}
            },
            "required": ["path"]
        }
    }]
    out = _render_tool_schema(tools)
    assert "path: string," in out
    assert "encoding?: string," in out
 
 
def test_render_tool_schema_no_args_emits_unit_signature():
    tools = [{"name": "ping", "description": "no args"}]
    out = _render_tool_schema(tools)
    assert "type ping = () => any;" in out

def test_render_tool_schema_inlines_property_descriptions():
    tools = [{
        "name": "read_file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to read"}
            },
            "required": ["path"]
        }
    }]
    out = _render_tool_schema(tools)
    assert "// Path to read" in out

def test_render_tool_schema_handles_enum():
    tools = [{
        "name": "set_mode",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["fast", "slow"]}
            },
            "required": ["mode"]
        }
    }]
    out = _render_tool_schema(tools)
    assert "mode: 'fast' | 'slow'" in out or 'mode: "fast" | "slow"' in out

def test_step_accepts_list_of_dicts():
    trajectory = [
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
    ]
    gen = ScriptedGenerator('<|action|>{"tool":"answer","text":"ok"}')
    result = step(trajectory, None, tokenizer=FakeTokenizer(), generate=gen)
    assert isinstance(result.trajectory, Trajectory)
 
 
def test_step_accepts_trajectory_instance():
    from telos.frames import goal, mission
    trajectory = Trajectory([goal("g"), mission("m")])
    gen = ScriptedGenerator('<|action|>{"tool":"answer","text":"ok"}')
    result = step(trajectory, None, tokenizer=FakeTokenizer(), generate=gen)
    assert isinstance(result.trajectory, Trajectory)
 
 
def test_step_accepts_list_of_frame_objects():
    from telos.frames import goal, mission
    trajectory = [goal("g"), mission("m")]
    gen = ScriptedGenerator('<|action|>{"tool":"answer","text":"ok"}')
    result = step(trajectory, None, tokenizer=FakeTokenizer(), generate=gen)
    assert isinstance(result.trajectory, Trajectory)

 
def test_step_appends_model_frames():
    trajectory = [
        {"type": "goal", "content": "be helpful"},
        {"type": "mission", "content": "answer briefly"},
    ]
    gen = ScriptedGenerator(
        '<|belief|>I know this.<|action|>{"tool":"answer","text":"42"}'
    )
    result = step(trajectory, None, tokenizer=FakeTokenizer(), generate=gen)
    assert isinstance(result, StepResult)
    assert result.stopped_on == "end"
    assert isinstance(result.new_frames, Trajectory)
    types_short = [f["type"] for f in result.new_frames.to_dict()]
    assert types_short == ["belief", "action", "end"]
 
 
def test_step_extended_trajectory_starts_with_input():
    trajectory = [
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
    ]
    gen = ScriptedGenerator('<|action|>{"tool":"answer","text":"ok"}')
    result = step(trajectory, None, tokenizer=FakeTokenizer(), generate=gen)
    extended_dicts = result.trajectory.to_dict()
    assert extended_dicts[:2] == trajectory
 
 
def test_step_with_tools_does_not_include_tools_in_returned_trajectory():
    trajectory = [
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
    ]
    tools = [{
        "name": "read_file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        }
    }]
    gen = ScriptedGenerator('<|action|>{"tool":"answer","text":"ok"}')
    result = step(trajectory, tools, tokenizer=FakeTokenizer(), generate=gen)
    types = [f["type"] for f in result.trajectory.to_dict()]
    assert "obs" not in types

 
def test_step_result_to_dict_is_json_serializable():
    trajectory = [
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
    ]
    gen = ScriptedGenerator('<|action|>{"tool":"answer","text":"ok"}')
    result = step(trajectory, None, tokenizer=FakeTokenizer(), generate=gen)
    d = result.to_dict()
    serialized = json.dumps(d)
    assert "trajectory" in d
    assert "new_frames" in d
    assert "stopped_on" in d
    assert "raw_text" in d
    assert json.loads(serialized) == d
 

def test_step_captures_parse_error():
    trajectory = [
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
    ]
    gen = ScriptedGenerator("this is not a frame")
    result = step(trajectory, None, tokenizer=FakeTokenizer(), generate=gen)
    assert result.stopped_on.startswith("parse_error")
    assert len(result.new_frames) == 0
    assert result.trajectory.to_dict() == trajectory
 
 
def test_step_strict_mode_rejects_runtime_owned_frame():
    trajectory = [
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
    ]
    gen = ScriptedGenerator('<|result|>{"ok":1,"value":1}')
    result = step(
        trajectory,
        None,
        tokenizer=FakeTokenizer(),
        generate=gen,
        strict=True,
    )
    assert "parse_error" in result.stopped_on
    assert len(result.new_frames) == 0