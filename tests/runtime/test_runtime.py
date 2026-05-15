import pytest
from telos.constants import FrameType
from telos.runtime import RunResult, Tool, ToolError, ToolRegistry, run
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

class ScriptedGenerator:
    """a generator that returns a pre-set text response."""
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.call_count = 0
 
    def __call__(self, input_ids, stop_token_id, max_new_tokens):
        if not self.responses:
            raise AssertionError("ScriptedGenerator exhausted")
        text = self.responses.pop(0)
        self.call_count += 1
        ids = [ord(c) for c in text] + [stop_token_id]
        if len(ids) > max_new_tokens:
            ids = ids[:max_new_tokens]
        return ids
 
def _trivial_schema(name, **props):
    return {
        "name": name,
        "parameters": {
            "type": "object",
            "properties": {k: {"type": "string"} for k in props},
            "required": list(props.keys()),
        },
    }

def test_registry_register_and_get():
    reg = ToolRegistry()
    reg.register(Tool("ping", lambda: "pong", _trivial_schema("ping")))
    assert reg.list_names() == ["ping"]
    assert reg.get("ping").name == "ping"

def test_registry_rejects_duplicate():
    reg = ToolRegistry()
    reg.register(Tool("ping", lambda: "pong", _trivial_schema("ping")))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(Tool("ping", lambda: "pong", _trivial_schema("ping")))
 
 
def test_registry_unknown_tool_raises():
    reg = ToolRegistry()
    with pytest.raises(ToolError, match="unknown tool"):
        reg.call("nope", {})
 
 
def test_registry_wraps_type_errors():
    reg = ToolRegistry()
    reg.register(Tool("needs_x", lambda x: x, _trivial_schema("needs_x", x="")))
    with pytest.raises(ToolError, match="bad arguments"):
        reg.call("needs_x", {"y": 1})

def test_registry_schemas_returns_list():
    reg = ToolRegistry()
    reg.register(Tool("a", lambda: 1, _trivial_schema("a")))
    reg.register(Tool("b", lambda: 2, _trivial_schema("b")))
    schemas = reg.schemas()
    assert len(schemas) == 2
    assert {s["name"] for s in schemas} == {"a", "b"}
 
 
def test_registry_calls_tool_successfully():
    reg = ToolRegistry()
    reg.register(Tool("upper", lambda text: text.upper(), _trivial_schema("upper", text="")))
    assert reg.call("upper", {"text": "hi"}) == "HI"


def test_registry_callable_invokes_tool():
    reg = ToolRegistry()
    reg.register(Tool("upper", lambda text: text.upper(), _trivial_schema("upper", text="")))
    assert reg("upper", {"text": "hi"}) == "HI"


def test_run_terminal_answer():
    """Model emits an answer action; run() stops with that answer."""
    reg = ToolRegistry()
    # No non-terminal tools needed; the model goes straight to answer.
 
    gen = ScriptedGenerator([
        '<|action|>{"tool":"answer","text":"42"}',
    ])
 
    initial = Trajectory([
        {"type": "goal", "content": "be helpful"},
        {"type": "mission", "content": "What is the answer?"},
    ])
    result = run(initial, reg, tokenizer=FakeTokenizer(), generate=gen)
 
    assert result.stopped_on == "terminal_action"
    assert result.final_answer == "42"
    assert result.iterations == 1
 
 
def test_run_terminal_fail():
    reg = ToolRegistry()
    gen = ScriptedGenerator([
        '<|action|>{"tool":"fail","reason":"cannot answer"}',
    ])
    initial = Trajectory([
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
    ])
    result = run(initial, reg, tokenizer=FakeTokenizer(), generate=gen)
 
    assert result.stopped_on == "terminal_action"
    assert result.final_answer is None  # only answer populates final_answer
 
 
def test_run_executes_tool_then_answers():
    """Model issues a tool call, gets a result, then answers."""
    reg = ToolRegistry()
    reg.register(Tool(
        "get_temp",
        lambda city: f"18C in {city}",
        _trivial_schema("get_temp", city=""),
    ))
 
    gen = ScriptedGenerator([
        '<|action|>{"tool":"get_temp","city":"Paris"}',
        '<|belief|>It is 18C in Paris.<|action|>{"tool":"answer","text":"18C"}',
    ])
 
    initial = Trajectory([
        {"type": "goal", "content": "weather assistant"},
        {"type": "mission", "content": "weather in Paris"},
    ])
    result = run(initial, reg, tokenizer=FakeTokenizer(), generate=gen)
    print(result.to_dict())
    assert result.stopped_on == "terminal_action"
    assert result.final_answer == "18C"
    assert result.iterations == 2
 
    # the trajectory should contain the tool result.
    found_result = False
    for f in result.trajectory:
        if f.type is FrameType.RESULT and isinstance(f.content, dict):
            if f.content.get("value") == "18C in Paris":
                found_result = True
                break
    assert found_result
 
 
def test_run_handles_tool_error():
    """when a tool raises ToolError, the runtime emits ok=0 and lets the model react."""
    reg = ToolRegistry()
 
    def broken(path):
        raise ToolError(f"no such file: {path}")
 
    reg.register(Tool("read_file", broken, _trivial_schema("read_file", path="")))
 
    gen = ScriptedGenerator([
        '<|action|>{"tool":"read_file","path":"/nope"}',
        '<|belief|>File missing.<|action|>{"tool":"answer","text":"file not found"}',
    ])
 
    initial = Trajectory([
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "read /nope"},
    ])
    result = run(initial, reg, tokenizer=FakeTokenizer(), generate=gen)
 
    assert result.stopped_on == "terminal_action"
    # the result frame should have ok=0.
    error_result = None
    for f in result.trajectory:
        if f.type is FrameType.RESULT and isinstance(f.content, dict):
            if f.content.get("ok") == 0:
                error_result = f
                break
    assert error_result is not None
    assert "no such file" in error_result.content["value"]


def test_run_stops_at_max_iterations():
    """If the model keeps issuing non-terminal actions, we cap it."""
    reg = ToolRegistry()
    reg.register(Tool("loop", lambda: "again", _trivial_schema("loop")))
 
    # Generator always returns the same non-terminal action.
    class InfiniteLoop:
        end_id_marker = "<|end|>"
 
        def __call__(self, input_ids, stop_token_id, max_new_tokens):
            text = '<|action|>{"tool":"loop"}'
            ids = [ord(c) for c in text] + [stop_token_id]
            return ids
 
    initial = Trajectory([
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
    ])
    result = run(
        initial, reg,
        tokenizer=FakeTokenizer(),
        generate=InfiniteLoop(),
        max_iterations=3,
    )
 
    assert result.stopped_on == "max_iterations"
    assert result.iterations == 3
 
 
def test_run_stops_on_parse_error():
    reg = ToolRegistry()
    gen = ScriptedGenerator([
        "not a valid frame at all",
    ])
    initial = Trajectory([
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
    ])
    result = run(initial, reg, tokenizer=FakeTokenizer(), generate=gen)
 
    assert result.stopped_on.startswith("parse_error")
    assert result.iterations == 1
 
 
def test_run_stops_on_no_action():
    """If the model emits only belief/think and no action, run stops."""
    reg = ToolRegistry()
    gen = ScriptedGenerator([
        "<|belief|>I have no idea what to do.",
    ])
    initial = Trajectory([
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
    ])
    result = run(initial, reg, tokenizer=FakeTokenizer(), generate=gen)
 
    assert result.stopped_on == "no_action"

def test_run_executes_batched_actions_in_one_step():
    """Model emits two actions in one block; both are executed and
    results appended in order."""
    reg = ToolRegistry()
    calls = []
 
    def echo(value):
        calls.append(value)
        return value
 
    reg.register(Tool("echo", echo, _trivial_schema("echo", value="")))
 
    gen = ScriptedGenerator([
        '<|action|>{"tool":"echo","value":"a"}<|action|>{"tool":"echo","value":"b"}',
        '<|action|>{"tool":"answer","text":"done"}',
    ])
 
    initial = Trajectory([
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
    ])
    result = run(initial, reg, tokenizer=FakeTokenizer(), generate=gen)
 
    assert result.stopped_on == "terminal_action"
    assert calls == ["a", "b"]

def test_run_result_to_dict_is_serializable():
    import json as _json
    reg = ToolRegistry()
    gen = ScriptedGenerator([
        '<|action|>{"tool":"answer","text":"ok"}',
    ])
    initial = Trajectory([
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
    ])
    result = run(initial, reg, tokenizer=FakeTokenizer(), generate=gen)
    d = result.to_dict()
    serialized = _json.dumps(d)
    assert _json.loads(serialized) == d