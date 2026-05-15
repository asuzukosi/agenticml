"""
telos sdk for stateless trajectory advancement
the public entry point is `step()`. given a trajectory and a tool schema, `step()` advances the trajectory by one generation cycle:
it injects the tool definitions into the prompt, calls the model, parses the model's output, and returns the extended trajectory.

sdk does not execute tools, when the model emits an action, the calleer si responsible for 
running the tool and appending the result frame before calling `step()` again.
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Callable, Optional, Iterable, Union
from telos.frames import Frame, parse, render
from telos.constants import FrameType
from telos.trajectory import Trajectory, FrameLike




def _ts_type(spec: dict) -> str:
    """translate a JSON Schema fragment to a rough TypeScript type."""
    t = spec.get("type")
    if t == "string":
        if "enum" in spec:
            return " | ".join(repr(v) for v in spec["enum"])
        return "string"
    if t in ("integer", "number"):
        return "number"
    if t == "boolean":
        return "boolean"
    if t == "array":
        item_type = _ts_type(spec.get("items", {}))
        return f"{item_type}[]"
    if t == "object":
        return "object"
    return "any"

def _render_tool_schema(tools: list[dict[str, Any]]) -> str:
    """ render an openai-style tool schema list to typescript naemspace """
    if not tools:
        return ""
    lines = ["tools:", "namespace tools {"]
    for t in tools:
        name = t.get("name", "<unnamed>")
        description = t.get("description", "")
        parameters: dict[str, Any] = t.get("parameters", {})
        properties: dict[str, Any] = parameters.get("properties", {})
        required: list[str] = parameters.get("required", [])
        
        if description:
            lines.append(f" // {description}")
        # emit a tyepscript-like signature
        if properties:
            lines.append(f" type {name} = (_: {{)")
            for pname, pspec in properties.items():
                ptype = _ts_type(pspec)
                optional  = "" if pname in required else "?"
                pdesc = pspec.get("description")
                if pdesc:
                    lines.append(f"   // {pdesc}")
                lines.append(f"   {pname}{optional}: {ptype},")
            lines.append("  }) => any;")
        else:
            lines.append(f" type {name} = () => any;")
    lines.append("}")
    return "\n".join(lines)

Generator  = Callable[[list[int], int, int], list[int]]

@dataclass
class StepResult:
    """
    return value of `step()`.
    """
    trajectory: Trajectory
    new_frames: Trajectory
    stopped_on: str
    raw_text: str

    def to_dict(self) -> dict:
        """a dictionary representation of the step result."""
        return {
            "trajectory": self.trajectory.to_dict(),
            "new_frames": self.new_frames.to_dict(),
            "stopped_on": self.stopped_on,
            "raw_text": self.raw_text,
        }
    
    def to_json(self) -> str:
        """a JSON representation of the step result."""
        return json.dumps(self.to_dict())

TrajectoryInput = Union[Trajectory, Iterable[FrameLike]]
 
def step(
    trajectory: TrajectoryInput,
    tools: Optional[list[dict]] = None,
    *,
    tokenizer,
    generate: Generator,
    max_new_tokens: int = 512,
    strict: bool = False,
) -> StepResult:
    """advance the trajectory by one generation cycle."""
    input_traj = trajectory if isinstance(trajectory, Trajectory) else Trajectory(trajectory)
 
    # build the prompt: original trajectory + tool definitions <|obs|>.
    prompt_frames = input_traj.to_frames()
    if tools:
        tool_body = _render_tool_schema(tools)
        prompt_frames.append(Frame(type=FrameType.OBS, content=tool_body))
 
    prompt_text = render(prompt_frames)
    prompt_ids = tokenizer.encode(prompt_text)
 
    # generate.
    new_ids = generate(prompt_ids, tokenizer.end_id, max_new_tokens)

    # determine stop reason.
    if new_ids and new_ids[-1] == tokenizer.end_id:
        stopped_on = "end"
    elif len(new_ids) >= max_new_tokens:
        stopped_on = "max_tokens"
    else:
        stopped_on = "other"
 
    # decode and parse the new tokens.
    raw_text = tokenizer.decode(new_ids)
    try:
        new_frame_objs = parse(raw_text, strict=strict)
    except Exception as e:
        return StepResult(
            trajectory=input_traj,
            new_frames=Trajectory(),
            stopped_on=f"parse_error: {e}",
            raw_text=raw_text,
        )
 
    new_traj = Trajectory(new_frame_objs)
    extended = input_traj + new_traj
 
    return StepResult(
        trajectory=extended,
        new_frames=new_traj,
        stopped_on=stopped_on,
        raw_text=raw_text,
    )