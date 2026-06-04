"""model backend protocol and per-step / per-run result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from telos.runtime.tools import ToolRegistry
from telos.sdk import StepResult
from telos.runtime.runtime import RunResult
from telos.trajectory import Trajectory


@dataclass
class BackendStepResult:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    inference_sec: float = 0.0
    step: Optional[StepResult] = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    new_messages: list[dict[str, Any]] = field(default_factory=list)
    stopped_on: str = ""
    raw_text: str = ""


@dataclass
class BackendRunResult:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    inference_sec: float = 0.0
    tool_sec: float = 0.0
    total_sec: float = 0.0
    run: Optional[RunResult] = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    stopped_on: str = ""
    iterations: int = 0
    final_answer: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ModelBackend(Protocol):
    """format-specific wrapper around sdk.step / runtime.run."""

    @property
    def format(self) -> str: ...

    def step(
        self,
        trajectory: Trajectory | list[dict],
        tools: Optional[list[dict]] = None,
        *,
        max_new_tokens: int = 512,
        strict: bool = True,
    ) -> BackendStepResult: ...

    def run(
        self,
        trajectory: Trajectory | list[dict],
        registry: ToolRegistry,
        *,
        max_iterations: int = 10,
        max_new_tokens: int = 512,
        strict: bool = True,
    ) -> BackendRunResult: ...
