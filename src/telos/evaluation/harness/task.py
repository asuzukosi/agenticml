"""eval task and benchmark result envelope."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class EvalTask:
    task_id: str
    domain: str
    frames: list[dict]
    gold: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict, *, gold: Optional[dict[str, Any]] = None) -> EvalTask:
        raw = row["frames"]
        frames = json.loads(raw) if isinstance(raw, str) else list(raw)
        return cls(str(row["id"]), str(row.get("domain") or "unknown"), frames, dict(gold or {}))


@dataclass
class TaskTiming:
    load_sec: float = 0.0
    inference_sec: float = 0.0
    tool_sec: float = 0.0
    total_sec: float = 0.0


@dataclass
class TaskTokens:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.total_tokens:
            self.total_tokens = self.prompt_tokens + self.generated_tokens


@dataclass
class TaskResult:
    task_id: str
    domain: str
    success: Optional[bool] = None
    metrics: dict[str, Any] = field(default_factory=dict)
    timing: TaskTiming = field(default_factory=TaskTiming)
    tokens: TaskTokens = field(default_factory=TaskTokens)
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkRunMeta:
    suite: str
    model: str
    format: str
    adapter_mode: str
    dataset: str
    split: str
    num_run: int
    sample_seed: int = 42
    adapter_id: Optional[str] = None


@dataclass
class BenchmarkResult:
    meta: BenchmarkRunMeta
    metrics: dict[str, Any] = field(default_factory=dict)
    tasks: list[TaskResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": asdict(self.meta),
            "metrics": self.metrics,
            "tasks": [asdict(t) for t in self.tasks],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
