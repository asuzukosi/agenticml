"""run-level aggregates from per-task results."""

from __future__ import annotations

from typing import Any

from telos.evaluation.harness.task import TaskResult


def aggregate_efficiency(tasks: list[TaskResult], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    n = len(tasks)
    if n == 0:
        return {"n": 0, **(extra or {})}

    def mean(vals: list[float]) -> float:
        return sum(vals) / len(vals)

    ok = [t for t in tasks if t.success is True]
    out = {
        "n": n,
        "avg_total_tokens": mean([float(t.tokens.total_tokens) for t in tasks]),
        "avg_generated_tokens": mean([float(t.tokens.generated_tokens) for t in tasks]),
        "avg_wall_sec": mean([t.timing.total_sec for t in tasks]),
        "avg_inference_sec": mean([t.timing.inference_sec for t in tasks]),
        "tokens_per_success": mean([float(t.tokens.total_tokens) for t in ok]) if ok else None,
        "sec_per_success": mean([t.timing.total_sec for t in ok]) if ok else None,
        "n_success": len(ok),
    }
    scored = [t for t in tasks if t.success is not None]
    if scored:
        out["success_rate"] = sum(t.success for t in scored) / len(scored)
    if extra:
        out.update(extra)
    return out
