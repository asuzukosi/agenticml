from __future__ import annotations

import json
from pathlib import Path

from telos.evaluation.harness.aggregate import aggregate_efficiency
from telos.evaluation.harness.runner import run_tasks, write_benchmark
from telos.evaluation.harness.task import BenchmarkRunMeta, EvalTask, TaskResult, TaskTiming, TaskTokens


def test_aggregate_efficiency():
    tasks = [
        TaskResult("a", "d", True, tokens=TaskTokens(total_tokens=100), timing=TaskTiming(total_sec=1.0)),
        TaskResult("b", "d", False, tokens=TaskTokens(total_tokens=200), timing=TaskTiming(total_sec=2.0)),
    ]
    m = aggregate_efficiency(tasks, {"valid_rate": 0.5})
    assert m["n"] == 2
    assert m["valid_rate"] == 0.5
    assert m["avg_total_tokens"] == 150.0


def test_run_tasks(tmp_path: Path):
    meta = BenchmarkRunMeta("test", "m", "telos", "merged", "ds", "eval", 2)
    tasks = [EvalTask("1", "d", []), EvalTask("2", "d", [])]

    def evaluate(task: EvalTask) -> TaskResult:
        return TaskResult(task.task_id, task.domain, True, tokens=TaskTokens(total_tokens=50))

    result = run_tasks(meta, tasks, evaluate, load_sec=2.0)
    assert result.tasks[0].timing.load_sec == 1.0
    assert result.metrics["avg_total_tokens"] == 50.0

    summary, details = write_benchmark(tmp_path / "run", result)
    assert json.loads(summary.read_text())["metrics"]["n"] == 2
    assert len(details.read_text().strip().splitlines()) == 2
