"""run eval tasks into the benchmark envelope."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable

from telos.evaluation.harness.aggregate import aggregate_efficiency
from telos.evaluation.harness.task import BenchmarkResult, BenchmarkRunMeta, EvalTask, TaskResult

TaskEvaluator = Callable[[EvalTask], TaskResult]


def run_tasks(
    meta: BenchmarkRunMeta,
    tasks: Iterable[EvalTask],
    evaluate: TaskEvaluator,
    *,
    load_sec: float = 0.0,
    extra_metrics: dict[str, Any] | None = None,
    desc: str | None = None,
) -> BenchmarkResult:
    task_list = list(tasks)
    load_each = load_sec / len(task_list) if load_sec and task_list else 0.0
    it = task_list
    if desc:
        try:
            from tqdm import tqdm

            it = tqdm(task_list, desc=desc, unit="task")
        except ImportError:
            pass

    results: list[TaskResult] = []
    for task in it:
        t0 = time.perf_counter()
        tr = evaluate(task)
        if tr.timing.total_sec <= 0:
            tr.timing.total_sec = time.perf_counter() - t0
        if load_each and tr.timing.load_sec <= 0:
            tr.timing.load_sec = load_each
        results.append(tr)

    return BenchmarkResult(
        meta=meta,
        metrics=aggregate_efficiency(results, extra_metrics),
        tasks=results,
    )


def write_benchmark(output_dir: Path, result: BenchmarkResult) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = output_dir / "summary.json"
    details = output_dir / "details.jsonl"
    summary.write_text(result.to_json())
    with details.open("w") as f:
        for t in result.tasks:
            f.write(json.dumps(asdict(t)) + "\n")
    return summary, details
