from __future__ import annotations

import json

from telos.evaluation.harness.task import BenchmarkResult, BenchmarkRunMeta, EvalTask, TaskResult, TaskTiming, TaskTokens


def test_eval_task_from_row():
    task = EvalTask.from_row(
        {"id": "ex-1", "domain": "code", "frames": '[{"type":"goal","content":"x"}]'},
        gold={"k": 1},
    )
    assert task.task_id == "ex-1"
    assert task.frames[0]["type"] == "goal"
    assert task.gold == {"k": 1}


def test_benchmark_result_json():
    br = BenchmarkResult(
        meta=BenchmarkRunMeta("fmt", "m", "telos", "merged", "ds", "eval", 1),
        metrics={"n": 1},
        tasks=[TaskResult("a", "d", True, tokens=TaskTokens(total_tokens=10))],
    )
    data = json.loads(br.to_json())
    assert data["metrics"]["n"] == 1
    assert data["tasks"][0]["task_id"] == "a"
