"""tests for benchmark results aggregation."""

from __future__ import annotations

import json
from pathlib import Path

from telos.evaluation.benchmarks.aggregate_results import (
    aggregate_results,
    load_result_rows,
    render_markdown,
    row_from_summary,
)


def _write_summary(path: Path, *, suite: str, fmt: str, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "meta": {
                    "suite": suite,
                    "model": f"org/{fmt}-model",
                    "format": fmt,
                    "num_run": 2,
                    "sample_seed": 42,
                },
                "metrics": {"n": 2, **metrics},
                "tasks": [],
            }
        )
    )


def test_row_from_summary_primary_fallback(tmp_path: Path):
    path = tmp_path / "bfcl" / "telos" / "summary.json"
    _write_summary(
        path,
        suite="bfcl",
        fmt="telos",
        metrics={"accuracy": 0.6, "avg_retry_count": 3.5, "avg_total_tokens": 1000.0},
    )
    row = row_from_summary("bfcl", "telos", path, json.loads(path.read_text()))
    assert row.primary == 0.6
    assert row.secondary == 3.5
    assert row.avg_total_tokens == 1000.0


def test_aggregate_writes_markdown_and_json(tmp_path: Path):
    root = tmp_path / "benchmarks"
    _write_summary(
        root / "format_validity" / "telos" / "summary.json",
        suite="format_validity",
        fmt="telos",
        metrics={"valid_rate": 1.0, "parse_rate": 1.0, "avg_wall_sec": 10.0},
    )
    _write_summary(
        root / "swe" / "telos" / "summary.json",
        suite="swe",
        fmt="telos",
        metrics={"avg_iterations": 3.5, "avg_wall_sec": 200.0},
    )

    md = tmp_path / "benchmark_results.md"
    js = tmp_path / "aggregate.json"
    rows = aggregate_results(root, markdown_path=md, json_path=js)

    assert len(rows) == 2
    assert md.is_file()
    text = md.read_text()
    assert "format_validity" in text
    assert "swe/telos" in text or "missing cells" in text
    assert js.is_file()
    assert len(json.loads(js.read_text())["rows"]) == 2


def test_render_markdown_lists_missing_cells(tmp_path: Path):
    rows = load_result_rows(tmp_path)
    md = render_markdown(rows, results_root=tmp_path)
    assert "bfcl/telos" in md
