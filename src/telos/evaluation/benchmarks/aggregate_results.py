"""aggregate per-suite summary.json envelopes into a published results table."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from telos.evaluation.benchmarks.common import repo_root
from telos.evaluation.benchmarks.run_all import MATRIX_FORMATS, MATRIX_SUITES

# (primary metric keys in priority order, secondary metric key, primary label, secondary label)
_SUITE_SPEC: dict[str, tuple[tuple[str, ...], str, str, str]] = {
    "format_validity": (("valid_rate", "format_validity_primary"), "parse_rate", "valid_rate", "parse_rate"),
    "bfcl": (("bfcl_primary", "accuracy", "bfcl_accuracy"), "avg_retry_count", "accuracy", "avg_retry_count"),
    "toolbench": (("toolbench_primary", "pass_rate", "structural_pass_rate"), "avg_steps", "pass_rate", "avg_steps"),
    "swe": (("swe_primary", "resolved_rate"), "avg_iterations", "resolved_rate", "avg_iterations"),
}

_FORMATS = tuple(MATRIX_FORMATS)
_SUITES = tuple(MATRIX_SUITES)


@dataclass(frozen=True)
class ResultRow:
    suite: str
    format: str
    model: str
    num_run: int
    sample_seed: int
    primary: Optional[float]
    primary_label: str
    secondary: Optional[float]
    secondary_label: str
    avg_total_tokens: Optional[float]
    tokens_per_success: Optional[float]
    avg_wall_sec: Optional[float]
    summary_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _metric(metrics: dict[str, Any], keys: tuple[str, ...]) -> Optional[float]:
    for key in keys:
        val = metrics.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _load_summary(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def discover_summaries(results_root: Path) -> list[tuple[str, str, Path]]:
    """return (suite, format, path) for each summary.json under results_root."""
    found: list[tuple[str, str, Path]] = []
    if not results_root.is_dir():
        return found
    for suite_dir in sorted(results_root.iterdir()):
        if not suite_dir.is_dir():
            continue
        suite = suite_dir.name
        for fmt in _FORMATS:
            path = suite_dir / fmt / "summary.json"
            if path.is_file():
                found.append((suite, fmt, path))
    return found


def row_from_summary(suite: str, fmt: str, path: Path, payload: dict[str, Any]) -> ResultRow:
    meta = payload.get("meta") or {}
    metrics = payload.get("metrics") or {}
    spec = _SUITE_SPEC.get(suite, ((f"{suite}_primary",), "", suite, ""))
    primary_keys, secondary_key, primary_label, secondary_label = spec
    primary = _metric(metrics, primary_keys)
    if primary is None and suite == "swe" and int(metrics.get("n_success") or 0) == 0:
        primary = 0.0
    return ResultRow(
        suite=suite,
        format=fmt,
        model=str(meta.get("model") or ""),
        num_run=int(meta.get("num_run") or metrics.get("n") or 0),
        sample_seed=int(meta.get("sample_seed") or 42),
        primary=primary,
        primary_label=primary_label,
        secondary=_metric(metrics, (secondary_key,)) if secondary_key else None,
        secondary_label=secondary_label,
        avg_total_tokens=_metric(metrics, ("avg_total_tokens",)),
        tokens_per_success=_metric(metrics, ("tokens_per_success",)),
        avg_wall_sec=_metric(metrics, ("avg_wall_sec",)),
        summary_path=str(path),
    )


def load_result_rows(results_root: Path) -> list[ResultRow]:
    rows: list[ResultRow] = []
    for suite, fmt, path in discover_summaries(results_root):
        payload = _load_summary(path)
        if payload is None:
            continue
        rows.append(row_from_summary(suite, fmt, path, payload))
    return rows


def _fmt_rate(val: Optional[float]) -> str:
    if val is None:
        return "—"
    return f"{val:.1%}"


def _fmt_num(val: Optional[float], *, digits: int = 1) -> str:
    if val is None:
        return "—"
    return f"{val:.{digits}f}"


_NUMERIC_LABELS = frozenset({"avg_iterations", "avg_retry_count", "avg_steps"})


def _fmt_primary(row: ResultRow) -> str:
    if row.primary_label in _NUMERIC_LABELS:
        return _fmt_num(row.primary)
    return _fmt_rate(row.primary)


def _fmt_secondary(row: ResultRow) -> str:
    if row.secondary_label in _NUMERIC_LABELS:
        return _fmt_num(row.secondary)
    return _fmt_rate(row.secondary)


def render_markdown(rows: list[ResultRow], *, results_root: Path) -> str:
    lines = [
        "# benchmark results",
        "",
        "aggregated from `results/benchmarks/<suite>/<format>/summary.json` envelopes.",
        f"regenerate: `telos eval-aggregate-results` (reads `{results_root}`).",
        "",
        "## matrix",
        "",
        "| suite | format | model | n | primary | secondary | avg_tokens | tok/success | avg_wall_sec |",
        "|-------|--------|-------|---|---------|-----------|------------|-------------|--------------|",
    ]
    order = {
        (s, f): i * len(_FORMATS) + j
        for i, s in enumerate(_SUITES)
        for j, f in enumerate(_FORMATS)
    }
    sorted_rows = sorted(rows, key=lambda r: order.get((r.suite, r.format), 99))

    if not sorted_rows:
        lines.append("| — | — | — | — | — | — | — | — | — |")
    else:
        for row in sorted_rows:
            model_short = row.model.split("/")[-1] if row.model else "—"
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.suite,
                        row.format,
                        model_short,
                        str(row.num_run),
                        f"{_fmt_primary(row)} ({row.primary_label})",
                        f"{_fmt_secondary(row)} ({row.secondary_label})" if row.secondary_label else "—",
                        _fmt_num(row.avg_total_tokens, digits=0),
                        _fmt_num(row.tokens_per_success, digits=0),
                        _fmt_num(row.avg_wall_sec),
                    ]
                )
                + " |"
            )

    present = {(r.suite, r.format) for r in rows}
    missing = [f"{s}/{f}" for s in _SUITES for f in _FORMATS if (s, f) not in present]
    lines.extend(["", "## coverage", ""])
    if missing:
        lines.append("missing cells (not run yet):")
        for cell in missing:
            lines.append(f"- `{cell}`")
    else:
        lines.append("all default matrix cells present (4 suites × 2 formats).")

    lines.extend(
        [
            "",
            "## per-suite metrics",
            "",
            "| suite | primary | secondary |",
            "|-------|---------|-----------|",
            "| format_validity | valid_rate | parse_rate |",
            "| bfcl | accuracy | avg_retry_count |",
            "| toolbench | pass_rate (structural) | avg_steps |",
            "| swe | resolved_rate | avg_iterations |",
            "",
            "## sources",
            "",
        ]
    )
    for row in sorted_rows:
        rel = Path(row.summary_path)
        try:
            rel = rel.relative_to(repo_root())
        except ValueError:
            pass
        lines.append(f"- `{row.suite}` / `{row.format}`: `{rel}`")
    lines.append("")
    return "\n".join(lines)


def aggregate_results(
    results_root: Optional[Path] = None,
    *,
    markdown_path: Optional[Path] = None,
    json_path: Optional[Path] = None,
) -> list[ResultRow]:
    root = results_root or (repo_root() / "results" / "benchmarks")
    rows = load_result_rows(root)

    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "results_root": str(root),
            "rows": [r.to_dict() for r in rows],
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n")

    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(rows, results_root=root))

    return rows
