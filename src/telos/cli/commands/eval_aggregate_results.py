from __future__ import annotations

import argparse
import json
from pathlib import Path

from telos.evaluation.benchmarks.aggregate_results import aggregate_results
from telos.evaluation.benchmarks.common import repo_root


def main(argv: list[str] | None = None) -> None:
    root = repo_root()
    default_results = root / "results" / "benchmarks"
    p = argparse.ArgumentParser(
        description="aggregate benchmark summary.json files into a results table",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=default_results,
        help="root containing <suite>/<format>/summary.json",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=root / "docs" / "benchmark_results.md",
        help="markdown table output",
    )
    p.add_argument(
        "--json",
        type=Path,
        default=default_results / "aggregate.json",
        help="machine-readable aggregate",
    )
    args = p.parse_args(argv)

    rows = aggregate_results(
        args.results_dir,
        markdown_path=args.output,
        json_path=args.json,
    )
    print(f"wrote {len(rows)} rows -> {args.output}")
    print(f"wrote json -> {args.json}")
    print(json.dumps({"n_rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
