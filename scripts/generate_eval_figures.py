#!/usr/bin/env python3
"""regenerate docs/figures/*.png from results/benchmarks/runpod_jun2026_aggregate.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AGG_PATH = ROOT / "results/benchmarks/runpod_jun2026_aggregate.json"
OUT_DIR = ROOT / "docs/figures"

AGENTICML_C = "#4C72B0"
CHATML_C = "#DD8452"


def parse_frac(s: str) -> float:
    a, b = s.split("/")
    return 100 * int(a) / int(b)


def main() -> None:
    agg = json.loads(AGG_PATH.read_text())
    cat = agg["bfcl_per_category"]
    fail = agg["bfcl_failure_shapes"]["agenticml"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    suites = ["Format Validity", "BFCL", "ToolBench"]
    agenticml_vals = [100, 4.4, 0]
    chatml_vals = [100, 60.0, 0]
    x = np.arange(len(suites))
    ax.bar(x - w / 2, agenticml_vals, w, label="AgenticML", color=AGENTICML_C)
    ax.bar(x + w / 2, chatml_vals, w, label="ChatML", color=CHATML_C)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Benchmark primary metrics")
    ax.set_xticks(x, suites)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")
    for i, (a, c) in enumerate(zip(agenticml_vals, chatml_vals)):
        if a > 0:
            ax.text(i - w / 2, a + 1, f"{a:.1f}%", ha="center", va="bottom", fontsize=8)
        if c > 0:
            ax.text(i + w / 2, c + 1, f"{c:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.savefig(OUT_DIR / "suite_primary_metrics.png")
    plt.close(fig)

    categories = list(cat["agenticml"].keys())
    labels = [c.replace("_", "\n") for c in categories]
    a_vals = [parse_frac(cat["agenticml"][c]) for c in categories]
    c_vals = [parse_frac(cat["chatml"][c]) for c in categories]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(categories))
    ax.bar(x - w / 2, a_vals, w, label="AgenticML", color=AGENTICML_C)
    ax.bar(x + w / 2, c_vals, w, label="ChatML", color=CHATML_C)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("BFCL accuracy by category (n=45 subset)")
    ax.set_xticks(x, labels, rotation=0, ha="center", fontsize=8)
    ax.set_ylim(0, 105)
    ax.legend()
    for i, c in enumerate(categories):
        ax.text(i - w / 2, a_vals[i] + 2, cat["agenticml"][c], ha="center", va="bottom", fontsize=7)
        ax.text(i + w / 2, c_vals[i] + 2, cat["chatml"][c], ha="center", va="bottom", fontsize=7)
    fig.savefig(OUT_DIR / "bfcl_per_category.png")
    plt.close(fig)

    labels_f = ["json shape\n(rejected call)", "multi-turn\nall empty", "empty\nno tool call"]
    sizes = [fail["json_tool_call"], fail["multi_turn_all_empty"], fail["empty_no_tool_call"]]
    colors = ["#8172B3", "#CCB974", "#64B5CD"]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.pie(sizes, labels=labels_f, autopct="%1.0f%%", startangle=90, colors=colors)
    ax.set_title("AgenticML BFCL failure shapes\n(43 failed / 45 tasks)")
    fig.savefig(OUT_DIR / "agenticml_bfcl_failures.png")
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    formats = ["AgenticML", "ChatML"]
    tokens = [140018, 14268]
    wall = [223.9, 3.9]
    colors_bar = [AGENTICML_C, CHATML_C]
    ax1.bar(formats, tokens, color=colors_bar)
    ax1.set_ylabel("Avg tokens per task")
    ax1.set_title("BFCL token cost")
    ax1.set_yscale("log")
    for i, v in enumerate(tokens):
        ax1.text(i, v * 1.15, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax2.bar(formats, wall, color=colors_bar)
    ax2.set_ylabel("Avg wall sec per task")
    ax2.set_title("BFCL latency")
    for i, v in enumerate(wall):
        ax2.text(i, v + 5, f"{v:.1f}s", ha="center", va="bottom", fontsize=8)
    fig.suptitle("BFCL efficiency", y=1.02)
    fig.savefig(OUT_DIR / "bfcl_efficiency.png")
    plt.close(fig)

    print(f"wrote {len(list(OUT_DIR.glob('*.png')))} figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
