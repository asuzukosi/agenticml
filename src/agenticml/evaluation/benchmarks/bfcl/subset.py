"""load pinned bfcl v4 subset via upstream bfcl_eval."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agenticml.evaluation.benchmarks.common import repo_root

BFCL_ROOT_REL = Path("third_party/gorilla/berkeley-function-call-leaderboard")
SUBSET_SOURCE = "agenticml.evaluation.benchmarks.bfcl.subset:SUBSET_IDS"

# pinned bfcl v4 subset (45 cases, seed 42); excludes irrelevance/relevance.
# multi_turn_base: 8 fast cases (2–3 turns, filesystem/trading/message). slow
# vehicle/travel sagas (67, 88, 97, 154) and 24 replaced with shorter ids.
SUBSET_IDS: dict[str, list[str]] = {
    "simple_python": [
        "simple_python_114",
        "simple_python_12",
        "simple_python_125",
        "simple_python_140",
        "simple_python_279",
        "simple_python_327",
        "simple_python_346",
        "simple_python_377",
        "simple_python_379",
        "simple_python_52",
        "simple_python_57",
        "simple_python_71",
    ],
    "parallel": [
        "parallel_108",
        "parallel_151",
        "parallel_22",
        "parallel_8",
    ],
    "multiple": [
        "multiple_23",
        "multiple_55",
        "multiple_59",
        "multiple_7",
    ],
    "parallel_multiple": [
        "parallel_multiple_129",
        "parallel_multiple_143",
        "parallel_multiple_154",
        "parallel_multiple_6",
    ],
    "live_simple": [
        "live_simple_101-60-0",
        "live_simple_112-68-0",
        "live_simple_142-94-1",
        "live_simple_214-117-6",
        "live_simple_229-120-0",
        "live_simple_3-2-1",
        "live_simple_81-42-0",
    ],
    "live_multiple": [
        "live_multiple_318-132-8",
        "live_multiple_440-144-0",
        "live_multiple_569-155-9",
        "live_multiple_689-164-5",
        "live_multiple_696-164-12",
        "live_multiple_865-182-2",
    ],
    "multi_turn_base": [
        "multi_turn_base_7",
        "multi_turn_base_13",
        "multi_turn_base_23",
        "multi_turn_base_26",
        "multi_turn_base_29",
        "multi_turn_base_91",
        "multi_turn_base_100",
        "multi_turn_base_104",
    ],
}

# agenticml is tool-first; bfcl irrelevance/relevance expect no tool call
EXCLUDED_CATEGORIES = frozenset({"irrelevance", "relevance", "live_irrelevance"})


def ensure_bfcl_on_path() -> Path:
    """prepend gorilla bfcl package root so `import bfcl_eval` works."""
    root = repo_root() / BFCL_ROOT_REL
    if not root.is_dir():
        raise FileNotFoundError(
            f"bfcl submodule not found at {root}; run: git submodule update --init third_party/gorilla"
        )
    try:
        import tree_sitter  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "bfcl needs tree_sitter. install gorilla editable:\n"
            "  pip install -e third_party/gorilla/berkeley-function-call-leaderboard"
        ) from e
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def ensure_bfcl_scoring() -> None:
    """gorilla scoring imports the full bfcl_eval package (eval_runner -> model_config)."""
    ensure_bfcl_on_path()
    try:
        from bfcl_eval.eval_checker.eval_runner import evaluate_task  # noqa: F401
    except ImportError as e:
        hint = "  pip install soundfile" if "soundfile" in str(e) else (
            "  pip install -e third_party/gorilla/berkeley-function-call-leaderboard"
        )
        raise ImportError(
            f"bfcl scoring import failed ({e}). try:\n{hint}\n"
            "see recipe.md §4 for full eval setup"
        ) from e


def load_subset_id_map(
    id_map: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    raw = id_map if id_map is not None else SUBSET_IDS
    out: dict[str, list[str]] = {}
    for cat, ids in raw.items():
        if not isinstance(cat, str) or not isinstance(ids, list) or not ids:
            continue
        if cat in EXCLUDED_CATEGORIES:
            continue
        if not all(isinstance(i, str) for i in ids):
            raise ValueError(f"{SUBSET_SOURCE}: {cat!r} ids must be strings")
        out[cat] = ids
    if not out:
        raise ValueError(f"{SUBSET_SOURCE}: no categories with ids")
    return out


@dataclass(frozen=True)
class BFCLSubset:
    categories: list[str]
    entries: list[dict[str, Any]]
    id_map: dict[str, list[str]]
    source: str = SUBSET_SOURCE


def load_subset(id_map: dict[str, list[str]] | None = None) -> BFCLSubset:
    """load prompt entries for the pinned subset (same schema as bfcl generate)."""
    ensure_bfcl_on_path()
    from bfcl_eval.utils import load_dataset_entry

    resolved = load_subset_id_map(id_map)
    categories: list[str] = []
    entries: list[dict[str, Any]] = []
    for category, test_ids in resolved.items():
        if not test_ids:
            continue
        entries.extend(
            [entry for entry in load_dataset_entry(category) if entry["id"] in test_ids]
        )
        categories.append(category)
    if len(entries) != sum(len(v) for v in resolved.values()):
        raise RuntimeError(
            f"expected {sum(len(v) for v in resolved.values())} entries, got {len(entries)}"
        )
    return BFCLSubset(categories=categories, entries=entries, id_map=resolved)
