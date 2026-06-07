"""shared toolbench path helpers and upstream name normalization."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from agenticml.evaluation.benchmarks.common import repo_root
from agenticml.evaluation.benchmarks.toolbench.subset import (
    TOOLBENCH_ROOT_REL,
    default_data_root,
    ensure_toolbench_data,
)

TOOLENV_REL = Path("data/toolenv/tools")
RESPONSE_EXAMPLES_REL = Path("data/toolenv/response_examples")
RESPONSE_CACHE_REL = Path("data/tool_response_cache")


def toolenv_path(data_root: Path | None = None) -> Path:
    return (data_root or default_data_root()) / TOOLENV_REL


def response_examples_path(data_root: Path | None = None) -> Path:
    return (data_root or default_data_root()) / RESPONSE_EXAMPLES_REL


def response_cache_path(data_root: Path | None = None) -> Path:
    return (data_root or default_data_root()) / RESPONSE_CACHE_REL


def ensure_toolenv(data_root: Path | None = None) -> Path:
    root = ensure_toolbench_data(data_root)
    tools = toolenv_path(root)
    if not tools.is_dir():
        raise FileNotFoundError(
            f"toolbench toolenv not found at {tools}; "
            "download OpenBMB data.zip (see recipe.md §4): "
            "hf download nullwwg/toolbench-data data.zip --repo-type dataset --local-dir third_party/ToolBench"
        )
    return root


def ensure_toolbench_on_path() -> Path:
    """prepend toolbench submodule root so `import toolbench` resolves."""
    root = repo_root() / TOOLBENCH_ROOT_REL
    if not (root / "toolbench").is_dir():
        raise FileNotFoundError(
            f"toolbench package not found at {root / 'toolbench'}; "
            "run: git submodule update --init third_party/ToolBench"
        )
    for sub in (root, root / "toolbench" / "inference"):
        s = str(sub)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root


def standardize(string: str) -> str:
    res = re.compile(r"[^\u4e00-\u9fa5^a-z^A-Z^0-9^_]")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+", "_", string).lower()
    while string and string[0] == "_":
        string = string[1:]
    while string and string[-1] == "_":
        string = string[:-1]
    if string and string[0].isdigit():
        string = "get_" + string
    return string


def change_name(name: str) -> str:
    reserved = {"from", "class", "return", "false", "true", "id", "and"}
    if name in reserved:
        return f"is_{name}"
    return name


def get_white_list(tool_root: Path) -> dict[str, dict[str, str]]:
    white_list: dict[str, dict[str, str]] = {}
    if not tool_root.is_dir():
        return white_list
    for cate in tool_root.iterdir():
        if not cate.is_dir():
            continue
        for file in cate.iterdir():
            if file.suffix != ".json":
                continue
            standard_tool_name = file.stem
            js_data = json.loads(file.read_text())
            origin_tool_name = js_data["tool_name"]
            white_list[standardize(origin_tool_name)] = {
                "description": js_data.get("tool_description", ""),
                "standard_tool_name": standard_tool_name,
            }
    return white_list


def contain(candidate_list: list[str], white_list: dict[str, dict[str, str]]) -> list[dict[str, str]] | bool:
    output: list[dict[str, str]] = []
    for cand in candidate_list:
        if cand not in white_list:
            return False
        output.append(white_list[cand])
    return output
