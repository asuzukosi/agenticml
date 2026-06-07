"""shared swe-bench helpers: paths, bash tool schema, observation formatting."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from agenticml.evaluation.benchmarks.common import repo_root

MINISWE_ROOT_REL = Path("third_party/mini-swe-agent")
MINISWE_SRC_REL = MINISWE_ROOT_REL / "src"
SWEBENCH_ROOT_REL = Path("third_party/SWE-bench")

BASH_TOOL_SCHEMA: dict[str, Any] = {
    "name": "bash",
    "description": "Execute a bash command in the SWE-bench container.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute",
            }
        },
        "required": ["command"],
    },
}

DEFAULT_GOAL = (
    "You are a software engineer that fixes bugs by running bash commands in a linux shell. "
    "Issue at least one bash command per step, read the output, then continue until the patch is ready to submit."
)

_OUTPUT_TRUNC = 10_000


def ensure_swebench_importable() -> None:
    """require the upstream swebench grader package and its dependencies."""
    try:
        import swebench  # noqa: F401
        return
    except ImportError:
        pass
    root = repo_root() / SWEBENCH_ROOT_REL
    pkg = root / "swebench"
    if pkg.is_dir():
        s = str(root)
        if s not in sys.path:
            sys.path.insert(0, s)
        try:
            import swebench  # noqa: F401
            return
        except ImportError:
            pass
    raise ImportError(
        "swebench is required for SWE grading. install eval dependencies:\n"
        "  pip install swebench\n"
        "or install the submodule package:\n"
        "  git submodule update --init third_party/SWE-bench\n"
        "  pip install -e third_party/SWE-bench\n"
        "see recipe.md §4"
    )


def ensure_miniswe_on_path() -> Path:
    """prepend mini-swe-agent src so `import minisweagent` resolves from the submodule."""
    root = repo_root() / MINISWE_SRC_REL
    pkg = root / "minisweagent"
    if not pkg.is_dir():
        raise FileNotFoundError(
            f"mini-swe-agent package not found at {pkg}; "
            "run: git submodule update --init third_party/mini-swe-agent"
        )
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def format_command_output(output: dict[str, Any]) -> str:
    """render mini-swe env output as agenticml result text (aligned with swebench.yaml observation template)."""
    exc = (output.get("exception_info") or "").strip()
    rc = output.get("returncode", -1)
    text = output.get("output") or ""
    parts: list[str] = []
    if exc:
        parts.append(f"<exception>{exc}</exception>")
    parts.append(f"<returncode>{rc}</returncode>")
    if len(text) < _OUTPUT_TRUNC:
        parts.append(f"<output>\n{text}</output>")
    else:
        elided = len(text) - _OUTPUT_TRUNC
        parts.extend(
            [
                "<warning>The output of your last command was too long. "
                "Try a more selective command.</warning>",
                f"<output_head>\n{text[:5000]}</output_head>",
                f"<elided_chars>{elided} characters elided</elided_chars>",
                f"<output_tail>\n{text[-5000:]}</output_tail>",
            ]
        )
    return "\n".join(parts)
