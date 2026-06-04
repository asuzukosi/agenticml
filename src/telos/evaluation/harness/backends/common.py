"""shared backend helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from telos.runtime.tools import ToolRegistry

GenerateFn = Callable[[list[int], int, int], list[int]]


@dataclass
class GenStats:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    inference_sec: float = 0.0


def wrap_generate(base: GenerateFn) -> tuple[GenerateFn, GenStats]:
    stats = GenStats()

    def generate(input_ids: list[int], stop_token_id: int, max_new_tokens: int) -> list[int]:
        stats.prompt_tokens += len(input_ids)
        t0 = time.perf_counter()
        out = base(input_ids, stop_token_id, max_new_tokens)
        stats.generated_tokens += len(out)
        stats.inference_sec += time.perf_counter() - t0
        return out

    return generate, stats


class TimedToolRegistry:
    def __init__(self, inner: ToolRegistry):
        self._inner = inner
        self.tool_sec = 0.0

    def schemas(self) -> list[dict[str, Any]]:
        return self._inner.schemas()

    def call(self, name: str, args: dict[str, Any]) -> Any:
        t0 = time.perf_counter()
        try:
            return self._inner.call(name, args)
        finally:
            self.tool_sec += time.perf_counter() - t0
