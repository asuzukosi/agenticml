"""telos backend: TelosTokenizer + HfGenerator + sdk step/run."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Union

from telos.evaluation.harness.backend import BackendRunResult, BackendStepResult
from telos.evaluation.harness.backends.common import GenStats, TimedToolRegistry, wrap_generate
from telos.evaluation.harness.load import AdapterMode
from telos.runtime.hf_generator import HfGenerator
from telos.runtime.runtime import run
from telos.runtime.tools import ToolRegistry
from telos.sdk import step
from telos.tokenizer import TelosTokenizer
from telos.trajectory import Trajectory


def _pad_and_stops(tt: TelosTokenizer) -> tuple[int, list[int]]:
    hf = getattr(tt, "hf", tt)
    pad = getattr(hf, "pad_token_id", None) or getattr(hf, "eos_token_id", None) or tt.end_id
    stops = [tt.end_id]
    eos = getattr(hf, "eos_token_id", None)
    if eos is not None and eos not in stops:
        stops.append(eos)
    return pad, stops


@dataclass
class TelosBackend:
    tokenizer: TelosTokenizer
    generator: HfGenerator

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        dtype: Optional[Any] = None,
        adapter_mode: Union[AdapterMode, str] = AdapterMode.MERGED,
        adapter_id: Optional[str] = None,
    ) -> TelosBackend:
        import torch

        dt = dtype or torch.bfloat16
        return cls(
            TelosTokenizer.from_pretrained(model_id),
            HfGenerator.from_pretrained(model_id, dtype=dt, adapter_mode=adapter_mode, adapter_id=adapter_id),
        )

    @property
    def format(self) -> str:
        return "telos"

    def _generate(self) -> tuple[Any, GenStats]:
        pad, stops = _pad_and_stops(self.tokenizer)

        def base(input_ids, _stop, max_new_tokens):
            return self.generator(input_ids, stops, max_new_tokens, pad_token_id=pad)

        return wrap_generate(base)

    def _stats_result(self, stats: GenStats, **kwargs) -> dict[str, Any]:
        return {
            "prompt_tokens": stats.prompt_tokens,
            "generated_tokens": stats.generated_tokens,
            "inference_sec": stats.inference_sec,
            **kwargs,
        }

    def step(
        self,
        trajectory: Trajectory | list[dict],
        tools: Optional[list[dict]] = None,
        *,
        max_new_tokens: int = 512,
        strict: bool = True,
    ) -> BackendStepResult:
        generate, stats = self._generate()
        result = step(trajectory, tools, tokenizer=self.tokenizer, generate=generate, max_new_tokens=max_new_tokens, strict=strict)
        return BackendStepResult(step=result, **self._stats_result(stats))

    def run(
        self,
        trajectory: Trajectory | list[dict],
        registry: ToolRegistry,
        *,
        max_iterations: int = 10,
        max_new_tokens: int = 512,
        strict: bool = True,
    ) -> BackendRunResult:
        generate, stats = self._generate()
        timed = TimedToolRegistry(registry)
        t0 = time.perf_counter()
        result = run(
            trajectory, timed, tokenizer=self.tokenizer, generate=generate,
            max_iterations=max_iterations, max_new_tokens=max_new_tokens, strict=strict,
        )
        return BackendRunResult(
            run=result,
            tool_sec=timed.tool_sec,
            total_sec=time.perf_counter() - t0,
            **self._stats_result(stats),
        )
