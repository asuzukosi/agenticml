"""chatml backend: messages + HfGenerator multi-step loop."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Optional, Union

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from telos.constants import TERMINAL_TOOLS
from telos.dataset_prep.telos_to_chatml import telos_to_chatml
from telos.evaluation.harness.backend import BackendRunResult, BackendStepResult
from telos.evaluation.harness.backends.common import GenStats, TimedToolRegistry
from telos.evaluation.harness.load import AdapterMode
from telos.runtime.hf_generator import HfGenerator
from telos.runtime.tools import ToolError, ToolRegistry
from telos.trajectory import Trajectory

_TOOL_RE = re.compile(r"<\|python_tag\|>(.+?)<\|(?:eom_id|eot_id)\|>", re.DOTALL)
_TEXT_RE = re.compile(
    r"(?:<\|start_header_id\|>assistant<\|end_header_id\|>)?\s*(.*?)<\|(?:eot_id|eom_id)\|>",
    re.DOTALL,
)


def _to_messages(trajectory: Trajectory | list[dict]) -> list[dict]:
    frames = trajectory.to_dict() if isinstance(trajectory, Trajectory) else list(trajectory)
    return frames if frames and "role" in frames[0] else telos_to_chatml(frames)


def _with_tools(messages: list[dict], tools: list[dict]) -> list[dict]:
    if not tools:
        return messages
    block = "available tools:\n" + json.dumps(tools)
    out = list(messages)
    for i, m in enumerate(out):
        if m.get("role") == "system":
            out[i] = {**m, "content": f"{m.get('content', '')}\n\n{block}".strip()}
            return out
    return [{"role": "system", "content": block}, *out]


def _parse(text: str) -> tuple[Optional[dict], str, str]:
    m = _TOOL_RE.search(text)
    if m:
        try:
            call = json.loads(m.group(1))
            if not isinstance(call, dict):
                return None, "", "parse_error: not object"
            return call, "", "tool_call"
        except json.JSONDecodeError as e:
            return None, "", f"parse_error: {e.msg}"
    m = _TEXT_RE.search(text)
    content = (m.group(1) if m else text).strip()
    return (None, content, "assistant_text" if content else "parse_error: empty")


def _tool_name_args(call: dict) -> tuple[str, dict]:
    name = call.get("name") or call.get("tool")
    if not name:
        raise ValueError("tool call missing name")
    raw = call.get("arguments", call.get("parameters", {}))
    args = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
    return str(name), args


def _assistant(call: Optional[dict], text: str, call_id: str) -> dict:
    if call is None:
        return {"role": "assistant", "content": text}
    name, args = _tool_name_args(call)
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}],
    }


@dataclass
class ChatMLBackend:
    tokenizer: PreTrainedTokenizerBase
    generator: HfGenerator

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        dtype: Optional[Any] = None,
        adapter_mode: Union[AdapterMode, str] = AdapterMode.MERGED,
        adapter_id: Optional[str] = None,
    ) -> ChatMLBackend:
        import torch

        dt = dtype or torch.bfloat16
        return cls(
            AutoTokenizer.from_pretrained(model_id),
            HfGenerator.from_pretrained(model_id, dtype=dt, adapter_mode=adapter_mode, adapter_id=adapter_id),
        )

    @property
    def format(self) -> str:
        return "chatml"

    def _stops(self) -> list[int]:
        ids = []
        for tok in ("<|eot_id|>", "<|eom_id|>"):
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != self.tokenizer.unk_token_id:
                ids.append(tid)
        if self.tokenizer.eos_token_id not in ids:
            ids.append(self.tokenizer.eos_token_id)
        return ids or [self.tokenizer.eos_token_id]

    def _generate(self, messages: list[dict], max_new_tokens: int, stats: GenStats) -> str:
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        stats.prompt_tokens += len(ids)
        t0 = time.perf_counter()
        gen = self.generator(
            ids, self._stops(), max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        stats.inference_sec += time.perf_counter() - t0
        stats.generated_tokens += len(gen)
        return self.tokenizer.decode(gen, skip_special_tokens=False)

    def step(
        self,
        trajectory: Trajectory | list[dict],
        tools: Optional[list[dict]] = None,
        *,
        max_new_tokens: int = 512,
        strict: bool = True,
    ) -> BackendStepResult:
        del strict
        messages = _with_tools(_to_messages(trajectory), tools or [])
        stats = GenStats()
        raw = self._generate(messages, max_new_tokens, stats)
        call, text, stop = _parse(raw)
        if stop.startswith("parse_error"):
            return BackendStepResult(messages=messages, stopped_on=stop, raw_text=raw, prompt_tokens=stats.prompt_tokens, generated_tokens=stats.generated_tokens, inference_sec=stats.inference_sec)
        msg = _assistant(call, text, f"call_{len(messages)}")
        return BackendStepResult(
            messages=messages + [msg], new_messages=[msg], stopped_on=stop, raw_text=raw,
            prompt_tokens=stats.prompt_tokens, generated_tokens=stats.generated_tokens, inference_sec=stats.inference_sec,
        )

    def run(
        self,
        trajectory: Trajectory | list[dict],
        registry: ToolRegistry,
        *,
        max_iterations: int = 10,
        max_new_tokens: int = 512,
        strict: bool = True,
    ) -> BackendRunResult:
        del strict
        messages = _with_tools(_to_messages(trajectory), registry.schemas())
        stats = GenStats()
        timed = TimedToolRegistry(registry)
        t0 = time.perf_counter()
        stop, iterations, answer = "max_iterations", 0, None

        for i in range(1, max_iterations + 1):
            iterations = i
            raw = self._generate(messages, max_new_tokens, stats)
            call, text, ps = _parse(raw)
            if ps.startswith("parse_error"):
                stop = ps
                break
            cid = f"call_{len(messages)}"
            messages.append(_assistant(call, text, cid))
            if call is None:
                stop = "terminal_action" if text else "no_action"
                answer = text or None
                break
            name, args = _tool_name_args(call)
            if name in TERMINAL_TOOLS:
                stop = "terminal_action"
                answer = args.get("text") or text if name == "answer" else None
                break
            try:
                payload = {"ok": 1, "value": timed.call(name, args)}
            except ToolError as e:
                payload = {"ok": 0, "value": str(e)}
            messages.append({"role": "tool", "tool_call_id": cid, "content": json.dumps(payload)})

        return BackendRunResult(
            messages=messages, stopped_on=stop, iterations=iterations, final_answer=answer,
            prompt_tokens=stats.prompt_tokens, generated_tokens=stats.generated_tokens,
            inference_sec=stats.inference_sec, tool_sec=timed.tool_sec, total_sec=time.perf_counter() - t0,
        )
