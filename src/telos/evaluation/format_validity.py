"""
format-validity evaluation for telos and chatml LoRA-trained models.

usage:
    telos eval-format-validity --format telos --model ... --dataset ... --output out.json
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from telos.frames import parse as telos_parse, render as telos_render
from telos.tokenizer import TelosTokenizer
from telos.trajectory import Trajectory
from telos.validators import validate_for_model_generation

TELOS_MODEL_TYPES = frozenset({"belief", "plan", "think", "action"})
_TOOL_CALL_RE = re.compile(r"<\|python_tag\|>(.+?)<\|(?:eom_id|eot_id)\|>", re.DOTALL)
_ASSISTANT_TEXT_RE = re.compile(
    r"(?:<\|start_header_id\|>assistant<\|end_header_id\|>)?\s*(.*?)<\|(?:eot_id|eom_id)\|>",
    re.DOTALL,
)


@dataclass
class ExampleResult:
    id: str
    domain: str
    parsed_ok: bool
    structurally_valid: bool
    parse_error: Optional[str] = None
    validation_errors: list[str] = field(default_factory=list)
    num_generated_tokens: int = 0
    generated_text_preview: str = ""


@dataclass(frozen=True)
class _FormatSpec:
    load_tokenizer: Callable[[str], Any]
    build_input_ids: Callable[[dict, Any], list[int]]
    decode_output: Callable[[Any, list[int]], str]
    pad_token_id: Callable[[Any], int]
    check_output: Callable[[dict, str], tuple[bool, bool, Optional[str], list[str]]]
    stop_token_ids: Callable[[Any], list[int]]


def _telos_load_tokenizer(model_id: str) -> TelosTokenizer:
    return TelosTokenizer.from_pretrained(model_id)


def _chatml_load_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_id)


def _inference_load_kwargs(dtype: torch.dtype) -> dict[str, Any]:
    """cuda:0 + cpu offload when gpu present; avoids multi-gpu auto split."""
    if not torch.cuda.is_available():
        return {"torch_dtype": dtype, "device_map": "cpu"}
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    cap = max(1, int(total_gib - 2))
    return {
        "torch_dtype": dtype,
        "device_map": "auto",
        "max_memory": {0: f"{cap}GiB", "cpu": "100GiB"},
        "offload_buffers": True,
    }


def load_model(
    model_id: str,
    adapter_mode: str,
    adapter_id: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
):
    load_kw = _inference_load_kwargs(dtype)
    if adapter_mode == "merged":
        return AutoModelForCausalLM.from_pretrained(model_id, **load_kw)
    if adapter_mode == "peft":
        if not adapter_id:
            raise ValueError("adapter_mode='peft' requires adapter_id")
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError("adapter_mode='peft' requires: pip install peft") from e
        base = AutoModelForCausalLM.from_pretrained(model_id, **load_kw)
        return PeftModel.from_pretrained(base, adapter_id)
    raise ValueError(f"adapter_mode must be 'merged' or 'peft', got: {adapter_mode!r}")


def _model_device(model) -> torch.device:
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device


def _loads_field(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _row_evaluable(row: dict, fmt: str) -> bool:
    """row has the columns and content needed to run generation for this format."""
    try:
        if fmt == "telos":
            if "frames" not in row:
                return False
            frames = _loads_field(row["frames"])
            if not isinstance(frames, list) or not frames:
                return False
            cut = _telos_cut_index(frames)
            return cut < len(frames)
        if "messages" not in row:
            return False
        messages = _loads_field(row["messages"])
        if not isinstance(messages, list) or not messages:
            return False
        return any(m.get("role") == "assistant" for m in messages)
    except (json.JSONDecodeError, TypeError, KeyError):
        return False


def _telos_cut_index(frames: list[dict]) -> int:
    return next(
        (i for i, f in enumerate(frames) if f.get("type") in TELOS_MODEL_TYPES),
        len(frames),
    )


def _telos_prelude_frames(frames: list[dict], cut: int) -> list:
    """dataset frames use short type names (goal); Trajectory coerces to FrameType."""
    prelude_dicts = [f for f in frames[:cut] if f.get("type") != "end"]
    return Trajectory(prelude_dicts).to_frames()


def _telos_input_ids(row: dict, tt: TelosTokenizer) -> list[int]:
    frames = _loads_field(row["frames"])
    cut = _telos_cut_index(frames)
    if cut >= len(frames):
        return []
    return tt.encode(telos_render(_telos_prelude_frames(frames, cut)))


def _telos_decode_output(tt: TelosTokenizer, token_ids: list[int]) -> str:
    return tt.decode(token_ids)


def _telos_pad_token_id(tt: TelosTokenizer) -> int:
    hf = tt.hf
    return hf.pad_token_id if hf.pad_token_id is not None else hf.eos_token_id


def _chatml_input_ids(row: dict, tokenizer: PreTrainedTokenizerBase) -> list[int]:
    messages = _loads_field(row["messages"])
    cut = next(
        (i for i, m in enumerate(messages) if m.get("role") == "assistant"),
        len(messages),
    )
    if cut >= len(messages):
        return []
    return tokenizer.apply_chat_template(
        messages[:cut],
        tokenize=True,
        add_generation_prompt=True,
    )


def _chatml_decode_output(tokenizer: PreTrainedTokenizerBase, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def _chatml_pad_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id


def _telos_stop_ids(tt: TelosTokenizer) -> list[int]:
    ids = [tt.end_id]
    eos = tt.hf.eos_token_id
    if eos is not None and eos not in ids:
        ids.append(eos)
    return ids


def _chatml_stop_ids(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    ids: list[int] = []
    for token in ("<|eot_id|>", "<|eom_id|>"):
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is not None and tid != tokenizer.unk_token_id:
            ids.append(tid)
    eos = tokenizer.eos_token_id
    if eos is not None and eos not in ids:
        ids.append(eos)
    return ids or [tokenizer.eos_token_id]


def _telos_check(row: dict, generated_text: str) -> tuple[bool, bool, Optional[str], list[str]]:
    frames = _loads_field(row["frames"])
    try:
        generated_frames = telos_parse(generated_text, strict=False)
    except Exception as e:
        return False, False, f"parse failure: {e}", []

    cut = _telos_cut_index(frames)
    full = _telos_prelude_frames(frames, cut)
    full.extend(generated_frames)

    try:
        violations = validate_for_model_generation(full)
    except Exception as e:
        return True, False, None, [f"validator crashed: {e}"]

    errs = [
        f"[{v.rule}] frame {v.frame_index}: {v.message}" if hasattr(v, "rule") else str(v)
        for v in violations
    ]
    return True, len(errs) == 0, None, errs


def _chatml_check(_row: dict, generated_text: str) -> tuple[bool, bool, Optional[str], list[str]]:
    errors: list[str] = []
    if not re.search(r"<\|(?:eot_id|eom_id)\|>", generated_text):
        return False, False, "missing stop token", ["no stop token emitted"]

    has_tool = False
    m = _TOOL_CALL_RE.search(generated_text)
    if m:
        try:
            call = json.loads(m.group(1).strip())
            if not isinstance(call, dict):
                errors.append("tool call payload is not a JSON object")
            elif "name" not in call:
                errors.append("tool call missing 'name' field")
            else:
                has_tool = True
        except json.JSONDecodeError as e:
            return True, False, None, [f"tool call JSON invalid: {e.msg}"]

    text_m = _ASSISTANT_TEXT_RE.search(generated_text)
    has_text = bool(text_m and text_m.group(1).strip())
    if not has_tool and not has_text:
        errors.append("generation has no tool call and no text content")
        return True, False, None, errors

    return True, len(errors) == 0, None, errors


FORMAT_SPECS: dict[str, _FormatSpec] = {
    "telos": _FormatSpec(
        _telos_load_tokenizer,
        _telos_input_ids,
        _telos_decode_output,
        _telos_pad_token_id,
        _telos_check,
        _telos_stop_ids,
    ),
    "chatml": _FormatSpec(
        _chatml_load_tokenizer,
        _chatml_input_ids,
        _chatml_decode_output,
        _chatml_pad_token_id,
        _chatml_check,
        _chatml_stop_ids,
    ),
}


def generate_completion(
    model,
    spec: _FormatSpec,
    tokenizer,
    input_ids: list[int],
    *,
    max_new_tokens: int = 1024,
    stop_token_ids: Optional[list[int]] = None,
) -> tuple[str, int]:
    device = _model_device(model)
    inputs = torch.tensor([input_ids], device=device, dtype=torch.long)
    input_len = inputs.shape[1]
    pad_id = spec.pad_token_id(tokenizer)
    default_eos = stop_token_ids[0] if stop_token_ids else pad_id
    with torch.no_grad():
        out = model.generate(
            inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=stop_token_ids or default_eos,
        )
    gen_ids = out[0][input_len:].tolist()
    return spec.decode_output(tokenizer, gen_ids), len(gen_ids)


def _eval_row(
    row: dict,
    model,
    tokenizer,
    spec: _FormatSpec,
    stop_ids: list[int],
    max_new_tokens: int,
) -> Optional[ExampleResult]:
    input_ids = spec.build_input_ids(row, tokenizer)
    if not input_ids:
        return None

    try:
        text, n_tok = generate_completion(
            model,
            spec,
            tokenizer,
            input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_ids,
        )
    except Exception as e:
        return ExampleResult(
            id=row["id"],
            domain=row["domain"],
            parsed_ok=False,
            structurally_valid=False,
            parse_error=f"generation error: {e}",
        )

    parsed_ok, valid, parse_err, val_errs = spec.check_output(row, text)
    return ExampleResult(
        id=row["id"],
        domain=row["domain"],
        parsed_ok=parsed_ok,
        structurally_valid=valid,
        parse_error=parse_err,
        validation_errors=val_errs,
        num_generated_tokens=n_tok,
        generated_text_preview=text[:200],
    )


def _prepare_eval_dataset(ds, fmt: str, num_examples: int, seed: int):
    """filter to evaluable rows, then sample (num_examples < 0 = full split)."""
    n_raw = len(ds)
    ds_ok = ds.filter(lambda row: _row_evaluable(row, fmt))
    n_ok = len(ds_ok)
    if n_ok == 0:
        raise ValueError(
            f"no evaluable rows for format {fmt!r} (need frames with model blocks, "
            f"or messages with an assistant turn)"
        )
    if n_ok < n_raw:
        print(f"filtered: {n_ok} / {n_raw} rows have a {fmt} generation target")

    if num_examples < 0:
        return ds_ok, n_raw, n_ok, n_ok

    k = min(num_examples, n_ok)
    if k >= n_ok:
        return ds_ok, n_raw, n_ok, n_ok

    indices = random.Random(seed).sample(range(n_ok), k)
    return ds_ok.select(indices), n_raw, n_ok, k


def evaluate_format_validity(
    model,
    spec: _FormatSpec,
    tokenizer,
    ds,
    *,
    format_name: str,
    max_new_tokens: int = 1024,
) -> list[ExampleResult]:
    stop_ids = spec.stop_token_ids(tokenizer)
    results: list[ExampleResult] = []
    n = len(ds)
    print(f"evaluating {n} {format_name} examples...")

    for row in tqdm(ds, total=n, desc=f"{format_name} format validity", unit="ex"):
        r = _eval_row(row, model, tokenizer, spec, stop_ids, max_new_tokens)
        if r is not None:
            results.append(r)

    return results


def aggregate(results: list[ExampleResult]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"n": 0}

    by_domain: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "parsed": 0, "valid": 0})
    error_counts: dict[str, int] = defaultdict(int)
    n_parsed = n_valid = 0

    for r in results:
        n_parsed += r.parsed_ok
        n_valid += r.structurally_valid
        by_domain[r.domain]["n"] += 1
        by_domain[r.domain]["parsed"] += int(r.parsed_ok)
        by_domain[r.domain]["valid"] += int(r.structurally_valid)
        if r.parse_error:
            error_counts[f"parse: {r.parse_error.split(':')[0][:50]}"] += 1
        for ve in r.validation_errors:
            error_counts[f"validation: {ve.split(':')[0][:50]}"] += 1

    return {
        "n": n,
        "parse_rate": n_parsed / n,
        "valid_rate": n_valid / n,
        "by_domain": dict(by_domain),
        "top_failures": dict(sorted(error_counts.items(), key=lambda x: -x[1])[:10]),
    }


def print_summary(summary: dict[str, Any]) -> None:
    n = summary.get("n", 0)
    if n == 0:
        print("no examples evaluated.")
        return
    print(f"examples evaluated: {n}")
    print(f"parse rate:         {summary['parse_rate']:.1%}  ({int(summary['parse_rate'] * n)}/{n})")
    print(f"structural valid:   {summary['valid_rate']:.1%}  ({int(summary['valid_rate'] * n)}/{n})\n")
    print("by domain:")
    print(f"  {'domain':<20} {'n':>6} {'parsed':>10} {'valid':>10}")
    for domain, d in sorted(summary["by_domain"].items()):
        dn = d["n"]
        print(
            f"  {domain:<20} {dn:>6} "
            f"{d['parsed'] / dn:>9.1%} {d['valid'] / dn:>9.1%}"
        )
    if summary.get("top_failures"):
        print("\ntop failure modes:")
        for failure, count in summary["top_failures"].items():
            print(f"  {count:>5}  {failure}")


def evaluate(
    model_id: str,
    dataset_id: str,
    split: str,
    fmt: str,
    output_path: Path,
    *,
    adapter_mode: str = "merged",
    adapter_id: Optional[str] = None,
    num_examples: int = 100,
    sample_seed: int = 42,
    max_new_tokens: int = 1024,
) -> None:
    if fmt not in FORMAT_SPECS:
        raise ValueError(f"format must be 'telos' or 'chatml', got: {fmt!r}")

    spec = FORMAT_SPECS[fmt]

    print(f"loading base model {model_id} (adapter_mode={adapter_mode})...")
    if adapter_mode == "peft":
        print(f"  adapter: {adapter_id}")
    model = load_model(model_id, adapter_mode, adapter_id)
    print(f"loading tokenizer for {fmt}...")
    tokenizer = spec.load_tokenizer(model_id)
    model.eval()

    print(f"loading dataset {dataset_id} split={split}...")
    ds_full = load_dataset(dataset_id, split=split)
    ds, split_len, evaluable_len, run_len = _prepare_eval_dataset(
        ds_full, fmt, num_examples, sample_seed
    )
    if num_examples < 0:
        print(f"evaluating full evaluable split ({run_len} / {split_len} rows)")
    elif run_len < evaluable_len:
        print(
            f"random sample: {run_len} / {evaluable_len} evaluable rows "
            f"({split_len} in split, num_examples={num_examples}, seed={sample_seed})"
        )
    else:
        print(f"evaluating all {run_len} evaluable rows ({split_len} in split)")

    results = evaluate_format_validity(
        model,
        spec,
        tokenizer,
        ds,
        format_name=fmt,
        max_new_tokens=max_new_tokens,
    )
    summary = aggregate(results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(
            {
                "model": model_id,
                "adapter": adapter_id,
                "adapter_mode": adapter_mode,
                "dataset": dataset_id,
                "split": split,
                "format": fmt,
                "split_size": split_len,
                "evaluable_size": evaluable_len,
                "num_examples": num_examples,
                "num_run": run_len,
                "sample_seed": sample_seed,
                "num_examples": len(results),
                "summary": summary,
                "results": [asdict(r) for r in results],
            },
            f,
            indent=2,
        )
    print(f"\nresults written to {output_path}\n")
    print_summary(summary)
