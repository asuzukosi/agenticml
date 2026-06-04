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
from transformers import AutoModelForCausalLM, AutoTokenizer

from telos.frames import parse as telos_parse, render as telos_render
from telos.trajectory import Trajectory
from telos.validators import validate as telos_validate

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
    build_prompt: Callable[[dict, Any], str]
    check_output: Callable[[dict, str], tuple[bool, bool, Optional[str], list[str]]]
    stop_token_ids: Callable[[Any], list[int]]


def load_model_and_tokenizer(
    model_id: str,
    adapter_mode: str,
    adapter_id: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
):
    if adapter_mode == "merged":
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        )
    elif adapter_mode == "peft":
        if not adapter_id:
            raise ValueError("adapter_mode='peft' requires adapter_id")
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError("adapter_mode='peft' requires: pip install peft") from e
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        base = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, adapter_id)
    else:
        raise ValueError(f"adapter_mode must be 'merged' or 'peft', got: {adapter_mode!r}")
    return model, tokenizer


def _telos_cut_index(frames: list[dict]) -> int:
    return next(
        (i for i, f in enumerate(frames) if f.get("type") in TELOS_MODEL_TYPES),
        len(frames),
    )


def _telos_prelude_frames(frames: list[dict], cut: int) -> list:
    """dataset frames use short type names (goal); Trajectory coerces to FrameType."""
    prelude_dicts = [f for f in frames[:cut] if f.get("type") != "end"]
    return Trajectory(prelude_dicts).to_frames()


def _telos_prompt(row: dict, _tokenizer) -> str:
    frames = json.loads(row["frames"])
    cut = _telos_cut_index(frames)
    if cut >= len(frames):
        return ""
    return telos_render(_telos_prelude_frames(frames, cut))


def _chatml_prompt(row: dict, tokenizer) -> str:
    messages = json.loads(row["messages"])
    cut = next((i for i, m in enumerate(messages) if m.get("role") == "assistant"), len(messages)) # find the index of the first assistant message
    if cut >= len(messages):
        return ""
    return tokenizer.apply_chat_template(
        messages[:cut], tokenize=False, add_generation_prompt=True # render the prompt and add the generation prompt to the first assistant message
    )


def _telos_stop_ids(tokenizer) -> list[int]:
    try:
        end_id = tokenizer.convert_tokens_to_ids("<|end|>")
        if end_id == tokenizer.unk_token_id:
            end_id = tokenizer.convert_tokens_to_ids("<|reserved_special_token_7|>")
        return [end_id, tokenizer.eos_token_id]
    except Exception:
        return [tokenizer.eos_token_id]


def _chatml_stop_ids(tokenizer) -> list[int]:
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
    frames = json.loads(row["frames"])
    try:
        generated_frames = telos_parse(generated_text, strict=False)
    except Exception as e:
        return False, False, f"parse failure: {e}", []

    cut = _telos_cut_index(frames)
    full = _telos_prelude_frames(frames, cut)
    full.extend(generated_frames)

    try:
        violations = telos_validate(full)
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
    "telos": _FormatSpec(_telos_prompt, _telos_check, _telos_stop_ids),
    "chatml": _FormatSpec(_chatml_prompt, _chatml_check, _chatml_stop_ids),
}


def generate_completion(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 1024,
    stop_token_ids: Optional[list[int]] = None,
) -> tuple[str, int]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_token_ids or tokenizer.eos_token_id,
        )
    gen_ids = out[0][input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=False), len(gen_ids)


def _eval_row(
    row: dict,
    model,
    tokenizer,
    spec: _FormatSpec,
    stop_ids: list[int],
    max_new_tokens: int,
) -> Optional[ExampleResult]:
    prompt = spec.build_prompt(row, tokenizer)
    if not prompt:
        return None

    try:
        text, n_tok = generate_completion(
            model, tokenizer, prompt,
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


def _random_sample_dataset(ds, sample_size: int, seed: int):
    n = len(ds)
    if sample_size <= 0 or sample_size >= n:
        return ds
    indices = random.Random(seed).sample(range(n), sample_size)
    return ds.select(indices)


def evaluate_format_validity(
    model,
    tokenizer,
    ds,
    fmt: str,
    *,
    max_new_tokens: int = 1024,
) -> list[ExampleResult]:
    spec = FORMAT_SPECS[fmt]
    stop_ids = spec.stop_token_ids(tokenizer)
    results: list[ExampleResult] = []
    n = len(ds)
    print(f"evaluating {n} {fmt} examples...")

    for row in tqdm(ds, total=n, desc=f"{fmt} format validity", unit="ex"):
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
    sample_size: int = 100,
    sample_seed: int = 42,
    max_new_tokens: int = 1024,
) -> None:
    if fmt not in FORMAT_SPECS:
        raise ValueError(f"format must be 'telos' or 'chatml', got: {fmt!r}")

    print(f"loading base model {model_id} (adapter_mode={adapter_mode})...")
    if adapter_mode == "peft":
        print(f"  adapter: {adapter_id}")
    model, tokenizer = load_model_and_tokenizer(model_id, adapter_mode, adapter_id)
    model.eval()

    print(f"loading dataset {dataset_id} split={split}...")
    ds_full = load_dataset(dataset_id, split=split)
    split_len = len(ds_full)
    ds = _random_sample_dataset(ds_full, sample_size, sample_seed)
    if len(ds) < split_len:
        print(
            f"random sample: {len(ds)} / {split_len} examples "
            f"(sample_size={sample_size}, seed={sample_seed})"
        )
    else:
        print(f"evaluating full split ({split_len} examples)")

    results = evaluate_format_validity(
        model, tokenizer, ds, fmt, max_new_tokens=max_new_tokens
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
                "sample_size": sample_size,
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
