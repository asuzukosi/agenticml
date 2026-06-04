"""verify telos reserved-token rows match seed-word mean pooling (read-only)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from telos.model_init.initialize_telos_embeddings import (
    TELOS_RESERVED_SLOT,
    TELOS_SEED_TOKENS,
    _seed_token_ids,
)


@dataclass
class MarkerEmbeddingReport:
    marker: str
    slot: int
    target_id: int
    seed_ids: list[int]
    n_seeds: int
    embed_norm: float
    lm_head_norm: float
    seed_mean_embed_norm: float
    embed_max_abs_err: float
    lm_head_max_abs_err: float


def verify_telos_embedding_init(
    model,
    tokenizer,
    *,
    atol: float = 0.05,
    min_norm_floor: float = 0.15,
    max_norm_ratio: float = 3.0,
) -> tuple[list[MarkerEmbeddingReport], list[str]]:
    """check reserved rows equal mean of seed rows."""
    embed = model.get_input_embeddings().weight
    lm_head = model.get_output_embeddings().weight
    reports: list[MarkerEmbeddingReport] = []
    errors: list[str] = []

    for marker, seeds in TELOS_SEED_TOKENS.items():
        slot = TELOS_RESERVED_SLOT[marker]
        reserved_name = f"<|reserved_special_token_{slot}|>"
        target_id = tokenizer.convert_tokens_to_ids(reserved_name)
        seed_ids = _seed_token_ids(tokenizer, seeds)

        if len(seed_ids) < 2:
            errors.append(f"{marker}: fewer than 2 single-token seeds ({len(seed_ids)})")

        expected_embed = embed[seed_ids].float().mean(dim=0)
        expected_head = lm_head[seed_ids].float().mean(dim=0)
        actual_embed = embed[target_id].float()
        actual_head = lm_head[target_id].float()

        embed_err = (actual_embed - expected_embed).abs().max().item()
        head_err = (actual_head - expected_head).abs().max().item()
        embed_norm = actual_embed.norm().item()
        head_norm = actual_head.norm().item()
        seed_mean_norm = expected_embed.norm().item()

        reports.append(
            MarkerEmbeddingReport(
                marker=marker,
                slot=slot,
                target_id=target_id,
                seed_ids=seed_ids,
                n_seeds=len(seed_ids),
                embed_norm=embed_norm,
                lm_head_norm=head_norm,
                seed_mean_embed_norm=seed_mean_norm,
                embed_max_abs_err=embed_err,
                lm_head_max_abs_err=head_err,
            )
        )

        if embed_err > atol:
            errors.append(f"{marker}: embed row differs from seed mean (max err {embed_err:.6f})")
        if head_err > atol:
            errors.append(f"{marker}: lm_head row differs from seed mean (max err {head_err:.6f})")
        if embed_norm < min_norm_floor:
            errors.append(
                f"{marker}: embed norm {embed_norm:.4f} below floor {min_norm_floor}"
            )
        if head_norm < min_norm_floor:
            errors.append(
                f"{marker}: lm_head norm {head_norm:.4f} below floor {min_norm_floor}"
            )
        if seed_mean_norm > 0:
            ratio = embed_norm / seed_mean_norm
            if ratio > max_norm_ratio or ratio < 1.0 / max_norm_ratio:
                errors.append(
                    f"{marker}: embed/seed-mean norm ratio {ratio:.3f} "
                    f"outside [1/{max_norm_ratio}, {max_norm_ratio}]"
                )
            head_ratio = head_norm / seed_mean_norm if seed_mean_norm else 0.0
            if head_ratio > max_norm_ratio * 1.5:
                errors.append(
                    f"{marker}: lm_head norm much larger than seed mean ({head_ratio:.3f}x)"
                )

    return reports, errors


def print_verification_report(reports: list[MarkerEmbeddingReport], errors: list[str]) -> None:
    print(
        f"{'marker':<14} {'slot':>4} {'id':>7} {'seeds':>5} "
        f"{'embed_norm':>10} {'head_norm':>10} {'seed_norm':>10} {'embed_err':>10}"
    )
    for r in reports:
        print(
            f"{r.marker:<14} {r.slot:>4} {r.target_id:>7} {r.n_seeds:>5} "
            f"{r.embed_norm:>10.4f} {r.lm_head_norm:>10.4f} {r.seed_mean_embed_norm:>10.4f} "
            f"{r.embed_max_abs_err:>10.6f}"
        )
    if errors:
        print("\nverification failures:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\nverification passed (all markers match seed means, norms ok)")


def run_verify_telos_embeddings(model_id: str) -> None:
    print(f"loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"loading model weights: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    embed = model.get_input_embeddings().weight
    lm_head = model.get_output_embeddings().weight
    print(f"embed_tokens device: {embed.device}, dtype: {embed.dtype}")
    print(f"lm_head device:      {lm_head.device}, dtype: {lm_head.dtype}")

    reports, errors = verify_telos_embedding_init(model, tokenizer)
    print_verification_report(reports, errors)
    if errors:
        raise RuntimeError(f"telos embedding verification failed ({len(errors)} issues)")
