from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model


TELOS_MODEL_MARKERS = ["<|belief|>", "<|plan|>", "<|think|>", "<|action|>", "<|end|>"]


@dataclass
class RunConfig:
    model_id: str
    dataset_id: str
    train_split: str
    eval_split: str
    output_dir: str
    project: str
    run_name: str
    # Memory-tight values empirically validated on 2x 4060 Ti 16 GB.
    # Effective batch size = 1 * 32 = 32 per machine, matches spec.
    max_length: int = 1536
    per_device_batch_size: int = 1
    grad_accum_steps: int = 32
    learning_rate: float = 2e-4
    max_grad_norm: float = 1.0
    num_epochs: float = 2.0
    warmup_ratio: float = 0.03
    seed: int = 42
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_lora_model(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",       # spec: include lm_head
        ],
    )
    model = get_peft_model(model, lora_cfg)
    return model


def make_training_args(cfg: RunConfig) -> TrainingArguments:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    kwargs = dict(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,
        weight_decay=0.1,
        bf16=True,
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        report_to=["wandb"],
        run_name=cfg.run_name,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    if world_size > 1:
        kwargs["fsdp"] = "full_shard auto_wrap"
        kwargs["fsdp_config"] = {
            "use_orig_params": True,
            "activation_checkpointing": True,
        }

    return TrainingArguments(**kwargs)


def maybe_init_wandb(cfg: RunConfig) -> None:
    os.environ.setdefault("WANDB_PROJECT", cfg.project)
    os.environ.setdefault("WANDB_NAME", cfg.run_name)


def print_trainable(model) -> None:
    trainable = 0
    total = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    pct = 100 * trainable / total
    print(f"trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")


def truncate(ids: List[int], labels: List[int], max_length: int):
    if len(ids) > max_length:
        ids = ids[:max_length]
        labels = labels[:max_length]
    return ids, labels


def _flat(v):
    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
        return v[0]
    return v


def causal_lm_collator(features: list[dict], pad_token_id: int) -> dict:
    features = [f for f in features if len(f.get("input_ids", [])) > 0]
    if not features:
        raise ValueError("Empty batch after filtering invalid features")

    for f in features:
        f["input_ids"] = _flat(f["input_ids"])
        f["attention_mask"] = _flat(f["attention_mask"])
        f["labels"] = _flat(f["labels"])

    max_len = max(len(f["input_ids"]) for f in features)

    input_ids, attention_mask, labels = [], [], []
    for f in features:
        n = len(f["input_ids"])
        pad = max_len - n
        input_ids.append(f["input_ids"] + [pad_token_id] * pad)
        attention_mask.append(f["attention_mask"] + [0] * pad)
        labels.append(f["labels"] + [-100] * pad)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def maybe_push_artifacts(
    *,
    model,
    tokenizer,
    output_dir: str,
    push_adapter: bool,
    push_merged: bool,
    adapter_repo_id: str | None,
    merged_repo_id: str | None,
) -> None:
    if push_adapter:
        if not adapter_repo_id:
            raise ValueError("--adapter-repo-id is required when --push-adapter is set")
        model.push_to_hub(adapter_repo_id)
        tokenizer.push_to_hub(adapter_repo_id)

    if push_merged:
        if not merged_repo_id:
            raise ValueError("--merged-repo-id is required when --push-merged is set")
        merged = model.merge_and_unload()
        merged.push_to_hub(merged_repo_id)
        tokenizer.push_to_hub(merged_repo_id)