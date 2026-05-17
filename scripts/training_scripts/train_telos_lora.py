from __future__ import annotations
import argparse
import json
import torch
from datasets import load_dataset
from transformers import Trainer
from telos.tokenizer import TelosTokenizer
from telos.trajectory import Trajectory
from scripts.training_scripts.train_lora_common import (
    TELOS_MODEL_MARKERS,
    RunConfig,
    build_lora_model,
    make_training_args,
    maybe_push_artifacts,
    maybe_init_wandb,
    print_trainable,
    set_seed,
    truncate,
    causal_lm_collator,
)
from functools import partial

def mask_telos_runtime_labels(input_ids: list[int], marker_ids: set[int], runtime_marker_ids: set[int]) -> list[int]:
    """mask telos labels."""
    labels = list(input_ids)
    in_model_block = False
    for i, tok in enumerate(input_ids):
        if tok in marker_ids:
            in_model_block = True
        elif tok in runtime_marker_ids:
            in_model_block = False
        if not in_model_block:
            labels[i] = -100
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="kosiasuzu/telos-agent-llama-3.1-8b-init")
    ap.add_argument("--dataset", default="kosiasuzu/telos-agent-trajectory-dataset")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-split", default="eval")
    ap.add_argument("--output-dir", default="outputs/telos-lora")
    ap.add_argument("--project", default="telos-agent")
    ap.add_argument("--run-name", default="telos-llama-3.1-8b-lora")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-eval", type=int, default=0)
    ap.add_argument("--push-adapter", action="store_true")
    ap.add_argument("--push-merged", action="store_true")
    ap.add_argument("--adapter-repo-id", default="")
    ap.add_argument("--merged-repo-id", default="")
    args = ap.parse_args()

    cfg = RunConfig(
        model_id=args.model_id,
        dataset_id=args.dataset,
        train_split=args.train_split,
        eval_split=args.eval_split,
        output_dir=args.output_dir,
        project=args.project,
        run_name=args.run_name,
        max_length=args.max_length,
    )

    set_seed(cfg.seed)
    maybe_init_wandb(cfg)
    print(f"loading tokenizer from {cfg.model_id}...")
    tt = TelosTokenizer.from_pretrained(cfg.model_id)
    tokenizer = tt.hf
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_lora_model(cfg.model_id)
    print_trainable(model)

    ds = load_dataset(cfg.dataset_id)
    train_ds = ds[cfg.train_split]
    eval_ds = ds[cfg.eval_split]
    if args.limit_train:
        train_ds = train_ds.select(range(min(args.limit_train, len(train_ds))))
    if args.limit_eval:
        eval_ds = eval_ds.select(range(min(args.limit_eval, len(eval_ds))))

    marker_ids = {tt.belief_id, tt.plan_id, tt.think_id, tt.action_id, tt.end_id}
    runtime_marker_ids = {tt.goal_id, tt.mission_id, tt.obs_id, tt.result_id, tt.feedback_id, tt.reward_id}

    def _tok(ex):
        frames = json.loads(ex["frames"])
        trajectory = Trajectory(frames)
        ids = tt.apply_trajectory_template(trajectory, tokenize=True, return_tensors="pt")
        # normalize to flat python list[int]
        if isinstance(ids, torch.Tensor):
            if ids.ndim == 2 and ids.shape[0] == 1:
                ids = ids[0]
            ids = ids.tolist()

        # handle accidental nested list
        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
            ids = ids[0]

        if not isinstance(ids, list) or len(ids) == 0 or not isinstance(ids[0], int):
            return {"input_ids": [], "labels": [], "attention_mask": []}
        labels = mask_telos_runtime_labels(ids, marker_ids, runtime_marker_ids)
        ids, labels = truncate(ids, labels, cfg.max_length)
        attn = [1] * len(ids)
        return {"input_ids": ids, "labels": labels, "attention_mask": attn}

    train_tok = train_ds.map(_tok, remove_columns=train_ds.column_names)
    train_tok = train_tok.filter(lambda x: len(x["input_ids"]) > 0)
    eval_tok = eval_ds.map(_tok, remove_columns=eval_ds.column_names)
    eval_tok = eval_tok.filter(lambda x: len(x["input_ids"]) > 0)

    targs = make_training_args(cfg)
    # add custom collator to handle telos runtime labels
    collator = partial(causal_lm_collator, pad_token_id=tokenizer.pad_token_id)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    trainer.save_state()
    maybe_push_artifacts(
        model=model,
        tokenizer=tokenizer,
        output_dir=cfg.output_dir,
        push_adapter=args.push_adapter,
        push_merged=args.push_merged,
        adapter_repo_id=args.adapter_repo_id or None,
        merged_repo_id=args.merged_repo_id or None,
    )


if __name__ == "__main__":
    main()