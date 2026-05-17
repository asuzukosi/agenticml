from __future__ import annotations

import argparse
import json
from functools import partial

from datasets import load_dataset
from transformers import AutoTokenizer, Trainer

from scripts.training_scripts.train_lora_common import (
    RunConfig,
    build_lora_model,
    causal_lm_collator,
    make_training_args,
    maybe_init_wandb,
    maybe_push_artifacts,
    print_trainable,
    set_seed,
    truncate,
)


def _find_subseq(haystack: list[int], needle: list[int], start: int = 0) -> int:
    if not needle:
        return -1
    n = len(needle)
    for i in range(start, len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return -1


def mask_assistant_only(input_ids: list[int], tokenizer) -> list[int]:
    labels = [-100] * len(input_ids)

    sh = tokenizer.encode("<|start_header_id|>", add_special_tokens=False)
    eh = tokenizer.encode("<|end_header_id|>", add_special_tokens=False)
    eot = tokenizer.encode("<|eot_id|>", add_special_tokens=False)
    assistant = tokenizer.encode("assistant", add_special_tokens=False)

    i = 0
    while i < len(input_ids):
        s = _find_subseq(input_ids, sh, i)
        if s == -1:
            break
        h = s + len(sh)
        eh_pos = _find_subseq(input_ids, eh, h)
        if eh_pos == -1:
            break
        role_ids = input_ids[h:eh_pos]

        content_start = eh_pos + len(eh)
        eot_pos = _find_subseq(input_ids, eot, content_start)
        if eot_pos == -1:
            eot_pos = len(input_ids)

        if role_ids == assistant:
            # Train on assistant content AND the closing <|eot_id|> so the
            # model learns to terminate turns. Without this, the model never
            # learns to emit <|eot_id|> at the right time.
            stop = min(eot_pos + len(eot), len(input_ids))
            for j in range(content_start, stop):
                labels[j] = input_ids[j]

        i = eot_pos + len(eot)

    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    # default to chatml-base-init: base Llama-3.1-8B with ChatML special
    # tokens pre-initialized via mean-pool. Mirrors telos-base-init.
    ap.add_argument("--model-id", default="kosiasuzu/chatml-agent-llama-3.1-8b-init")
    ap.add_argument(
        "--tokenizer-id",
        default="",
        help="Defaults to model-id (chatml-base-init carries the Instruct tokenizer)",
    )
    ap.add_argument("--dataset", default="kosiasuzu/telos-agent-trajectory-dataset")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-split", default="eval")
    ap.add_argument("--output-dir", default="outputs/chatml-lora")
    ap.add_argument("--project", default="telos-agent")
    ap.add_argument("--run-name", default="chatml-llama-3.1-8b-lora")
    ap.add_argument("--max-length", type=int, default=2048)   # matches Telos
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

    tok_id = args.tokenizer_id or cfg.model_id
    tokenizer = AutoTokenizer.from_pretrained(tok_id)
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

    def _tok(ex):
        try:
            messages = json.loads(ex["messages"])
        except Exception:
            return {"input_ids": [], "labels": [], "attention_mask": []}

        if not isinstance(messages, list) or len(messages) == 0:
            return {"input_ids": [], "labels": [], "attention_mask": []}

        ids = None
        assistant_mask = None
        try:
            out = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
            ids = out["input_ids"]
            assistant_mask = out.get("assistant_tokens_mask", None)
        except Exception:
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            except Exception:
                return {"input_ids": [], "labels": [], "attention_mask": []}

        if ids is None or len(ids) == 0:
            return {"input_ids": [], "labels": [], "attention_mask": []}

        if assistant_mask is not None and len(assistant_mask) == len(ids):
            labels = [tok if m else -100 for tok, m in zip(ids, assistant_mask)]
        else:
            labels = mask_assistant_only(ids, tokenizer)

        ids, labels = truncate(ids, labels, cfg.max_length)
        if len(ids) == 0:
            return {"input_ids": [], "labels": [], "attention_mask": []}
        if all(x == -100 for x in labels):
            return {"input_ids": [], "labels": [], "attention_mask": []}

        attn = [1] * len(ids)
        return {"input_ids": ids, "labels": labels, "attention_mask": attn}

    train_tok = train_ds.map(_tok, remove_columns=train_ds.column_names)
    train_tok = train_tok.filter(lambda x: len(x["input_ids"]) > 0)

    eval_tok = eval_ds.map(_tok, remove_columns=eval_ds.column_names)
    eval_tok = eval_tok.filter(lambda x: len(x["input_ids"]) > 0)
    print(f"chatml rows kept: train={len(train_tok)} eval={len(eval_tok)}")

    targs = make_training_args(cfg)
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