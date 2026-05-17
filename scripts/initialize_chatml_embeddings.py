"""
build chatml-base-init: Llama-3.1-8B base with ChatML special-token rows
in embed_tokens and lm_head pre-initialized via mean-pooling of seed tokens.
run once. push to HF or save locally, then point train_chatml_lora.py at it.
mirrors the procedure used for telos-base-init so the two runs differ only
in format, not in pre-training-of-format-tokens.
"""
from __future__ import annotations

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# chatml special tokens that exist in the Instruct tokenizer but are
# untrained in the base model's embedding table.
#
# seed tokens are chosen to give each marker a sensible starting point:
# tokens whose meaning roughly corresponds to the marker's role.
CHATML_TOKEN_SEEDS: dict[str, list[str]] = {
    "<|start_header_id|>": ["start", "begin", "role", "header"],
    "<|end_header_id|>": ["end", "stop", "header", "close"],
    "<|eot_id|>": ["end", "stop", "done", "finish"],
    "<|begin_of_text|>": ["begin", "start", "text"],
    "<|end_of_text|>": ["end", "stop", "text"],
    "<|python_tag|>": ["python", "tool", "call", "function"],
}


def mean_pool_embedding(
    token_ids: list[int],
    weight: torch.Tensor,
) -> torch.Tensor:
    """average the embedding rows for the given token ids."""
    rows = weight[token_ids]
    return rows.mean(dim=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument(
        "--instruct-tokenizer",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="tokenizer that already has chatml tokens in vocab",
    )
    ap.add_argument("--output-dir", default="outputs/chatml-base-init")
    ap.add_argument("--push-to-hub", action="store_true")
    ap.add_argument("--hub-repo-id", default="")
    args = ap.parse_args()

    print(f"loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16)

    print(f"loading instruct tokenizer: {args.instruct_tokenizer}")
    instruct_tok = AutoTokenizer.from_pretrained(args.instruct_tokenizer)

    # The base tokenizer needs to be replaced with the Instruct one so the
    # ChatML tokens are in the vocab. The base model's embedding table is
    # already sized for the full Llama-3.1 vocab (128256), which is what
    # the Instruct tokenizer uses — so no resizing needed. Verify:
    embed = model.get_input_embeddings().weight
    lm_head = model.get_output_embeddings().weight
    assert embed.shape[0] == len(instruct_tok), (
        f"Vocab size mismatch: embed={embed.shape[0]} tok={len(instruct_tok)}"
    )
    print(f"Vocab size OK: {embed.shape[0]}")

    print("\ninitializing chatml token rows via mean-pool:")
    with torch.no_grad():
        for marker, seed_words in CHATML_TOKEN_SEEDS.items():
            marker_id = instruct_tok.convert_tokens_to_ids(marker)
            if marker_id is None or marker_id == instruct_tok.unk_token_id:
                print(f"  SKIP {marker}: not in vocab")
                continue

            # encode each seed word and collect token ids
            seed_ids: list[int] = []
            for word in seed_words:
                ids = instruct_tok.encode(word, add_special_tokens=False)
                seed_ids.extend(ids)

            if not seed_ids:
                print(f"  SKIP {marker}: no seed ids")
                continue

            # mean-pool seed rows into embed_tokens and lm_head
            new_embed = mean_pool_embedding(seed_ids, embed)
            new_head = mean_pool_embedding(seed_ids, lm_head)

            embed[marker_id] = new_embed.to(embed.dtype)
            lm_head[marker_id] = new_head.to(lm_head.dtype)

            print(
                f"  OK   {marker:<24} id={marker_id:>6} "
                f"seeds={len(seed_ids):>2} "
                f"embed_norm={new_embed.norm().item():.3f} "
                f"head_norm={new_head.norm().item():.3f}"
            )

    print(f"\nsaving to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    instruct_tok.save_pretrained(args.output_dir)

    if args.push_to_hub:
        if not args.hub_repo_id:
            raise ValueError("--hub-repo-id required when --push-to-hub set")
        print(f"pushing to {args.hub_repo_id}")
        model.push_to_hub(args.hub_repo_id)
        instruct_tok.push_to_hub(args.hub_repo_id)

    print("done.")


if __name__ == "__main__":
    main()