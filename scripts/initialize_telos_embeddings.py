"""
initialize Telos reserved-token embeddings from semantically related tokens.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# map each Telos marker to a list of seed token strings to average.
# these are chosen for semantic role, not exact synonymy.
TELOS_SEED_TOKENS: dict[str, list[str]] = {
    "<|goal|>":     ["goal", "objective", "purpose", "aim" ],
    "<|mission|>":  ["mission", "task", "instruction", "assignment", "problem"],
    "<|obs|>":      ["observation", "context", "environment", "situation"],
    "<|belief|>":   ["belief", "state", "knowledge", "assumption"],
    "<|plan|>":     ["plan", "strategy", "approach", "method"],
    "<|think|>":    ["think", "reasoning", "thought", "reflection"],
    "<|action|>":   ["action", "call", "tool", "command", "invocation", "function"],
    "<|end|>":      ["end", "stop", "done", "complete", "finish", "terminate"],
    "<|result|>":   ["result", "output", "response", "outcome"],
    "<|feedback|>": ["feedback", "update", "progress", "comment"],
    "<|reward|>":   ["reward", "score", "bonus", "credit"],
}

# Telos marker -> reserved-token slot index in Llama-3.1's vocabulary.
TELOS_RESERVED_SLOT: dict[str, int] = {
    "<|goal|>":     0,
    "<|mission|>":  1,
    "<|obs|>":      2,
    "<|belief|>":   3,
    "<|plan|>":     4,
    "<|think|>":    5,
    "<|action|>":   6,
    "<|end|>":      7,
    "<|result|>":   8,
    "<|feedback|>": 9,
    "<|reward|>":   10,
}


def _seed_token_ids(tokenizer, words: list[str]) -> list[int]:
    """encode seed words and pull out single-token IDs only.
    multi-token encodings are skipped with a warning - using only the
    first token of a multi-token word would seed with a sub-word
    fragment, which is worse than averaging fewer clean tokens.
    """
    out: list[int] = []
    for w in words:
        # encode with a leading space so the tokenizer is more likely
        # to produce a single 'whole word' token; many BPE tokenizers
        # split "goal" but keep " goal" as one token.
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1:
            out.append(ids[0])
        else:
            print(f"  warning: {w!r} tokenized to {len(ids)} tokens, skipping")
    if not out:
        # fallback: take the first sub-word token from the first seed.
        first = tokenizer.encode(" " + words[0], add_special_tokens=False)
        out = [first[0]]
        print(f"  fallback: using first sub-word token only")
    return out


def init_telos_embeddings(model, tokenizer) -> None:
    """in-place modification of embed_tokens and lm_head rows."""
    if model.config.tie_word_embeddings:
        raise RuntimeError(
            "this script assumes untied embeddings; llama-3.1 has tie_word_embeddings=False. "
            "got tie_word_embeddings=True - the script would need a different code path."
        )

    embed = model.get_input_embeddings().weight
    lm_head = model.get_output_embeddings().weight
    reserved_slot_base = tokenizer.convert_tokens_to_ids("<|reserved_special_token_0|>")

    print(f"embed_tokens device: {embed.device}, dtype: {embed.dtype}")
    print(f"lm_head device:      {lm_head.device}, dtype: {lm_head.dtype}")
    print(f"reserved_slot_0 base id: {reserved_slot_base}")

    for marker, seeds in TELOS_SEED_TOKENS.items():
        # the reserved-slot offset is stored at known IDs; compute the
        # actual ID via the tokenizer to be safe.
        slot = TELOS_RESERVED_SLOT[marker]
        reserved_name = f"<|reserved_special_token_{slot}|>"
        target_id = tokenizer.convert_tokens_to_ids(reserved_name)
        print(f"\n{marker} -> {reserved_name} (id={target_id})")

        seed_ids = _seed_token_ids(tokenizer, seeds)
        print(f"  seed token ids: {seed_ids}")

        # average embeddings from the input matrix.
        seed_embed = embed[seed_ids].float().mean(dim=0)
        # average projection rows from the output matrix.
        seed_head = lm_head[seed_ids].float().mean(dim=0)

        # write back. cast to the matrix's dtype.
        with torch.no_grad():
            embed[target_id] = seed_embed.to(embed.dtype)
            lm_head[target_id] = seed_head.to(lm_head.dtype)

        new_norm_embed = embed[target_id].float().norm().item()
        new_norm_head = lm_head[target_id].float().norm().item()
        print(f"  embed norm: {new_norm_embed:.4f}")
        print(f"  lm_head norm: {new_norm_head:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model",
        default="meta-llama/Llama-3.1-8B",
        help="HuggingFace model id to extend. Default is Llama-3.1-8B-base; "
             "swap for Llama-3.1-70B-base or other Llama-3.x variants as needed.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="mark the repo private.",
    )
    args = parser.parse_args()

    print(f"loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("\ninitializing telos marker embeddings...")
    init_telos_embeddings(model, tokenizer)


    print(f"\npushing to huggingface hub: telos-llama-3.1-8b-init")
    commit_message = f"telos embedding init from base {args.base_model}"
    model.push_to_hub(
        "telos-agent-llama-3.1-8b-init",
        commit_message=commit_message,
        private=args.private,
    )
    tokenizer.push_to_hub(
        "telos-agent-llama-3.1-8b-init",
        commit_message=commit_message,
        private=args.private,
    )
    print(f"pushed.")
    print("done.")


if __name__ == "__main__":
    main()