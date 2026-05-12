from transformers import AutoTokenizer

MODEL = "meta-llama/Llama-3.1-8B"
tok = AutoTokenizer.from_pretrained(MODEL)

print(f"Vocab size: {tok.vocab_size}")
print(f"Total tokens including added: {len(tok)}")
print(f"Special tokens: {tok.special_tokens_map}")
print(f"Number of additional special tokens: {len(tok.additional_special_tokens) if tok.additional_special_tokens else 0}")

# Verify reserved tokens encode as single tokens
print("\n--- Reserved token check ---")
for i in range(11):
    marker = f"<|reserved_special_token_{i}|>"
    ids = tok.encode(marker, add_special_tokens=False)
    print(f"{marker}: ids={ids}, len={len(ids)}")

# Also verify the known Llama-3 special tokens
print("\n--- Known special token check ---")
for marker in ["<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]:
    ids = tok.encode(marker, add_special_tokens=False)
    print(f"{marker}: ids={ids}, len={len(ids)}")