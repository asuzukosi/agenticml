from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B"
tok = AutoTokenizer.from_pretrained(MODEL)
print(f"vocab size: {len(tok.vocab)}")
print(f"total tokens including added : {len(tok)}")
print(f"special tokens: {tok.special_tokens_map}")
print(f"all special tokens: {tok.all_special_tokens}")

for i in range(15):
    marker = f"<|extra_{i}|>"
    ids = tok.encode(marker, add_special_tokens=False)
    print(f"marker {i}: {marker} -> {ids}")