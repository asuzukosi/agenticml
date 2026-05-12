from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

# This is the standard way to access added/special tokens in HF
for token_id, token_obj in tokenizer.added_tokens_decoder.items():
    if "extra" in str(token_obj):
        print(f"ID: {token_id} | Token: {token_obj}")

# To get a specific one directly:
extra_0_id = tokenizer.convert_tokens_to_ids("<|extra_0|>")
print(f"\nDirect check for <|extra_0|>: {extra_0_id}")
