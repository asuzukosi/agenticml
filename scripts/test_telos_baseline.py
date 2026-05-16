from telos.tokenizer import TelosTokenizer
from telos.runtime.hf_generator import HfGenerator

tt = TelosTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
gen = HfGenerator.from_pretrained("meta-llama/Llama-3.1-8B")

prompt = (
    "<|goal|>You are a file-system assistant.\n"
    "<|mission|>What files are in /tmp?\n"
    "<|obs|>tools:\nnamespace tools {\n"
    "  type list_dir = (_: { path: string }) => any;\n"
    "  type answer = (_: { text: string }) => any;\n"
    "}\n"
    "<|action|>"
)

input_ids = tt.encode(prompt)
print(f"Prompt tokens: {len(input_ids)}")

new_ids = gen(input_ids, tt.end_id, max_new_tokens=100)
print(f"Generated {len(new_ids)} tokens")
print("Decoded:")
print(tt.decode(new_ids))