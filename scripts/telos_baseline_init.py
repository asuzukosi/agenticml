from telos.tokenizer import TelosTokenizer
from telos.runtime.hf_generator import HfGenerator

# point this at your huggingface repo or the local init directory
MODEL = "kosiasuzu/telos-agent-llama-3.1-8b-init"

tt = TelosTokenizer.from_pretrained(MODEL)
gen = HfGenerator.from_pretrained(MODEL)

prompt = (
    "<|goal|>You are a file-system assistant.\n"
    "<|mission|>What files are in /tmp?\n"
    "<|obs|>tools:\nnamespace tools {\n"
    "  type list_dir = (_: { path: string }) => any;\n"
    "  type answer = (_: { text: string }) => any;\n"
    "}\n"
)

input_ids = tt.encode(prompt)
print(f"Prompt tokens: {len(input_ids)}")

new_ids = gen(input_ids, tt.end_id, max_new_tokens=200)
print(f"Generated {len(new_ids)} tokens")
print("Decoded:")
print(tt.decode(new_ids))