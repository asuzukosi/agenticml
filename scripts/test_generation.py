# script: 04_basic_gen.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "meta-llama/Llama-3.1-8B"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

prompt = "The capital of France is"
inputs = tok(prompt, return_tensors="pt").to("cuda:0")

print(f"Input tokens: {inputs.input_ids.shape[1]}")

with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )

new_tokens = out[0, inputs.input_ids.shape[1]:]
print("Generated:")
print(tok.decode(new_tokens, skip_special_tokens=True))