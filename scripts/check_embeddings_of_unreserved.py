import torch
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", torch_dtype=torch.bfloat16, device_map="auto")
emb = m.get_input_embeddings().weight

# reserved tokens we care about
telos_ids = [128002, 128003, 128005, 128011, 128012, 128013, 128014, 128015, 128016, 128017, 128018]
# some normal content tokens for comparison
content_ids = [464, 1124, 856, 4320]  # arbitrary common tokens

for tid in telos_ids:
    print(f"reserved {tid}: norm={emb[tid].float().norm().item():.4f}")
print("---")
for tid in content_ids:
    print(f"content {tid}: norm={emb[tid].float().norm().item():.4f}")