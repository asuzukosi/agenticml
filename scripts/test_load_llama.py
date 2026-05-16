import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL = "meta-llama/Llama-3.1-8B"

print("loading tokenizer...")
tok  = AutoTokenizer.from_pretrained(MODEL)
print("loading model (16GB on first run, may take a while)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="auto"
)
print("model loaded")

print(f"model type :{model.dtype}")
# device layout lives on hf_device_map (nn.Module has no device_map)
dm = getattr(model, "hf_device_map", None)
print(f"hf device map :{dm}")
print(f"vram allocated :")
print("model word embeddings tied: ", model.config.tie_word_embeddings)
for i in range(torch.cuda.device_count()):
    allocated = torch.cuda.memory_allocated(i) / 1e9
    total = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"GPU {i}: allocated {allocated:.2f}GB / total {total:.2f}GB")