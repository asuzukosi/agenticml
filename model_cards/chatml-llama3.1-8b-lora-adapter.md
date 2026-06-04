---
library_name: peft
license: llama3.1
base_model: kosiasuzu/chatml-agent-llama-3.1-8b-init
tags:
  - llama-3.1
  - chatml
  - lora
  - peft
  - agent
  - telos-baseline
pipeline_tag: text-generation
---

# kosiasuzu/chatml-llama3.1-8b-lora-adapter

**LoRA adapter only** for the ChatML baseline in the Telos format study. Load on top of [`kosiasuzu/chatml-agent-llama-3.1-8b-init`](https://huggingface.co/kosiasuzu/chatml-agent-llama-3.1-8b-init).

For single-file inference, use: [`kosiasuzu/chatml-llama3.1-8b-lora-merged`](https://huggingface.co/kosiasuzu/chatml-llama3.1-8b-lora-merged).

## Model Details

### Model Description

- **Model ID:** `kosiasuzu/chatml-llama3.1-8b-lora-adapter`
- **Developed by:** Telos project / kosiasuzu
- **Model type:** PEFT LoRA adapter (CAUSAL_LM)
- **Base model:** `kosiasuzu/chatml-agent-llama-3.1-8b-init` (**required**)
- **License:** [Llama 3.1 Community License](https://llama.meta.com/llama3_1/license/)

**Init base:** Llama 3.1 8B base weights + Instruct tokenizer; ChatML special tokens mean-pooled in embedding table (`telos init-chatml-embeddings`).

**Training:** `messages` column on `kosiasuzu/telos-agent-trajectory-dataset` (ChatML derived from Telos `frames`); loss on assistant spans via `apply_chat_template`.

### Model Sources

- **Telos repository:** https://github.com/kosiasuzu/talos
- **Merged weights:** [`kosiasuzu/chatml-llama3.1-8b-lora-merged`](https://huggingface.co/kosiasuzu/chatml-llama3.1-8b-lora-merged)
- **Paired Telos adapter:** [`kosiasuzu/telos-llama3.1-8b-lora-adapter`](https://huggingface.co/kosiasuzu/telos-llama3.1-8b-lora-adapter)
- **Dataset:** [`kosiasuzu/telos-agent-trajectory-dataset`](https://huggingface.co/datasets/kosiasuzu/telos-agent-trajectory-dataset)

## Uses

### Direct Use

- PEFT load on `chatml-agent-llama-3.1-8b-init` for ChatML agent continuation or further tuning.

### Downstream Use

- Merge: `model.merge_and_unload()` → equivalent to merged hub checkpoint.
- Eval: `telos eval-format-validity --format chatml --adapter-mode peft`

### Out-of-Scope Use

- Do not load on `telos-agent-llama-3.1-8b-init` or raw `meta-llama/Llama-3.1-8B`.
- Not for Telos frame rendering (`--format telos` eval expects Telos-trained weights).

## How to Get Started with the Model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "kosiasuzu/chatml-agent-llama-3.1-8b-init"
adapter_id = "kosiasuzu/chatml-llama3.1-8b-lora-adapter"

tokenizer = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(
    base_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, adapter_id)
model.eval()

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello."},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False))
```

**CLI eval (PEFT mode):**

```bash
telos eval-format-validity \
  --format chatml \
  --model kosiasuzu/chatml-agent-llama-3.1-8b-init \
  --adapter-mode peft \
  --adapter-id kosiasuzu/chatml-llama3.1-8b-lora-adapter \
  --dataset kosiasuzu/telos-agent-trajectory-dataset \
  --split eval \
  --output results/chatml_format_validity_peft.json
```

## Training Details

Produced by `telos train-chatml-lora` with `--adapter-repo-id kosiasuzu/chatml-llama3.1-8b-lora-adapter`.

| LoRA setting | Value |
|--------------|--------|
| r | 32 |
| lora_alpha | 64 |
| lora_dropout | 0.05 |
| target_modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head |
| epochs | 2 |
| learning_rate | 2e-4 |
| max_length | 2048 |

Full training narrative: see [`chatml-llama3.1-8b-lora-merged.md`](chatml-llama3.1-8b-lora-merged.md).

## Evaluation

ChatML format validity on `eval` split. Results: TBD.

## Citation

Cite Telos repository + Llama 3.1. Adapter: `kosiasuzu/chatml-llama3.1-8b-lora-adapter`.

## Model Card Contact

- Hugging Face: [kosiasuzu](https://huggingface.co/kosiasuzu)
