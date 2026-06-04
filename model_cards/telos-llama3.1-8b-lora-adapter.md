---
library_name: peft
license: llama3.1
base_model: kosiasuzu/telos-agent-llama-3.1-8b-init
tags:
  - llama-3.1
  - telos
  - lora
  - peft
  - agent
pipeline_tag: text-generation
---

# kosiasuzu/telos-llama3.1-8b-lora-adapter

**LoRA adapter only** for Telos-format agent fine-tuning on Llama 3.1 8B. Load on top of [`kosiasuzu/telos-agent-llama-3.1-8b-init`](https://huggingface.co/kosiasuzu/telos-agent-llama-3.1-8b-init).

For single-file inference without PEFT, use the merged checkpoint: [`kosiasuzu/telos-llama3.1-8b-lora-merged`](https://huggingface.co/kosiasuzu/telos-llama3.1-8b-lora-merged).

## Model Details

### Model Description

- **Model ID:** `kosiasuzu/telos-llama3.1-8b-lora-adapter`
- **Developed by:** Telos project / kosiasuzu
- **Model type:** PEFT LoRA adapter (CAUSAL_LM)
- **Base model:** `kosiasuzu/telos-agent-llama-3.1-8b-init` (**required** at load time)
- **License:** [Llama 3.1 Community License](https://llama.meta.com/llama3_1/license/) (via base model)

**Training:** same run as the Telos merged model — `frames` column on `kosiasuzu/telos-agent-trajectory-dataset`, TelosTokenizer, assistant/model-block label masking only.

### Model Sources

- **Telos repository:** https://github.com/kosiasuzu/talos
- **Merged weights:** [`kosiasuzu/telos-llama3.1-8b-lora-merged`](https://huggingface.co/kosiasuzu/telos-llama3.1-8b-lora-merged)
- **Dataset:** [`kosiasuzu/telos-agent-trajectory-dataset`](https://huggingface.co/datasets/kosiasuzu/telos-agent-trajectory-dataset)

## Uses

### Direct Use

- Load with **PEFT** on the init base for Telos trajectory continuation or further fine-tuning.
- Swap or stack adapters only if you manage compatibility (this adapter is trained for one base revision).

### Downstream Use

- Merge locally: `model.merge_and_unload()` (as in `telos` training push path).
- Eval: `telos eval-format-validity --format telos --adapter-mode peft --model <init> --adapter-id kosiasuzu/telos-llama3.1-8b-lora-adapter`

### Out-of-Scope Use

- **Will not work** if loaded on `meta-llama/Llama-3.1-8B` or ChatML init — base must be `telos-agent-llama-3.1-8b-init`.
- Not intended for ChatML `apply_chat_template` workflows.

## How to Get Started with the Model

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_id = "kosiasuzu/telos-agent-llama-3.1-8b-init"
adapter_id = "kosiasuzu/telos-llama3.1-8b-lora-adapter"

from telos.tokenizer import TelosTokenizer

tt = TelosTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(
    base_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, adapter_id)
model.eval()

# use tt.encode(render(...)) and tt.decode(...) as in the merged model card (not tt.hf on wire text)
```

**CLI eval (PEFT mode):**

```bash
telos eval-format-validity \
  --format telos \
  --model kosiasuzu/telos-agent-llama-3.1-8b-init \
  --adapter-mode peft \
  --adapter-id kosiasuzu/telos-llama3.1-8b-lora-adapter \
  --dataset kosiasuzu/telos-agent-trajectory-dataset \
  --split eval \
  --output results/telos_format_validity_peft.json
```

## Training Details

Produced by `telos train-telos-lora` with `--adapter-repo-id kosiasuzu/telos-llama3.1-8b-lora-adapter`.

| LoRA setting | Value |
|--------------|--------|
| r | 32 |
| lora_alpha | 64 |
| lora_dropout | 0.05 |
| target_modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head |
| epochs | 2 |
| learning_rate | 2e-4 |
| max_length | 2048 |

Full training narrative: see [`telos-llama3.1-8b-lora-merged.md`](telos-llama3.1-8b-lora-merged.md).


## Model Card Contact

- Hugging Face: [kosiasuzu](https://huggingface.co/kosiasuzu)
