---
library_name: transformers
license: llama3.1
base_model: kosiasuzu/telos-agent-llama-3.1-8b-init
tags:
  - llama-3.1
  - telos
  - lora
  - agent
  - tool-use
pipeline_tag: text-generation
---

# kosiasuzu/telos-llama3.1-8b-lora-merged

**Telos-format** agent model: Llama 3.1 8B with Telos frame markers on reserved vocabulary slots, LoRA-fine-tuned on agent trajectories, then **merged** into dense weights for single-checkpoint inference.

Paired ChatML baseline for the same study: [`kosiasuzu/chatml-llama3.1-8b-lora-merged`](https://huggingface.co/kosiasuzu/chatml-llama3.1-8b-lora-merged).

## Model Details

### Model Description

- **Model ID:** `kosiasuzu/telos-llama3.1-8b-lora-merged`
- **Developed by:** Telos project / kosiasuzu
- **Model type:** Causal language model (Llama 3.1 8B), **PEFT LoRA merged** into dense weights
- **Language(s):** English (primary)
- **License:** [Llama 3.1 Community License](https://llama.meta.com/llama3_1/license/)
- **Finetuned from:** [`kosiasuzu/telos-agent-llama-3.1-8b-init`](https://huggingface.co/kosiasuzu/telos-agent-llama-3.1-8b-init)
- **Adapter (pre-merge):** [`kosiasuzu/telos-llama3.1-8b-lora-adapter`](https://huggingface.co/kosiasuzu/telos-llama3.1-8b-lora-adapter)

**Init base (`telos-agent-llama-3.1-8b-init`):** `meta-llama/Llama-3.1-8B` with Telos marker rows (`<|goal|>` … `<|reward|>`) initialized in `embed_tokens` / `lm_head` via mean-pooled seed embeddings (`telos init-telos-embeddings`).

**Training target:** supervised next-token prediction on the **`frames`** column (Telos frame sequences). Loss applies only inside **model-owned** blocks (`belief`, `plan`, `think`, `action`, `end`); runtime frames (`goal`, `mission`, `obs`, `result`, etc.) are masked.

### Model Sources

- **Telos repository:** https://github.com/kosiasuzu/talos
- **Dataset:** [`kosiasuzu/telos-agent-trajectory-dataset`](https://huggingface.co/datasets/kosiasuzu/telos-agent-trajectory-dataset)
- **Paired ChatML merged model:** [`kosiasuzu/chatml-llama3.1-8b-lora-merged`](https://huggingface.co/kosiasuzu/chatml-llama3.1-8b-lora-merged)

## Uses

### Direct Use

- Continue **Telos** agent trajectories: typed frames (`goal`, `mission`, `obs`, `belief`, `plan`, `think`, `action`, `result`, …) rendered with Telos markers.
- Research on agent-native serialization vs ChatML on **matched** training data.

### Downstream Use

- Load with `TelosTokenizer` + `transformers` generation on rendered prelude text.
- Evaluate with `telos eval-benchmarks --suite format_validity --format telos`.

### Out-of-Scope Use

- Not a drop-in replacement for Meta Instruct or generic chat APIs without Telos rendering.
- Simulated tool trajectories only; deploy with your own tool runtime and safety checks.
- High-stakes autonomous actions without human oversight.

## Bias, Risks, and Limitations

- Synthetic / project-generated trajectories; domain mix may not reflect production.
- Format validity ≠ task success or factual correctness.
- Llama 3.1 base limitations and biases apply.

### Recommendations

- Parse and validate generations with `telos.frames.parse` and `telos.validators.validate`.
- Compare against the ChatML merged model on the same `eval` split.
- Report metrics by `domain`.

## How to Get Started with the Model

```python
import json
import torch
from transformers import AutoModelForCausalLM

from telos.frames import parse, render
from telos.trajectory import Trajectory
from telos.tokenizer import TelosTokenizer

model_id = "kosiasuzu/telos-llama3.1-8b-lora-merged"

tt = TelosTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

# example: continue from runtime prelude (before first model frame)
frames = [
    {"type": "goal", "content": "You are a calculator assistant."},
    {"type": "mission", "content": "What is 17 * 23?"},
]
prelude = Trajectory(frames).to_frames()
prompt = render(prelude)

# use TelosTokenizer.encode/decode — not tt.hf on wire text (see README gotcha)
prompt_ids = tt.encode(prompt)
device = next(model.parameters()).device
inputs = torch.tensor([prompt_ids], device=device, dtype=torch.long)
end_id = tt.end_id
pad_id = tt.hf.pad_token_id or tt.hf.eos_token_id
with torch.no_grad():
    out = model.generate(
        inputs,
        attention_mask=torch.ones_like(inputs),
        max_new_tokens=512,
        do_sample=False,
        eos_token_id=[end_id, tt.hf.eos_token_id],
        pad_token_id=pad_id,
    )

gen_text = tt.decode(out[0][len(prompt_ids):].tolist())
generated_frames = parse(gen_text, strict=False)
print(generated_frames)
```

**Telos CLI eval** (`pip install -e ".[eval]"`):

```bash
telos eval-benchmarks --suite format_validity \
  --format telos \
  --model kosiasuzu/telos-llama3.1-8b-lora-merged \
  --adapter-mode merged \
  --dataset kosiasuzu/telos-agent-trajectory-dataset \
  --split eval \
  --output results/telos_format_validity_merged.json
```

## Training Details

### Training Data

- **Dataset:** `kosiasuzu/telos-agent-trajectory-dataset`
- **Field used:** `frames` (JSON string per row)
- **Tokenizer:** `TelosTokenizer` from the init checkpoint

### Training Procedure

```bash
torchrun --standalone --nproc_per_node=2 -m telos.cli.commands.train_telos_lora \
  --model-id kosiasuzu/telos-agent-llama-3.1-8b-init \
  --dataset kosiasuzu/telos-agent-trajectory-dataset \
  --adapter-repo-id kosiasuzu/telos-llama3.1-8b-lora-adapter \
  --merged-repo-id kosiasuzu/telos-llama3.1-8b-lora-merged
```

#### Preprocessing

- `TelosTokenizer.apply_trajectory_template` on parsed `frames`.
- Labels: train tokens in model blocks only (`mask_telos_runtime_labels`).
- **`max_length=2048`** (CLI default).

#### Training Hyperparameters

| Setting | Value |
|--------|--------|
| **Regime** | bf16, gradient checkpointing |
| **LoRA** | r=32, alpha=64, dropout=0.05 |
| **LoRA targets** | q/k/v/o_proj, gate/up/down_proj, lm_head |
| **Epochs** | 2 |
| **LR** | 2e-4, cosine, warmup 3% |
| **Batch** | per-device 1 × grad accum 32 |
| **Seed** | 42 |
| **Multi-GPU** | FSDP (with `torchrun`) |

## Evaluation

Upstream benchmark matrix (subset runs, seed 42). Full table: [`docs/benchmark_results.md`](../docs/benchmark_results.md). Regenerate: `telos eval-aggregate-results`.

| Suite | n | Primary | Secondary | avg_tokens | avg_wall_sec |
|-------|---|---------|-----------|------------|--------------|
| format_validity | 3 | valid 100% | parse 100% | 139 | 31s |
| bfcl | 5 | accuracy 60% | avg_retry 8.8 | 36,570 | 281s |
| toolbench | 2 | pass 0% | avg_steps 12 | 30,063 | 736s |
| swe (lite) | 2 | resolved 0% | avg_iter 3.5 | 10,884 | 221s |

Paired ChatML baseline on the same suites: see [`chatml-llama3.1-8b-lora-merged.md`](chatml-llama3.1-8b-lora-merged.md).

```bash
telos eval-benchmarks --suite <bfcl|toolbench|swe|format_validity> --format telos --model kosiasuzu/telos-llama3.1-8b-lora-merged
telos eval-run-all --dry-run
telos eval-aggregate-results
```

## Technical Specifications

- **Architecture:** Llama 3.1 8B causal LM, merged LoRA
- **Software:** `telos train-telos-lora`, `peft`, `transformers`


## Model Card Contact

- Hugging Face: [kosiasuzu](https://huggingface.co/kosiasuzu)
