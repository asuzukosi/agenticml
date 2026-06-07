---
library_name: transformers
license: llama3.1
base_model: kosiasuzu/agenticml-agent-llama-3.1-8b-init
tags:
  - llama-3.1
  - agenticml
  - lora
  - agent
  - tool-use
pipeline_tag: text-generation
---

# kosiasuzu/agenticml-llama3.1-8b-lora-merged

**AgenticML-format** agent model: Llama 3.1 8B with AgenticML frame markers on reserved vocabulary slots, LoRA-fine-tuned on agent trajectories, then **merged** into dense weights for single-checkpoint inference.

Paired ChatML baseline for the same study: [`kosiasuzu/chatml-llama3.1-8b-lora-merged`](https://huggingface.co/kosiasuzu/chatml-llama3.1-8b-lora-merged).

## Model Details

### Model Description

- **Model ID:** `kosiasuzu/agenticml-llama3.1-8b-lora-merged`
- **Developed by:** AgenticML project / kosiasuzu
- **Model type:** Causal language model (Llama 3.1 8B), **PEFT LoRA merged** into dense weights
- **Language(s):** English (primary)
- **License:** [Llama 3.1 Community License](https://llama.meta.com/llama3_1/license/)
- **Finetuned from:** [`kosiasuzu/agenticml-agent-llama-3.1-8b-init`](https://huggingface.co/kosiasuzu/agenticml-agent-llama-3.1-8b-init)

**Init base (`agenticml-agent-llama-3.1-8b-init`):** `meta-llama/Llama-3.1-8B` with AgenticML marker rows (`<|goal|>` … `<|reward|>`) initialized in `embed_tokens` / `lm_head` via mean-pooled seed embeddings (`agenticml init-embeddings --format agenticml`).

**Training target:** supervised next-token prediction on the **`frames`** column (AgenticML frame sequences). Loss applies only inside **model-owned** blocks (`belief`, `plan`, `think`, `action`, `end`); runtime frames (`goal`, `mission`, `obs`, `result`, etc.) are masked.

### Model Sources

- **AgenticML repository:** https://github.com/kosiasuzu/agenticml
- **Dataset:** [`kosiasuzu/agenticml-agent-trajectory-dataset`](https://huggingface.co/datasets/kosiasuzu/agenticml-agent-trajectory-dataset)
- **Paired ChatML merged model:** [`kosiasuzu/chatml-llama3.1-8b-lora-merged`](https://huggingface.co/kosiasuzu/chatml-llama3.1-8b-lora-merged)

## Uses

### Direct Use

- Continue **AgenticML** agent trajectories: typed frames (`goal`, `mission`, `obs`, `belief`, `plan`, `think`, `action`, `result`, …) rendered with AgenticML markers.
- Research on agent-native serialization vs ChatML on **matched** training data.

### Downstream Use

- Load with `AutoTokenizer.from_pretrained` + `apply_chat_template(trajectory.to_dict(), ...)`.
- Advance trajectories with `agenticml.sdk.step`; inject tool schemas via `with_tool_obs`.
- Evaluate with `agenticml eval-benchmarks --suite format_validity --format agenticml`.

### Out-of-Scope Use

- Not a drop-in replacement for Meta Instruct or generic chat APIs without AgenticML rendering.
- Simulated tool trajectories only; deploy with your own tool runtime and safety checks.
- High-stakes autonomous actions without human oversight.

## Bias, Risks, and Limitations

- Synthetic / project-generated trajectories; domain mix may not reflect production.
- Format validity ≠ task success or factual correctness.
- Llama 3.1 base limitations and biases apply.

### Recommendations

- Parse generations with `parse_reserved_wire` and validate with `agenticml.validators.validate`.
- Compare against the ChatML merged model on the same `eval` split.
- Report metrics by `domain`.

## Same task, ChatML baseline (side-by-side)

Paired checkpoint: [`kosiasuzu/chatml-llama3.1-8b-lora-merged`](https://huggingface.co/kosiasuzu/chatml-llama3.1-8b-lora-merged). Same dataset and training recipe; only serialization differs.

**Task:** read `main.py`, return line count.

| AgenticML (this model) | ChatML (baseline) |
|------------------------|-------------------|
| `<\|goal\|>You are a coding assistant.` | `{"role":"system","content":"You are a coding assistant...."}` |
| `<\|mission\|>How many lines are in main.py?` | `{"role":"user","content":"How many lines are in main.py?"}` |
| `<\|action\|>{"tool":"read_file","path":"main.py"}<\|end\|>` | assistant `tool_calls`: `read_file` + `{"path":"main.py"}` |
| `<\|result\|>{"tool":"read_file","value":"..."}` | `{"role":"tool","content":"{\"tool\":\"read_file\",...}"}` |
| `<\|action\|>{"tool":"answer","text":"main.py has 2 lines."}<\|end\|>` | `{"role":"assistant","content":"main.py has 2 lines."}` |

ChatML tool-call generation uses `<\|python_tag\|>{...}<\|eom_id\|>`. Conversion: [`bridge.py`](../src/agenticml/bridge.py). Full pipeline: [`recipe.md`](../recipe.md).

## How to Get Started with the Model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agenticml.agentic_template import parse_reserved_wire
from agenticml.constants import END_MARKER_TOKEN_ID
from agenticml.trajectory import Trajectory

model_id = "kosiasuzu/agenticml-llama3.1-8b-lora-merged"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

frames = [
    {"type": "goal", "content": "You are a calculator assistant."},
    {"type": "mission", "content": "What is 17 * 23?"},
]
prompt_ids = tokenizer.apply_chat_template(
    Trajectory(frames).to_dict(),
    tokenize=True,
    add_generation_prompt=False,
    add_special_tokens=False,
)
device = next(model.parameters()).device
inputs = torch.tensor([prompt_ids], device=device, dtype=torch.long)
pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
with torch.no_grad():
    out = model.generate(
        inputs,
        attention_mask=torch.ones_like(inputs),
        max_new_tokens=512,
        do_sample=False,
        eos_token_id=[END_MARKER_TOKEN_ID, tokenizer.eos_token_id],
        pad_token_id=pad_id,
    )

gen_text = tokenizer.decode(out[0][len(prompt_ids):].tolist())
print(parse_reserved_wire(gen_text, strict=False))
```

**AgenticML CLI eval** (`pip install -e ".[eval]"`):

```bash
agenticml eval-benchmarks --suite format_validity \
  --format agenticml \
  --model kosiasuzu/agenticml-llama3.1-8b-lora-merged \
  --output-dir results/benchmarks/format_validity
```

## Training Details

### Training Data

- **Dataset:** `kosiasuzu/agenticml-agent-trajectory-dataset`
- **Field used:** `frames` (JSON string per row)
- **Tokenizer:** hub `AutoTokenizer` from the init checkpoint (baked chat template)

### Training Procedure

```bash
torchrun --standalone --nproc_per_node=2 -m agenticml.cli.commands.train_on_format --format agenticml \
  --model-id kosiasuzu/agenticml-agent-llama-3.1-8b-init \
  --dataset kosiasuzu/agenticml-agent-trajectory-dataset \
  --hub-repo-id kosiasuzu/agenticml-llama3.1-8b-lora-merged
```

#### Preprocessing

- `tokenizer.apply_chat_template(frames, tokenize=True, ...)` on parsed `frames`.
- Labels: train tokens in model blocks only (`mask_agenticml_runtime_labels`).
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
| **Multi-GPU** | `torchrun` DDP |

## Evaluation

Full upstream benchmark run on RunPod L40S (June 2026): BFCL subset **n=45**, ToolBench G1 **n=10**, seed 42. Format validity **100%** parse/valid (prior validation smokes; no n=100 re-run). SWE-bench not run (no Docker). Full report: [`docs/report.md`](../docs/report.md).

| Suite | n | Primary | Secondary | avg_tokens | avg_wall_sec |
|-------|---|---------|-----------|------------|--------------|
| format_validity | smoke | valid **100%** | parse **100%** | — | — |
| bfcl | 45 | accuracy **4.4%** (2/45) | avg_retry **3.9** | 140,018 | 224s |
| toolbench | 10 | pass **0%** | avg_steps 1.0 | 2,121 | 30s |

**BFCL by category (agenticml):** simple_python 0/12, multiple 0/4, parallel 0/4, parallel_multiple 2/4, live 0/13, multi_turn 0/8. Most failures are wrong call **shape** (duplicate JSON arrays on single-call tasks), not missing tools — see runpod report.

Paired ChatML on the same suites: **60%** BFCL accuracy, **0%** ToolBench. See [`chatml-llama3.1-8b-lora-merged.md`](chatml-llama3.1-8b-lora-merged.md).

```bash
agenticml eval-benchmarks --suite bfcl --format agenticml \
  --model kosiasuzu/agenticml-llama3.1-8b-lora-merged
agenticml eval-run-all --suites bfcl toolbench format_validity
agenticml eval-aggregate-results
```

## Technical Specifications

- **Architecture:** Llama 3.1 8B causal LM, merged LoRA
- **Software:** `agenticml train-on-format --format agenticml`, `peft`, `transformers`


## Model Card Contact

- Hugging Face: [kosiasuzu](https://huggingface.co/kosiasuzu)
