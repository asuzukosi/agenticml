---
library_name: transformers
license: llama3.1
base_model: kosiasuzu/chatml-agent-llama-3.1-8b-init
tags:
  - llama-3.1
  - chatml
  - lora
  - agent
  - tool-use
  - agenticml-baseline
pipeline_tag: text-generation
---

# kosiasuzu/chatml-llama3.1-8b-lora-merged

ChatML baseline for the AgenticML agent-format study: **Llama 3.1 8B** with instruct-style ChatML templates and tool-call tokens, LoRA-fine-tuned on the same agent-trajectory corpus as the AgenticML-format model, then **merged** into full weights for single-checkpoint inference.

This checkpoint is the **ChatML+tools comparison arm**—same data and training recipe as `kosiasuzu/agenticml-llama3.1-8b-lora-merged`, differing only in serialization (HF `messages` + `apply_chat_template` vs AgenticML frames).

## Model Details

### Model Description

- **Model ID:** `kosiasuzu/chatml-llama3.1-8b-lora-merged`
- **Developed by:** AgenticML project / kosiasuzu
- **Model type:** Causal language model (Llama 3.1 8B), **PEFT LoRA merged** into dense weights
- **Language(s):** English (primary)
- **License:** [Llama 3.1 Community License](https://llama.meta.com/llama3_1/license/) (inherits from Meta Llama 3.1 base; confirm before commercial use)
- **Finetuned from:** [`kosiasuzu/chatml-agent-llama-3.1-8b-init`](https://huggingface.co/kosiasuzu/chatml-agent-llama-3.1-8b-init)

**Init base (`chatml-agent-llama-3.1-8b-init`):** `meta-llama/Llama-3.1-8B` weights + `meta-llama/Llama-3.1-8B-Instruct` tokenizer; ChatML special-token rows in `embed_tokens` / `lm_head` mean-pooled from seed words (`agenticml init-embeddings --format chatml`).

**Training target:** supervised next-token prediction on the **`messages`** column (ChatML conversation derived from AgenticML trajectories via `bridge.frames_to_messages` during `data-clean-push`). Labels are masked to **assistant** spans only.

### Model Sources

- **AgenticML repository:** https://github.com/kosiasuzu/agenticml
- **Dataset:** [`kosiasuzu/agenticml-agent-trajectory-dataset`](https://huggingface.co/datasets/kosiasuzu/agenticml-agent-trajectory-dataset)
- **Paired AgenticML merged model:** [`kosiasuzu/agenticml-llama3.1-8b-lora-merged`](https://huggingface.co/kosiasuzu/agenticml-llama3.1-8b-lora-merged)

## Uses

### Direct Use

- Continue multi-turn **ChatML** agent trajectories: system/user/tool messages, assistant generations with `<|eot_id|>` / tool payloads.
- Intended for **research comparison** against AgenticML-format models on the same tasks—not a general-purpose chat product.

### Downstream Use

- Load with `transformers` + `apply_chat_template` (instruct tokenizer bundled with this checkpoint). Do not use the base `meta-llama/Llama-3.1-8B` tokenizer — it has no `chat_template`.
- Run format-validity or task benchmarks with `agenticml eval-benchmarks --suite format_validity --format chatml`.

### Out-of-Scope Use

- Not validated for open-ended consumer chat, code execution without sandboxing, or high-stakes decisions (finance, health, legal).
- Do not assume parity with official Meta Instruct checkpoints; this is a **project-specific** init + LoRA on synthetic/agent SFT data.
- Tool calls in training data are **simulated**; real APIs need your own schemas and safety layers.

## Bias, Risks, and Limitations

- Trained largely on **synthetic** agent trajectories; domain mix is fixed in the generation prompt (coding, finance, travel, etc.) and may not match production traffic.
- Same **biases and safety gaps** as Llama 3.1 base and the synthetic generator.
- ChatML structural metrics (stop tokens, tool JSON shape) do not guarantee **correct** tool choice or factual answers.
- This repo is a **single merged checkpoint** for inference and eval (no separate adapter load path in the AgenticML CLI).

### Recommendations

- Use with explicit tool schemas and runtime validation.
- Compare against `kosiasuzu/agenticml-llama3.1-8b-lora-merged` on identical eval splits for fair format A/B tests.
- Report **domain-stratified** metrics (dataset includes a `domain` field).

## Same task, AgenticML format (side-by-side)

Paired checkpoint: [`kosiasuzu/agenticml-llama3.1-8b-lora-merged`](https://huggingface.co/kosiasuzu/agenticml-llama3.1-8b-lora-merged). Same dataset; ChatML uses `messages`, AgenticML uses `frames`.

**Task:** read `main.py`, return line count.

| ChatML (this model) | AgenticML (paired) |
|---------------------|-------------------|
| system + tools block in one message | separate `<\|goal\|>` + `<\|obs\|>` frames |
| user mission message | `<\|mission\|>` frame |
| `<\|python_tag\|>{"name":"read_file",...}<\|eom_id\|>` | `<\|action\|>{"tool":"read_file",...}<\|end\|>` |
| tool role with JSON content | `<\|result\|>` frame |
| plain assistant answer text | `<\|action\|>{"tool":"answer","text":"..."}<\|end\|>` |

Conversion: [`bridge.py`](../src/agenticml/bridge.py). Full pipeline: [`recipe.md`](../recipe.md).

## How to Get Started with the Model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "kosiasuzu/chatml-llama3.1-8b-lora-merged"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

messages = [
    {"role": "system", "content": "You are a helpful assistant with tools."},
    {"role": "user", "content": "What is 17 * 23?"},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
print(text)
```

**AgenticML CLI eval** (requires `pip install -e ".[eval]"` from the AgenticML repo):

```bash
agenticml eval-benchmarks --suite format_validity \
  --format chatml \
  --model kosiasuzu/chatml-llama3.1-8b-lora-merged \
  --output-dir results/benchmarks/format_validity
```

## Training Details

### Training Data

- **Dataset:** `kosiasuzu/agenticml-agent-trajectory-dataset`
- **Splits:** `train` / `eval` (stratified by `domain` at dataset build time)
- **Field used:** `messages` (JSON string per row: ChatML message list from AgenticML `frames`)
- **Rows without valid `messages` are dropped** during tokenization

### Training Procedure

**Command (representative):**

```bash
torchrun --standalone --nproc_per_node=2 -m agenticml.cli.commands.train_on_format --format chatml \
  --model-id kosiasuzu/chatml-agent-llama-3.1-8b-init \
  --dataset kosiasuzu/agenticml-agent-trajectory-dataset \
  --hub-repo-id kosiasuzu/chatml-llama3.1-8b-lora-merged
```

**Tokenizer:** training loads the instruct tokenizer from the ChatML init checkpoint (`init-embeddings --format chatml`). Hub push stores the tokenizer on this merged repo alongside the weights.

#### Preprocessing

- Instruct tokenizer: `tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)` when supported; else assistant-span masking via token-id subsequence search.
- Truncate to **`max_length=2048`** (CLI default for chatml train).

#### Training Hyperparameters

| Setting | Value |
|--------|--------|
| **Regime** | bf16, gradient checkpointing |
| **LoRA rank** | r=32, alpha=64, dropout=0.05 |
| **LoRA targets** | q/k/v/o_proj, gate/up/down_proj, **lm_head** |
| **Epochs** | 2 |
| **LR** | 2e-4, cosine, warmup 3% |
| **Batch** | per-device 1 × grad accum 32 (effective batch scales with world size) |
| **Max grad norm** | 1.0 |
| **Weight decay** | 0.1 |
| **Seed** | 42 |
| **Multi-GPU** | FSDP full_shard + auto_wrap (when using `torchrun`) |
| **Logging** | Weights & Biases (`project=agenticml-agent`, run name configurable) |

#### Speeds, Sizes, Times

Fill from your W&B / cluster logs (GPU type, world size, wall-clock, peak VRAM).

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

- **Held-out split:** `eval` on `kosiasuzu/agenticml-agent-trajectory-dataset`
- **Same examples** as AgenticML eval (paired `frames` / `messages` columns)

#### Factors

- **Domain** (`coding`, `finance`, `travel`, …) — stratification in dataset builder

#### Metrics

- **Format validity** (AgenticML harness, `--format chatml`): parseable stop structure, tool-call JSON when present, non-empty assistant content
- **Rates:** `parse_rate`, `structural_valid_rate` per domain (see eval JSON summary)

### Results

Full upstream benchmark run on RunPod L40S (June 2026): BFCL subset **n=45**, ToolBench G1 **n=10**, seed 42. Format validity **100%** parse/valid (prior validation smokes). Analysis: [`docs/report.md`](../docs/report.md).

| Suite | n | Primary | Secondary | avg_tokens | avg_wall_sec |
|-------|---|---------|-----------|------------|--------------|
| format_validity | smoke | valid **100%** | parse **100%** | — | — |
| bfcl | 45 | accuracy **60.0%** (27/45) | avg_retry **1.6** | 14,268 | 3.9s |
| toolbench | 10 | pass **0%** | avg_steps 9.8 | 21,542 | 11s |

**BFCL highlights:** simple_python 12/12, multiple 4/4, live_multiple 6/6; parallel 0/4 (single call when many required). ToolBench: all tasks failed structurally (`no finish`).

Paired AgenticML on the same suites: **4.4%** BFCL, **0%** ToolBench. See [`agenticml-llama3.1-8b-lora-merged.md`](agenticml-llama3.1-8b-lora-merged.md).

```bash
agenticml eval-benchmarks --suite bfcl --format chatml \
  --model kosiasuzu/chatml-llama3.1-8b-lora-merged
agenticml eval-run-all --suites bfcl toolbench format_validity
agenticml eval-aggregate-results
```

## Technical Specifications

### Model Architecture and Objective

- **Architecture:** Llama 3.1 8B causal LM
- **Objective:** causal LM fine-tuning with assistant-only labels (ChatML template)

### Compute Infrastructure

#### Hardware

Multi-GPU training via `torchrun` (typical: 2× GPU with FSDP); document your setup when publishing.

#### Software

- Python ≥3.10, `transformers` ≥4.43, `peft`, `datasets`, `torch`
- Training entrypoint: `agenticml train-on-format --format chatml` / `python -m agenticml.cli.commands.train_on_format`

## Citation

If you use this model in research, cite the AgenticML project repository and acknowledge Meta Llama 3.1.

**APA:** [Author]. (Year). AgenticML: Agent-native serialization for language agents. GitHub repository. Baseline model: kosiasuzu/chatml-llama3.1-8b-lora-merged.

**BibTeX:** Add when a formal report or arXiv entry is available.

## Model Card Contact

- Hugging Face: [kosiasuzu](https://huggingface.co/kosiasuzu)
- Issues: AgenticML repository issue tracker
