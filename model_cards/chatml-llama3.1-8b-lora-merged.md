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
  - telos-baseline
pipeline_tag: text-generation
---

# kosiasuzu/chatml-llama3.1-8b-lora-merged

ChatML baseline for the Telos agent-format study: **Llama 3.1 8B** with instruct-style ChatML templates and tool-call tokens, LoRA-fine-tuned on the same agent-trajectory corpus as the Telos-format model, then **merged** into full weights for single-checkpoint inference.

This checkpoint is the **ChatML+tools comparison arm**—same data and training recipe as `kosiasuzu/telos-llama3.1-8b-lora-merged`, differing only in serialization (HF `messages` + `apply_chat_template` vs Telos frames).

## Model Details

### Model Description

- **Model ID:** `kosiasuzu/chatml-llama3.1-8b-lora-merged`
- **Developed by:** Telos project / kosiasuzu
- **Model type:** Causal language model (Llama 3.1 8B), **PEFT LoRA merged** into dense weights
- **Language(s):** English (primary)
- **License:** [Llama 3.1 Community License](https://llama.meta.com/llama3_1/license/) (inherits from Meta Llama 3.1 base; confirm before commercial use)
- **Finetuned from:** [`kosiasuzu/chatml-agent-llama-3.1-8b-init`](https://huggingface.co/kosiasuzu/chatml-agent-llama-3.1-8b-init)
- **Adapter (pre-merge):** [`kosiasuzu/chatml-llama3.1-8b-lora-adapter`](https://huggingface.co/kosiasuzu/chatml-llama3.1-8b-lora-adapter) (optional; this repo is the fused artifact)

**Init base (`chatml-agent-llama-3.1-8b-init`):** `meta-llama/Llama-3.1-8B` weights + `meta-llama/Llama-3.1-8B-Instruct` tokenizer; ChatML special-token rows in `embed_tokens` / `lm_head` mean-pooled from seed words (`telos init-chatml-embeddings`).

**Training target:** supervised next-token prediction on the **`messages`** column (ChatML conversation derived from Telos trajectories via `telos data-telos-to-chatml`). Labels are masked to **assistant** spans only.

### Model Sources

- **Telos repository:** https://github.com/kosiasuzu/talos
- **Dataset:** [`kosiasuzu/telos-agent-trajectory-dataset`](https://huggingface.co/datasets/kosiasuzu/telos-agent-trajectory-dataset)
- **Paired Telos merged model:** [`kosiasuzu/telos-llama3.1-8b-lora-merged`](https://huggingface.co/kosiasuzu/telos-llama3.1-8b-lora-merged)

## Uses

### Direct Use

- Continue multi-turn **ChatML** agent trajectories: system/user/tool messages, assistant generations with `<|eot_id|>` / tool payloads.
- Intended for **research comparison** against Telos-format models on the same tasks—not a general-purpose chat product.

### Downstream Use

- Load with `transformers` + `apply_chat_template` (tokenizer bundled with this checkpoint).
- Run format-validity or task benchmarks with `telos eval-benchmarks --suite format_validity --format chatml`.

### Out-of-Scope Use

- Not validated for open-ended consumer chat, code execution without sandboxing, or high-stakes decisions (finance, health, legal).
- Do not assume parity with official Meta Instruct checkpoints; this is a **project-specific** init + LoRA on synthetic/agent SFT data.
- Tool calls in training data are **simulated**; real APIs need your own schemas and safety layers.

## Bias, Risks, and Limitations

- Trained largely on **synthetic** agent trajectories; domain mix is fixed in the generation prompt (coding, finance, travel, etc.) and may not match production traffic.
- Same **biases and safety gaps** as Llama 3.1 base and the synthetic generator.
- ChatML structural metrics (stop tokens, tool JSON shape) do not guarantee **correct** tool choice or factual answers.
- Merged weights fix adapter at train time; no on-the-fly LoRA switching.

### Recommendations

- Use with explicit tool schemas and runtime validation.
- Compare against `kosiasuzu/telos-llama3.1-8b-lora-merged` on identical eval splits for fair format A/B tests.
- Report **domain-stratified** metrics (dataset includes a `domain` field).

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

**Telos CLI eval** (requires `pip install -e ".[eval]"` from the Telos repo):

```bash
telos eval-benchmarks --suite format_validity \
  --format chatml \
  --model kosiasuzu/chatml-llama3.1-8b-lora-merged \
  --adapter-mode merged \
  --dataset kosiasuzu/telos-agent-trajectory-dataset \
  --split eval \
  --output results/chatml_format_validity_merged.json
```

## Training Details

### Training Data

- **Dataset:** `kosiasuzu/telos-agent-trajectory-dataset`
- **Splits:** `train` / `eval` (stratified by `domain` at dataset build time)
- **Field used:** `messages` (JSON string per row: ChatML message list from Telos `frames`)
- **Rows without valid `messages` are dropped** during tokenization

### Training Procedure

**Command (representative):**

```bash
torchrun --standalone --nproc_per_node=2 -m telos.cli.commands.train_chatml_lora \
  --model-id kosiasuzu/chatml-agent-llama-3.1-8b-init \
  --dataset kosiasuzu/telos-agent-trajectory-dataset \
  --adapter-repo-id kosiasuzu/chatml-llama3.1-8b-lora-adapter \
  --merged-repo-id kosiasuzu/chatml-llama3.1-8b-lora-merged
```

#### Preprocessing

- `tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)` when supported; else assistant-span masking via token-id subsequence search.
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
| **Logging** | Weights & Biases (`project=telos-agent`, run name configurable) |

#### Speeds, Sizes, Times

Fill from your W&B / cluster logs (GPU type, world size, wall-clock, peak VRAM).

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

- **Held-out split:** `eval` on `kosiasuzu/telos-agent-trajectory-dataset`
- **Same examples** as Telos eval (paired `frames` / `messages` columns)

#### Factors

- **Domain** (`coding`, `finance`, `travel`, …) — stratification in dataset builder

#### Metrics

- **Format validity** (Telos harness, `--format chatml`): parseable stop structure, tool-call JSON when present, non-empty assistant content
- **Rates:** `parse_rate`, `structural_valid_rate` per domain (see eval JSON summary)

### Results

Upstream benchmark matrix (subset runs, seed 42). Full table: [`docs/benchmark_results.md`](../docs/benchmark_results.md). Regenerate: `telos eval-aggregate-results`.

| Suite | n | Primary | Secondary | avg_tokens | avg_wall_sec |
|-------|---|---------|-----------|------------|--------------|
| bfcl | 5 | accuracy 60% | avg_retry 1.6 | 20,139 | 192s |
| toolbench | 2 | pass 0% | avg_steps 12 | 40,350 | 848s |
| format_validity | — | not run | — | — | — |
| swe (lite) | — | not run | — | — | — |

Paired Telos model on the same suites: [`telos-llama3.1-8b-lora-merged.md`](telos-llama3.1-8b-lora-merged.md).

```bash
telos eval-benchmarks --suite <bfcl|toolbench|swe|format_validity> --format chatml --model kosiasuzu/chatml-llama3.1-8b-lora-merged
telos eval-run-all --dry-run
telos eval-aggregate-results
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
- Training entrypoint: `telos train-chatml-lora` / `telos.cli.commands.train_chatml_lora`

## Citation

If you use this model in research, cite the Telos project repository and acknowledge Meta Llama 3.1.

**APA:** [Author]. (Year). Telos: Agent-native serialization for language agents. GitHub repository. Baseline model: kosiasuzu/chatml-llama3.1-8b-lora-merged.

**BibTeX:** Add when a formal report or arXiv entry is available.

## Model Card Contact

- Hugging Face: [kosiasuzu](https://huggingface.co/kosiasuzu)
- Issues: Telos repository issue tracker
