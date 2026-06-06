# end-to-end recipe: base llama → merged models → evaluation

This walkthrough rebuilds the AgenticML study from scratch: two **merged** checkpoints (AgenticML frames vs ChatML messages) trained on the **same** trajectory dataset, then scored on the same benchmark matrix.

All inference and eval commands load **merged** Hugging Face checkpoints only (`--model kosiasuzu/...-lora-merged`). There is no PEFT / adapter loading path in the CLI.

Copy-paste commands also live in [`command.txt`](command.txt).

**Convention:** run the paired `pytest` line(s) in each step **before** the `agenticml` commands below. Post-step checks (`verify-embeddings`, hub loads) run after the command.

---

## replication runbook

Use this section to reproduce the full pipeline on a fresh machine (e.g. RunPod L40S). One GPU is enough — use `agenticml train-on-format` directly, not `torchrun --nproc_per_node=2`.

### 0. clone and setup

```bash
git clone --recurse-submodules https://github.com/asuzukosi/agenticml.git
cd agenticml

# if you already cloned without submodules:
# git submodule update --init --recursive

python -m venv venv && source venv/bin/activate
pip install --upgrade pip

# install a cuda-matched torch wheel from https://pytorch.org/ first if needed, then:
pip install -e ".[dev,train,eval,data]"

hf auth login          # needs meta-llama access for Llama 3.1
wandb login            # optional; training logs to wandb when configured

pytest tests/test_frames.py tests/test_bridge.py tests/test_sdk.py -q
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

| Requirement | Notes |
|-------------|--------|
| Python ≥3.10 | |
| NVIDIA GPU + driver | Training and eval are impractical on CPU |
| Hugging Face access | `hf auth login` |
| Meta Llama 3.1 license | `meta-llama/Llama-3.1-8B` and `meta-llama/Llama-3.1-8B-Instruct` |
| OpenRouter (optional) | Synthetic data generation only (`OPENROUTER_API_KEY` in `.env`) |
| Docker (SWE only) | For `--suite swe` grading |

---

### 1. trajectory dataset

**Default — use the published Hub dataset** (fastest):

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('kosiasuzu/agenticml-agent-trajectory-dataset', split='train')
row = ds[0]
assert 'frames' in row and 'messages' in row
print('ok', len(ds), 'rows', row['id'])
"
```

Rows include `frames` (agenticml) and `messages` (chatml) after `data-clean-push`.

**If you only have local jsonl** (generated but not pushed):

```bash
pytest tests/test_bridge.py tests/evaluation/benchmarks/test_aggregate_results.py -q

agenticml data-clean-push \
  --input data/generated.jsonl \
  --repo-id kosiasuzu/agenticml-agent-trajectory-dataset
```

Then set `--dataset` in training to that repo id.

**Optional — generate synthetic trajectories** (requires `OPENROUTER_API_KEY` in `.env`):

```bash
pytest tests/test_bridge.py tests/evaluation/benchmarks/test_aggregate_results.py -q

agenticml data-synthetic-gen \
  --target 500 \
  --workers 4 \
  --out data/generated.jsonl

agenticml data-clean-push \
  --input data/generated.jsonl \
  --repo-id kosiasuzu/agenticml-agent-trajectory-dataset
```

---

### 2. init checkpoints

**Skip init** if `kosiasuzu/agenticml-agent-llama-3.1-8b-init` and `kosiasuzu/chatml-agent-llama-3.1-8b-init` are already on the Hub. Verify only:

```bash
agenticml verify-embeddings --format agenticml \
  --model kosiasuzu/agenticml-agent-llama-3.1-8b-init

agenticml verify-embeddings --format chatml \
  --model kosiasuzu/chatml-agent-llama-3.1-8b-init
```

Re-run init only if you want fresh weights under your own Hub account.

#### 2a. agenticml init

Maps AgenticML frame markers onto Llama reserved slots via mean-pooled seed embeddings.

```bash
pytest tests/test_agentic_template.py -q

agenticml init-embeddings --format agenticml \
  --base-model meta-llama/Llama-3.1-8B \
  --repo-id kosiasuzu/agenticml-agent-llama-3.1-8b-init

agenticml verify-embeddings --format agenticml \
  --model kosiasuzu/agenticml-agent-llama-3.1-8b-init
```

#### 2b. chatml init

Same base weights; instruct tokenizer vocab; ChatML special-token rows initialized.

```bash
pytest tests/evaluation/harness/backends/test_chatml_backend.py tests/test_tokenizer_helpers.py -q

agenticml init-embeddings --format chatml \
  --base-model meta-llama/Llama-3.1-8B \
  --instruct-tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --repo-id kosiasuzu/chatml-agent-llama-3.1-8b-init

agenticml verify-embeddings --format chatml \
  --model kosiasuzu/chatml-agent-llama-3.1-8b-init
```

---

### 3. fine-tune both formats (merged hub push only)

Both runs use the **same dataset**; AgenticML trains on `frames`, ChatML on `messages`. LoRA hub push: adapters to `<hub-repo-id>-adapter`, merged weights to `--hub-repo-id`.

Run a smoke pass first (~minutes), then the full run.

```bash
pytest tests/training/ -q

# smoke
agenticml train-on-format --format agenticml \
  --model-id kosiasuzu/agenticml-agent-llama-3.1-8b-init \
  --dataset kosiasuzu/agenticml-agent-trajectory-dataset \
  --output-dir outputs/agenticml-lora-smoke \
  --run-name agenticml-lora-smoke \
  --limit-train 32 --limit-eval 8

agenticml train-on-format --format chatml \
  --model-id kosiasuzu/chatml-agent-llama-3.1-8b-init \
  --dataset kosiasuzu/agenticml-agent-trajectory-dataset \
  --output-dir outputs/chatml-lora-smoke \
  --run-name chatml-lora-smoke \
  --limit-train 32 --limit-eval 8

# full training + hub push
agenticml train-on-format --format agenticml \
  --model-id kosiasuzu/agenticml-agent-llama-3.1-8b-init \
  --dataset kosiasuzu/agenticml-agent-trajectory-dataset \
  --output-dir outputs/agenticml-lora-full \
  --run-name agenticml-lora-full \
  --hub-repo-id kosiasuzu/agenticml-llama3.1-8b-lora-merged

agenticml train-on-format --format chatml \
  --model-id kosiasuzu/chatml-agent-llama-3.1-8b-init \
  --dataset kosiasuzu/agenticml-agent-trajectory-dataset \
  --output-dir outputs/chatml-lora-full \
  --run-name chatml-lora-full \
  --hub-repo-id kosiasuzu/chatml-llama3.1-8b-lora-merged
```

On a single GPU (e.g. L40S), omit `torchrun`; default LoRA settings (`batch=1`, `grad_accum=32`) target this setup.

**Multi-GPU alternative** (2+ GPUs):

```bash
torchrun --standalone --nproc_per_node=2 -m agenticml.cli.commands.train_on_format \
  --format agenticml \
  --model-id kosiasuzu/agenticml-agent-llama-3.1-8b-init \
  --dataset kosiasuzu/agenticml-agent-trajectory-dataset \
  --output-dir outputs/agenticml-lora-full \
  --run-name agenticml-lora-full \
  --hub-repo-id kosiasuzu/agenticml-llama3.1-8b-lora-merged

torchrun --standalone --nproc_per_node=2 -m agenticml.cli.commands.train_on_format \
  --format chatml \
  --model-id kosiasuzu/chatml-agent-llama-3.1-8b-init \
  --dataset kosiasuzu/agenticml-agent-trajectory-dataset \
  --output-dir outputs/chatml-lora-full \
  --run-name chatml-lora-full \
  --hub-repo-id kosiasuzu/chatml-llama3.1-8b-lora-merged
```

**Check after training:**

```bash
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('kosiasuzu/agenticml-llama3.1-8b-lora-merged')
print('agenticml merged ok', tok.convert_tokens_to_ids('<|reserved_special_token_7|>'))
"
```

---

### 4. eval dependencies

See [`docs/eval_dependencies.md`](docs/eval_dependencies.md) for BFCL / ToolBench / SWE layout.

```bash
pytest tests/evaluation/benchmarks/ -q \
  --ignore=tests/evaluation/benchmarks/swe/test_smoke_pipeline.py

git submodule update --init --recursive
pip install -e ".[eval-benchmarks]"
```

ToolBench cached data (one-time):

```bash
cd third_party/ToolBench
hf download nullwwg/toolbench-data data.zip --local-dir .
unzip -o data.zip
cd ../..
```

**Check after setup:**

```bash
agenticml eval-run-all --dry-run   # lists 8 matrix cells, no gpu
```

SWE also needs Docker running on the host.

---

### 5. per-suite smoke (catch errors before long runs)

Use **merged** model ids. Run these in order; each step should finish in minutes (except first SWE docker pull).

#### format validity (parse + structure)

```bash
agenticml eval-benchmarks --suite format_validity --format agenticml \
  --model kosiasuzu/agenticml-llama3.1-8b-lora-merged \
  --num-examples 5 --output-dir results/benchmarks/format_validity

agenticml eval-benchmarks --suite format_validity --format chatml \
  --model kosiasuzu/chatml-llama3.1-8b-lora-merged \
  --num-examples 5 --output-dir results/benchmarks/format_validity
```

#### toolbench (cached tools, no live api)

```bash
agenticml eval-benchmarks --suite toolbench --format agenticml \
  --model kosiasuzu/agenticml-llama3.1-8b-lora-merged \
  --num-examples 1 --output-dir results/benchmarks/toolbench
```

#### bfcl (subset)

```bash
agenticml eval-benchmarks --suite bfcl --format agenticml \
  --model kosiasuzu/agenticml-llama3.1-8b-lora-merged \
  --num-examples 3 --no-score --output-dir results/benchmarks/bfcl
```

#### swe (docker; inference only first)

```bash
# pre-pull images — can take 10–30+ min first time; see docs/eval_swe_bench.md
python -c "
from agenticml.evaluation.benchmarks.swe.env import pull_instance_image
from agenticml.evaluation.benchmarks.swe.subset import load_entries
for e in load_entries(1, seed=42):
    pull_instance_image(e)
"

SWE_VERBOSE=1 agenticml eval-benchmarks --suite swe --format agenticml \
  --model kosiasuzu/agenticml-llama3.1-8b-lora-merged \
  --num-examples 1 --max-iterations 20 --no-score \
  --output-dir results/benchmarks/swe
```

---

### 6. full benchmark matrix + publish

```bash
# inference-only smoke across suites (no swe docker grade)
agenticml eval-run-all --num-examples 3 --no-score \
  --suites bfcl toolbench format_validity swe

# full matrix (long; use --continue-on-error)
agenticml eval-run-all --continue-on-error

# re-grade bfcl/swe from saved rows
agenticml eval-run-all --score-only --suites bfcl swe

# publish table
agenticml eval-aggregate-results
```

Outputs: `results/benchmarks/<suite>/<format>/summary.json`. Aggregated markdown: [`docs/benchmark_results.md`](docs/benchmark_results.md).

#### publish hub model cards

Edit [`model_cards/`](model_cards/) locally, then push each file as the repo `README.md`:

```bash
hf upload kosiasuzu/agenticml-llama3.1-8b-lora-merged \
  model_cards/agenticml-llama3.1-8b-lora-merged.md README.md \
  --commit-message "update model card"

hf upload kosiasuzu/chatml-llama3.1-8b-lora-merged \
  model_cards/chatml-llama3.1-8b-lora-merged.md README.md \
  --commit-message "update model card"
```

---

## practical notes

| Topic | Guidance |
|--------|----------|
| **Repo** | Clone `asuzukosi/agenticml` for code; Hub ids in commands point at `kosiasuzu/...` unless you re-init/train to your account |
| **pip** | Run `pip install --upgrade pip` before editable install; older pip (24.x) fails on optional `file://` deps in metadata |
| **Submodules** | Required for BFCL / ToolBench / SWE eval, not for train-only |
| **Disk** | ToolBench `data.zip` is large; ensure enough volume |
| **SWE** | Needs Docker + first image pull can take 30+ min |
| **Time order** | Setup → dataset (or verify Hub) → init (or verify) → train agenticml → train chatml → eval setup → smoke evals → full matrix |

---

## quick reference: default merged models

| Format | Merged checkpoint |
|--------|-------------------|
| agenticml | `kosiasuzu/agenticml-llama3.1-8b-lora-merged` |
| chatml | `kosiasuzu/chatml-llama3.1-8b-lora-merged` |

Init bases (training only): `kosiasuzu/agenticml-agent-llama-3.1-8b-init`, `kosiasuzu/chatml-agent-llama-3.1-8b-init`.

ChatML inference: `AutoTokenizer.from_pretrained(kosiasuzu/chatml-llama3.1-8b-lora-merged)` (instruct chat template pushed with merged weights).

---

## format A/B: same task, two serializations

See [README — same task, two traces](README.md#same-task-two-traces-agenticml-vs-chatml) and model cards for side-by-side examples. Conversion logic: [`src/agenticml/bridge.py`](src/agenticml/bridge.py).
