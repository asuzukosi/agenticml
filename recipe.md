# End-to-End Recipe: Base Llama → Merged Models → Evaluation

This walkthrough rebuilds the AgenticML study from scratch: two **merged** checkpoints (AgenticML frames vs ChatML messages) trained on the **same** trajectory dataset, then scored on the same benchmark matrix.

All inference and eval commands load **merged** Hugging Face checkpoints only (`--model kosiasuzu/...-lora-merged`). There is no PEFT / adapter loading path in the CLI.

Copy-paste commands also live in [`command.txt`](command.txt).

**Convention:** run the paired `pytest` line(s) in each step **before** the `agenticml` commands below. Post-step checks (`verify-embeddings`, hub loads) run after the command.

---

## Replication Runbook

Use this section to reproduce the full pipeline on a fresh machine (e.g. RunPod L40S). One GPU is enough — use `agenticml train-on-format` directly, not `torchrun --nproc_per_node=2`.

### 0. Clone and Setup

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
export HF_TOKEN=...    # or add to ~/.bashrc; required for hub push + dataset downloads
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

### 1. Trajectory Dataset

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

### 2. Init Checkpoints

**Skip init** if `kosiasuzu/agenticml-agent-llama-3.1-8b-init` and `kosiasuzu/chatml-agent-llama-3.1-8b-init` are already on the Hub. Verify only:

```bash
agenticml verify-embeddings --format agenticml \
  --model kosiasuzu/agenticml-agent-llama-3.1-8b-init

agenticml verify-embeddings --format chatml \
  --model kosiasuzu/chatml-agent-llama-3.1-8b-init
```

Re-run init only if you want fresh weights under your own Hub account.

#### 2a. AgenticML Init

Maps AgenticML frame markers onto Llama reserved slots via mean-pooled seed embeddings.

```bash
pytest tests/test_agentic_template.py -q

agenticml init-embeddings --format agenticml \
  --base-model meta-llama/Llama-3.1-8B \
  --repo-id kosiasuzu/agenticml-agent-llama-3.1-8b-init

agenticml verify-embeddings --format agenticml \
  --model kosiasuzu/agenticml-agent-llama-3.1-8b-init
```

#### 2b. ChatML Init

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

### 3. Fine-Tune Both Formats (Merged Hub Push Only)

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

### 4. Eval Dependencies

Benchmark suites: **BFCL**, **ToolBench** (cached tool results), **SWE-bench-Lite** (`swebench` grader), plus **format validity** on our models.

#### Python Packages

```bash
pytest tests/evaluation/benchmarks/ -q \
  --ignore=tests/evaluation/benchmarks/swe/test_smoke_pipeline.py

git submodule update --init --recursive

# staged install: pip install -e ".[eval-benchmarks]" often fails on the editable
# bfcl_eval file:// dependency; install bfcl separately instead
pip install -e ".[eval]"
pip install -e third_party/gorilla/berkeley-function-call-leaderboard
pip install soundfile   # bfcl scoring (qwen-agent dependency)
pip install swebench   # optional; skip on hosts without Docker (no SWE grading)
# if toolbench fails with ModuleNotFoundError: termcolor, re-run pip install -e ".[eval]"
```

| extra | packages |
|-------|----------|
| `eval` | torch, datasets, tqdm, termcolor (toolbench upstream) |
| `eval-benchmarks` (metadata) | `agenticml[eval]`, [swebench](https://github.com/princeton-nlp/SWE-bench), editable `bfcl_eval` — install bfcl via staged command above |

Verify:

```bash
python -c "import bfcl_eval; print('bfcl ok')"
agenticml eval-run-all --dry-run   # lists matrix cells, no gpu
```

Scoring imports `bfcl_eval.eval_checker.eval_runner`, which loads gorilla’s full dependency set. `.[eval]` alone is not enough for BFCL.

#### third_party submodules

| path | upstream |
|------|----------|
| `third_party/gorilla` | https://github.com/ShishirPatil/gorilla — BFCL scoring and ChatML eval path |
| `third_party/SWE-bench` | https://github.com/princeton-nlp/SWE-bench — datasets and `run_evaluation` grader |
| `third_party/mini-swe-agent` | https://github.com/SWE-agent/mini-swe-agent — agent loop for SWE-bench runs |
| `third_party/ToolBench` | https://github.com/OpenBMB/ToolBench — upstream tool env + inference |

Benchmark code: `src/agenticml/evaluation/benchmarks/` with shared `BenchmarkSuite` interface. Tests mirror under `tests/evaluation/benchmarks/`.

#### ToolBench Data (~2 GB)

Eval needs the OpenBMB **on-disk tree** under `data/` (`test_instruction`, `test_query_ids`, `toolenv`, `tool_response_cache`). Not a generic HF parquet dataset.

```bash
cd third_party/ToolBench

# must pass --repo-type dataset; without it HF looks for a *model* repo and returns
# "Repository not found". OpenBMB Google Drive / Tsinghua links are often dead (404).
hf download nullwwg/toolbench-data data.zip --repo-type dataset --local-dir .

# extract with python (many minimal images, e.g. RunPod, have no unzip package)
python -c "import zipfile; zipfile.ZipFile('data.zip').extractall('.')"

cd ../..

# sanity check (needs test_instruction, toolenv, tool_response_cache under data/)
ls data/test_instruction/G1_instruction.json
ls data/toolenv/tools | head
```

| source | status |
|--------|--------|
| [`nullwwg/toolbench-data`](https://huggingface.co/datasets/nullwwg/toolbench-data) | community mirror of OpenBMB `data.zip` (preferred) |
| OpenBMB Google Drive / Tsinghua Cloud | often dead (404) |
| `Maurus/ToolBench` on HF | wrong format (flat table; no `toolenv` / cache tree) |

**Own mirror (optional):**

```bash
hf upload kosiasuzu/toolbench-data data.zip data.zip --repo-type dataset
hf download kosiasuzu/toolbench-data data.zip --repo-type dataset --local-dir third_party/ToolBench
```

Set `export TOOLBENCH_DATA=/path/to/ToolBench` when unzipped elsewhere. ToolBench eval uses pinned cache artifacts only (no live RapidAPI in the default setup).

#### Suite Reference

**Matrix runner:**

```bash
agenticml eval-run-all --dry-run
agenticml eval-run-all --num-examples 5
agenticml eval-aggregate-results   # writes results/benchmarks/aggregate_table.md
```

Default models: `kosiasuzu/agenticml-llama3.1-8b-lora-merged`, `kosiasuzu/chatml-llama3.1-8b-lora-merged`.

**BFCL** — subset IDs in `src/agenticml/evaluation/benchmarks/bfcl/subset.py` (45 cases, seed 42; excludes irrelevance; long multi-turn cases swapped for faster 2–3 turn examples):

```bash
agenticml eval-benchmarks --suite bfcl --format agenticml --model <hf_id> --num-examples 5
agenticml eval-benchmarks --suite bfcl --format chatml --model <hf_id> --num-examples 5
```

Writes gorilla result files under `results/benchmarks/bfcl/`, scores via upstream `evaluate_task`, envelope at `results/benchmarks/bfcl/<format>/summary.json`. Re-score: `--score-only`.

**Format validity** — always uses `kosiasuzu/agenticml-agent-trajectory-dataset` / `eval`:

```bash
agenticml eval-benchmarks --suite format_validity --format agenticml --model <hf_id> --num-examples 100
```

**ToolBench** — 10 pinned G1_instruction query IDs in `toolbench/subset.py`; cached env in `toolbench/cache.py`:

```bash
agenticml eval-benchmarks --suite toolbench --format agenticml --model <hf_id> --num-examples 3
```

Structural scoring in `toolbench/score.py`. Optional full GPT judge: `OPENAI_API_KEY` + `TOOLEVAL_GPT=1`.

**SWE-bench-Lite** — 30 instances in `swe/subset.py`; needs Docker. Full ops below.

SWE also needs Docker running on the host.

#### SWE-bench-Lite Evaluation

**SWE-bench-Lite** runs through [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) and grades with upstream [swebench](https://github.com/princeton-nlp/SWE-bench) `run_evaluation`. AgenticML and ChatML each use a **merged** checkpoint with format-specific backends (`AgenticMLBackend` / `ChatMLBackend`); the Docker agent loop and grader stay upstream.

**Subset:** Pinned instance IDs live in `SUBSET_IDS` in `src/agenticml/evaluation/benchmarks/swe/subset.py`: **30** tasks sampled with seed 42 from `princeton-nlp/SWE-Bench_Lite` / `test` (300 total). Loader: `agenticml.evaluation.benchmarks.swe.subset` (`load_subset_ids`, `load_subset`, `load_entries`).

```python
from agenticml.evaluation.benchmarks.swe.subset import load_subset

subset = load_subset()
print(subset.instance_ids[0], subset.entries[0]["repo"])
```

**Live progress** — per-step action logs (bash command + truncated output):

```bash
SWE_VERBOSE=1 agenticml eval-benchmarks --suite swe --format agenticml --model <model> --num-examples 1 --no-score
```

Other signals while a run is in flight:

- **tqdm** postfix shows the current `instance_id`
- **mini-swe** Docker logs: `DEBUG:minisweagent.environment` when containers start
- **GPU:** `watch -n1 nvidia-smi` — generation should show util spikes
- **Container shell:** `docker exec -it <minisweagent-name> bash` (repo at `/testbed`)

Full trajectory lands in per-instance JSON under `results/benchmarks/swe/<model_slug>/` after each task completes.

**GPU:** SWE inference runs the **8B model in a multi-turn loop** (up to 250 steps). It effectively requires a working GPU.

If you see `CUDA initialization: The NVIDIA driver on your system is too old`, PyTorch falls back to CPU and a single step can take hours. Verify before eval:

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'torch cuda:', torch.version.cuda)"
nvidia-smi
```

Fix by aligning **driver** and **PyTorch CUDA build** (reinstall torch for your driver, or update the NVIDIA driver). A 4060 Ti with ~16 GB is enough; CPU-only smoke is not practical.

| Component | Path / Package |
|-----------|----------------|
| SWE-bench grader | `swebench` (pyproject extra) + `third_party/SWE-bench` submodule |
| Agent loop | `third_party/mini-swe-agent` submodule |
| Dataset | Hugging Face `princeton-nlp/SWE-Bench_Lite` (cached on first load) |

**Docker:**

- Eval **requires Docker** (or Singularity on HPC — see mini-swe-agent docs).
- The user running `agenticml` must access Docker **without sudo** (`docker ps` works in the same shell).
- `docker run` **exit 126** — permission denied on `/var/run/docker.sock` (`sudo usermod -aG docker $USER`, re-login).
- `docker run` **exit 125** + `containerd.sock: connection refused` — **containerd is down**. Fix before eval:

```bash
sudo journalctl -u containerd -n 30 --no-pager
sudo systemctl stop docker containerd
sudo rm -f /run/containerd/containerd.sock
sudo containerd config default | sudo tee /etc/containerd/config.toml
sudo systemctl start containerd && sudo systemctl start docker
sudo systemctl status containerd   # must be active (running)
```

- **Pre-pull instance images** before `agenticml eval-benchmarks --suite swe`. First pull can take 10–30+ minutes per image; `docker run` during eval only waits ~600s.

```bash
python -c "
from agenticml.evaluation.benchmarks.swe.env import pull_instance_image
from agenticml.evaluation.benchmarks.swe.subset import load_entries
for e in load_entries(2, seed=42):
    print('pulling', e['instance_id'])
    pull_instance_image(e)
"
```

- Disk: SWE eval images are large (~tens of GB across many instances).

**mini-swe-agent (reference):**

```bash
cd third_party/mini-swe-agent
pip install -e .
mini-extra swebench \
  --subset lite \
  --split test \
  --model <provider/model> \
  --filter 'django__django-11099|sympy__sympy-12454' \
  -w 1 \
  -o /tmp/swe-out
```

Single-instance debug:

```bash
mini-extra swebench-single --subset lite --split test -i sympy__sympy-12454 --model <model>
```

Docs: `third_party/mini-swe-agent/docs/usage/swebench.md`

**Grading (reference):** After predictions exist as `preds.json` / `preds.jsonl`:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-Bench_Lite \
  --split test \
  --predictions_path /path/to/preds.jsonl \
  --max_workers 4 \
  --run_id agenticml-swe-smoke
```

Logs: `logs/run_evaluation/` under the working directory. Primary metric: **resolved rate** (instance-level pass).

**AgenticML harness:**

- Subset: `agenticml.evaluation.benchmarks.swe.subset`
- Prelude: `instance_to_prelude` builds goal/mission from `problem_statement` + SWE instructions
- Tools: `registry_from_env` maps AgenticML `bash` actions to mini-swe `env.execute` (captures `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` as `model_patch`)
- Loop: `run_agenticml_swe(backend, bridge, instance)` drives `AgenticMLBackend.step` until submit or limit
- Task entry: `agenticml.evaluation.benchmarks.swe.agenticml.run_one_task` / `swe.chatml.run_one_task` wire Docker env + loop; `io.pred_entry` for `preds.json`
- Grading: `agenticml.evaluation.benchmarks.swe.score` writes `preds.json`, calls `swebench.harness.run_evaluation`, and reads resolved rate from the upstream report

CLI:

```bash
# inference only (no docker grader) — smoke with 3 instances
agenticml eval-benchmarks --suite swe --format agenticml --model <model> --num-examples 3 --no-score

# full run: agent loop + swebench grader
agenticml eval-benchmarks --suite swe --format agenticml --model <model> --num-examples 3

# grade existing result rows
agenticml eval-benchmarks --suite swe --format agenticml --model <model> --score-only
```

Output: `results/benchmarks/swe/<format>/summary.json` (envelope), per-instance rows under `results/benchmarks/swe/<model_slug>/`, grader artifacts under `results/benchmarks/swe/score/<model_slug>/`.

**Smoke checklist:**

1. `docker info` succeeds.
2. `python -c "from agenticml.evaluation.benchmarks.swe.subset import load_subset; print(len(load_subset().entries))"` → `30`.
3. `mini-extra swebench-single --subset lite --split test -i astropy__astropy-14995 ...` completes one trajectory (optional).
4. Grade a tiny preds file with `run_evaluation` (optional).

#### Sync Benchmark Results (RunPod → Laptop)

RunPod SSH proxy often breaks `scp`. Use a private Hub dataset.

**Hub dataset:** `kosiasuzu/agenticml-runpod-benchmarks-jun2026` (`repo-type=dataset`)

On pod:

```bash
cd /agenticml && source venv/bin/activate
export HF_TOKEN=...   # or: huggingface-cli login

tar czf /tmp/benchmarks_done.tgz \
  results/benchmarks/bfcl \
  results/benchmarks/toolbench \
  results/benchmarks/score
ls -lh /tmp/benchmarks_done.tgz

huggingface-cli upload kosiasuzu/agenticml-runpod-benchmarks-jun2026 \
  /tmp/benchmarks_done.tgz \
  runpod-jun2026/benchmarks_done.tgz \
  --repo-type dataset \
  --commit-message "bfcl + toolbench full run"
```

On laptop:

```bash
cd ~/Developer/agenticml
huggingface-cli login   # once, if needed

hf download kosiasuzu/agenticml-runpod-benchmarks-jun2026 \
  runpod-jun2026/benchmarks_done.tgz \
  --repo-type dataset \
  --local-dir results/hf

mkdir -p results
tar xzf results/hf/runpod-jun2026/benchmarks_done.tgz -C .
# tarball paths are results/benchmarks/... — extract at repo root, not under results/benchmarks/
```

Verify full run (not stale smoke):

```bash
python -c "import json; d=json.load(open('results/benchmarks/bfcl/agenticml/summary.json')); print(len(d['tasks']), 'tasks')"
# expect 45
```

Use **`tmux`** for long `eval-run-all` jobs so SSH drops do not hide progress. Decisions log (subset trim, bridge fix, etc.): [`docs/report.md`](docs/report.md).

---

### 5. Per-Suite Smoke (Catch Errors Before Long Runs)

Use **merged** model ids. Run these in order; each step should finish in minutes (except first SWE docker pull).

#### Format Validity (Parse + Structure)

```bash
agenticml eval-benchmarks --suite format_validity --format agenticml \
  --model kosiasuzu/agenticml-llama3.1-8b-lora-merged \
  --num-examples 5 --output-dir results/benchmarks/format_validity

agenticml eval-benchmarks --suite format_validity --format chatml \
  --model kosiasuzu/chatml-llama3.1-8b-lora-merged \
  --num-examples 5 --output-dir results/benchmarks/format_validity
```

#### ToolBench (Cached Tools, No Live API)

```bash
agenticml eval-benchmarks --suite toolbench --format agenticml \
  --model kosiasuzu/agenticml-llama3.1-8b-lora-merged \
  --num-examples 1 --output-dir results/benchmarks/toolbench
```

#### BFCL (Subset)

```bash
agenticml eval-benchmarks --suite bfcl --format agenticml \
  --model kosiasuzu/agenticml-llama3.1-8b-lora-merged \
  --num-examples 3 --no-score --output-dir results/benchmarks/bfcl
```

#### SWE (Docker; Inference Only First)

```bash
# pre-pull images — can take 10–30+ min first time; see recipe.md § SWE-bench-Lite Evaluation
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

### 6. Full Benchmark Matrix + Publish

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

Outputs: `results/benchmarks/<suite>/<format>/summary.json`. Full results and figures: [`docs/report.md`](docs/report.md) (regenerate matrix: `agenticml eval-aggregate-results`; figures: `python scripts/generate_eval_figures.py`).

#### Publish Hub Model Cards

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

## Practical Notes

| Topic | Guidance |
|--------|----------|
| **Repo** | Clone `asuzukosi/agenticml` for code; Hub ids in commands point at `kosiasuzu/...` unless you re-init/train to your account |
| **pip** | Run `pip install --upgrade pip` before editable install; use staged eval install (see step 4), not `pip install -e ".[eval-benchmarks]"` alone |
| **HF auth** | `export HF_TOKEN=...` or `hf auth login` before training hub push and ToolBench download |
| **Submodules** | Required for BFCL / ToolBench / SWE eval, not for train-only |
| **ToolBench data** | `hf download nullwwg/toolbench-data ... --repo-type dataset` — **not** a model repo; extract with `python -c "import zipfile; ..."` if `unzip` is missing |
| **Disk** | ToolBench `data.zip` is ~2 GB; ensure enough volume |
| **SWE** | Needs Docker + first image pull can take 30+ min |
| **Time order** | Setup → dataset (or verify Hub) → init (or verify) → train agenticml → train chatml → eval setup → smoke evals → full matrix |

---

## Quick Reference: Default Merged Models

| Format | Merged checkpoint |
|--------|-------------------|
| agenticml | `kosiasuzu/agenticml-llama3.1-8b-lora-merged` |
| chatml | `kosiasuzu/chatml-llama3.1-8b-lora-merged` |

Init bases (training only): `kosiasuzu/agenticml-agent-llama-3.1-8b-init`, `kosiasuzu/chatml-agent-llama-3.1-8b-init`.

ChatML inference: `AutoTokenizer.from_pretrained(kosiasuzu/chatml-llama3.1-8b-lora-merged)` (instruct chat template pushed with merged weights).

---

## Format A/B: Same Task, Two Serializations

See [README — same task, two traces](README.md#same-task-two-traces-agenticml-vs-chatml) and model cards for side-by-side examples. Conversion logic: [`src/agenticml/bridge.py`](src/agenticml/bridge.py).
