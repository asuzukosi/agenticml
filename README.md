# Telos: Agent Native Serialization Format for Language Agents

A goal-native serialization format for language agents.
 
Telos is a research project exploring an alternative to chat-completion
interfaces for autonomous agents. Where ChatML and Harmony treat every
turn as a message with a role, Telos uses typed frames: goal, mission,
belief, plan, think, action, result, feedback, reward, that are
first-class objects in the serialization itself. Each frame is
delimited by a single special token from the model's reserved
vocabulary.
 
`The hypothesis`: an interface designed for agent loops, with explicit
state and tool semantics, produces more interpretable and more
recoverable agent behavior than a chat envelope retrofitted with tool
calls.

## Installation

- **Python** 3.10 or newer.

- **Editable install (library + runtime):**

  ```bash
  pip install -e .
  ```

- **Development (pytest, Weights & Biases):**

  ```bash
  pip install -e ".[dev]"
  ```

  Or mirror the same pins with `requirements.txt` (core + dev tools in that file; see comments there for optional PyTorch / NumPy):

  ```bash
  pip install -r requirements.txt
  pip install -e .
  ```

  **Weights & Biases:** authenticate once (stores credentials for later runs):

  ```bash
  wandb login
  ```

  In a script or notebook, initialize a run before logging (adjust `project` / `entity` / `name` as needed):

  ```python
  import wandb

  wandb.init(project="your-project", entity="your-entity", name="optional-run-name")
  ```

  Use `WANDB_MODE=offline` for local-only logging without syncing to the hub.

- **CLI pipelines (train, eval, data):** install the extras you need (PyTorch is not in the core install):

  ```bash
  pip install -e ".[train]"        # LoRA training, embedding init
  pip install -e ".[eval]"         # format validity eval
  pip install -e ".[data]"         # clean/push, telos→chatml
  pip install -e ".[data-gen]"     # synthetic jsonl via openrouter
  pip install -e ".[dev,train,eval,data,data-gen]"   # full workspace
  ```

  Prefer a platform-specific PyTorch wheel from [pytorch.org](https://pytorch.org/) when installing `train` or `eval`. The legacy `llm-scripts` extra is only `torch>=2.0`; use `train` for the full training stack.

- **`apply_trajectory_template(..., return_tensors="np")`:** install NumPy (`pip install -e ".[tokenizer-numpy]"` or any `numpy>=1.24`).

- **Hugging Face:** tokenizer and model tests expect access to the gated checkpoint `meta-llama/Llama-3.1-8B`. Authenticate with `huggingface-cli login` (or env token) before running those tests.

## Commands

After `pip install -e .` (plus extras as needed), use the `telos` CLI or `python -m telos`. Example invocations are in [`command.txt`](command.txt).

| Command | Extra | Purpose |
|---------|-------|---------|
| `telos train-telos-lora` | `train` | Fine-tune Telos-format LoRA |
| `telos train-chatml-lora` | `train` | Fine-tune ChatML LoRA |
| `telos init-telos-embeddings` | `train` | Initialize Telos reserved-token embeddings |
| `telos verify-telos-embeddings` | `train` | Verify Telos init checkpoint embedding rows |
| `telos init-chatml-embeddings` | `train` | Initialize ChatML marker embeddings |
| `telos eval-format-validity` | `eval` | Generate and score format validity |
| `telos data-clean-push` | `data` | Validate, split, push trajectory dataset |
| `telos data-telos-to-chatml` | `data` | Add ChatML `messages` column to jsonl |
| `telos data-synthetic-gen` | `data-gen` | Parallel synthetic trajectory generation |

Multi-GPU training uses `torchrun` on the command module, for example:

```bash
torchrun --standalone --nproc_per_node=2 -m telos.cli.commands.train_telos_lora --help
```

List commands: `telos` or `telos --help`. Per-command help: `telos <command> --help`.

## Tests

From the repository root:

```bash
pytest
```

`pyproject.toml` sets `testpaths` and `pythonpath=["src"]`. Some tests load the Llama tokenizer or full model and **skip** if the hub is unreachable or **CUDA / torch** are missing (for example `tests/runtime/test_hf_generator.py` and the causal-LM case in `tests/test_llama_gpu_integration.py`).

## What's in the box
 
The Telos package separates the format SDK (what other people would
build on) from the reference runtime (one possible implementation,
used here for our own evaluations).
 
```
src/telos/
├── constants.py        SDK: frame types, owner table, token map
├── frames.py           SDK: Frame dataclass, parser, renderer, sanitize
├── tokenizer.py        SDK: tokenizer adapter for Llama-3.1; encode/decode, apply_trajectory_template
├── trajectory.py       SDK: Trajectory container with dict/frame conversion
├── validators.py       SDK: sequence-level validation
├── sdk.py              SDK: step(), the stateless trajectory-advancement API
├── cli/                CLI entry and command wrappers
├── training/           LoRA training pipelines
├── evaluation/         Eval harnesses
├── dataset_prep/       Dataset validate, convert, generate, push
├── model_init/         Embedding initialization for Telos / ChatML bases
├── prompts/            Prompt constants for data generation
└── runtime/            Reference runtime (one possible implementation)
    ├── tools.py        ToolRegistry, Tool, ToolError
    ├── runtime.py      run(), RunResult, terminal-action handling
    └── hf_generator.py HfGenerator: AutoModelForCausalLM continuation helper
```
 
Anyone building a different runtime needs only the SDK modules; the
`runtime/` subpackage is not required for defining or parsing Telos-only artifacts.
 
## Frame types
 
| Marker         | Owner   | Payload | Purpose                                                         |
| -------------- | ------- | ------- | --------------------------------------------------------------- |
| `<\|goal\|>`     | runtime | prose   | Persistent role / objective. Always the first frame.            |
| `<\|mission\|>`  | runtime | prose   | Specific task instruction. Latest supersedes earlier.           |
| `<\|obs\|>`      | runtime | prose   | Runtime context (tool definitions, env, files, etc.).           |
| `<\|belief\|>`   | model   | prose   | Persistent model-internal state. Updated each step.             |
| `<\|plan\|>`     | model   | prose   | Strategy. Latest supersedes earlier.                            |
| `<\|think\|>`    | model   | prose   | Private reasoning. Stripped after consumption.                  |
| `<\|action\|>`   | model   | JSON    | Tool invocation. The model may emit multiple actions per turn.  |
| `<\|end\|>`      | model   | empty   | Generation stop token. Model yields control to runtime.         |
| `<\|result\|>`   | runtime | JSON    | Tool outcome. One per action.                                   |
| `<\|feedback\|>` | runtime | prose   | Tool progress or user follow-up. Persistent.                    |
| `<\|reward\|>`   | runtime | number  | Training-time reward signal. Stripped after consumption.        |
 
Markers are mapped to `<|reserved_special_token_0|>` through
`<|reserved_special_token_10|>` in the Llama-3.1 tokenizer. Aliasing
happens at the string level; the underlying tokenizer is unmodified.
`TelosTokenizer.apply_trajectory_template` renders a `Trajectory` to
wire text and optionally returns token ids (analogous to
`PreTrainedTokenizer.apply_chat_template`).

### Telos tokenizer gotcha (inference and eval)

Telos wire strings use markers like `<|goal|>`, but **training and the model see reserved-slot token ids** (`<|reserved_special_token_0|>`, …). `TelosTokenizer` rewrites markers at encode time and restores them on decode; the underlying Hugging Face tokenizer is unchanged.

- **Do:** `ids = tt.encode(render(frames))` (or `apply_trajectory_template(..., tokenize=True)`), then `model.generate` on `ids`; decode with `tt.decode(generated_ids)` before `parse()`.
- **Do not:** `AutoTokenizer(rendered_wire)` or `tt.hf(rendered_wire)` — that BPE-tokenizes the marker *strings* and does not match training.
- **Eval:** `telos eval-format-validity --format telos` uses `TelosTokenizer` automatically. ChatML eval uses `apply_chat_template` on the instruct tokenizer as usual.

Any prior Telos format-validity numbers run with raw `AutoTokenizer` on wire text should be discarded and re-run after this fix.
 
## Example trajectory
 
```
<|goal|>You are a coding assistant.
<|mission|>How many lines are in main.py?
<|obs|>tools:
namespace tools {
  // Read a file's contents.
  type read_file = (_: {
    path: string,
  }) => any;
  type answer = (_: {
    text: string,
  }) => any;
}
<|action|>{"tool":"read_file","path":"main.py"}<|end|>
<|result|>{"ok":1,"value":"def main():\n    print('hi')\n"}
<|belief|>main.py has 2 lines.
<|action|>{"tool":"answer","text":"main.py has 2 lines."}<|end|>
```
 
## Comparison to ChatML and Harmony
 
| Property                       | ChatML                                  | Harmony (gpt-oss)                                                                                            | Telos                                                                |
| ------------------------------ | --------------------------------------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| Special tokens in the format   | 2 (`<\|im_start\|>`, `<\|im_end\|>`)        | 7 (`<\|start\|>`, `<\|message\|>`, `<\|end\|>`, `<\|channel\|>`, `<\|constrain\|>`, `<\|return\|>`, `<\|call\|>`)  | 11 (one per frame type)                                              |
| How roles are encoded          | Inline strings after `<\|im_start\|>`    | Inline strings after `<\|start\|>`                                                                            | First token of each frame is its type marker; no separate role      |
| Where tool definitions live    | Inside a `system` message               | Inside a `developer` message                                                                                 | Inside an `<\|obs\|>` frame                                           |
| How tool calls are encoded     | Markup inside an `assistant` message    | An `assistant` message routed via `to=` plus `<\|call\|>` stop token                                          | A first-class `<\|action\|>` frame with a JSON payload                |
| Where tool results return      | A new `user` (or `tool`) role message   | A new message with role set to the tool name                                                                 | A first-class `<\|result\|>` frame                                    |
| Reasoning                      | Not part of the format                  | The `analysis` channel inside an `assistant` message                                                          | A first-class `<\|think\|>` frame                                     |
| Persistent agent state         | None                                    | None                                                                                                         | A first-class `<\|belief\|>` frame                                    |
| Training-time reward channel   | None                                    | None                                                                                                         | A first-class `<\|reward\|>` frame                                    |
| Generation stop signal         | `<\|im_end\|>` ends the assistant turn   | `<\|return\|>`, `<\|call\|>`, or `<\|end\|>` depending on intent                                               | `<\|end\|>` (single stop token)                                       |
 
## Design decisions
 
- **Base model: Llama-3.1-8B-base.** Llama-3 has 250 documented reserved
  special tokens; Qwen2.5 does not. Telos claims eleven of these
  reserved slots and aliases them to frame markers at the string level:
  | Telos marker     | Reserved token                     | Token ID |
  | ---------------- | ---------------------------------- | -------- |
  | `<\|goal\|>`       | `<\|reserved_special_token_0\|>`     | 128002   |
  | `<\|mission\|>`    | `<\|reserved_special_token_1\|>`     | 128003   |
  | `<\|obs\|>`        | `<\|reserved_special_token_2\|>`     | 128005   |
  | `<\|belief\|>`     | `<\|reserved_special_token_3\|>`     | 128011   |
  | `<\|plan\|>`       | `<\|reserved_special_token_4\|>`     | 128012   |
  | `<\|think\|>`      | `<\|reserved_special_token_5\|>`     | 128013   |
  | `<\|action\|>`     | `<\|reserved_special_token_6\|>`     | 128014   |
  | `<\|end\|>`        | `<\|reserved_special_token_7\|>`     | 128015   |
  | `<\|result\|>`     | `<\|reserved_special_token_8\|>`     | 128016   |
  | `<\|feedback\|>`   | `<\|reserved_special_token_9\|>`     | 128017   |
  | `<\|reward\|>`     | `<\|reserved_special_token_10\|>`    | 128018   |
- **No closing tokens.** Each frame extends until the next marker or
  `<|end|>`.
- **`<|end|>` is a generation stop, not an envelope.** The model
  signals "your turn" by emitting any model-owned marker; no warm-up
  token needed.
- **Batched actions per turn.** A model block may emit multiple
  actions; the runtime emits a result for each, in order.
- **No action IDs in v1.** Action-result correspondence is positional.
- **TypeScript-namespace tool schemas (Harmony-style).** The model has
  seen this format in pretraining.
- **Strict ownership at the parser.** Model output cannot contain
  runtime-owned frames in strict mode.
- **Sanitization at the runtime boundary.** External text (user
  messages, tool output) is stripped of Telos markers before being
  embedded in a frame.
## Next steps
 
- [x] HF generator (`HfGenerator` in `runtime/hf_generator.py`, wraps `AutoModelForCausalLM.generate`)
- [x] CLI pipelines (`telos` commands; see Commands above and `command.txt`)
- [x] Synthetic data generation (`telos data-synthetic-gen`)
- [x] LoRA fine-tune commands (`telos train-telos-lora`, `telos train-chatml-lora`)
- [x] Format validity eval (`telos eval-format-validity`)
- [x] Hand-authored seed trajectories
- [x] Synthetic data generation pipeline
- [x] LoRA fine-tune of Llama-3.1-8B-base on the Telos format
- [x] ChatML+tools baseline fine-tune on matched data
- [ ] Evaluation harness (BFCL subset, ToolBench subset, small SWE-bench-Lite subset, retry counting on failure-injected runs)