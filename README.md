# AgenticML: Agent Native Serialization Format for Language Agents

A goal-native serialization format for language agents.
 
AgenticML is a research project exploring an alternative to chat-completion interfaces for autonomous agents. Where ChatML and Harmony treat every turn as a message with a role, AgenticML uses typed frames: goal, mission, belief, plan, think, action, result, feedback, reward, that are first-class objects in the serialization itself. Each frame is delimited by a single special token from the model's reserved vocabulary.
 
`The hypothesis`: an interface designed for agent loops, with explicit state and tool semantics, produces more interpretable and more
recoverable agent behavior than a chat envelope retrofitted with tool calls.

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
  pip install -e ".[data]"         # clean/push dataset pipeline
  pip install -e ".[data-gen]"     # synthetic jsonl via openrouter
  pip install -e ".[dev,train,eval,data,data-gen]"   # full workspace
  ```

  Prefer a platform-specific PyTorch wheel from [pytorch.org](https://pytorch.org/) when installing `train` or `eval`.

- **Hugging Face:** authenticate with `hf auth login` (or env token). Init checkpoints and the trajectory dataset live under the `kosiasuzu/agenticml-*` repos on the Hub; Llama base access (`meta-llama/Llama-3.1-8B`) is still required for embedding init.

## Hugging Face artifacts

| Repo | Role |
|------|------|
| `kosiasuzu/agenticml-agent-llama-3.1-8b-init` | AgenticML marker init base (training) |
| `kosiasuzu/agenticml-llama3.1-8b-lora-merged` | AgenticML merged model (eval + inference) |
| `kosiasuzu/chatml-agent-llama-3.1-8b-init` | ChatML baseline init base (weights + instruct tokenizer) |
| `kosiasuzu/chatml-llama3.1-8b-lora-merged` | ChatML merged eval model (weights + instruct tokenizer) |
| `kosiasuzu/agenticml-agent-trajectory-dataset` | Training + format-validity dataset |

**End-to-end walkthrough:** [`recipe.md`](recipe.md) — base Llama → init → dataset → train both formats → eval, with smoke tests at each step.

Model cards: [`model_cards/`](model_cards/). Eval setup and SWE ops: [`recipe.md`](recipe.md). Results and report: [`docs/report.md`](docs/report.md).

All CLI commands use **merged** checkpoints for inference and eval (`--model ...-lora-merged`). LoRA training pushes adapters to `<hub-repo-id>-adapter` and merged weights to `--hub-repo-id`.

**ChatML init tokenizer:** `init-embeddings --format chatml` replaces the base tokenizer with the Llama 3.1 Instruct tokenizer on the init checkpoint. Training and inference load model + tokenizer from the init or merged hub repo (`--model-id`).

## Commands

After `pip install -e .` (plus extras as needed), use the `agenticml` CLI or `python -m agenticml`. Example invocations are in [`command.txt`](command.txt).

| Command | Extra | Purpose |
|---------|-------|---------|
| `agenticml train-on-format` | `train` | Fine-tune (`--format agenticml\|chatml`, `--mode lora\|full`) |
| `agenticml init-embeddings` | `train` | Initialize marker embeddings (`--format agenticml\|chatml`) |
| `agenticml verify-embeddings` | `train` | Verify init checkpoint embedding rows (`--format agenticml\|chatml`) |
| `agenticml eval-benchmarks --suite format_validity` | `eval` | Generate and score format validity |
| `agenticml eval-benchmarks` | `eval-benchmarks` | BFCL / ToolBench / SWE subset inference + scoring (agenticml or chatml) |
| `agenticml eval-run-all` | `eval-benchmarks` | Run full benchmark matrix (suites × formats) |
| `agenticml eval-aggregate-results` | `eval-benchmarks` | Aggregate summary envelopes to `results/benchmarks/aggregate_table.md` |
| `agenticml data-clean-push` | `data` | Validate, add ChatML `messages`, split, push dataset |
| `agenticml data-synthetic-gen` | `data-gen` | Parallel synthetic trajectory generation |

Multi-GPU training uses `torchrun` on the command module, for example:

```bash
torchrun --standalone --nproc_per_node=2 -m agenticml.cli.commands.train_on_format --format agenticml --help
```

List commands: `agenticml` or `agenticml --help`. Per-command help: `agenticml <command> --help`.

## Tests

From the repository root:

```bash
pytest
```

`pyproject.toml` sets `testpaths = ["tests"]` and `pythonpath = ["src", ...]`. The test tree mirrors `src/agenticml/`:

```
tests/
├── test_frames.py, test_bridge.py, test_trajectory.py, test_validators.py, test_sdk.py, test_agentic_template.py
├── evaluation/
│   ├── benchmarks/          # bfcl/, swe/, toolbench/, format_validity/, aggregate, run_all
│   └── harness/             # runner, task, backends/
└── runtime/                 # hf_generator, runtime
```

Some tests load Hub checkpoints or a GPU and **skip** when unavailable (`tests/runtime/test_hf_generator.py`). Run a fast subset without Hub/GPU:

```bash
pytest --ignore=tests/runtime
```

## What's in the box
 
The AgenticML package separates the format SDK (what other people would
build on) from the reference runtime (one possible implementation,
used here for our own evaluations).
 
```
src/agenticml/
├── constants.py        SDK: frame types, owner table, token map
├── frames.py           SDK: Frame dataclass, marker-wire parser
├── bridge.py           SDK: AgenticML ↔ ChatML conversion (single source of truth)
├── agentic_template.py Jinja wire render, marker aliasing, hub bake, tokenization
├── trajectory.py       SDK: Trajectory container with dict/frame conversion
├── validators.py       SDK: sequence-level validation
├── sdk.py              SDK: step(), the stateless trajectory-advancement API
├── cli/                CLI entry and command wrappers
├── training/           supervised training (LoRA + full FT)
├── evaluation/         Eval harnesses
├── dataset_prep/       Dataset validate, convert, generate, push
├── model_init/         Embedding initialization for AgenticML / ChatML bases
├── prompts/            Prompt constants for data generation
└── runtime/            Reference runtime (one possible implementation)
    ├── tools.py        ToolRegistry, Tool, ToolError
    ├── runtime.py      run(), RunResult, terminal-action handling
    └── hf_generator.py HfGenerator: AutoModelForCausalLM continuation helper
```
 
Anyone building a different runtime needs only the SDK modules; the
`runtime/` subpackage is not required for defining or parsing AgenticML-only artifacts.
 
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
 
Frame markers are `<|reserved_special_token_0|>` through
`<|reserved_special_token_10|>` on Llama 3.1 (slot 7 is generation stop /
block boundary). rendering goes through the hub tokenizer:
`tokenizer.apply_chat_template(trajectory.to_dict(), ...)`.
use `reserved_to_markers()` only for logs or docs.

- **Render / tokenize:** `render_trajectory(tokenizer, trajectory)` or
  `tokenizer.apply_chat_template(trajectory.to_dict(), tokenize=True, ...)`.
- **Decode:** `raw = tokenizer.decode(generated_ids)` then
  `parse_reserved_wire(raw)` (reserved → markers → frames).
- **Hub:** `AutoTokenizer.from_pretrained(model_id)` — AgenticML checkpoints ship
  the agentic template on the tokenizer; ChatML merged checkpoints ship the
  Llama 3.1 Instruct chat template (not the base model tokenizer).
 
## Example Trajectory (Marker View)

Compact single-line form — see [Same Task, Two Traces](#same-task-two-traces-agenticml-vs-chatml) for JSON trajectories, reserved-token wire, and per-step generation.

```
<|goal|>You are a coding assistant.<|mission|>How many lines are in main.py?<|obs|>tools:
namespace tools {
  type read_file = (_: { path: string }) => any;
  type answer = (_: { text: string }) => any;
}<|action|>{"tool":"read_file","path":"main.py"}<|end|><|result|>{"tool":"read_file","value":"def main():\n    print('hi')\n"}<|belief|>main.py has 2 lines.<|action|>{"tool":"answer","text":"main.py has 2 lines."}<|end|>
```

## Same Task, Two Traces (AgenticML vs ChatML)

The project trains and evaluates **paired** models on the **same** underlying trajectories. Each dataset row stores two views of one agent run: `frames` (AgenticML) and `messages` (ChatML). [`bridge.py`](src/agenticml/bridge.py) is the single source of truth for conversion.

**Task:** Count lines in `main.py` using `read_file`, then answer.

### 1. Structured trajectory (dataset fields)

These are the **inputs** to training and eval — not the raw token string yet.

**AgenticML** — typed frames (`trajectory.to_dict()` / dataset column `frames`):

```json
[
  {"type": "goal", "content": "You are a coding assistant."},
  {"type": "obs", "content": "tools:\nnamespace tools {\n  type read_file = (_: { path: string }) => any;\n  type answer = (_: { text: string }) => any;\n}"},
  {"type": "mission", "content": "How many lines are in main.py?"},
  {"type": "action", "content": {"tool": "read_file", "path": "main.py"}},
  {"type": "end", "content": null},
  {"type": "result", "content": {"tool": "read_file", "value": "def main():\n    print('hi')\n"}},
  {"type": "action", "content": {"tool": "answer", "text": "main.py has 2 lines."}},
  {"type": "end", "content": null}
]
```

**ChatML** — role messages (`bridge.frames_to_messages(frames)` / dataset column `messages`):

```json
[
  {
    "role": "system",
    "content": "You are a coding assistant.\n\ntools:\nnamespace tools {\n  type read_file = (_: { path: string }) => any;\n  type answer = (_: { text: string }) => any;\n}"
  },
  {"role": "user", "content": "How many lines are in main.py?"},
  {
    "role": "assistant",
    "content": "",
    "tool_calls": [{
      "id": "call_2",
      "type": "function",
      "function": {"name": "read_file", "arguments": "{\"path\": \"main.py\"}"}
    }]
  },
  {
    "role": "tool",
    "tool_call_id": "call_2",
    "content": "{\"tool\": \"read_file\", \"value\": \"def main():\\n    print('hi')\\n\"}"
  },
  {"role": "assistant", "content": "main.py has 2 lines."}
]
```

### 2. Frame ↔ message mapping

| AgenticML frame | ChatML message |
|-----------------|----------------|
| `goal` + `obs` (runtime context) | One `system` message (goal prose, then tool schemas) |
| `mission` | `user` |
| `think` (optional) | Prefix text on the next `assistant` message |
| `action` + `end` (tool call) | `assistant` with `tool_calls` |
| `result` | `tool` (JSON payload) |
| `action` + `end` (`answer` terminal tool) | Plain `assistant` text |
| `feedback` | Extra `user` turn after the first mission |

Runtime-owned frames (`goal`, `obs`, `mission`, `result`, …) are injected by the agent loop; model-owned frames (`think`, `action`, `end`, …) are generated.

### 3. Prompt wire (tokenizer input)

Both formats flatten the structured trajectory into one token stream. **Delimiters differ:** AgenticML uses one reserved token per frame type; ChatML uses role header / end-of-turn tokens.

#### AgenticML — marker view (logs, `parse()`)

Each payload runs until the **next frame token** (no closing tag per frame):

```
<|goal|>You are a coding assistant.
<|obs|>tools:
namespace tools {
  type read_file = (_: { path: string }) => any;
  type answer = (_: { text: string }) => any;
}
<|mission|>How many lines are in main.py?
<|action|>{"tool":"read_file","path":"main.py"}<|end|>
<|result|>{"tool":"read_file","value":"def main():\n    print('hi')\n"}
<|action|>{"tool":"answer","text":"main.py has 2 lines."}<|end|>
```

#### AgenticML — model wire (what Llama actually sees)

`render_trajectory()` / `apply_chat_template` on `frames` emits **reserved slots** (see [Design decisions](#design-decisions) for the full map). Same text, frame boundaries shown with `\n` for readability:

```
<|reserved_special_token_0|>You are a coding assistant.
<|reserved_special_token_2|>tools:
namespace tools {
  type read_file = (_: { path: string }) => any;
  type answer = (_: { text: string }) => any;
}
<|reserved_special_token_1|>How many lines are in main.py?
<|reserved_special_token_6|>{"tool":"read_file","path":"main.py"}<|reserved_special_token_7|>
<|reserved_special_token_8|>{"tool":"read_file","value":"def main():\n    print('hi')\n"}
<|reserved_special_token_6|>{"tool":"answer","text":"main.py has 2 lines."}<|reserved_special_token_7|>
```

Use `reserved_to_markers(wire)` when reading model output in docs or debug logs.

#### ChatML — prompt wire (history + context)

Llama 3.1 Instruct `apply_chat_template` on `messages` (date preamble omitted):

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a coding assistant.

tools:
namespace tools {
  type read_file = (_: { path: string }) => any;
  type answer = (_: { text: string }) => any;
}<|eot_id|><|start_header_id|>user<|end_header_id|>

How many lines are in main.py?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

<|python_tag|>{"name": "read_file", "parameters": "{\"path\": \"main.py\"}"}<|eom_id|><|eot_id|><|start_header_id|>ipython<|end_header_id|>

{"tool": "read_file", "value": "def main():\n    print('hi')\n"}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

main.py has 2 lines.<|eot_id|>
```

Role headers (`<|start_header_id|>…<|end_header_id|>`) and turn ends (`<|eot_id|>`, `<|eom_id|>`) play the same structural role as AgenticML frame tokens: they **segment** the stream into speaker spans instead of typed frames.

### 4. Model output (one generation step)

After the prompt ends at the last context frame / message, the model **generates** the next model-owned span.

**Step A — first tool call**

| | Generated text |
|---|----------------|
| **AgenticML** | `<\|reserved_special_token_6\|>{"tool":"read_file","path":"main.py"}<\|reserved_special_token_7\|>` |
| **AgenticML** (markers) | `<\|action\|>{"tool":"read_file","path":"main.py"}<\|end\|>` |
| **ChatML** | `<\|python_tag\|>{"name": "read_file", "arguments": "{\"path\": \"main.py\"}"}<\|eom_id\|>` |

The runtime parses that span, executes `read_file`, then **appends** a runtime frame / message (not generated by the model):

| | Runtime injection |
|---|-------------------|
| **AgenticML** | `<\|reserved_special_token_8\|>{"tool":"read_file","value":"def main():\\n    print('hi')\\n"}` |
| **ChatML** | `<\|start_header_id\|>ipython<\|end_header_id\|>\n\n{"tool": "read_file", "value": "..."}<\|eot_id\|>` |

**Step B — final answer**

| | Generated text |
|---|----------------|
| **AgenticML** | `<\|action\|>{"tool":"answer","text":"main.py has 2 lines."}<\|end\|>` |
| **ChatML** | `main.py has 2 lines.<\|eot_id\|>` (plain assistant text; terminal `answer` tool collapsed to text in ChatML) |

### Takeaway

AgenticML names the **kind of turn** (`action`, `result`, `end`, …) with dedicated tokens; ChatML names the **speaker** (`system`, `user`, `assistant`, `ipython`) and embeds tool JSON inside those spans. The benchmark harness converts between representations so both formats face the same tasks where possible. Full pipeline: [`recipe.md`](recipe.md).

## Comparison to ChatML and Harmony
 
| Property                       | ChatML                                  | Harmony (gpt-oss)                                                                                            | AgenticML                                                                |
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
  special tokens; Qwen2.5 does not. AgenticML claims eleven of these
  reserved slots and aliases them to frame markers at the string level:
  | AgenticML marker     | Reserved token                     | Token ID |
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
- **`<|end|>` is a stored frame and generation stop.** Each model turn
  ends with an explicit `end` frame in the trajectory; the model emits
  `<|end|>` to yield control to the runtime.
- **Batched actions per turn.** A model block may emit multiple
  actions; the runtime emits a result for each, in order.
- **No action IDs in v1.** Action-result correspondence is positional.
- **TypeScript-namespace tool schemas (Harmony-style).** The model has
  seen this format in pretraining.
- **Strict ownership at the parser.** Model output cannot contain
  runtime-owned frames in strict mode.
- **Tool schemas are caller-owned.** Use `with_tool_obs(trajectory, tools)`
  once before `step()`; `step()` tokenizes the trajectory as given.
