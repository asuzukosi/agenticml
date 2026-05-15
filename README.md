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
 
## What's in the box
 
The Telos package separates the format SDK (what other people would
build on) from the reference runtime (one possible implementation,
used here for our own evaluations).
 
```
src/telos/
├── constants.py        SDK: frame types, owner table, token map
├── frames.py           SDK: Frame dataclass, parser, renderer, sanitize
├── tokenizer.py        SDK: tokenizer adapter for Llama-3.1
├── trajectory.py       SDK: Trajectory container with dict/frame conversion
├── validators.py       SDK: sequence-level validation
├── sdk.py              SDK: step(), the stateless trajectory-advancement API
└── runtime/            Reference runtime (one possible implementation)
    ├── tools.py        ToolRegistry, Tool, ToolError
    └── runtime.py      run(), RunResult, terminal-action handling
```
 
Anyone building a different runtime needs only the SDK modules; the
`runtime/` subpackage is not a dependency.
 
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
 
- [ ] HF generator implementation (`AutoModelForCausalLM` wrapper matching the `Generator` signature)
- [ ] Hand-authored seed trajectories
- [ ] Synthetic data generation pipeline
- [ ] LoRA fine-tune of Llama-3.1-8B-base on the Telos format
- [ ] ChatML+tools baseline fine-tune on matched data
- [ ] Evaluation harness (BFCL subset, ToolBench subset, small SWE-bench-Lite subset, retry counting on failure-injected runs)