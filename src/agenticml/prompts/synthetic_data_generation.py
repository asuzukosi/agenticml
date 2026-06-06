"""prompt for synthetic agenticml trajectory generation."""

SYNTHETIC_DATA_GENERATION_PROMPT = '''You are generating training data for an agent serialization format called AgenticML. Your job is to write complete, varied trajectories that demonstrate an LLM agent solving real tasks.

IMPORTANT: An earlier attempt produced unusable output because it generated trajectories from a small bank of fixed phrases recombined under different goal/mission strings. That output is described under "PRIOR FAILURE MODES" below and must not be repeated. Each trajectory must be reasoned out individually as if you were the agent solving the task for the first time.

## The format

A trajectory is a JSON array of frame objects. Each frame has a "type" and "content". The 11 frame types:

- "goal" (runtime, prose): persistent role/objective. Always the FIRST frame.
- "mission" (runtime, prose): the specific task instruction.
- "obs" (runtime, prose): runtime context. Tool definitions live here.
- "belief" (model, prose): the agent's updated understanding after seeing results.
- "plan" (model, prose): the agent's strategy, used when the path is non-obvious.
- "think" (model, prose): private reasoning. Use sparingly.
- "action" (model, JSON object): a tool call. Object has "tool" and tool arguments.
- "end" (model, null): generation stop. Closes a model block.
- "result" (runtime, JSON object): tool outcome. {"tool": "<name>", "value": ...}. Put error text in "value" when the call failed.
- "feedback" (runtime, prose): user follow-up. Rare in single-turn trajectories.
- "reward" (runtime, number): training-time signal. Do not emit in generated data.

## Structural rules (trajectories violating these will be rejected)

1. First frame must be "goal".
2. Tool definitions go inside an "obs" frame, formatted as TypeScript namespace declarations (see exemplars below).
3. Each model block ends with an "end" frame. A model block is the contiguous sequence between two runtime frames.
4. Every "action" frame must be followed (eventually) by exactly one "result" frame before the next action.
5. Multiple actions may be emitted in one model block; each gets its own "result" in order.
6. The trajectory must end with a "result" frame after a terminal action (tool="answer" or tool="fail"), with content {"tool": "<same tool>", "value": null}.
7. Failed tool calls still use {"tool": "<name>", "value": "error message"} - the agent should react realistically (retry, fall back, escalate, halt).
8. Each action frame must include ALL required arguments specified by its tool schema. Do not emit actions with empty content objects.
9. Belief and plan frames may only appear inside a model block (between runtime frames). Never emit more than two belief or plan frames in succession without an intervening action.

## PRIOR FAILURE MODES (these will cause your output to be rejected)

The earlier attempt produced these patterns. Do not reproduce them under any circumstances:

1. **Placeholder result values.** Result content was always `"primary_result"`, `"fallback_result"`, `"secondary_result"`, `"batched_result_1"`, etc. EVERY result value must be realistic structured data: a number, a string of actual content, a dict with plausible fields, or a list of plausible items. If the tool is `list_transactions`, the result is an array of transaction objects with dates, merchants, and amounts. If the tool is `get_weather`, the result is an object with temperature, condition, etc. Realistic data is not optional.

2. **Recycled belief phrases.** The earlier corpus reused the same ~6 sentences across thousands of trajectories: "The evidence narrows the decision space", "I should ensure the final recommendation is grounded in observed evidence", "I should prioritize user-impacting risks", etc. Belief frames must be written fresh for each trajectory and must reference specific details from the result that preceded them. If a transactions result contained 7 transactions totaling 329 GBP, the belief should say so. Generic meta-commentary about decision-making is forbidden.

3. **Belief/plan loops.** The earlier corpus emitted dozens of consecutive belief/plan frames after the final action, padding the trajectory. A trajectory must move forward: belief frames lead to actions, not to more belief frames. After an answer action, the trajectory must end with one result frame and nothing else.

4. **Tool-result mismatches.** The earlier corpus emitted error messages like "bank api timeout" in a health-information trajectory. Error messages must match the tool that was called. A health symptom lookup failing should produce something like "lookup service unavailable" or "rate limited", not a banking error.

5. **Actions with empty content.** Actions like `{"tool": "fetch"}` or `{"tool": "draft"}` with no other arguments. Every action must include the arguments the tool schema requires, populated with realistic values.

6. **Generic mission text.** Missions like "Build cleaning plan" or "Compare options for this issue". Missions must be specific, like a real user's first message: "How many lines are in main.py?", "Send 1200 GBP to my brother for the holiday deposit", "I've been getting headaches every afternoon for a week".

7. **Identical goal text across trajectories of the same domain.** The earlier corpus had every cooking trajectory start with the same goal sentence. Vary goal phrasings, even within a domain. The role description should match the task.

## Three reference trajectories (these are correct)

### Example 1: simple single-tool task

```json
[
  {"type": "goal", "content": "You are a calculator assistant."},
  {"type": "mission", "content": "What is the square root of 144?"},
  {"type": "obs", "content": "tools:\nnamespace tools {\n  type python_eval = (_: { expression: string }) => any;\n  type answer = (_: { text: string }) => any;\n}"},
  {"type": "action", "content": {"tool": "python_eval", "expression": "144 ** 0.5"}},
  {"type": "end", "content": null},
  {"type": "result", "content": {"tool": "python_eval", "value": 12.0}},
  {"type": "action", "content": {"tool": "answer", "text": "The square root of 144 is 12."}},
  {"type": "end", "content": null},
  {"type": "result", "content": {"tool": "answer", "value": null}}
]
```

### Example 2: multi-step task with a recovered failure

```json
[
  {"type": "goal", "content": "You are a personal finance assistant."},
  {"type": "mission", "content": "How much did I spend on groceries last month, and how does it compare to my usual?"},
  {"type": "obs", "content": "tools:\nnamespace tools {\n  type list_transactions = (_: { account: string, category: string, start_date: string, end_date: string }) => any;\n  type get_category_average = (_: { category: string, months: number }) => any;\n  type answer = (_: { text: string }) => any;\n}"},
  {"type": "plan", "content": "Pull last month's grocery transactions to total them, then pull a multi-month average for comparison."},
  {"type": "action", "content": {"tool": "list_transactions", "account": "primary", "category": "groceries", "start_date": "2025-02-01", "end_date": "2025-02-28"}},
  {"type": "end", "content": null},
  {"type": "result", "content": {"tool": "list_transactions", "value": "bank api timeout"}},
  {"type": "belief", "content": "Bank API timed out. Retry once before falling back."},
  {"type": "action", "content": {"tool": "list_transactions", "account": "primary", "category": "groceries", "start_date": "2025-02-01", "end_date": "2025-02-28"}},
  {"type": "end", "content": null},
  {"type": "result", "content": {"tool": "list_transactions", "value": [{"date": "2025-02-09", "merchant": "Sainsbury's", "amount": 62.15}, {"date": "2025-02-16", "merchant": "Whole Foods", "amount": 84.92}, {"date": "2025-02-22", "merchant": "Sainsbury's", "amount": 51.30}]}},
  {"type": "belief", "content": "Three transactions totaling 198.37 GBP. Compare to historical average."},
  {"type": "action", "content": {"tool": "get_category_average", "category": "groceries", "months": 6}},
  {"type": "end", "content": null},
  {"type": "result", "content": {"tool": "get_category_average", "value": {"monthly_mean": 274.50}}},
  {"type": "belief", "content": "Feb at 198 GBP is well below the 6-month average of 275 GBP, by about 76 GBP."},
  {"type": "action", "content": {"tool": "answer", "text": "You spent 198.37 GBP on groceries in February, which is about 76 GBP below your 6-month average of 275 GBP - roughly 28% under your usual."}},
  {"type": "end", "content": null},
  {"type": "result", "content": {"tool": "answer", "value": null}}
]
```

### Example 3: task with a security halt

```json
[
  {"type": "goal", "content": "You are a personal banking assistant. You always confirm before money moves and stop when something looks unusual."},
  {"type": "mission", "content": "Send 1200 GBP from my main account to my brother's account for the holiday deposit."},
  {"type": "obs", "content": "tools:\nnamespace tools {\n  type list_payees = () => any;\n  type recent_transfers = (_: { payee: string, days: number }) => any;\n  type initiate_transfer = (_: { from_account: string, to_payee_id: string, amount_gbp: number, reference: string, confirm: boolean }) => any;\n  type answer = (_: { text: string }) => any;\n}"},
  {"type": "action", "content": {"tool": "list_payees"}},
  {"type": "end", "content": null},
  {"type": "result", "content": {"tool": "list_payees", "value": [{"id": "py_001", "name": "James Whitfield", "relation": "brother"}]}},
  {"type": "action", "content": {"tool": "recent_transfers", "payee": "py_001", "days": 30}},
  {"type": "end", "content": null},
  {"type": "result", "content": {"tool": "recent_transfers", "value": [{"date": "2025-03-12", "amount_gbp": 1200, "reference": "holiday deposit"}]}},
  {"type": "belief", "content": "Already sent 1200 GBP to James 2 days ago with the same reference. This looks like a duplicate. Halt and confirm with user."},
  {"type": "action", "content": {"tool": "answer", "text": "Stopping before I send. You already sent 1200 GBP to James 2 days ago with reference 'holiday deposit' - this looks like a duplicate. Did the first transfer fail to reach him, or is this a second deposit? I have not sent anything yet."}},
  {"type": "end", "content": null},
  {"type": "result", "content": {"tool": "answer", "value": null}}
]
```

## Distribution of trajectories to generate

Generate trajectories spread across these 20 domains. Use the listed `domain` slug in the output (third column).

| Domain                       | Share | Slug          |
|------------------------------|-------|---------------|
| Coding and dev tools         | 12%   | coding        |
| Personal finance             |  8%   | finance       |
| Scheduling and calendar      |  6%   | scheduling    |
| Cooking and meal planning    |  6%   | cooking       |
| Research and fact synthesis  |  8%   | research      |
| News and current events      |  4%   | news          |
| Travel planning              |  6%   | travel        |
| Writing and communication    |  8%   | writing       |
| Tutoring and education       |  6%   | education     |
| Health information           |  4%   | health        |
| Home and household ops       |  4%   | home          |
| Smart home / IoT             |  4%   | smart_home    |
| Shopping and gifts           |  4%   | shopping      |
| Music and media curation     |  4%   | media         |
| Photo and file curation      |  3%   | photo         |
| Interpersonal advice         |  3%   | interpersonal |
| Legal and contract review    |  3%   | legal         |
| Vehicle / repair diagnosis   |  2%   | vehicle       |
| Civic / non-profit / charity |  3%   | civic         |
| Customer support             |  2%   | support       |

## Required features distribution

For each trajectory, vary on these axes:

- 40% include at least one tool failure (error in result value) the agent recovers from realistically
- 30% include a "plan" frame because the path is non-obvious
- 20% emit multiple actions in a single model block (batched actions, e.g., looking up several items in parallel before reasoning)
- 5% end with a "fail" action because the task is genuinely impossible or out of scope
- 5% end with a security/safety halt where the agent refuses to proceed without user confirmation
- 60% are 10-20 frames; 30% are 20-35 frames; 10% are 35-50 frames

Failure types to draw from: API timeout, permission denied, rate limit exceeded, device offline, 404/missing resource, malformed response, policy violation requiring escalation, duplicate-action security halt, ambiguous user request requiring clarification, out-of-policy threshold exceeded. The failure type must match the tool that produced it (a calendar tool can be rate-limited or permission-denied, but should not return a bank API error).

## Output format

Output one trajectory per line as a JSON object (JSONL). Each object has three fields:

- "id": a unique slug like "fin_groceries_feb_001". Use a domain prefix and a short descriptive suffix.
- "domain": one of the slugs from the distribution table above.
- "frames": a JSON array of frame objects as described in this prompt.

Example output line:

```
{"id": "fin_groceries_feb_001", "domain": "finance", "frames": [{"type": "goal", "content": "You are a personal finance assistant."}, {"type": "mission", "content": "..."}, ...]}
```

Output ONLY JSONL. No prose before or after, no markdown fences, no outer array. Each line is one complete trajectory object. If you cannot complete all requested trajectories in a single response, generate as many as you can fit and stop at a complete trajectory boundary (never mid-trajectory).

Generate the trajectories now.'''
