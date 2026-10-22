# Vega training workflow

Experimental, and for a source checkout only: `vega train` reads and writes under `training/` at
the repository root, so run it from a clone, not a `pip install`.

## What gets trained

The Ollama model named `vega` is built from a profile in `models.toml` (a Modelfile, not
fine-tuned weights). The starter LoRA recipe (`training/config/lora.yaml`) uses the MLX-format
4-bit Qwen2.5-Coder 7B Instruct model on Apple Silicon. It writes adapters under `training/runs/adapters/vega`; it does not
overwrite the Ollama model.

## Workflow: collect → review → prepare → run

```sh
vega train collect -n 3           # attempt every practice task 3 times; keep verified passes
vega train review                 # list pending candidates
vega train review --accept FILE   # append one candidate to data/examples.jsonl
vega train review --reject FILE   # set one aside
vega train validate               # dataset format + eval/train separation
vega train prepare                # 80/10/10 splits
vega train run                    # MLX LoRA fine-tune
```

**Practice tasks.** `vega train collect` reads a pool of small practice problems,
`tasks/tasks.jsonl`, in the same schema as the eval cases (fixture `files`, `task`, and the same
checks), plus a `category` label. The task pool and its generator are not part of the public
release, so `collect` reports that the pool is missing until you supply your own file at that path.
The other steps work on examples you curate by hand (below).

**Keep training and evaluation separate.** Practice tasks must never reuse an eval case's id,
task text, fixture paths or function names — otherwise eval scores measure memorisation.
`vega train validate` (and `collect`, and `review --accept`) refuse any training task whose id or
text matches an eval case, and any dataset example whose user message contains an eval task.

**Collect.** `vega train collect [--repeat N] [--task ID-SUBSTRING]` runs Vega on each task in an
isolated temporary project with shell commands disabled, exactly as `vega eval` does, then runs the
task's checks. Only passing attempts produce a candidate, written to
`candidates/<task-id>-<n>.json` in the MLX `tools` format: system prompt, task, assistant tool
calls, tool results and final answer, plus the tool definitions that were offered. Before writing,
the trace is cleaned: the temporary project path becomes `/workspace/project`, and every harness
nudge (a user message starting with `(Vega harness)`) is removed together with the assistant turn
that provoked it, so examples teach the clean behaviour. A cleaned trace that no longer validates,
or that contains no tool calls, is not kept. Failures are never written.

**Review.** A passing check proves the outcome, not that the route was good. Read each candidate
before accepting it: look for wasted or wrong tool calls, edits outside the task, and a final
answer that claims more than the tool results show. Accept or reject one file at a time; there is
no bulk accept. Accepted files move to `candidates/accepted/`, rejected ones to
`candidates/rejected/`. `candidates/` is local working state and is ignored by Git. Your curated
examples go in `data/examples.jsonl` (not included in this repository).

## Curate examples by hand

Only add examples after reviewing and sanitizing them. Do not include secrets, private source code,
customer data, or raw conversation logs. Vega does not automatically collect chat history. Add a
text-only example with:

```sh
vega train add-example "User task" "Reviewed ideal answer"
```

For the workflows we want Vega to learn, curate a complete tool trace in a JSON file and import it
explicitly with `vega train add-trace path/to/reviewed-trace.json`. This command adds exactly the
selected trace; it never scans chat history. A trace follows MLX-LM's `tools` data format and includes
the tool schema, assistant tool call, tool result, and final answer. For example:

```json
{
  "messages": [
    {"role": "user", "content": "Rename the greeting to say hello."},
    {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "edit_file", "arguments": "{\"path\":\"src/greeting.py\",\"old_text\":\"return 'hi'\",\"new_text\":\"return 'hello'\"}"}}]},
    {"role": "tool", "tool_call_id": "call-1", "content": "Edited src/greeting.py (one exact text block replaced)."},
    {"role": "assistant", "content": "Updated the greeting in `src/greeting.py`."}
  ],
  "tools": [{"type": "function", "function": {"name": "edit_file", "description": "Replace one unique exact text block.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}}]
}
```

The validator checks message roles, tool schemas, JSON arguments, and matching tool-call results.
Keep examples representative of behavior we want: repository discovery, choosing the right tool,
targeted edits, debugging, verification, handling errors, and concise completion reports. Prefer at
least 50 diverse reviewed examples before a first experiment; tiny datasets can overfit and make the
model worse.

```sh
vega train status
vega train validate
vega train prepare
vega train run
```

`prepare` requires at least 10 valid examples and creates deterministic 80/10/10 train, validation,
and test splits. `train` runs the configured MLX LoRA recipe and records the base model, dataset
hash, timestamp, adapter location, and outcome in `training/runs.jsonl`. Review validation/test
results and run `vega eval` before deciding whether to use an adapter. Keep the base model and
previous adapters so a weaker experiment can be rolled back.

Install the optional training dependencies in the project virtual environment with:

```sh
pip install -e '.[train]'
```

The adapter is MLX format and is not automatically loaded by Ollama. Serving or converting an
adapter is a separate deployment step; keep `vega` on the known-good Ollama model until the adapter
has passed evaluations.
