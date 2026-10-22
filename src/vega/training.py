from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from . import evals, server
from .agent import CodingAgent

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = PROJECT_ROOT / "training"
EXAMPLES_FILE = TRAINING_DIR / "data" / "examples.jsonl"
PROCESSED_DIR = TRAINING_DIR / "data" / "processed"
RUNS_FILE = TRAINING_DIR / "runs.jsonl"
CONFIG_FILE = TRAINING_DIR / "config" / "lora.yaml"
TASKS_FILE = TRAINING_DIR / "tasks" / "tasks.jsonl"
CANDIDATES_DIR = TRAINING_DIR / "candidates"
WORKSPACE_PLACEHOLDER = "/workspace/project"
HARNESS_PREFIX = "(Vega harness)"
console = Console()
training_app = typer.Typer(help="Curate, evaluate, and fine-tune Vega.")
AgentFactory = Callable[[Path], CodingAgent]


def _read_examples() -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    if not EXAMPLES_FILE.exists():
        return examples
    for line_number, line in enumerate(EXAMPLES_FILE.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"Invalid JSON on {EXAMPLES_FILE}:{line_number}: {exc}") from exc
        _validate_example(item, line_number)
        examples.append(item)
    return examples


def _validate_example(item: Any, line_number: int) -> None:
    if not isinstance(item, dict) or not isinstance(item.get("messages"), list):
        raise typer.BadParameter(f"Example {line_number} must have a messages array.")
    messages = item["messages"]
    if len(messages) < 2 or any(not isinstance(m, dict) for m in messages):
        raise typer.BadParameter(f"Example {line_number} needs at least two message objects.")
    roles = [message.get("role") for message in messages]
    if not isinstance(roles[0], str) or roles[0] not in {"system", "user"} or "user" not in roles or roles[-1] != "assistant":
        raise typer.BadParameter(f"Example {line_number} must contain a user task and end with an assistant answer.")

    definitions = item.get("tools", [])
    if not isinstance(definitions, list):
        raise typer.BadParameter(f"Example {line_number} tools must be a list.")
    tool_names = set()
    for definition in definitions:
        function = definition.get("function") if isinstance(definition, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if not isinstance(definition, dict) or definition.get("type") != "function" or not isinstance(name, str) or not name:
            raise typer.BadParameter(f"Example {line_number} has an invalid tool definition.")
        tool_names.add(name)

    pending_calls: set[str] = set()
    seen_call_ids: set[str] = set()
    saw_tool_call = False
    for index, message in enumerate(messages, 1):
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str):
            raise typer.BadParameter(f"Example {line_number} message {index} needs a string role.")
        if role in {"system", "user"}:
            if not isinstance(content, str) or not content.strip():
                raise typer.BadParameter(f"Example {line_number} message {index} needs non-empty {role} content.")
        elif role == "assistant":
            if content is not None and not isinstance(content, str):
                raise typer.BadParameter(f"Example {line_number} assistant content must be text or null.")
            calls = message.get("tool_calls", [])
            if not isinstance(calls, list):
                raise typer.BadParameter(f"Example {line_number} assistant tool_calls must be a list.")
            if not calls and (not isinstance(content, str) or not content.strip()):
                raise typer.BadParameter(f"Example {line_number} assistant messages need text or tool calls.")
            for call in calls:
                if not isinstance(call, dict):
                    raise typer.BadParameter(f"Example {line_number} has an invalid tool call at message {index}.")
                function = call.get("function") if isinstance(call, dict) else None
                call_id = call.get("id") if isinstance(call, dict) else None
                name = function.get("name") if isinstance(function, dict) else None
                arguments = function.get("arguments") if isinstance(function, dict) else None
                if call.get("type") != "function" or not isinstance(call_id, str) or not call_id:
                    raise typer.BadParameter(f"Example {line_number} has an invalid tool call at message {index}.")
                if not isinstance(name, str) or name not in tool_names:
                    raise typer.BadParameter(f"Example {line_number} tool call {call_id} has no matching tool definition.")
                if call_id in seen_call_ids:
                    raise typer.BadParameter(f"Example {line_number} reuses tool call id {call_id!r}.")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError as exc:
                        raise typer.BadParameter(f"Example {line_number} tool call {call_id} has invalid JSON arguments: {exc}") from exc
                if not isinstance(arguments, dict):
                    raise typer.BadParameter(f"Example {line_number} tool call {call_id} arguments must be a JSON object.")
                pending_calls.add(call_id)
                seen_call_ids.add(call_id)
                saw_tool_call = True
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending_calls:
                raise typer.BadParameter(f"Example {line_number} tool reply at message {index} has no preceding tool call.")
            if not isinstance(content, str):
                raise typer.BadParameter(f"Example {line_number} tool reply at message {index} must have text content.")
            pending_calls.remove(call_id)
        else:
            raise typer.BadParameter(f"Example {line_number} has unsupported role {role!r} at message {index}.")

    if pending_calls:
        raise typer.BadParameter(f"Example {line_number} is missing tool replies for: {', '.join(sorted(pending_calls))}.")
    if saw_tool_call and not definitions:
        raise typer.BadParameter(f"Example {line_number} with tool calls must include top-level tool definitions.")


def _append_example(example: dict[str, Any]) -> int:
    _validate_example(example, len(_read_examples()) + 1)
    EXAMPLES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with EXAMPLES_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    return len(_read_examples())


def _dataset_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


# --- Practice tasks, eval separation and candidate collection --------------------------------

def _load_tasks() -> list[dict[str, Any]]:
    """Training practice tasks: the eval-case schema, kept in training/tasks/tasks.jsonl."""
    if not TASKS_FILE.exists():
        return []
    tasks = []
    for line_number, line in enumerate(TASKS_FILE.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"Invalid task at {TASKS_FILE}:{line_number}: {exc}") from exc
        if not isinstance(item.get("id"), str) or not isinstance(item.get("task"), str):
            raise typer.BadParameter(f"Training task {line_number} needs id and task strings.")
        tasks.append(item)
    return tasks


def _normalise_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _overlap_problems(
    tasks: list[dict[str, Any]], examples: list[dict[str, Any]], cases: list[dict[str, Any]],
) -> list[str]:
    """Ways the training data leaks evaluation cases; any problem makes eval scores meaningless."""
    case_ids = {case["id"]: case["id"] for case in cases}
    case_tasks = {_normalise_text(case["task"]): case["id"] for case in cases}
    problems = []
    for item in tasks:
        if item["id"] in case_ids:
            problems.append(f"training task id {item['id']!r} is also an eval case id")
        match = case_tasks.get(_normalise_text(item["task"]))
        if match:
            problems.append(
                f"training task {item['id']!r} has the same text as eval case {match!r}")
    for number, example in enumerate(examples, 1):
        for message in example.get("messages", []):
            if message.get("role") != "user" or not isinstance(message.get("content"), str):
                continue
            content = _normalise_text(message["content"])
            for case_task, case_id in case_tasks.items():
                if case_task == content or case_task in content:
                    problems.append(f"example {number} user message matches eval case {case_id!r}")
    return problems


def _eval_overlap(tasks: list[dict[str, Any]], examples: list[dict[str, Any]]) -> list[str]:
    cases = evals._load_cases() if evals.CASES_FILE.exists() else []
    return _overlap_problems(tasks, examples, cases)


class _ToolRecorder:
    """Wraps a model client to record the tool definitions actually offered to the model."""

    def __init__(self, client: Any):
        self._client = client
        self.offered: list[list[dict[str, Any]]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.offered.append(list(kwargs.get("tools") or []))
        return self._client.chat.completions.create(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def _replace_paths(value: Any, paths: list[str]) -> Any:
    if isinstance(value, str):
        for path in paths:
            value = value.replace(path, WORKSPACE_PLACEHOLDER)
        return value
    if isinstance(value, list):
        return [_replace_paths(item, paths) for item in value]
    if isinstance(value, dict):
        return {key: _replace_paths(item, paths) for key, item in value.items()}
    return value


def _clean_message(message: dict[str, Any]) -> dict[str, Any]:
    """Keep only the fields of the MLX tools format, with tool arguments as JSON strings."""
    role = message.get("role")
    content = message.get("content")
    if role == "tool":
        return {"role": "tool", "tool_call_id": message.get("tool_call_id"), "content": content}
    if role != "assistant":
        return {"role": role, "content": content}
    cleaned: dict[str, Any] = {"role": "assistant"}
    if isinstance(content, str) and content.strip():
        cleaned["content"] = content
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        calls.append({"id": call.get("id"), "type": "function",
                      "function": {"name": function.get("name"), "arguments": arguments}})
    if calls:
        cleaned["tool_calls"] = calls
    elif "content" not in cleaned:
        cleaned["content"] = content
    return cleaned


def build_trace(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]], project_paths: list[str],
) -> dict[str, Any]:
    """Turn an agent conversation into a candidate training example.

    Harness nudges and the assistant turn that provoked each one are removed, so the example shows
    the clean behaviour; temp project paths become a neutral placeholder.
    """
    kept: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        is_user = message.get("role") == "user" and isinstance(content, str)
        if is_user and content.startswith(HARNESS_PREFIX):
            if kept and kept[-1].get("role") == "assistant":
                kept.pop()
            continue
        kept.append(_clean_message(message))
    paths = sorted({path for path in project_paths if path}, key=len, reverse=True)
    return _replace_paths({"messages": kept, "tools": tools}, paths)


def _candidate_problem(example: dict[str, Any]) -> str:
    """Return why a trace cannot be a candidate, or an empty string."""
    try:
        _validate_example(example, 0)
    except typer.BadParameter as exc:
        return exc.message
    if not any(message.get("tool_calls") for message in example["messages"]):
        return "no tool calls"
    return ""


def _next_candidate_path(task_id: str) -> Path:
    places = [CANDIDATES_DIR, CANDIDATES_DIR / "accepted", CANDIDATES_DIR / "rejected"]
    number = 1
    while any((place / f"{task_id}-{number}.json").exists() for place in places):
        number += 1
    return CANDIDATES_DIR / f"{task_id}-{number}.json"


def _default_agent(project: Path) -> CodingAgent:
    return CodingAgent(project, allow_commands=False, max_steps=40)


def collect_candidates(
    repeat: int = 1, only: list[str] | None = None, agent_factory: AgentFactory | None = None,
) -> list[dict[str, Any]]:
    """Attempt each training task; write a candidate trace for every verified passing attempt."""
    tasks = _load_tasks()
    if not tasks:
        raise typer.BadParameter(
            f"No training tasks in {TASKS_FILE}. Run training/tasks/make_tasks.py.")
    problems = _eval_overlap(tasks, [])
    if problems:
        raise typer.BadParameter("Training tasks overlap the eval cases: " + "; ".join(problems))
    if only:
        tasks = [item for item in tasks if any(key in item["id"] for key in only)]
    if not tasks:
        raise typer.BadParameter("No training tasks matched.")
    factory = agent_factory or _default_agent
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in tasks:
        row = {"task": item["id"], "attempts": repeat, "passed": 0, "written": 0, "skipped": []}
        for attempt in range(repeat):
            label = f"{item['id']} attempt {attempt + 1}/{repeat}"
            with tempfile.TemporaryDirectory(prefix="vega-train-") as temp_dir:
                project = Path(temp_dir).resolve()  # macOS: /var -> /private/var
                evals._write_fixture(project, item.get("files", {}))
                agent = factory(project)
                recorder = _ToolRecorder(agent.client)
                agent.client = recorder
                response = agent.run(item["task"])
                try:
                    detail = evals._check(item, project, response)
                except subprocess.TimeoutExpired:
                    detail = f"python_check timed out after {evals.CHECK_TIMEOUT}s"
                if detail:
                    console.print(f"[dim]{label}: fail — {detail}[/dim]")
                    continue
                row["passed"] += 1
                tools = recorder.offered[-1] if recorder.offered else []
                example = build_trace(agent.messages, tools, [str(project), temp_dir])
            problem = _candidate_problem(example)
            if problem:
                row["skipped"].append(problem)
                console.print(f"[dim]{label}: pass, not kept — {problem}[/dim]")
                continue
            path = _next_candidate_path(item["id"])
            content = json.dumps(example, ensure_ascii=False, indent=2) + "\n"
            path.write_text(content, encoding="utf-8")
            row["written"] += 1
            console.print(f"[dim]{label}: pass → {path.name}[/dim]")
        rows.append(row)
    return rows


@training_app.command("collect")
def collect(
    repeat: int = typer.Option(1, "--repeat", "-n", min=1, help="Attempts per task."),
    task: list[str] = typer.Option(None, "--task", "-t", help="Only tasks whose id contains this."),
) -> None:
    """Attempt the practice tasks and save verified passing traces as candidates for review."""
    try:
        server.ensure_running()
    except RuntimeError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
    rows = collect_candidates(repeat=repeat, only=task)
    table = Table(title="Vega candidate collection")
    table.add_column("Task")
    table.add_column("Attempts", justify="right")
    table.add_column("Passed", justify="right")
    table.add_column("Candidates written", justify="right")
    table.add_column("Not kept")
    for row in rows:
        table.add_row(row["task"], str(row["attempts"]), str(row["passed"]), str(row["written"]),
                      "; ".join(row["skipped"])[:60])
    console.print(table)
    attempts = sum(row["attempts"] for row in rows)
    passed = sum(row["passed"] for row in rows)
    written = sum(row["written"] for row in rows)
    console.print(f"Passed {passed}/{attempts} attempts; wrote {written} candidates to "
                  f"{CANDIDATES_DIR}. Review them with `vega train review`.")


# --- Review ----------------------------------------------------------------------------------

def _pending_candidates() -> list[Path]:
    return sorted(CANDIDATES_DIR.glob("*.json")) if CANDIDATES_DIR.is_dir() else []


def _resolve_candidate(raw: Path) -> Path:
    for path in (raw, CANDIDATES_DIR / raw, CANDIDATES_DIR / raw.name):
        if path.is_file():
            path = path.resolve()
            if path.parent != CANDIDATES_DIR.resolve():
                raise typer.BadParameter(f"{raw} is not a pending candidate in {CANDIDATES_DIR}.")
            return path
    raise typer.BadParameter(f"No pending candidate named {raw} in {CANDIDATES_DIR}.")


def _move(path: Path, folder: str) -> Path:
    target_dir = CANDIDATES_DIR / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / path.name
    if target.exists():
        raise typer.BadParameter(f"{target} already exists; rename the candidate before moving it.")
    return Path(shutil.move(str(path), target))


def accept_candidate(raw: Path) -> tuple[int, Path]:
    path = _resolve_candidate(raw)
    try:
        example = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"{path.name} is not valid JSON: {exc}") from exc
    problem = _candidate_problem(example)
    if problem:
        raise typer.BadParameter(f"{path.name} is not a valid trace: {problem}")
    overlap = _eval_overlap([], [example])
    if overlap:
        raise typer.BadParameter(f"{path.name} would leak an eval case: {'; '.join(overlap)}")
    if example in _read_examples():
        raise typer.BadParameter(f"{path.name} is already in {EXAMPLES_FILE.name}.")
    total = _append_example(example)
    return total, _move(path, "accepted")


def reject_candidate(raw: Path) -> Path:
    return _move(_resolve_candidate(raw), "rejected")


@training_app.command("review")
def review(
    accept: Path = typer.Option(None, "--accept", help="Append this candidate to the dataset."),
    reject: Path = typer.Option(None, "--reject", help="Move this candidate to rejected/."),
) -> None:
    """List pending candidates, or accept or reject one of them (one file at a time)."""
    if accept and reject:
        raise typer.BadParameter("Use --accept or --reject, not both.")
    if accept:
        total, moved = accept_candidate(accept)
        console.print(f"[green]Accepted[/green] {moved.name}. Dataset now has {total} examples.")
        return
    if reject:
        moved = reject_candidate(reject)
        console.print(f"Rejected {moved.name}.")
        return
    pending = _pending_candidates()
    if not pending:
        console.print("No pending candidates. Run `vega train collect` to generate some.")
        return
    table = Table(title=f"Pending candidates ({len(pending)}) in {CANDIDATES_DIR}")
    table.add_column("File")
    table.add_column("Tool calls", justify="right")
    table.add_column("Tools used")
    table.add_column("Final answer")
    for path in pending:
        try:
            messages = json.loads(path.read_text(encoding="utf-8"))["messages"]
        except (json.JSONDecodeError, KeyError, TypeError):
            table.add_row(path.name, "?", "", "[red]unreadable[/red]")
            continue
        calls = [c["function"]["name"] for m in messages for c in m.get("tool_calls", [])]
        final = " ".join(str(messages[-1].get("content") or "").split())
        table.add_row(path.name, str(len(calls)), ", ".join(dict.fromkeys(calls)), final[:60])
    console.print(table)
    console.print("Read a candidate in full before deciding, then "
                  "`vega train review --accept FILE` or `--reject FILE`.")


@training_app.command("status")
def status() -> None:
    """Show model, curated data, prepared splits, and latest training run."""
    examples = _read_examples()
    table = Table(title="Vega training status")
    table.add_column("Item")
    table.add_column("Current value")
    table.add_row("Active API model", os.getenv("VEGA_MODEL", "vega"))
    tool_trace_count = sum(bool(item.get("tools")) for item in examples)
    table.add_row("Curated examples", str(len(examples)))
    table.add_row("Tool-use traces", str(tool_trace_count))
    table.add_row("Text-only examples", str(sum(not item.get("tools") for item in examples)))
    table.add_row("Prepared training set", str((PROCESSED_DIR / "train.jsonl").exists()))
    if RUNS_FILE.exists() and RUNS_FILE.stat().st_size:
        last = json.loads(RUNS_FILE.read_text(encoding="utf-8").splitlines()[-1])
        table.add_row("Latest training run", f"{last.get('timestamp')} · {last.get('status')} · {last.get('dataset_sha256', '')[:12]}")
    else:
        table.add_row("Latest training run", "none")
    console.print(table)
    if len(examples) < 50:
        console.print("[yellow]Keep collecting diverse reviewed examples; the guide recommends around 50+ before a first fine-tune.[/yellow]")
    if tool_trace_count < 50:
        console.print("[yellow]Tool-use workflows need more reviewed traces; text-only examples do not teach tool-call sequences.[/yellow]")


@training_app.command("add-example")
def add_example(
    prompt: str = typer.Argument(..., help="A sanitized user request."),
    ideal_response: str = typer.Argument(..., help="The reviewed response Vega should learn from."),
) -> None:
    """Add one explicitly curated user/assistant example to the fine-tuning dataset."""
    example = {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": ideal_response}]}
    total = _append_example(example)
    console.print(f"Added one reviewed text example. Total: {total}.")


@training_app.command("add-trace")
def add_trace(
    source: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="A reviewed JSON trace using MLX-LM's tools format."),
) -> None:
    """Add one explicitly reviewed tool-use trajectory from a JSON file; never imports chat logs automatically."""
    try:
        example = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Trace file must contain one JSON object: {exc}") from exc
    _validate_example(example, len(_read_examples()) + 1)
    if not any(message.get("tool_calls") for message in example["messages"]):
        raise typer.BadParameter("Trace has no assistant tool_calls. Use `add-example` for a text-only example.")
    total = _append_example(example)
    console.print(f"Added one reviewed tool-use trace. Total: {total}.")


@training_app.command("validate")
def validate() -> None:
    """Validate the curated dataset and its separation from the eval cases."""
    examples = _read_examples()
    tasks = _load_tasks()
    problems = _eval_overlap(tasks, examples)
    if problems:
        for problem in problems:
            console.print(f"[red]Eval overlap:[/red] {problem}")
        console.print("[red]Training data must not contain eval cases; "
                      "eval scores would be meaningless.[/red]")
        raise typer.Exit(1)
    trace_count = sum(bool(item.get("tools")) for item in examples)
    text_count = len(examples) - trace_count
    console.print(
        f"[green]Dataset valid:[/green] {len(examples)} reviewed examples "
        f"({trace_count} tool-use traces, {text_count} text-only) in {EXAMPLES_FILE.relative_to(PROJECT_ROOT)}"
    )
    console.print(f"[green]No eval overlap[/green] across {len(tasks)} training tasks "
                  "and the dataset.")


@training_app.command("prepare")
def prepare() -> None:
    """Create deterministic 80/10/10 train, validation, and test JSONL splits."""
    examples = _read_examples()
    if len(examples) < 10:
        raise typer.BadParameter(f"Need at least 10 valid examples before splitting; currently have {len(examples)}.")
    random.Random(42).shuffle(examples)
    holdout = max(1, round(len(examples) * 0.1))
    test_set = examples[:holdout]
    valid_set = examples[holdout:2 * holdout]
    train_set = examples[2 * holdout:]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train_set), ("valid", valid_set), ("test", test_set)):
        (PROCESSED_DIR / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    console.print(f"Prepared splits: train={len(train_set)}, valid={len(valid_set)}, test={len(test_set)}")


@training_app.command("run")
def train() -> None:
    """Run the configured MLX LoRA fine-tune and record its dataset hash and outcome."""
    required = [PROCESSED_DIR / f"{part}.jsonl" for part in ("train", "valid", "test")]
    if not all(path.exists() for path in required):
        raise typer.BadParameter("Prepared data is missing. Run `vega train validate` and `vega train prepare` first.")
    executable = shutil.which("mlx_lm.lora")
    if not executable:
        raise typer.BadParameter("MLX-LM is not installed in this environment. Install with `pip install -e '.[train]'`.")
    RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        "dataset_sha256": _dataset_hash(required),
        "config": str(CONFIG_FILE.relative_to(PROJECT_ROOT)),
        "status": "running",
        "adapter_path": "training/runs/adapters/vega",
    }
    with RUNS_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    console.print("Starting MLX LoRA fine-tuning. This can take time and memory on a 16 GB Mac.")
    result = subprocess.run([executable, "--config", str(CONFIG_FILE)], cwd=PROJECT_ROOT)
    record["status"] = "complete" if result.returncode == 0 else f"failed:{result.returncode}"
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    with RUNS_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    if result.returncode:
        raise typer.Exit(result.returncode)
    console.print("[green]Training completed.[/green] Review the adapter and run evaluations before using it.")
