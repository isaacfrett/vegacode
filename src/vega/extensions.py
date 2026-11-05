"""Optional extensions: a small, fixed set of hook points with plain defaults.

Vega works on its own; every function here does nothing special unless an extension package is
importable. When it is (an editable install that also has it on the path), its module-level
functions of the same names replace the defaults below. The package is never required: if it is
missing, or importing it fails for any reason, Vega behaves exactly as without it.
VEGA_NO_EXTENSIONS=1 ignores it even when present.

Hook points (each one is called from one or two places in core, named in its docstring):

    before_start(command, *, model_given) -> int   cli: `vega`, chat, run, auto start
    cli_commands(app) -> None                      cli: once, after the core commands
    turn_started(agent, task, **context) -> None   agent loop: before its first write
    system_context(agent, task, **context) -> str  agent loop: once a turn, before its requests
    checks_recorded(source, **context) -> None     a full check run finished
    final_answer(answer, *, source, **context) -> str   the answer a path is about to return
    writer_context(project, target, task, *, source, **context) -> str   a writer prompt's extra
    unchanged_offer(project, target, task, *, source, **context) -> str
                                                   project pipeline: a writer prompt's last line
    unchanged_reply(reply, target, *, source, **context) -> bool
                                                   project pipeline: a reply to that offer
    planner_request(schema, system, *, source, **context) -> (schema, system)
                                                   project pipeline: the planner's JSON request
    stalled_round(source, **context) -> dict | None   write_loop.run_loop: a repair round that
                                                   left every file unchanged, what to do instead
    side_model(job, *, agent, **context) -> str | None   planner, clarify, acceptance, task_checks
                                                   (and lab side calls): the model for a side job
    render_markdown(text) -> str                   ui, tui: an answer about to be shown
    attempt_started(agent, *, suite, case) -> None evals: an eval attempt's agent, before it runs
    attempt_finished(attempt, *, agent, suite, case) -> None   evals: its record, before it is kept
    attempt_wait(*, sleep, first, log) -> dict | list | None
                                                   evals: before every attempt, idle time it spent
    acted_tracked() -> list[str]                   evals: the switches whose acting each attempt's
                                                   `acted` record covers (features.mark_acted)
    switch_names() -> list[str]                    features.disabled: switch names the extension
                                                   owns that VEGA_FEATURES_OFF may name
    clarify_verdict(task, verdict, **context) -> Verdict   clarify: the gate's rule verdict
    clarify_answer(agent, task, **context) -> str | None   clarify.answered: the reply to Vega's
                                                   questions, the task to run instead
    keep_answer(answer, *, reason, **context) -> bool      agent loop: before a reply is sent back
    present_questions(ui, questions, **context) -> str | None   cli chat, tui: questions asked
    run_case(case, **context) -> dict | None       evals: an attempt, before the harness runs it
    write_findings(tool, relative, before, after, *, project) -> (refusal, note)
                                                   edits.check: a computed write that passed it
    repair_write(tool, relative, before, after, *, project, problem) -> dict
                                                   edits.check: a write that would not parse
    resolve_undefined(tool, relative, before, after, *, project, names, offers) -> dict
                                                   edits.check: a write refused only for
                                                   undefined names, the text to write instead
    write_base(tool, relative, *, project, definition) -> str | None
                                                   edits.plan_definition: the text it edits
    tool_refused(tool, args, refusal, *, project) -> dict
                                                   tools.plan_write, tools.execute: a refused call
    outgoing_write(tool, relative, after, *, project) -> dict
                                                   tools.plan_write, remember: a computed write's
                                                   text, the text to save instead
    tool_guard(agent, name, args, **context) -> str | None
                                                   agent loop: a tool call, a refusal instead
    chat_opened(ui, *, project, **context) -> bool tui: started, before the first message
    chat_message(ui, text, **context) -> str | None   tui: a typed message, before its turn
    warm() -> None                                 tui: background start-up, imports ahead
    write_starting(project, *, tool, **context) -> None
                                                   tools.execute: a write or a non-read-only
                                                   shell command, just before it runs
    agent_event(agent, kind, data) -> None         agent: each progress event it emits
    answer_context(agent, task, **context) -> str  question pipeline, agent loop: extra context
    question_excerpts(project, task, *, index, shown, remaining, **context) -> list[str]
                                                   question pipeline: excerpt blocks added after
                                                   the named files and search hits
    memory_context(agent, task, *, step, **context) -> str   agent loop: text kept outside the
                                                   history, added to each request
    budget(name, default, *, num_ctx) -> int       every prompt-size budget use (pipelines, routing)
    finish_problems(agent, task, **context) -> list   agent loop: a would-be final answer, what it owes
    lost_reply(agent, task, **context) -> str | None   agent loop: a reply with tokens but no text
                                                   and no call, the note to send back
    route_task(task, *, project, **context) -> Route | None   routing.route_task: before its rules
    read_only_task(task, **context) -> bool        agent.asks_for_change, FinishGate.for_task: the
                                                   task forbids changing any file
    compact_history(messages, goal_index, num_ctx, log, **context) -> Compaction | None
                                                   agent._fit_context: instead of context.compact
    compact_share(share, *, num_ctx, reply_tokens, projected) -> float
                                                   context.compact_at: the share of the window at
                                                   which the history is compacted
    code_snippet(lines, picked, start, end, *, default) -> str | None
                                                   index.CodeIndex._hit: a search hit's snippet
    routed_answer(agent, task, answer, *, before, route) -> None
                                                   agent: a pipeline's answer to a routed turn
    question_sentence(sentence, *, before, after) -> bool   verbs: a question sentence's change
                                                   verb is asked about, not for
    change_request(task) -> bool                   question_pipeline.is_question: a question-shaped
                                                   request that still asks for a change
    page_measured(page, *, html) -> dict           web_pipeline: a rendered page, the extension's
                                                   own measurements kept with the snapshot
    page_failure(check, failure, *, before, gone) -> str   web_pipeline: a failing page check's
                                                   line (lost ids), the line to use instead
    example_script(script) -> str                  pipeline.run_examples: the examples' check script
    examples_failed(failures, *, path) -> None     pipeline.run_examples: its failure lines
    model_profiles() -> dict                       models.load: profiles added to models.toml's
    catalog_models() -> dict                       model_store.load_catalog: entries added to the
                                                   shipped catalogue
    tool_list(tools, task, **context) -> list      agent loop (and its warm-up, the TUI gauge): the
                                                   tool definitions a request offers
    system_prompt(prompt) -> str                   agent.system_prompt: the agent loop's rules
    allow_command(command, project, **context) -> bool   agent loop: a run_command in a session
                                                   whose commands are disabled, run anyway
    deliverable_prompt(project, task, wanted, *, source, **context) -> str
                                                   deliverable pipeline: text after the inputs
    deliverable_runner(project, task, *, source, **context) -> dict
                                                   deliverable pipeline: extra script-runner options
    deliverable_direct(task, *, source, **context) -> str   deliverable pipeline: write directly
    deliverable_repair(feedback, *, source, **context) -> str   deliverable pipeline: a script
                                                   error's feedback
    deliverable_files(files, *, source, **context) -> dict   deliverable pipeline: a round's
                                                   deliverables before they are written
    check_call(agent, command, **context) -> (command, result) | None
                                                   agent loop: a model's run_check, before it runs
    suite_rerun(spec, project, **context) -> str | None   acceptance: instead of a whole suite
    targeted_first(spec, project, *, changed, stop, item) -> dict | None
                                                   acceptance: a large suite's first targeted runs
    timeout_hint(hint, *, command, project, timeout) -> str   targeted_tests.run_check_hint: the
                                                   note after a large suite's run_check timeout
    check_result(agent, command, result, *, changed, read) -> str   agent loop: a run_check result
                                                   the core ran, the text to use instead
    baseline_gate(agent, **context) -> bool        agent loop: baseline_aware_gate on for it
    test_baseline(agent, command, *, existing, **context) -> Baseline | None
                                                   agent loop: the baseline at the first edit
    round_score(source, *, checks, default, **context) -> tuple
                                                   write_loop.run_loop: a round's rank
    finish_check(agent, task, **context) -> dict | None   agent loop: a would-be final answer
                                                   after a write, one check run and its note
    tool_shown(agent, name, args, result, **context) -> str | None
                                                   agent loop: a tool result, the text the model
                                                   sees instead (before condensing and the cap)
    compaction_pins(agent, **context) -> str       agent._fit_context: text kept across a
                                                   compaction, after pin_editing_target's
    larger_window(agent, *, step, **context) -> dict | None
                                                   agent._fit_context: each loop step, a larger
                                                   window to switch to (CodingAgent.grow_window)
    check_workspace(root) -> (clone, temp) | None  sandbox.run_check: where a check's clone and its
                                                   temp folder go inside its workspace
    check_python() -> Path | None                  sandbox, evals: an interpreter that stands in for
                                                   Vega's own in checks and project commands
    case_environment(case, *, suite) -> context manager   evals: around an attempt (and the
                                                   fixture guard), its record's extra fields
    command_run(argv, *, cwd, env, timeout, stop, shell) -> CompletedProcess | None
                                                   sandbox.run_stoppable: every command run, run
                                                   (and watched) by the extension instead

Hooks receive keyword context they may ignore; new keys can be added without breaking them.
"""

from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType
from typing import Any

PACKAGE = "vega_lab"
DISABLE_ENV = "VEGA_NO_EXTENSIONS"

_UNSET: Any = object()
_module: Any = _UNSET
load_error = ""  # why the extension package did not load ("" when it loaded or is absent)


def load() -> ModuleType | None:
    """The extension package, or None (absent, disabled or broken). Imported once."""
    global _module, load_error
    if _module is _UNSET:
        _module, load_error = None, ""
        if not os.environ.get(DISABLE_ENV):
            try:
                _module = importlib.import_module(PACKAGE)
            except ModuleNotFoundError as exc:
                if exc.name != PACKAGE:  # present, but something it imports is missing
                    load_error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:  # noqa: BLE001 - a broken extension never stops Vega
                load_error = f"{type(exc).__name__}: {exc}"
    return _module


def reset() -> None:
    """Forget the loaded package (tests; the next hook call imports again)."""
    global _module, load_error
    _module, load_error = _UNSET, ""


def active() -> bool:
    return load() is not None


def _impl(name: str) -> Any:
    module = load()
    return getattr(module, name, None) if module is not None else None


def _quiet(name: str, fallback: Any, *args: Any, warn: bool = False, **kwargs: Any) -> Any:
    """Call hook `name`; any error is logged (and with `warn` printed to stderr) and the default
    result is used instead: an extension never fails a turn or the CLI."""
    fn = _impl(name)
    if fn is None:
        return fallback
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - see the docstring
        from . import debuglog

        debuglog.record("stage", stage=f"extensions.{name}.error", error=str(exc)[:300])
        if warn:
            print(f"warning: extension hook {name} failed ({type(exc).__name__}: {exc}); "
                  "going on without it", file=sys.stderr)
        return fallback


def before_start(command: str | None, *, model_given: bool) -> int:
    """Called when `vega` (command None), `vega chat`, `vega run` or `vega auto` starts, before
    the model is selected. Returns 0 to go on, else the exit code to stop with (after saying
    why). Default: 0."""
    code = _quiet("before_start", 0, command, model_given=model_given, warn=True)
    return code if isinstance(code, int) else 0


def cli_commands(app: Any) -> None:
    """Called once when the CLI module loads, after the core commands are registered, with the
    Typer app (to add commands). Default: nothing."""
    _quiet("cli_commands", None, app, warn=True)


def turn_started(agent: Any, task: str, **context: Any) -> None:
    """Called by the agent loop for a task that may write files, before its first write, with
    context `spec` (the turn's acceptance.Spec, or None) and `stop` (the turn's stop Event).
    Default: nothing."""
    _quiet("turn_started", None, agent, task, **context)


def system_context(agent: Any, task: str, **context: Any) -> str:
    """Called by the agent loop once a turn, before its first model request, with context
    `conversation` (question_pipeline.is_conversation(task)) and `route` (the turn's route record,
    or None). Returns text the loop adds to each of this turn's requests as a trailing message;
    it is never stored in the history, so the cached prompt prefix stays the same. Default: ""
    (nothing added)."""
    result = _quiet("system_context", "", agent, task, **context)
    return result if isinstance(result, str) else ""


def checks_recorded(source: str, **context: Any) -> None:
    """Called after a full check run: `source` is the pipeline's name (write_loop's
    LoopConfig.name) with context `checks` (a write_loop.Checks), `round`, `project`, `task` (the
    request; "" when the pipeline does not set it), `last` (no repair round follows), `ask` (the
    host's model request, for a hook's own one-off request), `originals` (the loop's targets
    before the loop; None: new) and `request` (the request, also where `task` is "": the web
    pipeline, for evidence providers), or
    "acceptance" (the agent loop's full acceptance run) with `agent`, `result` and `changed`.

    A hook may add a failing check of its own before the loop decides whether to repair: a
    pipeline's with `checks.add_failure(source, label, line)` (counted as failing, the line goes
    to the writer as repair feedback), the agent loop's by appending a failing acceptance.Item to
    `result.items` (the finish gate sends it back like its own failures).

    A pipeline that records its checks one by one fills `checks.items` (else it is None): one
    plain dict per check (write_loop.item), {"source", "label", "ok", "detail"} and optional
    keys. `source` names the checker: "acceptance" (an acceptance.Item: also `kind`, `output`,
    `note`), "tests" (a run of the project's tests, `ran` False when none ran; with `before`,
    one test that failed before the change too), "command" (a command the task names; with
    `tests`, the project's test command), "examples" (a docstring example), "format", "write"
    (a planned write not saved), "load" (a module load), "rendered" (a rendered-page check of
    a task probe), "page" (a rendered-page check made on every page: script errors, CSS
    accepted, ids kept), "names" (a name the task gives), "imports" (an unresolved import), "written tests" (tests
    Vega wrote from the task) or "task checks" (a task_checks rule: also `kind`, `dropped`).
    Default: nothing."""
    _quiet("checks_recorded", None, source, **context)


def final_answer(answer: str, *, source: str, **context: Any) -> str:
    """Called with the answer a path is about to return; returns the answer to use. `source`:
    the pipeline's name (context `result`, `config`, `changed`; the direct and project pipelines
    also `project` and `task`; write_loop.answer_for also
    `left`, target -> why it was not written, already said at the end of `answer`, and `what`,
    "code" or "content", the answer's word for what was written; the project
    pipeline also `written`, the files whose content differs from before, and `notes`, the
    lines it adds after its outcome; every code pipeline also `agent`, the turn's agent), or "agent" (context `agent`, `task`, `spec`, `written`
    (the files the turn wrote), `stop`, and `baseline`: the tests failing before the first
    edit, or None). Default: the answer unchanged."""
    result = _quiet("final_answer", answer, answer, source=source, **context)
    return result if isinstance(result, str) else answer


def writer_context(project: Any, target: str, task: str, *, source: str, **context: Any) -> str:
    """Called by the direct and project pipelines while building the writer's prompt for one file
    (`target`, posix, relative to `project`), with context `prompt` (the prompt so far), `budget`
    (the characters it may still take) and, from the project pipeline, `plan` (its items,
    {"path", "change"}). Returns text appended to that prompt. Default: "" (nothing added)."""
    result = _quiet("writer_context", "", project, target, task, source=source, **context)
    return result if isinstance(result, str) else ""


def render_markdown(text: str) -> str:
    """Called by the terminal and TUI renderers with an answer (or the part added to a streamed
    one) about to be shown as Markdown; returns the text to render. Default: unchanged."""
    result = _quiet("render_markdown", text, text)
    return result if isinstance(result, str) else text


def attempt_started(agent: Any, *, suite: str, case: dict[str, Any]) -> None:
    """Called by the eval runner (evals._run_attempt) for each attempt of Vega itself, with the
    attempt's agent once it is built and metered (its `client` already wrapped), just before it
    runs the case's task. The attempt's debug log (VEGA_DEBUG) is already started. Default:
    nothing."""
    _quiet("attempt_started", None, agent, suite=suite, case=case)


def attempt_finished(attempt: dict[str, Any], *, agent: Any, suite: str,
                     case: dict[str, Any]) -> None:
    """Called by the eval runner with an attempt's record (a dict, graded) before it is kept; may
    add keys to it. Default: nothing."""
    _quiet("attempt_finished", None, attempt, agent=agent, suite=suite, case=case)


def attempt_wait(*, sleep: Any, first: bool, log: Any, **context: Any) -> list[dict[str, Any]]:
    """Called by the eval runner's pacer (evals._Pacer) before every attempt, after the cooldown
    and the owner's pause, with `sleep` (the runner's: tests pass a fake), `first` (no attempt ran
    yet in this run) and `log` (one line to the console). The extension may idle (with `sleep`)
    and returns {"kind": name, "seconds": idled} for each wait, or a list of them, or None
    (default). It never stops the run. Each kind is recorded in the run's `resources` as
    `<kind>_waits` and `<kind>_wait_seconds`; attempt seconds never include it. Returns the
    well-formed waits (a list, maybe empty)."""
    result = _quiet("attempt_wait", None, sleep=sleep, first=first, log=log, **context)
    rows = result if isinstance(result, (list, tuple)) else [result]
    out = []
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("kind"), str) and row["kind"].isidentifier() \
                and isinstance(row.get("seconds"), (int, float)) and row["seconds"] >= 0:
            out.append({**row, "seconds": float(row["seconds"])})
    return out


def acted_tracked() -> list[str]:
    """The extension's switches whose acting is tracked in this process (each attempt's `acted`
    record, evals._with_acted; they mark it through features.mark_acted). Default: none."""
    result = _quiet("acted_tracked", [])
    return [str(name) for name in result] if isinstance(result, (list, tuple)) else []


def switch_names() -> list[str]:
    """Called by features.disabled when VEGA_FEATURES_OFF names a switch the core does not have:
    the names the extension owns there (switches it took over, which it then reads from
    VEGA_FEATURES_OFF itself). Default: none, so every such name is an unknown-switch error."""
    result = _quiet("switch_names", [])
    return [str(name) for name in result] if isinstance(result, (list, tuple)) else []


def clarify_verdict(task: str, verdict: Any, **context: Any) -> Any:
    """Called by the clarify gate (clarify.run) with the request and its rule verdict (a
    clarify.Verdict), context `agent` and `follow_up`; returns the verdict to use. A returned
    verdict may carry `questions` to ask. Default: the verdict unchanged (also when the hook
    returns something that is not a verdict)."""
    result = _quiet("clarify_verdict", verdict, task, verdict, **context)
    kind = getattr(result, "kind", None)
    return result if kind in ("clear", "vague", "unsure") else verdict


def clarify_answer(agent: Any, task: str, **context: Any) -> str | None:
    """Called by clarify.answered on the turn right after Vega asked its clarifying questions,
    with the user's reply (`task`) and context `pending` (the request the questions were about)
    and `defaults` (the recommended options as text, "" when none). The agent's pending state is
    already cleared; a hook that wants the questions to stay open sets it again. Returns the task
    the turn runs instead (e.g. the request with the recommended options for "yes, use the
    defaults"), or None (default): the core's own reading (the request with the reply as the
    answers)."""
    result = _quiet("clarify_answer", None, agent, task, **context)
    return result if isinstance(result, str) and result.strip() else None


def keep_answer(answer: str, *, reason: str, **context: Any) -> bool:
    """Called by the agent loop before one of its end-of-turn checks sends a would-be final
    answer back to the model; `reason` names the check ("asked the user for project content",
    "the task is not done yet", "plan written as text": a plan the loop would record and send
    back to act on), context `agent`, `task` and `wrote` (a file was written this
    turn). True keeps the answer as the turn's final answer. Default: False (sent back)."""
    return _quiet("keep_answer", False, answer, reason=reason, **context) is True


def present_questions(ui: Any, questions: list[Any], **context: Any) -> str | None:
    """Called by the line chat (`ui`: its rich Console) and the TUI (`ui`: the App) when a turn's
    answer asks the user questions (agent.questions_asked: (question, options) items), with
    context `text` (the answer as written), `agent` and `send` (a callable taking the reply to
    send as the user's next message, for a UI that answers later). Returns the reply to send now,
    "" when the hook showed the questions and has nothing to send now, or None (default): the UI
    shows the answer and the user types a reply as usual."""
    result = _quiet("present_questions", None, ui, questions, **context)
    return result if isinstance(result, str) else None


def run_case(case: dict[str, Any], **context: Any) -> dict[str, Any] | None:
    """Called by the eval harness (evals.run_attempt) for every attempt, inside the attempt's
    isolated knowledge library, with context `suite`, `agent_factory`, `archive` and
    `transcript`. Returns the attempt record to use, or None to let the harness run the case
    itself. Default: None."""
    result = _quiet("run_case", None, case, **context)
    return result if isinstance(result, dict) else None


def write_findings(tool: str, relative: str, before: str | None, after: str, *, project: Any,
                   **context: Any) -> tuple[str, str]:
    """Called by edits.check (the atomic write check of the agent loop's write tools and of the
    pipelines' saves) for a computed write that parses and adds no static findings: `tool` (the
    write tool, or "pipeline"), the project-relative path, the file's text before (None: a new
    file) and after, with context `pending` (files the same pipeline may still write, possibly
    empty). Returns (refusal, note): a non-empty refusal refuses the write, its text said after
    "Write not applied: " and before the retry hint; a note is added to an allowed write's
    result. Default: ("", "")."""
    result = _quiet("write_findings", ("", ""), tool, relative, before, after, project=project,
                    **context)
    if (isinstance(result, tuple) and len(result) == 2
            and all(isinstance(part, str) for part in result)):
        return result
    return "", ""


def repair_write(tool: str, relative: str, before: str | None, after: str, *, project: Any,
                 problem: Any, **context: Any) -> dict[str, Any]:
    """Called by edits.check, for a caller that can write a repaired text (the agent loop's write
    tools, write_loop.save), when a computed write adds a parse error: `problem` (edits.Problem:
    message, line, column, kind), with context `region` (the 1-based lines the write changed),
    `refusal` (the refusal text about to be returned) and `args` (the tool's arguments).
    Returns {"after": text, "note": text} to write a repaired text instead (edits.check uses it
    only when it parses, then checks it like any write and adds the note to the result), or
    {"refusal": text} to refuse with that text instead; {} for the plain refusal. Default: {}."""
    result = _quiet("repair_write", {}, tool, relative, before, after, project=project,
                    problem=problem, **context)
    return result if isinstance(result, dict) else {}


def resolve_undefined(tool: str, relative: str, before: str | None, after: str, *, project: Any,
                      names: list[str], offers: dict[str, tuple[str, str]],
                      **context: Any) -> dict[str, Any]:
    """Called by edits.check, for a caller that can write a repaired text (the agent loop's write
    tools, write_loop.save), when a computed write parses and its only new findings are undefined
    names (F821) none of which the layer rules explain: `names` (in order), `offers` (import_offer:
    name -> (the project file defining it, the import line)), with context `args`. Returns
    {"after": text, "note": text} to write that text instead (edits.check checks it like any
    write and uses it only when it passes, adding the note to the result), or {} (default) for
    the plain refusal."""
    result = _quiet("resolve_undefined", {}, tool, relative, before, after, project=project,
                    names=names, offers=offers, **context)
    return result if isinstance(result, dict) else {}


def write_base(tool: str, relative: str, *, project: Any, **context: Any) -> str | None:
    """Called by edits.plan_definition (replace_definition) before it reads the file it edits,
    with context `definition` (its name) and `disk` (the file's text, None when it does not
    exist). Returns the text to edit instead (a draft the extension kept from a refused write;
    the result is written, and checked, as a whole), or None for the file on disk. Default:
    None."""
    result = _quiet("write_base", None, tool, relative, project=project, **context)
    return result if isinstance(result, str) else None


def tool_refused(tool: str, args: dict[str, Any], refusal: str, *, project: Any,
                 **context: Any) -> dict[str, Any]:
    """Called with a tool call the tool refused before any check, and the text it would return:
    by tools.plan_write when a write tool's call is refused while it is computed (old_text not
    found or ambiguous, a line range outside the file, replace_definition on a file it cannot
    edit, a name its code does not define, or arguments it cannot use: then `refusal` is the
    "Tool error: <exception>" text tools.execute would return), and by tools.execute for an
    unknown tool name or another tool that raised on its arguments. Context `atomic` (plan_write's
    flag). Returns {"plan": tools.WritePlan} (write tools from plan_write only: used as the
    computed write, then checked like any other write), {"refusal": text} to return that text
    instead, or {} (default) for the plain refusal."""
    result = _quiet("tool_refused", {}, tool, args, refusal, project=project, **context)
    return result if isinstance(result, dict) else {}


def outgoing_write(tool: str, relative: str, after: str, *, project: Any,
                   **context: Any) -> dict[str, Any]:
    """Called by tools.plan_write (every write tool, after its checks) and the `remember` tool
    with the computed write's project-relative path and whole new text. Returns {"after": text,
    "note": text} to save that text instead (the note is added to the tool's result), or {}
    (default) to save it as computed."""
    result = _quiet("outgoing_write", {}, tool, relative, after, project=project, **context)
    return result if isinstance(result, dict) else {}


def tool_guard(agent: Any, name: str, args: dict[str, Any], **context: Any) -> str | None:
    """Called by the agent loop with a tool call it is about to run (context `task`). Returns a
    refusal text to give the model instead of running the call, or None (default) to run it."""
    result = _quiet("tool_guard", None, agent, name, args, **context)
    return result if isinstance(result, str) and result.strip() else None


def chat_opened(ui: Any, *, project: Any, **context: Any) -> bool:
    """Called by the TUI once, when it has started (before the first message), with context
    `agent`, `sessions` (the folder of chat transcripts) and callables `say(text)` (a note in the
    conversation), `switch(path)` (VegaApp.switch_project) and `send(text)` (run a turn). Returns
    True when the extension asked the user something itself (the TUI then leaves out its own
    "not a project" note). Default: False."""
    return _quiet("chat_opened", False, ui, project=project, **context) is True


def chat_message(ui: Any, text: str, **context: Any) -> str | None:
    """Called by the TUI with a message the user typed (not a /command), before the turn starts,
    with the context of chat_opened. Returns None (default) to run it as a turn, "" when the
    extension handled it (the TUI shows the message and runs no turn), or other text to run
    instead."""
    result = _quiet("chat_message", None, ui, text, **context)
    return result if isinstance(result, str) else None


def warm() -> None:
    """Called once from the TUI's background start-up thread: the package imports what its hooks
    will need, so the first turn does not pay for it (~0.3 s of imports, 2026-10-28). Default:
    nothing."""
    _quiet("warm", None)


def write_starting(project: Any, *, tool: str, **context: Any) -> None:
    """Called by tools.execute just before a write it is about to make (write_file, edit_file,
    ..., remember; the agent loop's and the pipelines' writes) or a shell command that is not
    read-only inspection, with `tool` the tool's name. Lets an extension take a project snapshot
    only for turns that write (a question never pays for one). Default: nothing."""
    _quiet("write_starting", None, project, tool=tool, **context)


def agent_event(agent: Any, kind: str, data: dict[str, Any]) -> None:
    """Called by the agent (CodingAgent._emit) with each progress event it emits, except streamed
    text: `kind` is "tool_call", "tool_result" (data `name`, `args`, `result`; the pipelines'
    writes and check runs too), "plan", "notice", "discarded", "request" or "final" (data `text`:
    the turn's answer, emitted once at the end of every turn). Default: nothing."""
    _quiet("agent_event", None, agent, kind, data)


def answer_context(agent: Any, task: str, **context: Any) -> str:
    """Called once per turn by the question pipeline (context `route` "question") and the agent
    loop (context `route`, the turn's routing record or None) before the model answers; returns
    extra text for the model's request ("" adds nothing). The question pipeline appends it to the
    question; the agent loop sends it as a trailing user message on each request of the turn
    (kept out of the conversation history). Default: ""."""
    result = _quiet("answer_context", "", agent, task, **context)
    return result if isinstance(result, str) else ""


def question_excerpts(project: Any, task: str, *, index: Any, shown: list[tuple[str, int, int]],
                      remaining: int, **context: Any) -> list[str]:
    """Called once per question by question_pipeline.excerpts after the files the question names
    and the search hits, before the project map, with `index` (the CodeIndex), `shown` (the
    (path, first line, last line) ranges already shown) and `remaining` (characters left of the
    excerpt budget), and context `agent` (or None). Returns more excerpt blocks, each a
    "path lines a-b ...:" header and numbered lines; the pipeline adds them in order while they
    fit in `remaining`. Default: [] (nothing added)."""
    result = _quiet("question_excerpts", [], project, task, index=index, shown=shown,
                    remaining=remaining, **context)
    return [block for block in result if isinstance(block, str) and block.strip()] \
        if isinstance(result, list) else []


def memory_context(agent: Any, task: str, *, step: int, **context: Any) -> str:
    """Called by the agent loop before each model request of a turn (`step`: the loop step,
    from 0), after any compaction. Returns text the loop adds to that request as a trailing user
    message; it is never stored in the history, so compaction never drops it and the cached
    prompt prefix stays the same. Unlike system_context it may change during the turn (e.g.
    notes gathered once the turn is under way). Default: "" (nothing added)."""
    result = _quiet("memory_context", "", agent, task, step=step, **context)
    return result if isinstance(result, str) else ""


def budget(name: str, default: int, *, num_ctx: int | None = None, **context: Any) -> int:
    """Called where a pipeline sizes a prompt from a fixed budget (characters, unless the name
    says otherwise): `name` is "module.CONSTANT" (e.g. "project_pipeline.FILE_CHARS"), `default`
    the constant, `num_ctx` the model's context window in tokens (None: unknown), or context
    `agent` (anything with a `num_ctx` attribute), read only when an extension defines the
    hook, so without one the window is never resolved for this. Returns the budget to use, a
    positive whole number. Default: `default`."""
    if _impl("budget") is None:
        return default
    if num_ctx is None and context.get("agent") is not None:
        try:
            num_ctx = getattr(context["agent"], "num_ctx", None)
        except Exception:  # noqa: BLE001 - an unknown window is the default budget
            num_ctx = None
    result = _quiet("budget", default, name, default, num_ctx=num_ctx, **context)
    return result if isinstance(result, int) and not isinstance(result, bool) and result > 0 \
        else default


def finish_problems(agent: Any, task: str, **context: Any) -> list[str]:
    """Called by the agent loop's end-of-turn checks on a change turn's would-be final answer
    (at most agent.FINISH_PROBLEM_ROUNDS times a turn), with context `wrote` (a file was written
    this turn), `answer` (the reply) and `round` (1, 2, ...). Returns what the turn still owes,
    one line each (e.g. a file the request names that does not exist); non-empty sends the answer
    back with them, through the finish gate's note. Default: [] (nothing owed)."""
    result = _quiet("finish_problems", [], agent, task, **context)
    return [str(line) for line in result if str(line).strip()] if isinstance(result, list) else []


def lost_reply(agent: Any, task: str, **context: Any) -> str | None:
    """Called by the agent loop when a reply had more than agent.LOST_REPLY_TOKENS completion tokens
    but no text and no tool call (the server dropped a tool call it could not parse), with context
    `reply_tokens`, `count` (lost replies so far this turn, this one included), `finish` (the
    finish reason) and `tools` (the offered tool names). Returns a note to send back to the model
    (the step is retried with it), or None (default): the reply is handled as an empty answer."""
    result = _quiet("lost_reply", None, agent, task, **context)
    return result if isinstance(result, str) and result.strip() else None


def read_only_task(task: str, **context: Any) -> bool:
    """Called by the finish gate (agent.asks_for_change, FinishGate.for_task) with the turn's task.
    True: the task forbids changing any file, beyond what routing.read_only reads, so the gate
    never asks for an edit and treats every changed file as outside the task's scope. Default:
    False (routing.read_only alone decides)."""
    return _quiet("read_only_task", False, task, **context) is True


def route_task(task: str, *, project: Any, **context: Any) -> Any:
    """Called by routing.route_task before its rules, with context `can_write`, `num_ctx` and
    `agent` (the agent routing its turn; None when routing is called without one).
    Returns a routing.Route to use instead (e.g. a read-only advice turn: name "agent", mode
    "advise", which the agent loop runs with no write tool, see CodingAgent._run_advice), or None
    (default): the rules decide."""
    from .routing import Route

    result = _quiet("route_task", None, task, project=project, **context)
    return result if isinstance(result, Route) else None


def compact_history(messages: list[dict[str, Any]], goal_index: int, num_ctx: int, log: Any,
                    **context: Any) -> Any:
    """Called by CodingAgent._fit_context once the next request would pass context.COMPACT_AT of
    the window, with the arguments of context.compact (`extra_tokens`, `scale`, `ledger`,
    `pinned`) and `agent`. Returns a context.Compaction to use instead of the core's (its
    `messages` replace the history; the goal message must be the same object), or None (default):
    context.compact decides."""
    from .context import Compaction

    result = _quiet("compact_history", None, messages, goal_index, num_ctx, log, **context)
    if not isinstance(result, Compaction) or not result.messages:
        return None
    if not any(message is messages[goal_index] for message in result.messages):
        return None  # the agent finds its goal by identity; a result without it is not usable
    return result


def compact_share(share: float, *, num_ctx: int, reply_tokens: int | None = None,
                  projected: int | None = None) -> float:
    """Called by context.compact_at (the compaction trigger, the TUI's context gauge) with the
    core's share (context.COMPACT_AT), the window `num_ctx`, the reply cap `reply_tokens` (None:
    the session's) and `projected` (the prompt tokens about to be compared, None when only the
    share is wanted). Returns the share to compact at, above 0 and below 1. Default: `share`."""
    result = _quiet("compact_share", share, share, num_ctx=num_ctx, reply_tokens=reply_tokens,
                    projected=projected)
    return result if isinstance(result, float) and 0 < result < 1 else share


def code_snippet(lines: list[str], picked: list[int], start: int, end: int, *,
                 default: str) -> str:
    """Called by index.CodeIndex._hit for every search hit (search_code shows them) with the
    file's `lines`, the 1-based line numbers the core picked, the chunk's `start`/`end` lines and
    the core's snippet (`default`: `N: <stripped line>` per picked line). Returns the snippet to
    show instead, or None (default): `default`."""
    if _impl("code_snippet") is None:
        return default
    result = _quiet("code_snippet", None, lines, picked, start, end, default=default)
    return result if isinstance(result, str) and result.strip() else default


def routed_answer(agent: Any, task: str, answer: str, *, before: int, route: str) -> None:
    """Called by CodingAgent._run_routed after a pipeline answered a routed turn (not the agent
    loop), with `before` (len(agent.messages) when the pipeline started) and the route's name. The
    question and project pipelines add their exchange to agent.messages themselves; the others do
    not. The extension may add it (or anything else) to agent.messages. Default: nothing."""
    _quiet("routed_answer", None, agent, task, answer, before=before, route=route)


def question_sentence(sentence: str, *, before: str, after: str) -> bool:
    """Called by verbs (routing, question_pipeline and clarify read change verbs through it) for a
    change verb in a question sentence that the core would count as a request, with the sentence
    and its text `before` and `after` the verb. True: the verb is asked about, not requested
    ("Which tests should we add?"). Default: False."""
    if _impl("question_sentence") is None:
        return False
    return _quiet("question_sentence", False, sentence, before=before, after=after) is True


def change_request(task: str) -> bool:
    """Called by question_pipeline.is_question for a question-shaped request that no change verb
    of the core's marks as a change. True: it asks for a change anyway (e.g. "can you build ...?"),
    so it is not answered as a question. Default: False."""
    return _quiet("change_request", False, task) is True


def page_measured(page: Any, *, html: str) -> dict[str, Any]:
    """Called by the web pipeline for each page it renders (before and after a round's writes),
    with the rendered page (page_render: `eval`, `call`) and its file name. Returns measurements
    to keep with the snapshot for page_failure (a dict, {} for none). Default: {}."""
    if _impl("page_measured") is None:
        return {}
    result = _quiet("page_measured", {}, page, html=html)
    return result if isinstance(result, dict) else {}


def page_failure(check: str, failure: str, *, before: dict[str, Any], gone: list[str]) -> str:
    """Called by the web pipeline for a failing rendered check (`check` "<page> ids": elements no
    longer in the page) with its failure line, the page's page_measured result from before the
    change and the lost ids. Returns the line to use (e.g. with where each lost id was). Default:
    `failure`."""
    result = _quiet("page_failure", failure, check, failure, before=before, gone=gone)
    return result if isinstance(result, str) and result.strip() else failure


def example_script(script: str) -> str:
    """Called by pipeline.run_examples with CHECK_SCRIPT, the Python it runs in the sandbox on the
    docstring examples (argv[1]: JSON [module, call, expected, source] rows; its last line: a JSON
    list of failure lines). Returns the script to run instead. Default: `script`."""
    result = _quiet("example_script", script, script)
    return result if isinstance(result, str) and result.strip() else script


def examples_failed(failures: list[str], *, path: str) -> None:
    """Called by pipeline.run_examples with the failure lines the script printed for `path`."""
    _quiet("examples_failed", None, failures, path=path)


def model_profiles() -> dict[str, dict[str, Any]]:
    """Called by models.load: model profiles to add, {name: the table models.toml would hold}. A
    name models.toml already has keeps its own. Default: none."""
    if _impl("model_profiles") is None:
        return {}
    result = _quiet("model_profiles", {})
    return {str(k): v for k, v in result.items() if isinstance(v, dict)} if isinstance(result, dict) else {}


def catalog_models() -> dict[str, dict[str, Any]]:
    """Called by model_store.load_catalog for the shipped catalogue: entries to add, {name: the
    table model_catalog.toml would hold}. A name the catalogue already has keeps its own.
    Default: none."""
    if _impl("catalog_models") is None:
        return {}
    result = _quiet("catalog_models", {})
    return {str(k): v for k, v in result.items() if isinstance(v, dict)} if isinstance(result, dict) else {}


def tool_list(tools: list[dict[str, Any]], task: str, **context: Any) -> list[dict[str, Any]]:
    """Called by the agent loop with the tool definitions a turn's requests offer (also its
    warm-up and the TUI's context gauge, with task "") and context `agent`. Returns the list to
    offer instead: definitions left out or reworded, never new tools. Default: `tools`."""
    result = _quiet("tool_list", tools, tools, task, **context)
    if not isinstance(result, list) or not all(isinstance(spec, dict) for spec in result):
        return tools
    names = {str(spec.get("function", {}).get("name")) for spec in tools}
    return result if all(str(spec.get("function", {}).get("name")) in names for spec in result) else tools


def system_prompt(prompt: str) -> str:
    """Called by agent.system_prompt with the agent loop's rules (SYSTEM with switched-off
    features' lines removed). Returns the rules to use (e.g. without the lines of tools tool_list
    leaves out). Default: `prompt`."""
    result = _quiet("system_prompt", prompt, prompt)
    return result if isinstance(result, str) and result.strip() else prompt


def allow_command(command: str, project: Any, **context: Any) -> bool:
    """Called by the agent loop for a run_command in a session whose commands are disabled (evals),
    with context `agent`. True: run it anyway (e.g. a read-only git inspection that stands in for
    a tool left out). Default: False (refused as disabled)."""
    return _quiet("allow_command", False, command, project, **context) is True


def unchanged_offer(project: Any, target: str, task: str, *, source: str, **context: Any) -> str:
    """Called by the project pipeline while building the writer's prompt for an existing file
    (`target`, posix, relative to `project`), with context `current` (its text now), `original`
    (its text before this run, None when the run created it), `failure` (the failing output the
    writer is shown, "" in the first round) and `change` (the plan item's change). Returns text
    added at the end of the prompt that lets the writer answer that the file needs no change
    (unchanged_reply then recognises such a reply). Default: "" (no offer)."""
    result = _quiet("unchanged_offer", "", project, target, task, source=source, **context)
    return result if isinstance(result, str) else ""


def unchanged_reply(reply: str, target: str, *, source: str, **context: Any) -> bool:
    """Called by the project pipeline with a writer's reply to a prompt that carried an
    unchanged_offer. True: the reply says `target` needs no change, and the file is kept exactly
    as it is (handled like a reply that repeats the file unchanged). Default: False."""
    return _quiet("unchanged_reply", False, reply, target, source=source, **context) is True


def planner_request(schema: dict[str, Any], system: str, *, source: str,
                    **context: Any) -> tuple[dict[str, Any], str]:
    """Called by the project pipeline before each planner call with the plan's JSON schema and the
    planner's system prompt, context `task` and `replan` (a repair round's plan). Returns the
    (schema, system) to send; the reply is read as before (only its "files"). Default: both
    unchanged."""
    result = _quiet("planner_request", (schema, system), schema, system, source=source, **context)
    if (isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], dict)
            and isinstance(result[1], str) and result[1].strip()):
        return result
    return schema, system


def side_model(job: str, *, agent: Any, **context: Any) -> str:
    """Called before a side-job model call (one that is not the main writer or agent turn: "planner",
    "clarify", "acceptance", "task_checks", ...) for the model name to send it to; the request
    still goes through the agent's client and sampling. Default: the agent's own model."""
    model = getattr(agent, "model", None)
    result = _quiet("side_model", None, job, agent=agent, **context)
    return result if isinstance(result, str) and result.strip() else model


def stalled_round(source: str, **context: Any) -> dict[str, Any] | None:
    """Called by write_loop.run_loop (`source`: the loop's name, e.g. "direct_pipeline") when a
    repair round rewrote every file byte-identically, with context `project`, `targets` (the
    round's files), `saved`, `failures` (the last checked round's failing lines), `output` (its
    test output and report), `stalls` (unchanged rounds in a row), `previous` (the action taken
    on the stall before, {"note", "branch"}), `can_widen` (the loop takes more files),
    `can_handover` (the caller can hand the task over), `can_sample` (a round may carry its own
    temperature), `task` and `round`. Returns None (the default: the core note, then stop on the
    second) or an action: "note" (or "note_for": the files added -> note) for the writer,
    "widen" + "why" (files to add to the next round and their change text), "handover"
    ({"paths", "failure", "reason"}: end the loop and let the caller hand the task over),
    "temperature" (the next round's), "extra_round" (the round does not use up a repair, at
    most write_loop.MAX_STALL_ROUNDS a loop), "branch"/"reason" (for the debug log)."""
    result = _quiet("stalled_round", None, source, **context)
    return result if isinstance(result, dict) else None


def deliverable_prompt(project: Any, task: str, wanted: list[str], *, source: str,
                       **context: Any) -> str:
    """Called by the deliverable pipeline once, while building its first prompt, with context
    `shown` (the input files as shown) and `fixture` (the folder the task maps to `/`, or "").
    Returns text added after the input files. Default: "" (nothing added)."""
    result = _quiet("deliverable_prompt", "", project, task, wanted, source=source, **context)
    return result if isinstance(result, str) else ""


def deliverable_runner(project: Any, task: str, *, source: str, **context: Any) -> dict[str, Any]:
    """Called by the deliverable pipeline before each script run, with context `fixture`. Returns
    extra options for deliverable_runner.py (e.g. `path_map`); {} adds none. Default: {}."""
    result = _quiet("deliverable_runner", {}, project, task, source=source, **context)
    return result if isinstance(result, dict) else {}


def deliverable_direct(task: str, *, source: str, **context: Any) -> str:
    """Called by the deliverable pipeline before its first request (context `rounds` 0) and after
    each round that ended on a script error (`rounds`, `script_errors`), with context `project`,
    `wanted` and `shown`. A non-empty text switches the run to direct writing: the text goes to
    the writer (in the first prompt, or after the script error), and from then on a reply with a
    `File:` block for every deliverable is taken as written, its script not run. Default: ""."""
    result = _quiet("deliverable_direct", "", task, source=source, **context)
    return result if isinstance(result, str) else ""


def deliverable_repair(feedback: str, *, source: str, **context: Any) -> str:
    """Called by the deliverable pipeline with the feedback for a round that ended on a script
    error, context `project`, `task`, `wanted`, `error` and `script`. Returns the feedback to
    send (e.g. with the files that do exist where the script opened a missing one). Default: the
    feedback unchanged."""
    result = _quiet("deliverable_repair", feedback, feedback, source=source, **context)
    return result if isinstance(result, str) and result.strip() else feedback


def deliverable_files(files: dict[str, str], *, source: str, **context: Any) -> dict[str, str]:
    """Called by the deliverable pipeline with a round's deliverables {path: content} before they
    are checked and written, context `project`, `task` and `fixture`. Returns the files to use
    (e.g. a snapshot copy's path `root/etc/x` written as the real path `/etc/x`). Default: the
    files unchanged."""
    result = _quiet("deliverable_files", files, files, source=source, **context)
    if isinstance(result, dict) and set(result) == set(files) and all(
            isinstance(v, str) for v in result.values()):
        return result
    return files


def check_call(agent: Any, command: str, **context: Any) -> tuple[str, str] | None:
    """Called by the agent loop before it runs a model's run_check `command`, with context `task`,
    `changed` (files written this turn) and `read` (files read this turn). Returns (the command
    actually run, its result in run_check's shape) to use instead; None runs `command` as usual.
    Default: None."""
    result = _quiet("check_call", None, agent, command, **context)
    if (isinstance(result, tuple) and len(result) == 2 and all(isinstance(r, str) for r in result)
            and result[1]):
        return result
    return None


def suite_rerun(spec: Any, project: Any, **context: Any) -> str | None:
    """Called by acceptance (targeted_tests) just before it runs a large project's whole suite,
    with the turn's acceptance.Spec (`spec.suite_output`: the last whole-suite output this turn)
    and context `changed`. Returns a test command to run instead (e.g. only what failed in the
    whole-suite run earlier this turn); None runs the whole suite. Default: None."""
    result = _quiet("suite_rerun", None, spec, project, **context)
    return result if isinstance(result, str) and result.strip() else None


def targeted_first(spec: Any, project: Any, *, changed: list[str], stop: Any, item: Any) -> dict[str, Any] | None:
    """Called by acceptance._targeted_items (a large pytest suite, before the test files related to
    the turn's changes run) with the acceptance Spec, the files `changed`, the `stop` event and
    `item` (acceptance._command_item: runs a command as an acceptance Item). Returns {"items":
    [Item, ...] run first, "stop": True when they decide (a failure) so nothing else runs, "suite":
    an Item that stands in for the whole-suite run after the related files passed, or None} or None
    (default): the related files, then the whole suite."""
    from .acceptance import Item

    result = _quiet("targeted_first", None, spec, project, changed=changed, stop=stop, item=item)
    if not isinstance(result, dict) or not all(isinstance(i, Item) for i in result.get("items") or []):
        return None
    suite = result.get("suite")
    return {"items": list(result.get("items") or []), "stop": result.get("stop") is True,
            "suite": suite if isinstance(suite, Item) else None}


def timeout_hint(hint: str, *, command: str, project: Any, timeout: float) -> str:
    """Called by targeted_tests.run_check_hint with the note sandbox.run_check adds after a large
    suite's whole run timed out (a time limit, not a failure, and what to run instead). Returns
    the note to use. Default: `hint`."""
    result = _quiet("timeout_hint", hint, hint, command=command, project=project, timeout=timeout)
    return result if isinstance(result, str) and result.strip() else hint


def check_result(agent: Any, command: str, result: str, *, changed: list[str], read: list[str]) -> str:
    """Called by the agent loop with a model's run_check result that the core ran (not an
    extension's check_call), with `changed` (files written this turn) and `read` (files read).
    Returns the result the loop uses instead (e.g. a timeout's hint naming real test files).
    Default: `result`."""
    shown = _quiet("check_result", result, agent, command, result, changed=changed, read=read)
    return shown if isinstance(shown, str) and shown else result


def baseline_gate(agent: Any, **context: Any) -> bool:
    """Called by the agent loop (once per agent) where the baseline_aware_gate switch is checked:
    True turns that gate on for this agent's project although the switch is off. Default:
    False."""
    return _quiet("baseline_gate", False, agent, **context) is True


def test_baseline(agent: Any, command: str, *, existing: Any, **context: Any) -> Any:
    """Called by the agent loop just before the session's first edit, with the gate on, the
    project's test command and the baseline so far (`existing`: the model's own whole-suite run
    before any edit, or None). Returns a baseline_gate.Baseline to use instead (e.g. a lazy one
    whose failing tests are checked on demand); None keeps the old behaviour (`existing`, else
    one sandboxed run now). Default: None."""
    from . import baseline_gate as gate

    result = _quiet("test_baseline", None, agent, command, existing=existing, **context)
    return result if isinstance(result, gate.Baseline) else None


def round_score(source: str, *, checks: Any, default: tuple[int, ...], **context: Any
                ) -> tuple[int, ...]:
    """Called by write_loop.run_loop after a round's checks (and checks_recorded), with the
    round's `checks` (a write_loop.Checks), `default` (the loop's own rank: config.score, else
    (failing checks,)), `round`, `project` and `task`. Returns the rank to keep the best round by
    (lower is better; every round of a loop must get the same shape), or None: `default`.
    Default: `default`."""
    result = _quiet("round_score", None, source, checks=checks, default=default, **context)
    if isinstance(result, tuple) and result and all(isinstance(n, int) for n in result):
        return result
    return default


def finish_check(agent: Any, task: str, **context: Any) -> dict[str, Any] | None:
    """Called by the agent loop's end-of-turn checks on a change turn's would-be final answer
    after a write, with context `written` (the files the turn wrote), `answer` (the reply),
    `writes` (successful edits so far), `stop` and `baseline` (the tests failing before the first
    edit, or None). Returns None (default: the core's own finish checks run: the acceptance
    finish gate, verify_before_finish and claim_check) or {"note": text}: the extension checked
    the turn and owns that finish; a non-empty note sends the answer back with it, "" lets it
    stand (the core's three finish checks are skipped either way)."""
    result = _quiet("finish_check", None, agent, task, **context)
    if isinstance(result, dict) and isinstance(result.get("note", ""), str):
        return result
    return None


def tool_shown(agent: Any, name: str, args: dict[str, Any], result: str,
               **context: Any) -> str | None:
    """Called by the agent loop for every tool result of a call that ran, before the core
    condenses it (repair_feedback) and caps it, with context `calls` (calls run so far this turn,
    this one included), `writes` (saved writes so far) and `goal` (the turn's goal message, the
    same object for the whole turn). Returns the text the model sees instead, or None (default:
    the core's text). The raw result still feeds the ledger, the work log and the debug log."""
    shown = _quiet("tool_shown", None, agent, name, args, result, **context)
    return shown if isinstance(shown, str) and shown else None


def compaction_pins(agent: Any, **context: Any) -> str:
    """Called by CodingAgent._fit_context when it compacts, with context `goal`. Returns text kept
    across the compaction (appended to the pinned text, which the compacted history keeps
    verbatim), or "" (default)."""
    text = _quiet("compaction_pins", "", agent, **context)
    return text if isinstance(text, str) else ""


GROW_KEYS = ("model", "num_ctx", "reason", "keep_tokens")


def larger_window(agent: Any, *, step: int, **context: Any) -> dict[str, Any] | None:
    """Called by CodingAgent._fit_context before every agent-loop request, with context `budget`
    (the turn's context.Budget), `over` (the request would compact), `compactions` (this turn's
    compactions so far), `projected` (its projected prompt tokens) and `threshold` (the tokens at
    which it compacts). Returns {"model", "num_ctx", "reason", optional "keep_tokens"}: the
    agent switches to that model with that window for the rest of its life (grow_window), or
    None (default): the window stays. A result without a model name or with a window that is
    not larger than the current one is ignored."""
    result = _quiet("larger_window", None, agent, step=step, **context)
    if not isinstance(result, dict):
        return None
    model, num_ctx = result.get("model"), result.get("num_ctx")
    if not isinstance(model, str) or not model or not isinstance(num_ctx, int):
        return None
    current = getattr(agent, "num_ctx", None)
    if isinstance(current, int) and num_ctx <= current:
        return None
    keep = result.get("keep_tokens")
    return {"model": model, "num_ctx": num_ctx, "reason": str(result.get("reason") or ""),
            "keep_tokens": keep if isinstance(keep, int) and keep > 0 else None}


class CaseEnvironmentError(RuntimeError):
    """Raised by an extension's case_environment when the attempt's environment cannot be made:
    the attempt is a fixture error (and the fixture guard is skipped)."""


def check_python() -> Any:
    """Called wherever a check or project command picks its Python (sandbox._python_for,
    find_python_tool, run_check's PATH, path scrubbing, an eval's python_check and commands). Returns
    an existing interpreter (a Path at <env>/bin/python) that stands in for Vega's own, e.g. an eval
    attempt's isolated environment, or None (default): Vega's own."""
    from pathlib import Path

    if _impl("check_python") is None:
        return None
    result = _quiet("check_python", None)
    if result is None:
        return None
    path = Path(result)
    return path if path.is_absolute() and path.exists() else None


def case_environment(case: dict[str, Any], *, suite: str) -> Any:
    """Called by the eval runner around each attempt (and around the fixture guard's checks, with
    case {}), with the case and its suite. Returns a context manager whose value is a dict of fields
    added to the attempt's record ({} for none); entering it may raise CaseEnvironmentError (the
    attempt is then a fixture error). Default: a context that changes nothing."""
    import contextlib

    fn = _impl("case_environment")
    if fn is None:
        return contextlib.nullcontext({})
    try:
        manager = fn(case, suite=suite)
    except Exception as exc:  # noqa: BLE001 - see _quiet
        from . import debuglog

        debuglog.record("stage", stage="extensions.case_environment.error", error=str(exc)[:300])
        return contextlib.nullcontext({})
    return manager if hasattr(manager, "__enter__") and hasattr(manager, "__exit__") \
        else contextlib.nullcontext({})


def check_workspace(root: Any) -> tuple[Any, Any] | None:
    """Called by sandbox.run_check with the run's fresh workspace folder (removed afterwards).
    Returns (clone folder, temp folder): two different folders inside `root`, neither inside the
    other (the temp folder is then writable in the sandbox profile and labelled <tmp> in the
    output), or None (default): the clone is `root` itself and its temp folder `root/.vega-tmp`."""
    from pathlib import Path

    result = _quiet("check_workspace", None, root)
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        return None
    try:
        base, clone, temp = Path(root).resolve(), Path(result[0]).resolve(), Path(result[1]).resolve()
    except (TypeError, OSError):
        return None
    if base not in clone.parents or base not in temp.parents or clone == temp \
            or clone in temp.parents or temp in clone.parents:
        return None
    return clone, temp


def command_run(argv: Any, *, cwd: Any, env: Any, timeout: float, stop: Any,
                shell: bool) -> Any:
    """Called by sandbox.run_stoppable (every run_check, run_command, pipeline check and eval
    check) before it starts the command. Returns a subprocess.CompletedProcess when the extension
    ran the command itself (watching it while it ran), or None (default: the core runs it). The
    extension raises what run_stoppable raises (subprocess.TimeoutExpired, sandbox.StoppedRun) and
    an OSError from starting the process; those pass through. Any other error of the hook is
    logged and the core runs the command (a hook must not fail after the command started)."""
    import subprocess

    fn = _impl("command_run")
    if fn is None:
        return None
    from .sandbox import StoppedRun

    try:
        result = fn(argv, cwd=cwd, env=env, timeout=timeout, stop=stop, shell=shell)
    except (subprocess.TimeoutExpired, StoppedRun, OSError):
        raise
    except Exception as exc:  # noqa: BLE001 - see the docstring
        from . import debuglog

        debuglog.record("stage", stage="extensions.command_run.error", error=str(exc)[:300])
        return None
    return result if isinstance(result, subprocess.CompletedProcess) else None
