from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import json
import os
import sys
import threading
import time

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from . import external, features, models, server
from .model_store import WATCH_FAMILIES
from .resources import Guard
from .transcripts import SessionLog
from .agent import CodingAgent
from .autopilot import autopilot_command
from .evals import MODEL_TEMPERATURE, SCREEN_MIN, compare_runs, merge_runs, run_ab, run_evaluations
from .tools import execute, is_project
from .training import training_app
from .ui import TerminalRenderer, undo_note

if TYPE_CHECKING:
    from .loops import LoopDef

app = typer.Typer(
    help="Vega, a local-first coding agent. Run `vega` with no command to start a chat.",
    invoke_without_command=True,
    no_args_is_help=False,
)
server_app = typer.Typer(help="Manage the local model server.")
model_app = typer.Typer(help="Choose and build the base model behind Vega (profiles in models.toml).")
lab_app = typer.Typer(help="Planned experiments: run them back to back (attended) and keep a scoreboard.")
debug_app = typer.Typer(help="Debug mode: watch what Vega sends the model and gets back, live.")
loops_app = typer.Typer(
    help="Loops: watchers that check your work on their own (Bug Watcher first). Deterministic "
    "checks run any time; model work only while the screen is locked.",
    invoke_without_command=True,
)
context_app = typer.Typer(help="Project context files: VEGA.md, session digests "
                          "(.vega/sessions.md) and notes, loaded at the start of each chat.")
learn_app = typer.Typer(help="Vega's knowledge library: learn web pages, PDFs (books) and documents, "
                        "then search them while working.")
app.add_typer(server_app, name="server")
app.add_typer(model_app, name="model")
app.add_typer(lab_app, name="lab")
app.add_typer(learn_app, name="learn")
web_app = typer.Typer(help="Web research: the search backend (self-hosted SearXNG), permission and "
                      "a manual search.")
app.add_typer(web_app, name="web")
adapters_app = typer.Typer(help="Site adapters Vega wrote for official APIs (you approved each): "
                           "list, show, remove, verify.")
web_app.add_typer(adapters_app, name="adapters")
app.add_typer(context_app, name="context")
app.add_typer(debug_app, name="debug")
app.add_typer(loops_app, name="loops")
app.add_typer(training_app, name="train")


def _require_checkout(command: str) -> None:
    """Developer tools write under the repository; in an installed package that would be the
    Python library folder, so they stop with a pointer instead."""
    if not server.source_checkout():
        console.print(f"`vega {command}` is a developer tool: it reads and writes files in a source "
                      "checkout of Vega (git clone https://github.com/isaacfrett/vegacode) and does "
                      "not work from an installed package.", highlight=False)
        raise typer.Exit(1)


@lab_app.callback()
def _lab_needs_checkout() -> None:
    _require_checkout("lab")


@training_app.callback()
def _train_needs_checkout() -> None:
    _require_checkout("train")
console = Console()

ProjectOption = typer.Option(Path.cwd(), "--project", "-p", help="Project root directory.")
ModelOption = typer.Option(
    None, "--model", "-m",
    help="Model profile from models.toml (built as vega-<profile> on demand, or its own endpoint "
    "server) or an Ollama model.",
)


def _resolve_project(project: Path) -> Path:
    project = project.expanduser().resolve()
    if not project.is_dir():
        raise typer.BadParameter(f"Not a directory: {project}")
    return project


def _ensure_server(*values: str | None) -> None:
    """Start Ollama on demand for the models involved (default: the current one); an endpoint
    profile's server is never started by Vega."""
    try:
        for value in values or (None,):
            if models.uses_ollama(value):
                server.ensure_running(value)
                return
    except RuntimeError as exc:
        console.print(f"[yellow]{exc}[/yellow]")


def _select_model(value: str | None) -> None:
    """Point this process (agents, evals, server checks) at the chosen model via VEGA_MODEL."""
    if not value:
        return
    _ensure_server(value)
    try:
        os.environ["VEGA_MODEL"] = models.resolve(value)
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    models.apply_feature_defaults(os.environ["VEGA_MODEL"])  # defaults follow the chosen model


ReviewOption = typer.Option(
    False, "--review", help="Show each file edit as a diff and ask before applying it.",
)
UNDO_DEPTH = 20
HELP_TEXT = (
    "Ask for engineering work in plain language. Vega inspects the repository, makes focused "
    "edits, and reports checks.\n"
    "  /status          Git status\n"
    "  /diff            tracked changes against HEAD\n"
    "  /undo [--force]  restore files changed in the last turn (repeat to go further back)\n"
    "  /review on|off   ask before applying each edit (currently {review})\n"
    "  /teach on|off    explain decisions while working (teach mode)\n"
    "  /reset           clear chat context\n"
    "  /exit            leave the session"
)


PlainOption = typer.Option(
    False, "--plain", help="Use the line-mode chat instead of the full-screen terminal app.",
)
TeachOption = typer.Option(
    False, "--teach", help="Teach mode: Vega states its plan, says why before each change or "
    "check, and ends with what to take away (also /teach on, or VEGA_TEACH=1).",
)


def _use_tui(plain: bool) -> bool:
    """The full-screen app needs a real terminal on both ends; pipes and --plain get line mode."""
    if plain:
        return False
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _start_chat(project: Path, *, review: bool, plain: bool, teach: bool = False) -> None:
    from . import debuglog

    log = debuglog.start(f"chat-{project.name}")
    if log is not None:
        console.print(f"[dim]Debug log: {log.path} — watch it with `vega debug watch` in another terminal.[/dim]")
    if not _use_tui(plain):
        _interactive_session(project, review=review, teach=teach)
        return
    _ensure_server()
    from .tui import run_tui  # imported here: Textual is only needed for the full-screen chat

    run_tui(project, review=review, undo_depth=UNDO_DEPTH, teach=True if teach else None)


@app.callback()
def main(
    ctx: typer.Context, project: Path = ProjectOption, review: bool = ReviewOption,
    model: str = ModelOption, plain: bool = PlainOption, teach: bool = TeachOption,
    debug: bool = typer.Option(False, "--debug", help="Log every model request, reply and tool "
                               "call live to ~/.vega/debug; watch it with `vega debug watch`."),
) -> None:
    """Start an interactive chat when no command is given."""
    if debug:
        os.environ["VEGA_DEBUG"] = "1"
    _select_model(model)
    models.apply_feature_defaults(server.model_name())
    ctx.obj = {"plain": plain, "teach": teach}
    if ctx.invoked_subcommand is None:
        _start_chat(_resolve_project(project), review=review, plain=plain, teach=teach)


@app.command()
def chat(
    ctx: typer.Context, project: Path = ProjectOption, review: bool = ReviewOption,
    model: str = ModelOption, plain: bool = PlainOption, teach: bool = TeachOption,
) -> None:
    """Start an interactive chat with Vega (same as running `vega` alone)."""
    _select_model(model)
    plain = plain or bool((ctx.obj or {}).get("plain"))
    teach = teach or bool((ctx.obj or {}).get("teach"))
    _start_chat(_resolve_project(project), review=review, plain=plain, teach=teach)


@app.command()
def run(
    task: str = typer.Argument(..., help="Task for Vega to complete, then exit."),
    project: Path = ProjectOption,
    review: bool = ReviewOption,
    model: str = ModelOption,
    teach: bool = TeachOption,
) -> None:
    """Run a single task non-interactively."""
    project = _resolve_project(project)
    _select_model(model)
    _ensure_server()
    CodingAgent(project, stream=True, on_event=TerminalRenderer(console), review=review,
                teach=True if teach else None, web_research=True).run(task)


def _undo(agent: CodingAgent, force: bool) -> None:
    report = agent.undo(force=force)
    if report is None:
        console.print("[dim]Nothing to undo: no earlier turn in this session changed files.[/dim]")
        return
    if not report.applied:
        console.print(
            "[yellow]Not undone. These files changed after Vega last wrote them, so restoring them "
            "would discard those later edits:[/yellow]"
        )
        for path in report.conflicts:
            console.print(f"  {path}")
        console.print("Use [bold]/undo --force[/bold] to restore them anyway.")
        return
    for path in report.restored:
        console.print(f"[green]restored[/green] {path}")
    for path in report.removed:
        console.print(f"[green]removed[/green]  {path} [dim](created by Vega)[/dim]")
    if report.conflicts:
        console.print(f"[yellow]Overwrote later changes in: {', '.join(report.conflicts)}[/yellow]")
    remaining = len(agent.checkpoints)
    console.print(f"[dim]{remaining} earlier turn(s) can still be undone.[/dim]")
    agent.messages.append(undo_note(report.restored + report.removed))


def _set_review(agent: CodingAgent, argument: str) -> None:
    if argument in {"on", "off"}:
        agent.review = argument == "on"
    elif argument:
        console.print("[yellow]Usage: /review on|off[/yellow]")
        return
    state = "on: each edit is shown as a diff for approval" if agent.review else "off"
    console.print(f"Review mode {state}.")


def _set_teach(agent: CodingAgent, argument: str) -> None:
    if argument in {"on", "off"}:
        agent.set_teach(argument == "on")
    elif argument:
        console.print("[yellow]Usage: /teach on|off[/yellow]")
        return
    state = ("on: Vega states its plan, says why before each change or check, and ends with what "
             "to take away" if agent.teach else "off")
    console.print(f"Teach mode {state}.")


def _interactive_session(project: Path, *, review: bool = False, teach: bool = False) -> None:
    _ensure_server()
    agent = CodingAgent(
        project, stream=True, on_event=TerminalRenderer(console),
        review=review, undo_depth=UNDO_DEPTH, teach=True if teach else None,
        web_research=True,
    )
    session_log = SessionLog(server.STATE_DIR / "sessions", project)
    # Load the model and cache the fixed prompt while the user types the first message.
    threading.Thread(target=agent.warm_up, daemon=True).start()
    mode = " · review on" if review else ""
    console.print(f"[bold]Vega terminal chat[/bold] · project: {project}{mode}")
    if not is_project(project):
        console.print(
            "[yellow]This folder doesn't look like a project (no Git repository or project file), "
            "so Vega only sees a partial view. Start Vega inside a project, or use "
            "[bold]vega -p <path>[/bold].[/yellow]"
        )
    console.print(
        "Ask Vega to investigate, debug, implement, refactor, review, or document work. "
        "Use [bold]/undo[/bold] to revert the last turn's edits, [bold]/review on[/bold] to "
        "approve each edit, [bold]/help[/bold] for all commands, or [bold]/exit[/bold] to leave."
    )
    try:
        _chat_loop(agent, project, session_log)
    finally:
        session_log.close()  # the session's final digest in .vega/sessions.md (project_context)


def _chat_loop(agent: CodingAgent, project: Path, session_log: SessionLog) -> None:
    while True:
        try:
            message = console.input("[bold cyan]you> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Vega session ended.[/dim]")
            break
        if not message:
            continue
        command, _, argument = message.lower().partition(" ")
        argument = argument.strip()
        if command in {"/exit", "/quit"}:
            console.print("[dim]Vega session ended.[/dim]")
            break
        if command == "/reset":
            agent.reset()
            console.print("[green]Conversation context cleared.[/green]")
            continue
        if command == "/help":
            console.print(HELP_TEXT.format(review="on" if agent.review else "off"), markup=False)
            continue
        if command == "/status":
            console.print(execute("git_status", {}, project))
            continue
        if command == "/diff":
            console.print(execute("git_diff", {}, project))
            continue
        if command == "/undo":
            if argument not in {"", "--force"}:
                console.print("[yellow]Usage: /undo [--force][/yellow]")
                continue
            _undo(agent, force=argument == "--force")
            continue
        if command == "/review":
            _set_review(agent, argument)
            continue
        if command == "/teach":
            _set_teach(agent, argument)
            continue
        agent.request_log.clear()
        started = time.monotonic()
        answer = agent.run(message)
        session_log.record_turn(message, answer, time.monotonic() - started, list(agent.request_log))
        session_log.save(agent.messages)


@app.command()
def config() -> None:
    """Show active model connection settings."""
    route = models.connection()
    console.print(f"Base URL: {route.base_url}"
                  + (f" (endpoint of profile {route.profile})" if route.is_endpoint else ""))
    console.print(f"Model: {server.model_name()}"
                  + (f" (sent as {route.model!r})" if route.model != server.model_name() else ""))
    console.print(f"Default model built from profile: {models.active_profile() or 'unknown'}")
    console.print(f"State directory: {server.STATE_DIR}")
    from . import teaching, view

    console.print("Teach mode: " + ("on (VEGA_TEACH)" if teaching.from_env() else
                                    "on (saved with /teach)" if view.load().teach else "off"))


UsageSwitch = typer.Argument("", help="on or off; leave out to see the summary.")
UsageDays = typer.Option(14, "--days", help="Summary window in days.")


@app.command("usage")
def usage_log(switch: str = UsageSwitch, days: float = UsageDays) -> None:
    """The opt-in local usage log (off by default): events only, kept in ~/.vega/usage.jsonl and
    never sent anywhere."""
    from . import usage

    if switch not in {"", "on", "off"}:
        raise typer.BadParameter("use on or off")
    if switch:
        usage.set_enabled(switch == "on")
    console.print(usage.summary(days), markup=False, highlight=False)


@app.command("eval")
def evaluate(
    runs: list[str] = typer.Argument(
        None,
        help="With --compare: two runs as numbers (1 = first; after `--`, -1 = last) or "
        "timestamps.",
    ),
    repeat: int = typer.Option(3, "--repeat", "-n", min=1, help="Attempts per case."),
    case: list[str] = typer.Option(None, "--case", "-c", help="Only run cases whose id contains this."),
    suite: str = typer.Option(
        None, "--suite", "-s", help="fixes (default), goals, selfbuild, all, or a benchmark: humaneval, humaneval40, "
        "mbpp, mbpp40 (generate with evals/import_evalplus.py).",
    ),
    compare: bool = typer.Option(
        False, "--compare",
        help="Compare two recorded runs instead of running (default: last two).",
    ),
    model: str = ModelOption,
    cooldown: float = typer.Option(
        0, "--cooldown", min=0, help="Seconds of idle between attempts (heat is guarded directly; default 0).",
    ),
    ab: str = typer.Option(
        None, "--ab",
        help="Paired A/B: A = current settings, B = A with these comma-separated features off.",
    ),
    ab_model: str = typer.Option(
        None, "--ab-model", help="Paired A/B: B uses this model profile (or Ollama model) instead.",
    ),
    agent: str = typer.Option(
        None, "--agent",
        help="Agent that attempts the cases: vega (default) or aider (same model, task text and "
        "checks; see evals/README.md).",
    ),
    ab_agent: str = typer.Option(
        None, "--ab-agent",
        help="Paired A/B: B runs this agent (e.g. aider) instead; all A attempts run first.",
    ),
    ab_on: str = typer.Option(
        None, "--ab-on",
        help="Paired A/B: B = A with these comma-separated opt-in features on (e.g. constrained); "
        "combines with --ab.",
    ),
    temperature: str = typer.Option(
        None, "--temperature",
        help="Sampling temperature sent with every request (default: the profile's temperature, "
        "else 0); `model` sends none (the server's own).",
    ),
    seed: int = typer.Option(
        None, "--seed", help="Sampling seed sent with every request (default: 7).",
    ),
    category: str = typer.Option(
        None, "--category", "-a",
        help="Only cases tagged with this capability area (learn, understand, solve, reason, "
        "adapt, apply).",
    ),
    shard: str = typer.Option(
        None, "--shard", help="Run only shard k/n of the cases (e.g. 1/2) to split a run across processes.",
    ),
    merge: bool = typer.Option(
        False, "--merge", help="Merge the given shard runs (numbers or timestamps) into one run.",
    ),
    early_stop: bool = typer.Option(
        False, "--early-stop",
        help="A/B only: shuffle cases and stop at the halfway look if p < 0.005.",
    ),
    off: str = typer.Option(
        None, "--off",
        help="Comma-separated features to switch off for this run (see `vega features`).",
    ),
    guard: bool = typer.Option(
        True, "--guard/--no-guard",
        help="Pause, then stop, if free memory, free disk or swap growth cross the VEGA_GUARD_* limits.",
    ),
    screen: bool = typer.Option(
        False, "--screen",
        help="Screen: only the cases that told two setups apart in evals/runs.jsonl (topped up to "
        "--screen-min); A/Bs stop for futility at half. Recorded as a subset, never a full result.",
    ),
    screen_min: int = typer.Option(
        SCREEN_MIN, "--screen-min", min=1, help="With --screen: run at least this many cases.",
    ),
) -> None:
    """Run evaluation suites in isolated temporary projects, or compare recorded runs."""
    _require_checkout("eval")
    if merge:
        merge_runs(list(runs or []))
        return
    if shard is not None:
        os.environ["VEGA_EVAL_SHARD"] = shard
    if category is not None:
        os.environ["VEGA_EVAL_CATEGORY"] = category
    if compare:
        if runs and len(runs) != 2:
            raise typer.BadParameter("--compare takes two runs, or none for the last two.")
        first, second = runs if runs else (None, None)
        compare_runs(first, second, suite=suite)
        return
    if runs:
        raise typer.BadParameter("Run arguments are only used with --compare.")
    if off is not None:
        os.environ[features.ENV] = off
    if temperature is not None:
        if temperature.strip().lower() != MODEL_TEMPERATURE:
            try:
                float(temperature)
            except ValueError as exc:
                raise typer.BadParameter("--temperature takes a number or `model`.") from exc
        os.environ["VEGA_TEMPERATURE"] = temperature.strip()
    if seed is not None:
        os.environ["VEGA_SEED"] = str(seed)
    try:
        features.disabled()
        features.enabled()
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if agent is not None:
        os.environ[external.AGENT_ENV] = agent.strip().lower()
    _ensure_server(model, ab_model)
    _select_model(model)
    try:
        paired = (ab is not None or ab_on is not None or ab_model is not None
                  or ab_agent is not None)
        if paired and early_stop and shard:
            raise typer.BadParameter("--early-stop needs the whole run; it cannot be combined with --shard.")
        if paired:
            watcher = Guard(log=lambda text: console.print(f"[yellow]{text}[/yellow]")) if guard else None
            b_off = [name.strip() for name in (ab or "").split(",") if name.strip()]
            b_on = [name.strip() for name in (ab_on or "").split(",") if name.strip()]
            run_ab(
                suite or "fixes", b_off, repeat=repeat, only=case, guard=watcher, cooldown=cooldown,
                early_stop=early_stop, b_on=b_on, b_model=ab_model, screen=screen,
                screen_min=screen_min, b_agent=ab_agent,
            )
        else:
            run_evaluations(repeat=repeat, only=case, suite=suite or "fixes", guard=guard,
                            cooldown=cooldown, screen=screen, screen_min=screen_min)
    finally:
        if server.unload():
            console.print("[dim]Model unloaded from memory.[/dim]")


@app.command("latency")
def latency_command(
    repeat: int = typer.Option(3, "--repeat", "-n", min=1, help="Warm runs of each request."),
    cold: bool = typer.Option(True, "--cold/--no-cold", help="Unload the model first and time one "
                              "cold request (includes loading the model)."),
) -> None:
    """Chat response time: everyday requests timed the way the chat runs them (first feedback,
    answer, tokens/s), each answer checked. Appends to evals/latency.jsonl."""
    _require_checkout("latency")
    from . import latency

    _ensure_server()
    latency.benchmark(repeat, cold=cold, guard=Guard(log=lambda text: console.print(
        f"[yellow]{text}[/yellow]")))


@app.command("auto")
def auto(
    goal: str = typer.Argument(..., help="End goal for Vega to work toward unattended."),
    project: Path = ProjectOption,
    branch: str = typer.Option(None, "--branch", help="Branch to create (default vega/<goal>-<time>)."),
    max_rounds: int = typer.Option(8, "--max-rounds", min=1, help="Maximum agent rounds."),
    check: str = typer.Option(None, "--check", help="Check command (default: detected; 'none' to disable)."),
    model: str = ModelOption,
) -> None:
    """Work toward a goal on a new Git branch, committing each round whose check passes."""
    _select_model(model)
    autopilot_command(goal, _resolve_project(project), branch, max_rounds, check)


@server_app.command("start")
def server_start() -> None:
    """Start the local Ollama server in the background."""
    try:
        console.print(server.start())
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc


@server_app.command("stop")
def server_stop() -> None:
    """Stop the server if Vega started it."""
    console.print(server.stop())


@server_app.command("status")
def server_status() -> None:
    """Show server and model state."""
    table = Table(title="Vega server")
    table.add_column("Item")
    table.add_column("Value")
    for item, value in server.status_rows():
        table.add_row(item, value)
    console.print(table)


@server_app.command("build")
def server_build() -> None:
    """Rebuild the default `vega` model from its active profile in models.toml."""
    try:
        code = server.build_model()
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if code:
        raise typer.Exit(code)
    console.print(f"[green]Model {server.model_name()!r} is ready.[/green]")


@model_app.command("list")
def model_list(
    notes: bool = typer.Option(False, "--notes", help="Also print each model's description."),
) -> None:
    """Profiles, installed Ollama models and the catalogue: size on disk, whether each fits this
    machine's RAM, which is active and default, and the latest eval scores where any exist."""
    from . import model_store

    store = model_store.inventory()
    if store.error:
        console.print(f"[yellow]{store.error}[/yellow]", highlight=False)
    _machine_line(store)
    table = Table(title="Vega models", title_justify="left")
    compact = console.width < 110  # 80-column terminals: one scores column, no Ollama tag
    # No scores anywhere (the public release ships none): no columns of dashes, no footer for them
    scored = any(row.scores for row in store.rows)
    columns = ((("Model", "left"), ("Size", "right"), ("RAM", "left"), ("On disk", "left"),
                ("Scores", "left")) if compact else
               (("Model", "left"), ("Ollama model", "left"), ("Size", "right"), ("RAM", "left"),
                ("On disk", "left"), ("fix", "right"), ("goal", "right"), ("HE+", "right"),
                ("MBPP+", "right")))
    if not scored:
        columns = columns[:4] if compact else columns[:5]
    for column, justify in columns:
        table.add_column(column, justify=justify, no_wrap=column not in {"Ollama model", "Scores", "Model"})
    for row in store.rows:
        marks = [label for label, on in (("active", row.active), ("default", row.default),
                                         ("shipped", row.shipped_default),
                                         ("over 10B", row.over_limit)) if on]
        label = row.name + (f" ({', '.join(marks)})" if marks else "")
        style = "bold" if row.active else ""
        size = "—" if row.size_gb is None else \
            f"{'~' if row.size_estimated else ''}{row.size_gb:.1f} GB"
        scores = [row.scores.get(s, "—") for s in ("fixes", "goals", "humaneval40", "mbpp40")]
        if compact:
            summary = " · ".join(f"{name} {value}" for name, value in zip(("fix", "goal", "HE+", "MBPP+"), scores)
                                 if value != "—") or "—"
            table.add_row(Text(label, style=style), size, _fit_text(row.fit),
                          _disk_state(row, store.ollama_up), *([summary] if scored else []))
        else:
            table.add_row(Text(label, style=style), row.tag, size, _fit_text(row.fit),
                          _disk_state(row, store.ollama_up), *(scores if scored else []))
    console.print(table)
    legend = ("Scores: the catalogue's measured results (two seeds summed for goals), else the "
              "latest run in evals/runs.jsonl. " if scored else "")
    console.print(f"[dim]RAM: {model_store.FIT_RULE}. {legend}"
                  "On disk: built = vega-<profile> exists. ~ = estimated from the parameter count.[/dim]", highlight=False)
    if notes:
        for row in store.rows:
            if row.note:
                console.print(f"  [bold]{row.name}[/bold]: {row.note}", highlight=False)
    if not store.ollama_up:
        console.print("[dim]Ollama is not running; downloads unknown (vega server start).[/dim]")


def _machine_line(store) -> None:  # store: model_store.Inventory
    from . import model_store

    ram = f"{store.total_ram_gb:.0f} GB RAM" if store.total_ram_gb else "RAM unknown"
    disk = (f"{store.free_disk_gb:.1f} GB free disk" if store.free_disk_gb is not None
            else "free disk unknown")
    console.print(f"This machine: {ram}, {disk} where Ollama keeps models "
                  f"({model_store.models_dir()}).", highlight=False)


def _fit_text(verdict: str) -> Text:
    return Text(verdict, style={"fits": "green", "tight": "yellow", "won't fit": "red"}.get(
        verdict, "dim"))


def _disk_state(row, ollama_up: bool) -> str:  # row: model_store.ModelRow
    if row.endpoint:
        return "endpoint"
    if not ollama_up:
        return "?"
    if row.built:
        return "built"
    return "downloaded" if row.downloaded else "not downloaded"


@model_app.command("use")
def model_use(profile: str = typer.Argument(..., help="Profile name from models.toml.")) -> None:
    """Rebuild the default `vega` model from a profile (pulls the base model if needed). An
    endpoint profile is only recorded as active: `vega` then talks to its server."""
    try:
        code = models.use(profile)
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if code:
        raise typer.Exit(code)
    chosen = models.get(profile)
    where = f" on {chosen.endpoint}; no Ollama build" if chosen.is_endpoint else ""
    console.print(f"[green]`vega` now runs profile {profile!r} ({chosen.base}{where}).[/green]")


@model_app.command("build")
def model_build(profile: str = typer.Argument(..., help="Profile name from models.toml.")) -> None:
    """Build vega-<profile> alongside the default, for comparison with --model."""
    try:
        chosen = models.get(profile)
        if chosen.is_endpoint:
            console.print(f"Nothing to build: {profile!r} runs on {chosen.endpoint}. Use it with "
                          f"--model {profile}.", highlight=False)
            return
        code = models.build(chosen, models.ollama_name(profile))
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if code:
        raise typer.Exit(code)
    console.print(f"[green]Built {models.ollama_name(profile)}; use it with --model {profile}.[/green]")


RequestFieldOption = typer.Option(
    None, "--request", help="Extra request field key=value, sent with every request (repeatable), "
    "e.g. reasoning_effort=none.")


@model_app.command("add")
def model_add(
    name: str = typer.Argument(..., help="New profile name (lowercase, digits, - and _)."),
    base: str = typer.Option(..., "--base", help="Ollama model to build from (e.g. "
                             "granite4.1:8b), or with --endpoint the model that server serves."),
    num_ctx: int = typer.Option(None, "--num-ctx", help="Context window in tokens (default 8192)."),
    description: str = typer.Option("", "--description", help="One line shown by `vega model list`."),
    endpoint: str = typer.Option(
        "", "--endpoint", help="Another local OpenAI-compatible server instead of Ollama, e.g. "
        "http://127.0.0.1:1234/v1 (LM Studio), :8080/v1 (llama.cpp), :8080/v1 (mlx_lm.server)."),
    api_model: str = typer.Option("", "--api-model", help="With --endpoint: the model name that "
                                  "server expects, if not the --base name."),
    request: list[str] = RequestFieldOption,
    allow_remote: bool = typer.Option(False, "--allow-remote", help="Allow an endpoint that is not "
                                      "on this machine or the local network (Vega is local-first)."),
) -> None:
    """Add a model profile to models.toml (other profiles and comments are left as they are)."""
    try:
        options = dict(models.parse_request_option(item) for item in request or [])
        profile = models.add_profile(
            name, base=base, num_ctx=num_ctx, description=description, endpoint=endpoint,
            api_model=api_model, request=options, allow_remote=allow_remote)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    where = (f"served by {profile.endpoint} as {profile.served_name!r}" if profile.is_endpoint
             else f"built from Ollama model {profile.base!r} as {models.ollama_name(name)}")
    console.print(f"[green]Added profile {name!r}[/green] to {models.PROFILES_FILE} ({where}).",
                  highlight=False)
    console.print("Next:")
    if not profile.is_endpoint:
        console.print(f"  ollama pull {profile.base}      # if it is not downloaded yet",
                      highlight=False)
    console.print(f"  vega model check {name}      # does it work through Vega? (~1-2 min)",
                  highlight=False)
    console.print(f"  vega eval --model {name}     # score it without changing the default",
                  highlight=False)
    console.print(f"  vega model use {name}        # make it the default `vega` model",
                  highlight=False)


@model_app.command("remove")
def model_remove(name: str = typer.Argument(..., help="Profile name from models.toml.")) -> None:
    """Remove a profile from models.toml. Model weights are never deleted (the command to do that
    is printed instead)."""
    try:
        profile = models.remove_profile(name)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"Removed profile {name!r} from {models.PROFILES_FILE}.", highlight=False)
    if profile.is_endpoint:
        console.print(f"Its model stays on {profile.endpoint}; remove it there if unused.",
                      highlight=False)
    else:
        console.print("Model files were not deleted. To free the disk space:", highlight=False)
        console.print(f"  ollama rm {models.ollama_name(name)}", highlight=False)
        console.print(f"  ollama rm {profile.base}      # only if no other profile uses it",
                      highlight=False)


@model_app.command("check")
def model_check_command(
    model: str = typer.Argument(..., help="Profile name (or an Ollama model name)."),
    guard: bool = typer.Option(True, "--guard/--no-guard", help="Pause or stop between items when "
                               "memory, swap or disk run low."),
) -> None:
    """Check a model works through Vega's interface: chat, thinking tags, the question and direct
    pipelines, a tool call in the agent loop, truncation continuation and request options (about
    a minute on a 7-8B model). Exits non-zero when an item fails."""
    from . import model_check

    _select_model(model)
    resolved = server.model_name()
    route = models.connection(resolved)
    console.print(f"Checking {resolved} at {route.base_url}"
                  + (f" (sent as {route.model!r})" if route.model != resolved else "")
                  + (" — endpoint profile: Ollama steps skipped." if route.is_endpoint else "."),
                  highlight=False)
    watcher = Guard(log=lambda text: console.print(f"[yellow]{text}[/yellow]")) if guard else None

    def show(item: model_check.Item) -> None:
        mark = {True: "[green]pass[/green]", False: "[red]FAIL[/red]", None: "[dim]skip[/dim]"}
        console.print(f"  {item.name:<12} {mark[item.passed]}  {item.seconds:5.1f}s  "
                      f"{item.detail}", highlight=False)

    report = model_check.run(resolved, before_item=watcher.before_attempt if watcher else None,
                             on_item=show)
    failed = [item.name for item in report.items if item.passed is False]
    first = "—" if report.first_reply is None else f"{report.first_reply:.1f}s"
    total = sum(item.seconds for item in report.items)
    console.print(f"First reply: {first} (time to the first streamed token; includes loading the "
                  f"model if it was not loaded) · items took {total:.0f}s in all", highlight=False)
    if report.memory:
        console.print(f"Memory: {report.memory}", highlight=False)
    if report.stopped:
        console.print(f"[yellow]Stopped early: {report.stopped}[/yellow]")
    if report.ok:
        console.print(f"[green]{resolved} works through Vega's interface.[/green]")
        return
    console.print(f"[red]Failed: {', '.join(failed) or 'stopped before finishing'}.[/red]")
    raise typer.Exit(1)


@model_app.command("pull")
def model_pull(
    target: str = typer.Argument(..., help="Profile, catalogue name (vega model list) or Ollama "
                                 "model name."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask before downloading."),
    size_gb: float = typer.Option(None, "--size-gb", help="Download size when the registry and "
                                  "the catalogue do not know it."),
    build: bool = typer.Option(True, "--build/--no-build", help="Build vega-<profile> after."),
) -> None:
    """Download a model through Ollama, then build vega-<profile>. Shows the size and free disk
    first and refuses when less than 12 GB would stay free (VEGA_GUARD_MIN_DISK_GB)."""
    from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn

    from . import model_store

    _ensure_ollama()
    try:
        plan = model_store.plan_pull(
            target, size_lookup=(lambda _tag: size_gb) if size_gb else None)
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if plan.already:
        console.print(f"{plan.tag} is already downloaded ({plan.size_gb or 0:.1f} GB).",
                      highlight=False)
    else:
        size = "unknown size" if plan.size_gb is None else \
            f"{plan.size_gb:.1f} GB ({plan.size_source})"
        console.print(f"Download {plan.tag}: {size}. Free disk {plan.free_gb:.1f} GB now, "
                      f"{plan.free_after_gb:.1f} GB after (Vega keeps at least "
                      f"{model_store.min_free_gb():.0f} GB free).", highlight=False)
    for warning in plan.warnings:
        console.print(f"[yellow]{warning}[/yellow]", highlight=False)
    if plan.refusal:
        console.print(f"[red]Refused:[/red] {plan.refusal}", highlight=False)
        raise typer.Exit(1)
    if not plan.already and not yes and not typer.confirm(f"Download {plan.tag}?", default=False):
        console.print("Nothing downloaded.")
        raise typer.Exit(1)
    if not plan.already:
        columns = (TextColumn("{task.description}"), BarColumn(), DownloadColumn(),
                   TimeRemainingColumn())
        with Progress(*columns, console=console, transient=False) as progress:
            task = progress.add_task(f"pulling {plan.tag}", total=None)

            def show(event: model_store.Progress) -> None:
                progress.update(task, completed=event.completed, total=event.total or None,
                                description=event.status[:40] or f"pulling {plan.tag}")

            try:
                model_store.pull(plan, show)
            except model_store.OllamaError as exc:
                console.print(f"[red]Download failed:[/red] {exc}", highlight=False)
                raise typer.Exit(1) from exc
        console.print(f"[green]Downloaded {plan.tag}.[/green]", highlight=False)
    if not build or plan.profile is None:
        return
    try:
        built = model_store.finish_pull(plan)
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]", highlight=False)
        raise typer.Exit(1) from exc
    console.print(f"[green]Built {built}[/green]: try it with --model {plan.profile}, make it the "
                  f"default with `vega model use {plan.profile}`.", highlight=False)


@model_app.command("rm")
def model_rm(
    target: str = typer.Argument(..., help="Profile name or Ollama model name."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask before deleting."),
) -> None:
    """Delete a model's downloads from Ollama (a profile: vega-<profile> and its base model unless
    another model still uses it). Shows the space it frees; never the active or default model.
    Blobs other models share stay on disk (Ollama only deletes unreferenced ones). The profile
    stays in models.toml (`vega model remove` drops it)."""
    from . import model_store

    _ensure_ollama()
    try:
        plan = model_store.plan_remove(target)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    for note in plan.kept:
        console.print(f"Keeping {note}.", highlight=False)
    if plan.refusal:
        console.print(f"[red]Refused:[/red] {plan.refusal}", highlight=False)
        raise typer.Exit(1)
    freed = plan.freed
    amount = "?" if freed is None else str(freed)
    console.print(f"Delete {', '.join(plan.names)}: frees {amount}.", highlight=False)
    if plan.affects:
        console.print(f"Profiles on the same weights will need `vega model pull` again: "
                      f"{', '.join(plan.affects)}.", highlight=False)
    if freed and freed.shared_with:
        console.print(f"Weights shared with {', '.join(freed.shared_with)} stay on disk.",
                      highlight=False)
    if not yes and not typer.confirm("Delete?", default=False):
        console.print("Nothing deleted.")
        raise typer.Exit(1)
    try:
        removed = model_store.remove(plan)
    except (model_store.OllamaError, ValueError) as exc:
        console.print(f"[red]Delete failed:[/red] {exc}", highlight=False)
        raise typer.Exit(1) from exc
    console.print(f"[green]Deleted {', '.join(removed)}.[/green] Free disk now "
                  f"{model_store.free_disk_gb():.1f} GB.", highlight=False)


@model_app.command("recommend")
def model_recommend() -> None:
    """The best catalogue model for this machine's RAM, from Vega's measured scores where any
    exist, else the shipped default (or the smallest model that fits)."""
    from . import model_store, resources

    ram = resources.total_memory_gb()
    choice = model_store.recommend(ram, tags=model_store.installed())
    if choice is None:
        console.print(f"[red]No catalogue model fits {ram or 0:.0f} GB of RAM.[/red]")
        raise typer.Exit(1)
    entry = choice.entry
    state = "downloaded" if choice.installed else f"{entry.download_gb:.1f} GB to download"
    console.print(f"[bold]{entry.name}[/bold] ({entry.tag}, {entry.params_b:g}B, {entry.licence}): "
                  f"{choice.fit}, {state}.", highlight=False)
    console.print(choice.reason, highlight=False)
    _, profiles = models.load()
    if not choice.installed:
        console.print(f"  vega model pull {entry.name}", highlight=False)
    console.print(f"  vega model use {entry.name}" + ("" if entry.name in profiles else
                  "      # after the pull adds its profile"), highlight=False)
    if choice.others:
        # "not measured" only means something next to models that were (none in the public release)
        any_measured = choice.entry.measured or any(o.measured for o, _ in choice.others)
        console.print("Next: " + ", ".join(
            f"{other.name} ({verdict}, {other.download_gb:.1f} GB"
            + ("" if other.measured or not any_measured else ", not measured") + ")"
            for other, verdict in choice.others), highlight=False)


QueueOption = typer.Option(None, "--queue", help="Queue the standard comparison for this "
                           "candidate (repeatable); runs nothing.")


@model_app.command("watch")
def model_watch(
    online: bool = typer.Option(False, "--online", help="Also ask ollama.com's library for new "
                                "tags of: " + ", ".join(WATCH_FAMILIES) + " (under 10B)."),
    queue: list[str] = QueueOption,
) -> None:
    """New models to try. Offline by default: catalogue models without a profile and models in
    Ollama's local list that no profile uses. --online also checks the Ollama library."""
    from . import model_store

    candidates = model_store.watch_offline()
    errors: list[str] = []
    if online:
        found, errors = model_store.watch_online()
        candidates += found
    if not candidates:
        console.print("No candidates: every catalogue and installed model has a profile.")
    table = Table(title="Candidates", title_justify="left")
    columns = ("Model", "From", "Params", "Size", *(("New",) if online else ()), "Profile name")
    for column in columns:
        table.add_column(column, no_wrap=True)
    for item in candidates:
        cells = [item.tag, item.source, "—" if item.params_b is None else f"{item.params_b:g}B",
                 "—" if item.size_gb is None else f"{item.size_gb:.1f} GB"]
        if online:
            cells.append("new since the last check" if item.new and item.source == "online"
                         else "")
        table.add_row(*cells, item.name)
    if candidates:
        console.print(table)
        console.print("Queue the standard comparison (HumanEval+ 40, MBPP+ 40, goals, fixes vs "
                      "the default): vega model queue <model>", highlight=False)
    for error in errors:
        console.print(f"[yellow]{error}[/yellow]", highlight=False)
    for target in queue or []:
        _queue(target)


@model_app.command("queue")
def model_queue(
    target: str = typer.Argument(..., help="Profile, catalogue name or Ollama model name."),
) -> None:
    """Write the standard comparison against the default (HumanEval+ 40, MBPP+ 40, goals, fixes)
    to evals/experiments.toml. Runs nothing: `vega lab run <ids>` runs them (attended)."""
    _require_checkout("model queue")
    _queue(target)


def _queue(target: str) -> None:
    from . import model_store

    try:
        profile, written, skipped = model_store.queue_comparison(target)
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if written:
        console.print(f"Queued for profile {profile!r}: {', '.join(written)}", highlight=False)
        console.print(f"  vega lab run {' '.join(written)}     # attended; pull the model first",
                      highlight=False)
    if skipped:
        console.print(f"Already queued: {', '.join(skipped)}", highlight=False)


def _ensure_ollama() -> None:
    if not server.is_running():
        console.print("[red]Ollama is not running.[/red] Start it with `vega server start`.")
        raise typer.Exit(1)


@app.command("models")
def models_command(
    notes: bool = typer.Option(False, "--notes", help="Also print each model's description."),
) -> None:
    """Same as `vega model list`."""
    model_list(notes=notes)


@app.command("features")
def list_features() -> None:
    """List feature switches and whether each is on (VEGA_FEATURES_OFF turns them off,
    VEGA_FEATURES_ON turns opt-in ones on)."""
    try:
        off = features.disabled()
        opted_in = features.enabled()
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    table = Table(title="Vega feature switches")
    for column in ("Feature", "State", "What it controls"):
        table.add_column(column)
    for name, description in features.FEATURES.items():
        state = "[red]off[/red]" if name in off else "[green]on[/green]"
        table.add_row(name, state, description)
    for name, description in features.OPT_IN.items():
        state = "[green]on[/green]" if name in opted_in else "[dim]off (opt-in)[/dim]"
        table.add_row(name, state, description)
    console.print(table)
    console.print(f"[dim]Turn off with {features.ENV}=a,b or `vega eval --off a,b`; turn opt-in "
                  f"features on with {features.ENV_ON}=a,b or compare them with "
                  "`vega eval --ab-on a,b`.[/dim]")


@context_app.command("show")
def context_show(
    project: Path = ProjectOption,
    full: bool = typer.Option(False, "--full", help="Also print the exact text the model gets."),
) -> None:
    """What a chat session in this project loads into the system prompt, with token counts."""
    from . import project_context as pc

    project = _resolve_project(project)
    if not features.on("project_context"):
        console.print("[yellow]The project_context feature is off (VEGA_FEATURES_OFF); chat "
                      "sessions load only .vega/notes.md.[/yellow]")
    text, pieces = pc.section(project, include_notes=features.on("remember"))
    if not pieces:
        console.print("Nothing to load: no VEGA.md, AGENTS.md, CLAUDE.md, session digests or "
                      "notes. `vega context init` drafts VEGA.md.")
        return
    budget = pc.token_budget()
    console.print(f"Project: {project}", soft_wrap=True)
    table = Table(title=f"Context files (budget {budget:,} tokens, {pc.TOKENS_ENV})")
    for column in ("#", "Piece", "File", "Tokens", "Loaded", "Status"):
        table.add_column(column, justify="right" if column in {"#", "Tokens", "Loaded"} else "left")
    for number, piece in enumerate(pieces, 1):
        style = {"loaded": "green", "truncated": "yellow", "dropped": "red"}[piece.status]
        table.add_row(str(number), piece.label, piece.source, f"{piece.tokens:,}",
                      f"{piece.loaded_tokens:,}", f"[{style}]{piece.status}[/{style}]")
    console.print(table)
    loaded = sum(piece.loaded_tokens for piece in pieces)
    console.print(f"File contents loaded: {loaded:,} tokens · whole section with its framing: "
                  f"~{pc.estimate_tokens(text):,} of {budget:,} tokens (estimate: characters / "
                  f"{pc.CHARS_PER_TOKEN})")
    if full:
        console.print(text.strip(), markup=False, highlight=False)


@context_app.command("init")
def context_init(
    project: Path = ProjectOption,
    force: bool = typer.Option(False, "--force", help="Replace the generated block even if it "
                               "was edited."),
) -> None:
    """Create VEGA.md, or refresh its generated block (between the vega:auto markers)."""
    from . import project_context as pc

    project = _resolve_project(project)
    result = pc.ensure_vega_md(project, explicit=True, force=force)
    messages = {
        "created": f"[green]Created {pc.VEGA_FILE}[/green] ({result.detail}). Edit the "
                   "Conventions section freely.",
        "updated": f"[green]Refreshed the generated block in {pc.VEGA_FILE}.[/green]",
        "unchanged": f"{pc.VEGA_FILE} is up to date.",
        "kept-user-edits": f"[yellow]Left {pc.VEGA_FILE} as is: {result.detail}.[/yellow]",
        "no-markers": f"[yellow]Left {pc.VEGA_FILE} as is: {result.detail}.[/yellow]",
        "skipped": f"[yellow]Skipped: {result.detail}.[/yellow]",
    }
    console.print(messages[result.action])
    if result.action in {"created", "updated", "unchanged"}:
        console.print((project / pc.VEGA_FILE).read_text(encoding="utf-8"), markup=False,
                      highlight=False)


@context_app.command("forget")
def context_forget(
    project: Path = ProjectOption,
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation."),
) -> None:
    """Delete the session digests (.vega/sessions.md). VEGA.md and notes are kept."""
    from . import project_context as pc

    project = _resolve_project(project)
    count = len(pc.read_digests(project))
    if not count:
        console.print(f"No session digests in {pc.DIGESTS_PATH}.")
        return
    if not yes and not typer.confirm(f"Delete {count} session digest(s) in {pc.DIGESTS_PATH}?",
                                     default=False):
        console.print("Kept them.")
        return
    pc.forget_digests(project)
    console.print(f"Deleted {count} session digest(s).")


@lab_app.command("list")
def lab_list() -> None:
    """Planned experiments with their status and latest result."""
    from . import lab

    lab.list_experiments()


@lab_app.command("run")
def lab_run(
    ids: list[str] = typer.Argument(None, help="Experiment ids (default: every pending one)."),
    again: bool = typer.Option(False, "--again", help="Also rerun experiments that have results."),
    screen: bool = typer.Option(
        False, "--screen", help="Screen each experiment on its suite's discriminating cases "
        "(recorded as a screen; the experiment stays pending until a full run).",
    ),
) -> None:
    """Run experiments one after another with the resource guard on. Ctrl-C stops the queue."""
    from . import lab

    _ensure_server()
    try:
        lab.run_queue(list(ids or []), include_done=again, screen=screen)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        server.unload()


@lab_app.command("report")
def lab_report() -> None:
    """Write docs/scoreboard.md from the recorded runs."""
    from . import lab

    lab.write_scoreboard()


@lab_app.command("tag")
def lab_tag(
    run: str = typer.Argument(..., help="Run number (after `--`, -1 = last) or timestamp."),
    experiment: str = typer.Argument(..., help="Experiment id from evals/experiments.toml."),
) -> None:
    """Attach an experiment id to a recorded run."""
    from . import lab

    try:
        lab.tag(run, experiment)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@web_app.command("status")
def web_status(project: Path = ProjectOption) -> None:
    """Show the web research settings in effect for a project (no network request)."""
    from . import features, settings

    project = _resolve_project(project)
    backend = settings.get("web.backend", project)
    console.print(f"Feature web_research: {'on' if features.on('web_research') else 'off'}")
    console.print(f"Backend (web.backend): {backend}")
    console.print(f"SearXNG URL (web.searxng_url): {settings.get('web.searxng_url', project)}")
    console.print(f"Permission (web.permission): {settings.get('web.permission', project)} "
                  "(ask = once per session)")
    console.print(f"Settings: {settings.user_file()} · project: {settings.project_file(project)}")
    if backend == "none":
        console.print("Web research is off: no backend. See README 'Web research', then run "
                      "[bold]vega web set web.backend searxng[/bold].")


@web_app.command("set")
def web_set(
    key: str = typer.Argument(..., help="web.backend, web.searxng_url, web.permission or "
                              "web.official_domains (comma-separated)."),
    value: str = typer.Argument(..., help="The new value."),
    for_project: bool = typer.Option(False, "--this-project", help="Store it for this project only."),
    project: Path = ProjectOption,
) -> None:
    """Change a web research setting."""
    from . import settings

    if not key.startswith("web."):
        key = "web." + key
    if key not in settings.DEFAULTS:
        raise typer.BadParameter(f"Unknown setting {key}; known: "
                                 + ", ".join(k for k in settings.DEFAULTS if k.startswith("web.")))
    try:
        path = settings.set_value(key, value, _resolve_project(project) if for_project else None)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"{key} = {settings.get(key, _resolve_project(project) if for_project else None)}"
                  f" · saved in {path}")


@web_app.command("search")
def web_search_command(
    query: str = typer.Argument(..., help="A short question in plain words."),
    project: Path = ProjectOption,
) -> None:
    """Search with the configured backend, as the web_search tool would (you asked: no prompt)."""
    from . import web

    project = _resolve_project(project)
    session = web.WebSession(project, permission="allow", cache=False)
    console.print(session.call("web_search", {"query": query}), markup=False, highlight=False)


@adapters_app.command("list")
def adapters_list() -> None:
    """Stored site adapters: site, API domains, state, dates."""
    from . import adapter_builder

    found = adapter_builder.load_all(include_unusable=True)
    if not found:
        console.print(f"No site adapters yet ({adapter_builder.adapters_dir()}). Vega offers to "
                      "build one when a site blocks reading and has an official API.")
        return
    for adapter in found:
        meta = adapter.meta
        console.print(f"[bold]{adapter.site}[/bold]  {adapter_builder.state_of(adapter)}  "
                      f"API {', '.join(adapter.domains)}  created {meta.get('created', '?')}  "
                      f"verified {meta.get('verified', '?')}  model {meta.get('model', '?')}",
                      highlight=False)
    console.print(f"Folder: {adapter_builder.adapters_dir()}")


@adapters_app.command("show")
def adapters_show(site: str = typer.Argument(..., help="The site (e.g. example.com) or a URL.")) -> None:
    """An adapter's provenance and code."""
    from . import adapter_builder

    adapter = adapter_builder.find(site, include_unusable=True)
    if adapter is None:
        raise typer.BadParameter(f"No adapter for {site}.")
    console.print(json.dumps(adapter.meta, indent=2), markup=False, highlight=False)
    console.print(f"State: {adapter_builder.state_of(adapter)}\n")
    console.print(adapter.code, markup=False, highlight=False)


@adapters_app.command("remove")
def adapters_remove(site: str = typer.Argument(..., help="The site (e.g. example.com).")) -> None:
    """Delete an adapter (its code and provenance)."""
    from . import adapter_builder

    if not adapter_builder.remove(site):
        raise typer.BadParameter(f"No adapter for {site}.")
    console.print(f"Removed the adapter for {site}.")


@adapters_app.command("verify")
def adapters_verify(site: str = typer.Argument(..., help="The site (e.g. example.com).")) -> None:
    """Run an adapter on its verification pages again (uses the network: the site's API)."""
    from . import adapter_builder, web

    adapter = adapter_builder.find(site, include_unusable=True)
    if adapter is None:
        raise typer.BadParameter(f"No adapter for {site}.")
    if adapter_builder.state_of(adapter) not in {"ok", "failing"}:
        raise typer.BadParameter(f"The adapter for {site} is {adapter_builder.state_of(adapter)}; "
                                 "remove it and let Vega build it again.")
    ok, feedback = adapter_builder.reverify(adapter, web.SearxngBackend().api_get)
    console.print(("[green]Verified.[/green] " if ok else "[red]Failed.[/red] ") + "", end="")
    console.print(feedback, markup=False, highlight=False)
    if not ok:
        raise typer.Exit(1)


@learn_app.command("add")
def learn_add(
    sources: list[str] = typer.Argument(..., help="URLs, files (.pdf .md .txt .rst .html) or folders."),
    title: str = typer.Option(None, "--title", "-t", help="Title to cite (one source only)."),
) -> None:
    """Learn material: fetch or read it, split it into sections and add it to the library."""
    from . import knowledge

    failed = False
    for target in sources:
        try:
            added = knowledge.learn(target, title if len(sources) == 1 else None)
        except knowledge.LearnError as exc:
            console.print(f"[red]{exc}[/red]")
            failed = True
            continue
        for source in added:
            console.print(f"Learned [bold]{source.title}[/bold] ({source.kind}): {source.chunks} "
                          f"sections, {source.chars:,} characters · id {source.id}")
    console.print(f"Library: {knowledge.library_dir()}")
    if failed:
        raise typer.Exit(1)


@learn_app.command("list")
def learn_list() -> None:
    """What Vega has learned."""
    from . import knowledge

    found = knowledge.sources()
    if not found:
        console.print("Nothing learned yet. Add material with: vega learn add <url | file | folder>")
        return
    table = Table(title=f"Knowledge library · {knowledge.library_dir()}")
    for column in ("id", "title", "kind", "sections", "characters", "added", "from"):
        table.add_column(column, justify="right" if column in {"sections", "characters"} else "left")
    for s in found:
        kind = f"{s.kind} ({s.trust} trust)" if s.trust else s.kind  # web research pages
        table.add_row(s.id, s.title, kind, str(s.chunks), f"{s.chars:,}", s.added, s.origin)
    console.print(table)


@learn_app.command("search")
def learn_search(query: str = typer.Argument(..., help="Words to look up.")) -> None:
    """Search the library the way Vega's search_knowledge tool does."""
    from . import knowledge

    console.print(knowledge.search_text(query), markup=False)


@learn_app.command("forget")
def learn_forget(source_id: str = typer.Argument(..., help="Source id (or its start) from vega learn list.")) -> None:
    """Remove one source from the library."""
    from . import knowledge

    removed = knowledge.forget(source_id)
    if removed is None:
        console.print(f"[red]No learned source with id {source_id}.[/red]")
        raise typer.Exit(1)
    console.print(f"Forgot {removed.title} ({removed.chunks} sections).")


@lab_app.command("pause")
def lab_pause() -> None:
    """Pause the running eval or experiment before its next attempt (nothing is lost)."""
    from . import evals

    path = evals.control_file("pause")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("paused\n", encoding="utf-8")
    console.print("Paused: the run waits before its next attempt. vega lab resume continues; "
                  "vega lab stop ends it and keeps the results so far.")


@lab_app.command("resume")
def lab_resume() -> None:
    """Resume a paused eval or experiment."""
    from . import evals

    evals.control_file("pause").unlink(missing_ok=True)
    console.print("Resumed.")


@lab_app.command("stop")
def lab_stop() -> None:
    """End the running eval or experiment after its current attempt, saving the results so far."""
    from . import evals

    path = evals.control_file("stop")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("stop\n", encoding="utf-8")
    console.print("Stopping after the current attempt; finished cases are recorded.")


@debug_app.command("watch")
def debug_watch(
    log: Path = typer.Argument(None, help="A log file; default: follow the newest, switching to newer ones."),
    full: bool = typer.Option(False, "--full", help="Show whole messages, including the system prompt."),
    once: bool = typer.Option(False, "--once", help="Print the log and exit instead of following it."),
) -> None:
    """Follow a debug log live (run in a second terminal next to `vega --debug`)."""
    from . import debuglog

    try:
        debuglog.watch(log, full=full, follow=not once)
    except KeyboardInterrupt:
        pass


@debug_app.command("list")
def debug_list(limit: int = typer.Option(15, "--limit", "-n")) -> None:
    """The newest debug logs."""
    from . import debuglog

    logs = sorted(debuglog.folder().glob("*.jsonl"), reverse=True)[:limit]
    if not logs:
        console.print("No debug logs yet. Start Vega with `vega --debug` or VEGA_DEBUG=1.")
        return
    table = Table("log", "events", "size")
    for path in logs:
        with path.open(encoding="utf-8") as handle:
            events = sum(1 for _ in handle)
        table.add_row(path.name, str(events), f"{path.stat().st_size / 1024:.0f} KB")
    console.print(table)


def _loop_or_exit(name: str) -> LoopDef:
    from . import loops

    try:
        return loops.find(name)
    except LookupError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


def _print_findings(items: list[dict], *, show_loop: bool) -> None:
    from .loops import inbox

    if not items:
        console.print("[dim]Inbox empty.[/dim]")
        return
    for item in items:
        state = inbox.state_of(item)
        style = "bold" if state in inbox.NEEDS_YOU else "dim"
        where = ", ".join(f"{w.get('path')}" + (f":{w['line']}" if w.get("line") else "")
                          for w in item.get("where", [])[:3])
        loop = f"{item['loop']} · " if show_loop else ""
        console.print(f"[{style}]{state:<11}[/{style}] {item['title']}", markup=True,
                      highlight=False)
        evidence = item.get("evidence", {})
        seen = len(item.get("seen_again", []))
        console.print(f"            [dim]{loop}{item['id']} · commit {item['commit'][:8]} · "
                      f"{evidence.get('type')} ({item.get('confidence')}) · {where}"
                      + (f" · seen again {seen}x" if seen else "") + "[/dim]", highlight=False)
        if evidence.get("type") == "failing_test" and evidence.get("passes_at"):
            console.print(f"            [dim]fails at {evidence.get('fails_at', '')[:8]}, passes at "
                          f"{evidence['passes_at'][:8]} · {evidence.get('path', '')}[/dim]",
                          highlight=False)


@loops_app.callback()
def loops_main(ctx: typer.Context) -> None:
    """`vega loops` alone lists the loops."""
    if ctx.invoked_subcommand is None:
        loops_list()


@loops_app.command("list")
def loops_list() -> None:
    """One row per loop: state, status, trigger, last run."""
    from .loops import status

    rows, errors = status.rows()
    head = status.header(rows)
    console.print(f"[bold]Loops[/bold]  [dim]{head.text()}[/dim]", highlight=False)
    if not head.model_allowed:
        console.print(f"[dim]  model work held: {', '.join(head.reasons)}[/dim]", highlight=False)
    if not rows:
        console.print("[dim]No loops defined. `vega loops start` creates Bug Watcher for Vega's "
                      "repo.[/dim]")
    width = max((len(r.name) for r in rows), default=0) + 2
    for r in rows:
        status_style = "bold yellow" if r.needs_you else "dim"
        text = f"({r.status})"
        console.print(f" {r.symbol} {r.name:<{width}}[{status_style}]{text:<22}[/{status_style}]"
                      f"[dim]{r.when()}[/dim]", highlight=False)
        if r.error:
            console.print(f"   [red]{r.error}[/red]", highlight=False)
    for error in errors:
        console.print(f"[red]definition error:[/red] {error}", highlight=False)


@loops_app.command("show")
def loops_show(name: str = typer.Argument(..., help="Loop name, e.g. 'Bug Watcher'.")) -> None:
    """A loop's definition, last result and its inbox."""
    from .loops import inbox, state, status

    loop = _loop_or_exit(name)
    r = status.row(loop)
    info = state.load_state(loop.name)
    console.print(f"{r.symbol} [bold]{loop.name}[/bold] ({r.status}) · {r.when()}", highlight=False)
    console.print(f"[dim]  skill {loop.skill} · project {loop.project} · budget "
                  f"{loop.minutes_per_run:g} min/run, {loop.runs_per_day} runs/day · {loop.source}[/dim]",
                  highlight=False)
    if info.last_result:
        console.print(f"[dim]  last: {info.last_result}[/dim]", highlight=False)
    if info.pending_model:
        console.print(f"[dim]  model pass waiting for idle: "
                      f"{', '.join(c[:8] for c in info.pending_model)}[/dim]", highlight=False)
    _print_findings(inbox.items(loop.name), show_loop=False)


@loops_app.command("inbox")
def loops_inbox(
    all_items: bool = typer.Option(False, "--all", help="Include finished findings."),
) -> None:
    """Findings from every loop, newest first."""
    from .loops import inbox

    _print_findings(inbox.items(open_only=not all_items), show_loop=True)


@loops_app.command("start")
def loops_start(
    foreground: bool = typer.Option(False, "--foreground", help="Run in this terminal (Ctrl-C stops)."),
) -> None:
    """Start the loops daemon (creates the Bug Watcher definition the first time)."""
    from . import loops
    from .loops import scheduler, state

    created = loops.ensure_default()
    if created is not None:
        console.print(f"Created the Bug Watcher loop for Vega's repo: [bold]{created}[/bold]")
        console.print(created.read_text(encoding="utf-8"), markup=False, highlight=False)
    try:
        if foreground:
            scheduler.Daemon(log=lambda text: console.print(f"[dim]{time.strftime('%H:%M:%S')}[/dim] "
                                                            f"{text}", highlight=False)).serve()
            return
        pid = scheduler.start_background()
    except state.AlreadyRunning as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(1) from exc
    for _ in range(40):
        if state.daemon_pid() == pid:
            break
        time.sleep(0.25)
    else:
        console.print(f"[red]The daemon did not start; see {state.daemon_log()}[/red]")
        raise typer.Exit(1)
    console.print(f"Loops daemon started (pid {pid}); log {state.daemon_log()}. "
                  "`vega loops stop` ends it.")


@loops_app.command("stop")
def loops_stop() -> None:
    """Stop the loops daemon (it finishes the current step first)."""
    from .loops import scheduler

    console.print(scheduler.stop_daemon())


@loops_app.command("pause")
def loops_pause(name: str = typer.Argument(None, help="Loop name (default: every loop).")) -> None:
    """Pause one loop or all of them (e.g. during a video call); model work stops at its next step."""
    from .loops import state

    if name is not None:
        name = _loop_or_exit(name).name
    state.set_paused(name, True)
    console.print(f"Paused {name or 'every loop'}.")


@loops_app.command("resume")
def loops_resume(name: str = typer.Argument(None, help="Loop name (default: every loop).")) -> None:
    """Resume one loop, or clear every pause."""
    from .loops import state

    if name is not None:
        name = _loop_or_exit(name).name
        if state.ALL in state.paused_names():
            console.print("[yellow]Every loop is paused; `vega loops resume` with no name clears "
                          "that.[/yellow]")
    state.set_paused(name, False)
    console.print(f"Resumed {name or 'every loop'}.")


@loops_app.command("run")
def loops_run(
    name: str = typer.Argument(..., help="Loop name."),
    commit: str = typer.Option(None, "--commit", help="Commit to check (default: the branch head)."),
) -> None:
    """Run a loop now. Tests and lint always run; the model pass only if the machine is idle."""
    from .loops import scheduler

    loop = _loop_or_exit(name)
    try:
        job, report = scheduler.run_now(loop, commit=commit)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"{loop.name}: {job.reason}", highlight=False)
    if report is None:
        console.print("[yellow]Not run: this loop is already running a job.[/yellow]")
        raise typer.Exit(1)
    for note in report.notes:
        console.print(f"[dim]  {note}[/dim]", highlight=False)
    console.print(report.summary(), highlight=False)
    if report.debug_log:
        console.print(f"[dim]Debug log: {report.debug_log}[/dim]", highlight=False)
    if report.filed:
        _print_findings(report.filed, show_loop=False)
    if report.error:
        raise typer.Exit(1)
