"""Experiment queue and scoreboard: run planned A/Bs back to back (attended) and summarise them.

Experiments are declared in evals/experiments.toml. Each run is tagged with its experiment id
(VEGA_EXPERIMENT), so `vega lab list` and `vega lab report` can tie results back to hypotheses.
Runs here only while the user is present (no unattended mode on a personal machine).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from . import evals, external, features, models
from .resources import Guard
from .stats import rate_text

EXPERIMENTS_FILE = evals.PROJECT_ROOT / "evals" / "experiments.toml"
SCOREBOARD_FILE = evals.PROJECT_ROOT / "docs" / "scoreboard.md"
console = Console()


@dataclass
class Experiment:
    id: str
    hypothesis: str
    suite: str = "fixes"
    ab: list[str] = field(default_factory=list)  # features switched off for B
    ab_on: list[str] = field(default_factory=list)  # opt-in features switched on for B
    ab_model: str | None = None  # model profile (or Ollama model) for B
    ab_agent: str | None = None  # agent for B (e.g. "aider"; A stays `agent`)
    agent: str | None = None  # agent for the run (A in an A/B); default vega
    model: str | None = None  # model profile (or Ollama model) for both arms; default: current
    repeat: int = 1
    seed: int | None = 7
    early_stop: bool = True
    screen: bool = False  # run only the suite's discriminating cases (never a full result)
    env: dict[str, str] = field(default_factory=dict)  # extra settings for both arms


def load(path: Path | None = None) -> list[Experiment]:
    path = path or EXPERIMENTS_FILE
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    experiments = []
    for raw in data.get("experiment", []):
        experiments.append(Experiment(
            id=raw["id"], hypothesis=raw["hypothesis"], suite=raw.get("suite", "fixes"),
            ab=list(raw.get("ab", [])), ab_on=list(raw.get("ab_on", [])), ab_model=raw.get("ab_model"),
            ab_agent=raw.get("ab_agent"), agent=raw.get("agent"), model=raw.get("model"),
            repeat=int(raw.get("repeat", 1)),
            seed=raw.get("seed", 7), early_stop=bool(raw.get("early_stop", True)),
            screen=bool(raw.get("screen", False)),
            env={key: str(value) for key, value in raw.get("env", {}).items()},
        ))
    ids = [experiment.id for experiment in experiments]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate experiment ids: {', '.join(sorted(duplicates))}")
    return experiments


def results(runs: list[dict[str, Any]] | None = None, *, screens: bool = False
            ) -> dict[str, dict[str, Any]]:
    """Latest recorded full run per experiment id (latest screen instead with `screens`): a screen
    never counts as an experiment's result."""
    latest: dict[str, dict[str, Any]] = {}
    for run in runs if runs is not None else evals.load_runs():
        if run.get("experiment") and bool(run.get("screen")) == screens:
            latest[run["experiment"]] = run
    return latest


def verdict(run: dict[str, Any]) -> str:
    if run.get("mode") != "ab":
        text = rate_text(run.get("passed", 0), run.get("total", 0))
        return f"screen, not a full result: {text}" if run.get("screen") else text
    paired = run["paired"]
    low, high = paired["diff_ci"]
    threshold = evals.EARLY_STOP_P if run.get("interim", {}).get("stopped") else 0.05
    # B either switches features off (ab) or switches opt-in features on (ab_on).
    variants = run.get("variants", {})
    b_on = (variants.get("B", {}).get("features_on") or []) != (
        variants.get("A", {}).get("features_on") or [])
    b_label, a_label = ("B (feature on)", "A (feature off)") if b_on else (
        "B (feature off)", "A (feature on)")
    agents = [variants.get(name, {}).get("agent") or "vega" for name in ("A", "B")]
    if agents[0] != agents[1]:
        a_label, b_label = f"A ({agents[0]})", f"B ({agents[1]})"
    if paired["mcnemar_p"] < threshold and not low <= 0 <= high:
        winner = f"{b_label} better" if paired["b_only"] > paired["a_only"] else f"{a_label} better"
    else:
        winner = "no detectable difference"
    text = (f"{winner}: B−A {paired['diff_b_minus_a'] * 100:+.0f} pts "
            f"(CI {low * 100:+.0f}..{high * 100:+.0f}), p={paired['mcnemar_p']:.2f}, "
            f"{paired['pairs']} pairs")
    if run.get("stopped") == "futility":
        text += ", stopped for futility"
    return f"screen, not a full result: {text}" if run.get("screen") else text


def list_experiments(path: Path | None = None) -> None:
    done = results()
    screened = results(screens=True)
    table = Table(title="Vega experiments")
    for column in ("Id", "Suite", "B changes", "Status", "Result"):
        table.add_column(column, overflow="fold")
    for experiment in load(path):
        run = done.get(experiment.id)
        status = run["timestamp"][:16].replace("T", " ") if run else "pending"
        if not run and experiment.id in screened:
            run = screened[experiment.id]
            status = "screened " + run["timestamp"][:16].replace("T", " ")
        changes = [f"-{name}" for name in experiment.ab] + [f"+{name}" for name in experiment.ab_on]
        if experiment.ab_model:
            changes.append(f"model {experiment.ab_model}")
        if experiment.ab_agent:
            changes.append(f"agent {experiment.ab_agent}")
        table.add_row(experiment.id, experiment.suite, ", ".join(changes) or "—", status,
                      verdict(run) if run else experiment.hypothesis)
    console.print(table)


def run_queue(ids: list[str] | None = None, *, include_done: bool = False, path: Path | None = None,
              screen: bool = False) -> None:
    """Run the given (or all pending) experiments one after another; Ctrl-C stops the queue.
    `screen` (or an experiment's `screen = true`) runs only each suite's discriminating cases."""
    experiments = load(path)
    done = results()
    if ids:
        unknown = set(ids) - {experiment.id for experiment in experiments}
        if unknown:
            raise ValueError(f"Unknown experiment(s): {', '.join(sorted(unknown))}")
        queue = [experiment for experiment in experiments if experiment.id in ids]
    else:
        queue = [e for e in experiments if include_done or e.id not in done]
    if not queue:
        console.print("Nothing to run: every experiment has a recorded result.")
        return
    console.print(f"Queue: {', '.join(e.id for e in queue)} (Ctrl-C stops after the current attempt).")
    for experiment in queue:
        saved = {key: os.environ.get(key) for key in (
            *experiment.env, "VEGA_EXPERIMENT", "VEGA_SEED", "VEGA_MODEL", external.AGENT_ENV,
            features.ENV_ON)}
        os.environ.update(experiment.env)
        os.environ["VEGA_EXPERIMENT"] = experiment.id
        if experiment.seed is not None:
            os.environ["VEGA_SEED"] = str(experiment.seed)
        if experiment.agent:
            os.environ[external.AGENT_ENV] = experiment.agent
        if experiment.model:  # both arms; a profile's default opt-in features follow it
            os.environ["VEGA_MODEL"] = evals._resolve_model(experiment.model)
            models.apply_feature_defaults(os.environ["VEGA_MODEL"])
        console.rule(f"{experiment.id}: {experiment.hypothesis}")
        evals.install_stop_signal()
        try:
            guard = Guard(log=lambda text: console.print(f"[yellow]{text}[/yellow]"))
            if experiment.ab or experiment.ab_on or experiment.ab_model or experiment.ab_agent:
                record = evals.run_ab(experiment.suite, experiment.ab, repeat=experiment.repeat,
                                      guard=guard, early_stop=experiment.early_stop,
                                      b_on=experiment.ab_on, b_model=experiment.ab_model,
                                      b_agent=experiment.ab_agent,
                                      screen=screen or experiment.screen)
            else:
                record = evals.run_suite(experiment.suite, repeat=experiment.repeat, guard=guard,
                                         screen=screen or experiment.screen)
        except KeyboardInterrupt:
            console.print("[yellow]Queue stopped by user.[/yellow]")
            return
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        if record.get("stopped") and record["stopped"] != "futility":  # futility: the run is done
            console.print(f"[yellow]Queue stopped: {record['stopped']}[/yellow]")
            return
    write_scoreboard()


def tag(run_key: str, experiment_id: str, path: Path | None = None) -> None:
    """Attach an experiment id to an already recorded run (e.g. one started before tagging)."""
    import json

    if experiment_id not in {experiment.id for experiment in load()}:
        raise ValueError(f"Unknown experiment {experiment_id!r}; add it to experiments.toml first.")
    runs_path = path or evals.RUNS_FILE
    runs = evals.load_runs(runs_path)
    index = evals.select_run(runs, run_key)
    runs[index]["experiment"] = experiment_id
    runs_path.write_text("".join(json.dumps(run, ensure_ascii=False) + "\n" for run in runs),
                         encoding="utf-8")
    console.print(f"Run #{index + 1} tagged as {experiment_id}.")


def write_scoreboard(path: Path | None = None) -> Path:
    """docs/scoreboard.md: every experiment's latest verdict plus its failure-class shift."""
    path = path or SCOREBOARD_FILE
    done = results()
    lines = [
        "# Vega scoreboard",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `vega lab report` "
        "from `evals/runs.jsonl`. A/B rows: A = current settings, B = A with the experiment's "
        "features switched off (or opt-in features switched on); positive B − A means B is "
        "better.",
        "",
        "| Experiment | Hypothesis | Suite | Result | A median s | B median s | Failure types A → B |",
        "|---|---|---|---|---|---|---|",
    ]
    for experiment in load():
        run = done.get(experiment.id)
        if not run:
            lines.append(f"| {experiment.id} | {experiment.hypothesis} | {experiment.suite} | pending "
                         "| | | |")
            continue
        if run.get("mode") == "ab":
            a, b = (run["variants"][name]["summary"] for name in ("A", "B"))
            shift = (_classes(a) or "—") + " → " + (_classes(b) or "—")
            lines.append(
                f"| {experiment.id} | {experiment.hypothesis} | {experiment.suite} | {verdict(run)} | "
                f"{_num(a.get('median_seconds'))} | {_num(b.get('median_seconds'))} | {shift} |")
        else:
            summary = run.get("summary", {})
            lines.append(f"| {experiment.id} | {experiment.hypothesis} | {experiment.suite} | "
                         f"{verdict(run)} | {_num(summary.get('median_seconds'))} | | "
                         f"{_classes(summary)} |")
    lines += _profile_table(done)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"Wrote {evals._display_path(path)}")
    return path


def _profile_table(done: dict[str, dict[str, Any]]) -> list[str]:
    """Capability profile: each A/B experiment's pass counts per primary area, A → B."""
    from . import categories

    area_ids = list(categories.areas())
    rows = []
    for experiment in load():
        run = done.get(experiment.id)
        if not run or run.get("mode") != "ab" or "areas" not in run["variants"]["A"]:
            continue
        a, b = run["variants"]["A"]["areas"], run["variants"]["B"]["areas"]
        cells = []
        for area in area_ids:
            if area in a or area in b:
                pa, ta = a.get(area, [0, 0])
                pb, tb = b.get(area, [0, 0])
                cells.append(f"{pa}/{ta} → {pb}/{tb}")
            else:
                cells.append("—")
        rows.append(f"| {experiment.id} | " + " | ".join(cells) + " |")
    if not rows:
        return []
    header = "| Experiment | " + " | ".join(categories.name(area) for area in area_ids) + " |"
    return ["", "## Capability profile (A → B, passes per primary area)", "", header,
            "|---|" + "---|" * len(area_ids), *rows]


def _classes(summary: dict[str, Any]) -> str:
    return ", ".join(f"{name} {count}" for name, count in summary.get("failure_classes", {}).items())


def _num(value: float | None) -> str:
    return f"{value:.1f}" if isinstance(value, (int, float)) else ""
