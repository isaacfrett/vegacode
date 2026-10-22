"""External agents (Aider) through the eval harness: command, settings, scoring and A/B order.

No model runs here: a fake `aider` executable on PATH stands in for the real one."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from rich.console import Console

from vega import evals, external, features, lab

PYTEST = "python -m pytest -q -p no:cacheprovider"


def _sandbox_works() -> bool:
    if not evals.sandbox.available():
        return False
    probe = subprocess.run(["sandbox-exec", "-p", "(version 1)(allow default)", "true"],
                           capture_output=True, check=False)
    return probe.returncode == 0


needs_sandbox = pytest.mark.skipif(
    not _sandbox_works(), reason="working macOS sandbox-exec required")
posix_only = pytest.mark.skipif(os.name != "posix", reason="fake executable uses a shebang")

FAKE_AIDER = r'''#!{python}
"""Fake aider 0.86.2: logs how it was called, then edits like the real one would."""
import json, os, sys
from pathlib import Path

args = sys.argv[1:]
if args == ["--version"]:
    print("aider 0.86.2")
    sys.exit(0)

def value(flag):
    return args[args.index(flag) + 1] if flag in args else None

files = [args[i + 1] for i, a in enumerate(args) if a == "--file"]
task = Path(value("--message-file")).read_text()
settings = json.loads(Path(value("--model-settings-file")).read_text())
log = {{"argv": args, "cwd": os.getcwd(), "files": files, "task": task, "settings": settings,
       "git": Path(".git").is_dir(), "ollama": os.environ.get("OLLAMA_API_BASE")}}
Path(os.environ["FAKE_AIDER_LOG"]).write_text(json.dumps(log))
Path(".aider.tags.cache.v4").mkdir()  # Aider's own cache: must not count as an edit
mode = os.environ.get("FAKE_AIDER_MODE", "fix")
print("Model: ollama_chat/vega with whole edit format")
print("Tokens: 1.2k sent, 85 received.")
if mode == "fix":
    Path("shop/price.py").write_text(
        "def total(items, discount=0.0):\n    return sum(items) * (1 - discount)\n")
    print("Applied edit to shop/price.py")
    print("Tokens: 12k sent, 1.5k received.")
elif mode == "edit_tests":
    Path("tests/test_price.py").write_text("def test_total():\n    pass\n")
    print("Applied edit to tests/test_price.py")
elif mode == "malformed":
    print("The LLM did not conform to the edit format.")
history = value("--chat-history-file")
Path(history).write_text(
    "\n# aider chat started at 2026-10-15\n\n#### " + task.replace("\n", "  \n#### ")
    + "\n\nI added a discount parameter.\n\n> Tokens: 1.2k sent, 85 received.\n")
'''

FILES = {
    "shop/price.py": "def total(items):\n    return sum(items)\n",
    "tests/test_price.py": "from shop.price import total\n\ndef test_total():\n"
                           "    assert total([1, 2]) == 3\n",
}
CASE = {
    "id": "discount", "task": "Support a discount: total(items, discount=0.5).", "files": FILES,
    "python_check": "from shop.price import total; assert total([10, 10], discount=0.5) == 10",
    "file_unchanged": ["tests/test_price.py"], "expected_files": ["shop/price.py"],
}


@pytest.fixture
def fake_aider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake `aider` first on PATH (the tools venv lives in the test's state dir: empty)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "aider"
    script.write_text(FAKE_AIDER.format(python=sys.executable))
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv(external.AIDER_ENV, raising=False)
    monkeypatch.setenv("FAKE_AIDER_LOG", str(tmp_path / "aider-call.json"))
    monkeypatch.setenv(external.AGENT_ENV, "aider")
    monkeypatch.setenv("VEGA_MODEL", "vega-granite-8b")
    monkeypatch.setenv("VEGA_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("VEGA_TEMPERATURE", raising=False)
    monkeypatch.delenv("VEGA_SEED", raising=False)
    external._version_of.cache_clear()
    return tmp_path / "aider-call.json"


def _setup(**overrides) -> external.ModelSetup:
    values = {"model": "vega-granite-8b", "num_ctx": 8192, "num_ctx_source": "server",
              "ollama_base": "http://localhost:11434"}
    return external.ModelSetup(**{**values, **overrides})


# --- Command and settings ------------------------------------------------------------------------

def test_aider_command_is_non_interactive_and_keeps_its_files_outside_the_project(tmp_path) -> None:
    command = external.aider_command("/venv/bin/aider", _setup(), workdir=tmp_path,
                                     files=["a.py", "pkg/b.py"], git=True)
    assert command[0] == "/venv/bin/aider"
    joined = " ".join(command)
    assert "--model ollama_chat/vega-granite-8b" in joined
    assert "--weak-model ollama_chat/vega-granite-8b" in joined
    for flag in ("--yes-always", "--no-auto-commits", "--no-dirty-commits", "--no-gitignore",
                 "--no-suggest-shell-commands", "--no-auto-test", "--no-check-update",
                 "--no-analytics", "--no-stream", "--no-restore-chat-history", "--no-pretty",
                 "--no-detect-urls"):
        assert flag in command
    assert "--no-git" not in command  # git (and so the repo map) on
    assert command[command.index("--message-file") + 1] == str(tmp_path / "task.txt")
    assert command[command.index("--chat-history-file") + 1] == str(tmp_path / "chat.history.md")
    assert [command[i + 1] for i, part in enumerate(command) if part == "--file"] == [
        "a.py", "pkg/b.py"]
    no_git = external.aider_command("aider", _setup(), workdir=tmp_path, files=[], git=False)
    assert "--no-git" in no_git and "--file" not in no_git


def test_model_settings_pin_the_context_window_and_follow_vega_sampling() -> None:
    plain = external.model_settings(_setup())[0]
    assert plain["name"] == "ollama_chat/vega-granite-8b"
    assert plain["extra_params"] == {"num_ctx": 8192}  # never Aider's per-request size
    assert plain["use_temperature"] is False  # the Modelfile's temperature, as Vega
    pinned = external.model_settings(_setup(temperature=0.0, seed=7,
                                            request={"reasoning_effort": "none"}))[0]
    assert pinned["use_temperature"] == 0.0
    assert pinned["extra_params"] == {"num_ctx": 8192, "seed": 7, "reasoning_effort": "none"}
    meta = external.model_metadata(_setup(num_ctx=16384))["ollama_chat/vega-granite-8b"]
    assert meta["max_input_tokens"] == 16384 and meta["input_cost_per_token"] == 0.0


def test_model_setup_uses_vegas_context_and_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VEGA_NUM_CTX", "12288")
    monkeypatch.setenv("VEGA_TEMPERATURE", "0")
    monkeypatch.setenv("VEGA_SEED", "11")
    monkeypatch.setenv("VEGA_BASE_URL", "http://127.0.0.1:11500/v1")
    setup = external.model_setup("vega-granite-8b")
    assert (setup.num_ctx, setup.num_ctx_source) == (12288, "VEGA_NUM_CTX")
    assert (setup.temperature, setup.seed) == (0.0, 11)
    assert setup.ollama_base == "http://127.0.0.1:11500"


def test_chat_files_adds_everything_that_fits_or_nothing() -> None:
    small = {"a.py": b"x = 1\n", "b/c.py": b"y = 2\n", "blob.bin": b"\xff\xfe\x00"}
    assert external.chat_files(small, 8192, "auto") == ["a.py", "b/c.py"]
    big = {"a.py": b"x" * (8192 * 4)}  # ~8K tokens: more than 40% of an 8K window
    assert external.chat_files(big, 8192, "auto") == []
    assert external.chat_files(big, 8192, "all") == ["a.py"]
    assert external.chat_files(small, 8192, "none") == []
    with pytest.raises(ValueError):
        external.chat_files(small, 8192, "some")


def test_time_cap_scales_with_max_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(external.STEP_SECONDS_ENV, raising=False)
    assert external.time_cap(80) == 80 * external.DEFAULT_STEP_SECONDS
    monkeypatch.setenv(external.STEP_SECONDS_ENV, "10")
    assert external.time_cap(40) == 400
    assert external.time_cap(1) == 60  # floor


def test_parse_output_reads_aiders_report_lines() -> None:
    output = ("Model: ollama_chat/x with diff edit format\nTokens: 950 sent, 40 received.\n"
              "Applied edit to a.py\nTokens: 2.3k sent, 1.1k cache hit, 1.5k received.\n"
              "Tokens: 12k sent, 120 received.\nThe LLM did not conform to the edit format.\n")
    stats = external.parse_output(output)
    assert stats["model_calls"] == 3
    assert stats["prompt_tokens"] == 950 + 2300 + 12000
    assert stats["completion_tokens"] == 40 + 1500 + 120
    assert (stats["writes_ok"], stats["writes_failed"]) == (1, 1)
    assert stats["edit_format"] == "diff" and stats["model_error"] is False
    error = external.parse_output("litellm.APIConnectionError: Ollama_chatException - refused")
    assert error["model_error"] is True and error["prompt_tokens"] is None


def test_assistant_text_drops_the_echoed_task_and_aiders_own_output() -> None:
    history = ("\n# aider chat started at 2026\n\n#### Fix the bug  \n#### in app.py\n\n"
               "The answer is 42.\n\n> Applied edit to app.py\n")
    assert external.assistant_text(history) == "The answer is 42."


def test_unknown_or_missing_agent_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    with pytest.raises(ValueError, match="Unknown agent"):
        external.check_agent("cline")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv(external.AIDER_ENV, raising=False)
    with pytest.raises(ValueError, match="not installed"):
        external.check_agent("aider")
    assert external.check_agent("vega") == "vega"


# --- Scored by the normal checks -----------------------------------------------------------------

@posix_only
def test_fake_aider_edit_is_scored_by_the_normal_checks(fake_aider: Path) -> None:
    attempt = evals.run_attempt(CASE, suite="goals")
    call = json.loads(fake_aider.read_text())
    assert attempt["passed"] is True and attempt["failure_class"] == ""
    assert call["task"] == CASE["task"]  # the same task text, nothing added
    assert call["files"] == ["shop/price.py", "tests/test_price.py"]  # small fixture: all of it
    assert call["git"] is True  # a throwaway repo, so the repo map sees every file
    assert call["ollama"] == "http://localhost:11434"
    assert call["settings"][0]["extra_params"]["num_ctx"] == 8192
    assert Path(call["cwd"]).name.startswith("vega-eval-")  # the attempt's sandbox copy
    assert attempt["agent"] == "aider" and attempt["agent_version"] == "0.86.2"
    assert attempt["diff"]["files"] == ["shop/price.py"]  # no .git, no .aider cache
    assert attempt["unexpected_files"] == []
    assert attempt["model_calls"] == attempt["steps"] == 2
    assert (attempt["prompt_tokens"], attempt["completion_tokens"]) == (13200, 1585)
    assert attempt["tokens_approx"] is True and attempt["edit_format"] == "whole"
    assert attempt["max_steps"] == 80 and attempt["hit_step_limit"] is False
    assert attempt["writes_ok"] == 1 and attempt["exit_code"] == 0
    assert attempt["agent_seconds"] >= 0 and attempt["model_seconds"] is None


@posix_only
def test_fake_aider_failures_get_the_usual_labels(fake_aider: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_AIDER_MODE", "edit_tests")
    wrong = evals.run_attempt(CASE, suite="goals")
    assert not wrong["passed"] and wrong["detail"] == "tests/test_price.py was modified"
    assert wrong["unexpected_files"] == ["tests/test_price.py"]
    assert wrong["failure_class"] == "wrong_file"
    monkeypatch.setenv("FAKE_AIDER_MODE", "malformed")
    malformed = evals.run_attempt(CASE, suite="goals")
    assert malformed["diff"]["files"] == [] and malformed["failure_class"] == "edit_failed"


@posix_only
def test_fake_aider_answer_is_checked_without_the_echoed_task(fake_aider: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_AIDER_MODE", "answer")
    question = {"id": "q", "task": "Which word? Say discount.", "files": {"a.py": "x = 1\n"},
                "response_contains": ["discount parameter"]}
    assert evals.run_attempt(question)["passed"] is True
    echoed = {**question, "response_contains": ["Which word"]}  # only in the task
    assert evals.run_attempt(echoed)["failure_class"] == "wrong_answer"


@posix_only
@needs_sandbox
def test_fake_aider_is_scored_by_hidden_and_regression_tests(fake_aider: Path) -> None:
    case = {"id": "discount", "task": "Support a discount.", "files": FILES,
            "hidden_files": {"hidden_tests/test_discount.py": (
                "from shop.price import total\n\ndef test_discount():\n"
                "    assert total([10, 10], discount=0.5) == 10\n")},
            "check_command": f"{PYTEST} hidden_tests", "regression_command": f"{PYTEST} tests",
            "expected_files": ["shop/price.py"]}
    attempt = evals.run_attempt(case, suite="goals")
    assert attempt["passed"] is True and attempt["regression_passed"] is True
    assert attempt["hidden"] == [1, 1]
    assert "hidden_tests/test_discount.py" not in json.loads(fake_aider.read_text())["files"]


# --- Recording and paired order ------------------------------------------------------------------

@pytest.fixture
def suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cases = tmp_path / "fixes.jsonl"
    cases.write_text("".join(json.dumps({"id": i, "task": "t", "files": {"x.txt": "1"},
                                         "response_contains": ["ok"]}) + "\n" for i in "abc"))
    monkeypatch.setattr(evals, "suite_file", lambda name: cases)
    monkeypatch.setattr(evals, "RUNS_FILE", tmp_path / "runs.jsonl")
    monkeypatch.setattr(evals, "_git_commit", lambda: "abc1234")
    monkeypatch.setattr(evals, "_server_fingerprint", lambda model: {})
    monkeypatch.setattr(evals, "console", Console(record=True, width=200))
    monkeypatch.setattr(external, "aider_executable", lambda: "/tools/aider")
    monkeypatch.setattr(evals, "_agent_version",
                        lambda name: "0.86.2" if name == "aider" else None)
    monkeypatch.setenv("VEGA_MODEL", "vega-granite-8b")
    monkeypatch.delenv(features.ENV, raising=False)
    monkeypatch.delenv(external.AGENT_ENV, raising=False)
    monkeypatch.delenv("VEGA_EVAL_SHARD", raising=False)
    evals.control_file("stop").parent.mkdir(parents=True, exist_ok=True)
    evals.control_file("stop").unlink(missing_ok=True)


def _result(passed: bool) -> dict:
    return {"passed": passed, "detail": "" if passed else "x", "seconds": 1.0, "failure": ""}


def test_ab_agent_runs_every_vega_attempt_before_any_aider_attempt(suite, monkeypatch) -> None:
    seen: list[tuple[str, str, str]] = []

    def fake_attempt(case, **kwargs):
        seen.append((case["id"], os.environ[external.AGENT_ENV], os.environ["VEGA_MODEL"]))
        return _result(os.environ[external.AGENT_ENV] == "vega" or case["id"] == "a")

    monkeypatch.setattr(evals, "run_attempt", fake_attempt)
    record = evals.run_ab("fixes", [], repeat=2, b_agent="aider")
    assert [agent for _, agent, _ in seen] == ["vega"] * 6 + ["aider"] * 6
    assert {model for _, _, model in seen} == {"vega-granite-8b"}  # same model on both sides
    assert [case for case, _, _ in seen[:6]] == [case for case, _, _ in seen[6:]]
    assert record["order"] == "grouped"
    assert record["variants"]["A"]["agent"] == "vega"
    assert record["variants"]["B"]["agent"] == "aider"
    assert record["variants"]["B"]["agent_version"] == "0.86.2"
    assert record["variants"]["A"]["agent_version"] is None
    assert (record["paired"]["a_only"], record["paired"]["both"]) == (4, 2)
    assert external.AGENT_ENV not in os.environ  # restored
    assert "B (aider)" in lab.verdict(record) or "no detectable" in lab.verdict(record)


def test_ab_without_any_difference_is_refused(suite) -> None:
    with pytest.raises(evals.typer.BadParameter, match="--ab-agent"):
        evals.run_ab("fixes", [], repeat=1, b_agent="vega")


def test_run_records_the_agent_and_its_version(suite, monkeypatch) -> None:
    monkeypatch.setenv(external.AGENT_ENV, "aider")
    seen = []
    monkeypatch.setattr(evals, "run_attempt",
                        lambda case, **kw: seen.append(os.environ.get(external.AGENT_ENV)) or _result(True))
    record = evals.run_suite("fixes", repeat=1)
    assert (record["agent"], record["agent_version"]) == ("aider", "0.86.2")
    assert seen == ["aider"] * 3
    saved = json.loads(evals.RUNS_FILE.read_text().splitlines()[-1])
    assert saved["agent"] == "aider"
    monkeypatch.delenv(external.AGENT_ENV)
    assert evals.run_suite("fixes", repeat=1)["agent"] == "vega"


def test_lab_experiment_pins_the_model_and_runs_aider_as_b(tmp_path, monkeypatch) -> None:
    path = tmp_path / "experiments.toml"
    path.write_text('[[experiment]]\nid = "goals-agent-aider"\nsuite = "goals"\n'
                    'hypothesis = "h"\nmodel = "granite-8b"\nab_agent = "aider"\n')
    calls = []

    def fake_ab(suite, b_off, **kwargs):
        calls.append((suite, kwargs["b_agent"], os.environ["VEGA_MODEL"]))
        return {}

    monkeypatch.setattr(evals, "run_ab", fake_ab)
    monkeypatch.setattr(evals, "_resolve_model", lambda value: f"vega-{value}")
    monkeypatch.setattr(lab, "results", lambda runs=None, screens=False: {})
    monkeypatch.setattr(lab, "write_scoreboard", lambda path=None: path)
    monkeypatch.setattr(lab, "console", Console(record=True, width=200))
    lab.run_queue(path=path)
    assert calls == [("goals", "aider", "vega-granite-8b")]
    assert "VEGA_MODEL" not in os.environ  # restored
