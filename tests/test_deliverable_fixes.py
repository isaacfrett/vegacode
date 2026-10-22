"""Deliverable diagnosis 2026-10-20 fixes: script vs File-block disagreement, format rules in the
pipeline's checks, script-error values, host-path scrubbing, code under a deliverable's label and
the sandbox read guard."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vega import deliverable_pipeline as dp
from vega import features, sandbox
from vega.agent import CodingAgent

needs_sandbox = pytest.mark.skipif(not sandbox.available(), reason="needs the macOS sandbox")

SSHD_TASK = ("`root/` is a snapshot of a server (`root/` stands for `/`). Write the answer to "
             "`answers/sshd.json` as one JSON object with exactly these keys: `port`, "
             "`password_auth`.")
SSHD_SCRIPT = ("```python\nimport json, os\nos.makedirs('answers', exist_ok=True)\n"
               "json.dump({'port': '22', 'password_auth': 'yes'}, open('answers/sshd.json', 'w'))"
               "\n```")
SSHD_BLOCK = 'File: answers/sshd.json\n```json\n{"port": "22", "password_auth": "no"}\n```'


def scripted(root: Path, replies: list[str]) -> tuple[CodingAgent, list[dict]]:
    agent = CodingAgent(root, allow_commands=False)
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content=replies.pop(0), tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return agent, calls


def _off(monkeypatch, *names: str) -> None:
    monkeypatch.setenv(features.ENV, ",".join(names))


def _fixture(root: Path) -> None:
    (root / "root" / "etc" / "ssh").mkdir(parents=True)
    (root / "root" / "etc" / "ssh" / "sshd_config").write_text("PasswordAuthentication no\n")


# -- deliverable_disagreement ---------------------------------------------------------------------

def test_disagreement_quotes_differing_words_not_computed_numbers() -> None:
    script = {"a.json": json.dumps({"port": "22", "auth": "yes", "score": 1.5, "ids": [1, 2]})}
    block = {"a.json": json.dumps({"port": "22", "auth": "no", "score": 2.0, "ids": [2, 1]})}
    assert dp.disagreement(script, block) == [
        ('a.json: your script and your File: block disagree:\n'
        '  - "auth": the script wrote "yes", your File: block has "no"')]
    numbers_only = {"a.json": json.dumps({"score": 2.0})}
    assert dp.disagreement({"a.json": json.dumps({"score": 1.5})}, numbers_only) == []
    assert dp.disagreement({"a.md": "x\n"}, {"a.md": "y\n"}) == []  # text: the script stands
    assert dp.disagreement(script, {"a.json": script["a.json"]}) == []


@needs_sandbox
def test_a_disagreeing_script_and_block_get_one_repair_round(tmp_path: Path) -> None:
    _fixture(tmp_path)
    blocks_only = SSHD_BLOCK
    agent, calls = scripted(tmp_path, [SSHD_SCRIPT + "\n\n" + SSHD_BLOCK, blocks_only])
    answer = dp.run(agent, SSHD_TASK)
    repair = calls[1]["messages"][-1]["content"]
    assert ('"password_auth": the script wrote "yes", your File: block has "no"' in repair
            and "reply with File: blocks only" in repair)
    assert json.loads((tmp_path / "answers" / "sshd.json").read_text())["password_auth"] == "no"
    assert answer.startswith("Wrote answers/sshd.json")


@needs_sandbox
def test_the_disagreement_round_is_asked_once_then_the_script_stands(tmp_path: Path) -> None:
    _fixture(tmp_path)
    both = SSHD_SCRIPT + "\n\n" + SSHD_BLOCK
    agent, calls = scripted(tmp_path, [both, both, both])
    dp.run(agent, SSHD_TASK)
    assert len(calls) == 2  # round 2 had no failing check: kept
    assert json.loads((tmp_path / "answers" / "sshd.json").read_text())["password_auth"] == "yes"


@needs_sandbox
def test_disagreement_off_keeps_the_script_silently(tmp_path: Path, monkeypatch) -> None:
    _off(monkeypatch, "deliverable_disagreement")
    _fixture(tmp_path)
    agent, calls = scripted(tmp_path, [SSHD_SCRIPT + "\n\n" + SSHD_BLOCK])
    dp.run(agent, SSHD_TASK)
    assert len(calls) == 1
    assert json.loads((tmp_path / "answers" / "sshd.json").read_text())["password_auth"] == "yes"


# -- deliverable_code_label (harness bug 6) --------------------------------------------------------

LABELLED = ("**Python script to generate `answers/sshd.json`**\n\n```python\nimport json\n"
            "json.dump({}, open('answers/sshd.json', 'w'))\n```")


def test_a_script_under_a_json_label_is_the_script(monkeypatch) -> None:
    found, scripts = dp.file_blocks(LABELLED, ["answers/sshd.json"])
    assert found == {} and len(scripts) == 1 and scripts[0].startswith("import json")
    plain = "File: answers/a.json\n```\nimport json\nprint(1)\n```"
    assert dp.file_blocks(plain, ["answers/a.json"])[1] == ["import json\nprint(1)\n"]
    data = 'File: answers/a.json\n```\n{"import json": 1}\n```'
    assert dp.file_blocks(data, ["answers/a.json"]) == ({"answers/a.json": '{"import json": 1}\n'},
                                                         [])
    _off(monkeypatch, "deliverable_code_label")
    assert dp.file_blocks(LABELLED, ["answers/sshd.json"])[0] == {
        "answers/sshd.json": "import json\njson.dump({}, open('answers/sshd.json', 'w'))\n"}


def test_a_refused_json_deliverable_is_not_asked_for_in_the_same_form(monkeypatch) -> None:
    refusal = ("Write not applied: answers/sshd.json would not parse.\nYour code became lines 1-9 "
               "of the would-be answers/sshd.json (shown above). Reply again with the corrected "
               "code for answers/sshd.json in the same form as before.")
    fixed = dp._same_form(refusal)
    assert "same form" not in fixed and "in a `File: answers/sshd.json` block" in fixed
    python = refusal.replace("answers/sshd.json", "tool.py")
    assert dp._same_form(python) == python
    _off(monkeypatch, "deliverable_code_label")
    assert dp._same_form(refusal) == refusal


# -- format_rules in the pipeline's checks ---------------------------------------------------------

def test_format_rules_failures_reach_the_repair_prompt(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("x = 1\n")
    task = ("Review `app.py`. Report every real bug as a JSON list in `review.json`, one object "
            'per bug: `{"file": "<path>", "lines": [<first>, <last>]}`.')
    nulls = 'File: review.json\n```json\n[{"file": "app.py", "lines": [null, null]}]\n```'
    good = 'File: review.json\n```json\n[{"file": "app.py", "lines": [1, 1]}]\n```'
    agent, calls = scripted(tmp_path, [nulls, good])
    answer = dp.run(agent, task)
    assert ('review.json item 0 "lines" item 0 is null: the task\'s template says '
            "[<first>, <last>]") in calls[1]["messages"][-1]["content"]
    assert answer == "Wrote review.json; 2 of 2 checks pass (1 format check, 1 format rule)."


# -- script_error_values, sandbox_path_scrub --------------------------------------------------------

@needs_sandbox
def test_a_failing_script_shows_the_values_on_its_failing_line(tmp_path: Path) -> None:
    (tmp_path / "ids.txt").write_text("Q-01\nQ-02\n")
    task = "Read `ids.txt` and write `out.json` as {\"ids\": [...]}."
    broken = ("```python\nimport json\nselected = open('ids.txt').read().split()\n"
              "ids = [i['id'] for i in selected]\njson.dump({'ids': ids}, open('out.json', 'w'))"
              "\n```")
    agent, calls = scripted(tmp_path, [broken] * (dp.SCRIPT_REPAIRS + 1))
    answer = dp.run(agent, task)
    feedback = calls[1]["messages"][-1]["content"]
    assert "Values at the failing line(s) of your script:" in feedback
    assert "selected = ['Q-01', 'Q-02']" in feedback
    assert "    ids = [i['id'] for i in selected]" in feedback  # the source line in the traceback
    assert ".vega-deliverable/run.py" not in feedback  # the runner's frames are dropped
    assert answer.startswith("Not written: out.json.")
    assert answer.rstrip().endswith("TypeError: string indices must be integers, not 'str'")
    assert str(Path.home()) not in feedback + answer


def test_host_paths_are_scrubbed(tmp_path: Path) -> None:
    lib = Path(sys.base_prefix) / "lib" / "json" / "decoder.py"
    venv = Path(sys.prefix) / "lib" / "site-packages" / "x.py"
    text = f'File "{lib}", line 3\nFile "{venv}"\n{Path.home()}/notes\n/Users/dana/Library/x\n'
    scrubbed = sandbox.scrub_paths(text, tmp_path / "clone", tmp_path)
    assert str(Path.home()) not in scrubbed
    assert "<python>/lib/json/decoder.py" in scrubbed or "<venv>" in scrubbed
    assert "/Users/dana/Library/x" in scrubbed  # a fixture's path is not the host's


@needs_sandbox
def test_run_check_output_has_no_host_paths(tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_text("import json\njson.loads('{bad')\n")
    output = sandbox.run_check("python bad.py", tmp_path)
    assert "JSONDecodeError" in output and str(Path.home()) not in output


# -- sandbox_read_guard -----------------------------------------------------------------------------

READ_HOST = ("```python\nimport json, os\nos.makedirs('answers', exist_ok=True)\n"
             "text = open('/etc/ssh/sshd_config').read()\n"
             "json.dump({'port': '22', 'password_auth': 'no'}, open('answers/sshd.json', 'w'))\n```")


@needs_sandbox
def test_a_script_cannot_read_the_hosts_files_and_is_told_where_the_copy_is(tmp_path: Path) -> None:
    _fixture(tmp_path)
    files, error = dp._run_script(tmp_path, READ_HOST.split("\n", 1)[1].rsplit("```", 1)[0],
                                  ["answers/sshd.json"], dp.fixture_root(SSHD_TASK))
    assert files == {}
    assert "PermissionError: Vega's sandbox: a deliverable script reads only the project's files" \
        in error
    assert "read root/etc/ssh/sshd_config instead" in error
    shutil.rmtree(tmp_path / dp.SCRIPT_DIR, ignore_errors=True)


@needs_sandbox
@pytest.mark.parametrize("path", ["/etc/hosts", str(Path.home() / ".zshrc"), "/private/tmp"])
def test_the_profile_denies_host_reads_even_without_the_audit_hook(tmp_path: Path,
                                                                    path: str) -> None:
    """The kernel-level layer: a subprocess (cat, ls) is not seen by Python's audit hook."""
    (tmp_path / "probe.py").write_text(
        "import subprocess, sys\n"
        f"r = subprocess.run(['cat' if '{path}' != '/private/tmp' else 'ls', '{path}'], "
        "capture_output=True, text=True)\nprint('DENIED' if 'not permitted' in r.stderr "
        "else 'READ' if r.returncode == 0 else 'MISSING')\n")
    guarded = sandbox.run_check("python probe.py", tmp_path, private_home=True)
    assert "DENIED" in guarded or "MISSING" in guarded
    assert "READ" not in guarded


@needs_sandbox
def test_host_inspection_still_works_under_the_guard(tmp_path: Path) -> None:
    """os-host tasks: lsof, sw_vers, uname and platform read the kernel or /System, not the
    guarded folders; zoneinfo and local time still load."""
    (tmp_path / "probe.py").write_text(
        "import os, platform, subprocess, time, zoneinfo\n"
        "zoneinfo.ZoneInfo('Europe/Berlin'); time.localtime()\n"
        "lsof = subprocess.run(['lsof', '-nP', '-a', '-p', str(os.getpid()), '-d', 'cwd', '-Fn'],"
        " capture_output=True, text=True)\n"
        "print('lsof', lsof.returncode, 'n' in lsof.stdout)\n"
        "print('uname', subprocess.run(['uname', '-r'], capture_output=True).returncode)\n"
        "print('sw_vers', subprocess.run(['sw_vers'], capture_output=True).returncode)\n"
        "print('mac_ver', bool(platform.mac_ver()[0]))\n")
    output = sandbox.run_check("python probe.py", tmp_path, private_home=True)
    assert output.startswith("exit code: 0"), output
    assert "lsof 0 True" in output and "uname 0" in output and "sw_vers 0" in output
    assert "mac_ver True" in output


@needs_sandbox
def test_the_guard_switch_restores_host_reads(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "probe.py").write_text("print(len(open('/etc/hosts').read()) > 0)\n")
    assert "PermissionError" in sandbox.run_check("python probe.py", tmp_path, private_home=True)
    _off(monkeypatch, "sandbox_read_guard")
    assert "True" in sandbox.run_check("python probe.py", tmp_path, private_home=True)
