"""fence_lines, short_tracebacks and output_head_tail.

The shapes: a Markdown heading finder whose code holds `startswith("```")`; a broken `binderapp add`
so argparse exits with SystemExit: 2 deep in its own frames; 13 failing tests whose default
tracebacks overflow the sandbox's 8,000-character tail.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from switches import switch_off

from vega import check_feedback, pipeline, project_pipeline, sandbox, write_loop

needs_sandbox = pytest.mark.skipif(not sandbox.available(), reason="needs macOS sandbox-exec")

# -- fence_lines ---------------------------------------------------------------------------------

HEADINGS = '''import re
from dataclasses import dataclass

HEADING = re.compile(r"^(#{1,6})\\s+(.*?)\\s*#*\\s*$")


@dataclass
class Heading:
    level: int
    text: str
    line: int


def find_headings(markdown):
    """ATX headings outside fenced code blocks, with 0-based line numbers."""
    headings, fenced = [], False
    for number, line in enumerate(markdown.splitlines()):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        match = None if fenced else HEADING.match(line)
        if match:
            headings.append(Heading(len(match.group(1)), match.group(2), number))
    return headings


def slug(text):
    return re.sub(r"[^a-z0-9 -]", "", text.lower()).strip().replace(" ", "-")
'''
REPLY = f"File: wikigen/headings.py\n```python\n{HEADINGS}```\n\nThe slug drops punctuation.\n"


def test_a_backtick_fence_inside_the_code_does_not_end_the_block() -> None:
    code, problem = write_loop.pick_block(REPLY, "wikigen/headings.py", ["wikigen/headings.py"])
    assert problem == ""
    assert code == HEADINGS
    ast.parse(code)  # was cut at `startswith("` -> unterminated string literal, column 37


def test_the_old_parse_cut_the_file_mid_string(monkeypatch: pytest.MonkeyPatch) -> None:
    switch_off(monkeypatch, "fence_lines")  # the bug this switch fixes
    code, _ = write_loop.pick_block(REPLY, "wikigen/headings.py", ["wikigen/headings.py"])
    assert code.endswith('startswith("')
    with pytest.raises(SyntaxError):
        ast.parse(code)


def test_line_fences_still_separate_blocks_and_read_labels() -> None:
    reply = ("File: wikigen/headings.py\n```python\n" + HEADINGS + "```\n\n"
             "File: wikigen/build.py\n```python\nfrom wikigen.headings import find_headings\n```\n")
    blocks = write_loop.reply_blocks(reply)
    assert [(lang, label) for lang, label, _ in blocks] == [
        ("python", "wikigen/headings.py"), ("python", "wikigen/build.py")]
    assert blocks[0][2] == HEADINGS
    assert blocks[1][2] == "from wikigen.headings import find_headings\n"


def test_a_longer_fence_closes_only_with_a_fence_as_long() -> None:
    reply = f"File: wikigen/headings.py\n````python\n{HEADINGS}````\n"
    assert write_loop.reply_blocks(reply) == [("python", "wikigen/headings.py", HEADINGS)]


def test_inline_fences_still_parse_when_no_block_has_line_fences() -> None:
    assert write_loop.reply_blocks("Here: ```python\nx = 1```") == [("python", "", "x = 1")]
    assert write_loop.code_blocks("```python\nx = 1\n```  \n") == ["x = 1\n"]


def test_code_blocks_for_the_project_pipeline_fallback() -> None:
    reply = f"```python\n{HEADINGS}```\n```toml\n[project]\n```\n"
    assert write_loop.code_blocks(reply) == [HEADINGS]
    assert write_loop.code_blocks(reply, any_language=True) == [HEADINGS, "[project]\n"]
    assert write_loop.code_blocks("```json\n{}\n```") == []


def test_a_file_holding_backticks_is_shown_in_a_longer_fence(tmp_path: Path) -> None:
    assert write_loop.fence_for("x = 1\n") == "```"
    assert write_loop.fence_for(HEADINGS) == "````"
    (tmp_path / "wikigen").mkdir()
    (tmp_path / "wikigen" / "headings.py").write_text(HEADINGS, encoding="utf-8")
    for inside in (False, True):
        shown = write_loop.render_inputs(tmp_path, ["wikigen/headings.py"], budget=10_000,
                                         header="Inputs", fence="python", label_inside=inside)
        [(_, _, body)] = write_loop.reply_blocks(shown)
        assert body.rstrip("\n").endswith("return re.sub(r\"[^a-z0-9 -]\", \"\", text.lower())"
                                          ".strip().replace(\" \", \"-\")")


def test_a_cut_off_reply_is_inside_its_block_despite_inner_backticks() -> None:
    cut = 'File: wikigen/headings.py\n```python\n    if line.lstrip().startswith("```"):\n'
    assert write_loop.inside_open_block(cut)
    assert not write_loop.inside_open_block(cut + "        pass\n```\n")
    assert write_loop.inside_open_block("Here: ```python\nx = 1\n")  # no line fence: old count


# -- short_tracebacks / output_head_tail ----------------------------------------------------------

def test_pytest_commands_get_short_tracebacks(monkeypatch: pytest.MonkeyPatch) -> None:
    command = "python -m pytest -q -p no:cacheprovider -rfE"
    assert check_feedback.short_tracebacks(command) == command + " --tb=short"
    assert check_feedback.short_tracebacks("pytest -q --tb=line") == "pytest -q --tb=line"
    assert check_feedback.short_tracebacks("go test ./...") == "go test ./..."
    switch_off(monkeypatch, "short_tracebacks")  # off: the command is unchanged
    assert check_feedback.short_tracebacks(command) == command


def test_clipped_output_keeps_head_and_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    output = "\n".join(f"line {n:05d}" for n in range(3000))
    clipped = sandbox.clip_output(output, 8000)
    head, _, tail = clipped.partition("\n... [")
    assert head.startswith("line 00000") and tail.endswith("line 02999")
    assert tail.split("\n", 1)[0].endswith("characters of output cut here] ...")
    assert len(clipped) < 8000 + 60
    assert sandbox.clip_output("short", 8000) == "short"
    switch_off(monkeypatch, "output_head_tail")  # off: only the tail, as before
    assert sandbox.clip_output(output, 8000) == ("... [earlier output truncated]\n"
                                                 + output[-8000:])


def test_condense_closes_the_block_at_the_cut() -> None:
    output = "\n".join([
        "exit code: 1", "FF", "=" * 20 + " FAILURES " + "=" * 20,
        "_" * 10 + " test_a " + "_" * 10, "tests/test_x.py:3: in test_a", "    assert f() == 1",
        "E   assert 2 == 1", sandbox.CUT_LINE.format(5000),
        "E   AssertionError: from a block whose header was cut", "tests/test_x.py:9: in test_b",
        "_" * 10 + " test_c " + "_" * 10, "tests/test_x.py:12: in test_c", "    assert g()",
        "E   assert False", "=" * 10 + " short test summary info " + "=" * 10,
        "FAILED tests/test_x.py::test_a - assert 2 == 1", "FAILED tests/test_x.py::test_b",
        "FAILED tests/test_x.py::test_c - assert False", "3 failed in 0.10s", "x" * 900])
    text = check_feedback.condense(output, note=False)
    a = text.split("FAILED tests/test_x.py::test_a", 1)[1].split("FAILED", 1)[0]
    assert "E assert 2 == 1" in a and "header was cut" not in a
    assert "E assert False" in text and "test_b" in text


BINDERAPP_CLI = '''import argparse


def build_parser():
    parser = argparse.ArgumentParser(prog="binderapp")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add")
    for name in ("isbn", "title", "author", "year", "extra"):
        add.add_argument(name)
    add.add_argument("--tag", action="append", default=[])
    commands.add_parser("list")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return 0
'''
BINDERAPP_TESTS = '''from binderapp.cli import main


def test_add_and_list():
    assert main(["add", "978-0-14-118776-1", "Dune", "Frank Herbert", "1965", "--tag", "SF"]) == 0
    assert main(["list"]) == 0


def test_checkout_and_checkin():
    assert main(["add", "0441013597", "Dune", "Frank Herbert", "1965"]) == 0


def test_list_alone():
    assert main(["list"]) == 0
'''


def _dutylist_tests() -> str:
    """13 failing tests with dict diffs; each test's body is long, as pytest's default traceback
    prints the whole test function (dutylist: 13 failed / 7 passed, about 12k characters)."""
    parts = ["def plan(shifts):\n    return {s: 'zoe' for s in shifts}\n"]
    for number in range(13):
        setup = "\n".join(f"    shift_{k} = ('s{k}', '2026-03-{9 + k % 3:02d}T09:00', "
                          f"'2026-03-{9 + k % 3:02d}T17:00', 'bar')" for k in range(14))
        parts.append(f"def test_rule_{number:02d}():\n{setup}\n"
                     f"    assert plan(['s1', 's2', 's3']) == {{'s1': 'ana', 's2': 'zoe', "
                     f"'s3': 'ana{number}'}}\n")
    parts += [f"def test_ok_{n}():\n    assert True\n" for n in range(7)]
    return "\n\n".join(parts)


def _project(tmp_path: Path, name: str, files: dict[str, str]) -> Path:
    root = tmp_path / name
    for relative, text in files.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text, encoding="utf-8")
    return root


@needs_sandbox
def test_binderapp_shape_repair_text_names_the_argparse_error(tmp_path: Path) -> None:
    project = _project(tmp_path, "binderapp", {"src/binderapp/__init__.py": "", "src/binderapp/cli.py": BINDERAPP_CLI,
                                           "tests/test_binderapp.py": BINDERAPP_TESTS})
    count, text = project_pipeline._run_tests(project)
    assert count == 2
    assert "the following arguments are required: extra" in text
    assert "src/binderapp/cli.py" in text


@needs_sandbox
def test_dutylist_shape_repair_text_has_the_assertion_diffs(tmp_path: Path) -> None:
    project = _project(tmp_path, "dutylist", {"tests/test_dutylist.py": _dutylist_tests()})
    count, text = project_pipeline._run_tests(project)
    assert count == 13
    assert text.count("E AssertionError: assert {") >= 5
    assert "Also failing (8):" in text


@needs_sandbox
def test_dutylist_shape_without_the_switches_is_bare_ids(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    switch_off(monkeypatch, "short_tracebacks", "output_head_tail")  # the bug they fix
    project = _project(tmp_path, "dutylist", {"tests/test_dutylist.py": _dutylist_tests()})
    count, text = project_pipeline._run_tests(project)
    assert count == 13
    assert "E AssertionError" not in text and "Failing:" in text


@needs_sandbox
def test_direct_pipeline_test_output_has_the_assertions(tmp_path: Path) -> None:
    project = _project(tmp_path, "dutylist", {"test_dutylist.py": _dutylist_tests()})
    result = pipeline.run_checks(project, [project / "test_dutylist.py"], [])
    assert "E AssertionError: assert {" in result.test_output
    assert result.test_output.splitlines()[-1].startswith("13 failed, 7 passed")
