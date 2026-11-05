"""Grading a "what should we do next?" answer (nextsteps suite): `project_unchanged` and
`recommend` in vega.evals, without a model."""

from __future__ import annotations

from pathlib import Path

from vega import evals

FILES = {
    "shop.py": "def find_items(conn, text):\n    return conn.execute('...')\n\n\n"
               "def low_stock(conn):\n    return []\n",
    "tests/test_shop.py": "from shop import find_items\n\n\ndef test_find_items():\n    pass\n",
    "README.md": "# shop\n\n- Search (`find_items`)\n- Ideas: a web dashboard someday\n",
    "requirements.txt": "pytest>=8\n",
}
SPEC = {"top": [r"find_items", r"sql[\s-]*injection"], "names": ["shop.py", "test_shop.py",
                                                                 "find_items", "low_stock"]}
GOOD = """I read `shop.py`, `tests/test_shop.py` and README.md.

1. **Fix the SQL injection in `find_items`** — it concatenates the search text.
2. **Test `low_stock`** — nothing covers it.

Want me to start on #1?"""


def _project(tmp_path: Path) -> Path:
    evals._write_fixture(tmp_path, FILES)
    return tmp_path


def _grade(tmp_path: Path, reply: str, **spec) -> str:
    case = {"id": "c", "task": "what's next?", "files": FILES, "recommend": {**SPEC, **spec}}
    return evals.run_checks(case, tmp_path, reply).acceptance


def test_project_unchanged_catches_edits_deletes_and_new_files(tmp_path):
    case = {"id": "c", "task": "t", "files": FILES, "project_unchanged": True}
    project = _project(tmp_path)
    (project / "__pycache__").mkdir()  # caches and Vega's own notes do not count
    (project / "__pycache__" / "shop.cpython-312.pyc").write_bytes(b"x")
    (project / ".vega").mkdir()
    (project / ".vega" / "notes.md").write_text("n")
    assert evals.run_checks(case, project, "").passed
    (project / "shop.py").write_text("changed\n")
    assert "shop.py was modified" in evals.run_checks(case, project, "").acceptance
    (project / "shop.py").write_text(FILES["shop.py"])
    (project / "PLAN.md").write_text("plan\n")
    assert "created PLAN.md" in evals.run_checks(case, project, "").acceptance
    (project / "PLAN.md").unlink()
    (project / "README.md").unlink()
    assert "README.md was deleted" in evals.run_checks(case, project, "").acceptance
    # without the field nothing is checked
    assert evals.run_checks({"id": "c", "task": "t", "files": FILES}, project, "").passed


def test_recommendations_split_numbered_bullets_headings_and_paragraphs():
    lead, items = evals.recommendations("Intro.\n\n1. One\n   more\n2. Two\n\nWant me to?")
    assert lead == "Intro." and items == ["1. One\n   more", "2. Two"]
    _, items = evals.recommendations("- a\n  - sub\n- b")
    assert items == ["- a\n  - sub", "- b"]
    _, items = evals.recommendations("### Fix x\nwhy\n### Then y\nwhy")
    assert [item.splitlines()[0] for item in items] == ["### Fix x", "### Then y"]
    _, items = evals.recommendations("**1. Bold** first\n**2. Bold** second")
    assert len(items) == 2
    assert evals.recommendations("First para.\n\nSecond para.") == ("", ["First para.",
                                                                       "Second para."])


def test_invented_names(tmp_path):
    project = _project(tmp_path)
    assert evals.invented_names(GOOD, project) == []
    assert evals.invented_names("Refactor `services/billing.py` next.", project) == [
        "services/billing.py"]
    assert evals.invented_names("Speed up `warm_cache()` in shop.py.", project) == ["warm_cache"]
    assert evals.invented_names("Then call purge_all() nightly.", project) == ["purge_all"]
    # proposed as new, builtins, stdlib, exception classes, tools, plain words: fine
    fine = ("Add a new `tests/test_cli.py`. Write tests for shop.py in tests/test_low.py. "
            "There is no CHANGELOG.md. Use `json.loads`, `len()`, `sqlite3.connect` and "
            "`pytest.raises`; it raises `JSONDecodeError`. Implement `export_csv()`. "
            "`requirements.txt` lists pytest; `done` and `--json` are flags.")
    assert evals.invented_names(fine, project) == []
    # a name the README mentions is grounded even without code
    assert evals.invented_names("Build the web dashboard from README.md.", project) == []


def test_recommend_passes_a_grounded_ranked_offer(tmp_path):
    _project(tmp_path)
    assert _grade(tmp_path, GOOD) == ""


def test_recommend_failures(tmp_path):
    _project(tmp_path)
    generic = "1. Add more tests.\n2. Improve the docs.\n\nWant me to start?"
    assert "generic" in _grade(tmp_path, generic)
    wrong = GOOD.replace("1. **Fix the SQL injection in `find_items`** — it concatenates the "
                         "search text.\n2. **Test `low_stock`** — nothing covers it.",
                         "1. **Test `low_stock`** — nothing covers it.\n2. **Dashboard** — "
                         "from the README.\n3. **Fix the SQL injection in `find_items`**.")
    assert "misses the planted signal" in _grade(tmp_path, wrong)
    assert "does not have" in _grade(tmp_path, GOOD.replace("README.md.", "README.md and "
                                                            "`api/routes.py`."))
    assert "no offer" in _grade(tmp_path, GOOD.rsplit("\n\n", 1)[0])
    assert "words (cap 20)" in _grade(tmp_path, GOOD, max_words=20)
    assert "asked where" in _grade(tmp_path, GOOD + " Where is the project located?")
    assert "how to verify" in _grade(tmp_path, "How should I verify it? " + GOOD)
    assert _grade(tmp_path, GOOD, within=1) == ""
    assert "misses" in _grade(tmp_path, GOOD, top=[r"low_stock"], within=1)


def test_recommend_avoid_and_top_anywhere(tmp_path):
    _project(tmp_path)
    avoid = [r"\b(?:add|implement)\w*\b[^.\n]{0,30}\blow[- ]stock"]
    stale = GOOD.replace("**Test `low_stock`** — nothing covers it.",
                         "**Add a low-stock report** — the roadmap lists it.")
    assert "recommends finished work" in _grade(tmp_path, stale, avoid=avoid)
    done = GOOD.replace("**Test `low_stock`** — nothing covers it.",
                        "**Add low-stock report: already done** — tick it off in the roadmap.")
    assert _grade(tmp_path, done, avoid=avoid) == ""
    healthy = ("The project is in good shape: `shop.py` and `tests/test_shop.py` cover the "
               "README.\n\nOptional: test `low_stock`.\n\nWant me to start on that?")
    assert _grade(tmp_path, healthy, top=[r"good shape"], top_anywhere=True) == ""
    late = ("Next:\n\n1. Test `low_stock`.\n2. Rename `find_items`.\n3. It is in good shape."
            "\n\nWant me to?")
    assert "misses" in _grade(tmp_path, late, top=[r"good shape"])  # not in the first two
    assert _grade(tmp_path, late, top=[r"good shape"], top_anywhere=True) == ""


def test_recommend_counts_as_a_read_only_answer():
    assert evals._is_read_only({"recommend": {"top": ["x"]}})
