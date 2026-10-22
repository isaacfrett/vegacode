"""Helpers behind the research fixes that need no web: web URLs and JSON sources, related files,
uninstalled third-party imports, and stated JSON template keys (web_fetch_first, direct_context,
acceptance_uninstalled, json_template). Scripted fake models only.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vega import acceptance, json_template, routing, sandbox, web
from vega.agent import CodingAgent

NEWS = "https://driverco.example/news/2.3"


def test_web_urls_and_json_sources(tmp_path: Path) -> None:
    assert web.is_web_url("https://driverco.example/news/2.3")
    assert web.is_web_url("driverco.example/news/2.3")
    assert not web.is_web_url("docs/PLATFORM.md")
    assert not web.is_web_url("requirements.txt")
    assert not web.is_web_url("http://localhost:8000/x")
    assert not web.is_web_url("http://127.0.0.1/x")
    (tmp_path / "A.json").write_text(json.dumps({"sources": [NEWS, "docs/x.md", 3]}))
    (tmp_path / "B.json").write_text("[1]")
    assert web.json_sources(tmp_path, ["A.json", "B.json", "missing.json", "x.py"]) == [NEWS]


def test_vendored_and_relative_imports_resolve(tmp_path: Path) -> None:
    (tmp_path / "third_party" / "tinyq").mkdir(parents=True)
    (tmp_path / "third_party" / "tinyq" / "__init__.py").write_text("def push(x): ...\n")
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "__init__.py").write_text("")
    (tmp_path / "app" / "util.py").write_text("def norm(x):\n    return x\n")
    (tmp_path / "app" / "jobs.py").write_text(
        "import json\nimport tinyq\nfrom .util import norm\n\n\ndef run(x):\n"
        "    raise NotImplementedError\n")
    files = ["third_party/tinyq/__init__.py", "app/__init__.py", "app/util.py", "app/jobs.py"]
    assert routing.related_files(tmp_path, "app/jobs.py", "Implement `run`.", files,
                                 exclude={"app/jobs.py"}) == \
        ["third_party/tinyq/__init__.py", "app/util.py"]


@pytest.mark.skipif(not sandbox.available(), reason="needs the macOS sandbox")
def test_a_missing_third_party_module_is_not_checkable(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "store.py").write_text("import nosuchdbx_not_installed\n\n\ndef connect():\n"
                                       "    return 1\n")
    spec = acceptance.extract(tmp_path, "Fix `connect` in store.py.")
    [item] = [i for i in acceptance.run(spec, tmp_path, ["store.py"]).items if i.kind == "import"]
    assert item.ok and "not checkable here (module not installed: nosuchdbx_not_installed)" \
        in item.detail
    # a module of the project that is missing is still a failure
    (tmp_path / "shop").mkdir()
    (tmp_path / "shop" / "__init__.py").write_text("")
    (tmp_path / "shop" / "cart.py").write_text("from shop import nothere\n\n\ndef total():\n"
                                               "    return 1\n")
    spec = acceptance.extract(tmp_path, "Fix `total` in shop/cart.py.")
    [item] = [i for i in acceptance.run(spec, tmp_path, ["shop/cart.py"]).items
              if i.kind == "import"]
    assert not item.ok
    monkeypatch.setenv("VEGA_FEATURES_OFF", "acceptance_uninstalled")
    spec = acceptance.extract(tmp_path, "Fix `connect` in store.py.")
    [item] = [i for i in acceptance.run(spec, tmp_path, ["store.py"]).items if i.kind == "import"]
    assert not item.ok and "ModuleNotFoundError" in item.detail


def test_template_keys_are_read_from_the_task() -> None:
    assert json_template.stated(
        "Write `DECISION.json` with exactly the keys `choice`, `min_version`, `reasons` and "
        "`sources`.") == {"DECISION.json": ["choice", "min_version", "reasons", "sources"]}
    assert json_template.stated(
        "Write `answers/port.json`, one JSON object with exactly the keys `pid` (a JSON number) "
        "and `command` (the name), or both `null`.") == {"answers/port.json": ["pid", "command"]}
    assert json_template.stated("Write out.json with keys a, b and c.") == \
        {"out.json": ["a", "b", "c"]}
    assert json_template.stated("Write out.json with keys for each user.") == {}
    assert json_template.stated("Write `answers/x.json` with exactly these keys:\n"
                                "- `cause`: one of \"a\", \"b\"\n- `when_utc`: ISO time\n\nDone.") \
        == {"answers/x.json": ["cause", "when_utc"]}


def test_official_results_are_marked_by_name_or_docs_host() -> None:
    urls = ["https://nvd.example/CVE-2026-00001", "https://advisories.kettlehttp.example/KH-1",
            "https://forum.qa.example/t/kettlehttp", "https://docs.python.org/3/x",
            "https://kettlehttp-tips.example/x"]
    official = web.official_urls("CVE-2026-00001 kettlehttp fix release", urls)
    assert official == {urls[1], urls[3], urls[4]}  # a label holding the name, a docs host
    assert web.official_urls("fix release docs", ["https://releases.example/x"]) == set()


def test_deliverable_pipeline_repairs_missing_template_keys(tmp_path: Path) -> None:
    (tmp_path / "backlog.csv").write_text("id,value\nA,1\nB,2\n")
    task = ("Rank the backlog in `backlog.csv` by value, highest first, and write `ranking.json` "
            "as {\"ranking\": [ids], \"top\": id}.")
    replies = ["File: ranking.json\n```json\n{\"ranking\": [\"B\", \"A\"]}\n```",
               "File: ranking.json\n```json\n{\"ranking\": [\"B\", \"A\"], \"top\": \"B\"}\n```"]
    prompts: list[str] = []
    agent = CodingAgent(tmp_path, allow_commands=False)

    def create(**kwargs: Any) -> Any:
        prompts.append(kwargs["messages"][-1]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=replies.pop(0), tool_calls=None),
            finish_reason="stop")], usage=None)

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    answer = agent.run(task)
    assert 'ranking.json lacks the key(s) the task states: "top"' in prompts[1]
    assert answer.startswith("Wrote ranking.json; 2 of 2 checks pass")
    assert json.loads((tmp_path / "ranking.json").read_text())["top"] == "B"
