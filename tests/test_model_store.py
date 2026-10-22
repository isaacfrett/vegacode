"""Model downloads and the catalogue (model_store, `vega model list|pull|rm|recommend|watch|queue`)
against a fake Ollama server on localhost: no real model is pulled, deleted or run."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
from fake_ollama import GB, FakeOllama, tag
from private_data import require_eval_data
from typer.testing import CliRunner

from vega import cli, lab, model_store, models, resources, server

PROFILES = """default = "granite-8b"

[profiles.granite-8b]
base = "granite4.1:8b"
num_ctx = 8192

[profiles.coder-7b]
base = "qwen2.5-coder:7b-instruct-q4_K_M"
num_ctx = 8192

[profiles.coder-7b-16k]
base = "qwen2.5-coder:7b-instruct-q4_K_M"
num_ctx = 16384

[profiles.lms]
base = "some-model"
endpoint = "http://127.0.0.1:1234/v1"
"""

# Made-up scores in the scores file's format: the real file is not part of the test data. Granite ranks first, as the shipped default does.
SCORES = """[models.granite-8b]
measured_on = "2026-01-02"
[models.granite-8b.scores]
fixes = { passed = 7, total = 10, commit = "c1", experiment = "e1" }
goals = { passed = 3, total = 8, commit = "c1", experiment = "e2" }
humaneval40 = { passed = 9, total = 10, commit = "c1", experiment = "e3" }
mbpp40 = { passed = 6, total = 10, commit = "c1", experiment = "e4" }

[models.coder-7b]
measured_on = "2026-01-02"
[models.coder-7b.scores]
fixes = { passed = 5, total = 10, commit = "c1", experiment = "e1" }
goals = { passed = 1, total = 8, commit = "c1", experiment = "e2" }

[models.qwen3-8b-nothink]
measured_on = "2026-01-02"
[models.qwen3-8b-nothink.scores]
goals = { passed = 1, total = 8, commit = "c1", experiment = "e5" }
humaneval40 = { passed = 8, total = 10, commit = "c1", experiment = "e6" }

[models.dropped-model]
measured_on = "2026-01-02"
[models.dropped-model.scores]
fixes = { passed = 1, total = 10, commit = "c1", experiment = "e1" }
"""

GRANITE, QWEN = "granite4.1:8b", "qwen2.5-coder:7b-instruct-q4_K_M"
W_GRANITE, W_QWEN = 5_347_914_400, 4_683_073_000  # the weights blobs


def layers(weights: str, size: int, *small: str) -> dict[str, int]:
    out = {weights: size, "sha256:template": 6843}
    out.update({digest: 200 for digest in small})
    return out


def installed_models() -> tuple[list[dict], dict[str, dict[str, int]]]:
    blobs = {
        "vega:latest": layers("sha256:granite", W_GRANITE, "sha256:sys-vega"),
        "vega-granite-8b:latest": layers("sha256:granite", W_GRANITE, "sha256:sys-g"),
        GRANITE: layers("sha256:granite", W_GRANITE),
        "vega-coder-7b:latest": layers("sha256:qwen", W_QWEN, "sha256:sys-c"),
        "vega-coder-7b-16k:latest": layers("sha256:qwen", W_QWEN, "sha256:sys-c16"),
        QWEN: layers("sha256:qwen", W_QWEN),
        "smollm2:1.7b": {"sha256:smol": 1_800_000_000, "sha256:smol-t": 1000},
    }
    parents = {"vega:latest": GRANITE, "vega-granite-8b:latest": GRANITE,
               "vega-coder-7b:latest": QWEN, "vega-coder-7b-16k:latest": QWEN}
    tags = [tag(name, sum(b.values()), parents.get(name, ""),
                "1.7B" if "smol" in name else "8.8B") for name, b in blobs.items()]
    return tags, blobs


def write_manifests(root: Path, blobs: dict[str, dict[str, int]]) -> None:
    for name, layer in blobs.items():
        model, _, version = name.partition(":")
        path = root / "manifests" / "registry.ollama.ai" / "library" / model / version
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "schemaVersion": 2, "config": {"digest": f"sha256:cfg-{model}", "size": 100},
            "layers": [{"digest": d, "size": s} for d, s in layer.items()]}))


@pytest.fixture
def scores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The made-up scores file in place of the real one."""
    path = tmp_path / "model_scores.toml"
    path.write_text(SCORES)
    monkeypatch.setattr(model_store, "SCORES_FILE", path)
    return path


@pytest.fixture
def no_scores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No scores file and no runs, as in the public release."""
    monkeypatch.setattr(model_store, "SCORES_FILE", tmp_path / "absent" / "model_scores.toml")
    monkeypatch.setattr(model_store, "RUNS_FILE", tmp_path / "absent" / "runs.jsonl")


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scores: Path):
    profiles = tmp_path / "models.toml"
    profiles.write_text(PROFILES)
    monkeypatch.setattr(models, "PROFILES_FILE", profiles)
    server.STATE_DIR.mkdir(parents=True, exist_ok=True)
    (server.STATE_DIR / "active_model.json").write_text(json.dumps({"profile": "granite-8b"}))
    tags, blobs = installed_models()
    store = tmp_path / "ollama-models"
    write_manifests(store, blobs)
    monkeypatch.setenv("OLLAMA_MODELS", str(store))
    ollama = FakeOllama(models=tags, blobs=blobs, store=store).start()
    monkeypatch.setenv("VEGA_BASE_URL", ollama.url + "/v1")
    monkeypatch.setattr(model_store, "free_disk_gb", lambda path=None: 40.0)
    monkeypatch.setattr(resources, "total_memory_gb", lambda: 16.0)
    monkeypatch.setattr(model_store, "registry_size_gb", lambda tag, fetch=None: None)
    runs = tmp_path / "runs.jsonl"
    runs.write_text(json.dumps({
        "mode": "ab", "suite": "fixes", "experiment": "x", "variants": {
            "A": {"model": "vega", "passed": 1, "total": 2},
            "B": {"model": "vega-coder-7b-16k", "passed": 2, "total": 2}}}) + "\n")
    monkeypatch.setattr(model_store, "RUNS_FILE", runs)
    experiments = tmp_path / "experiments.toml"
    experiments.write_text('[[experiment]]\nid = "old"\nhypothesis = "h"\n')
    monkeypatch.setattr(lab, "EXPERIMENTS_FILE", experiments)
    built: list[str] = []
    monkeypatch.setattr(models, "build", lambda profile, target: built.append(target) or 0)
    ollama.built = built
    ollama.experiments = experiments
    yield ollama
    ollama.stop()


# -- catalogue and rules --------------------------------------------------------------------------

def test_shipped_catalogue_parses_with_one_default_and_no_scores(no_scores: None) -> None:
    catalog = model_store.load_catalog()
    default = model_store.shipped_default(catalog)
    assert (default.name, default.tag) == ("granite-8b", "granite4.1:8b")
    assert catalog["coder-7b"].fallback
    assert catalog["qwen3-8b-nothink"].request == {"reasoning_effort": "none"}
    for entry in catalog.values():
        assert entry.params_b < 10
        assert abs(entry.ram_gb - model_store.ram_needed_gb(entry.download_gb)) < 0.06, entry.name
        assert entry.licence
        assert not entry.scores and not entry.measured_on, entry.name
    # results live in the scores file only: the shipped catalogue carries none
    raw = tomllib.loads(model_store.CATALOG_FILE.read_text(encoding="utf-8"))["models"]
    assert not any({"scores", "measured_on"} & set(entry) for entry in raw.values())


def test_scores_merge_from_the_scores_file(scores: Path) -> None:
    catalog = model_store.load_catalog()
    granite = catalog["granite-8b"]
    assert {s: str(v) for s, v in granite.scores.items()} == {
        "fixes": "7/10", "goals": "3/8", "humaneval40": "9/10", "mbpp40": "6/10"}
    assert granite.measured_on == "2026-01-02" and granite.scores["fixes"].experiment == "e1"
    assert granite.tag == "granite4.1:8b" and granite.default  # catalogue fields kept
    assert not catalog["phi4-mini"].scores
    assert "dropped-model" not in catalog  # scores for a model the catalogue dropped: ignored


def test_scores_need_a_date_and_a_missing_file_means_none(scores: Path, tmp_path: Path) -> None:
    scores.write_text('[models.granite-8b.scores]\nfixes = { passed = 1, total = 2 }\n')
    with pytest.raises(ValueError, match="no measured_on"):
        model_store.load_catalog()
    assert model_store.load_scores(tmp_path / "nowhere.toml") == {}
    catalog = model_store.load_catalog(scores_path=tmp_path / "nowhere.toml")
    assert not any(entry.measured for entry in catalog.values())


def test_real_scores_file_is_dated_and_sourced() -> None:
    require_eval_data()  # the real scores file travels with the eval data
    catalog = model_store.load_catalog(scores_path=model_store.scores_file())
    default = model_store.shipped_default(catalog)
    assert set(default.scores) == set(model_store.COMPARE_SUITES)
    for entry in catalog.values():
        if entry.scores:
            assert entry.measured_on
            assert all(score.commit and score.experiment for score in entry.scores.values())


def test_catalogue_rejects_big_models_and_two_defaults(tmp_path: Path) -> None:
    entry = ('[models.{n}]\ntag = "x:{p}b"\nfamily = "x"\nparams_b = {p}\ndownload_gb = 1.0\n'
             'ram_gb = 3.25\nlicence = "MIT"\n{d}\n')
    path = tmp_path / "c.toml"
    path.write_text(entry.format(n="a", p=14, d="default = true"))
    with pytest.raises(ValueError, match="under 10B"):
        model_store.load_catalog(path)
    path.write_text(entry.format(n="a", p=3, d="default = true")
                    + entry.format(n="b", p=4, d="default = true"))
    with pytest.raises(ValueError, match="exactly one"):
        model_store.load_catalog(path)


def test_fit_rule() -> None:
    assert model_store.fit(5.3, 16) == "fits"  # needs 8.6 GB <= 9.6
    assert model_store.fit(9.0, 16) == "tight"  # 13.25 GB <= 13.6 (the 14B: heavy swap)
    assert model_store.fit(10.0, 16) == "won't fit"
    assert model_store.fit(5.3, 8) == "won't fit"
    assert model_store.fit(2.1, 8) == "fits"
    assert model_store.fit(None, 16) == model_store.fit(5.0, None) == "?"
    assert model_store.params_from_tag("qwen2.5-coder:7b-instruct-q4_K_M") == 7
    assert model_store.params_from_tag("phi4-mini:3.8b") == 3.8
    assert model_store.params_from_tag("granite4:micro") is None


# -- list -----------------------------------------------------------------------------------------

def test_inventory_rows_markers_sizes_and_scores(fake: FakeOllama) -> None:
    store = model_store.inventory()
    assert store.ollama_up and store.total_ram_gb == 16.0
    granite = store.row("granite-8b")
    assert granite.active and granite.default and granite.shipped_default and granite.built
    assert granite.fit == "fits" and abs(granite.size_gb - 5.35) < 0.01
    assert granite.scores["fixes"] == "7/10" and granite.scores["goals"] == "3/8"
    sixteen = store.row("coder-7b-16k")
    assert sixteen.scores == {"fixes": "2/2"}  # a named A/B arm in runs.jsonl
    assert store.row("lms").endpoint and store.row("lms").fit == "?"
    phi = store.row("phi4-mini")
    assert phi.kind == "catalogue" and not phi.downloaded and phi.size_gb == 2.5
    smol = store.row("smollm2:1.7b")
    assert smol.kind == "ollama" and smol.params_b == 1.7
    names = [row.name for row in store.rows]
    assert not any(name.startswith("vega") for name in names)  # builds belong to their profile


def test_list_command_shows_fit_markers_and_scores(fake: FakeOllama) -> None:
    result = CliRunner().invoke(cli.app, ["model", "list"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "This machine: 16 GB RAM, 40.0 GB free disk" in result.output
    assert "granite-8b (active, default, shipped)" in result.output
    assert "7/10" in result.output and "fits" in result.output and "MBPP+" in result.output
    assert "Scores: the catalogue's measured results" in result.output
    assert "not downloaded" in result.output and "built" in result.output
    assert CliRunner().invoke(cli.app, ["models"], env={"COLUMNS": "200"}).exit_code == 0


@pytest.mark.parametrize("width", [200, 80])
def test_list_without_scores_has_no_score_columns(fake: FakeOllama, no_scores: None,
                                                  monkeypatch, width: int) -> None:
    from vega import cli as cli_module

    monkeypatch.setattr(cli_module.console, "width", width)
    result = CliRunner().invoke(cli.app, ["models"], env={"COLUMNS": str(width)})
    assert result.exit_code == 0, result.output
    assert "granite-8b (active, default, shipped)" in result.output and "fits" in result.output
    for gone in ("Scores", "MBPP+", "HE+", " fix ", "goal", "runs.jsonl"):
        assert gone not in result.output, gone
    assert "On disk: built = vega-<profile> exists" in result.output


def test_list_is_readable_at_80_columns(fake: FakeOllama, monkeypatch) -> None:
    from vega import cli as cli_module

    monkeypatch.setattr(cli_module.console, "width", 80)
    result = CliRunner().invoke(cli.app, ["model", "list"])
    assert result.exit_code == 0, result.output
    assert "Scores" in result.output and "Ollama model" not in result.output
    assert "fix 7/10" in result.output and "…" not in result.output.split("Vega models")[1].split("RAM:")[0]


def test_list_without_ollama_still_lists_profiles(fake: FakeOllama, monkeypatch) -> None:
    monkeypatch.setenv("VEGA_BASE_URL", "http://127.0.0.1:9/v1")
    store = model_store.inventory()
    assert not store.ollama_up and store.row("granite-8b").size_gb == 5.3  # catalogue size


# -- pull -----------------------------------------------------------------------------------------

def test_pull_refused_when_free_disk_would_drop_below_12_gb(fake: FakeOllama, monkeypatch) -> None:
    monkeypatch.setattr(model_store, "free_disk_gb", lambda path=None: 15.0)
    plan = model_store.plan_pull("llama31-8b")
    assert plan.size_gb == 4.9 and plan.size_source == "catalogue"
    assert "leave 10.1 GB free" in plan.refusal
    result = CliRunner().invoke(cli.app, ["model", "pull", "llama31-8b", "--yes"])
    assert result.exit_code == 1 and "Refused" in result.output and "15.0 GB now" in result.output
    assert fake.calls("POST", "/api/pull") == []


def test_pull_refuses_big_and_unknown_size_models(fake: FakeOllama) -> None:
    assert "under 10B" in model_store.plan_pull("qwen2.5-coder:14b").refusal
    plan = model_store.plan_pull("newthing:3b")
    assert "unknown" in plan.refusal and plan.profile is None
    sized = model_store.plan_pull("newthing:3b", size_lookup=lambda tag: 2.0)
    assert sized.refusal == "" and sized.size_source == "registry"
    assert any("vega model add" in warning for warning in sized.warnings)


def test_pull_streams_progress_adds_catalogue_profile_and_builds(fake: FakeOllama) -> None:
    fake.pull_sizes["phi4-mini:3.8b"] = 2_500_000_000
    plan = model_store.plan_pull("phi4-mini")
    assert plan.refusal == "" and plan.new_profile is not None and not plan.already
    seen: list[model_store.Progress] = []
    model_store.pull(plan, seen.append)
    assert seen[0].status == "pulling manifest" and seen[-1].status == "success"
    assert seen[-1].completed == seen[-1].total == 2_500_000_100
    assert fake.calls("POST", "/api/pull") == [{"model": "phi4-mini:3.8b", "stream": True}]
    assert model_store.finish_pull(plan) == "vega-phi4-mini"
    assert fake.built == ["vega-phi4-mini"]
    assert models.get("phi4-mini").base == "phi4-mini:3.8b"


def test_pull_command_confirms_and_reports_errors(fake: FakeOllama) -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["model", "pull", "gemma3-4b"], input="n\n")
    assert result.exit_code == 1 and "Nothing downloaded" in result.output
    assert "3.3 GB (catalogue)" in result.output and "36.7 GB after" in result.output
    assert fake.calls("POST", "/api/pull") == []
    fake.pull_error = "pull model manifest: file does not exist"
    result = runner.invoke(cli.app, ["model", "pull", "gemma3-4b", "--yes"])
    assert result.exit_code == 1 and "file does not exist" in result.output
    fake.pull_error = ""
    result = runner.invoke(cli.app, ["model", "pull", "gemma3-4b", "--yes"])
    assert result.exit_code == 0, result.output
    assert "Built vega-gemma3-4b" in result.output and "gemma3:4b" in fake.names()


def test_pull_of_downloaded_profile_only_builds(fake: FakeOllama) -> None:
    result = CliRunner().invoke(cli.app, ["model", "pull", "coder-7b"])
    assert result.exit_code == 0, result.output
    assert "already downloaded" in result.output and fake.calls("POST", "/api/pull") == []
    assert fake.built == ["vega-coder-7b"]


# -- rm -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("target", ["granite-8b", "granite4.1:8b", "vega", "vega-granite-8b"])
def test_rm_refuses_the_active_default_and_its_weights(fake: FakeOllama, target: str) -> None:
    plan = model_store.plan_remove(target)
    assert plan.refusal and plan.names == []
    result = CliRunner().invoke(cli.app, ["model", "rm", target, "--yes"])
    assert result.exit_code == 1 and "Refused" in result.output
    assert fake.calls("DELETE", "/api/delete") == []


def test_rm_refuses_a_non_active_default(fake: FakeOllama) -> None:
    (server.STATE_DIR / "active_model.json").write_text(json.dumps({"profile": "coder-7b"}))
    assert "default" in model_store.plan_remove("granite-8b").refusal
    assert "active" in model_store.plan_remove("coder-7b").refusal


def test_rm_profile_keeps_a_base_another_build_uses(fake: FakeOllama) -> None:
    plan = model_store.plan_remove("coder-7b")
    assert plan.names == ["vega-coder-7b:latest"]
    assert any("vega-coder-7b-16k" in note for note in plan.kept)
    assert plan.freed.exact and plan.freed.gb < 0.001  # only its own small blobs
    assert QWEN in plan.freed.shared_with


def test_rm_confirmation_flow_and_shared_blobs(fake: FakeOllama) -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["model", "rm", "coder-7b-16k"], input="n\n")
    assert result.exit_code == 1 and "Nothing deleted" in result.output
    assert fake.calls("DELETE", "/api/delete") == []
    assert runner.invoke(cli.app, ["model", "rm", "coder-7b", "--yes"]).exit_code == 0
    plan = model_store.plan_remove("coder-7b-16k")  # now the last user of the qwen weights
    assert plan.names == ["vega-coder-7b-16k:latest", QWEN]
    # the weights, its system blob and both configs; the template is shared with Granite's
    assert abs(plan.freed.gb - (W_QWEN + 200 + 100 + 100) / GB) < 1e-6
    result = runner.invoke(cli.app, ["model", "rm", "coder-7b-16k"], input="y\n")
    assert result.exit_code == 0, result.output
    assert "frees 4.7 GB" in result.output
    assert [b["model"] for b in fake.calls("DELETE", "/api/delete")] == [
        "vega-coder-7b:latest", "vega-coder-7b-16k:latest", QWEN]
    assert "sha256:qwen" in fake.deleted_blobs and "sha256:granite" not in fake.deleted_blobs
    assert "sha256:template" not in fake.deleted_blobs  # Granite's models still reference it
    assert models.get("coder-7b-16k")  # the profile stays; `vega model remove` drops it


def test_rm_plain_ollama_model_and_estimate_without_manifests(fake: FakeOllama) -> None:
    plan = model_store.plan_remove("smollm2:1.7b")
    assert plan.names == ["smollm2:1.7b"] and abs(plan.freed.gb - 1.8000011) < 1e-6
    estimate = model_store.space_freed(["vega-coder-7b:latest"], fake.models, None)
    assert not estimate.exact and estimate.gb == 0 and QWEN in estimate.shared_with
    alone = model_store.space_freed(["smollm2:1.7b"], fake.models, None)
    assert abs(alone.gb - 1.800001) < 0.001
    assert "not downloaded" in model_store.plan_remove("gemma3-4b").refusal


# -- recommend ------------------------------------------------------------------------------------

def test_recommend_by_machine(scores: Path) -> None:
    sixteen = model_store.recommend(16.0)
    assert sixteen.entry.name == "granite-8b" and sixteen.fit == "fits"
    assert "fix 7/10" in sixteen.reason and "measured 2026-01-02" in sixteen.reason
    eight = model_store.recommend(8.0)
    assert eight.entry.params_b < 5 and "untested" in eight.reason
    assert model_store.recommend(3.0) is None


def test_recommend_without_scores_names_the_default(no_scores: None) -> None:
    sixteen = model_store.recommend(16.0)
    assert sixteen.entry.name == "granite-8b"
    assert sixteen.reason == "granite-8b is Vega's default model and fits 16 GB."
    eight = model_store.recommend(8.0)
    assert eight.entry.params_b < 5 and eight.fit == "fits"
    assert eight.reason == (f"{eight.entry.name} fits 8 GB; Vega's default, granite-8b, needs "
                            "about 8.6 GB.")
    for reason in (sixteen.reason, eight.reason):
        assert "untested" not in reason and "queue" not in reason


def test_recommend_command_without_scores(fake: FakeOllama, no_scores: None) -> None:
    result = CliRunner().invoke(cli.app, ["model", "recommend"])
    assert result.exit_code == 0, result.output
    assert "granite-8b is Vega's default model and fits 16 GB." in result.output
    assert "measured" not in result.output


def test_recommend_command(fake: FakeOllama) -> None:
    result = CliRunner().invoke(cli.app, ["model", "recommend"])
    assert result.exit_code == 0, result.output
    assert "granite-8b" in result.output and "downloaded" in result.output


# -- watch and queue ------------------------------------------------------------------------------

def test_watch_offline_lists_catalogue_and_local_models_without_profiles(fake: FakeOllama) -> None:
    found = {c.tag: c for c in model_store.watch_offline()}
    assert "smollm2:1.7b" in found and found["smollm2:1.7b"].source == "ollama"
    assert found["phi4-mini:3.8b"].source == "catalogue"
    assert GRANITE not in found and QWEN not in found and "vega:latest" not in found
    result = CliRunner().invoke(cli.app, ["model", "watch"], env={"COLUMNS": "200"})
    assert result.exit_code == 0 and "smollm2:1.7b" in result.output
    assert not [p for m, p, _ in fake.requests if not p.startswith("/api/tags")]


SEARCH = ('<a href="/library/granite4.1">granite4.1</a><a href="/library/granite5">granite5</a>'
          '<a href="/library/qwen3">not this family</a>')
TAGS = {
    "granite4.1": '<a href="/library/granite4.1:8b">8b</a><a href="/library/granite4.1:3b">3b</a>',
    "granite5": ('<a href="/library/granite5:latest">latest</a><a href="/library/granite5:8b">8b'
                 '</a><a href="/library/granite5:20b">20b</a>'),
}


def test_watch_online_is_mocked_and_marks_new_tags(fake: FakeOllama) -> None:
    fetched: list[str] = []

    def fetch(url: str) -> str:
        fetched.append(url)
        if "/search?q=" in url:
            return SEARCH
        return TAGS[url.split("/library/")[1].removesuffix("/tags")]

    found, errors = model_store.watch_online(("granite",), fetch=fetch)
    assert errors == []
    assert [(c.tag, c.new) for c in found] == [("granite4.1:3b", True), ("granite5:8b", True)]
    assert fetched[0] == "https://ollama.com/search?q=granite"
    again, _ = model_store.watch_online(("granite",), fetch=fetch)
    assert [c.new for c in again] == [False, False]

    def broken(url: str) -> str:
        raise OSError("offline")

    assert model_store.watch_online(("gemma",), fetch=broken) == ([], ["gemma: offline"])


def test_queue_writes_the_standard_comparison_once(fake: FakeOllama) -> None:
    profile, written, skipped = model_store.queue_comparison("phi4-mini")
    assert profile == "phi4-mini" and skipped == []
    assert written == ["humaneval40-model-phi4-mini", "mbpp40-model-phi4-mini",
                       "goals-model-phi4-mini", "fixes-model-phi4-mini"]
    experiments = {e.id: e for e in lab.load(fake.experiments)}
    assert experiments["goals-model-phi4-mini"].ab_model == "phi4-mini"
    assert experiments["goals-model-phi4-mini"].suite == "goals" and "old" in experiments
    assert models.get("phi4-mini").base == "phi4-mini:3.8b"
    assert model_store.queue_comparison("phi4-mini")[1] == []
    result = CliRunner().invoke(cli.app, ["model", "queue", "smollm2:1.7b"])
    assert result.exit_code == 0 and "vega lab run" in result.output
    assert models.get("smollm2-1-7b").base == "smollm2:1.7b"
    with pytest.raises(ValueError, match="current default"):
        model_store.queue_comparison("granite-8b")
    with pytest.raises(ValueError, match="under 10B"):
        model_store.queue_comparison("qwen2.5-coder:14b")
    tomllib.loads(fake.experiments.read_text())
    assert fake.calls("POST", "/api/pull") == []  # queueing downloads and runs nothing


def test_watch_command_online_flag_uses_the_library_check(fake: FakeOllama, monkeypatch) -> None:
    calls: list[bool] = []

    def online(*args, **kwargs):
        calls.append(True)
        return [model_store.Candidate("granite5:8b", "online", None, 8.0, "granite5-8b")], []

    monkeypatch.setattr(model_store, "watch_online", online)
    runner = CliRunner()
    offline = runner.invoke(cli.app, ["model", "watch"], env={"COLUMNS": "200"})
    assert "granite5" not in offline.output and calls == []
    result = runner.invoke(cli.app, ["model", "watch", "--online", "--queue", "granite5:8b"],
                           env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert calls == [True] and "granite5:8b" in result.output and "new since" in result.output
    assert "fixes-model-granite5-8b" in result.output
    assert models.get("granite5-8b").base == "granite5:8b"
