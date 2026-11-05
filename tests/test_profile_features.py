"""Per-profile default features (constrained decoding for the 7B profiles)."""

from __future__ import annotations

import os

from vega import features, models


def test_repo_7b_profiles_default_to_constrained() -> None:
    _, profiles = models.load()
    assert profiles["coder-7b"].features_on == ["constrained"]
    assert profiles["coder-14b"].features_on == []


def test_defaults_apply_follow_the_model_and_never_override_the_user(monkeypatch) -> None:
    monkeypatch.setattr(models, "active_profile", lambda: "coder-7b")
    monkeypatch.setattr(models, "_APPLIED", None)
    assert models.apply_feature_defaults("vega") == ["constrained"]
    assert features.on("constrained")
    assert models.apply_feature_defaults("vega-coder-14b") == []  # 14B profile: no default
    assert not features.on("constrained") and features.ENV_ON not in os.environ
    monkeypatch.setenv(features.ENV_ON, "")  # the user's own choice (here: nothing on)
    assert models.apply_feature_defaults("vega") == [] and not features.on("constrained")


def test_unprofiled_models_get_no_defaults(monkeypatch) -> None:
    monkeypatch.setattr(models, "_APPLIED", None)
    assert models.apply_feature_defaults("llama3.1:8b") == []


def test_profile_request_options_reach_every_request(tmp_path, monkeypatch) -> None:
    from vega import models
    from vega.agent import CodingAgent

    toml = tmp_path / "models.toml"
    toml.write_text('default = "a"\n[profiles.a]\nbase = "x"\n'
                    '[profiles.nothink]\nbase = "y"\nrequest = { reasoning_effort = "none" }\n')
    monkeypatch.setattr(models, "PROFILES_FILE", toml)
    assert models.request_options("vega-nothink") == {"reasoning_effort": "none"}
    assert models.request_options("vega-a") == {} and models.request_options("other") == {}
    monkeypatch.setenv("VEGA_MODEL", "vega-nothink")
    assert CodingAgent(tmp_path).sampling["reasoning_effort"] == "none"


def test_a_profile_can_switch_a_default_off_and_never_overrides_the_user(monkeypatch) -> None:
    _, profiles = models.load()
    assert profiles["granite-8b"].features_off == ["plan"]  # measured 2026-10-18
    monkeypatch.setattr(models, "_APPLIED", None)
    monkeypatch.setattr(models, "_APPLIED_OFF", None)
    monkeypatch.delenv(features.ENV, raising=False)
    assert models.apply_feature_defaults("vega-granite-8b") == ["-plan"]
    assert not features.on("plan")
    assert models.apply_feature_defaults("vega-coder-7b") == ["constrained"]  # switching back
    assert features.on("plan") and features.ENV not in os.environ
    monkeypatch.setenv(features.ENV, "")  # the user's own choice: nothing off
    assert models.apply_feature_defaults("vega-granite-8b") == []
    assert features.on("plan")


def test_profile_off_defaults_are_added_to_the_users_off_list(tmp_path, monkeypatch) -> None:
    # 2026-11-04: VEGA_FEATURES_OFF=x replaced Granite's `plan` default, so A/B arms changed two things
    toml = tmp_path / "models.toml"
    toml.write_text('default = "g"\n[profiles.g]\nbase = "x"\nfeatures_off = ["plan"]\n')
    monkeypatch.setattr(models, "PROFILES_FILE", toml)
    monkeypatch.setattr(models, "active_profile", lambda: "g")
    for name in ("_APPLIED_OFF", "_USER_OFF"):
        monkeypatch.setattr(models, name, None)
    assert models.apply_feature_defaults("vega") == ["-plan"] and not features.on("plan")
    monkeypatch.setenv(features.ENV, "lint")  # an A/B arm's own switch
    assert models.apply_feature_defaults("vega") == ["-plan"]
    assert not features.on("plan") and not features.on("lint")
    assert models.apply_feature_defaults("vega") == ["-plan"]  # idempotent: env now holds both
    assert os.environ[features.ENV] == "lint,plan"
    monkeypatch.setenv("VEGA_PROFILE_FEATURES_OFF", "0")
    monkeypatch.setenv(features.ENV, "lint")
    assert models.apply_feature_defaults("vega") == [] and features.on("plan")
