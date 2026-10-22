"""The `vegacode` wheel works without a checkout: the default model profiles ship inside the
package, while a checkout (and VEGA_MODELS_FILE) keep reading the files they always read."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from vega import models, server

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
KEYWORDS_MD = ROOT / "docs" / "release" / "keywords.md"
# The public tree (built by the release builder) ships README.public.md as README.md and
# release/github/ as .github/; this repo has release/ and keeps both under their source names.
_PUBLIC_TREE = not (ROOT / "release").is_dir() and (ROOT / ".github" / "scripts").is_dir()
PUBLIC_README = (ROOT / "README.md" if _PUBLIC_TREE
                 else ROOT / "docs" / "release" / "README.public.md")
CHECK_SCRIPT = (ROOT / (".github" if _PUBLIC_TREE else "release/github") / "scripts"
                / "check_keywords.py")
# The self-build eval snapshots only src/, tests/ and a few root files; the release files are not in
# it.
needs_release_files = pytest.mark.skipif(
    not (KEYWORDS_MD.is_file() and PUBLIC_README.is_file() and CHECK_SCRIPT.is_file()),
    reason="release files (docs/release/, release/github/) not in this tree")


def _python(code: str, *, cwd: Path, pythonpath: Path, home: Path,
            extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items()
           if key not in ("PYTHONPATH", "VEGA_MODELS_FILE", "VEGA_HOME")}
    env.update({"PYTHONPATH": str(pythonpath), "VEGA_HOME": str(home), **(extra_env or {})})
    return subprocess.run([sys.executable, "-c", code], check=False, cwd=cwd, env=env, text=True,
                          capture_output=True, timeout=120)


@pytest.fixture
def installed(tmp_path: Path) -> Path:
    """The package laid out as the wheel installs it: site-packages/vega with
    default_models.toml inside it and no repository around it."""
    site = tmp_path / "lib" / "python3" / "site-packages"
    shutil.copytree(ROOT / "src" / "vega", site / "vega",
                    ignore=shutil.ignore_patterns("__pycache__"))
    target = PYPROJECT["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]["models.toml"]
    shutil.copyfile(ROOT / "models.toml", site / target)
    return site


def test_installed_package_reads_packaged_profiles_without_writing(installed: Path,
                                                                   tmp_path: Path) -> None:
    home, cwd = tmp_path / "vh", tmp_path / "empty"
    cwd.mkdir()
    result = _python("from vega import models; d, p = models.load(); "
                     "print(models.PROFILES_FILE); print(d, len(p))",
                     cwd=cwd, pythonpath=installed, home=home)
    assert result.returncode == 0, result.stderr
    where, summary = result.stdout.splitlines()
    assert Path(where) == home / "models.toml"
    expected = tomllib.loads((ROOT / "models.toml").read_text(encoding="utf-8"))
    assert summary == f"{expected['default']} {len(expected['profiles'])}"
    assert not (home / "models.toml").exists()  # reading never copies the defaults out


def test_installed_package_first_edit_writes_user_profiles(installed: Path, tmp_path: Path) -> None:
    home, cwd = tmp_path / "vh", tmp_path / "empty"
    cwd.mkdir()
    result = _python("from vega import models; models.add_profile('extra', base='foo:1b'); "
                     "print(sorted(models.load()[1]))",
                     cwd=cwd, pythonpath=installed, home=home)
    assert result.returncode == 0, result.stderr
    written = (home / "models.toml").read_text(encoding="utf-8")
    shipped = (ROOT / "models.toml").read_text(encoding="utf-8")
    assert written.startswith(shipped.rstrip("\n"))  # comments and order kept byte for byte
    assert "[profiles.extra]" in written and "'extra'" in result.stdout
    assert shipped == (installed / "vega" / "default_models.toml").read_text(encoding="utf-8")


def test_checkout_reads_repo_root_profiles(tmp_path: Path) -> None:
    result = _python("from vega import models; print(models.PROFILES_FILE)",
                     cwd=tmp_path, pythonpath=ROOT / "src", home=tmp_path / "vh")
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == ROOT / "models.toml"


def test_models_file_override_wins(installed: Path, tmp_path: Path) -> None:
    custom = tmp_path / "mine.toml"
    custom.write_text('default = "a"\n[profiles.a]\nbase = "x:1b"\n', encoding="utf-8")
    for path in (ROOT / "src", installed):
        result = _python("from vega import models; print(models.PROFILES_FILE, "
                         "sorted(models.load()[1]))", cwd=tmp_path, pythonpath=path,
                         home=tmp_path / "vh", extra_env={"VEGA_MODELS_FILE": str(custom)})
        assert result.returncode == 0, result.stderr
        assert result.stdout.split() == [str(custom), "['a']"]


def test_missing_user_file_without_packaged_defaults_still_errors(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "state" / "models.toml"
    monkeypatch.setattr(models, "PROFILES_FILE", path)
    monkeypatch.setattr(models, "USER_PROFILES_FILE", path)
    monkeypatch.setattr(models, "packaged_profiles_text", lambda: None)
    monkeypatch.setattr(server, "STATE_DIR", tmp_path / "state")
    with pytest.raises(ValueError, match="No model profiles file"):
        models.load()


def test_missing_explicit_file_does_not_fall_back(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "PROFILES_FILE", tmp_path / "gone.toml")
    monkeypatch.setattr(models, "packaged_profiles_text", lambda: 'default = "a"\n')
    with pytest.raises(ValueError, match="No model profiles file"):
        models.load()


def test_distribution_metadata() -> None:
    project = PYPROJECT["project"]
    assert project["name"] == "vegacode"
    assert project["scripts"] == {"vega": "vega.cli:app", "vegacode": "vega.cli:app"}
    assert project["readme"] == "README.md" and project["requires-python"] == ">=3.11"
    assert all(url.startswith("https://github.com/isaacfrett/vegacode")
               for url in project["urls"].values())
    assert project["license"] == "AGPL-3.0-only"
    assert project["license-files"] == ["LICENSE", "NOTICE"]
    wheel = PYPROJECT["tool"]["hatch"]["build"]["targets"]
    assert wheel["wheel"]["force-include"] == {"models.toml": "vega/default_models.toml"}
    assert models.PACKAGED_PROFILES == "default_models.toml"
    assert not {"tests", "evals", "docs", "training"} & set(wheel["sdist"]["only-include"])
    assert {"LICENSE", "NOTICE"} <= set(wheel["sdist"]["only-include"])


@needs_release_files
def test_release_keyword_check_passes() -> None:
    """The keyword check (run by the public CI) agrees with the files as they are."""
    result = subprocess.run([sys.executable, str(CHECK_SCRIPT), "--readme", str(PUBLIC_README)],
                            check=False, cwd=ROOT,
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@needs_release_files
def test_keyword_check_reports_drift() -> None:
    """The check fails when a required phrase or topic goes missing."""
    sys.path.insert(0, str(CHECK_SCRIPT.parent))
    try:
        import check_keywords
    finally:
        sys.path.remove(str(CHECK_SCRIPT.parent))
    keywords = KEYWORDS_MD.read_text(encoding="utf-8")
    readme = PUBLIC_README.read_text(encoding="utf-8")
    project = dict(PYPROJECT["project"])
    assert check_keywords.check(readme, project, keywords) == []
    problems = check_keywords.check(readme.replace("Ollama", "a runtime"), project, keywords)
    assert "README first paragraph: missing 'Ollama'" in problems
    project["keywords"] = [k for k in project["keywords"] if k != "local LLM"]
    assert any("'local-llm'" in p for p in check_keywords.check(readme, project, keywords))
    assert any("personal path" in p for p in
               check_keywords.check(readme + "\nsee /Users/someone/x\n", PYPROJECT["project"],
                                    keywords))
