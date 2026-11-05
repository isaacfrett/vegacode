"""`vega update` and `vega uninstall` (vega.lifecycle) against fake installs in temp folders.

Nothing here reaches the network, PyPI, Ollama, launchctl or the real home folder: the Host's
outside-world callables are replaced, and every path lives under tmp_path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from vega import _bootstrap as boot
from vega import cli, lifecycle

PY = "python3.12"


def _file(path: Path, text: str = "x", mode: int = 0o644) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    return path


def _env_with_package(prefix: Path, version: str = "0.3.1") -> tuple[Path, Path]:
    """A fake environment with vegacode in its site-packages: (package dir, dist-info)."""
    site = prefix / "lib" / PY / "site-packages"
    package = site / "vega"
    _file(package / "__init__.py", "# vega\n" * 100)
    info = site / f"vegacode-{version}.dist-info"
    _file(info / "METADATA", f"Name: vegacode\nVersion: {version}\n")
    _file(info / "RECORD", "vega/__init__.py,,\n"
          f"vegacode-{version}.dist-info/METADATA,,\nvegacode-{version}.dist-info/RECORD,,\n"
          "../../../bin/vega,,\n")
    _file(prefix / "bin" / "python", "#!/bin/sh\n", 0o755)
    _file(prefix / "bin" / "vega", "#!/bin/sh\n", 0o755)
    (prefix / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    return package, info


class Fake:
    """A machine with one vegacode install; records every command and HTTP request."""

    def __init__(self, tmp_path: Path, kind: str = "launcher", *, version: str = "0.3.1",
                 pypi: str = "0.3.2", env: dict[str, str] | None = None,
                 tags: list[dict[str, Any]] | None = None, system: str = "Darwin") -> None:
        self.home = tmp_path / "home"
        self.home.mkdir(parents=True, exist_ok=True)
        self.env = {"HOME": str(self.home), "SHELL": "/bin/zsh", **(env or {})}
        self.commands: list[list[str]] = []
        self.captures: list[list[str]] = []
        self.fetched: list[str] = []
        self.deleted: list[str] = []
        self.pypi = pypi
        self.new_version = pypi  # what `vega --version` says after an upgrade
        self.run_code = 0
        self.tags = tags
        base = Path("/usr/local/fake-python")
        direct = None
        user_site = self.home / ".local" / "lib" / PY / "site-packages"
        if kind == "launcher":
            prefix = boot.venv_dir(self.env, self.home)
        elif kind == "pipx":
            prefix = self.home / ".local" / "pipx" / "venvs" / "vegacode"
        elif kind == "uv":
            prefix = self.home / ".local" / "share" / "uv" / "tools" / "vegacode"
        elif kind == "venv":
            prefix = tmp_path / "project" / ".venv"
        elif kind == "user":
            prefix = base = tmp_path / "python"
        elif kind == "dev":
            prefix = tmp_path / "checkout" / ".venv"
        else:
            raise ValueError(kind)
        package, info = _env_with_package(prefix, version)
        if kind == "pipx":
            _file(prefix / "pipx_metadata.json", "{}")
        if kind == "uv":
            _file(prefix / "uv-receipt.toml", "[tool]\n")
        if kind == "user":
            site = user_site
            package = site / "vega"
            _file(package / "__init__.py")
            info = site / f"vegacode-{version}.dist-info"
            _file(info / "RECORD", "vega/__init__.py,,\n")
        if kind == "dev":
            checkout = tmp_path / "checkout"
            _file(checkout / "pyproject.toml", "[project]\nname = 'vegacode'\n")
            package = checkout / "src" / "vega"
            _file(package / "__init__.py")
            direct = {"url": checkout.as_uri(), "dir_info": {"editable": True}}
        self.prefix = prefix
        self.host = lifecycle.Host(
            env=self.env, home=self.home, system=system, prefix=prefix,
            base_prefix=base if kind == "user" else Path("/usr/local/fake-python"),
            executable=prefix / "bin" / "python", package_dir=package, version=version,
            dist_info=info, direct_url=direct, user_site=user_site, python_version="3.12",
            run=self.run, capture=self.capture, fetch=self.fetch,
            which=lambda name: f"/opt/bin/{name}", ollama_tags=lambda: self.tags,
            ollama_delete=self.deleted.append, ollama_blobs=lambda: None)

    def run(self, argv: list[str]) -> int:
        self.commands.append(argv)
        return self.run_code

    def capture(self, argv: list[str]) -> tuple[int, str]:
        self.captures.append(argv)
        return 0, f"vegacode {self.new_version}\n"

    def fetch(self, url: str) -> bytes:
        self.fetched.append(url)
        return json.dumps({"info": {"name": "vegacode", "version": self.pypi}}).encode()


def _no(question: str, default: bool) -> bool:
    raise AssertionError(f"asked: {question}")


# --- detection ----------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["launcher", "pipx", "uv", "venv", "user", "dev"])
def test_detects_each_install_type(tmp_path: Path, kind: str) -> None:
    fake = Fake(tmp_path, kind)
    install = lifecycle.detect(fake.host)
    assert install.kind == kind
    assert install.version == "0.3.1"
    if kind == "dev":
        assert install.source == tmp_path / "checkout"


def test_a_checkout_run_without_installing_is_dev(tmp_path: Path) -> None:
    fake = Fake(tmp_path, "venv")
    checkout = tmp_path / "repo"
    _file(checkout / "pyproject.toml")
    fake.host.package_dir = checkout / "src" / "vega"
    fake.host.direct_url = None
    install = lifecycle.detect(fake.host)
    assert install.kind == "dev" and install.source == checkout


def test_system_pip_is_the_fallback(tmp_path: Path) -> None:
    fake = Fake(tmp_path, "venv")
    fake.host.base_prefix = fake.host.prefix
    fake.host.user_site = tmp_path / "elsewhere"
    assert lifecycle.detect(fake.host).kind == "system"


# --- versions and PyPI --------------------------------------------------------------------------

@pytest.mark.parametrize(("new", "old", "newer"), [
    ("0.3.2", "0.3.1", True), ("0.3.1", "0.3.1", False), ("0.10.0", "0.9.9", True),
    ("1.0", "1.0.0", False), ("1.0", "1.0rc1", True), ("1.0rc1", "1.0b2", True),
    ("1.0a1", "1.0.dev3", True), ("1.0.post1", "1.0", True), ("0.3.1", "0.3.2", False),
    ("junk", "0.3.1", False),
])
def test_version_comparison(new: str, old: str, newer: bool) -> None:
    assert boot.version_newer(new, old) is newer


def test_latest_version_reads_the_mocked_pypi_json(tmp_path: Path) -> None:
    fake = Fake(tmp_path, pypi="0.4.0")
    assert lifecycle.latest_version(fake.host) == "0.4.0"
    assert fake.fetched == [lifecycle.PYPI_URL]


def test_pypi_unreachable_or_garbled_is_a_clear_error(tmp_path: Path) -> None:
    import urllib.error

    fake = Fake(tmp_path)

    def offline(url: str) -> bytes:
        raise urllib.error.URLError("no network")
    fake.host.fetch = offline
    with pytest.raises(lifecycle.UpdateError, match="Could not reach PyPI"):
        lifecycle.latest_version(fake.host)
    fake.host.fetch = lambda url: b"<html>"
    with pytest.raises(lifecycle.UpdateError, match="could not be read"):
        lifecycle.latest_version(fake.host)


# --- vega update --------------------------------------------------------------------------------

def test_update_refuses_in_a_dev_checkout(tmp_path: Path) -> None:
    fake = Fake(tmp_path, "dev")
    out: list[str] = []
    assert lifecycle.update(fake.host, check=False, yes=True, ask=_no, interactive=True,
                            say=out.append) == 1
    assert "source checkout" in out[0] and "git" in out[0]
    assert fake.fetched == [] and fake.commands == []


def test_update_check_reports_and_changes_nothing(tmp_path: Path) -> None:
    fake = Fake(tmp_path, "launcher", pypi="0.3.2")
    out: list[str] = []
    assert lifecycle.update(fake.host, check=True, yes=False, ask=_no, interactive=True,
                            say=out.append) == 0
    text = "\n".join(out)
    assert "vegacode 0.3.1" in text and "vegacode 0.3.2" in text and "0.3.1 -> 0.3.2" in text
    assert fake.commands == []


def test_update_when_current(tmp_path: Path) -> None:
    fake = Fake(tmp_path, "pipx", pypi="0.3.1")
    out: list[str] = []
    assert lifecycle.update(fake.host, check=False, yes=True, ask=_no, interactive=True,
                            say=out.append) == 0
    assert "up to date" in out[-1] and fake.commands == []


@pytest.mark.parametrize(("kind", "expected"), [
    ("launcher", ["{py}", "-m", "pip", "install", "--upgrade", "--disable-pip-version-check",
                  "vegacode==0.3.2"]),
    ("venv", ["{py}", "-m", "pip", "install", "--upgrade", "--disable-pip-version-check",
              "vegacode==0.3.2"]),
    ("user", ["{py}", "-m", "pip", "install", "--upgrade", "--disable-pip-version-check",
              "--user", "vegacode==0.3.2"]),
    ("pipx", ["/opt/bin/pipx", "install", "--force", "vegacode==0.3.2"]),
    # the same Python: --force alone rebuilds on the first Python uv finds (macOS 3.9 there
    # left only the launcher, 2026-11-04 lifecycle test)
    ("uv", ["/opt/bin/uv", "tool", "install", "--force", "--python", "3.12",
            "vegacode==0.3.2"]),
])
def test_update_upgrades_with_the_matching_installer_and_verifies(
        tmp_path: Path, kind: str, expected: list[str]) -> None:
    fake = Fake(tmp_path, kind)
    asked: list[str] = []

    def ask(question: str, default: bool) -> bool:
        asked.append(question)
        return True
    out: list[str] = []
    assert lifecycle.update(fake.host, check=False, yes=False, ask=ask, interactive=True,
                            say=out.append) == 0
    assert asked == ["Update vegacode 0.3.1 -> 0.3.2?"]
    assert fake.commands == [[part.replace("{py}", str(fake.host.executable))
                              for part in expected]]
    assert fake.captures and fake.captures[-1][-1] == "--version"
    assert "Updated to vegacode 0.3.2" in out[-1]


def test_update_reports_a_version_that_did_not_change(tmp_path: Path) -> None:
    fake = Fake(tmp_path)
    fake.new_version = "0.3.1"
    out: list[str] = []
    assert lifecycle.update(fake.host, check=False, yes=True, ask=_no, interactive=False,
                            say=out.append) == 1
    assert "not 0.3.2" in out[-1]


def test_update_refused_or_without_a_terminal_changes_nothing(tmp_path: Path) -> None:
    fake = Fake(tmp_path)
    out: list[str] = []
    assert lifecycle.update(fake.host, check=False, yes=False, ask=lambda q, d: False,
                            interactive=True, say=out.append) == 1
    assert lifecycle.update(fake.host, check=False, yes=False, ask=_no, interactive=False,
                            say=out.append) == 1
    assert fake.commands == [] and "--yes" in out[-1]


# --- PATH lines ---------------------------------------------------------------------------------

def test_rc_removal_touches_only_vega_marked_lines(tmp_path: Path) -> None:
    line = 'export PATH="$HOME/.local/bin:$PATH"'
    text = ("# mine\n" + line + "\nalias ll='ls -l'\n\n" + boot.PATH_MARK + "\n" + line + "\n"
            "export EDITOR=vim\n")
    new, removed = lifecycle.strip_path_lines(text)
    assert removed == [boot.PATH_MARK, line]
    assert new == "# mine\n" + line + "\nalias ll='ls -l'\nexport EDITOR=vim\n"
    # a marker whose next line the user changed is left alone
    edited = boot.PATH_MARK + '\nexport PATH="$HOME/bin:$PATH"\n'
    assert lifecycle.strip_path_lines(edited) == (edited, [])


def test_rc_removal_undoes_exactly_what_the_launcher_wrote(tmp_path: Path) -> None:
    home = tmp_path / "home"
    original = "# my zshrc\nsetopt autocd\n"
    _file(home / ".zshrc", original)
    written = boot.add_to_path(home / ".local" / "bin", {"SHELL": "/bin/zsh", "PATH": "/usr/bin"},
                               home, "Darwin")
    assert written == home / ".zshrc"
    fake = Fake(tmp_path)
    group = lifecycle.path_group(fake.host)
    assert group is not None and group.default is True
    group.action()
    assert (home / ".zshrc").read_text(encoding="utf-8") == original


def test_fish_file_written_by_vega_is_deleted_when_empty(tmp_path: Path) -> None:
    fake = Fake(tmp_path)
    boot.add_to_path(fake.home / ".local" / "bin", {"SHELL": "/usr/bin/fish", "PATH": ""},
                     fake.home, "Linux")
    fish = fake.home / ".config/fish/conf.d/vega.fish"
    assert fish.exists()
    lifecycle.path_group(fake.host).action()
    assert not fish.exists()


def test_path_line_is_kept_by_default_when_other_commands_use_the_folder(tmp_path: Path) -> None:
    fake = Fake(tmp_path)
    _file(fake.home / ".zshrc", f"\n{boot.PATH_MARK}\n{lifecycle.PATH_LINES[0]}\n")
    _file(fake.home / ".local/bin/uv", mode=0o755)
    group = lifecycle.path_group(fake.host)
    assert group.default is False and "uv" in group.note


# --- vega uninstall -----------------------------------------------------------------------------

MODEL_TAGS = [
    {"name": "granite4.1:8b", "size": 5_300_000_000},
    {"name": "vega:latest", "size": 5_300_000_000,
     "details": {"parent_model": "granite4.1:8b"}},
    {"name": "llama3.2:3b", "size": 2_000_000_000},  # the user's own: never listed
]


def _launcher_machine(tmp_path: Path) -> Fake:
    """The stock-Mac path: launcher venv, links, a marked PATH line, a LaunchAgent, ~/.vega,
    granite pulled by first-run setup, and the Python 3.9 launcher copy."""
    agents = tmp_path / "LaunchAgents"
    fake = Fake(tmp_path, "launcher", tags=MODEL_TAGS,
                env={lifecycle.LAUNCH_AGENTS_ENV: str(agents)})
    links = fake.home / ".local" / "bin"
    links.mkdir(parents=True)
    for name in boot.COMMANDS:
        (links / name).symlink_to(fake.prefix / "bin" / "vega")
    _file(fake.home / ".zshrc", "# mine\n")
    boot.add_to_path(links, {"SHELL": "/bin/zsh", "PATH": "/usr/bin"}, fake.home, "Darwin")
    _file(agents / "dev.vega.research.plist", "<plist/>")
    _file(agents / "com.other.thing.plist", "<plist/>")
    _file(fake.home / ".vega" / "setup.json", json.dumps({"profile": "granite-8b",
                                                          "complete": True}))
    _file(fake.home / ".vega" / "sessions" / "a.jsonl", "x" * 5000)
    old = fake.home / "Library/Python/3.9/lib/python/site-packages"
    _file(old / "vega" / "_launch.py", "x" * 2000)
    _file(old / "vega" / "native" / "macos" / "DesignRender.swift", "x")
    _file(old / "vegacode-0.3.1.dist-info/RECORD",
          "vega/_launch.py,,\nvega/native/macos/DesignRender.swift,,\n"
          "vegacode-0.3.1.dist-info/RECORD,,\n../../../bin/vega,,\n")
    _file(fake.home / "Library/Python/3.9/bin/vega", "#!/usr/bin/python3\nvega._launch\n", 0o755)
    return fake


def test_dry_run_lists_every_group_with_sizes_and_changes_nothing(tmp_path: Path) -> None:
    fake = _launcher_machine(tmp_path)
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    out: list[str] = []
    assert lifecycle.uninstall(fake.host, dry_run=True, yes=False, remove_data=False,
                               remove_models=False, ask=_no, ask_text=_no, interactive=True,
                               say=out.append) == 0
    text = "\n".join(out)
    assert "1. Vega itself: vegacode 0.3.1 (the launcher's own environment)" in text and "~/.local/share/vega/venv" in text
    assert "~/.local/bin/vega, ~/.local/bin/vegacode  (links)" in text
    assert "Library/Python/3.9/lib/python/site-packages/vegacode-0.3.1.dist-info" in text
    assert "~/.zshrc: export PATH=\"$HOME/.local/bin:$PATH\"" in text
    assert "dev.vega.research.plist" in text and "com.other.thing" not in text
    assert "~/.vega" in text and "[default: keep]" in text
    assert "5.3 GB  granite4.1:8b" in text and "vega:latest  (built on granite4.1:8b" in text
    assert "frees 5.3 GB" in text and "[default: keep]" in text
    assert "llama3.2" not in text
    assert "never removed" in text and "Dry run: nothing was changed." in text
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before
    assert fake.commands == [] and fake.deleted == []


def test_uninstall_yes_removes_vega_keeps_data_and_models_and_is_idempotent(
        tmp_path: Path) -> None:
    fake = _launcher_machine(tmp_path)
    out: list[str] = []
    kwargs: dict[str, Any] = {"dry_run": False, "yes": True, "remove_data": False,
                              "remove_models": False, "ask": _no, "ask_text": _no,
                              "interactive": False, "say": out.append}
    assert lifecycle.uninstall(fake.host, **kwargs) == 0
    assert not fake.prefix.exists() and not (fake.home / ".local/share/vega").exists()
    assert not (fake.home / ".local/bin/vega").is_symlink()
    assert (fake.home / ".zshrc").read_text(encoding="utf-8") == "# mine\n"
    assert not (tmp_path / "LaunchAgents/dev.vega.research.plist").exists()
    assert (tmp_path / "LaunchAgents/com.other.thing.plist").exists()
    assert not (fake.home / "Library/Python/3.9/bin/vega").exists()
    assert not (fake.home / "Library/Python/3.9/lib/python/site-packages/vega").exists()
    # folders that hold only folders go too (vega/native once vega/native/macos is empty);
    # site-packages and the prefix stay
    assert (fake.home / "Library/Python/3.9/lib/python/site-packages").is_dir()
    assert (fake.home / ".vega/setup.json").exists() and fake.deleted == []
    assert out[-1].startswith("Uninstall: removed Vega itself")
    assert "kept ~/.vega" in out[-1] and "Ollama models" in out[-1]
    # again: only the kept groups are left, nothing more is removed
    out.clear()
    assert lifecycle.uninstall(fake.host, **kwargs) == 0
    assert "Vega itself" not in "\n".join(out) and out[-1].startswith("Uninstall: kept")


def test_models_need_a_typed_yes_naming_each_model(tmp_path: Path) -> None:
    fake = _launcher_machine(tmp_path)
    prompts: list[str] = []

    def typed(answer: str):
        def ask_text(question: str) -> str:
            prompts.append(question)
            return answer
        return ask_text
    out: list[str] = []
    lifecycle.uninstall(fake.host, dry_run=False, yes=False, remove_data=False,
                        remove_models=False, ask=lambda q, d: d, ask_text=typed("y"),
                        interactive=True, say=out.append)
    assert "granite4.1:8b (5.3 GB)" in prompts[0] and "vega:latest (shares those weights)" in prompts[0]
    assert fake.deleted == []  # "y" is not "yes"
    assert (fake.home / ".vega").exists()  # data defaults to kept
    lifecycle.uninstall(fake.host, dry_run=False, yes=False, remove_data=True,
                        remove_models=False, ask=lambda q, d: d, ask_text=typed("yes"),
                        interactive=True, say=out.append)
    assert fake.deleted == ["granite4.1:8b", "vega:latest"]
    assert not (fake.home / ".vega").exists()


def test_uninstall_without_a_terminal_or_yes_changes_nothing(tmp_path: Path) -> None:
    fake = _launcher_machine(tmp_path)
    out: list[str] = []
    assert lifecycle.uninstall(fake.host, dry_run=False, yes=False, remove_data=False,
                               remove_models=False, ask=_no, ask_text=_no, interactive=False,
                               say=out.append) == 1
    assert fake.prefix.exists() and "--yes" in out[-1]


@pytest.mark.parametrize("kind", ["venv", "user"])
def test_pip_installs_remove_only_the_package(tmp_path: Path, kind: str) -> None:
    fake = Fake(tmp_path, kind)
    _install, groups = lifecycle.plan(fake.host)
    package = groups[0]
    assert package.key == "package" and "dependencies stay" in package.items[0].text
    package.action()
    assert fake.commands == [[str(fake.host.executable), "-m", "pip", "uninstall", "-y",
                              "--disable-pip-version-check", "vegacode"]]
    assert fake.prefix.exists()


def test_pipx_uninstall_goes_through_pipx(tmp_path: Path) -> None:
    fake = Fake(tmp_path, "pipx")
    _install, groups = lifecycle.plan(fake.host)
    groups[0].action()
    assert fake.commands == [["/opt/bin/pipx", "uninstall", "vegacode"]]


def test_uninstall_leaves_a_dev_checkout_alone(tmp_path: Path) -> None:
    fake = Fake(tmp_path, "dev")
    _install, groups = lifecycle.plan(fake.host)
    assert groups[0].key == "package" and groups[0].items == []
    assert "source checkout" in groups[0].note


def test_ollama_not_running_is_said_not_guessed(tmp_path: Path) -> None:
    fake = Fake(tmp_path, tags=None)
    _install, groups = lifecycle.plan(fake.host)
    models = next(g for g in groups if g.key == "models")
    assert models.items == [] and "not running" in models.note


def test_cli_wires_both_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _launcher_machine(tmp_path)
    monkeypatch.setattr(lifecycle.Host, "current", classmethod(lambda cls: fake.host))
    result = CliRunner().invoke(cli.app, ["uninstall", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Dry run: nothing was changed." in result.output
    result = CliRunner().invoke(cli.app, ["update", "--check"])
    assert result.exit_code == 0 and "0.3.1 -> 0.3.2" in result.output
