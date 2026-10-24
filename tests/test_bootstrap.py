"""The Python 3.9 bootstrap (vega._bootstrap, vega._launch): `pip3 install vegacode` with macOS's
own Python installs only the launcher, which sets Vega up under a Python 3.11+ in its own venv.
Everything outside the process is faked here; one test runs the real system Python 3.9 when the
machine has one."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from vega import _bootstrap as boot
from vega import _launch

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PYTHON = Path("/usr/bin/python3")


def _exe(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _fake_probe(versions: dict[str, tuple[int, ...]], no_venv: tuple[str, ...] = ()):
    calls: list[str] = []

    def probe(path: Path):
        calls.append(str(path))
        if path.name == "python" and path.parent.name == "bin":  # a venv's interpreter
            return boot.Python(path, (3, 12, 7), True) if path.exists() else None
        version = versions.get(str(path))
        return None if version is None else boot.Python(path, version, str(path) not in no_venv)

    probe.calls = calls  # type: ignore[attr-defined]
    return probe


# --- discovery ---------------------------------------------------------------------------------

def test_candidates_cover_path_homebrew_python_org_uv_and_pyenv(tmp_path: Path) -> None:
    home = tmp_path / "home"
    uv = _exe(home / ".local/share/uv/python/cpython-3.12.13-macos-aarch64-none/bin/python3")
    pyenv = _exe(home / ".pyenv/versions/3.11.9/bin/python3")
    found = [str(p) for p in boot.candidates({"PATH": "/a:/b"}, home, "Darwin")]
    for expected in ("/a/python3.13", "/b/python3.11", "/a/python3",
                     "/opt/homebrew/bin/python3.12", "/usr/local/bin/python3.12",
                     "/opt/homebrew/opt/python@3.12/bin/python3.12",
                     "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12",
                     str(home / ".local/bin/python3.12"), str(uv), str(pyenv)):
        assert expected in found
    assert found.index("/a/python3.13") < found.index("/opt/homebrew/bin/python3.13")
    override = boot.candidates({"VEGA_BOOTSTRAP_PYTHON": "/x/py", "PATH": ""}, home, "Linux")
    assert str(override[0]) == "/x/py"
    assert "/usr/bin/python3.12" in [str(p) for p in override]  # Linux distro Pythons


def test_find_python_prefers_supported_newest_and_skips_old_and_duplicates(tmp_path: Path) -> None:
    bins = tmp_path / "bin"
    p39, p311, p312, p314 = (_exe(bins / n) for n in ("python3", "python3.11", "python3.12",
                                                       "python3.13"))
    (bins / "dup").mkdir()
    os.symlink(p312, bins / "dup" / "python3.12")  # same interpreter twice on PATH
    probe = _fake_probe({str(p39): (3, 9, 6), str(p311): (3, 11, 9), str(p312): (3, 12, 7),
                         str(p314): (3, 14, 0)})
    env = {"PATH": f"{bins}:{bins / 'dup'}"}
    best, no_venv = boot.find_python(env, tmp_path / "home", "Linux", probe)
    assert best is not None and best.path == p312  # 3.14 is newer but untested
    assert no_venv == []
    assert sum(Path(c).resolve() == p312 for c in probe.calls) == 1


def test_find_python_reports_pythons_without_venv(tmp_path: Path) -> None:
    py = _exe(tmp_path / "bin" / "python3.12")
    probe = _fake_probe({str(py): (3, 12, 3)}, no_venv=(str(py),))
    best, no_venv = boot.find_python({"PATH": str(py.parent)}, tmp_path, "Linux", probe)
    assert best is None and [p.path for p in no_venv] == [py]


def test_find_python_never_runs_the_macos_stub() -> None:
    probe = _fake_probe({})
    boot.find_python({"PATH": "/usr/bin"}, Path("/nonexistent-home"), "Darwin", probe)
    assert "/usr/bin/python3" not in probe.calls


# --- what to install, where --------------------------------------------------------------------

def test_venv_dir_default_xdg_and_override(tmp_path: Path) -> None:
    home = tmp_path / "h"
    assert boot.venv_dir({}, home) == home / ".local/share/vega/venv"
    assert boot.venv_dir({"XDG_DATA_HOME": "/d"}, home) == Path("/d/vega/venv")
    assert boot.venv_dir({"XDG_DATA_HOME": "/d", "VEGA_BOOTSTRAP_VENV": "/v"}, home) == Path("/v")


def test_install_spec_reuses_the_source_this_copy_came_from(tmp_path: Path) -> None:
    assert boot.install_spec("0.3.1", None) == (["vegacode==0.3.1"], "PyPI")
    assert boot.install_spec(None, None) == (["vegacode"], "PyPI (latest)")
    wheel = tmp_path / "my dir" / "vegacode-0.3.1-py3-none-any.whl"
    wheel.parent.mkdir()
    wheel.write_bytes(b"")
    url = {"url": "file://" + str(wheel).replace(" ", "%20"), "archive_info": {}}
    assert boot.install_spec("0.3.1", url) == ([str(wheel)], str(wheel))
    gone = {"url": "file:///nowhere/vegacode-0.3.1.whl", "archive_info": {}}
    assert boot.install_spec("0.3.1", gone)[0] == ["vegacode==0.3.1"]  # file gone: PyPI
    folder = {"url": "file://" + str(tmp_path), "dir_info": {"editable": True}}
    assert boot.install_spec("0.3.1", folder)[0] == [str(tmp_path)]
    git = {"url": "https://github.com/x/vegacode", "vcs_info": {"vcs": "git", "commit_id": "abc"}}
    assert boot.install_spec("0.3.1", git)[0] == [
        "vegacode @ git+https://github.com/x/vegacode@abc"]
    web = {"url": "https://example.com/vegacode-0.3.1.whl", "archive_info": {}}
    assert boot.install_spec("0.3.1", web)[0] == [
        "vegacode @ https://example.com/vegacode-0.3.1.whl"]


# --- links and PATH ----------------------------------------------------------------------------

def _venv_bin(tmp_path: Path) -> Path:
    source = tmp_path / "venv" / "bin"
    for name in ("vega", "vegacode"):
        _exe(source / name)
    return source


def _never(question: str) -> bool:
    raise AssertionError("asked: " + question)


def test_link_commands_creates_links_and_is_idempotent(tmp_path: Path) -> None:
    source, links = _venv_bin(tmp_path), tmp_path / "local" / "bin"
    commands, notes = boot.link_commands(source, links, _never, interactive=False)
    assert notes == [] and commands == {"vega": links / "vega", "vegacode": links / "vegacode"}
    assert os.readlink(links / "vega") == str(source / "vega")
    assert boot.link_commands(source, links, _never, interactive=False) == (commands, [])


def test_link_commands_leaves_foreign_commands_unless_the_user_agrees(tmp_path: Path) -> None:
    source, links = _venv_bin(tmp_path), tmp_path / "local" / "bin"
    links.mkdir(parents=True)
    other = _exe(tmp_path / "other" / "vega")
    os.symlink(other, links / "vega")
    (links / "vegacode").write_text("my own script\n", encoding="utf-8")
    commands, notes = boot.link_commands(source, links, _never, interactive=False)
    assert commands == {"vega": source / "vega", "vegacode": source / "vegacode"}
    assert len(notes) == 2 and all("Left" in n for n in notes)
    assert os.readlink(links / "vega") == str(other)  # untouched
    asked: list[str] = []
    commands, notes = boot.link_commands(source, links, lambda q: asked.append(q) or True,
                                         interactive=True)
    assert len(asked) == 2 and str(other) in asked[0]
    assert os.readlink(links / "vega") == str(source / "vega")
    assert os.readlink(links / "vegacode") == str(source / "vegacode")


def test_link_commands_replaces_the_old_python_launcher_script(tmp_path: Path) -> None:
    source, links = _venv_bin(tmp_path), tmp_path / "local" / "bin"
    links.mkdir(parents=True)
    (links / "vega").write_text("#!/usr/bin/python3\nimport re\nimport sys\n"
                                "from vega._launch import main\n", encoding="utf-8")
    commands, notes = boot.link_commands(source, links, _never, interactive=False)
    assert commands["vega"] == links / "vega" and "launcher" in notes[0]
    assert os.readlink(links / "vega") == str(source / "vega")


@pytest.mark.parametrize(("shell", "system", "expected"), [
    ("/bin/zsh", "Darwin", """echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc"""),
    ("/bin/bash", "Darwin", """echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bash_profile"""),
    ("/bin/bash", "Linux", """echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc"""),
    ("/usr/bin/fish", "Linux", "fish_add_path $HOME/.local/bin"),
    ("", "Linux", """echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.profile"""),
])
def test_path_hint_names_the_line_for_the_shell(tmp_path: Path, shell: str, system: str,
                                               expected: str) -> None:
    folder = tmp_path / ".local" / "bin"
    lines = boot.path_hint(folder, {"PATH": "/usr/bin", "SHELL": shell}, tmp_path, system)
    assert "    " + expected in lines and "~/.local/bin is not on your PATH" in lines[0]
    on_path = {"PATH": f"/usr/bin:{folder}", "SHELL": shell}
    assert boot.path_hint(folder, on_path, tmp_path, system) == []


# --- whole runs --------------------------------------------------------------------------------

class Machine:
    """A fake machine: one Python 3.12 on PATH; `python -m venv` and `pip install` act on disk."""

    def __init__(self, tmp_path: Path, *, env: dict[str, str] | None = None,
                 tty: bool = False, answers: list[bool] | None = None,
                 version: str | None = "0.3.1", pythons: dict[str, tuple[int, ...]] | None = None,
                 system: str = "Darwin") -> None:
        self.home = tmp_path / "home"
        self.bins = tmp_path / "bin"
        self.python = _exe(self.bins / "python3.12")
        self.env = {"PATH": str(self.bins), "SHELL": "/bin/zsh", **(env or {})}
        self.boot = boot.Bootstrap(env=self.env, home=self.home, system=system)
        self.commands: list[list[str]] = []
        self.execs: list[tuple[Path, list[str]]] = []
        self.questions: list[str] = []
        self.answers = list(answers or [])
        self.installed_version = version
        self.boot.interactive = lambda: tty
        self.boot.ask = self._ask
        self.boot.run_command = self._run
        self.boot.exec_command = lambda path, args: self.execs.append((path, args)) or 0
        self.boot.probe = _fake_probe(pythons if pythons is not None
                                      else {str(self.python): (3, 12, 7)})
        self.boot.version = lambda: version
        self.boot.direct_url = lambda: None

    @property
    def venv(self) -> Path:
        return self.home / ".local/share/vega/venv"

    def _ask(self, question: str, default: bool = False) -> bool:
        self.questions.append(question)
        return self.answers.pop(0)

    def _run(self, command: list[str]) -> int:
        self.commands.append(command)
        if command[1:3] == ["-m", "venv"]:
            venv = Path(command[-1])
            _exe(venv / "bin" / "python")
            (venv / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
        elif command[1:4] == ["-m", "pip", "install"]:
            venv = Path(command[0]).parents[1]
            for name in ("vega", "vegacode"):
                _exe(venv / "bin" / name)
            site = venv / "lib/python3.12/site-packages"
            for old in site.glob("vegacode-*.dist-info"):
                old.rmdir()
            (site / f"vegacode-{self.installed_version}.dist-info").mkdir(parents=True)
        return 0


def test_without_a_terminal_it_explains_and_changes_nothing(tmp_path: Path, capsys) -> None:
    machine = Machine(tmp_path)
    assert machine.boot.main(["--help"]) == 1
    err = capsys.readouterr().err
    assert "Vega needs Python 3.11 or newer" in err and "No terminal to ask in" in err
    assert f"{machine.python} -m venv {machine.venv}" in err
    assert "pip install vegacode==0.3.1" in err
    assert machine.commands == [] and machine.execs == [] and not machine.home.exists()


def test_consented_setup_makes_a_venv_installs_links_and_runs_vega(tmp_path: Path,
                                                                  capsys) -> None:
    machine = Machine(tmp_path, env={"VEGA_BOOTSTRAP_YES": "1"})
    assert machine.boot.main(["run", "fix it"]) == 0
    venv_python = str(machine.venv / "bin" / "python")
    assert machine.commands == [
        [str(machine.python), "-m", "venv", str(machine.venv)],
        [venv_python, "-m", "pip", "install", "--disable-pip-version-check", "vegacode==0.3.1"],
    ]
    # PEP 668: pip only ever runs in Vega's own venv, never in the Python that was found.
    assert all(c[0] == venv_python for c in machine.commands if "pip" in c)
    link = machine.home / ".local/bin/vega"
    assert machine.execs == [(link, ["run", "fix it"])]
    assert os.readlink(link) == str(machine.venv / "bin/vega")
    err = capsys.readouterr().err
    assert "~/.local/bin is not on your PATH" in err and "~/.zshrc" in err


def test_second_run_goes_straight_to_vega(tmp_path: Path, capsys) -> None:
    machine = Machine(tmp_path, env={"VEGA_BOOTSTRAP_YES": "1"})
    machine.boot.main([])
    capsys.readouterr()
    machine.commands.clear()
    machine.execs.clear()
    assert machine.boot.main(["--version"]) == 0
    assert machine.commands == []
    assert machine.execs == [(machine.home / ".local/bin/vega", ["--version"])]
    assert capsys.readouterr().err == ""


def test_a_new_launcher_version_updates_the_venv_without_recreating_it(tmp_path: Path) -> None:
    machine = Machine(tmp_path, env={"VEGA_BOOTSTRAP_YES": "1"}, version="0.3.1")
    machine.boot.main([])
    machine.commands.clear()
    machine.installed_version = "0.3.2"
    machine.boot.version = lambda: "0.3.2"
    machine.boot.env = {k: v for k, v in machine.env.items() if k != "VEGA_BOOTSTRAP_YES"}
    assert machine.boot.main([]) == 0
    assert machine.commands == [[str(machine.venv / "bin/python"), "-m", "pip", "install",
                                 "--disable-pip-version-check", "vegacode==0.3.2"]]


def test_interactive_consent_and_refusal(tmp_path: Path, capsys) -> None:
    refused = Machine(tmp_path / "a", tty=True, answers=[False])
    assert refused.boot.main([]) == 1
    assert refused.questions == ["Set up Vega now?"] and refused.commands == []
    assert "Nothing changed" in capsys.readouterr().err
    agreed = Machine(tmp_path / "b", tty=True, answers=[True])
    assert agreed.boot.main([]) == 0 and len(agreed.commands) == 2


def test_a_broken_venv_is_recreated_with_clear(tmp_path: Path) -> None:
    machine = Machine(tmp_path, env={"VEGA_BOOTSTRAP_YES": "1"})
    machine.venv.mkdir(parents=True)
    (machine.venv / "pyvenv.cfg").write_text("home = /gone\n", encoding="utf-8")
    assert machine.boot.main([]) == 0
    assert machine.commands[0] == [str(machine.python), "-m", "venv", "--clear", str(machine.venv)]


def test_refuses_a_folder_that_is_not_a_venv(tmp_path: Path, capsys) -> None:
    machine = Machine(tmp_path, env={"VEGA_BOOTSTRAP_YES": "1"})
    machine.venv.mkdir(parents=True)
    (machine.venv / "notes.txt").write_text("mine", encoding="utf-8")
    assert machine.boot.main([]) == 1 and machine.commands == []
    assert "is not a virtual environment" in capsys.readouterr().err


def test_venv_override_and_local_source(tmp_path: Path) -> None:
    wheel = tmp_path / "vegacode-0.3.1-py3-none-any.whl"
    wheel.write_bytes(b"")
    target = tmp_path / "elsewhere"
    machine = Machine(tmp_path, env={"VEGA_BOOTSTRAP_YES": "1", "VEGA_BOOTSTRAP_VENV": str(target)})
    machine.boot.direct_url = lambda: {"url": wheel.as_uri(), "archive_info": {}}
    assert machine.boot.main([]) == 0
    assert machine.commands[1][-1] == str(wheel) and machine.commands[1][0].startswith(str(target))


def test_no_python_offers_brew_only_in_a_terminal(tmp_path: Path, capsys,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(boot, "find_brew", lambda env: "/opt/homebrew/bin/brew")
    monkeypatch.setattr(boot, "find_uv", lambda env, home: None)
    quiet = Machine(tmp_path / "a", pythons={})
    assert quiet.boot.main([]) == 1 and quiet.commands == [] and quiet.questions == []
    err = capsys.readouterr().err
    assert "brew install python@3.12" in err
    assert "uv tool install --python 3.12 vegacode" in err

    declined = Machine(tmp_path / "b", pythons={}, tty=True, answers=[False])
    assert declined.boot.main([]) == 1 and declined.commands == []
    assert declined.questions == ["Run it now?"]

    agreed = Machine(tmp_path / "c", pythons={}, tty=True, answers=[True, True])
    real_run = agreed._run

    def run(command: list[str]) -> int:
        if command[1:] == ["install", "python@3.12"]:  # brew "installs" a Python
            agreed.commands.append(command)
            agreed.boot.probe = _fake_probe({str(agreed.python): (3, 12, 7)})
            return 0
        return real_run(command)

    agreed.boot.run_command = run
    assert agreed.boot.main([]) == 0
    assert agreed.commands[0] == ["/opt/homebrew/bin/brew", "install", "python@3.12"]
    assert agreed.commands[1][1:3] == ["-m", "venv"]


def test_no_python_offers_uv_before_brew(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(boot, "find_brew", lambda env: "/opt/homebrew/bin/brew")
    machine = Machine(tmp_path, pythons={}, tty=True, answers=[True, True])
    uv = _exe(machine.home / ".local" / "bin" / "uv")
    managed = machine.home / ".local/share/uv/python/cpython-3.12.13-macos-aarch64-none/bin/python3"
    real_run = machine._run

    def run(command: list[str]) -> int:
        if command[1:] == ["python", "install", "3.12"]:  # uv "installs" a managed Python
            machine.commands.append(command)
            _exe(managed)
            machine.boot.probe = _fake_probe({str(managed): (3, 12, 13)})
            return 0
        return real_run(command)

    machine.boot.run_command = run
    assert machine.boot.main([]) == 0
    assert machine.commands[0] == [str(uv), "python", "install", "3.12"]
    assert machine.commands[1] == [str(managed), "-m", "venv", str(machine.venv)]


def test_linux_hints_and_windows_instructions(tmp_path: Path, capsys,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(boot.shutil, "which", lambda name, path=None: "/usr/bin/apt-get"
                        if name == "apt-get" else None)
    linux = Machine(tmp_path / "a", pythons={}, system="Linux", tty=True)
    assert linux.boot.main([]) == 1 and linux.questions == []  # never runs apt
    assert "sudo apt install python3.12 python3.12-venv" in capsys.readouterr().err
    windows = Machine(tmp_path / "b", system="Windows")
    assert windows.boot.main([]) == 1 and windows.commands == []
    assert "py -3.12 -m pip install vegacode" in capsys.readouterr().err


# --- the launcher ------------------------------------------------------------------------------

def test_launcher_hands_to_the_cli_on_311_and_the_bootstrap_below(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from vega import cli

    monkeypatch.setattr(cli, "app", lambda: "cli")
    assert _launch.main() == "cli"
    monkeypatch.setattr(_launch.sys, "version_info", (3, 9, 6))
    monkeypatch.setattr(boot, "run", lambda args: ("bootstrap", args))
    monkeypatch.setattr(_launch.sys, "argv", ["vega", "--help"])
    assert _launch.main() == ("bootstrap", ["--help"])


def test_launcher_imports_only_the_standard_library() -> None:
    code = (f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); before = set(sys.modules); "
            "import vega._launch, vega._bootstrap; "
            "print(sorted(m for m in set(sys.modules) - before "
            "if m.split('.')[0] not in sys.stdlib_module_names and m.split('.')[0] != 'vega'))")
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                            check=True, timeout=60)
    assert result.stdout.strip() == "[]"


def _system_python_version() -> tuple[int, int] | None:
    if not SYSTEM_PYTHON.exists():
        return None
    try:
        out = subprocess.run([str(SYSTEM_PYTHON), "-c", "import sys; print(*sys.version_info[:2])"],
                             capture_output=True, text=True, timeout=60, check=True).stdout
        major, minor = out.split()
        return int(major), int(minor)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


_SYSTEM_VERSION = _system_python_version()


@pytest.mark.skipif(_SYSTEM_VERSION is None or _SYSTEM_VERSION >= (3, 11),
                    reason="needs a Python older than 3.11 at /usr/bin/python3")
def test_real_old_python_imports_the_launcher_and_explains(tmp_path: Path) -> None:
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path / "home"),
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT / "src")}
    imported = subprocess.run([str(SYSTEM_PYTHON), "-c", ("import vega, vega._launch, "
                               "vega._bootstrap")], capture_output=True, text=True,
                              env=env, timeout=60, check=False)
    assert imported.returncode == 0, imported.stderr
    result = subprocess.run([str(SYSTEM_PYTHON), "-m", "vega", "--help"], capture_output=True,
                            text=True, env=env, stdin=subprocess.DEVNULL, timeout=120, check=False)
    assert result.returncode == 1, result.stderr
    assert "Vega needs Python 3.11 or newer; this is Python 3." in result.stderr
    assert not (tmp_path / "home" / ".local").exists()


def test_distribution_metadata_lets_old_pythons_install_only_the_launcher() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["requires-python"] == ">=3.9"
    assert project["scripts"] == {"vega": "vega._launch:main", "vegacode": "vega._launch:main"}
    marker = 'python_version >= "3.11"'
    requirements = project["dependencies"] + [
        req for extra in project["optional-dependencies"].values() for req in extra]
    assert requirements and all(req.endswith("; " + marker) for req in requirements)
