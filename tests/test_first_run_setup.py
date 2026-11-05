"""First-run setup (first_run: `vega setup`, the check when `vega`
starts) with fakes for the machine (platform, RAM, disk, PATH, subprocess, HTTP download,
codesign, network) and a fake Ollama server on localhost (tags, a streaming pull). Nothing is
installed, downloaded or run."""

from __future__ import annotations

import contextlib
import json
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fake_ollama import GB, FakeOllama, tag
from typer.testing import CliRunner

from vega import cli, first_run, model_store, models, server

PROFILES = """default = "granite-8b"

[profiles.granite-8b]
base = "granite4.1:8b"
num_ctx = 8192

[profiles.coder-7b]
base = "qwen2.5-coder:7b-instruct-q4_K_M"
num_ctx = 8192
"""
GRANITE = "granite4.1:8b"
SIGNER = "Developer ID Application: Infra Technologies, Inc (3MU9H2V9Y9)"
GOOD_SIGNATURE = (f"Executable=/x/Ollama.app/Contents/MacOS/Ollama\nAuthority={SIGNER}\n"
                  "Authority=Developer ID Certification Authority\nAuthority=Apple Root CA\n"
                  "TeamIdentifier=3MU9H2V9Y9\n")


class FakeSystem(first_run.System):
    """The machine: answers from fields; commands are recorded and simulated."""

    def __init__(self, tmp: Path, *, os: str = "darwin", arch: str = "arm64", ram: float = 16.0,
                 free: float = 140.0, brew: bool = True, ollama: bool = True,
                 online: bool = True, codesign: str = GOOD_SIGNATURE,
                 verify_ok: bool = True) -> None:
        self.tmp, self.os, self._arch, self.ram, self.free = tmp, os, arch, ram, free
        self.brew, self.is_online, self.codesign, self.verify_ok = brew, online, codesign, verify_ok
        self.ollama = "/usr/local/bin/ollama" if ollama else None
        self.runs: list[list[str]] = []
        self.captures: list[list[str]] = []
        self.downloads: list[str] = []
        (tmp / "home").mkdir(exist_ok=True)

    def platform_system(self) -> str:
        return self.os

    def arch(self) -> str:
        return self._arch

    def os_label(self) -> str:
        return {"darwin": "macOS 15.2", "linux": "Ubuntu 24.04", "windows": "Windows 11"}[self.os]

    def apple_silicon(self) -> bool:
        return self.os == "darwin" and self._arch == "arm64"

    def ram_gb(self) -> float | None:
        return self.ram

    def models_dir(self) -> Path:
        return self.tmp / "home" / ".ollama" / "models"

    def free_gb(self, path: Path) -> float | None:
        return self.free

    def which(self, name: str) -> str | None:
        return {"brew": "/opt/homebrew/bin/brew" if self.brew else None,
                "curl": "/usr/bin/curl"}.get(name)

    def home(self) -> Path:
        return self.tmp / "home"

    def online(self) -> bool:
        return self.is_online

    def temp_dir(self) -> Path:
        work = self.tmp / "work"
        work.mkdir()
        return work

    def run(self, argv: list[str]) -> int:
        self.runs.append(argv)
        if argv[1:] == ["install", "ollama"]:
            self.ollama = "/opt/homebrew/bin/ollama"
        elif argv[:2] == ["sh", "-c"]:
            self.ollama = "/usr/local/bin/ollama"
        elif argv[:3] == ["ditto", "-x", "-k"]:
            cli_path = Path(argv[4]) / "Ollama.app" / "Contents" / "Resources" / "ollama"
            cli_path.parent.mkdir(parents=True)
            cli_path.write_text("#!/bin/sh\n")
        elif argv[0] == "ditto":
            shutil.copytree(argv[1], argv[2])
        return 0

    def capture(self, argv: list[str]) -> tuple[int, str]:
        self.captures.append(argv)
        if argv[:2] == ["codesign", "--verify"]:
            return (0, "valid on disk\n") if self.verify_ok else (1, "a sealed resource is missing")
        if argv[:2] == ["codesign", "-dv"]:
            return 0, self.codesign
        if argv[0] == "spctl":
            return 0, f"{argv[-1]}: accepted\nsource=Notarized Developer ID\n"
        return 1, ""

    def download(self, url: str, dest: Path, on_progress: Callable[[int, int], None]) -> None:
        self.downloads.append(url)
        dest.write_bytes(b"zip")
        on_progress(3, 3)

    def ollama_executable(self) -> str | None:
        if self.ollama:
            return self.ollama
        app_cli = self.home() / "Applications" / "Ollama.app" / "Contents" / "Resources" / "ollama"
        return str(app_cli) if app_cli.exists() else None


class FakeUI:
    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers)
        self.lines: list[str] = []
        self.asked: list[str] = []
        self.progress_events: list[tuple[int, int, str]] = []

    def say(self, text: str = "", style: str = "") -> None:
        self.lines.append(text)

    def confirm(self, question: str) -> bool:
        self.asked.append(question)
        return self.answers.pop(0)

    @contextlib.contextmanager
    def progress(self, description: str) -> Iterator[Callable[[int, int, str], None]]:
        yield lambda done, total, status: self.progress_events.append((done, total, status))

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@pytest.fixture
def ollama(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeOllama]:
    """A fake Ollama with nothing downloaded; profiles in a temp file; no scores (the public
    release); `ollama create` replaced by adding the built model to the fake's list."""
    monkeypatch.delenv("VEGA_SKIP_SETUP", raising=False)
    monkeypatch.delenv("VEGA_GUARD_MIN_DISK_GB", raising=False)
    profiles = tmp_path / "models.toml"
    profiles.write_text(PROFILES)
    monkeypatch.setattr(models, "PROFILES_FILE", profiles)
    monkeypatch.setattr(model_store, "SCORES_FILE", tmp_path / "absent" / "scores.toml")
    monkeypatch.setattr(model_store, "registry_size_gb", lambda tag, fetch=None: None)
    fake = FakeOllama(pull_sizes={GRANITE: 5 * GB}).start()
    monkeypatch.setenv("VEGA_BASE_URL", fake.url + "/v1")
    fake.built = []

    def build(profile: models.Profile, target: str) -> int:
        fake.built.append((profile.name, target))
        fake.models.append(tag(f"{target}:latest", 5 * GB, profile.base))
        return 0

    monkeypatch.setattr(models, "build", build)
    yield fake
    fake.stop()


def everything_installed(fake: FakeOllama) -> None:
    fake.models += [tag(GRANITE, 5 * GB), tag("vega:latest", 5 * GB, GRANITE)]


def pulls(fake: FakeOllama) -> list[str]:
    return [body["model"] for body in fake.calls("POST", "/api/pull")]


# -- everything present ---------------------------------------------------------------------------

def test_everything_present_asks_nothing_and_marks_setup_complete(ollama, tmp_path) -> None:
    everything_installed(ollama)
    system, ui = FakeSystem(tmp_path), FakeUI()
    assert first_run.ensure_ready(interactive=True, system=system, ui=ui) == 0
    assert ui.lines == [] and ui.asked == [] and system.runs == []
    assert pulls(ollama) == [] and ollama.built == []
    state = first_run.read_state()
    assert state["complete"] is True and state["profile"] == "granite-8b"


def test_setup_command_reports_ready_without_downloading(ollama, tmp_path) -> None:
    everything_installed(ollama)
    ui = FakeUI()
    result = first_run.run_setup(ui, FakeSystem(tmp_path))
    assert result.profile == "granite-8b" and not result.downloaded
    assert "  Ollama    installed: /usr/local/bin/ollama" in ui.lines
    assert "Vega's model is ready (profile granite-8b, granite4.1:8b)." in ui.text
    assert ui.asked == []


def test_another_model_server_needs_no_setup(ollama, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VEGA_BASE_URL", "http://gpu-box.example:11434/v1")
    system = FakeSystem(tmp_path, ollama=False)
    assert first_run.ensure_ready(interactive=False, system=system, ui=FakeUI()) == 0
    assert system.runs == []


# -- installing Ollama ----------------------------------------------------------------------------

def test_macos_with_homebrew_installs_with_brew_then_downloads_the_pick(ollama, tmp_path) -> None:
    system = FakeSystem(tmp_path, ollama=False, brew=True)
    ui = FakeUI(True, True)
    assert first_run.ensure_ready(interactive=True, system=system, ui=ui) == 0, ui.text
    assert system.runs == [["/opt/homebrew/bin/brew", "install", "ollama"]]
    assert ui.asked == ["  Install Ollama now?", "  Download Granite 8B now (5.3 GB)?"]
    assert "Machine   macOS 15.2, Apple Silicon, 16 GB RAM" in ui.text
    assert pulls(ollama) == [GRANITE]
    assert ollama.built == [("granite-8b", "vega")]
    assert ui.progress_events[-1][0] == ui.progress_events[-1][1] > 0
    assert models.active_profile() == "granite-8b"
    assert first_run.read_state()["complete"] is True
    assert "Vega is ready (model profile granite-8b)." in ui.text


def test_the_whole_plan_is_shown_before_the_first_question(ollama, tmp_path) -> None:
    """What will be installed and downloaded, how big and where, then yes/no questions only."""
    system = FakeSystem(tmp_path, ollama=False, brew=True)
    ui = FakeUI(False)
    assert first_run.ensure_ready(interactive=True, system=system, ui=ui) == 1
    plan = ui.lines[ui.lines.index("Vega will set up:"):]
    assert plan[:4] == [
        "Vega will set up:",
        "  Ollama    installed by Homebrew (brew install ollama; about 50 MB, in /opt/homebrew)",
        "  Model     Granite 8B: a 5.3 GB download into ~/.ollama/models",
        "            the best fit for this machine: fits your 16 GB RAM; 140 GB free",
    ]
    assert "  Then      builds Vega's model from it (`vega`, the default from now on)" in plan
    assert ui.asked == ["  Install Ollama now?"]  # asked after the plan, answered no
    assert system.runs == [] and pulls(ollama) == []


def test_macos_without_homebrew_installs_the_signed_app_in_user_applications(
        ollama, tmp_path) -> None:
    system = FakeSystem(tmp_path, ollama=False, brew=False)
    ui = FakeUI(True, True)
    assert first_run.ensure_ready(interactive=True, system=system, ui=ui) == 0, ui.text
    assert system.downloads == [first_run.MAC_APP_URL]
    assert first_run.MAC_APP_URL.startswith("https://")
    assert ("  Ollama    the Ollama app from ollama.com (about 200 MB download, HTTPS), into "
            "~/Applications/Ollama.app") in ui.lines
    assert "no administrator password needed" in ui.text
    app = tmp_path / "home" / "Applications" / "Ollama.app"
    assert (app / "Contents" / "Resources" / "ollama").is_file()
    verify = ["codesign", "--verify", "--deep", "--strict", "--verbose=2"]
    assert any(argv[:5] == verify for argv in system.captures)
    assert "Signature valid: Infra Technologies, Inc, Team ID 3MU9H2V9Y9; notarized by Apple." in ui.text
    assert first_run.read_state()["ollama_team_id"] == "3MU9H2V9Y9"
    assert not (tmp_path / "work").exists()  # the temporary download is removed
    assert pulls(ollama) == [GRANITE]


@pytest.mark.parametrize(("codesign", "verify_ok", "pinned", "why"), [
    (GOOD_SIGNATURE, False, "", "codesign --verify failed"),
    ("Authority=Developer ID Application: Someone Else (ZZZ9)\nTeamIdentifier=ZZZ9\n", True, "",
     "its Team ID is ZZZ9, not Ollama's 3MU9H2V9Y9"),
    ("Authority=Apple Development: Someone (3MU9H2V9Y9)\nTeamIdentifier=3MU9H2V9Y9\n", True, "",
     "not a Developer ID Application certificate"),
    (GOOD_SIGNATURE, True, "OTHER00000", "its Team ID is 3MU9H2V9Y9, not Ollama's OTHER00000"),
])
def test_macos_app_with_a_bad_signature_is_not_installed(ollama, tmp_path, monkeypatch,
                                                         codesign, verify_ok, pinned, why) -> None:
    if pinned:
        monkeypatch.setenv(first_run.TEAM_ID_ENV, pinned)
    system = FakeSystem(tmp_path, ollama=False, brew=False, codesign=codesign,
                        verify_ok=verify_ok)
    ui = FakeUI(True)
    assert first_run.ensure_ready(interactive=True, system=system, ui=ui) == 1
    assert why in ui.text
    assert f"To do it yourself: Download Ollama from {first_run.DOWNLOAD_PAGE}" in ui.text
    assert not (tmp_path / "home" / "Applications" / "Ollama.app").exists()
    assert not (tmp_path / "work").exists()
    assert pulls(ollama) == []


def test_linux_runs_the_official_script_and_says_it_asks_for_sudo(ollama, tmp_path) -> None:
    system = FakeSystem(tmp_path, os="linux", arch="x86_64", ollama=False)
    ui = FakeUI(True, True)
    assert first_run.ensure_ready(interactive=True, system=system, ui=ui) == 0, ui.text
    assert system.runs == [["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"]]
    assert "asks for your password (sudo)" in ui.text
    assert "Machine   Ubuntu 24.04, x86_64, 16 GB RAM" in ui.text
    assert pulls(ollama) == [GRANITE]


def test_windows_is_refused_before_anything_runs(ollama, tmp_path) -> None:
    system = FakeSystem(tmp_path, os="windows", arch="AMD64", ollama=False)
    ui = FakeUI()
    assert first_run.ensure_ready(interactive=True, system=system, ui=ui) == 1
    assert "Windows is not supported" in ui.text
    assert system.runs == [] and ui.asked == [] and pulls(ollama) == []


def test_declining_the_install_changes_nothing_and_prints_the_manual_step(ollama, tmp_path) -> None:
    system = FakeSystem(tmp_path, ollama=False, brew=True)
    ui = FakeUI(False)
    assert first_run.ensure_ready(interactive=True, system=system, ui=ui) == 1
    assert system.runs == []
    assert "Ollama was not installed." in ui.text
    assert "To do it yourself: brew install ollama   then run `vega` again" in ui.text


def test_offline_without_ollama_stops_with_the_manual_step(ollama, tmp_path) -> None:
    system = FakeSystem(tmp_path, ollama=False, online=False)
    ui = FakeUI()
    assert first_run.ensure_ready(interactive=True, system=system, ui=ui) == 1
    assert "this machine is offline" in ui.text and system.runs == [] and ui.asked == []


# -- model choice ---------------------------------------------------------------------------------

def choice(ram: float, free: float, tags: list[dict] | None = None) -> first_run.Choice:
    return first_run.choose_model(model_store.load_catalog(), ram, free, tags or [])


def test_model_choice_by_ram(ollama) -> None:
    assert choice(16, 140).options[0].entry.name == "granite-8b"
    eight = choice(8, 140).options[0]
    assert eight.entry.name == "granite-micro" and eight.fit == "fits"
    assert choice(32, 140).options[0].entry.name == "granite-8b"


def test_model_choice_by_disk_falls_back_to_a_smaller_model(ollama) -> None:
    # 17 GB free: Granite (5.3 GB) would leave 11.7 GB, under the 12 GB Vega keeps free
    pick = choice(16, 17.0).options[0]
    assert pick.entry.name == "coder-7b"  # the catalogue's fallback
    assert all(17.0 - o.entry.download_gb >= 12 for o in choice(16, 17.0).options)
    # 15 GB free: only models up to 3 GB are left
    assert choice(16, 15.0).options[0].entry.download_gb <= 3.0


def test_an_already_downloaded_model_needs_no_disk(ollama) -> None:
    pick = choice(16, 13.0, [tag(GRANITE, 5 * GB)]).options[0]
    assert pick.entry.name == "granite-8b" and pick.installed
    assert pick.line(16, 13.0) == ("Granite 8B · already downloaded · fits your 16 GB RAM; "
                                   "13 GB free")


def test_nothing_fits_says_what_is_needed(ollama) -> None:
    with pytest.raises(first_run.SetupError) as disk:
        choice(16, 13.0)
    assert "The smallest, Granite Micro 3B, needs 2.1 GB of disk" in str(disk.value)
    assert "free up 1.1 GB more" in str(disk.value)
    assert "OLLAMA_MODELS" in disk.value.manual
    with pytest.raises(first_run.SetupError) as ram:
        choice(4, 140)
    assert "about 4.6 GB of RAM to run (this machine has 4 GB)" in str(ram.value)


def test_nothing_fits_stops_setup_before_any_download(ollama, tmp_path) -> None:
    system, ui = FakeSystem(tmp_path, free=13.0), FakeUI()
    assert first_run.ensure_ready(interactive=True, system=system, ui=ui) == 1
    assert "No model Vega offers fits this machine." in ui.text
    assert pulls(ollama) == [] and ui.asked == []


def test_saying_no_to_the_model_downloads_nothing_and_names_the_others(ollama, tmp_path) -> None:
    ui = FakeUI(False)
    assert first_run.ensure_ready(interactive=True, system=FakeSystem(tmp_path), ui=ui) == 1
    assert ui.asked == ["  Download Granite 8B now (5.3 GB)?"]
    assert "No model was downloaded." in ui.text and pulls(ollama) == []
    assert "pick another model with `vega setup --model NAME`: coder-7b (Qwen2.5-Coder 7B" in ui.text


def test_another_model_by_name_downloads_that_one_and_adds_its_profile(ollama, tmp_path) -> None:
    name = "granite-micro"
    entry = model_store.load_catalog()[name]
    ui = FakeUI(True)
    result = first_run.run_setup(ui, FakeSystem(tmp_path), model=name)
    assert result.profile == name, ui.text
    assert ui.asked == [f"  Download {entry.display} now ({entry.download_gb:.1f} GB)?"]
    assert pulls(ollama) == [entry.tag]
    assert ollama.built == [(name, "vega")]
    assert name in models.load()[1]  # a catalogue model gets its profile
    assert models.active_profile() == name


# -- interrupted download, skip and repair --------------------------------------------------------

def test_interrupted_download_resumes_the_same_model_on_the_next_run(ollama, tmp_path) -> None:
    ollama.pull_cut = 1
    ui = FakeUI(True)
    assert first_run.ensure_ready(interactive=True, system=FakeSystem(tmp_path), ui=ui) == 1
    assert "The download stopped" in ui.text
    assert "Run `vega` again (or `vega setup`) to continue" in ui.text
    state = first_run.read_state()
    assert state["profile"] == "granite-8b" and state["complete"] is False
    # next start: the same model, no new choice; one confirmation for the download
    again = FakeUI(True)
    assert first_run.ensure_ready(interactive=True, system=FakeSystem(tmp_path), ui=again) == 0
    assert again.asked == ["  Download Granite 8B now (5.3 GB)?"]
    assert "a part fetched before is kept" in again.text
    assert pulls(ollama) == [GRANITE, GRANITE]
    assert first_run.read_state()["complete"] is True


def test_ctrl_c_during_the_download_is_a_pause(ollama, tmp_path, monkeypatch) -> None:
    def interrupted(plan, on_progress):
        raise KeyboardInterrupt

    monkeypatch.setattr(model_store, "pull", interrupted)
    ui = FakeUI(True)
    assert first_run.ensure_ready(interactive=True, system=FakeSystem(tmp_path), ui=ui) == 1
    assert "Download paused." in ui.text and "keeps the part already downloaded" in ui.text


def test_complete_setup_is_skipped_on_later_starts(ollama, tmp_path) -> None:
    everything_installed(ollama)
    first_run.write_state(complete=True, profile="granite-8b")
    system, ui = FakeSystem(tmp_path), FakeUI()
    assert first_run.ensure_ready(interactive=True, system=system, ui=ui) == 0
    assert ui.lines == [] and system.runs == []
    assert [p for m, p, _ in ollama.requests if m == "POST"] == []


def test_a_deleted_vega_model_is_rebuilt_without_downloading(ollama, tmp_path) -> None:
    ollama.models.append(tag(GRANITE, 5 * GB))  # the weights are here, `vega` is gone
    first_run.write_state(complete=True, profile="granite-8b")
    ui = FakeUI()
    assert first_run.ensure_ready(interactive=True, system=FakeSystem(tmp_path), ui=ui) == 0
    assert "Vega: repairing setup" in ui.text
    assert ollama.built == [("granite-8b", "vega")]
    assert pulls(ollama) == [] and ui.asked == []


def test_a_deleted_model_is_downloaded_again_after_one_confirmation(ollama, tmp_path) -> None:
    first_run.write_state(complete=True, profile="granite-8b")
    (server.STATE_DIR / "active_model.json").write_text(json.dumps({"profile": "granite-8b"}))
    ui = FakeUI(True)
    assert first_run.ensure_ready(interactive=True, system=FakeSystem(tmp_path), ui=ui) == 0
    assert ui.asked == ["  Download Granite 8B now (5.3 GB)?"]  # the same model again
    assert pulls(ollama) == [GRANITE]


def test_ollama_gone_is_reinstalled_and_the_model_kept(ollama, tmp_path) -> None:
    everything_installed(ollama)
    first_run.write_state(complete=True, profile="granite-8b")
    system = FakeSystem(tmp_path, ollama=False, brew=True)
    ui = FakeUI(True)
    assert first_run.ensure_ready(interactive=True, system=system, ui=ui) == 0, ui.text
    assert system.runs == [["/opt/homebrew/bin/brew", "install", "ollama"]]
    assert pulls(ollama) == [] and ollama.built == []


# -- without a terminal, the CLI and --version ----------------------------------------------------

def test_without_a_terminal_setup_is_not_run_and_the_command_is_named(ollama, tmp_path) -> None:
    system, ui = FakeSystem(tmp_path, ollama=False), FakeUI()
    assert first_run.ensure_ready(interactive=False, system=system, ui=ui) == 1
    assert ui.lines == ["Vega is not set up yet: Ollama is not installed.",
                        ("Run this once (it installs and downloads what is missing), then try "
                         "again:\n  vega setup --yes")]
    assert system.runs == [] and ui.asked == []


def test_vega_run_without_a_terminal_fails_with_the_setup_command(ollama, tmp_path,
                                                                  monkeypatch) -> None:
    monkeypatch.setattr(first_run, "System", lambda: FakeSystem(tmp_path, ollama=False))
    result = CliRunner().invoke(cli.app, ["run", "fix the bug", "-p", str(tmp_path)])
    assert result.exit_code == 1
    assert "Vega is not set up yet: Ollama is not installed." in result.output
    assert "vega setup --yes" in result.output


def test_setup_command_yes_runs_without_questions(ollama, tmp_path, monkeypatch) -> None:
    system = FakeSystem(tmp_path, ollama=False, brew=True)
    monkeypatch.setattr(first_run, "System", lambda: system)
    result = CliRunner().invoke(cli.app, ["setup", "--yes"], env={"COLUMNS": "120"})
    assert result.exit_code == 0, result.output
    assert "Install Ollama now? yes (--yes)" in result.output
    assert system.runs == [["/opt/homebrew/bin/brew", "install", "ollama"]]
    assert pulls(ollama) == [GRANITE] and models.active_profile() == "granite-8b"


def test_after_setup_plain_vega_finds_its_default_model(ollama, tmp_path) -> None:
    """Setup makes the chosen profile the default: the `vega` model exists and maps to it, so a
    later `vega` (no --model) neither asks nor downloads."""
    ui = FakeUI(True)
    assert first_run.ensure_ready(interactive=True, system=FakeSystem(tmp_path), ui=ui) == 0
    assert server.model_name() == "vega" and "vega:latest" in ollama.names()
    assert models.profile_for_model("vega").name == "granite-8b"
    later = FakeUI()
    assert first_run.ensure_ready(interactive=True, system=FakeSystem(tmp_path), ui=later) == 0
    assert later.lines == [] and later.asked == [] and pulls(ollama) == [GRANITE]


@pytest.mark.parametrize(("argv", "checked"), [
    ([], None), (["chat"], "chat"), (["run", "x"], "run"), (["model", "--help"], ""),
    (["server", "--help"], ""),
])
def test_setup_is_checked_when_work_with_a_model_starts(monkeypatch, argv, checked) -> None:
    """`vega` alone (the launcher's exec after its own setup), chat, run and auto check setup
    first; model and server commands do not."""
    seen: list = []

    def check(command, *, model_given):
        seen.append(command)
        return 3  # stop before anything else runs

    monkeypatch.setattr(first_run, "before_start", check)
    monkeypatch.delenv("VEGA_SKIP_SETUP", raising=False)
    monkeypatch.setenv("VEGA_BASE_URL", "http://127.0.0.1:9/v1")  # nothing may reach a server
    result = CliRunner().invoke(cli.app, argv)
    if checked == "":
        assert seen == []
    else:
        assert seen == [checked] and result.exit_code == 3


def test_setup_command_without_a_terminal_needs_yes(ollama, tmp_path, monkeypatch) -> None:
    system = FakeSystem(tmp_path, ollama=False)
    monkeypatch.setattr(first_run, "System", lambda: system)
    result = CliRunner().invoke(cli.app, ["setup"])
    assert result.exit_code == 1 and "vega setup --yes" in result.output
    assert system.runs == []
