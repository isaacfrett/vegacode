"""First-run setup: `pip install vegacode`, then `vega` handles the rest.

When `vega` (chat), `vega run` or `vega auto` starts and something Vega needs is missing, this
module finds out what and sets it up. It first shows the whole plan (what it installs and
downloads, how big, where), then asks yes/no before installing Ollama and yes/no before the model
download; nothing is installed or downloaded without that yes:

1. The machine: OS and architecture, RAM, free disk where Ollama keeps models, network.
2. Ollama: used when installed. Otherwise installed after its yes: macOS `brew install ollama`
   (with Homebrew), else the official app archive from
   ollama.com over HTTPS, its code signature checked (codesign --verify --deep --strict, a
   Developer ID Application certificate with Ollama's Team ID, Gatekeeper's assessment), into
   ~/Applications (no admin rights); Linux the official install script (it asks for sudo itself);
   Windows is not supported. Then the server is started (server.start).
3. A model: the best catalogue model that fits the RAM (model_store.recommend) and the disk (its
   download plus the 12 GB Vega keeps free), else a smaller one; when none fits, what is needed.
   Declining it names the alternatives (`vega setup --model NAME`).
4. The download after its yes, with progress (model_store.pull; Ollama keeps partial downloads, so
   running setup again continues), then the profile is made active and the default (models.use
   builds `vega`, which plain `vega` runs), and the chat (or the command) goes on.
5. VEGA_HOME/setup.json records completion. Later starts only check that Ollama and the `vega`
   model are still there, and repair just the missing piece.

Without a terminal nothing is asked: Vega prints the one command to run (`vega setup --yes`).
VEGA_SKIP_SETUP=1 skips the startup check (CI, scripts that manage their own server).

Wiring: cli.main calls at_start (`vega`, chat, run, auto) and `vega setup` is setup_command
(promoted from the private lab into core on 2026-11-04, unchanged).
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import typer

from . import model_store, models, resources, server

SKIP_ENV = "VEGA_SKIP_SETUP"
TEAM_ID_ENV = "VEGA_OLLAMA_TEAM_ID"  # override the expected Team ID (if Ollama re-signs its app)
MAC_APP_URL = "https://ollama.com/download/Ollama-darwin.zip"
LINUX_INSTALL = "curl -fsSL https://ollama.com/install.sh | sh"
DOWNLOAD_PAGE = "https://ollama.com/download"
# The genuine Ollama.app (ollama.com/download/Ollama-darwin.zip) is signed "Developer ID Application:
# Infra Technologies, Inc (3MU9H2V9Y9)" and notarized; read off a real download on 2026-10-22.
OLLAMA_TEAM_ID = "3MU9H2V9Y9"
SETUP_COMMAND = "vega setup --yes"
# `vega`, chat, run and auto start work with a model: setup is checked before them (cli.main).
START_COMMANDS = {None, "chat", "run", "auto"}


def version() -> str:
    """The installed vegacode version (setup.json, the download's User-Agent)."""
    from importlib import metadata

    try:
        return metadata.version("vegacode")
    except metadata.PackageNotFoundError:
        return "unknown"


class SetupError(Exception):
    """Setup stopped. `manual` is the exact step the user can take instead."""

    def __init__(self, message: str, manual: str = "") -> None:
        super().__init__(message)
        self.manual = manual


# --- State ----------------------------------------------------------------------------------------

def state_path() -> Path:
    return server.STATE_DIR / "setup.json"  # resolved on use: tests and VEGA_HOME move it


def read_state() -> dict[str, Any]:
    try:
        data = json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_state(**fields: Any) -> dict[str, Any]:
    state = {**read_state(), **fields}
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)
    return state


# --- The machine and the outside world (replaced by fakes in tests) ------------------------------

@dataclass
class Machine:
    system: str  # "darwin", "linux", "windows" or another platform.system() value, lowercased
    os_label: str  # "macOS 15.2", "Ubuntu 24.04 LTS"
    arch: str  # "arm64", "x86_64"
    ram_gb: float | None
    models_dir: Path
    free_gb: float | None
    apple_silicon: bool = False

    def describe(self) -> str:
        chip = "Apple Silicon" if self.apple_silicon else self.arch
        ram = f"{self.ram_gb:.0f} GB RAM" if self.ram_gb else "RAM unknown"
        return f"{self.os_label}, {chip}, {ram}"


class System:
    """Everything setup reads from or does to the machine. Tests replace the machine-level methods
    (platform, RAM, disk, PATH, subprocess, HTTP, codesign, network); the Ollama methods talk to
    whatever VEGA_BASE_URL points at (a fake server in tests)."""

    # machine
    def platform_system(self) -> str:
        return platform.system().lower()

    def arch(self) -> str:
        return platform.machine() or "unknown"

    def os_label(self) -> str:
        name = self.platform_system()
        if name == "darwin":
            release = platform.mac_ver()[0]
            return f"macOS {release}".strip()
        if name == "linux":
            with contextlib.suppress(OSError):
                for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
                    if line.startswith("PRETTY_NAME="):
                        return line.split("=", 1)[1].strip().strip('"')
            return "Linux"
        return platform.system() or "unknown OS"

    def apple_silicon(self) -> bool:
        if self.platform_system() != "darwin":
            return False
        if self.arch() == "arm64":
            return True
        # an x86_64 Python under Rosetta still runs on Apple Silicon
        code, out = self.capture(["sysctl", "-n", "hw.optional.arm64"])
        return code == 0 and out.strip() == "1"

    def ram_gb(self) -> float | None:
        return resources.total_memory_gb()

    def models_dir(self) -> Path:
        return model_store.models_dir()

    def free_gb(self, path: Path) -> float | None:
        try:
            return model_store.free_disk_gb(path)
        except OSError:
            return None

    def which(self, name: str) -> str | None:
        return shutil.which(name)

    def home(self) -> Path:
        return Path.home()

    def online(self) -> bool:
        try:
            with socket.create_connection(("ollama.com", 443), timeout=4):
                return True
        except OSError:
            return False

    def run(self, argv: list[str]) -> int:
        """Run a command with the terminal attached, so its own progress and prompts (sudo) show."""
        try:
            return subprocess.run(argv, check=False).returncode
        except OSError as exc:
            raise SetupError(f"Could not run {argv[0]}: {exc}") from exc

    def capture(self, argv: list[str]) -> tuple[int, str]:
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=120, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return 127, str(exc)
        return done.returncode, (done.stdout or "") + (done.stderr or "")

    def download(self, url: str, dest: Path, on_progress: Callable[[int, int], None]) -> None:
        if not url.startswith("https://"):
            raise SetupError(f"Refusing a download over plain HTTP: {url}")
        request = urllib.request.Request(url, headers={"User-Agent": f"vegacode/{version()}"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if not response.geturl().startswith("https://"):
                    raise SetupError(f"The download was redirected off HTTPS: {response.geturl()}")
                total = int(response.headers.get("Content-Length") or 0)
                done = 0
                with dest.open("wb") as out:
                    while chunk := response.read(1 << 20):
                        out.write(chunk)
                        done += len(chunk)
                        on_progress(done, total)
        except (urllib.error.URLError, OSError) as exc:
            raise SetupError(f"Downloading {url} failed: {exc}") from exc

    def temp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="vega-ollama-"))

    # Ollama and Vega's profiles
    def ollama_executable(self) -> str | None:
        return server.ollama_executable()

    def server_local(self) -> bool:
        return server.is_local()

    def server_running(self) -> bool:
        return server.is_running()

    def start_server(self) -> str:
        return server.start()

    def installed_tags(self) -> list[dict[str, Any]] | None:
        return model_store.installed()

    def catalog(self) -> dict[str, model_store.CatalogEntry]:
        return model_store.load_catalog()

    def profiles(self) -> tuple[str, dict[str, models.Profile]]:
        return models.load()

    def active_profile(self) -> str | None:
        return models.active_profile()

    def plan_pull(self, target: str, *, tags: list[dict[str, Any]], free_gb: float | None
                  ) -> model_store.PullPlan:
        return model_store.plan_pull(target, tags=tags, free_gb=free_gb)

    def pull(self, plan: model_store.PullPlan,
             on_progress: Callable[[model_store.Progress], None]) -> None:
        model_store.pull(plan, on_progress)

    def add_profile(self, entry: model_store.CatalogEntry) -> None:
        model_store.add_catalog_profile(entry)

    def use_profile(self, name: str) -> int:
        return models.use(name)


def machine(system: System) -> Machine:
    where = system.models_dir()
    return Machine(system.platform_system(), system.os_label(), system.arch(), system.ram_gb(),
                   where, system.free_gb(where), system.apple_silicon())


# --- What is there --------------------------------------------------------------------------------

@dataclass
class Status:
    applicable: bool = True  # False: another server or an endpoint profile; nothing to set up
    note: str = ""
    ollama: str | None = None
    server: bool = False
    server_error: str = ""
    model_checked: bool = True
    built: bool = False  # the `vega` model exists in Ollama
    profile: str = ""  # the profile `vega` is (to be) built from
    chosen: bool = False  # that profile was chosen before (active or recorded), not just default
    base: str = ""
    base_downloaded: bool = False

    @property
    def missing(self) -> list[str]:
        if not self.applicable:
            return []
        if not self.ollama:
            return ["Ollama is not installed"]
        if not self.server:
            return [f"the Ollama server does not start ({self.server_error or 'unknown error'})"]
        if self.model_checked and not self.built:
            if self.base_downloaded:
                return [f"Vega's model is not built from {self.profile}"]
            return ["no model is downloaded yet" if not self.chosen else
                    f"the model {self.base} ({self.profile}) is not downloaded"]
        return []

    @property
    def ready(self) -> bool:
        return not self.missing


VEGA_TAG = model_store.canonical(models.DEFAULT_MODEL)


def check(system: System, *, check_model: bool = True, start_server: bool = True) -> Status:
    """What Vega needs and whether it is there. Local only: never downloads anything."""
    status = Status(model_checked=check_model)
    if not system.server_local():
        status.applicable = False
        status.note = f"Vega uses the model server at {server.base_url()}; nothing to set up here."
        return status
    try:
        default, profiles = system.profiles()
    except (OSError, ValueError) as exc:
        raise SetupError(f"Vega's model profiles cannot be read: {exc}") from exc
    recorded = read_state().get("profile")
    active = system.active_profile()
    name = active if active in profiles else recorded if recorded in profiles else default
    status.profile, status.chosen = name, name in {active, recorded}
    profile = profiles[name]
    if profile.is_endpoint:
        status.applicable = False
        status.note = f"Profile {name} runs on {profile.endpoint}; nothing to set up here."
        return status
    status.base = profile.base
    status.ollama = system.ollama_executable()
    if not status.ollama:
        return status
    status.server = system.server_running()
    if not status.server and start_server:
        try:
            system.start_server()
            status.server = True
        except RuntimeError as exc:
            status.server_error = str(exc)
    if status.server and check_model:
        names = {model_store.canonical(t.get("name", "")) for t in system.installed_tags() or []}
        status.built = VEGA_TAG in names
        status.base_downloaded = model_store.canonical(profile.base) in names
    return status


# --- Model choice ---------------------------------------------------------------------------------

@dataclass
class Option:
    entry: model_store.CatalogEntry
    fit: str
    installed: bool

    def size_text(self) -> str:
        return "already downloaded" if self.installed else \
            f"{self.entry.download_gb:.1f} GB download"

    def fit_text(self, ram_gb: float | None) -> str:
        ram = f"your {ram_gb:.0f} GB RAM" if ram_gb else "unknown RAM"
        return {"fits": f"fits {ram}", "tight": f"fits {ram} only tightly"}.get(
            self.fit, f"RAM fit unknown ({ram})")

    def line(self, ram_gb: float | None, free_gb: float | None) -> str:
        """`Granite 8B · 5.3 GB download · fits your 16 GB RAM; 140 GB free`"""
        free = f"; {free_gb:.0f} GB free" if free_gb is not None else ""
        return f"{self.entry.display} · {self.size_text()} · {self.fit_text(ram_gb)}{free}"


@dataclass
class Choice:
    options: list[Option]  # the pick first
    reason: str = ""


def disk_ok(entry: model_store.CatalogEntry, free_gb: float | None, have: set[str]) -> bool:
    if model_store.canonical(entry.tag) in have or free_gb is None:
        return True
    return free_gb - entry.download_gb >= model_store.min_free_gb()


def choose_model(catalog: dict[str, model_store.CatalogEntry], ram_gb: float | None,
                 free_gb: float | None, tags: list[dict[str, Any]]) -> Choice:
    """The best catalogue model for this RAM and disk (model_store.recommend's ranking), with up to
    three alternatives. Raises SetupError saying what is needed when nothing fits."""
    have = {model_store.canonical(t.get("name", "")) for t in tags}
    best = model_store.recommend(ram_gb, tags=tags, catalog=catalog,
                                 allowed=lambda entry: disk_ok(entry, free_gb, have))
    if best is None:
        raise nothing_fits(catalog, ram_gb, free_gb)
    options = [Option(best.entry, best.fit, best.installed)]
    options += [Option(entry, verdict, model_store.canonical(entry.tag) in have)
                for entry, verdict in best.others]
    return Choice(options, best.reason)


def nothing_fits(catalog: dict[str, model_store.CatalogEntry], ram_gb: float | None,
                 free_gb: float | None) -> SetupError:
    smallest = min(catalog.values(), key=lambda entry: entry.download_gb)
    needs = []
    if model_store.fit(smallest.download_gb, ram_gb) == "won't fit":
        needs.append(f"about {smallest.ram_gb:.1f} GB of RAM to run (this machine has "
                     f"{ram_gb:.0f} GB)")
    margin = model_store.min_free_gb()
    disk_short = free_gb is not None and free_gb - smallest.download_gb < margin
    if disk_short:
        short = smallest.download_gb + margin - free_gb
        needs.append(f"{smallest.download_gb:.1f} GB of disk for the download plus the "
                     f"{margin:.0f} GB Vega keeps free; {free_gb:.1f} GB is free now, so free up "
                     f"{short:.1f} GB more")
    if not needs:  # every model is ruled out by RAM or disk, just not the smallest alone
        needs.append("more RAM or free disk than this machine has")
    message = (f"No model Vega offers fits this machine. The smallest, {smallest.display}, needs "
               + " and ".join(needs) + ".")
    manual = ("Free disk space where Ollama keeps models (or point OLLAMA_MODELS at a larger "
              "disk), then run `vega setup`.") if disk_short else \
        "Vega needs a machine with more memory for its smallest model."
    return SetupError(message, manual)


# --- The conversation with the user ---------------------------------------------------------------

class UI(Protocol):
    def say(self, text: str = "", style: str = "") -> None: ...

    def confirm(self, question: str) -> bool: ...

    def progress(self, description: str) -> contextlib.AbstractContextManager[
        Callable[[int, int, str], None]]: ...


class ConsoleUI:
    """Setup in the terminal (before the full-screen app opens: installers may ask for a sudo
    password, which needs the real terminal). `yes` answers every question with its default."""

    def __init__(self, console: Any = None, *, yes: bool = False) -> None:
        from rich.console import Console

        self.console = console or Console()
        self.yes = yes

    def say(self, text: str = "", style: str = "") -> None:
        self.console.print(text, style=style or None, highlight=False, markup=False)

    def confirm(self, question: str) -> bool:
        if self.yes:
            self.say(f"{question} yes (--yes)")
            return True
        from rich.prompt import Confirm

        return Confirm.ask(question, default=True, console=self.console)

    @contextlib.contextmanager
    def progress(self, description: str) -> Iterator[Callable[[int, int, str], None]]:
        from rich.progress import (
            BarColumn,
            DownloadColumn,
            Progress,
            TextColumn,
            TimeRemainingColumn,
            TransferSpeedColumn,
        )

        columns = (TextColumn("  {task.description:<24}"), BarColumn(), DownloadColumn(),
                   TransferSpeedColumn(), TimeRemainingColumn())
        with Progress(*columns, console=self.console, transient=False) as bar:
            task = bar.add_task(description, total=None)

            def update(completed: int, total: int, status: str = "") -> None:
                bar.update(task, completed=completed, total=total or None,
                           description=(status or description)[:24])

            yield update


# --- Installing Ollama ----------------------------------------------------------------------------

@dataclass
class InstallPlan:
    kind: str  # "brew", "mac-app", "linux-script"
    summary: list[str]  # what will happen, line by line, shown before asking
    manual: str  # the same thing done by hand, shown when it fails


def install_plan(system: System, where: Machine) -> InstallPlan:
    if where.system == "windows":
        raise SetupError(
            "Windows is not supported: Vega has never been run or tested on Windows, so setup "
            "stops here.",
            "Use macOS or Linux (on Windows, WSL 2 with Ubuntu is Linux, but Vega is untested "
            "there too).")
    if where.system == "darwin":
        brew = system.which("brew")
        if brew:
            prefix = Path(brew).parent.parent
            return InstallPlan("brew", [
                f"installed by Homebrew (brew install ollama; about 50 MB, in {prefix})",
            ], "brew install ollama   then run `vega` again")
        target = system.home() / "Applications" / "Ollama.app"
        return InstallPlan("mac-app", [
            ("the Ollama app from ollama.com (about 200 MB download, HTTPS), into "
             f"{_tilde(target, system.home())}"),
            (f"its code signature checked first (Developer ID, Ollama's Team ID {OLLAMA_TEAM_ID}, "
             "Gatekeeper); no administrator password needed"),
        ], f"Download Ollama from {DOWNLOAD_PAGE}, open it once, then run `vega` again")
    if where.system == "linux":
        return InstallPlan("linux-script", [
            f"Ollama's official install script ({LINUX_INSTALL}), into /usr/local",
            "with an `ollama` system service; the script asks for your password (sudo) itself.",
            "Read it first at https://ollama.com/install.sh.",
        ], f"{LINUX_INSTALL}   then run `vega` again")
    raise SetupError(f"Vega cannot install Ollama on {where.os_label}.",
                     f"Install Ollama from {DOWNLOAD_PAGE}, then run `vega` again")


def _tilde(path: Path, home: Path) -> str:
    try:
        return "~/" + str(path.relative_to(home))
    except ValueError:
        return str(path)


def install_ollama(ui: UI, system: System, where: Machine, *, plan: InstallPlan) -> str:
    """Install Ollama as `plan` says (already confirmed). Returns the `ollama` command's path."""
    if plan.kind == "brew":
        brew = system.which("brew") or "brew"
        if system.run([brew, "install", "ollama"]) != 0:
            raise SetupError("`brew install ollama` failed (its output is above).", plan.manual)
    elif plan.kind == "linux-script":
        if not system.which("curl"):
            raise SetupError("The install script needs curl, which is not installed.",
                             f"Install curl (e.g. sudo apt install curl), then: {LINUX_INSTALL}")
        if system.run(["sh", "-c", LINUX_INSTALL]) != 0:
            raise SetupError("Ollama's install script failed (its output is above).", plan.manual)
    else:
        install_mac_app(ui, system, plan)
    found = system.ollama_executable()
    if not found:
        raise SetupError("Ollama was installed, but Vega cannot find the `ollama` command.",
                         plan.manual)
    return found


@dataclass
class Signature:
    ok: bool
    authority: str = ""
    team_id: str = ""
    notarized: bool = False
    problem: str = ""


def check_signature(system: System, app: Path) -> Signature:
    """codesign's verdict on the app bundle, who signed it and Gatekeeper's assessment."""
    code, out = system.capture(["codesign", "--verify", "--deep", "--strict", "--verbose=2",
                                str(app)])
    if code != 0:
        return Signature(False, problem=f"codesign --verify failed: {out.strip()[:300]}")
    _, info = system.capture(["codesign", "-dv", "--verbose=4", str(app)])
    authorities = re.findall(r"^Authority=(.+)$", info, re.MULTILINE)
    team = re.search(r"^TeamIdentifier=(\S+)$", info, re.MULTILINE)
    leaf = authorities[0].strip() if authorities else ""
    sig = Signature(True, leaf, team.group(1) if team else "")
    if not leaf.startswith("Developer ID Application:"):
        return Signature(False, leaf, sig.team_id,
                         problem=f"it is signed by {leaf or 'nobody'!r}, not a Developer ID "
                                 "Application certificate")
    pinned = os.environ.get(TEAM_ID_ENV, "").strip() or OLLAMA_TEAM_ID
    if sig.team_id != pinned:
        return Signature(False, leaf, sig.team_id,
                         problem=f"its Team ID is {sig.team_id or 'missing'}, not Ollama's "
                                 f"{pinned}")
    if not sig.team_id or f"({sig.team_id})" not in leaf:
        return Signature(False, leaf, sig.team_id,
                         problem="its Team ID does not match the signing certificate")
    code, out = system.capture(["spctl", "--assess", "--type", "execute", "--verbose=2",
                                str(app)])
    if "rejected" in out:
        return Signature(False, leaf, sig.team_id, problem=f"Gatekeeper rejects it: {out.strip()}")
    sig.notarized = code == 0 and "Notarized" in out
    return sig


def install_mac_app(ui: UI, system: System, plan: InstallPlan) -> None:
    target = system.home() / "Applications" / "Ollama.app"
    if target.exists():
        raise SetupError(f"{target} already exists but has no working `ollama` command; Vega "
                         "does not overwrite it.", f"Remove or repair {target}, then run "
                         "`vega setup`")
    work = system.temp_dir()
    try:
        archive = work / "Ollama-darwin.zip"
        with ui.progress("Ollama app") as update:
            system.download(MAC_APP_URL, archive, lambda done, total: update(done, total, ""))
        unpacked = work / "unpacked"
        if system.run(["ditto", "-x", "-k", str(archive), str(unpacked)]) != 0:
            raise SetupError("The Ollama archive could not be unpacked.", plan.manual)
        app = unpacked / "Ollama.app"
        if not app.is_dir():
            raise SetupError("The Ollama archive does not contain Ollama.app.", plan.manual)
        sig = check_signature(system, app)
        if not sig.ok:
            raise SetupError(f"Not installing the downloaded Ollama app: {sig.problem}.",
                             plan.manual)
        signer = sig.authority.removeprefix("Developer ID Application:").strip()
        signer = signer.removesuffix(f"({sig.team_id})").strip()
        ui.say(f"  Signature valid: {signer}, Team ID {sig.team_id}; "
               + ("notarized by Apple." if sig.notarized else
                  "Gatekeeper did not confirm notarization."))
        target.parent.mkdir(parents=True, exist_ok=True)
        if system.run(["ditto", str(app), str(target)]) != 0:
            raise SetupError(f"Copying Ollama.app to {target.parent} failed.", plan.manual)
        write_state(ollama_team_id=sig.team_id, ollama_signer=sig.authority)
    finally:
        shutil.rmtree(work, ignore_errors=True)


# --- The setup run --------------------------------------------------------------------------------

@dataclass
class Result:
    profile: str = ""
    installed_ollama: bool = False
    downloaded: str = ""
    notes: list[str] = field(default_factory=list)


def run_setup(ui: UI, system: System | None = None, *, model: str | None = None,
              first: bool = True) -> Result:
    """Set up whatever is missing: the plan first (what is installed and downloaded, how big,
    where), then a yes/no before installing Ollama and before the model download. Raises
    SetupError (with the manual step) when it cannot finish or the user says no."""
    system = system or System()
    result = Result()
    status = check(system, start_server=False)
    ui.say("Vega setup" if first else "Vega: repairing setup", style="bold")
    if not status.applicable:
        ui.say(f"  {status.note}")
        write_state(complete=True, completed_at=_now(), version=version())
        return result
    where = machine(system)
    home = system.home()
    ui.say(f"  Machine   {where.describe()}")
    free = "unknown" if where.free_gb is None else f"{where.free_gb:.1f} GB free"
    ui.say(f"  Disk      {free} in {_tilde(where.models_dir, home)} (Ollama's models)")
    online = system.online()
    ui.say(f"  Network   {'online' if online else 'offline'}")
    ui.say()

    install = None
    tags: list[dict[str, Any]] = []
    if status.ollama:
        _start_server(ui, system, status.ollama)
        status = check(system, start_server=False)
        tags = system.installed_tags() or []
        if model is None and status.built:
            ui.say(f"  Ollama    installed: {status.ollama}")
            ui.say(f"  Model     Vega's model is ready (profile {status.profile}, {status.base}).")
            return _finish(ui, result, status.profile)
    else:
        install = install_plan(system, where)
        if not online:
            raise SetupError("Ollama is not installed and this machine is offline, so Vega "
                             "cannot download it.", f"Connect to the internet and run `vega` "
                             f"again, or: {install.manual}")
    pick = _pick(system, status, model, where, tags, online=online)

    # The plan, before anything is asked.
    ui.say("Vega will set up:", style="bold")
    if install is None:
        ui.say(f"  Ollama    installed: {_tilde(Path(status.ollama or ''), home)}")
    else:
        ui.say(f"  Ollama    {install.summary[0]}")
        for line in install.summary[1:]:
            ui.say(f"            {line}")
    ui.say(f"  Model     {pick.plan_line(where, home)}")
    if pick.why:
        ui.say(f"            {pick.why}")
    ui.say("  Then      builds Vega's model from it (`vega`, the default from now on)")
    ui.say(f"  Vega keeps at least {model_store.min_free_gb():.0f} GB of disk free after a "
           "download.", style="dim")
    ui.say()

    # 1. Ollama
    if install is not None:
        if not ui.confirm("  Install Ollama now?"):
            raise SetupError("Ollama was not installed.", install.manual)
        status.ollama = install_ollama(ui, system, where, plan=install)
        result.installed_ollama = True
        ui.say(f"  Installed: {_tilde(Path(status.ollama), home)}", style="green")
        _start_server(ui, system, status.ollama)
        where = machine(system)  # the Linux service keeps models elsewhere
        tags = system.installed_tags() or []
        if model is None and check(system, start_server=False).built:  # reinstalled: all there
            ui.say(f"  Vega's model is ready (profile {status.profile}, {status.base}).")
            return _finish(ui, result, status.profile)

    # 2. The model
    _download(ui, system, pick, tags, where.free_gb, result, online=online)

    # 3. Vega's model, the default from now on
    ui.say(f"  Building Vega's model (`vega`) from profile {pick.name}...")
    if system.use_profile(pick.name) != 0:
        raise SetupError(f"Building Vega's model from {pick.name} failed (Ollama's output is "
                         "above).", f"vega model use {pick.name}")
    return _finish(ui, result, pick.name)


def _start_server(ui: UI, system: System, ollama: str) -> None:
    if system.server_running():
        return
    ui.say("  Starting the Ollama server...")
    try:
        system.start_server()
    except RuntimeError as exc:
        raise SetupError(f"The Ollama server did not start: {exc}",
                         f"Run `{ollama} serve` in another terminal to see why, then run `vega` "
                         "again") from exc


CHOSEN_BEFORE = "the model chosen before"


@dataclass
class Pick:
    """The model setup downloads (or reuses) and builds `vega` from."""

    name: str  # profile / catalogue name
    display: str
    size_gb: float | None  # the download, from the catalogue (None: unknown)
    installed: bool = False  # the weights are already downloaded
    why: str = ""  # shown under the plan line
    alternatives: list[Option] = field(default_factory=list)

    def size_text(self) -> str:
        return "unknown size" if self.size_gb is None else f"{self.size_gb:.1f} GB"

    def plan_line(self, where: Machine, home: Path) -> str:
        if self.installed:
            return f"{self.display}: already downloaded"
        return (f"{self.display}: a {self.size_text()} download into "
                f"{_tilde(where.models_dir, home)}")


def _pick(system: System, status: Status, model: str | None, where: Machine,
          tags: list[dict[str, Any]], *, online: bool) -> Pick:
    catalog = system.catalog()
    have = {model_store.canonical(t.get("name", "")) for t in tags}

    def from_name(name: str, why: str = "") -> Pick:
        entry = catalog.get(name)
        if entry is not None:
            return Pick(name, entry.display, entry.download_gb,
                        model_store.canonical(entry.tag) in have, why)
        base = system.profiles()[1][name].base
        return Pick(name, name, None, model_store.canonical(base) in have, why)

    if model is not None:
        if model not in catalog and model not in system.profiles()[1]:
            raise SetupError(f"Unknown model {model!r}. Choose one of: {', '.join(catalog)}.")
        return from_name(model)
    if status.chosen or status.base_downloaded:  # repair: the model chosen before, or here already
        return from_name(status.profile, CHOSEN_BEFORE if status.chosen else "")
    if not online:
        raise SetupError("This machine is offline, so no model can be downloaded.",
                         "Connect to the internet and run `vega` again.")
    choice = choose_model(catalog, where.ram_gb, where.free_gb, tags)
    best = choice.options[0]
    free = f"; {where.free_gb:.0f} GB free" if where.free_gb is not None else ""
    return Pick(best.entry.name, best.entry.display, best.entry.download_gb, best.installed,
                f"the best fit for this machine: {best.fit_text(where.ram_gb)}{free}",
                choice.options[1:])


def _download(ui: UI, system: System, pick: Pick, tags: list[dict[str, Any]],
              free_gb: float | None, result: Result, *, online: bool) -> None:
    try:
        plan = system.plan_pull(pick.name, tags=tags, free_gb=free_gb)
    except (ValueError, OSError) as exc:
        raise SetupError(str(exc), f"vega model pull {pick.name}") from exc
    if plan.refusal:
        raise SetupError(plan.refusal, "Free disk space, then run `vega setup`.")
    if plan.new_profile is not None:
        system.add_profile(plan.new_profile)
    write_state(profile=pick.name, complete=False)  # a stopped download resumes with this model
    if plan.already:
        ui.say(f"  {pick.display} ({plan.tag}) is already downloaded.")
        return
    if not online:
        raise SetupError(f"This machine is offline, so {plan.tag} cannot be downloaded.",
                         "Connect to the internet and run `vega` again.")
    size = f"{plan.size_gb:.1f} GB" if plan.size_gb else pick.size_text()
    if pick.why == CHOSEN_BEFORE:
        ui.say(f"  {pick.display} is not downloaded yet; a part fetched before is kept.")
    if not ui.confirm(f"  Download {pick.display} now ({size})?"):
        raise SetupError("No model was downloaded.", _alternatives(pick))
    try:
        with ui.progress(plan.tag) as update:
            system.pull(plan, lambda event: update(event.completed, event.total, plan.tag))
    except KeyboardInterrupt as exc:
        raise SetupError("Download paused.", "Run `vega` again (or `vega setup`) to continue: "
                         "Ollama keeps the part already downloaded.") from exc
    except model_store.OllamaError as exc:
        raise SetupError(f"The download stopped: {exc}", "Run `vega` again (or `vega setup`) "
                         "to continue: Ollama keeps the part already downloaded.") from exc
    result.downloaded = plan.tag
    ui.say(f"  Downloaded {pick.display} ({plan.tag}).", style="green")


def _alternatives(pick: Pick) -> str:
    if not pick.alternatives:
        return f"Run `vega` again when you are ready (it offers {pick.display} again)."
    shown = "; ".join(f"{o.entry.name} ({o.entry.display}, {o.size_text()})"
                      for o in pick.alternatives)
    return (f"Run `vega` again when you are ready, or pick another model with "
            f"`vega setup --model NAME`: {shown}.")


def _finish(ui: UI, result: Result, profile: str) -> Result:
    result.profile = profile
    write_state(complete=True, profile=profile, completed_at=_now(), version=version())
    ui.say()
    ui.say(f"Vega is ready (model profile {profile}).", style="bold green")
    return result


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --- At startup -----------------------------------------------------------------------------------

def ensure_ready(*, interactive: bool, check_model: bool = True, system: System | None = None,
                 ui: UI | None = None) -> int:
    """Called when `vega`, `vega chat`, `vega run` or `vega auto` starts: 0 when Vega can go on
    (setup ran first if something was missing), else an exit code after saying why."""
    if os.environ.get(SKIP_ENV):
        return 0
    system = system or System()
    ui = ui or ConsoleUI()
    try:
        status = check(system, check_model=check_model)
    except SetupError as exc:
        report(ui, exc)
        return 1
    if status.ready:
        if status.applicable and not read_state().get("complete") and check_model:
            write_state(complete=True, profile=status.profile, completed_at=_now(),
                        version=version())
        return 0
    first = not read_state().get("complete")
    if not interactive:
        ui.say(f"Vega is not set up yet: {'; '.join(status.missing)}." if first else
               f"Vega's setup is incomplete: {'; '.join(status.missing)}.")
        ui.say(f"Run this once (it installs and downloads what is missing), then try again:\n"
               f"  {SETUP_COMMAND}")
        return 1
    try:
        run_setup(ui, system, first=first)
    except SetupError as exc:
        report(ui, exc)
        return 1
    ui.say()
    return 0


def report(ui: UI, exc: SetupError) -> None:
    ui.say(str(exc), style="red")
    if exc.manual:
        ui.say(f"To do it yourself: {exc.manual}")


def interactive_terminal() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


# --- Entry points (cli.main, `vega setup`) ---------------------------------------------------------

def at_start(ctx: typer.Context, model: str | None) -> None:
    """cli.main calls this first. For `vega`, chat, run and auto: check (and at a terminal set
    up) Ollama and the `vega` model; with an explicit model only Ollama. Exits when it cannot."""
    if ctx.invoked_subcommand not in START_COMMANDS or ctx.resilient_parsing:
        return
    code = before_start(ctx.invoked_subcommand,
                        model_given=bool(model or os.environ.get("VEGA_MODEL")))
    if code:
        raise typer.Exit(code)


def before_start(command: str | None, *, model_given: bool) -> int:
    """`vega`, chat, run or auto is starting: check (and at a terminal repair) Ollama and the
    `vega` model; with an explicit model only Ollama. 0 to go on, else the exit code."""
    return ensure_ready(interactive=interactive_terminal(), check_model=not model_given)


def setup_command(
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask: install Ollama if it is "
                             "missing and download the recommended model."),
    model: str = typer.Option(None, "--model", "-m", help="Download this catalogue model or "
                              "profile instead of the automatic choice."),
) -> None:
    """Set Vega up (also runs by itself when `vega` starts and something is missing): install
    Ollama if needed, pick the best model for this machine's RAM and disk, download it and build
    Vega's model. Safe to run again: it repairs only what is missing."""
    from .cli import console

    if not yes and not interactive_terminal():
        console.print("`vega setup` asks before installing or downloading. Without a terminal, "
                      f"run:\n  {SETUP_COMMAND}", highlight=False)
        raise typer.Exit(1)
    ui = ConsoleUI(console, yes=yes)
    try:
        run_setup(ui, model=model, first=not read_state().get("complete"))
    except SetupError as exc:
        report(ui, exc)
        raise typer.Exit(1) from exc
    console.print("Start a chat in a project folder with `vega`.", highlight=False)
