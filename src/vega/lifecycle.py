"""`vega update` and `vega uninstall` for people who installed `vegacode` from PyPI.

How Vega was installed decides what both commands do (detect):

- launcher: the environment the Python 3.9 launcher made (vega._bootstrap: ~/.local/share/vega/venv,
  with ~/.local/bin/vega and vegacode linked to it and a marked PATH line in the shell's startup
  file). Vega owns that whole environment.
- pipx / uv: `pipx install vegacode` / `uv tool install vegacode`; that tool upgrades and removes.
- venv: `pip install vegacode` inside someone's virtual environment; only the package is Vega's.
- user / system: `pip install [--user] vegacode` into a Python's own site-packages.
- dev: an editable install or a source checkout. `vega update` refuses (use git) and `vega
  uninstall` leaves the package alone.

`vega update` asks PyPI's JSON API for the latest vegacode, shows both versions, asks, upgrades
with the installer that matches, then runs the new `vega --version` to check it took.

`vega uninstall` lists everything it can remove, with sizes, in five groups, and asks per group:
Vega itself, the PATH lines (only those under vega._bootstrap.PATH_MARK), the LaunchAgents
dev.vega.* (macOS), the data folder (VEGA_HOME, ~/.vega; kept by default) and the Ollama models
Vega pulled or built (kept by default; removing them takes a typed "yes" and names each model and
its size). It never removes Ollama, Python or Homebrew. Running it again removes nothing new.

Everything outside the process goes through `Host`, which the tests replace with a fake install
in a temp folder. The macOS-only parts are the LaunchAgents (launch_agents, unload_agent).
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import site
import subprocess
import sys
import sysconfig
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import _bootstrap as boot

PACKAGE = "vegacode"
PYPI_URL = "https://pypi.org/pypi/vegacode/json"
AGENT_PREFIX = "dev.vega."
LAUNCH_AGENTS_ENV = "VEGA_LAUNCH_AGENTS_DIR"  # as vega_lab.night: tests move the folder
NEVER_REMOVED = "Ollama itself, Python and Homebrew are never removed."
# Startup files the launcher may have written its PATH line to (vega._bootstrap.shell_rc).
RC_FILES = (".zshrc", ".bash_profile", ".bashrc", ".profile", ".config/fish/conf.d/vega.fish")
PATH_LINES = ('export PATH="$HOME/.local/bin:$PATH"', "fish_add_path $HOME/.local/bin")

KINDS = {
    "launcher": "the launcher's own environment",
    "pipx": "pipx",
    "uv": "uv tool",
    "venv": "pip, in a virtual environment",
    "user": "pip --user",
    "system": "pip, into a Python's own site-packages",
    "dev": "a source checkout (editable install)",
}


# --- sizes and paths ----------------------------------------------------------------------------

def size_of(path: Path) -> int:
    """Bytes under `path` (a file or a folder; symlinks are not followed)."""
    try:
        if path.is_symlink() or path.is_file():
            return path.lstat().st_size
    except OSError:
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def human(size: float) -> str:
    """Decimal units, as Ollama and the model catalogue show sizes."""
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if size >= scale:
            return f"{size / scale:.1f} {unit}"
    return f"{int(size)} B"


def inside(path: Path, folder: Path) -> bool:
    try:
        Path(os.path.realpath(path)).relative_to(os.path.realpath(folder))
        return True
    except ValueError:
        return False


# --- the outside world --------------------------------------------------------------------------

def _fetch_pypi(url: str = PYPI_URL, timeout: float = 10.0) -> bytes:
    request = urllib.request.Request(url, headers={"Accept": "application/json",
                                                   "User-Agent": "vegacode-update"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _run(argv: list[str]) -> int:
    print("$ " + " ".join(argv), flush=True)
    return subprocess.run(argv, check=False).returncode


def _capture(argv: list[str]) -> tuple[int, str]:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)
    return done.returncode, (done.stdout or "") + (done.stderr or "")


def _ollama_tags() -> list[dict[str, Any]] | None:
    from . import model_store

    return model_store.installed()


def _ollama_delete(name: str) -> None:
    from . import model_store

    model_store.delete(name)


def _ollama_blobs() -> dict[str, dict[str, int]] | None:
    from . import model_store

    return model_store.manifest_blobs()


def _own_distribution() -> tuple[str | None, Path | None, dict | None]:
    """(version, dist-info folder, PEP 610 direct_url.json) of the running vegacode."""
    from importlib import metadata

    try:
        dist = metadata.distribution(PACKAGE)
    except metadata.PackageNotFoundError:
        return None, None, None
    path = getattr(dist, "_path", None)
    try:
        text = dist.read_text("direct_url.json")
        direct = json.loads(text) if text else None
    except ValueError:
        direct = None
    return dist.version, Path(path) if path else None, direct


@dataclass
class Host:
    """This machine and this process, as the two commands see them."""

    env: Mapping[str, str]
    home: Path
    system: str  # platform.system(): "Darwin", "Linux", "Windows"
    prefix: Path  # sys.prefix
    base_prefix: Path  # sys.base_prefix
    executable: Path  # sys.executable
    package_dir: Path  # the imported vega package folder
    version: str | None
    dist_info: Path | None
    direct_url: dict | None
    user_site: Path | None
    python_version: str = f"{sys.version_info[0]}.{sys.version_info[1]}"  # this interpreter's
    run: Callable[[list[str]], int] = _run
    capture: Callable[[list[str]], tuple[int, str]] = _capture
    fetch: Callable[[str], bytes] = _fetch_pypi
    which: Callable[[str], str | None] = shutil.which
    ollama_tags: Callable[[], list[dict[str, Any]] | None] = _ollama_tags
    ollama_delete: Callable[[str], None] = _ollama_delete
    ollama_blobs: Callable[[], dict[str, dict[str, int]] | None] = _ollama_blobs

    @classmethod
    def current(cls) -> Host:
        version, dist_info, direct = _own_distribution()
        env = os.environ
        try:
            user_site = Path(site.getusersitepackages())
        except (AttributeError, OSError):
            user_site = None
        return cls(env=env, home=Path(env.get("HOME") or Path.home()), system=platform.system(),
                   prefix=Path(sys.prefix), base_prefix=Path(sys.base_prefix),
                   executable=Path(sys.executable), package_dir=Path(__file__).resolve().parent,
                   version=version, dist_info=dist_info, direct_url=direct, user_site=user_site)

    @property
    def vega_home(self) -> Path:
        return Path(self.env.get("VEGA_HOME") or self.home / ".vega").expanduser()

    def tilde(self, path: Path) -> str:
        return boot.tilde(path, self.home)


# --- how Vega was installed ---------------------------------------------------------------------

@dataclass
class Install:
    kind: str  # a KINDS key
    version: str
    prefix: Path  # the environment (sys.prefix)
    python: Path
    source: Path | None = None  # the checkout, for "dev"

    @property
    def label(self) -> str:
        return KINDS[self.kind]


def _checkout_of(package_dir: Path) -> Path | None:
    """The source checkout `package_dir` (…/src/vega) sits in, when it is one."""
    root = package_dir.parent.parent
    if package_dir.parent.name == "src" and (root / "pyproject.toml").is_file():
        return root
    return None


def detect(host: Host) -> Install:
    version = host.version or "unknown"
    direct = host.direct_url or {}
    editable = bool((direct.get("dir_info") or {}).get("editable"))
    in_site_packages = any(part in ("site-packages", "dist-packages")
                           for part in host.package_dir.parts)
    if editable or not in_site_packages:
        source = None
        url = str(direct.get("url") or "")
        if url.startswith("file://"):
            from urllib.parse import unquote, urlparse

            source = Path(unquote(urlparse(url).path))
        return Install("dev", version, host.prefix, host.executable,
                       source or _checkout_of(host.package_dir))
    prefix = Path(os.path.realpath(host.prefix))
    if prefix == Path(os.path.realpath(boot.venv_dir(host.env, host.home))):
        return Install("launcher", version, host.prefix, host.executable)
    if (host.prefix / "pipx_metadata.json").exists():
        return Install("pipx", version, host.prefix, host.executable)
    if (host.prefix / "uv-receipt.toml").exists():
        return Install("uv", version, host.prefix, host.executable)
    if os.path.realpath(host.prefix) != os.path.realpath(host.base_prefix):
        return Install("venv", version, host.prefix, host.executable)
    if host.user_site and host.dist_info and inside(host.dist_info, host.user_site):
        return Install("user", version, host.prefix, host.executable)
    return Install("system", version, host.prefix, host.executable)


def command_path(install: Install, system: str, name: str = "vega") -> Path:
    """The `vega` command of this install (next to its interpreter, or the Python's scripts
    folder for a --user install)."""
    exe = name + (".exe" if system == "Windows" else "")
    beside = install.python.parent / exe
    if beside.exists() or install.kind != "user":
        return beside
    scripts = sysconfig.get_path("scripts", f"{os.name}_user")
    return Path(scripts) / exe if scripts else beside


# --- vega update --------------------------------------------------------------------------------

class UpdateError(Exception):
    pass


def latest_version(host: Host) -> str:
    """The newest vegacode release on PyPI (its JSON API's info.version)."""
    try:
        data = json.loads(host.fetch(PYPI_URL).decode("utf-8"))
        version = str(data["info"]["version"])
    except (urllib.error.URLError, OSError) as exc:
        raise UpdateError(f"Could not reach PyPI ({exc}). Check the network and try again.") \
            from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise UpdateError(f"PyPI's answer for {PACKAGE} could not be read ({exc}).") from exc
    if boot.version_key(version) is None:
        raise UpdateError(f"PyPI lists {PACKAGE} {version!r}, which is not a version Vega reads.")
    return version


def upgrade_command(install: Install, host: Host, version: str) -> list[str]:
    pin = f"{PACKAGE}=={version}"
    if install.kind == "pipx":
        pipx = host.which("pipx")
        if not pipx:
            raise UpdateError("Vega was installed with pipx, but `pipx` is not on PATH. Run "
                              "`pipx upgrade vegacode` where pipx is available.")
        return [pipx, "install", "--force", pin]
    if install.kind == "uv":
        uv = host.which("uv")
        if not uv:
            raise UpdateError("Vega was installed with `uv tool`, but `uv` is not on PATH. Run "
                              "`uv tool upgrade vegacode` where uv is available.")
        # --force rebuilds the environment with whatever Python uv finds first unless told; with
        # macOS's 3.9 first on PATH that leaves only the launcher. Keep this install's Python.
        # (pipx install --force reuses the existing environment, so it keeps its Python.)
        return [uv, "tool", "install", "--force", "--python", host.python_version, pin]
    command = [str(install.python), "-m", "pip", "install", "--upgrade",
               "--disable-pip-version-check"]
    if install.kind == "user":
        command.append("--user")
    return command + [pin]


def installed_version_of(install: Install, host: Host) -> str | None:
    """What the install's `vega --version` prints now (`vegacode X`), or None."""
    code, output = host.capture([str(command_path(install, host.system)), "--version"])
    match = re.search(r"vegacode\s+(\S+)", output)
    return match.group(1) if code == 0 and match else None


def update(host: Host, *, check: bool, yes: bool, ask: Callable[[str, bool], bool],
           interactive: bool, say: Callable[[str], None] = print) -> int:
    install = detect(host)
    if install.kind == "dev":
        where = f" ({install.source})" if install.source else ""
        say(f"This is a source checkout{where}; use git to update it (git pull), not `vega update`.")
        return 1
    try:
        latest = latest_version(host)
    except UpdateError as exc:
        say(str(exc))
        return 1
    say(f"Installed: vegacode {install.version} ({install.label}, {host.tilde(install.prefix)})")
    say(f"On PyPI:   vegacode {latest}")
    if not boot.version_newer(latest, install.version):
        say("Vega is up to date.")
        return 0
    try:
        command = upgrade_command(install, host, latest)
    except UpdateError as exc:
        say(str(exc))
        return 1
    if install.kind == "system":
        say("Note: this Python's own site-packages; if pip refuses (an externally managed "
            "Python), reinstall Vega with pipx or the launcher.")
    if check:
        say(f"An update is available: {install.version} -> {latest}. Run `vega update` to "
            "install it.")
        return 0
    say("Will run: " + " ".join(command))
    if not yes:
        if not interactive:
            say("No terminal to ask in: run `vega update --yes` to update.")
            return 1
        if not ask(f"Update vegacode {install.version} -> {latest}?", True):
            say("Nothing changed.")
            return 1
    if host.run(command) != 0:
        say(f"The upgrade failed (the output above says why); vegacode {install.version} is "
            "still installed.")
        return 1
    now = installed_version_of(install, host)
    if now != latest:
        say(f"The upgrade ran, but `vega --version` now says {now or 'nothing readable'}, not "
            f"{latest}.")
        return 1
    say(f"Updated to vegacode {latest} (checked: `{command_path(install, host.system)} --version` "
        f"says vegacode {now}).")
    return 0


# --- vega uninstall: what is there --------------------------------------------------------------

@dataclass
class Item:
    text: str
    size: int | None = None


@dataclass
class Group:
    key: str  # package, path, agents, data, models
    title: str
    items: list[Item]
    default: bool  # the answer when the user just presses Enter (and with --yes)
    action: Callable[[], list[str]]  # removes it; returns notes
    strong: bool = False  # needs a typed "yes"
    note: str = ""
    summary: str = ""  # how the one-line summary names it
    total: int | None = None  # bytes freed, when not the items' sum (models sharing weights)
    empty: str = "nothing to remove"  # shown when there are no items

    @property
    def size(self) -> int:
        return self.total if self.total is not None else sum(item.size or 0 for item in self.items)


def launcher_copies(host: Host, install: Install | None) -> list[Path]:
    """vegacode dist-info folders of other Pythons' user site-packages: the copy macOS's
    Python 3.9 pip installed (only the launcher). Not the running one."""
    patterns = [host.home / "Library" / "Python", host.home / ".local" / "lib"]
    found: list[Path] = []
    for root in patterns:
        if not root.is_dir():
            continue
        for info in sorted(root.glob("*/**/site-packages/vegacode-*.dist-info")):
            if host.dist_info and os.path.realpath(info) == os.path.realpath(host.dist_info):
                continue
            found.append(info)
    return found


def _record_base(site_packages: Path) -> Path:
    """The installation prefix a RECORD's paths must stay in (e.g. ~/Library/Python/3.9)."""
    return site_packages.parents[2] if len(site_packages.parents) > 2 else site_packages


def record_files(dist_info: Path) -> list[Path]:
    """Files a dist-info's RECORD lists, kept to its installation prefix (site-packages' grand-
    parent, e.g. ~/Library/Python/3.9), so a bad RECORD cannot point anywhere else."""
    site_packages = dist_info.parent
    base = _record_base(site_packages)
    try:
        lines = (dist_info / "RECORD").read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    files: list[Path] = []
    for line in lines:
        name = line.split(",")[0].strip()
        if not name:
            continue
        path = Path(os.path.normpath(site_packages / name))
        if inside(path.parent, base) and path not in files:
            files.append(path)
    return files


def remove_record(dist_info: Path) -> None:
    """Remove what a dist-info's RECORD lists, then the folders that left empty (a package's
    vega/native/macos empties vega/native too), up to but not including site-packages and the
    installation prefix."""
    files = record_files(dist_info)
    stop = {dist_info.parent, _record_base(dist_info.parent)}
    folders = set()
    for path in files:
        if path.is_symlink() or path.is_file():
            path.unlink()
        folders.add(path.parent)
    for folder in sorted(folders, key=lambda p: len(p.parts), reverse=True):
        try:
            if (folder / "__pycache__").is_dir():
                shutil.rmtree(folder / "__pycache__")
            while folder not in stop and folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()
                folder = folder.parent
        except OSError:
            pass
    if dist_info.exists():
        shutil.rmtree(dist_info)


def own_links(host: Host, target_root: Path) -> list[Path]:
    """~/.local/bin/vega and vegacode when they are links into `target_root`."""
    links = []
    for name in boot.COMMANDS:
        link = boot.bin_dir(host.home) / name
        if link.is_symlink() and inside(Path(os.path.realpath(link)).parent, target_root):
            links.append(link)
    return links


def package_group(host: Host, install: Install) -> Group | None:
    copies = launcher_copies(host, install)
    items: list[Item] = []
    steps: list[Callable[[], None]] = []
    note = ""
    prefix_exists = install.prefix.exists()
    if install.kind == "dev":
        note = (f"Vega runs from a source checkout ({install.source or install.prefix}); left as "
                "it is. Delete the checkout yourself if you want it gone.")
    elif not prefix_exists:
        pass
    elif install.kind == "launcher":
        items.append(Item(f"{host.tilde(install.prefix)}  (Vega's own environment)",
                          size_of(install.prefix)))
        links = own_links(host, install.prefix)
        if links:
            items.append(Item(", ".join(host.tilde(link) for link in links) + "  (links)"))

        def remove_launcher_env() -> None:
            for link in own_links(host, install.prefix):
                link.unlink()
            shutil.rmtree(install.prefix, ignore_errors=True)
            parent = install.prefix.parent  # ~/.local/share/vega, when now empty
            if parent.name == "vega" and parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        steps.append(remove_launcher_env)
    elif install.kind in ("pipx", "uv"):
        tool = "pipx" if install.kind == "pipx" else "uv"
        items.append(Item(f"{host.tilde(install.prefix)}  ({tool}'s environment for vegacode)",
                          size_of(install.prefix)))
        argv = ([tool, "uninstall", PACKAGE] if tool == "pipx"
                else ["uv", "tool", "uninstall", PACKAGE])

        def remove_tool() -> None:
            found = host.which(tool)
            if not found:
                raise RuntimeError(f"`{tool}` is not on PATH; run `{' '.join(argv)}` yourself.")
            if host.run([found] + argv[1:]) != 0:
                raise RuntimeError(f"`{' '.join(argv)}` failed.")
        steps.append(remove_tool)
    else:  # pip into a venv, --user or a Python's own site-packages: only the package
        where = {"venv": f"the virtual environment {host.tilde(install.prefix)} stays",
                 "user": "your Python stays", "system": "your Python stays"}[install.kind]
        size = sum(size_of(p) for p in record_files(host.dist_info)) if host.dist_info else None
        items.append(Item(f"the vegacode package ({where}; its dependencies stay too)", size))
        command = [str(install.python), "-m", "pip", "uninstall", "-y",
                   "--disable-pip-version-check", PACKAGE]

        def remove_pip() -> None:
            if host.dist_info and not host.dist_info.exists():
                return
            if host.run(command) != 0:
                raise RuntimeError("`pip uninstall vegacode` failed.")
        steps.append(remove_pip)
    for info in copies:
        items.append(Item(f"{host.tilde(info.parent)}/{info.name}  (the launcher an older "
                          "Python's pip installed)", sum(size_of(p) for p in record_files(info))))
        steps.append(lambda info=info: remove_record(info))
    if not items:
        if note:
            return Group("package", "Vega itself", [], False, list, note=note,
                         summary="Vega itself", empty="left as it is")
        return None

    def act() -> list[str]:
        for step in steps:
            step()
        return []
    return Group("package", f"Vega itself: vegacode {install.version} ({install.label})", items, True, act, note=note, summary="Vega itself")


def strip_path_lines(text: str) -> tuple[str, list[str]]:
    """`text` without Vega's PATH blocks (its marker comment plus the PATH line right after it,
    and the blank line add_to_path put before them). Unmarked lines, even identical ones, stay."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    removed: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        following = lines[i + 1].strip() if i + 1 < len(lines) else None
        if line.strip() == boot.PATH_MARK and following in PATH_LINES:
            if out and not out[-1].strip():
                out.pop()
            removed += [line.rstrip("\n"), lines[i + 1].rstrip("\n")]
            i += 2
            continue
        out.append(line)
        i += 1
    return "".join(out), removed


def rc_edits(host: Host) -> list[tuple[Path, str, list[str]]]:
    edits = []
    for name in RC_FILES:
        path = host.home / name
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        new, removed = strip_path_lines(text)
        if removed:
            edits.append((path, new, removed))
    return edits


def path_group(host: Host) -> Group | None:
    edits = rc_edits(host)
    if not edits:
        return None
    items = [Item(f"{host.tilde(path)}: {removed[-1].strip()}  (under Vega's marker comment)")
             for path, _, removed in edits]
    folder = boot.bin_dir(host.home)
    others = sorted(p.name for p in folder.iterdir()
                    if p.name not in boot.COMMANDS and not p.name.startswith(".")) \
        if folder.is_dir() else []
    note = ""
    if others:
        shown = ", ".join(others[:5]) + (", ..." if len(others) > 5 else "")
        note = (f"{host.tilde(folder)} also holds other commands ({shown}); without this line new "
                "terminals will not find them. Kept unless you say so.")

    def act() -> list[str]:
        for path, _, _ in edits:  # re-read: the file may have changed since the listing
            text = path.read_text(encoding="utf-8")
            new, removed = strip_path_lines(text)
            if not removed:
                continue
            if path.name == "vega.fish" and not new.strip():
                path.unlink()
                continue
            temp = path.with_name(path.name + ".vega-tmp")
            temp.write_text(new, encoding="utf-8")
            shutil.copymode(path, temp)
            os.replace(temp, path)
        return []
    return Group("path", "PATH line Vega added", items, not others, act, note=note,
                 summary="PATH line" + ("s" if len(edits) > 1 else ""))


# macOS only: user LaunchAgents dev.vega.* (vega lab night / research schedules).
def launch_agents_dir(host: Host) -> Path | None:
    if host.env.get(LAUNCH_AGENTS_ENV):
        return Path(host.env[LAUNCH_AGENTS_ENV])
    return host.home / "Library" / "LaunchAgents" if host.system == "Darwin" else None


def launch_agents(host: Host) -> list[Path]:
    folder = launch_agents_dir(host)
    if folder is None or not folder.is_dir():
        return []
    return sorted(folder.glob(AGENT_PREFIX + "*.plist"))


def unload_agent(host: Host, plist: Path) -> None:
    if host.env.get(LAUNCH_AGENTS_ENV) or host.system != "Darwin":
        return  # a test folder: nothing is loaded from it
    label = plist.name[: -len(".plist")]
    host.capture(["launchctl", "bootout", f"gui/{os.getuid()}/{label}"])


def agents_group(host: Host) -> Group | None:
    agents = launch_agents(host)
    if not agents:
        return None

    def act() -> list[str]:
        for plist in agents:
            if plist.exists():
                unload_agent(host, plist)
                plist.unlink()
        return []
    return Group("agents", "Background jobs (LaunchAgents)",
                 [Item(host.tilde(p) + "  (stopped first)") for p in agents], True, act,
                 summary=f"{len(agents)} LaunchAgent" + ("s" if len(agents) > 1 else ""))


def data_group(host: Host) -> Group | None:
    home = host.vega_home
    if not home.exists():
        return None

    def act() -> list[str]:
        shutil.rmtree(home, ignore_errors=True)
        return []
    return Group("data", "Your Vega data",
                 [Item(f"{host.tilde(home)}  (settings, sessions, logs, knowledge, model "
                       "profiles)", size_of(home))], False, act, summary=host.tilde(home))


def vega_model_names(host: Host, tags: list[dict[str, Any]]) -> list[str]:
    """The Ollama models Vega made or pulled: `vega` and `vega-*` (built from profiles) and the
    base it set up (setup.json's profile from the catalogue; active_model.json's base)."""
    from . import model_store

    def read(name: str) -> dict:
        try:
            data = json.loads((host.vega_home / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    bases: set[str] = set()
    profile = read("setup.json").get("profile")
    if profile:
        try:
            entry = model_store.load_catalog().get(str(profile))
        except (OSError, ValueError):
            entry = None
        if entry:
            bases.add(model_store.canonical(entry.tag))
    base = read("active_model.json").get("base")
    if base:
        bases.add(model_store.canonical(str(base)))
    names = []
    for tag in tags:
        name = model_store.canonical(str(tag.get("name", "")))
        short = name.split(":")[0]
        if short == "vega" or short.startswith("vega-") or name in bases:
            names.append(name)
    return names


def models_group(host: Host) -> Group | None:
    from . import model_store

    tags = host.ollama_tags()
    if tags is None:
        return Group("models", "Ollama models", [], False, list,
                     note="Ollama is not running, so its models were not checked. Start it "
                          "(`ollama serve`) and run `vega uninstall` again to remove them.",
                     summary="Ollama models", empty="not checked")
    names = vega_model_names(host, tags)
    if not names:
        return None
    by_name = {model_store.canonical(str(t.get("name", ""))): t for t in tags}
    items = []
    for name in names:
        tag = by_name.get(name, {})
        parent = (tag.get("details") or {}).get("parent_model")
        if parent and model_store.canonical(str(parent)) in names:  # its weights are listed
            items.append(Item(f"{name}  (built on {parent}; shares its weights)"))
        else:
            items.append(Item(name, int(tag.get("size") or 0)))
    try:
        freed = model_store.space_freed(names, tags, host.ollama_blobs())
    except (OSError, ValueError):
        freed = None

    def act() -> list[str]:
        notes = []
        for name in names:
            try:
                host.ollama_delete(name)
            except Exception as exc:  # noqa: BLE001 - one model failing must not stop the rest
                notes.append(f"Could not remove {name}: {exc}")
        return notes
    group = Group("models", "Ollama models Vega pulled or built", items,
                  False, act, strong=True, summary="Ollama models")
    if freed is not None:
        group.total = int(freed.gb * model_store.GB)
        group.note = f"Removing them frees {freed}."
    return group


def plan(host: Host) -> tuple[Install, list[Group]]:
    install = detect(host)
    groups = [package_group(host, install), path_group(host), agents_group(host),
              data_group(host), models_group(host)]
    return install, [g for g in groups if g is not None]


def render(groups: list[Group], host: Host) -> list[str]:

    actionable = [g for g in groups if g.items]
    lines = ["What `vega uninstall` can remove:" if actionable
             else "Nothing of Vega's is left to remove.", ""]
    for number, group in enumerate(groups, 1):
        if group.items:
            default = "remove" if group.default else "keep"
            size = f", {human(group.size)}" if group.size else ""
            lines.append(f"  {number}. {group.title}{size}  [default: {default}]")
        else:
            lines.append(f"  {number}. {group.title}: {group.empty}")
        for item in group.items:
            lines.append(f"     {human(item.size) if item.size else '':>8}  {item.text}")
        if group.note:
            lines.append(f"     {'':>8}  Note: {group.note}")
    lines += ["", NEVER_REMOVED]
    return lines


# --- vega uninstall: doing it -------------------------------------------------------------------

def uninstall(host: Host, *, dry_run: bool, yes: bool, remove_data: bool, remove_models: bool,
              ask: Callable[[str, bool], bool], ask_text: Callable[[str], str],
              interactive: bool, say: Callable[[str], None] = print) -> int:
    _install, groups = plan(host)
    for line in render(groups, host):
        say(line)
    actionable = [g for g in groups if g.items]
    if dry_run:
        say("")
        say("Dry run: nothing was changed.")
        return 0
    if not actionable:
        return 0
    if not yes and not interactive:
        say("")
        say("No terminal to ask in: run `vega uninstall --yes` (keeps your data and models), "
            "adding --remove-data / --remove-models to remove those too.")
        return 1
    if not yes:
        say("")
    chosen: list[Group] = []
    for group in actionable:
        if group.key == "data" and remove_data or group.key == "models" and remove_models:
            chosen.append(group)
        elif yes:
            if group.default:
                chosen.append(group)
        elif group.strong:
            names = ", ".join(item.text.split()[0] + (f" ({human(item.size)})" if item.size
                                                      else " (shares those weights)")
                              for item in group.items)
            answer = ask_text(f"Remove these Ollama models: {names}? Type yes to remove them, "
                              "anything else keeps them: ")
            if answer.strip().lower() == "yes":
                chosen.append(group)
        elif ask(f"Remove {group.title[0].lower() + group.title[1:]}?", group.default):
            chosen.append(group)
    # Vega's own environment goes last: this process runs from it.
    chosen.sort(key=lambda g: g.key == "package")
    removed, failed = [], []
    for group in chosen:
        try:
            for note in group.action():
                say(note)
            removed.append(group)
        except (OSError, RuntimeError) as exc:
            say(f"Could not remove {group.summary}: {exc}")
            failed.append(group)
    removed.sort(key=actionable.index)  # the summary follows the listing
    kept = [g for g in actionable if g not in chosen]
    say("")
    say(summary_line(removed, kept, failed))
    return 1 if failed else 0


def summary_line(removed: list[Group], kept: list[Group], failed: list[Group]) -> str:
    def names(groups: list[Group]) -> str:
        return ", ".join(g.summary + (f" ({human(g.size)})" if g.size else "") for g in groups)

    parts = []
    if removed:
        parts.append("removed " + names(removed))
    if kept:
        parts.append("kept " + names(kept))
    if failed:
        parts.append("failed " + names(failed))
    return ("Uninstall: " + "; ".join(parts) + ".") if parts else "Uninstall: nothing changed."
