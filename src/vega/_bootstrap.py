"""Set Vega up under Python 3.11+ when it was pip-installed into an older Python.

macOS's own `pip3` is Python 3.9, so `pip3 install vegacode` there installs only the `vega`
package (every dependency is marked `python_version >= "3.11"`). Running `vega` (or
`python3 -m vega`) then lands here, which:

1. finds a Python 3.11+ already on the machine (PATH, Homebrew, python.org, uv, pyenv), or offers
   `uv python install 3.12` when uv is there, else `brew install python@3.12` (asks first; only
   in a terminal), and otherwise says how to get one;
2. with the user's yes (or VEGA_BOOTSTRAP_YES=1), creates a dedicated virtual environment
   (~/.local/share/vega/venv; XDG_DATA_HOME and VEGA_BOOTSTRAP_VENV move it) and pip-installs the
   same vegacode into it: the same version from PyPI, or the same wheel, folder or Git commit when
   this copy came from one. It never pip-installs into the Python it found, so Homebrew's
   externally managed Python (PEP 668) is left alone;
3. links ~/.local/bin/vega and ~/.local/bin/vegacode to the environment's commands (a different
   file already there is replaced only after asking), says how to put ~/.local/bin on PATH, and
4. runs the real `vega` with the original arguments.

Run again, it finds the environment ready and goes straight to step 4. Windows gets instructions
only. Python 3.9 syntax and the standard library only.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Callable, NamedTuple
from urllib.parse import unquote, urlparse

MIN_PYTHON = (3, 11)
SUPPORTED = ((3, 11), (3, 12), (3, 13))  # the versions CI tests; preferred over newer ones
BREW_FORMULA = "python@3.12"
UV_PYTHON = "3.12"
COMMANDS = ("vega", "vegacode")
YES_ENV = "VEGA_BOOTSTRAP_YES"  # consent to creating the environment without a terminal
VENV_ENV = "VEGA_BOOTSTRAP_VENV"  # where the environment goes
PYTHON_ENV = "VEGA_BOOTSTRAP_PYTHON"  # an interpreter to try first
UV_INSTALL = "curl -LsSf https://astral.sh/uv/install.sh | sh"
_PROBE = ("import sys\ntry:\n    import venv, ensurepip\n    ok = 1\nexcept ImportError:\n"
          "    ok = 0\nprint('%d.%d.%d %d' % (sys.version_info[:3] + (ok,)))")

Version = tuple[int, ...]


class Python(NamedTuple):
    path: Path
    version: Version
    has_venv: bool  # venv + ensurepip importable (Debian/Ubuntu split them into python3.X-venv)


def say(text: str = "") -> None:
    print(text, file=sys.stderr, flush=True)


def dotted(version: Sequence[int]) -> str:
    return ".".join(str(part) for part in version)


def tilde(path: Path, home: Path) -> str:
    try:
        return "~/" + str(Path(path).relative_to(home))
    except ValueError:
        return str(path)


# --- where things go ---------------------------------------------------------------------------

def venv_dir(env: Mapping[str, str], home: Path) -> Path:
    if env.get(VENV_ENV):
        return Path(env[VENV_ENV]).expanduser()
    data = env.get("XDG_DATA_HOME") or str(home / ".local" / "share")
    return Path(data).expanduser() / "vega" / "venv"


def bin_dir(home: Path) -> Path:
    return home / ".local" / "bin"


def venv_bin(venv: Path, system: str) -> Path:
    return venv / ("Scripts" if system == "Windows" else "bin")


# --- finding a Python 3.11+ --------------------------------------------------------------------

def candidates(env: Mapping[str, str], home: Path, system: str) -> list[Path]:
    """Interpreters worth probing, best guesses first (duplicates and missing files included;
    find_python skips them)."""
    minors = [m for _, m in reversed(SUPPORTED)]
    names = [f"python3.{m}" for m in minors] + ["python3"]
    found: list[Path] = []
    if env.get(PYTHON_ENV):
        found.append(Path(env[PYTHON_ENV]).expanduser())
    for folder in env.get("PATH", "").split(os.pathsep):
        if folder:
            found += [Path(folder) / name for name in names]
    brew_prefixes = (["/opt/homebrew", "/usr/local"] if system == "Darwin"
                     else ["/home/linuxbrew/.linuxbrew", "/usr/local"])
    for prefix in brew_prefixes:
        found += [Path(prefix) / "bin" / name for name in names]
        found += [Path(prefix) / "opt" / f"python@3.{m}" / "bin" / f"python3.{m}"
                  for m in minors]
    if system == "Darwin":  # python.org installers
        found += [Path(f"/Library/Frameworks/Python.framework/Versions/3.{m}/bin/python3.{m}")
                  for m in minors]
    else:
        found += [Path("/usr/bin") / name for name in names[:-1]]
    found += [bin_dir(home) / name for name in names[:-1]]  # `uv python install` links these
    data = Path(env.get("XDG_DATA_HOME") or str(home / ".local" / "share")).expanduser()
    uv_dir = Path(env.get("UV_PYTHON_INSTALL_DIR") or str(data / "uv" / "python")).expanduser()
    pyenv_dir = Path(env.get("PYENV_ROOT") or str(home / ".pyenv")).expanduser() / "versions"
    for root, pattern in ((uv_dir, "cpython-3.*/bin/python3"), (pyenv_dir, "3.*/bin/python3")):
        if root.is_dir():
            found += sorted(root.glob(pattern), reverse=True)
    return found


def probe(path: Path) -> Python | None:
    """Run the interpreter once to read its version and whether it can make a venv."""
    try:
        result = subprocess.run([str(path), "-I", "-c", _PROBE], capture_output=True, text=True,
                                timeout=20, check=False)
        version_text, ok = result.stdout.split()
        return Python(path, tuple(int(part) for part in version_text.split(".")), ok == "1")
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def find_python(env: Mapping[str, str], home: Path, system: str,
                probe_fn: Callable[[Path], Python | None] = probe,
                ) -> tuple[Python | None, list[Python]]:
    """The best Python 3.11+ that can make a venv, and the 3.11+ ones that cannot."""
    skip = {os.path.realpath(sys.executable)}
    if system == "Darwin":
        skip.add("/usr/bin/python3")  # the 3.9 stub; without the CLT it opens an install dialog
    usable: list[Python] = []
    no_venv: list[Python] = []
    for path in candidates(env, home, system):
        real = os.path.realpath(path)
        if real in skip or str(path) in skip or not os.path.isfile(real) \
                or not os.access(real, os.X_OK):
            continue
        skip.add(real)
        found = probe_fn(path)
        if found is None or found.version[:2] < MIN_PYTHON:
            continue
        (usable if found.has_venv else no_venv).append(found)
    usable.sort(key=lambda p: (p.version[:2] in SUPPORTED, p.version), reverse=True)
    return (usable[0] if usable else None), no_venv


# --- what to install ---------------------------------------------------------------------------

def own_version() -> str | None:
    from importlib import metadata

    try:
        return metadata.version("vegacode")
    except metadata.PackageNotFoundError:
        return None


def own_direct_url() -> dict | None:
    """PEP 610 direct_url.json of this vegacode, when it was installed from a file, folder or
    URL rather than an index."""
    from importlib import metadata

    try:
        text = metadata.distribution("vegacode").read_text("direct_url.json")
        return json.loads(text) if text else None
    except (metadata.PackageNotFoundError, ValueError):
        return None


def install_spec(version: str | None, direct_url: dict | None) -> tuple[list[str], str]:
    """pip arguments that install the same vegacode, and where it comes from (for the message)."""
    url = str((direct_url or {}).get("url") or "")
    vcs = (direct_url or {}).get("vcs_info")
    if url and isinstance(vcs, dict) and vcs.get("vcs") and vcs.get("commit_id"):
        return (["vegacode @ {}+{}@{}".format(vcs["vcs"], url, vcs["commit_id"])],
                "{} {} at {}".format(vcs["vcs"], url, str(vcs["commit_id"])[:12]))
    if url.startswith("file://"):
        local = Path(unquote(urlparse(url).path))
        if local.exists():
            return [str(local)], str(local)
    elif url and "archive_info" in (direct_url or {}):
        return ["vegacode @ " + url], url
    if version:
        return ["vegacode==" + version], "PyPI"
    return ["vegacode"], "PyPI (latest)"


def venv_vegacode_version(venv: Path) -> str | None:
    """The vegacode version installed in the environment, read from its dist-info folder name
    (no interpreter start)."""
    for pattern in ("lib/python3.*/site-packages/vegacode-*.dist-info",
                    "Lib/site-packages/vegacode-*.dist-info"):
        for info in sorted(venv.glob(pattern)):
            return info.name[len("vegacode-"):-len(".dist-info")]
    return None


def venv_python_ok(venv: Path, system: str, probe_fn: Callable[[Path], Python | None] = probe,
                   ) -> bool:
    exe = venv_bin(venv, system) / ("python.exe" if system == "Windows" else "python")
    found = probe_fn(exe) if exe.exists() else None
    return bool(found and found.version[:2] >= MIN_PYTHON)


# --- commands on PATH --------------------------------------------------------------------------

def is_own_launcher(path: Path) -> bool:
    """A pip-written `vega` script for this launcher (e.g. `pip install --user` on Linux puts
    the Python 3.9 one in ~/.local/bin): safe to replace with the link."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(4096)
    except OSError:
        return False
    return head.startswith(b"#!") and b"vega._launch" in head


def link_commands(source_bin: Path, links: Path, ask: Callable[[str], bool],
                  interactive: bool) -> tuple[dict[str, Path], list[str]]:
    """Point links/vega and links/vegacode at the environment's commands. Returns the command
    to run for each name and notes for the user. Never replaces someone else's file unasked."""
    commands: dict[str, Path] = {}
    notes: list[str] = []
    for name in COMMANDS:
        target = source_bin / name
        if not target.exists():
            continue
        link = links / name
        commands[name] = target
        if link.is_symlink() and os.path.realpath(link) == os.path.realpath(target):
            commands[name] = link
            continue
        if link.exists() or link.is_symlink():
            if is_own_launcher(link) and not link.is_symlink():
                notes.append(f"Replaced {link} (the Python {dotted(sys.version_info[:2])} "
                             "launcher) with a link to Vega.")
            else:
                where = os.path.realpath(link) if link.is_symlink() else "a file"
                question = (f"{link} already exists and points to {where}. "
                            f"Replace it with a link to {target}?")
                if not (interactive and ask(question)):
                    notes.append(f"Left {link} as it is; Vega's command is {target}.")
                    continue
        links.mkdir(parents=True, exist_ok=True)
        temporary = link.with_name("." + name + ".vega-link")
        if temporary.is_symlink() or temporary.exists():
            temporary.unlink()
        temporary.symlink_to(target)
        os.replace(str(temporary), str(link))
        commands[name] = link
    return commands, notes


def on_path(folder: Path, env: Mapping[str, str]) -> bool:
    wanted = os.path.realpath(folder)
    return any(entry and os.path.realpath(os.path.expanduser(entry)) == wanted
               for entry in env.get("PATH", "").split(os.pathsep))


def path_hint(folder: Path, env: Mapping[str, str], home: Path, system: str) -> list[str]:
    """Lines telling the user how to put `folder` on PATH for their shell ([] when it is)."""
    if on_path(folder, env):
        return []
    try:
        shown = "$HOME/" + str(folder.relative_to(home))
    except ValueError:
        shown = str(folder)
    shell = os.path.basename(env.get("SHELL", ""))
    if shell == "fish":
        command = f"fish_add_path {shown}"
    else:
        rc = {"zsh": "~/.zshrc", "bash": "~/.bash_profile" if system == "Darwin" else "~/.bashrc"
              }.get(shell, "~/.profile")
        command = f"echo 'export PATH=\"{shown}:$PATH\"' >> {rc}"
    return [(f"{tilde(folder, home)} is not on your PATH yet, so a new terminal will not find "
             "`vega`. Add it with:"), "", "    " + command, "",
            f"then open a new terminal. Until then, run {tilde(folder / 'vega', home)}."]


# --- getting a Python when there is none -------------------------------------------------------

def find_brew(env: Mapping[str, str]) -> str | None:
    found = shutil.which("brew", path=env.get("PATH"))
    if found:
        return found
    for path in ("/opt/homebrew/bin/brew", "/usr/local/bin/brew"):
        if os.access(path, os.X_OK):
            return path
    return None


def find_uv(env: Mapping[str, str], home: Path) -> str | None:
    found = shutil.which("uv", path=env.get("PATH"))
    if found:
        return found
    for path in (bin_dir(home) / "uv", home / ".cargo" / "bin" / "uv"):
        if os.access(path, os.X_OK):
            return str(path)
    return None


def no_python_help(system: str, brew: str | None, env: Mapping[str, str]) -> list[str]:
    lines = ["No Python 3.11 or newer was found. Install one, then run `python3 -m vega` again:",
             ""]
    if system == "Darwin":
        if brew:
            lines.append(f"    brew install {BREW_FORMULA}")
        lines.append("    or the installer from https://www.python.org/downloads/macos/")
    elif shutil.which("apt-get", path=env.get("PATH")):
        lines.append("    sudo apt install python3.12 python3.12-venv"
                     "   (older releases: the deadsnakes PPA)")
    elif shutil.which("dnf", path=env.get("PATH")):
        lines.append("    sudo dnf install python3.12")
    else:
        lines.append("    Python 3.12 from your package manager, or "
                     "https://www.python.org/downloads/")
    lines += ["", "Or skip all of this with uv, which fetches its own Python:", "",
              "    " + UV_INSTALL, "    uv tool install --python 3.12 vegacode"]
    return lines


# --- the run -----------------------------------------------------------------------------------

def _interactive() -> bool:
    try:
        return sys.stdin.isatty() and sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


def _ask(question: str, default: bool = False) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    sys.stderr.write(question + suffix)
    sys.stderr.flush()
    try:
        answer = input().strip().lower()
    except EOFError:
        return False
    return default if not answer else answer in ("y", "yes")


def _run(command: list[str]) -> int:
    say("$ " + " ".join(command))
    return subprocess.run(command, check=False).returncode


def _exec(command: Path, args: list[str]) -> int:
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(str(command), [str(command)] + list(args))
    return 0  # not reached


class Bootstrap:
    """One run, with the outside world behind attributes the tests replace."""

    def __init__(self, env: Mapping[str, str] | None = None, home: Path | None = None,
                 system: str | None = None) -> None:
        self.env: Mapping[str, str] = os.environ if env is None else env
        self.home = Path(self.env.get("HOME") or Path.home()) if home is None else home
        self.system = system or platform.system()
        self.interactive = _interactive
        self.ask = _ask
        self.run_command = _run
        self.exec_command = _exec
        self.probe = probe
        self.version = own_version
        self.direct_url = own_direct_url

    def manual_steps(self, python: Python, venv: Path, spec: list[str]) -> list[str]:
        vbin = venv_bin(venv, self.system)
        quoted = " ".join(f"'{part}'" if " " in part else part for part in spec)
        links = bin_dir(self.home)
        return [f"    {python.path} -m venv {venv}",
                f"    {vbin / 'python'} -m pip install {quoted}",
                f"    mkdir -p {links} && ln -sf {vbin / 'vega'} {vbin / 'vegacode'} {links}/"]

    def main(self, args: list[str]) -> int:
        here = f"Python {dotted(sys.version_info[:3])} ({sys.executable})"
        if self.system == "Windows":
            for line in [f"Vega needs Python 3.11 or newer; this is {here}.", "",
                         ("Install Python 3.12 from https://www.python.org/downloads/windows/, "
                         "then: py -3.12 -m pip install vegacode"), "",
                         "Vega has not been tested on Windows."]:
                say(line)
            return 1
        venv = venv_dir(self.env, self.home)
        vbin = venv_bin(venv, self.system)
        wanted = self.version()
        installed = venv_vegacode_version(venv)
        ready = (vbin / "vega").exists() and installed is not None \
            and (wanted is None or installed == wanted) and venv_python_ok(venv, self.system,
                                                                          self.probe)
        if not ready:
            code = self.setup(venv, wanted, installed, here)
            if code:
                return code
        commands, notes = link_commands(vbin, bin_dir(self.home), self.ask, self.interactive())
        for note in notes:
            say(note)
        if not ready:
            hint = path_hint(bin_dir(self.home), self.env, self.home, self.system)
            for line in hint:
                say(line)
            say("Vega is set up. Starting it...\n")
        return self.exec_command(commands.get("vega", vbin / "vega"), args)

    def setup(self, venv: Path, wanted: str | None, installed: str | None,
              here: str) -> int:
        spec, source = install_spec(wanted, self.direct_url())
        upgrade = installed is not None and venv_python_ok(venv, self.system, self.probe)
        if upgrade:
            say(f"Updating Vega's environment ({tilde(venv, self.home)}) from vegacode "
                f"{installed} to {wanted or 'this copy'}.")
            return self.install(venv, spec)
        say(f"Vega needs Python 3.11 or newer; this is {here}.")
        say("pip installed only a small launcher. It can set Vega up in its own environment.")
        say()
        python, no_venv = find_python(self.env, self.home, self.system, self.probe)
        if python is None:
            python = self.get_python(no_venv)
            if python is None:
                return 1
        vbin = venv_bin(venv, self.system)
        plan = [("Python", f"{python.path} ({dotted(python.version)})"),
                ("Environment", tilde(venv, self.home)),
                ("Install", f"vegacode from {source} with its dependencies (about 120 MB)"),
                ("Commands", ", ".join(tilde(bin_dir(self.home) / n, self.home)
                                       for n in COMMANDS))]
        for label, value in plan:
            say(f"  {label + ':':<12} {value}")
        say()
        say(f"Nothing is installed into {python.path} itself, and nothing outside those "
            "folders changes.")
        if venv.exists() and any(venv.iterdir()) and not (venv / "pyvenv.cfg").exists():
            say(f"{venv} exists and is not a virtual environment; move it, or set {VENV_ENV} "
                "to another folder.")
            return 1
        if self.env.get(YES_ENV) == "1":
            consent = True
        elif self.interactive():
            consent = self.ask("Set up Vega now?", True)
        else:
            say(f"No terminal to ask in. Run `python3 -m vega` in a terminal (or set {YES_ENV}=1), "
                "or do it by hand:")
            say()
            for line in self.manual_steps(python, venv, spec):
                say(line)
            return 1
        if not consent:
            say("Nothing changed. To do it by hand:")
            for line in self.manual_steps(python, venv, spec):
                say(line)
            return 1
        create = [str(python.path), "-m", "venv"]
        if (venv / "pyvenv.cfg").exists():
            create.append("--clear")  # its Python is gone or too old (e.g. Homebrew upgraded it)
        if self.run_command(create + [str(venv)]):
            say(f"Could not create the environment at {venv}.")
            return 1
        if not (vbin / "python").exists():
            # `python -m venv` exited 0 without making one: stop before pip.
            say(f"The environment at {venv} has no Python; stopping.")
            return 1
        return self.install(venv, spec)

    def install(self, venv: Path, spec: list[str]) -> int:
        vbin = venv_bin(venv, self.system)
        command = [str(vbin / "python"), "-m", "pip", "install", "--disable-pip-version-check"]
        if self.run_command(command + spec) or not (vbin / "vega").exists():
            say(f"Installing vegacode into {venv} failed (the output above says why).")
            return 1
        return 0

    def get_python(self, no_venv: list[Python]) -> Python | None:
        for python in no_venv:
            say(f"{python.path} is Python {dotted(python.version)} but cannot create virtual "
                "environments; on Debian/Ubuntu: "
                f"sudo apt install python3.{python.version[1]}-venv")
        brew = find_brew(self.env) if self.system == "Darwin" else None
        uv = find_uv(self.env, self.home)
        # One offer, uv first (no sudo, nothing outside its own folder), then Homebrew.
        offer = ([uv, "python", "install", UV_PYTHON] if uv
                 else [brew, "install", BREW_FORMULA] if brew else None)
        if offer and self.interactive():
            tool = "uv" if uv else "Homebrew"
            say(f"No Python 3.11 or newer was found. {tool} can install one:")
            say()
            say("    " + " ".join(offer))
            say()
            if self.ask("Run it now?"):
                if self.run_command(offer) == 0:
                    python, _ = find_python(self.env, self.home, self.system, self.probe)
                    if python is not None:
                        return python
                    say(f"{tool} finished but no Python 3.11+ was found afterwards.")
                else:
                    say(f"`{' '.join(offer[1:])}` failed.")
        for line in no_python_help(self.system, brew, self.env):
            say(line)
        return None


def run(args: list[str]) -> int:
    return Bootstrap().main(args)
