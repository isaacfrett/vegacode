"""The deliverable pipeline's script runner, copied into the sandboxed clone and run there as
`python .vega-deliverable/run.py <targets json> <script path> <options json>` from the project root.
Standard library only, and no vega import: it runs under the sandbox's interpreter, not Vega's.

It makes the deliverables' folders, runs the model's script as if it were `<name>` in the project
root, then prints `<marker>` and a JSON object {target: content} of the deliverables it wrote.

Options (2026-10-20):
- `values` (script_error_values): on an exception the traceback starts at the script (the runner's
  own frames dropped) with its source lines, followed by the values of the names on the failing
  line of the script's innermost frames (`values_frames`, each repr cut at `values_width`).
- `guard` (sandbox_read_guard): an audit hook refuses reads (open, listdir, scandir, glob, and
  subprocess arguments) of host paths outside the project: home folders, /etc, /var, /Volumes and
  the like. Python-level only and platform independent; the macOS sandbox profile enforces the
  same at the kernel level. `fixture_root`: the task maps a folder of the project to `/`
  ("`root/` stands for `/`"), so a refused absolute path says where its copy is.
"""

import json
import linecache
import os
import re
import sys
import traceback
from pathlib import Path

# host areas a deliverable script never reads (the interpreter's own paths are allowed below)
DENIED = ("/Users", "/home", "/root", "/etc", "/private/etc", "/var", "/private/var",
          "/private/tmp", "/tmp", "/Volumes", "/mnt", "/media", "/Library/Keychains")
# inside DENIED, still needed by the interpreter and the tools a script may run (time zones,
# xcode-select's choice for git)
ALLOWED_INSIDE = ("/etc/localtime", "/private/etc/localtime", "/var/db/timezone",
                  "/private/var/db/timezone", "/private/var/select", "/var/select",
                  "/usr/share/zoneinfo")
_WORDS = re.compile(r"(?<![.\w])([A-Za-z_]\w*)")


def _under(path, roots):
    return any(path == root or path.startswith(root.rstrip(os.sep) + os.sep) for root in roots)


def _windows_denied():
    system = os.environ.get("SystemDrive", "C:")
    return (os.path.normcase(system + "\\Users"), os.path.normcase(system + "\\ProgramData"))


def install_guard(cwd, fixture_root="", real_home=""):
    """Refuse reads of host paths outside `cwd` and the interpreter's own folders."""
    allowed = [os.path.realpath(cwd)]
    for entry in os.scandir(cwd):  # the clone's linked folders (.git, .venv, node_modules)
        if entry.is_symlink():
            allowed.append(os.path.realpath(entry.path))
    for root in (sys.prefix, sys.base_prefix, sys.exec_prefix, *sys.path):
        if root:
            allowed += [os.path.abspath(root), os.path.realpath(root)]
    allowed += [os.path.realpath(p) for p in ALLOWED_INSIDE] + list(ALLOWED_INSIDE)
    allowed += [os.path.realpath(os.environ.get("TMPDIR", cwd))]
    # never a root that would open a whole home folder (a python in ~/bin, a `/` on sys.path)
    homes = {os.path.realpath(home) for home in (real_home, os.path.expanduser("~")) if home}
    allowed = [root for root in dict.fromkeys(allowed)
               if root == os.path.realpath(cwd) or (
                   root not in ("/", os.sep, "/Users", "/home")
                   and not any(_under(home, [root]) for home in homes))]
    denied = _windows_denied() if os.name == "nt" else DENIED
    denied = tuple(dict.fromkeys([*denied, *(os.path.realpath(d) for d in denied)]))

    def refused(raw):
        if isinstance(raw, int) or raw is None:
            return None
        try:
            text = os.fsdecode(raw)
        except TypeError:
            return None
        full = os.path.realpath(os.path.join(cwd, os.path.expanduser(text)))
        check = os.path.normcase(full)
        if not _under(check, denied) or _under(check, [os.path.normcase(a) for a in allowed]):
            return None
        hint = ""
        if fixture_root and text.startswith("/"):
            hint = (" The task says `" + fixture_root + "/` stands for `/`: read "
                    + fixture_root + text + " instead.")
        return PermissionError(
            "Vega's sandbox: a deliverable script reads only the project's files, and " + text
            + " is outside the project (it would read this computer, not the task's inputs)."
            + hint)

    def hook(event, args):
        if event in ("open", "os.listdir", "os.scandir", "glob.glob", "os.chdir") and args:
            error = refused(args[0])
        elif event == "subprocess.Popen" and len(args) > 1:
            argv = args[1] if isinstance(args[1], (list, tuple)) else [args[1]]
            error = next((e for e in (refused(a) for a in argv[1:]
                                      if isinstance(a, (str, bytes)) and str(a)[:1] in "/~")
                          if e), None)
        else:
            return
        if error is not None:
            raise error

    sys.addaudithook(hook)


def _value(value, width):
    try:
        text = repr(value)
    except Exception:  # noqa: BLE001 - a broken __repr__ in the model's script
        return "<unprintable>"
    if len(text) <= width:
        return text
    size = f"{len(value)} items" if hasattr(value, "__len__") else "cut"
    return text[:width] + f"... ({type(value).__name__}, {size})"


def report(exc, path, frames, width):
    """Print the traceback from the script's first frame, then the failing lines' values."""
    tb = exc.__traceback__
    while tb is not None and tb.tb_frame.f_code.co_filename != path:
        tb = tb.tb_next
    traceback.print_exception(type(exc), exc, tb or exc.__traceback__)
    script_frames = []
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == path:
            script_frames.append((tb.tb_frame, tb.tb_lineno))
        tb = tb.tb_next
    shown = []
    for frame, number in script_frames[-frames:]:
        line = linecache.getline(path, number).strip()
        scope = dict(frame.f_globals)
        if frame.f_locals is not frame.f_globals:
            scope.update(frame.f_locals)
        values = []
        for name in dict.fromkeys(_WORDS.findall(line)):
            if name not in scope or name.startswith("__"):
                continue
            value = scope[name]
            if callable(value) or type(value).__name__ == "module":
                continue
            values.append(f"    {name} = {_value(value, width)}")
        where = "" if frame.f_code.co_name == "<module>" else f" (in {frame.f_code.co_name})"
        shown.append(f"  line {number}{where}: {line}" + ("\n" + "\n".join(values) if values
                                                          else ""))
    if shown:
        print("\nValues at the failing line(s) of your script:\n" + "\n".join(shown),
              file=sys.stderr)


def main():
    targets = json.loads(sys.argv[1])
    source = Path(sys.argv[2]).read_text(encoding="utf-8")
    options = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    for target in targets:  # the deliverables' folders exist before the script runs
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    path = os.path.abspath(options.get("name", "deliverable_script.py"))  # as if in the root
    namespace = {"__name__": "__main__", "__file__": path}
    code = compile(source, path, "exec")
    if options.get("guard"):
        install_guard(os.getcwd(), options.get("fixture_root") or "",
                      options.get("real_home") or "")
    if options.get("values"):
        linecache.cache[path] = (len(source), None, source.splitlines(True), path)
        try:
            exec(code, namespace)  # noqa: S102 - the model's script, in the sandbox
        except Exception as exc:  # noqa: BLE001 - reported to the model
            report(exc, path, int(options.get("values_frames", 3)),
                   int(options.get("values_width", 160)))
            sys.exit(1)
    else:
        exec(code, namespace)  # noqa: S102
    print(options.get("marker", ""))
    print(json.dumps({t: Path(t).read_text(encoding="utf-8") for t in targets
                      if Path(t).is_file()}))


if __name__ == "__main__":
    main()
