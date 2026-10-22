"""`python -m vega.loops serve`: the background daemon process started by `vega loops start`."""

from __future__ import annotations

import sys
from datetime import datetime

from .scheduler import Daemon
from .state import AlreadyRunning


def _log(text: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {text}", flush=True)


def main(argv: list[str]) -> int:
    if argv != ["serve"]:
        print("usage: python -m vega.loops serve", file=sys.stderr)
        return 2
    try:
        Daemon(log=_log).serve()
    except AlreadyRunning as exc:
        _log(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
