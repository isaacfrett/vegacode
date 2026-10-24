"""The `vega` / `vegacode` commands and `python -m vega` start here.

On Python 3.11+ this hands straight to vega.cli (nothing else is imported). On an older Python
(macOS's own python3 is 3.9) pip installs only this package, without its dependencies, and
vega._bootstrap sets Vega up under a newer Python instead.

Python 3.9 syntax and the standard library only: this module runs on interpreters Vega does not
support.
"""

import sys

MIN_PYTHON = (3, 11)


def main():
    if sys.version_info >= MIN_PYTHON:
        from vega.cli import app

        return app()
    from vega._bootstrap import run

    return run(sys.argv[1:])
