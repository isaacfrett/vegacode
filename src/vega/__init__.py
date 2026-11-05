"""Vega: a local-first coding agent."""


def version() -> str:
    """The installed vegacode version (`vega --version`), else the source checkout's
    pyproject.toml version, else "unknown"."""
    from importlib import metadata

    try:
        return metadata.version("vegacode")
    except metadata.PackageNotFoundError:
        pass
    try:  # a source checkout run without installing it
        import tomllib

        from .server import PROJECT_ROOT

        data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        return str(data["project"]["version"])
    except (OSError, ValueError, KeyError):
        return "unknown"
