"""Model profiles from models.toml: swap the base model behind Vega with one command."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import subprocess
import tomllib
from dataclasses import dataclass, field, replace
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import server

# Where the profiles live, first match wins:
#   1. VEGA_MODELS_FILE, if set;
#   2. models.toml at the repository root, in a checkout (the file the repo tracks);
#   3. VEGA_HOME/models.toml for an installed package (pip install vegacode). Until the user adds or
#      removes a profile that file does not exist and the defaults shipped inside the package
#      (vega/default_models.toml, the repo-root models.toml copied in at build time) are read;
#      the first edit writes them there, so an upgrade never overwrites the user's profiles.
CHECKOUT_PROFILES_FILE = server.PROJECT_ROOT / "models.toml"
USER_PROFILES_FILE = server.STATE_DIR / "models.toml"
PACKAGED_PROFILES = "default_models.toml"  # resource name inside the `vega` package


def _profiles_file() -> Path:
    override = os.getenv("VEGA_MODELS_FILE")
    if override:
        return Path(override)
    return CHECKOUT_PROFILES_FILE if CHECKOUT_PROFILES_FILE.is_file() else USER_PROFILES_FILE


PROFILES_FILE = _profiles_file()
_WIRING = re.compile(r"^[a-z0-9][a-z0-9._-]{0,47}$")  # an Ollama renderer / parser name
DEFAULT_MODEL = "vega"
_SYSTEM = (
    "You are Vega, a careful software engineering coding assistant. The caller supplies the full "
    "instructions and tools for each session."
)


@dataclass(frozen=True)
class Profile:
    name: str
    base: str
    description: str = ""
    num_ctx: int = 8192
    parameters: dict[str, Any] = field(default_factory=dict)
    features_on: list[str] = field(default_factory=list)  # opt-in features this model needs
    features_off: list[str] = field(default_factory=list)  # defaults measured worse with this model
    # extra fields sent with every request, e.g. {"reasoning_effort": "none"} to turn off a thinking
    # model's reasoning (Vega's own system message replaces the Modelfile's, so it cannot go there)
    request: dict[str, Any] = field(default_factory=dict)
    # Another local OpenAI-compatible server (LM Studio, llama.cpp's llama-server, mlx_lm.server)
    # instead of Ollama: Vega sends requests straight to it, with no Ollama build step.
    endpoint: str = ""
    api_model: str = ""  # the model name that server expects (default: base)
    allow_remote: bool = False  # endpoint may be a non-local address (local-first: off by default)
    # Ollama's built-in prompt renderer and reply parser (Modelfile RENDERER / PARSER, e.g. "lfm2"):
    # for a Hugging Face GGUF, whose hf.co template has no tool format, so tool calls need the same
    # wiring as the library build of that family. Empty: whatever the base model has (a library model
    # built FROM keeps its own renderer and parser; checked on Ollama 0.34.3, 2026-11-04).
    renderer: str = ""
    parser: str = ""

    @property
    def is_endpoint(self) -> bool:
        return bool(self.endpoint)

    @property
    def served_name(self) -> str:
        """The model name sent in requests to an endpoint profile's server."""
        return self.api_model or self.base

    def modelfile(self) -> str:
        lines = [f"FROM {self.base}"]
        lines += [f"RENDERER {self.renderer}"] if self.renderer else []
        lines += [f"PARSER {self.parser}"] if self.parser else []
        lines += ["", f"PARAMETER num_ctx {self.num_ctx}"]
        for key, value in self.parameters.items():
            lines.append(f"PARAMETER {key} {value}")
        lines += ["", f'SYSTEM """{_SYSTEM}"""', ""]
        return "\n".join(lines)


def packaged_profiles_text() -> str | None:
    """The default profiles shipped in the installed package, or None (a source checkout has
    none: it reads the repo-root models.toml instead)."""
    try:
        return (resources.files("vega") / PACKAGED_PROFILES).read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError, ModuleNotFoundError):
        return None


def _profiles_text() -> str:
    """models.toml's text; for the per-user file not yet written, the packaged defaults."""
    try:
        return PROFILES_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        if PROFILES_FILE == USER_PROFILES_FILE:
            packaged = packaged_profiles_text()
            if packaged is not None:
                return packaged
        raise


def load() -> tuple[str, dict[str, Profile]]:
    """(default profile name, profiles by name)."""
    try:
        data = tomllib.loads(_profiles_text())
    except FileNotFoundError as exc:
        raise ValueError(f"No model profiles file at {PROFILES_FILE}.") from exc
    profiles: dict[str, Profile] = {}
    listed = dict(data.get("profiles", {}))
    from . import extensions  # an extension's own profiles (extensions.model_profiles): the file's win

    for name, raw in extensions.model_profiles().items():
        listed.setdefault(name, raw)
    for name, raw in listed.items():
        if not isinstance(raw, dict) or not isinstance(raw.get("base"), str):
            raise ValueError(f"Profile {name!r} in {PROFILES_FILE.name} needs a `base` model.")
        profiles[name] = Profile(
            name=name, base=raw["base"], description=str(raw.get("description", "")),
            num_ctx=int(raw.get("num_ctx", 8192)), parameters=dict(raw.get("parameters", {})),
            features_on=list(raw.get("features_on", [])),
            features_off=list(raw.get("features_off", [])),
            request=dict(raw.get("request", {})),
            endpoint=str(raw.get("endpoint", "")).strip(),
            api_model=str(raw.get("api_model", "")).strip(),
            allow_remote=bool(raw.get("allow_remote", False)),
            renderer=str(raw.get("renderer", "")).strip(), parser=str(raw.get("parser", "")).strip(),
        )
        for key in ("renderer", "parser"):  # one Modelfile word each: never a second line
            value = getattr(profiles[name], key)
            if value and not _WIRING.match(value):
                raise ValueError(f"Profile {name!r} in {PROFILES_FILE.name}: {key} {value!r} is not an Ollama "
                                 f"{key} name (lowercase letters, digits, . _ -).")
    default = data.get("default")
    if default not in profiles:
        raise ValueError(f"`default` in {PROFILES_FILE.name} must name one of: {', '.join(profiles)}.")
    return default, profiles


def get(name: str) -> Profile:
    _, profiles = load()
    if name not in profiles:
        raise ValueError(f"Unknown model profile {name!r}. Known: {', '.join(profiles)}.")
    return profiles[name]


def ollama_name(profile: str) -> str:
    """Ollama model name a profile is built as for side-by-side use."""
    return f"vega-{profile}"


def _active_file() -> Path:
    return server.STATE_DIR / "active_model.json"


def active_profile() -> str | None:
    """Profile the default `vega` model was last built from, if recorded."""
    try:
        return json.loads(_active_file().read_text(encoding="utf-8")).get("profile")
    except (OSError, ValueError):
        return None


def build(profile: Profile, target: str) -> int:
    """Create (or refresh) Ollama model `target` from a profile; pulls the base model if needed."""
    server.ensure_running()
    executable = server.ollama_executable()
    if not executable:
        raise RuntimeError("Ollama is not installed. Run `vega setup` (or install it from "
                           "https://ollama.com/download).")
    modelfiles = server.STATE_DIR / "modelfiles"
    modelfiles.mkdir(parents=True, exist_ok=True)
    path = modelfiles / f"{target}.Modelfile"
    path.write_text(profile.modelfile(), encoding="utf-8")
    return subprocess.run([executable, "create", target, "-f", str(path)]).returncode


def use(name: str) -> int:
    """Rebuild the default `vega` model from a profile and remember which profile it is. An
    endpoint profile has nothing to build: it is only recorded as active (the `vega` name then
    routes to its server)."""
    profile = get(name)
    if profile.is_endpoint:
        check_endpoint(profile.endpoint, allow_remote=profile.allow_remote)
        code = 0
    else:
        code = build(profile, DEFAULT_MODEL)
    if code == 0:
        server.STATE_DIR.mkdir(parents=True, exist_ok=True)
        record = {"profile": name, "base": profile.base}
        if profile.is_endpoint:
            record["endpoint"] = profile.endpoint
        _active_file().write_text(json.dumps(record), "utf-8")
    return code


def resolve(value: str) -> str:
    """`--model` value → the model name Vega runs under: a profile name maps to vega-<profile>
    (an Ollama profile is built on demand; an endpoint profile needs no build, and requests under
    that name go to its server, see `connection`); anything else is used as an Ollama model name
    as-is."""
    _, profiles = load()
    if value not in profiles:
        return value
    target = ollama_name(value)
    if profiles[value].is_endpoint:
        check_endpoint(profiles[value].endpoint, allow_remote=profiles[value].allow_remote)
        return target
    if server.is_running() and not server.has_model(target):
        if build(profiles[value], target):
            raise RuntimeError(f"Could not build {target} from profile {value!r}.")
    return target


def latest_scores(runs_file: Path) -> dict[str, str]:
    """Latest `fixes`-suite score per Ollama model name, e.g. {"vega-coder-14b": "7/10"}."""
    scores: dict[str, str] = {}
    try:
        lines = runs_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return scores
    for line in lines:
        try:
            run = json.loads(line)
        except ValueError:
            continue
        if run.get("suite", "fixes") == "fixes" and run.get("model") and run.get("mode") != "ab":
            scores[run["model"]] = f"{run.get('passed')}/{run.get('total')}"
    return scores


def profile_for_model(model: str) -> Profile | None:
    """The profile behind an Ollama model name: `vega` is the active (or default) profile,
    `vega-<name>` is profile <name> and a window alias `vega-<name>-16k` is profile <name> with
    num_ctx 16384 (see window_alias); other model names have no profile."""
    try:
        default, profiles = load()
    except (ValueError, OSError):
        return None
    if model in {DEFAULT_MODEL, f"{DEFAULT_MODEL}:latest"}:
        return profiles.get(active_profile() or default)
    name = model.removesuffix(":latest")
    if name.startswith("vega-"):
        found = profiles.get(name[len("vega-"):])
        if found is not None:
            return found
        split = _split_window_alias(name)
        if split is not None and split[0].startswith("vega-"):
            base = profiles.get(split[0][len("vega-"):])
            if base is not None and not base.is_endpoint:
                return replace(base, num_ctx=split[1])
    return None


# --- Window aliases: the same weights served with another context window ---------------------------
# Vega never sends num_ctx with a request: Ollama's OpenAI-compatible endpoint (/v1, which Vega
# uses) takes the window from the model's Modelfile (`PARAMETER num_ctx`). A budget of 16K
# (VEGA_NUM_CTX=16384) against a model built with 8K let prompts reach 8,191 tokens and the server
# then silently cut them (context shift, system prompt and tools lost; server.log 2026-10-13..28).
# So a window other than the model's own is served by an alias built from the same profile with
# only num_ctx changed: `vega-granite-8b-16k` (`vega model build granite-8b --num-ctx 16384`, or
# `vega model fit-window` for the current model and VEGA_NUM_CTX). Same blob, no download.
# Not options.num_ctx per request: Ollama's native API takes it, but whether 0.34.3's /v1 layer
# honours it is unverified (checking needs a model call), and an alias's window can be read back
# from /api/show before a run starts (context.match_window), which a request field cannot.

_WINDOW_SUFFIX = re.compile(r"^(?P<base>.+)-(?:(?P<k>\d+)k|ctx(?P<n>\d+))$")


def window_suffix(num_ctx: int) -> str:
    """`-16k` for 16384; `-ctx12000` for a window that is not a whole number of K."""
    return f"-{num_ctx // 1024}k" if num_ctx % 1024 == 0 else f"-ctx{num_ctx}"


def _split_window_alias(name: str) -> tuple[str, int] | None:
    """(model name without the window suffix, window) for `vega-granite-8b-16k`, else None."""
    match = _WINDOW_SUFFIX.match(name)
    if not match:
        return None
    window = int(match["k"]) * 1024 if match["k"] else int(match["n"])
    return (match["base"], window) if window >= 512 else None


def window_profile_name(model: str) -> str | None:
    """The profile an Ollama model name is built from (`vega` → the active or default profile,
    `vega-<p>` and `vega-<p>-16k` → p), or None (not a profile's model, or an endpoint)."""
    try:
        default, profiles = load()
    except (ValueError, OSError):
        return None
    name = model.removesuffix(":latest")
    if name == DEFAULT_MODEL:
        chosen = active_profile() or default
    elif name.startswith("vega-") and name[len("vega-"):] in profiles:
        chosen = name[len("vega-"):]
    else:
        split = _split_window_alias(name)
        chosen = split[0][len("vega-"):] if split and split[0].startswith("vega-") else None
    profile = profiles.get(chosen) if chosen else None
    return chosen if profile is not None and not profile.is_endpoint else None


def window_alias(model: str, num_ctx: int) -> str | None:
    """The Ollama model that serves `model`'s weights with a num_ctx-token window:
    `vega-<profile><suffix>` (e.g. vega-granite-8b-16k), or None when `model` is not built from a
    profile (Vega cannot build an alias for an arbitrary Ollama model)."""
    profile = window_profile_name(model)
    if profile is None:
        return None
    _, profiles = load()
    if profiles[profile].num_ctx == num_ctx:
        return ollama_name(profile)
    return ollama_name(profile) + window_suffix(num_ctx)


def build_window_alias(profile_name: str, num_ctx: int) -> tuple[str, int]:
    """Build (or refresh) the alias of a profile with another window: (model name, exit code).
    The profile's own window builds plain vega-<profile>."""
    profile = get(profile_name)
    if profile.is_endpoint:
        raise ValueError(f"{profile_name!r} runs on {profile.endpoint}: set the window in that "
                         "server; Vega builds nothing for an endpoint profile.")
    if num_ctx < 512:
        raise ValueError("num_ctx must be at least 512.")
    target = ollama_name(profile_name)
    if num_ctx != profile.num_ctx:
        target += window_suffix(num_ctx)
    return target, build(replace(profile, num_ctx=num_ctx), target)


_APPLIED: str | None = None  # what apply_feature_defaults last set (vs a user's own setting)
_APPLIED_OFF: str | None = None  # the same for the profile's switched-off defaults
_USER_OFF: str | None = None  # the user's own VEGA_FEATURES_OFF that the defaults are added to


def apply_feature_defaults(model: str) -> list[str]:
    """Apply the profile's feature defaults: opt-ins it turns on (e.g. constrained decoding for the
    7B) and defaults it turns off (e.g. the plan tool for Granite), unless the user set
    VEGA_FEATURES_ON / VEGA_FEATURES_OFF themselves. Returns what was applied ("-name" = off)."""
    import os

    from . import features

    global _APPLIED, _APPLIED_OFF, _USER_OFF
    profile = profile_for_model(model)
    applied: list[str] = []
    current = os.environ.get(features.ENV_ON)
    if current is None or current == _APPLIED:  # else the user chose; never override
        names = [name for name in (profile.features_on if profile else []) if name in features.OPT_IN]
        if names:
            os.environ[features.ENV_ON] = _APPLIED = ",".join(names)
        elif current is not None:  # a previous profile's defaults no longer apply
            os.environ.pop(features.ENV_ON, None)
            _APPLIED = None
        applied += names
    # The profile's switched-off defaults are ADDED to the user's own VEGA_FEATURES_OFF, never
    # replaced by it (2026-11-04: an A/B arm with VEGA_FEATURES_OFF=x on Granite also turned `plan`
    # back on, so every core inverted A/B changed two things). VEGA_PROFILE_FEATURES_OFF=0 or an
    # explicitly empty VEGA_FEATURES_OFF="" drops them.
    current_off = os.environ.get(features.ENV)
    if current_off is not None and current_off != _APPLIED_OFF:
        _USER_OFF = current_off  # the user set it since the last apply
    elif current_off is None:
        _USER_OFF = None
    user = [n for n in (_USER_OFF or "").split(",") if n.strip()] if _USER_OFF is not None else []
    use_profile = (_USER_OFF != ""  # an explicitly empty list is the user's "nothing off"
                   and os.environ.get("VEGA_PROFILE_FEATURES_OFF", "1").strip().lower() not in ("0", "off", "no"))
    off = [name for name in (profile.features_off if profile and use_profile else [])
           if name in features.FEATURES and name not in user]
    if user or off:
        os.environ[features.ENV] = _APPLIED_OFF = ",".join(user + off)
    elif _USER_OFF is not None:  # the user's own (empty) choice stays as set
        os.environ[features.ENV] = _APPLIED_OFF = _USER_OFF
    elif current_off is not None:
        os.environ.pop(features.ENV, None)
        _APPLIED_OFF = None
    applied += [f"-{name}" for name in off]
    return applied


def request_options(model: str) -> dict[str, Any]:
    """Extra request fields the model's profile asks for ({} without a profile)."""
    profile = profile_for_model(model)
    return dict(profile.request) if profile else {}


# --- Endpoints: other local OpenAI-compatible servers ---------------------------------------------

def endpoint_profile(value: str | None = None) -> Profile | None:
    """The endpoint profile behind a model name (`vega`, `vega-<profile>`) or a profile name, or
    None when it runs on Ollama (or is not a profile at all)."""
    value = value or server.model_name()
    try:
        _, profiles = load()
    except (ValueError, OSError):
        return None
    profile = profiles.get(value) or profile_for_model(value)
    return profile if profile is not None and profile.is_endpoint else None


def uses_ollama(value: str | None = None) -> bool:
    """True when the model runs on Ollama (so Ollama-only steps such as starting the server,
    building, unloading and /api/show apply)."""
    return endpoint_profile(value) is None


@dataclass(frozen=True)
class Connection:
    base_url: str  # OpenAI-compatible base URL requests go to
    model: str  # the model name sent in requests
    profile: str | None = None  # the endpoint profile, when not Ollama

    @property
    def is_endpoint(self) -> bool:
        return self.profile is not None


def connection(model: str | None = None) -> Connection:
    """Where requests for `model` go: an endpoint profile's own server and model name, otherwise
    VEGA_BASE_URL (Ollama by default) with the name unchanged."""
    model = model or server.model_name()
    profile = endpoint_profile(model)
    if profile is None:
        return Connection(server.base_url(), model)
    check_endpoint(profile.endpoint, allow_remote=profile.allow_remote)
    return Connection(profile.endpoint, profile.served_name, profile.name)


def is_local_host(host: str) -> bool:
    """Loopback or private-network address (or `localhost`): the machine or the local network."""
    host = host.strip("[]").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False  # a hostname: cannot tell where it resolves without a lookup
    return address.is_loopback or address.is_private or address.is_link_local


def check_endpoint(url: str, *, allow_remote: bool = False) -> str:
    """Validate an endpoint URL (http(s), a host, local unless allowed); returns it normalised."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Endpoint {url!r} must be an http(s) URL such as "
                         "http://127.0.0.1:1234/v1.")
    if not allow_remote and not is_local_host(parsed.hostname):
        raise ValueError(
            f"Endpoint {url!r} is not a loopback or private-network address. Vega is local-first; "
            "add the profile with --allow-remote to send code to another machine anyway.")
    return url.strip().rstrip("/")


# --- Editing models.toml --------------------------------------------------------------------------

_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
_HEADER = re.compile(r"^\s*\[\[?\s*([^\]]+?)\s*\]\]?\s*(#.*)?$")


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)  # a JSON string is a valid TOML basic string
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{key} = {_toml_value(item)}" for key, item in value.items()) + " }"
    raise ValueError(f"Cannot write {value!r} to models.toml.")


def parse_request_option(text: str) -> tuple[str, Any]:
    """`key=value` → (key, value): the value as a TOML literal (0.2, true, "x", [1, 2]) or, if it
    is not one, as a plain string."""
    key, sep, raw = text.partition("=")
    key, raw = key.strip(), raw.strip()
    if not sep or not re.fullmatch(r"[A-Za-z0-9_-]+", key):
        raise ValueError(f"--request takes key=value (e.g. reasoning_effort=none), got {text!r}.")
    try:
        return key, tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return key, raw


def _write_checked(text: str) -> None:
    """Write models.toml only if the result still loads (atomic replace, so never half-written)."""
    tomllib.loads(text)
    existed = PROFILES_FILE.is_file()  # False: the first edit of an installed package's defaults
    previous = _profiles_text()
    PROFILES_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp = PROFILES_FILE.with_suffix(".toml.tmp")
    temp.write_text(text, encoding="utf-8")
    try:
        os.replace(temp, PROFILES_FILE)
        try:
            load()
        except ValueError:
            if existed:
                PROFILES_FILE.write_text(previous, encoding="utf-8")
            else:
                PROFILES_FILE.unlink(missing_ok=True)
            raise
    finally:
        temp.unlink(missing_ok=True)


def add_profile(
    name: str, *, base: str, num_ctx: int | None = None, description: str = "",
    endpoint: str = "", api_model: str = "", request: dict[str, Any] | None = None,
    allow_remote: bool = False,
) -> Profile:
    """Append a [profiles.NAME] table to models.toml; the rest of the file (comments, order,
    other profiles) is left byte for byte as it was."""
    if not _NAME.match(name):
        raise ValueError(f"Profile name {name!r}: use lowercase letters, digits, - and _ "
                         "(it becomes the Ollama model vega-<name>), at most 48 characters.")
    if not base.strip():
        raise ValueError("--base needs the model name (Ollama model, or the endpoint's model).")
    _, profiles = load()
    if name in profiles:
        raise ValueError(f"Profile {name!r} already exists in {PROFILES_FILE.name}; remove it "
                         "first or choose another name.")
    if num_ctx is not None and num_ctx < 512:
        raise ValueError("--num-ctx must be at least 512 tokens.")
    if endpoint:
        endpoint = check_endpoint(endpoint, allow_remote=allow_remote)
    elif api_model:
        raise ValueError("--api-model only applies with --endpoint.")
    lines = [f"[profiles.{name}]", f"base = {_toml_value(base.strip())}"]
    if description:
        lines.append(f"description = {_toml_value(description)}")
    lines.append(f"num_ctx = {num_ctx or 8192}")
    if endpoint:
        lines.append(f"endpoint = {_toml_value(endpoint)}")
        if api_model:
            lines.append(f"api_model = {_toml_value(api_model)}")
        if allow_remote:
            lines.append("allow_remote = true")
    if request:
        lines.append(f"request = {_toml_value(request)}")
    text = _profiles_text()
    text = text.rstrip("\n") + "\n\n" + "\n".join(lines) + "\n"
    _write_checked(text)
    return get(name)


def remove_profile(name: str) -> Profile:
    """Delete [profiles.NAME] (and its sub-tables) from models.toml, keeping everything else.
    Refuses the default and the active profile. Never deletes model weights."""
    default, profiles = load()
    if name not in profiles:
        raise ValueError(f"Unknown model profile {name!r}. Known: {', '.join(profiles)}.")
    if name == default:
        raise ValueError(f"{name!r} is the default in {PROFILES_FILE.name}; set another default "
                         "first.")
    if name == active_profile():
        raise ValueError(f"{name!r} is the active profile behind `vega`; run "
                         "`vega model use <other>` first.")
    profile = profiles[name]
    lines = _profiles_text().splitlines(keepends=True)
    own = (f"profiles.{name}", f'profiles."{name}"')

    def mine(table: str) -> bool:
        return any(table == key or table.startswith(key + ".") for key in own)

    keep: list[str] = []
    index = 0
    while index < len(lines):
        match = _HEADER.match(lines[index])
        if not (match and mine(match.group(1).strip())):
            keep.append(lines[index])
            index += 1
            continue
        end = index + 1  # the table runs to the next header that is not one of its sub-tables
        while end < len(lines):
            header = _HEADER.match(lines[end])
            if header and not mine(header.group(1).strip()):
                break
            end += 1
        start_next = end  # comment lines right above the next table belong to it
        while start_next > index + 1 and lines[start_next - 1].lstrip().startswith("#"):
            start_next -= 1
        keep.extend(lines[start_next:end])
        index = end
    _write_checked("".join(keep).rstrip("\n") + "\n")
    return profile
