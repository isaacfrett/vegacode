"""Local model downloads: what is installed, whether it fits this machine, pulling and removing
models safely, the curated catalogue (model_catalog.toml), recommendations and new-model tracking.

Profiles (models.toml, models.py) say how Vega runs a model; this module manages the weights on
disk through Ollama's HTTP API (/api/tags, /api/pull, /api/delete). Rules:

- Fit: a model needs `size x 1.25 + 2 GB` of RAM (weights, an 8K context cache, runtime). It
  *fits* when that is at most 60% of total RAM, is *tight* up to 85%, and *won't fit* above.
- Disk: a pull is refused when it would leave less than 12 GB free on the disk Ollama stores
  models on (VEGA_GUARD_MIN_DISK_GB changes the floor, as for the resource guard).
- Removal: never the active or default profile (or the weights they are built from); only after
  confirmation. Ollama deletes a model's blobs only when no other model still references them
  (server/routes.go DeleteHandler -> manifest.RemoveNamed -> RemoveUnreferencedBlobs, which skips
  every digest a remaining manifest retains), so a shared weights file stays. The space shown
  before removal is computed the same way from Ollama's manifests on disk when they are readable.
- Size rule: models under 10B parameters only (catalogue entries, and any tag naming its size).
- Network: only `pull` and `watch --online` go online; everything else is local.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import models, server

CATALOG_FILE = Path(os.getenv("VEGA_MODEL_CATALOG", Path(__file__).with_name("model_catalog.toml")))
GB = 1e9  # sizes are shown in GB (10^9 bytes), as Ollama and the library show them
MAX_PARAMS_B = 10.0
GB_PER_BILLION = 0.62  # a 4-bit (Q4_K_M) download per billion parameters: 8.8B -> 5.35 GB measured
FIT_RATIO, TIGHT_RATIO = 0.60, 0.85
WATCH_FAMILIES = ("granite", "qwen", "llama", "gemma", "phi", "mistral", "deepseek-coder")
COMPARE_SUITES = ("humaneval40", "mbpp40", "goals", "fixes")
SUITE_LABELS = {"fixes": "fix", "goals": "goal", "humaneval40": "HE+", "mbpp40": "MBPP+"}
LIBRARY = "https://ollama.com"
REGISTRY = "https://registry.ollama.ai"
SHARED_NAMED_BYTES = 100_000_000
RUNS_FILE: Path | None = None  # default: <project>/evals/runs.jsonl
# Measured scores live apart from the shipped catalogue, in an optional local scores file: the
# packaged catalogue lists models without results (since 2026-10-21).
SCORES_FILE: Path | None = None  # default: <project>/evals/model_scores.toml


def min_free_gb() -> float:
    try:
        return float(os.environ.get("VEGA_GUARD_MIN_DISK_GB", "12"))
    except ValueError:
        return 12.0


def ram_needed_gb(size_gb: float) -> float:
    return size_gb * 1.25 + 2.0


def fit(size_gb: float | None, total_ram_gb: float | None) -> str:
    """`fits`, `tight`, `won't fit`, or `?` when the size or the machine's RAM is unknown."""
    if not size_gb or not total_ram_gb:
        return "?"
    need = ram_needed_gb(size_gb)
    if need <= total_ram_gb * FIT_RATIO:
        return "fits"
    if need <= total_ram_gb * TIGHT_RATIO:
        return "tight"
    return "won't fit"


FIT_RULE = (f"needs size x 1.25 + 2 GB of RAM; fits at <= {FIT_RATIO:.0%} of total RAM, tight at "
            f"<= {TIGHT_RATIO:.0%}, otherwise won't fit")


def params_from_tag(tag: str) -> float | None:
    """Parameter count in billions from a tag such as `qwen3:8b` or `x:7b-instruct-q4_K_M`."""
    match = re.search(r"(?:^|[:\-_])(\d+(?:\.\d+)?)b(?:$|[\-_.])", tag.split("/")[-1].lower())
    return float(match.group(1)) if match else None


def parse_parameter_size(text: str) -> float | None:
    """Ollama's details.parameter_size (`8.8B`, `494M`) in billions."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([BMK])\s*", text or "", re.IGNORECASE)
    if not match:
        return None
    scale = {"B": 1.0, "M": 1e-3, "K": 1e-6}[match.group(2).upper()]
    return float(match.group(1)) * scale


def canonical(name: str) -> str:
    """Ollama's display name: `x` -> `x:latest`."""
    return name if ":" in name.split("/")[-1] else f"{name}:latest"


# --- Catalogue ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Score:
    passed: int
    total: int
    commit: str = ""
    experiment: str = ""

    def __str__(self) -> str:
        return f"{self.passed}/{self.total}"

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


@dataclass(frozen=True)
class CatalogEntry:
    name: str
    tag: str
    family: str
    params_b: float
    download_gb: float
    ram_gb: float
    licence: str
    size_source: str = "library"
    default: bool = False
    fallback: bool = False
    num_ctx: int = 8192
    temperature: float | None = None
    request: dict[str, Any] = field(default_factory=dict)
    features_on: list[str] = field(default_factory=list)
    measured_on: str = ""
    notes: str = ""
    scores: dict[str, Score] = field(default_factory=dict)

    @property
    def measured(self) -> bool:
        return bool(self.scores)


def scores_file() -> Path:
    return SCORES_FILE or server.PROJECT_ROOT / "evals" / "model_scores.toml"


def load_scores(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """{catalogue name: {measured_on, scores}} from the private scores file; {} when it is absent
    (the public release), so a missing file means no scores rather than an error."""
    path = path or scores_file()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        return dict(tomllib.loads(text).get("models", {}))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path.name}: {exc}") from exc


def load_catalog(path: Path | None = None, scores_path: Path | None = None
                 ) -> dict[str, CatalogEntry]:
    """The catalogue with measured scores merged in from the scores file where it exists. Scores
    for a name the catalogue no longer lists are ignored (a dropped model keeps its history)."""
    path = path or CATALOG_FILE
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    measured = load_scores(scores_path)
    entries: dict[str, CatalogEntry] = {}
    for name, raw in data.get("models", {}).items():
        raw = {**raw, **measured.get(name, {})}
        try:
            scores = {suite: Score(int(s["passed"]), int(s["total"]), str(s.get("commit", "")),
                                   str(s.get("experiment", "")))
                      for suite, s in raw.get("scores", {}).items()}
            entry = CatalogEntry(
                name=name, tag=str(raw["tag"]), family=str(raw["family"]),
                params_b=float(raw["params_b"]), download_gb=float(raw["download_gb"]),
                ram_gb=float(raw["ram_gb"]), licence=str(raw["licence"]),
                size_source=str(raw.get("size_source", "library")),
                default=bool(raw.get("default", False)), fallback=bool(raw.get("fallback", False)),
                num_ctx=int(raw.get("num_ctx", 8192)),
                temperature=float(raw["temperature"]) if "temperature" in raw else None,
                request=dict(raw.get("request", {})), features_on=list(raw.get("features_on", [])),
                measured_on=str(raw.get("measured_on", "")), notes=str(raw.get("notes", "")),
                scores=scores,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Catalogue entry {name!r} in {path.name}: {exc}") from exc
        if entry.params_b >= MAX_PARAMS_B:
            raise ValueError(f"Catalogue entry {name!r} has {entry.params_b}B parameters; Vega's "
                             f"catalogue lists models under {MAX_PARAMS_B:.0f}B only.")
        if entry.scores and not entry.measured_on:
            raise ValueError(f"Catalogue entry {name!r} has scores but no measured_on date.")
        entries[name] = entry
    defaults = [e.name for e in entries.values() if e.default]
    if len(defaults) != 1:
        raise ValueError(f"{path.name} must mark exactly one model `default = true` "
                         f"(found {len(defaults)}).")
    return entries


def shipped_default(catalog: dict[str, CatalogEntry]) -> CatalogEntry:
    return next(entry for entry in catalog.values() if entry.default)


def catalog_by_tag(catalog: dict[str, CatalogEntry]) -> dict[str, CatalogEntry]:
    return {canonical(entry.tag): entry for entry in catalog.values()}


# --- Ollama's local API ---------------------------------------------------------------------------

class OllamaError(RuntimeError):
    pass


def _request(method: str, path: str, payload: dict | None = None, timeout: float = 10.0):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(server._ollama_root() + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=timeout)


def installed() -> list[dict[str, Any]] | None:
    """Ollama's /api/tags `models` (name, size, digest, details), or None if unreachable."""
    try:
        with _request("GET", "/api/tags", timeout=3) as response:
            data = json.loads(response.read().decode())
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return [item for item in data.get("models", []) if isinstance(item, dict)]


def pull_stream(name: str) -> Iterator[dict[str, Any]]:
    """Stream /api/pull progress objects (`status`, `digest`, `total`, `completed`). Raises
    OllamaError when Ollama reports an error."""
    try:
        response = _request("POST", "/api/pull", {"model": name, "stream": True}, timeout=600)
    except urllib.error.HTTPError as exc:
        raise OllamaError(_error_text(exc)) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise OllamaError(f"Ollama is not reachable at {server._ollama_root()}: {exc}") from exc
    with response:
        for line in response:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("error"):
                raise OllamaError(str(event["error"]))
            yield event


def delete(name: str) -> None:
    try:
        with _request("DELETE", "/api/delete", {"model": name}, timeout=60):
            pass
    except urllib.error.HTTPError as exc:
        raise OllamaError(_error_text(exc)) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise OllamaError(f"Ollama is not reachable at {server._ollama_root()}: {exc}") from exc


def _error_text(exc: urllib.error.HTTPError) -> str:
    try:
        return str(json.loads(exc.read().decode()).get("error") or exc)
    except (ValueError, OSError):
        return str(exc)


# --- Disk: where Ollama keeps models, free space and shared blobs ---------------------------------

def models_dir() -> Path:
    """Ollama's model store: OLLAMA_MODELS, else ~/.ollama/models (macOS, Windows, a user
    install on Linux), else the Linux service's /usr/share/ollama/.ollama/models."""
    configured = os.environ.get("OLLAMA_MODELS")
    if configured:
        return Path(configured).expanduser()
    home = Path.home() / ".ollama" / "models"
    service = Path("/usr/share/ollama/.ollama/models")
    return service if not home.exists() and service.exists() else home


def free_disk_gb(path: Path | None = None) -> float:
    path = path or models_dir()
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free / GB


def manifest_blobs(root: Path | None = None) -> dict[str, dict[str, int]] | None:
    """{model name: {blob digest: bytes}} from Ollama's manifests on disk, or None when they cannot
    be read (another user's store, a remote server, or a manifest format this does not know)."""
    root = (root or models_dir()) / "manifests"
    if not root.is_dir():
        return None
    out: dict[str, dict[str, int]] = {}
    try:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            parts = path.relative_to(root).parts
            if len(parts) != 4:
                continue
            host, namespace, model, tag = parts
            prefix = "" if (host, namespace) == ("registry.ollama.ai", "library") else (
                f"{namespace}/" if host == "registry.ollama.ai" else f"{host}/{namespace}/")
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data.get("layers"), list):
                return None  # a manifest list or another format: do not guess
            blobs = {layer["digest"]: int(layer.get("size", 0)) for layer in data["layers"]}
            config = data.get("config") or {}
            if config.get("digest"):
                blobs[config["digest"]] = int(config.get("size", 0))
            out[f"{prefix}{model}:{tag}"] = blobs
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return out


@dataclass
class Freed:
    gb: float
    exact: bool  # computed from manifests (else estimated from /api/tags)
    shared_with: list[str] = field(default_factory=list)  # remaining models sharing its weights

    def __str__(self) -> str:
        amount = f"{self.gb:.1f} GB" if self.gb >= 0.1 else (
            f"{self.gb * 1000:.0f} MB" if self.gb >= 0.001 else "under 1 MB")
        return amount + ("" if self.exact else " (estimate)")


def space_freed(names: list[str], tags: list[dict[str, Any]],
                blobs: dict[str, dict[str, int]] | None = None) -> Freed:
    """What removing `names` frees: the blobs no remaining model references (how Ollama deletes)."""
    removing = {canonical(n) for n in names}
    remaining = [t for t in tags if canonical(t.get("name", "")) not in removing]
    if blobs is not None and all(n in blobs for n in removing):
        kept = {d for name, layer in blobs.items() if name not in removing for d in layer}
        sizes: dict[str, int] = {}
        for name in removing:
            sizes.update(blobs[name])
        freed = sum(size for digest, size in sizes.items() if digest not in kept)
        # models that keep a large shared blob (weights) on disk; small shared blobs such as a
        # licence or template are not worth naming
        sharing = sorted(name for name, layer in blobs.items() if name not in removing
                         and sum(size for digest, size in layer.items()
                                 if digest in sizes) >= SHARED_NAMED_BYTES)
        return Freed(freed / GB, True, sharing)
    # Estimate: a model built FROM another (details.parent_model) shares its weights blob.
    def source(item: dict[str, Any]) -> str:
        return canonical((item.get("details") or {}).get("parent_model") or item.get("name", ""))

    by_name = {canonical(t.get("name", "")): t for t in tags}
    groups: dict[str, float] = {}
    for name in removing:
        item = by_name.get(name, {"name": name})
        groups.setdefault(source(item), float(item.get("size", 0)) / GB)
    still = {source(t) for t in remaining} | {canonical(t.get("name", "")) for t in remaining}
    sharing = sorted(canonical(t.get("name", "")) for t in remaining
                     if source(t) in groups or canonical(t.get("name", "")) in groups)
    total = sum(size for src, size in groups.items() if src not in still)
    return Freed(total, False, sharing)


# --- Inventory: profiles, installed models and the catalogue together -----------------------------

@dataclass
class ModelRow:
    name: str  # profile name, or the Ollama model name for a model without a profile
    tag: str  # the Ollama model holding the weights
    kind: str  # "profile", "ollama" (installed, no profile) or "catalogue" (not a profile)
    size_gb: float | None
    downloaded: bool
    built: bool = False  # vega-<profile> exists
    fit: str = "?"
    active: bool = False
    default: bool = False
    endpoint: str = ""
    scores: dict[str, str] = field(default_factory=dict)
    catalogued: bool = False
    shipped_default: bool = False
    params_b: float | None = None
    licence: str = ""
    note: str = ""
    size_estimated: bool = False  # from the parameter count (4-bit), not Ollama or the catalogue

    @property
    def over_limit(self) -> bool:
        return self.params_b is not None and self.params_b >= MAX_PARAMS_B


@dataclass
class Inventory:
    rows: list[ModelRow]
    total_ram_gb: float | None
    free_disk_gb: float | None
    ollama_up: bool
    default: str = ""
    active: str | None = None
    error: str = ""

    def row(self, name: str) -> ModelRow | None:
        return next((r for r in self.rows if r.name == name), None) or next(
            (r for r in self.rows if canonical(r.tag) == canonical(name)), None)


def run_scores(runs_file: Path) -> dict[str, dict[str, str]]:
    """Latest score per explicitly named Ollama model and suite from runs.jsonl, including the
    arms of A/B runs (`vega-granite-8b` -> {"fixes": "7/10"}). Runs under the plain `vega` name
    are not attributed: which profile `vega` was then is not recorded in the run."""
    out: dict[str, dict[str, str]] = {}
    try:
        lines = runs_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            run = json.loads(line)
        except ValueError:
            continue
        suite = run.get("suite", "fixes")
        arms = list((run.get("variants") or {}).values()) if run.get("mode") == "ab" else [run]
        for arm in arms:
            model = str(arm.get("model") or "").removesuffix(":latest")
            if model and model != models.DEFAULT_MODEL and arm.get("total"):
                out.setdefault(model, {})[suite] = f"{arm.get('passed')}/{arm.get('total')}"
    return out


def inventory(*, tags: list[dict[str, Any]] | None = None, total_ram_gb: float | None = None,
              catalog: dict[str, CatalogEntry] | None = None,
              runs_file: Path | None = None, read_tags: bool = True) -> Inventory:
    """Every profile, every installed Ollama model without a profile and every catalogue model,
    with size, fit, markers and scores. Never raises for a missing server (rows say unknown)."""
    from . import resources

    if total_ram_gb is None:
        total_ram_gb = resources.total_memory_gb()
    error = ""
    try:
        catalog = catalog if catalog is not None else load_catalog()
    except (OSError, ValueError) as exc:
        catalog, error = {}, f"catalogue: {exc}"
    if tags is None and read_tags:
        tags = installed()
    up = tags is not None
    tags = tags or []
    by_name = {canonical(t.get("name", "")): t for t in tags}
    try:
        default, profiles = models.load()
    except (OSError, ValueError) as exc:
        default, profiles, error = "", {}, (error + "; " if error else "") + str(exc)
    active = models.active_profile()
    scores = run_scores(runs_file or RUNS_FILE or server.PROJECT_ROOT / "evals" / "runs.jsonl")
    by_tag = catalog_by_tag(catalog)
    try:
        free = free_disk_gb()
    except OSError:
        free = None

    def size_of(tag: str, entry: CatalogEntry | None) -> float | None:
        item = by_name.get(canonical(tag))
        if item and item.get("size"):
            return float(item["size"]) / GB
        return entry.download_gb if entry else None

    def score_map(name: str, entry: CatalogEntry | None) -> dict[str, str]:
        merged = dict(scores.get(models.ollama_name(name), {}))
        if entry:  # curated (summed seeds, named commits) wins over the latest single run
            merged.update({suite: str(score) for suite, score in entry.scores.items()})
        return merged

    rows: list[ModelRow] = []
    used: set[str] = set()
    for name, profile in profiles.items():
        same = catalog.get(name)  # the catalogue's own entry: its scores and settings apply
        same = same if same and canonical(same.tag) == canonical(profile.base) else None
        entry = same or by_tag.get(canonical(profile.base))  # same weights: size and licence
        size = None if profile.is_endpoint else size_of(profile.base, entry)
        params = entry.params_b if entry else params_from_tag(profile.base)
        detail = (by_name.get(canonical(profile.base)) or {}).get("details") or {}
        params = parse_parameter_size(detail.get("parameter_size", "")) or params
        estimated = size is None and params is not None and not profile.is_endpoint
        if estimated:
            size = params * GB_PER_BILLION
        rows.append(ModelRow(
            name=name, tag=profile.base, kind="profile", size_gb=size, size_estimated=estimated,
            downloaded=profile.is_endpoint or canonical(profile.base) in by_name,
            built=canonical(models.ollama_name(name)) in by_name,
            fit="?" if profile.is_endpoint else fit(size, total_ram_gb),
            active=name == (active or default), default=name == default,
            endpoint=profile.endpoint, scores=score_map(name, same), catalogued=entry is not None,
            shipped_default=bool(same and same.default), params_b=params,
            licence=entry.licence if entry else "", note=profile.description))
        used.add(canonical(profile.base))
    for name, entry in catalog.items():
        if name in profiles or canonical(entry.tag) in used:
            continue
        size = size_of(entry.tag, entry)
        rows.append(ModelRow(
            name=name, tag=entry.tag, kind="catalogue", size_gb=size,
            downloaded=canonical(entry.tag) in by_name, fit=fit(size, total_ram_gb),
            scores=score_map(name, entry), catalogued=True, shipped_default=entry.default,
            params_b=entry.params_b, licence=entry.licence, note=entry.notes))
        used.add(canonical(entry.tag))
    for key, item in by_name.items():
        bare = key.removesuffix(":latest")
        if key in used or bare == models.DEFAULT_MODEL or bare.startswith("vega-"):
            continue
        size = float(item.get("size", 0)) / GB or None
        detail = item.get("details") or {}
        rows.append(ModelRow(
            name=key, tag=key, kind="ollama", size_gb=size, downloaded=True,
            fit=fit(size, total_ram_gb), scores=dict(scores.get(bare, {})),
            params_b=parse_parameter_size(detail.get("parameter_size", "")) or params_from_tag(key),
            note="installed in Ollama; no Vega profile"))
    return Inventory(rows, total_ram_gb, free, up, default, active or default or None, error)


# --- Pull -----------------------------------------------------------------------------------------

@dataclass
class PullPlan:
    target: str  # what the user asked for
    tag: str  # the Ollama model to pull
    profile: str | None  # profile to build afterwards (added from the catalogue if new)
    size_gb: float | None
    size_source: str  # "registry", "catalogue", "installed" or "unknown"
    free_gb: float
    params_b: float | None
    already: bool  # the weights are already downloaded
    new_profile: CatalogEntry | None = None  # catalogue entry to add as a profile first
    refusal: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def free_after_gb(self) -> float:
        return self.free_gb - (0.0 if self.already else (self.size_gb or 0.0))


def registry_size_gb(tag: str, fetch: Callable[[str, dict[str, str]], bytes] | None = None
                     ) -> float | None:
    """Exact download size of an Ollama library model from its registry manifest (online; only
    called by `pull`, which downloads from the same place anyway). None when unknown."""
    name, _, version = canonical(tag).partition(":")
    path = name if "/" in name else f"library/{name}"
    url = f"{REGISTRY}/v2/{path}/manifests/{version}"
    headers = {"Accept": "application/vnd.docker.distribution.manifest.v2+json"}
    try:
        raw = (fetch or _fetch_bytes)(url, headers)
        data = json.loads(raw)
        layers = data.get("layers") or []
        total = sum(int(layer.get("size", 0)) for layer in layers)
        total += int((data.get("config") or {}).get("size", 0))
    except (OSError, ValueError, TypeError, urllib.error.URLError):
        return None
    return total / GB if total else None


def _fetch_bytes(url: str, headers: dict[str, str] | None = None) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "vega-model-store",
                                                   **(headers or {})})
    with urllib.request.urlopen(request, timeout=15) as response:
        return response.read()


def plan_pull(target: str, *, tags: list[dict[str, Any]] | None = None,
              catalog: dict[str, CatalogEntry] | None = None, free_gb: float | None = None,
              size_lookup: Callable[[str], float | None] | None = None) -> PullPlan:
    """Resolve a profile, catalogue name or Ollama model name to what will be downloaded, and
    whether the disk and size rules allow it (refusal is set when they do not)."""
    catalog = catalog if catalog is not None else load_catalog()
    tags = tags if tags is not None else (installed() or [])
    by_name = {canonical(t.get("name", "")): t for t in tags}
    _, profiles = models.load()
    entry: CatalogEntry | None = None
    new_profile = None
    if target in profiles:
        profile = profiles[target]
        if profile.is_endpoint:
            raise ValueError(f"{target!r} runs on {profile.endpoint}; download it in that server.")
        tag, name = profile.base, target
        entry = catalog_by_tag(catalog).get(canonical(tag))
    elif target in catalog:
        entry, name = catalog[target], target
        tag = entry.tag
        new_profile = entry
    else:
        tag, name = target, None
        entry = catalog_by_tag(catalog).get(canonical(tag))
        if entry is not None:
            match = next((p for p, prof in profiles.items()
                          if canonical(prof.base) == canonical(tag)), None)
            name = match or entry.name
            new_profile = None if match else entry
    params = entry.params_b if entry else params_from_tag(tag)
    already = canonical(tag) in by_name
    free = free_gb if free_gb is not None else free_disk_gb()
    if already:
        size, source = float(by_name[canonical(tag)].get("size", 0)) / GB, "installed"
    else:
        size = (size_lookup or registry_size_gb)(tag)
        source = "registry" if size else "unknown"
        if not size and entry:
            size, source = entry.download_gb, "catalogue"
    plan = PullPlan(target, tag, name, size, source, free, params, already, new_profile)
    if params is not None and params >= MAX_PARAMS_B:
        plan.refusal = (f"{tag} has {params:g}B parameters; Vega uses models under "
                        f"{MAX_PARAMS_B:.0f}B only.")
    elif not already and size is None:
        plan.refusal = (f"The download size of {tag} is unknown (the registry did not answer and it "
                        "is not in the catalogue), so the 12 GB free-disk rule cannot be checked. "
                        "Pass --size-gb N with the size from ollama.com.")
    elif not already and plan.free_after_gb < min_free_gb():
        plan.refusal = (f"Downloading {tag} ({size:.1f} GB) would leave {plan.free_after_gb:.1f} GB "
                        f"free; Vega keeps at least {min_free_gb():.0f} GB free. Free space first "
                        "(vega model rm <name>).")
    if params is None:
        plan.warnings.append(f"The parameter count of {tag} is unknown; Vega uses models under "
                             f"{MAX_PARAMS_B:.0f}B only.")
    if name is None:
        plan.warnings.append(f"{tag} is not a profile; after the download add one with "
                             f"`vega model add NAME --base {tag}`.")
    return plan


@dataclass
class Progress:
    status: str
    completed: int
    total: int


def pull(plan: PullPlan, on_progress: Callable[[Progress], None] | None = None) -> None:
    """Download plan.tag through Ollama (skipped when already downloaded), reporting bytes summed
    over all layers."""
    if plan.already:
        return
    totals: dict[str, int] = {}
    done: dict[str, int] = {}
    for event in pull_stream(plan.tag):
        digest = event.get("digest")
        if digest:
            totals[digest] = int(event.get("total") or totals.get(digest, 0))
            done[digest] = int(event.get("completed") or done.get(digest, 0))
        if on_progress:
            on_progress(Progress(str(event.get("status", "")), sum(done.values()),
                                 sum(totals.values())))
        if event.get("status") == "success":
            return
    raise OllamaError(f"The download of {plan.tag} ended before Ollama reported success.")


def finish_pull(plan: PullPlan) -> str | None:
    """After the download: add the catalogue profile if new, then build vega-<profile>. Returns
    the built Ollama model name (None when the target has no profile)."""
    if plan.profile is None:
        return None
    if plan.new_profile is not None:
        add_catalog_profile(plan.new_profile)
    profile = models.get(plan.profile)
    target = models.ollama_name(plan.profile)
    if models.build(profile, target):
        raise OllamaError(f"Building {target} from {profile.base} failed.")
    return target


def add_catalog_profile(entry: CatalogEntry) -> models.Profile:
    _, profiles = models.load()
    if entry.name in profiles:
        return profiles[entry.name]
    profile = models.add_profile(entry.name, base=entry.tag, num_ctx=entry.num_ctx,
                                 description=f"From the catalogue: {entry.notes}".strip(),
                                 request=entry.request or None)
    extra = []
    if entry.features_on:
        extra.append(f"features_on = {models._toml_value(entry.features_on)}")
    if entry.temperature is not None:
        extra.append(f"[profiles.{entry.name}.parameters]\ntemperature = {entry.temperature!r}")
    if extra:
        text = models._profiles_text().rstrip("\n") + "\n"
        models._write_checked(text + "\n".join(extra) + "\n")
        profile = models.get(entry.name)
    return profile


# --- Remove ---------------------------------------------------------------------------------------

@dataclass
class RemovePlan:
    target: str
    names: list[str]  # Ollama models to delete, builds first
    freed: Freed | None
    kept: list[str] = field(default_factory=list)  # related models left in place, with reasons
    affects: list[str] = field(default_factory=list)  # other profiles that will need a pull again
    refusal: str = ""


def plan_remove(target: str, *, tags: list[dict[str, Any]] | None = None,
                blobs: dict[str, dict[str, int]] | None = None, read_blobs: bool = True
                ) -> RemovePlan:
    """What `vega model rm target` would delete and free. A profile removes vega-<profile> and its
    base model unless another remaining model is built from that base. Refuses the active and the
    default profile, the `vega` model, and the weights they are built from."""
    tags = tags if tags is not None else installed()
    if tags is None:
        return RemovePlan(target, [], None, refusal="Ollama is not running; start it with "
                          "`vega server start` to remove models.")
    default, profiles = models.load()
    active = models.active_profile() or default
    protected = {default, active}
    by_name = {canonical(t.get("name", "")): t for t in tags}
    protected_names = {canonical(models.DEFAULT_MODEL)}
    protected_bases = set()
    for name in protected:
        if name in profiles:
            protected_names.add(canonical(models.ollama_name(name)))
            protected_bases.add(canonical(profiles[name].base))
    if target in profiles:
        if target in protected:
            role = "active" if target == active else "default"
            other = "`vega model use <other>`" if target == active else \
                f"another `default` in {models.PROFILES_FILE.name}"
            return RemovePlan(target, [], None, refusal=f"{target!r} is the {role} model; "
                              f"switch first ({other}).")
        profile = profiles[target]
        if profile.is_endpoint:
            return RemovePlan(target, [], None, refusal=f"{target!r} runs on {profile.endpoint}; "
                              "its weights live in that server, not Ollama.")
        candidates = [canonical(models.ollama_name(target)), canonical(profile.base)]
    else:
        candidates = [canonical(target)]
    names = [n for n in candidates if n in by_name]
    if not names:
        return RemovePlan(target, [], None, refusal=f"Nothing to remove: {target} is not "
                          "downloaded.")
    bad = [n for n in names if n in protected_names or n in protected_bases]
    if bad and target not in profiles:
        return RemovePlan(target, [], None, refusal=f"{bad[0]} is the active or default model "
                          "(or the weights it is built from); switch first with "
                          "`vega model use <other>`.")
    kept = []
    remaining = [n for n in by_name if n not in names]
    for name in list(names):
        if name in protected_bases:
            names.remove(name)
            kept.append(f"{name} (the active or default model is built from it)")
            continue
        users = [r for r in remaining if canonical(
            (by_name[r].get("details") or {}).get("parent_model") or "") == name]
        users += [canonical(models.ollama_name(p)) for p, prof in profiles.items()
                  if p != target and canonical(prof.base) == name
                  and canonical(models.ollama_name(p)) in remaining and
                  canonical(models.ollama_name(p)) not in users]
        if users and name != candidates[0]:
            names.remove(name)
            kept.append(f"{name} (still used by {', '.join(sorted(set(users)))})")
    if not names:
        return RemovePlan(target, [], None, kept, refusal="Nothing to remove: " + "; ".join(kept))
    if blobs is None and read_blobs:
        blobs = manifest_blobs()
    affects = [p for p, prof in profiles.items() if p != target and not prof.is_endpoint
               and canonical(prof.base) in names]
    return RemovePlan(target, names, space_freed(names, tags, blobs), kept, affects=affects)


def remove(plan: RemovePlan) -> list[str]:
    if plan.refusal:
        raise ValueError(plan.refusal)
    removed = []
    for name in plan.names:
        delete(name)
        removed.append(name)
    return removed


# --- Recommend ------------------------------------------------------------------------------------

@dataclass
class Recommendation:
    entry: CatalogEntry
    fit: str
    installed: bool
    reason: str
    others: list[tuple[CatalogEntry, str]] = field(default_factory=list)


def composite(entry: CatalogEntry) -> float | None:
    """Mean pass rate over the comparison suites Vega measured (None when unmeasured)."""
    rates = [entry.scores[s].rate for s in COMPARE_SUITES if s in entry.scores]
    return sum(rates) / len(rates) if rates else None


def recommend(total_ram_gb: float | None, *, tags: list[dict[str, Any]] | None = None,
              catalog: dict[str, CatalogEntry] | None = None) -> Recommendation | None:
    """Best catalogue model for this machine: measured models that fit first (by mean pass rate on
    fixes, goals, HumanEval+ and MBPP+, the shipped default winning ties), then tight ones, then
    unmeasured ones (the default first, then smallest). Installed models win exact ties. With no
    scores at all (the public release) that is the default when it fits."""
    catalog = catalog if catalog is not None else load_catalog()
    have = {canonical(t.get("name", "")) for t in (tags or [])}
    ranked = []
    for entry in catalog.values():
        verdict = fit(entry.download_gb, total_ram_gb)
        if verdict == "won't fit":
            continue
        score = composite(entry)
        key = (verdict != "fits", score is None, -(score or 0.0), not entry.default,
               canonical(entry.tag) not in have, entry.download_gb)
        ranked.append((key, entry, verdict))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    _, best, verdict = ranked[0]
    score = composite(best)
    ram = f"{total_ram_gb:.0f} GB" if total_ram_gb else "unknown RAM"
    how = "fits" if verdict == "fits" else "fits only tightly in"
    if score is None and not any(entry.measured for entry in catalog.values()):
        # nothing measured: say what the pick is, not that it is untested (the public release)
        default = shipped_default(catalog)
        if best is default:
            reason = f"{best.name} is Vega's default model and {how} {ram}."
        else:
            reason = (f"{best.name} {how} {ram}; Vega's default, {default.name}, needs about "
                      f"{default.ram_gb:.1f} GB.")
    elif score is None:
        reason = (f"No measured model fits {ram}; {best.name} is the smallest untested one that "
                  "does. Queue a comparison with `vega model queue` before relying on it.")
    else:
        parts = ", ".join(f"{SUITE_LABELS[s]} {best.scores[s]}" for s in COMPARE_SUITES
                          if s in best.scores)
        how = "fit" if verdict == "fits" else "fit only tightly in"
        reason = (f"Highest measured scores among models that {how} {ram}: {parts} "
                  f"(measured {best.measured_on}).")
    others = [(entry, v) for _, entry, v in ranked[1:4]]
    return Recommendation(best, verdict, canonical(best.tag) in have, reason, others)


# --- Watch: new models to try ---------------------------------------------------------------------

@dataclass
class Candidate:
    tag: str
    source: str  # "catalogue", "ollama" (installed, no profile) or "online"
    size_gb: float | None = None
    params_b: float | None = None
    name: str = ""  # catalogue name / suggested profile name
    note: str = ""
    new: bool = True  # not reported by an earlier online check


def _watch_file() -> Path:
    return server.STATE_DIR / "model_watch.json"


def suggested_profile_name(tag: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "-", tag.lower().replace(":latest", "")).strip("-")
    return name[:48] or "model"


def watch_offline(*, tags: list[dict[str, Any]] | None = None,
                  catalog: dict[str, CatalogEntry] | None = None) -> list[Candidate]:
    """Models that could be compared without any network: catalogue models that are not profiles
    yet, and models in Ollama's local list that no profile uses (excluding Vega's own builds)."""
    catalog = catalog if catalog is not None else load_catalog()
    tags = tags if tags is not None else (installed() or [])
    _, profiles = models.load()
    bases = {canonical(p.base) for p in profiles.values()}
    out: list[Candidate] = []
    for entry in catalog.values():
        if entry.name in profiles or canonical(entry.tag) in bases:
            continue
        out.append(Candidate(entry.tag, "catalogue", entry.download_gb, entry.params_b,
                             entry.name, entry.notes))
    known = bases | {canonical(e.tag) for e in catalog.values()}
    for item in tags:
        name = canonical(item.get("name", ""))
        bare = name.removesuffix(":latest")
        if name in known or bare == models.DEFAULT_MODEL or bare.startswith("vega-"):
            continue
        detail = item.get("details") or {}
        params = parse_parameter_size(detail.get("parameter_size", "")) or params_from_tag(name)
        if params is not None and params >= MAX_PARAMS_B:
            continue
        out.append(Candidate(name, "ollama", float(item.get("size", 0)) / GB or None, params,
                             suggested_profile_name(name), "installed in Ollama; no profile"))
    return out


_LIBRARY_LINK = re.compile(r'href="/library/([a-z0-9][a-z0-9._-]*)(?::([A-Za-z0-9._-]+))?"')


def watch_online(families: tuple[str, ...] = WATCH_FAMILIES, *,
                 fetch: Callable[[str], str] | None = None,
                 catalog: dict[str, CatalogEntry] | None = None,
                 remember: bool = True) -> tuple[list[Candidate], list[str]]:
    """Opt-in: ask ollama.com's library for each family's models and their tags, keeping tags that
    name a size under 10B and are not catalogued, profiles or installed. Returns (candidates,
    errors). Remembers what it saw so the next check marks only newer tags as new."""
    fetch = fetch or (lambda url: _fetch_bytes(url).decode("utf-8", "replace"))
    catalog = catalog if catalog is not None else load_catalog()
    _, profiles = models.load()
    known = {canonical(e.tag) for e in catalog.values()} | {canonical(p.base)
                                                             for p in profiles.values()}
    known |= {canonical(t.get("name", "")) for t in (installed() or [])}
    try:
        seen = set(json.loads(_watch_file().read_text(encoding="utf-8")).get("seen", []))
    except (OSError, ValueError):
        seen = set()
    found: dict[str, Candidate] = {}
    errors: list[str] = []
    for family in families:
        try:
            page = fetch(f"{LIBRARY}/search?q={family}")
        except (OSError, urllib.error.URLError, ValueError) as exc:
            errors.append(f"{family}: {exc}")
            continue
        names = sorted({m.group(1) for m in _LIBRARY_LINK.finditer(page)
                        if m.group(1).startswith(family)})
        for library_name in names:
            try:
                tags_page = fetch(f"{LIBRARY}/library/{library_name}/tags")
            except (OSError, urllib.error.URLError, ValueError) as exc:
                errors.append(f"{library_name}: {exc}")
                continue
            for match in _LIBRARY_LINK.finditer(tags_page):
                if match.group(1) != library_name or not match.group(2):
                    continue
                tag = f"{library_name}:{match.group(2)}"
                params = params_from_tag(tag)
                if params is None or params >= MAX_PARAMS_B or canonical(tag) in known:
                    continue
                found.setdefault(tag, Candidate(tag, "online", None, params,
                                                suggested_profile_name(tag), new=tag not in seen))
    if remember and found:
        server.STATE_DIR.mkdir(parents=True, exist_ok=True)
        _watch_file().write_text(json.dumps({
            "seen": sorted(seen | set(found)),
            "checked": datetime.now(UTC).isoformat(timespec="seconds"),
        }, indent=2) + "\n", encoding="utf-8")
    return sorted(found.values(), key=lambda c: (not c.new, c.tag)), errors


# --- Queue the standard comparison ----------------------------------------------------------------

def queue_comparison(target: str, *, experiments_file: Path | None = None,
                     catalog: dict[str, CatalogEntry] | None = None,
                     suites: tuple[str, ...] = COMPARE_SUITES) -> tuple[str, list[str], list[str]]:
    """Write the standard comparison against the current default (HumanEval+ 40, MBPP+ 40, goals,
    fixes) to evals/experiments.toml for `vega lab run`; runs nothing. A catalogue name or Ollama
    model without a profile gets one first. Returns (profile, ids written, ids already there)."""
    from . import lab

    catalog = catalog if catalog is not None else load_catalog()
    default, profiles = models.load()
    if target in profiles:
        profile = target
    elif target in catalog:
        profile = add_catalog_profile(catalog[target]).name
    else:
        entry = catalog_by_tag(catalog).get(canonical(target))
        match = next((p for p, prof in profiles.items()
                      if canonical(prof.base) == canonical(target)), None)
        if match:
            profile = match
        elif entry:
            profile = add_catalog_profile(entry).name
        else:
            params = params_from_tag(target)
            if params is not None and params >= MAX_PARAMS_B:
                raise ValueError(f"{target} has {params:g}B parameters; Vega compares models "
                                 f"under {MAX_PARAMS_B:.0f}B only.")
            profile = suggested_profile_name(target)
            models.add_profile(profile, base=target,
                               description="Queued by vega model watch; not evaluated yet.")
    if profile == (models.active_profile() or default):
        raise ValueError(f"{profile!r} is the current default; compare another model against it.")
    path = experiments_file or lab.EXPERIMENTS_FILE
    existing = {e.id for e in lab.load(path)} if path.exists() else set()
    base = models.get(profile).base
    today = datetime.now().strftime("%Y-%m-%d")
    written, skipped, blocks = [], [], []
    for suite in suites:
        experiment = f"{suite}-model-{profile}"
        if experiment in existing:
            skipped.append(experiment)
            continue
        hypothesis = (f"New-model check ({today}): {base} as profile {profile} (B) vs the current "
                      f"default (A), same harness and defaults")
        blocks.append(f'[[experiment]]\nid = "{experiment}"\nsuite = "{suite}"\n'
                      f"hypothesis = {json.dumps(hypothesis)}\nab_model = \"{profile}\"\nseed = 7\n")
        written.append(experiment)
    if blocks:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        addition = "\n".join(blocks)
        text = text.rstrip("\n") + ("\n\n" if text else "") + addition
        tomllib.loads(text)
        path.write_text(text, encoding="utf-8")
        lab.load(path)  # still valid (unique ids)
    return profile, written, skipped


__all__ = [
    "FIT_RULE",
    "Candidate",
    "CatalogEntry",
    "Inventory",
    "ModelRow",
    "PullPlan",
    "RemovePlan",
    "add_catalog_profile",
    "finish_pull",
    "fit",
    "inventory",
    "load_catalog",
    "plan_pull",
    "plan_remove",
    "pull",
    "queue_comparison",
    "recommend",
    "remove",
    "watch_offline",
    "watch_online",
]
