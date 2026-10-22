"""Code index: symbol extraction, repo map, BM25 search and the two tools built on them."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from vega.index import build_index, extract_symbols, tokenize
from vega.tools import execute, tool_specs


def write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


PY_SOURCE = '''\
MAX_RETRIES = 3
helper_value = 1


class Cart(Base):
    """A cart."""

    def add(self, item: str, qty: int = 1) -> None:
        self.items.append((item, qty))

    async def checkout(self):
        return True


def calculate_total(prices: list[float]) -> float:
    return sum(prices)
'''


def test_python_symbols_include_methods_signatures_and_constants() -> None:
    symbols = {s.name: s for s in extract_symbols("shop.py", PY_SOURCE)}
    assert symbols["MAX_RETRIES"].kind == "constant"
    assert "helper_value" not in symbols
    assert symbols["Cart"].kind == "class" and symbols["Cart"].line == 5
    assert symbols["Cart"].signature == "class Cart(Base)"
    add = symbols["add"]
    assert add.kind == "method" and add.depth == 1 and add.line == 8
    assert add.signature == "def add(self, item: str, qty: int=1) -> None"
    assert symbols["checkout"].signature.startswith("async def checkout")
    total = symbols["calculate_total"]
    assert (total.kind, total.line, total.end_line) == ("function", 15, 16)


def test_python_syntax_error_falls_back_to_regex() -> None:
    source = "class Broken:\n    def ok(self):\n        pass\n\ndef bad(:\n    pass\n"
    symbols = extract_symbols("broken.py", source)
    assert [(s.name, s.kind, s.line) for s in symbols] == [
        ("Broken", "class", 1), ("ok", "function", 2), ("bad", "function", 5),
    ]


def test_js_and_ts_regex_symbols() -> None:
    source = """\
export class Store extends Base {
  constructor(opts) {
    this.opts = opts;
  }
  async loadItems(id: string): Promise<Item[]> {
    if (id) {
      return [];
    }
  }
}
export function renderPage(props) {
  return null;
}
export const useCart = async (id: string) => fetch(id);
interface CartItem { id: string }
export type Price = number;
"""
    found = {(s.name, s.kind) for s in extract_symbols("store.ts", source)}
    assert {
        ("Store", "class"), ("constructor", "method"), ("loadItems", "method"),
        ("renderPage", "function"), ("useCart", "function"), ("CartItem", "interface"),
        ("Price", "type"),
    } <= found
    assert not any(name in {"if", "return"} for name, _ in found)


def test_other_language_regexes() -> None:
    go = extract_symbols("main.go", "type Server struct {}\nfunc (s *Server) Start() error {\n}\n")
    assert [s.name for s in go] == ["Server", "Start"]
    rust = extract_symbols(
        "lib.rs", "pub struct Pool;\nimpl Pool {\n    pub async fn get(&self) {}\n}\n",
    )
    assert [s.name for s in rust] == ["Pool", "Pool", "get"]
    gd = extract_symbols(
        "player.gd", "class_name Player\nconst SPEED = 5\nfunc _ready():\n\tpass\n",
    )
    assert [s.name for s in gd] == ["Player", "SPEED", "_ready"]
    c = extract_symbols("x.c", "struct node {\nint v;\n};\nstatic int add(int a, int b)\n{\n"
                               "  return a + b;\n}\nint x = add(1, 2);\n")
    assert [s.name for s in c] == ["node", "add"]


def test_tokenize_splits_snake_and_camel_case() -> None:
    assert tokenize("calculate_total") == ["calculate_total", "calculate", "total"]
    assert tokenize("parseHTTPResponse") == ["parsehttpresponse", "parse", "http", "response"]
    assert tokenize("getUser_byID") == ["getuser_byid", "get", "user", "by", "id"]
    assert tokenize("plain words") == ["plain", "words"]


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_gitignore_is_respected(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    write(tmp_path, ".gitignore", "build/\nsecret.py\n")
    write(tmp_path, "app.py", "def main():\n    pass\n")
    write(tmp_path, "secret.py", "def hidden():\n    pass\n")
    write(tmp_path, "build/out.py", "def generated():\n    pass\n")
    (tmp_path / "blob.bin").write_bytes(b"\0\1\2binary")
    paths = {f.path for f in build_index(tmp_path).files}
    assert "app.py" in paths
    assert not {"secret.py", "build/out.py", "blob.bin"} & paths


def test_walk_fallback_skips_ignored_dirs_and_large_files(tmp_path: Path) -> None:
    write(tmp_path, "keep.py", "x = 1\n")
    write(tmp_path, "node_modules/pkg/index.js", "function f() {}\n")
    write(tmp_path, "big.txt", "a" * 1_100_000)
    paths = {f.path for f in build_index(tmp_path).files}
    assert paths == {"keep.py"}


def test_bm25_ranks_definition_above_mentions(tmp_path: Path) -> None:
    write(tmp_path, "billing.py", PY_SOURCE)
    for i in range(5):
        write(tmp_path, f"notes_{i}.py", f"def unrelated_{i}():\n    # the total is shown here\n"
                                         f"    print('report {i}')\n")
    index = build_index(tmp_path)
    for query in ("calculate_total", "total", "calculate total"):
        hits = index.search(query)
        assert hits[0].path == "billing.py", query
        assert hits[0].start_line == 15 and hits[0].end_line == 16
        assert hits[0].score > hits[1].score
    assert "calculate_total" in index.search("calculate_total")[0].snippet


def test_search_windows_non_python_and_limits(tmp_path: Path) -> None:
    body = "\n".join(f"line {i}" for i in range(100)) + "\nneedleToken here\n"
    write(tmp_path, "data.txt", body)
    index = build_index(tmp_path)
    hits = index.search("needle", limit=5)
    assert hits[0].path == "data.txt" and hits[0].start_line == 81 and hits[0].end_line == 101
    assert index.search("") == [] and index.search("absentword") == []


def test_repo_map_lists_symbols_and_truncates_keeping_every_file(tmp_path: Path) -> None:
    write(tmp_path, "shop.py", PY_SOURCE)
    many = "\n".join(f"def function_number_{i}(a, b, c):\n    return a\n" for i in range(60))
    for n in range(8):
        write(tmp_path, f"pkg/module_{n}.py", many)
    index = build_index(tmp_path)
    full = index.repo_map(max_chars=1_000_000)
    assert "shop.py\n  1: MAX_RETRIES\n  5: class Cart(Base)\n    8: def add(" in full
    small = index.repo_map(max_chars=1500)
    assert len(small) <= 1500
    for f in index.files:
        assert f.path in small
    assert "more symbols" in small
    assert index.repo_map(prefix="pkg/").splitlines()[0].startswith("pkg/")
    assert "shop.py" not in index.repo_map(prefix="pkg/")


def test_tools_via_execute(tmp_path: Path) -> None:
    names = {spec["function"]["name"] for spec in tool_specs()}
    assert {"repo_map", "search_code", "search_text"} <= names
    write(tmp_path, "src/billing.py", PY_SOURCE)
    write(tmp_path, "README.md", "Totals are computed elsewhere.\n")
    result = execute("search_code", {"query": "calculate total"}, tmp_path)
    first = result.splitlines()[0]
    assert first.startswith("src/billing.py:15-16 (")
    assert "def calculate_total" in result
    mapped = execute("repo_map", {}, tmp_path)
    assert "src/billing.py" in mapped and "README.md" in mapped
    only_src = execute("repo_map", {"path": "src"}, tmp_path)
    assert "src/billing.py" in only_src and "README.md" not in only_src
    assert "No code matched" in execute("search_code", {"query": "zzzqqq"}, tmp_path)
    escaped = execute("repo_map", {"path": "../"}, tmp_path)
    assert "must stay inside the project" in escaped
    assert "must stay inside the project" in execute("repo_map", {"path": "/etc"}, tmp_path)


def test_symlink_outside_project_is_not_indexed(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    outside = tmp_path / "outside"
    write(outside, "secret.py", "def leaked_secret():\n    pass\n")
    write(project, "app.py", "def main():\n    pass\n")
    (project / "link.py").symlink_to(outside / "secret.py")
    result = execute("search_code", {"query": "leaked_secret"}, project)
    assert result.startswith("No code matched")


@pytest.mark.parametrize("git", [True, False])
def test_vega_state_dir_is_hidden_from_search_and_listing(tmp_path: Path, git: bool) -> None:
    # Selfbuild 2026-10-21: after `remember`, search_text returned .vega/notes.md as its first hit
    # and the agent edited its own note instead of the code.
    from vega.tools import project_overview

    if git:
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    write(tmp_path, "app.py", "def approval_prompt():\n    return 'approval prompt'\n")
    saved = execute("remember", {"note": "The approval prompt lives in app.py (approval_prompt)."}, tmp_path)
    assert (tmp_path / ".vega" / "notes.md").is_file(), saved
    write(tmp_path, ".vega/sessions.md", "- approval prompt work\n")

    for name, args in [
        ("search_text", {"query": "approval prompt"}),
        ("search_code", {"query": "approval prompt"}),
        ("repo_map", {}),
        ("list_files", {}),
        ("find_files", {"query": "notes"}),
        ("find_files", {"query": ".vega"}),
    ]:
        assert ".vega/" not in execute(name, args, tmp_path), name
    assert "app.py" in execute("search_text", {"query": "approval prompt"}, tmp_path)
    assert "app.py" in execute("search_code", {"query": "approval prompt"}, tmp_path)
    assert ".vega" not in {f.path.split("/")[0] for f in build_index(tmp_path).files}
    files_section = project_overview(tmp_path).split("Git", 1)[0]  # git status itself stays as git says
    assert "app.py" in files_section and ".vega/" not in files_section

    # An explicit read still works: the notes feature and the user's own requests keep access.
    read = execute("read_file", {"path": ".vega/notes.md"}, tmp_path)
    assert "approval prompt lives in app.py" in read
