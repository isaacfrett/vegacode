"""Feature switches for isolating what helps or hurts the model, without deleting code.

Everything in FEATURES is on by default. Turn features off with VEGA_FEATURES_OFF=lint,plan (or
`vega eval --off lint,plan`); eval runs record which were off so results stay attributable.
Experimental features in OPT_IN are off by default; turn them on with VEGA_FEATURES_ON=constrained
(or compare them with `vega eval --ab-on constrained`).
"""

from __future__ import annotations

import os

FEATURES: dict[str, str] = {
    "lint": "ruff findings appended to edit results (parse checks always stay on)",
    "plan": "update_plan tool, its prompt line, plan injection and the unfinished-plan nudge",
    "remember": "remember tool, its prompt line and loading .vega/notes.md",
    # added 2026-10-15: familiarity between chat sessions. Deterministic (no model call)
    "manage_gitignore": "keep .vega/ out of git: a marked Vega block in .gitignore, added once per repo "
                        "(never touches other lines; not re-added after you remove it)",
    "project_context": "VEGA.md (drafted on a project's first chat session, generated block kept "
    "current), a digest of each chat session in .vega/sessions.md, and loading VEGA.md (else "
    "AGENTS.md/CLAUDE.md), the last 2 digests and notes under one token budget at session start",
    "compaction": "summarising older history when nearing the context limit",
    "result_cap": "per-tool-result size cap (20% of the context window)",
    "code_index": "repo_map and search_code tools and their prompt mention",
    "nudge_act": "send back a reply that used no tools",
    "nudge_malformed": "send back a tool call written as unparseable text",
    "nudge_parse": "send back when an edited file no longer parses",
    "nudge_claim": "send back a claimed change when no edit succeeded",
    "nudge_length": "send back a reply cut off by the length limit once (else stop cleanly)",
    "repeat_guard": "do not re-run a call identical to one that already failed; say so instead",
    # added 2026-10-21: a task re-sent one refused write over a hundred times
    "repeat_stop": "end the agent-loop turn after 3 guarded repeats of failed calls in a row",
    "unchanged_rounds": "a repair round whose saved writes leave every file byte-identical runs no "
    "checks: the writer is told so with the failing check once; a second such round in a row ends "
    "the loop (best round kept)",
    "kv_q8": "flash attention + q8_0 KV cache when Vega starts Ollama (needs a server restart)",
    "knowledge": "the search_knowledge tool and library summary, when the library holds anything",
    # on by default since 2026-10-14 after paired A/Bs
    "direct_pipeline": "self-contained write tasks (implement functions in 1-2 small files) go "
    "write → run docstring examples and tests → repair (≤2) instead of the agent loop",
    # a UX fix (2026-10-14), switchable so its effect on later prompts can be measured
    # on by default since 2026-10-15 after paired A/Bs
    "project_pipeline": "multi-file change tasks: plan files from the repo map → write each file "
    "as a code block → run the project's tests → repair (≤3 rounds) instead of the agent loop",
    "missing_symbol_hint": "a search for `def name`/`class Name` that nothing defines says the "
    "definition does not exist yet and should be written",
    # on by default since 2026-10-21 from an offline ranking check against search_code's BM25:
    # agents on larger projects often edited the wrong file; unmeasured on a model
    "issue_localize": "change tasks on the agent loop in projects with 25+ files: a short 'Likely "
    "places' block in the request (top 3 files holding the task's quoted strings, names, "
    "/commands and file names, rarer and defined names weighted up, with the functions around "
    "the matches); no model call",
    # on by default since 2026-10-15 after a paired A/B on code-understanding tasks; it also
    # answers much sooner
    "question_pipeline": "questions and read-only requests: find the relevant code (named files, "
    "then the search index) and answer in one model call; falls back to the agent loop when the "
    "excerpts do not hold the answer",
    "keep_docstrings": "pipelines put back a user's docstring that the model's version of a "
    "function, class or module left out",
    # on by default since 2026-10-15 (a vague build request was met with repeated searches).
    # Rules only, no model call; checked offline not to flag ordinary, well-specified tasks
    "clarify_first": "a new request with an open-ended thing to build and nothing pinning it down "
    "(no file, module, input or output) gets 2-4 numbered questions with options from the project "
    "instead of a guess; no tool runs",
    # on by default since 2026-10-15 (after compaction the model repeated the same searches it
    # had made before it)
    "turn_ledger": "this turn's read-only calls are kept as one-line outcomes in the compaction "
    "summary; an identical search/listing/check with no write since returns that line instead of "
    "re-running (read_file always runs), and after two such repeats a change-approach note",
    # added 2026-10-17: "compute a file from the inputs" tasks did poorly in the agent loop.
    # On by default (was opt-in), measured offline by route replay; unmeasured on a model
    "deliverable_pipeline": "tasks that ask for a JSON/Markdown/CSV/TXT/lock file worked out from "
    "the project's inputs: all inputs in one prompt; the file directly, or a Python script that "
    "computes it run in the sandbox; format check and ≤2 repairs; inputs never changed",
    "deliverable_wide": "deliverable routing also takes an existing file the task asks to update "
    "or regenerate (`Update requirements.lock`), and a task whose only code file is a helper "
    "script it lets you write and run (`scratch/check.py`); the writer sees binary inputs by name and "
    "size and may first ask to run the commands the task names (`Run: python tools/x.py ...`, "
    "≤2 rounds, in the sandbox)",
    # added 2026-10-21: on a large test suite, "`python -m pytest -q`: Check timed out after
    # 60s." was reported as a failing check; the model chased the timeout, not the bug
    "targeted_tests": "on a project with 30+ pytest files, acceptance runs the test files related "
    "to this turn's changes first (edited tests, tests named after or importing a changed module, "
    "≤12, run_check's timeout), then the whole suite with a longer timeout scaled to its measured "
    "time (≤600 s); a whole suite that does not finish is reported as slow, not failing, and a "
    "model's own whole-suite run_check timeout says so and what to run instead",
    # added 2026-10-18 as a config of write_loop. On by default, measured offline by route
    # replay; unmeasured on a model
    "nonpython_pipeline": "a task that changes or creates ONE non-Python file (Go, Java, Ruby, R, "
    "JS, YAML, SQL, ...) and names the command that checks it (`python check.py`, `node --test`, "
    "`go test ./...`): write the whole file, run that command in the sandbox after every write, "
    "≤3 repairs, the round with the fewest failing checks kept",
    # agent-loop answers finished with a task-named file missing, a named function still a stub,
    # a file outside "do not change any other file" modified, or acceptance failing
    "finish_gate": "agent loop: a final answer is sent back (≤2 per turn, saying exactly what is "
    "missing) while a file the task says to write/create does not exist, a function it names is "
    "still a NotImplementedError/`...`/pass stub, a file outside its \"change only X\"/\"do not "
    "change any other file\" scope was modified this turn, or the last acceptance run failed with "
    "no write since",
    # added 2026-10-15/16 from failures seen with Granite 4.1 8B: on by default, unmeasured
    # until an A/B with VEGA_FEATURES_OFF
    "plan_pending_ok": "update_plan accepts a plan whose open items are all pending and starts the "
    "first one (says so in the result); two or more in_progress items are still rejected",
    "data_preview": "read_file of a whole large data file (> 4 KB or > 60 lines) returns a "
    "structured preview (size, lines, CSV header and inferred column types, first 20 rows; JSON "
    "shape and first items) and how to read more; line ranges and code files are unchanged",
    "empty_final": "send back an empty final reply once, asking for the answer (what was done, "
    "what is left); a second empty reply ends the turn as before",
    # added 2026-10-16: the model rarely recovered after a failing check, often claimed success,
    # and often ended without running the tests after the last edit. On by default, unmeasured
    # until the A/B
    "verify_before_finish": "agent loop: before accepting the final answer of a turn that changed "
    "files, if the project has a test/check command and none ran after the last write, Vega runs "
    "it once in the sandbox and gives the result to the model (≤2 rounds per turn; never for "
    "questions or conversation)",
    "repair_feedback": "a failing check is condensed for the 8k window: failing test ids, their "
    "assertion/exception lines, the failing test line and the nearest line of the project's own "
    "code; pytest noise dropped; how to see one test's whole traceback (agent loop and the "
    "project pipeline's repair prompt)",
    "claim_check": "agent loop: a final answer that claims success ('tests pass', 'verified') "
    "while the last check after the last write failed, or none ran, is sent back once with "
    "that run's result; the model corrects the answer or keeps fixing",
    # failure diagnosis (2026-10-16), redesign A: many failures were the verification loop
    # (agent-loop attempts almost never had visible tests). On by default so it can be A/B'd off;
    # unmeasured until that A/B
    "acceptance": "Vega checks the work against the task itself: before the first write it reads "
    "the task (rules, no model call) for the names and parameters it requires, what they return, "
    "the input files and the commands it names; after each write a one-line result of the "
    "failing import/name/call/static checks goes to the model; at finish the full set (plus "
    "commands and visible tests) is the gate (≤3 feedback rounds, one per failing state); the "
    "project pipeline does not stop and the direct pipeline keeps repairing while it fails",
    # since 2026-10-16: the sandbox never touches the user's machine
    "acceptance_ignores_no_commands": "acceptance runs its checks in the sandbox even when the "
    "task says 'Do not run commands' (off: only static checks for such tasks)",
    # failure diagnosis (2026-10-16), redesign B: a large share of failures were the edit tools
    # (breaking the file and editing on, or rebuilding old_text from memory). On by default so
    # each can be A/B'd off; unmeasured on a model until that A/B
    "atomic_writes": "every write tool computes the whole new file and refuses it (file unchanged) "
    "when it does not parse (Python, JSON, TOML, INI; YAML with PyYAML) or adds undefined names "
    "(F821) or redefinitions (F811) the file did not have; the result gives line, column, the 3 "
    "lines and how to retry. A file already broken may be written if no new parse error is added",
    "edit_by_name": "a replace_definition tool (function, class or Class.method by name; code "
    "cleaned, re-indented, decorators and docstring kept, merged like merge_code) offered as the "
    "recommended edit for Python code in the tool descriptions and system-prompt rule 3",
    "fence_strip": "a code file's write (content/new_text) wrapped in ``` fences or starting with "
    "a line that only names the file is cleaned before it is checked, with a note in the result",
    "pin_editing_target": "context compaction keeps the current text of the definitions written "
    "this turn (from disk, newest first, ≤1,500 characters) in the summary",
    # failure diagnosis (2026-10-16), redesign C: some failures were the project pipeline
    # trusting proxies (old tests green, longest block, "no change" items, import errors scored
    # as one failure). On by default so it can be A/B'd off; unmeasured on a model until that A/B
    "pipeline_signals": "pipelines trust the task, not proxies: the project pipeline ranks a test "
    "file or task name that will not import (or a config pytest cannot read) below any number of "
    "failing tests; asks the writer for one file per reply labelled `File: path` and takes the "
    "block for that file (a reply only for another file is sent back once with the reason); drops "
    "plan items that say no change and retries an empty plan once before falling back with a "
    "notice; direct, project and deliverable pipeline writes get the atomic check (parse + new "
    "F821/F811) with the refusal fed back to the writer as its repair input; merge_code keeps "
    "the reply's new or changed module-level assignments (TypeVars, constants)",
    # since 2026-10-16: web research, asked once per session, self-hosted SearXNG.
    # Inert until a backend is configured (web.backend) and the user allows it (vega/web.py)
    "web_research": "web_search and fetch_page tools (a configured backend, permission asked once "
    "per session; only the query is sent, never code or secrets; fetched pages cached in the "
    "knowledge library with URL, dates and trust tier); a task asking to look things up, research "
    "or cite sources goes to the agent loop while the tools are available",
    # since 2026-10-17: stackoverflow.com answers 403 to any non-browser client; no
    # impersonation or bot-check bypass, official APIs instead (vega/web.py site_adapter)
    "web_site_apis": "fetch_page reads Stack Exchange, GitHub, PyPI, npm, crates.io, Wikipedia "
    "and MDN pages through their official APIs (or the raw source) instead of the HTML page; a "
    "site that blocks automated reading is reported as such, never worked around",
    # since 2026-10-17: adapt to a site's API the way Vega solves code. Inert without
    # web permission and a model; every adapter is approved by the user (vega/adapter_builder.py)
    "web_adapter_builder": "a page that is blocked or too short on a site no adapter covers: find "
    "the site's OFFICIAL API (OpenAPI/Swagger, developer docs, APIs.guru), have the model write an "
    "adapter, verify it in a no-network sandbox against that page and 2-3 sibling pages (≤3 "
    "repair rounds), and store it in ~/.vega/adapters only after the user approves it; approved "
    "adapters load after the built-in ones",
    "web_citations": "agent loop: a final answer of a turn that used web research and cites none "
    "of the pages it used is sent back once with those URLs, asking for citations",
    # research diagnosis (2026-10-18): most failures acted on search snippets without fetching
    # the decisive page. On by default so each can be A/B'd off; unmeasured on a model
    "web_fetch_first": "web research: only pages fetched this turn count as read. The citation "
    "send-back (web_citations) names any search result the answer cites, and any web URL in the "
    "`sources` of a JSON file written this turn, that was never fetched, asking to fetch it and "
    "check the answer against it (≤2 send-backs per turn); the turn's first write after a "
    "web_search with nothing fetched is not run once: the result names the top 2 results to "
    "fetch first",
    "web_official_hint": "web_search results whose host holds the queried product's name (or is "
    "a known official-docs host) are marked \"(official site)\"; the fetch-first notes list them "
    "first",
    # added 2026-10-18: decision tasks fetched the 2 suggested results, not the decisive page
    # lower in the ranking (vega/web_facets.py). Default on; unmeasured on a model
    "web_facet_fetch": "web research: the fetch-first notes name, per facet of the task (each "
    "option whose site a result is on, each stated version/date/identifier a result mentions, "
    "topic words like licence/pricing; repo files read this turn add values and topics), the "
    "best unfetched result covering it (official site first, ≤4), plus an option's newer "
    "changelog/news/release/advisory/pricing page; at the finish of a turn that read pages, an "
    "option or version of the task (or a version/date the answer or written files state) that "
    "no page read covers is sent back once with the result that covers it",
    # added 2026-10-19: some research failures were a write wrecking an existing file. Default
    # on, measured offline by replay; unmeasured on a model
    "write_guard": "agent loop: a write is not run once (the result names what would be lost, "
    "shows the file and says to use edit_file or repeat the same call) when write_file replaces "
    "an existing file not read this turn and loses lines, when a dependency/config file "
    "(requirements*.txt, *.toml, *.cfg/*.ini, package.json, *.lock) loses a package or key, or "
    "when a .py file loses a module-level name (a def/class/assignment deleted or swallowed by an "
    "unclosed string); the same call repeated runs; pipeline writes are never held",
    # added 2026-10-20: every write of a file holding ``` was lost (the block ended inside the
    # code), and repairs worked from bare test ids. Default on, measured offline by replay;
    # unmeasured on a model
    "fence_lines": "pipelines read a reply's code blocks by whole-line fences (a ``` inside the "
    "code, e.g. `startswith(\"```\")`, no longer ends the block; a closing fence is at least as "
    "long as the opening one), and a file shown in a prompt that holds ``` is fenced with ````",
    "short_tracebacks": "pipeline pytest runs (project pipeline, direct pipeline, a task-named "
    "pytest check) use --tb=short, so a library-raised error (argparse SystemExit: 2) or 10+ "
    "failures still fit the sandbox output with their assertion lines; the direct pipeline's "
    "repair shows the condensed failures instead of the last 12 lines",
    "output_head_tail": "sandboxed check output over the 8,000-character limit (agent loop and "
    "pipelines alike) keeps its head (the first failures' details) and its tail (the summary), "
    "cut in the middle with a marked line, instead of only the tail",
    "direct_context": "direct pipeline: the writer also sees the project modules the target "
    "imports (`from ledger import Ledger` -> ledger/__init__.py, vendored packages too) "
    "and the docs/README files that mention a distinctive token of the task (an error code like "
    "AB-12, a project package's name), within the inputs' budget",
    # added 2026-10-20: a share of multi-file goal failures were routing. Default on, measured
    # offline by route replay; unmeasured on a model
    "route_cli_target": "routing: a task that would go to the direct pipeline with one target "
    "also changes the CLI file a backticked command in it runs (`app export --csv` -> the file "
    "whose argparse prog= is `app`, or the module a pyproject [project.scripts] entry `app` "
    "runs); with that second target it goes to the project pipeline (kept direct otherwise)",
    "route_prose_class": "routing: as route_cli_target, for a CamelCase project class the task "
    "names in prose without backticks (\"picked up by ApiClient\") that exactly one project "
    "file defines",
    # added 2026-10-21: a `python -m pkg` task could not pass while pkg/__main__.py was never
    # created. Measured offline by route replay only; unmeasured on a model
    "route_module_entry": "routing: a task that would go to the direct pipeline, where the task, "
    "a doc or test it names, or a visible test runs a project package as `python -m pkg` and "
    "pkg/__main__.py does not exist, also targets pkg/__main__.py and leaves the direct pipeline "
    "(which cannot create files): project pipeline with tests or acceptance checks, else the "
    "agent loop",
    "question_change_cues":"routing: \"report\" is a question cue only as a verb (\"report the "
    "marker\", \"Only report it\"), not a noun (\"The stock report must mark\"); a task that "
    "states a requirement with \"must\" (outside a negation, not of the answer) is a change, "
    "never the question pipeline (a whole-scope \"do not change any file\" still is)",
    "question_code_guard": "question pipeline: an answer with a code block (a def/class, or 6+ "
    "lines) to a task that states \"must\" requirements or asks for a change falls back to the "
    "agent loop instead of ending the turn with code nobody wrote to a file",
    "acceptance_uninstalled": "acceptance: importing a module that fails with "
    "ModuleNotFoundError for a module that is not part of the project is \"not checkable here "
    "(module not installed)\", not a failure",
    "json_template": "a task that states `NAME.json as {...}` or \"NAME.json with keys ...\": the "
    "finish gate (agent loop) and the deliverable pipeline's checks require every stated key, and "
    "`sources` (when stated) to be a list of strings; the feedback names the missing keys",
    # added 2026-10-17. On by default so each can be A/B'd off; unmeasured on a model
    "teach_root_cause": "teach mode: the takeaway of a bug fix names the root-cause mechanism first "
    "(the language or runtime behaviour that made the old code wrong, named precisely), then the "
    "fix, then the rejected candidate (direct pipeline writer and agent loop instructions)",
    "clarify_model": "clarify_first: a short change request the rules cannot settle (no anchor, "
    "e.g. 'Add alerts to the dashboard.') gets one short model call with a constrained "
    "JSON reply, seeing the README, the loaded project context and the file list, that decides "
    "whether to ask (default since 2026-10-17; unmeasured until an A/B)",
    "failing_checks_quoted": "direct pipeline: when it gives up with checks still failing, the "
    "final answer quotes the first 3 failing check lines (e.g. \"No module named 'yamlkit'\") "
    "instead of only their count",
    "acceptance_prose_calls": "acceptance: a name written with empty parentheses in prose "
    "(\"calls `format_total()`\") names the function; it is not a zero-parameter signature "
    "requirement (only `def f()` is)",
    "acceptance_stubs": "acceptance: a task-named function whose body still raises "
    "NotImplementedError (or is only `...`/`pass` with a TODO comment) fails with a 'still a stub' "
    "item; @abstractmethod and base-class methods a subclass overrides are exempt",
    "sandbox_toolchains": "run_check's sandbox gives Go a writable GOCACHE/GOPATH/GOMODCACHE inside "
    "the clone (GOPROXY=off, GOTOOLCHAIN=local), and run_check also allows javac, java, Rscript "
    "and `node <script.js>` (a script file inside the project)",
    # added 2026-10-17. On by default so each can be A/B'd off; unmeasured on a model
    "project_inputs": "project pipeline: the route's non-Python read-only inputs (text notes, "
    "docs, data files) are shown to the planner and the writer, as bounded excerpts "
    "that fit the window",
    "project_large_files": "project pipeline: a planned Python file over 9,000 characters is "
    "written through the large_projects path (outline + relevant definitions, the reply merged "
    "by definition) instead of being left unchanged, even with large_projects off",
    "project_honest_answer": "project pipeline: the final answer names exactly the files written "
    "and the planned files left unchanged with the reason, and never says the tests pass when "
    "nothing was written or no test ran",
    "project_no_tests": "project pipeline: pytest exit code 5 (no tests collected) counts as no "
    "tests, not as a failing test",
    "project_run_script": "project pipeline: when the task runs a script that writes an output "
    "(`python report.py` writes out.json), the script is run in the sandbox after each "
    "round's writes and its output files are saved, instead of the writer hand-writing them",
    # added 2026-10-17: the model misread a boundary comparison (`500 < 500` read as true)
    "question_run_code": "question pipeline: for 'what does f return / what would X be for "
    "input Y' with literal inputs in the question, the project's function (and the project "
    "functions it calls) is run on them in the sandbox and the results are given to the model "
    "as evidence before it answers",
    # added 2026-10-20. Default on, measured offline by replay; unmeasured on a model
    "project_route_targets": "project pipeline: the Python files routing resolved as targets "
    "(app/cli.py for `app export --csv`, app/client.py for ApiClient) are named to the planner "
    "as required, and added to the plan when it leaves one out; the planner's own files are kept",
    "project_auto_imports": "project pipeline: before a write is checked, an undefined name the "
    "code uses is imported when it is a stdlib module used as one (re.sub -> import re; this ran "
    "only under opt-in import_grounding before), a name exactly one project module defines "
    "(ParseError -> from .errors import ParseError), or a well-known stdlib name "
    "(DecimalException -> from decimal import DecimalException)",
    # added 2026-10-21. Default on from an offline replay; unmeasured on a model.
    # Needs project_auto_imports (it extends that step)
    "project_layer_imports": "project pipeline auto-imports: the stdlib name table is derived "
    "from the stdlib (date, datetime as class or module by use, Path, List, ...; ambiguous and "
    "generic names left out), a test file gets `import pytest`, a name in a module the task says "
    "the file must not import is not imported, and a write refused for a name whose only home is "
    "a sibling layer or a forbidden module says so (remove or move the code) instead of "
    "\"define or import\"",
    # added 2026-10-21: a visible test still calling the old signature failed work that was
    # otherwise right. Offline replay only; unmeasured on a model
    "plan_contradicting_tests": "project pipeline: when the task states new parameters for an "
    "existing function or class (`Service(repository)` where it takes data_dir now) and "
    "a visible test file calls it the old way, and the task does not ask for test changes, that "
    "test file is added to the plan to update only those calls; its write is refused when it "
    "drops a test, lowers a test's assertion count, or changes an assertion that does not call "
    "the changed name",
    "project_reexport_guard": "project pipeline: after a round's writes, a package __init__.py "
    "that re-exports names its source module does not define at module level (from .models "
    "import total, a method) has those names dropped, so the package imports",
    "project_hold_back": "project pipeline: a file saved in a round that imports a module (or "
    "names from it) whose write was refused in the same round is put back as it was before the "
    "round and counted as not saved, with the reason, so the package stays importable",
    "clarify_greedy": "clarify_model: the ask/act check call is sent at temperature 0 (the rest "
    "of the turn keeps the run's sampling), so one request gets one verdict; off sends the turn's "
    "own sampling (unmeasured until an A/B)",
    # task_checks added 2026-10-20 (task_checks.py). Default on from offline replays only: it
    # caught wrong-logic solutions and never fired on a correct one; unmeasured on a model.
    # restore_later_files (write_loop): seen while replaying those checks (2026-10-20)
    "restore_later_files": "write loop: when an earlier round is kept, files first written in a "
    "later round are put back as they were before the loop (they were left changed, so the kept "
    "round's package could fail its own tests)",
    "task_checks": "project pipeline: checks from the task's own words, run after each round "
    "with no model call: a backticked command of the project's CLI (`app show --verbose`) run "
    "through its main() (argparse rejecting a written word, or a new traceback, fails), and "
    "\"APP_PORT=9000 overrides `port`\" through the settings loader; kept only if they fail "
    "because the behaviour is missing, dropped when they keep failing while all else passes",
    # added 2026-10-21. Default on from offline replays only; unmeasured on a model
    "task_module_commands": "the task's own commands, run properly: a bare `python -m pkg` is a "
    "name, so acceptance checks the full invocation the task or its README gives instead "
    "(`python -m pkg \"input\" --after ...`) and never fails the bare name on "
    "argparse's usage exit; task_checks also run `python -m pkg ...` commands (each segment of "
    "a backticked pipeline), feeding the ones that read stdin a sample of the project's data",
    # added 2026-10-21: a task's own reproduction command can name an example file the project
    # does not have; once the module existed the command exited 1 ("notes.txt: No such file or
    # directory", the behaviour the task asks for) and acceptance kept failing it through every
    # repair round. Default on from offline replays; unmeasured on a model
    "command_absent_inputs": "acceptance: a task command whose input file argument the project "
    "does not have and the task never mentions outside the command (`python -m pkg count "
    "notes.txt`) passes when it exits non-zero only to report that file missing; any other "
    "failure (no `__main__`, a usage error, a traceback elsewhere) still fails",
    # added 2026-10-20: CSS/HTML/JS tasks all ran in the agent loop, which handled them poorly;
    # measured offline by route and check replays; default on, unmeasured on a model
    "web_pipeline": "a change to only CSS/HTML/JS files (and package.json, at most 5) goes to the "
    "non-Python pipeline's front-end config: CSS merged rule by rule (rules left out are kept), "
    "then syntax checks (CSS, HTML nesting, `node --check`), each Node module loaded once in the "
    "sandbox, the task's check commands and visible tests, and on macOS the page rendered in WebKit (script errors, declarations the engine "
    "drops, and the measured rules the task names: reduced motion, dark scheme, contrast, layout "
    "shift), ≤3 repairs, the round with the fewest failing checks kept",
    # added 2026-10-19/20. Default on, measured offline by replay; unmeasured on a model
    "finish_gate_answer_file": "finish gate: a JSON object in the final answer with the keys of "
    "the task's JSON template is that output file even when no heading names it, and is compared "
    "with the file by value (scalars, `sources` as a set); a one-line block that is a dependency "
    "pin or an assignment (`requests==2.32.3`, `API_VERSION = \"v2\"`) is checked too; "
    "the send-back names the values that differ",
    "finish_gate_normalised": "finish gate: a block the answer shows for a file counts as written "
    "when, comments and all whitespace removed, it is part of the file (each part between "
    "elided `...` lines), not only when 80% of its lines are the file's lines",
    # added 2026-10-21: change requests often finished in prose with no write (failure_class
    # no_edit). Default on from an offline replay; unmeasured on a model
    "finish_gate_no_edit": "finish gate: a final answer to a task routed as a change (not a "
    "question) with no file written or edited this turn (and no run_command that may write) is "
    "sent back once: find the responsible file and edit it, or say explicitly why no change is "
    "needed; the next answer is accepted",
    "cite_named_pages": "web research: once per turn, the citation send-back lists the pages a "
    "fetch-first or finish note named for a facet of the task, that were fetched, and that the "
    "written JSON's `sources` leave out",
    "constraint_docs": "agent loop: documents the task names as its rules (\"following "
    "SECURITY.md\", \"under docs/RULES.md\", \"against the guidelines in docs/\") hold "
    "the turn's first write once while one is unread, listing them; at the finish of a decision "
    "task (a `choice` key, or choose/pick/decide) their rule sentences are quoted back once with "
    "the choice written",
    "facet_same_option": "web_facet_fetch: an option's newer release/news page is picked only "
    "when its version of that option (a version right after the option's name, or the page's "
    "version on the option's own site) is above the pages read, and never when it is dated "
    "before the newest page of the option already read",
    # added 2026-10-20. Default on, measured offline by replay; unmeasured on a model
    "deliverable_disagreement": "deliverable pipeline: when one reply has a script and a `File:` "
    "block for the same deliverable and they differ, one repair round quotes the differing "
    "fields (script vs block) instead of silently keeping the script's output",
    "format_rules": "deliverable pipeline: checks from the task's own format rules: a stated path "
    "prefix (`starting /Users/dana/`), \"at most N characters\" per line of a text deliverable, "
    "and each item's value shapes for a \"JSON list, one object per ...: `{...}`\" template "
    "(no nulls, numbers where the placeholder is a number, fixed-length lists, string caps)",
    "script_error_values": "deliverable pipeline: a failed script's traceback starts at the "
    "script, shows its source lines and the values of the names on the failing line(s)",
    "sandbox_path_scrub": "sandboxed command output has host paths replaced: the clone and the "
    "project by `.`, Vega's venv by <venv>, the interpreter by <python>, the user's home by `~`",
    "deliverable_code_label": "deliverable pipeline: a Python script under a `File: x.json` label "
    "is run as the script, and a refused non-code deliverable is not asked for again \"in the "
    "same form as before\"",
    "noop_edit_refused": "edit_file with new_text identical to old_text is refused with a note "
    "instead of reported as an exact-match edit",
    "sandbox_env_scrub": "sandboxed checks run without Vega's own VEGA_* environment (an eval's "
    "pinned seed/temperature and A/B feature switches are the harness's settings, not the "
    "project's)",
    "sandbox_read_guard": "deliverable scripts (and their `Run:` commands) may not read host "
    "files outside the project and the interpreter: a macOS sandbox profile denies reads of "
    "/Users, /Volumes, /private/etc, /private/var, /private/tmp, /Library/Keychains, and the "
    "runner's audit hook refuses them in Python on every platform (naming the fixture copy when "
    "the task maps `root/` to `/`)",
}
ENV = "VEGA_FEATURES_OFF"
OPT_IN: dict[str, str] = {
    "constrained": "every reply is one JSON action forced by a schema (structured outputs) instead "
    "of native tool calls",
    "search_streak": "after 4 searches in a row find nothing, show the file list and suggest "
    "reading the likely file",
    "show_on_miss": "when edit_file's old_text is not found, show the file's current lines",
    "references_hint": "after an edit that replaces a name, list where the old name still appears",
    "append_guard": "refuse an append (empty old_text) that redefines an existing top-level "
    "function or class; show the existing definition to edit instead",
    "spec_tests": "project pipeline: first write pytest tests from the task (kept outside the "
    "project in .vega-spec/, used only if they fail before the change) and repair against them too",
    "rollback_breakers": "project pipeline: when a round makes test files that loaded before fail "
    "to load (import or syntax errors), put touched files back to their original one at a time, "
    "keeping each revert that removes the breakage without failing more spec tests or task names "
    "(no model calls; assertion failures never trigger it: they may be intended)",
    "import_grounding": "project pipeline: show the writer the project's real modules, and after "
    "each round report imports of project modules or names that do not exist (checked statically)",
    "plan_churn_guard": "after 2 update_plan calls in a row with no other tool between, the result "
    "tells the model to do the next step; from 4 in a row the update is refused",
    "large_projects": "project pipeline on large projects (up to 600 Python files): a focused "
    "planner map; files over 9,000 characters shown as an outline plus the relevant definitions, "
    "the reply merged method by method; longer test timeout; tests failing before the change "
    "ranked after new failures",
    "review_diff": "before finishing a turn that changed files, show the model its diff beside the "
    "request once and ask it to check each example",
    "task_checks_model": "project pipeline: one constrained-JSON model call writes up to 5 small "
    "checks of the task's explicit statements (each quoting the task's words), gated like "
    "task_checks: kept only if they fail on the unchanged project by an assertion or a name the "
    "task places there, dropped when they keep failing while all else passes",
    "acceptance_model": "acceptance: when the rules find no required names or commands in the "
    "task, one short constrained-JSON model call lists them (kept only if they resolve in the "
    "project)",
    # added 2026-10-21: agent-loop bug fixes wrote source first, usually with no check before
    # it, and never wrote a test that reproduced the report. Opt-in until an A/B
    "repro_first": "agent loop: a bug report (fix/bug/error/doesn't/never ...) on a Python project "
    "with a pytest directory and 3+ source files: the turn's first source write is held once "
    "until the model writes one failing test in tests/test_vega_repro.py (Vega runs it when "
    "written, before any source edit, ≤2 times); at the end of the turn that file stays only if "
    "it passes, else it is removed; not for tasks that ask for tests or limit the files to change",
}
ENV_ON = "VEGA_FEATURES_ON"
# Switches whose code was removed (2026-10-17), with what replaced them.
# Still an error in VEGA_FEATURES_OFF/ON (an A/B arm would silently not differ), but a named one;
# the experiment records keep the experiments that used them.
RETIRED: dict[str, str] = {
    "shape_routing": "routing.route_task is the only router",
    "data_inputs": "routing.route_task classifies data inputs",
    "direct_long_tasks": "routing.route_task budgets the direct writer by file size",
    "parse_guard": "replaced by atomic_writes",
    "replace_function": "replaced by edit_by_name (replace_definition)",
    "repeat_unchanged": "replaced by turn_ledger",
}


def _retired_note(names: set[str]) -> str:
    retired = sorted(names & RETIRED.keys())
    return "".join(f" {name} was retired on 2026-10-17: {RETIRED[name]}." for name in retired)


def disabled() -> frozenset[str]:
    """Features switched off via VEGA_FEATURES_OFF; unknown names are an error, not a silent no-op."""
    names = {name.strip() for name in os.environ.get(ENV, "").split(",") if name.strip()}
    unknown = names - FEATURES.keys()
    if unknown:
        raise ValueError(
            f"Unknown feature(s) in {ENV}: {', '.join(sorted(unknown))}.{_retired_note(unknown)} "
            f"Known: {', '.join(FEATURES)}."
        )
    return frozenset(names)


def enabled() -> frozenset[str]:
    """Opt-in features switched on via VEGA_FEATURES_ON; unknown names are an error. A feature
    that has become a default (e.g. project_pipeline) is accepted and ignored: it is on already,
    so older experiments and scripts that switch it on keep working."""
    names = {name.strip() for name in os.environ.get(ENV_ON, "").split(",") if name.strip()}
    names -= FEATURES.keys()
    unknown = names - OPT_IN.keys()
    if unknown:
        raise ValueError(
            f"Unknown opt-in feature(s) in {ENV_ON}: {', '.join(sorted(unknown))}."
            f"{_retired_note(unknown)} Known: {', '.join(OPT_IN)}."
        )
    return frozenset(names)


def on(name: str) -> bool:
    if name in FEATURES:
        return name not in disabled()
    if name in OPT_IN:
        return name in enabled()
    raise KeyError(f"Unknown feature {name!r}")
