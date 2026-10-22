"""Helpers for driving the full-screen app headless with a scripted model (no server)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from vega.agent import CodingAgent
from vega.transcripts import SessionLog
from vega.tui import PromptInput, VegaApp


def chunk(content: str | None = None, tool_calls: list | None = None,
          prompt_tokens: int | None = None) -> SimpleNamespace:
    usage = SimpleNamespace(prompt_tokens=prompt_tokens) if prompt_tokens else None
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=tool_calls),
                                 finish_reason=None)],
        usage=usage,
    )


def calls(*specs: tuple[str, dict[str, Any]], prompt_tokens: int | None = None) -> list:
    """One streamed reply holding these tool calls."""
    out = []
    for index, (name, args) in enumerate(specs):
        fn = SimpleNamespace(name=name, arguments=json.dumps(args))
        out.append(chunk(tool_calls=[SimpleNamespace(index=index, id=f"c{index}-{name}",
                                                     function=fn)]))
    if prompt_tokens:
        out.append(chunk(prompt_tokens=prompt_tokens))
    return out


def answer(text: str, prompt_tokens: int | None = None) -> list:
    out = [chunk(text)]
    if prompt_tokens:
        out.append(chunk(prompt_tokens=prompt_tokens))
    return out


def scripted_agent(project: Path, replies: list, *, gate: Callable[[], None] | None = None,
                   **options: Any) -> tuple[CodingAgent, list[list[dict]]]:
    """A streaming agent whose model replays `replies`. gate() runs before each reply (tests use
    it to hold the worker at a known point)."""
    options.setdefault("undo_depth", 20)
    agent = CodingAgent(project, stream=True, num_ctx=16384, **options)
    seen: list[list[dict]] = []

    def create(**kwargs: Any) -> Any:
        seen.append(list(kwargs["messages"]))
        if gate is not None:
            gate()
        reply = replies.pop(0)
        return iter(reply)

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return agent, seen


def make_app(project: Path, agent: CodingAgent, tmp: Path, **options: Any) -> VegaApp:
    options.setdefault("warm_up", False)
    # Most tests were written against the classic layout; the app's own default is workbench
    # (tests/test_tui_view.py covers that).
    options.setdefault("layout", "classic")
    options.setdefault("memory_probe", lambda: 5.1 * 1024)
    options.setdefault("total_memory_gb", 16.0)
    return VegaApp(project, agent=agent, session_log=SessionLog(tmp / "sessions", project),
                   **options)


def drive(app: VegaApp, script: Callable[[Any], Awaitable[None]],
          size: tuple[int, int] = (120, 36)) -> None:
    async def main() -> None:
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            await script(pilot)

    asyncio.run(main())


async def wait_until(pilot: Any, condition: Callable[[], bool], seconds: float = 10.0) -> None:
    for _ in range(int(seconds / 0.02)):
        if condition():
            return
        await pilot.pause(0.02)
    raise AssertionError("condition not reached in time")


async def send(pilot: Any, text: str, *, wait: bool = True) -> None:
    app = pilot.app
    app.query_one(PromptInput).load_text(text)
    await pilot.press("enter")
    if wait:
        await wait_until(pilot, lambda: not app.working)
        await pilot.pause()
