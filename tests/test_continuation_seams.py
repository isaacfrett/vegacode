"""Joining a reply cut off at the token limit with its continuation."""

from __future__ import annotations

import ast

import pytest
from switches import switch_off

from vega.pipeline import join_continuation


def test_a_cut_inside_the_indent_is_not_indented_twice() -> None:
    # recorded from granite-8b at 32-token replies: "   " then "    return ..." (7 spaces)
    text = '```python\ndef squares(n):\n    """Squares."""\n   '
    joined = join_continuation(text, "    return [i * i for i in range(1, n + 1)]\n```")
    code = joined.removeprefix("```python\n").removesuffix("```")
    ast.parse(code)
    assert "\n    return [" in joined and "\n       return" not in joined


def test_a_cut_mid_line_gets_one_space_not_the_restarted_indent() -> None:
    joined = join_continuation("    The squares of integers from 1", "    up to n, inclusive.")
    assert joined == "    The squares of integers from 1 up to n, inclusive."


def test_a_restarted_line_replaces_the_cut_one() -> None:
    # recorded from granite-8b: the doctest line cut at "[1, 4" came back whole
    text = '    >>> squares(3)\n    [1, 4'
    assert join_continuation(text, '    [1, 4, 9]\n    """') == '    >>> squares(3)\n    [1, 4, 9]\n    """'


def test_a_cut_after_a_space_drops_the_restarted_indent() -> None:
    assert join_continuation("integers from ", "    1 to n") == "integers from 1 to n"


@pytest.mark.parametrize("text, part", [
    ("def f():\n", "    return 1\n"),        # cut at a line start: the indent belongs to the line
    ("return [1, 4,", "9]"),                 # exact continuation, no leading space
    ("x = foo(a,", " b)"),                   # one space is the token's own
])
def test_clean_continuations_are_joined_unchanged(text: str, part: str) -> None:
    assert join_continuation(text, part) == text + part


def test_switch_off_joins_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    switch_off(monkeypatch, "continuation_seams")
    assert join_continuation("   ", "    return 1") == "       return 1"
