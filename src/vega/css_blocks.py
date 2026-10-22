"""CSS by blocks: parse a stylesheet into rules, merge a writer's rules into a file, check syntax.

Why (2026-10-20): the agent loop rewrote styles.css
whole and dropped rules, and an `edit_file` with `old_text="#hero {"` appended a second copy of
declarations after a rule and left the file malformed. The web pipeline writes CSS through
`merge` instead: each rule the reply sends replaces the file's rule with the same selector (or
at-rule prelude), new rules go in where the reply puts them, and every rule the reply leaves out
is kept. A conditional group rule (@media, @supports, @container, @layer, @scope) is merged rule
by rule inside, so `@media (max-width: 600px) { .a {...} }` does not drop that block's other rules.
A rule sent with an empty body (`.old {}`) deletes the file's rule.

No third-party parser: a small tokenizer that knows comments, strings and nested braces, enough to
find rule boundaries and report what a browser would drop (an unclosed block, a stray `}`, a
declaration without a colon, an unknown media feature).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

GROUP_RULES = {"media", "supports", "container", "layer", "scope", "document", "starting-style"}
MEDIA_FEATURES = {
    "width", "height", "aspect-ratio", "orientation", "resolution", "device-width",
    "device-height", "device-aspect-ratio", "color", "color-index", "monochrome", "color-gamut",
    "hover", "any-hover", "pointer", "any-pointer", "grid", "scan", "update", "overflow-block",
    "overflow-inline", "display-mode", "dynamic-range", "video-dynamic-range", "forced-colors",
    "inverted-colors", "prefers-color-scheme", "prefers-contrast", "prefers-reduced-motion",
    "prefers-reduced-transparency", "prefers-reduced-data", "scripting",
    "-webkit-device-pixel-ratio", "-webkit-min-device-pixel-ratio",
    "-webkit-max-device-pixel-ratio", "min-resolution", "max-resolution",
}
MEDIA_VALUES = {"prefers-reduced-motion": {"reduce", "no-preference"},
                "prefers-color-scheme": {"light", "dark"},
                "prefers-contrast": {"more", "less", "no-preference", "custom"},
                "orientation": {"portrait", "landscape"},
                "hover": {"none", "hover"}, "any-hover": {"none", "hover"},
                "pointer": {"none", "coarse", "fine"}, "any-pointer": {"none", "coarse", "fine"}}


@dataclass
class Rule:
    """One top-level statement: `prelude { body }` or `@name prelude;`. `text` is its source from
    the end of the previous statement (leading whitespace and comments included)."""

    prelude: str
    body: str | None  # None for a statement without a block (`@import ...;`)
    text: str
    line: int  # 1-based line of the prelude
    body_line: int = 0

    @property
    def key(self) -> str:
        return normalise(self.prelude)

    @property
    def at_name(self) -> str:
        match = re.match(r"@([\w-]+)", self.prelude.strip())
        return match.group(1).lower() if match else ""

    @property
    def group(self) -> bool:
        return self.body is not None and self.at_name in GROUP_RULES


@dataclass
class Sheet:
    rules: list[Rule] = field(default_factory=list)
    tail: str = ""  # whitespace and comments after the last rule
    errors: list[str] = field(default_factory=list)


def normalise(prelude: str) -> str:
    """A selector or at-rule prelude without comments and with whitespace collapsed (around
    commas and combinators too), so `.a,\n.b` and `.a, .b` are the same rule."""
    text = re.sub(r"/\*.*?\*/", " ", prelude, flags=re.DOTALL)
    text = " ".join(text.split())
    text = re.sub(r"\s*([,>+~])\s*", r"\1", text)
    text = re.sub(r"\(\s*", "(", re.sub(r"\s*\)", ")", text))
    text = re.sub(r"\s*:\s*(?=[^()]*\))", ":", text)  # (max-width : 600px)
    return text.lower() if text.startswith("@") else text


def _skip_string(text: str, i: int) -> int:
    quote = text[i]
    i += 1
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == quote or text[i] == "\n":
            return i + 1
        i += 1
    return len(text)


def _line(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def parse(text: str, *, line_offset: int = 0) -> Sheet:
    """Top-level statements of a stylesheet (or of a group rule's body), with syntax errors."""
    sheet = Sheet()
    i = start = 0
    prelude_start: int | None = None
    n = len(text)
    while i < n:
        char = text[i]
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end < 0:
                sheet.errors.append(f"line {_line(text, i) + line_offset}: comment never closed")
                i = n
                break
            i = end + 2
            continue
        if char in "\"'":
            if prelude_start is None:
                prelude_start = i
            j = _skip_string(text, i)
            if j > n or text[j - 1] not in "\"'":
                sheet.errors.append(f"line {_line(text, i) + line_offset}: string never closed")
            i = j
            continue
        if char.isspace():
            i += 1
            continue
        if prelude_start is None:
            prelude_start = i
        if char == "{":
            close = _matching(text, i)
            prelude = text[prelude_start:i]
            line = _line(text, prelude_start) + line_offset
            if close is None:
                sheet.errors.append(f"line {line}: `{_short(prelude)} {{` is never closed (a "
                                    "missing `}`)")
                close = n
            if not prelude.strip():
                sheet.errors.append(f"line {line}: a block with no selector")
            body = text[i + 1:close]
            sheet.rules.append(Rule(prelude.strip(), body, text[start:close + 1], line,
                                    _line(text, i + 1) + line_offset))
            i = start = close + 1
            prelude_start = None
            continue
        if char == ";":
            prelude = text[prelude_start:i]
            line = _line(text, prelude_start) + line_offset
            if prelude.strip().startswith("@"):
                sheet.rules.append(Rule(prelude.strip(), None, text[start:i + 1], line))
            else:
                sheet.errors.append(f"line {line}: `{_short(prelude)};` outside any rule (a "
                                    "declaration needs a selector and braces)")
                sheet.rules.append(Rule(prelude.strip(), None, text[start:i + 1], line))
            i = start = i + 1
            prelude_start = None
            continue
        if char == "}":
            sheet.errors.append(f"line {_line(text, i) + line_offset}: a `}}` with no block open "
                                "(an extra brace, or declarations left after a rule)")
            i += 1
            start = i
            prelude_start = None
            continue
        i += 1
    if prelude_start is not None and text[prelude_start:].strip():
        leftover = text[prelude_start:]
        sheet.errors.append(f"line {_line(text, prelude_start) + line_offset}: "
                            f"`{_short(leftover)}` at the end is not a complete rule")
        sheet.rules.append(Rule(leftover.strip(), None, text[start:], _line(text, prelude_start)))
        start = n
    sheet.tail = text[start:]
    return sheet


def _short(text: str, limit: int = 60) -> str:
    text = " ".join(re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL).split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _matching(text: str, open_index: int) -> int | None:
    """Index of the `}` closing the `{` at open_index (comments and strings skipped)."""
    depth = 0
    i = open_index
    n = len(text)
    while i < n:
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end < 0:
                return None
            i = end + 2
            continue
        char = text[i]
        if char in "\"'":
            i = _skip_string(text, i)
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


# -- declarations -----------------------------------------------------------------------------------

_DECLARATION = re.compile(r"^\s*(--[\w-]+|-?[A-Za-z_][\w-]*)\s*:(.*)$", re.DOTALL)


@dataclass
class Declaration:
    selector: str
    prop: str
    value: str
    line: int


def declarations(text: str) -> tuple[list[Declaration], list[str]]:
    """Every declaration in a stylesheet (in rules, nested rules and group rules; not in
    @keyframes steps or @font-face, which are checked for syntax only), and the syntax errors:
    the parse errors plus pieces of a rule body that are not `property: value`."""
    sheet = parse(text)
    found: list[Declaration] = []
    errors = list(sheet.errors)
    _walk(sheet, found, errors, text, 0)
    return found, errors


def _walk(sheet: Sheet, found: list[Declaration], errors: list[str], text: str,
          offset: int) -> None:
    for rule in sheet.rules:
        if rule.body is None:
            continue
        if rule.group:
            inner = parse(rule.body, line_offset=rule.body_line - 1)
            errors.extend(inner.errors)
            _walk(inner, found, errors, text, offset)
            continue
        collect = not rule.at_name  # keyframes / font-face: syntax only
        _body(rule.prelude, rule.body, rule.body_line, found if collect else None, errors)


def _body(selector: str, body: str, first_line: int, found: list[Declaration] | None,
          errors: list[str]) -> None:
    """Declarations of one rule body; nested rules (`&:hover { ... }`, keyframe steps) recursed."""
    i = start = 0
    n = len(body)
    while i < n:
        if body.startswith("/*", i):
            end = body.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        char = body[i]
        if char in "\"'":
            i = _skip_string(body, i)
            continue
        if char == "(":
            depth = 0
            while i < n:  # url(...), var(...): a ; or { inside does not end a declaration
                if body[i] == "(":
                    depth += 1
                elif body[i] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            i += 1
            continue
        if char == ";":
            _declaration(selector, body[start:i], first_line + body.count("\n", 0, start),
                         found, errors)
            i = start = i + 1
            continue
        if char == "{":
            close = _matching(body, i)
            close = n if close is None else close
            nested = body[start:i].strip()
            _body(nested, body[i + 1:close], first_line + body.count("\n", 0, i), found, errors)
            i = start = close + 1
            continue
        i += 1
    _declaration(selector, body[start:], first_line + body.count("\n", 0, start), found, errors)


def _declaration(selector: str, piece: str, line: int, found: list[Declaration] | None,
                 errors: list[str]) -> None:
    cleaned = re.sub(r"/\*.*?\*/", " ", piece, flags=re.DOTALL)
    if not cleaned.strip():
        return
    line += piece[:len(piece) - len(piece.lstrip())].count("\n")
    match = _DECLARATION.match(cleaned)
    if match is None or not match.group(2).strip():
        errors.append(f"line {line}: `{_short(cleaned)}` in `{_short(selector, 40)}` is not a "
                      "`property: value` declaration")
        return
    if found is not None:
        found.append(Declaration(" ".join(selector.split()), match.group(1).lower(),
                                 " ".join(match.group(2).split()), line))


# -- media queries ------------------------------------------------------------------------------------

_FEATURE = re.compile(r"\(\s*([-\w]+)\s*(?::\s*([^()]*)|[<>=][^()]*)?\)")


def media_problems(text: str) -> list[str]:
    """Unknown media features (`prefers-reduce-motion`) and values (`prefers-color-scheme:
    darker`) in @media rules: a browser treats such a query as false, silently."""
    problems: list[str] = []

    def visit(sheet: Sheet) -> None:
        for rule in sheet.rules:
            if rule.at_name in {"media", "import"} or (rule.at_name == "custom-media"):
                for match in _FEATURE.finditer(rule.prelude):
                    name = match.group(1).lower()
                    base = re.sub(r"^(min|max)-", "", name)
                    if name not in MEDIA_FEATURES and base not in MEDIA_FEATURES:
                        problems.append(f"line {rule.line}: unknown media feature `{name}` in "
                                        f"`{_short(rule.prelude)}` (the query never matches)")
                        continue
                    value = (match.group(2) or "").strip().lower()
                    allowed = MEDIA_VALUES.get(name)
                    if allowed and value and value not in allowed:
                        problems.append(f"line {rule.line}: `{name}: {value}` is not a valid "
                                        f"value (one of {', '.join(sorted(allowed))})")
            if rule.group and rule.body is not None:
                visit(parse(rule.body, line_offset=rule.body_line - 1))

    visit(parse(text))
    return problems


def check(text: str) -> list[str]:
    """Syntax errors and media query problems of a stylesheet ([] when clean)."""
    _found, errors = declarations(text)
    return errors + media_problems(text)


# -- merging ------------------------------------------------------------------------------------------

def merge(original: str, reply: str) -> str:
    """The file after applying the reply's rules to `original` (see the module docstring).
    Same-key rules are matched in order (the 2nd `.a` of the reply replaces the 2nd `.a` of the
    file); a new rule goes right after the file rule matched just before it in the reply, else
    before the next matched one, else at the end."""
    if not original.strip():
        return reply
    old = parse(original)
    new = parse(reply)
    if old.errors or new.errors:
        # a reply that does not parse is written as sent (the checks report it and the loop keeps
        # the best round); a file that does not parse is replaced by a reply that does
        return reply
    return _merge_sheets(old, new)


def _merge_sheets(old: Sheet, new: Sheet) -> str:
    slots: list[list[str]] = [[rule.text] for rule in old.rules]  # each old rule's output text
    before: list[str] = []  # new rules that go before every old rule
    after: list[str] = []  # new rules that go at the end
    used: dict[str, int] = {}
    positions: dict[str, list[int]] = {}
    for index, rule in enumerate(old.rules):
        positions.setdefault(rule.key, []).append(index)
    matches: list[int | None] = []
    for rule in new.rules:
        taken = used.get(rule.key, 0)
        candidates = positions.get(rule.key, [])
        if taken < len(candidates):
            used[rule.key] = taken + 1
            matches.append(candidates[taken])
        else:
            matches.append(None)
    for number, rule in enumerate(new.rules):
        target = matches[number]
        if target is not None:
            current = old.rules[target]
            if rule.body is not None and not re.sub(r"/\*.*?\*/", "", rule.body,
                                                    flags=re.DOTALL).strip() \
                    and current.body is not None and current.body.strip():
                slots[target][0] = _leading(current.text)  # `.old {}`: delete the rule
            elif rule.group and current.group:
                slots[target][0] = _replace_body(current, _merge_sheets(
                    parse(current.body or ""), parse(rule.body or "")))
            else:
                slots[target][0] = _with_leading(current.text, rule.text)
            continue
        text = _spaced(rule.text)
        previous = next((matches[k] for k in range(number - 1, -1, -1)
                         if matches[k] is not None), None)
        following = next((matches[k] for k in range(number + 1, len(matches))
                          if matches[k] is not None), None)
        if previous is not None:
            slots[previous].append(text)
        elif following is not None:
            before.append(text) if following == 0 else slots[following - 1].append(text)
        else:
            after.append(text)
    out = "".join(before) + "".join("".join(slot) for slot in slots) + "".join(after)
    return out + old.tail


def _leading(text: str) -> str:
    """The whitespace before a rule (its comments go with the deleted rule)."""
    match = re.match(r"\s*", text)
    return match.group(0) if match else ""


def _with_leading(old_text: str, new_text: str) -> str:
    """The reply's rule, laid out where the old one was: the old rule's leading whitespace kept,
    the reply's own leading comments kept, its leading blank lines dropped."""
    lead = _leading(old_text)
    body = new_text.lstrip()
    return lead + body


def _spaced(text: str) -> str:
    """A new rule on its own line."""
    body = text.lstrip()
    return "\n" + body


def _replace_body(rule: Rule, body: str) -> str:
    """A group rule's text with its body replaced."""
    text = rule.text
    prelude_end = text.find(rule.prelude) + len(rule.prelude)
    open_index = text.find("{", prelude_end)
    return text[:open_index + 1] + body + "}"
