"""css_blocks: rule-by-rule merging of a writer's CSS, and the syntax checks the web pipeline runs."""

from __future__ import annotations

from vega import css_blocks

ORIGINAL = """* { box-sizing: border-box; }
/* layout */
main { max-width: 640px; margin: 0 auto; }
@keyframes spin { to { transform: rotate(360deg); } }
.card { flex: 1;
        transition: transform 200ms ease, box-shadow 200ms ease; }
@media (max-width: 600px) { .a { color: red; } .b { color: blue; } }
.spacer { height: 1600px; }
"""


def test_a_partial_reply_replaces_its_rules_and_keeps_every_other_rule() -> None:
    reply = (".card { flex: 1; transition: transform 200ms ease; }\n"
             "@media (prefers-reduced-motion: reduce) {\n  .card { transition: none; }\n}\n")
    merged = css_blocks.merge(ORIGINAL, reply)
    assert ".card { flex: 1; transition: transform 200ms ease; }" in merged
    assert "box-shadow 200ms" not in merged  # the old .card rule was replaced, not duplicated
    for kept in ("* { box-sizing", "main { max-width", "@keyframes spin", ".spacer {",
                 ".b { color: blue; }"):
        assert kept in merged
    # the new rule goes right after the rule the reply sent before it
    assert merged.index("prefers-reduced-motion") > merged.index(".card {")
    assert merged.index("prefers-reduced-motion") < merged.index("@media (max-width")
    assert css_blocks.check(merged) == []


def test_a_media_block_is_merged_rule_by_rule_and_prelude_spacing_does_not_matter() -> None:
    merged = css_blocks.merge(ORIGINAL, "@media (max-width:600px) { .a { color: pink; } }\n")
    assert ".a { color: pink; }" in merged and ".b { color: blue; }" in merged
    assert merged.count("@media") == 1


def test_an_empty_rule_deletes_and_a_whole_file_reply_is_the_file() -> None:
    assert ".spacer" not in css_blocks.merge(ORIGINAL, ".spacer {}\n")
    whole = ORIGINAL.replace("1600px", "800px")
    assert css_blocks.merge(ORIGINAL, whole).strip() == whole.strip()


def test_selectors_written_differently_are_the_same_rule() -> None:
    merged = css_blocks.merge("h1,\nh2 { margin: 0; }\np { color: red; }\n",
                              "h1, h2 { margin: 4px; }\n")
    assert merged.count("margin") == 1 and "margin: 4px" in merged and "p { color: red; }" in merged


def test_repeated_selectors_are_matched_in_order() -> None:
    original = ".a { color: red; }\n.b { x: 1; }\n.a { margin: 0; }\n"
    merged = css_blocks.merge(original, ".a { color: blue; }\n.a { margin: 2px; }\n")
    assert merged.count(".a {") == 2
    assert "color: blue" in merged and "margin: 2px" in merged and ".b { x: 1; }" in merged


def test_declarations_left_after_a_rule_are_reported() -> None:
    # a real corruption: edit_file(old_text="#hero {") appended a second
    # copy of the declarations after the rule
    broken = "#hero { min-height: 10px; }\n  aspect-ratio: 16 / 9;\n}\n.body p { margin: 0; }\n"
    problems = css_blocks.check(broken)
    assert any("outside any rule" in p for p in problems)
    assert any("`}` with no block open" in p for p in problems)


def test_unclosed_blocks_bad_declarations_and_unknown_media_features() -> None:
    problems = css_blocks.check(".x { colr red; }\n@media (prefers-reduce-motion: reduce) {"
                                " .y { color: red; } }\n.z { color: red\n")
    assert any("`colr red` in `.x` is not a `property: value`" in p for p in problems)
    assert any("unknown media feature `prefers-reduce-motion`" in p for p in problems)
    assert any("`.z {` is never closed" in p for p in problems)
    assert css_blocks.check("@media (prefers-color-scheme: darker) { a { color: red; } }")
    assert css_blocks.check('a { background: url("x;y.png"); content: "{"; }\n'
                            "@media (min-width: 600px) and (prefers-color-scheme: dark) {"
                            " a:hover { color: red; } }\n") == []


def test_declarations_are_listed_with_their_selector_and_line() -> None:
    found, errors = css_blocks.declarations("a {\n  color: red;\n}\n@media (min-width: 1px) {"
                                            " b { margin: 0 } }\n@font-face { font-family: X; }")
    assert errors == []
    assert [(d.selector, d.prop, d.value, d.line) for d in found] == [
        ("a", "color", "red", 2), ("b", "margin", "0", 4)]
