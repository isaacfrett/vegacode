"""format_rules: checks from the format rules a deliverable task states (a JSON answer, a
commit message, a review file)."""

from __future__ import annotations

import json

from vega import features, format_rules

MAC = ("Write the answer to `answers/tally.json` as one JSON object with exactly these keys, paths "
       "absolute as on the Mac (starting `/Users/dana/`, no `~`): `bundle_id`, `database` (its "
       "ledger.sqlite), `preferences` (the preferences plist it reads), `log_file` (its log).")
COMMIT = ("Put the message in `commit_message.txt`. 1. Line 1 is `<type>(<scope>): <summary>`: "
          "the whole line is at most 72 characters. 2. Line 2 is empty. 3. Then a body of one or "
          "more lines, each at most 72 characters, that names every function.")
REVIEW = ("Report every real bug, as a JSON list in `review.json`, one object per bug: "
          '`{"file": "<path relative to the project root>", "lines": [<first>, <last>], '
          '"category": "<category>", "explanation": "<one line, at most 200 characters>"}`, '
          "where lines are 1-based line numbers.")
SECURITY = ("Write your findings to `security_report.json` as a JSON list with one object per "
            'finding, exactly `{"file": "<path relative to the project root, with / '
            'separators>", "line": <int>, "category": "<category>"}`, where category is one of x.')


def _mac(**values: str) -> dict[str, str]:
    answer = {"bundle_id": "com.example.tally",
              "database": "/Users/dana/Library/Containers/x/Data/Library/Application Support/"
                          "Tally/ledger.sqlite",
              "preferences": "/Users/dana/Library/Containers/x/Data/Library/Preferences/x.plist",
              "log_file": "/Users/dana/Library/Containers/x/Data/Library/Logs/Tally/tally.log"}
    answer.update(values)
    return {"answers/tally.json": json.dumps(answer)}


def test_a_stated_path_prefix_is_checked_on_path_values() -> None:
    assert format_rules.rules(MAC, ["answers/tally.json"]).prefix == "/Users/dana/"
    assert format_rules.problems(MAC, _mac()) == []  # bundle_id is not a path
    rooted = format_rules.problems(MAC, _mac(database="root/Users/dana/Library/x/ledger.sqlite"))
    assert rooted == [('answers/tally.json: "database" is "root/Users/dana/Library/x/ledger.sqlite"'
                      ", but the task says paths start with /Users/dana/")]
    tilde = format_rules.problems(MAC, _mac(log_file="~/Library/Logs/Tally/tally.log"))
    assert len(tilde) == 1 and "no `~`" in tilde[0]


def test_lines_over_the_stated_length_are_named() -> None:
    assert format_rules.rules(COMMIT, ["commit_message.txt"]).line_max == 72
    long = "fix(invoicing): ensure due dates fall on weekdays\n\n" + "x" * 106 + "\n\nRefs: B-1\n"
    found = format_rules.problems(COMMIT, {"commit_message.txt": long})
    assert len(found) == 1 and found[0].startswith(
        "commit_message.txt line 3 is 106 characters; the task says at most 72: rewrap it")
    short = "fix(invoicing): ensure due dates fall on weekdays\n\nbody\n\nRefs: B-1\n"
    assert format_rules.problems(COMMIT, {"commit_message.txt": short}) == []
    # no line rule: a JSON deliverable, or "at most N characters" not said of lines
    assert format_rules.rules(COMMIT, ["out.json"]).line_max is None
    assert format_rules.rules("Write `a.txt`: a title of at most 10 characters.",
                              ["a.txt"]).line_max is None


def test_list_of_object_templates_check_each_items_shape() -> None:
    shapes = format_rules.rules(REVIEW, ["review.json"]).lists["review.json"]
    assert shapes["lines"].kind == "list" and shapes["lines"].fixed
    good = {"file": "a.py", "lines": [3, 4], "category": "off-by-one", "explanation": "x"}
    assert format_rules.problems(REVIEW, {"review.json": json.dumps([good])}) == []
    bad = [good, {**good, "lines": [None, None]}, {**good, "lines": ["3", 4]},
           {**good, "lines": [3]}, {**good, "explanation": "y" * 201},
           {k: v for k, v in good.items() if k != "category"}, "just text"]
    found = format_rules.problems(REVIEW, {"review.json": json.dumps(bad)})
    assert found == [
        'review.json item 1 "lines" item 0 is null: the task\'s template says [<first>, <last>]',
        ('review.json item 2 "lines" item 0 is "3", not a number: the task\'s template says '
        "[<first>, <last>]"),
        ('review.json item 3 "lines" has 1 item(s), not 2: the task\'s template says '
        "[<first>, <last>]"),
        ('review.json item 4 "explanation" is 201 characters (at most 200): the task\'s template '
        'says "<one line, at most 200 characters>"'),
        'review.json item 5 lacks "category"',
        ('review.json item 6 is "just text", not an object {"file": "<path relative to the project '
        'root>", "lines": [<first>, <last>], "category": "<category>", "explanation": "<one line, '
        'at most 200 characters>"}')]
    assert format_rules.problems(SECURITY, {"security_report.json": json.dumps(
        [{"file": "a.py", "line": None, "category": "xss"}])}) == [
        'security_report.json item 0 "line" is null: the task\'s template says <int>']


def test_object_templates_and_untemplated_tasks_are_left_alone() -> None:
    """A "nothing listens" answer is `{"pid": null, "command": null}`."""
    task = ("Write `answers/port.json`, one JSON object with exactly the keys `pid` (a JSON number) "
            "and `command`, or both `null` if nothing listens.")
    found = format_rules.rules(task, ["answers/port.json"])
    assert found.count == 0
    assert format_rules.problems(task, {"answers/port.json": '{"pid": null}'}) == []


def test_the_switch_turns_the_checks_off(monkeypatch) -> None:
    rooted = _mac(database="root/Users/dana/x.sqlite")
    assert format_rules.problems(MAC, rooted)
    monkeypatch.setenv(features.ENV, "format_rules")
    assert format_rules.problems(MAC, rooted) == []
