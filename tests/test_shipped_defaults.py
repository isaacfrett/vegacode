"""Guard: the suite tests what ships (systems review 2026-10-17, recommendation #1).

Until 2026-10-17 tests/conftest.py switched 9 default features off for every test (BASE_OFF), so
~78 tests only passed in a configuration no user runs, and interactions between the defaults went
untested. conftest now leaves every default on; a test that is not about a default turns that one
feature off itself (tests/switches.py) with a comment saying why. These tests fail if conftest
turns a default off (or an opt-in on) for the whole suite again.
"""

from __future__ import annotations

import ast
from pathlib import Path

from vega import features

# A default feature conftest may turn off for every test, with the written reason. Keep it empty:
# prefer switching the feature off in the tests that need it.
ALLOWED_OFF: dict[str, str] = {}

CONFTEST = Path(__file__).with_name("conftest.py")
_SWITCH_NAMES = {"VEGA_FEATURES_OFF", "VEGA_FEATURES_ON"}
_SWITCH_ATTRS = {"ENV", "ENV_ON"}  # features.ENV / features.ENV_ON


def test_every_default_feature_is_on_in_a_plain_test() -> None:
    off = features.disabled()
    assert not off - ALLOWED_OFF.keys(), (
        f"conftest turns default feature(s) off for every test: {sorted(off - ALLOWED_OFF.keys())}. "
        "Switch them off in the tests that need it (tests/switches.py), or add them to "
        "ALLOWED_OFF with the reason.")
    assert all(reason.strip() for reason in ALLOWED_OFF.values()), "every allowance needs a reason"
    assert all(features.on(name) for name in features.FEATURES if name not in ALLOWED_OFF)


def test_no_opt_in_feature_is_on_in_a_plain_test() -> None:
    assert features.enabled() == frozenset(), "conftest must not turn opt-in features on globally"


def _names_a_switch(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return node.value in _SWITCH_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr in _SWITCH_ATTRS
    if isinstance(node, ast.Name):
        return node.id in _SWITCH_ATTRS
    return False


def test_conftest_never_sets_a_feature_switch() -> None:
    """conftest may clear the switches (delenv) but never set them, nor patch vega.features."""
    tree = ast.parse(CONFTEST.read_text(encoding="utf-8"))
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            name = node.func.attr
            if name == "setenv" and node.args and _names_a_switch(node.args[0]):
                problems.append(f"line {node.lineno}: setenv of a feature switch")
            if name == "setattr" and node.args and (
                    (isinstance(node.args[0], ast.Name) and node.args[0].id == "features")
                    or (isinstance(node.args[0], ast.Constant)
                        and str(node.args[0].value).startswith("vega.features"))):
                problems.append(f"line {node.lineno}: patches vega.features")
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and _names_a_switch(target.slice):
                    problems.append(f"line {node.lineno}: assigns a feature switch")
    assert not problems, problems
