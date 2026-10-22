"""Text checks in vega.evals._check: response_contains (substring) and response_matches (regex)."""

from __future__ import annotations

from vega import evals


def test_response_matches_all_patterns_case_insensitive(tmp_path):
    case = {"files": {}, "response_matches": [r"\bORBIT_CACHE_HOME\b", r"(?<![\d.])45(?!\.?\d)"]}
    assert evals._check(case, tmp_path, "orbit_cache_home, 45 seconds.") == ""
    assert "does not match" in evals._check(case, tmp_path, "ORBIT_CACHE_HOME, 450 ms")
    assert "does not match" in evals._check(case, tmp_path, "45 seconds")


def test_response_contains_still_substring(tmp_path):
    case = {"files": {}, "response_contains": ["md5"]}
    assert evals._check(case, tmp_path, "It uses hashlib.MD5.") == ""
    assert evals._check(case, tmp_path, "sha1") == "response missing 'md5'"
