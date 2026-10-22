"""Small, dependency-free statistics for eval results: how sure are we about a pass rate or a gap?"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

Z95 = 1.959964


def wilson(passed: int, total: int, z: float = Z95) -> tuple[float, float]:
    """95% Wilson score interval for a pass rate; well-behaved at small n and at 0% or 100%."""
    if total <= 0:
        return 0.0, 1.0
    p = passed / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value from discordant pairs (only A passed / only B passed)."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_diff_ci(
    pairs_by_case: Sequence[Sequence[tuple[bool, bool]]], iterations: int = 2000, seed: int = 7,
) -> tuple[float, float]:
    """95% interval for pass-rate(B) - pass-rate(A), resampling whole cases (attempts on one case
    are correlated, so resampling attempts would overstate certainty)."""
    cases = [list(pairs) for pairs in pairs_by_case if pairs]
    if not cases:
        return 0.0, 0.0
    rng = random.Random(seed)
    diffs = []
    for _ in range(iterations):
        sample = [cases[rng.randrange(len(cases))] for _ in cases]
        pairs = [pair for case in sample for pair in case]
        diffs.append(sum(b for _, b in pairs) / len(pairs) - sum(a for a, _ in pairs) / len(pairs))
    diffs.sort()
    return diffs[int(0.025 * iterations)], diffs[int(0.975 * iterations) - 1]


def bootstrap_mean_diff_ci(
    pairs_by_case: Sequence[Sequence[tuple[float, float]]], iterations: int = 2000, seed: int = 7,
) -> tuple[float, float, float]:
    """(mean B − A, 95% interval) of a per-attempt score in [0, 1] (e.g. the share of hidden tests
    passed), resampling whole cases like bootstrap_diff_ci."""
    cases = [list(pairs) for pairs in pairs_by_case if pairs]
    if not cases:
        return 0.0, 0.0, 0.0
    flat = [pair for case in cases for pair in case]
    mean = sum(b - a for a, b in flat) / len(flat)
    rng = random.Random(seed)
    diffs = []
    for _ in range(iterations):
        pairs = [pair for _ in cases for pair in cases[rng.randrange(len(cases))]]
        diffs.append(sum(b - a for a, b in pairs) / len(pairs))
    diffs.sort()
    return mean, diffs[int(0.025 * iterations)], diffs[int(0.975 * iterations) - 1]


def pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def rate_text(passed: int, total: int) -> str:
    """"10/60 (17%, 95% CI 9–28%)"."""
    if not total:
        return "0/0"
    low, high = wilson(passed, total)
    return f"{passed}/{total} ({pct(passed / total)}, 95% CI {pct(low)}–{pct(high)})"
