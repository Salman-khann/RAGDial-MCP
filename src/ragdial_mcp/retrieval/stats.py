"""Small-sample statistics for benchmark results.

Golden sets are usually small (the bundled one has 30 questions), so a 0.01
difference in mean nDCG is often one question moving one rank. These helpers
put error bars on means and paired differences, so a "winner" is only called
when the data supports it. Everything is deterministic (fixed seed).
"""

from __future__ import annotations

from math import comb

import numpy as np

N_BOOT = 2000
SEED = 7


def mean_ci(values: list[float], level: float = 0.95) -> tuple[float, float]:
    """Percentile-bootstrap confidence interval for the mean."""
    x = np.asarray(values, dtype=float)
    if x.size < 2:
        m = float(x.mean()) if x.size else 0.0
        return m, m
    rng = np.random.default_rng(SEED)
    means = x[rng.integers(0, x.size, size=(N_BOOT, x.size))].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def paired_diff_ci(a: list[float], b: list[float], level: float = 0.95) -> tuple[float, float]:
    """Bootstrap CI for mean(b - a) over the same questions."""
    return mean_ci([y - x for x, y in zip(a, b)], level)


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test: how surprising is ``wins`` vs ``losses`` if there were no real difference?"""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return min(1.0, p)
