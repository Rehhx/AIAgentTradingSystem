"""
analytics/reality_check.py
--------------------------
Data-snooping tests: is the BEST strategy's outperformance real, or just the
luckiest of many tried? White's Reality Check (2000) and Hansen's SPA (2005)
give a p-value for the null "no strategy beats the benchmark" while accounting
for the number of strategies searched, using a stationary block bootstrap
(Politis & Romano 1994) to preserve serial correlation.

  p small  -> the best strategy's edge survives data-snooping
  p large  -> the apparent edge is consistent with luck across many trials

Hansen's SPA studentizes each strategy by its own standard error and recenters
poor strategies, giving more power than White's RC when the search included many
weak candidates (which it always does).
"""
from __future__ import annotations

import numpy as np


def _stationary_bootstrap_indices(T: int, n_boot: int, avg_block: int, rng) -> np.ndarray:
    """Politis-Romano stationary bootstrap: returns a (T x n_boot) index matrix.
    Each column resamples rows in geometrically-distributed blocks (mean length
    avg_block) wrapping circularly, preserving autocorrelation up to ~avg_block."""
    p = 1.0 / max(1, avg_block)
    idx = np.empty((T, n_boot), dtype=np.int64)
    idx[0] = rng.integers(0, T, size=n_boot)
    for t in range(1, T):
        cont = rng.random(n_boot) >= p                 # continue current block?
        nxt = (idx[t - 1] + 1) % T                     # ... then take the next row
        rnd = rng.integers(0, T, size=n_boot)          # ... else jump to a random row
        idx[t] = np.where(cont, nxt, rnd)
    return idx


def whites_reality_check(returns_matrix, benchmark=None, n_boot: int = 2000,
                         avg_block: int = 10, seed: int = 0) -> dict:
    """White's Reality Check.

    returns_matrix : (T x N) strategy returns
    benchmark      : (T,) benchmark returns, or None for a zero benchmark
    Null: max_k E[strategy_k - benchmark] <= 0.
    """
    M = np.asarray(returns_matrix, dtype=float)
    T, N = M.shape
    b = np.zeros(T) if benchmark is None else np.asarray(benchmark, dtype=float)
    f = M - b[:, None]                                  # performance vs benchmark
    fbar = f.mean(axis=0)
    stat = np.sqrt(T) * np.max(fbar)

    rng = np.random.default_rng(seed)
    idx = _stationary_bootstrap_indices(T, n_boot, avg_block, rng)
    boot = np.empty(n_boot)
    for j in range(n_boot):
        fb = f[idx[:, j]].mean(axis=0)
        boot[j] = np.sqrt(T) * np.max(fb - fbar)        # recentered (null-imposed)
    return {"statistic": float(stat), "p_value": float(np.mean(boot >= stat)),
            "n_strategies": int(N), "best_mean_excess": float(np.max(fbar))}


def hansen_spa(returns_matrix, benchmark=None, n_boot: int = 2000,
               avg_block: int = 10, seed: int = 0) -> dict:
    """Hansen's SPA test (consistent variant): studentized + recentered RC."""
    M = np.asarray(returns_matrix, dtype=float)
    T, N = M.shape
    b = np.zeros(T) if benchmark is None else np.asarray(benchmark, dtype=float)
    f = M - b[:, None]
    fbar = f.mean(axis=0)
    sigma = f.std(axis=0, ddof=1)
    sigma = np.where(sigma == 0, np.nan, sigma)
    stat = float(np.nanmax(np.sqrt(T) * fbar / sigma))

    # consistent recentering: a strategy too far below zero is set to 0 (excluded)
    keep = fbar >= -np.sqrt((sigma ** 2 / T) * 2 * np.log(np.log(max(T, 3))))
    g = np.where(keep, fbar, 0.0)

    rng = np.random.default_rng(seed)
    idx = _stationary_bootstrap_indices(T, n_boot, avg_block, rng)
    boot = np.empty(n_boot)
    for j in range(n_boot):
        fb = f[idx[:, j]].mean(axis=0)
        boot[j] = np.nanmax(np.sqrt(T) * (fb - g) / sigma)
    return {"statistic": stat, "p_value": float(np.mean(boot >= stat)),
            "n_strategies": int(N)}
