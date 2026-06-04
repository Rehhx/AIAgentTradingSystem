"""
analytics/pbo.py
----------------
Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric
Cross-Validation (CSCV). Bailey, Borwein, Lopez de Prado & Zhu (2015),
"The Probability of Backtest Overfitting", J. Computational Finance.

Idea: line up the return series of EVERY strategy you tried as columns of a
matrix M (T observations x N strategies). Chop the T rows into S equal
time-blocks. For every way of choosing S/2 blocks as the in-sample (IS) set (the
rest are out-of-sample, OOS):
  - pick the strategy that's best IS,
  - see where it ranks OOS.
If the IS winner is consistently a middling-or-worse OOS performer, your
selection process is overfitting.

  PBO = fraction of splits where the IS-best strategy lands in the BOTTOM HALF OOS

  PBO ~ 0.0   selection generalizes (the IS winner stays a winner OOS)
  PBO ~ 0.5   pure noise (IS rank tells you nothing about OOS rank)
  PBO ~ 1.0   systematic overfitting (IS winner is reliably an OOS loser)
"""
from __future__ import annotations

import itertools

import numpy as np


def _sharpe(block: np.ndarray) -> np.ndarray:
    """per-column (per-strategy) Sharpe of a rows x strategies block. Annualization
    cancels in the cross-strategy ranking, so it is omitted."""
    mu = block.mean(axis=0)
    sd = block.std(axis=0, ddof=1)
    sd = np.where(sd == 0, np.nan, sd)
    return mu / sd


def cscv_pbo(returns_matrix, n_splits: int = 16, metric=None) -> dict:
    """Combinatorially Symmetric Cross-Validation PBO.

    returns_matrix : (T observations x N strategies) array-like, N >= 2
    n_splits       : number of contiguous time-blocks S (must be even)
    metric         : block -> per-strategy score (default: Sharpe)
    """
    M = np.asarray(returns_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        raise ValueError("returns_matrix must be (T observations x N strategies), N>=2")
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even")
    T, N = M.shape
    if T < n_splits * 2:
        raise ValueError(f"need at least {n_splits*2} observations for {n_splits} splits")
    metric = metric or _sharpe

    rows = (T // n_splits) * n_splits                      # trim to a multiple of S
    blocks = np.array_split(M[:rows], n_splits, axis=0)
    idx = list(range(n_splits))

    logits, n_oos_loss, n_total = [], 0, 0
    for combo in itertools.combinations(idx, n_splits // 2):
        cset = set(combo)
        is_rows = np.concatenate([blocks[i] for i in combo], axis=0)
        oos_rows = np.concatenate([blocks[i] for i in idx if i not in cset], axis=0)
        is_perf = metric(is_rows)
        oos_perf = metric(oos_rows)
        if np.all(np.isnan(is_perf)):
            continue
        n_star = int(np.nanargmax(is_perf))                # best strategy in-sample
        # OOS relative rank of that strategy: 0 = worst .. N-1 = best
        ranks = np.argsort(np.argsort(np.nan_to_num(oos_perf, nan=-np.inf)))
        omega = (ranks[n_star] + 1) / (N + 1)              # in (0, 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        lam = np.log(omega / (1 - omega))                  # logit of the OOS rank
        logits.append(lam)
        if lam <= 0:                                       # IS winner in bottom half OOS
            n_oos_loss += 1
        n_total += 1

    logits = np.asarray(logits)
    return {
        "pbo": (n_oos_loss / n_total) if n_total else float("nan"),
        "n_splits": n_splits,
        "n_combinations": n_total,
        "logits_mean": float(np.mean(logits)) if len(logits) else float("nan"),
        "logits_median": float(np.median(logits)) if len(logits) else float("nan"),
    }
