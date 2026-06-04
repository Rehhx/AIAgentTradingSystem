"""
ml/cv.py
--------
PurgedKFold — K-fold cross-validation for time series with OVERLAPPING labels
(Lopez de Prado, "Advances in Financial Machine Learning", ch. 7).

Each observation i has a decision time t0 (its index) and a label-end time t1 (how
far forward its label looked). Ordinary K-fold leaks the future: a training row
whose label window [t0, t1] overlaps the test fold's time span shares information
with the test labels. PurgedKFold fixes this:

  * folds are CONTIGUOUS in time (no shuffling),
  * training rows whose label window overlaps the test span are PURGED,
  * an EMBARGO drops a buffer of rows immediately after the test fold, killing
    leakage that flows through serial correlation.

The result is an honest generalization estimate — the input the Tier-1A deflated
Sharpe needs to mean anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class PurgedKFold:
    def __init__(self, n_splits: int = 5, t1: pd.Series | None = None,
                 pct_embargo: float = 0.0):
        if t1 is None:
            raise ValueError("t1 (label-end times, indexed like X) is required")
        if not isinstance(t1, pd.Series):
            raise TypeError("t1 must be a pd.Series of label-end times")
        self.n_splits = int(n_splits)
        self.t1 = t1
        self.pct_embargo = float(pct_embargo)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(self.t1)
        if len(X) != n:
            raise ValueError("X and t1 must align (same length)")
        idx = np.arange(n)
        embargo = int(n * self.pct_embargo)
        t0_times = self.t1.index.values          # decision times (sorted ascending)
        t1_times = self.t1.values                # label-end times

        for test_idx in np.array_split(idx, self.n_splits):
            i0, i1 = int(test_idx[0]), int(test_idx[-1])
            test_t0 = t0_times[i0]                # first decision time in the fold
            test_t1 = t1_times[test_idx].max()    # last label-end time in the fold

            # keep training rows that do NOT overlap the test span:
            #   label finished before the test began, OR decided after test labels ended
            train_mask = (t1_times < test_t0) | (t0_times > test_t1)

            # embargo: drop the `embargo` rows immediately after the test block
            if embargo > 0:
                lo, hi = min(i1 + 1, n), min(i1 + 1 + embargo, n)
                train_mask[lo:hi] = False

            train_idx = idx[train_mask]
            yield train_idx, test_idx
