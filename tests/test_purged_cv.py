"""PurgedKFold leakage controls + triple-barrier labeling (ml.cv, ml.labels)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from ml.cv import PurgedKFold
from ml.labels import triple_barrier_labels, get_daily_vol


# --- helpers ---------------------------------------------------------------

def _t1(n, horizon=5):
    """n observations, each labeled over the next `horizon` bars (overlapping)."""
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    ends = [idx[min(i + horizon, n - 1)] for i in range(n)]
    return pd.Series(ends, index=idx)


# --- PurgedKFold ----------------------------------------------------------

def test_train_and_test_are_disjoint():
    t1 = _t1(120, horizon=5)
    X = pd.DataFrame({"f": np.arange(120)}, index=t1.index)
    for tr, te in PurgedKFold(n_splits=5, t1=t1).split(X):
        assert set(tr).isdisjoint(set(te))


def test_every_observation_tested_exactly_once():
    t1 = _t1(120, horizon=5)
    X = pd.DataFrame({"f": np.arange(120)}, index=t1.index)
    seen = np.concatenate([te for _, te in PurgedKFold(n_splits=6, t1=t1).split(X)])
    assert sorted(seen.tolist()) == list(range(120))


def test_no_training_label_overlaps_the_test_span():
    # the core guarantee: purge removes any train row whose [t0,t1] touches the test span
    n, horizon = 200, 8
    t1 = _t1(n, horizon=horizon)
    X = pd.DataFrame({"f": np.arange(n)}, index=t1.index)
    t0 = t1.index
    for tr, te in PurgedKFold(n_splits=5, t1=t1).split(X):
        test_t0, test_t1 = t0[te[0]], t1.iloc[te].max()
        for i in tr:
            # train row must end before the test starts OR begin after the test ends
            assert (t1.iloc[i] < test_t0) or (t0[i] > test_t1)


def test_embargo_drops_the_buffer_after_each_fold():
    n = 200
    t1 = _t1(n, horizon=3)
    X = pd.DataFrame({"f": np.arange(n)}, index=t1.index)
    embargo = int(n * 0.05)               # 10 rows
    splits = list(PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.05).split(X))
    for tr, te in splits[:-1]:            # last fold has no rows after it to embargo
        i1 = int(te[-1])
        buffer = set(range(i1 + 1, min(i1 + 1 + embargo, n)))
        assert buffer.isdisjoint(set(tr))


def test_embargo_zero_keeps_more_rows_than_positive_embargo():
    t1 = _t1(200, horizon=5)
    X = pd.DataFrame({"f": np.arange(200)}, index=t1.index)
    n0 = sum(len(tr) for tr, _ in PurgedKFold(5, t1=t1, pct_embargo=0.0).split(X))
    n1 = sum(len(tr) for tr, _ in PurgedKFold(5, t1=t1, pct_embargo=0.05).split(X))
    assert n1 < n0


def test_requires_t1():
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=5)


# --- triple-barrier labels ------------------------------------------------

def _series(vals):
    return pd.Series(vals, index=pd.date_range("2020-01-01", periods=len(vals), freq="B"))


def test_rising_series_hits_upper_barrier():
    px = _series([100 * (1.01 ** i) for i in range(15)])      # +1%/day
    vol = _series([0.005] * 15)                                # barrier at +/-0.5%
    lab = triple_barrier_labels(px, horizon=5, pt=1.0, sl=1.0, vol=vol)
    assert (lab["label"].iloc[:-1] == 1).all()                 # last bar has no forward window
    assert (lab["t1"] >= lab.index).all()                      # label end is forward in time


def test_falling_series_hits_lower_barrier():
    px = _series([100 * (0.99 ** i) for i in range(15)])
    vol = _series([0.005] * 15)
    lab = triple_barrier_labels(px, horizon=5, pt=1.0, sl=1.0, vol=vol)
    assert (lab["label"].iloc[:-1] == -1).all()                # last bar has no forward window


def test_flat_series_times_out_at_vertical_barrier():
    px = _series([100.0] * 15)
    vol = _series([0.01] * 15)
    lab = triple_barrier_labels(px, horizon=5, pt=1.0, sl=1.0, vol=vol)
    assert (lab["label"] == 0).all()


def test_zero_vol_rows_are_skipped():
    px = _series([100, 101, 102, 103, 104, 105])
    vol = _series([0.0, 0.0, 0.01, 0.01, 0.01, 0.01])
    lab = triple_barrier_labels(px, horizon=2, pt=1.0, sl=1.0, vol=vol)
    assert lab.index[0] == px.index[2]                         # first two (vol=0) dropped


def test_get_daily_vol_is_positive_and_aligned():
    rng = np.random.default_rng(0)
    px = _series(100 * np.cumprod(1 + rng.normal(0, 0.01, 300)))
    v = get_daily_vol(px, span=20).dropna()
    assert (v > 0).all() and len(v) > 250
