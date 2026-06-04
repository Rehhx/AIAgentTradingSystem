"""Probability of Backtest Overfitting via CSCV (analytics.pbo)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from analytics.pbo import cscv_pbo


def test_pbo_low_when_one_strategy_truly_best():
    # strategy 0 has a real, persistent edge -> IS winner stays OOS winner -> PBO ~ 0
    rng = np.random.default_rng(0)
    M = rng.normal(0, 0.01, (2000, 8))
    M[:, 0] += 0.002                                  # +0.2%/day genuine alpha
    out = cscv_pbo(M, n_splits=10)
    assert out["pbo"] < 0.2
    assert out["logits_median"] > 0                  # IS winner ranks high OOS


def test_pbo_about_half_on_pure_noise():
    # all strategies iid noise -> IS rank says nothing about OOS rank -> PBO ~ 0.5
    rng = np.random.default_rng(1)
    M = rng.normal(0, 0.01, (2000, 10))
    out = cscv_pbo(M, n_splits=10)
    assert 0.3 <= out["pbo"] <= 0.7


def test_pbo_validation():
    with pytest.raises(ValueError):
        cscv_pbo(np.zeros((100, 1)))                 # need >= 2 strategies
    with pytest.raises(ValueError):
        cscv_pbo(np.zeros((100, 3)), n_splits=7)     # odd split count
    with pytest.raises(ValueError):
        cscv_pbo(np.zeros((10, 3)), n_splits=10)     # too few observations


def test_pbo_reports_combination_count():
    rng = np.random.default_rng(2)
    M = rng.normal(0, 0.01, (1000, 6))
    out = cscv_pbo(M, n_splits=10)
    # C(10,5) = 252 symmetric splits
    assert out["n_combinations"] == 252
    assert out["n_splits"] == 10
