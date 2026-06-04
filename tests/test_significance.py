"""Deflated/Probabilistic Sharpe + min track-record length (analytics.significance)."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from analytics.significance import (
    norm_cdf, norm_ppf, sharpe_stats, probabilistic_sharpe_ratio,
    expected_max_sharpe, deflated_sharpe_ratio, min_track_record_length,
    dsr_from_trials,
)


def test_norm_ppf_cdf_roundtrip():
    for p in [0.01, 0.1, 0.5, 0.9, 0.975, 0.99]:
        assert abs(norm_cdf(norm_ppf(p)) - p) < 1e-6


def test_norm_ppf_known_values():
    assert abs(norm_ppf(0.975) - 1.959963985) < 1e-5      # the 97.5% z
    assert abs(norm_ppf(0.5)) < 1e-9


def test_psr_is_half_at_benchmark():
    # observed SR exactly equals the benchmark -> coin flip
    assert abs(probabilistic_sharpe_ratio(0.1, 1000, 0.0, 3.0, 0.1) - 0.5) < 1e-9


def test_psr_increases_with_observations():
    a = probabilistic_sharpe_ratio(0.1, 100, 0.0, 3.0, 0.0)
    b = probabilistic_sharpe_ratio(0.1, 1000, 0.0, 3.0, 0.0)
    assert b > a > 0.5


def test_negative_skew_fat_tails_lower_psr():
    base = probabilistic_sharpe_ratio(0.1, 500, 0.0, 3.0, 0.0)
    bad = probabilistic_sharpe_ratio(0.1, 500, -1.0, 8.0, 0.0)   # left-skew, fat tails
    assert bad < base


def test_expected_max_sharpe_monotone_in_trials():
    v = 0.01
    assert expected_max_sharpe(1, v) == 0.0          # one trial -> no selection bias
    assert expected_max_sharpe(100, v) > expected_max_sharpe(10, v) > 0.0


def test_expected_max_sharpe_zero_variance():
    assert expected_max_sharpe(50, 0.0) == 0.0       # identical strategies -> no spread


def test_deflation_never_raises_psr():
    sr, n, sk, ku = 0.08, 1000, 0.0, 3.0
    d = deflated_sharpe_ratio(sr, n, sk, ku, n_trials=20, sr_variance=0.001)
    assert d["sr_star"] > 0
    assert d["dsr"] <= d["psr_vs_zero"] + 1e-12      # the hurdle can only hurt


def test_mintrl_decreases_with_sharpe():
    a = min_track_record_length(0.05, 0.0, 3.0, 0.0, 0.95)
    b = min_track_record_length(0.10, 0.0, 3.0, 0.0, 0.95)
    assert b < a
    assert math.isinf(min_track_record_length(-0.1, 0.0, 3.0, 0.0))   # no edge -> never


def test_dsr_from_trials_bounds():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0008, 0.01, 1200)               # ~1.3 annualized Sharpe
    trials = rng.normal(0.0, 0.03, 20)               # 20 trials' per-period SRs
    out = dsr_from_trials(r, list(trials), periods=252)
    assert 0.0 <= out["dsr"] <= 1.0
    assert out["dsr"] <= out["psr_vs_zero"] + 1e-9
    assert out["n_trials"] == 20
    assert out["sr_star_period"] > 0


def test_sharpe_stats_on_normal_returns():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, 5000)
    s = sharpe_stats(r)
    assert abs(s["skew"]) < 0.2
    assert abs(s["kurt"] - 3.0) < 0.3                # non-excess kurtosis ~ 3
    assert s["n"] == 5000


def test_sharpe_stats_degenerate():
    assert sharpe_stats([0.0, 0.0, 0.0])["sr"] == 0.0
    assert sharpe_stats([0.01])["sr"] == 0.0          # too few obs
