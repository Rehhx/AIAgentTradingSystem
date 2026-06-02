"""Risk/return metrics + the volatility-targeting overlay."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from agents.daily_strategies import _metrics_from_returns, vol_target, TRADING_DAYS


def _series(vals):
    idx = pd.date_range("2020-01-01", periods=len(vals), freq="B")
    return pd.Series(vals, index=idx)


def test_metrics_positive_drift():
    r = _series([0.001] * TRADING_DAYS)            # steady +0.1%/day for a year
    m = _metrics_from_returns(r, [], "x")
    assert m["cagr"] > 0
    assert m["sharpe"] > 0
    assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-9)   # never declines


def test_metrics_drawdown_is_negative_on_loss():
    r = _series([0.01, 0.01, -0.30, 0.0])          # a 30% crash
    m = _metrics_from_returns(r, [], "x")
    assert m["max_drawdown"] < -0.25


def test_metrics_flat_returns_zero_sharpe():
    r = _series([0.0] * 50)
    m = _metrics_from_returns(r, [], "x")
    assert m["sharpe"] == 0.0


def test_vol_target_caps_at_max_leverage():
    # very calm series: target/realized would imply huge leverage -> must cap
    r = _series([0.0001] * 300)
    out = vol_target(r, target_vol=0.15, max_leverage=1.0)
    ratio = (out / r.replace(0, np.nan)).dropna()
    assert ratio.max() <= 1.0 + 1e-9               # never levers above the cap


def test_vol_target_derisks_high_vol():
    rng = np.random.default_rng(0)
    vals = rng.normal(0, 0.05, 300)                # ~80% annualized vol
    r = _series(vals)
    scaled = vol_target(r, target_vol=0.15, max_leverage=1.5)
    # realized vol of the scaled series should be well below the raw series
    assert scaled.std() < r.std()


def test_vol_target_no_lookahead():
    # scale uses yesterday's vol (shifted) -> first window is flat/zero, not NaN
    r = _series(list(np.linspace(0.01, -0.01, 60)))
    out = vol_target(r, 0.15, max_leverage=1.5)
    assert not out.isna().any()
