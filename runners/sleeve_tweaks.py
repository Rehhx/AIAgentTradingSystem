"""
runners/sleeve_tweaks.py
------------------------
No-leverage attempts to turn existing sleeves into stronger ones:
  1. recovery_xs  -- run the high-quality recovery sleeve cross-sectionally on the
     FULL S&P 500 (it currently only runs on the quality-10, so it rarely fires).
  2. xs concentration -- does top-5 momentum beat top-10 robustly (walk-forward)?
Each is judged the usual way: standalone Sharpe/CAGR/DD + marginal effect on the
deployed book at a sensible weight. $100k, 6 bps, adjusted data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    sig_recovery, backtest_cross_sectional, _metrics_from_returns,
    walk_forward_folds, split_metrics, SIDE_COST,
)
from data.sp500 import sp500_tickers, load_daily
from runners.diversifier_screen import build_base, overlays, W
from runners.final_tests import lowvol_factor
from runners.lowvol_defensive import make_defensive


def recovery_xs(hold=120, cap=0.10):
    """cross-sectional recovery on the full S&P 500: hold every name currently in
    its recovery window, equal weight (1/active, capped), cash when none active."""
    data = load_daily(sp500_tickers(), start="2016-01-01")
    sigs, rets = {}, {}
    for t, d in data.items():
        if len(d) < 260:
            continue
        sigs[t] = sig_recovery(d, {"hold_days": hold})
        rets[t] = d["close"].pct_change()
    sig_df = pd.DataFrame(sigs).fillna(0.0)
    ret_df = pd.DataFrame(rets).reindex(columns=sig_df.columns).reindex(sig_df.index).fillna(0.0)
    active = sig_df.sum(axis=1)
    w = sig_df.div(active.where(active > 0, 1.0), axis=0).clip(upper=cap)
    port = (w.shift(1) * ret_df).sum(axis=1)
    turn = w.diff().abs().sum(axis=1).fillna(0.0)
    return (port - turn * SIDE_COST).fillna(0.0)


def stat(label, r):
    m = _metrics_from_returns(r, [], label)
    pos = sum(1 for f in walk_forward_folds(r, 5) if f["sharpe"] > 0)
    print(f"  {label:30s} Sharpe {m['sharpe']:5.2f} | CAGR {m['cagr']:6.1%} | DD {m['max_drawdown']:6.1%} | WF {pos}/5")
    return r


def main():
    print("building book + sleeve tweaks (scans S&P 500; ~1-2 min) ...\n")
    panel = build_base()
    idx = panel.index
    lvd = make_defensive(lowvol_factor()).reindex(idx).fillna(0)
    base = sum(panel[c].fillna(0) * W[c] for c in W) * 0.90 + lvd * 0.10
    bm = _metrics_from_returns(overlays(base, idx), [], "base")
    print(f"deployed book: Sharpe {bm['sharpe']:.2f} | CAGR {bm['cagr']:.1%} | DD {bm['max_drawdown']:.1%}\n")

    print("1) RECOVERY on full S&P 500 vs quality-10:")
    rec10 = stat("recovery (quality-10)", panel["rec"].fillna(0))
    rec500 = stat("recovery_xs (full 500)", recovery_xs())
    corr = float(panel["rec"].corr(rec500.reindex(idx)))
    for wt in (0.08, 0.12):
        b2 = overlays(base * (1 - wt) + rec500.reindex(idx).fillna(0) * wt, idx)
        m2 = _metrics_from_returns(b2, [], "x")
        print(f"     + recovery_xs @ {wt:.0%}: book Sharpe {m2['sharpe']:.2f} ({m2['sharpe']-bm['sharpe']:+.2f}) "
              f"| CAGR {m2['cagr']:.1%} | DD {m2['max_drawdown']:.1%}  (corr to rec10 {corr:.2f})")

    print("\n2) MOMENTUM concentration (xs_dualmom top-k):")
    sp = sp500_tickers()
    for k in (5, 10, 15):
        r = backtest_cross_sectional(sp, mode="momentum", lookback=252, skip=21, k=k, market_filter=True)["_returns"]
        stat(f"xs_dualmom top-{k}", r)


if __name__ == "__main__":
    main()
