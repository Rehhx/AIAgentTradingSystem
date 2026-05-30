"""
runners/final_tests.py
----------------------
The last two genuinely-different return sources, to exhaust the discovery space:
  - crypto_trend : absolute-momentum (long/flat) on BTC + ETH
  - lowvol_factor: cross-sectional low-volatility anomaly (hold the 30 lowest-vol
                   S&P 500 names, monthly rebalance, equal weight)
Same bar as every other candidate: standalone metrics + correlation to the
deployed book + marginal effect when added at 12%.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    backtest_book, _metrics_from_returns, sig_abs_momentum,
    walk_forward_folds, split_metrics, SIDE_COST,
)
from runners.diversifier_screen import build_base, overlays, W


def crypto_trend():
    return backtest_book(sig_abs_momentum, ["BTC-USD", "ETH-USD"], {"lookback": 126}, label="crypto")["_returns"]


def lowvol_factor():
    from data.sp500 import sp500_tickers, load_daily
    data = load_daily(sp500_tickers(), start="2016-01-01")
    closes = pd.DataFrame({t: d["close"] for t, d in data.items()}).sort_index()
    rets = closes.pct_change()
    vol = rets.rolling(60).std()
    k = 30
    rb = list(closes.index[60::21])
    w = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
    for dt in rb:
        v = vol.loc[dt].dropna()
        if len(v) < k:
            continue
        row = pd.Series(0.0, index=closes.columns)
        row[v.nsmallest(k).index] = 1.0 / k
        w.loc[dt] = row
    w = w.ffill().fillna(0.0)
    port = (w.shift(1) * rets).sum(axis=1)
    turn = w.diff().abs().sum(axis=1).fillna(0.0)
    return (port - turn * SIDE_COST).fillna(0.0)


def score(name, r, base_combo, bm, index):
    m = _metrics_from_returns(r, [], name)
    pos = sum(1 for f in walk_forward_folds(r, 5) if f["sharpe"] > 0)
    rr = r.reindex(index)
    corr = float(base_combo.corr(rr))
    combo2 = base_combo * 0.88 + rr.fillna(0) * 0.12
    m2 = _metrics_from_returns(overlays(combo2, index), [], name)
    d_sh = m2["sharpe"] - bm["sharpe"]
    d_dd = m2["max_drawdown"] - bm["max_drawdown"]
    verdict = "ADD" if (d_sh >= 0.02 and m2["cagr"] >= bm["cagr"] - 0.005) else "reject"
    print(f"{name:20s} Sharpe {m['sharpe']:5.2f} | CAGR {m['cagr']:6.1%} | DD {m['max_drawdown']:6.1%} "
          f"| WF {pos}/5 | corr {corr:+.2f}")
    print(f"{'':20s} +book@12%: Sharpe {m2['sharpe']:.2f} ({d_sh:+.2f}) | CAGR {m2['cagr']:.1%} "
          f"| DD {m2['max_drawdown']:.1%} ({d_dd:+.1%}) -> {verdict}\n")


def main():
    print("building deployed book + last two candidates ...\n")
    panel = build_base()
    base_combo = sum(panel[c].fillna(0) * W[c] for c in W)
    bm = _metrics_from_returns(overlays(base_combo, panel.index), [], "base")
    print(f"deployed portfolio_full: Sharpe {bm['sharpe']:.2f} | CAGR {bm['cagr']:.1%} | DD {bm['max_drawdown']:.1%}\n")
    score("crypto_trend", crypto_trend(), base_combo, bm, panel.index)
    score("lowvol_factor", lowvol_factor(), base_combo, bm, panel.index)


if __name__ == "__main__":
    main()
