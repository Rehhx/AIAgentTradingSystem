"""
runners/reversal_test.py
------------------------
New non-leverage candidate: LONG-ONLY cross-sectional short-term reversal across
the full S&P 500 — buy the biggest N-day losers (when the market is up; they tend
to bounce), diversified over k names. Reversal is the natural edge in choppy /
sideways markets (the lean-year problem) and is often uncorrelated to momentum.
Tested honestly through the gate: Sharpe + walk-forward + correlation + 2018-2020.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    backtest_book, backtest_cross_sectional, vol_target, _metrics_from_returns,
    sig_rsi2_meanrev, sig_donchian, sig_trend_5020, DEPLOY_PARAMS, QUALITY_UNIVERSE,
    walk_forward_folds, split_metrics,
)
from data.sp500 import sp500_tickers


def fold2(r):
    return walk_forward_folds(r, 5)[1]["return_pct"]


def main():
    sp = sp500_tickers()
    print(f"\n== long-only cross-sectional reversal (full S&P 500, buy biggest losers, market-filtered) ==")
    print(f"{'config':22s} {'Sharpe':>6s} {'CAGR':>7s} {'maxDD':>7s} {'OOS':>6s} {'2018-20':>8s}")
    print("-" * 64)
    cands = {}
    for lbl, kw in [("rev 5d, k=30", dict(lookback=5, k=30)),
                    ("rev 5d, k=50", dict(lookback=5, k=50)),
                    ("rev 3d, k=30", dict(lookback=3, k=30)),
                    ("rev 10d, k=30", dict(lookback=10, k=30))]:
        m = backtest_cross_sectional(sp, mode="reversal", market_filter=True, label=lbl, **kw)
        r = m["_returns"]; cands[lbl] = r
        s = split_metrics(r)
        print(f"{lbl:22s} {m['sharpe']:6.2f} {m['cagr']:7.1%} {m['max_drawdown']:7.1%} "
              f"{s['test_sharpe']:6.2f} {fold2(r):+8.1%}")

    best = max(cands, key=lambda k: _metrics_from_returns(cands[k], [], k)["sharpe"])
    rev = cands[best]
    bm = _metrics_from_returns(rev, [], best)
    gate = (bm["sharpe"] >= 0.8 and bm["max_drawdown"] >= -0.15)
    print(f"\nbest: {best} | Sharpe {bm['sharpe']} | standalone gate: {'PASS' if gate else 'FAIL'}")

    # correlation to the equity portfolio + portfolio impact
    U = QUALITY_UNIVERSE
    rsi = backtest_book(sig_rsi2_meanrev, U, DEPLOY_PARAMS["rsi2_meanrev"])["_returns"]
    don = backtest_book(sig_donchian, U)["_returns"]
    trd = backtest_book(sig_trend_5020, U)["_returns"]
    xs = backtest_cross_sectional(sp, mode="momentum", lookback=252, skip=21, k=10, market_filter=True)["_returns"]

    def build(sleeves):
        panel = pd.concat(sleeves, axis=1, sort=True)
        iv = {c: 1.0 / (panel[c].std() or 1e-9) for c in panel.columns}
        w = np.array([iv[c] for c in panel.columns]); w /= w.sum()
        return vol_target((panel.fillna(0.0) * w).sum(axis=1), 0.16, max_leverage=1.6)

    base = build({"rsi": rsi, "don": don, "trd": trd, "xs": xs})
    corr = round(float(rev.reindex(base.index).fillna(0).corr(base)), 2)
    corr_rsi = round(float(rev.reindex(rsi.index).fillna(0).corr(rsi)), 2)
    print(f"correlation of reversal to: portfolio {corr} | RSI-2 sleeve {corr_rsi}")

    print("\n== portfolio impact (risk-parity, vol-target 16%/1.6x) ==")
    for label, r in [("portfolio (4 sleeves)", base),
                     ("+ reversal (5 sleeves)", build({"rsi": rsi, "don": don, "trd": trd, "xs": xs, "rev": rev}))]:
        m = _metrics_from_returns(r, [], label)
        print(f"  {label:28s} Sharpe {m['sharpe']:.2f} | CAGR {m['cagr']:.1%} | "
              f"DD {m['max_drawdown']:.1%} | 2018-2020 {fold2(r):+.1%}")


if __name__ == "__main__":
    main()
