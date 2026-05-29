"""
runners/pead_test.py
--------------------
Test the PEAD (post-earnings-drift) sleeve on the full S&P 500: Sharpe,
walk-forward, correlation to the equity book (is it genuinely uncorrelated?),
the 2018-2020 lean fold, and whether the allocator would admit it.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    backtest_book, backtest_cross_sectional, vol_target, _metrics_from_returns,
    sig_rsi2_meanrev, sig_donchian, sig_trend_5020, sig_pead,
    DEPLOY_PARAMS, QUALITY_UNIVERSE, walk_forward_folds, split_metrics,
)
from data.sp500 import sp500_tickers


def fold2(r):
    return walk_forward_folds(r, 5)[1]["return_pct"]


def main():
    sp = sp500_tickers()
    print(f"\n== PEAD sleeve (full S&P 500, buy earnings-gap beats, hold the drift) ==")
    print(f"{'config':24s} {'Sharpe':>6s} {'CAGR':>7s} {'maxDD':>7s} {'OOS':>6s} {'2018-20':>8s} {'trades':>7s}")
    print("-" * 72)
    cands = {}
    for lbl, kw in [("gap5% hold40", dict(gap_pct=0.05, vol_mult=2.0, hold_days=40)),
                    ("gap5% hold60", dict(gap_pct=0.05, vol_mult=2.0, hold_days=60)),
                    ("gap4% hold40", dict(gap_pct=0.04, vol_mult=1.8, hold_days=40)),
                    ("gap7% hold30", dict(gap_pct=0.07, vol_mult=2.5, hold_days=30))]:
        m = backtest_book(sig_pead, sp, kw, label=lbl)
        r = m["_returns"]; cands[lbl] = r
        s = split_metrics(r)
        print(f"{lbl:24s} {m['sharpe']:6.2f} {m['cagr']:7.1%} {m['max_drawdown']:7.1%} "
              f"{s['test_sharpe']:6.2f} {fold2(r):+8.1%} {m['total_trades']:7d}")

    best = max(cands, key=lambda k: _metrics_from_returns(cands[k], [], k)["sharpe"])
    pead = cands[best]
    bm = _metrics_from_returns(pead, [], best)
    gate = (bm["sharpe"] >= 0.8 and bm["max_drawdown"] >= -0.15)
    print(f"\nbest: {best} | Sharpe {bm['sharpe']} | standalone gate: {'PASS' if gate else 'FAIL'}")

    # correlation + portfolio impact
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
    corr = round(float(pead.reindex(base.index).fillna(0).corr(base)), 2)
    print(f"correlation of PEAD to the equity portfolio: {corr}  (lower = better diversifier)")

    print("\n== portfolio impact (risk-parity, vol-target 16%/1.6x) ==")
    for label, r in [("portfolio (4 sleeves)", base),
                     ("+ PEAD (5 sleeves)", build({"rsi": rsi, "don": don, "trd": trd, "xs": xs, "pead": pead}))]:
        m = _metrics_from_returns(r, [], label)
        print(f"  {label:28s} Sharpe {m['sharpe']:.2f} | CAGR {m['cagr']:.1%} | "
              f"DD {m['max_drawdown']:.1%} | 2018-2020 {fold2(r):+.1%}")


if __name__ == "__main__":
    main()
