"""
runners/cross_sectional.py
--------------------------
Build + evaluate the CROSS-SECTIONAL book (rank the universe daily, hold the
top-k strongest [momentum] or weakest [reversal], equal weight, long-only) and
compare it to / blend it with the existing books. Shows $ P&L on $100k for every
book so they're directly comparable.

Usage:
  python runners\\cross_sectional.py
  python runners\\cross_sectional.py --universe SPY,QQQ,GLD,MSFT,AAPL,GOOGL,AMZN,JPM,UNH,XOM
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    STRATEGIES_DAILY, DEPLOY_PARAMS, QUALITY_UNIVERSE,
    backtest_book, backtest_blended, backtest_cross_sectional,
    sleeve_returns, daily_bars, _metrics_from_returns, split_metrics,
)
from agents.risk_agent import RiskAgent


def _row(name, m, v):
    gate = "PASS" if v["passed"] else "FAIL"
    return (f"{name:26s} {m['sharpe']:7.2f} {m['pnl_dollars']:12,.0f} "
            f"{m['cagr']:7.1%} {m['max_drawdown']:7.1%} {m['win_rate']:8.1%} "
            f"{m['total_trades']:7d} {gate:>6s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default=",".join(QUALITY_UNIVERSE))
    args = ap.parse_args()
    universe = [t.strip().upper() for t in args.universe.split(",") if t.strip()]
    risk = RiskAgent()

    print(f"\nCross-sectional book | universe({len(universe)})={', '.join(universe)}\n")

    # 1) sweep cross-sectional configs
    print("== cross-sectional sweep ==")
    print(f"{'config':26s} {'Sharpe':>7s} {'$ PnL':>12s} {'CAGR':>7s} {'maxDD':>7s} "
          f"{'winRate':>8s} {'trades':>7s} {'RISK':>6s}")
    print("-" * 84)
    xs = {}
    grid = [
        ("xs_mom_12-1_k3",      dict(mode="momentum", lookback=252, skip=21, k=3)),
        ("xs_mom_6-1_k3",       dict(mode="momentum", lookback=126, skip=21, k=3)),
        ("xs_mom_3-0_k3",       dict(mode="momentum", lookback=63,  skip=0,  k=3)),
        ("xs_rev_5_k3",         dict(mode="reversal", lookback=5,   k=3)),
        # DUAL MOMENTUM: relative strength + market trend filter (cash in bears)
        ("xs_dualmom_12-1_k3",  dict(mode="momentum", lookback=252, skip=21, k=3, market_filter=True)),
        ("xs_dualmom_6-1_k3",   dict(mode="momentum", lookback=126, skip=21, k=3, market_filter=True)),
        ("xs_dualmom_3-0_k3",   dict(mode="momentum", lookback=63,  skip=0,  k=3, market_filter=True)),
    ]
    for name, kw in grid:
        m = backtest_cross_sectional(universe, label=name, **kw)
        v = risk.evaluate(m)
        xs[name] = (m, v)
        print(_row(name, m, v))

    best_xs = max(xs, key=lambda k: xs[k][0]["sharpe"])
    bm = xs[best_xs][0]
    print(f"\nbest cross-sectional: {best_xs} (Sharpe {bm['sharpe']}, "
          f"OOS {split_metrics(bm['_returns'])['test_sharpe']})")

    # 2) all books PnL on $100k (directly comparable)
    print("\n== ALL BOOKS -- $ P&L on $100,000 ==")
    print(f"{'book':26s} {'Sharpe':>7s} {'$ PnL':>12s} {'CAGR':>7s} {'maxDD':>7s} "
          f"{'winRate':>8s} {'trades':>7s} {'RISK':>6s}")
    print("-" * 84)
    books = {}
    for name, fn in STRATEGIES_DAILY.items():
        books[name] = backtest_book(fn, universe, DEPLOY_PARAMS.get(name), label=name)
    books["blended"] = backtest_blended(universe, DEPLOY_PARAMS, label="blended")
    books["trend_tilt"] = backtest_blended(
        universe, DEPLOY_PARAMS, label="trend_tilt",
        weights={"trend_5020": 0.5, "rsi2_meanrev": 0.5})
    books[best_xs] = bm
    for name, m in sorted(books.items(), key=lambda kv: -kv[1]["sharpe"]):
        print(_row(name, m, risk.evaluate(m)))

    # 3) does cross-sectional improve the blend? (correlation + 4-way blend)
    core_rets = {n: backtest_book(fn, universe, DEPLOY_PARAMS.get(n), label=n)["_returns"]
                 for n, fn in STRATEGIES_DAILY.items()}
    core_rets[best_xs] = bm["_returns"]
    panel = pd.concat(core_rets, axis=1); panel.columns = list(core_rets)
    corr_to_blend = panel[list(STRATEGIES_DAILY)].mean(axis=1).corr(bm["_returns"])
    print(f"\ncorrelation of {best_xs} to the core-3 blend: {corr_to_blend:.2f}")

    # 4-way equal blend (core3 + best XS)
    w = {n: 1.0 for n in core_rets}
    names = list(w); wv = np.array([w[n] for n in names]); wv /= wv.sum()
    p4 = (panel[names] * wv).sum(axis=1, min_count=1)
    # gather trades across the 4 sleeves
    tr = []
    for n, fn in STRATEGIES_DAILY.items():
        for t in universe:
            try:
                _, x = sleeve_returns(daily_bars(t), fn, DEPLOY_PARAMS.get(n))
                tr.extend(x)
            except Exception:
                pass
    from agents.daily_strategies import _xs_trades  # reuse for xs sleeve count
    m4 = _metrics_from_returns(p4, tr, "blended+xs")
    v4 = risk.evaluate(m4)
    base = books["blended"]
    print("\n== blend comparison ==")
    print(_row("blended (core-3)", base, risk.evaluate(base)))
    print(_row("blended + cross-sectional", m4, v4))
    impr = m4["sharpe"] > base["sharpe"] + 0.03 or m4["max_drawdown"] > base["max_drawdown"] + 0.01
    print(f"\n-> adding cross-sectional {'IMPROVES' if impr else 'does NOT materially improve'} the blend")


if __name__ == "__main__":
    main()
