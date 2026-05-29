"""
runners/strategy_lab.py
-----------------------
Research bench for NEW daily strategies + blend tuning. A new strategy earns its
place only if it improves the BLENDED book (higher Sharpe or lower drawdown) — a
mediocre but uncorrelated sleeve can do that; a great but correlated one can't.

Does three things:
  1. Backtests the 3 core + N candidate strategies standalone (quality universe).
  2. Prints the return-correlation matrix (low correlation = good diversifier).
  3. Tests candidate blends — equal-weight, inverse-volatility (risk-parity),
     and core+candidate combos — and ranks them through the risk gate.

Usage:
  python runners\\strategy_lab.py
  python runners\\strategy_lab.py --universe SPY,QQQ,GLD,MSFT,AAPL,GOOGL,AMZN,JPM,UNH,XOM
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    STRATEGIES_DAILY, CANDIDATE_STRATEGIES, DEPLOY_PARAMS, QUALITY_UNIVERSE,
    backtest_book, sleeve_returns, daily_bars, _metrics_from_returns,
    walk_forward_folds, split_metrics, TRADING_DAYS, INITIAL_CAP,
)
from agents.risk_agent import RiskAgent


def book_returns_and_trades(name, fn, universe, params):
    """returns (daily_return_series, trades_list) for a one-strategy book."""
    sleeves, trades = [], []
    for t in universe:
        try:
            d = daily_bars(t)
        except Exception:
            continue
        net, trs = sleeve_returns(d, fn, params)
        sleeves.append(net.rename(t))
        trades.extend(trs)
    ret = pd.concat(sleeves, axis=1).mean(axis=1)
    return ret, trades


def blend(rets: dict, trades: dict, weights: dict, label: str):
    names = [n for n in weights if weights[n] > 0]
    w = np.array([weights[n] for n in names], float); w /= w.sum()
    panel = pd.concat([rets[n] for n in names], axis=1); panel.columns = names
    port = (panel * w).sum(axis=1, min_count=1)
    all_tr = [t for n in names for t in trades[n]]
    m = _metrics_from_returns(port, all_tr, label)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default=",".join(QUALITY_UNIVERSE))
    args = ap.parse_args()
    universe = [t.strip().upper() for t in args.universe.split(",") if t.strip()]
    risk = RiskAgent()

    allfns = {**STRATEGIES_DAILY, **CANDIDATE_STRATEGIES}
    core = list(STRATEGIES_DAILY)
    cands = list(CANDIDATE_STRATEGIES)

    print(f"\nStrategy lab | universe({len(universe)}) | core={core} | candidates={cands}\n")

    rets, trades, standalone = {}, {}, {}
    for name, fn in allfns.items():
        r, tr = book_returns_and_trades(name, fn, universe, DEPLOY_PARAMS.get(name))
        rets[name], trades[name] = r, tr
        standalone[name] = _metrics_from_returns(r, tr, name)

    # 1) standalone
    print(f"{'strategy':16s} {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s} {'winRate':>8s} {'trades':>7s}")
    print("-" * 60)
    for name in allfns:
        m = standalone[name]
        tag = "  (candidate)" if name in cands else ""
        print(f"{name:16s} {m['sharpe']:7.2f} {m['cagr']:7.1%} {m['max_drawdown']:7.1%} "
              f"{m['win_rate']:8.1%} {m['total_trades']:7d}{tag}")

    # 2) correlation matrix of daily returns
    panel = pd.concat([rets[n] for n in allfns], axis=1); panel.columns = list(allfns)
    corr = panel.corr()
    print("\nReturn correlation matrix (lower = better diversifier):")
    print(corr.round(2).to_string())

    # 3) candidate blends
    def eqw(names): return {n: 1.0 for n in names}
    def invvol(names):
        return {n: 1.0 / (rets[n].std() or 1e-9) for n in names}

    blends = {
        "core3_equal (current)":  eqw(core),
        "core3_invvol":           invvol(core),
    }
    # core + each single candidate (equal)
    for c in cands:
        blends[f"core3+{c}"] = eqw(core + [c])
    # core + all candidates
    blends["core3+all_cands_equal"] = eqw(core + cands)
    blends["core3+all_cands_invvol"] = invvol(core + cands)

    print(f"\n{'blend':28s} {'Sharpe':>7s} {'$PnL':>11s} {'CAGR':>7s} {'maxDD':>7s} "
          f"{'winRate':>8s} {'RISK':>6s}")
    print("-" * 78)
    results = {}
    for label, w in blends.items():
        m = blend(rets, trades, w, label)
        v = risk.evaluate(m)
        results[label] = (m, v)
        gate = "PASS" if v["passed"] else "FAIL"
        print(f"{label:28s} {m['sharpe']:7.2f} {m['pnl_dollars']:11,.0f} "
              f"{m['cagr']:7.1%} {m['max_drawdown']:7.1%} {m['win_rate']:8.1%} {gate:>6s}")

    # recommend the best PASSING blend by Sharpe, tiebreak lower DD
    passing = {k: v for k, v in results.items() if v[1]["passed"]}
    pool = passing or results
    best = max(pool, key=lambda k: (pool[k][0]["sharpe"], pool[k][0]["max_drawdown"]))
    bm = results[best][0]
    base = results["core3_equal (current)"][0]
    print(f"\n=== BEST BLEND: {best} ===")
    print(f"  Sharpe {bm['sharpe']} (current {base['sharpe']}) | "
          f"CAGR {bm['cagr']:.1%} (cur {base['cagr']:.1%}) | "
          f"maxDD {bm['max_drawdown']:.1%} (cur {base['max_drawdown']:.1%})")
    print(f"  split: {split_metrics(bm['_returns'])}")
    impr = (bm['sharpe'] > base['sharpe'] + 0.03) or (bm['max_drawdown'] > base['max_drawdown'] + 0.01)
    print(f"  -> {'IMPROVES the current blend' if impr else 'no material improvement over current'}")


if __name__ == "__main__":
    main()
