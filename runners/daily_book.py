"""
runners/daily_book.py
---------------------
Board-ready report for the daily strategy books. Builds all three strategies as
standalone $100k equal-weight portfolios, plus the blended book, then:

  - prints a ranked table (Sharpe, $PnL, CAGR, maxDD, win-rate, trades)
  - runs each book through the real RiskAgent.evaluate (config.RISK gates)
  - prints 70/30 in/out-of-sample Sharpe and a 5-fold walk-forward
  - writes results/daily_book.json

Usage:
    python runners\\daily_book.py                       # default 8-name universe
    python runners\\daily_book.py --universe SPY,QQQ,GLD
    python runners\\daily_book.py --universe all        # all 20 parquet tickers
    python runners\\daily_book.py --folds 6
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.daily_strategies import (
    STRATEGIES_DAILY, DEPLOY_PARAMS, DEFAULT_UNIVERSE,
    backtest_book, backtest_blended, walk_forward_folds, split_metrics,
)
from agents.risk_agent import RiskAgent
from config import RISK, RESULTS_DIR


def _clean(m: dict) -> dict:
    return {k: v for k, v in m.items() if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default=",".join(DEFAULT_UNIVERSE))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    if args.universe.strip().lower() == "all":
        from data.loader import DATA_DIR
        universe = sorted(p.stem for p in Path(DATA_DIR).glob("*.parquet"))
    elif args.universe.strip().lower() == "sp500":
        from data.sp500 import sp500_tickers
        universe = sp500_tickers()
    else:
        universe = [t.strip().upper() for t in args.universe.split(",") if t.strip()]

    print(f"\nUniverse ({len(universe)}): {', '.join(universe)}")
    print(f"Cost basis: 6 bps round-trip | start: $100,000 | long/flat daily\n")

    risk = RiskAgent()
    books = {}

    # three standalone books (RSI-2 uses the walk-forward-tuned DEPLOY_PARAMS)
    for name, fn in STRATEGIES_DAILY.items():
        books[name] = backtest_book(fn, universe, DEPLOY_PARAMS.get(name) or None, label=name)
    # blended
    books["blended_book"] = backtest_blended(universe, DEPLOY_PARAMS)

    # ---- ranked table ----
    hdr = (f"{'book':16s} {'Sharpe':>7s} {'$ PnL':>12s} {'CAGR':>7s} "
           f"{'maxDD':>7s} {'winRate':>8s} {'trades':>7s} {'expo':>6s} {'RISK':>6s}")
    print(hdr)
    print("-" * len(hdr))
    rows = sorted(books.items(), key=lambda kv: -kv[1].get("sharpe", -99))
    summary = {}
    for name, m in rows:
        verdict = risk.evaluate(m)
        gate = "PASS" if verdict["passed"] else "FAIL"
        print(f"{name:16s} {m['sharpe']:7.2f} {m['pnl_dollars']:12,.0f} "
              f"{m['cagr']:7.1%} {m['max_drawdown']:7.1%} {m['win_rate']:8.1%} "
              f"{m['total_trades']:7d} {m['exposure_pct']:6.0%} {gate:>6s}")
        summary[name] = {
            **_clean(m),
            "risk_passed": verdict["passed"],
            "risk_failures": verdict["failures"],
            "split": split_metrics(m["_returns"]),
            "walk_forward": walk_forward_folds(m["_returns"], args.folds),
        }

    # ---- detail: in/out-of-sample + folds ----
    for name, m in rows:
        s = summary[name]
        print(f"\n=== {name} ===")
        print(f"  in-sample (first 70%)  Sharpe: {s['split']['train_sharpe']:+.2f}")
        print(f"  out-of-sample (last 30%) Sharpe: {s['split']['test_sharpe']:+.2f}")
        if s["risk_failures"]:
            print(f"  risk gate FAILS: {s['risk_failures']}")
        else:
            print(f"  risk gate: PASS (Sharpe>={RISK['min_sharpe']}, "
                  f"DD>={RISK['max_drawdown']:.0%}, WR>={RISK['min_win_rate']:.0%}, "
                  f"trades>={RISK['min_trades']})")
        print(f"  walk-forward ({args.folds} folds):")
        for f in s["walk_forward"]:
            mark = "+" if f["sharpe"] > 0 else "-"
            rng = f"{f.get('start','?')}..{f.get('end','?')}"
            print(f"    [{mark}] fold {f['fold']} {rng}: Sharpe {f['sharpe']:+.2f}, "
                  f"ret {f['return_pct']:+.1%}")

    # ---- persist ----
    out = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "universe": universe,
        "cost_bps_round_trip": 6,
        "initial_capital": 100_000,
        "risk_thresholds": RISK,
        "books": summary,
    }
    out_path = Path(RESULTS_DIR) / "daily_book.json"
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
