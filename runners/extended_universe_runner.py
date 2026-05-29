"""
agents/extended_universe_runner.py
----------------------------------
test the top strategies across an expanded ticker universe so we can see
which strategies have edge that generalizes vs. those that worked only on
the original 5 (SPY/QQQ/TSLA/NVDA/AAPL).

extended set: AAPL, AMD, AMZN, GOOGL, MSFT (mega-cap tech)
              + JPM, GS, CAT (non-tech for diversification check)

strategies tested:
  - bb_squeeze              (current #1)
  - overnight_gap_fade      (current #2)
  - extreme_bar_fade        (current #3)
  - half_hour_continuation  (NEW — academic edge)
  - bb_band_touch_revert_v2 (NEW — RSI-gated v1)

output saved to results/extended_universe.json + console table.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from agents.backtesting_agent import BacktestingAgent
from data.loader import DATA_DIR

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")


EXTENDED_TICKERS = ["AAPL", "AMD", "AMZN", "GOOGL", "MSFT", "JPM", "GS", "CAT"]

STRATEGIES_TO_TEST = [
    "bb_squeeze",
    "overnight_gap_fade",
    "extreme_bar_fade",
    "half_hour_continuation",
    "bb_band_touch_revert_v2",
]


def main():
    agent = BacktestingAgent(data_dir=DATA_DIR)
    all_results = {}

    print(f"\n{'='*72}")
    print(f"  extended-universe backtest  |  {len(STRATEGIES_TO_TEST)} strategies × {len(EXTENDED_TICKERS)} tickers")
    print(f"{'='*72}\n")

    for strat in STRATEGIES_TO_TEST:
        print(f"--- {strat} ---")
        r = agent.run({"payload": {
            "name":    strat,
            "tickers": EXTENDED_TICKERS,
            "start":   "2022-01-01",
            "end":     "2025-01-01",
        }})

        if not r.get("success"):
            print(f"  FAILED: {r.get('reason')}")
            print()
            continue

        per_ticker = r["per_ticker"]
        agg        = r["aggregate"]
        all_results[strat] = {
            "aggregate": agg,
            "per_ticker": {t: {
                "sharpe": rt["sharpe"], "max_drawdown": rt["max_drawdown"],
                "win_rate": rt["win_rate"], "total_trades": rt["total_trades"],
            } for t, rt in per_ticker.items()},
        }

        # print per-ticker breakdown
        for t, rt in per_ticker.items():
            print(f"  {t:<6} sharpe={rt['sharpe']:>6.2f}  "
                  f"dd={rt['max_drawdown']:>7.2%}  "
                  f"wr={rt['win_rate']:>5.1%}  "
                  f"trades={rt['total_trades']:>5}")
        print(f"  {'AGG':<6} sharpe={agg['sharpe']:>6.2f}  "
              f"dd={agg['max_drawdown']:>7.2%}  "
              f"wr={agg['win_rate']:>5.1%}  "
              f"trades={agg['total_trades']:>5}")
        print()

    # cross-strategy ranking
    print(f"\n{'='*72}")
    print(f"  ranking by aggregate sharpe on extended universe")
    print(f"{'='*72}")
    ranked = sorted(all_results.items(), key=lambda kv: kv[1]["aggregate"]["sharpe"], reverse=True)
    for i, (strat, r) in enumerate(ranked, 1):
        a = r["aggregate"]
        print(f"  {i}. {strat:<26} sharpe={a['sharpe']:>6.2f}  "
              f"dd={a['max_drawdown']:>7.2%}  wr={a['win_rate']:>5.1%}  trades={a['total_trades']}")

    # find strategies that consistently work on >=4 tickers (positive sharpe)
    print(f"\n  per-ticker positive-sharpe count (out of {len(EXTENDED_TICKERS)}):")
    for strat, r in all_results.items():
        positives = sum(1 for t, rt in r["per_ticker"].items() if rt["sharpe"] > 0)
        print(f"    {strat:<26} {positives}/{len(EXTENDED_TICKERS)} tickers with sharpe>0")

    out = Path("results/extended_universe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "tickers":     EXTENDED_TICKERS,
            "strategies":  all_results,
            "run_at":      datetime.now(timezone.utc).isoformat(),
        }, f, indent=2, default=str)
    print(f"\n  saved to {out}")


if __name__ == "__main__":
    main()
