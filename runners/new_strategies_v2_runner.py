"""
agents/new_strategies_v2_runner.py
----------------------------------
test the two newest strategies on a focused universe:
  - noise_area_breakout  (Zarattini 2024, SSRN 4824172, claimed SPY Sharpe ~1.33)
  - trend_ride           (custom: 50-bar EMA + 30-bar breakout, designed for
                          bull/bear regimes via market_regime_affinity)

both run with dump_trades=True so the user can audit every fill.

usage:
    python agents/new_strategies_v2_runner.py
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


TICKERS = ["SPY", "QQQ", "MSFT", "AAPL", "CAT"]


def run(agent, name, tickers, apply_market_regime, label=None):
    label = label or name
    print(f"--- {label} ---")
    r = agent.run({"payload": {
        "name":    name,
        "tickers": tickers,
        "start":   "2022-01-01",
        "end":     "2025-01-01",
        "dump_trades":         True,
        "apply_market_regime": apply_market_regime,
    }})
    if not r.get("success"):
        print(f"  FAILED: {r.get('reason')}\n")
        return None
    per_t = r["per_ticker"]
    agg   = r["aggregate"]
    for t, rt in per_t.items():
        csv = rt.get("trades_csv", "")
        print(f"  {t:<6} sharpe={rt['sharpe']:>6.2f}  "
              f"dd={rt['max_drawdown']:>7.2%}  "
              f"wr={rt['win_rate']:>5.1%}  "
              f"trades={rt['total_trades']:>5}  "
              f"csv={csv}")
    print(f"  {'AGG':<6} sharpe={agg['sharpe']:>6.2f}  "
          f"dd={agg['max_drawdown']:>7.2%}  "
          f"wr={agg['win_rate']:>5.1%}  "
          f"trades={agg['total_trades']:>5}\n")
    return {
        "label":   label,
        "name":    r["strategy"],
        "params":  r["params"],
        "aggregate": agg,
        "per_ticker": {t: {
            "sharpe": rt["sharpe"], "max_drawdown": rt["max_drawdown"],
            "win_rate": rt["win_rate"], "total_trades": rt["total_trades"],
            "trades_csv": rt.get("trades_csv"),
        } for t, rt in per_t.items()},
    }


def main():
    agent = BacktestingAgent(data_dir=DATA_DIR)
    out = {}

    print(f"\n{'='*72}")
    print(f"  noise_area_breakout + trend_ride  |  {len(TICKERS)} tickers")
    print(f"{'='*72}\n")

    # Zarattini noise-area — no market gate (paper says edge is in any regime)
    out["noise_area_breakout"] = run(
        agent, "noise_area_breakout", TICKERS, apply_market_regime=False,
        label="noise_area_breakout (no market gate)",
    )

    # custom trend_ride — gated to bull/bear only
    out["trend_ride_gated"] = run(
        agent, "trend_ride", TICKERS, apply_market_regime=True,
        label="trend_ride (gated to bull/bear regimes)",
    )

    # trend_ride without gate — see how much the market filter helps
    out["trend_ride_ungated"] = run(
        agent, "trend_ride", TICKERS, apply_market_regime=False,
        label="trend_ride (no market gate)",
    )

    # ranking
    valid = [(k, v) for k, v in out.items() if v]
    ranked = sorted(valid, key=lambda kv: kv[1]["aggregate"]["sharpe"], reverse=True)
    print(f"\n{'='*72}")
    print(f"  ranking by aggregate sharpe")
    print(f"{'='*72}")
    for i, (key, v) in enumerate(ranked, 1):
        a = v["aggregate"]
        print(f"  {i}. {v['label']:<48} sharpe={a['sharpe']:>6.2f}  "
              f"dd={a['max_drawdown']:>7.2%}  trades={a['total_trades']}")

    print(f"\n  per-ticker positive-sharpe count (out of {len(TICKERS)}):")
    for key, v in valid:
        positives = sum(1 for t, rt in v["per_ticker"].items() if rt["sharpe"] > 0)
        print(f"    {v['label']:<48} {positives}/{len(TICKERS)} tickers with sharpe>0")

    path = Path("results/new_strategies_v2.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"tickers": TICKERS, "results": out,
                   "run_at": datetime.now(timezone.utc).isoformat()},
                  f, indent=2, default=str)
    print(f"\n  saved to {path}")
    print(f"  per-trade CSVs in results/trades/")


if __name__ == "__main__":
    main()
