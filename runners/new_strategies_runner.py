"""
agents/new_strategies_runner.py
-------------------------------
isolated test of the 2 new strategies (now that the v2 lookup bug is fixed):
  1. bb_band_touch_revert_v2 — RSI confluence + tighter stop
  2. half_hour_continuation  — paired with a stronger threshold sweep

both tested on the same 8-ticker extended universe used previously so we can
compare apples-to-apples against the original strategies and v1.
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

TICKERS = ["SPY", "QQQ", "AAPL", "AMD", "MSFT", "JPM", "GS", "CAT"]


def run_strategy(agent, name, params=None, label=None):
    label = label or name
    print(f"--- {label} ---")
    payload = {
        "name":    name,
        "tickers": TICKERS,
        "start":   "2022-01-01",
        "end":     "2025-01-01",
    }
    if params:
        payload["params"] = params

    r = agent.run({"payload": payload})
    if not r.get("success"):
        print(f"  FAILED: {r.get('reason')}\n")
        return None

    per_t = r["per_ticker"]
    agg   = r["aggregate"]
    for t, rt in per_t.items():
        print(f"  {t:<6} sharpe={rt['sharpe']:>6.2f}  "
              f"dd={rt['max_drawdown']:>7.2%}  "
              f"wr={rt['win_rate']:>5.1%}  "
              f"trades={rt['total_trades']:>5}")
    print(f"  {'AGG':<6} sharpe={agg['sharpe']:>6.2f}  "
          f"dd={agg['max_drawdown']:>7.2%}  "
          f"wr={agg['win_rate']:>5.1%}  "
          f"trades={agg['total_trades']:>5}\n")
    return {
        "label":  label,
        "name":   r["strategy"],
        "params": r["params"],
        "aggregate": agg,
        "per_ticker": {t: {
            "sharpe": rt["sharpe"], "max_drawdown": rt["max_drawdown"],
            "win_rate": rt["win_rate"], "total_trades": rt["total_trades"],
        } for t, rt in per_t.items()},
    }


def main():
    agent = BacktestingAgent(data_dir=DATA_DIR)
    out = {}

    print(f"\n{'='*72}")
    print(f"  isolated new-strategy test  |  {len(TICKERS)} tickers")
    print(f"{'='*72}\n")

    # 1. v2 with default params
    out["bb_v2_default"] = run_strategy(agent, "bb_band_touch_revert_v2",
                                        label="bb_band_touch_revert_v2 (RSI 70/30)")

    # 2. v2 with stricter RSI (75/25) — only the truly overextended
    out["bb_v2_strict"] = run_strategy(
        agent, "bb_band_touch_revert_v2",
        params={"rsi_overbought": 75, "rsi_oversold": 25},
        label="bb_band_touch_revert_v2 (RSI 75/25)",
    )

    # 3. half_hour_continuation with stronger threshold (10 bps)
    out["hhc_10bps"] = run_strategy(
        agent, "half_hour_continuation",
        params={"threshold_bps": 10},
        label="half_hour_continuation (thr=10bps)",
    )

    # 4. half_hour_continuation with even stronger threshold (25 bps)
    out["hhc_25bps"] = run_strategy(
        agent, "half_hour_continuation",
        params={"threshold_bps": 25},
        label="half_hour_continuation (thr=25bps)",
    )

    # ranking
    print(f"\n{'='*72}")
    print(f"  ranking by aggregate sharpe")
    print(f"{'='*72}")
    valid = [(k, v) for k, v in out.items() if v]
    ranked = sorted(valid, key=lambda kv: kv[1]["aggregate"]["sharpe"], reverse=True)
    for i, (key, v) in enumerate(ranked, 1):
        a = v["aggregate"]
        print(f"  {i}. {v['label']:<40} sharpe={a['sharpe']:>6.2f}  "
              f"dd={a['max_drawdown']:>7.2%}  wr={a['win_rate']:>5.1%}  trades={a['total_trades']}")

    print(f"\n  per-ticker positive-sharpe count (out of {len(TICKERS)}):")
    for key, v in valid:
        positives = sum(1 for t, rt in v["per_ticker"].items() if rt["sharpe"] > 0)
        print(f"    {v['label']:<40} {positives}/{len(TICKERS)} tickers with sharpe>0")

    path = Path("results/new_strategies_isolated.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "tickers":     TICKERS,
            "results":     out,
            "run_at":      datetime.now(timezone.utc).isoformat(),
        }, f, indent=2, default=str)
    print(f"\n  saved to {path}")


if __name__ == "__main__":
    main()
