"""
agents/noise_area_aapl_wf.py
----------------------------
walk-forward noise_area_breakout on AAPL — the one ticker where the
strategy posted +0.34 on the full 2022-2025 window. tests whether that
positive sharpe survives a 70/30 train/test split.

usage:
    python agents/noise_area_aapl_wf.py
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from agents.backtesting_agent import walk_forward_optimize

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")


PARAM_GRID = {
    "sigma_mult":    [0.5, 1.0, 1.5, 2.0],
    "lookback_days": [7, 14, 21, 30],
    "stop_atr_mult": [1.5, 2.0, 2.5],
    # one_entry_per_day is structural; not sweeping it
}


def main():
    print(f"\n{'='*60}")
    print(f"  walk-forward noise_area_breakout on AAPL")
    print(f"  ({len(PARAM_GRID['sigma_mult']) * len(PARAM_GRID['lookback_days']) * len(PARAM_GRID['stop_atr_mult'])} combos)")
    print(f"{'='*60}")

    res = walk_forward_optimize(
        strategy_name = "noise_area_breakout",
        param_grid    = PARAM_GRID,
        tickers       = ["AAPL"],
        start         = "2022-01-01",
        end           = "2025-01-01",
        train_pct     = 0.7,
    )

    if not res.get("success"):
        print(f"FAILED: {res.get('reason')}")
        return

    print(f"\nbest params  : {res['best_params']}")
    print(f"train sharpe : {res['train_sharpe']}")
    print(f"test  sharpe : {res['test_sharpe']}")
    print(f"overfit gap  : {res['overfit_gap']}")

    print(f"\ntop 8 train configs:")
    for r in res["all_results"][:8]:
        print(f"  {r['params']} -> train_sharpe={r['train_sharpe']}")

    path = Path("results/noise_area_aapl_wf.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "strategy":  "noise_area_breakout",
            "ticker":    "AAPL",
            "best_params":  res["best_params"],
            "train_sharpe": res["train_sharpe"],
            "test_sharpe":  res["test_sharpe"],
            "overfit_gap":  res["overfit_gap"],
            "all_results":  res["all_results"],
            "run_at":       datetime.now(timezone.utc).isoformat(),
        }, f, indent=2, default=str)
    print(f"\nsaved to {path}")


if __name__ == "__main__":
    main()
