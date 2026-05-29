"""
agents/bb_squeeze_msft_cat_wf.py
---------------------------------
walk-forward bb_squeeze on the two tickers where it showed positive Sharpe:
MSFT (+0.26) and CAT (+0.34). sweeps bb_period / bb_std / kc_mult and
reports train vs test sharpe per ticker so we can see if the edge holds
out-of-sample.

usage:
    python agents/bb_squeeze_msft_cat_wf.py
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
    "bb_period":     [15, 20, 25, 30],
    "bb_std":        [2.0, 2.5, 3.0],
    "kc_mult":       [1.0, 1.5, 2.0],
    "stop_atr_mult": [1.5, 2.0, 2.5],
}


def main():
    out = {}
    for ticker in ["MSFT", "CAT"]:
        print(f"\n{'='*60}\n  walk-forward bb_squeeze on {ticker}\n{'='*60}")
        res = walk_forward_optimize(
            strategy_name = "bb_squeeze",
            param_grid    = PARAM_GRID,
            tickers       = [ticker],
            start         = "2022-01-01",
            end           = "2025-01-01",
            train_pct     = 0.7,
        )
        if not res.get("success"):
            print(f"FAILED: {res.get('reason')}")
            continue

        print(f"  best params  : {res['best_params']}")
        print(f"  train sharpe : {res['train_sharpe']}")
        print(f"  test  sharpe : {res['test_sharpe']}")
        print(f"  overfit gap  : {res['overfit_gap']}")

        # top 5 train results
        print(f"  top 5 train configs:")
        for r in res["all_results"][:5]:
            print(f"    {r['params']} -> {r['train_sharpe']}")

        out[ticker] = {
            "best_params":  res["best_params"],
            "train_sharpe": res["train_sharpe"],
            "test_sharpe":  res["test_sharpe"],
            "overfit_gap":  res["overfit_gap"],
            "top5":         res["all_results"][:5],
        }

    path = Path("results/bb_squeeze_wf_msft_cat.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"results": out, "run_at": datetime.now(timezone.utc).isoformat()},
                  f, indent=2, default=str)
    print(f"\nsaved to {path}")


if __name__ == "__main__":
    main()
