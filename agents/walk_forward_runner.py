"""
agents/walk_forward_runner.py
-----------------------------
runs walk_forward_optimize for every active strategy and writes results to
results/walk_forward_results.json so the orchestrator can pick up best params.

usage:
    python agents/walk_forward_runner.py
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.backtesting_agent import (
    STRATEGIES,
    walk_forward_optimize,
)

log = logging.getLogger("walk_forward_runner")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# grids kept intentionally compact — full cartesian product of these is what
# every active strategy will be searched over.
GRIDS = {
    "bb_squeeze":    {"bb_period": [15, 20, 30], "bb_std": [1.5, 2.0, 2.5]},
    "orb":           {"orb_minutes": [5, 15, 30]},
    "momentum":      {"lookback_bars": [10, 20, 40], "volume_zscore_min": [1.0, 1.5]},
    "ema_crossover": {"fast_period": [5, 9], "slow_period": [21, 34]},
}


def main():
    out = []
    for name, grid in GRIDS.items():
        _, default_params = STRATEGIES.get(name, (None, {}))
        if not default_params.get("active", True):
            print(f"skipping {name} (inactive)")
            continue
        print("=" * 60)
        print(f"  walk-forward optimizing: {name}")
        print(f"  grid: {grid}")
        print("=" * 60)
        result = walk_forward_optimize(name, grid)
        if not result["success"]:
            print(f"  FAILED: {result.get('reason')}")
            continue
        print(f"  best params : {result['best_params']}")
        print(f"  train sharpe: {result['train_sharpe']:.3f}")
        print(f"  test  sharpe: {result['test_sharpe']:.3f}")
        print(f"  overfit gap : {result['overfit_gap']:.3f}  (small = good)")
        print()
        for row in result["all_results"][:5]:
            print(f"    {row['params']}  train_sharpe={row['train_sharpe']:.3f}")
        print()
        out.append(result)

    output_path = Path("results/walk_forward_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {"results": out, "run_at": datetime.now(timezone.utc).isoformat()},
            f, indent=2,
        )
    print(f"saved results to {output_path}")


if __name__ == "__main__":
    main()
