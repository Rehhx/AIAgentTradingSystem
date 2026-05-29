"""
agents/spy_bb_squeeze_full_runner.py
------------------------------------
the "everything we know" run for the lead candidate:

  - ticker:        SPY only (highest train Sharpe in the per-ticker WF)
  - strategy:      bb_squeeze
  - params:        bb_period=30, bb_std=2.5 (from prior per-ticker WF)
  - regime gate:   STRATEGY_REGIME_AFFINITY["bb_squeeze"] = {"breakout"}
  - embedding gate: precomputed SPY quality cache, sweep thresholds
  - validation:    train/test split — best threshold on train, measure on test

if both train and test Sharpe come out positive without a big overfit gap,
this is the first real candidate ready for risk_review promotion.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from agents.backtesting_agent import walk_forward_with_gate

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def main():
    print("=" * 70)
    print("  SPY × bb_squeeze — all filters, walk-forward gate threshold")
    print("=" * 70)
    print(f"  params override : bb_period=30, bb_std=2.5  (from per-ticker WF)")
    print(f"  regime gate     : breakout only (single regime)")
    print(f"  embedding gate  : sweep thresholds 0.50, 0.52, 0.55, 0.58, 0.60")
    print(f"  train/test      : 70 / 30 of 2022-2025")
    print()

    result = walk_forward_with_gate(
        strategy_name           = "bb_squeeze",
        ticker                  = "SPY",
        params_override         = {"bb_period": 30, "bb_std": 2.5},
        quality_min_pct_pos_grid = [0.50, 0.52, 0.55, 0.58, 0.60],
    )

    if not result.get("success"):
        print(f"FAILED: {result.get('reason')}")
        return

    print("\n  threshold grid on train:")
    for r in result["all_train_results"]:
        print(f"    thr={r['threshold']}  train_sharpe={r['train_sharpe']:>+7.3f}  trades={r['train_trades']}")

    print(f"\n  best threshold : {result['best_threshold']}")
    print(f"  train sharpe   : {result['train_sharpe']:>+7.3f}    ({result['train_trades']} trades)")
    print(f"  test  sharpe   : {result['test_sharpe']:>+7.3f}    ({result['test_trades']} trades, "
          f"wr={result['test_wr']:.2%}, dd={result['test_dd']:.2%})")
    print(f"  overfit gap    : {result['overfit_gap']:>+7.3f}    (positive = test worse than train)")
    print()

    # interpretation
    train, test, gap = result["train_sharpe"], result["test_sharpe"], result["overfit_gap"]
    if train > 0 and test > 0 and abs(gap) < 1.0:
        verdict = ">>> CANDIDATE: positive train AND test, small overfit gap"
    elif test > 0:
        verdict = ">>> PASSED: positive test sharpe (train was negative — luckier than expected)"
    elif train > 0 and test >= -0.5:
        verdict = ">>> NEAR-MISS: train positive, test near zero — possibly tradeable with more data"
    elif gap > 1.5:
        verdict = ">>> OVERFIT: train was tuned to past noise; do not promote"
    elif test >= -0.5:
        verdict = ">>> NEAR-MISS: test sharpe near zero, structurally close"
    else:
        verdict = ">>> NOT READY: test sharpe still bleeding"
    print(f"  {verdict}")

    out = Path("results/spy_bb_squeeze_full.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  saved to {out}")


if __name__ == "__main__":
    main()
