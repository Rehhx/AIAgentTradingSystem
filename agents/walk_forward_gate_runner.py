"""
agents/walk_forward_gate_runner.py
----------------------------------
walk-forward the embedding gate threshold for a (strategy, ticker) pair.
catches threshold overfit explicitly: a good train sharpe with a bad test
sharpe means the threshold was tuned to past noise, not real conviction.

usage:
    python agents/walk_forward_gate_runner.py bb_squeeze NVDA
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


def main(strategy: str, ticker: str):
    print(f"walk-forwarding gate threshold for {strategy} x {ticker}")
    print(f"train = first 70% of 2022-2025 data, test = last 30%")
    print()

    result = walk_forward_with_gate(strategy, ticker)
    if not result["success"]:
        print(f"FAILED: {result.get('reason')}")
        return

    print("threshold grid (train):")
    for r in result["all_train_results"]:
        print(f"  thr={r['threshold']}  train_sharpe={r['train_sharpe']:>7.3f}  trades={r['train_trades']}")
    print()
    print(f"best threshold:  {result['best_threshold']}")
    print(f"train sharpe:    {result['train_sharpe']:.3f} ({result['train_trades']} trades)")
    print(f"test  sharpe:    {result['test_sharpe']:.3f} ({result['test_trades']} trades, wr={result['test_wr']:.2%}, dd={result['test_dd']:.2%})")
    print(f"overfit gap:     {result['overfit_gap']:+.3f}  (positive = test worse than train)")
    print()

    if result["test_sharpe"] >= 0 and result["overfit_gap"] < 1.0:
        print(">>> CANDIDATE: positive test sharpe, no major overfit")
    elif result["overfit_gap"] > 1.5:
        print(">>> WARNING: large overfit gap, threshold may have curve-fit to train period")
    elif result["test_sharpe"] >= -0.5:
        print(">>> NEAR-MISS: test sharpe near zero, structurally close to tradeable")
    else:
        print(">>> NOT READY: test sharpe too negative")

    out = Path(f"results/walk_forward_gate_{ticker}_{strategy}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    strategy = sys.argv[1] if len(sys.argv) > 1 else "bb_squeeze"
    ticker   = sys.argv[2] if len(sys.argv) > 2 else "NVDA"
    main(strategy, ticker)
