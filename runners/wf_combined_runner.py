"""
agents/wf_combined_runner.py
----------------------------
two walk-forwards in one run:

  1. extreme_bar_fade — aggregate sweep across 5 tickers, single (best params)
  2. bb_squeeze       — per-ticker walk-forward, surfaces ticker-specific tuning

train = first 70% of 2022-2025 data. test = last 30%.
expect ~5-10 minutes total compute, no API calls.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


EXTREME_GRID  = {"body_atr_mult": [2.0, 3.0, 4.0], "hold_bars": [5, 10, 20]}
GAP_GRID      = {"gap_threshold_pct": [0.005, 0.010, 0.015], "exit_time_et": ["11:00", "12:00", "13:00"]}
BB_GRID       = {"bb_period":     [15, 20, 30],    "bb_std":    [1.5, 2.0, 2.5]}
TICKERS       = ["SPY", "QQQ", "TSLA", "NVDA", "AAPL"]


def fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)


def main():
    out = {"run_at": datetime.now(timezone.utc).isoformat()}

    # ------------------------------------------------------------------
    # 0. overnight_gap_fade — aggregate sweep (NEW)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("  overnight_gap_fade — aggregate walk-forward (5 tickers)")
    print(f"  grid: {GAP_GRID}")
    print("=" * 70)
    r0 = walk_forward_optimize("overnight_gap_fade", GAP_GRID, tickers=TICKERS)
    if r0.get("success"):
        print(f"\n  best params  : {r0['best_params']}")
        print(f"  train sharpe : {r0['train_sharpe']:.3f}")
        print(f"  test  sharpe : {r0['test_sharpe']:.3f}")
        print(f"  overfit gap  : {r0['overfit_gap']:+.3f}")
    else:
        print(f"  FAILED: {r0.get('reason')}")
    out["overnight_gap_fade_aggregate"] = r0

    # ------------------------------------------------------------------
    # 1. extreme_bar_fade — aggregate sweep
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  extreme_bar_fade — aggregate walk-forward (5 tickers)")
    print(f"  grid: {EXTREME_GRID}")
    print("=" * 70)
    r1 = walk_forward_optimize("extreme_bar_fade", EXTREME_GRID, tickers=TICKERS)
    if r1.get("success"):
        print(f"\n  best params  : {r1['best_params']}")
        print(f"  train sharpe : {r1['train_sharpe']:.3f}")
        print(f"  test  sharpe : {r1['test_sharpe']:.3f}")
        print(f"  overfit gap  : {r1['overfit_gap']:+.3f}")
        print(f"\n  top 5 train results:")
        for row in r1["all_results"][:5]:
            print(f"    {row['params']}  train_sharpe={row['train_sharpe']:.3f}")
    else:
        print(f"  FAILED: {r1.get('reason')}")
    out["extreme_bar_fade_aggregate"] = r1

    # ------------------------------------------------------------------
    # 2. bb_squeeze — per-ticker walk-forward
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  bb_squeeze — per-ticker walk-forward")
    print(f"  grid: {BB_GRID}")
    print("=" * 70)
    per_ticker = {}
    for t in TICKERS:
        print(f"\n  --- {t} ---")
        r = walk_forward_optimize("bb_squeeze", BB_GRID, tickers=[t])
        if not r.get("success"):
            print(f"  FAILED: {r.get('reason')}")
            per_ticker[t] = r
            continue
        print(f"    best params : {r['best_params']}")
        print(f"    train sharpe: {r['train_sharpe']:.3f}    test sharpe: {r['test_sharpe']:.3f}    "
              f"overfit gap: {r['overfit_gap']:+.3f}")
        per_ticker[t] = r
    out["bb_squeeze_per_ticker"] = per_ticker

    # ------------------------------------------------------------------
    # summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  summary")
    print("=" * 70)
    print(f"\n  overnight_gap_fade (aggregate):")
    if r0.get("success"):
        print(f"    best={r0['best_params']}  train={r0['train_sharpe']:+.3f}  "
              f"test={r0['test_sharpe']:+.3f}  gap={r0['overfit_gap']:+.3f}")

    print(f"\n  extreme_bar_fade (aggregate):")
    if r1.get("success"):
        print(f"    best={r1['best_params']}  train={r1['train_sharpe']:+.3f}  "
              f"test={r1['test_sharpe']:+.3f}  gap={r1['overfit_gap']:+.3f}")

    print(f"\n  bb_squeeze per-ticker:")
    print(f"  {'ticker':<8} {'best params':<35} {'train':>8} {'test':>8} {'gap':>8}")
    print("  " + "-" * 70)
    for t, r in per_ticker.items():
        if r.get("success"):
            params_str = ", ".join(f"{k}={v}" for k, v in r["best_params"].items())
            print(f"  {t:<8} {params_str:<35} {r['train_sharpe']:>+8.3f} {r['test_sharpe']:>+8.3f} {r['overfit_gap']:>+8.3f}")
        else:
            print(f"  {t:<8} (failed)")

    output_path = Path("results/wf_combined_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  saved to {output_path}")


if __name__ == "__main__":
    main()
