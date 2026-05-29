"""
agents/bb_squeeze_regime_wf.py
------------------------------
regime-specific walk-forward for bb_squeeze. for each market regime
(bull, bear, high_vol, neutral) separately:
  - sweep params on the train split, restricting entries to that regime
  - measure best params on the test split (also restricting to regime)
  - report train vs test Sharpe, plus the regime-adjusted Sharpe

if any regime shows train > 0 AND test > 0 with overfit_gap < 1.0,
bb_squeeze has REAL conditional edge in that regime.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from agents.backtesting_agent import walk_forward_with_market_regime

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")


PARAM_GRID = {
    "bb_period":     [15, 20, 25],
    "bb_std":        [2.0, 2.5],
    "kc_mult":       [1.5, 2.0],
    "stop_atr_mult": [2.0, 2.5],
}

TICKERS  = ["SPY", "QQQ", "AAPL", "MSFT", "CAT"]
REGIMES  = ["bull", "bear", "high_vol", "neutral"]


def main():
    out = {}
    for regime in REGIMES:
        print(f"\n{'='*60}\n  bb_squeeze WF — regime={regime}\n{'='*60}")
        res = walk_forward_with_market_regime(
            strategy_name = "bb_squeeze",
            param_grid    = PARAM_GRID,
            target_regime = regime,
            tickers       = TICKERS,
            start         = "2022-01-01",
            end           = "2025-01-01",
            train_pct     = 0.7,
        )
        if not res.get("success"):
            print(f"FAILED: {res.get('reason')}")
            out[regime] = {"error": res.get("reason")}
            continue

        print(f"  best params      : {res['best_params']}")
        print(f"  train sharpe     : {res['train_sharpe']}  ({res['train_trades']} trades)")
        print(f"  test  sharpe     : {res['test_sharpe']}  ({res['test_trades']} trades)")
        print(f"  overfit gap      : {res['overfit_gap']}")
        print(f"  regime fraction  : {res['regime_fraction']:.2%}")
        print(f"  adjusted test SR : {res['regime_adjusted_test_sharpe']} (sharpe / sqrt(fraction))")
        out[regime] = {k: v for k, v in res.items() if k != "all_results"}

    # cross-regime summary
    print(f"\n{'='*72}")
    print(f"  CROSS-REGIME SUMMARY for bb_squeeze")
    print(f"{'='*72}")
    print(f"  {'regime':<10}  {'train':>8}  {'test':>8}  {'gap':>6}  {'adj.test':>10}  {'verdict'}")
    for regime in REGIMES:
        r = out.get(regime, {})
        if "error" in r:
            print(f"  {regime:<10}  FAILED: {r['error']}")
            continue
        verdict = "REAL EDGE" if r["test_sharpe"] > 0 and r["overfit_gap"] < 1.0 else "fail"
        print(f"  {regime:<10}  {r['train_sharpe']:>8.2f}  {r['test_sharpe']:>8.2f}  "
              f"{r['overfit_gap']:>6.2f}  {r['regime_adjusted_test_sharpe'] if r['regime_adjusted_test_sharpe'] is not None else '-':>10}  {verdict}")

    path = Path("results/bb_squeeze_regime_wf.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"strategy": "bb_squeeze", "results": out,
                   "run_at": datetime.now(timezone.utc).isoformat()},
                  f, indent=2, default=str)
    print(f"\nsaved to {path}")


if __name__ == "__main__":
    main()
