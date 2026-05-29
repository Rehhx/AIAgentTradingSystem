"""
agents/closed_loop_runner.py
----------------------------
end-to-end test of the closed loop: autonomous idea -> code_agent generates
implementation -> registers into STRATEGIES -> backtest -> risk.

uses one idea (lunch_lull_breakout_continuation) which is single-ticker and
the most implementable. cost: one Claude Code subprocess call (~30-90s).

usage:
    python agents/closed_loop_runner.py
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


IDEA = {
    "name":        "lunch_lull_breakout_continuation",
    "hypothesis":  ("Narrow 11:30-13:30 ET consolidation breakouts continue into the "
                    "close, because midday consolidation reflects a real liquidity vacuum "
                    "(institutional desks at lunch, gamma-positive dealer hedging). When "
                    "afternoon flow arrives, the breakout direction accelerates."),
    "description": ("Long when price breaks above the 11:30-13:30 ET range AFTER 13:30, "
                    "Short when it breaks below. Exit at close (15:55). Stop at "
                    "consolidation midpoint. Only fire on days where the 11:30-13:30 range "
                    "is less than 0.5 * 20-day ATR."),
    "params": {
        "consolidation_start_et":  "11:30",
        "consolidation_end_et":    "13:30",
        "entry_window_start_et":   "13:30",
        "entry_window_end_et":     "14:30",
        "exit_time_et":            "15:55",
        "max_range_pct_of_atr":    0.5,
        "breakout_volume_mult":    1.8,
        "min_consolidation_bars":  12,
    },
    "source_agent": "autonomous_agent",
}


def main():
    orch = Orchestrator()
    print(f"\nfeeding 1 novel idea through closed loop\n")
    print("=" * 70)
    print(f"  idea: {IDEA['name']}")
    print("=" * 70)

    sid = orch._run_strategy_lifecycle(IDEA)

    # _run_strategy_lifecycle returns sid only on full success; on failure we
    # need to find the just-created strategy. take the MOST RECENT one with
    # this name (created_at desc) to avoid grabbing a stale entry from a
    # previous run.
    if sid:
        final = orch.store.get_strategy(sid)
    else:
        matches = [s for s in orch.store._data["strategies"].values()
                   if s["name"] == IDEA["name"]]
        final = max(matches, key=lambda s: s.get("created_at", "")) if matches else None

    if not final:
        print("ERROR: strategy not found in store")
        return

    print(f"\n  status:    {final['status']}")
    print(f"  id:        {final['id']}")
    print(f"  code_path: {final.get('code_path')}")
    bt = final.get("backtest_results")
    if bt:
        if isinstance(bt, dict) and "sharpe" in bt:
            print(f"  backtest:  sharpe={bt['sharpe']:.3f}  "
                  f"dd={bt.get('max_drawdown', 0):.2%}  "
                  f"wr={bt.get('win_rate', 0):.2%}  "
                  f"trades={bt.get('total_trades', 0)}")
        else:
            print(f"  backtest:  {bt}")
    risk = final.get("risk_results")
    if risk:
        if isinstance(risk, dict) and "failures" in risk:
            print(f"  risk:      passed={risk.get('passed')}")
            if risk.get("failures"):
                print(f"             failures: {risk['failures']}")
            if risk.get("warnings"):
                print(f"             warnings: {risk['warnings']}")
        else:
            print(f"  risk:      {risk}")

    print(f"\n  history ({len(final.get('history', []))} transitions):")
    for h in final.get("history", []):
        keys = list(h.get("changes", {}).keys())
        print(f"    {h['timestamp']}  changed: {keys}")

    if final.get("code_path") and Path(final["code_path"]).exists():
        print(f"\n  generated code at {final['code_path']}:")
        print("  " + "-" * 60)
        code = Path(final["code_path"]).read_text()
        for line in code.splitlines()[:50]:
            print(f"  {line}")
        if len(code.splitlines()) > 50:
            print(f"  ... ({len(code.splitlines()) - 50} more lines)")


if __name__ == "__main__":
    main()
