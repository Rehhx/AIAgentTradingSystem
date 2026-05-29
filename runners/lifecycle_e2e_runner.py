"""
agents/lifecycle_e2e_runner.py
------------------------------
end-to-end test of the orchestrator's strategy lifecycle on real ideas from
the autonomous_agent. ideas are loaded from the most recent autonomous output
file so we don't burn fresh Claude tokens every time.

each idea is registered, dispatched to backtest, then to risk, then (if
approved) to code generation. final state is read from results/store.json.

usage:
    python agents/lifecycle_e2e_runner.py
"""

import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from orchestrator import Orchestrator, AgentName

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# the 3 ideas the autonomous_agent generated in the prior smoke test.
# checked in as a fixture so this script is deterministic and doesn't burn
# new tokens every run.
AUTONOMOUS_IDEAS = [
    {
        "name":        "opening_auction_imbalance_fade",
        "hypothesis":  "Cash open over-extends opening auction imbalance; reverts within 30-45 min.",
        "params":      {"imbalance_lookback_min": 5, "entry_minute_after_open": 10,
                        "extension_threshold_atr": 1.25, "exit_minute": 45, "stop_atr": 0.75},
        "source_agent": "autonomous_agent",
    },
    {
        "name":        "qqq_spy_dispersion_snapback",
        "hypothesis":  "QQQ-vs-SPY beta-adjusted residual mean-reverts intraday.",
        "params":      {"beta_lookback_min": 60, "zscore_lookback_min": 30,
                        "entry_z": 2.0, "exit_z": 0.3, "max_hold_min": 20, "stop_z": 3.5},
        "source_agent": "autonomous_agent",
    },
    {
        "name":        "lunch_lull_breakout_continuation",
        "hypothesis":  "Narrow 11:30-13:30 ET consolidation breakouts continue into close.",
        "params":      {"max_range_pct_of_atr": 0.5, "breakout_volume_mult": 1.8,
                        "min_consolidation_bars": 12},
        "source_agent": "autonomous_agent",
    },
]


def main():
    orch = Orchestrator()
    print(f"\nfeeding {len(AUTONOMOUS_IDEAS)} autonomous ideas through the lifecycle\n")

    registered_ids = []
    for idea in AUTONOMOUS_IDEAS:
        print("=" * 70)
        print(f"  idea: {idea['name']}")
        print("=" * 70)
        sid = orch._run_strategy_lifecycle(idea)
        if sid:
            registered_ids.append(sid)
            print(f"  -> reached approved/paper_trading | id={sid}")
        else:
            # idea was rejected somewhere; find its id by name in the store
            for s in orch.store._data["strategies"].values():
                if s["name"] == idea["name"] and s["source_agent"] == "autonomous_agent":
                    registered_ids.append(s["id"])
                    break
        print()

    print("=" * 70)
    print("  final state from store.json")
    print("=" * 70)
    for sid in registered_ids:
        s = orch.store.get_strategy(sid)
        if not s:
            continue
        print(f"\n  {s['name']}  (id={sid})")
        print(f"    status:   {s['status']}")
        bt = s.get("backtest_results")
        if bt:
            if isinstance(bt, dict) and "sharpe" in bt:
                print(f"    backtest: sharpe={bt['sharpe']:.3f}  dd={bt.get('max_drawdown', 0):.2%}  "
                      f"wr={bt.get('win_rate', 0):.2%}  trades={bt.get('total_trades', 0)}")
            else:
                print(f"    backtest: {bt}")
        else:
            print(f"    backtest: (none)")
        risk = s.get("risk_results")
        if risk:
            if isinstance(risk, dict) and "failures" in risk:
                print(f"    risk:     passed={risk.get('passed')}  failures={risk.get('failures')}")
            else:
                print(f"    risk:     {risk}")
        print(f"    history:  {len(s.get('history', []))} transitions")

    summary = orch.store.summary()
    print(f"\n  store summary: {json.dumps(summary, indent=2, default=str)}")


if __name__ == "__main__":
    main()
