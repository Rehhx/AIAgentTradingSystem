"""
agents/full_pipeline_runner.py
------------------------------
top-level autonomous research pipeline. one command, end-to-end:

  1. autonomous_agent (Claude SDK)   -> N strategy ideas
  2. for each idea:
       code_agent (Claude SDK)        -> python implementation + register
       backtesting_agent              -> real metrics on 1m bar data
       risk_agent                     -> threshold gate (config.RISK)
       param_tuner (optional retries) -> relax knobs and re-submit if rejected
  3. anything that passes risk        -> status = paper_trading
  4. final summary printed and persisted in store.json

cost: roughly one autonomous SDK call (~30-60s) + one code SDK call per idea
(~30-90s each) + ~30s per ticker per backtest. for 3 ideas across 5 tickers
expect 5-10 minutes and a handful of Claude/OpenAI cents.

usage:
    python agents/full_pipeline_runner.py [--ideas N] [--max-retries 0]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from orchestrator import Orchestrator, AgentName, StrategyStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ideas",       type=int, default=2,
                        help="ideas to ask the autonomous agent for (default 2)")
    parser.add_argument("--max-retries", type=int, default=0,
                        help="param-tuning retries on risk rejection (default 0, no retry)")
    args = parser.parse_args()

    orch = Orchestrator()

    print(f"\n{'=' * 70}")
    print(f"  full pipeline: {args.ideas} ideas, up to {args.max_retries} retries each")
    print(f"{'=' * 70}\n")

    # step 1 — generate ideas
    print(f"[1/3] dispatching autonomous_agent to generate {args.ideas} ideas...")
    auto_result = orch.dispatch(AgentName.AUTONOMOUS, "generate", {
        "seed": f"Generate {args.ideas} novel intraday equity strategy hypotheses for SPY/QQQ on 1-5 minute bars.",
    })
    if not auto_result.get("success"):
        print(f"  FAILED: {auto_result.get('reason')}")
        return

    ideas = auto_result.get("ideas", [])
    print(f"  got {len(ideas)} ideas: {[i.get('name') for i in ideas]}")
    if not ideas:
        print("  nothing to run; exiting")
        return

    # step 2 — run each through the lifecycle
    print(f"\n[2/3] running lifecycle for each idea...")
    outcomes = []
    for i, idea in enumerate(ideas, 1):
        idea.setdefault("source_agent", "autonomous_agent")
        print(f"\n  ({i}/{len(ideas)}) {idea.get('name')}")
        sid = orch._run_strategy_lifecycle(idea)

        # find the strategy in store regardless of success
        matches = [s for s in orch.store._data["strategies"].values()
                   if s["name"] == idea.get("name")]
        record = max(matches, key=lambda s: s.get("created_at", "")) if matches else None

        if record:
            status = record["status"]
            bt     = record.get("backtest_results") or {}
            risk   = record.get("risk_results") or {}
            sharpe = bt.get("sharpe") if isinstance(bt, dict) else None
            print(f"      status: {status}")
            if isinstance(sharpe, (int, float)):
                print(f"      sharpe: {sharpe:.3f}  trades: {bt.get('total_trades', 0)}")
            if isinstance(risk, dict) and risk.get("failures"):
                print(f"      failures: {risk['failures']}")
            outcomes.append({"id": record["id"], "name": idea.get("name"), "status": status,
                             "sharpe": sharpe, "trades": bt.get("total_trades", 0) if isinstance(bt, dict) else 0})

            # param-tuning retry — disabled by default. when enabled, asks the
            # autonomous_agent to relax knobs on the rejected strategy.
            if args.max_retries > 0 and status in (StrategyStatus.REJECTED, "rejected"):
                outcomes[-1]["retries"] = _retry_with_tuned_params(orch, idea, record, args.max_retries)

    # step 3 — summary
    print(f"\n[3/3] pipeline summary:")
    print(f"  {'name':<45} {'status':<20} {'sharpe':>8} {'trades':>8}")
    print(f"  {'-' * 45} {'-' * 20} {'-' * 8} {'-' * 8}")
    for o in outcomes:
        sh = f"{o['sharpe']:.3f}" if isinstance(o.get('sharpe'), (int, float)) else "-"
        print(f"  {o['name']:<45} {str(o['status']):<20} {sh:>8} {o['trades']:>8}")
    print()

    summary = orch.store.summary()
    print(f"  store summary: total_strategies={summary['total_strategies']}, "
          f"by_status={summary['by_status']}")
    paper = [o for o in outcomes if "paper" in str(o.get("status", "")).lower()]
    if paper:
        print(f"\n  PROMOTED TO PAPER_TRADING:")
        for o in paper:
            print(f"    {o['name']}  (id={o['id']})")


def _retry_with_tuned_params(orch, idea: dict, record: dict, max_retries: int) -> list:
    """
    on risk rejection, ask the autonomous_agent to suggest relaxed params for
    the same hypothesis and re-submit through the lifecycle. bounded by
    max_retries to cap cost.
    """
    history = []
    failures = record.get("risk_results", {}).get("failures", [])
    if not failures:
        return history

    for attempt in range(1, max_retries + 1):
        print(f"      retry {attempt}/{max_retries} — asking autonomous_agent for relaxed params")
        prompt_seed = (
            f"The strategy '{idea['name']}' was rejected with these failures: {failures}. "
            f"Original params were: {idea.get('params')}. "
            f"Suggest a single revised version of the SAME hypothesis with relaxed params "
            f"that would generate more trades while preserving the structural edge. "
            f"Return JSON in the same schema as autonomous_agent."
        )
        result = orch.dispatch(AgentName.AUTONOMOUS, "tune", {"seed": prompt_seed})
        revised = (result.get("ideas") or [{}])[0]
        if not revised.get("params"):
            print(f"      retry {attempt}: agent didn't return usable params")
            break

        # carry the original hypothesis text; only swap params
        retry_idea = {**idea, "params": revised["params"]}
        sid = orch._run_strategy_lifecycle(retry_idea)
        matches = [s for s in orch.store._data["strategies"].values()
                   if s["name"] == idea.get("name")]
        latest = max(matches, key=lambda s: s.get("created_at", "")) if matches else None
        history.append({"attempt": attempt, "params": revised["params"],
                        "status": latest["status"] if latest else "unknown"})
        if latest and "paper" in str(latest["status"]).lower():
            print(f"      retry {attempt}: PASSED — promoted to paper_trading")
            return history
        failures = (latest.get("risk_results") or {}).get("failures", []) if latest else failures
    return history


if __name__ == "__main__":
    main()
