"""
agents/research_runner_round2.py
--------------------------------
second-round invocation of research_agent via the claude sdk. asks for
strategies that match the empirical edge we found in round 1: low-vol
large-cap stocks (the regime where bb_squeeze posted positive Sharpe on
MSFT/CAT) AND options-based strategies (since the user explicitly asked).

caches results to results/research_ideas_round2.json so the next backtest
doesn't burn another tokens.

usage:
    python agents/research_runner_round2.py
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from agents.research_agent import ResearchAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


QUERY = (
    "Find 4-6 quantitative trading strategies that fit these constraints: "
    "(a) at least 2 should target low-volatility large-cap US stocks (think MSFT, "
    "AAPL, JPM, CAT) on 1-5 minute bars — we have empirical evidence that breakout "
    "strategies work on this subset of names. "
    "(b) at least 2 should be OPTIONS strategies on SPY/QQQ tradeable via Alpaca's "
    "paper options API — focus on directional 0DTE/short-dated, gamma scalping, or "
    "volatility-risk-premium harvesting. "
    "Avoid strategies that need order book / level-2 data (we only have 1-min OHLCV). "
    "Cite specific papers, quantitative blog posts, or backtest writeups for each."
)


def main():
    agent = ResearchAgent()
    print("calling research_agent (this may take 30-90 seconds)...")
    result = agent.run({"payload": {"query": QUERY}})

    if not result.get("success"):
        print(f"FAILED: {result.get('reason')}")
        return

    strategies = result.get("strategies_found") or []

    # save FIRST — print can fail on Windows cp1252 if the model emits unicode
    # math symbols (σ, Δ, etc.). don't lose the data because of a print bug.
    out = Path("results/research_ideas_round2.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"strategies": strategies, "query": QUERY,
                   "run_at": datetime.now(timezone.utc).isoformat()},
                  f, indent=2, default=str, ensure_ascii=False)
    print(f"saved to {out}")

    print(f"\nresearch_agent returned {len(strategies)} strategies\n")

    def _safe(s):
        """encode to cp1252-safe by replacing characters that don't fit."""
        if not isinstance(s, str):
            return s
        return s.encode("ascii", "replace").decode("ascii")

    for i, s in enumerate(strategies, 1):
        print(f"--- {i}. {_safe(s.get('name', '?'))} ---")
        print(f"  description : {_safe(s.get('description', '?'))}")
        hypo = s.get('hypothesis', '?')
        print(f"  hypothesis  : {_safe(hypo)[:200]}{'...' if len(hypo) > 200 else ''}")
        print(f"  timeframe   : {_safe(s.get('timeframe', '?'))}")
        print(f"  direction   : {_safe(s.get('direction', '?'))}")
        print(f"  source_url  : {_safe(s.get('source_url', '?'))}")
        print()


if __name__ == "__main__":
    main()
