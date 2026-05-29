"""
agents/research_runner.py
-------------------------
fire research_agent once via the Claude SDK. asks for fresh intraday equity
strategy ideas with cited sources. cache results to results/research_ideas.json
so we don't burn tokens every time.

usage:
    python agents/research_runner.py
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


def main():
    agent = ResearchAgent()
    result = agent.run({"payload": {
        "query": ("Find 3-5 intraday US equity trading strategies that fire at least "
                  "10 times per day per ticker on 1-5 minute bars. Focus on strategies "
                  "with documented microstructure rationale: VWAP-related, opening/closing "
                  "auction dynamics, gamma hedging flows, options expiry effects. Cite "
                  "specific papers or quant blogs."),
    }})

    if not result.get("success"):
        print(f"FAILED: {result.get('reason')}")
        return

    strategies = result.get("strategies_found") or []
    print(f"\nresearch_agent returned {len(strategies)} strategies\n")
    for i, s in enumerate(strategies, 1):
        print(f"--- {i} ---")
        print(json.dumps(s, indent=2))
        print()

    out = Path("results/research_ideas.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"strategies": strategies,
                   "run_at": datetime.now(timezone.utc).isoformat()},
                  f, indent=2, default=str)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
