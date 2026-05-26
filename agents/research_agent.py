"""
agents/research_agent.py
------------------------
scans arxiv, quant blogs, and known strategy databases for new strategy
ideas. embeds findings into research_store for similarity search.

implementation note:
  - this is a scaffold. the production version uses claude-agent-sdk's
    Agent class with web_search + web_fetch tools. it sends a structured
    prompt asking for strategies in JSON, validates them with pydantic,
    and pushes each into research_store.add_document().
  - the orchestrator already handles ideas->backtest, so this agent's
    only job is to return a dict with {"strategies_found": [...]}.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ANTHROPIC_API_KEY

log = logging.getLogger("research_agent")


RESEARCH_PROMPT = """\
You are a quantitative research analyst. Search recent arxiv papers, quant
blogs (QuantConnect, SSRN, Robot Wealth), and known strategy databases for
intraday trading strategies on US equities using 1-minute bar data.

Return up to 5 strategy ideas as JSON. Each must have:
  - name:        short identifier, snake_case
  - description: 1-2 sentences
  - hypothesis:  the market microstructure / behavioral reason it works
  - params:      dict of default parameters
  - timeframe:   "1min", "5min", "15min", or "1h"
  - direction:   "long_only", "short_only", or "both"
  - source_url:  the paper or blog you got it from

Only include strategies you can cite a source for.
"""


class ResearchAgent:
    """matches BaseAgent.run(task) contract used by orchestrator."""

    def __init__(self, store=None, research_store=None):
        self.store          = store
        self.research_store = research_store
        self.log            = logging.getLogger("research_agent")

    def run(self, task: dict) -> dict:
        if not ANTHROPIC_API_KEY:
            return self._failure("ANTHROPIC_API_KEY not configured")

        try:
            from agents._claude_sdk import ask_claude
            response = ask_claude(
                prompt        = task.get("payload", {}).get("query", "intraday equity strategies 2024"),
                system_prompt = RESEARCH_PROMPT,
                allowed_tools = ["WebSearch", "WebFetch"],
                model         = "claude-opus-4-7",
            )
            strategies = self._parse_strategies(response)
        except ImportError:
            self.log.warning("claude_agent_sdk not installed — returning empty strategy list")
            strategies = []
        except Exception as e:
            return self._failure(f"sdk call failed: {e}")

        # optionally index into research_store for future similarity search
        if self.research_store is not None and strategies:
            for s in strategies:
                try:
                    self.research_store.add_document(s)
                except Exception as e:
                    self.log.warning(f"research_store.add_document failed: {e}")

        return self._success(strategies_found=strategies)

    def _parse_strategies(self, response) -> list:
        """expects the SDK response to contain JSON-formatted strategies."""
        import json, re
        text = response if isinstance(response, str) else str(response)
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return []
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    def _success(self, **kw): return {"success": True, "agent": "research_agent", **kw}
    def _failure(self, reason, **kw):
        self.log.warning(f"research failed | {reason}")
        return {"success": False, "agent": "research_agent", "reason": reason, **kw}
