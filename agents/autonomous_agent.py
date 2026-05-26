"""
agents/autonomous_agent.py
--------------------------
generates novel strategy ideas from first principles. no web searches, no
literature lookup — just structured reasoning about market microstructure,
time-of-day effects, volatility regimes, and cross-asset relationships.

implementation note:
  - scaffold for claude-agent-sdk Agent class with NO tools (pure reasoning).
  - the prompt is the product: it conditions claude to think like a quant
    researcher generating hypotheses, not summarizing existing strategies.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ANTHROPIC_API_KEY

log = logging.getLogger("autonomous_agent")


AUTONOMOUS_PROMPT = """\
You are a senior quant researcher inventing trading hypotheses from first
principles. You do NOT search the web or cite existing strategies. You reason
from market structure.

For each idea, consider:
  - the market microstructure that would create the pattern
  - the behavioral or structural reason participants would leave it on the table
  - the timeframe it operates on (1-60 minute bars)
  - what regime makes it work and what regime breaks it
  - what would falsify the hypothesis

Return up to 3 ideas as JSON. Each must have:
  - name:           snake_case
  - hypothesis:     1-2 sentences on WHY this should work
  - mechanism:      the microstructure explanation
  - timeframe:      "1min", "5min", "15min", or "1h"
  - direction:      "long_only", "short_only", or "both"
  - params:         dict of starting parameters
  - regime_fit:     {"works_in": [...], "breaks_in": [...]}  using labels:
                    trending | mean_reversion | chop | breakout
  - falsifier:      what test would prove the hypothesis wrong

Quality over quantity. Reject vague ideas you can't defend mechanically.
"""


class AutonomousAgent:
    """matches BaseAgent.run(task) contract used by orchestrator."""

    def __init__(self, store=None):
        self.store = store
        self.log   = logging.getLogger("autonomous_agent")

    def run(self, task: dict) -> dict:
        if not ANTHROPIC_API_KEY:
            return self._failure("ANTHROPIC_API_KEY not configured")

        prompt_seed = task.get("payload", {}).get("seed", "Generate three novel intraday equity strategy hypotheses.")

        try:
            from agents._claude_sdk import ask_claude
            response = ask_claude(
                prompt        = prompt_seed,
                system_prompt = AUTONOMOUS_PROMPT,
                allowed_tools = [],   # no tools — pure reasoning
                model         = "claude-opus-4-7",
            )
            ideas = self._parse_ideas(response)
        except ImportError:
            self.log.warning("claude_agent_sdk not installed — returning empty idea list")
            ideas = []
        except Exception as e:
            return self._failure(f"sdk call failed: {e}")

        return self._success(ideas=ideas)

    def _parse_ideas(self, response) -> list:
        import json, re
        text = response if isinstance(response, str) else str(response)
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return []
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    def _success(self, **kw): return {"success": True, "agent": "autonomous_agent", **kw}
    def _failure(self, reason, **kw):
        self.log.warning(f"autonomous failed | {reason}")
        return {"success": False, "agent": "autonomous_agent", "reason": reason, **kw}
