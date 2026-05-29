"""
agents/options_research_agent.py
--------------------------------
specialized research agent for OPTIONS strategies. distinct from the general
research_agent (which proposes equity strategies on OHLCV) — options have
their own structural patterns: variance risk premium, gamma scalping, OPEX
flows, 0DTE microstructure, vol regime arbitrage.

returns ideas matched to options_agent's execution contract — each idea
specifies underlying, intent (bullish/bearish), moneyness, DTE range, and
sizing logic so options_code_agent can translate it into runnable form.

usage:
    agent = OptionsResearchAgent()
    result = agent.run({"payload": {"query": "0DTE SPY strategies with edge"}})
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ANTHROPIC_API_KEY

log = logging.getLogger("options_research_agent")


OPTIONS_RESEARCH_PROMPT = """\
You are a quantitative options strategist. Your job has TWO parts:

  PART A (discovery) — Find 2-3 documented options strategies (cite source):
    - vol risk premium, dispersion, gamma scalping, OPEX-flow, 0DTE momentum,
      calendar spreads, ratio backspreads, etc.
  PART B (invention) — Invent 2-3 NEW options strategies of your own design.
    Mark with source_url = "novel_invention".

EXECUTION CONTEXT — these strategies will trade via Alpaca's paper options
endpoint. Constraints:
  - SPY/QQQ are the cleanest underlyings (deepest options chains)
  - Single-leg or simple spreads (call/put long, vertical, basic condor)
    — multi-leg complex structures (butterflies, ratios) require manual entry
  - DTE 0 to 30 days available; weeklies + monthlies
  - You can size by # of contracts; we cap position at $5k risk per trade

EXISTING REGISTRY (don't propose duplicates):
{existing_strategies}

Recent findings to inform your designs:
  - Equity strategies in our system bleed at ~6bps round-trip cost.
  - Options widen the cost basis: $0.05 bid-ask × 100 multiplier = $5 per
    contract per side = ~50-100 bps round-trip on a $1-5 option. So options
    strategies need MORE edge to clear costs, not less.
  - 0DTE SPY options now ~45% of SPY options volume — gamma exposure spikes
    around big strikes.

Return JSON list of 4-6 ideas. Each must have:
  - name:            short snake_case identifier
  - description:     1-2 sentences
  - hypothesis:      market microstructure / behavioral reason it works
  - underlying:      ticker (SPY, QQQ, etc.)
  - signal_rule:     entry condition on UNDERLYING price/indicators
                     (so it can be backtested with OHLCV)
  - direction_logic: how signal_value (-1/0/+1) maps to bullish/bearish intent
  - moneyness:       "atm" | "5pct_otm" | "10pct_otm" | "5pct_itm" | etc.
  - dte_target:      target days-to-expiry (int) or "0DTE"
  - max_hold_bars:   exit force-flat after N 1-min bars
  - profit_target_pct: as % of premium paid (e.g. 50 = take profit at 1.5x)
  - stop_loss_pct:   as % of premium paid (e.g. 50 = exit at 0.5x)
  - structure:       "long_call" | "long_put" | "call_vertical" |
                     "put_vertical" | "iron_condor" | "calendar"
  - source_url:      paper/blog URL or "novel_invention"
  - kind:            "discovery" or "invention"

Prefer ideas that fire 20-200 times per year (real edge has signal density,
not curve-fit one-offs).
"""


def _existing_summary() -> str:
    """reuse the equity-registry summary so we don't duplicate THAT either."""
    try:
        from agents.research_agent import _existing_strategies_summary
        return _existing_strategies_summary()
    except Exception:
        return "  (unable to load registry)"


class OptionsResearchAgent:
    """matches BaseAgent.run(task) contract — usable from orchestrator."""

    def __init__(self, store=None):
        self.store = store
        self.log   = logging.getLogger("options_research_agent")

    def run(self, task: dict) -> dict:
        if not ANTHROPIC_API_KEY:
            return self._failure("ANTHROPIC_API_KEY not configured")

        query = task.get("payload", {}).get("query",
            "Find options strategies on SPY/QQQ tradeable via Alpaca paper.")

        try:
            from agents._claude_sdk import ask_claude
            system_prompt = OPTIONS_RESEARCH_PROMPT.format(
                existing_strategies=_existing_summary(),
            )
            response = ask_claude(
                prompt        = query,
                system_prompt = system_prompt,
                allowed_tools = ["WebSearch", "WebFetch"],
                model         = "claude-opus-4-7",
                debug_log_path = "results/options_research_raw.txt",
            )
            self.log.info(f"options_research_agent received {len(response)} chars from SDK")
        except ImportError:
            return self._failure("claude_agent_sdk not installed")
        except Exception as e:
            return self._failure(f"sdk call failed: {e}")

        # reuse research_agent's robust parser
        from agents.research_agent import ResearchAgent
        ideas = ResearchAgent()._parse_strategies(response)
        if not ideas and response:
            self.log.warning(f"parser found 0 ideas in {len(response)}-char response")

        return self._success(ideas=ideas, raw_len=len(response or ""))

    def _success(self, **kw): return {"success": True, "agent": "options_research_agent", **kw}
    def _failure(self, reason, **kw):
        self.log.warning(f"options_research_agent failed | {reason}")
        return {"success": False, "agent": "options_research_agent", "reason": reason, **kw}
