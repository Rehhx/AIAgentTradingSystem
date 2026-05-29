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


RESEARCH_PROMPT_TEMPLATE = """\
You are a quantitative research analyst.

=== MISSION (urgent) ===
We need a DEPLOYABLE strategy targeting a 10-20% ANNUAL RETURN that passes our
risk gate: Sharpe >= 0.8, max drawdown >= -15%, win rate >= 45%, AND >= 100
trades/year. There is a hard deadline — propose ideas that can be validated fast.

DECISIVE LESSON: 1-MINUTE INTRADAY IS DEAD. At our 6 bps round-trip cost, intraday
strategies bleed (dozens of -1 to -30 Sharpe examples in the ledger below). What
WORKS is DAILY / multi-day holds (2-20 trading days), where cost is negligible.
Our deployed winners are all daily. Propose DAILY / SWING strategies with a real
economic mechanism (mean reversion, trend, carry, seasonality, cross-sectional
momentum) — NOT intraday scalps.

Your job has TWO parts:

  PART A (discovery) — Search arxiv, SSRN, QuantConnect, Robot Wealth, etc.
  for 2-3 strategies that match the user's query and are NOT in our existing
  registry below. Cite the source URL for each.

  PART B (invention) — Design 2-3 NEW strategies of your own, not from any
  paper. Use your understanding of market microstructure to invent novel
  ideas. Mark these with source_url = "novel_invention". The mechanism field
  is critical: explain WHY this should work.

EXISTING STRATEGIES IN OUR REGISTRY (do NOT propose duplicates or trivial
variants of these — we have already tested them):
{existing_strategies}

Recent backtest findings to inform your designs:
- DAILY holds work: RSI-2 mean reversion (buy short-term dips in an uptrend)
  gives out-of-sample Sharpe ~1.1 at ~112 trades/yr; a blended daily book
  (RSI-2 + Donchian breakout + 50/200 trend) passes risk at Sharpe 1.09, +8.6%/yr.
- 1-minute intraday strategies ALL fail: cost drag dominates (trade count tracks
  loss magnitude almost perfectly). Do NOT propose them.
- To reach 10-20%/yr we likely need trend / cross-sectional momentum on daily
  bars (these run hotter) paired with mean reversion for drawdown control.

Return JSON list of up to 6 ideas. Each must have:
  - name:        short identifier, snake_case
  - description: 1-2 sentences
  - hypothesis:  market microstructure / behavioral reason it works
  - params:      dict of default parameters
  - timeframe:   "1d" or "swing" (PREFERRED, 2-20 day hold). Avoid intraday.
  - direction:   "long_only", "short_only", or "both"
  - source_url:  paper URL, or "novel_invention" for Part B
  - kind:        "discovery" or "invention"

For invention entries: prioritize DAILY / multi-day strategies that fire
100-500 times per YEAR with a verifiable structural mechanism, target a 10-20%
annual return, and could pass the risk gate. Do NOT curve-fit intraday indicators.
"""


def _existing_strategies_summary() -> str:
    """build a compact text summary of registered STRATEGIES for the prompt."""
    try:
        from agents.backtesting_agent import STRATEGIES, STRATEGY_REGIME_AFFINITY
    except Exception:
        return "  (unable to load registry)"
    lines = []
    for key, entry in STRATEGIES.items():
        params = entry[1] if len(entry) > 1 else {}
        # only show non-execution params
        keys_to_show = [k for k in params if k not in
                        ("active", "stop_atr_mult", "max_hold_bars",
                         "disable_atr_stop")][:4]
        param_str = ", ".join(f"{k}={params[k]}" for k in keys_to_show)
        affinity = ", ".join(sorted(STRATEGY_REGIME_AFFINITY.get(key, []))) or "any"
        lines.append(f"  - {key}({param_str})  [regimes: {affinity}]")
    base = "\n".join(lines) if lines else "  (registry empty)"

    # append the persistent ledger so agents still know about strategies whose
    # code was removed (tried & rejected) and won't re-propose them.
    try:
        from agents.strategy_ledger import ledger_summary_for_prompt
        led = ledger_summary_for_prompt()
        if led and "no prior" not in led:
            base += "\n\nPERSISTENT LEDGER — every strategy already evaluated:\n" + led
    except Exception:
        pass
    return base


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
            # build the system prompt with the current registry inlined so the
            # SDK knows what NOT to propose and what we've already tried.
            system_prompt = RESEARCH_PROMPT_TEMPLATE.format(
                existing_strategies=_existing_strategies_summary(),
            )
            response = ask_claude(
                prompt        = task.get("payload", {}).get("query", "intraday equity strategies 2024"),
                system_prompt = system_prompt,
                allowed_tools = ["WebSearch", "WebFetch"],
                model         = "claude-opus-4-7",
                debug_log_path = "results/research_raw_response.txt",
            )
            self.log.info(f"research_agent received {len(response)} chars from SDK")
            strategies = self._parse_strategies(response)
            if not strategies and response:
                self.log.warning(f"parser found 0 strategies in {len(response)}-char response; "
                                 f"raw saved to results/research_raw_response.txt")
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
        """
        try several extraction strategies — model output may wrap the JSON in
        ```json ... ``` fences, embed it in prose, or include multiple arrays.
        """
        import json, re
        text = response if isinstance(response, str) else str(response)
        if not text.strip():
            return []

        # 1. fenced ```json ... ``` block
        fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                pass

        # 2. greedy outer brackets — for cases where the array is the entire body
        greedy = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
        if greedy:
            try:
                return json.loads(greedy.group(0))
            except json.JSONDecodeError:
                pass

        # 3. try to find an array by walking forward from each '['
        for i, ch in enumerate(text):
            if ch != "[":
                continue
            depth = 0
            for j in range(i, len(text)):
                if text[j] == "[":
                    depth += 1
                elif text[j] == "]":
                    depth -= 1
                    if depth == 0:
                        candidate = text[i:j + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                                return parsed
                        except json.JSONDecodeError:
                            pass
                        break
        return []

    def _success(self, **kw): return {"success": True, "agent": "research_agent", **kw}
    def _failure(self, reason, **kw):
        self.log.warning(f"research failed | {reason}")
        return {"success": False, "agent": "research_agent", "reason": reason, **kw}
