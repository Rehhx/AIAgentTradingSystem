"""
agents/options_code_agent.py
----------------------------
generates options strategy modules from options_research_agent specs.

IMPORTANT — backtest interpretation:
  the modules generated here are backtested using UNDERLYING price moves.
  we DON'T have a historical options chain, so we can't compute exact option
  P&L over the backtest period. instead we use the UNDERLYING signal as a
  proxy. the displayed Sharpe/PnL is for HOLDING THE UNDERLYING in the
  signal's direction — it represents the directional edge, not the actual
  option payoff.

  for actual options paper-trading via options_agent, the generated module
  ALSO exposes options_intent(signal, df_row, params) returning the request
  dict that options_agent.run() consumes. so the same code is dual-use:
    - backtest: signals() on OHLCV
    - live:     options_intent() on the latest signal

contract — each generated module exports:
  def signals(df: pd.DataFrame, params: dict) -> pd.Series        # ints {-1,0,1}
  def options_intent(signal: int, last_close: float, params: dict) -> dict
      # returns: {"underlying", "side", "intent", "qty", "moneyness", "dte_max", "structure"}
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ANTHROPIC_API_KEY

log = logging.getLogger("options_code_agent")


OPTIONS_CODE_PROMPT = """\
You are a python engineer implementing an OPTIONS trading strategy.

CONTRACT — your module must export TWO functions:

    def signals(df: pd.DataFrame, params: dict) -> pd.Series
        # signal on the UNDERLYING (-1=bearish bias, +1=bullish, 0=flat)
        # this is what the backtest engine uses
        # df has columns: open, high, low, close, volume

    def options_intent(signal: int, last_close: float, params: dict) -> dict
        # translates a current signal into an options order request
        # returns dict shaped for agents.options_agent.run():
        # {{
        #   "underlying":  str,           # e.g. "SPY"
        #   "side":        "buy" | "sell",
        #   "intent":      "bullish" | "bearish",
        #   "qty":         int,           # # contracts
        #   "moneyness":   "atm" | "5pct_otm" | ...
        #   "dte_max":     int,
        #   "structure":   "long_call" | "long_put" | "call_vertical" | ...
        # }}
        # return {{}} if signal == 0 (no trade)

ALLOWED IMPORTS: pandas, numpy. NOTHING else.
NO print, no __main__, no markdown fences.
Use params.get(...) for every tunable.

MODERN PANDAS (>=2.1): use .ffill() / .bfill() directly. The keyword arg
.fillna(method='ffill') was REMOVED in pandas 2.1 and will crash at runtime.

COMMON BUGS THAT FAIL VALIDATION — avoid these:
  - .rolling(window=W, min_periods=M) REQUIRES M <= W (never min_periods > window).
  - .clip()/.pct_change()/.rolling()/.diff() are Series/DataFrame methods ONLY —
    never call them on df.index. Do indicator math on df['close']/df['high']/etc.
  - DO NOT OVERTRADE: ~6bps round-trip cost punishes signals that flip every few
    bars. Use hysteresis so positions persist; target tens-to-hundreds of trades,
    not thousands.

DO NOT re-implement any of these existing strategies:
{existing_strategies}

STRATEGY SPEC:
{spec}

Note: the spec's signal_rule field tells you the entry condition on the
underlying. Translate it into the signals() function. The other fields
(moneyness, dte_target, structure, profit_target_pct, stop_loss_pct) feed
into options_intent() — they don't change signal generation.

Output only the module contents.
"""


class OptionsCodeAgent:
    OUTPUT_DIR = Path("strategies")

    def __init__(self, store=None):
        self.store = store
        self.log   = logging.getLogger("options_code_agent")

    def generate_from_spec(self, spec: dict, strategy_id: str = None) -> dict:
        from agents.backtesting_agent import (
            STRATEGIES, register_strategy, is_registered,
        )

        name = (spec.get("name") or "").lower().strip().replace(" ", "_")
        if not name:
            return self._failure("spec missing 'name'")
        if is_registered(name):
            return self._failure(f"duplicate: '{name}' already in STRATEGIES",
                                 duplicate=True, matched=name)
        if not ANTHROPIC_API_KEY:
            return self._failure("ANTHROPIC_API_KEY not configured")

        try:
            from agents._claude_sdk import ask_claude
            from agents.research_agent import _existing_strategies_summary
            code_text = ask_claude(
                prompt        = "Write the module now.",
                system_prompt = OPTIONS_CODE_PROMPT.format(
                    spec=spec,
                    existing_strategies=_existing_strategies_summary(),
                ),
                allowed_tools = [],
                model         = "claude-opus-4-7",
            )
        except ImportError:
            return self._failure("claude_agent_sdk not installed")
        except Exception as e:
            return self._failure(f"sdk call failed: {e}")

        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        sid = strategy_id or f"opt{abs(hash(name)) % 100000:05d}"
        out_path = self.OUTPUT_DIR / f"{sid}_{name}.py"
        out_path.write_text(self._strip_fences(code_text), encoding="utf-8")

        ok, why, signals_fn = self._validate(out_path)
        if not ok:
            return self._failure(f"validation failed: {why}",
                                 code_path=str(out_path),
                                 duplicate="duplicates" in why)

        try:
            register_strategy(
                name           = name,
                signal_fn      = signals_fn,
                default_params = spec.get("params", {}),
                overwrite      = False,
            )
        except Exception as e:
            return self._failure(f"register_strategy failed: {e}",
                                 code_path=str(out_path))

        self.log.info(f"options_code_agent generated + registered '{name}' at {out_path}")
        return self._success(
            name        = name,
            code_path   = str(out_path),
            registered  = True,
            params      = spec.get("params", {}),
            underlying  = spec.get("underlying", "SPY"),
            structure   = spec.get("structure", "long_call"),
        )

    def _validate(self, path: Path):
        """validates BOTH signals() and options_intent() exist + work."""
        import importlib.util, numpy as np, pandas as pd

        spec_loader = importlib.util.spec_from_file_location(f"gen_opt_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec_loader)
        try:
            spec_loader.loader.exec_module(mod)
        except Exception as e:
            return False, f"import error: {e}", None
        if not hasattr(mod, "signals"):
            return False, "no signals() function", None
        if not hasattr(mod, "options_intent"):
            return False, "no options_intent() function", None

        idx = pd.date_range("2024-01-01", periods=500, freq="1min", tz="UTC")
        df  = pd.DataFrame({
            "open":   np.random.randn(500).cumsum() + 100,
            "high":   np.random.randn(500).cumsum() + 101,
            "low":    np.random.randn(500).cumsum() + 99,
            "close":  np.random.randn(500).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 500),
        }, index=idx)
        try:
            sig = mod.signals(df, {})
        except Exception as e:
            return False, f"signals() raised: {e}", None
        if not isinstance(sig, pd.Series) or len(sig) != len(df):
            return False, "signals() returned wrong shape/type", None

        # also verify options_intent doesn't crash with a sample signal
        try:
            intent = mod.options_intent(1, 100.0, {})
            if not isinstance(intent, dict):
                return False, "options_intent() must return a dict", None
        except Exception as e:
            return False, f"options_intent() raised: {e}", None

        return True, "ok", mod.signals

    @staticmethod
    def _strip_fences(text: str) -> str:
        t = text.strip()
        if t.startswith("```"):
            t = t.split("\n", 1)[1] if "\n" in t else t[3:]
            if t.rstrip().endswith("```"):
                t = t.rstrip()[:-3]
        return t.strip() + "\n"

    def _success(self, **kw): return {"success": True, "agent": "options_code_agent", **kw}
    def _failure(self, reason, **kw):
        self.log.warning(f"options_code_agent failed | {reason}")
        return {"success": False, "agent": "options_code_agent", "reason": reason, **kw}
