"""
agents/code_agent.py
--------------------
takes an approved strategy spec and writes a self-contained python module
implementing it. the output file lives in strategies/ and exposes a single
function `signals(df, params) -> pd.Series` that the execution agent can
import and call on live data.

implementation note:
  - scaffold for claude-agent-sdk Agent class with file_write tool.
  - validates the generated code by attempting an import + a single call
    against synthetic data before approving it.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ANTHROPIC_API_KEY

log = logging.getLogger("code_agent")


CODE_PROMPT = """\
You are a python engineer implementing a quantitative trading strategy.

Given the strategy spec below, write a self-contained python module that:
  1. Imports only pandas, numpy, and the project's data.loader if needed.
  2. Exposes one function:
        def signals(df: pd.DataFrame, params: dict) -> pd.Series
     where df has columns: open, high, low, close, volume (DatetimeIndex)
     and returns a series of int {{-1, 0, 1}} aligned to df.index.
  3. Uses params.get(...) with sensible defaults for every tunable knob.
  4. Has no print statements, no side effects, no global state.
  5. Includes a brief docstring explaining the entry/exit rules.

Do NOT generate __main__ blocks, plotting, or test code. Output only the
contents of the module — no markdown fences.

STRATEGY SPEC:
{spec}
"""


class CodeAgent:
    """matches BaseAgent.run(task) contract used by orchestrator."""

    OUTPUT_DIR = Path("strategies")

    def __init__(self, store=None):
        self.store = store
        self.log   = logging.getLogger("code_agent")

    def run(self, task: dict) -> dict:
        strategy_id = task.get("strategy_id")
        if not strategy_id or self.store is None:
            return self._failure("strategy_id and store required")

        strategy = self.store.get_strategy(strategy_id)
        if not strategy:
            return self._failure(f"strategy {strategy_id} not found")

        if not ANTHROPIC_API_KEY:
            return self._failure("ANTHROPIC_API_KEY not configured")

        spec = {
            "name":        strategy["name"],
            "description": strategy.get("description", ""),
            "params":      strategy.get("params", {}),
        }

        try:
            from agents._claude_sdk import ask_claude
            code_text = ask_claude(
                prompt        = "Write the module now.",
                system_prompt = CODE_PROMPT.format(spec=spec),
                allowed_tools = [],
                model         = "claude-opus-4-7",
            )
        except ImportError:
            return self._failure("claude_agent_sdk not installed")
        except Exception as e:
            return self._failure(f"sdk call failed: {e}")

        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = self.OUTPUT_DIR / f"{strategy_id}_{strategy['name']}.py"
        out_path.write_text(self._strip_fences(code_text))

        # smoke-test the generated module, then register it for backtest
        ok, why, signals_fn = self._validate(out_path)
        if not ok:
            return self._failure(f"generated code failed validation: {why}",
                                 code_path=str(out_path))

        # register the generated function into the runtime STRATEGIES dict
        # so subsequent backtests can find it by name. orchestrator's
        # _run_strategy_lifecycle dispatches code_agent BEFORE backtest for
        # novel strategies, so the next dispatch will hit this registration.
        from agents.backtesting_agent import register_strategy
        try:
            register_strategy(
                name           = strategy["name"],
                signal_fn      = signals_fn,
                default_params = strategy.get("params", {}),
                overwrite      = True,
            )
        except Exception as e:
            return self._failure(f"register_strategy failed: {e}",
                                 code_path=str(out_path))

        from orchestrator import StrategyStatus  # local import — avoid cycle
        self.store.update_strategy(
            strategy_id,
            status    = StrategyStatus.IMPLEMENTING,
            code_path = str(out_path),
        )
        return self._success(code_path=str(out_path), registered=True)

    def _validate(self, path: Path):
        """import the generated module and call signals() on synthetic data.
        returns (ok, reason, signals_fn) — signals_fn is None on failure."""
        import importlib.util, numpy as np, pandas as pd

        spec = importlib.util.spec_from_file_location(f"gen_strategy_{path.stem}", path)
        mod  = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            return False, f"import error: {e}", None

        if not hasattr(mod, "signals"):
            return False, "no signals() function found", None

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
        return True, "ok", mod.signals

    @staticmethod
    def _strip_fences(text: str) -> str:
        """remove ```python ... ``` markdown fences if the model added them."""
        t = text.strip()
        if t.startswith("```"):
            t = t.split("\n", 1)[1] if "\n" in t else t[3:]
            if t.rstrip().endswith("```"):
                t = t.rstrip()[:-3]
        return t.strip() + "\n"

    def _success(self, **kw): return {"success": True, "agent": "code_agent", **kw}
    def _failure(self, reason, **kw):
        self.log.warning(f"code agent failed | {reason}")
        return {"success": False, "agent": "code_agent", "reason": reason, **kw}
