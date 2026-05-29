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

DO NOT re-implement any of these existing strategies (we already have them
in the registry — generating duplicates is wasted work):
{existing_strategies}

Your implementation must encode a DIFFERENT mechanism than the above. If the
spec describes something equivalent to one of these, write a one-line module
with: `raise NotImplementedError("duplicates existing <name>")` so the
validator catches it and we can skip.

Do NOT generate __main__ blocks, plotting, or test code. Output only the
contents of the module — no markdown fences.

MODERN PANDAS (>=2.1): use .ffill() / .bfill() directly. The keyword arg
.fillna(method='ffill') was REMOVED in pandas 2.1 and will crash at runtime.

COMMON BUGS THAT FAIL VALIDATION — avoid these:
  - .rolling(window=W, min_periods=M) REQUIRES M <= W. Never set min_periods
    larger than the window. When in doubt use min_periods=W or omit it.
  - .clip()/.pct_change()/.rolling()/.diff() are Series/DataFrame methods ONLY.
    NEVER call them on df.index (a DatetimeIndex has no such methods). Do
    indicator math on df['close']/df['high']/etc., never on the index.
  - df.index is a tz-aware (UTC) DatetimeIndex. NEVER compare it to a tz-naive
    timestamp — `df.index >= pd.Timestamp("2020-01-01")` raises "Cannot compare
    tz-naive and tz-aware". Use pd.Timestamp("2020-01-01", tz="UTC"), or call
    df.index.tz_localize(None) first, or df.index.tz_convert("America/New_York")
    for time-of-day / session logic. Build helper date columns from the index
    the same tz-aware way.

DO NOT OVERTRADE: the engine charges ~6bps round-trip. A signal that flips
every few bars bleeds to death (real examples here: 40k-400k trades, Sharpe
-5 to -30). Use hysteresis / persistence so a position is held across noise
instead of churning. Target tens-to-hundreds of round-trips over the test
window, not thousands.

OUTPUT FORMAT: respond with the module SOURCE CODE as plain text only. Do not
use any tools, do not try to write files, do not add explanation before or
after. The first character of your reply must be the first character of the
module (an import, a comment, or a docstring).

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
            from agents.research_agent import _existing_strategies_summary
            code_text = ask_claude(
                prompt        = "Write the module now.",
                system_prompt = CODE_PROMPT.format(
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

    def generate_from_spec(self, spec: dict, strategy_id: str = None) -> dict:
        """
        store-free entrypoint used by the auto-pipeline. takes a spec dict
        (must have at least "name"; "description", "hypothesis", "params"
        recommended) and returns a result dict:

          { success, name, code_path, registered, reason? }

        does the duplicate-check itself: skips if the name already maps to a
        registered strategy by exact or substring match.
        """
        from agents.backtesting_agent import STRATEGIES, register_strategy, is_registered

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
        except ImportError:
            return self._failure("claude_agent_sdk not installed")

        system_prompt = CODE_PROMPT.format(
            spec=spec, existing_strategies=_existing_strategies_summary())
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        sid = strategy_id or f"gen{abs(hash(name)) % 100000:05d}"
        out_path = self.OUTPUT_DIR / f"{sid}_{name}.py"

        # retry loop: a one-shot SDK error (e.g. "max turns") or a code bug
        # (tz comparison, unterminated string) is fed back and re-attempted
        # instead of being thrown away.
        max_attempts, prev_code, last_err, signals_fn, ok = 3, None, None, None, False
        for attempt in range(1, max_attempts + 1):
            user_prompt = ("Write the module now." if attempt == 1
                           else self._fix_prompt(prev_code, last_err))
            try:
                code_text = ask_claude(
                    prompt        = user_prompt,
                    system_prompt = system_prompt,
                    allowed_tools = [],
                    model         = "claude-opus-4-7",
                    max_turns     = 2,          # code-gen is single-shot text
                )
            except Exception as e:
                last_err = f"sdk call failed: {e}"
                self.log.warning(f"code_agent attempt {attempt}/{max_attempts}: {last_err}")
                continue                        # transient — retry

            prev_code = self._strip_fences(code_text)
            out_path.write_text(prev_code, encoding="utf-8")
            ok, why, signals_fn = self._validate(out_path)
            if ok:
                break
            last_err = why
            if "duplicates" in why:             # intentional dedupe — not a bug
                return self._failure(f"validation failed: {why}",
                                     code_path=str(out_path), duplicate=True)
            self.log.warning(f"code_agent attempt {attempt}/{max_attempts} "
                             f"failed: {why}")

        if not ok:
            return self._failure(f"failed after {max_attempts} attempts: {last_err}",
                                 code_path=str(out_path))

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

        self.log.info(f"code_agent generated + registered '{name}' at {out_path}")
        return self._success(
            name        = name,
            code_path   = str(out_path),
            registered  = True,
            params      = spec.get("params", {}),
        )

    @staticmethod
    def _fix_prompt(prev_code: str, error: str) -> str:
        """corrective prompt: feed the failure + prior code back for a fix."""
        return (
            "Your previous attempt FAILED with this error:\n\n"
            f"    {error}\n\n"
            + (f"Here is the code you wrote:\n\n{prev_code}\n\n" if prev_code else "")
            + "Return a corrected, COMPLETE module that fixes this specific error "
            "and satisfies the contract (export signals(df, params) -> pd.Series of "
            "ints aligned to df.index). Mind the COMMON BUGS section. Output only "
            "the module source as plain text — no tools, no fences, no commentary."
        )

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
        """extract the python module from the model's reply. Handles markdown
        fences anywhere, and trims leading prose so the file starts at real
        code (avoids 'unterminated string literal at line 1')."""
        import re
        t = text.strip()
        # 1) if a fenced code block exists anywhere, take the first one
        m = re.search(r"```(?:python|py)?[ \t]*\n(.*?)```", t, re.DOTALL)
        if m:
            t = m.group(1)
        else:
            if t.startswith("```"):
                t = t.split("\n", 1)[1] if "\n" in t else ""
            if t.rstrip().endswith("```"):
                t = t.rstrip()[:-3]
        # 2) drop any leading prose lines before the first real code line
        lines = t.splitlines()
        starts = ("import ", "from ", "def ", "class ", "#", '"""', "'''", "@",
                  "raise ", "import\t")
        for i, ln in enumerate(lines):
            s = ln.strip()
            if s and (s.startswith(starts) or s.startswith('"') or s.startswith("'")):
                lines = lines[i:]
                break
        return "\n".join(lines).strip() + "\n"

    def _success(self, **kw): return {"success": True, "agent": "code_agent", **kw}
    def _failure(self, reason, **kw):
        self.log.warning(f"code agent failed | {reason}")
        return {"success": False, "agent": "code_agent", "reason": reason, **kw}
