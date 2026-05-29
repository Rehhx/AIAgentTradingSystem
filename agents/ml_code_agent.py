"""
agents/ml_code_agent.py
-----------------------
specialized code_agent for ML strategy proposals from ml_research_agent.

separated from agents/code_agent.py because ML strategies need a different
contract than rule-based strategies:

  rule-based:  signals(df, params) -> pd.Series   (pure indicator-based)
  ML strategy: signals(df, params) -> pd.Series   (TRAINS a model on the
                                                   first half of df, predicts
                                                   on the rest)

the generated modules are self-contained — they import only pandas, numpy,
sklearn / xgboost (already installed), and the project's build_features
helpers from ml_research_agent. they fit cleanly into the existing backtest
engine because they still expose signals(df, params).

the train-inside-signals pattern is intentional:
  - keeps the backtest engine unchanged (no new train/predict split logic)
  - the engine's 70/30 walk-forward later validates these on held-out data
  - inference cost is paid once per backtest invocation, not per bar
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ANTHROPIC_API_KEY

log = logging.getLogger("ml_code_agent")


ML_CODE_PROMPT = """\
You are a python ML engineer implementing a SELF-CONTAINED trading model
that fits inside this project's existing backtest engine.

CONTRACT — your module must export ONE function:

    def signals(df: pd.DataFrame, params: dict) -> pd.Series

  - df has columns: open, high, low, close, volume (DatetimeIndex, UTC)
  - returns a series of int {{-1, 0, 1}} aligned to df.index
  - YOUR FUNCTION trains the model internally and predicts in the same call:
      1. build features for every bar
      2. fit on the first train_pct of bars (default 0.5)
      3. predict on the remaining bars
      4. convert probability/score to int signal:
            long  if p >= prob_long  (default 0.55)
            short if p <= prob_short (default 0.45)
            flat  otherwise
      5. return a Series of ints aligned to df.index (0 for bars before train_end)

ALLOWED IMPORTS:
  - pandas, numpy
  - sklearn (Ridge, RandomForestClassifier, GradientBoostingClassifier...)
  - xgboost.XGBClassifier
  - lightgbm.LGBMClassifier if available
  - the project's build_features helpers (`from agents.ml_research_agent import
    build_features, build_target`) — already exports an 18-feature matrix
  - NO torch (we don't have GPU); if the spec asks for a transformer or TCN,
    substitute a GradientBoostingClassifier on the same features as a proxy
    and document the substitution in the docstring.

CONSTRAINTS:
  - No print, no plotting, no __main__ blocks, no markdown fences.
  - Use params.get(...) for every tunable.
  - Robust to short df (return all-zeros if len(df) < 1000).
  - Time-aware split — no leakage. Train indices < test indices.
  - MODERN PANDAS (>=2.1): use .ffill() / .bfill() directly. The keyword
    arg .fillna(method='ffill') was REMOVED in pandas 2.1 and will crash.
    Use .ffill() instead. Same for bfill.
  - Use np.errstate(invalid='ignore') around log/divide if doing
    np.log(ret) or x/y on possibly-zero denominators.

COMMON BUGS THAT FAIL VALIDATION — these have actually broken past runs, avoid them:
  - .rolling(window=W, min_periods=M) REQUIRES M <= W. Never set min_periods
    larger than the window (e.g. window=5, min_periods=10 is an error). When in
    doubt use min_periods=W, or omit min_periods entirely.
  - .clip(), .pct_change(), .rolling(), .ewm(), .diff() are Series/DataFrame
    methods ONLY. NEVER call them on df.index (it is a DatetimeIndex with no
    such methods — '.clip' will raise AttributeError). Do all indicator math on
    df['close'] / df['high'] / df['low'] / df['volume'], never on the index.
  - Build every feature aligned to df.index and return EXACTLY len(df) values.

DO NOT OVERTRADE — this is why most ML strategies here fail. The engine charges
~6bps round-trip; a signal that flips every few bars bleeds to death (we have
many real examples at -7 to -30 Sharpe with 40k-400k trades). Make the signal
PERSIST:
  - use hysteresis: go long when p >= prob_long, and only return to flat when
    p < 0.5 (a dead-band), NOT the instant p dips below prob_long. Same on the
    short side. This holds a position across noise instead of churning.
  - aim for a signal that changes state on the order of hundreds of times over
    the sample, not every bar. Do not emit a fresh nonzero signal every bar.

DO NOT re-implement any of these existing strategies:
{existing_strategies}

STRATEGY SPEC:
{spec}

Output only the module contents. No code fences. No extra text.
"""


class MLCodeAgent:
    """generates ML strategy modules. uses code_agent's validator."""

    OUTPUT_DIR = Path("strategies")

    def __init__(self, store=None):
        self.store = store
        self.log   = logging.getLogger("ml_code_agent")

    def generate_from_spec(self, spec: dict, strategy_id: str = None,
                           max_attempts: int = 3) -> dict:
        """same shape as code_agent.generate_from_spec; specialized prompt.

        if the generated module fails validation with a code bug (not a
        duplicate), the error is fed back to Claude and generation is retried
        up to ``max_attempts`` times. this turns one-shot failures like
        'min_periods 10 must be <= window 5' or ''Index' object has no attribute
        'clip'' into self-correcting iterations instead of dead runs.
        """
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
        except ImportError:
            return self._failure("claude_agent_sdk not installed")

        system_prompt = ML_CODE_PROMPT.format(
            spec=spec,
            existing_strategies=_existing_strategies_summary(),
        )
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        sid = strategy_id or f"ml{abs(hash(name)) % 100000:05d}"
        out_path = self.OUTPUT_DIR / f"{sid}_{name}.py"

        prev_code, last_err, signals_fn = None, None, None
        ok = False
        for attempt in range(1, max_attempts + 1):
            user_prompt = ("Write the module now." if attempt == 1
                           else self._fix_prompt(prev_code, last_err))
            try:
                code_text = ask_claude(
                    prompt        = user_prompt,
                    system_prompt = system_prompt,
                    allowed_tools = [],
                    model         = "claude-opus-4-7",
                    max_turns     = 2,          # single-shot text; avoids tool-loop max-turns errors
                )
            except Exception as e:
                return self._failure(f"sdk call failed: {e}", code_path=str(out_path))

            prev_code = self._strip_fences(code_text)
            out_path.write_text(prev_code, encoding="utf-8")

            ok, why, signals_fn = self._validate(out_path)
            if ok:
                break
            last_err = why
            if "duplicates" in why:          # not a code bug — retry won't help
                return self._failure(f"validation failed: {why}",
                                     code_path=str(out_path), duplicate=True)
            self.log.warning(f"ml_code_agent attempt {attempt}/{max_attempts} "
                             f"failed validation: {why} — retrying" if attempt < max_attempts
                             else f"ml_code_agent attempt {attempt}/{max_attempts} "
                                  f"failed validation: {why} — giving up")

        if not ok:
            return self._failure(f"validation failed after {max_attempts} attempts: {last_err}",
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

        self.log.info(f"ml_code_agent generated + registered '{name}' at {out_path}")
        return self._success(
            name        = name,
            code_path   = str(out_path),
            registered  = True,
            params      = spec.get("params", {}),
        )

    def _validate(self, path: Path):
        """ML modules need more data than rule-based to fit a model. give
        the validator 1500 bars instead of 500."""
        import importlib.util, numpy as np, pandas as pd

        spec = importlib.util.spec_from_file_location(f"gen_ml_{path.stem}", path)
        mod  = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            return False, f"import error: {e}", None

        if not hasattr(mod, "signals"):
            return False, "no signals() function found", None

        # bigger synthetic dataset so feature warmup + train/test split is viable
        idx = pd.date_range("2024-01-01", periods=1500, freq="1min", tz="UTC")
        rng = np.random.default_rng(42)
        close = 100 + rng.standard_normal(1500).cumsum()
        df  = pd.DataFrame({
            "open":   close + rng.standard_normal(1500) * 0.05,
            "high":   close + np.abs(rng.standard_normal(1500)) * 0.1,
            "low":    close - np.abs(rng.standard_normal(1500)) * 0.1,
            "close":  close,
            "volume": rng.integers(1000, 10000, 1500),
        }, index=idx)
        try:
            sig = mod.signals(df, {})
        except Exception as e:
            return False, f"signals() raised: {e}", None
        if not isinstance(sig, pd.Series) or len(sig) != len(df):
            return False, "signals() returned wrong shape/type", None
        return True, "ok", mod.signals

    @staticmethod
    def _fix_prompt(prev_code: str, error: str) -> str:
        """build a corrective prompt that feeds the validation error + the
        broken code back to Claude for a fixed module."""
        return (
            "Your previous module FAILED validation with this error:\n\n"
            f"    {error}\n\n"
            "Here is the code you wrote:\n\n"
            f"{prev_code}\n\n"
            "Return a corrected, COMPLETE module that fixes this specific error "
            "and still satisfies the full contract (export signals(df, params), "
            "return a pd.Series of ints aligned to df.index). Pay attention to "
            "the COMMON BUGS section. Output only the module contents — no code "
            "fences, no commentary."
        )

    @staticmethod
    def _strip_fences(text: str) -> str:
        t = text.strip()
        if t.startswith("```"):
            t = t.split("\n", 1)[1] if "\n" in t else t[3:]
            if t.rstrip().endswith("```"):
                t = t.rstrip()[:-3]
        return t.strip() + "\n"

    def _success(self, **kw): return {"success": True, "agent": "ml_code_agent", **kw}
    def _failure(self, reason, **kw):
        self.log.warning(f"ml_code_agent failed | {reason}")
        return {"success": False, "agent": "ml_code_agent", "reason": reason, **kw}
