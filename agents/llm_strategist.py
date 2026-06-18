"""
agents/llm_strategist.py
------------------------
LLM-INVENTED strategy mechanisms for the agent lab. Each run, Claude proposes a
fresh batch of original daily/multi-day equity signals as small Python functions;
we compile and vet each one before it is ever backtested.

Because this EXECUTES model-written code, there are two hard guards:

  1. AST SANDBOX (safe_compile) — the code must be a single `def signal(d, params)`
     using only pandas (pd), numpy (np) and a tiny builtin whitelist. No imports,
     no attribute dunders, no eval/exec/open/IO, no pandas .eval/.query/.to_*/read_*.
  2. LOOK-AHEAD PROBE (lookahead_safe) — recompute the signal on data truncated at
     several cut points; if a past day's value changes once the future is hidden,
     the function peeks ahead and is rejected. This catches global .max()/.mean(),
     negative .shift(), .iloc[i+k], etc. that the AST check can't see.

If the LLM (claude-agent-sdk) is unavailable, propose_batch() returns [] and the
lab falls back to its deterministic parameter-search batch — the button always works.
"""
from __future__ import annotations

import ast
import json
import re

import numpy as np
import pandas as pd

MODEL = "claude-opus-4-8"

# ---- AST sandbox -----------------------------------------------------------
_ALLOWED_BUILTINS = {
    "abs": abs, "min": min, "max": max, "len": len, "range": range, "float": float,
    "int": int, "bool": bool, "round": round, "sum": sum, "enumerate": enumerate,
    "zip": zip, "sorted": sorted, "list": list, "dict": dict, "tuple": tuple,
    "set": set, "map": map, "filter": filter, "True": True, "False": False, "None": None,
}
_FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "input", "exit", "quit", "help", "breakpoint",
    "memoryview", "__builtins__", "system", "popen", "environ", "importlib",
}
_FORBIDDEN_ATTRS = {
    "eval", "query", "to_pickle", "to_csv", "to_json", "to_parquet", "to_hdf",
    "to_feather", "to_excel", "to_sql", "read_csv", "read_pickle", "read_parquet",
    "read_json", "system", "popen", "communicate", "__class__", "__globals__",
    "__subclasses__", "__bases__", "__mro__", "__dict__", "__getattribute__",
}


def safe_compile(code: str):
    """compile a `def signal(d, params)` from `code` in a restricted namespace.
    raises ValueError if the code uses anything outside the sandbox."""
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("imports are not allowed")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise ValueError("global/nonlocal not allowed")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in _FORBIDDEN_ATTRS:
                raise ValueError(f"attribute '{node.attr}' is not allowed")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise ValueError(f"name '{node.id}' is not allowed")
    ns = {"pd": pd, "np": np, "__builtins__": _ALLOWED_BUILTINS}
    exec(compile(tree, "<llm_signal>", "exec"), ns)
    fn = ns.get("signal")
    if not callable(fn):
        raise ValueError("code must define a callable `signal(d, params)`")
    return fn


def _as_unit_series(out, index) -> pd.Series:
    """coerce a signal output to a clean [0,1] float series aligned to index."""
    s = out if isinstance(out, pd.Series) else pd.Series(out, index=index)
    s = pd.to_numeric(s, errors="coerce").reindex(index).fillna(0.0).clip(0.0, 1.0)
    return s.astype(float)


def lookahead_safe(fn, d: pd.DataFrame, params: dict, cuts=(0.5, 0.7, 0.85)) -> bool:
    """recompute on truncated history; a clean (causal) signal gives the SAME value
    for a given past day whether or not later bars exist. If any cut disagrees on
    the last in-sample day, the function peeks ahead -> reject."""
    try:
        full = _as_unit_series(fn(d, params), d.index)
    except Exception:
        return False
    n = len(d)
    for frac in cuts:
        k = int(n * frac)
        if k < 220 or k >= n:
            continue
        try:
            part = _as_unit_series(fn(d.iloc[:k], params), d.iloc[:k].index)
        except Exception:
            return False
        # compare the last 5 shared days (warmup-insensitive)
        a, b = full.iloc[k - 5:k].to_numpy(), part.iloc[k - 5:k].to_numpy()
        if not np.allclose(a, b, atol=1e-9, equal_nan=True):
            return False
    return True


def validate_spec(spec: dict, probe_df: pd.DataFrame):
    """compile + sanity + look-ahead check one proposed strategy.
    returns (fn, reason_if_rejected). fn is None when rejected."""
    code = spec.get("code", "")
    if not isinstance(code, str) or "def signal" not in code:
        return None, "no signal() definition"
    try:
        fn = safe_compile(code)
    except (ValueError, SyntaxError) as e:
        return None, f"sandbox reject: {e}"
    params = spec.get("params") or {}
    try:
        out = _as_unit_series(fn(probe_df, params), probe_df.index)
    except Exception as e:
        return None, f"runtime error: {str(e)[:60]}"
    if out.abs().sum() == 0 or out.nunique() <= 1:
        return None, "signal never trades / constant"
    if not lookahead_safe(fn, probe_df, params):
        return None, "look-ahead detected"
    return fn, None


# ---- LLM proposal ----------------------------------------------------------
_SYSTEM = """You are a senior quant researcher inventing ORIGINAL daily/multi-day
equity trading signals. Invent genuinely new mechanisms from first principles — do
NOT reproduce textbook indicators (RSI, MACD, Bollinger, Donchian, classic 50/200).

Return ONLY a JSON array of objects, each:
{
 "name": "snake_case_unique_name",
 "family": "reversion|trend|volatility|structure",
 "thesis": "one sentence, the economic/behavioral rationale",
 "params": {"lookback": 20, ...},          // small dict of numeric params
 "code": "def signal(d, params):\\n    ..."  // a COMPLETE python function
}

Hard rules for `code`:
- Signature exactly: def signal(d, params): ... return a pandas Series.
- `d` is a daily OHLCV DataFrame with columns open, high, low, close, volume.
- Use ONLY pandas (as pd) and numpy (as np). NO imports, NO file/network/system
  access, NO eval/exec, NO .query/.eval/.to_*/.read_*.
- Return a Series aligned to d.index with values in [0,1] (0=flat, 1=full long,
  fractions allowed). End with .fillna(0).
- CAUSAL ONLY: use rolling windows and POSITIVE shifts. Never use the whole series'
  global .max()/.min()/.mean(), never .shift(negative), never future indices. Do NOT
  shift the final signal yourself (the backtester enters next day).
- Read params via params.get("name", default) so defaults exist.
Output the JSON array and nothing else."""


def propose_batch(n: int, avoid=None, seed=None, model: str = MODEL) -> list[dict]:
    """ask Claude for n fresh strategy specs. Returns [] on any failure (no key,
    sdk missing, bad JSON) so the caller can fall back."""
    avoid = sorted(set(avoid or []))
    prompt = (f"Invent {n} NEW, mutually-distinct equity signals as specified. "
              f"Make them decorrelated from trend-following and from each other. ")
    if avoid:
        prompt += "Do NOT duplicate these existing mechanisms: " + ", ".join(avoid) + ". "
    if seed is not None:
        prompt += f"(diversity seed {seed}) "
    try:
        from agents._claude_sdk import ask_claude
        text = ask_claude(prompt=prompt, system_prompt=_SYSTEM, allowed_tools=[], model=model)
    except Exception:
        return []
    return _parse(text)


def _parse(text: str) -> list[dict]:
    if not isinstance(text, str):
        return []
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for s in data if isinstance(data, list) else []:
        if isinstance(s, dict) and s.get("name") and s.get("code"):
            out.append(s)
    return out
