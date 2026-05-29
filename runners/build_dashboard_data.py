"""
runners/build_dashboard_data.py
-------------------------------
Generate docs/data.json for the static dashboard (docs/index.html). Pulls the
REAL backtest numbers — deploy-ensemble equity curve vs SPY, per-book metrics,
walk-forward folds, the AI-agent roster + activity, and strategy-ledger counts —
into one small committable JSON (results/ itself is gitignored).

Run after any material change, then commit docs/data.json:
  python runners\\build_dashboard_data.py
"""
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    backtest_book, backtest_cross_sectional, vol_target, _metrics_from_returns,
    sig_rsi2_meanrev, sig_donchian, sig_trend_5020, DEPLOY_PARAMS, QUALITY_UNIVERSE,
    walk_forward_folds, daily_bars, INITIAL_CAP,
)


def m_of(r, lbl):
    return _metrics_from_returns(r, [], lbl)


def book_row(name, m, note=""):
    gate = (m["sharpe"] >= 0.8 and m["max_drawdown"] >= -0.15)
    return {"name": name, "sharpe": m["sharpe"], "cagr": m["cagr"],
            "max_drawdown": m["max_drawdown"], "pnl_dollars": m["pnl_dollars"],
            "risk_pass": bool(gate), "note": note}


AGENTS = [
    ("research_agent",   "research",  "Discovers + invents equity strategies (web + first principles)"),
    ("autonomous_agent", "research",  "Pure first-principles strategy invention"),
    ("ml_research_agent","research",  "Proposes daily ML/DL approaches"),
    ("options_research_agent","research","Options strategy ideas (verticals, condors, 0DTE)"),
    ("code_agent",       "build",     "Rule-based spec -> signals() module (validation-retry)"),
    ("ml_code_agent",    "build",     "ML spec -> train-in-signals module (validation-retry)"),
    ("options_code_agent","build",    "Options spec -> signals + intent module"),
    ("backtesting_agent","validate",  "$100k engine + walk-forward + regime + risk"),
    ("risk_agent",       "validate",  "Gates by Sharpe/DD/WR/trades thresholds"),
    ("execution_agent",  "execute",   "Alpaca paper orders (fractional + vol-target)"),
    ("options_agent",    "execute",   "Alpaca paper options orders"),
    ("monitor_agent",    "execute",   "Live PnL + concentration/drawdown alerts"),
]


def agent_activity():
    """mark agents active if they appear in the latest pipeline run + attach a
    work count; otherwise 'ready'."""
    latest = sorted(glob.glob("results/auto_pipeline_*.json"))
    ran, counts = set(), {}
    if latest:
        try:
            d = json.loads(Path(latest[-1]).read_text())
            ag = d.get("agents", {})
            for k, v in ag.items():
                ran.add(k)
                if isinstance(v, dict):
                    counts[k] = len(v)
        except Exception:
            pass
    # ledger-derived counts for the code agents
    led = {}
    try:
        led = json.loads(Path("results/strategy_ledger.json").read_text())
    except Exception:
        pass
    n_gen = sum(1 for v in led.values() if v.get("kind") == "rule")
    n_ml = sum(1 for v in led.values() if v.get("kind") == "ml")
    out = []
    for name, layer, role in AGENTS:
        detail = ""
        if name == "code_agent" and n_gen:
            detail = f"{n_gen} rule strategies generated"
        elif name == "ml_code_agent" and n_ml:
            detail = f"{n_ml} ML strategies generated"
        elif name in counts:
            detail = f"{counts[name]} outputs last run"
        status = "active" if (name in ran or detail) else "ready"
        out.append({"name": name, "layer": layer, "role": role,
                    "status": status, "detail": detail})
    return out


def ledger_stats():
    try:
        led = json.loads(Path("results/strategy_ledger.json").read_text())
    except Exception:
        return {"tried": 0, "deployed": 0, "dead": 0}
    return {
        "tried": len(led),
        "deployed": sum(1 for v in led.values() if v.get("status") in ("deployed", "kept")),
        "dead": sum(1 for v in led.values() if v.get("status") == "dead"),
    }


def main():
    U = QUALITY_UNIVERSE
    print("computing component books ...")
    r_rsi = backtest_book(sig_rsi2_meanrev, U, DEPLOY_PARAMS["rsi2_meanrev"], label="rsi")["_returns"]
    r_don = backtest_book(sig_donchian, U, None, label="don")["_returns"]
    r_trd = backtest_book(sig_trend_5020, U, None, label="trd")["_returns"]
    print("computing full-S&P-500 cross-sectional sleeve ...")
    from data.sp500 import sp500_tickers
    r_xs = backtest_cross_sectional(sp500_tickers(), mode="momentum", lookback=252,
                                    skip=21, k=10, market_filter=True, label="xs")["_returns"]

    panel = pd.concat([r_rsi, r_don, r_trd, r_xs], axis=1, sort=True)
    panel.columns = ["rsi", "don", "trd", "xs"]
    ens = vol_target(panel.mean(axis=1), target_vol=0.12, max_leverage=1.0)
    me = m_of(ens, "ensemble")

    # equity curves (monthly) — ensemble vs SPY buy & hold over the same window
    eq = me["_equity"]
    spy = daily_bars("SPY")["close"]
    spy = spy[spy.index >= eq.index.min()]
    spy_eq = INITIAL_CAP * (spy / spy.iloc[0])
    eqm = eq.resample("ME").last().dropna()
    spym = spy_eq.resample("ME").last().dropna()
    equity = [{"date": d.strftime("%Y-%m"),
               "strategy": round(float(eqm.loc[d]), 0),
               "spy": round(float(spym.loc[d]), 0) if d in spym.index else None}
              for d in eqm.index]

    wf = [{"period": f"{f.get('start','?')[:7]}..{f.get('end','?')[:7]}",
           "sharpe": f["sharpe"], "ret": f["return_pct"]}
          for f in walk_forward_folds(ens, 5)]

    books = [
        book_row("Ensemble + vol-target (DEPLOY)", me, "all 4 passers, dual-momentum on S&P 500"),
        book_row("trend_5020 + vol-target", m_of(vol_target(r_trd, 0.12), "t"), "50/200 trend, de-risked"),
        book_row("RSI-2 + Donchian + trend (blended)", m_of(panel[["rsi","don","trd"]].mean(axis=1), "b"), "core-3 equal weight"),
        book_row("Donchian breakout", m_of(r_don, "d"), "20/10 channel"),
        book_row("RSI-2 mean reversion", m_of(r_rsi, "r"), "buy dips in uptrend"),
    ]

    # preserve a manually-set repo_url across regenerations
    repo_url = ""
    try:
        repo_url = json.loads(Path("docs/data.json").read_text()).get("repo_url", "")
    except Exception:
        pass

    data = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "span": f"{eq.index.min().date()} .. {eq.index.max().date()}",
        "repo_url": repo_url,
        "headline": {
            "name": "Ensemble + vol-target",
            "sharpe": me["sharpe"], "cagr": me["cagr"],
            "max_drawdown": me["max_drawdown"], "pnl_dollars": me["pnl_dollars"],
            "final_capital": me["final_capital"],
            "wf_folds_positive": sum(1 for f in wf if f["sharpe"] > 0),
            "wf_total": len(wf),
        },
        "books": books,
        "equity": equity,
        "walk_forward": wf,
        "agents": agent_activity(),
        "ledger": ledger_stats(),
    }
    out = Path("docs/data.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"wrote {out} | ensemble Sharpe {me['sharpe']} CAGR {me['cagr']:.1%} "
          f"DD {me['max_drawdown']:.1%} | {len(equity)} equity points")


if __name__ == "__main__":
    main()
