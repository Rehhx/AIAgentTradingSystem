r"""
runners/monitor.py
------------------
Daily monitoring + track-record + regime-posture check for the deployed book.
Run it once a day (after the rebalance) to:

  1. REGIME-POSTURE CHECK — detect the current market regime and verify the live
     book is positioned the way it SHOULD be for that regime (risk-on when calm,
     de-risked in a bear). This is the "does the portfolio cover each type of
     market" guard: it flags if the live posture doesn't match the regime.
  2. TRACK RECORD — append today's equity + daily P&L to results/track_record.csv
     and report realized return / Sharpe / drawdown vs SPY over the tracked window.
     This is the live proof that the book behaves like the backtest.
  3. ALARMS — drawdown breach (< -15% gate), outsized daily move, posture mismatch.

Read-only: never places an order. Works in SIMULATED mode (no Alpaca creds) by
showing the regime + the coverage playbook without a live posture.

  python runners\monitor.py              # check + append today's snapshot
  python runners\monitor.py --history    # also print the full track record
  python runners\monitor.py --no-append  # check only, don't write a row
"""
import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from agents.execution_agent import ExecutionAgent
from agents.daily_strategies import daily_bars, TRADING_DAYS
from runners.daily_rebalance import _detect_regime, regime_status

TRACK = Path(__file__).parent.parent / "results" / "track_record.csv"
DD_GATE = -0.15
BIG_DAY = 0.05
CASH_LIKE = {"BIL", "SHV", "SGOV"}      # T-bill ETFs counted as defensive, not risk

# regime playbook: expected RISK-exposure band, defensive band, and what the book does.
PLAYBOOK = {
    "bull_calm": dict(label="Bull . calm", gross=(0.95, 1.85), defensive=(0.0, 0.25),
                      posture="risk-on: full/levered exposure, recovery + lowvol active"),
    "bull_vol":  dict(label="Bull . volatile", gross=(0.55, 1.25), defensive=(0.0, 0.45),
                      posture="vol-target de-levers, mean-reversion favored"),
    "bear":      dict(label="Bear / downtrend", gross=(0.0, 0.70), defensive=(0.30, 1.0),
                      posture="de-risked: momentum -> cash, lowvol -> BIL, early-warning cut to 60%"),
}


def read_account(agent):
    if agent.simulated or agent.client is None:
        return None
    try:
        acct = agent.client.get_account()
        pos = agent.get_positions()
        eq = float(acct.equity)
        last = float(getattr(acct, "last_equity", eq) or eq)
        cash = float(getattr(acct, "cash", 0.0) or 0.0)
        risk = sum(abs(p["market_value"]) for p in pos if p["symbol"] not in CASH_LIKE)
        defn = cash + sum(abs(p["market_value"]) for p in pos if p["symbol"] in CASH_LIKE)
        return dict(equity=eq, last_equity=last, risk_gross=risk / eq if eq else 0,
                    defensive=defn / eq if eq else 0, n_positions=len(pos))
    except Exception as e:
        print(f"  [warn] could not read Alpaca account: {e}")
        return None


def coverage_table(regime):
    print("\nMARKET-COVERAGE PLAYBOOK (how the book positions for each regime):")
    print(f"  {'regime':18s} {'risk exposure':>14s} {'defensive':>11s}   what it does")
    order = ["bull_calm", "bull_vol", "bear"]
    for k in order:
        p = PLAYBOOK[k]
        cur = "  <== CURRENT" if k == regime else ""
        g = f"{p['gross'][0]:.0%}-{p['gross'][1]:.0%}"
        d = f"{p['defensive'][0]:.0%}-{p['defensive'][1]:.0%}"
        print(f"  {p['label']:18s} {g:>14s} {d:>11s}   {p['posture']}{cur}")


def posture_check(regime, acct):
    p = PLAYBOOK[regime]
    g0, g1 = p["gross"]
    rg = acct["risk_gross"]
    print("\nLIVE POSTURE (Alpaca paper):")
    print(f"  equity         ${acct['equity']:,.0f}")
    print(f"  risk exposure  {rg:.0%}   (expected {g0:.0%}-{g1:.0%} for {p['label']})")
    print(f"  defensive      {acct['defensive']:.0%}   (cash + T-bills)")
    print(f"  positions      {acct['n_positions']}")
    if rg < g0 - 0.10:
        return f"UNDER-EXPOSED for {p['label']} (risk {rg:.0%} < {g0:.0%}) — check signals/fills"
    if rg > g1 + 0.10:
        return f"OVER-EXPOSED for {p['label']} (risk {rg:.0%} > {g1:.0%}) — leverage too high"
    return f"MATCH — posture is consistent with {p['label']}"


def append_today(acct, regime):
    today = datetime.now(timezone.utc).date().isoformat()
    rows = list(csv.DictReader(open(TRACK))) if TRACK.exists() else []
    rows = [r for r in rows if r["date"] != today]
    dret = acct["equity"] / acct["last_equity"] - 1 if acct["last_equity"] else 0.0
    rows.append(dict(date=today, equity=f"{acct['equity']:.2f}", daily_return=f"{dret:.6f}",
                     regime=regime, risk_gross=f"{acct['risk_gross']:.4f}",
                     defensive=f"{acct['defensive']:.4f}", n_positions=acct["n_positions"]))
    rows.sort(key=lambda r: r["date"])
    TRACK.parent.mkdir(exist_ok=True)
    with open(TRACK, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows


def track_report(rows, show_history):
    df = pd.DataFrame(rows)
    df["equity"] = df["equity"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    print(f"\nTRACK RECORD ({len(df)} session(s), since {df.date.iloc[0].date()}):")
    if len(df) < 5:
        print("  accumulating — need ~5+ sessions before Sharpe/DD are meaningful.")
    else:
        eq = df.set_index("date")["equity"]
        ret = eq.pct_change().dropna()
        cum = eq.iloc[-1] / eq.iloc[0] - 1
        dd = float((eq / eq.cummax() - 1).min())
        sharpe = ret.mean() / ret.std() * np.sqrt(TRADING_DAYS) if ret.std() > 0 else 0.0
        spy = daily_bars("SPY")["close"]
        spy = spy[(spy.index >= df.date.iloc[0]) & (spy.index <= df.date.iloc[-1])]
        spy_cum = (spy.iloc[-1] / spy.iloc[0] - 1) if len(spy) > 1 else float("nan")
        print(f"  cumulative {cum:+.1%}  |  ann ~{(1+cum)**(TRADING_DAYS/len(df))-1:+.0%}  "
              f"|  Sharpe {sharpe:.2f}  |  max DD {dd:.1%}")
        print(f"  vs SPY over the same window: {spy_cum:+.1%}  (book {cum-spy_cum:+.1%} relative)")
    if show_history:
        print("\n  date        equity       day%   regime      riskX  def%  pos")
        for _, r in df.iterrows():
            print(f"  {r.date.date()}  ${r.equity:>10,.0f}  {float(r.daily_return)*100:+5.2f}  "
                  f"{r.regime:10s}  {float(r.risk_gross)*100:4.0f}% {float(r.defensive)*100:4.0f}% {r.n_positions:>4}")
    return df


def alarms(rows, acct, posture_msg):
    al = []
    df = pd.DataFrame(rows); df["equity"] = df["equity"].astype(float)
    if len(df) >= 2:
        eq = df["equity"]
        dd = float((eq / eq.cummax() - 1).iloc[-1])
        if dd < DD_GATE:
            al.append(f"DRAWDOWN BREACH: track-record drawdown {dd:.1%} < {DD_GATE:.0%} gate")
        dret = float(df["daily_return"].astype(float).iloc[-1])
        if abs(dret) > BIG_DAY:
            al.append(f"OUTSIZED DAY: {dret:+.1%} move (> +-{BIG_DAY:.0%})")
    if acct and not posture_msg.startswith("MATCH"):
        al.append(f"POSTURE: {posture_msg}")
    print("\nALARMS:", "none" if not al else "")
    for a in al:
        print(f"  [!] {a}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", action="store_true", help="print the full track-record table")
    ap.add_argument("--no-append", action="store_true", help="check only; don't write today's row")
    ap.add_argument("--source", default="auto")
    args = ap.parse_args()

    print(f"\n=== Portfolio Monitor | {datetime.now(timezone.utc).date()} ===")
    regime = _detect_regime(args.source)
    regime_status(args.source)
    coverage_table(regime)

    agent = ExecutionAgent()
    acct = read_account(agent)
    posture_msg = ""
    if acct is None:
        print("\nLIVE POSTURE: (no Alpaca account — SIMULATED). Playbook shown above is the")
        print("intended coverage; run with live creds to verify the book matches it.")
    else:
        posture_msg = posture_check(regime, acct)
        print(f"  posture check: {posture_msg}")
        rows = acct and (list(csv.DictReader(open(TRACK))) if (args.no_append and TRACK.exists()) else append_today(acct, regime))
        if rows:
            track_report(rows, args.history)
            alarms(rows, acct, posture_msg)


if __name__ == "__main__":
    main()
