r"""
runners/tracking_dashboard.py
-----------------------------
The "is it actually tracking?" report. Ties together:
  - the BACKTEST EXPECTATION (deployed book's daily mean/vol -> expected band),
  - the LIVE track record (results/track_record.csv from monitor.py),
  - the REAL fill quality (results/fill_quality.csv from fill_tracker.py),
  - the current regime,
and judges whether live performance is within the backtest's expected range. This
is the tool that turns the forward pilot into the proof the board needs -- it tells
you, day by day, if the live book is behaving like the backtest or diverging.

Conservative expectation uses the NO-CRYPTO book (crypto is opt-in/often in cash),
and the honest worst-case drawdown (~-25% bootstrap / ~-32% GFC), not the -13.7%.

  python runners\tracking_dashboard.py            # console report + writes TRACKING.md
  python runners\tracking_dashboard.py --history  # include the per-session table
"""
import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows cp1252 console
except Exception:
    pass

import numpy as np
import pandas as pd

from agents.daily_strategies import TRADING_DAYS
from agents.execution_agent import ExecutionAgent
from runners.diversifier_screen import build_base, overlays, W
from runners.final_tests import lowvol_factor
from runners.lowvol_defensive import make_defensive
from runners.daily_rebalance import _detect_regime

ROOT = Path(__file__).parent.parent
TRACK = ROOT / "results" / "track_record.csv"
FILLS = ROOT / "results" / "fill_quality.csv"
OUT = ROOT / "TRACKING.md"


def backtest_expectation():
    panel = build_base(); idx = panel.index
    lvd = make_defensive(lowvol_factor()).reindex(idx).fillna(0)
    book = overlays(sum(panel[c].fillna(0) * W[c] for c in W) * 0.90 + lvd * 0.10, idx).fillna(0)
    mu, sd = float(book.mean()), float(book.std())
    return dict(mu=mu, sd=sd, ann=(1 + mu) ** TRADING_DAYS - 1,
                vol=sd * np.sqrt(TRADING_DAYS), sharpe=mu / sd * np.sqrt(TRADING_DAYS) if sd else 0)


def live_record():
    if not TRACK.exists():
        return None
    df = pd.read_csv(TRACK)
    df["equity"] = df["equity"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def fill_summary():
    if not FILLS.exists():
        return None
    rows = [r for r in csv.DictReader(open(FILLS)) if r.get("slippage_bps")]
    if not rows:
        return None
    s = [float(r["slippage_bps"]) for r in rows]
    return dict(n=len(s), avg=sum(s) / len(s), worst=max(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", type=int, default=1, choices=[1, 2])
    ap.add_argument("--history", action="store_true")
    args = ap.parse_args()

    print("computing backtest expectation + reading live records ...")
    exp = backtest_expectation()
    df = live_record()
    fills = fill_summary()
    try:
        regime = _detect_regime("auto")
    except Exception:
        regime = "unknown"

    L = []
    L.append(f"# Live Paper-vs-Backtest Tracking — {datetime.now(timezone.utc).date()}")
    L.append("")
    L.append(f"Current regime: **{regime.upper()}**")
    L.append("")
    L.append("## Backtest expectation (deployed book, no-crypto base)")
    L.append(f"- Expected annual return **{exp['ann']:.1%}** · vol **{exp['vol']:.1%}** · Sharpe **{exp['sharpe']:.2f}**")
    L.append(f"- Expected daily: **{exp['mu']*100:+.3f}% ± {exp['sd']*100:.2f}%**")
    L.append(f"- Honest worst-case drawdown: **~−25% (bootstrap p5) / ~−32% (2008 GFC)** — *not* −13.7%")
    L.append("")

    if df is None or len(df) < 2:
        n = 0 if df is None else len(df)
        L.append(f"## Live track record\n\n_Accumulating — {n} session(s) logged. Need ~20+ trading "
                 "days before live vs expected is statistically meaningful. The monitor appends one "
                 "row per day._")
    else:
        eq = df.set_index("date")["equity"]
        ret = eq.pct_change().dropna()
        n = len(ret)
        live_cum = eq.iloc[-1] / eq.iloc[0] - 1
        live_dd = float((eq / eq.cummax() - 1).min())
        live_sharpe = ret.mean() / ret.std() * np.sqrt(TRADING_DAYS) if ret.std() > 0 else 0
        # expected cumulative + 2-sigma band after n days
        exp_cum = (1 + exp["mu"]) ** n - 1
        band = 2 * exp["sd"] * np.sqrt(n)
        lo, hi = exp_cum - band, exp_cum + band
        verdict = ("ON TRACK" if lo <= live_cum <= hi else
                   ("ABOVE expectation" if live_cum > hi else "BELOW expectation — investigate"))
        L.append(f"## Live track record ({n} sessions since {df['date'].iloc[0].date()})")
        L.append("")
        L.append(f"- Live cumulative **{live_cum:+.2%}** · realized Sharpe {live_sharpe:.2f} · current drawdown {live_dd:.1%}")
        L.append(f"- Expected after {n} days: **{exp_cum:+.2%}** (2σ band {lo:+.1%} … {hi:+.1%})")
        L.append(f"- **Tracking: {verdict}**")
        if args.history:
            L.append("\n| date | equity | day % | regime |\n|---|---|---|---|")
            for _, r in df.iterrows():
                L.append(f"| {r['date'].date()} | ${r['equity']:,.0f} | {float(r['daily_return'])*100:+.2f}% | {r['regime']} |")

    L.append("")
    L.append("## Real fill quality (paper vs perfect liquidity)")
    if fills:
        L.append(f"- Avg slippage **{fills['avg']:+.1f} bps** ({fills['n']} fills) vs backtest's 6 bps assumption · worst {fills['worst']:+.1f}")
        L.append("- Real costs exceed the backtest — expect realized return below the headline (see cost stress in STRATEGIES.md).")
    else:
        L.append("- _No fills priced yet — run `fill_tracker.py` after live orders execute._")
    L.append("")
    L.append("## Alarms")
    alarms = []
    if df is not None and len(df) >= 2:
        eq = df["equity"].astype(float)
        dd = float((eq / eq.cummax() - 1).iloc[-1])
        if dd < -0.15:
            alarms.append(f"drawdown {dd:.1%} exceeds the −15% gate")
    if fills and fills["avg"] > 30:
        alarms.append(f"avg slippage {fills['avg']:.0f} bps is high — execution eating returns")
    L.append("\n".join(f"- ⚠️ {a}" for a in alarms) if alarms else "- none")
    L.append("")
    L.append("---\n*Honest note: a backtest can't prove a non-stationary future. This dashboard is "
             "the forward proof — it shows, live, whether the book tracks the backtest. Only a real-money "
             "pilot (after key rotation) closes the gap fully.*")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n(wrote {OUT.name})")


if __name__ == "__main__":
    main()
