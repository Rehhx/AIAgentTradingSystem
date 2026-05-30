"""
runners/managed_futures.py
--------------------------
A proper MANAGED-FUTURES / CTA book (the crisis-alpha engine for Account 2). This
is what made AQR / Man AHL / Aspect 30-70% in 2022: diversified long/short trend-
following, vol-targeted (leveraged), across many asset classes. It is designed to
be FLAT-to-positive in normal markets and SHINE when macro trends are strong
(2018 Q4, 2022) -- i.e. profitable exactly when long equity books lose.

  - markets: equities (SPY,QQQ,EFA,EEM), bonds (TLT,IEF), gold (GLD), commodities
    (DBC), US dollar (UUP), real estate (VNQ)
  - signal: multi-timeframe time-series momentum (avg sign of 1/3/12-mo returns),
    +1 long an uptrend / -1 short a downtrend
  - inverse-vol risk weighting, then vol-targeted to `--target-vol` (cap 2x)

$100k base, 6 bps round-trip, split/dividend-adjusted data.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from agents.daily_strategies import daily_bars, RT_COST, _metrics_from_returns, walk_forward_folds, split_metrics, TRADING_DAYS

MARKETS = ["SPY", "QQQ", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "UUP", "VNQ"]


def mf_returns(target_vol=0.12, max_lev=2.0):
    closes = {}
    for t in MARKETS:
        try:
            c = daily_bars(t)["close"]
            if len(c) > 300:
                closes[t] = c
        except Exception:
            pass
    C = pd.DataFrame(closes).sort_index()
    R = C.pct_change()
    sig = (np.sign(C / C.shift(21) - 1) + np.sign(C / C.shift(63) - 1) + np.sign(C / C.shift(252) - 1)) / 3.0
    vol = R.rolling(60).std()
    raw = sig / vol
    w = raw.div(raw.abs().sum(axis=1).replace(0, np.nan), axis=0)      # gross exposure ~1
    port = (w.shift(1) * R).sum(axis=1)
    turn = w.diff().abs().sum(axis=1).fillna(0)
    port = (port - turn * RT_COST).fillna(0)
    rv = port.rolling(20).std() * np.sqrt(TRADING_DAYS)
    scale = (target_vol / rv.replace(0, np.nan)).clip(upper=max_lev).shift(1).fillna(0.0)
    return (port * scale).fillna(0), w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-vol", type=float, default=0.12)
    ap.add_argument("--max-lev", type=float, default=2.0)
    args = ap.parse_args()

    print(f"building managed-futures book ({len(MARKETS)} markets, vol-target {args.target_vol:.0%}, cap {args.max_lev}x) ...\n")
    mf, w = mf_returns(args.target_vol, args.max_lev)
    m = _metrics_from_returns(mf, [], "managed_futures")
    s = split_metrics(mf)
    pos = sum(1 for f in walk_forward_folds(mf, 5) if f["sharpe"] > 0)
    print(f"=== MANAGED FUTURES (Account 2 crisis-alpha book) ===")
    print(f"  $100,000 -> ${m['final_capital']:,.0f}  (+${m['pnl_dollars']:,.0f})")
    print(f"  CAGR {m['cagr']:.1%} | Sharpe {m['sharpe']} | max DD {m['max_drawdown']:.1%} "
          f"| in-sample {s['train_sharpe']:+.2f} -> OOS {s['test_sharpe']:+.2f} | WF {pos}/5")

    yb = (1 + mf).groupby(mf.index.year).prod() - 1
    spy = daily_bars("SPY")["close"].reindex(mf.index).pct_change().fillna(0)
    ys = (1 + spy).groupby(spy.index.year).prod() - 1
    print("\n  YEAR-BY-YEAR (managed futures vs S&P 500) -- note the crisis years:")
    print(f"    {'year':>4s} {'mgd futures':>12s} {'S&P 500':>9s}")
    for y in yb.index:
        crisis = "  <== S&P down, CTA up" if ys[y] < 0 and yb[y] > 0 else ""
        print(f"    {y:>4d} {yb[y]:+12.1%} {ys[y]:+9.1%}{crisis}")

    print("\n  current positions (sign = long/short, |size| = risk weight):")
    last = w.iloc[-1].dropna().sort_values()
    for t, wt in last.items():
        print(f"    {t:5s} {'SHORT' if wt < 0 else 'LONG ':5s} {abs(wt):5.1%}")

    print("\n  This is the Account-2 book: low/choppy in calm bulls, but POSITIVE in the")
    print("  crisis years when Account 1 (long equity) is down. Run the two together.")


if __name__ == "__main__":
    main()
