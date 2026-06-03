"""
runners/sector_momentum.py  (Account 1 research)
------------------------------------------------
Sector-momentum rotation: rank the 9 SPDR sectors by trailing momentum, hold the
top-K (long-only, equal weight), rotate to T-bills when SPY < 200d (dual momentum).
Sectors are less noisy than single names, so sector momentum often clears a higher
Sharpe than stock momentum — and it may diversify the existing single-name sleeves.

Earns a seat only if (a) Sharpe is competitive (>=1.0 after vol-target) AND (b) it
isn't ~1.0 correlated to the deployed cross-sectional/trend sleeves. 2005-2026,
6 bps, vol-targeted.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from agents.daily_strategies import _metrics_from_returns, vol_target, walk_forward_folds, TRADING_DAYS

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY", "XLB"]
LOOKBACK, TOPK, COST = 126, 3, 0.0006


def _px(t):
    s = yf.Ticker(t).history(start="2004-06-01", end="2026-06-03", auto_adjust=True)["Close"]
    s.index = (s.index.tz_localize(None) if s.index.tz is None else s.index.tz_convert("UTC").tz_localize(None)).normalize()
    return s


def main():
    print("pulling 9 SPDR sectors + SPY + BIL (2005-2026) ...\n")
    C = pd.DataFrame({t: _px(t) for t in SECTORS}).sort_index()
    spy = _px("SPY").reindex(C.index).ffill()
    try:
        bil = _px("BIL").pct_change().reindex(C.index).fillna(0.0)
    except Exception:
        bil = pd.Series(0.0, index=C.index)
    C = C.dropna(how="all").ffill()
    R = C.pct_change().fillna(0.0)

    mom = C / C.shift(LOOKBACK) - 1                       # 6-month momentum
    rank = mom.rank(axis=1, ascending=False)
    hold = rank.le(TOPK).astype(float)                   # top-K sectors
    hold = hold.div(hold.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)   # equal weight
    market_on = (spy > spy.rolling(200).mean()).astype(float)   # dual-momentum cash filter
    W = hold.mul(market_on, axis=0)

    gross = (W.shift(1) * R).sum(axis=1)
    cash_w = (1 - W.shift(1).sum(axis=1)).clip(lower=0)
    turn = W.diff().abs().sum(axis=1).fillna(0)
    raw = gross + cash_w * bil - turn * COST
    vt = vol_target(raw, target_vol=0.12, max_leverage=1.5)

    m = _metrics_from_returns(vt, [], "sector_mom")
    folds = walk_forward_folds(vt, 5); pos = sum(1 for f in folds if f["sharpe"] > 0)
    corr_spy = float(vt.corr(spy.pct_change().fillna(0)))

    print("=" * 66)
    print("SECTOR-MOMENTUM ROTATION  (top-3 of 9, dual-momentum, vol-target 12%)")
    print("=" * 66)
    print(f"  CAGR {m['cagr']:.1%} | Sharpe {m['sharpe']:.2f} | maxDD {m['max_drawdown']:.1%} "
          f"| corr SPY {corr_spy:+.2f} | WF {pos}/5")
    print("\n  WALK-FORWARD folds:")
    for f in folds:
        mk = "+" if f["sharpe"] > 0 else "-"
        print(f"    [{mk}] {f.get('start','?')}..{f.get('end','?')}  Sharpe {f['sharpe']:+.2f}  ret {f['return_pct']:+.1%}")
    print("\n  YEAR-BY-YEAR:")
    yb = (1 + vt).groupby(vt.index.year).prod() - 1
    ys = (1 + spy.pct_change().fillna(0)).groupby(vt.index.year).prod() - 1
    for y in yb.index:
        print(f"    {y}  sector {yb[y]:>+6.1%}   SPY {ys.get(y,0):>+6.1%}{'  <-bear' if ys.get(y,0)<0 else ''}")

    print("\n" + "=" * 66)
    v = ("STRONG — test as a sleeve" if m["sharpe"] >= 1.3 and pos >= 4 else
         "decent — only adds if low-corr to existing momentum" if m["sharpe"] >= 1.0 else
         "WEAK/REJECT")
    print(f"  Sharpe {m['sharpe']:.2f} vs the book's 1.55 -> {v}")
    print("  Sector momentum is correlated to the existing trend/xs sleeves, so even a")
    print("  good Sharpe only helps if it diversifies. The blend test is the real decider.")


if __name__ == "__main__":
    main()
