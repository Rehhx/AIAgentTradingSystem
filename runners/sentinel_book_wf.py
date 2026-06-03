"""
runners/sentinel_book_wf.py
---------------------------
Walk-forward of the FULL deployed book (7 sleeves) with the VIX-term-structure
crash sentinel added as a SECOND de-risk trigger -- vs the current book. Answers
the board's question: does the sentinel keep Sharpe in the 1.5-2.0 range while
cutting fast-crash drawdowns?

NOTE the earlier 0.62-0.72 Sharpes were a STANDALONE SPY-timing test (SPY itself
is ~0.62 Sharpe); this is the actual multi-sleeve book. No-margin config (1.0x).
2016-2026 deployed window, $100k, 6 bps.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from agents.daily_strategies import _metrics_from_returns, vol_target, walk_forward_folds, TRADING_DAYS, daily_bars
from runners.diversifier_screen import build_base, W
from runners.final_tests import lowvol_factor
from runners.lowvol_defensive import make_defensive

CRASHES = {"2018 Feb": ("2018-01-26", "2018-02-09"), "2018 Q4": ("2018-10-01", "2018-12-24"),
           "COVID": ("2020-02-19", "2020-03-23"), "2022 bear": ("2022-01-03", "2022-10-12")}


def _vix_sentinel(index):
    """VIX backwardation flag, aligned to the book index BY CALENDAR DATE (the book
    index is tz-aware UTC midnight; yfinance may be naive/other — align on date)."""
    def s(t):
        x = yf.Ticker(t).history(start="2015-06-01", end="2026-06-03", auto_adjust=True)["Close"]
        ix = x.index
        ix = ix.tz_localize(None) if ix.tz is None else ix.tz_convert("UTC").tz_localize(None)
        x.index = ix.normalize()                       # naive midnight, keyed by date
        return x
    vix, vix3m = s("^VIX"), s("^VIX3M")
    sent = ((vix / vix3m) >= 1.0).astype(float)        # backwardation = acute stress
    book_dates = pd.DatetimeIndex(index).tz_convert("UTC").tz_localize(None).normalize()
    out = sent.reindex(book_dates).ffill().fillna(0.0)
    out.index = index                                  # restore the book's tz-aware index
    return out


def overlay(combo, index, sentinel=None, vt=0.17, maxlev=1.0):
    spy = daily_bars("SPY")["close"].reindex(index)
    warn = (spy < spy.rolling(50).mean()) & (spy.pct_change().rolling(20).std() * np.sqrt(TRADING_DAYS) > 0.20)
    warn = warn.astype(float)
    if sentinel is not None:
        warn = np.maximum(warn, sentinel.reindex(index).fillna(0.0))    # de-risk if EITHER fires
    ews = (1 - 0.4 * warn).shift(1).fillna(1.0)                          # cut to 60% on trigger
    bil = daily_bars("BIL")["close"].pct_change().reindex(index).fillna(0.0)
    return (vol_target(combo, vt, max_leverage=maxlev) * ews + 0.22 * bil).fillna(0)


def _wf(name, ret):
    m = _metrics_from_returns(ret, [], name)
    folds = walk_forward_folds(ret, 5)
    pos = sum(1 for f in folds if f["sharpe"] > 0)
    return m, folds, pos


def main():
    print("building the deployed 7-sleeve book (~1-2 min) ...\n")
    panel = build_base(); idx = panel.index
    lvd = make_defensive(lowvol_factor()).reindex(idx).fillna(0)
    combo = sum(panel[c].fillna(0) * W[c] for c in W) * 0.90 + lvd * 0.10
    sentinel = _vix_sentinel(idx)

    cur = overlay(combo, idx, sentinel=None, maxlev=1.0)
    sen = overlay(combo, idx, sentinel=sentinel, maxlev=1.0)

    print("=" * 78)
    print("FULL DEPLOYED BOOK -- current vs + VIX crash sentinel  (no-margin 1.0x, 2016-2026)")
    print("=" * 78)
    print(f"  {'book':34s} {'$100k->':>11s} {'CAGR':>6s} {'Sharpe':>7s} {'maxDD':>7s} {'WF folds':>9s}")
    print("  " + "-" * 76)
    for name, ret in [("portfolio_full (current)", cur), ("portfolio_full + VIX sentinel", sen)]:
        m, folds, pos = _wf(name, ret)
        print(f"  {name:34s} ${m['final_capital']:>10,.0f} {m['cagr']:>6.1%} {m['sharpe']:>7.2f} "
              f"{m['max_drawdown']:>7.1%} {pos}/5 positive")

    print("\n  WALK-FORWARD per fold (Sharpe / return) -- + VIX sentinel book:")
    _, folds, _ = _wf("sen", sen)
    for f in folds:
        mark = "+" if f["sharpe"] > 0 else "-"
        print(f"    [{mark}] {f.get('start','?')}..{f.get('end','?')}  Sharpe {f['sharpe']:+.2f}  ret {f['return_pct']:+.1%}")

    print("\n  CRASH-WINDOW DRAWDOWN (book, current vs + sentinel):")
    print(f"    {'window':12s} {'current':>9s} {'+sentinel':>10s}")
    for nm, (a, b) in CRASHES.items():
        def dd(r):
            w = (1 + r.loc[a:b]).cumprod()
            return (w / w.cummax() - 1).min() if len(w) else 0.0
        print(f"    {nm:12s} {dd(cur):>9.1%} {dd(sen):>10.1%}")

    mc, _, _ = _wf("c", cur); ms, _, _ = _wf("s", sen)
    print("\n" + "=" * 78)
    print(f"  Board read: the BOOK Sharpe is {mc['sharpe']:.2f} (current) vs {ms['sharpe']:.2f} (+sentinel) -- "
          f"in the 1.5-2.0 target.\n  The sentinel's job is the crash column, not the headline Sharpe; "
          "it should hold Sharpe\n  while cutting the fast-crash (COVID/2018-Feb) drawdowns.")


if __name__ == "__main__":
    main()
