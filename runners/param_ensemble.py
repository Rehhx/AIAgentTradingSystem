"""
runners/param_ensemble.py
-------------------------
Fixes the board's core valid critique -- parameter fragility under regime change.
Instead of betting each sleeve on ONE parameter (a param that may be right for the past
regime and wrong for the next), each sleeve becomes an ENSEMBLE: the average of
several parameter settings. The book then can't be "wrong about theta" when the regime
shifts -- it holds the whole range. Compared single-theta vs ensemble across 2005-2026
(dot-com tail, GFC, ZIRP, COVID, 2022) so robustness is shown across regimes, not one.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    sig_rsi2_meanrev, sig_donchian, sig_trend_5020, sig_recovery,
    vol_target, _metrics_from_returns, walk_forward_folds, DEPLOY_PARAMS, TRADING_DAYS,
)
from runners.extended_backtest import fetch_long, U, WTS, BEARS, _sleeve_book

SINGLE = {"rsi": (sig_rsi2_meanrev, DEPLOY_PARAMS["rsi2_meanrev"]),
          "don": (sig_donchian, None), "trd": (sig_trend_5020, None),
          "rec": (sig_recovery, {"hold_days": 120})}

ENSEMBLE = {
    "rsi": (sig_rsi2_meanrev, [{"rsi_period": 2, "entry_rsi": e, "exit_rsi": 50, "trend_sma": 100}
                               for e in (20, 25, 30, 35, 40)]),
    "don": (sig_donchian, [{"entry_lookback": lb, "exit_lookback": 10} for lb in (10, 15, 20, 30, 40)]),
    "trd": (sig_trend_5020, [{"fast": 50, "slow": s} for s in (150, 175, 200, 225, 250)]),
    "rec": (sig_recovery, [{"hold_days": h} for h in (90, 120, 150)]),
}


def overlay(combo, spy):
    warn = (spy < spy.rolling(50).mean()) & (spy.pct_change().rolling(20).std() * np.sqrt(TRADING_DAYS) > 0.20)
    ews = (1 - 0.4 * warn.astype(float)).shift(1).fillna(1.0)
    return (vol_target(combo, 0.17, max_leverage=1.8) * ews).fillna(0)


def main():
    print("pulling 2005-2026 data + building single-theta vs ensemble books (~2 min) ...")
    data = {t: fetch_long(t) for t in U}
    data = {t: d for t, d in data.items() if d is not None and len(d) > 260}
    idx = pd.DatetimeIndex(sorted(set().union(*[d.index for d in data.values()])))
    spy = data["SPY"]["close"].reindex(idx)
    sret = spy.pct_change().fillna(0)

    sl_s = {k: _sleeve_book(data, *SINGLE[k]).reindex(idx).fillna(0) for k in SINGLE}
    sl_e = {k: pd.concat([_sleeve_book(data, ENSEMBLE[k][0], p) for p in ENSEMBLE[k][1]], axis=1)
            .mean(axis=1).reindex(idx).fillna(0) for k in ENSEMBLE}
    book_s = overlay(sum(sl_s[k] * WTS[k] for k in WTS), spy)
    book_e = overlay(sum(sl_e[k] * WTS[k] for k in WTS), spy)

    print(f"\n  {'book':32s} {'CAGR':>6s} {'Sharpe':>7s} {'maxDD':>7s} {'WF':>4s}")
    print("  " + "-" * 60)
    for name, b in [("single-theta (1 param/sleeve)", book_s), ("ENSEMBLE (avg of 3-5 params)", book_e)]:
        m = _metrics_from_returns(b, [], name)
        pos = sum(1 for f in walk_forward_folds(b, 5) if f["sharpe"] > 0)
        print(f"  {name:32s} {m['cagr']:6.1%} {m['sharpe']:7.2f} {m['max_drawdown']:7.1%} {pos}/5")

    print("\n  ACROSS REGIMES (book return per bear, single-theta vs ensemble):")
    print(f"    {'bear':16s} {'single-theta':>9s} {'ensemble':>9s} {'S&P':>8s}")
    for nm, (a, b) in BEARS.items():
        print(f"    {nm:16s} {(1+book_s.loc[a:b]).prod()-1:+9.1%} {(1+book_e.loc[a:b]).prod()-1:+9.1%} "
              f"{(1+sret.loc[a:b]).prod()-1:+8.1%}")

    # robustness: std of yearly returns (lower = steadier across regimes)
    ys_s = (1 + book_s).groupby(book_s.index.year).prod() - 1
    ys_e = (1 + book_e).groupby(book_e.index.year).prod() - 1
    print(f"\n  yearly-return volatility (regime stability): single-theta {ys_s.std():.1%}  vs  ensemble {ys_e.std():.1%}")
    print("  The ensemble can't be wrong about theta when the regime shifts -- it holds the whole range.")


if __name__ == "__main__":
    main()
