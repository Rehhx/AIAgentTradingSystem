"""
runners/regime_fix_test.py
--------------------------
Tests the trend-aware leverage cap: never lever above 1.0x when SPY < its 200-day.
Fixes the "Bear . calm" gap (vol-target levering UP into a quiet downtrend) found
by regime_coverage.py. Compares current vs fixed book by regime + overall.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from agents.daily_strategies import daily_bars, vol_target, _metrics_from_returns, walk_forward_folds
from runners.diversifier_screen import build_base, overlays, W
from runners.final_tests import lowvol_factor
from runners.lowvol_defensive import make_defensive

TD = 252


def vol_target_trendcap(returns, spy, target_vol=0.17, window=20, maxlev=1.8):
    r = returns.fillna(0.0)
    rv = r.rolling(window).std() * np.sqrt(TD)
    above200 = (spy > spy.rolling(200).mean()).reindex(r.index).fillna(True)
    cap = np.where(above200.to_numpy(), maxlev, 1.0)             # no leverage below 200d
    scale = np.minimum((target_vol / rv.replace(0, np.nan)).to_numpy(), cap)
    scale = pd.Series(scale, index=r.index).shift(1).fillna(0.0)
    return r * scale


def overlays_fixed(combo, index, vt=0.17, maxlev=1.8):
    spy = daily_bars("SPY")["close"].reindex(index)
    warn = (spy < spy.rolling(50).mean()) & (spy.pct_change().rolling(20).std() * np.sqrt(TD) > 0.20)
    ews = (1 - 0.4 * warn.astype(float)).shift(1).fillna(1.0)
    bil = daily_bars("BIL")["close"].pct_change().reindex(index).fillna(0.0)
    return vol_target_trendcap(combo, spy, vt, maxlev=maxlev) * ews + 0.22 * bil


def by_regime(book, sret, reg):
    out = {}
    for r in ["Bull . calm", "Bull . stormy", "Bear . calm", "Bear . stormy"]:
        m = reg == r
        out[r] = book[m].mean() * TD if m.sum() else float("nan")
    return out


def summary(label, book):
    m = _metrics_from_returns(book, [], label)
    pos = sum(1 for f in walk_forward_folds(book, 5) if f["sharpe"] > 0)
    return f"Sharpe {m['sharpe']:.2f} | CAGR {m['cagr']:.1%} | DD {m['max_drawdown']:.1%} | WF {pos}/5"


def main():
    print("building book (current vs trend-capped leverage) ...\n")
    panel = build_base()
    base = sum(panel[c].fillna(0) * W[c] for c in W)
    idx = panel.index
    lvd = make_defensive(lowvol_factor()).reindex(idx).fillna(0)
    combo = base * 0.90 + lvd * 0.10

    cur = overlays(combo, idx).fillna(0)
    fix = overlays_fixed(combo, idx).fillna(0)

    spy = daily_bars("SPY")["close"].reindex(idx)
    sret = spy.pct_change().fillna(0)
    trend_up = spy > spy.rolling(200).mean()
    calm = (sret.rolling(20).std() * np.sqrt(TD)) < 0.20
    reg = pd.Series("", index=idx)
    reg[trend_up & calm] = "Bull . calm"; reg[trend_up & ~calm] = "Bull . stormy"
    reg[~trend_up & calm] = "Bear . calm"; reg[~trend_up & ~calm] = "Bear . stormy"

    rc, rf = by_regime(cur, sret, reg), by_regime(fix, sret, reg)
    print(f"{'regime':16s} {'current':>10s} {'trend-capped':>14s}")
    print("-" * 44)
    for r in ["Bull . calm", "Bull . stormy", "Bear . calm", "Bear . stormy"]:
        print(f"{r:16s} {rc[r]:10.1%} {rf[r]:14.1%}")

    print(f"\noverall:")
    print(f"  current      : {summary('cur', cur)}")
    print(f"  trend-capped : {summary('fix', fix)}")


if __name__ == "__main__":
    main()
