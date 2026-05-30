"""
runners/lowvol_defensive.py
---------------------------
A market-filtered ("defensive") low-vol sleeve: hold the 30 lowest-vol S&P names
ONLY while SPY > its 200-day; otherwise rotate to T-bills (BIL). This makes the
sleeve sit out slow bears (2018 Q4, 2022) instead of riding them down. It will
NOT dodge a fast crash (COVID) -- the 200-day is too slow -- but that's the
honest limit of any trend filter.

Compares plain vs defensive lowvol: standalone metrics, walk-forward, the three
bear windows, and the marginal effect on the deployed book at 10%.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from agents.daily_strategies import daily_bars, RT_COST, _metrics_from_returns, walk_forward_folds, split_metrics
from runners.diversifier_screen import build_base, overlays, W
from runners.final_tests import lowvol_factor

WINDOWS = {
    "2018 Q4 selloff": ("2018-10-01", "2018-12-24"),
    "COVID crash":     ("2020-02-19", "2020-03-23"),
    "2022 bear":       ("2022-01-01", "2022-10-12"),
}


def make_defensive(port):
    """rotate the long-only lowvol return series to BIL whenever SPY < 200-day."""
    spy = daily_bars("SPY")["close"]
    risk_on = (spy > spy.rolling(200).mean()).shift(1).reindex(port.index).fillna(True)
    bil = daily_bars("BIL")["close"].pct_change().reindex(port.index).fillna(0.0)
    switch = risk_on.astype(int).diff().abs().fillna(0)        # rotation cost on each regime flip
    return port.where(risk_on, bil) - switch * RT_COST


def cum(r, a, b):
    w = r.loc[a:b].fillna(0.0)
    eq = (1 + w).cumprod()
    dd = float((eq / eq.cummax() - 1).min()) if len(eq) else float("nan")
    return (float(eq.iloc[-1] - 1) if len(eq) else float("nan"), dd)


def standalone(name, r):
    m = _metrics_from_returns(r, [], name)
    pos = sum(1 for f in walk_forward_folds(r, 5) if f["sharpe"] > 0)
    print(f"  {name:24s} Sharpe {m['sharpe']:5.2f} | CAGR {m['cagr']:6.1%} | DD {m['max_drawdown']:6.1%} | WF {pos}/5")
    return m


def book_effect(name, sleeve, base, idx, bm):
    m = _metrics_from_returns(overlays(base * 0.90 + sleeve.reindex(idx).fillna(0) * 0.10, idx), [], name)
    folds = overlays(base * 0.90 + sleeve.reindex(idx).fillna(0) * 0.10, idx)
    pos = sum(1 for f in walk_forward_folds(folds, 5) if f["sharpe"] > 0)
    print(f"  {name:24s} Sharpe {m['sharpe']:.2f} ({m['sharpe']-bm['sharpe']:+.2f}) | "
          f"CAGR {m['cagr']:.1%} | DD {m['max_drawdown']:.1%} | WF {pos}/5")


def main():
    print("building book + plain/defensive lowvol (scans S&P 500; ~1 min) ...\n")
    panel = build_base()
    base = sum(panel[c].fillna(0) * W[c] for c in W)
    idx = panel.index
    lv = lowvol_factor()
    lvd = make_defensive(lv)

    print("STANDALONE sleeve:")
    standalone("lowvol (plain, long-only)", lv)
    standalone("lowvol (defensive 200d)", lvd)

    print("\nBEAR / VOL WINDOWS (total return over window, drawdown within):")
    print(f"  {'window':18s} {'SPY':>16s} {'plain lowvol':>16s} {'defensive lowvol':>18s}")
    for nm, (a, b) in WINDOWS.items():
        sp, lp, ld = cum(daily_bars('SPY')['close'].pct_change().reindex(idx), a, b), cum(lv, a, b), cum(lvd, a, b)
        f = lambda x: f"{x[0]:+5.1%}(dd{x[1]:+4.0%})"
        print(f"  {nm:18s} {f(sp):>16s} {f(lp):>16s} {f(ld):>18s}")

    bm = _metrics_from_returns(overlays(base, idx), [], "base")
    print(f"\nMARGINAL EFFECT ON BOOK (baseline Sharpe {bm['sharpe']:.2f}, CAGR {bm['cagr']:.1%}, DD {bm['max_drawdown']:.1%}):")
    book_effect("+ plain lowvol @10%", lv, base, idx, bm)
    book_effect("+ defensive lowvol @10%", lvd, base, idx, bm)


if __name__ == "__main__":
    main()
