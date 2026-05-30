"""
runners/lowvol_stress.py
------------------------
Does lowvol_factor help on the DOWN side? Measures the sleeve and the book WITH
vs WITHOUT it through the real bear / high-vol windows, plus stress-day capture.
lowvol_factor is long-only (no cash exit) -> it CUSHIONS (low beta), not hedges.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from agents.daily_strategies import daily_bars
from runners.diversifier_screen import build_base, overlays, W
from runners.final_tests import lowvol_factor

WINDOWS = {
    "2018 Q4 selloff":  ("2018-10-01", "2018-12-24"),
    "COVID crash":      ("2020-02-19", "2020-03-23"),
    "2022 bear":        ("2022-01-01", "2022-10-12"),
}


def cum_and_dd(r, a, b):
    w = r.loc[a:b].fillna(0.0)
    eq = (1 + w).cumprod()
    if len(eq) == 0:
        return float("nan"), float("nan")
    dd = float((eq / eq.cummax() - 1).min())
    return float(eq.iloc[-1] - 1), dd


def main():
    print("building book + lowvol (scans S&P 500; ~1 min) ...\n")
    panel = build_base()
    base = sum(panel[c].fillna(0) * W[c] for c in W)
    idx = panel.index
    lv = lowvol_factor().reindex(idx).fillna(0)

    spy_ret = daily_bars("SPY")["close"].pct_change().reindex(idx).fillna(0)
    book_base = overlays(base, idx)
    book_lv = overlays(base * 0.90 + lv * 0.10, idx)

    print(f"{'window':18s} {'SPY':>16s} {'lowvol sleeve':>16s} {'book (now)':>18s} {'book + lowvol':>18s}")
    print("-" * 92)
    for name, (a, b) in WINDOWS.items():
        sp = cum_and_dd(spy_ret, a, b)
        l = cum_and_dd(lv, a, b)
        bb = cum_and_dd(book_base, a, b)
        bl = cum_and_dd(book_lv, a, b)
        fmt = lambda x: f"{x[0]:+5.1%} (dd{x[1]:+4.0%})"
        print(f"{name:18s} {fmt(sp):>16s} {fmt(l):>16s} {fmt(bb):>18s} {fmt(bl):>18s}")
    print("  (each cell = total return over the window (max drawdown within it))")

    # stress-day capture: on the worst SPY days, what does each return on average?
    worst = spy_ret < spy_ret.quantile(0.05)        # bottom 5% of SPY days
    print(f"\nstress days (worst 5% of SPY days, n={int(worst.sum())}, avg SPY {spy_ret[worst].mean():+.2%}):")
    print(f"  lowvol sleeve avg : {lv[worst].mean():+.2%}   (beta to SPY on these days: "
          f"{np.polyfit(spy_ret[worst], lv[worst], 1)[0]:.2f})")
    print(f"  book now avg      : {book_base[worst].mean():+.2%}")
    print(f"  book + lowvol avg : {book_lv[worst].mean():+.2%}")

    # full-sample downside: average return on ALL down-SPY days
    down = spy_ret < 0
    print(f"\nall down-SPY days (n={int(down.sum())}):")
    print(f"  book now avg      : {book_base[down].mean():+.3%}")
    print(f"  book + lowvol avg : {book_lv[down].mean():+.3%}")


if __name__ == "__main__":
    main()
