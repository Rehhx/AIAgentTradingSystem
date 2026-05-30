"""
runners/tail_hedge.py
---------------------
The one regime the long-biased book can't cover is a CRASH (it cushions, doesn't
profit). A protective-put overlay is the only thing that makes the book flat-to-
positive in a crash -- but it BLEEDS premium the other ~90% of the time. This
quantifies that trade: how much CAGR you give up in calm years to buy how much
crash protection.

Model (monthly): buy 1-month `otm`-OTM SPY puts on a fraction `h` of book notional,
priced Black-Scholes at realized vol + VRP markup (you PAY the premium as a buyer).
MODELED (no historical chains) -- same honest caveat as runners/options_income.py.
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
from runners.lowvol_defensive import make_defensive
from runners.options_income import bs_put

TD = 252
CRASHES = {"2018 Q4": ("2018-10-01", "2018-12-31"),
           "COVID":   ("2020-02-01", "2020-03-31"),
           "2022 bear": ("2022-01-01", "2022-10-31")}


def metrics_m(r):
    eq = (1 + r).cumprod()
    yrs = len(r) / 12
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = float((eq / eq.cummax() - 1).min())
    sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    return cagr, dd, sharpe


def put_overlay(spy, otm=0.10, vrp=0.03):
    """monthly P&L per unit hedged notional from buying 1-month OTM puts."""
    spm = spy.resample("ME").last()
    rvm = (spy.pct_change().rolling(21).std() * np.sqrt(TD)).resample("ME").last()
    out = {}
    months = list(spm.index)
    for a, b in zip(months[:-1], months[1:]):
        S0, S1 = spm[a], spm[b]
        iv = max(0.08, (rvm[a] if rvm[a] == rvm[a] else 0.15) + vrp)
        K = S0 * (1 - otm)
        prem = bs_put(S0, K, 21 / TD, 0.04, iv) / S0
        out[b] = max(0.0, K - S1) / S0 - prem      # payoff minus premium paid
    return pd.Series(out)


def main():
    print("building deployed book + modeling protective-put overlay ...\n")
    panel = build_base()
    base = sum(panel[c].fillna(0) * W[c] for c in W)
    idx = panel.index
    lvd = make_defensive(lowvol_factor()).reindex(idx).fillna(0)
    bookd = overlays(base * 0.90 + lvd * 0.10, idx).fillna(0)
    bookm = (1 + bookd).resample("ME").prod() - 1

    spy = daily_bars("SPY")["close"].reindex(idx)
    hedge = put_overlay(spy, otm=0.10).reindex(bookm.index).fillna(0)

    print("protective-put overlay (1-month 10% OTM SPY puts, modeled):")
    print(f"  {'hedge frac':12s} {'CAGR':>7s} {'maxDD(mo)':>10s} {'Sharpe':>7s}   {'crash-window returns':>20s}")
    print("-" * 92)
    for h in (0.0, 0.5, 1.0, 1.5):
        bh = bookm + h * hedge
        cagr, dd, sh = metrics_m(bh)
        # crash-window cumulative returns
        cw = []
        bh_d = bookd.copy()  # daily for window slicing approximated via monthly
        for nm, (a, b) in CRASHES.items():
            w = bh.loc[a:b]
            cw.append(f"{nm} {((1+w).prod()-1):+.0%}")
        tag = "  <- no hedge" if h == 0 else ""
        print(f"  {h:11.1f}x {cagr:7.1%} {dd:10.1%} {sh:7.2f}   {' | '.join(cw)}{tag}")

    print("\n  Read: higher hedge fraction = smaller crash losses (even gains) but lower CAGR")
    print("  in calm years (premium bleed). The book ALREADY cushions crashes; the hedge")
    print("  only earns its keep if the board wants crash-POSITIVE, not just crash-cushioned.")


if __name__ == "__main__":
    main()
