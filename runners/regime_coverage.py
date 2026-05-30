"""
runners/regime_coverage.py
--------------------------
"Does the book cover all sides of the market?" Tags every trading day by regime
(trend x volatility) and measures the DEPLOYED 7-sleeve book vs SPY in each, plus
up/down capture and the worst-case stress days. The point is to find the GAP --
the regime where the book is still exposed -- not to flatter it.
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

TD = 252


def main():
    print("building deployed 7-sleeve book + tagging regimes ...\n")
    panel = build_base()
    base = sum(panel[c].fillna(0) * W[c] for c in W)
    idx = panel.index
    lvd = make_defensive(lowvol_factor()).reindex(idx).fillna(0)
    book = overlays(base * 0.90 + lvd * 0.10, idx).fillna(0)

    spy = daily_bars("SPY")["close"].reindex(idx)
    sret = spy.pct_change().fillna(0)
    trend_up = spy > spy.rolling(200).mean()
    calm = (sret.rolling(20).std() * np.sqrt(TD)) < 0.20

    reg = pd.Series("", index=idx)
    reg[trend_up & calm] = "Bull . calm"
    reg[trend_up & ~calm] = "Bull . stormy"
    reg[~trend_up & calm] = "Bear . calm"
    reg[~trend_up & ~calm] = "Bear . stormy"

    print(f"{'regime':16s} {'% days':>7s} {'SPY ann':>9s} {'book ann':>9s} {'book vol':>9s} {'book worst day':>15s}")
    print("-" * 70)
    for r in ["Bull . calm", "Bull . stormy", "Bear . calm", "Bear . stormy"]:
        mask = reg == r
        if mask.sum() == 0:
            continue
        b, s = book[mask], sret[mask]
        print(f"{r:16s} {mask.mean():7.0%} {s.mean()*TD:9.1%} {b.mean()*TD:9.1%} "
              f"{b.std()*np.sqrt(TD):9.1%} {b.min():15.1%}")

    up, dn = sret > 0, sret < 0
    print(f"\nup/down capture vs SPY:")
    print(f"  up-market days   : book {book[up].mean()*TD:+.0%} ann vs SPY {sret[up].mean()*TD:+.0%}  "
          f"-> {book[up].mean()/sret[up].mean():.0%} upside capture")
    print(f"  down-market days : book {book[dn].mean()*TD:+.0%} ann vs SPY {sret[dn].mean()*TD:+.0%}  "
          f"-> {book[dn].mean()/sret[dn].mean():.0%} downside capture (lower=better)")

    # acute stress: worst 1% of SPY days (the fast-crash flank)
    worst = sret < sret.quantile(0.01)
    print(f"\nacute stress (worst 1% SPY days, n={int(worst.sum())}, avg SPY {sret[worst].mean():.1%}):")
    print(f"  book avg on those days: {book[worst].mean():.1%}  (worst single day {book.min():.1%})")
    beta = np.polyfit(sret[worst], book[worst], 1)[0]
    print(f"  book beta to SPY on crash days: {beta:.2f}  (still long -> NOT crash-hedged)")

    # full-period drawdown the book has actually taken
    eq = (1 + book).cumprod()
    print(f"\nfull-period: book max drawdown {float((eq/eq.cummax()-1).min()):.1%} "
          f"| SPY max drawdown {float((( 1+sret).cumprod()/(1+sret).cumprod().cummax()-1).min()):.1%}")


if __name__ == "__main__":
    main()
