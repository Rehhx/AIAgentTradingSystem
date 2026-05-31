"""
runners/regime_switch_backtest.py
---------------------------------
Does REGIME SWITCHING beat a static blend? Dynamically rotate between the two
engines by detected regime: risk-on (SPY > 200d AND 20d vol < 20%) -> heavy
Account 1 (growth); risk-off -> heavy Account 2 (crisis-alpha managed futures).
Compared to the static 70/30 blend and each engine alone. The 200-day signal
lags, so this is an honest test of whether switching adds value or just whipsaws.
$100k base, 6 bps.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from agents.daily_strategies import _metrics_from_returns, walk_forward_folds, daily_bars, TRADING_DAYS
from runners.diversifier_screen import build_base, overlays, W
from runners.final_tests import lowvol_factor, crypto_trend
from runners.lowvol_defensive import make_defensive
from runners.managed_futures import mf_returns


def main():
    print("building both engines + regime switch (~1-2 min) ...\n")
    panel = build_base(); idx = panel.index
    lvd = make_defensive(lowvol_factor()).reindex(idx).fillna(0)
    combo = sum(panel[c].fillna(0) * W[c] for c in W) * 0.90 + lvd * 0.10
    cr = crypto_trend().reindex(idx).fillna(0)
    a1 = overlays(combo * 0.95 + cr * 0.05, idx).fillna(0)        # Account 1 (growth)
    a2 = mf_returns(0.12, 1.5)[0].reindex(idx).fillna(0)          # Account 2 (crisis-alpha)

    spy = daily_bars("SPY")["close"].reindex(idx)
    risk_on = ((spy > spy.rolling(200).mean()) &
               (spy.pct_change().rolling(20).std() * np.sqrt(TRADING_DAYS) < 0.20)).shift(1).fillna(True)
    print(f"  risk-on {risk_on.mean():.0%} of days / risk-off {1-risk_on.mean():.0%}\n")

    def switch(bull_a1, off_a1):
        w1 = pd.Series(np.where(risk_on, bull_a1, off_a1), index=idx)
        return w1 * a1 + (1 - w1) * a2

    variants = {
        "Account 1 only (growth)": a1,
        "Account 2 only (crisis)": a2,
        "Static 70/30 blend": 0.70 * a1 + 0.30 * a2,
        "Regime-switch 90/10 <-> 40/60": switch(0.90, 0.40),
        "Regime-switch 95/5 <-> 20/80": switch(0.95, 0.20),
    }

    print(f"  {'strategy':32s} {'CAGR':>6s} {'Sharpe':>7s} {'maxDD':>7s} {'2018':>7s} {'2022':>7s} {'WF':>4s}")
    print("  " + "-" * 76)
    for name, r in variants.items():
        m = _metrics_from_returns(r, [], name)
        pos = sum(1 for f in walk_forward_folds(r, 5) if f["sharpe"] > 0)
        y = (1 + r).groupby(r.index.year).prod() - 1
        print(f"  {name:32s} {m['cagr']:6.1%} {m['sharpe']:7.2f} {m['max_drawdown']:7.1%} "
              f"{y.get(2018,0):+7.1%} {y.get(2022,0):+7.1%} {pos}/5")

    print("\n  Verdict: regime-switch earns its complexity only if it beats the STATIC 70/30")
    print("  on Sharpe or drawdown. The 200d signal lags, so switching often just whipsaws.")


if __name__ == "__main__":
    main()
