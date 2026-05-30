"""
runners/grind_fix_test.py
-------------------------
Targeted fix for the "Bear . calm" gap (book -9.8% vs SPY -2.9% in quiet
downtrends). The earlier 200d leverage cap backfired because Bear.calm also holds
recovery bounces. This rule de-risks ONLY when SPY is below BOTH its 50d AND 200d
(a confirmed grind-down) -- recovery bounces (below 200d but back above 50d) are
left fully invested. De-risked capital earns T-bills (BIL).
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


def overlays_grind(combo, idx, f, vt=0.17, maxlev=1.8):
    spy = daily_bars("SPY")["close"].reindex(idx)
    below50 = spy < spy.rolling(50).mean()
    below200 = spy < spy.rolling(200).mean()
    grind = (below50 & below200).shift(1).fillna(False).astype(bool)   # confirmed downtrend
    warn = (spy < spy.rolling(50).mean()) & (spy.pct_change().rolling(20).std() * np.sqrt(TD) > 0.20)
    ews = (1 - 0.4 * warn.astype(float)).shift(1).fillna(1.0)
    bil = daily_bars("BIL")["close"].pct_change().reindex(idx).fillna(0.0)
    eq = vol_target(combo, vt, max_leverage=maxlev) * ews
    eq2 = eq.where(~grind, eq * f)                                  # scale exposure to f in grind
    extra = pd.Series(np.where(grind.to_numpy(), (1 - f) * bil.to_numpy(), 0.0), index=idx)
    return eq2 + 0.22 * bil + extra                                 # freed capital -> BIL


def regimes(idx):
    spy = daily_bars("SPY")["close"].reindex(idx)
    sret = spy.pct_change().fillna(0)
    up = spy > spy.rolling(200).mean()
    calm = (sret.rolling(20).std() * np.sqrt(TD)) < 0.20
    reg = pd.Series("", index=idx)
    reg[up & calm] = "Bull . calm"; reg[up & ~calm] = "Bull . stormy"
    reg[~up & calm] = "Bear . calm"; reg[~up & ~calm] = "Bear . stormy"
    return reg


def main():
    print("building book + grind-down de-risk variants ...\n")
    panel = build_base()
    base = sum(panel[c].fillna(0) * W[c] for c in W)
    idx = panel.index
    lvd = make_defensive(lowvol_factor()).reindex(idx).fillna(0)
    combo = base * 0.90 + lvd * 0.10
    reg = regimes(idx)

    variants = {"current (no fix)": overlays(combo, idx)}
    for f in (0.5, 0.3, 0.0):
        variants[f"grind de-risk to {f:.0%}"] = overlays_grind(combo, idx, f)

    print(f"{'variant':22s} {'Bear.calm':>10s} {'Bear.storm':>11s} {'Bull.calm':>10s} "
          f"{'Sharpe':>7s} {'CAGR':>7s} {'DD':>7s} {'WF':>4s}")
    print("-" * 90)
    for name, b in variants.items():
        b = b.fillna(0)
        m = _metrics_from_returns(b, [], name)
        pos = sum(1 for fld in walk_forward_folds(b, 5) if fld["sharpe"] > 0)
        bc = b[reg == "Bear . calm"].mean() * TD
        bs = b[reg == "Bear . stormy"].mean() * TD
        bu = b[reg == "Bull . calm"].mean() * TD
        print(f"{name:22s} {bc:10.1%} {bs:11.1%} {bu:10.1%} "
              f"{m['sharpe']:7.2f} {m['cagr']:7.1%} {m['max_drawdown']:7.1%} {pos}/5")

    # recovery-capture check: the 2020 spring rebound fold should stay strong
    print("\nrecovery-capture check (2020-03..2020-12 cumulative, the post-COVID rebound):")
    for name, b in variants.items():
        w = b.fillna(0).loc["2020-03-23":"2020-12-31"]
        print(f"  {name:22s} {((1+w).prod()-1):+.1%}")


if __name__ == "__main__":
    main()
