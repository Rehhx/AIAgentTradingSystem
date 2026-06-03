"""
runners/combined_system.py
--------------------------
The honest path to a higher SYSTEM Sharpe: not a mythical standalone sleeve, but
combining uncorrelated engines. Blends the three genuine engines we have over the
common window and measures the combined Sharpe + correlation matrix + walk-forward:

  E1  Equity book (7 sleeves + VIX sentinel, no-margin)   -- the workhorse
  E2  Managed futures, time-series momentum               -- crisis alpha (2008)
  E3  Managed futures, cross-sectional momentum           -- crisis alpha (2022)

Combination is RISK-BASED (inverse-vol), not optimized weights (a fitted optimizer
overfit and lost OOS — LESSONS 8). Diversification can lift Sharpe even though E2/E3
are weak standalone, because they're ~uncorrelated and crisis-positive. 2016-2026.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from agents.daily_strategies import _metrics_from_returns, walk_forward_folds, TRADING_DAYS
from runners.diversifier_screen import build_base, W
from runners.sentinel_book_wf import overlay, _vix_sentinel
from runners.futures_xsmom import _prices as mf_prices, time_series_mom, cross_sectional_mom
from runners.final_tests import lowvol_factor
from runners.lowvol_defensive import make_defensive

CRASHES = {"2018 Q4": ("2018-10-01", "2018-12-24"), "COVID": ("2020-02-19", "2020-03-23"),
           "2022 bear": ("2022-01-03", "2022-10-12")}


def _naive(s):
    s = s.copy()
    s.index = pd.DatetimeIndex(s.index)
    s.index = (s.index.tz_convert("UTC").tz_localize(None) if s.index.tz else s.index).normalize()
    return s


def main():
    print("building 3 engines (equity+sentinel, MF time-series, MF cross-sectional) ~2 min ...\n")
    # E1 — equity book + sentinel
    panel = build_base(); idx = panel.index
    lvd = make_defensive(lowvol_factor()).reindex(idx).fillna(0)
    combo = sum(panel[c].fillna(0) * W[c] for c in W) * 0.90 + lvd * 0.10
    e1 = _naive(overlay(combo, idx, sentinel=_vix_sentinel(idx), maxlev=1.0))
    # E2/E3 — managed futures
    C = mf_prices(); R = C.pct_change(); vol = R.rolling(60).std()
    e2 = _naive(time_series_mom(C, R, vol))
    e3 = _naive(cross_sectional_mom(C, R, vol))

    common = e1.index.intersection(e2.index).intersection(e3.index)
    E = pd.DataFrame({"equity+sentinel": e1, "MF time-series": e2, "MF cross-sec": e3}).reindex(common).fillna(0.0)
    print(f"  common window {common[0].date()}..{common[-1].date()} ({len(common)} days)\n")

    print("  CORRELATION MATRIX (low/negative off-diagonal = diversification):")
    cm = E.corr()
    print(cm.round(2).to_string().replace("\n", "\n    "))
    print()
    print(f"  {'engine':22s} {'CAGR':>6s} {'Sharpe':>7s} {'maxDD':>7s}")
    for c in E.columns:
        m = _metrics_from_returns(E[c], [], c)
        print(f"  {c:22s} {m['cagr']:>6.1%} {m['sharpe']:>7.2f} {m['max_drawdown']:>7.1%}")

    # inverse-vol (risk-parity) weights, plus a couple of fixed equity-tilted blends
    iv = 1.0 / E.std()
    iv = iv / iv.sum()
    blends = {
        "inverse-vol (risk parity)": (E * iv.values).sum(axis=1),
        "70/15/15 equity-tilt":      0.70 * E["equity+sentinel"] + 0.15 * E["MF time-series"] + 0.15 * E["MF cross-sec"],
        "60/20/20":                  0.60 * E["equity+sentinel"] + 0.20 * E["MF time-series"] + 0.20 * E["MF cross-sec"],
        "85/7.5/7.5 equity-heavy":   0.85 * E["equity+sentinel"] + 0.075 * E["MF time-series"] + 0.075 * E["MF cross-sec"],
    }

    print("\n" + "=" * 72)
    print("COMBINED SYSTEM — blends of the 3 engines (2016-2026)")
    print("=" * 72)
    print(f"  {'blend':30s} {'CAGR':>6s} {'Sharpe':>7s} {'maxDD':>7s} {'WF':>5s} {'COVID':>7s} {'2022':>7s}")
    print("  " + "-" * 70)
    base = _metrics_from_returns(E["equity+sentinel"], [], "e1")
    print(f"  {'equity+sentinel ALONE':30s} {base['cagr']:>6.1%} {base['sharpe']:>7.2f} {base['max_drawdown']:>7.1%}"
          f" {'5/5':>5s}")
    best = None
    for name, r in blends.items():
        m = _metrics_from_returns(r, [], name)
        folds = walk_forward_folds(r, 5); pos = sum(1 for f in folds if f["sharpe"] > 0)
        def tot(a, b):
            w = (1 + r.loc[a:b]).cumprod()
            return (w.iloc[-1] / w.iloc[0] - 1) if len(w) else 0
        cov = tot(*CRASHES["COVID"]); y22 = (1 + r.loc["2022-01-01":"2022-12-31"]).prod() - 1
        print(f"  {name:30s} {m['cagr']:>6.1%} {m['sharpe']:>7.2f} {m['max_drawdown']:>7.1%} {pos}/5"
              f" {cov:>+7.1%} {y22:>+7.1%}")
        if best is None or m["sharpe"] > best[1]:
            best = (name, m["sharpe"])

    print("\n" + "=" * 72)
    print(f"  Best blend: {best[0]} at Sharpe {best[1]:.2f}  (vs equity-alone {base['sharpe']:.2f})")
    print("  Honest read: diversification lifts the SYSTEM Sharpe and cuts the crash. If the")
    print("  best blend is ~1.6-1.8, that's the real ceiling — reaching a TRUE 2.0 needs either")
    print("  a genuinely new uncorrelated alpha (scarce) or leverage (which re-adds tail risk).")


if __name__ == "__main__":
    main()
