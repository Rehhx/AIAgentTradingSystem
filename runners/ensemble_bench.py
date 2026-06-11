"""
runners/ensemble_bench.py
-------------------------
Strategy search benchmarked against OUR equity ensemble book (no SPY). The book
(7 sleeves + crash sentinel, no-margin) is the bar to clear. Two ways a candidate
"wins":

  1. standalone Sharpe at or near the ensemble's, OR
  2. (the real prize) it DIVERSIFIES -- a lower-Sharpe but low-correlation sleeve
     can RAISE the blended Sharpe. So we measure each candidate's marginal
     contribution: Sharpe of (85% ensemble + 15% candidate) vs the ensemble alone.

A candidate earns a look if it improves the blend (delta > 0) or its standalone
Sharpe is within ~0.25 of the ensemble. 2016-2026, vol-targeted, costs included.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    STRATEGIES_DAILY, CANDIDATE_STRATEGIES, DEPLOY_PARAMS, QUALITY_UNIVERSE,
    backtest_book, vol_target, _metrics_from_returns, TRADING_DAYS,
)
from runners.diversifier_screen import build_base, W
from runners.sentinel_book_wf import overlay, _vix_sentinel
from runners.final_tests import lowvol_factor
from runners.lowvol_defensive import make_defensive


def _naive(s):
    s = s.copy()
    s.index = pd.DatetimeIndex(s.index)
    s.index = (s.index.tz_convert("UTC").tz_localize(None) if s.index.tz else s.index).normalize()
    return s


def _sh(r):
    r = r.dropna()
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def build_ensemble():
    """the deployed equity book: 7 sleeves + lowvol + VIX crash sentinel, no-margin."""
    panel = build_base()
    idx = panel.index
    lvd = make_defensive(lowvol_factor()).reindex(idx).fillna(0)
    combo = sum(panel[c].fillna(0) * W[c] for c in W) * 0.90 + lvd * 0.10
    return _naive(overlay(combo, idx, sentinel=_vix_sentinel(idx), maxlev=1.0))


def candidate_series(name, fn):
    b = backtest_book(fn, QUALITY_UNIVERSE, DEPLOY_PARAMS.get(name), label=name)
    if "error" in b:
        return None
    return _naive(vol_target(b["_returns"], target_vol=0.12, max_leverage=1.0))


def main():
    print("building the equity ensemble benchmark (build_base ~1-2 min) ...\n")
    ens = build_ensemble()
    ens_sh = _sh(ens)
    em = _metrics_from_returns(ens, [], "ensemble")
    print("=" * 78)
    print(f"BENCHMARK = equity ensemble book   Sharpe {ens_sh:.2f} | CAGR {em['cagr']:.1%} "
          f"| maxDD {em['max_drawdown']:.1%}")
    print("=" * 78)
    print(f"  {'candidate':18s} {'Sharpe':>7s} {'maxDD':>7s} {'corr->ens':>10s} "
          f"{'blend Sharpe':>12s} {'delta':>7s}")
    print("  " + "-" * 70)

    cands = {**STRATEGIES_DAILY, **CANDIDATE_STRATEGIES}
    rows = []
    for name, fn in cands.items():
        try:
            r = candidate_series(name, fn)
        except Exception as e:
            print(f"  skip {name}: {e}")
            continue
        if r is None:
            continue
        common = ens.index.intersection(r.index)
        if len(common) < 250:
            continue
        e, c = ens.reindex(common).fillna(0.0), r.reindex(common).fillna(0.0)
        sh = _sh(c)
        dd = _metrics_from_returns(c, [], name)["max_drawdown"]
        corr = float(c.corr(e))
        blend = 0.85 * e + 0.15 * c            # marginal contribution of a 15% sleeve
        bsh = _sh(blend)
        rows.append({"name": name, "sh": sh, "dd": dd, "corr": corr,
                     "bsh": bsh, "delta": bsh - _sh(e)})

    rows.sort(key=lambda x: x["delta"], reverse=True)
    for x in rows:
        flag = " <-- improves blend" if x["delta"] > 0.005 else (
               "  ~ close" if x["sh"] >= ens_sh - 0.25 else "")
        print(f"  {x['name']:18s} {x['sh']:>7.2f} {x['dd']:>7.1%} {x['corr']:>10.2f} "
              f"{x['bsh']:>12.2f} {x['delta']:>+7.2f}{flag}")

    improvers = [x for x in rows if x["delta"] > 0.005]
    close = [x for x in rows if x["sh"] >= ens_sh - 0.25]
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if improvers:
        best = improvers[0]
        print(f"  {len(improvers)} candidate(s) RAISE the blended Sharpe. Best: {best['name']} "
              f"-> {best['bsh']:.2f} (vs {ens_sh:.2f}), corr {best['corr']:+.2f}.")
        print("  Low correlation is doing the work, not standalone Sharpe. Worth a")
        print("  proper walk-forward + deflated-Sharpe before adding to the book.")
    else:
        print(f"  No candidate raises the blend at 15% weight. The ensemble ({ens_sh:.2f})")
        print("  already absorbs these mechanisms -- consistent with the diversification")
        print("  ceiling. To move it you need a genuinely uncorrelated return stream")
        print("  (managed futures / market-neutral), not another long-equity sleeve.")
    if close:
        print(f"  Standalone-close (within 0.25): {', '.join(x['name'] for x in close)}")


if __name__ == "__main__":
    main()
