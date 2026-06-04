"""
runners/rigor_report.py  (BUILD_PLAN.md Tier 1A)
------------------------------------------------
The honest scorecard. Takes EVERY strategy we tried (deployed + candidates),
backtests each as an equal-weight book over the quality universe, and runs the
statistical-rigor battery on the whole set:

  * per-strategy annualized Sharpe              -- the naive numbers
  * Deflated Sharpe Ratio on the BEST one       -- corrects for N trials + non-normal
  * Probability of Backtest Overfitting (CSCV)  -- does selection generalize OOS?
  * White's Reality Check + Hansen's SPA        -- does the best beat buy-&-hold SPY?

This is the difference between "I backtested some strategies" and "I corrected
for the multiple-testing problem". The numbers it prints feed RESEARCH.md (1B).
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
    backtest_book, daily_bars, TRADING_DAYS,
)
from analytics.significance import dsr_from_trials, sharpe_stats
from analytics.pbo import cscv_pbo
from analytics.reality_check import whites_reality_check, hansen_spa


def build_trials() -> pd.DataFrame:
    """one aligned return column per strategy we tried (deployed + candidates)."""
    funcs = {**STRATEGIES_DAILY, **CANDIDATE_STRATEGIES}
    series = {}
    for name, fn in funcs.items():
        try:
            b = backtest_book(fn, QUALITY_UNIVERSE, DEPLOY_PARAMS.get(name), label=name)
            if "error" not in b:
                series[name] = b["_returns"]
        except Exception as e:
            print(f"  skip {name}: {e}")
    df = pd.concat(series, axis=1).dropna(how="all").fillna(0.0)
    return df


def main():
    print("backtesting every strategy we tried (deployed + candidates) ~30s ...\n")
    R = build_trials()
    names = list(R.columns)
    print(f"  {len(names)} strategies x {len(R)} days "
          f"({R.index[0].date()}..{R.index[-1].date()})\n")

    stats = {n: sharpe_stats(R[n].to_numpy()) for n in names}
    ann = {n: stats[n]["sr"] * np.sqrt(TRADING_DAYS) for n in names}
    order = sorted(names, key=lambda n: ann[n], reverse=True)

    print(f"  {'strategy':18s} {'Sharpe(ann)':>11s} {'skew':>7s} {'kurt':>7s}")
    print("  " + "-" * 46)
    for n in order:
        print(f"  {n:18s} {ann[n]:>11.2f} {stats[n]['skew']:>7.2f} {stats[n]['kurt']:>7.2f}")

    best = order[0]
    trial_srs = [stats[n]["sr"] for n in names]

    print("\n" + "=" * 64)
    print(f"DEFLATED SHARPE RATIO  -- best strategy: {best}")
    print("=" * 64)
    d = dsr_from_trials(R[best].to_numpy(), trial_srs, periods=TRADING_DAYS)
    print(f"  naive Sharpe (annual)         {d['sr_annual']:>8.2f}")
    print(f"  trials searched (N)           {d['n_trials']:>8d}")
    print(f"  E[max Sharpe under null]      {d['sr_star_annual']:>8.2f}   (the deflation hurdle)")
    print(f"  PSR vs 0                      {d['psr_vs_zero']:>8.1%}")
    print(f"  DEFLATED SHARPE (DSR)         {d['dsr']:>8.1%}    <- P(true SR > selection hurdle)")
    mtrl = d["min_track_record_length"]
    yrs = mtrl / TRADING_DAYS if np.isfinite(mtrl) else float("inf")
    print(f"  min track-record length       {mtrl:>8.0f} obs (~{yrs:.1f} yrs)")
    verdict = ("PASS -- survives selection bias" if d["dsr"] >= 0.95 else
               "MARGINAL" if d["dsr"] >= 0.90 else
               "FAIL -- not distinguishable from luck-of-N-trials")
    print(f"  verdict: {verdict}")

    print("\n" + "=" * 64)
    print("PROBABILITY OF BACKTEST OVERFITTING  (CSCV)")
    print("=" * 64)
    n_splits = 16 if len(R) >= 16 * 20 else 10
    p = cscv_pbo(R.to_numpy(), n_splits=n_splits)
    print(f"  splits={p['n_splits']}  combinations={p['n_combinations']}")
    print(f"  PBO = {p['pbo']:.1%}    (0%=generalizes, 50%=noise, 100%=overfit)")
    print(f"  median OOS-rank logit {p['logits_median']:+.2f}")

    print("\n" + "=" * 64)
    print("DATA-SNOOPING  -- does the best strategy beat buy-&-hold SPY?")
    print("=" * 64)
    spy = daily_bars("SPY")["close"].pct_change().reindex(R.index).fillna(0.0)
    rc = whites_reality_check(R.to_numpy(), benchmark=spy.to_numpy(), n_boot=2000)
    spa = hansen_spa(R.to_numpy(), benchmark=spy.to_numpy(), n_boot=2000)
    print(f"  White's Reality Check   p = {rc['p_value']:.3f}")
    print(f"  Hansen's SPA            p = {spa['p_value']:.3f}")
    print("  (p < 0.05 => the best strategy's edge over passive SPY survives snooping)")

    print("\nDone. Put these figures in RESEARCH.md (Tier 1B).")


if __name__ == "__main__":
    main()
