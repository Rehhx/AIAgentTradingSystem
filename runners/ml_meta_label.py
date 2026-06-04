"""
runners/ml_meta_label.py  (Tier 3 follow-up — the principled next try)
----------------------------------------------------------------------
Meta-labeling (Lopez de Prado, AFML ch. 3). Directional prediction failed in
ml_signal.py (OOS AUC ~0.50 — forecasting the sign of a daily move is a coin
flip). So we DON'T predict direction. We let the PROVEN trend rule (50/200) pick
the direction, and use ML only to decide whether to TAKE each long it proposes:

  primary model  : sig_trend_5020   -> says WHEN to be long
  meta model (ML): P(this long trade is profitable | features) -> says IF to take it

That turns an impossible forecasting problem into a tractable precision/filtering
one: the model can add value by SKIPPING low-quality trades even if it can't
predict direction. Validated the same way as everything else: PurgedKFold OOS,
judged against the RAW rule and buy-hold (not zero). One fixed model config — no
grid search — so there's no selection bias to deflate. Honest result either way.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from agents.daily_strategies import (
    daily_bars, sleeve_returns, sig_trend_5020, SIDE_COST, TRADING_DAYS,
    _metrics_from_returns,
)
from analytics.significance import sharpe_stats, probabilistic_sharpe_ratio
from ml.cv import PurgedKFold
from ml.labels import get_daily_vol, triple_barrier_labels
from ml.features import make_features


def _oos_proba(model, X, y, cv):
    p = pd.Series(np.nan, index=X.index)
    for tr, te in cv.split(X):
        if y.iloc[tr].nunique() < 2:        # need both classes to fit
            continue
        model.fit(X.iloc[tr], y.iloc[tr])
        p.iloc[te] = model.predict_proba(X.iloc[te])[:, 1]
    return p


def _net(pos: pd.Series, close: pd.Series) -> pd.Series:
    """long/flat position -> costed daily return (enter next day, 3bps/side)."""
    ret = close.pct_change()
    pos = pos.reindex(close.index).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs())
    return pos.shift(1).fillna(0.0) * ret - turn.shift(1).fillna(0.0) * SIDE_COST


def _ann(r):
    s = sharpe_stats(r.to_numpy())
    return s["sr"] * np.sqrt(TRADING_DAYS)


def main(symbol: str = "SPY"):
    print(f"meta-labeling the trend rule on {symbol} ...\n")
    d = daily_bars(symbol)
    close = d["close"]

    primary = sig_trend_5020(d).reindex(close.index).fillna(0.0)     # 0/1 direction
    vol = get_daily_vol(close, span=20)
    labels = triple_barrier_labels(close, horizon=10, pt=1.5, sl=1.5, vol=vol)
    X_all = make_features(close)
    feat_cols = list(X_all.columns)

    # events = bars where the primary rule wants to be LONG; meta-label = did it pay?
    df = labels.join(X_all, how="inner")
    df["primary"] = primary.reindex(df.index).fillna(0.0)
    df = df[df["primary"] == 1.0].dropna()
    X, y, t1 = df[feat_cols], (df["ret"] > 0).astype(int), df["t1"]
    base = y.mean()
    print(f"  {len(df)} long-signal events ({df.index[0].date()}..{df.index[-1].date()})")
    print(f"  base rate P(long trade profitable) = {base:.1%}\n")

    cv = PurgedKFold(n_splits=6, t1=t1, pct_embargo=0.01)
    model = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                       learning_rate=0.05, subsample=0.8, random_state=0)
    proba = _oos_proba(model, X, y, cv).dropna()
    yz = y.reindex(proba.index)
    take = proba > 0.5                                  # take the trade only if meta approves
    auc = roc_auc_score(yz, proba)
    coverage = float(take.mean())
    precision = float(yz[take].mean()) if take.sum() else float("nan")

    print("  META-MODEL (filtering quality):")
    print(f"    OOS AUC                         {auc:.3f}   (0.50 = no skill)")
    print(f"    trades kept (coverage)          {coverage:.1%}")
    print(f"    precision P(profit | take)      {precision:.1%}  vs base {base:.1%} "
          f"-> lift {precision - base:+.1%}")

    # build the meta-filtered position: long only when primary long AND meta approves
    meta_pos = pd.Series(0.0, index=close.index)
    meta_pos.loc[proba.index[take.values]] = 1.0

    win = close.index[(close.index >= proba.index.min()) & (close.index <= proba.index.max())]
    bh = close.pct_change().reindex(win).fillna(0.0)
    raw_net = _net(primary, close).reindex(win).fillna(0.0)
    meta_net = _net(meta_pos, close).reindex(win).fillna(0.0)

    print("\n" + "=" * 60)
    print("META-FILTERED TREND  vs  RAW TREND  vs  BUY-HOLD  (same window)")
    print("=" * 60)
    print(f"  {'strategy':22s} {'Sharpe':>7s} {'maxDD':>7s} {'CAGR':>7s}")
    print("  " + "-" * 46)
    rows = [("buy & hold", bh), ("raw trend_5020", raw_net), ("meta-filtered trend", meta_net)]
    M = {}
    for name, r in rows:
        m = _metrics_from_returns(r, [], name)
        M[name] = m
        print(f"  {name:22s} {m['sharpe']:>7.2f} {m['max_drawdown']:>7.1%} {m['cagr']:>7.1%}")

    raw_sh = M["raw trend_5020"]["sharpe"]
    meta_sh = M["meta-filtered trend"]["sharpe"]
    # PSR that the meta strategy's Sharpe exceeds the RAW rule's Sharpe (the real bar)
    s = sharpe_stats(meta_net.to_numpy())
    raw_period = sharpe_stats(raw_net.to_numpy())["sr"]
    psr_vs_raw = probabilistic_sharpe_ratio(s["sr"], s["n"], s["skew"], s["kurt"],
                                            sr_benchmark=raw_period)

    print("\n" + "=" * 60)
    print("HONEST VERDICT (does ML filtering beat the raw rule?)")
    print("=" * 60)
    print(f"  meta Sharpe {meta_sh:.2f}  vs  raw-rule Sharpe {raw_sh:.2f}  "
          f"(dSharpe {meta_sh - raw_sh:+.2f})")
    print(f"  maxDD {M['meta-filtered trend']['max_drawdown']:.1%} vs raw "
          f"{M['raw trend_5020']['max_drawdown']:.1%}")
    print(f"  P(meta Sharpe > raw Sharpe)     {psr_vs_raw:.1%}")
    if meta_sh > raw_sh + 0.10 and psr_vs_raw >= 0.90:
        print("  -> ML meta-filter ADDS value: higher risk-adjusted return than the raw")
        print("     rule, confirmed against the right benchmark. Worth wiring in.")
    elif meta_sh >= raw_sh - 0.05:
        print("  -> ROUGHLY NEUTRAL: the filter neither clearly helps nor hurts. Not")
        print("     worth the added complexity/overfitting surface. Keep the raw rule.")
    else:
        print("  -> ML meta-filter HURTS: it removes good trades, not just bad ones.")
        print("     Honest negative. The raw rule stands.")


if __name__ == "__main__":
    main()
