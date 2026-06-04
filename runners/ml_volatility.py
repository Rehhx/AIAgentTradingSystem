"""
runners/ml_volatility.py  (Tier 3 — where ML actually works)
------------------------------------------------------------
Direction is unpredictable (ml_signal.py, ml_meta_label.py: AUC ~0.50). VOLATILITY
is not — it clusters, so tomorrow's vol is forecastable from today's. This runs the
SAME rigorous pipeline (trailing-only features, PurgedKFold OOS) against forward
realized volatility and compares the model to the naive persistence baseline
("forward vol = recent vol").

Two purposes:
  1. Prove the machinery is sound — a clear OOS win here means the earlier negatives
     were about the SIGNAL (direction is noise), not a broken pipeline.
  2. Show the CORRECT use of ML in this book: forecasting vol feeds the vol-targeting
     / de-risk overlay (size down BEFORE the spike, not after) — risk, not direction.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

from agents.daily_strategies import daily_bars, TRADING_DAYS
from ml.cv import PurgedKFold

H = 10  # forecast horizon (days)


def main(symbol: str = "SPY"):
    print(f"forecasting {H}-day forward realized volatility on {symbol} ...\n")
    close = daily_bars(symbol)["close"]
    ret = close.pct_change()
    ann = np.sqrt(TRADING_DAYS)

    def rv(w):
        return ret.rolling(w).std() * ann

    # label: realized vol over the NEXT H days (annualized), aligned to decision time t
    fwd = ret.rolling(H).std().shift(-H) * ann

    # trailing-only features (all <= t): vol at several horizons, term structure,
    # downside vol, vol-of-vol, and recent return (leverage effect: down -> higher vol)
    F = pd.DataFrame(index=close.index)
    F["rv_5"], F["rv_10"], F["rv_21"], F["rv_63"] = rv(5), rv(10), rv(21), rv(63)
    F["rv_ratio"] = F["rv_5"] / F["rv_63"]
    F["abs_ret_5"] = ret.abs().rolling(5).mean() * ann
    F["down_rv_21"] = ret.clip(upper=0).rolling(21).std() * ann
    F["vov_21"] = rv(21).rolling(21).std()
    F["ret_5"], F["ret_21"] = close.pct_change(5), close.pct_change(21)

    df = F.join(fwd.rename("y")).dropna()
    n, idx = len(df), df.index
    X, y = df[F.columns], df["y"]
    baseline = df["rv_10"]                          # persistence forecast: fwd vol ~ trailing 10d vol
    t1 = pd.Series([idx[min(i + H, n - 1)] for i in range(n)], index=idx)
    print(f"  {n} obs ({idx[0].date()}..{idx[-1].date()}), {X.shape[1]} features\n")

    cv = PurgedKFold(n_splits=6, t1=t1, pct_embargo=0.01)
    model = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                      learning_rate=0.03, subsample=0.8, random_state=0)
    pred = pd.Series(np.nan, index=idx)
    for tr, te in cv.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        pred.iloc[te] = model.predict(X.iloc[te])
    pred = pred.dropna()
    yz, bz = y.reindex(pred.index), baseline.reindex(pred.index)

    r2_model, r2_base = r2_score(yz, pred), r2_score(yz, bz)
    corr_model = float(np.corrcoef(yz, pred)[0, 1])
    corr_base = float(np.corrcoef(yz, bz)[0, 1])

    print("=" * 60)
    print("FORWARD-VOL FORECAST -- out-of-sample (purged CV)")
    print("=" * 60)
    print(f"  {'predictor':24s} {'OOS R^2':>9s} {'corr':>7s}")
    print("  " + "-" * 42)
    print(f"  {'persistence (recent vol)':24s} {r2_base:>9.3f} {corr_base:>7.3f}")
    print(f"  {'gradient-boosted ML':24s} {r2_model:>9.3f} {corr_model:>7.3f}")
    print()
    print(f"  contrast: DIRECTION model AUC ~0.50 (no skill) vs VOL R^2 {r2_model:.2f}")
    print("  same features, same CV -- the pipeline is sound; direction is just noise.")

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if r2_model > 0.2 and r2_model >= r2_base - 0.02:
        edge = r2_model - r2_base
        print(f"  Volatility IS predictable (OOS R^2 {r2_model:.2f}). ML matches/beats the")
        print(f"  persistence baseline by {edge:+.3f} R^2 (it captures the asymmetry:")
        print("  vol jumps on down moves). This is the right home for ML in the book --")
        print("  feed the forecast into vol_target/the de-risk overlay to cut exposure")
        print("  BEFORE a vol spike. ML for RISK, not direction.")
    else:
        print(f"  OOS R^2 {r2_model:.2f}. Even vol is hard to beat-the-baseline on here;")
        print("  persistence is tough to improve. Use the simple trailing-vol estimate.")


if __name__ == "__main__":
    main()
