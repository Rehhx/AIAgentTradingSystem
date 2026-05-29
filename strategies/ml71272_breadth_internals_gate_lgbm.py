"""
breadth_internals_gate_lgbm
===========================

Self-contained daily-horizon ML gate model. Trains a LightGBM (or
GradientBoostingClassifier fallback) classifier on synthetic "market internals"
features derived from a single OHLCV series — since we only have one symbol's
bars inside this engine, we proxy the breadth panel with rolling-window
statistics of price/volume that mimic the *information content* of the spec's
breadth features:

  - advance/decline slope        -> trailing up-bar fraction slope
  - % above 50/200 MA            -> close vs SMA-50 / SMA-200 distance
  - McClellan oscillator         -> EMA(19) - EMA(39) of up-down proxy
  - McClellan summation          -> cumulative oscillator
  - new highs - new lows         -> rolling argmax/argmin diff
  - equal-weight vs cap-weight   -> short-term mean return vs vol-weighted
  - sector dispersion            -> rolling std of returns of sub-windows
  - HY-SPY correlation           -> rolling corr of (close, |ret|*close) proxy
  - VIX percentile               -> rolling vol percentile

Target: sign of forward N-day return with magnitude > 0.5 * ATR (dead-zone
classification). Two heads (long / short) are derived from a single 3-class
target collapsed into P(up) - P(down).

Substitution note: spec mentions LightGBM; we fall back to
GradientBoostingClassifier if lightgbm is unavailable. No torch needed.

Hysteresis is applied to avoid overtrading: enter at prob_long / prob_short,
exit only when score crosses the dead-band (0.5).
"""

import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMClassifier as _Booster
    _HAS_LGBM = True
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier as _Booster
    _HAS_LGBM = False


def _safe_div(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        out = a / b
    return out.replace([np.inf, -np.inf], np.nan)


def _build_breadth_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)

    with np.errstate(invalid="ignore", divide="ignore"):
        ret = close.pct_change()
        logret = np.log(close.replace(0, np.nan)).diff()

    up = (ret > 0).astype(float)
    dn = (ret < 0).astype(float)

    f = pd.DataFrame(index=df.index)

    # advance/decline line slope proxy: slope of cumulative (up - down)
    ad_line = (up - dn).cumsum()
    f["ad_slope_10"] = ad_line.diff(10)
    f["ad_slope_30"] = ad_line.diff(30)

    # pct above 50 / 200 SMA proxy: distance of close from SMA
    sma50 = close.rolling(50, min_periods=20).mean()
    sma200 = close.rolling(200, min_periods=50).mean()
    f["above_sma50"] = _safe_div(close - sma50, sma50)
    f["above_sma200"] = _safe_div(close - sma200, sma200)

    # McClellan oscillator proxy: EMA19 - EMA39 of (up - down)
    ud = (up - dn).fillna(0.0)
    ema19 = ud.ewm(span=19, adjust=False).mean()
    ema39 = ud.ewm(span=39, adjust=False).mean()
    mcc = ema19 - ema39
    f["mcclellan_osc"] = mcc
    f["mcclellan_sum"] = mcc.cumsum()

    # new highs - new lows over 20d (rolling argmax/argmin proxy)
    win = 20
    rolling_max = close.rolling(win, min_periods=5).max()
    rolling_min = close.rolling(win, min_periods=5).min()
    new_high = (close >= rolling_max).astype(float)
    new_low = (close <= rolling_min).astype(float)
    f["nh_nl_5"] = (new_high - new_low).rolling(5, min_periods=1).sum()

    # equal-weight vs cap-weight proxy: ratio of mean return to volume-weighted return
    vw_num = (ret * vol).rolling(5, min_periods=2).sum()
    vw_den = vol.rolling(5, min_periods=2).sum()
    vw_ret = _safe_div(vw_num, vw_den)
    ew_ret = ret.rolling(5, min_periods=2).mean()
    f["ew_vs_vw_5"] = ew_ret - vw_ret

    # sector dispersion proxy: rolling std of returns across sub-windows
    f["dispersion_20"] = ret.rolling(20, min_periods=5).std()
    f["dispersion_60"] = ret.rolling(60, min_periods=10).std()

    # HY-SPY correlation proxy: corr between price and |return| (risk-on/off)
    abs_ret = ret.abs()
    f["hy_corr_60"] = close.rolling(60, min_periods=20).corr(abs_ret)

    # VIX percentile proxy: 252d percentile rank of realized vol
    rv = ret.rolling(20, min_periods=5).std()
    f["vix_pct_252"] = rv.rolling(252, min_periods=40).rank(pct=True)

    # ATR proxy on daily-ish scale, used for the target dead-zone
    tr = pd.concat(
        [
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    f["atr_14"] = tr.rolling(14, min_periods=3).mean()

    # short momentum to give the booster some direction baseline
    f["mom_5"] = logret.rolling(5, min_periods=2).sum()
    f["mom_20"] = logret.rolling(20, min_periods=5).sum()
    f["mom_60"] = logret.rolling(60, min_periods=10).sum()

    return f


def _build_target(df: pd.DataFrame, horizon: int, atr: pd.Series, k: float) -> pd.Series:
    close = df["close"].astype(float)
    fwd = close.shift(-horizon) / close - 1.0
    thresh = _safe_div(k * atr, close).fillna(np.nan)
    y = pd.Series(0, index=df.index, dtype=int)
    y[fwd > thresh] = 1
    y[fwd < -thresh] = -1
    return y


def signals(df: pd.DataFrame, params: dict) -> pd.Series:
    if df is None or len(df) < 1000:
        return pd.Series(0, index=df.index if df is not None else None, dtype=int)

    train_pct = float(params.get("train_pct", 0.5))
    prob_long = float(params.get("prob_long", 0.55))
    prob_short = float(params.get("prob_short", 0.45))
    horizon = int(params.get("horizon", 10))
    atr_k = float(params.get("atr_k", 0.5))
    n_estimators = int(params.get("n_estimators", 300))
    max_depth = int(params.get("max_depth", 4))
    learning_rate = float(params.get("learning_rate", 0.05))
    min_hold_bars = int(params.get("min_hold_bars", 5))

    # Build features
    feats = _build_breadth_features(df)
    atr = feats["atr_14"]

    # Build target (3-class with dead-zone)
    y = _build_target(df, horizon=horizon, atr=atr, k=atr_k)

    # Feature matrix excluding target leakage; atr stays as a level feature
    X = feats.copy()
    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    n = len(df)
    train_end = int(n * train_pct)
    # Avoid leakage: drop the last `horizon` rows of training set (target uses future)
    train_cut = max(50, train_end - horizon - 1)

    X_train = X.iloc[:train_cut]
    y_train = y.iloc[:train_cut]

    # Need at least 2 classes for classification
    classes_present = np.unique(y_train.values)
    if len(classes_present) < 2:
        return pd.Series(0, index=df.index, dtype=int)

    # Collapse to binary: predict P(up) using y==1 vs y!=1; and separately P(down)
    # We use a single 3-class model when possible.
    try:
        if _HAS_LGBM:
            model = _Booster(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                num_leaves=int(params.get("num_leaves", 31)),
                min_child_samples=int(params.get("min_child_samples", 30)),
                subsample=float(params.get("subsample", 0.8)),
                colsample_bytree=float(params.get("colsample_bytree", 0.8)),
                reg_lambda=float(params.get("reg_lambda", 1.0)),
                verbosity=-1,
                n_jobs=1,
            )
        else:
            model = _Booster(
                n_estimators=min(n_estimators, 200),
                max_depth=max_depth,
                learning_rate=learning_rate,
            )
        model.fit(X_train.values, y_train.values)
    except Exception:
        return pd.Series(0, index=df.index, dtype=int)

    # Predict on the entire test region
    X_test = X.iloc[train_end:]
    if len(X_test) == 0:
        return pd.Series(0, index=df.index, dtype=int)

    try:
        proba = model.predict_proba(X_test.values)
        classes = list(model.classes_)
    except Exception:
        return pd.Series(0, index=df.index, dtype=int)

    def _col(cls):
        return classes.index(cls) if cls in classes else None

    up_idx = _col(1)
    dn_idx = _col(-1)

    p_up = proba[:, up_idx] if up_idx is not None else np.zeros(len(X_test))
    p_dn = proba[:, dn_idx] if dn_idx is not None else np.zeros(len(X_test))

    # Score in [-1, 1]: positive = up edge, negative = down edge
    score = pd.Series(p_up - p_dn, index=X_test.index)

    # Convert thresholds (centered around 0.5 for class prob) into score space.
    # prob_long > 0.5 => score must exceed (prob_long - 0.5)*2 roughly; we use directly
    long_enter = prob_long - 0.5
    short_enter = 0.5 - prob_short
    # Dead band exits: score must cross 0 (sign change) to flip
    long_exit = 0.0
    short_exit = 0.0

    # Hysteresis state machine with a minimum holding period
    out_test = np.zeros(len(score), dtype=int)
    state = 0
    held = 0
    s_vals = score.values
    for i in range(len(s_vals)):
        v = s_vals[i]
        if np.isnan(v):
            out_test[i] = state
            held += 1
            continue
        if state == 0:
            if v >= long_enter:
                state = 1
                held = 0
            elif v <= -short_enter:
                state = -1
                held = 0
        elif state == 1:
            if held >= min_hold_bars and v <= long_exit:
                # allow direct flip to short on strong opposite signal
                if v <= -short_enter:
                    state = -1
                else:
                    state = 0
                held = 0
            else:
                held += 1
        elif state == -1:
            if held >= min_hold_bars and v >= short_exit:
                if v >= long_enter:
                    state = 1
                else:
                    state = 0
                held = 0
            else:
                held += 1
        out_test[i] = state

    # Assemble full series, zeros before train_end
    out = pd.Series(0, index=df.index, dtype=int)
    out.iloc[train_end:] = out_test

    return out
