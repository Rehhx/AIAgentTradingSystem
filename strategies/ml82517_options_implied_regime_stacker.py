"""
options_implied_regime_stacker
==============================

Self-contained ML strategy that proxies options-implied regime indicators
(VIX, VIX term structure, SKEW, VVIX, PCR) from OHLCV bars and uses a
GradientBoostingClassifier meta-label to gate directional exposure.

Substitution note: the spec calls for live options-panel data (VIX9D / VIX /
VIX3M / SKEW / VVIX / equity & index PCR). Because this module is constrained
to OHLCV input only, each options-implied feature is reconstructed from a
realised-volatility / return-distribution / volume proxy on the bar stream:
    - vix_level            -> annualised realised vol (long window)
    - vix9d_vix_ratio      -> short-window RV / long-window RV
    - vix_vix3m_ratio      -> long-window RV / very-long-window RV
    - skew_index           -> rolling return skewness (sign-flipped, scaled)
    - skew_minus_vix       -> skewness proxy minus rv proxy
    - put_call_ratio_*     -> down-volume / up-volume EMA proxies
    - spx_25d_put_iv_atm   -> downside / upside semi-variance ratio
    - vvix_level           -> rolling std of the RV proxy (vol of vol)
    - vvix_vix_ratio       -> vvix proxy / vix proxy
    - rsi2_signal_strength -> RSI(2) deviation from 50, scaled
    - trend_5020_signal_strength -> normalised EMA(50)-EMA(200) gap

Hysteresis is applied to the predicted probability so the signal persists
across noise (long if p >= prob_long, exit only when p < 0.5; symmetric
short).
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier


def _safe_log_ret(close: pd.Series) -> pd.Series:
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.log(close / close.shift(1))
    return r.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=max(int(span), 2), adjust=False, min_periods=1).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    period = max(int(period), 2)
    roll_up = up.ewm(alpha=1.0 / period, adjust=False, min_periods=1).mean()
    roll_dn = dn.ewm(alpha=1.0 / period, adjust=False, min_periods=1).mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = roll_up / roll_dn.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.fillna(50.0)


def _build_feature_panel(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float).clip(lower=0.0)

    ret = _safe_log_ret(close)

    short_w = int(params.get("rv_short_window", 30))
    mid_w = int(params.get("rv_mid_window", 120))
    long_w = int(params.get("rv_long_window", 480))

    rv_short = ret.rolling(short_w, min_periods=short_w).std()
    rv_mid = ret.rolling(mid_w, min_periods=mid_w).std()
    rv_long = ret.rolling(long_w, min_periods=long_w).std()

    # VIX-level proxy: annualised long-window RV scaled to a VIX-like range.
    vix_level = (rv_mid * np.sqrt(252.0 * 390.0) * 100.0)

    # Term-structure-style ratios.
    with np.errstate(invalid="ignore", divide="ignore"):
        vix9d_vix_ratio = rv_short / rv_mid.replace(0.0, np.nan)
        vix_vix3m_ratio = rv_mid / rv_long.replace(0.0, np.nan)

    # Skew / downside-tail proxies.
    skew_index = -ret.rolling(mid_w, min_periods=mid_w).skew() * 50.0 + 100.0
    skew_minus_vix = skew_index - vix_level

    # PCR proxies via signed-volume.
    up_vol = volume.where(ret > 0, 0.0)
    dn_vol = volume.where(ret < 0, 0.0)
    pcr_short_span = int(params.get("pcr_short_span", 60))
    pcr_long_span = int(params.get("pcr_long_span", 240))
    up_ema_s = _ema(up_vol, pcr_short_span)
    dn_ema_s = _ema(dn_vol, pcr_short_span)
    up_ema_l = _ema(up_vol, pcr_long_span)
    dn_ema_l = _ema(dn_vol, pcr_long_span)
    with np.errstate(invalid="ignore", divide="ignore"):
        pcr_equity = dn_ema_s / up_ema_s.replace(0.0, np.nan)
        pcr_index = dn_ema_l / up_ema_l.replace(0.0, np.nan)

    # 25d put-IV minus ATM proxy: downside / upside semi-variance.
    down = ret.clip(upper=0.0).pow(2)
    up = ret.clip(lower=0.0).pow(2)
    down_var = down.rolling(mid_w, min_periods=mid_w).mean()
    up_var = up.rolling(mid_w, min_periods=mid_w).mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        put_atm_proxy = (down_var - up_var) / (down_var + up_var).replace(0.0, np.nan)

    # VVIX proxies: vol of the RV series.
    vvix_window = int(params.get("vvix_window", 240))
    vvix_level = rv_mid.rolling(vvix_window, min_periods=vvix_window).std() * 1000.0
    with np.errstate(invalid="ignore", divide="ignore"):
        vvix_vix_ratio = vvix_level / vix_level.replace(0.0, np.nan)

    # Base-signal strengths used by the original ensemble.
    rsi2 = _rsi(close, 2)
    rsi2_strength = (50.0 - rsi2) / 50.0  # +1 = oversold, -1 = overbought

    ema_fast = _ema(close, int(params.get("trend_fast", 50)))
    ema_slow = _ema(close, int(params.get("trend_slow", 200)))
    with np.errstate(invalid="ignore", divide="ignore"):
        trend_5020_strength = (ema_fast - ema_slow) / ema_slow.replace(0.0, np.nan)

    feats = pd.DataFrame({
        "vix_level": vix_level,
        "vix9d_vix_ratio": vix9d_vix_ratio,
        "vix_vix3m_ratio": vix_vix3m_ratio,
        "skew_index": skew_index,
        "skew_minus_vix": skew_minus_vix,
        "put_call_ratio_equity_5d_ema": pcr_equity,
        "put_call_ratio_index_5d_ema": pcr_index,
        "spx_25d_put_iv_minus_atm": put_atm_proxy,
        "vvix_level": vvix_level,
        "vvix_vix_ratio": vvix_vix_ratio,
        "rsi2_signal_strength": rsi2_strength,
        "trend_5020_signal_strength": trend_5020_strength,
    }, index=df.index)

    feats = feats.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return feats


def _build_target(df: pd.DataFrame, params: dict) -> pd.Series:
    """Binary meta-label: positive expectancy on the natural hold horizon."""
    horizon = int(params.get("target_horizon_bars", 390 * 3))  # ~3 trading days
    close = df["close"].astype(float)
    fwd = close.shift(-horizon) / close - 1.0
    y = (fwd > 0).astype(int)
    return y


def signals(df: pd.DataFrame, params: dict) -> pd.Series:
    if df is None or len(df) < 1000:
        return pd.Series(0, index=df.index if df is not None else None, dtype=int)

    train_pct = float(params.get("train_pct", 0.5))
    prob_long = float(params.get("prob_long", 0.55))
    prob_short = float(params.get("prob_short", 0.45))
    exit_long = float(params.get("exit_long", 0.50))
    exit_short = float(params.get("exit_short", 0.50))

    feats = _build_feature_panel(df, params)
    y = _build_target(df, params)

    n = len(df)
    train_end = int(n * train_pct)
    if train_end < 200 or train_end >= n - 50:
        return pd.Series(0, index=df.index, dtype=int)

    # Drop the last `horizon` bars of the training window so labels are valid.
    horizon = int(params.get("target_horizon_bars", 390 * 3))
    train_cut = max(train_end - horizon, 100)

    X_train = feats.iloc[:train_cut].values
    y_train = y.iloc[:train_cut].values

    # Need both classes present.
    if len(np.unique(y_train)) < 2:
        return pd.Series(0, index=df.index, dtype=int)

    model = GradientBoostingClassifier(
        n_estimators=int(params.get("n_estimators", 200)),
        max_depth=int(params.get("max_depth", 3)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        subsample=float(params.get("subsample", 0.8)),
        random_state=int(params.get("random_state", 42)),
    )

    try:
        model.fit(X_train, y_train)
    except Exception:
        return pd.Series(0, index=df.index, dtype=int)

    X_test = feats.iloc[train_end:].values
    try:
        proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        return pd.Series(0, index=df.index, dtype=int)

    # Hysteresis pass to generate a persistent signal.
    out = np.zeros(n, dtype=int)
    state = 0
    test_idx_offset = train_end
    # Use the trend_5020 strength as a tiebreaker direction hint.
    trend_dir = feats["trend_5020_signal_strength"].iloc[train_end:].values

    for i, p in enumerate(proba):
        if state == 0:
            if p >= prob_long and trend_dir[i] >= 0:
                state = 1
            elif p <= prob_short and trend_dir[i] <= 0:
                state = -1
            else:
                state = 0
        elif state == 1:
            # Stay long until the meta-probability falls below the dead-band.
            if p < exit_long:
                state = 0
                # Allow an immediate flip if the short trigger is also live.
                if p <= prob_short and trend_dir[i] < 0:
                    state = -1
        elif state == -1:
            if p > exit_short:
                state = 0
                if p >= prob_long and trend_dir[i] > 0:
                    state = 1
        out[test_idx_offset + i] = state

    return pd.Series(out, index=df.index, dtype=int)
