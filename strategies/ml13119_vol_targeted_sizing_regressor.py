import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


def _detect_bars_per_day(idx):
    """Best-effort bars-per-trading-day from the DatetimeIndex spacing."""
    if len(idx) < 2:
        return 390
    try:
        deltas = np.diff(idx.values.astype('datetime64[s]').astype(np.int64))
        deltas = deltas[deltas > 0]
        if len(deltas) == 0:
            return 390
        median_sec = float(np.median(deltas))
        if median_sec <= 0:
            return 390
        if median_sec >= 24 * 3600 * 0.9:
            return 1
        return max(1, int(round(6.5 * 3600.0 / median_sec)))
    except Exception:
        return 390


def _build_vol_features(df, bpd):
    """Build the 10 volatility features specified in the strategy spec."""
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    open_ = df['open'].astype(float)

    with np.errstate(invalid='ignore', divide='ignore'):
        log_ret = np.log(close.replace(0, np.nan) / close.shift(1).replace(0, np.nan))
    log_ret = log_ret.replace([np.inf, -np.inf], np.nan)

    w5 = max(int(5 * bpd), 10)
    w20 = max(int(20 * bpd), 20)
    w60 = max(int(60 * bpd), 40)
    w14 = max(int(14 * bpd), 14)

    mp5 = max(2, w5 // 5)
    mp20 = max(5, w20 // 5)
    mp60 = max(10, w60 // 5)
    mp14 = max(5, w14 // 5)

    rv5 = log_ret.rolling(window=w5, min_periods=mp5).std()
    rv20 = log_ret.rolling(window=w20, min_periods=mp20).std()
    rv60 = log_ret.rolling(window=w60, min_periods=mp60).std()

    pc = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - pc).abs(),
        (low - pc).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(window=w14, min_periods=mp14).mean()
    with np.errstate(invalid='ignore', divide='ignore'):
        atr_pct = atr14 / close.replace(0, np.nan)
    atr_pct = atr_pct.replace([np.inf, -np.inf], np.nan)

    with np.errstate(invalid='ignore', divide='ignore'):
        log_hl = np.log(high.replace(0, np.nan) / low.replace(0, np.nan))
        log_co = np.log(close.replace(0, np.nan) / open_.replace(0, np.nan))
    log_hl = log_hl.replace([np.inf, -np.inf], np.nan)
    log_co = log_co.replace([np.inf, -np.inf], np.nan)

    gk_term = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
    gk20 = gk_term.rolling(window=w20, min_periods=mp20).mean()
    gk_vol = np.sqrt(gk20.clip(lower=0))

    park_term = (log_hl ** 2) / (4.0 * np.log(2.0))
    park20 = park_term.rolling(window=w20, min_periods=mp20).mean()
    park_vol = np.sqrt(park20.clip(lower=0))

    vol_of_vol = rv5.rolling(window=w20, min_periods=mp20).std()

    ann_factor = float(np.sqrt(252.0 * max(1, bpd)))
    vix_proxy = rv20 * ann_factor
    vix_premium = (rv20 - rv5) * ann_factor

    abs_ret = log_ret.abs()
    lag = max(1, int(bpd))
    abs_lag = abs_ret.shift(lag)
    abs_autocorr = abs_ret.rolling(window=w5, min_periods=mp5).corr(abs_lag)

    feats = pd.DataFrame({
        'rv5': rv5,
        'rv20': rv20,
        'rv60': rv60,
        'vix_proxy': vix_proxy,
        'vix_premium': vix_premium,
        'atr_pct': atr_pct,
        'gk_vol': gk_vol,
        'park_vol': park_vol,
        'vol_of_vol': vol_of_vol,
        'abs_autocorr': abs_autocorr,
    }, index=df.index)

    return feats, log_ret, w5


def signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Vol-targeted sizing regressor adapted as a directional gate.

    Trains a GradientBoostingRegressor to predict log(forward 5-bar-window
    realized volatility) from 10 vol features (rv5 / rv20 / rv60, ATR%,
    Garman-Klass, Parkinson, vol-of-vol, abs-return autocorr lag-1 day,
    plus an annualized-rv20 'vix proxy' and its premium vs rv5).

    SUBSTITUTION NOTE: the spec lists 'vix_level' and
    'vix_minus_realized_premium' but this engine does not load a separate
    VIX series, so we proxy them with annualized rv20 (and rv20 - rv5).
    Every other listed feature is built as specified.

    SIGNAL-CONTRACT ADAPTATION: the spec describes this model as a SIZING
    OVERLAY on existing strategies (multiply size by target_vol /
    predicted_vol, clipped to [0.3, 2.0]). The backtest engine requires
    a {-1, 0, 1} signal, so we convert the vol forecast into a directional
    gate: compute the spec's size factor s = clip(target_vol /
    predicted_vol, 0.3, 2.0) which is > 1.0 in calm regimes and < 1.0 in
    stressed regimes. Combine with a slow long-window momentum direction
    sign(close - close.shift(N_long)) (distinct from the EMA-crossover
    strategy already in the project). We go long only when momentum is
    up AND the calm gate is favorable, short only when momentum is down
    AND calm. A dead-band hysteresis (enter at prob_long / prob_short,
    exit only when probability crosses back through 0.5) keeps the signal
    persistent across noise so turnover stays low.
    """
    n = len(df)
    if n < 1000:
        return pd.Series(0, index=df.index, dtype=int)

    train_pct = float(params.get('train_pct', 0.5))
    prob_long = float(params.get('prob_long', 0.55))
    prob_short = float(params.get('prob_short', 0.45))
    vol_clip_lo = float(params.get('vol_clip_lo', 0.3))
    vol_clip_hi = float(params.get('vol_clip_hi', 2.0))
    n_estimators = int(params.get('n_estimators', 200))
    max_depth = int(params.get('max_depth', 3))
    learning_rate = float(params.get('learning_rate', 0.05))
    subsample = float(params.get('subsample', 0.8))
    trend_lookback_days = float(params.get('trend_lookback_days', 40.0))
    trend_dead_band = float(params.get('trend_dead_band', 0.001))
    max_train_rows = int(params.get('max_train_rows', 60000))

    bpd = _detect_bars_per_day(df.index)

    feats, log_ret, w5 = _build_vol_features(df, bpd)

    fwd_vol = log_ret.rolling(window=w5, min_periods=max(2, w5 // 5)).std().shift(-w5)
    with np.errstate(invalid='ignore', divide='ignore'):
        target = np.log(fwd_vol.replace(0, np.nan))
    target = target.replace([np.inf, -np.inf], np.nan)

    train_end = int(n * train_pct)
    if train_end <= 100 or train_end >= n - 100:
        return pd.Series(0, index=df.index, dtype=int)

    cutoff = max(0, train_end - w5 - 1)
    X_all = feats.values.astype(float)
    y_all = target.values.astype(float)

    train_mask = np.zeros(n, dtype=bool)
    train_mask[:cutoff] = True
    finite_X = np.isfinite(X_all).all(axis=1)
    finite_y = np.isfinite(y_all)
    train_mask &= finite_X & finite_y

    if int(train_mask.sum()) < 200:
        return pd.Series(0, index=df.index, dtype=int)

    train_idx = np.where(train_mask)[0]
    if len(train_idx) > max_train_rows:
        # uniformly subsample to bound training time while preserving order
        step = max(1, len(train_idx) // max_train_rows)
        train_idx = train_idx[::step]

    X_train = X_all[train_idx]
    y_train = y_all[train_idx]

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=42,
    )
    try:
        model.fit(X_train, y_train)
    except Exception:
        return pd.Series(0, index=df.index, dtype=int)

    X_pred = pd.DataFrame(X_all, index=df.index, columns=feats.columns)
    X_pred = X_pred.ffill().bfill().fillna(0.0)

    try:
        pred_log_vol = model.predict(X_pred.values)
    except Exception:
        return pd.Series(0, index=df.index, dtype=int)

    pred_vol = np.exp(np.clip(pred_log_vol, -25.0, 5.0))
    pred_vol = pd.Series(pred_vol, index=df.index)
    pred_vol = pred_vol.where(np.isfinite(pred_vol) & (pred_vol > 0)).ffill().bfill().fillna(1.0)

    # target_vol = in-sample median predicted vol so size factor ~ 1 on average
    in_sample = pred_vol.iloc[:train_end].values
    target_vol = float(np.nanmedian(in_sample)) if len(in_sample) else 1.0
    if not np.isfinite(target_vol) or target_vol <= 0:
        target_vol = float(np.nanmedian(pred_vol.values))
    if not np.isfinite(target_vol) or target_vol <= 0:
        target_vol = 1.0

    size_factor = (target_vol / pred_vol).clip(lower=vol_clip_lo, upper=vol_clip_hi)

    # Calm score in [0, 1] from the clipped size factor:
    #   size > 1 (low predicted vol)  -> calm score high  -> take a position
    #   size < 1 (high predicted vol) -> calm score low   -> stand aside
    denom = max(1e-9, (vol_clip_hi - vol_clip_lo))
    calm = ((size_factor - vol_clip_lo) / denom).clip(lower=0.0, upper=1.0)

    # Long-window momentum direction (price vs. price N days back), NOT a
    # short-window EMA crossover. With trend_lookback_days = 40 this is
    # distinct from any existing strategy: it is a position-vs-anchor sign,
    # gated by the ML-predicted vol regime.
    close = df['close'].astype(float)
    lookback = max(int(trend_lookback_days * bpd), 50)
    with np.errstate(invalid='ignore', divide='ignore'):
        mom = (close - close.shift(lookback)) / close.shift(lookback).replace(0, np.nan)
    mom = mom.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    trend_pos = (mom > trend_dead_band).values
    trend_neg = (mom < -trend_dead_band).values

    raw_dir = np.zeros(n, dtype=float)
    raw_dir[trend_pos] = 1.0
    raw_dir[trend_neg] = -1.0

    # Map to probability in [0, 1]
    #   p = 0.5 + 0.5 * sign(trend) * calm
    p = 0.5 + 0.5 * raw_dir * calm.values
    p = np.clip(p, 0.0, 1.0)

    # Hysteresis state machine to suppress flips:
    #   from flat:  long if p >= prob_long, short if p <= prob_short
    #   from long:  flat if p < 0.5 (dead band, not at prob_long)
    #   from short: flat if p > 0.5
    out = np.zeros(n, dtype=int)
    state = 0
    for i in range(train_end, n):
        pi = p[i]
        if state == 1:
            if pi < 0.5:
                state = 0
        elif state == -1:
            if pi > 0.5:
                state = 0
        if state == 0:
            if pi >= prob_long:
                state = 1
            elif pi <= prob_short:
                state = -1
        out[i] = state

    return pd.Series(out, index=df.index, dtype=int)
