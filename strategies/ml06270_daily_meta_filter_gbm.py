import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import GradientBoostingClassifier
    _HAS_SK = True
except Exception:
    _HAS_SK = False


def _rsi(close, period):
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    a = 1.0 / max(int(period), 1)
    ru = up.ewm(alpha=a, adjust=False).mean()
    rd = dn.ewm(alpha=a, adjust=False).mean()
    with np.errstate(invalid='ignore', divide='ignore'):
        rs = ru / rd.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def _atr(high, low, close, period):
    pc = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - pc).abs(),
                    (low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _adx(high, low, close, period):
    up = high.diff()
    dn = -low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=high.index)
    atr = _atr(high, low, close, period)
    a = 1.0 / max(int(period), 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        plus_di = 100.0 * plus_dm.ewm(alpha=a, adjust=False).mean() / atr.replace(0.0, np.nan)
        minus_di = 100.0 * minus_dm.ewm(alpha=a, adjust=False).mean() / atr.replace(0.0, np.nan)
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=a, adjust=False).mean().fillna(0.0)


def _build_meta_features(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    idx = df.index

    feat = pd.DataFrame(index=idx)

    # --- base strategy signal-strength proxies ----------------------------
    rsi2 = _rsi(close, 2)
    feat['rsi2'] = rsi2
    feat['rsi2_signal_strength'] = (50.0 - rsi2) / 50.0  # higher = more oversold

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    with np.errstate(invalid='ignore', divide='ignore'):
        feat['trend_5020_signal_strength'] = (ema20 - ema50) / ema50.replace(0.0, np.nan)
    feat['trend_above'] = (ema20 > ema50).astype(float)

    dc_n = 20
    dc_hi = high.rolling(dc_n, min_periods=dc_n).max()
    dc_lo = low.rolling(dc_n, min_periods=dc_n).min()
    dc_range = (dc_hi - dc_lo)
    with np.errstate(invalid='ignore', divide='ignore'):
        feat['donchian_signal_strength'] = (close - dc_lo) / dc_range.replace(0.0, np.nan)
    feat['donchian_break'] = (close >= dc_hi.shift(1)).astype(float)

    # --- realized-vol regime (VIX / VIX-term-structure proxy) -------------
    ret = close.pct_change()
    rv5 = ret.rolling(5, min_periods=5).std()
    rv20 = ret.rolling(20, min_periods=20).std()
    rv60 = ret.rolling(60, min_periods=60).std()
    feat['rv_5'] = rv5
    feat['rv_20'] = rv20
    feat['rv_60'] = rv60
    with np.errstate(invalid='ignore', divide='ignore'):
        feat['rv_5_over_20'] = rv5 / rv20.replace(0.0, np.nan)
        feat['rv_20_over_60'] = rv20 / rv60.replace(0.0, np.nan)
    feat['rv20_change_5'] = rv20 - rv20.shift(5)

    # --- % above 200-bar MA (long-term regime) ----------------------------
    ma200 = close.rolling(200, min_periods=200).mean()
    with np.errstate(invalid='ignore', divide='ignore'):
        feat['pct_above_200ma'] = (close - ma200) / ma200.replace(0.0, np.nan)

    # --- ATR pct regime bucket --------------------------------------------
    atr14 = _atr(high, low, close, 14)
    with np.errstate(invalid='ignore', divide='ignore'):
        atr_pct = atr14 / close.replace(0.0, np.nan)
    feat['atr_pct'] = atr_pct
    feat['atr_pct_rank_60'] = atr_pct.rolling(60, min_periods=60).rank(pct=True)

    # --- ADX 14 (trend quality) -------------------------------------------
    feat['adx_14'] = _adx(high, low, close, 14)

    # --- skew proxy (substitute for SKEW index) ---------------------------
    feat['skew_60'] = ret.rolling(60, min_periods=60).skew()
    feat['kurt_60'] = ret.rolling(60, min_periods=60).kurt()

    # --- cross-asset proxies (TLT/SPY, HYG/IEF unavailable): use own
    #     short vs long momentum spread + volume regime as substitutes ----
    mom5 = close.pct_change(5)
    mom20 = close.pct_change(20)
    mom60 = close.pct_change(60)
    feat['mom_5'] = mom5
    feat['mom_20'] = mom20
    feat['mom_60'] = mom60
    feat['mom_5_minus_20'] = mom5 - mom20
    feat['mom_20_minus_60'] = mom20 - mom60

    vol_ma20 = volume.rolling(20, min_periods=20).mean()
    with np.errstate(invalid='ignore', divide='ignore'):
        feat['vol_ratio_20'] = volume / vol_ma20.replace(0.0, np.nan)

    # --- day-of-week one-hot ---------------------------------------------
    dow = idx.dayofweek
    for d in range(5):
        feat[f'dow_{d}'] = (dow == d).astype(float)

    # --- bar-range regime -------------------------------------------------
    with np.errstate(invalid='ignore', divide='ignore'):
        bar_rng = (high - low) / close.replace(0.0, np.nan)
    feat['range_pct'] = bar_rng
    rng_mu = bar_rng.rolling(60, min_periods=60).mean()
    rng_sd = bar_rng.rolling(60, min_periods=60).std()
    with np.errstate(invalid='ignore', divide='ignore'):
        feat['range_z_60'] = (bar_rng - rng_mu) / rng_sd.replace(0.0, np.nan)

    return feat


def signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    daily_meta_filter_gbm
    ---------------------
    Lopez-de-Prado-style meta-label classifier on (daily-ish) bars. Predicts the
    probability that holding long for `hold_bars` bars beats the round-trip
    cost; long when p >= prob_long, short when p <= prob_short, with a 0.50
    dead-band exit (hysteresis) to keep positions persistent and avoid the
    overtrading death-spiral that has buried prior ML attempts here.

    Spec-listed features that depend on cross-asset / vol-index data (VIX,
    VIX9D/VIX3M term structure, SKEW, TLT/SPY spread, HYG/IEF ratio) are not
    available from a single OHLCV frame, so they are substituted by self-
    derived regime proxies: a realized-vol cone (rv5 / rv20 / rv60 and
    ratios), ATR-pct percentile rank, return skew & kurt, and multi-horizon
    momentum spreads. This substitution is intentional and documented here.

    Spec asked for "gradient_boosting_with_meta_label"; we use sklearn
    GradientBoostingClassifier — no torch / transformer / TCN substitution
    needed.
    """
    n = len(df)
    if n < 1000 or not _HAS_SK:
        return pd.Series(0, index=df.index, dtype=int)

    train_pct = float(params.get('train_pct', 0.5))
    prob_long = float(params.get('prob_long', 0.55))
    prob_short = float(params.get('prob_short', 0.45))
    exit_band = float(params.get('exit_band', 0.50))
    hold_bars = int(params.get('hold_bars', 5))
    cost_bps = float(params.get('cost_bps', 6.0))
    n_estimators = int(params.get('n_estimators', 150))
    max_depth = int(params.get('max_depth', 3))
    learning_rate = float(params.get('learning_rate', 0.05))
    subsample = float(params.get('subsample', 0.85))
    min_hold = int(params.get('min_hold_bars', 2))

    feat = _build_meta_features(df)

    close = df['close']
    fwd_ret = close.shift(-hold_bars) / close - 1.0
    cost_thresh = cost_bps / 10000.0
    y = (fwd_ret > cost_thresh).astype('float')
    y[fwd_ret.isna()] = np.nan  # forward unknown at tail

    train_end = int(n * train_pct)
    if train_end < 250 or (n - train_end) < 50:
        return pd.Series(0, index=df.index, dtype=int)

    # Sanitize feature matrix
    Xdf = feat.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    X = Xdf.values.astype(np.float64)
    X = np.where(np.isfinite(X), X, 0.0)

    y_vals = y.values
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:train_end] = True
    valid_train = train_mask & np.isfinite(y_vals)

    if valid_train.sum() < 200:
        return pd.Series(0, index=df.index, dtype=int)

    Xtr = X[valid_train]
    ytr = y_vals[valid_train].astype(int)

    if len(np.unique(ytr)) < 2:
        return pd.Series(0, index=df.index, dtype=int)

    clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=42,
    )
    clf.fit(Xtr, ytr)

    proba = np.full(n, 0.5, dtype=np.float64)
    proba[train_end:] = clf.predict_proba(X[train_end:])[:, 1]

    # ---- hysteretic state machine ---------------------------------------
    # Enter long when p >= prob_long, hold until p < exit_band.
    # Enter short when p <= prob_short, hold until p > 1 - exit_band.
    out = np.zeros(n, dtype=np.int64)
    state = 0
    bars_in_state = 0
    for i in range(train_end, n):
        p = proba[i]
        bars_in_state += 1
        if state == 0:
            if p >= prob_long:
                state = 1
                bars_in_state = 0
            elif p <= prob_short:
                state = -1
                bars_in_state = 0
        elif state == 1:
            if bars_in_state >= min_hold and p < exit_band:
                if p <= prob_short:
                    state = -1
                else:
                    state = 0
                bars_in_state = 0
        else:  # state == -1
            if bars_in_state >= min_hold and p > (1.0 - exit_band):
                if p >= prob_long:
                    state = 1
                else:
                    state = 0
                bars_in_state = 0
        out[i] = state

    return pd.Series(out, index=df.index, dtype=int)
