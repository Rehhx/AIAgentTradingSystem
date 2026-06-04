"""
ml/labels.py
------------
The triple-barrier method (Lopez de Prado, AFML ch. 3). For each day t0 we set
three barriers and label by which is touched first:

  * upper barrier  = +pt * volatility(t0)   -> label +1 (profit-take hit)
  * lower barrier  = -sl * volatility(t0)   -> label -1 (stop-loss hit)
  * vertical (time) barrier after `horizon` -> label = sign of the terminal return

This produces PATH-DEPENDENT labels with an explicit end time t1 (the touch time),
which is exactly what PurgedKFold needs to purge overlapping windows. Volatility is
injectable so the labeling is unit-testable with deterministic series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def get_daily_vol(close: pd.Series, span: int = 20) -> pd.Series:
    """exponentially-weighted std of daily returns (the barrier scale)."""
    return close.pct_change().ewm(span=span).std()


def triple_barrier_labels(close: pd.Series, horizon: int = 10, pt: float = 1.5,
                          sl: float = 1.5, vol: pd.Series | None = None,
                          min_ret: float = 0.0) -> pd.DataFrame:
    """Return a DataFrame indexed by t0 with columns:
        t1     -- time the first barrier was touched (label-end time)
        ret    -- realized return from t0 to t1
        label  -- +1 upper / -1 lower / sign of terminal return at the vertical
    `pt`/`sl` are multiples of `vol` (defaults to get_daily_vol(close))."""
    if vol is None:
        vol = get_daily_vol(close)
    close = close.astype(float)
    idx = close.index
    arr = close.to_numpy()
    v = vol.reindex(idx).to_numpy()
    n = len(arr)
    rows = []
    for i in range(n):
        if not np.isfinite(v[i]) or v[i] <= 0:
            continue
        up, dn = pt * v[i], -sl * v[i]
        end = min(i + horizon, n - 1)
        p0 = arr[i]
        touch_j, label = end, 0
        for j in range(i + 1, end + 1):
            r = arr[j] / p0 - 1.0
            if r >= up:
                touch_j, label = j, 1
                break
            if r <= dn:
                touch_j, label = j, -1
                break
        else:
            rt = arr[end] / p0 - 1.0
            label = 1 if rt > min_ret else (-1 if rt < -min_ret else 0)
        rows.append((idx[i], idx[touch_j], arr[touch_j] / p0 - 1.0, label))
    return pd.DataFrame(rows, columns=["t0", "t1", "ret", "label"]).set_index("t0")
