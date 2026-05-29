"""
sector_dispersion_revert
========================

Weekly cross-sectional mean reversion across the 11 SPDR sector ETFs
(XLE/XLF/XLK/XLV/XLY/XLP/XLI/XLB/XLU/XLRE/XLC).

Entry/exit rules
----------------
Cross-sectional mode (when ``data.loader`` exposes the sector panel):
  * Each rebalance day (default Monday), rank the 11 sector ETFs by their
    trailing ``rank_lookback_days`` simple return.
  * If the asset described by ``df`` is among the ``long_n`` worst
    performers, emit +1 (long the loser, expecting reversion up).
  * If it is among the ``short_n`` best performers, emit -1 (short the
    winner, expecting reversion down).
  * Otherwise emit 0.
  * The position is held for ``hold_days`` trading sessions, then flat
    (unless replaced by a newer rebalance).

Single-asset fallback (used when the cross-sectional panel cannot be
constructed):
  * On the same weekly cadence, compute the z-score of the asset's own
    trailing N-day return against a ``baseline_lookback_days`` rolling
    window. If z is in the lower tail (<= -z_entry) go long; if in the
    upper tail (>= +z_entry) go short; else flat. Hold ``hold_days``.

The weekly rebalance + multi-day hold structure caps trade count to
roughly one round-trip per week per direction, so 6 bps round-trip
slippage is comfortably absorbed.
"""

import numpy as np
import pandas as pd


_SECTOR_TICKERS = (
    "XLE", "XLF", "XLK", "XLV", "XLY", "XLP",
    "XLI", "XLB", "XLU", "XLRE", "XLC",
)

_WEEKDAY = {
    "monday": 0, "tuesday": 1, "wednesday": 2,
    "thursday": 3, "friday": 4,
}


def _try_load_sector_panel(index):
    """Best-effort attempt to build a daily close panel for sector ETFs
    via the project's data.loader. Returns None on any failure."""
    try:
        from data import loader  # type: ignore
    except Exception:
        return None

    try:
        start = pd.Timestamp(index.min()).normalize()
        end = pd.Timestamp(index.max()).normalize()
    except Exception:
        return None

    candidates = ("load", "load_symbol", "load_daily", "get", "fetch")
    loader_fn = None
    for name in candidates:
        if hasattr(loader, name):
            loader_fn = getattr(loader, name)
            break
    if loader_fn is None:
        return None

    closes = {}
    for tkr in _SECTOR_TICKERS:
        try:
            sub = None
            for kwargs in (
                {"start": start, "end": end},
                {"start_date": start, "end_date": end},
                {},
            ):
                try:
                    sub = loader_fn(tkr, **kwargs)
                    break
                except TypeError:
                    continue
                except Exception:
                    sub = None
                    break
            if sub is None or len(sub) == 0:
                continue
            if isinstance(sub, pd.Series):
                series = sub
            else:
                col = None
                for c in ("close", "Close", "adj_close", "Adj Close"):
                    if c in sub.columns:
                        col = c
                        break
                if col is None:
                    continue
                series = sub[col]
            series = series[~series.index.duplicated(keep="last")]
            if not isinstance(series.index, pd.DatetimeIndex):
                series.index = pd.to_datetime(series.index)
            series = series.groupby(series.index.normalize()).last()
            closes[tkr] = series.astype(float)
        except Exception:
            continue

    if len(closes) < 6:
        return None

    panel = pd.DataFrame(closes).sort_index()
    panel = panel.ffill(limit=3)
    return panel


def _identify_ticker(df, panel):
    """Match df['close'] against the sector panel by daily return
    correlation. Returns the matching ticker symbol or None."""
    try:
        own = df["close"].astype(float)
        own_daily = own.groupby(own.index.normalize()).last()
        own_ret = own_daily.pct_change().dropna()
    except Exception:
        return None

    best = None
    best_corr = -1.0
    for tkr in panel.columns:
        other = panel[tkr].pct_change().dropna()
        common = own_ret.index.intersection(other.index)
        if len(common) < 50:
            continue
        a = own_ret.reindex(common).values
        b = other.reindex(common).values
        if np.std(a) <= 0 or np.std(b) <= 0:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        if not np.isfinite(c):
            continue
        if c > best_corr:
            best_corr = c
            best = tkr

    if best is not None and best_corr > 0.999:
        return best
    return None


def signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """Cross-sectional sector mean reversion. See module docstring."""
    rank_lookback = int(params.get("rank_lookback_days", 5))
    hold_days = int(params.get("hold_days", 5))
    long_n = int(params.get("long_n", 3))
    short_n = int(params.get("short_n", 3))
    rebal_day = str(params.get("rebal_day", "monday")).lower()
    baseline_lookback = int(params.get("baseline_lookback_days", 60))
    z_entry = float(params.get("zscore_entry", 1.0))

    rank_lookback = max(2, rank_lookback)
    hold_days = max(1, hold_days)
    long_n = max(0, long_n)
    short_n = max(0, short_n)
    baseline_lookback = max(20, baseline_lookback)
    target_wd = _WEEKDAY.get(rebal_day, 0)

    idx = df.index
    out = pd.Series(0, index=idx, dtype="int64")
    if len(df) < rank_lookback + hold_days + 5:
        return out
    if not isinstance(idx, pd.DatetimeIndex):
        return out

    bar_dates = pd.Series(idx.normalize(), index=idx)
    first_bar_of_day = ~bar_dates.duplicated(keep="first")
    weekdays = pd.Series(idx.weekday, index=idx)

    iso = idx.isocalendar()
    week_key = pd.Series(
        iso["year"].astype(str).values + "-" + iso["week"].astype(str).values,
        index=idx,
    )

    on_target_wd = (weekdays == target_wd) & first_bar_of_day

    rebal_mask = pd.Series(False, index=idx)
    groups = pd.DataFrame({
        "wk": week_key.values,
        "first_day": first_bar_of_day.values,
        "on_target": on_target_wd.values,
    }, index=idx)
    for _wk, sub in groups.groupby("wk", sort=False):
        chosen = sub.index[sub["on_target"].values]
        if len(chosen) == 0:
            chosen = sub.index[sub["first_day"].values]
            if len(chosen) == 0:
                continue
        rebal_mask.loc[chosen[0]] = True

    rebal_bars = idx[rebal_mask.values]
    if len(rebal_bars) == 0:
        return out

    own_daily = df["close"].groupby(idx.normalize()).last().astype(float)

    panel = _try_load_sector_panel(idx)
    own_ticker = _identify_ticker(df, panel) if panel is not None else None

    direction_by_rebal = {}

    if panel is not None and own_ticker is not None:
        panel_ret_n = panel.pct_change(rank_lookback)
        for b in rebal_bars:
            d = pd.Timestamp(b).normalize()
            avail = panel_ret_n.index[panel_ret_n.index <= d]
            if len(avail) == 0:
                continue
            row = panel_ret_n.loc[avail[-1]].dropna()
            if (own_ticker not in row.index
                    or len(row) < (long_n + short_n + 1)):
                continue
            ranked = row.sort_values()
            losers = set(ranked.index[:long_n]) if long_n > 0 else set()
            winners = (set(ranked.index[-short_n:])
                       if short_n > 0 else set())
            if own_ticker in losers:
                direction_by_rebal[b] = 1
            elif own_ticker in winners:
                direction_by_rebal[b] = -1
            else:
                direction_by_rebal[b] = 0
    else:
        ret_n = own_daily.pct_change(rank_lookback)
        mp = max(20, baseline_lookback // 3)
        mp = min(mp, baseline_lookback)
        mu = ret_n.rolling(baseline_lookback, min_periods=mp).mean()
        sd = ret_n.rolling(baseline_lookback, min_periods=mp).std()
        z = (ret_n - mu) / sd.replace(0.0, np.nan)
        z = z.replace([np.inf, -np.inf], np.nan)
        for b in rebal_bars:
            d = pd.Timestamp(b).normalize()
            avail = z.index[z.index <= d]
            if len(avail) == 0:
                continue
            val = z.loc[avail[-1]]
            if not np.isfinite(val):
                continue
            if val <= -z_entry:
                direction_by_rebal[b] = 1
            elif val >= z_entry:
                direction_by_rebal[b] = -1
            else:
                direction_by_rebal[b] = 0

    if not direction_by_rebal:
        return out

    unique_days = pd.DatetimeIndex(sorted(set(idx.normalize())))
    day_pos = {d: i for i, d in enumerate(unique_days)}

    pos_by_day = pd.Series(0, index=unique_days, dtype="int64")
    for b in sorted(direction_by_rebal.keys()):
        sign = direction_by_rebal[b]
        sd0 = pd.Timestamp(b).normalize()
        if sd0 not in day_pos:
            later = unique_days[unique_days >= sd0]
            if len(later) == 0:
                continue
            sd0 = later[0]
        i0 = day_pos[sd0]
        i1 = min(i0 + hold_days - 1, len(unique_days) - 1)
        if sign == 0:
            pos_by_day.iloc[i0:i1 + 1] = 0
        else:
            pos_by_day.iloc[i0:i1 + 1] = sign

    bar_day = idx.normalize()
    aligned = pos_by_day.reindex(bar_day).fillna(0).astype("int64")
    aligned.index = idx
    return aligned
