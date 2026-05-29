"""
data/yfinance_loader.py
-----------------------
free supplementary data source via Yahoo Finance. matches the schema of
data/loader.py so it can be used interchangeably with the local parquet files.

LIMITATIONS (yahoo finance is free but rate-limited):
  - 1-minute bars: only the last ~7 days
  - 5-minute bars: only the last ~60 days
  - 15-min / 30-min / 1-hour bars: ~730 days
  - daily bars: many years

so this is useful for:
  - extending the universe to tickers we don't have parquet for
  - filling recent gaps (e.g. paper-trading a new ticker before backtesting)
  - prototyping on indices not in our parquet set (^VIX, ^DJI, ^GSPC)

install:
    pip install yfinance

NO ACCOUNT REQUIRED. Yahoo's API is unofficial — they can rate-limit or
break at any time. For production, consider Polygon (paid) or Alpaca
historical-data API (free up to 5 yrs but only equities they cover).
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger("data.yfinance_loader")

# yahoo's accepted interval strings → max_lookback (days)
INTERVAL_LIMITS = {
    "1m":  7,
    "2m":  60,
    "5m":  60,
    "15m": 60,
    "30m": 60,
    "1h":  730,
    "1d":  10000,
}


def load_yf(
    ticker: str,
    interval: str  = "1m",
    period:   Optional[str] = None,    # e.g. "5d", "60d" — passed to yf
    start:    Optional[str] = None,    # e.g. "2025-01-01"
    end:      Optional[str] = None,
    session:  str = "regular",
) -> pd.DataFrame:
    """
    pull OHLCV bars from yahoo finance. returns the same shape as
    data.loader.load_ticker:

        index   : UTC DatetimeIndex
        columns : open, high, low, close, volume

    if both period and start are given, start wins (period is ignored).

    NOTE: yahoo's 1m endpoint silently returns empty if start is older than
    the 7-day window. callers should fall back to a coarser interval.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError("yfinance not installed. run: pip install yfinance") from e

    if interval not in INTERVAL_LIMITS:
        raise ValueError(f"bad interval {interval}. allowed: {list(INTERVAL_LIMITS)}")

    if start is None and period is None:
        # default to the maximum lookback for the chosen interval
        days = INTERVAL_LIMITS[interval]
        period = f"{min(days, 60)}d" if interval not in ("1d",) else "5y"

    yf_ticker = yf.Ticker(ticker)
    kwargs = {"interval": interval, "auto_adjust": False, "prepost": session != "regular"}
    if start:
        kwargs["start"] = start
        if end:
            kwargs["end"] = end
    elif period:
        kwargs["period"] = period

    df = yf_ticker.history(**kwargs)
    if df.empty:
        raise FileNotFoundError(
            f"yfinance returned 0 rows for {ticker} interval={interval} "
            f"start={start} period={period}. likely the start date is outside "
            f"the {INTERVAL_LIMITS[interval]}-day lookback window for this interval."
        )

    # normalize to data.loader schema: lowercase columns, UTC index, OHLCV only
    df.index = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
    df.index.name = "timestamp"
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    df = df[["open", "high", "low", "close", "volume"]].dropna()

    if session == "regular":
        # filter to 9:30-16:00 ET
        idx_et = df.index.tz_convert("America/New_York")
        mask   = ((idx_et.time >= pd.Timestamp("2000-01-01 09:30").time()) &
                  (idx_et.time <= pd.Timestamp("2000-01-01 16:00").time()))
        df = df[mask]

    log.info(f"yfinance {ticker} | {len(df):,} rows | {df.index[0].date()} -> {df.index[-1].date()}")
    return df


def save_to_parquet(df: pd.DataFrame, ticker: str, out_dir: Path) -> Path:
    """
    write a yfinance dataframe back out in the same parquet layout the local
    loader expects, so a freshly downloaded ticker can be backtested
    immediately by the existing engine.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}.parquet"

    write_df = df.copy()
    write_df.index.name = None
    write_df = write_df.reset_index()
    write_df = write_df.rename(columns={
        "timestamp": "EventAt", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "volume": "Volume",
    })
    # add the columns data/loader.py drops anyway but expects to exist
    write_df["symbol"]   = ticker.encode("utf-8")
    write_df["Interval"] = 1
    write_df["Source"]   = 0
    write_df["AggCount"] = 1

    write_df.to_parquet(out_path)
    log.info(f"wrote {len(write_df)} rows to {out_path}")
    return out_path


if __name__ == "__main__":
    # quick smoke test: pull last 5 days of 1m SPY
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    df = load_yf("SPY", interval="5m", period="30d")
    print(f"\nSPY 5m last 30d:")
    print(f"  shape   : {df.shape}")
    print(f"  range   : {df.index[0]} -> {df.index[-1]}")
    print(f"  sample  :")
    print(df.tail(3))
