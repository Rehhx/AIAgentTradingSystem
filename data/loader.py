"""
data/loader.py
--------------
shared data access layer for all agents.
reads 1m parquet files from alpaca data dump.

real schema (confirmed from GOOGL.parquet):
  columns : symbol (bytes), Interval (int32), EventAt (datetime64[ns,UTC]),
             Open, High, Low, Close (float64), Volume (int64),
             Source (int32), AggCount (int64)
  rows    : ~1.18M per ticker | GOOGL: Jan 2016 → Sep 2025 | 2,476 trading days
  index   : RangeIndex — EventAt is a plain column, NOT the index
  extras  : includes pre/after-hours | symbol stored as bytes b'GOOGL'
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger("data.loader")

DATA_DIR = Path(r"C:\Users\pcagm\Downloads\StockData")

KEEP_COLS    = ["Open", "High", "Low", "Close", "Volume"]
MARKET_OPEN  = "09:30"
MARKET_CLOSE = "16:00"
PRE_MARKET   = "04:00"
AFTER_HOURS  = "20:00"


def load_ticker(
    ticker: str,
    data_dir: Path = DATA_DIR,
    start: Optional[str] = None,
    end: Optional[str] = None,
    session: str = "regular",          # "regular" | "extended" | "all"
    resample_to: Optional[str] = None, # "5min" | "15min" | "1h" | "1d"
) -> pd.DataFrame:
    """
    load a single ticker's 1m bar data.

    returns clean dataframe:
        index   : timestamp (UTC DatetimeIndex)
        columns : open, high, low, close, volume
    """
    path = data_dir / f"{ticker}.parquet"
    if not path.exists():
        available = [p.stem for p in data_dir.glob("*.parquet")]
        raise FileNotFoundError(
            f"{ticker}.parquet not found in {data_dir}\navailable: {available}"
        )

    df = pd.read_parquet(path)

    # set EventAt as DatetimeIndex
    df["EventAt"] = pd.to_datetime(df["EventAt"], utc=True)
    df = df.set_index("EventAt")
    df.index.name = "timestamp"

    # keep only OHLCV, lowercase
    df = df[KEEP_COLS].rename(columns=str.lower).sort_index()

    # date range filter
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]

    # session filter — convert index to ET, filter by time, convert back
    if session in ("regular", "extended"):
        open_t  = MARKET_OPEN  if session == "regular" else PRE_MARKET
        close_t = MARKET_CLOSE if session == "regular" else AFTER_HOURS
        idx_et  = df.index.tz_convert("America/New_York")
        mask    = (
            (idx_et.time >= pd.Timestamp(f"2000-01-01 {open_t}").time()) &
            (idx_et.time <= pd.Timestamp(f"2000-01-01 {close_t}").time())
        )
        df = df[mask]

    df = df.dropna()

    if resample_to:
        df = resample(df, resample_to)

    log.info(
        f"{ticker} | {len(df):,} rows | "
        f"{df.index[0].date()} to {df.index[-1].date()} | "
        f"session={session}" + (f" resampled to {resample_to}" if resample_to else "")
    )
    return df


def load_multiple(tickers: list, data_dir: Path = DATA_DIR, **kwargs) -> dict:
    result = {}
    for ticker in tickers:
        try:
            result[ticker] = load_ticker(ticker, data_dir, **kwargs)
        except FileNotFoundError as e:
            log.warning(str(e))
    return result


def resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    return df.resample(freq, label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna()


def available_tickers(data_dir: Path = DATA_DIR) -> list:
    if not data_dir.exists():
        return []
    return sorted([p.stem for p in data_dir.glob("*.parquet")])


def inspect(ticker: str, data_dir: Path = DATA_DIR) -> dict:
    df     = load_ticker(ticker, data_dir, session="all")
    df_reg = load_ticker(ticker, data_dir, session="regular")
    return {
        "ticker":             ticker,
        "total_rows":         len(df),
        "regular_hours_rows": len(df_reg),
        "columns":            list(df.columns),
        "start":              str(df.index[0].date()),
        "end":                str(df.index[-1].date()),
        "trading_days":       df.index.normalize().nunique(),
        "price_range":        f"${df['close'].min():.2f} → ${df['close'].max():.2f}",
        "avg_daily_volume":   int(df_reg["volume"].sum() / df_reg.index.normalize().nunique()),
    }