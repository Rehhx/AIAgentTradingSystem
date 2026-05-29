"""
data/sp500.py
-------------
S&P 500 universe + free daily-bar loader (yfinance), with on-disk caching.

The RSI-2 book is a DAILY strategy, so we don't need the 1-minute parquet — we
can pull split/dividend-adjusted daily bars for every S&P 500 name straight from
yfinance and trade the whole index.

CAVEAT (survivorship bias): sp500_tickers() returns TODAY's constituents.
Backtesting today's members over history overstates returns because names that
were dropped from the index (losers, bankruptcies) are excluded. Treat the
historical numbers as optimistic; the live/forward signal is unaffected.

Cache layout:
  data/cache/sp500_tickers.json          constituent list
  data/cache/sp500_daily/<TICKER>.parquet  per-ticker adjusted daily OHLCV
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("data.sp500")

CACHE_DIR    = Path("data/cache")
DAILY_DIR    = CACHE_DIR / "sp500_daily"
TICKER_CACHE = CACHE_DIR / "sp500_tickers.json"


def sp500_tickers(refresh: bool = False) -> list[str]:
    """current S&P 500 symbols (yfinance format: BRK.B -> BRK-B). Cached.

    Primary source is a plain CSV (no lxml dependency); falls back to Wikipedia
    via pandas.read_html if the CSV is unreachable."""
    if TICKER_CACHE.exists() and not refresh:
        return json.loads(TICKER_CACHE.read_text())

    syms = None
    csv_url = ("https://raw.githubusercontent.com/datasets/"
               "s-and-p-500-companies/main/data/constituents.csv")
    try:
        df = pd.read_csv(csv_url)
        col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        syms = [str(s).replace(".", "-").strip() for s in df[col].tolist()]
    except Exception as e:
        log.warning(f"CSV constituent source failed ({e}); trying Wikipedia")
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        syms = [str(s).replace(".", "-").strip() for s in tables[0]["Symbol"].tolist()]

    syms = [s for s in syms if s and s.lower() != "nan"]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TICKER_CACHE.write_text(json.dumps(syms))
    log.info(f"fetched {len(syms)} S&P 500 tickers")
    return syms


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                            "Close": "close", "Volume": "volume"})
    idx = df.index
    df.index = idx.tz_convert("UTC") if getattr(idx, "tz", None) else idx.tz_localize("UTC")
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    return df[cols].dropna()


def load_daily(tickers: list[str], start: str = "2016-01-01", end: str | None = None,
               refresh: bool = False, batch: int = 80) -> dict[str, pd.DataFrame]:
    """return {ticker: daily OHLCV DataFrame} (adjusted, UTC index). Cached per
    ticker; only missing/uncached names are downloaded, in batches."""
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    out, need = {}, []
    for t in tickers:
        fp = DAILY_DIR / f"{t}.parquet"
        if fp.exists() and not refresh:
            try:
                out[t] = pd.read_parquet(fp)
                continue
            except Exception:
                pass
        need.append(t)

    if need:
        import yfinance as yf
        log.info(f"downloading {len(need)} tickers from yfinance ...")
        for i in range(0, len(need), batch):
            chunk = need[i:i + batch]
            try:
                data = yf.download(chunk, start=start, end=end, auto_adjust=True,
                                   group_by="ticker", progress=False, threads=True)
            except Exception as e:
                log.warning(f"batch download failed ({e}); retrying one-by-one")
                data = None
            for t in chunk:
                try:
                    if data is not None and isinstance(data.columns, pd.MultiIndex):
                        sub = data[t].copy()
                    elif data is not None and len(chunk) == 1:
                        sub = data.copy()
                    else:
                        sub = yf.download(t, start=start, end=end, auto_adjust=True,
                                          progress=False)
                    sub = _normalize(sub)
                    if len(sub) < 220:
                        continue
                    sub.to_parquet(DAILY_DIR / f"{t}.parquet")
                    out[t] = sub
                except Exception as e:
                    log.warning(f"  {t}: skipped ({e})")
    log.info(f"loaded {len(out)}/{len(tickers)} tickers")
    return out
