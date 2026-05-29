"""
data/multi_source.py
--------------------
adapters for free supplementary market data sources.

what's in here:
  - Finnhub:  real-time quotes, company news, news sentiment, earnings calendar
              free tier 60 req/min, signup at finnhub.io
  - FRED:     macroeconomic series (VIX, treasury yields, USD index, oil...)
              free, generous limits, signup at fred.stlouisfed.org/docs/api

each function returns a normalized dict / DataFrame matching the rest of the
project's conventions. quiet failures (return empty dict or None) when keys
are missing — never raise on missing env var so the orchestrator stays alive.

usage examples at bottom under `if __name__ == "__main__":`.
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FINNHUB_API_KEY, FRED_API_KEY

log = logging.getLogger("data.multi_source")

FINNHUB_BASE = "https://finnhub.io/api/v1"
FRED_BASE    = "https://api.stlouisfed.org/fred/series/observations"


# ---------------------------------------------------------------------------
# finnhub adapters
# ---------------------------------------------------------------------------

def finnhub_quote(symbol: str) -> dict:
    """
    latest quote: current price, prior close, day high/low, day open, timestamp.
    returns {} if no API key or request fails. always normalized to {price,
    change, change_pct, day_high, day_low, day_open, prev_close, ts}.
    """
    if not FINNHUB_API_KEY:
        return {}
    try:
        import requests
        r = requests.get(f"{FINNHUB_BASE}/quote",
                         params={"symbol": symbol, "token": FINNHUB_API_KEY},
                         timeout=10)
        r.raise_for_status()
        d = r.json()
    except Exception as e:
        log.warning(f"finnhub quote failed for {symbol}: {e}")
        return {}

    # finnhub keys: c (current), pc (prev close), d (change), dp (change pct),
    # h (high), l (low), o (open), t (timestamp unix)
    return {
        "symbol":     symbol,
        "price":      d.get("c"),
        "prev_close": d.get("pc"),
        "change":     d.get("d"),
        "change_pct": d.get("dp"),
        "day_high":   d.get("h"),
        "day_low":    d.get("l"),
        "day_open":   d.get("o"),
        "ts":         datetime.fromtimestamp(d.get("t", 0), tz=timezone.utc).isoformat()
                      if d.get("t") else None,
        "source":     "finnhub",
    }


def finnhub_news(symbol: str, lookback_days: int = 3) -> list:
    """
    company news from the last N days. returns a list of dicts with
    {headline, summary, source, url, datetime, image}.
    """
    if not FINNHUB_API_KEY:
        return []
    try:
        import requests
        today = datetime.now(timezone.utc).date()
        start = today - timedelta(days=lookback_days)
        r = requests.get(f"{FINNHUB_BASE}/company-news",
                         params={"symbol": symbol,
                                 "from": start.isoformat(),
                                 "to":   today.isoformat(),
                                 "token": FINNHUB_API_KEY},
                         timeout=10)
        r.raise_for_status()
        items = r.json() or []
    except Exception as e:
        log.warning(f"finnhub news failed for {symbol}: {e}")
        return []

    out = []
    for item in items[:20]:   # cap to 20 headlines per symbol
        out.append({
            "headline":  item.get("headline"),
            "summary":   item.get("summary"),
            "source":    item.get("source"),
            "url":       item.get("url"),
            "datetime":  datetime.fromtimestamp(item.get("datetime", 0),
                                                tz=timezone.utc).isoformat()
                         if item.get("datetime") else None,
        })
    return out


def finnhub_sentiment(symbol: str) -> dict:
    """
    news sentiment + buzz score for a symbol.
    returns {sentiment, articles_in_last_week, buzz, sector_avg_buzz}.
    """
    if not FINNHUB_API_KEY:
        return {}
    try:
        import requests
        r = requests.get(f"{FINNHUB_BASE}/news-sentiment",
                         params={"symbol": symbol, "token": FINNHUB_API_KEY},
                         timeout=10)
        r.raise_for_status()
        d = r.json()
    except Exception as e:
        log.warning(f"finnhub sentiment failed for {symbol}: {e}")
        return {}

    sent = d.get("sentiment", {}) or {}
    buzz = d.get("buzz", {}) or {}
    return {
        "symbol":                symbol,
        "bullish_pct":           sent.get("bullishPercent"),
        "bearish_pct":           sent.get("bearishPercent"),
        "articles_in_last_week": buzz.get("articlesInLastWeek"),
        "buzz":                  buzz.get("buzz"),
        "weekly_avg":            buzz.get("weeklyAverage"),
        "company_news_score":    d.get("companyNewsScore"),
        "sector_avg_news_score": d.get("sectorAverageNewsScore"),
        "source":                "finnhub",
    }


def finnhub_earnings_calendar(lookahead_days: int = 7) -> list:
    """upcoming earnings releases in the next N days (returns list of dicts)."""
    if not FINNHUB_API_KEY:
        return []
    try:
        import requests
        today = datetime.now(timezone.utc).date()
        end   = today + timedelta(days=lookahead_days)
        r = requests.get(f"{FINNHUB_BASE}/calendar/earnings",
                         params={"from": today.isoformat(),
                                 "to":   end.isoformat(),
                                 "token": FINNHUB_API_KEY},
                         timeout=10)
        r.raise_for_status()
        data = r.json() or {}
    except Exception as e:
        log.warning(f"finnhub earnings calendar failed: {e}")
        return []

    items = data.get("earningsCalendar") or []
    return [{
        "symbol":   e.get("symbol"),
        "date":     e.get("date"),
        "hour":     e.get("hour"),
        "eps_est":  e.get("epsEstimate"),
        "rev_est":  e.get("revenueEstimate"),
        "quarter":  e.get("quarter"),
        "year":     e.get("year"),
    } for e in items]


# ---------------------------------------------------------------------------
# fred adapters
# ---------------------------------------------------------------------------

# series of interest — each is one HTTP call to FRED.
# add more from https://fred.stlouisfed.org/categories
FRED_SERIES = {
    "vix":              "VIXCLS",      # CBOE volatility index, close
    "treasury_10y":     "DGS10",       # 10-year treasury yield
    "treasury_2y":      "DGS2",        # 2-year treasury yield
    "yield_curve_10_2": "T10Y2Y",      # 10y minus 2y (recession indicator)
    "treasury_bill_3m": "DTB3",        # 3-month T-bill yield
    "fed_funds_rate":   "DFF",         # effective federal funds rate
    "dollar_index":     "DTWEXBGS",    # trade-weighted dollar
    "breakeven_10y":    "T10YIE",      # 10-year breakeven inflation
    "wti_oil":          "DCOILWTICO",  # WTI crude oil price
    "gold_vol":         "GVZCLS",      # CBOE gold volatility index
}


def fred_latest(series_id: str) -> Optional[dict]:
    """
    latest observation for a FRED series.
    returns {series_id, date, value, units?} or None on failure.
    """
    if not FRED_API_KEY:
        return None
    try:
        import requests
        r = requests.get(FRED_BASE, params={
            "series_id":   series_id,
            "api_key":     FRED_API_KEY,
            "file_type":   "json",
            "sort_order":  "desc",
            "limit":       1,
        }, timeout=10)
        r.raise_for_status()
        data = r.json() or {}
    except Exception as e:
        log.warning(f"fred {series_id} failed: {e}")
        return None

    obs = (data.get("observations") or [])
    if not obs:
        return None
    o = obs[0]
    val = o.get("value")
    # fred uses "." for missing data points
    val_f = float(val) if val not in (None, ".", "") else None
    return {
        "series_id": series_id,
        "date":      o.get("date"),
        "value":     val_f,
    }


def fred_history(series_id: str, start: str = None, end: str = None) -> pd.DataFrame:
    """
    historical observations of a FRED series as a DataFrame indexed by date.
    use this when you want a feature column (e.g. VIX values aligned to your
    daily bars) instead of just a snapshot.
    """
    if not FRED_API_KEY:
        return pd.DataFrame()
    try:
        import requests
        params = {
            "series_id": series_id,
            "api_key":   FRED_API_KEY,
            "file_type": "json",
        }
        if start: params["observation_start"] = start
        if end:   params["observation_end"]   = end
        r = requests.get(FRED_BASE, params=params, timeout=15)
        r.raise_for_status()
        data = r.json() or {}
    except Exception as e:
        log.warning(f"fred history {series_id} failed: {e}")
        return pd.DataFrame()

    obs = data.get("observations") or []
    if not obs:
        return pd.DataFrame()
    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"]  = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().set_index("date").sort_index()
    df.columns = [series_id]
    return df


def fred_snapshot() -> dict:
    """
    latest values for every series in FRED_SERIES.
    returns a single dict mapping nickname -> value (or None).
    """
    out = {}
    for nickname, sid in FRED_SERIES.items():
        latest = fred_latest(sid)
        out[nickname] = {
            "value": latest["value"] if latest else None,
            "date":  latest["date"]  if latest else None,
            "series_id": sid,
        }
    return out


# ---------------------------------------------------------------------------
# local CSV fallback for FRED (use when no API key, or for offline access)
# ---------------------------------------------------------------------------
# format expected:
#   observation_date,SERIES_ID
#   2016-05-26,13.43
#   2016-05-30,
#   2016-05-31,14.19
# blank values are skipped (FRED leaves these on bank holidays etc).

LOCAL_FRED_CSVS = {
    # nickname -> full path. extend as you download more FRED series.
    "vix": Path(r"C:\Users\pcagm\Downloads\VIXCLS (1).csv"),
}


def load_local_fred_csv(path: Path, series_label: str = None) -> pd.DataFrame:
    """
    load a FRED CSV download. returns DataFrame indexed by date with one
    column (the series). bank-holiday blanks are dropped, dates are UTC-naive.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # FRED export columns: observation_date, <series_id>
    if "observation_date" not in df.columns or len(df.columns) < 2:
        log.warning(f"unexpected CSV format at {path}: cols={list(df.columns)}")
        return pd.DataFrame()
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    series_col = [c for c in df.columns if c != "observation_date"][0]
    df[series_col] = pd.to_numeric(df[series_col], errors="coerce")
    df = df.dropna().set_index("observation_date").sort_index()
    if series_label and series_label != series_col:
        df.columns = [series_label]
    return df


def vix_history(start: str = None, end: str = None) -> pd.DataFrame:
    """
    VIX history with three-tier fallback:
      1. local CSV at LOCAL_FRED_CSVS["vix"] if present (fastest, offline)
      2. FRED API if key is set
      3. empty DataFrame if neither available

    returns DataFrame indexed by date with single column "VIXCLS".
    """
    local_path = LOCAL_FRED_CSVS.get("vix")
    if local_path and local_path.exists():
        df = load_local_fred_csv(local_path, series_label="VIXCLS")
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        return df

    return fred_history("VIXCLS", start=start, end=end)


# ---------------------------------------------------------------------------
# convenience: combined market pulse
# ---------------------------------------------------------------------------

def market_pulse(equity_symbols: list = None) -> dict:
    """
    one-shot snapshot used by runners/market_pulse.py.
    pulls FRED macro + Finnhub quotes/news/sentiment for the equity universe.
    """
    equity_symbols = equity_symbols or ["SPY", "QQQ"]
    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "macro":        fred_snapshot(),
        "equities":     {},
    }
    for sym in equity_symbols:
        snapshot["equities"][sym] = {
            "quote":     finnhub_quote(sym),
            "sentiment": finnhub_sentiment(sym),
            "news":      finnhub_news(sym, lookback_days=2)[:5],   # top 5 headlines
        }
    snapshot["earnings_calendar"] = finnhub_earnings_calendar(lookahead_days=7)
    return snapshot


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not FINNHUB_API_KEY and not FRED_API_KEY:
        print("Neither FINNHUB_API_KEY nor FRED_API_KEY is set in .env — nothing to fetch.")
        sys.exit(0)

    print("--- Finnhub SPY quote ---")
    print(json.dumps(finnhub_quote("SPY"), indent=2))

    print("\n--- Finnhub SPY sentiment ---")
    print(json.dumps(finnhub_sentiment("SPY"), indent=2))

    print("\n--- Finnhub SPY top 3 news ---")
    for n in finnhub_news("SPY")[:3]:
        print(f"  [{n['datetime']}] {n['headline']}")

    print("\n--- FRED macro snapshot ---")
    print(json.dumps(fred_snapshot(), indent=2, default=str))
