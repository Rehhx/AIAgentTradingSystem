"""
data/finnhub_fundamentals.py
----------------------------
Fundamental data via Finnhub (free tier: stock/metric = current snapshot of 130+
company financials). Builds a cross-sectional QUALITY/VALUE composite used to tilt
or filter the technical sleeves (e.g. "quality momentum": hold momentum names that
also pass a quality screen).

>>> HONEST DATA LIMIT <<<
The free tier returns the CURRENT fundamentals, not clean point-in-time history,
so this is a LIVE screen/tilt -- a rigorous historical backtest of a fundamental
factor needs paid point-in-time data (else look-ahead bias). Use it to rank/filter
today's holdings, not to claim a backtested fundamental edge.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import FINNHUB_API_KEY

CACHE = Path(__file__).parent / "cache" / "finnhub_fund"

# metric field -> higher_is_better (for the composite z-score)
FACTORS = {
    "roeTTM": True,                          # quality: return on equity
    "netProfitMarginTTM": True,              # quality: profitability
    "currentRatioQuarterly": True,           # quality: liquidity/solvency
    "totalDebt/totalEquityQuarterly": False, # quality: leverage (lower better)
    "revenueGrowthTTMYoy": True,             # growth
    "peTTM": False,                          # value (lower better)
}


def get_metrics(ticker: str, refresh: bool = False) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{ticker.replace('/', '-')}.json"
    if f.exists() and not refresh:
        try:
            return json.loads(f.read_text())
        except Exception:
            pass
    try:
        r = requests.get("https://finnhub.io/api/v1/stock/metric",
                         params={"symbol": ticker, "metric": "all", "token": FINNHUB_API_KEY},
                         timeout=15)
        if r.status_code != 200:
            return {}
        m = r.json().get("metric", {}) or {}
        f.write_text(json.dumps(m))
        return m
    except Exception:
        return {}


def fundamentals_frame(tickers) -> pd.DataFrame:
    rows = {t: {k: get_metrics(t).get(k) for k in FACTORS} for t in tickers}
    df = pd.DataFrame(rows).T
    for k in FACTORS:
        df[k] = pd.to_numeric(df[k], errors="coerce")
    return df


def quality_score(df: pd.DataFrame) -> pd.Series:
    """cross-sectional composite: mean of per-factor z-scores (signed so higher=better)."""
    z = pd.DataFrame(index=df.index)
    for k, higher_better in FACTORS.items():
        col = df[k]
        sd = col.std()
        zc = (col - col.mean()) / (sd if sd and sd > 0 else 1.0)
        z[k] = zc if higher_better else -zc
    return z.mean(axis=1, skipna=True)
