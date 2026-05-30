"""
runners/extended_backtest.py
----------------------------
Stress-test the CORE equity engine on 2005-2026 -- crucially including the 2008
GFC and 2011, the real bears our 2016-2026 window never had. Uses the per-ticker
sleeves that have long history (RSI-2, Donchian, 50/200 trend, recovery) on the
quality names available since 2005, with the live overlays (vol-target 17%/1.8x +
early-warning de-risk). The cross-sectional/PEAD/crypto sleeves are dropped here
(no clean survivorship-free data pre-2016), so this is the core engine only.

Exposes core_engine() so the board report can fold in the GFC-inclusive numbers.
$100k base, 6 bps.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from agents.daily_strategies import (
    sleeve_returns, sig_rsi2_meanrev, sig_donchian, sig_trend_5020, sig_recovery,
    vol_target, _metrics_from_returns, DEPLOY_PARAMS, TRADING_DAYS,
)

U = ["SPY", "QQQ", "GLD", "MSFT", "AAPL", "GOOGL", "AMZN", "JPM", "UNH", "XOM"]
WTS = {"rsi": 0.34, "don": 0.27, "trd": 0.17, "rec": 0.22}     # core-4, renormalized
SIGS = {"rsi": (sig_rsi2_meanrev, DEPLOY_PARAMS["rsi2_meanrev"]),
        "don": (sig_donchian, None), "trd": (sig_trend_5020, None),
        "rec": (sig_recovery, {"hold_days": 120})}
BEARS = {"2008 GFC": ("2007-10-01", "2009-03-09"), "2011 EU crisis": ("2011-05-01", "2011-10-03"),
         "2015-16 selloff": ("2015-08-01", "2016-02-11"), "2018 Q4": ("2018-10-01", "2018-12-24"),
         "COVID": ("2020-02-19", "2020-03-23"), "2022 bear": ("2022-01-01", "2022-10-12")}


def fetch_long(t, start="2005-01-01"):
    raw = yf.Ticker(t).history(start=start, end="2026-06-01", interval="1d", auto_adjust=True)
    if raw.empty:
        return None
    raw = raw.rename(columns={"Open": "open", "High": "high", "Low": "low",
                              "Close": "close", "Volume": "volume"})
    idx = raw.index
    raw.index = idx.tz_convert("UTC") if idx.tz else idx.tz_localize("UTC")
    return raw[["open", "high", "low", "close", "volume"]].dropna()


def _sleeve_book(data, sig, params):
    rets = [sleeve_returns(d, sig, params)[0] for d in data.values() if len(d) > 260]
    return pd.concat(rets, axis=1, sort=True).mean(axis=1)


def core_engine(start="2005-01-01"):
    """returns (book_returns, spy_returns) for the core equity engine since `start`."""
    data = {t: fetch_long(t, start) for t in U}
    data = {t: d for t, d in data.items() if d is not None and len(d) > 260}
    idx = pd.DatetimeIndex(sorted(set().union(*[d.index for d in data.values()])))
    sl = {k: _sleeve_book(data, *SIGS[k]).reindex(idx).fillna(0) for k in SIGS}
    combo = sum(sl[k] * WTS[k] for k in WTS)
    spy = data["SPY"]["close"].reindex(idx)
    warn = (spy < spy.rolling(50).mean()) & (spy.pct_change().rolling(20).std() * np.sqrt(TRADING_DAYS) > 0.20)
    ews = (1 - 0.4 * warn.astype(float)).shift(1).fillna(1.0)
    book = (vol_target(combo, 0.17, max_leverage=1.8) * ews).fillna(0)
    return book, spy.pct_change().fillna(0)


def main():
    print("pulling 2005-2026 data from yfinance + building the core engine (~1-2 min) ...")
    book, sret = core_engine()
    m = _metrics_from_returns(book, [], "core")
    ms = _metrics_from_returns(sret, [], "spy")
    print(f"\n=== CORE ENGINE 2005-2026 ({book.index[0].date()}..{book.index[-1].date()}) ===")
    print(f"  book: $100k -> ${m['final_capital']:,.0f} | CAGR {m['cagr']:.1%} | Sharpe {m['sharpe']} | maxDD {m['max_drawdown']:.1%}")
    print(f"  SPY : $100k -> ${ms['final_capital']:,.0f} | CAGR {ms['cagr']:.1%} | Sharpe {ms['sharpe']} | maxDD {ms['max_drawdown']:.1%}")
    print("\n  THE BEAR MARKETS (book vs SPY, total return over the window):")
    for nm, (a, b) in BEARS.items():
        print(f"    {nm:16s} book {(1+book.loc[a:b]).prod()-1:+7.1%}  vs  SPY {(1+sret.loc[a:b]).prod()-1:+7.1%}")
    print("\n  YEAR-BY-YEAR:")
    yb = (1 + book).groupby(book.index.year).prod() - 1
    ys = (1 + sret).groupby(sret.index.year).prod() - 1
    for y in yb.index:
        print(f"    {y}  book {yb[y]:+6.1%}  SPY {ys[y]:+6.1%}{'  <-- bear' if ys[y] < 0 else ''}")


if __name__ == "__main__":
    main()
