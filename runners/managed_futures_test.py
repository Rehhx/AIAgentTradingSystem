"""
runners/managed_futures_test.py
-------------------------------
Prototype of a managed-futures / CTA book using liquid ETF proxies (we lack true
futures data): time-series momentum LONG and SHORT across asset classes — equity,
bonds, commodities, gold, dollar, REITs. Shorting is the key: it lets the book
PROFIT when assets fall, the genuine "crisis alpha" that's uncorrelated to a
long-equity book. Inverse-vol risk weighting, vol-targeted.

Reports Sharpe / CAGR / DD, the per-fold returns, the 2022 bear (CTAs' big year),
and — most important — correlation to our equity portfolio.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    backtest_book, backtest_cross_sectional, vol_target, _metrics_from_returns,
    sig_rsi2_meanrev, sig_donchian, sig_trend_5020, DEPLOY_PARAMS, QUALITY_UNIVERSE,
    walk_forward_folds, daily_bars,
)
from data.sp500 import sp500_tickers

ETFS = ["SPY", "QQQ", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "UUP", "VNQ"]


def win(r, lo, hi):
    s = r[(r.index >= lo) & (r.index <= hi)]
    return (1 + s).prod() - 1


def main():
    closes = {}
    for t in ETFS:
        try:
            closes[t] = daily_bars(t)["close"]
        except Exception:
            pass
    panel = pd.concat(closes, axis=1); panel.columns = list(closes)
    rets = panel.pct_change()

    # time-series momentum: long uptrends / SHORT downtrends (12-month sign)
    mom = panel / panel.shift(252) - 1
    sig = np.sign(mom).shift(1)
    # inverse-vol risk weighting (each instrument equal risk), normalized to gross 1
    iv = 1.0 / rets.rolling(60).std()
    w = iv.div(iv.abs().sum(axis=1), axis=0)
    pos = (sig * w).fillna(0.0)
    gross = (pos * rets).sum(axis=1)
    turnover = pos.diff().abs().sum(axis=1).fillna(0)
    borrow = pos.clip(upper=0).abs().sum(axis=1) * (0.02 / 252)   # 2%/yr on shorts
    raw = gross - turnover * 0.0003 - borrow
    mf = vol_target(raw, target_vol=0.12, max_leverage=2.0)

    m = _metrics_from_returns(mf, [], "managed_futures")
    print(f"\n== managed-futures prototype (long/short TS-momentum, {len(closes)} ETFs, vol-target 12%) ==")
    print(f"  Sharpe {m['sharpe']} | CAGR {m['cagr']:.1%} | maxDD {m['max_drawdown']:.1%} | $PnL +${m['pnl_dollars']:,.0f}")
    print("  per-fold: " + "  ".join(f"{f.get('start','?')[:4]}-{f.get('end','?')[:4]}:{f['return_pct']:+.0%}"
                                     for f in walk_forward_folds(mf, 5)))
    print(f"  2018 (Q4 crash yr): {win(mf,'2018-01-01','2018-12-31'):+.1%}  |  "
          f"2022 (bond+stock bear): {win(mf,'2022-01-01','2022-12-31'):+.1%}")

    # correlation to the equity portfolio (the whole point)
    U = QUALITY_UNIVERSE
    rsi = backtest_book(sig_rsi2_meanrev, U, DEPLOY_PARAMS["rsi2_meanrev"])["_returns"]
    don = backtest_book(sig_donchian, U)["_returns"]
    trd = backtest_book(sig_trend_5020, U)["_returns"]
    xs = backtest_cross_sectional(sp500_tickers(), mode="momentum", lookback=252, skip=21, k=10, market_filter=True)["_returns"]
    panelq = pd.concat({"rsi": rsi, "don": don, "trd": trd, "xs": xs}, axis=1, sort=True)
    ivq = {c: 1.0 / (panelq[c].std() or 1e-9) for c in panelq.columns}
    wq = np.array([ivq[c] for c in panelq.columns]); wq /= wq.sum()
    eqport = vol_target((panelq.fillna(0.0) * wq).sum(axis=1), 0.16, max_leverage=1.6)
    corr = round(float(mf.reindex(eqport.index).fillna(0).corr(eqport)), 2)
    print(f"\n  correlation to the equity portfolio: {corr}   (near 0 = true diversifier)")

    # 50/50 blend with the equity book
    j = pd.concat([eqport, mf], axis=1, sort=True).fillna(0.0); j.columns = ["eq", "mf"]
    for label, series in [("equity portfolio", eqport),
                          ("70% equity + 30% MF", 0.7 * j["eq"] + 0.3 * j["mf"])]:
        mm = _metrics_from_returns(series, [], label)
        print(f"  {label:24s} Sharpe {mm['sharpe']:.2f} | CAGR {mm['cagr']:.1%} | DD {mm['max_drawdown']:.1%} "
              f"| 2018-2020 {walk_forward_folds(series,5)[1]['return_pct']:+.1%}")


if __name__ == "__main__":
    main()
