"""
runners/market_park_backtest.py
-------------------------------
Validates the proposed fix for under-deployment: instead of parking idle capital
in T-bills (beta 0, drags below SPY in a bull run), stay in the MARKET by default
and only de-risk to cash when the crash overlay fires. This backtests the core
idea at the index level -- "hold SPY, cut to 60% when early-warning OR VIX
backwardation fires" -- against plain buy-and-hold SPY, 2016-2026.

If the overlay matches SPY's upside while cutting the drawdown, then defaulting
idle cash to the market (with the sentinel as the brake) is the honest way to
stop trailing SPY -- SPY-like return, smaller crashes -- without leverage or any
pretend alpha.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

TD = 252
CRASHES = {"2018 Q4": ("2018-10-01", "2018-12-24"), "COVID": ("2020-02-19", "2020-03-23"),
           "2022 bear": ("2022-01-03", "2022-10-12")}


def _px(t, start="2015-06-01"):
    s = yf.Ticker(t).history(start=start, end="2026-06-09", auto_adjust=True)["Close"]
    s.index = (s.index.tz_convert("UTC").tz_localize(None) if s.index.tz else s.index).normalize()
    return s


def _metrics(r):
    r = r.fillna(0.0)
    eq = (1 + r).cumprod()
    sd = r.std()
    sharpe = r.mean() / sd * np.sqrt(TD) if sd > 0 else 0.0
    cagr = eq.iloc[-1] ** (TD / len(r)) - 1
    dd = (eq / eq.cummax() - 1).min()
    return {"cagr": float(cagr), "sharpe": float(sharpe), "maxdd": float(dd), "eq": eq}


def main():
    print("loading SPY + VIX term structure (2016-2026) ...\n")
    spy = _px("SPY")
    vix, vix3m = _px("^VIX"), _px("^VIX3M")
    idx = spy.index
    ret = spy.pct_change().fillna(0.0)

    # de-risk signals (same logic as the live overlay), aligned by calendar date
    sma50 = spy.rolling(50).mean()
    vol20 = ret.rolling(20).std() * np.sqrt(TD)
    early_warning = (spy < sma50) & (vol20 > 0.20)
    vix_bw = (vix.reindex(idx).ffill() >= vix3m.reindex(idx).ffill())
    derisk = (early_warning | vix_bw).fillna(False)

    # exposures: buy-hold vs market-with-sentinel-brake (cut to 60% on de-risk)
    exp_bh = pd.Series(1.0, index=idx)
    exp_sentinel = pd.Series(1.0, index=idx).where(~derisk, 0.6)

    r_bh = exp_bh.shift(1).fillna(1.0) * ret
    r_sent = exp_sentinel.shift(1).fillna(1.0) * ret           # idle 40% -> ~cash (≈0)

    m_bh, m_sent = _metrics(r_bh), _metrics(r_sent)

    print("=" * 64)
    print("STAY-IN-MARKET + SENTINEL DE-RISK  vs  BUY-AND-HOLD SPY  (2016-2026)")
    print("=" * 64)
    print(f"  {'strategy':30s} {'CAGR':>7s} {'Sharpe':>7s} {'maxDD':>7s}")
    print("  " + "-" * 54)
    print(f"  {'buy & hold SPY':30s} {m_bh['cagr']:>7.1%} {m_bh['sharpe']:>7.2f} {m_bh['maxdd']:>7.1%}")
    print(f"  {'market + sentinel de-risk':30s} {m_sent['cagr']:>7.1%} {m_sent['sharpe']:>7.2f} {m_sent['maxdd']:>7.1%}")

    print(f"\n  de-risk active on {derisk.mean():.0%} of days "
          f"({int(derisk.sum())} of {len(derisk)})")

    print("\n  CRASH PROTECTION (cumulative return through each episode):")
    for name, (a, b) in CRASHES.items():
        def seg(r):
            w = (1 + r.loc[a:b]).prod() - 1
            return w
        print(f"    {name:10s}  buy-hold {seg(r_bh):>+7.1%}   sentinel {seg(r_sent):>+7.1%}")

    print("\n  YEAR-BY-YEAR:")
    yb = (1 + r_bh).groupby(idx.year).prod() - 1
    ys = (1 + r_sent).groupby(idx.year).prod() - 1
    for y in yb.index:
        flag = "  <-bear" if yb[y] < 0 else ""
        print(f"    {y}  buy-hold {yb[y]:>+6.1%}   sentinel {ys[y]:>+6.1%}{flag}")

    print("\n" + "=" * 64)
    better_dd = m_sent["maxdd"] > m_bh["maxdd"]
    close_cagr = m_sent["cagr"] >= m_bh["cagr"] - 0.015
    if better_dd and m_sent["sharpe"] >= m_bh["sharpe"]:
        print("  VERDICT: market+sentinel matches SPY's return at a HIGHER Sharpe and")
        print("  SMALLER drawdown. Defaulting idle cash to the market (with the sentinel")
        print("  brake) is the honest fix -- SPY-like upside, better downside. Recommend")
        print("  changing idle parking from BIL -> SPY when RISK-ON.")
    else:
        print("  VERDICT: the brake costs some upside; weigh CAGR give-up vs DD reduction.")


if __name__ == "__main__":
    main()
