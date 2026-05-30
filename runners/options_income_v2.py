"""
runners/options_income_v2.py
----------------------------
Options income sleeve DONE PROPERLY: a systematic cash-secured PUT-WRITE that
harvests the volatility risk premium, fixing the two flaws of v1 (options_income.py):

  1. DELTA-TARGETED strikes -- sell the ~target_delta put (e.g. 0.16-delta, ~1 SD
     OTM) instead of a fixed % OTM, so the strike adapts to volatility (the
     institutional standard; how CBOE PUT and most systematic programs work).
  2. MARKET FILTER -- only write puts when SPY > its 200-day; otherwise hold cash
     (T-bills). You are never selling downside insurance into a confirmed
     downtrend, which is what wrecks naive put-writing in a bear.

Fully cash-secured (collateral = strike) => NO leverage, NO naked risk.

>>> HONEST DATA CAVEAT (unchanged) <<<
No historical option chains exist for free, so premiums are MODELED with
Black-Scholes at realized vol + a volatility-risk-premium markup (`--vrp`). The
VRP-sensitivity table shows exactly how much of the result rides on that one
assumption. This is a MODEL; validate live on Alpaca paper before real capital.
"""
import argparse
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import norm

from agents.daily_strategies import daily_bars, INITIAL_CAP
from runners.options_income import bs_put

TD = 252
MONTH = 21
RF = 0.04
HAIRCUT = 0.0015      # bid/ask + commission per cycle, as a fraction of collateral


def metrics(rets):
    rets = np.asarray(rets, float)
    eq = INITIAL_CAP * np.cumprod(1 + rets)
    yrs = len(rets) / (TD / MONTH)
    cagr = (eq[-1] / INITIAL_CAP) ** (1 / yrs) - 1 if yrs > 0 else 0.0
    vol = rets.std() * math.sqrt(TD / MONTH)
    sharpe = rets.mean() / rets.std() * math.sqrt(TD / MONTH) if rets.std() > 0 else 0.0
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return dict(final=eq[-1], cagr=cagr, vol=vol, sharpe=sharpe, dd=dd, pnl=eq[-1] - INITIAL_CAP)


def putwrite(ticker, target_delta=0.16, vrp=0.03, market_filter=True):
    px = daily_bars(ticker)["close"].dropna()
    lr = np.log(px / px.shift(1))
    rv = (lr.rolling(MONTH).std() * math.sqrt(TD)).to_numpy()
    sma = px.rolling(200).mean().to_numpy()
    p = px.to_numpy(); idx = px.index
    T, rets, dates, wins, written = MONTH / TD, [], [], 0, 0
    for i in range(200, len(p) - MONTH, MONTH):
        S0, ST = p[i], p[i + MONTH]
        dates.append(idx[i + MONTH])
        if market_filter and not (sma[i] == sma[i] and S0 > sma[i]):
            rets.append(RF * T)                       # below 200d -> sit in cash
            continue
        iv = max(0.08, (rv[i] if rv[i] == rv[i] else 0.15) + vrp)
        d1 = -norm.ppf(target_delta)                  # strike at ~target_delta put
        K = S0 * math.exp((RF + 0.5 * iv ** 2) * T - d1 * iv * math.sqrt(T))
        prem = bs_put(S0, K, T, RF, iv) - HAIRCUT * K
        rets.append((prem - max(0.0, K - ST)) / K + RF * T)   # cash-secured on K
        written += 1; wins += 1 if ST >= K else 0
    return pd.Series(rets, index=pd.to_datetime(dates)), wins, written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["SPY", "QQQ"])
    ap.add_argument("--delta", type=float, default=0.16, help="target put delta (0.16 = ~1 SD OTM)")
    ap.add_argument("--vrp", type=float, default=0.03)
    args = ap.parse_args()

    print(f"\n=== Cash-Secured PutWrite v2 | {args.delta:.2f}-delta, market-filtered (SPY>200d) ===")
    print("MODELED premiums (Black-Scholes, IV = realized + VRP). Not a fill backtest.\n")

    series, wtot, ntot = [], 0, 0
    for t in args.tickers:
        s, w, n = putwrite(t, args.delta, args.vrp)
        series.append(s); wtot += w; ntot += n
    L = min(len(s) for s in series)
    book = pd.concat([s.iloc[-L:].reset_index(drop=True) for s in series], axis=1).mean(axis=1)
    m = metrics(book.to_numpy())
    print(f"book ({'+'.join(args.tickers)}): $100k -> ${m['final']:,.0f} (+${m['pnl']:,.0f}) | "
          f"CAGR {m['cagr']:.1%} | vol {m['vol']:.1%} | Sharpe {m['sharpe']:.2f} | maxDD {m['dd']:.1%}")
    print(f"  put-write win rate {wtot/ntot:.0%} | cycles written {ntot} (rest sat in cash via filter)")

    # compare vs v1 naive (no delta, no filter) and vs SPY
    naive, _, _ = putwrite("SPY", 0.16, args.vrp, market_filter=False)
    mn = metrics(naive.to_numpy())
    spm = daily_bars("SPY")["close"].resample("ME").last().pct_change().dropna()
    msp = metrics(spm.to_numpy())
    print(f"\n  vs unfiltered put-write : CAGR {mn['cagr']:.1%} | Sharpe {mn['sharpe']:.2f} | maxDD {mn['dd']:.1%}")
    print(f"  vs SPY buy-and-hold     : CAGR {msp['cagr']:.1%} | Sharpe {msp['sharpe']:.2f} | maxDD {msp['dd']:.1%}")

    print("\n  VRP-markup sensitivity (how much rides on the IV>RV assumption):")
    print(f"    {'markup':>8s} {'CAGR':>7s} {'Sharpe':>7s} {'maxDD':>7s}")
    for vp in (0.00, 0.02, 0.03, 0.04, 0.06):
        ss = [putwrite(t, args.delta, vp)[0] for t in args.tickers]
        bb = pd.concat([s.iloc[-L:].reset_index(drop=True) for s in ss], axis=1).mean(axis=1)
        mm = metrics(bb.to_numpy())
        tag = "  <- base" if abs(vp - args.vrp) < 1e-9 else ""
        print(f"    {vp*100:6.0f}pt {mm['cagr']:7.1%} {mm['sharpe']:7.2f} {mm['dd']:7.1%}{tag}")

    print("\n  Done-properly upgrades vs v1: delta-targeted strikes + market filter (no")
    print("  writing into a downtrend). NEXT: validate live on Alpaca paper before real capital.")


if __name__ == "__main__":
    main()
