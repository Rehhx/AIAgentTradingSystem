"""
runners/account_vs_spy.py
-------------------------
Read-only: how has an account done OVERALL (since inception) versus the S&P 500?
Pulls the Alpaca portfolio-history equity curve and compares it to SPY over the
exact same window -- total return, annualized vol/Sharpe, beta, and up/down
capture -- then breaks down the current invested-vs-cash split (the usual reason
a vol-targeted book lags a fully-invested index in an up market). No orders.

  python runners/account_vs_spy.py --account 1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings

warnings.filterwarnings("ignore")

import argparse

import numpy as np
import pandas as pd
import requests

from config import alpaca_keys, ALPACA_PAPER

BASE = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
TD = 252


def _headers(account):
    key, secret = alpaca_keys(account)
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}, key


def portfolio_history(account, period="1A", timeframe="1D"):
    h, key = _headers(account)
    if not key:
        return None
    r = requests.get(f"{BASE}/v2/account/portfolio/history", headers=h,
                     params={"period": period, "timeframe": timeframe,
                             "extended_hours": "true"}, timeout=30)
    r.raise_for_status()
    j = r.json()
    eq = pd.Series(j["equity"], index=pd.to_datetime(j["timestamp"], unit="s"))
    eq = eq[eq > 0]                                   # drop pre-funding zeros
    eq.index = eq.index.tz_localize("UTC").tz_convert("America/New_York").normalize().tz_localize(None)
    return eq.groupby(eq.index).last()                # one point per day


def spy_window(start, end):
    import yfinance as yf
    s = yf.Ticker("SPY").history(start=start, end=end + pd.Timedelta(days=2),
                                 auto_adjust=True)["Close"]
    s.index = s.index.tz_localize(None).normalize() if s.index.tz is None \
        else s.index.tz_convert("UTC").tz_localize(None).normalize()
    return s


def _stats(ret):
    sd = ret.std()
    return {"vol": float(sd * np.sqrt(TD)),
            "sharpe": float(ret.mean() / sd * np.sqrt(TD)) if sd > 0 else 0.0}


def current_allocation(account):
    from alpaca.trading.client import TradingClient
    key, secret = alpaca_keys(account)
    c = TradingClient(api_key=key, secret_key=secret, paper=ALPACA_PAPER)
    acct = c.get_account()
    eq, cash = float(acct.equity), float(acct.cash)
    rows = []
    for p in c.get_all_positions():
        rows.append((p.symbol.replace(".", "-"), float(p.market_value),
                     float(p.unrealized_pl)))
    return eq, cash, sorted(rows, key=lambda r: -abs(r[1]))


def main(account: int = 1):
    eq = portfolio_history(account)
    if eq is None or len(eq) < 2:
        print(f"account {account}: no portfolio history (unfunded or no keys)")
        return
    start, end = eq.index[0], eq.index[-1]
    spy = spy_window(start.date().isoformat(), end.date()).reindex(eq.index).ffill()

    acc_ret = eq.pct_change().dropna()
    spy_ret = spy.pct_change().reindex(acc_ret.index).fillna(0.0)

    acc_tot = float(eq.iloc[-1] / eq.iloc[0] - 1)
    spy_tot = float(spy.iloc[-1] / spy.iloc[0] - 1)
    a, s = _stats(acc_ret), _stats(spy_ret)
    var = spy_ret.var()
    beta = float(acc_ret.cov(spy_ret) / var) if var else float("nan")
    corr = float(acc_ret.corr(spy_ret))
    up = spy_ret > 0
    dn = spy_ret < 0
    up_cap = float(acc_ret[up].mean() / spy_ret[up].mean()) if up.any() and spy_ret[up].mean() else float("nan")
    dn_cap = float(acc_ret[dn].mean() / spy_ret[dn].mean()) if dn.any() and spy_ret[dn].mean() else float("nan")

    print("=" * 64)
    print(f"ACCOUNT {account}  vs  S&P 500 (SPY)   {start.date()} .. {end.date()}  "
          f"({len(eq)} days)")
    print("=" * 64)
    print(f"  {'metric':22s} {'account':>12s} {'SPY':>12s}")
    print("  " + "-" * 48)
    print(f"  {'total return':22s} {acc_tot:>+12.2%} {spy_tot:>+12.2%}")
    print(f"  {'annualized vol':22s} {a['vol']:>12.1%} {s['vol']:>12.1%}")
    print(f"  {'annualized Sharpe':22s} {a['sharpe']:>12.2f} {s['sharpe']:>12.2f}")
    print(f"  {'beta to SPY':22s} {beta:>12.2f} {1.0:>12.2f}")
    print(f"  {'correlation':22s} {corr:>12.2f}")
    print(f"  {'up-capture':22s} {up_cap:>12.0%}")
    print(f"  {'down-capture':22s} {dn_cap:>12.0%}")
    print(f"\n  gap vs SPY (total):  {acc_tot - spy_tot:>+.2%}")

    eq0, cash, rows = current_allocation(account)
    invested = sum(mv for _, mv, _ in rows)
    print("\n" + "=" * 64)
    print(f"  CURRENT ALLOCATION   equity ${eq0:,.0f}")
    print("=" * 64)
    print(f"  cash               ${cash:>11,.0f}  ({cash/eq0:>5.1%})")
    for sym, mv, upl in rows:
        tag = "  <- cash ETF (drag)" if sym in ("BIL", "SHV", "SGOV") else ""
        print(f"  {sym:8s}           ${mv:>11,.0f}  ({mv/eq0:>5.1%})  unrl {upl:>+8,.0f}{tag}")
    gross = (cash + invested)
    print(f"\n  invested {invested/eq0:>5.1%} | cash+ST {1 - invested/eq0:>5.1%} "
          f"-> effective equity beta ~{beta:.2f}")

    print("\n  READ: a vol-targeted, partly-in-cash book has beta < 1, so in an up")
    print("  market it CAPTURES less of SPY's gain (up-capture < 100%) by design --")
    print("  the trade-off is lower vol and smaller drawdowns. It lags on RETURN, not")
    print("  necessarily on RISK-ADJUSTED return (compare the Sharpe column).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", type=int, default=1)
    main(ap.parse_args().account)
