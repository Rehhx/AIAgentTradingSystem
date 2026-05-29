"""
runners/options_income.py
-------------------------
NO-LEVERAGE options INCOME sleeve: systematic covered-call (BuyWrite / CBOE BXM)
and cash-secured put-write (PutWrite / CBOE PUT). These harvest the VOLATILITY
RISK PREMIUM -- option buyers systematically overpay for insurance, so a
disciplined seller earns the implied-minus-realized spread. Both are fully
collateralized (covered call = you own the shares; put-write = cash secures the
strike), so NO leverage and NO naked short risk.

  BuyWrite  : own SPY, sell a 1-month ~2% OTM call each month -> stock upside
              capped at the strike, but you pocket the call premium every month.
  PutWrite  : hold cash, sell a 1-month ~2% OTM put each month, cash secures the
              strike -> keep the premium unless assigned below the strike.

>>> HONEST DATA CAVEAT <<<
We do NOT have historical option chains (yfinance gives only current chains).
So premiums are MODELED with Black-Scholes using realized vol PLUS a documented
volatility-risk-premium markup (`--vrp`, in vol points). Empirically, SPX
implied vol runs ~2-4 points above subsequently-realized vol; we show a
sensitivity table across markups so you can see exactly how much of the result
is the VRP assumption. This is a MODEL, not a fill-level backtest -- validate it
live on Alpaca paper (agents/options_agent.py can place these) before trusting.

$100k base, monthly (~21 trading day) non-overlapping windows, adjusted data.
Costs: $0.65/contract + ~$0.03 slippage modeled as a flat premium haircut.
"""
import argparse
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from agents.daily_strategies import daily_bars, INITIAL_CAP

TDAYS = 252
MONTH = 21               # trading days per option cycle
R_FREE = 0.04            # risk-free for BS + cash-secured collateral yield
PREMIUM_HAIRCUT = 0.0015  # ~bid/ask + commission as a fraction of notional per cycle


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S, K, T, r, sig):
    if T <= 0 or sig <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)


def bs_put(S, K, T, r, sig):
    return bs_call(S, K, T, r, sig) - S + K * math.exp(-r * T)


def _metrics(rets: np.ndarray, periods_per_year=TDAYS / MONTH):
    rets = np.asarray(rets, float)
    eq = INITIAL_CAP * np.cumprod(1 + rets)
    years = len(rets) / periods_per_year
    cagr = (eq[-1] / INITIAL_CAP) ** (1 / years) - 1 if years > 0 else 0.0
    vol = rets.std() * math.sqrt(periods_per_year)
    sharpe = (rets.mean() / rets.std() * math.sqrt(periods_per_year)) if rets.std() > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = float((eq / peak - 1).min())
    return dict(final=float(eq[-1]), cagr=cagr, vol=vol, sharpe=sharpe, maxdd=dd,
                pnl=float(eq[-1] - INITIAL_CAP))


def run_strategy(ticker, kind, otm, vrp):
    """Return (monthly_returns, underlying_monthly_returns, dates, detail)."""
    px = daily_bars(ticker)["close"].dropna()
    logret = np.log(px / px.shift(1))
    rv = (logret.rolling(MONTH).std() * math.sqrt(TDAYS)).to_numpy()
    p = px.to_numpy()
    idx = px.index
    starts = list(range(MONTH, len(p) - MONTH, MONTH))
    T = MONTH / TDAYS
    cash_yield = R_FREE * T
    rets, under, dates, detail = [], [], [], []
    for i in starts:
        S0, ST = p[i], p[i + MONTH]
        iv = max(0.08, (rv[i] if not math.isnan(rv[i]) else 0.15) + vrp)
        under.append(ST / S0 - 1)
        dates.append(idx[i])
        if kind == "putwrite":
            K = S0 * (1 - otm)
            prem = bs_put(S0, K, T, R_FREE, iv) / S0 - PREMIUM_HAIRCUT
            assigned = ST < K
            payoff = prem - max(0.0, K - ST) / S0 + cash_yield      # cash-secured
        else:  # buywrite / covered call
            K = S0 * (1 + otm)
            prem = bs_call(S0, K, T, R_FREE, iv) / S0 - PREMIUM_HAIRCUT
            assigned = ST > K                                       # called away
            payoff = (min(ST, K) - S0) / S0 + prem                  # capped stock + premium
        rets.append(payoff)
        detail.append({"premium": prem, "assigned": assigned})
    return np.array(rets), np.array(under), dates, detail


def print_detail(book, under, dates, detail_list, kind):
    """Year-by-year book vs underlying + per-cycle statistics."""
    import pandas as pd
    n = len(book)
    ser = pd.Series(book, index=pd.to_datetime(dates[-n:]))
    und = pd.Series(under[-n:], index=pd.to_datetime(dates[-n:]))
    yb = (1 + ser).groupby(ser.index.year).prod() - 1
    yu = (1 + und).groupby(und.index.year).prod() - 1
    print("\n  year-by-year (option book vs underlying buy-hold, same windows):")
    print(f"    {'year':>4s} {'options':>9s} {'underlying':>11s}   edge")
    for y in yb.index:
        e = yb[y] - yu[y]
        print(f"    {y:>4d} {yb[y]:+9.1%} {yu[y]:+11.1%}   {e:+.1%}")
    # per-cycle stats averaged across the underlyings
    assigned = np.mean([[d["assigned"] for d in dl] for dl in detail_list], axis=0)
    prem = np.mean([[d["premium"] for d in dl] for dl in detail_list], axis=0)
    prem = prem[-n:]; assigned = assigned[-n:]
    verb = "assigned (put ITM)" if kind == "putwrite" else "called away (call ITM)"
    print(f"\n  per-cycle ({n} monthly cycles):")
    print(f"    win rate (premium kept, not {verb.split()[0]}): {(assigned < 0.5).mean():.0%}")
    print(f"    avg premium collected / cycle: {prem.mean():.2%}  (~{prem.mean()*12:.1%} annualized gross)")
    print(f"    {verb} frequency: {(assigned >= 0.5).mean():.0%} of cycles")
    print(f"    best month {book.max():+.1%} | worst month {book.min():+.1%} | "
          f"positive months {(book > 0).mean():.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["putwrite", "buywrite"], default="putwrite")
    ap.add_argument("--tickers", nargs="+", default=["SPY", "QQQ"])
    ap.add_argument("--otm", type=float, default=0.02, help="strike distance OTM (0.02 = 2%)")
    ap.add_argument("--vrp", type=float, default=0.03, help="vol-risk-premium markup in vol points")
    args = ap.parse_args()

    label = "Cash-Secured PutWrite (CBOE PUT-style)" if args.kind == "putwrite" else "Covered Call / BuyWrite (CBOE BXM-style)"
    print(f"\n=== {label} | {args.otm:.0%} OTM, 1-month cycles, NO leverage ===")
    print("MODELED premiums (Black-Scholes, IV = realized vol + VRP markup). Not a fill backtest.\n")

    # equal-weight across the chosen underlyings
    all_rets, all_det, spy_b, dates0 = [], [], None, None
    for t in args.tickers:
        r, u, dts, det = run_strategy(t, args.kind, args.otm, args.vrp)
        all_rets.append(r); all_det.append(det)
        if t == "SPY":
            spy_b = u
        dates0 = dts
    n = min(len(r) for r in all_rets)
    book = np.mean([r[-n:] for r in all_rets], axis=0)
    m = _metrics(book)
    print(f"book ({'+'.join(args.tickers)}, equal weight):")
    print(f"  $100,000 -> ${m['final']:,.0f}  (+${m['pnl']:,.0f}) | CAGR {m['cagr']:.1%} "
          f"| vol {m['vol']:.1%} | Sharpe {m['sharpe']:.2f} | maxDD {m['maxdd']:.1%}")

    if spy_b is not None:
        bm = _metrics(spy_b[-n:])
        corr = float(np.corrcoef(book, spy_b[-n:])[0, 1])
        print(f"\n  vs SPY buy-and-hold (same windows):")
        print(f"    SPY:      CAGR {bm['cagr']:.1%} | vol {bm['vol']:.1%} | Sharpe {bm['sharpe']:.2f} | maxDD {bm['maxdd']:.1%}")
        print(f"    options:  CAGR {m['cagr']:.1%} | vol {m['vol']:.1%} | Sharpe {m['sharpe']:.2f} | maxDD {m['maxdd']:.1%}")
        print(f"    -> vol cut {(1-m['vol']/bm['vol'])*100:.0f}%, correlation to SPY {corr:.2f} "
              f"(income/low-vol equity, NOT market-neutral)")
        print_detail(book, spy_b, dates0, all_det, args.kind)

    print("\n  VRP-markup sensitivity (how much of the edge is the IV>RV assumption):")
    print(f"    {'markup':>8s} {'CAGR':>7s} {'Sharpe':>7s} {'maxDD':>7s}")
    for vp in (0.00, 0.02, 0.03, 0.04, 0.06):
        rr = []
        for t in args.tickers:
            r, _, _, _ = run_strategy(t, args.kind, args.otm, vp)
            rr.append(r[-n:])
        mm = _metrics(np.mean(rr, axis=0))
        tag = "  <- base" if abs(vp - args.vrp) < 1e-9 else ""
        print(f"    {vp*100:6.0f}pt {mm['cagr']:7.1%} {mm['sharpe']:7.2f} {mm['maxdd']:7.1%}{tag}")
    print("\n  At 0pt (IV=RV, no premium edge) the strategy should barely clear its risk —")
    print("  the spread between that row and the base row IS the volatility risk premium.")
    print("\nNEXT: validate live on Alpaca paper via agents/options_agent.py before real capital.")


if __name__ == "__main__":
    main()
