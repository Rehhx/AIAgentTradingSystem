"""
runners/options_coveredcall.py  (Account 3 research)
----------------------------------------------------
Covered-call (buy-write) income overlay: hold the index, sell a 1-month OTM call
each month, harvest the premium. Caps upside, cushions downside. Economically a
covered call ~ a short put (put-call parity), and we already rejected naked
put-writing (-24% DD) — so the honest question is whether, ON TOP of owning the
stock, the premium income improves RISK-ADJUSTED return or just caps the upside.

Priced with Black-Scholes on the real VIX (genuine implied vol). SPY proxy for the
equity book's beta. Monthly European approximation, 2007-2026, ~3 bps/roll.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from agents.daily_strategies import _metrics_from_returns, TRADING_DAYS
from runners.options_leverage import bs_call

ROLL_COST = 0.0003


def _series(t):
    s = yf.Ticker(t).history(start="2007-07-01", end="2026-06-03", auto_adjust=True)["Close"]
    s.index = (s.index.tz_localize(None) if s.index.tz is None else s.index.tz_convert("UTC").tz_localize(None)).normalize()
    return s


def covered_call(spy, vix, otm):
    """Monthly buy-write at `otm` above spot. Returns a monthly return series."""
    m_start = spy.resample("MS").first()
    m_end = spy.resample("MS").last()
    iv = (vix.resample("MS").first() / 100.0).clip(lower=0.05)
    rows = {}
    for dt in m_start.index:
        S0 = float(m_start[dt]); S1 = float(m_end.get(dt, S0)); sig = float(iv.get(dt, 0.18))
        if not np.isfinite(S0) or S0 <= 0:
            continue
        K = S0 * (1 + otm)
        prem = bs_call(S0, K, 21 / TRADING_DAYS, sig, 0.03)        # 1-month call premium
        capped = min(S1, K)
        rows[dt] = (capped - S0) / S0 + prem / S0 - ROLL_COST     # capped stock + premium yield
    return pd.Series(rows)


def main():
    print("pulling SPY + real VIX (2007-2026) ...\n")
    spy, vix = _series("SPY"), _series("^VIX")
    idx = spy.index.intersection(vix.index)
    spy, vix = spy.reindex(idx).ffill(), vix.reindex(idx).ffill()
    bh_m = spy.resample("MS").last().pct_change().dropna()        # buy-hold monthly

    def ann(r):
        m = r.mean(); sd = r.std()
        sharpe = m / sd * np.sqrt(12) if sd else 0
        eq = (1 + r).cumprod()
        cagr = eq.iloc[-1] ** (12 / len(r)) - 1
        dd = (eq / eq.cummax() - 1).min()
        return cagr, sharpe, dd, eq

    print("=" * 72)
    print("COVERED CALL (buy-write) vs SPY BUY-HOLD  (monthly, real-VIX priced)")
    print("=" * 72)
    print(f"  {'strategy':28s} {'CAGR':>6s} {'Sharpe':>7s} {'maxDD':>7s}")
    print("  " + "-" * 56)
    cg, sh, dd, _ = ann(bh_m)
    print(f"  {'SPY buy & hold':28s} {cg:>6.1%} {sh:>7.2f} {dd:>7.1%}")
    results = {}
    for otm in (0.02, 0.03, 0.05):
        cc = covered_call(spy, vix, otm).reindex(bh_m.index).dropna()
        cg, sh, dd, _ = ann(cc)
        results[otm] = (cg, sh, dd)
        print(f"  {'covered call ' + f'{int(otm*100)}% OTM':28s} {cg:>6.1%} {sh:>7.2f} {dd:>7.1%}")

    # crash + bull-year behaviour for the 3% strike
    cc3 = covered_call(spy, vix, 0.03).reindex(bh_m.index).dropna()
    yb = (1 + bh_m).groupby(bh_m.index.year).prod() - 1
    yc = (1 + cc3).groupby(cc3.index.year).prod() - 1
    print("\n  YEAR-BY-YEAR (3% OTM cc vs buy-hold) — note capped upside in big bull years:")
    for y in [2008, 2013, 2017, 2019, 2021, 2022, 2023, 2024]:
        print(f"    {y}  cc {yc.get(y,0):>+6.1%}  vs  buy-hold {yb.get(y,0):>+6.1%}")

    best = max(results, key=lambda k: results[k][1])
    bh_sh = ann(bh_m)[1]
    print("\n" + "=" * 72)
    print(f"  Best covered-call Sharpe ({int(best*100)}% OTM): {results[best][1]:.2f} vs buy-hold {bh_sh:.2f}")
    print("  Read: covered calls cut drawdown and vol but cap the big up-years. They earn")
    print("  a seat ONLY if Sharpe beats buy-hold meaningfully; otherwise it's a vol-")
    print("  reduction tool, not alpha — and the book already controls vol via targeting.")


if __name__ == "__main__":
    main()
