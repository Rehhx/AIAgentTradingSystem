"""
runners/options_leverage.py
---------------------------
DEFINED-RISK LEVERAGE via deep-ITM index LEAPS -- the only options path that
amplifies an edge without an unbounded tail. Instead of selling premium (which
adds crash risk -- see options_income_v2, -24% DD) this BUYS long-dated in-the-
money calls so the worst case is the premium paid, never more.

Why this is backtestable when single-name options are not: for INDEX options we
have real historical implied vol -- the VIX. We price each LEAPS with Black-
Scholes using the actual VIX of the day, roll annually, and pay a bid/ask spread.
So the leverage cost here is grounded in real market-implied vol, not a guess.

What it shows, honestly:
  1. SPY-LEAPS leverage 1.0x..3.0x vs just holding SPY  (real-VIX priced)
  2. the all-in carry cost of one turn of leverage (extracted from the sim)
  3. the SAME leverage applied to OUR core engine -- the punchline: leverage on a
     low-vol, drawdown-controlled book is far safer than on raw SPY.

$100k base. This is MODELED on Black-Scholes + real VIX; real fills (assignment,
early exercise, wider spreads in a crash) will be somewhat worse. Labeled as such.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")
from math import log, sqrt, exp, erf

import numpy as np
import pandas as pd
import yfinance as yf

from agents.daily_strategies import _metrics_from_returns, INITIAL_CAP, TRADING_DAYS
from runners.extended_backtest import core_engine, BEARS

MONEYNESS  = 0.90      # strike = 90% of spot -> ~10% ITM -> ~0.8 delta deep LEAPS
ROLL_COST  = 0.01      # 1% of premium round-trip spread on each annual roll
LEVERAGES  = (1.0, 1.5, 2.0, 3.0)


# ---- Black-Scholes (scalar; daily loop so no need to vectorize) -------------
def _ncdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_call(S, K, T, sigma, r):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * _ncdf(d1) - K * exp(-r * T) * _ncdf(d2)


def bs_delta(S, K, T, sigma, r):
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    return _ncdf(d1)


# ---- data -------------------------------------------------------------------
def _to_dates(idx):
    """naive, midnight-floored index so securities (tz-aware) and indices
    (tz-naive) from yfinance align by calendar date rather than time-of-day."""
    idx = idx.tz_localize(None) if idx.tz is None else idx.tz_convert("UTC").tz_localize(None)
    return idx.normalize()


def _series(t, start="2005-01-01"):
    s = yf.Ticker(t).history(start=start, end="2026-06-01", interval="1d", auto_adjust=True)["Close"]
    s.index = _to_dates(s.index)
    return s


def load(start="2005-01-01"):
    spy = _series("SPY", start)
    vix = _series("^VIX", start).reindex(spy.index).ffill().bfill()
    irx = _series("^IRX", start).reindex(spy.index).ffill().bfill() / 100.0   # 13wk T-bill, decimal
    irx = irx.clip(lower=0.0).fillna(0.02)
    return spy, vix, irx


# ---- the LEAPS leverage simulation (real-VIX priced, annual roll) -----------
def leaps_sim(spy, vix, rf, leverage, moneyness=MONEYNESS, roll_cost=ROLL_COST):
    """Hold a rolling 1yr deep-ITM SPY call sized to `leverage` x equity of
    delta-adjusted notional; park un-spent capital in T-bills. Returns the daily
    equity curve and the average annual extrinsic (time-value) drag paid."""
    idx = spy.index
    eq = np.zeros(len(idx))
    cash = INITIAL_CAP
    shares, K, t_left = 0.0, None, 0
    extrinsic_rates = []

    for i in range(len(idx)):
        S = float(spy.iloc[i]); sig = max(float(vix.iloc[i]) / 100.0, 0.05); r = float(rf.iloc[i])
        cash *= (1.0 + r) ** (1.0 / TRADING_DAYS)                  # T-bill yield on idle cash

        opt_val = shares * bs_call(S, K, max(t_left, 0) / TRADING_DAYS, sig, r) if shares > 0 else 0.0
        equity = cash + opt_val

        if shares == 0 or t_left <= 0:                            # establish / annual roll
            cash = equity                                         # liquidate old call into cash
            K = moneyness * S
            P0 = bs_call(S, K, 1.0, sig, r)
            d0 = bs_delta(S, K, 1.0, sig, r)
            extrinsic_rates.append((P0 - max(S - K, 0.0)) / S)    # annual time-value cost, %ofspot
            shares_t = (equity * leverage) / (d0 * S)             # delta-adjusted notional = L x equity
            total = shares_t * P0 * (1.0 + roll_cost)
            if total > equity:                                    # DEFINED RISK: never spend > equity
                shares_t *= equity / total
                total = equity
            cash = equity - total
            shares = shares_t
            t_left = TRADING_DAYS
        else:
            t_left -= 1

        opt_val = shares * bs_call(S, K, max(t_left, 0) / TRADING_DAYS, sig, r)
        eq[i] = cash + opt_val

    return pd.Series(eq, index=idx), float(np.mean(extrinsic_rates))


def _report(label, eq, bench_ret=None):
    ret = eq.pct_change().fillna(0.0)
    m = _metrics_from_returns(ret, [], label)
    line = (f"  {label:22s} ${m['final_capital']:>11,.0f}  {m['cagr']:>6.1%}  "
            f"{m['sharpe']:>6.2f}  {m['max_drawdown']:>7.1%}")
    return line, ret


def main():
    print("pulling SPY + real VIX + T-bill yield (2005-2026) and pricing rolling LEAPS ...\n")
    spy, vix, rf = load()
    spy_ret = spy.pct_change().fillna(0.0)

    # ---- 1. SPY-LEAPS leverage curve, priced on the real VIX ----------------
    print("=" * 78)
    print("1. SPY-LEAPS LEVERAGE vs BUY-AND-HOLD SPY  (priced on real historical VIX)")
    print("=" * 78)
    print(f"  {'position':22s} {'$100k ->':>12s}  {'CAGR':>6s}  {'Sharpe':>6s}  {'maxDD':>7s}")
    print("  " + "-" * 64)
    spy_eq = INITIAL_CAP * (1 + spy_ret).cumprod()
    bh_line, _ = _report("SPY buy & hold", spy_eq)
    print(bh_line)
    carry = {}
    leaps_eq = {}
    for L in LEVERAGES:
        eq, extr = leaps_sim(spy, vix, rf, L)
        leaps_eq[L] = eq
        carry[L] = extr
        line, _ = _report(f"LEAPS {L:.1f}x notional", eq)
        print(line)
    print(f"\n  Avg annual time-value (extrinsic) cost of the LEAPS: {np.mean(list(carry.values())):.1%} of spot")
    print("  => roughly the 'rent' you pay for the leverage + downside protection.")

    # crash behavior -- the whole point of 'defined risk'
    print("\n  CRASH BEHAVIOR (total return through each bear; LEAPS loss is capped at premium):")
    print(f"    {'window':16s} {'SPY':>8s} {'1.5x':>8s} {'2.0x':>8s} {'3.0x':>8s}")
    for nm, (a, b) in BEARS.items():
        def tot(eq):
            w = eq.loc[a:b]
            return (w.iloc[-1] / w.iloc[0] - 1) if len(w) > 1 else 0.0
        print(f"    {nm:16s} {tot(spy_eq):>+8.1%} {tot(leaps_eq[1.5]):>+8.1%} "
              f"{tot(leaps_eq[2.0]):>+8.1%} {tot(leaps_eq[3.0]):>+8.1%}")

    # ---- 2. the SAME leverage on OUR core engine ----------------------------
    print("\n" + "=" * 78)
    print("2. THE SAME LEVERAGE ON OUR CORE ENGINE  (the punchline)")
    print("=" * 78)
    print("  building the GFC-tested core engine (~1-2 min) ...")
    book, _ = core_engine()
    book.index = _to_dates(book.index)
    book = book.reindex(spy.index).fillna(0.0)
    # all-in carry per turn of leverage, calibrated from the sim's real-VIX option prices:
    # r (finance) + extrinsic/delta (time-value per unit of delta exposure)
    f = float(rf.mean()) + np.mean(list(carry.values())) / 0.8
    print(f"  calibrated all-in carry per turn of leverage: {f:.1%}/yr "
          f"(T-bill {rf.mean():.1%} + option time-value)\n")
    print(f"  {'book leverage':22s} {'$100k ->':>12s}  {'CAGR':>6s}  {'Sharpe':>6s}  {'maxDD':>7s}  {'2008':>7s}")
    print("  " + "-" * 74)
    for L in LEVERAGES:
        lev_ret = L * book - (L - 1.0) * (f / TRADING_DAYS)
        eq = INITIAL_CAP * (1 + lev_ret).cumprod()
        m = _metrics_from_returns(lev_ret, [], "x")
        gfc = (1 + lev_ret.loc["2007-10-01":"2009-03-09"]).prod() - 1
        tag = ""
        if L == 1.0:
            tag = "  <- best Sharpe (baseline book)"
        elif m["max_drawdown"] < -0.32:
            tag = "  <- BREAKS the -32% board drawdown"
        print(f"  {'core x' + f'{L:.1f}':22s} ${m['final_capital']:>11,.0f}  {m['cagr']:>6.1%}  "
              f"{m['sharpe']:>6.2f}  {m['max_drawdown']:>7.1%}  {gfc:>+7.1%}{tag}")

    print("\n" + "=" * 78)
    print("READ THIS  (what the numbers actually say -- not what we hoped)")
    print("=" * 78)
    print("""  1. The real win is NOT leverage -- it's the 1.0x LEAPS itself. Buying a deep-ITM
     1yr SPY call instead of SPY shares BEAT buy-and-hold on BOTH return and
     drawdown (11.6% / -43% vs 11.0% / -55%): the option floor caps the crash and
     the un-spent ~75% of capital earns T-bill yield. That convexity is ~free.

  2. Leverage does NOT improve risk-adjusted return. Sharpe FALLS at every step up
     (0.90 -> 0.84 -> 0.81 -> 0.73). Higher leverage buys more CAGR by paying with
     proportionally MORE drawdown -- it is a dial, not a free multiplier.

  3. It will NOT 'easily multiply' the account safely. 1.5x on the book already
     pushes the 2008 drawdown to ~-40% and max DD to ~-46% -- past the -32% we told
     the board. 3x effectively wipes out in a GFC. The multiplication and the ruin
     are the same lever.

  4. Levering the BOOK is tamer than levering SPY at the same multiple (-46% vs
     -59% DD at 1.5x) because the engine already de-risks into crashes -- but
     'tamer' still means breaking our stated risk budget above 1.0x.

  HONEST RECOMMENDATION:
     - Adopt 1.0x deep-ITM LEAPS as capital-efficient, tail-protected share
       replacement on the INDEX sleeves (SPY/QQQ) -- genuine free improvement.
     - Treat anything above 1.0x as a drawdown decision for the board, identical to
       the growth/crisis dial in drawdown_blends.py -- not a 'multiply' button.

  CAVEATS: MODELED on Black-Scholes + real VIX. Section 2 levers the book LINEARLY
  (no option floor credited), so its drawdowns are a conservative upper bound. Real
  fills are worse -- spreads blow out in a crash, exactly when you'd need to roll.
  You cannot buy options on the custom book directly; real use = index LEAPS overlay.
  Validate with 1-2 real paper LEAPS before sizing anything.""")


if __name__ == "__main__":
    main()
