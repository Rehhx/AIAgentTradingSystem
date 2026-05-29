"""
runners/trade_stats.py
----------------------
Quantify whether a strategy's win rate is statistically meaningful given how many
trades you have — built to interpret the Go system's "20% win rate, 2023-2025,
sparse trading" result.

Two things people conflate:
  1. Is the win rate REAL or just noise?  -> binomial confidence interval + test.
     Sparse trading => wide CI => "low significance" even if the edge is real.
  2. Is a 20% win rate even BAD?           -> no, IF the payoff ratio compensates.
     Breakeven win rate = 1 / (1 + avg_win/avg_loss). A trend strategy can win 20%
     of the time and still print money if winners are ~5x losers.

Input: a trades CSV with a P&L or return column (auto-detected: pnl, pnl_dollars,
return_pct, ret, profit). Or --demo to run on our RSI-2 trade logs.

Usage:
  python runners\\trade_stats.py --csv results/trades/daily_all_trades.csv
  python runners\\trade_stats.py --csv my_go_trades.csv --null 0.5
  python runners\\trade_stats.py --demo
"""
import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd


def _wilson(k: int, n: int, z: float = 1.96):
    """Wilson score 95% CI for a proportion — correct for small samples."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    h = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _binom_p(k: int, n: int, p0: float) -> float:
    """two-sided binomial test p-value (scipy if available, else normal approx)."""
    try:
        from scipy.stats import binomtest
        return float(binomtest(k, n, p0, alternative="two-sided").pvalue)
    except Exception:
        if n == 0:
            return 1.0
        from math import erf, sqrt
        mu, sd = n*p0, sqrt(n*p0*(1-p0))
        if sd == 0:
            return 1.0
        z = abs(k - mu) / sd
        return 2 * (1 - 0.5*(1 + erf(z/sqrt(2))))


def _pick_pnl(df: pd.DataFrame) -> pd.Series:
    cols = {c.lower(): c for c in df.columns}
    for cand in ("pnl_dollars", "pnl", "profit", "return_pct", "ret", "return"):
        if cand in cols:
            return pd.to_numeric(df[cols[cand]], errors="coerce").dropna()
    raise SystemExit(f"no P&L column found. columns: {list(df.columns)}")


def analyze(pnl: pd.Series, null: float, label: str, years: float | None):
    n = len(pnl)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    k = len(wins)
    wr = k / n if n else 0.0
    avg_w = wins.mean() if len(wins) else 0.0
    avg_l = abs(losses.mean()) if len(losses) else 0.0
    payoff = (avg_w / avg_l) if avg_l > 0 else float("inf")
    breakeven_wr = 1 / (1 + payoff) if payoff not in (0, float("inf")) else (0.0 if payoff == float("inf") else 1.0)
    pf = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    expectancy = pnl.mean()
    lo, hi = _wilson(k, n)
    p_vs_null = _binom_p(k, n, null)
    p_vs_be = _binom_p(k, n, breakeven_wr) if 0 < breakeven_wr < 1 else float("nan")

    print(f"\n========== {label} ==========")
    print(f"  trades              : {n}" + (f"  (~{n/years:.0f}/yr)" if years else ""))
    print(f"  win rate            : {wr:.1%}   95% CI [{lo:.1%}, {hi:.1%}]  (Wilson)")
    print(f"  avg win / avg loss  : {avg_w:,.4g} / {avg_l:,.4g}   payoff {payoff:.2f}x")
    print(f"  breakeven win rate  : {breakeven_wr:.1%}  (= 1/(1+payoff))")
    print(f"  profit factor       : {pf:.2f}     expectancy/trade: {expectancy:,.4g}")
    print(f"  total P&L           : {pnl.sum():,.4g}")
    print("  --- significance ---")
    print(f"  vs null {null:.0%}        : p = {p_vs_null:.3g} "
          f"({'SIGNIFICANT' if p_vs_null < 0.05 else 'not significant'} at 5%)")
    if 0 < breakeven_wr < 1:
        edge = "ABOVE breakeven (profitable)" if wr > breakeven_wr else "BELOW breakeven (losing)"
        print(f"  vs breakeven {breakeven_wr:.0%}    : p = {p_vs_be:.3g} | win rate is {edge}")
    # interpretation
    print("  --- interpretation ---")
    ci_w = hi - lo
    if n < 30:
        print(f"  * Only {n} trades -> CI width {ci_w:.0%}. Win rate is NOT reliably "
              f"estimable; you need ~100+ trades to trust it. This is your 'sparse "
              f"trading => low significance' problem.")
    elif ci_w > 0.25:
        print(f"  * CI width {ci_w:.0%} is wide -> still sample-limited; treat WR cautiously.")
    if expectancy > 0 and wr < 0.4:
        print(f"  * A {wr:.0%} win rate is FINE here: payoff {payoff:.1f}x clears the "
              f"{breakeven_wr:.0%} breakeven. This looks like a trend/momentum profile.")
    if expectancy <= 0:
        print(f"  * Negative expectancy: a {wr:.0%} win rate with payoff {payoff:.1f}x "
              f"loses money. If this is a MEAN-REVERSION strategy (should win ~65%), "
              f"suspect an inverted signal or bad/unadjusted data.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--demo", action="store_true", help="use our RSI-2 trade logs")
    ap.add_argument("--null", type=float, default=0.5, help="null win rate to test against")
    ap.add_argument("--years", type=float, default=None, help="span for per-year rate")
    ap.add_argument("--by", default=None, help="column to group by (e.g. ticker, strategy)")
    args = ap.parse_args()

    if args.demo:
        fs = glob.glob("results/trades/daily_rsi2_meanrev_*.csv")
        df = pd.concat([pd.read_csv(f) for f in fs]) if fs else pd.DataFrame()
        if df.empty:
            raise SystemExit("no demo trades found; run dump_daily_trades.py first")
        label = "RSI-2 (demo)"
    elif args.csv:
        df = pd.read_csv(args.csv)
        label = Path(args.csv).stem
    else:
        raise SystemExit("provide --csv <file> or --demo")

    pnl = _pick_pnl(df)
    analyze(pnl, args.null, label, args.years)

    if args.by and args.by in df.columns:
        print(f"\n--- by {args.by} ---")
        for g, sub in df.groupby(args.by):
            try:
                analyze(_pick_pnl(sub), args.null, f"{label} :: {g}", args.years)
            except Exception:
                pass


if __name__ == "__main__":
    main()
