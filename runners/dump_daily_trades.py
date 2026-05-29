"""
runners/dump_daily_trades.py
----------------------------
Write per-trade CSV logs for the daily strategy books — entry/exit dates,
prices, holding period, return and $ P&L for every round-trip.

Outputs (under results/trades/):
  daily_<strategy>_<ticker>.csv     one file per strategy+ticker
  daily_all_trades.csv              every trade across the book, sorted by entry

$ P&L is computed on the book's equal-weight sleeve notional = $100k / N names
(the capital actually allocated to each position in the portfolio).

Usage:
  python runners\\dump_daily_trades.py --book blended --universe SPY,QQQ,GLD,MSFT,JPM,GOOGL
  python runners\\dump_daily_trades.py --book rsi2_meanrev --universe SPY,QQQ,GLD,MSFT,JPM,GOOGL
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from agents.daily_strategies import (
    STRATEGIES_DAILY, DEPLOY_PARAMS, DEFAULT_UNIVERSE, daily_bars,
    sleeve_returns, INITIAL_CAP,
)

BOOKS = {
    "rsi2_meanrev": ["rsi2_meanrev"],
    "donchian":     ["donchian"],
    "trend_5020":   ["trend_5020"],
    "blended":      ["rsi2_meanrev", "donchian", "trend_5020"],
    "all":          ["rsi2_meanrev", "donchian", "trend_5020"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="blended", choices=list(BOOKS))
    ap.add_argument("--universe", default=",".join(DEFAULT_UNIVERSE))
    ap.add_argument("--outdir", default="results/trades")
    args = ap.parse_args()

    if args.universe.strip().lower() == "all":
        from data.loader import DATA_DIR
        universe = sorted(p.stem for p in Path(DATA_DIR).glob("*.parquet"))
    else:
        universe = [t.strip().upper() for t in args.universe.split(",") if t.strip()]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    strategies = BOOKS[args.book]
    # capital actually allocated to each (strategy x ticker) sleeve in this book:
    # $100k split equally across strategies, then equally across names.
    notional = INITIAL_CAP / (len(strategies) * len(universe))

    print(f"\nbook={args.book} | universe({len(universe)})={', '.join(universe)}")
    print(f"per-sleeve notional (= $100k / {len(strategies)} strat / "
          f"{len(universe)} names): ${notional:,.0f}\n")

    all_rows = []
    for strat in strategies:
        fn = STRATEGIES_DAILY[strat]
        for t in universe:
            try:
                d = daily_bars(t)
            except Exception as e:
                print(f"  [skip] {t}: {e}")
                continue
            _, trades = sleeve_returns(d, fn, DEPLOY_PARAMS.get(strat) or None)
            rows = []
            for tr in trades:
                ret = tr["ret"]
                rows.append({
                    "strategy":   strat,
                    "ticker":     t,
                    "side":       "long",
                    "entry_date": pd.Timestamp(tr["entry"]).date().isoformat(),
                    "exit_date":  pd.Timestamp(tr["exit"]).date().isoformat(),
                    "hold_days":  int(tr["bars"]),
                    "entry_px":   round(tr["entry_px"], 4),
                    "exit_px":    round(tr["exit_px"], 4),
                    "return_pct": round(ret * 100, 4),
                    "pnl_dollars": round(ret * notional, 2),
                    "status":     "open" if tr.get("open") else "closed",
                })
            if rows:
                df = pd.DataFrame(rows)
                fpath = outdir / f"daily_{strat}_{t}.csv"
                df.to_csv(fpath, index=False, encoding="utf-8")
                all_rows.extend(rows)
                wins = (df["pnl_dollars"] > 0).sum()
                print(f"  {strat:14s} {t:6s} | {len(df):3d} trades | "
                      f"WR {wins/len(df):5.1%} | $PnL {df['pnl_dollars'].sum():>10,.0f} "
                      f"-> {fpath.name}")

    if all_rows:
        combined = pd.DataFrame(all_rows).sort_values(["entry_date", "ticker", "strategy"])
        cpath = outdir / "daily_all_trades.csv"
        combined.to_csv(cpath, index=False, encoding="utf-8")
        print(f"\nTOTAL: {len(combined)} trades across {combined['ticker'].nunique()} "
              f"names, {combined['strategy'].nunique()} strategies")
        print(f"  date range: {combined['entry_date'].min()} .. {combined['exit_date'].max()}")
        print(f"  aggregate $PnL (equal-weight sleeves): "
              f"${combined['pnl_dollars'].sum():,.0f}")
        print(f"  win rate: {(combined['pnl_dollars'] > 0).mean():.1%}")
        print(f"\nWrote {cpath}")


if __name__ == "__main__":
    main()
