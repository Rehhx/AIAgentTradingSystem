"""
runners/data_quality_check.py
-----------------------------
Validate that the daily price data feeding a strategy is clean. Built to debug
"weird results" (e.g. an unexpected 20% win rate) — the most common cause is bad
data, not a bad strategy. Runs the same checks you'd want on the feed your Go
system uses.

Checks per ticker (daily bars):
  - coverage: row count, first/last date
  - gaps: business days in range with no bar (holidays expected; many gaps = bad)
  - duplicates: repeated index timestamps
  - NaN / non-positive prices, zero-volume days
  - OHLC integrity: high>=low, high>=max(open,close), low<=min(open,close)
  - extreme moves: |daily return| > 35%  -> usually an UNADJUSTED SPLIT or bad tick
  - stale prices: >=3 identical consecutive closes -> halted/illiquid/stale feed
  - adjustment check (--check-adjustment): compares yfinance adjusted vs raw
    close; large divergence means a feed using RAW close will be WRONG (splits/divs)

Usage:
  python runners\\data_quality_check.py --tickers SPY,QQQ,AAPL,NVDA,TSLA
  python runners\\data_quality_check.py --tickers SPY,AAPL --check-adjustment
  python runners\\data_quality_check.py --csv mydata.csv          # your own OHLCV csv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

EXTREME_RET = 0.35      # |daily return| above this is almost certainly bad data
STALE_RUN   = 3         # identical consecutive closes


def _load_csv(path: str) -> dict:
    df = pd.read_csv(path)
    # find a date column + ohlcv columns case-insensitively
    cols = {c.lower(): c for c in df.columns}
    datecol = next((cols[c] for c in ("date", "timestamp", "time", "eventat") if c in cols), df.columns[0])
    df[datecol] = pd.to_datetime(df[datecol], utc=True, errors="coerce")
    df = df.set_index(datecol).sort_index()
    ren = {}
    for want in ("open", "high", "low", "close", "volume"):
        if want in cols:
            ren[cols[want]] = want
    df = df.rename(columns=ren)
    return {"csv:" + Path(path).stem: df}


def check_one(t: str, d: pd.DataFrame) -> dict:
    flags = []
    n = len(d)
    if n == 0:
        return {"ticker": t, "rows": 0, "verdict": "FAIL", "flags": ["no rows"]}
    close = d["close"]
    # gaps — work in tz-naive normalized dates to avoid tz arithmetic issues
    idx = d.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    dates = pd.DatetimeIndex(idx).normalize()
    present = set(dates)
    bdays = pd.bdate_range(dates.min(), dates.max())
    missing = sum(1 for b in bdays if b not in present)
    yrs = max((dates.max() - dates.min()).days / 365.25, 1e-9)
    gaps_per_yr = missing / yrs
    # duplicates
    dups = int(d.index.duplicated().sum())
    # bad prices
    n_nan = int(close.isna().sum())
    n_nonpos = int((close <= 0).sum())
    # OHLC integrity
    if {"open", "high", "low"}.issubset(d.columns):
        hi, lo, op = d["high"], d["low"], d["open"]
        viol = int(((hi < lo) | (hi < close - 1e-9) | (hi < op - 1e-9) |
                    (lo > close + 1e-9) | (lo > op + 1e-9)).sum())
    else:
        viol = -1
    # extreme moves (split / bad tick)
    ret = close.pct_change()
    extreme = ret[ret.abs() > EXTREME_RET]
    # stale
    same = (close.diff() == 0)
    stale_runs = int(((same) & (same.shift(1)) & (same.shift(2))).sum())
    # zero volume
    n_zerovol = int((d["volume"] <= 0).sum()) if "volume" in d.columns else -1

    if gaps_per_yr > 15: flags.append(f"{gaps_per_yr:.0f} missing days/yr")
    if dups: flags.append(f"{dups} duplicate dates")
    if n_nan: flags.append(f"{n_nan} NaN closes")
    if n_nonpos: flags.append(f"{n_nonpos} non-positive closes")
    if viol > 0: flags.append(f"{viol} OHLC violations")
    if len(extreme): flags.append(f"{len(extreme)} moves >{EXTREME_RET:.0%} (split/unadjusted?)")
    if stale_runs > 5: flags.append(f"{stale_runs} stale runs")
    if n_zerovol > 5: flags.append(f"{n_zerovol} zero-volume days")

    verdict = "OK" if not flags else ("FAIL" if (viol > 0 or n_nan or n_nonpos or dups
                                                 or len(extreme) > 3) else "WARN")
    return {
        "ticker": t, "rows": n,
        "range": f"{d.index.min().date()}..{d.index.max().date()}",
        "gaps_per_yr": round(gaps_per_yr, 1), "dups": dups,
        "nan": n_nan, "nonpos": n_nonpos, "ohlc_viol": viol,
        "extreme_moves": len(extreme),
        "biggest_move": (f"{ret.abs().max():.0%} on {ret.abs().idxmax().date()}"
                         if n > 1 else "n/a"),
        "stale_runs": stale_runs, "zero_vol": n_zerovol,
        "verdict": verdict, "flags": flags,
    }


def adjustment_check(tickers, start):
    import yfinance as yf
    print("\n=== adjustment check (adjusted vs raw close divergence) ===")
    print("large divergence => a feed using RAW/unadjusted close will be WRONG\n")
    for t in tickers:
        try:
            adj = yf.Ticker(t).history(start=start, interval="1d", auto_adjust=True)["Close"]
            raw = yf.Ticker(t).history(start=start, interval="1d", auto_adjust=False)["Close"]
            j = pd.concat([adj.rename("adj"), raw.rename("raw")], axis=1).dropna()
            if j.empty:
                print(f"  {t:6s}: no data"); continue
            div = (j["raw"] / j["adj"] - 1).abs().max()
            note = "OK (use adjusted)" if div > 0.02 else "minimal"
            print(f"  {t:6s}: max raw-vs-adjusted divergence {div:6.1%}  -> {note}")
        except Exception as e:
            print(f"  {t:6s}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="SPY,QQQ,GLD,AAPL,NVDA,TSLA")
    ap.add_argument("--csv", default=None, help="validate an OHLCV csv instead")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--check-adjustment", action="store_true")
    args = ap.parse_args()

    if args.csv:
        data = _load_csv(args.csv)
    else:
        from data.sp500 import load_daily
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        data = load_daily(tickers, start=args.start)

    print(f"\nData-quality check | {len(data)} series | start={args.start}\n")
    hdr = (f"{'ticker':10s} {'rows':>5s} {'range':>23s} {'gap/yr':>7s} "
           f"{'extreme':>7s} {'biggestMove':>18s} {'verdict':>7s}")
    print(hdr); print("-" * len(hdr))
    reports = []
    for t, d in data.items():
        r = check_one(t, d)
        reports.append(r)
        print(f"{t:10s} {r['rows']:>5} {r.get('range',''):>23s} "
              f"{r.get('gaps_per_yr',0):>7} {r.get('extreme_moves',0):>7} "
              f"{str(r.get('biggest_move','')):>18s} {r['verdict']:>7s}")
        if r["flags"]:
            print(f"           flags: {'; '.join(r['flags'])}")

    if args.check_adjustment and not args.csv:
        adjustment_check([t.strip().upper() for t in args.tickers.split(",")], args.start)

    bad = [r for r in reports if r["verdict"] != "OK"]
    print(f"\n{len(reports)-len(bad)}/{len(reports)} clean. "
          f"{len(bad)} with warnings/failures.")
    Path("results/data_quality.json").write_text(
        json.dumps(reports, indent=2, default=str), encoding="utf-8")
    print("Wrote results/data_quality.json")


if __name__ == "__main__":
    main()
