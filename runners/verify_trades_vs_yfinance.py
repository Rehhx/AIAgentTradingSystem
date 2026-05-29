"""
runners/verify_trades_vs_yfinance.py
-------------------------------------
Cross-check backtest fills against an independent data source. For every trade in
a backtest trade log, pull yfinance daily OHLC for that ticker on the entry and
exit dates and verify the backtest's price falls within that day's [low, high]
range. A price OUTSIDE the day's range means the backtest could not actually have
filled there — usually a sign that the backtest's price feed (local parquet)
disagrees with yfinance (most often a SPLIT/DIVIDEND ADJUSTMENT difference).

What it reports per ticker and overall:
  - % of entries / exits whose price is within the day's [low, high]
  - mean |% difference| between the backtest price and yfinance's close
    (our daily strategy fills at the close, so this should be ~0 if the feeds
     agree; a large, systematic gap = adjustment mismatch)
  - the worst mismatches, so you can eyeball them

Run it against BOTH adjustment modes to diagnose:
  --adjust auto  (default) compares to split/dividend-ADJUSTED yfinance
  --adjust raw             compares to UNADJUSTED yfinance
If the backtest matches 'raw' but not 'auto', the parquet is unadjusted and the
backtest carries split risk (fake gaps on split days).

Usage:
  python runners\\verify_trades_vs_yfinance.py
  python runners\\verify_trades_vs_yfinance.py --csv results/trades/daily_all_trades.csv
  python runners\\verify_trades_vs_yfinance.py --adjust raw
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

TOL = 0.001          # 0.1% slack on the [low, high] band for rounding


def fetch_ohlc(ticker: str, start, end, adjust: str) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.Ticker(ticker).history(
        start=str(pd.Timestamp(start).date()),
        end=str((pd.Timestamp(end) + pd.Timedelta(days=3)).date()),
        interval="1d", auto_adjust=(adjust == "auto"))
    if raw.empty:
        return raw
    raw = raw.rename(columns={"Open": "open", "High": "high", "Low": "low",
                              "Close": "close"})
    # index -> tz-naive date for matching
    idx = raw.index
    idx = idx.tz_convert("UTC").tz_localize(None) if getattr(idx, "tz", None) else idx
    raw.index = pd.DatetimeIndex(idx).normalize()
    return raw[["open", "high", "low", "close"]]


def check_leg(px: float, row) -> tuple[bool, float]:
    """is px within [low,high]? and % diff vs that day's close."""
    if row is None or pd.isna(px):
        return False, float("nan")
    lo, hi, cl = float(row["low"]), float(row["high"]), float(row["close"])
    within = (lo * (1 - TOL)) <= px <= (hi * (1 + TOL))
    pct_vs_close = (px / cl - 1) if cl else float("nan")
    return within, pct_vs_close


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/trades/daily_all_trades.csv")
    ap.add_argument("--adjust", choices=["auto", "raw"], default="auto",
                    help="auto = split/div-adjusted yfinance; raw = unadjusted")
    ap.add_argument("--show-worst", type=int, default=12)
    args = ap.parse_args()

    trades = pd.read_csv(args.csv)
    for c in ("entry_date", "exit_date"):
        trades[c] = pd.to_datetime(trades[c]).dt.normalize()
    print(f"\nVerifying {len(trades)} trades from {args.csv}")
    print(f"reference: yfinance daily ({'ADJUSTED' if args.adjust=='auto' else 'RAW/unadjusted'})\n")

    rows, per_ticker = [], {}
    for t, sub in trades.groupby("ticker"):
        try:
            ohlc = fetch_ohlc(t, sub["entry_date"].min(), sub["exit_date"].max(), args.adjust)
        except Exception as e:
            print(f"  [skip] {t}: {e}")
            continue
        if ohlc.empty:
            print(f"  [skip] {t}: no yfinance data")
            continue
        ein = eout = n = 0
        diffs = []
        for _, tr in sub.iterrows():
            er = ohlc.loc[tr["entry_date"]] if tr["entry_date"] in ohlc.index else None
            xr = ohlc.loc[tr["exit_date"]] if tr["exit_date"] in ohlc.index else None
            ew, ed = check_leg(tr.get("entry_px"), er)
            xw, xd = check_leg(tr.get("exit_px"), xr)
            ein += int(ew); eout += int(xw); n += 1
            if not np.isnan(ed):
                diffs.append(abs(ed))
            rows.append({"ticker": t, "entry_date": str(tr["entry_date"].date()),
                         "entry_px": tr.get("entry_px"),
                         "entry_in_range": ew, "entry_pct_vs_close": round(ed, 4) if not np.isnan(ed) else None,
                         "exit_in_range": xw})
        per_ticker[t] = {
            "trades": n,
            "entry_in_range_pct": round(ein / n, 3) if n else 0,
            "exit_in_range_pct": round(eout / n, 3) if n else 0,
            "mean_abs_pct_vs_close": round(float(np.mean(diffs)), 4) if diffs else None,
        }

    hdr = f"{'ticker':8s} {'trades':>6s} {'entry_in_rng':>13s} {'exit_in_rng':>12s} {'mean|%vs close|':>17s}"
    print(hdr); print("-" * len(hdr))
    for t, r in sorted(per_ticker.items()):
        d = r["mean_abs_pct_vs_close"]
        print(f"{t:8s} {r['trades']:>6} {r['entry_in_range_pct']:>12.1%} "
              f"{r['exit_in_range_pct']:>11.1%} "
              f"{(f'{d:.2%}' if d is not None else 'n/a'):>17s}")

    rdf = pd.DataFrame(rows)
    tot = len(rdf)
    ein_pct = rdf["entry_in_range"].mean() if tot else 0
    eout_pct = rdf["exit_in_range"].mean() if tot else 0
    print("-" * len(hdr))
    print(f"OVERALL: {ein_pct:.1%} of entries and {eout_pct:.1%} of exits fall "
          f"within the day's low-high range.")

    # worst mismatches by |Δ vs close|
    worst = rdf.dropna(subset=["entry_pct_vs_close"]).reindex(
        rdf["entry_pct_vs_close"].abs().sort_values(ascending=False).index).head(args.show_worst)
    if len(worst):
        print(f"\nWorst {len(worst)} entry mismatches (backtest px vs yfinance close):")
        print(f"  {'ticker':8s} {'date':12s} {'bt_px':>10s} {'%vs close':>10s} {'in_range':>9s}")
        for _, w in worst.iterrows():
            print(f"  {w['ticker']:8s} {w['entry_date']:12s} {w['entry_px']:>10.2f} "
                  f"{w['entry_pct_vs_close']:>10.1%} {str(bool(w['entry_in_range'])):>9s}")

    out = {"csv": args.csv, "adjust": args.adjust, "total_trades": tot,
           "entry_in_range_pct": round(float(ein_pct), 4),
           "exit_in_range_pct": round(float(eout_pct), 4),
           "per_ticker": per_ticker}
    Path("results/trade_verification.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("\nWrote results/trade_verification.json")
    if ein_pct < 0.9:
        print("\n[!] Many entries fall outside the day's range. If --adjust auto, "
              "try --adjust raw: if that matches better, your backtest parquet is "
              "UNADJUSTED and carries split risk.")


if __name__ == "__main__":
    main()
