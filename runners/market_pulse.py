"""
runners/market_pulse.py
-----------------------
one-shot snapshot of "what's the market doing right now" using free APIs.
pulls FRED macro indicators + Finnhub quotes/news/sentiment for the
user-specified ticker universe, then writes a single JSON to
results/market_pulse.json and prints a human-readable summary.

usage:
    python runners/market_pulse.py                    # SPY + QQQ defaults
    python runners/market_pulse.py SPY QQQ NVDA TSLA  # custom symbols
    python runners/market_pulse.py --all              # every ticker in DATA_DIR

requires FINNHUB_API_KEY and FRED_API_KEY in .env. missing keys produce
empty sections rather than failing — partial snapshots are still useful.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from data.multi_source import market_pulse
from config import FINNHUB_API_KEY, FRED_API_KEY

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*", default=None,
                    help="ticker symbols to fetch quotes/news for (default: SPY QQQ)")
    ap.add_argument("--all", action="store_true",
                    help="use every ticker parquet in DATA_DIR")
    ap.add_argument("--out", default="results/market_pulse.json",
                    help="output JSON path")
    args = ap.parse_args()

    if args.all:
        from data.loader import available_tickers
        symbols = available_tickers()
    elif args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        symbols = ["SPY", "QQQ"]

    print(f"\n{'='*72}\n  MARKET PULSE  |  {len(symbols)} symbols\n{'='*72}")
    if not FINNHUB_API_KEY:
        print("  (FINNHUB_API_KEY not set — equity quotes/news/sentiment will be empty)")
    if not FRED_API_KEY:
        print("  (FRED_API_KEY not set — macro section will be empty)")
    print()

    snap = market_pulse(symbols)

    # --- macro ---
    print("--- macro snapshot (FRED) ---")
    macro = snap.get("macro") or {}
    for nickname, info in macro.items():
        val  = info.get("value")
        date = info.get("date")
        if val is not None:
            print(f"  {nickname:<22} {val:>10.4f}  ({date})")
        else:
            print(f"  {nickname:<22}      n/a  (no data / missing key)")

    # --- equities ---
    print("\n--- equities ---")
    equities = snap.get("equities") or {}
    for sym, e in equities.items():
        q    = e.get("quote") or {}
        sent = e.get("sentiment") or {}
        if q.get("price") is not None:
            chg_pct = q.get("change_pct") or 0
            print(f"  {sym:<6} ${q['price']:>9.2f}  "
                  f"{chg_pct:>+7.2f}%  "
                  f"O={q.get('day_open'):>7.2f} H={q.get('day_high'):>7.2f} L={q.get('day_low'):>7.2f}")
        else:
            print(f"  {sym:<6} (no quote data)")

        if sent.get("buzz") is not None:
            bull = sent.get("bullish_pct")
            bear = sent.get("bearish_pct")
            bull_str = f"{bull*100:.1f}%" if isinstance(bull, (int, float)) else "?"
            bear_str = f"{bear*100:.1f}%" if isinstance(bear, (int, float)) else "?"
            print(f"         sentiment: bullish={bull_str}  bearish={bear_str}  "
                  f"buzz={sent.get('buzz'):.2f}  "
                  f"articles_week={sent.get('articles_in_last_week')}")

        news = e.get("news") or []
        if news:
            print(f"         top headlines:")
            for n in news[:3]:
                hd = (n.get("headline") or "")[:90]
                print(f"           - {hd}")

    # --- earnings ---
    earnings = snap.get("earnings_calendar") or []
    relevant = [e for e in earnings if e.get("symbol") in symbols]
    if relevant:
        print(f"\n--- upcoming earnings (next 7 days) ---")
        for e in relevant[:10]:
            print(f"  {e['symbol']:<6} {e['date']}  EPS est={e.get('eps_est')}  hour={e.get('hour')}")

    # --- save ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  full snapshot saved to {out_path}")


if __name__ == "__main__":
    main()
