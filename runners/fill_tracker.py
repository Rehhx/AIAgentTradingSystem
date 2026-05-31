r"""
runners/fill_tracker.py
-----------------------
Measures REAL execution quality: for each filled order it compares the fill price
to the prevailing market price (the 1-minute bar at fill time) and logs the
slippage in basis points. This is the tool that turns "paper overstates fills"
from an opinion into a measured number, per ticker.

  - On a PAPER account it should read ~0 bps (paper fills at the quote = the
    "perfect liquidity" you correctly distrust).
  - On a LIVE account it shows the real spread/impact cost you actually pay.

Slippage convention: positive bps = cost (paid above mid on a buy, or sold below
mid). Appends to results/fill_quality.csv (deduped by order id).

  python runners\fill_tracker.py --account 1 --days 7
"""
import argparse
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.execution_agent import ExecutionAgent
from config import ALPACA_API_KEY_2, ALPACA_API_SECRET_2

OUT = Path(__file__).parent.parent / "results" / "fill_quality.csv"


def ref_price(symbol, when, key, secret):
    """prevailing price near `when` (1-min bar close); None if unavailable."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        cli = StockHistoricalDataClient(key, secret)
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute,
                               start=when - timedelta(minutes=10), end=when + timedelta(minutes=10))
        bars = cli.get_stock_bars(req).data.get(symbol, [])
        if not bars:
            return None
        nearest = min(bars, key=lambda b: abs((b.timestamp - when).total_seconds()))
        return float(nearest.close)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", type=int, default=1, choices=[1, 2])
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()

    if args.account == 2:
        key, secret = ALPACA_API_KEY_2, ALPACA_API_SECRET_2
        agent = ExecutionAgent(api_key=key, api_secret=secret)
    else:
        from config import ALPACA_API_KEY, ALPACA_API_SECRET
        key, secret = ALPACA_API_KEY, ALPACA_API_SECRET
        agent = ExecutionAgent()
    if agent.simulated or agent.client is None:
        print("No live Alpaca account for this slot — nothing to track.")
        return

    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    after = datetime.now(timezone.utc) - timedelta(days=args.days)
    orders = agent.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, after=after, limit=500))
    fills = [o for o in orders if getattr(o, "filled_at", None) and getattr(o, "filled_avg_price", None)]
    print(f"Account #{args.account}: {len(fills)} filled order(s) in the last {args.days} days.\n")
    if not fills:
        print("  (no fills yet — once live orders execute, re-run to measure slippage)")
        return

    seen = set()
    if OUT.exists():
        seen = {r["order_id"] for r in csv.DictReader(open(OUT))}
    rows, new = [], []
    for o in fills:
        oid = str(o.id)
        fp = float(o.filled_avg_price)
        ref = ref_price(o.symbol, o.filled_at, key, secret) if "/" not in o.symbol else None
        sign = 1 if "buy" in str(o.side).lower() else -1
        slip = round(sign * (fp - ref) / ref * 10000, 1) if ref else None
        rec = dict(order_id=oid, date=str(o.filled_at)[:10], symbol=o.symbol,
                   side=str(o.side).split(".")[-1], qty=float(o.filled_qty),
                   fill_px=round(fp, 4), ref_px=round(ref, 4) if ref else "",
                   slippage_bps=slip if slip is not None else "")
        rows.append(rec)
        if oid not in seen:
            new.append(rec)

    OUT.parent.mkdir(exist_ok=True)
    write_header = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(new)

    measured = [r for r in rows if r["slippage_bps"] != ""]
    print(f"  {'symbol':8s} {'side':5s} {'fill':>10s} {'ref':>10s} {'slip(bps)':>10s}")
    for r in measured[:40]:
        print(f"  {r['symbol']:8s} {r['side']:5s} {r['fill_px']:>10.2f} {r['ref_px']:>10.2f} {r['slippage_bps']:>10.1f}")
    if measured:
        avg = sum(r["slippage_bps"] for r in measured) / len(measured)
        cost = [r["slippage_bps"] for r in measured]
        print(f"\n  AVG slippage: {avg:+.1f} bps  ({len(measured)} fills priced) | "
              f"worst {max(cost):+.1f} | best {min(cost):+.1f}")
        print("  (~0 on paper = perfect liquidity; positive on live = real cost you pay)")
    print(f"\n  appended {len(new)} new fill(s) to {OUT.name}")


if __name__ == "__main__":
    main()
