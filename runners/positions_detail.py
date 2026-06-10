"""
runners/positions_detail.py
----------------------------
Read-only position inspector for one account. For OPTIONS it parses the OCC
symbol, fetches the underlying spot, and compares each contract's mark to its
INTRINSIC value -- an in-the-money option cannot really be worth less than
intrinsic, so a mark below intrinsic flags a stale/bad paper-feed quote rather
than a real loss. No orders are placed.

  python runners/positions_detail.py --account 3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings

warnings.filterwarnings("ignore")

import argparse
import re

from config import alpaca_keys, ALPACA_PAPER

OCC = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")


def occ_parse(sym: str):
    m = OCC.match(sym)
    if not m:
        return None
    root, ymd, cp, strike = m.groups()
    return {"root": root, "exp": f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:6]}",
            "type": "call" if cp == "C" else "put", "strike": int(strike) / 1000.0}


def spot_prices(roots):
    import yfinance as yf
    out = {}
    for r in roots:
        try:
            out[r] = float(yf.Ticker(r).history(period="1d")["Close"].iloc[-1])
        except Exception:
            out[r] = None
    return out


def main(account: int):
    key, secret = alpaca_keys(account)
    if not key or not secret:
        print(f"account {account}: no keys configured")
        return
    from alpaca.trading.client import TradingClient
    client = TradingClient(api_key=key, secret_key=secret, paper=ALPACA_PAPER)
    acct = client.get_account()
    eq, last = float(acct.equity), float(acct.last_equity)
    print(f"ACCOUNT {account}  equity ${eq:,.2f} | prior close ${last:,.2f} "
          f"| today {eq/last-1:+.2%} | cash ${float(acct.cash):,.2f}\n")

    positions = client.get_all_positions()
    if not positions:
        print("  (no open positions)")
        return

    roots = {occ_parse(p.symbol)["root"] for p in positions if occ_parse(p.symbol)}
    spots = spot_prices(roots) if roots else {}

    print(f"  {'symbol':22s} {'qty':>5s} {'entry':>9s} {'mark':>9s} {'mkt val':>11s} "
          f"{'unrl P/L':>10s} {'%':>7s}  note")
    print("  " + "-" * 95)
    for p in positions:
        sym = p.symbol
        qty, entry = float(p.qty), float(p.avg_entry_price)
        mark = float(p.current_price); mv = float(p.market_value)
        upl, uplpc = float(p.unrealized_pl), float(p.unrealized_plpc)
        note = ""
        info = occ_parse(sym)
        if info:
            spot = spots.get(info["root"])
            if spot is not None:
                intrinsic = (max(0.0, spot - info["strike"]) if info["type"] == "call"
                             else max(0.0, info["strike"] - spot))
                note = (f"{info['root']} {info['type']} K{info['strike']:.0f} exp{info['exp']} "
                        f"| spot {spot:.2f} intrinsic {intrinsic:.2f}/sh")
                if mark < intrinsic * 0.98:
                    note += "  <<< MARK BELOW INTRINSIC (stale/bad quote)"
        print(f"  {sym:22s} {qty:>5.0f} {entry:>9.2f} {mark:>9.2f} {mv:>11,.0f} "
              f"{upl:>+10,.0f} {uplpc:>+7.1%}  {note}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", type=int, default=3)
    a = ap.parse_args()
    main(a.account)
