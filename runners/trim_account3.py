"""
runners/trim_account3.py
------------------------
Trim the Account 3 LEAPS book back to the 1.0x defined-risk target: keep N
contracts per underlying (default 1) and sell the rest, so premium-at-risk drops
from ~99% of the account to ~25-30% with a big T-bill cash buffer.

Options MARKET orders are only accepted during regular hours (9:30-16:00 ET), so
this refuses to fire when the market is closed. Dry-run by default.

  python runners\\trim_account3.py              # dry-run: show the plan
  python runners\\trim_account3.py --live       # execute (must be during market hours)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.execution_agent import ExecutionAgent
from config import alpaca_keys


def _option_positions(client):
    out = []
    for p in client.get_all_positions():
        is_opt = "option" in str(getattr(p, "asset_class", "")).lower() or len(p.symbol) > 15
        if is_opt:
            out.append(p)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=1, help="contracts to keep per underlying")
    ap.add_argument("--live", action="store_true", help="actually submit the sells")
    args = ap.parse_args()

    k, s = alpaca_keys(3)
    a = ExecutionAgent(api_key=k, api_secret=s)
    if a.simulated:
        print("Account 3 is simulated (no ALPACA_*_3 keys) — nothing to do.")
        return

    clk = a.client.get_clock()
    print(f"Account 3 LEAPS trim | keep {args.keep}/underlying | market open: {clk.is_open} (now {clk.timestamp})")
    opts = _option_positions(a.client)
    if not opts:
        print("  no option positions found.")
        return

    plan = []
    for p in opts:
        held = int(float(p.qty))
        sell = max(0, held - args.keep)
        mv = float(p.market_value)
        print(f"  {p.symbol:22s} hold {held} (${mv:,.0f})  ->  sell {sell}, keep {args.keep}")
        if sell > 0:
            plan.append((p.symbol, sell))

    if not plan:
        print("  already at/below target — nothing to sell.")
        return
    if not args.live:
        print("\nDRY-RUN — re-run with --live DURING market hours (9:30-16:00 ET) to execute.")
        return
    if not clk.is_open:
        print("\nMarket CLOSED — option market orders need regular hours. Re-run after the 9:30 ET open.")
        return

    print("\nSubmitting…")
    try:
        a.client.cancel_orders()
    except Exception:
        pass
    for sym, qty in plan:
        res = a.submit_option(sym, qty, "sell")
        f = res.get("fill", {})
        print(f"  SELL {qty}x {sym} -> {f.get('status', res.get('reason'))}")


if __name__ == "__main__":
    main()
