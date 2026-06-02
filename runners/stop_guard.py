"""
runners/stop_guard.py
---------------------
Software stop-loss enforcer. It LIQUIDATES positions that have breached their
stop, covering the window Alpaca's NATIVE stops can't: pre/post market and
overnight (native stop / trailing-stop orders only trigger 9:30-16:00 ET).

For each LONG position it tracks a high-water mark (results/position_highs.json)
and sells if price falls below EITHER:
  - trailing level = high_water * (1 - trail_pct)     (rides winners, locks gains)
  - hard floor     = avg_entry  * (1 - hard_pct)      (catastrophe backstop)

How it sells:
  - regular session  -> market sell (full position)
  - extended session -> whole-share DAY limit sell with extended_hours=True
  - no session open  -> flags the breach to act when the next session opens

It cancels the symbol's existing (native) stop first so it never double-sells.
Run standalone on a schedule, or call enforce_stops() from the dashboard checker.

  python runners\\stop_guard.py --account 1 --dry-run      # report breaches only
  python runners\\stop_guard.py --account 1                # liquidate breaches LIVE
"""
import json
import sys
from datetime import time as dtime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.execution_agent import ExecutionAgent

STATE = Path(__file__).parent.parent / "results" / "position_highs.json"


def _load():
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {}


def _save(d):
    try:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(d, indent=2))
    except Exception:
        pass


def session(agent) -> str:
    """'regular' | 'extended' | 'closed' from the Alpaca clock (best-effort)."""
    try:
        clk = agent.client.get_clock()
        if clk.is_open:
            return "regular"
        now = clk.timestamp                       # tz-aware ET
        t = now.time()
        weekday = now.weekday() < 5
        pre = dtime(4, 0) <= t < dtime(9, 30)
        post = dtime(16, 0) <= t < dtime(20, 0)
        return "extended" if (weekday and (pre or post)) else "closed"
    except Exception:
        return "closed"


def _liquidate(agent, sym, qty, px, sess, buf):
    # cancel this symbol's open orders (native stop) first so we don't double-sell
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        alp = sym.replace("-", ".") if ("-" in sym and len(sym.rsplit("-", 1)[-1]) == 1) else sym
        for o in agent.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN)):
            if str(o.symbol) == alp:
                agent.client.cancel_order_by_id(str(o.id))
    except Exception:
        pass
    if sess == "extended":                        # ext hours: whole-share DAY limit
        sig = {"ticker": sym, "side": "sell", "qty": abs(int(qty)),
               "order_type": "limit", "time_in_force": "day",
               "extended_hours": True, "limit_price": round(px * (1 - buf), 2)}
    else:                                         # regular: market, full (fractional ok on a long sell)
        sig = {"ticker": sym, "side": "sell", "qty": abs(qty),
               "order_type": "market", "time_in_force": "day"}
    res = agent.run({"payload": {"signal": sig}, "strategy_id": "stop_guard"})
    return res.get("fill", {}).get("status", res.get("reason"))


def enforce_stops(agent, trail_pct=20.0, hard_pct=15.0, ext_buffer=0.005,
                  sanity_pct=40.0, do_liquidate=True):
    """Check every long position; liquidate (or just flag) the ones that breached.
    SAFETY: a 'breach' worse than sanity_pct below entry is treated as suspect data
    (split / feed glitch — e.g. a large-cap showing -92%) and is NEVER auto-sold;
    it's flagged SUSPECT for manual review. Returns human-readable log lines."""
    if agent.simulated or agent.client is None:
        return ["stop guard: simulated account — inactive"]
    sess = session(agent)
    highs = _load()
    log, held = [], set()
    for p in agent.get_positions():
        sym, qty = p["symbol"], p["qty"]
        if qty <= 0:                              # longs only (shorts use MF vol-targeting)
            continue
        held.add(sym)
        px = p["market_value"] / qty if qty else 0.0
        entry = p["avg_entry_price"]
        hw = max(highs.get(sym, entry), px, entry)
        highs[sym] = hw
        trail_level = hw * (1 - trail_pct / 100.0)
        hard_level = entry * (1 - hard_pct / 100.0)
        sanity_floor = entry * (1 - sanity_pct / 100.0)
        if px <= trail_level or px <= hard_level:
            why = "trailing" if px <= trail_level else "hard"
            if px <= sanity_floor:                # implausible move -> likely bad data, DO NOT sell
                drop = (px / entry - 1) if entry else 0
                log.append(f"SUSPECT {sym} {px:.2f} is {drop:+.0%} vs entry {entry:.2f} — likely a "
                           f"split/data glitch, NOT liquidating. Review manually.")
            elif not do_liquidate:
                log.append(f"BREACH {sym} {px:.2f} <= {why} stop (entry {entry:.2f}, hi {hw:.2f}) — NOT selling (dry-run)")
            elif sess == "closed":
                log.append(f"BREACH {sym} {px:.2f} ({why}) — no session open, will act next session")
            else:
                status = _liquidate(agent, sym, qty, px, sess, ext_buffer)
                log.append(f"LIQUIDATE {sym} {qty:g} @ {px:.2f} ({why} stop, {sess}) -> {status}")
                highs.pop(sym, None)
    highs = {k: v for k, v in highs.items() if k in held}     # prune closed names
    _save(highs)
    if not any((k in l for l in log for k in ("BREACH", "LIQUIDATE", "SUSPECT"))):
        log.append(f"stop guard: {len(held)} long(s) checked, none breached ({sess} session)")
    return log


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--trail-pct", type=float, default=20.0)
    ap.add_argument("--hard-pct", type=float, default=15.0)
    ap.add_argument("--sanity-pct", type=float, default=40.0,
                    help="drops worse than this vs entry are treated as bad data (not sold)")
    ap.add_argument("--dry-run", action="store_true", help="report breaches, do not sell")
    args = ap.parse_args()
    from config import alpaca_keys
    k, s = alpaca_keys(args.account)
    agent = ExecutionAgent(api_key=k, api_secret=s)
    print(f"Stop guard | account #{args.account} | trail {args.trail_pct}% | hard {args.hard_pct}% "
          f"| {'DRY-RUN' if args.dry_run else 'LIVE'}")
    for line in enforce_stops(agent, args.trail_pct, args.hard_pct,
                              sanity_pct=args.sanity_pct, do_liquidate=not args.dry_run):
        print("  ", line)


if __name__ == "__main__":
    main()
