"""
runners/cleanup_strategies.py
-----------------------------
Record every strategy the system has touched into the persistent ledger
(agents/strategy_ledger.py), then DELETE the unprofitable generated modules in
strategies/. The 3 daily winners live in agents/daily_strategies.py and are
untouched; the dead code is removed but its NAME + verdict is retained in the
ledger so the AI agents know it's been tried.

Run with --apply to actually delete (default is a dry-run preview).

Usage:
  python runners\\cleanup_strategies.py            # preview
  python runners\\cleanup_strategies.py --apply     # record ledger + delete files
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.strategy_ledger import record, load_ledger, LEDGER_PATH

STRAT_DIR = Path("strategies")

# the strategies we KEEP (daily books, in agents/daily_strategies.py — not files here)
KEEP = {
    "rsi2_meanrev": ("deployed", 0.95, "tuned RSI-2, passes risk, 109 trades/yr"),
    "blended_book": ("deployed", 1.09, "equal-weight RSI-2+Donchian+trend, passes risk"),
    "donchian":     ("kept", 0.51, "daily breakout; retained, fails risk standalone"),
    "trend_5020":   ("kept", 1.07, "daily 50/200 trend; retained, 15% CAGR but -28% DD"),
}


def best_sharpe_map() -> dict:
    """pull the best/last known Sharpe per strategy from champions.json."""
    m = {}
    cf = Path("results/champions.json")
    if cf.exists():
        c = json.loads(cf.read_text())
        for bucket in ("champions", "retired_this_run"):
            for row in c.get(bucket, []):
                n = row.get("name")
                if n and n not in m:
                    m[n] = row.get("sharpe")
    return m


def derive_name(path: Path) -> str:
    parts = path.stem.split("_", 1)
    return parts[1] if len(parts) == 2 else path.stem


def kind_of(path: Path) -> str:
    s = path.name
    if s.startswith("ml"):
        return "ml"
    if s.startswith("opt"):
        return "options"
    return "rule"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually delete files")
    args = ap.parse_args()

    sharpes = best_sharpe_map()

    # 1) record the keepers
    for name, (status, sh, note) in KEEP.items():
        record(name, status=status, sharpe=sh, kind="daily", note=note,
               source_file="agents/daily_strategies.py")

    # 2) record built-ins as dead (confirmed in LESSONS.md) — code stays in engine
    for n in ["orb", "momentum", "bb_squeeze", "rsi_reversion", "vwap_reversion",
              "ema_crossover", "noise_area_breakout", "qqq_spy_dispersion",
              "bb_band_touch_revert", "half_hour_continuation", "trend_ride"]:
        record(n, status="dead", sharpe=sharpes.get(n), kind="builtin",
               note="1m intraday; cost drag — dead per LESSONS.md (code kept in engine)")

    # 3) the generated files to remove
    files = sorted(p for p in STRAT_DIR.glob("*.py")
                   if not p.name.startswith("_") and p.stem != "__init__")
    print(f"\n{'ACTION':8s} {'name':40s} {'kind':8s} {'sharpe':>8s}")
    print("-" * 70)
    removed = 0
    for p in files:
        name = derive_name(p)
        if name in KEEP:                 # safety: never delete a keeper
            continue
        sh = sharpes.get(name)
        record(name, status="dead", sharpe=sh, kind=kind_of(p),
               note="removed from strategies/ — unprofitable", source_file=p.name)
        action = "DELETE" if args.apply else "(would)"
        print(f"{action:8s} {name:40s} {kind_of(p):8s} "
              f"{(f'{sh:.2f}' if isinstance(sh,(int,float)) else '?'):>8s}")
        if args.apply:
            p.unlink()
        removed += 1

    led = load_ledger()
    print(f"\nLedger now has {len(led)} strategies recorded at {LEDGER_PATH}")
    print(f"{'DELETED' if args.apply else 'WOULD DELETE'} {removed} files from strategies/")
    if not args.apply:
        print("\nDry-run only. Re-run with --apply to record the ledger and delete.")


if __name__ == "__main__":
    main()
