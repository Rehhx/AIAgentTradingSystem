"""
runners/traded_stocks.py
------------------------
Which stocks actually got traded by each daily strategy, and which are COMMON
across all strategies. Reads per-ticker trade counts + $PnL straight from the
portfolio backtester (agents/daily_strategies.py).

Usage:
  python runners\\traded_stocks.py                       # recommended 6-name universe
  python runners\\traded_stocks.py --universe all        # all 20 parquet tickers
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.daily_strategies import STRATEGIES_DAILY, DEFAULT_UNIVERSE, backtest_book


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="SPY,QQQ,GLD,MSFT,JPM,GOOGL")
    args = ap.parse_args()

    if args.universe.strip().lower() == "all":
        from data.loader import DATA_DIR
        universe = sorted(p.stem for p in Path(DATA_DIR).glob("*.parquet"))
    else:
        universe = [t.strip().upper() for t in args.universe.split(",") if t.strip()]

    print(f"\nUniverse ({len(universe)}): {', '.join(universe)}")
    print("Trades per stock, per strategy (full history, 6bps costs, $100k/sleeve)\n")

    strat_names = list(STRATEGIES_DAILY)               # rsi2_meanrev, donchian, trend_5020
    per_ticker = {name: backtest_book(fn, universe, label=name)["per_ticker"]
                  for name, fn in STRATEGIES_DAILY.items()}

    # matrix: ticker -> {strategy: trades}
    hdr = f"{'ticker':7s} " + " ".join(f"{s:>14s}" for s in strat_names) + f"{'traded_by':>11s}"
    print(hdr)
    print("-" * len(hdr))
    traded_sets = {s: set() for s in strat_names}
    for t in universe:
        cells, n_strats = [], 0
        for s in strat_names:
            pm = per_ticker[s].get(t, {})
            tr = pm.get("total_trades", 0)
            pnl = pm.get("pnl_dollars", 0.0)
            if tr > 0:
                traded_sets[s].add(t)
                n_strats += 1
            cells.append(f"{tr:>4d}/{pnl:>8,.0f}")
        flag = "ALL 3" if n_strats == 3 else f"{n_strats}/3"
        print(f"{t:7s} " + " ".join(f"{c:>14s}" for c in cells) + f"{flag:>11s}")
    print("  (cell = trades / standalone $PnL on $100k)\n")

    common = set(universe)
    for s in strat_names:
        common &= traded_sets[s]
    for s in strat_names:
        print(f"  {s:14s} traded {len(traded_sets[s])} names: "
              f"{', '.join(sorted(traded_sets[s]))}")
    print(f"\n  COMMON to all 3 strategies ({len(common)}): "
          f"{', '.join(sorted(common)) if common else '(none)'}")


if __name__ == "__main__":
    main()
