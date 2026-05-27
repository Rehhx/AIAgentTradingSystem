"""
agents/strategy_ticker_grid.py
------------------------------
runs every active strategy against every ticker, with and without the
embedding-based regime quality gate. reuses the per-ticker parquet caches
created by embedding_gate_runner so no OpenAI calls are needed.

output: results/strategy_ticker_grid.json — a (strategies x tickers) grid of
{baseline_sharpe, gated_sharpe, delta_sharpe, baseline_trades, gated_trades, ...}.

usage:
    python agents/strategy_ticker_grid.py
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# load .env in case the cache for a ticker doesn't exist yet and the embedding
# precompute needs to fire — won't actually be needed if all caches are present.
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from data.loader import load_ticker, DATA_DIR
from agents.backtesting_agent import (
    ATR_STOP_MULT,
    STRATEGIES,
    STRATEGY_REGIME_AFFINITY,
    precompute_regime_quality,
    regime_label_series,
    run_backtest,
)

log = logging.getLogger("strategy_ticker_grid")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

TICKERS    = ["SPY", "QQQ", "TSLA", "NVDA", "AAPL"]
CACHE_DIR  = Path("vector_stores/.cache")
RESULT_OUT = Path("results/strategy_ticker_grid.json")


def run_cell(strategy_name: str, ticker: str) -> dict:
    """one strategy x ticker cell: baseline + gated backtest."""
    fn, default_params = STRATEGIES[strategy_name][0], STRATEGIES[strategy_name][1]
    stop_mult = default_params.get("stop_atr_mult", ATR_STOP_MULT)
    allowed   = STRATEGY_REGIME_AFFINITY.get(strategy_name)

    df = load_ticker(
        ticker, data_dir=DATA_DIR,
        start="2022-01-01", end="2025-01-01", session="regular",
    )

    signal   = fn(df, default_params)
    regime_s = regime_label_series(df)

    # baseline = heuristic regime gate only
    baseline = run_backtest(
        df, signal,
        stop_atr_mult=stop_mult,
        regime_series=regime_s,
        allowed_regimes=allowed,
    )

    # gated = heuristic + embedding-based quality gate (uses cached parquet)
    quality_df       = precompute_regime_quality(ticker, df, step=60)
    quality_aligned  = quality_df.reindex(df.index, method="ffill")
    gated = run_backtest(
        df, signal,
        stop_atr_mult=stop_mult,
        regime_series=regime_s,
        allowed_regimes=allowed,
        quality_series=quality_aligned,
        quality_min_pct_pos=0.55,
    )

    return {
        "baseline_sharpe":  baseline["sharpe"],
        "baseline_trades":  baseline["total_trades"],
        "baseline_wr":      baseline["win_rate"],
        "baseline_dd":      baseline["max_drawdown"],
        "gated_sharpe":     gated["sharpe"],
        "gated_trades":     gated["total_trades"],
        "gated_wr":         gated["win_rate"],
        "gated_dd":         gated["max_drawdown"],
        "delta_sharpe":     round(gated["sharpe"] - baseline["sharpe"], 4),
    }


def main():
    active = [name for name, (_, p) in STRATEGIES.items() if p.get("active", True)]
    print(f"active strategies: {active}")
    print(f"tickers:           {TICKERS}")
    print(f"grid size:         {len(active)} x {len(TICKERS)} = {len(active) * len(TICKERS)} cells")
    print()

    # check which caches exist up front
    missing = [t for t in TICKERS if not (CACHE_DIR / f"{t}_regime_quality_step60.parquet").exists()]
    if missing:
        print(f"WARNING: missing regime quality cache for: {missing}")
        print(f"those cells will trigger fresh OpenAI precomputes (~24 min, ~$0.05 each)")
        print()

    grid = {}
    for strategy_name in active:
        grid[strategy_name] = {}
        for ticker in TICKERS:
            print(f"  {strategy_name:<16} x {ticker:<6} ... ", end="", flush=True)
            try:
                cell = run_cell(strategy_name, ticker)
                grid[strategy_name][ticker] = cell
                print(
                    f"baseline sharpe={cell['baseline_sharpe']:>7.3f} ({cell['baseline_trades']:>4}t)  "
                    f"-> gated sharpe={cell['gated_sharpe']:>7.3f} ({cell['gated_trades']:>4}t)  "
                    f"delta={cell['delta_sharpe']:+.3f}"
                )
            except FileNotFoundError as e:
                print(f"DATA MISSING ({e})")
            except Exception as e:
                log.exception(f"cell failed: {e}")
                print(f"FAILED ({e})")

    # render summary table
    print()
    print("=" * 95)
    print("  gated sharpe grid (positive in **bold** when reading)")
    print("=" * 95)
    header = f"  {'strategy':<16}" + "".join(f" | {t:>11}" for t in TICKERS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for s in active:
        row = f"  {s:<16}"
        for t in TICKERS:
            cell = grid.get(s, {}).get(t)
            if cell is None:
                row += " | " + "-" * 11
            else:
                row += f" | {cell['gated_sharpe']:>7.3f} ({cell['gated_trades']:>2})"
        print(row)
    print()

    # count near-positive cells (sharpe > -0.5 with at least 20 trades)
    candidates = []
    for s, ticker_rows in grid.items():
        for t, cell in ticker_rows.items():
            if cell["gated_sharpe"] > -0.5 and cell["gated_trades"] >= 20:
                candidates.append((s, t, cell["gated_sharpe"], cell["gated_trades"]))
    candidates.sort(key=lambda x: x[2], reverse=True)
    print(f"near-tradeable cells (gated sharpe > -0.5, trades >= 20):")
    if candidates:
        for s, t, sh, tr in candidates:
            print(f"  {s:<16} x {t:<6} sharpe={sh:>6.3f} trades={tr}")
    else:
        print("  none")

    RESULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_OUT, "w") as f:
        json.dump({
            "grid":       grid,
            "candidates": candidates,
            "run_at":     datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)
    print(f"\nsaved to {RESULT_OUT}")


if __name__ == "__main__":
    main()
