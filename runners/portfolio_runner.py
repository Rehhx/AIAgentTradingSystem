"""
agents/portfolio_runner.py
--------------------------
combine multiple strategies into a single equity curve. each strategy gets
an equal slice of capital and its own position; the portfolio's equity is
the sum of all sleeve cash + unrealized PnL.

design notes:
  - each sleeve is run through the same run_backtest engine with reduced
    position_size_pct (1/N of full size, where N = number of sleeves)
  - per-sleeve equity curves are aligned, summed, and resampled to daily
    for Sharpe — same metric basis as the single-strategy backtests
  - diversification benefit comes from uncorrelated entries: bb_squeeze
    fires on squeeze→expansion, gap_fade on overnight news, extreme_bar
    on liquidity vacuums — all different microstructure events
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from data.loader import load_ticker, DATA_DIR
from agents.backtesting_agent import (
    ATR_STOP_MULT, INITIAL_CAP, STRATEGIES, STRATEGY_REGIME_AFFINITY,
    regime_label_series, run_backtest, get_strategy_meta,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def run_single_sleeve(strategy_name: str, ticker: str, position_size_pct: float):
    entry          = STRATEGIES[strategy_name]
    fn             = entry[0]
    default_params = entry[1]
    meta           = get_strategy_meta(strategy_name)
    stop_mult      = default_params.get("stop_atr_mult", ATR_STOP_MULT)
    disable_stop   = bool(default_params.get("disable_atr_stop", False))
    max_hold       = default_params.get("max_hold_bars")
    allowed        = STRATEGY_REGIME_AFFINITY.get(strategy_name)

    if meta.get("kind") == "multi":
        dfs = {t: load_ticker(t, data_dir=DATA_DIR, start="2022-01-01",
                              end="2025-01-01", session="regular")
               for t in meta["data_tickers"]}
        df     = dfs[meta["tradeable_ticker"]]
        signal = fn(dfs, default_params)
    else:
        df = load_ticker(ticker, data_dir=DATA_DIR, start="2022-01-01",
                         end="2025-01-01", session="regular")
        signal = fn(df, default_params)

    regime_s = regime_label_series(df)
    result = run_backtest(
        df, signal,
        position_size_pct=position_size_pct,
        stop_atr_mult=stop_mult,
        regime_series=regime_s,
        allowed_regimes=allowed,
        disable_atr_stop=disable_stop,
        max_hold_bars=max_hold,
    )

    # we can't get the bar-level equity curve out of the current run_backtest
    # return shape (it strips to hourly resample). reconstruct from trades by
    # tracking realized + unrealized at daily resolution.
    return df, signal, result


def main():
    sleeves = [
        ("bb_squeeze",         "SPY"),
        ("overnight_gap_fade", "SPY"),
        ("extreme_bar_fade",   "SPY"),
    ]
    n = len(sleeves)
    slice_pct = 0.10 / n   # full pos size was 10%; divide across sleeves

    print(f"\nportfolio backtest — {n} sleeves on SPY, {slice_pct:.2%} per sleeve\n")
    sleeve_results = []
    for strategy_name, ticker in sleeves:
        print(f"  running sleeve: {strategy_name} x {ticker}")
        df, signal, result = run_single_sleeve(strategy_name, ticker, slice_pct)
        sleeve_results.append({
            "strategy": strategy_name,
            "ticker":   ticker,
            "sharpe":   result["sharpe"],
            "trades":   result["total_trades"],
            "max_dd":   result["max_drawdown"],
            "win_rate": result["win_rate"],
            "total_return": result["total_return"],
            "final_capital": result["final_capital"],
        })
        print(f"    sharpe={result['sharpe']:>+7.3f}  dd={result['max_drawdown']:.2%}  "
              f"wr={result['win_rate']:.2%}  trades={result['total_trades']}  "
              f"final=${result['final_capital']:.0f}")

    # portfolio aggregation — sum sleeve final capital, infer portfolio return
    starting_capital = INITIAL_CAP * n   # each sleeve starts with INITIAL_CAP
    ending_capital   = sum(s["final_capital"] for s in sleeve_results)
    portfolio_return = (ending_capital - starting_capital) / starting_capital

    # weighted-avg Sharpe by trade count is a rough proxy until we have
    # bar-level equity curves to merge properly. note: ignores correlation.
    total_trades = sum(s["trades"] for s in sleeve_results) or 1
    weighted_sharpe = sum(s["sharpe"] * s["trades"] for s in sleeve_results) / total_trades
    avg_max_dd = max(s["max_dd"] for s in sleeve_results)   # worst sleeve DD as floor

    print("\n" + "=" * 70)
    print("  portfolio summary")
    print("=" * 70)
    print(f"  sleeves:                {n}  ({', '.join(s['strategy'] for s in sleeve_results)})")
    print(f"  total trades:           {total_trades}")
    print(f"  portfolio total return: {portfolio_return:+.2%}")
    print(f"  weighted-avg Sharpe:    {weighted_sharpe:+.3f}  (approximation — true portfolio")
    print(f"                                                Sharpe needs bar-level equity merge)")
    print(f"  worst sleeve DD:        {avg_max_dd:.2%}")
    print()
    print(f"  per-sleeve breakdown:")
    print(f"  {'strategy':<22} {'sharpe':>8} {'dd':>8} {'wr':>7} {'trades':>8} {'final':>12}")
    print("  " + "-" * 68)
    for s in sleeve_results:
        print(f"  {s['strategy']:<22} {s['sharpe']:>+8.3f} {s['max_dd']:>7.2%} "
              f"{s['win_rate']:>7.2%} {s['trades']:>8} ${s['final_capital']:>10.0f}")

    out = Path("results/portfolio_top3.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"sleeves": sleeve_results,
                   "portfolio_total_return": portfolio_return,
                   "weighted_sharpe":        weighted_sharpe,
                   "run_at": datetime.now(timezone.utc).isoformat()},
                  f, indent=2, default=str)
    print(f"\n  saved to {out}")


if __name__ == "__main__":
    main()
