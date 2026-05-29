"""
agents/dispersion_runner.py
---------------------------
runs the QQQ/SPY dispersion snapback strategy proposed by autonomous_agent.

cross-asset signals can't ride through the standard STRATEGIES registry
(which assumes one-ticker-in, one-signal-out), so this runner loads both
tickers, builds the signal, and calls run_backtest on QQQ as the tradeable
instrument.

usage:
    python agents/dispersion_runner.py
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from data.loader import load_ticker, DATA_DIR
from agents.backtesting_agent import (
    signals_qqq_spy_dispersion,
    regime_label_series,
    run_backtest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def main():
    params = {
        "beta_lookback_min":  60,
        "zscore_lookback_min": 30,
        "entry_z":             2.0,
        "exit_z":              0.3,
    }

    print(f"loading QQQ and SPY 2022-2025...")
    qqq = load_ticker("QQQ", data_dir=DATA_DIR, start="2022-01-01", end="2025-01-01", session="regular")
    spy = load_ticker("SPY", data_dir=DATA_DIR, start="2022-01-01", end="2025-01-01", session="regular")
    print(f"  QQQ {len(qqq):,} bars, SPY {len(spy):,} bars")

    print(f"\nbuilding dispersion signal with params {params}...")
    signal = signals_qqq_spy_dispersion(qqq, spy, params)
    nonzero = (signal != 0).sum()
    print(f"  {nonzero:,} bars with active position ({nonzero/len(signal):.1%} of bars)")

    print(f"\nrunning backtest on QQQ as the tradeable leg...")
    # cross-asset has no regime affinity defined; run without the regime gate
    # for the first measurement, then optionally retry with it.
    bt = run_backtest(qqq, signal, stop_atr_mult=2.0)

    print(f"\nresults:")
    print(f"  total return:   {bt['total_return']:.4%}")
    print(f"  sharpe:         {bt['sharpe']:.3f}")
    print(f"  max drawdown:   {bt['max_drawdown']:.2%}")
    print(f"  win rate:       {bt['win_rate']:.2%}")
    print(f"  total trades:   {bt['total_trades']}")
    print(f"  avg bars held:  {bt['avg_bars_held']:.1f}")
    print(f"  profit factor:  {bt['profit_factor']:.3f}")

    print(f"\nretry with the regime gate (chop + mean_reversion only)...")
    regime_s = regime_label_series(qqq)
    bt_gated = run_backtest(
        qqq, signal,
        stop_atr_mult   = 2.0,
        regime_series   = regime_s,
        allowed_regimes = {"chop", "mean_reversion"},
    )
    print(f"  sharpe={bt_gated['sharpe']:.3f}  dd={bt_gated['max_drawdown']:.2%}  "
          f"wr={bt_gated['win_rate']:.2%}  trades={bt_gated['total_trades']}")

    out_path = Path("results/dispersion_qqq_spy.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "strategy":  "qqq_spy_dispersion_snapback",
        "tickers":   {"tradeable": "QQQ", "anchor": "SPY"},
        "params":    params,
        "ungated":   {k: v for k, v in bt.items() if k not in ("trades", "equity_curve")},
        "gated":     {k: v for k, v in bt_gated.items() if k not in ("trades", "equity_curve")},
        "run_at":    datetime.now(timezone.utc).isoformat(),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
