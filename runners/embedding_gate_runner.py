"""
agents/embedding_gate_runner.py
-------------------------------
proof-of-concept: precompute embedding-based regime quality for one ticker,
run a backtest with the quality gate on, compare to the baseline.

this is the heavy version of the regime gate — instead of the local heuristic
that classifies each window by trend/vol thresholds, this hits the regime
store with find_similar() and gates entries on the forward-return distribution
of the k most-similar historical patterns.

cost: ~1 OpenAI embedding call per query row. for SPY at step=60 over 3y
that's ~4800 calls — about $0.05 in API cost and 15-25 min of latency,
one time per ticker (results are cached to vector_stores/.cache).

usage:
    python agents/embedding_gate_runner.py SPY bb_squeeze

after the first run the parquet cache makes subsequent backtests instant.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# load .env so OPENAI_API_KEY is available to vector_stores.client
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

log = logging.getLogger("embedding_gate_runner")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def main(ticker: str, strategy_name: str):
    if strategy_name not in STRATEGIES:
        raise SystemExit(f"unknown strategy: {strategy_name}")

    fn, default_params = STRATEGIES[strategy_name][0], STRATEGIES[strategy_name][1]
    stop_mult = default_params.get("stop_atr_mult", ATR_STOP_MULT)
    allowed   = STRATEGY_REGIME_AFFINITY.get(strategy_name)

    df = load_ticker(ticker, data_dir=DATA_DIR,
                     start="2022-01-01", end="2025-01-01", session="regular")

    print(f"loaded {len(df):,} bars for {ticker}")

    # baseline: heuristic regime gate (no embedding)
    signal   = fn(df, default_params)
    regime_s = regime_label_series(df)
    baseline = run_backtest(
        df, signal,
        stop_atr_mult=stop_mult,
        regime_series=regime_s,
        allowed_regimes=allowed,
    )
    print(f"\nbaseline (heuristic regime gate):")
    print(f"  sharpe={baseline['sharpe']:.3f}  dd={baseline['max_drawdown']:.2%}  "
          f"wr={baseline['win_rate']:.2%}  trades={baseline['total_trades']}")

    # precompute (or load cached) embedding-based quality
    print(f"\nprecomputing regime quality for {ticker} (this may take 15-25 min on first run)...")
    quality_df = precompute_regime_quality(ticker, df, step=60)
    quality_aligned = quality_df.reindex(df.index, method="ffill")

    # gated run
    gated = run_backtest(
        df, signal,
        stop_atr_mult=stop_mult,
        regime_series=regime_s,
        allowed_regimes=allowed,
        quality_series=quality_aligned,
        quality_min_pct_pos=0.55,
    )
    print(f"\nwith embedding quality gate (pct_positive >= 0.55):")
    print(f"  sharpe={gated['sharpe']:.3f}  dd={gated['max_drawdown']:.2%}  "
          f"wr={gated['win_rate']:.2%}  trades={gated['total_trades']}")

    print(f"\ndelta: sharpe {gated['sharpe'] - baseline['sharpe']:+.3f}, "
          f"trades {gated['total_trades'] - baseline['total_trades']:+d}")

    out_path = Path(f"results/embedding_gate_{ticker}_{strategy_name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "ticker":   ticker,
            "strategy": strategy_name,
            "baseline": {k: v for k, v in baseline.items() if k not in ("trades", "equity_curve")},
            "gated":    {k: v for k, v in gated.items()    if k not in ("trades", "equity_curve")},
            "run_at":   datetime.now(timezone.utc).isoformat(),
        }, f, indent=2, default=str)
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    ticker        = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    strategy_name = sys.argv[2] if len(sys.argv) > 2 else "bb_squeeze"
    main(ticker, strategy_name)
