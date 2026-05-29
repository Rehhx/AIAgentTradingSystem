"""
agents/ml_sweep_runner.py
-------------------------
runs ml_research_agent (XGBoost) on every active ticker; reports classification
metrics (accuracy, AUC) AND the trading Sharpe of using the model's predicted
probability as a {-1, 0, 1} signal through the same backtest engine the rule
strategies use. apples-to-apples comparison.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from agents.ml_research_agent import MLResearchAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

TICKERS = ["SPY", "QQQ", "TSLA", "NVDA", "AAPL"]


def main():
    agent = MLResearchAgent()
    rows  = []

    for ticker in TICKERS:
        print(f"\n--- {ticker} ---")
        result = agent.run({"payload": {
            "ticker":              ticker,
            "model":               "xgboost",
            "forward_bars":        5,
            "train_pct":           0.7,
            "prob_long_threshold": 0.55,
            "prob_short_threshold": 0.45,
        }})
        if not result["success"]:
            print(f"  FAILED: {result.get('reason')}")
            continue
        m = result["metrics"]
        print(f"  accuracy={m['accuracy']:.4f}  auc={m['auc']:.4f}")
        print(f"  trading sharpe={m['trading_sharpe']:.3f}  trades={m['trading_trades']}  wr={m['trading_wr']:.2%}")
        print(f"  top features (10):")
        for name, imp in m["top_features"][:10]:
            print(f"    {name:<25} {imp:.4f}")
        rows.append({"ticker": ticker, **{k: m[k] for k in
                    ("accuracy", "auc", "trading_sharpe", "trading_trades", "trading_wr", "trading_dd")}})

    print("\n" + "=" * 80)
    print(f"  {'ticker':<8} {'acc':>6} {'auc':>6} {'sharpe':>8} {'trades':>8} {'wr':>7}")
    print("  " + "-" * 50)
    for r in rows:
        print(f"  {r['ticker']:<8} {r['accuracy']:>6.3f} {r['auc']:>6.3f} "
              f"{r['trading_sharpe']:>8.3f} {r['trading_trades']:>8} {r['trading_wr']:>7.2%}")

    out = Path("results/ml_sweep_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"rows": rows, "run_at": datetime.now(timezone.utc).isoformat()},
                  f, indent=2, default=str)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
