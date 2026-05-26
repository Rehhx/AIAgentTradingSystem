"""
agents/ml_research_agent.py
---------------------------
trains ML models on 1m bar data to predict short-horizon return direction,
then evaluates the predictions as a trading signal through the same backtest
engine the rule-based strategies use (so sharpe is apples-to-apples).

model zoo:
  - xgboost    (implemented, default)
  - lstm       (stub — uncomment torch in requirements.txt to enable)
  - transformer (stub)

features:
  - technical indicators from backtesting_agent (RSI, ATR, EMA ratios, BB pos, vol z)
  - regime label (one-hot from regime_label_series)
  - lagged returns at multiple horizons
  - time-of-day buckets

target:
  - sign of forward N-bar return (binary classification — easier and more robust
    than magnitude regression on 1m data)

evaluation:
  - walk-forward train/test split (no leakage)
  - report accuracy, AUC, AND trading-Sharpe of using model probability as signal
    (signal = long when p_up > 0.55, short when p_up < 0.45)
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_ticker, DATA_DIR
from agents.backtesting_agent import (
    atr, rsi, ema, bollinger_bands, volume_zscore,
    regime_label_series,
    run_backtest,
    ATR_STOP_MULT,
)

log = logging.getLogger("ml_research_agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

REGIME_CLASSES = ["trending", "chop", "breakout", "mean_reversion", "unknown"]


# ---------------------------------------------------------------------------
# feature engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    construct a tabular feature matrix from raw OHLCV bars.
    every column is point-in-time — no lookahead.
    """
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

    feats = pd.DataFrame(index=df.index)
    feats["ret_1"]   = close.pct_change(1)
    feats["ret_5"]   = close.pct_change(5)
    feats["ret_15"]  = close.pct_change(15)
    feats["ret_60"]  = close.pct_change(60)

    feats["rsi_14"]  = rsi(close, 14)
    feats["atr_14"]  = atr(df, 14)
    feats["atr_pct"] = feats["atr_14"] / close

    ema9, ema21, ema50 = ema(close, 9), ema(close, 21), ema(close, 50)
    feats["ema9_21_ratio"]  = ema9 / ema21 - 1
    feats["ema21_50_ratio"] = ema21 / ema50 - 1
    feats["close_to_ema21"] = close / ema21 - 1

    bb_lo, bb_mid, bb_up = bollinger_bands(close, 20, 2.0)
    feats["bb_position"]  = (close - bb_mid) / (bb_up - bb_lo).replace(0, np.nan)
    feats["bb_width_pct"] = (bb_up - bb_lo) / bb_mid

    feats["vol_z_20"] = volume_zscore(vol, 20)
    feats["vol_z_60"] = volume_zscore(vol, 60)

    # regime as one-hot (numeric for xgboost)
    regimes = regime_label_series(df)
    for cls in REGIME_CLASSES:
        feats[f"regime_{cls}"] = (regimes == cls).astype(int)

    # time-of-day in ET (open/midday/close buckets)
    et_hour = df.index.tz_convert("America/New_York").hour
    feats["tod_open"]   = ((et_hour >= 9)  & (et_hour < 11)).astype(int)
    feats["tod_midday"] = ((et_hour >= 11) & (et_hour < 14)).astype(int)
    feats["tod_close"]  = ((et_hour >= 14) & (et_hour < 16)).astype(int)

    return feats


def build_target(df: pd.DataFrame, forward_bars: int = 5) -> pd.Series:
    """
    binary: 1 if close N bars ahead > current close, else 0.
    shift(-N) is the standard forward-return target — note this introduces
    NaN at the tail; we drop those rows when training.
    """
    fwd_ret = df["close"].shift(-forward_bars) / df["close"] - 1
    return (fwd_ret > 0).astype(int).where(fwd_ret.notna())


# ---------------------------------------------------------------------------
# xgboost trainer
# ---------------------------------------------------------------------------

def train_xgboost(X_train, y_train, X_test, y_test) -> dict:
    """trains an XGBoost classifier and returns metrics + predictions."""
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, roc_auc_score

    model = xgb.XGBClassifier(
        n_estimators      = 200,
        max_depth         = 5,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        objective         = "binary:logistic",
        eval_metric       = "auc",
        tree_method       = "hist",
        n_jobs            = -1,
        random_state      = 42,
    )
    model.fit(X_train, y_train)

    p_test  = model.predict_proba(X_test)[:, 1]
    yhat    = (p_test > 0.5).astype(int)

    return {
        "model":        model,
        "probs_test":   p_test,
        "preds_test":   yhat,
        "accuracy":     float(accuracy_score(y_test, yhat)),
        "auc":          float(roc_auc_score(y_test, p_test)),
        "feature_importance": dict(zip(X_train.columns, model.feature_importances_.tolist())),
    }


def train_lstm(*args, **kwargs):
    raise NotImplementedError(
        "LSTM trainer not enabled — uncomment torch in requirements.txt and implement."
    )


def train_transformer(*args, **kwargs):
    raise NotImplementedError(
        "Transformer trainer not enabled — uncomment torch in requirements.txt and implement."
    )


TRAINERS = {
    "xgboost":     train_xgboost,
    "lstm":        train_lstm,
    "transformer": train_transformer,
}


# ---------------------------------------------------------------------------
# main agent class
# ---------------------------------------------------------------------------

class MLResearchAgent:
    """
    matches BaseAgent.run(task) contract used by orchestrator. orchestrator
    delegates to this real class; the stub in orchestrator.py can be replaced
    with a thin wrapper that holds an instance of this.
    """

    def __init__(self, store=None, data_dir: Path = DATA_DIR):
        self.store    = store
        self.data_dir = data_dir
        self.log      = logging.getLogger("ml_research_agent")

    def run(self, task: dict) -> dict:
        payload      = task.get("payload", {})
        ticker       = payload.get("ticker", "SPY")
        model_name   = payload.get("model", "xgboost")
        forward_bars = payload.get("forward_bars", 5)
        train_pct    = payload.get("train_pct", 0.7)
        start        = payload.get("start", "2022-01-01")
        end          = payload.get("end", "2025-01-01")
        prob_long    = payload.get("prob_long_threshold", 0.55)
        prob_short   = payload.get("prob_short_threshold", 0.45)

        if model_name not in TRAINERS:
            return {"success": False, "agent": "ml_research_agent",
                    "reason": f"unknown model {model_name}"}

        self.log.info(f"training {model_name} on {ticker} | forward={forward_bars}b train_pct={train_pct}")

        df = load_ticker(ticker, data_dir=self.data_dir, start=start, end=end, session="regular")
        X  = build_features(df)
        y  = build_target(df, forward_bars=forward_bars)

        # drop rows with NaNs (early bars without warmup + tail without target)
        valid = X.dropna().index.intersection(y.dropna().index)
        X, y  = X.loc[valid], y.loc[valid]

        cut = int(len(X) * train_pct)
        X_train, X_test = X.iloc[:cut], X.iloc[cut:]
        y_train, y_test = y.iloc[:cut], y.iloc[cut:]

        result = TRAINERS[model_name](X_train, y_train, X_test, y_test)

        # convert predictions into a signal series and run through backtest engine
        signal = pd.Series(0, index=df.index)
        prob_aligned = pd.Series(result["probs_test"], index=X_test.index)
        signal.loc[prob_aligned.index] = np.where(
            prob_aligned >= prob_long, 1,
            np.where(prob_aligned <= prob_short, -1, 0)
        )

        test_df_full = df.loc[X_test.index[0]:]   # backtest on the test segment
        signal_test  = signal.loc[test_df_full.index]
        bt = run_backtest(test_df_full, signal_test, stop_atr_mult=ATR_STOP_MULT)

        metrics = {
            "model":         model_name,
            "ticker":        ticker,
            "forward_bars":  forward_bars,
            "n_train":       len(X_train),
            "n_test":        len(X_test),
            "accuracy":      result["accuracy"],
            "auc":           result["auc"],
            "trading_sharpe": bt["sharpe"],
            "trading_trades": bt["total_trades"],
            "trading_wr":     bt["win_rate"],
            "trading_dd":     bt["max_drawdown"],
            "top_features":   sorted(
                result["feature_importance"].items(),
                key=lambda x: x[1], reverse=True,
            )[:10],
        }

        if self.store is not None:
            self.store.log_model_score(model_name, ticker, metrics)

        return {"success": True, "agent": "ml_research_agent", "metrics": metrics}


# ---------------------------------------------------------------------------
# standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    model  = sys.argv[2] if len(sys.argv) > 2 else "xgboost"

    agent  = MLResearchAgent()
    result = agent.run({"payload": {"ticker": ticker, "model": model}})

    if not result["success"]:
        print(f"FAILED: {result.get('reason')}")
        sys.exit(1)

    m = result["metrics"]
    print(f"\n{model} on {ticker}:")
    print(f"  n_train={m['n_train']:,}  n_test={m['n_test']:,}")
    print(f"  accuracy={m['accuracy']:.4f}  auc={m['auc']:.4f}")
    print(f"  trading_sharpe={m['trading_sharpe']:.3f}  trades={m['trading_trades']}  wr={m['trading_wr']:.2%}")
    print(f"\n  top features:")
    for name, imp in m["top_features"]:
        print(f"    {name:<25} {imp:.4f}")

    out = Path(f"results/ml_research_{ticker}_{model}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"result": result, "run_at": datetime.now(timezone.utc).isoformat()},
                  f, indent=2, default=str)
    print(f"\nsaved to {out}")
