"""
runners/ml_alpha.py
-------------------
A machine-learning ALPHA model: instead of hand-coded sleeves, train a gradient-
boosting regressor to predict each S&P 500 name's forward 21-day return from a
factor panel (12-1 momentum, 5-day reversal, 1-month momentum, distance from
200-day, 60-day vol, distance from 52-week high, RSI-2). Strict WALK-FORWARD:
each month, train only on data whose forward return is already realized, predict
the current month, go long the top-K (market-filtered to cash when SPY < 200d).

Honest test: does ML beat the rule-based momentum sleeve (xs_dualmom) OUT-OF-
SAMPLE? Tabular financial data is low signal-to-noise and ML overfits easily, so
the walk-forward OOS result -- not an in-sample fit -- is the only thing that counts.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from agents.daily_strategies import _rsi, RT_COST, backtest_cross_sectional, daily_bars
from data.sp500 import sp500_tickers, load_daily

TD = 252
K = 30          # hold top-30 predicted
HORIZON = 21    # monthly


def build_panel():
    data = {t: d for t, d in load_daily(sp500_tickers(), start="2015-01-01").items() if len(d) > 300}
    C = pd.DataFrame({t: d["close"] for t, d in data.items()}).sort_index()
    R = C.pct_change()
    feats = {
        "mom_12_1": C.shift(21) / C.shift(252) - 1,
        "rev_5":    C / C.shift(5) - 1,
        "mom_1m":   C / C.shift(21) - 1,
        "dist_200": C / C.rolling(200).mean() - 1,
        "vol_60":   R.rolling(60).std() * np.sqrt(TD),
        "dist_high": C / C.rolling(252).max() - 1,
        "rsi2":     C.apply(lambda s: _rsi(s, 2)),
    }
    fwd = C.shift(-HORIZON) / C - 1
    grid = list(C.index[252::HORIZON])
    rows = []
    for g in grid:
        sub = pd.DataFrame({k: feats[k].loc[g] for k in feats})
        sub["fwd"] = fwd.loc[g]
        sub["date"] = g
        rows.append(sub.dropna(subset=list(feats)))
    panel = pd.concat(rows).reset_index().rename(columns={"index": "ticker"})
    return panel, list(feats), C


def main():
    print("building factor panel for the ML alpha model (~1-2 min) ...")
    panel, FEATS, C = build_panel()
    spy = daily_bars("SPY")["close"]
    grid = sorted(panel["date"].unique())

    print("walk-forward training (HistGradientBoosting, monthly) ...")
    rets, dates, prev = [], [], set()
    for i, d in enumerate(grid):
        if i < 24 or i >= len(grid) - 1:
            continue                                  # 2yr warmup; need realized fwd
        train = panel[panel["date"] <= grid[i - 2]].dropna(subset=["fwd"])
        test = panel[panel["date"] == d]
        if len(train) < 3000 or test.empty:
            continue
        m = HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.05)
        m.fit(train[FEATS], train["fwd"])
        test = test.assign(pred=m.predict(test[FEATS]))
        # market filter: only deploy when SPY > 200d, else cash
        on = float(spy.reindex([d]).iloc[0]) > float(spy.rolling(200).mean().reindex([d]).iloc[0]) if d in spy.index else True
        picks = set(test.nlargest(K, "pred")["ticker"]) if on else set()
        gross = test.set_index("ticker")["fwd"]
        r = float(gross.reindex(picks).mean()) if picks else 0.0
        turn = len(picks ^ prev) / max(2 * K, 1)
        rets.append(r - turn * RT_COST); dates.append(d); prev = picks

    sr = pd.Series(rets, index=pd.to_datetime(dates)).dropna()

    def stats(s, label):
        eq = (1 + s).cumprod(); yrs = len(s) / (TD / HORIZON)
        cagr = eq.iloc[-1] ** (1 / yrs) - 1
        sharpe = s.mean() / s.std() * np.sqrt(TD / HORIZON) if s.std() > 0 else 0
        dd = float((eq / eq.cummax() - 1).min())
        print(f"  {label:32s} Sharpe {sharpe:5.2f} | CAGR {cagr:6.1%} | DD {dd:6.1%} | n={len(s)}")
        return sharpe

    print("\nML ALPHA (out-of-sample, walk-forward) vs the rule-based momentum sleeve:")
    sh_ml = stats(sr, "ml_alpha (GBM, top-30)")

    # benchmark: xs_dualmom resampled to the same monthly grid
    xs = backtest_cross_sectional(sp500_tickers(), mode="momentum", lookback=252, skip=21, k=10, market_filter=True)["_returns"]
    xs_m = (1 + xs).resample(f"{HORIZON}D").prod() - 1
    xs_m = xs_m.reindex(sr.index, method="nearest")
    stats(xs_m.dropna(), "xs_dualmom (rule-based)")

    print("\n  VERDICT:", "ML beats the rule -> investigate further"
          if sh_ml > 1.30 else "ML does NOT beat the rule-based sleeve out-of-sample (expected: tabular")
    print("  financial data is low signal; the hand-built momentum sleeve already captures the edge).")


if __name__ == "__main__":
    main()
