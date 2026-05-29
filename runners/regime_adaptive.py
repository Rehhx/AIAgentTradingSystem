"""
runners/regime_adaptive.py
--------------------------
A SEPARATE, more aggressive book that ensembles regime-switching + conditional
leverage — kept apart from the validated `blended_plus` deploy so the proven
book stays untouched.

Regime is read from SPY (200-day trend + 20-day realized vol), decided on the
prior close (no lookahead):

  BULL-CALM  (SPY>200d, vol<16%) : offense — momentum/trend heavy, leverage 1.5x
  BULL-VOL   (SPY>200d, vol>=16%): stay long but de-lever to 1.0x, mean-reversion tilt
  BEAR       (SPY<200d)          : defense — gold + cash, NO leverage

Leverage is therefore CONDITIONAL: full 1.5x only in calm uptrends, never in a
downtrend or a vol spike. Compares against the static ensemble + the
leverage-only ensemble.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    backtest_book, backtest_cross_sectional, vol_target, _metrics_from_returns,
    sig_rsi2_meanrev, sig_donchian, sig_trend_5020, DEPLOY_PARAMS, QUALITY_UNIVERSE,
    walk_forward_folds, split_metrics, daily_bars,
)
from data.sp500 import sp500_tickers

# per-regime sleeve weights (invested fraction; remainder is cash) + leverage
REGIME = {
    "bull_calm": ({"xs": 0.40, "trd": 0.30, "don": 0.15, "rsi": 0.15, "gld": 0.0}, 1.5),
    "bull_vol":  ({"rsi": 0.45, "don": 0.20, "trd": 0.20, "xs": 0.15, "gld": 0.0}, 1.0),
    "bear":      ({"rsi": 0.0,  "don": 0.0,  "trd": 0.0,  "xs": 0.0,  "gld": 0.40}, 1.0),
}


def folds(r):
    return "  ".join(f"{f.get('start','?')[:4]}-{f.get('end','?')[:4]}:{f['return_pct']:+.0%}"
                     for f in walk_forward_folds(r, 5))


def show(label, r):
    m = _metrics_from_returns(r, [], label)
    gate = "PASS" if (m["sharpe"] >= 0.8 and m["max_drawdown"] >= -0.15) else "FAIL"
    s = split_metrics(r)
    print(f"{label:26s} SR {m['sharpe']:5.2f} | CAGR {m['cagr']:6.1%} | DD {m['max_drawdown']:6.1%} "
          f"| OOS {s['test_sharpe']:+.2f} | {gate}\n    folds {folds(r)}")
    return m


def main():
    U = QUALITY_UNIVERSE
    print("\nbuilding sleeves ...")
    sl = {
        "rsi": backtest_book(sig_rsi2_meanrev, U, DEPLOY_PARAMS["rsi2_meanrev"])["_returns"],
        "don": backtest_book(sig_donchian, U)["_returns"],
        "trd": backtest_book(sig_trend_5020, U)["_returns"],
        "xs":  backtest_cross_sectional(sp500_tickers(), mode="momentum", lookback=252,
                                        skip=21, k=10, market_filter=True)["_returns"],
    }
    panel = pd.concat(sl, axis=1, sort=True); panel.columns = list(sl)
    idx = panel.index
    gld = daily_bars("GLD")["close"].pct_change().reindex(idx).fillna(0.0)
    panel["gld"] = gld

    # regime from SPY (decided on prior close)
    spy = daily_bars("SPY")["close"]
    trend_up = (spy > spy.rolling(200).mean()).reindex(idx).ffill().fillna(False)
    vol = (spy.pct_change().rolling(20).std() * np.sqrt(252)).reindex(idx).ffill().fillna(0.2)
    regime = pd.Series("bull_calm", index=idx)
    regime[~trend_up] = "bear"
    regime[trend_up & (vol >= 0.16)] = "bull_vol"
    regime = regime.shift(1).fillna("bear")

    # build the regime-adaptive return series
    cols = list(panel.columns)
    port = pd.Series(0.0, index=idx)
    for reg, (w, lev) in REGIME.items():
        mask = (regime == reg)
        contrib = sum(w.get(c, 0.0) * panel[c] for c in cols) * lev
        port[mask] = contrib[mask]

    print(f"\nregime mix: " + ", ".join(f"{r}={(regime==r).mean():.0%}" for r in REGIME))
    print(f"\n{'book':26s} metrics")
    print("-" * 96)

    raw_ens = panel[["rsi", "don", "trd", "xs"]].mean(axis=1)
    show("static ensemble (deploy)", vol_target(raw_ens, 0.12, max_leverage=1.0))
    show("ensemble + leverage 1.5x", vol_target(raw_ens, 0.12, max_leverage=1.5))
    show("REGIME-ADAPTIVE (raw)", port)
    show("REGIME-ADAPTIVE + vol-cap", vol_target(port, 0.13, max_leverage=1.6))

    print("\nLeverage is conditional (1.5x only in calm uptrends). Bear = gold+cash, no leverage.")


if __name__ == "__main__":
    main()
