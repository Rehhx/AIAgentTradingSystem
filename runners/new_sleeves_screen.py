"""
runners/new_sleeves_screen.py
-----------------------------
Screen the three NEW no-leverage equity sleeves (high_momentum, bollinger_revert,
ma_pullback) the same way every other candidate is screened:

  1. Standalone gate: Sharpe >= 0.8, max DD >= -15%, trades >= 50, and
     walk-forward positive in >= 4/5 folds with positive out-of-sample.
  2. Marginal value: does ADDING the sleeve (small weight) to the deployed
     portfolio_full book actually improve Sharpe / CAGR / DD? A sleeve that
     passes the gate but doesn't help the book is noise, not alpha.

No leverage is introduced here — these are long/flat sleeves combined by weight.
$100k base, 6 bps round-trip, split/dividend-adjusted data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    backtest_book, backtest_cross_sectional, vol_target, _metrics_from_returns,
    sig_rsi2_meanrev, sig_donchian, sig_trend_5020, sig_recovery, sig_pead,
    sig_high_momentum, sig_bollinger_revert, sig_ma_pullback,
    DEPLOY_PARAMS, QUALITY_UNIVERSE, walk_forward_folds, split_metrics, daily_bars,
)
from data.sp500 import sp500_tickers

NEW = {
    "high_momentum":    (sig_high_momentum,    {}),
    "bollinger_revert": (sig_bollinger_revert, {}),
    "ma_pullback":      (sig_ma_pullback,      {}),
}
W = {"rsi": 0.28, "don": 0.22, "trd": 0.14, "xs": 0.08, "rec": 0.18, "pead": 0.10}


def wf_ok(r, folds=5):
    s = split_metrics(r)
    pos = sum(1 for f in walk_forward_folds(r, folds) if f["sharpe"] > 0)
    return (s["test_sharpe"] > 0 and pos >= folds - 1), pos, s["test_sharpe"]


def build_book():
    U, sp = QUALITY_UNIVERSE, sp500_tickers()
    S = {
        "rsi":  backtest_book(sig_rsi2_meanrev, U, DEPLOY_PARAMS["rsi2_meanrev"])["_returns"],
        "don":  backtest_book(sig_donchian, U)["_returns"],
        "trd":  backtest_book(sig_trend_5020, U)["_returns"],
        "xs":   backtest_cross_sectional(sp, mode="momentum", lookback=252, skip=21, k=10, market_filter=True)["_returns"],
        "rec":  backtest_book(sig_recovery, U, {"hold_days": 120})["_returns"],
        "pead": backtest_book(sig_pead, sp, {"gap_pct": 0.05, "vol_mult": 2.0, "hold_days": 60})["_returns"],
    }
    panel = pd.concat(S, axis=1, sort=True)
    panel.columns = list(S)
    return panel


def overlays(combo, index, vt=0.17, maxlev=1.8):
    spy = daily_bars("SPY")["close"].reindex(index)
    warn = (spy < spy.rolling(50).mean()) & (spy.pct_change().rolling(20).std() * np.sqrt(252) > 0.20)
    ews = (1 - 0.4 * warn.astype(float)).shift(1).fillna(1.0)
    bil = daily_bars("BIL")["close"].pct_change().reindex(index).fillna(0.0)
    return vol_target(combo, vt, max_leverage=maxlev) * ews + 0.22 * bil


def main():
    print("building deployed book + scoring 3 new no-leverage sleeves ...\n")
    panel = build_book()
    base_combo = sum(panel[c].fillna(0) * W[c] for c in W)
    base = overlays(base_combo, panel.index)
    bm = _metrics_from_returns(base, [], "base")
    print(f"deployed portfolio_full: Sharpe {bm['sharpe']:.2f} | CAGR {bm['cagr']:.1%} | DD {bm['max_drawdown']:.1%}\n")

    print(f"{'new sleeve':18s} {'Sharpe':>6s} {'CAGR':>7s} {'maxDD':>7s} {'trades':>6s} {'WF':>9s}  gate    +book effect")
    print("-" * 96)
    new_rets = {}
    for name, (fn, prm) in NEW.items():
        m = backtest_book(fn, QUALITY_UNIVERSE, prm, label=name)
        r = m["_returns"]; new_rets[name] = r
        ok, pos, oos = wf_ok(r)
        gate = (m["sharpe"] >= 0.8 and m["max_drawdown"] >= -0.15 and m["total_trades"] >= 50 and ok)
        # marginal effect: add the sleeve at 8% (renormalize the rest to 92%)
        combo2 = base_combo * 0.92 + r.reindex(panel.index).fillna(0) * 0.08
        b2 = overlays(combo2, panel.index)
        m2 = _metrics_from_returns(b2, [], name)
        eff = (f"Sharpe {m2['sharpe']:+.2f} ({m2['sharpe']-bm['sharpe']:+.2f}) "
               f"CAGR {m2['cagr']:.1%} DD {m2['max_drawdown']:.1%}")
        print(f"{name:18s} {m['sharpe']:6.2f} {m['cagr']:7.1%} {m['max_drawdown']:7.1%} "
              f"{m['total_trades']:6d} {pos}/5 {oos:+.2f}  {'PASS ' if gate else 'fail '}  {eff}")

    print("\nverdict: a sleeve earns a spot only if it PASSES the gate AND lifts the book's")
    print("Sharpe (or holds Sharpe while cutting DD). Otherwise it's noise — leave it out.")


if __name__ == "__main__":
    main()
