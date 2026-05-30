"""
runners/diversifier_screen.py
-----------------------------
The equity-pattern space on our 10 names is saturated (see new_sleeves_screen.py).
Genuine additive return must come from a DIFFERENT ASSET CLASS whose cycle doesn't
line up with the equity book. This screens no-leverage, long/flat ABSOLUTE-MOMENTUM
sleeves on other asset classes and asks the only question that matters for a
diversifier:

    does ADDING it to portfolio_full raise Sharpe or cut drawdown
    (a diversifier earns its seat on risk-adjusted improvement, not headline CAGR)?

For each candidate we report: standalone Sharpe/CAGR/DD/trades, walk-forward,
CORRELATION to the deployed book, and the marginal effect of adding it at 12%.
$100k base, 6 bps round-trip, split/dividend-adjusted daily data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    backtest_book, backtest_cross_sectional, vol_target, _metrics_from_returns,
    sig_rsi2_meanrev, sig_donchian, sig_trend_5020, sig_recovery, sig_pead,
    sig_abs_momentum, DEPLOY_PARAMS, QUALITY_UNIVERSE,
    walk_forward_folds, split_metrics, daily_bars,
)
from data.sp500 import sp500_tickers

W = {"rsi": 0.28, "don": 0.22, "trd": 0.14, "xs": 0.08, "rec": 0.18, "pead": 0.10}

# candidate diversifiers: (universe, signal-params). All sig_abs_momentum -> each
# asset is long when its own 6-month trend is up, else flat (no leverage, no shorts).
CANDIDATES = {
    "bond_trend (TLT,IEF)":              (["TLT", "IEF"], {"lookback": 126}),
    "commodity_trend (GLD,DBC,SLV)":     (["GLD", "DBC", "SLV"], {"lookback": 126}),
    "intl_trend (EFA,EEM,VEA,VWO)":      (["EFA", "EEM", "VEA", "VWO"], {"lookback": 126}),
    "allweather_trend (7 asset ETFs)":   (["TLT", "IEF", "GLD", "DBC", "EFA", "EEM", "VNQ"], {"lookback": 126}),
}


def loadable(universe):
    """keep only tickers with enough adjusted daily history so backtest_book won't choke."""
    ok = []
    for t in universe:
        try:
            if len(daily_bars(t)) >= 260:
                ok.append(t)
        except Exception:
            pass
    return ok


def wf_ok(r, folds=5):
    s = split_metrics(r)
    pos = sum(1 for f in walk_forward_folds(r, folds) if f["sharpe"] > 0)
    return pos, s["test_sharpe"]


def build_base():
    U, sp = QUALITY_UNIVERSE, sp500_tickers()
    S = {
        "rsi":  backtest_book(sig_rsi2_meanrev, U, DEPLOY_PARAMS["rsi2_meanrev"])["_returns"],
        "don":  backtest_book(sig_donchian, U)["_returns"],
        "trd":  backtest_book(sig_trend_5020, U)["_returns"],
        "xs":   backtest_cross_sectional(sp, mode="momentum", lookback=252, skip=21, k=10, market_filter=True)["_returns"],
        "rec":  backtest_book(sig_recovery, U, {"hold_days": 120})["_returns"],
        "pead": backtest_book(sig_pead, sp, {"gap_pct": 0.05, "vol_mult": 2.0, "hold_days": 60})["_returns"],
    }
    panel = pd.concat(S, axis=1, sort=True); panel.columns = list(S)
    return panel


def overlays(combo, index, vt=0.17, maxlev=1.8):
    spy = daily_bars("SPY")["close"].reindex(index)
    warn = (spy < spy.rolling(50).mean()) & (spy.pct_change().rolling(20).std() * np.sqrt(252) > 0.20)
    ews = (1 - 0.4 * warn.astype(float)).shift(1).fillna(1.0)
    bil = daily_bars("BIL")["close"].pct_change().reindex(index).fillna(0.0)
    return vol_target(combo, vt, max_leverage=maxlev) * ews + 0.22 * bil


def main():
    print("building deployed book + scoring asset-class diversifiers ...\n")
    panel = build_base()
    base_combo = sum(panel[c].fillna(0) * W[c] for c in W)
    base = overlays(base_combo, panel.index)
    bm = _metrics_from_returns(base, [], "base")
    print(f"deployed portfolio_full: Sharpe {bm['sharpe']:.2f} | CAGR {bm['cagr']:.1%} | DD {bm['max_drawdown']:.1%}\n")

    print(f"{'diversifier':30s} {'Shrp':>5s} {'CAGR':>6s} {'maxDD':>7s} {'trd':>4s} {'WF':>4s} {'corr':>5s}  +book @12%")
    print("-" * 104)
    for name, (uni, prm) in CANDIDATES.items():
        u = loadable(uni)
        if not u:
            print(f"{name:30s}  (no data)")
            continue
        m = backtest_book(sig_abs_momentum, u, prm, label=name)
        r = m["_returns"]
        pos, oos = wf_ok(r)
        # correlation to the (pre-overlay) equity book, on overlapping dates
        rr = r.reindex(panel.index)
        corr = float(base_combo.corr(rr))
        # marginal effect: add at 12% (renormalize the rest to 88%)
        combo2 = base_combo * 0.88 + rr.fillna(0) * 0.12
        m2 = _metrics_from_returns(overlays(combo2, panel.index), [], name)
        d_sh, d_dd = m2["sharpe"] - bm["sharpe"], m2["max_drawdown"] - bm["max_drawdown"]
        verdict = "ADD" if (d_sh >= 0.02 or d_dd >= 0.01) else "noise"
        eff = (f"Sharpe {m2['sharpe']:.2f} ({d_sh:+.2f}) CAGR {m2['cagr']:.1%} "
               f"DD {m2['max_drawdown']:.1%} ({d_dd:+.1%}) -> {verdict}")
        print(f"{name:30s} {m['sharpe']:5.2f} {m['cagr']:6.1%} {m['max_drawdown']:7.1%} "
              f"{m['total_trades']:4d} {pos}/5 {corr:5.2f}  {eff}")

    print("\nA diversifier earns a seat if it lifts the book's Sharpe by >=0.02 OR cuts DD by")
    print(">=1pt. Low correlation to the equity book is the mechanism; headline CAGR is not.")


if __name__ == "__main__":
    main()
