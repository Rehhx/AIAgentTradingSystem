"""
runners/portfolio_allocator.py
------------------------------
A "portfolio of strategies" that AUTO-INCLUDES any strategy which passes the
bar — so winners get added automatically and dead ones are dropped, without
hand-editing weights.

How it works:
  1. CANDIDATES is a registry: name -> function returning a daily return series.
     New strategies (incl. ones on DIFFERENT tickers/universes) just get added
     here — see the cross-sectional book ranking the full S&P 500 vs the
     quality-10 sleeves below.
  2. Each candidate is scored: Sharpe, CAGR, max DD, and walk-forward (positive
     out-of-sample + positive in >= 4/5 folds).
  3. Only candidates that PASS (Sharpe >= min_sharpe, DD >= dd_budget, WF ok)
     are admitted. Admitted strategies are combined RISK-PARITY (inverse-vol) so
     no single sleeve dominates, then scaled to a target volatility to aim for
     the desired return band.

SAFETY: this selects + combines for the PAPER/backtest portfolio. Auto-promoting
to LIVE capital still needs a human gate + a live paper-trading period — a buggy
or overfit strategy must never auto-deploy to real money.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    backtest_book, backtest_cross_sectional, vol_target, _metrics_from_returns,
    sig_rsi2_meanrev, sig_donchian, sig_trend_5020, sig_recovery, sig_pead,
    sig_trend_multi, DEPLOY_PARAMS, QUALITY_UNIVERSE,
    walk_forward_folds, split_metrics, TRADING_DAYS,
)
from data.sp500 import sp500_tickers


# ---- candidate registry: add new strategies here; passers auto-join -----------
# different universes for genuine diversification (uncorrelated return streams)
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
DEFENSIVE   = ["XLP", "XLU", "XLV", "GLD"]   # staples / utilities / healthcare / gold

def _rsi():   return backtest_book(sig_rsi2_meanrev, QUALITY_UNIVERSE, DEPLOY_PARAMS["rsi2_meanrev"])["_returns"]
def _don():   return backtest_book(sig_donchian, QUALITY_UNIVERSE)["_returns"]
def _trd():   return backtest_book(sig_trend_5020, QUALITY_UNIVERSE)["_returns"]
def _xs500(): return backtest_cross_sectional(sp500_tickers(), mode="momentum", lookback=252, skip=21, k=10, market_filter=True)["_returns"]
def _sector(): return backtest_cross_sectional(SECTOR_ETFS, mode="momentum", lookback=126, skip=21, k=4, market_filter=True)["_returns"]
def _defensive(): return backtest_book(sig_rsi2_meanrev, DEFENSIVE, DEPLOY_PARAMS["rsi2_meanrev"])["_returns"]
def _recovery(): return backtest_book(sig_recovery, QUALITY_UNIVERSE, {"hold_days": 120})["_returns"]
def _pead(): return backtest_book(sig_pead, sp500_tickers(), {"gap_pct": 0.05, "vol_mult": 2.0, "hold_days": 60})["_returns"]
def _trendmulti(): return backtest_book(sig_trend_multi, QUALITY_UNIVERSE)["_returns"]

CANDIDATES = {
    "rsi2_meanrev (quality-10)":      _rsi,
    "donchian (quality-10)":          _don,
    "trend_5020 (quality-10)":        _trd,
    "xs_dualmom (full S&P 500)":      _xs500,
    "recovery_thrust (quality-10)":   _recovery,    # captures bull-run snapbacks
    "pead (full S&P 500)":            _pead,         # event-driven earnings drift
    "trend_multi (quality-10)":       _trendmulti,   # multi-speed trend
    "sector_momentum (11 ETFs)":      _sector,       # different universe: sector rotation
    "defensive_meanrev (staples/util/hlth/gold)": _defensive,  # low-beta ballast
    # add new strategies / different universes here -> they auto-join if they pass
}


def wf_ok(r, folds=5):
    s = split_metrics(r)
    pos = sum(1 for f in walk_forward_folds(r, folds) if f["sharpe"] > 0)
    return (s["test_sharpe"] > 0 and pos >= folds - 1), pos, s["test_sharpe"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-sharpe", type=float, default=0.8)
    ap.add_argument("--dd-budget", type=float, default=-0.15, help="max drawdown a candidate may have")
    ap.add_argument("--target-vol", type=float, default=0.0, help="scale the combined book to this vol (0=off)")
    ap.add_argument("--max-leverage", type=float, default=1.0)
    args = ap.parse_args()

    print("\nscoring candidate strategies ...\n")
    print(f"{'candidate':30s} {'Sharpe':>6s} {'CAGR':>7s} {'maxDD':>7s} {'WF':>10s}  verdict")
    print("-" * 80)
    rets, admitted = {}, []
    for name, fn in CANDIDATES.items():
        r = fn()
        rets[name] = r
        m = _metrics_from_returns(r, [], name)
        ok, pos, oos = wf_ok(r)
        passes = (m["sharpe"] >= args.min_sharpe and m["max_drawdown"] >= args.dd_budget and ok)
        if passes:
            admitted.append(name)
        print(f"{name:30s} {m['sharpe']:6.2f} {m['cagr']:7.1%} {m['max_drawdown']:7.1%} "
              f"{pos}/5 {oos:+.2f}  {'ADMIT' if passes else 'reject'}")

    if not admitted:
        print("\nno candidate passed — nothing to allocate.")
        return

    # risk-parity (inverse-vol) weights over the admitted strategies
    panel = pd.concat([rets[n] for n in admitted], axis=1, sort=True)
    panel.columns = admitted
    inv_vol = {n: 1.0 / (panel[n].std() or 1e-9) for n in admitted}
    w = np.array([inv_vol[n] for n in admitted]); w /= w.sum()
    combined = (panel.fillna(0.0) * w).sum(axis=1)
    if args.target_vol > 0:
        combined = vol_target(combined, args.target_vol, max_leverage=args.max_leverage)

    m = _metrics_from_returns(combined, [], "PORTFOLIO")
    okc, posc, oosc = wf_ok(combined)
    print(f"\n=== AUTO-ALLOCATED PORTFOLIO ({len(admitted)} strategies, risk-parity) ===")
    for n, wi in zip(admitted, w):
        print(f"   {wi:5.1%}  {n}")
    print(f"\n   Sharpe {m['sharpe']} | CAGR {m['cagr']:.1%} | maxDD {m['max_drawdown']:.1%} "
          f"| $PnL ${m['pnl_dollars']:,.0f} | WF {posc}/5 (OOS {oosc:+.2f})")
    if args.target_vol > 0:
        print(f"   (scaled to {args.target_vol:.0%} vol, leverage cap {args.max_leverage}x)")
    print("\nAdd new strategies to CANDIDATES; any that pass auto-join this portfolio.")
    print("SAFETY: paper-trade + human sign-off before any LIVE capital.")


if __name__ == "__main__":
    main()
