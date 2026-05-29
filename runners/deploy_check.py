"""
runners/deploy_check.py
-----------------------
Pre-deployment gate. For every deployable strategy/book it reports $ P&L on
$100k, the RISK gate (Sharpe >= 0.8, max DD >= -15%), and the WALK-FORWARD
verdict (out-of-sample Sharpe > 0 and positive in >= 4/5 contiguous folds).

Then it builds the ENSEMBLE that utilizes every strategy that passed — the exact
combination the live rebalancer trades for `--book blended_plus --xs-universe
sp500 --vol-target 0.12`: equal-weight RSI-2 + Donchian + 50/200 trend (quality-10)
+ cross-sectional dual-momentum (full S&P 500), with a 12% vol-target overlay —
and reports its PnL / risk / walk-forward so you know the deployed book passes.

Usage:
  python runners\\deploy_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    sig_rsi2_meanrev, sig_donchian, sig_trend_5020, DEPLOY_PARAMS, QUALITY_UNIVERSE,
    backtest_book, backtest_cross_sectional, vol_target, _metrics_from_returns,
    walk_forward_folds, split_metrics,
)

U = QUALITY_UNIVERSE


def gate_ok(m) -> bool:
    return m["sharpe"] >= 0.8 and m["max_drawdown"] >= -0.15


def wf_verdict(r, folds=5):
    s = split_metrics(r)
    fl = walk_forward_folds(r, folds)
    pos = sum(1 for f in fl if f["sharpe"] > 0)
    passed = s["test_sharpe"] > 0 and pos >= folds - 1
    return passed, pos, s["test_sharpe"], fl


def show(rows):
    print(f"{'strategy':30s} {'Sharpe':>6s} {'$ PnL':>12s} {'CAGR':>7s} {'maxDD':>7s} "
          f"{'RISK':>5s} {'WF(folds/OOS)':>15s}")
    print("-" * 86)
    for name, m, r in rows:
        gate = "PASS" if gate_ok(m) else "FAIL"
        wp, pos, oos, _ = wf_verdict(r)
        wf = f"{'PASS' if wp else 'FAIL'} {pos}/5 {oos:+.2f}"
        print(f"{name:30s} {m['sharpe']:6.2f} {m['pnl_dollars']:12,.0f} {m['cagr']:7.1%} "
              f"{m['max_drawdown']:7.1%} {gate:>5s} {wf:>15s}")


def main():
    print(f"\nPRE-DEPLOY CHECK | quality-10 + full-S&P-500 cross-sectional | $100k, adjusted, 6bps\n")

    # component return series
    r_rsi = backtest_book(sig_rsi2_meanrev, U, DEPLOY_PARAMS["rsi2_meanrev"], label="rsi")["_returns"]
    r_don = backtest_book(sig_donchian, U, None, label="don")["_returns"]
    r_trd = backtest_book(sig_trend_5020, U, None, label="trd")["_returns"]
    print("  (running full-S&P-500 cross-sectional sleeve ...)")
    from data.sp500 import sp500_tickers
    r_xs = backtest_cross_sectional(sp500_tickers(), mode="momentum", lookback=252,
                                    skip=21, k=10, market_filter=True, label="xs500")["_returns"]

    def M(r, lbl): return _metrics_from_returns(r, [], lbl)

    # --- individual strategies (raw + vol-targeted where needed) ---
    print("== individual strategies ==")
    rows = [
        ("rsi2_meanrev", M(r_rsi, "rsi"), r_rsi),
        ("donchian", M(r_don, "don"), r_don),
        ("trend_5020 (raw)", M(r_trd, "trd"), r_trd),
        ("trend_5020 +voltarget", M(vol_target(r_trd, 0.12), "trdvt"), vol_target(r_trd, 0.12)),
        ("xs_dualmom_500 (raw)", M(r_xs, "xs"), r_xs),
        ("xs_dualmom_500 +voltarget", M(vol_target(r_xs, 0.10), "xsvt"), vol_target(r_xs, 0.10)),
    ]
    show(rows)

    # --- the ENSEMBLE that uses everything that passed (= live blended_plus) ---
    print("\n== ENSEMBLE (all passing strategies; = live --book blended_plus --xs-universe sp500 --vol-target 0.12) ==")
    panel = pd.concat([r_rsi, r_don, r_trd, r_xs], axis=1, sort=True)
    panel.columns = ["rsi", "don", "trd", "xs"]
    raw_ens = panel.mean(axis=1)                       # equal-weight the 4 sleeves
    vt_ens = vol_target(raw_ens, target_vol=0.12, max_leverage=1.0)
    rows2 = [
        ("ensemble (raw, equal-weight)", M(raw_ens, "ens"), raw_ens),
        ("ENSEMBLE +voltarget 12% (DEPLOY)", M(vt_ens, "ensvt"), vt_ens),
    ]
    show(rows2)

    me = M(vt_ens, "ensvt")
    wp, pos, oos, fl = wf_verdict(vt_ens)
    print(f"\n=== DEPLOY BOOK: ensemble + vol-target 12% ===")
    print(f"  $ PnL ${me['pnl_dollars']:,.0f} on $100k | CAGR {me['cagr']:.1%} | "
          f"Sharpe {me['sharpe']} | maxDD {me['max_drawdown']:.1%}")
    print(f"  RISK gate: {'PASS' if gate_ok(me) else 'FAIL'}   "
          f"WALK-FORWARD: {'PASS' if wp else 'FAIL'} (positive {pos}/5 folds, OOS Sharpe {oos:+.2f})")
    for f in fl:
        mk = "+" if f["sharpe"] > 0 else "-"
        print(f"    [{mk}] {f.get('start','?')}..{f.get('end','?')}: Sharpe {f['sharpe']:+.2f}, ret {f['return_pct']:+.1%}")
    print(f"\n  deploy: python runners\\daily_rebalance.py --book blended_plus "
          f"--xs-universe sp500 --vol-target 0.12 --live")


if __name__ == "__main__":
    main()
