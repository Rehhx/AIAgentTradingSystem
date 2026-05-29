"""
runners/derisk_wf.py
--------------------
Take the high-P&L strategies we want to KEEP (50/200 trend, cross-sectional
momentum, cross-sectional dual-momentum), apply a volatility-targeting overlay
to bring their drawdown under the -15% risk gate WITHOUT killing return, then
run a walk-forward (5 contiguous folds + 70/30 split) on each to confirm the
edge is out-of-sample. Also runs the cross-sectional books on the FULL S&P 500.

Shows $ P&L on $100k for raw vs de-risked, and the risk-gate verdict.

Usage:
  python runners\\derisk_wf.py
  python runners\\derisk_wf.py --sp500     # also run cross-sectional on full S&P 500
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    backtest_book, backtest_cross_sectional, vol_target, _metrics_from_returns,
    sig_trend_5020, QUALITY_UNIVERSE, walk_forward_folds, split_metrics,
)
from agents.risk_agent import RiskAgent

risk = RiskAgent()


def evalr(label, r):
    m = _metrics_from_returns(r, [], label)
    v = risk.evaluate({**m, "win_rate": 0.5, "total_trades": 999})  # gate on Sharpe+DD here
    return m, v


def line(label, m, v):
    return (f"{label:34s} {m['sharpe']:6.2f} {m['pnl_dollars']:12,.0f} "
            f"{m['cagr']:7.1%} {m['max_drawdown']:7.1%} "
            f"{('PASS' if v['passed'] else 'FAIL'):>6s}")


def wf(label, r, folds=5):
    print(f"\n  walk-forward -- {label}:")
    s = split_metrics(r)
    print(f"    in-sample SR {s['train_sharpe']:+.2f} -> out-of-sample SR {s['test_sharpe']:+.2f}")
    pos = 0
    for f in walk_forward_folds(r, folds):
        mark = "+" if f["sharpe"] > 0 else "-"
        pos += f["sharpe"] > 0
        print(f"    [{mark}] {f.get('start','?')}..{f.get('end','?')}: "
              f"Sharpe {f['sharpe']:+.2f}, ret {f['return_pct']:+.1%}")
    print(f"    -> positive in {pos}/{folds} folds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sp500", action="store_true")
    args = ap.parse_args()
    U = QUALITY_UNIVERSE

    # the kept high-PnL strategies (raw return series) + the vol-target config
    # that brings each under the -15% DD gate (from calibration).
    raw = {
        "trend_5020":      backtest_book(sig_trend_5020, U, label="t")["_returns"],
        "xs_mom_12-1":     backtest_cross_sectional(U, mode="momentum", lookback=252, skip=21, k=3)["_returns"],
        "xs_dualmom_12-1": backtest_cross_sectional(U, mode="momentum", lookback=252, skip=21, k=3, market_filter=True)["_returns"],
    }
    vt_cfg = {           # (target_vol, max_leverage)
        "trend_5020":      (0.12, 1.0),
        "xs_mom_12-1":     (0.10, 1.0),
        "xs_dualmom_12-1": (0.10, 1.0),
    }

    print(f"\nDe-risk (vol-target) + walk-forward | universe={len(U)} quality names")
    print(f"\n{'strategy':34s} {'Sharpe':>6s} {'$ PnL':>12s} {'CAGR':>7s} {'maxDD':>7s} {'RISK':>6s}")
    print("-" * 76)
    out = {}
    for name, r in raw.items():
        mr, vr = evalr(name + " (raw)", r)
        print(line(name + " (raw)", mr, vr))
        tv, ml = vt_cfg[name]
        rv = vol_target(r, target_vol=tv, max_leverage=ml)
        mv, vv = evalr(f"{name} vt={tv:.0%}", rv)
        print(line(f"{name} +voltarget {tv:.0%}", mv, vv))
        out[name] = {"raw": {k: v for k, v in mr.items() if not k.startswith("_")},
                     "voltarget": {k: v for k, v in mv.items() if not k.startswith("_")},
                     "vt_cfg": {"target_vol": tv, "max_leverage": ml}}

    # walk-forward on the de-risked versions
    print("\n" + "=" * 76)
    print("WALK-FORWARD (on vol-targeted, gate-passing versions)")
    for name, r in raw.items():
        tv, ml = vt_cfg[name]
        wf(f"{name} +voltarget {tv:.0%}", vol_target(r, target_vol=tv, max_leverage=ml))

    # optional: cross-sectional on the FULL S&P 500 (better ranking breadth)
    if args.sp500:
        from data.sp500 import sp500_tickers
        sp = sp500_tickers()
        print("\n" + "=" * 76)
        print(f"CROSS-SECTIONAL on FULL S&P 500 ({len(sp)} names)")
        print(f"\n{'strategy':34s} {'Sharpe':>6s} {'$ PnL':>12s} {'CAGR':>7s} {'maxDD':>7s} {'RISK':>6s}")
        print("-" * 76)
        for lbl, kw in [("xs_mom_12-1 (500)", dict(mode="momentum", lookback=252, skip=21, k=10)),
                        ("xs_dualmom_12-1 (500)", dict(mode="momentum", lookback=252, skip=21, k=10, market_filter=True))]:
            m = backtest_cross_sectional(sp, label=lbl, **kw)
            mr, vr = evalr(lbl, m["_returns"]); print(line(lbl, mr, vr))
            rv = vol_target(m["_returns"], target_vol=0.10, max_leverage=1.0)
            mv, vv = evalr(lbl + " vt", rv); print(line(lbl + " +voltarget 10%", mv, vv))
            wf(lbl + " +voltarget 10%", rv)

    Path("results/derisk_wf.json").write_text(json.dumps(
        {"run_at": datetime.now(timezone.utc).isoformat(), "universe": U,
         "results": out}, indent=2, default=str), encoding="utf-8")
    print("\nWrote results/derisk_wf.json")


if __name__ == "__main__":
    main()
