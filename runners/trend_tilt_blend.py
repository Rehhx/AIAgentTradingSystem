"""
runners/trend_tilt_blend.py
---------------------------
Find a trend-tilted blend that reaches a 10-15% annual return at the lowest
achievable drawdown. The equal-weight blend returns ~8.6%/yr at -13.9% DD; pure
50/200 trend returns ~15%/yr at -28% DD. We sweep capital tilts between them.

Uses the tuned RSI-2 params (DEPLOY_PARAMS) for the mean-reversion sleeve.
Reports each tilt's Sharpe / $PnL / CAGR / DD / win-rate, risk-gate verdict, and
70/30 + 5-fold walk-forward for the recommended tilt.

Usage:
  python runners\\trend_tilt_blend.py --universe SPY,QQQ,GLD,MSFT,JPM,GOOGL
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.daily_strategies import (
    DEPLOY_PARAMS, DEFAULT_UNIVERSE, backtest_blended,
    walk_forward_folds, split_metrics,
)
from agents.risk_agent import RiskAgent
from config import RISK, RESULTS_DIR

# candidate tilts: (trend, rsi2, donchian)
TILTS = {
    "equal_1/3":          {"trend_5020": 1/3, "rsi2_meanrev": 1/3, "donchian": 1/3},
    "trend40_rsi40_dc20": {"trend_5020": 0.40, "rsi2_meanrev": 0.40, "donchian": 0.20},
    "trend50_rsi30_dc20": {"trend_5020": 0.50, "rsi2_meanrev": 0.30, "donchian": 0.20},
    "trend50_rsi50":      {"trend_5020": 0.50, "rsi2_meanrev": 0.50, "donchian": 0.0},
    "trend60_rsi40":      {"trend_5020": 0.60, "rsi2_meanrev": 0.40, "donchian": 0.0},
    "trend70_rsi30":      {"trend_5020": 0.70, "rsi2_meanrev": 0.30, "donchian": 0.0},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default=",".join(DEFAULT_UNIVERSE))
    args = ap.parse_args()
    universe = [t.strip().upper() for t in args.universe.split(",") if t.strip()]

    print(f"\nTrend-tilted blend sweep | universe({len(universe)})={', '.join(universe)}")
    print(f"sub-strategy params: RSI-2 tuned {DEPLOY_PARAMS['rsi2_meanrev']}\n")

    risk = RiskAgent()
    hdr = (f"{'tilt':22s} {'Sharpe':>7s} {'$ PnL':>11s} {'CAGR':>7s} "
           f"{'maxDD':>7s} {'winRate':>8s} {'RISK':>6s}")
    print(hdr); print("-" * len(hdr))

    results = {}
    for name, w in TILTS.items():
        m = backtest_blended(universe, DEPLOY_PARAMS, label=name, weights=w)
        v = risk.evaluate(m)
        gate = "PASS" if v["passed"] else "FAIL"
        print(f"{name:22s} {m['sharpe']:7.2f} {m['pnl_dollars']:11,.0f} "
              f"{m['cagr']:7.1%} {m['max_drawdown']:7.1%} {m['win_rate']:8.1%} {gate:>6s}")
        results[name] = {**{k: v2 for k, v2 in m.items() if not k.startswith("_")},
                         "risk_passed": v["passed"], "risk_failures": v["failures"],
                         "split": split_metrics(m["_returns"]),
                         "walk_forward": walk_forward_folds(m["_returns"], 5),
                         "_returns_obj": m["_returns"]}

    # recommend: highest CAGR among tilts that reach >=10% CAGR, preferring the
    # smallest drawdown; fall back to best Sharpe if none reach 10%.
    reach10 = {k: r for k, r in results.items() if r["cagr"] >= 0.10}
    if reach10:
        rec = min(reach10, key=lambda k: -results[k]["cagr"] + abs(results[k]["max_drawdown"]))
        rec = max(reach10, key=lambda k: results[k]["sharpe"])
    else:
        rec = max(results, key=lambda k: results[k]["sharpe"])

    r = results[rec]
    print(f"\n=== RECOMMENDED TILT: {rec} ===")
    print(f"  weights: {r['weights']}")
    print(f"  Sharpe {r['sharpe']} | CAGR {r['cagr']:.1%} | $PnL ${r['pnl_dollars']:,.0f} "
          f"| maxDD {r['max_drawdown']:.1%} | WR {r['win_rate']:.1%}")
    print(f"  in-sample SR {r['split']['train_sharpe']} -> OOS SR {r['split']['test_sharpe']}")
    print(f"  risk gate: {'PASS' if r['risk_passed'] else 'FAIL ' + str(r['risk_failures'])}")
    print(f"  walk-forward (5 folds):")
    for f in r["walk_forward"]:
        mark = "+" if f["sharpe"] > 0 else "-"
        print(f"    [{mark}] {f.get('start','?')}..{f.get('end','?')}: "
              f"Sharpe {f['sharpe']:+.2f}, ret {f['return_pct']:+.1%}")

    out = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "universe": universe, "risk_thresholds": RISK,
        "recommended": rec,
        "tilts": {k: {kk: vv for kk, vv in r.items() if kk != "_returns_obj"}
                  for k, r in results.items()},
    }
    fp = Path(RESULTS_DIR) / "trend_tilt_blend.json"
    fp.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {fp}")


if __name__ == "__main__":
    main()
