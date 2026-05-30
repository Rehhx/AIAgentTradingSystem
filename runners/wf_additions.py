"""
runners/wf_additions.py
-----------------------
Full backtest + walk-forward of the two candidate additions vs the current book:
  - portfolio_full (deployed baseline)
  - portfolio_full + lowvol_factor @ 10%
  - portfolio_full + crypto_trend  @ 5%
Same overlays as live (vol-target 17% / 1.8x, early-warning, BIL cash yield),
$100k base, 6 bps, adjusted data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

from agents.daily_strategies import (
    _metrics_from_returns, walk_forward_folds, split_metrics,
)
from runners.diversifier_screen import build_base, overlays, W
from runners.final_tests import crypto_trend, lowvol_factor


def report(label, r):
    m = _metrics_from_returns(r, [], label)
    s = split_metrics(r)
    print(f"\n=== {label} ===")
    print(f"  $100,000 -> ${m['final_capital']:,.0f}   (+${m['pnl_dollars']:,.0f}, {m['total_return']*100:.0f}%)")
    print(f"  CAGR {m['cagr']:.1%} | Sharpe {m['sharpe']} | max DD {m['max_drawdown']:.1%} | win-rate {m['win_rate']:.0%}")
    print(f"  in-sample (70%) Sharpe {s['train_sharpe']:+.2f} -> out-of-sample (30%) Sharpe {s['test_sharpe']:+.2f}")
    print("  walk-forward (5 contiguous folds):")
    pos = 0
    for f in walk_forward_folds(r, 5):
        mark = "+" if f["sharpe"] > 0 else "-"; pos += f["sharpe"] > 0
        print(f"    [{mark}] {f.get('start','?')[:7]}..{f.get('end','?')[:7]}: "
              f"Sharpe {f['sharpe']:+.2f}, return {f['return_pct']:+.1%}")
    print(f"    -> positive in {pos}/5 folds")


def main():
    print("building book + both additions (scans S&P 500; ~1-2 min) ...")
    panel = build_base()
    base_combo = sum(panel[c].fillna(0) * W[c] for c in W)
    idx = panel.index

    lv = lowvol_factor().reindex(idx).fillna(0)
    cr = crypto_trend().reindex(idx).fillna(0)

    report("portfolio_full (deployed baseline)", overlays(base_combo, idx))
    report("portfolio_full + lowvol_factor @ 10%", overlays(base_combo * 0.90 + lv * 0.10, idx))
    report("portfolio_full + crypto_trend @ 5%", overlays(base_combo * 0.95 + cr * 0.05, idx))


if __name__ == "__main__":
    main()
