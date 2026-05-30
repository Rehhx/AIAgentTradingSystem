"""
runners/full_backtest.py
------------------------
Definitive backtest of the DEPLOYED book (portfolio_full, 7 sleeves): per-sleeve
standalone metrics, the combined book's PnL / Sharpe / drawdown, in-sample ->
out-of-sample split, and 5-fold walk-forward. Overlays match live exactly:
vol-target 17% (<=1.8x conditional leverage), early-warning de-risk, idle -> BIL.
$100k base, 6 bps round-trip, split/dividend-adjusted daily data, 2016-2026.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

from agents.daily_strategies import _metrics_from_returns, walk_forward_folds, split_metrics
from runners.diversifier_screen import build_base, overlays, W
from runners.final_tests import lowvol_factor
from runners.lowvol_defensive import make_defensive

SLEEVE_NAMES = {"rsi": "rsi2_meanrev", "don": "donchian", "trd": "trend_5020",
                "xs": "xs_dualmom", "rec": "recovery", "pead": "pead"}


def report(label, r):
    m = _metrics_from_returns(r, [], label)
    s = split_metrics(r)
    print(f"\n=== {label} ===")
    print(f"  $100,000 -> ${m['final_capital']:,.0f}   (+${m['pnl_dollars']:,.0f}, {m['total_return']*100:.0f}%)")
    print(f"  CAGR {m['cagr']:.1%} | Sharpe {m['sharpe']} | max DD {m['max_drawdown']:.1%}")
    print(f"  in-sample (70%) Sharpe {s['train_sharpe']:+.2f} -> out-of-sample (30%) Sharpe {s['test_sharpe']:+.2f}")
    print("  walk-forward (5 contiguous folds):")
    pos = 0
    for f in walk_forward_folds(r, 5):
        mark = "+" if f["sharpe"] > 0 else "-"; pos += f["sharpe"] > 0
        print(f"    [{mark}] {f.get('start','?')[:7]}..{f.get('end','?')[:7]}: "
              f"Sharpe {f['sharpe']:+.2f}, return {f['return_pct']:+.1%}")
    print(f"    -> positive in {pos}/5 folds")
    return m


def main():
    print("building the deployed 7-sleeve book (scans S&P 500 a few times; ~1-2 min) ...")
    panel = build_base()
    idx = panel.index
    lvd = make_defensive(lowvol_factor()).reindex(idx).fillna(0)

    print("\n-- per-sleeve standalone (Sharpe / CAGR / max DD) --")
    for k, nm in SLEEVE_NAMES.items():
        m = _metrics_from_returns(panel[k].fillna(0), [], nm)
        print(f"  {nm:14s} {m['sharpe']:5.2f} | {m['cagr']:6.1%} | {m['max_drawdown']:6.1%}")
    ml = _metrics_from_returns(lvd, [], "lowvol_def")
    print(f"  {'lowvol_def':14s} {ml['sharpe']:5.2f} | {ml['cagr']:6.1%} | {ml['max_drawdown']:6.1%}")

    base = sum(panel[c].fillna(0) * W[c] for c in W)
    book = overlays(base * 0.90 + lvd * 0.10, idx)
    report("DEPLOYED portfolio_full (7 sleeves, vol-target 17% / 1.8x)", book)


if __name__ == "__main__":
    main()
