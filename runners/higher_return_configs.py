"""
runners/higher_return_configs.py
--------------------------------
The honest return/risk frontier for "bigger numbers". There is no hidden alpha
left on this universe (validated ~26 ways), so more return = more risk via the
only two real levers: a crypto sleeve and more leverage. This prices each option
with full walk-forward so the trade-off is explicit. $100k, adjusted data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

from agents.daily_strategies import _metrics_from_returns, walk_forward_folds
from runners.diversifier_screen import build_base, overlays, W
from runners.final_tests import lowvol_factor, crypto_trend
from runners.lowvol_defensive import make_defensive

GATE_DD = -0.15


def line(label, r):
    m = _metrics_from_returns(r, [], label)
    folds = walk_forward_folds(r, 5)
    pos = sum(1 for f in folds if f["sharpe"] > 0)
    lean = next((f["return_pct"] for f in folds if f.get("start", "").startswith("2018")), float("nan"))
    gate = "PASS" if m["max_drawdown"] >= GATE_DD else "FAIL"
    print(f"  {label:32s} ${m['pnl_dollars']:>9,.0f} | {m['cagr']:5.1%} | {m['sharpe']:.2f} | "
          f"{m['max_drawdown']:6.1%} | 2018-20 {lean:+5.1%} | {pos}/5 | {gate}")


def main():
    print("building book + crypto, pricing the higher-return frontier ...\n")
    panel = build_base()
    idx = panel.index
    lvd = make_defensive(lowvol_factor()).reindex(idx).fillna(0)
    book = sum(panel[c].fillna(0) * W[c] for c in W) * 0.90 + lvd * 0.10
    cr = crypto_trend().reindex(idx).fillna(0)

    print(f"  {'configuration':32s} {'$PnL':>11s} | {'CAGR':>5s} | Shrp | {'maxDD':>6s} | {'lean yrs':>8s} | WF  | gate")
    print("  " + "-" * 104)
    line("DEPLOYED (1.8x, no crypto)", overlays(book, idx, vt=0.17, maxlev=1.8))
    line("+ crypto 5%", overlays(book * 0.95 + cr * 0.05, idx, vt=0.17, maxlev=1.8))
    line("leverage 2.0x (no crypto)", overlays(book, idx, vt=0.19, maxlev=2.0))
    line("+ crypto 5% + leverage 2.0x", overlays(book * 0.95 + cr * 0.05, idx, vt=0.19, maxlev=2.0))
    line("+ crypto 8% + leverage 2.0x", overlays(book * 0.92 + cr * 0.08, idx, vt=0.19, maxlev=2.0))

    print("\n  Honest notes:")
    print("  - crypto's standalone CAGR (45%) is front-loaded in the 2017 BTC bull; do NOT")
    print("    extrapolate it forward. Its OUT-OF-SAMPLE Sharpe contribution is weaker/recent.")
    print("  - crypto is a GOVERNANCE decision (a traditional board may forbid it) and trades")
    print("    24/7 with -50%+ drawdowns; only the small sizing keeps the book inside the gate.")
    print("  - leverage amplifies a sudden gap-down ~linearly; the gate-passing DD assumes the")
    print("    de-risk overlays react in time, which a 1-day crash can outrun.")


if __name__ == "__main__":
    main()
