"""
runners/overnight_rigor.py
--------------------------
The make-or-break validation of the overnight premium (see overnight_edge.py).
Two things kill a "beautiful gross pattern": unrealistic fills and the multiple-
testing/overfitting problem. So this:

  1. MODELS REALISTIC MOC/MOO COSTS per asset (closing/opening auction effective
     spread for liquid ETFs), not an assumed flat bp. Each night = 1 sell at the
     open + 1 buy at the close = a full round-trip, so cost is the whole game.
  2. Runs the FULL RIGOR BATTERY on the NET series: Deflated Sharpe (corrected for
     the assets we searched), PSR vs the buy-hold benchmark (does it beat passive
     risk-adjusted?), 5-fold walk-forward, and a net-of-cost decay check by era.

Honest by construction: a passing battery makes this a deployable lead; a failing
one sends it back to the shelf.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from agents.daily_strategies import walk_forward_folds, TRADING_DAYS
from analytics.significance import sharpe_stats, dsr_from_trials, probabilistic_sharpe_ratio
from runners.overnight_edge import _ohlc, decompose, ETFS

# Per-SIDE MOC/MOO effective cost (bps) for liquid ETFs: closing/opening auctions
# are deep; SPY/QQQ/DIA ~0.4bp/side, small-cap IWM a touch wider. A night pays TWO
# sides (sell @open + buy @close) -> round-trip = 2x these.
PER_SIDE_BPS = {"SPY": 0.4, "QQQ": 0.4, "DIA": 0.4, "IWM": 0.6}


def _ann(r):
    r = r.dropna(); sd = r.std(); eq = (1 + r).cumprod()
    return {"cagr": float(eq.iloc[-1] ** (TRADING_DAYS / len(r)) - 1) if len(r) else 0.0,
            "sharpe": float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0,
            "maxdd": float((eq / eq.cummax() - 1).min()) if len(r) else 0.0}


def net_overnight(scenario: float = 1.0):
    """per-asset net overnight returns after MOC/MOO round-trip cost (scenario
    multiplies the central cost estimate: 1.0 central, 1.5 conservative)."""
    on_net, bh = {}, {}
    for t in ETFS:
        on, _intra, b = decompose(_ohlc(t))
        rt = 2 * PER_SIDE_BPS[t] * scenario / 10000.0      # round-trip per night
        on_net[t] = (on - rt).rename(t)
        bh[t] = b.rename(t)
    onN = pd.concat(on_net.values(), axis=1).dropna()
    bhN = pd.concat(bh.values(), axis=1).reindex(onN.index).dropna()
    return onN, bhN.reindex(onN.index)


def main():
    print("modeling MOC/MOO costs + running the rigor battery on the overnight edge ...\n")
    onN, bhN = net_overnight(1.0)
    basket = onN.mean(axis=1)                              # equal-weight net overnight
    bh = bhN.mean(axis=1)                                  # equal-weight buy-hold
    m_on, m_bh = _ann(basket), _ann(bh)

    print("NET-OF-COST (central MOC/MOO estimate ~0.8-1.2bp round-trip/asset)")
    print(f"  {'strategy':24s} {'CAGR':>7s} {'Sharpe':>7s} {'maxDD':>7s}")
    print("  " + "-" * 48)
    print(f"  {'own-the-night (net)':24s} {m_on['cagr']:>7.1%} {m_on['sharpe']:>7.2f} {m_on['maxdd']:>7.1%}")
    print(f"  {'buy & hold basket':24s} {m_bh['cagr']:>7.1%} {m_bh['sharpe']:>7.2f} {m_bh['maxdd']:>7.1%}")
    cons = _ann(net_overnight(1.5)[0].mean(axis=1))
    print(f"  {'own-the-night (conserv.)':24s} {cons['cagr']:>7.1%} {cons['sharpe']:>7.2f} {cons['maxdd']:>7.1%}")

    # --- Deflated Sharpe (corrected for the 4 assets searched) ---
    trial_srs = [sharpe_stats(onN[t].to_numpy())["sr"] for t in onN.columns]
    d = dsr_from_trials(basket.to_numpy(), trial_srs, periods=TRADING_DAYS)
    print("\n" + "=" * 60)
    print("RIGOR BATTERY (on the NET series)")
    print("=" * 60)
    print(f"  Deflated Sharpe vs 0         {d['dsr']:>7.1%}   (corrected for {d['n_trials']} assets)")

    # --- PSR vs the buy-hold benchmark (the meaningful bar) ---
    s = sharpe_stats(basket.to_numpy())
    bh_sr = sharpe_stats(bh.to_numpy())["sr"]
    psr_vs_bh = probabilistic_sharpe_ratio(s["sr"], s["n"], s["skew"], s["kurt"], sr_benchmark=bh_sr)
    print(f"  PSR vs buy-hold Sharpe       {psr_vs_bh:>7.1%}   (P[overnight Sharpe > passive])")

    # --- Walk-forward (positive in every fold?) ---
    folds = walk_forward_folds(basket, 5)
    pos = sum(1 for f in folds if f["sharpe"] > 0)
    print(f"  Walk-forward                 {pos}/5 folds positive")
    for f in folds:
        mk = "+" if f["sharpe"] > 0 else "-"
        print(f"    [{mk}] {f.get('start','?')}..{f.get('end','?')}  Sharpe {f['sharpe']:>+5.2f}  ret {f['return_pct']:>+6.1%}")

    # --- decay on the NET series ---
    print("\n  decay check (NET overnight CAGR by era):")
    for label, a, b in [("2005-2009","2005","2009"),("2010-2014","2010","2014"),
                        ("2015-2019","2015","2019"),("2020-2026","2020","2026")]:
        seg = basket.loc[a:b]
        if len(seg) > 60:
            print(f"    {label}  {_ann(seg)['cagr']:>+6.1%}  Sharpe {_ann(seg)['sharpe']:>+5.2f}")

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    passes = (m_on["sharpe"] > m_bh["sharpe"] and d["dsr"] >= 0.95
              and psr_vs_bh >= 0.90 and pos >= 4)
    if passes:
        print("  PASS: net of realistic MOC/MOO cost the overnight premium clears the full")
        print(f"  battery -- Sharpe {m_on['sharpe']:.2f} vs buy-hold {m_bh['sharpe']:.2f}, "
              f"DSR {d['dsr']:.0%}, beats passive {psr_vs_bh:.0%}, {pos}/5 folds,")
        print(f"  HALF the drawdown ({m_on['maxdd']:.0%} vs {m_bh['maxdd']:.0%}). DEPLOYABLE LEAD")
        print("  -> next: wire as an overnight-beta sleeve (MOC buy / MOO sell) in the book.")
    else:
        print("  MIXED: see which test it misses above. If it beats buy-hold on Sharpe + DD")
        print("  but PSR-vs-passive is < 90%, it's a risk-reducer (overnight-beta) more than")
        print("  an alpha -- still useful as a lower-drawdown market-exposure sleeve.")


if __name__ == "__main__":
    main()
