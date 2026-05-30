"""
runners/weight_optimize.py
--------------------------
Are the deployed sleeve weights any good, or just hand-picked? This re-derives
them with a WALK-FORWARD optimizer (anchored/expanding): on each training window
find the long-only weights that maximize Sharpe (capped per sleeve to avoid
concentration), then score them OUT-OF-SAMPLE on the next block. Compare the
optimized weights' OOS performance to the current hand-set weights.

Honest test: if the optimizer doesn't beat the hand weights out-of-sample, that
VALIDATES the deployed mix (and shows Sharpe-maxing overfits). If it does, we have
a robustly better weighting to adopt. $100k base, adjusted data, 7 sleeves.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from agents.daily_strategies import _metrics_from_returns, walk_forward_folds, TRADING_DAYS
from runners.diversifier_screen import build_base, overlays, W
from runners.final_tests import lowvol_factor
from runners.lowvol_defensive import make_defensive

ORDER = ["rsi", "don", "trd", "xs", "rec", "pead", "lowvol"]
W_CUR = np.array([0.252, 0.198, 0.126, 0.072, 0.162, 0.090, 0.10])   # deployed weights
CAP = 0.35                                                            # max per sleeve (anti-concentration)


def neg_sharpe(w, R):
    r = R @ w
    sd = r.std()
    return 0.0 if sd == 0 else -(r.mean() / sd * np.sqrt(TRADING_DAYS))


def optimize(R):
    n = R.shape[1]
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    res = minimize(neg_sharpe, np.full(n, 1.0 / n), args=(R,), method="SLSQP",
                   bounds=[(0.0, CAP)] * n, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-9})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def oos_sharpe(r):
    sd = r.std()
    return 0.0 if sd == 0 else float(r.mean() / sd * np.sqrt(TRADING_DAYS))


def main():
    print("building 7 sleeves + walk-forward weight optimization ...\n")
    panel = build_base()
    idx = panel.index
    lvd = make_defensive(lowvol_factor()).reindex(idx).fillna(0)
    P = pd.concat([panel[["rsi", "don", "trd", "xs", "rec", "pead"]], lvd.rename("lowvol")], axis=1).fillna(0)
    Rall = P.to_numpy()
    n = len(Rall)

    # anchored expanding folds: 4 OOS blocks (test = blocks 2..5, train = all prior)
    edges = [int(n * k / 5) for k in range(6)]
    print(f"{'OOS block':18s} {'opt OOS Sharpe':>15s} {'current OOS Sharpe':>19s}  winner")
    print("-" * 70)
    opt_oos, cur_oos, opt_weights = [], [], []
    for i in range(1, 5):
        tr = Rall[:edges[i]]
        te = Rall[edges[i]:edges[i + 1]]
        w = optimize(tr)
        opt_weights.append(w)
        so, sc = oos_sharpe(te @ w), oos_sharpe(te @ W_CUR)
        opt_oos.append(so); cur_oos.append(sc)
        lab = f"{P.index[edges[i]].date()}..{P.index[edges[i+1]-1].date()}"
        print(f"{lab:18s} {so:15.2f} {sc:19.2f}  {'optimized' if so > sc else 'current'}")

    print(f"\n  avg OOS Sharpe: optimized {np.mean(opt_oos):.2f}  vs  current {np.mean(cur_oos):.2f}")
    avg_w = np.mean(opt_weights, axis=0)
    print(f"\n  avg optimizer weights vs deployed:")
    print(f"    {'sleeve':10s} {'optimized':>10s} {'deployed':>9s}")
    for s, wo, wc in zip(ORDER, avg_w, W_CUR):
        print(f"    {s:10s} {wo:10.1%} {wc:9.1%}")

    # full-period book with each weighting (+ live overlays)
    def book(w):
        return overlays(pd.Series(Rall @ w, index=idx), idx)
    mc = _metrics_from_returns(book(W_CUR), [], "current")
    mo = _metrics_from_returns(book(avg_w), [], "optimized")
    print(f"\n  full-period book (+ overlays):")
    print(f"    current  : Sharpe {mc['sharpe']:.2f} | CAGR {mc['cagr']:.1%} | DD {mc['max_drawdown']:.1%}")
    print(f"    optimized: Sharpe {mo['sharpe']:.2f} | CAGR {mo['cagr']:.1%} | DD {mo['max_drawdown']:.1%}")

    verdict = ("optimizer wins OOS -> consider adopting"
               if np.mean(opt_oos) > np.mean(cur_oos) + 0.05
               else "hand weights hold up OOS -> deployed mix validated (Sharpe-max overfits)")
    print(f"\n  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
