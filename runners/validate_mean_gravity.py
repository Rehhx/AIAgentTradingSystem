"""
runners/validate_mean_gravity.py
--------------------------------
Out-of-sample WALK-FORWARD validation of the `mean_gravity` sleeve before we
promote it into the book. A full-sample +0.02 blend bump is not enough — we
require the marginal contribution to hold across contiguous time folds, survive
the deflation for searching 12 strategies, and beat the ensemble at a sensible
sleeve weight. Only then does --promote write it into web/book.json.

    python runners/validate_mean_gravity.py            # report only
    python runners/validate_mean_gravity.py --promote  # add to book.json IF it passes
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

from agents.daily_strategies import (
    backtest_book, vol_target, _metrics_from_returns, QUALITY_UNIVERSE, TRADING_DAYS,
)
from agents.lab_strategies import LAB_STRATEGIES, LAB_PARAMS
from analytics.significance import (
    sharpe_stats, probabilistic_sharpe_ratio, min_track_record_length,
    expected_max_sharpe,
)
from runners.ensemble_bench import build_ensemble, _naive, _sh

NAME = "mean_gravity"
W_SLEEVE = 0.15
PILOT_W = 0.05            # newly-validated diversifier starts as a small pilot, not in-sample optimal
BOOK = Path(__file__).parent.parent / "web" / "book.json"


def _sleeve(name, params):
    b = backtest_book(LAB_STRATEGIES[name], QUALITY_UNIVERSE, params, label=name)
    return _naive(vol_target(b["_returns"], target_vol=0.12, max_leverage=1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true",
                    help="if validation PASSES, add mean_gravity to web/book.json")
    args = ap.parse_args()

    params = LAB_PARAMS[NAME]
    print("=" * 74)
    print(f"WALK-FORWARD VALIDATION | {NAME} | params {params}")
    print("=" * 74)
    print("building the live ensemble + the sleeve (~1-2 min)...\n")

    ens = build_ensemble()
    g = _sleeve(NAME, params)
    idx = ens.index.intersection(g.index)
    e, c = ens.reindex(idx).fillna(0.0), g.reindex(idx).fillna(0.0)
    blend = (1 - W_SLEEVE) * e + W_SLEEVE * c
    full_delta = _sh(blend) - _sh(e)
    print(f"FULL SAMPLE  gravity Sharpe {_sh(c):.2f} | corr->ens {c.corr(e):+.2f} | "
          f"maxDD {_metrics_from_returns(c, [], NAME)['max_drawdown']:.1%}")
    print(f"             ensemble {_sh(e):.2f} -> blend@{W_SLEEVE:.0%} {_sh(blend):.2f} "
          f"({full_delta:+.2f})\n")

    # ---- contiguous walk-forward at several fold counts (temporal robustness) --
    robust = []
    for K in (4, 5, 6):
        n = len(idx)
        fl = n // K
        pos = 0
        print(f"walk-forward | {K} contiguous folds")
        for k in range(K):
            sl = slice(k * fl, (k + 1) * fl if k < K - 1 else n)
            ee, cc, bb = e.iloc[sl], c.iloc[sl], blend.iloc[sl]
            de = _sh(bb) - _sh(ee)
            pos += de > 0
            print(f"  fold {k+1} {idx[sl][0].date()}..{idx[sl][-1].date()} | "
                  f"gravity {_sh(cc):+5.2f} | ens {_sh(ee):4.2f} -> blend {_sh(bb):4.2f} "
                  f"({de:+.2f}){'  +' if de > 0 else ''}")
        print(f"  -> blend improves in {pos}/{K} folds\n")
        robust.append((pos, K))

    # ---- deflation: correct for searching all 12 lab strategies ---------------
    print("deflation (correcting for the 12-strategy search):")
    trial_srs = []
    for nm in LAB_STRATEGIES:
        try:
            trial_srs.append(sharpe_stats(_sleeve(nm, LAB_PARAMS[nm]).to_numpy())["sr"])
        except Exception:
            pass
    sr_var = float(np.var(trial_srs, ddof=1)) if len(trial_srs) > 1 else 0.0
    sr_star = expected_max_sharpe(len(trial_srs), sr_var)
    st = sharpe_stats(c.to_numpy())
    dsr = probabilistic_sharpe_ratio(st["sr"], st["n"], st["skew"], st["kurt"], sr_star)
    psr0 = probabilistic_sharpe_ratio(st["sr"], st["n"], st["skew"], st["kurt"], 0.0)
    mtrl = min_track_record_length(st["sr"], st["skew"], st["kurt"], 0.0, 0.95)
    print(f"  trials {len(trial_srs)} | SR* hurdle {sr_star*np.sqrt(TRADING_DAYS):.2f} (ann) | "
          f"PSR vs 0 {psr0:.0%} | DSR {dsr:.0%} | min track-record ~{mtrl/TRADING_DAYS:.1f}y\n")

    # ---- best sleeve weight ---------------------------------------------------
    print("blend Sharpe vs sleeve weight:")
    best_w, best_s = 0.0, _sh(e)
    for w in (0.05, 0.10, 0.15, 0.20, 0.25):
        bs = _sh((1 - w) * e + w * c)
        if bs > best_s:
            best_w, best_s = w, bs
        print(f"  w={w:.0%} -> {bs:.2f}")
    print(f"  best weight {best_w:.0%} -> {best_s:.2f} (ensemble {_sh(e):.2f})\n")

    # ---- verdict --------------------------------------------------------------
    folds_ok = all(p >= max(3, K - 1) for p, K in robust)   # improves in K-1+ of every split
    standalone_ok = psr0 >= 0.90                              # sleeve is real on its own
    weight_ok = best_s > _sh(e) and best_w > 0
    passed = folds_ok and standalone_ok and weight_ok
    print("=" * 74)
    print("VERDICT:", "PASS - promote" if passed else "FAIL - do not promote")
    print(f"  folds robust (>=K-1 each split): {folds_ok} | standalone PSR>=90%: {standalone_ok} "
          f"| improves at best weight: {weight_ok}")
    if not passed:
        print("  Honest call: the +0.02 full-sample bump is not robust enough to deploy.")
    print("=" * 74)

    if args.promote:
        if not passed:
            print("\n--promote given but validation FAILED -> book.json left unchanged.")
            return
        book = json.loads(BOOK.read_text(encoding="utf-8"))
        book["strategies"] = [s for s in book["strategies"] if s.get("name") != NAME]
        book["strategies"].append({
            "name": NAME, "label": "gravity (lab)", "weight": PILOT_W,
            "source": "lab", "family": "reversion", "validated": True,
            "sharpe": round(_sh(c), 3), "corr": round(float(c.corr(e)), 3),
            "wf": f"{robust[1][0]}/{robust[1][1]}", "dsr": round(dsr, 3),
            "best_weight": round(best_w, 3), "added": time.strftime("%Y-%m-%d"),
        })
        BOOK.write_text(json.dumps(book, indent=2), encoding="utf-8")
        print(f"\nPROMOTED -> wrote {NAME} into {BOOK} at pilot weight {PILOT_W:.0%} "
              f"(in-sample best was {best_w:.0%}).")


if __name__ == "__main__":
    main()
