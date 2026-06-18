"""
runners/agent_lab.py
--------------------
The autonomous 12-AGENT strategy lab. Each agent owns one ORIGINAL mechanism
(agents/lab_strategies.py) and walks it through the full desk loop:

    research  -> state the first-principles hypothesis
    build     -> compile the signal (long/flat, shift=1 - no look-ahead)
    validate  -> standalone Sharpe + maxDD, correlation to the live ensemble,
                 marginal contribution to an 85/15 blend, walk-forward folds,
                 and a deflated Sharpe that corrects for searching 12 trials
    execute   -> DRY-RUN only: what a 15% paper sleeve would allocate. No live
                 order is ever placed from here.

The bar is OUR equity ensemble book (Sharpe ~1.59), not buy-&-hold SPY. A
candidate wins by RAISING the blended Sharpe (decorrelation), not by posting a
big standalone number. Every candidate gets a recommendation - PROMOTE / REVIEW
/ REJECT - and the whole set is written to web/candidates.json so a human can
approve or disapprove each one in the cockpit. The machine proposes; the human
disposes.

    python runners/agent_lab.py                 # human-readable, streams to stdout
    python runners/agent_lab.py --emit web/candidates.json
"""
import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings

warnings.filterwarnings("ignore")

try:                                  # never die on a glyph the console can't map
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    backtest_book, vol_target, walk_forward_folds, _metrics_from_returns,
    daily_bars, QUALITY_UNIVERSE, TRADING_DAYS,
)
from agents.lab_strategies import LAB_STRATEGIES, LAB_AGENTS, sample_params
from analytics.significance import (
    sharpe_stats, expected_max_sharpe, probabilistic_sharpe_ratio,
)
from runners.ensemble_bench import build_ensemble, _naive, _sh

NOMINAL_BOOK = 100_000.0      # the equity book size we size a paper sleeve against
SLEEVE_W = 0.15               # candidate sleeve weight in the marginal-blend test


def _say(phase, agent, msg, flush=True):
    """one streamed line: '<phase> | <agent> | <msg>'. The cockpit colours by phase."""
    tag = {"research": "research", "build": "build", "validate": "validate",
           "execute": "execute", "verdict": "verdict"}[phase]
    print(f"{tag} | {agent:<14s} | {msg}", flush=flush)


def candidate_series(name, params):
    fn = LAB_STRATEGIES[name]
    b = backtest_book(fn, QUALITY_UNIVERSE, params, label=name)
    if "error" in b:
        return None
    return _naive(vol_target(b["_returns"], target_vol=0.12, max_leverage=1.0)), b


def _verdict(delta, corr, wf_pos, wf_n, dsr):
    """machine recommendation - the human still approves/rejects in the UI."""
    if delta >= 0.01 and corr < 0.6 and wf_pos >= max(4, wf_n - 1) and dsr >= 0.90:
        return "PROMOTE", "raises the blend, decorrelated, robust across folds + deflation"
    if delta > 0.0:
        why = []
        if corr >= 0.6:
            why.append("corr too high")
        if wf_pos < max(4, wf_n - 1):
            why.append(f"only {wf_pos}/{wf_n} folds")
        if dsr < 0.90:
            why.append(f"DSR {dsr:.0%}")
        return "REVIEW", "improves the blend but " + (", ".join(why) or "needs more track record")
    return "REJECT", "does not improve the blended book (the ensemble already owns this)"


def run(emit_path=None, seed=None):
    t0 = time.time()
    seed = int(time.time()) if seed is None else int(seed)
    rng = random.Random(seed)
    print("=" * 78, flush=True)
    print("AUTONOMOUS AGENT LAB | 12 agents | research -> build -> validate -> execute",
          flush=True)
    print("=" * 78, flush=True)
    print(f"batch seed {seed} - each run samples a fresh parameter set per agent", flush=True)
    print("booting benchmark: building the live equity ensemble (~1-2 min)...\n", flush=True)

    ens = build_ensemble()
    ens_sh = _sh(ens)
    em = _metrics_from_returns(ens, [], "ensemble")
    print(f"BENCHMARK | equity ensemble book | Sharpe {ens_sh:.2f} | "
          f"CAGR {em['cagr']:.1%} | maxDD {em['max_drawdown']:.1%}\n", flush=True)

    # first pass: sample THIS run's params + build every candidate's return series
    # (the param draw is what makes each click a different batch of strategies)
    series, books, order, used = {}, {}, [], {}
    for spec in LAB_AGENTS:
        name, agent = spec["strategy"], spec["agent"]
        params = sample_params(name, rng)
        used[name] = params
        try:
            res = candidate_series(name, params)
        except Exception as e:
            _say("build", agent, f"FAILED to compile {name}: {e}")
            continue
        if res is None:
            _say("build", agent, f"no data for {name}")
            continue
        r, b = res
        series[name], books[name] = r, b
        order.append(spec)

    # deflation set: variance of per-period Sharpes across all trials searched
    per = {n: sharpe_stats(series[n].to_numpy()) for n in series}
    sr_var = float(np.var([per[n]["sr"] for n in per], ddof=1)) if len(per) > 1 else 0.0
    sr_star = expected_max_sharpe(len(per), sr_var)      # the deflation hurdle

    results = []
    for i, spec in enumerate(order, 1):
        name, agent = spec["strategy"], spec["agent"]
        r, b = series[name], books[name]
        print(f"\n[{i}/{len(order)}] {agent} | {name} ({spec['family']})", flush=True)

        # ---- research --------------------------------------------------------
        _say("research", agent, f"hypothesis: {spec['thesis']}")

        # ---- build -----------------------------------------------------------
        params = used[name]
        _say("build", agent, f"compiled signal | params {params} | long/flat | "
                             f"shift=1 (no look-ahead) | {len(r)} bars")

        # ---- validate --------------------------------------------------------
        common = ens.index.intersection(r.index)
        e = ens.reindex(common).fillna(0.0)
        c = r.reindex(common).fillna(0.0)
        sh = _sh(c)
        dd = _metrics_from_returns(c, [], name)["max_drawdown"]
        corr = float(c.corr(e)) if c.std() > 0 else 0.0
        blend = (1 - SLEEVE_W) * e + SLEEVE_W * c
        bsh = _sh(blend)
        delta = bsh - _sh(e)
        folds = walk_forward_folds(c, n_folds=5)
        fold_srs = [round(f["sharpe"], 2) for f in folds]
        wf_pos = sum(1 for f in folds if f["sharpe"] > 0)
        st = per[name]
        dsr = probabilistic_sharpe_ratio(st["sr"], st["n"], st["skew"], st["kurt"],
                                         sr_benchmark=sr_star)
        _say("validate", agent,
             f"Sharpe {sh:.2f} | maxDD {dd:.1%} | corr->ens {corr:+.2f} | "
             f"blend {bsh:.2f} ({delta:+.2f}) | DSR {dsr:.0%}")
        _say("validate", agent,
             f"walk-forward {len(folds)} folds | +ve in {wf_pos}/{len(folds)} | "
             f"fold Sharpes {fold_srs}")

        # ---- execute (DRY-RUN) ----------------------------------------------
        try:
            target = float(LAB_STRATEGIES[name](daily_bars("SPY"), params).iloc[-1])
        except Exception:
            target = 0.0
        sleeve_usd = NOMINAL_BOOK * SLEEVE_W
        _say("execute", agent,
             f"dry-run paper sleeve @ {SLEEVE_W:.0%} (${sleeve_usd:,.0f}) | "
             f"current target exposure {target:.0%} | NO live order")

        # ---- verdict ---------------------------------------------------------
        verdict, reason = _verdict(delta, corr, wf_pos, len(folds), dsr)
        _say("verdict", agent, f"{verdict} - {reason}")

        results.append({
            "agent": agent, "strategy": name, "family": spec["family"],
            "thesis": spec["thesis"], "params": params,
            "sharpe": round(sh, 3), "maxdd": round(dd, 4), "corr": round(corr, 3),
            "blend": round(bsh, 3), "delta": round(delta, 4),
            "wf_pos": wf_pos, "wf_n": len(folds), "wf_folds": fold_srs,
            "dsr": round(dsr, 4), "verdict": verdict, "reason": reason,
        })

    # ---- summary -------------------------------------------------------------
    results.sort(key=lambda x: x["delta"], reverse=True)
    promote = [x for x in results if x["verdict"] == "PROMOTE"]
    review = [x for x in results if x["verdict"] == "REVIEW"]
    print("\n" + "=" * 78, flush=True)
    print("SUMMARY | candidates ranked by marginal contribution to the blend", flush=True)
    print("=" * 78, flush=True)
    print(f"  {'agent':14s} {'strategy':18s} {'corr':>6s} {'blendD':>7s} "
          f"{'WF':>4s} {'DSR':>5s}  verdict", flush=True)
    print("  " + "-" * 72, flush=True)
    for x in results:
        print(f"  {x['agent']:14s} {x['strategy']:18s} {x['corr']:>+6.2f} "
              f"{x['delta']:>+7.2f} {x['wf_pos']:>2d}/5 {x['dsr']:>4.0%}  {x['verdict']}",
              flush=True)
    print(f"\n  {len(promote)} PROMOTE | {len(review)} REVIEW | "
          f"{len(results) - len(promote) - len(review)} REJECT", flush=True)
    print(f"  elapsed {time.time() - t0:.0f}s | decisions await human approval in the cockpit",
          flush=True)

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "benchmark": {"sharpe": round(ens_sh, 3), "cagr": round(em["cagr"], 4),
                      "maxdd": round(em["max_drawdown"], 4)},
        "sr_star_annual": round(sr_star * np.sqrt(TRADING_DAYS), 3),
        "candidates": results,
    }
    if emit_path:
        out = Path(emit_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  wrote {out}", flush=True)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", default=None, help="write candidates JSON to this path")
    ap.add_argument("--seed", default=None, type=int,
                    help="fix the batch seed (default: time-based -> a new batch each run)")
    args = ap.parse_args()
    run(emit_path=args.emit, seed=args.seed)


if __name__ == "__main__":
    main()
