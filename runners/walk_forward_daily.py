"""
runners/walk_forward_daily.py
-----------------------------
Anchored (expanding-window) walk-forward with REAL parameter optimization, for
the three daily strategies. This is the honest test: parameters are chosen only
on past data, then judged on the next unseen year.

Scheme (per the desk's spec):
  - train on 2016 .. (Y-1), optimize parameters (>= 4 years of train data)
  - test on year Y (out-of-sample)
  - roll forward: year Y is folded into the next training set, Y+1 becomes the
    new test year. Training window GROWS (anchored start = 2016).

Manager constraint: the chosen parameters must trade >= --min-trades-per-year
(default 100) in-sample, otherwise that parameter set is ineligible. Among the
eligible sets we pick the one with the best in-sample Sharpe.

Objective optimized in-sample: Sharpe, subject to the trades/year floor.

Outputs:
  results/walk_forward_daily.json     full per-fold detail + chosen params
  WALK_FORWARD_SETTINGS.md            human-readable settings for all 3 algos

Usage:
  python runners\\walk_forward_daily.py --universe SPY,QQQ,GLD,MSFT,JPM,GOOGL
  python runners\\walk_forward_daily.py --universe all --min-trades-per-year 100
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    STRATEGIES_DAILY, daily_bars, sleeve_returns, INITIAL_CAP, TRADING_DAYS,
)

# parameter grids — RSI-2 includes loose thresholds so the optimizer can reach
# the >=100 trades/year floor by trading dips more frequently.
GRIDS = {
    "rsi2_meanrev": {
        "rsi_period": [2, 3],
        "entry_rsi":  [5, 10, 15, 20, 25, 30],
        "exit_rsi":   [50, 60, 70, 80],
        "trend_sma":  [100, 200],
    },
    "donchian": {
        "entry_lookback": [10, 20, 40, 55],
        "exit_lookback":  [5, 10, 20],
    },
    "trend_5020": {
        "fast": [10, 20, 50],
        "slow": [100, 150, 200],
    },
}


def combos(grid: dict) -> list[dict]:
    keys = list(grid)
    out = []
    for vals in product(*[grid[k] for k in keys]):
        d = dict(zip(keys, vals))
        if "fast" in d and d["fast"] >= d["slow"]:
            continue
        if "exit_lookback" in d and d["exit_lookback"] > d["entry_lookback"]:
            continue
        out.append(d)
    return out


def _sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 5 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS))


def _maxdd(r: pd.Series) -> float:
    if len(r) == 0:
        return 0.0
    eq = (1 + r.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def precompute(strat: str, universe: list) -> dict:
    """for every param combo, compute the book's full-history daily return
    series + a trade table (with entry dates). Folds just slice these."""
    fn = STRATEGIES_DAILY[strat]
    daily = {t: daily_bars(t) for t in universe}
    out = {}
    for params in combos(GRIDS[strat]):
        sleeves, trades = [], []
        for t, d in daily.items():
            net, trs = sleeve_returns(d, fn, params)
            sleeves.append(net.rename(t))
            for x in trs:
                trades.append({"entry": x["entry"], "ret": x["ret"], "ticker": t})
        book = pd.concat(sleeves, axis=1).mean(axis=1)
        tdf = pd.DataFrame(trades)
        if not tdf.empty:
            tdf["entry"] = pd.to_datetime(tdf["entry"])
        out[json.dumps(params, sort_keys=True)] = (params, book, tdf)
    return out


def _slice(s: pd.Series, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.Series:
    return s[(s.index >= lo) & (s.index <= hi)]


def _count(tdf: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp):
    if tdf.empty:
        return 0, 0.0
    sub = tdf[(tdf["entry"] >= lo) & (tdf["entry"] <= hi)]
    n = len(sub)
    wr = float((sub["ret"] > 0).mean()) if n else 0.0
    return n, wr


def walk_forward(strat: str, universe: list, init_train_years: int,
                 min_tpy: float) -> dict:
    pre = precompute(strat, universe)
    any_book = next(iter(pre.values()))[1]
    start = any_book.index.min()
    last  = any_book.index.max()
    first_test_year = start.year + init_train_years
    test_years = list(range(first_test_year, last.year + 1))

    folds, oos_slices, oos_trade_frames = [], [], []
    for Y in test_years:
        train_lo = start
        train_hi = pd.Timestamp(f"{Y-1}-12-31", tz="UTC")
        test_lo  = pd.Timestamp(f"{Y}-01-01", tz="UTC")
        test_hi  = pd.Timestamp(f"{Y}-12-31", tz="UTC")
        train_years = max((train_hi - train_lo).days / 365.25, 1e-9)

        best, best_sh, best_meets = None, -1e9, False
        # pass 1: combos meeting the trades/year floor in-sample
        for key, (params, book, tdf) in pre.items():
            tr = _slice(book, train_lo, train_hi)
            sh = _sharpe(tr)
            n, _ = _count(tdf, train_lo, train_hi)
            tpy = n / train_years
            meets = tpy >= min_tpy
            if meets and sh > best_sh:
                best, best_sh, best_meets = key, sh, True
        # fallback: if nothing meets the floor, take best Sharpe overall (flagged)
        if best is None:
            for key, (params, book, tdf) in pre.items():
                tr = _slice(book, train_lo, train_hi)
                sh = _sharpe(tr)
                if sh > best_sh:
                    best, best_sh, best_meets = key, sh, False

        params, book, tdf = pre[best]
        tr = _slice(book, train_lo, train_hi)
        te = _slice(book, test_lo, test_hi)
        n_tr, _    = _count(tdf, train_lo, train_hi)
        n_te, wr_te = _count(tdf, test_lo, test_hi)
        test_span = max((min(test_hi, last) - test_lo).days / 365.25, 1e-9)

        oos_slices.append(te)
        if not tdf.empty:
            oos_trade_frames.append(
                tdf[(tdf["entry"] >= test_lo) & (tdf["entry"] <= test_hi)])

        folds.append({
            "test_year": Y,
            "train": f"{train_lo.date()}..{train_hi.date()} ({train_years:.1f}y)",
            "params": params,
            "train_sharpe": round(best_sh, 3),
            "train_trades_per_year": round(n_tr / train_years, 1),
            "meets_trade_floor": best_meets,
            "oos_sharpe": round(_sharpe(te), 3),
            "oos_return_pct": round(float((1 + te.fillna(0)).prod() - 1) * 100, 2),
            "oos_trades": n_te,
            "oos_trades_per_year": round(n_te / test_span, 1),
            "oos_win_rate": round(wr_te, 4),
        })

    oos = pd.concat(oos_slices).sort_index() if oos_slices else pd.Series(dtype=float)
    oos = oos[~oos.index.duplicated()]
    oos_years = max(len(oos) / TRADING_DAYS, 1e-9)
    total_oos_trades = int(sum(len(f) for f in oos_trade_frames))
    oos_wr = (float(pd.concat(oos_trade_frames)["ret"].gt(0).mean())
              if oos_trade_frames and sum(len(f) for f in oos_trade_frames) else 0.0)
    final_cap = INITIAL_CAP * float((1 + oos.fillna(0)).prod())

    return {
        "strategy": strat,
        "universe": universe,
        "init_train_years": init_train_years,
        "min_trades_per_year": min_tpy,
        "folds": folds,
        "oos_aggregate": {
            "sharpe": round(_sharpe(oos), 3),
            "total_return_pct": round((final_cap / INITIAL_CAP - 1) * 100, 2),
            "pnl_dollars": round(final_cap - INITIAL_CAP, 2),
            "final_capital": round(final_cap, 2),
            "max_drawdown": round(_maxdd(oos), 4),
            "win_rate": round(oos_wr, 4),
            "total_trades": total_oos_trades,
            "trades_per_year": round(total_oos_trades / oos_years, 1),
            "oos_span": f"{oos.index.min().date()}..{oos.index.max().date()}" if len(oos) else "n/a",
        },
        # params to DEPLOY going forward = last fold (trained on the most data)
        "recommended_params": folds[-1]["params"] if folds else {},
        "recommended_meets_floor": folds[-1]["meets_trade_floor"] if folds else False,
    }


def write_markdown(reports: dict, path: Path, min_tpy: float, universe: list):
    L = []
    L.append("# Walk-Forward Settings — 3 Daily Algorithms\n")
    L.append(f"**Generated:** {datetime.now(timezone.utc).date()}  ")
    L.append(f"**Universe:** {', '.join(universe)}  ")
    L.append(f"**Method:** anchored expanding-window walk-forward "
             f"(train 2016→Y-1, test Y, roll forward)  ")
    L.append(f"**Manager constraint:** >= {min_tpy:.0f} trades/year, enforced "
             f"during in-sample parameter selection  ")
    L.append(f"**Costs:** 6 bps round-trip · **Start capital:** $100,000\n")
    L.append("> Out-of-sample = stitched test years only, each traded with "
             "parameters chosen *before* that year. This is the honest number.\n")

    for strat, rep in reports.items():
        agg = rep["oos_aggregate"]
        L.append(f"\n## {strat}\n")
        L.append("**Recommended deploy parameters** "
                 "(trained on all data through the last full year):\n")
        L.append("```json")
        L.append(json.dumps(rep["recommended_params"], indent=2))
        L.append("```")
        floor = "YES" if rep["recommended_meets_floor"] else "NO (floor not reachable)"
        L.append(f"- meets >= {min_tpy:.0f} trades/yr: **{floor}**\n")
        L.append("**Out-of-sample (walk-forward) performance:**\n")
        L.append(f"| Sharpe | $PnL | Total ret | Max DD | Win rate | Trades/yr |")
        L.append(f"|--------|------|-----------|--------|----------|-----------|")
        L.append(f"| {agg['sharpe']} | ${agg['pnl_dollars']:,.0f} | "
                 f"{agg['total_return_pct']}% | {agg['max_drawdown']:.1%} | "
                 f"{agg['win_rate']:.1%} | {agg['trades_per_year']} |\n")
        L.append("**Per-fold detail:**\n")
        L.append("| Test yr | Train | Params | Train SR | Train tr/yr | "
                 "OOS SR | OOS ret | OOS tr/yr | floor |")
        L.append("|---------|-------|--------|----------|-------------|"
                 "--------|---------|-----------|-------|")
        for f in rep["folds"]:
            p = ", ".join(f"{k}={v}" for k, v in f["params"].items())
            L.append(f"| {f['test_year']} | {f['train']} | {p} | "
                     f"{f['train_sharpe']} | {f['train_trades_per_year']} | "
                     f"{f['oos_sharpe']} | {f['oos_return_pct']}% | "
                     f"{f['oos_trades_per_year']} | "
                     f"{'Y' if f['meets_trade_floor'] else 'n'} |")
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="SPY,QQQ,GLD,MSFT,JPM,GOOGL")
    ap.add_argument("--init-train-years", type=int, default=4)
    ap.add_argument("--min-trades-per-year", type=float, default=100)
    args = ap.parse_args()

    if args.universe.strip().lower() == "all":
        from data.loader import DATA_DIR
        universe = sorted(p.stem for p in Path(DATA_DIR).glob("*.parquet"))
    else:
        universe = [t.strip().upper() for t in args.universe.split(",") if t.strip()]

    print(f"\nWalk-forward (anchored) | universe({len(universe)})={', '.join(universe)}")
    print(f"init train years={args.init_train_years} | "
          f"min trades/year floor={args.min_trades_per_year:.0f}\n")

    reports = {}
    for strat in STRATEGIES_DAILY:
        print(f"  optimizing {strat} ...", flush=True)
        reports[strat] = walk_forward(strat, universe, args.init_train_years,
                                      args.min_trades_per_year)

    # console summary
    print(f"\n{'strategy':14s} {'OOS Sharpe':>10s} {'OOS $PnL':>11s} "
          f"{'maxDD':>7s} {'winRate':>8s} {'tr/yr':>7s} {'floor':>6s}")
    print("-" * 70)
    for strat, rep in reports.items():
        a = rep["oos_aggregate"]
        floor = "PASS" if rep["recommended_meets_floor"] else "FAIL"
        print(f"{strat:14s} {a['sharpe']:10.2f} {a['pnl_dollars']:11,.0f} "
              f"{a['max_drawdown']:7.1%} {a['win_rate']:8.1%} "
              f"{a['trades_per_year']:7.0f} {floor:>6s}")

    out_json = Path("results/walk_forward_daily.json")
    out_json.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(),
        "method": "anchored expanding-window walk-forward with param optimization",
        "reports": reports,
    }, indent=2, default=str), encoding="utf-8")
    write_markdown(reports, Path("WALK_FORWARD_SETTINGS.md"),
                   args.min_trades_per_year, universe)
    print(f"\nWrote {out_json} and WALK_FORWARD_SETTINGS.md")


if __name__ == "__main__":
    main()
