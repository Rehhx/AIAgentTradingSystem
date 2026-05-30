"""
runners/trade_failure_ml.py
---------------------------
"Learn why trades fail" — instead of inventing new strategies, mine the LOSING
trades of a deployed sleeve to find an entry filter that removes the worst ones.

Method (deliberately simple + interpretable, to avoid the classic ML-overfit trap):
  1. Replay a sleeve's trades; for each ENTRY record features known at decision
     time (no lookahead): RSI, distance from trend, market regime/vol, the name's
     own vol, distance from its 52-week high, 5-day momentum into entry.
  2. Label win = trade return > 0.
  3. Univariate read: win-rate by tercile of each feature (the human-readable
     "why trades fail").
  4. Walk-forward logistic regression: train on trades before 2022, TEST on
     2022+ (out-of-sample). Report OOS AUC + coefficients.
  5. Filter test: on the OOS trades, drop those the model rates low-probability;
     compare expectancy (avg return/trade) and win-rate with vs without the filter.
     Honest verdict: a filter only counts if it improves the OUT-OF-SAMPLE set.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    daily_bars, _rsi, RT_COST, DEPLOY_PARAMS, QUALITY_UNIVERSE,
    sig_rsi2_meanrev, sig_donchian, sig_recovery,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

TD = 252
FEATS = ["rsi", "dist_trend", "mkt_bull", "mkt_vol", "name_vol", "dist_high", "ret5"]
CUTOFF = "2022-01-01"


def collect(sig_fn, params, universe):
    spy = daily_bars("SPY")["close"]
    spy200 = spy.rolling(200).mean()
    spyvol = spy.pct_change().rolling(20).std() * np.sqrt(TD)
    rows = []
    for t in universe:
        d = daily_bars(t); c = d["close"]
        pos = sig_fn(d, params).fillna(0).to_numpy()
        rsi = _rsi(c, 2).to_numpy()
        sma = c.rolling(100).mean().to_numpy()
        vol = (c.pct_change().rolling(20).std() * np.sqrt(TD)).to_numpy()
        hi = c.rolling(252).max().to_numpy()
        ret5 = c.pct_change(5).to_numpy()
        cv = c.to_numpy(); idx = c.index
        sb = (spy > spy200).reindex(idx).fillna(False).to_numpy()
        sv = spyvol.reindex(idx).to_numpy()
        for i in range(1, len(pos) - 1):
            if pos[i] > 0 and pos[i - 1] == 0:           # signal fires at i -> enter close i+1
                j = i + 1
                while j < len(pos) and pos[j - 1] > 0:    # held while prior-bar signal on
                    j += 1
                if j >= len(cv):
                    break
                ret = cv[j] / cv[i + 1] - 1 - RT_COST
                if not (sma[i] == sma[i] and hi[i] == hi[i]):
                    continue
                rows.append(dict(
                    ticker=t, entry=idx[i + 1], ret=ret, win=int(ret > 0),
                    rsi=rsi[i], dist_trend=cv[i] / sma[i] - 1, mkt_bull=int(sb[i]),
                    mkt_vol=sv[i], name_vol=vol[i], dist_high=cv[i] / hi[i] - 1, ret5=ret5[i],
                ))
    return pd.DataFrame(rows).dropna().sort_values("entry").reset_index(drop=True)


def analyze(name, sig_fn, params, universe):
    df = collect(sig_fn, params, universe)
    print(f"\n{'='*78}\n{name}: {len(df)} trades | overall win-rate {df.win.mean():.0%} "
          f"| avg return/trade {df.ret.mean():+.2%}\n{'='*78}")
    if len(df) < 200:
        print("  too few trades for a reliable model — skipping."); return

    print("\n  why trades fail (win-rate by feature tercile, low->high):")
    print(f"    {'feature':12s} {'low':>14s} {'mid':>14s} {'high':>14s}")
    for f in FEATS:
        if df[f].nunique() <= 2:
            g = df.groupby(df[f].astype(int)).win.agg(["mean", "count"])
            cells = "  ".join(f"{int(k)}={v['mean']:.0%}(n{int(v['count'])})" for k, v in g.iterrows())
            print(f"    {f:12s} {cells}")
            continue
        q = pd.qcut(df[f], 3, labels=["low", "mid", "high"], duplicates="drop")
        wr = df.groupby(q).win.mean()
        rr = df.groupby(q).ret.mean()
        cells = [f"{wr.get(b, float('nan')):.0%}/{rr.get(b, float('nan')):+.1%}" for b in ["low", "mid", "high"]]
        print(f"    {f:12s} {cells[0]:>14s} {cells[1]:>14s} {cells[2]:>14s}")
    print("    (cells = win-rate / avg-return per tercile)")

    tr, te = df[df.entry < CUTOFF], df[df.entry >= CUTOFF]
    if len(te) < 60:
        print("\n  not enough post-2022 trades for an OOS test."); return
    sc = StandardScaler().fit(tr[FEATS])
    lr = LogisticRegression(max_iter=1000).fit(sc.transform(tr[FEATS]), tr.win)
    p_te = lr.predict_proba(sc.transform(te[FEATS]))[:, 1]
    auc = roc_auc_score(te.win, p_te)
    print(f"\n  walk-forward logistic model (train <{CUTOFF}, n={len(tr)}; test >= n={len(te)}):")
    print(f"    out-of-sample AUC = {auc:.3f}  ({'has signal' if auc>0.55 else 'no real signal (~coin flip)'})")
    coef = sorted(zip(FEATS, lr.coef_[0]), key=lambda x: -abs(x[1]))
    print("    feature pull on win-probability (+ helps, - hurts):")
    for f, w in coef:
        print(f"      {f:12s} {w:+.2f}")

    thr = np.median(lr.predict_proba(sc.transform(tr[FEATS]))[:, 1])
    kept = te[p_te >= thr]
    print(f"\n  filter test (drop OOS trades the model rates below the train-median odds):")
    print(f"    no filter : {len(te):4d} trades | win {te.win.mean():.0%} | expectancy {te.ret.mean():+.2%}/trade")
    print(f"    filtered  : {len(kept):4d} trades | win {kept.win.mean():.0%} | expectancy {kept.ret.mean():+.2%}/trade "
          f"({len(te)-len(kept)} dropped)")
    lift = kept.ret.mean() - te.ret.mean()
    print(f"    -> OOS expectancy {'IMPROVES' if lift>0.001 else 'no robust improvement'} ({lift:+.2%}/trade)")


def main():
    analyze("RSI-2 mean-reversion", sig_rsi2_meanrev, DEPLOY_PARAMS["rsi2_meanrev"], QUALITY_UNIVERSE)
    analyze("Donchian breakout", sig_donchian, None, QUALITY_UNIVERSE)
    analyze("Recovery-thrust", sig_recovery, {"hold_days": 120}, QUALITY_UNIVERSE)


if __name__ == "__main__":
    main()
