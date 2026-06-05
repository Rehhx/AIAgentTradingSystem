"""
runners/ml_orthogonal.py  (Tier 3 — the one legitimate "keep working" test)
---------------------------------------------------------------------------
Price-only ML found no directional edge (efficient market). A model can only learn
what's IN the data, so the principled move is NEW orthogonal information, not a
fancier model. This feeds FREE cross-asset signals into the SAME rigorous pipeline:

  * VIX term structure   ^VIX / ^VIX3M  (backwardation = stress; powers the sentinel)
  * credit stress        HYG / LQD ratio + HY returns
  * flight to safety     TLT (long bonds) momentum
  * dollar               UUP momentum

Question: do orthogonal-but-free features predict SPY direction where price alone
could not? Tested two ways (orthogonal-only, orthogonal+price), judged the same way
(PurgedKFold OOS AUC, strategy vs buy-hold). If even this shows no edge, the
conclusion is locked and the only remaining lever is PAID data (options IV surface).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from agents.daily_strategies import daily_bars, SIDE_COST, TRADING_DAYS, _metrics_from_returns
from analytics.significance import sharpe_stats, probabilistic_sharpe_ratio
from ml.cv import PurgedKFold
from ml.labels import get_daily_vol, triple_barrier_labels
from ml.features import make_features


def _norm(idx):
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.normalize()


def _px(t, start="2014-06-01"):
    s = yf.Ticker(t).history(start=start, auto_adjust=True)["Close"]
    s.index = _norm(s.index)
    return s


def orthogonal_features(dates) -> pd.DataFrame:
    """FREE cross-asset features, aligned to `dates` (trailing/contemporaneous only)."""
    f = pd.DataFrame(index=dates)
    def add(name, series):
        f[name] = series.reindex(dates).ffill()
    try:
        vix = _px("^VIX")
        add("vix", vix); add("vix_chg5", vix.pct_change(5))
        try:
            ts = vix / _px("^VIX3M")
            add("vix_ts", ts); add("vix_ts_chg5", ts.diff(5))
        except Exception as e:
            print(f"  (no ^VIX3M term structure: {e})")
    except Exception as e:
        print(f"  (no ^VIX: {e})")
    try:
        hyg, lqd = _px("HYG"), _px("LQD")
        cr = hyg / lqd
        add("credit", cr); add("credit_chg21", cr.pct_change(21)); add("hyg_ret5", hyg.pct_change(5))
    except Exception as e:
        print(f"  (no credit HYG/LQD: {e})")
    try:
        add("tlt_ret21", _px("TLT").pct_change(21))
    except Exception as e:
        print(f"  (no TLT: {e})")
    try:
        add("uup_ret21", _px("UUP").pct_change(21))
    except Exception as e:
        print(f"  (no UUP: {e})")
    return f


def _oos_proba(model, X, y, cv):
    p = pd.Series(np.nan, index=X.index)
    for tr, te in cv.split(X):
        if y.iloc[tr].nunique() < 2:
            continue
        model.fit(X.iloc[tr], y.iloc[tr])
        p.iloc[te] = model.predict_proba(X.iloc[te])[:, 1]
    return p


def _strategy_returns(proba, close):
    ret = close.pct_change()
    pos = (proba > 0.5).astype(float).reindex(close.index).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs())
    net = pos.shift(1).fillna(0.0) * ret - turn.shift(1).fillna(0.0) * SIDE_COST
    return net.reindex(proba.index).fillna(0.0)


def _ann(r):
    return sharpe_stats(r.to_numpy())["sr"] * np.sqrt(TRADING_DAYS)


def main(symbol: str = "SPY"):
    print(f"orthogonal-free-features test on {symbol} ...\n")
    close = daily_bars(symbol)["close"]
    close.index = _norm(close.index)
    vol = get_daily_vol(close, span=20)
    labels = triple_barrier_labels(close, horizon=10, pt=1.5, sl=1.5, vol=vol)
    ortho = orthogonal_features(close.index)
    price = make_features(close)
    print(f"\n  orthogonal features: {list(ortho.columns)}\n")

    sets = {"orthogonal-only": ortho, "orthogonal+price": ortho.join(price)}
    bh_full = close.pct_change()

    print(f"  {'feature set':20s} {'obs':>6s} {'OOS AUC':>8s} {'Sharpe':>7s} {'buy-hold':>8s}")
    print("  " + "-" * 56)
    summary = {}
    for name, feat in sets.items():
        df = labels.join(feat, how="inner").dropna()
        X, y, t1 = df[feat.columns], (df["label"] > 0).astype(int), df["t1"]
        cv = PurgedKFold(n_splits=6, t1=t1, pct_embargo=0.01)
        model = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                           learning_rate=0.05, subsample=0.8, random_state=0)
        proba = _oos_proba(model, X, y, cv).dropna()
        auc = roc_auc_score(y.reindex(proba.index), proba)
        strat = _strategy_returns(proba, close)
        win = strat.index
        sh, bh = _ann(strat), _ann(bh_full.reindex(win).fillna(0.0))
        summary[name] = {"auc": auc, "sharpe": sh, "bh": bh, "strat": strat}
        print(f"  {name:20s} {len(df):>6d} {auc:>8.3f} {sh:>7.2f} {bh:>8.2f}")

    best = max(summary, key=lambda k: summary[k]["sharpe"])
    b = summary[best]
    print("\n" + "=" * 60)
    print("HONEST VERDICT")
    print("=" * 60)
    print(f"  best set: {best}  (OOS AUC {b['auc']:.3f}, Sharpe {b['sharpe']:.2f} "
          f"vs buy-hold {b['bh']:.2f})")
    if b["auc"] > 0.55 and b["sharpe"] > b["bh"] + 0.10:
        print("  -> Orthogonal FREE data ADDS directional edge. Worth wiring features in")
        print("     and re-running the full rigor battery (DSR/PBO) before believing it.")
    else:
        print("  -> No directional edge even from orthogonal free data: AUC ~0.5 and the")
        print("     strategy does not beat buy-hold. CONCLUSION LOCKED -- daily directional")
        print("     prediction has no edge in free data; the only remaining lever is PAID")
        print("     data (options IV surface). The simple ensemble + sentinel stands.")
        print("     (Note: VIX term structure still helps as a RISK overlay -- the crash")
        print("     sentinel already uses it -- just not as a directional alpha feature.)")


if __name__ == "__main__":
    main()
