"""
runners/futures_xsmom.py  (Account 2 research)
----------------------------------------------
The managed-futures book uses TIME-SERIES momentum (each asset long/short on its
OWN trend). This tests CROSS-SECTIONAL momentum (rank the asset-class ETFs against
each other; long the strongest, short the weakest) — a distinct risk premium that
often diversifies trend. A diversifier earns its seat only if it has LOW correlation
to the existing book and lifts the blended Sharpe or cuts crash drawdown.

10 asset-class ETFs, inverse-vol legs, vol-targeted 12%, 6 bps. 2006-2026.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from agents.daily_strategies import _metrics_from_returns, TRADING_DAYS

MKT = ["SPY", "QQQ", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "UUP", "VNQ"]
COST = 0.0006


def _prices():
    def f(t):
        s = yf.Ticker(t).history(start="2006-06-01", end="2026-06-03", auto_adjust=True)["Close"]
        s.index = (s.index.tz_localize(None) if s.index.tz is None else s.index.tz_convert("UTC").tz_localize(None)).normalize()
        return s
    C = pd.DataFrame({t: f(t) for t in MKT}).sort_index().dropna(how="all")
    return C


def _vol_target(core, target=0.12, max_lev=1.5):
    rv = core.rolling(20).std() * np.sqrt(TRADING_DAYS)
    scale = (target / rv.replace(0, np.nan)).clip(upper=max_lev).shift(1).fillna(0)
    return (core * scale).fillna(0)


def time_series_mom(C, R, vol):
    # avg sign of 1/3/12-month returns, inverse-vol weighted (the deployed approach)
    sig = (np.sign(C / C.shift(21) - 1) + np.sign(C / C.shift(63) - 1) + np.sign(C / C.shift(252) - 1)) / 3
    W = (sig / vol).div((sig / vol).abs().sum(axis=1).replace(0, np.nan), axis=0)
    core = (W.shift(1) * R).sum(axis=1) - W.diff().abs().sum(axis=1).fillna(0) * COST
    return _vol_target(core)


def cross_sectional_mom(C, R, vol, k=3):
    # 12-1 month momentum, ranked across assets; long top-k, short bottom-k, inverse-vol
    score = C.shift(21) / C.shift(252) - 1
    rank = score.rank(axis=1, ascending=False)
    n = score.notna().sum(axis=1)
    longs = rank.le(k).astype(float)
    shorts = rank.gt(n.values[:, None] - k).astype(float)
    raw = longs - shorts
    W = (raw / vol)
    W = W.div(W.abs().sum(axis=1).replace(0, np.nan), axis=0)
    core = (W.shift(1) * R).sum(axis=1) - W.diff().abs().sum(axis=1).fillna(0) * COST
    return _vol_target(core)


def _row(name, r):
    m = _metrics_from_returns(r, [], name)
    y = (1 + r).groupby(r.index.year).prod() - 1
    return m, y


def main():
    print("pulling 10 asset-class ETFs (2006-2026) ...\n")
    C = _prices(); R = C.pct_change(); vol = R.rolling(60).std()
    ts = time_series_mom(C, R, vol)
    xs = cross_sectional_mom(C, R, vol)
    idx = ts.index.intersection(xs.index)
    ts, xs = ts.reindex(idx).fillna(0), xs.reindex(idx).fillna(0)
    blend = 0.5 * ts + 0.5 * xs

    corr = float(ts.corr(xs))
    print("=" * 76)
    print("FUTURES: time-series vs cross-sectional momentum  (vol-target 12%, 6bps)")
    print("=" * 76)
    print(f"  {'book':34s} {'CAGR':>6s} {'Sharpe':>7s} {'maxDD':>7s} {'2008':>7s} {'2022':>7s}")
    print("  " + "-" * 72)
    rows = [("time-series mom (deployed)", ts), ("cross-sectional mom (new)", xs),
            ("50/50 blend", blend)]
    for name, r in rows:
        m, y = _row(name, r)
        print(f"  {name:34s} {m['cagr']:>6.1%} {m['sharpe']:>7.2f} {m['max_drawdown']:>7.1%} "
              f"{y.get(2008,0):>+7.1%} {y.get(2022,0):>+7.1%}")

    mt = _metrics_from_returns(ts, [], "ts"); mb = _metrics_from_returns(blend, [], "b")
    print(f"\n  Correlation XS vs TS: {corr:+.2f}   (low = genuine diversification)")
    print(f"  Blend Sharpe {mb['sharpe']:.2f} vs TS-only {mt['sharpe']:.2f}  "
          f"=> {'ADDITIVE — wire as a 2nd MF sleeve' if mb['sharpe'] >= mt['sharpe'] + 0.05 or mb['max_drawdown'] > mt['max_drawdown'] + 0.02 else 'marginal/redundant — reject'}")
    print("\n  Read: cross-sectional momentum earns a seat in Account 2 only if it's lowly")
    print("  correlated to the deployed time-series book AND the blend beats TS-only on")
    print("  Sharpe or drawdown. Otherwise it's the 31st rejected strategy.")


if __name__ == "__main__":
    main()
