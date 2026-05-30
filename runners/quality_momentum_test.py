"""
runners/quality_momentum_test.py
--------------------------------
Can a QUALITY filter improve the momentum sleeve? The fundamental version isn't
backtestable (Finnhub free = no point-in-time history), so test the concept with a
BACKTESTABLE quality proxy: low realized volatility (low-vol correlates with
fundamental quality and screens out fragile high-flyers).

  pure momentum   : top-10 by 12-1 momentum (the deployed xs_dualmom logic)
  quality momentum: top-30 by momentum, then keep the 10 lowest-vol of those

Same methodology for both, monthly, market-filtered (cash when SPY<200d). If
quality-momentum doesn't beat pure momentum out-of-sample, we skip the idea.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from agents.daily_strategies import _metrics_from_returns, walk_forward_folds, RT_COST, daily_bars, TRADING_DAYS
from data.sp500 import sp500_tickers, load_daily

H = 21


def run(quality=False):
    data = load_daily(sp500_tickers(), start="2015-01-01")
    C = pd.DataFrame({t: d["close"] for t, d in data.items() if len(d) > 300}).sort_index()
    R = C.pct_change()
    mom = C.shift(21) / C.shift(252) - 1
    vol = R.rolling(60).std()
    spy = daily_bars("SPY")["close"].reindex(C.index)
    on = (spy > spy.rolling(200).mean())
    grid = list(C.index[252::H])
    rets, dates, prev = [], [], set()
    for g in grid[:-1]:
        nxt = C.index[min(C.index.get_loc(g) + H, len(C.index) - 1)]
        if not bool(on.reindex([g]).iloc[0]):
            rets.append(0.0); dates.append(g); prev = set(); continue
        m = mom.loc[g].dropna()
        if quality:
            pool = m.nlargest(30).index
            v = vol.loc[g].reindex(pool).dropna()
            picks = list(v.nsmallest(10).index)
        else:
            picks = list(m.nlargest(10).index)
        fwd = (C.loc[nxt, picks] / C.loc[g, picks] - 1).mean()
        turn = len(set(picks) ^ prev) / 20
        rets.append(float(fwd) - turn * RT_COST); dates.append(g); prev = set(picks)
    return pd.Series(rets, index=pd.to_datetime(dates))


def stat(label, s):
    eq = (1 + s).cumprod(); yrs = len(s) / (TRADING_DAYS / H)
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    sharpe = s.mean() / s.std() * np.sqrt(TRADING_DAYS / H) if s.std() > 0 else 0
    dd = float((eq / eq.cummax() - 1).min())
    pos = sum(1 for f in walk_forward_folds(s, 5) if f["sharpe"] > 0)
    print(f"  {label:24s} Sharpe {sharpe:5.2f} | CAGR {cagr:6.1%} | DD {dd:6.1%} | WF {pos}/5")
    return sharpe, dd


def main():
    print("testing quality-filtered momentum vs pure momentum (monthly, market-filtered) ...\n")
    pure = run(quality=False)
    qual = run(quality=True)
    sp, dp = stat("pure momentum", pure)
    sq, dq = stat("quality momentum", qual)
    print()
    if sq > sp + 0.05 or (abs(sq - sp) <= 0.05 and dq > dp + 0.02):
        print(f"  VERDICT: quality filter IMPROVES momentum (Sharpe {sp:.2f}->{sq:.2f}, DD {dp:.1%}->{dq:.1%}) -> worth wiring")
    else:
        print(f"  VERDICT: quality filter does NOT improve momentum (Sharpe {sp:.2f}->{sq:.2f}, DD {dp:.1%}->{dq:.1%}) -> SKIP")


if __name__ == "__main__":
    main()
