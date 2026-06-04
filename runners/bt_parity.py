"""
runners/bt_parity.py  (BUILD_PLAN.md Tier 2A validation)
-------------------------------------------------------
Proves the event-driven engine (backtest/) reproduces the vectorized backtest
(agents/daily_strategies) it is meant to replace. Runs the SAME strategy
(trend_5020) on the SAME data (SPY) two ways and compares the return series.

If the event engine is correct and look-ahead-free, its equity curve should track
the vectorized `signal.shift(1)` book to floating-point tolerance: identical
annualized Sharpe (to 2 dp), identical total return (to a few bps), correlation
~1.0. A material gap would mean a look-ahead leak or a cost/timing bug.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from agents.daily_strategies import (
    daily_bars, sleeve_returns, sig_trend_5020, TRADING_DAYS,
)
from backtest import (
    HistoricDataHandler, FunctionStrategy, Portfolio, SimulatedExecution, Backtest,
)
from backtest.execution import SIDE_COST


def _sharpe(r):
    r = r.fillna(0.0)
    return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0


def main(symbol: str = "SPY"):
    d = daily_bars(symbol)
    print(f"parity check: trend_5020 on {symbol}  ({len(d)} bars "
          f"{d.index[0].date()}..{d.index[-1].date()})\n")

    # --- vectorized (the reference) ---
    vec, _ = sleeve_returns(d, sig_trend_5020)

    # --- event-driven engine ---
    data = HistoricDataHandler({symbol: d[["close"]]})
    bt = Backtest(data, FunctionStrategy(sig_trend_5020),
                  Portfolio(data, 100_000.0),
                  SimulatedExecution(data, commission_rate=SIDE_COST, slippage_rate=0.0))
    res = bt.run()
    eng = res["returns"]

    # align on common dates
    common = vec.index.intersection(eng.index)
    v, e = vec.reindex(common).fillna(0.0), eng.reindex(common).fillna(0.0)

    vs, es = _sharpe(v), _sharpe(e)
    vtot = float((1 + v).prod() - 1)
    etot = float((1 + e).prod() - 1)
    corr = float(np.corrcoef(v.to_numpy(), e.to_numpy())[0, 1])
    max_diff = float((v - e).abs().max())

    print(f"  {'metric':22s} {'vectorized':>12s} {'event-engine':>13s}")
    print("  " + "-" * 49)
    print(f"  {'annualized Sharpe':22s} {vs:>12.3f} {es:>13.3f}")
    print(f"  {'total return':22s} {vtot:>12.2%} {etot:>13.2%}")
    print(f"  {'# fills':22s} {'-':>12s} {res['counts']['fills']:>13d}")
    print()
    print(f"  return-series correlation : {corr:.6f}")
    print(f"  max per-bar abs diff      : {max_diff:.2e}")
    print()

    ok = (round(vs, 2) == round(es, 2)) and abs(vtot - etot) < 0.005 and corr > 0.999
    print("  PARITY: " + ("PASS -- engine reproduces the vectorized book"
                          if ok else "MISMATCH -- investigate timing/cost/look-ahead"))
    return ok


if __name__ == "__main__":
    main()
