"""
runners/full_backtest.py
------------------------
Full backtest + walk-forward of the deployed book (portfolio_full), plus a test
of an EARLY-WARNING de-risk overlay (cut exposure when SPY breaks its 50-day AND
vol spikes, before the 200-day confirms a bear). $100k base, 6 bps, adjusted data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    backtest_book, backtest_cross_sectional, vol_target, _metrics_from_returns,
    sig_rsi2_meanrev, sig_donchian, sig_trend_5020, sig_recovery, sig_pead,
    DEPLOY_PARAMS, QUALITY_UNIVERSE, walk_forward_folds, split_metrics, daily_bars,
)
from data.sp500 import sp500_tickers

WEIGHTS = {"rsi": 0.28, "don": 0.22, "trd": 0.14, "xs": 0.08, "rec": 0.18, "pead": 0.10}


def report(label, r):
    m = _metrics_from_returns(r, [], label)
    s = split_metrics(r)
    print(f"\n=== {label} ===")
    print(f"  $100,000 -> ${m['final_capital']:,.0f}   (+${m['pnl_dollars']:,.0f}, {m['total_return']*100:.0f}%)")
    print(f"  CAGR {m['cagr']:.1%} | Sharpe {m['sharpe']} | max DD {m['max_drawdown']:.1%} "
          f"| win-rate {m['win_rate']:.0%} | trades {m['total_trades']}")
    print(f"  in-sample (70%) Sharpe {s['train_sharpe']:+.2f} -> out-of-sample (30%) Sharpe {s['test_sharpe']:+.2f}")
    print("  walk-forward (5 contiguous folds):")
    pos = 0
    for f in walk_forward_folds(r, 5):
        mark = "+" if f["sharpe"] > 0 else "-"; pos += f["sharpe"] > 0
        print(f"    [{mark}] {f.get('start','?')[:7]}..{f.get('end','?')[:7]}: "
              f"Sharpe {f['sharpe']:+.2f}, return {f['return_pct']:+.1%}")
    print(f"    -> positive in {pos}/5 folds")
    return m


def main():
    U, sp = QUALITY_UNIVERSE, sp500_tickers()
    print("building 6 sleeves (this scans the full S&P 500 twice) ...")
    S = {
        "rsi":  backtest_book(sig_rsi2_meanrev, U, DEPLOY_PARAMS["rsi2_meanrev"]),
        "don":  backtest_book(sig_donchian, U),
        "trd":  backtest_book(sig_trend_5020, U),
        "xs":   backtest_cross_sectional(sp, mode="momentum", lookback=252, skip=21, k=10, market_filter=True),
        "rec":  backtest_book(sig_recovery, U, {"hold_days": 120}),
        "pead": backtest_book(sig_pead, sp, {"gap_pct": 0.05, "vol_mult": 2.0, "hold_days": 60}),
    }
    print("\n-- per-sleeve standalone --")
    for k, m in S.items():
        print(f"  {k:5s} Sharpe {m['sharpe']:5.2f} | CAGR {m['cagr']:6.1%} | DD {m['max_drawdown']:6.1%} | trades {m['total_trades']}")

    panel = pd.concat({k: v["_returns"] for k, v in S.items()}, axis=1, sort=True)
    combo = sum(panel[k].fillna(0) * WEIGHTS[k] for k in WEIGHTS)
    bil = daily_bars("BIL")["close"].pct_change().reindex(panel.index).fillna(0.0)
    book = vol_target(combo, 0.15, max_leverage=1.6) + 0.28 * bil
    report("portfolio_full (deployed)", book)

    # early-warning overlay: de-risk to 60% when SPY < 50d AND 20d vol > 20%
    spy = daily_bars("SPY")["close"].reindex(panel.index)
    warn = (spy < spy.rolling(50).mean()) & (spy.pct_change().rolling(20).std() * np.sqrt(252) > 0.20)
    scale = (1.0 - 0.4 * warn.astype(float)).shift(1).reindex(book.index).fillna(1.0)
    report("portfolio_full + EARLY-WARNING overlay", book * scale)


if __name__ == "__main__":
    main()
