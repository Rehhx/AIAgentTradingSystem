"""
runners/regime_levers.py
------------------------
Explore how to (a) lift the weak 2016-2020 stretch and (b) push CAGR toward
15-20%, using REGIME-AWARE tweaks rather than new signals:

  1. leverage in calm regimes  -> vol-target with max_leverage 1.5 / 2.0
  2. bear-market rotation      -> when SPY < 200d SMA, park in TLT (bonds) /
                                  GLD instead of cash ("crisis alpha")
  3. both combined

Reports Sharpe / CAGR / maxDD + per-fold returns (so we can see the 2018-2020
fold specifically). This is analysis, not a deploy change.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    backtest_book, backtest_cross_sectional, vol_target, _metrics_from_returns,
    sig_rsi2_meanrev, sig_donchian, sig_trend_5020, DEPLOY_PARAMS, QUALITY_UNIVERSE,
    walk_forward_folds, daily_bars,
)
from data.sp500 import sp500_tickers


def fold_line(r):
    return "  ".join(f"{f.get('start','?')[:4]}-{f.get('end','?')[:4]}:{f['return_pct']:+.0%}"
                     for f in walk_forward_folds(r, 5))


def show(label, r):
    m = _metrics_from_returns(r, [], label)
    gate = "PASS" if (m["sharpe"] >= 0.8 and m["max_drawdown"] >= -0.15) else "FAIL"
    print(f"{label:30s} SR {m['sharpe']:5.2f} | CAGR {m['cagr']:6.1%} | "
          f"DD {m['max_drawdown']:6.1%} | {gate} | folds {fold_line(r)}")
    return m


def main():
    U = QUALITY_UNIVERSE
    print("\nbuilding ensemble components ...")
    r_rsi = backtest_book(sig_rsi2_meanrev, U, DEPLOY_PARAMS["rsi2_meanrev"])["_returns"]
    r_don = backtest_book(sig_donchian, U)["_returns"]
    r_trd = backtest_book(sig_trend_5020, U)["_returns"]
    r_xs = backtest_cross_sectional(sp500_tickers(), mode="momentum", lookback=252,
                                    skip=21, k=10, market_filter=True)["_returns"]
    panel = pd.concat([r_rsi, r_don, r_trd, r_xs], axis=1, sort=True)
    panel.columns = ["rsi", "don", "trd", "xs"]
    raw = panel.mean(axis=1)

    # SPY regime + defensive assets
    spy = daily_bars("SPY")["close"]
    bull = (spy > spy.rolling(200).mean()).reindex(raw.index).ffill().fillna(False)
    bull_y = bull.shift(1).fillna(False)            # decide on prior close
    def defret(t):
        c = daily_bars(t)["close"]
        return c.pct_change().reindex(raw.index).fillna(0.0)
    tlt = defret("TLT")
    gld = defret("GLD")

    print(f"\n{'variant':30s} {'Sharpe':>6s}   {'CAGR':>6s}   {'maxDD':>6s}  gate   per-fold returns")
    print("-" * 110)

    base = vol_target(raw, 0.12, max_leverage=1.0)
    show("ensemble (current deploy)", base)
    show("  + leverage <=1.5", vol_target(raw, 0.12, max_leverage=1.5))
    show("  + leverage <=2.0", vol_target(raw, 0.12, max_leverage=2.0))

    # bear rotation: offense when bull, bonds/gold when bear (50/50 TLT+GLD)
    defensive = 0.5 * tlt + 0.5 * gld
    switch = base.where(bull_y, defensive)
    show("regime-switch (bear->TLT/GLD)", switch)
    switch_lev = vol_target(raw, 0.12, max_leverage=1.5).where(bull_y, defensive)
    show("regime-switch + lev<=1.5", switch_lev)

    print("\nNote: leverage figures assume margin (Alpaca paper supports it). "
          "Bear-rotation needs TLT/GLD added to the live universe.")


if __name__ == "__main__":
    main()
