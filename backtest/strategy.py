"""
backtest/strategy.py
--------------------
The pluggable strategy interface. A strategy receives the visible price history
(closes through the current bar — never the future) and returns a TARGET position
for the symbol as a fraction of equity in [-1, 1] (0 = flat, 1 = fully long).

FunctionStrategy adapts the existing vectorized signal functions in
agents/daily_strategies (each: DataFrame -> position series) to the event loop by
computing the function on the visible history and taking the last value. Because
`history` only contains bars <= now, this is look-ahead-free, and taking the last
value (the signal for the current close) reproduces the vectorized convention.
"""
from __future__ import annotations

import pandas as pd


class Strategy:
    def calculate_signal(self, symbol: str, history: pd.Series):
        """history: closes through the current bar. Return target weight in
        [-1, 1], or None to leave the position unchanged."""
        raise NotImplementedError


class FunctionStrategy(Strategy):
    def __init__(self, sig_fn, params: dict | None = None):
        self.sig_fn = sig_fn
        self.params = params or {}

    def calculate_signal(self, symbol: str, history: pd.Series):
        if len(history) < 2:
            return 0.0
        d = pd.DataFrame({"close": history.to_numpy()}, index=history.index)
        pos = self.sig_fn(d, self.params)
        v = pos.iloc[-1]
        return float(v) if pd.notna(v) else 0.0
