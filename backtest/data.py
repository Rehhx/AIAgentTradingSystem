"""
backtest/data.py
----------------
The look-ahead firewall. HistoricDataHandler streams bars one at a time along a
unified timeline (the union of every symbol's dates). The ONLY way a strategy
reads prices is `get_history(symbol)` / `latest_close(symbol)`, both of which are
hard-bounded by the current cursor:

    get_history(s)  ==  closes.iloc[: cursor + 1]      # never index > cursor

so a strategy physically cannot read a future bar. This is the property that
makes the engine trustworthy (and is asserted directly in the tests).
"""
from __future__ import annotations

import pandas as pd

from backtest.events import MarketEvent


class HistoricDataHandler:
    def __init__(self, bars: dict[str, pd.DataFrame]):
        """bars: {symbol -> DataFrame with a 'close' column and a DatetimeIndex}."""
        if not bars:
            raise ValueError("need at least one symbol")
        self.symbols = list(bars)
        timeline = pd.DatetimeIndex(sorted(set().union(*[b.index for b in bars.values()])))
        self.timeline = timeline
        # align every symbol onto the shared timeline (missing days -> NaN close)
        self._bars = {s: b.reindex(timeline) for s, b in bars.items()}
        self._cursor = -1
        self.continue_backtest = True

    def update_bars(self) -> MarketEvent | None:
        """advance the cursor by one bar; stops the backtest past the end."""
        self._cursor += 1
        if self._cursor >= len(self.timeline):
            self.continue_backtest = False
            return None
        return MarketEvent()

    def now(self):
        return self.timeline[self._cursor]

    def get_history(self, symbol: str) -> pd.Series:
        """closes from the start through the CURRENT bar only — never the future."""
        return self._bars[symbol]["close"].iloc[: self._cursor + 1]

    def latest_close(self, symbol: str):
        if self._cursor < 0:
            return None
        v = self._bars[symbol]["close"].iloc[self._cursor]
        return float(v) if pd.notna(v) else None
