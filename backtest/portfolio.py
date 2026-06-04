"""
backtest/portfolio.py
---------------------
Position + cash accounting and mark-to-market. Two responsibilities:

  update_timeindex()  -- mark holdings at the current bar's close and RECORD
                         equity. Called BEFORE any same-bar fill, so the recorded
                         equity reflects the position held INTO this bar (decided
                         last bar). This ordering is what reproduces the vectorized
                         shift(1) convention and pushes each trade's cost onto the
                         next bar's return — matching agents/daily_strategies.

  generate_order()    -- turn a target weight into a signed-share order, but ONLY
                         when the target weight changes (it holds fixed shares
                         between signal changes, so a steady long position incurs
                         no drift-rebalancing turnover — again matching the
                         vectorized long/flat book).
"""
from __future__ import annotations

import pandas as pd

from backtest.events import OrderEvent

INITIAL_CAPITAL = 100_000.0


class Portfolio:
    def __init__(self, data, initial_capital: float = INITIAL_CAPITAL):
        self.data = data
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.positions = {s: 0.0 for s in data.symbols}        # signed shares
        self.target_weight = {s: 0.0 for s in data.symbols}    # last target seen
        self.equity_curve: list[tuple] = []

    # -- marking -----------------------------------------------------------
    def _market_value(self) -> float:
        mv = 0.0
        for s in self.data.symbols:
            px = self.data.latest_close(s)
            if px is not None:
                mv += self.positions[s] * px
        return mv

    def current_equity(self) -> float:
        return self.cash + self._market_value()

    def update_timeindex(self) -> None:
        """record equity at the current close (pre-fill)."""
        self.equity_curve.append((self.data.now(), self.current_equity()))

    # -- order generation --------------------------------------------------
    def generate_order(self, signal) -> OrderEvent | None:
        s = signal.symbol
        if abs(signal.target - self.target_weight[s]) < 1e-12:
            return None                                       # target unchanged -> hold
        px = self.data.latest_close(s)
        if px is None or px <= 0:
            return None
        target_shares = signal.target * self.current_equity() / px
        delta = target_shares - self.positions[s]
        self.target_weight[s] = signal.target
        if abs(delta) < 1e-12:
            return None
        return OrderEvent(s, self.data.now(), delta)

    # -- fills -------------------------------------------------------------
    def update_fill(self, fill) -> None:
        self.cash -= fill.quantity * fill.fill_price          # buy: cash down
        self.cash -= fill.commission
        self.positions[fill.symbol] += fill.quantity

    # -- results -----------------------------------------------------------
    def results(self) -> dict:
        s = pd.Series({ts: eq for ts, eq in self.equity_curve})
        s.index = pd.DatetimeIndex(s.index)
        returns = s.pct_change().fillna(0.0)
        return {"equity": s, "returns": returns}
