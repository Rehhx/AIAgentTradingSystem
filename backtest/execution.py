"""
backtest/execution.py
---------------------
Turns an OrderEvent into a FillEvent with an explicit cost model:

  fill price  = bar close * (1 +/- slippage_rate)        (adverse to the trade)
  commission  = |shares| * close * commission_rate

The commission convention is chosen to match the vectorized book: there a trade
costs `turnover_weight * SIDE_COST`, and turnover_weight * equity == |shares| *
price, so commission_rate == SIDE_COST (3 bps) reproduces it exactly. Slippage
defaults to 0 so parity runs are clean; set it to model real fills.
"""
from __future__ import annotations

from backtest.events import FillEvent

SIDE_COST = 0.0003     # 3 bps per side, same basis as agents/daily_strategies


class SimulatedExecution:
    def __init__(self, data, commission_rate: float = SIDE_COST,
                 slippage_rate: float = 0.0):
        self.data = data
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

    def execute_order(self, order) -> FillEvent | None:
        px = self.data.latest_close(order.symbol)
        if px is None or px <= 0 or order.quantity == 0:
            return None
        direction = 1.0 if order.quantity > 0 else -1.0
        fill_price = px * (1 + self.slippage_rate * direction)   # adverse slippage
        commission = abs(order.quantity) * px * self.commission_rate
        return FillEvent(order.symbol, self.data.now(), order.quantity,
                         fill_price, commission)
