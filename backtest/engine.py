"""
backtest/engine.py
------------------
The event loop. Per bar, in strict order:

  1. update_bars()        advance the cursor (or stop at the end)
  2. update_timeindex()   mark + record equity on the OLD position (pre-fill)
  3. strategy             read history <= now, emit a SignalEvent per symbol
  4. drain the queue      SignalEvent -> OrderEvent -> FillEvent -> book the fill

Step 2 BEFORE step 4 is deliberate: it makes a position decided at close[t] earn
close[t]->close[t+1] and pushes its trade cost onto the next bar — reproducing the
vectorized shift(1) book (validated in runners/bt_parity.py).
"""
from __future__ import annotations

from collections import deque

from backtest.events import SignalEvent, OrderEvent, FillEvent


class Backtest:
    def __init__(self, data, strategy, portfolio, execution):
        self.data = data
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution = execution
        self.events: deque = deque()
        self.counts = {"signals": 0, "orders": 0, "fills": 0}

    def run(self) -> dict:
        while True:
            self.data.update_bars()
            if not self.data.continue_backtest:
                break

            self.portfolio.update_timeindex()                 # mark OLD position

            for s in self.data.symbols:
                target = self.strategy.calculate_signal(s, self.data.get_history(s))
                if target is not None:
                    self.events.append(SignalEvent(s, self.data.now(), target))

            while self.events:
                ev = self.events.popleft()
                if isinstance(ev, SignalEvent):
                    self.counts["signals"] += 1
                    order = self.portfolio.generate_order(ev)
                    if order is not None:
                        self.events.append(order)
                elif isinstance(ev, OrderEvent):
                    self.counts["orders"] += 1
                    fill = self.execution.execute_order(ev)
                    if fill is not None:
                        self.events.append(fill)
                elif isinstance(ev, FillEvent):
                    self.counts["fills"] += 1
                    self.portfolio.update_fill(ev)

        out = self.portfolio.results()
        out["counts"] = self.counts
        return out
