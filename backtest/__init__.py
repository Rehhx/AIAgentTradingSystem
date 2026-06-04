"""
backtest — event-driven, look-ahead-free backtesting engine (BUILD_PLAN.md Tier 2A)
-----------------------------------------------------------------------------------
A small event-driven engine in the classic MarketEvent -> SignalEvent ->
OrderEvent -> FillEvent style. Its defining property: the DataHandler can only
expose bars up to the current cursor, so a strategy *cannot* see the future —
look-ahead bias is impossible by construction, not by discipline.

The mark-to-market is recorded BEFORE the same-bar fill, which makes the engine
reproduce the vectorized `signal.shift(1)` convention exactly (a position decided
at close[t] earns close[t]->close[t+1], and its trade cost lands on the next bar
— see runners/bt_parity.py for the validation against agents/daily_strategies).
"""
from backtest.events import MarketEvent, SignalEvent, OrderEvent, FillEvent
from backtest.data import HistoricDataHandler
from backtest.strategy import Strategy, FunctionStrategy
from backtest.portfolio import Portfolio
from backtest.execution import SimulatedExecution
from backtest.engine import Backtest

__all__ = [
    "MarketEvent", "SignalEvent", "OrderEvent", "FillEvent",
    "HistoricDataHandler", "Strategy", "FunctionStrategy",
    "Portfolio", "SimulatedExecution", "Backtest",
]
