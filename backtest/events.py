"""
backtest/events.py
------------------
The four event types that flow through the engine queue. Each stage of the
pipeline consumes one type and may emit the next:

  MarketEvent  -- a new bar is available (emitted by the DataHandler)
  SignalEvent  -- a strategy's target exposure for a symbol (emitted by Strategy)
  OrderEvent   -- a signed-share order to reach that target (emitted by Portfolio)
  FillEvent    -- the executed order with price + commission (emitted by Execution)
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass
class MarketEvent:
    type: str = "MARKET"


@dataclass
class SignalEvent:
    symbol: str
    timestamp: dt.datetime
    target: float          # target position as a fraction of equity, in [-1, 1]
    type: str = "SIGNAL"


@dataclass
class OrderEvent:
    symbol: str
    timestamp: dt.datetime
    quantity: float        # signed shares: +buy, -sell (may be fractional)
    type: str = "ORDER"


@dataclass
class FillEvent:
    symbol: str
    timestamp: dt.datetime
    quantity: float        # signed shares actually filled
    fill_price: float
    commission: float
    type: str = "FILL"
