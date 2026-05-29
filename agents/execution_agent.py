"""
agents/execution_agent.py
-------------------------
the only agent allowed to send orders to alpaca. wraps alpaca-py's
TradingClient with the orchestrator's task interface.

design:
  - paper-trading only (config.ALPACA_PAPER is True by default)
  - if alpaca creds missing, falls back to "simulated" mode so the
    rest of the pipeline can be tested end-to-end without an account
  - logs every fill to results store via store.log_trade()

task payload contract:
    {
      "signal": {
        "ticker":     "AAPL",
        "side":       "buy" | "sell",
        "qty":        int,
        "order_type": "market" | "limit",   # default market
        "limit_price": float,                # required if limit
        "time_in_force": "day" | "gtc",      # default day
      },
      "strategy_id": "abc12345"   # optional, for traceability
    }
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_PAPER

log = logging.getLogger("execution_agent")


class ExecutionAgent:
    """matches BaseAgent.run(task) contract used by orchestrator."""

    def __init__(self, store=None):
        self.store    = store
        self.client   = None
        self.simulated = False
        self.log      = logging.getLogger("execution_agent")

        if not ALPACA_API_KEY or not ALPACA_API_SECRET:
            self.log.warning("ALPACA creds missing — running in SIMULATED mode")
            self.simulated = True
            return

        try:
            from alpaca.trading.client import TradingClient
            self.client = TradingClient(
                api_key    = ALPACA_API_KEY,
                secret_key = ALPACA_API_SECRET,
                paper      = ALPACA_PAPER,
            )
            self.log.info(f"alpaca trading client ready | paper={ALPACA_PAPER}")
        except Exception as e:
            self.log.exception(f"failed to init alpaca client — falling back to SIMULATED: {e}")
            self.simulated = True

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def run(self, task: dict) -> dict:
        sig         = task.get("payload", {}).get("signal", {})
        strategy_id = task.get("strategy_id")

        ticker = sig.get("ticker")
        side   = sig.get("side", "buy").lower()
        qty    = float(sig.get("qty", 0) or 0)        # may be fractional
        notional = float(sig.get("notional", 0) or 0)  # dollar-sized (fractional)

        if not ticker or side not in ("buy", "sell") or (qty <= 0 and notional <= 0):
            return self._failure(f"invalid signal: {sig}")

        order_type = sig.get("order_type", "market").lower()
        tif        = sig.get("time_in_force", "day").lower()
        amount     = f"${notional:,.0f}" if notional > 0 else f"{qty:g} sh"

        if self.simulated:
            fill = self._simulated_fill(ticker, side, notional or qty, order_type)
        else:
            try:
                fill = self._submit_alpaca(ticker, side, qty, notional, order_type,
                                           tif, sig.get("limit_price"))
            except Exception as e:
                self.log.exception(f"alpaca order failed: {e}")
                return self._failure(f"alpaca submit error: {e}")

        fill["strategy_id"] = strategy_id
        if self.store is not None:
            self.store.log_trade(fill)
        self.log.info(f"executed | {side} {amount} {ticker} -> {fill.get('status')}")
        return self._success(fill=fill)

    def get_positions(self) -> list:
        """live positions snapshot — used by monitor_agent too."""
        if self.simulated or self.client is None:
            return []
        try:
            return [
                {"symbol": p.symbol, "qty": float(p.qty),
                 "avg_entry_price": float(p.avg_entry_price),
                 "unrealized_pl": float(p.unrealized_pl),
                 "market_value": float(p.market_value)}
                for p in self.client.get_all_positions()
            ]
        except Exception as e:
            self.log.exception(f"get_positions failed: {e}")
            return []

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _submit_alpaca(self, ticker, side, qty, notional, order_type, tif, limit_price):
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        side_enum = OrderSide.BUY if side == "buy" else OrderSide.SELL
        tif_enum  = TimeInForce.DAY if tif == "day" else TimeInForce.GTC

        if order_type == "limit":
            if limit_price is None:
                raise ValueError("limit_price required for limit orders")
            req = LimitOrderRequest(
                symbol=ticker, qty=qty, side=side_enum,
                time_in_force=tif_enum, limit_price=float(limit_price),
            )
        elif notional and notional > 0:
            # fractional (dollar-sized) market order — Alpaca requires DAY tif
            req = MarketOrderRequest(
                symbol=ticker, notional=round(float(notional), 2),
                side=side_enum, time_in_force=TimeInForce.DAY,
            )
        else:
            req = MarketOrderRequest(
                symbol=ticker, qty=qty, side=side_enum, time_in_force=tif_enum,
            )

        order = self.client.submit_order(req)
        return {
            "ticker":     ticker,
            "side":       side,
            "qty":        qty,
            "notional":   notional or None,
            "order_id":   str(order.id),
            "client_order_id": getattr(order, "client_order_id", None),
            "status":     str(getattr(order, "status", "submitted")),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "mode":       "alpaca_paper" if ALPACA_PAPER else "alpaca_live",
        }

    def _simulated_fill(self, ticker, side, qty, order_type):
        return {
            "ticker":       ticker,
            "side":         side,
            "qty":          qty,
            "fill_price":   None,
            "status":       "simulated",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "mode":         "simulated",
        }

    def _success(self, **kwargs):
        return {"success": True, "agent": "execution_agent", **kwargs}

    def _failure(self, reason: str, **kwargs):
        self.log.warning(f"execution failed | {reason}")
        return {"success": False, "agent": "execution_agent", "reason": reason, **kwargs}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    agent = ExecutionAgent()
    print(agent.run({"payload": {"signal": {"ticker": "SPY", "side": "buy", "qty": 1}}}))
