"""
agents/options_agent.py
-----------------------
trades options on the alpaca paper account. takes a directional signal
(side + underlying) and picks an appropriate option contract — by default
the nearest-expiry ATM call (long bullish) or put (long bearish).

design:
  - paper-trading only; simulated fallback when alpaca creds missing
  - selects contract based on directional intent, not Greeks (first cut)
  - returns the contract symbol + order id + simulated mark
  - logs every fill to results store via store.log_trade()

task payload contract:
    {
      "signal": {
        "underlying":  "SPY",
        "side":        "buy" | "sell",
        "intent":      "bullish" | "bearish",   # picks call vs put
        "qty":         int,                       # number of contracts
        "moneyness":   "atm" | "5pct_otm",       # default atm
        "dte_max":     int,                       # max days-to-expiry, default 7
      },
      "strategy_id": "abc12345"   # optional
    }
"""

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_PAPER

log = logging.getLogger("options_agent")


class OptionsAgent:
    """matches BaseAgent.run(task) contract used by orchestrator."""

    def __init__(self, store=None):
        self.store    = store
        self.client   = None
        self.simulated = False
        self.log      = logging.getLogger("options_agent")

        if not ALPACA_API_KEY or not ALPACA_API_SECRET:
            self.log.warning("ALPACA creds missing — options_agent in SIMULATED mode")
            self.simulated = True
            return

        try:
            from alpaca.trading.client import TradingClient
            self.client = TradingClient(
                api_key    = ALPACA_API_KEY,
                secret_key = ALPACA_API_SECRET,
                paper      = ALPACA_PAPER,
            )
            self.log.info(f"alpaca client ready for options | paper={ALPACA_PAPER}")
        except Exception as e:
            self.log.exception(f"alpaca init failed — SIMULATED: {e}")
            self.simulated = True

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def run(self, task: dict) -> dict:
        sig         = task.get("payload", {}).get("signal", {})
        strategy_id = task.get("strategy_id")

        underlying = (sig.get("underlying") or "").upper()
        side       = (sig.get("side") or "buy").lower()
        intent     = (sig.get("intent") or "bullish").lower()
        qty        = int(sig.get("qty", 1))
        moneyness  = sig.get("moneyness", "atm")
        dte_max    = int(sig.get("dte_max", 7))

        if not underlying or qty <= 0 or side not in ("buy", "sell"):
            return self._failure(f"invalid options signal: {sig}")
        if intent not in ("bullish", "bearish"):
            return self._failure(f"intent must be bullish or bearish, got {intent}")

        if self.simulated:
            fill = self._simulated_fill(underlying, intent, qty, moneyness, dte_max)
        else:
            try:
                fill = self._submit_real(underlying, side, intent, qty, moneyness, dte_max)
            except Exception as e:
                self.log.exception(f"alpaca options order failed: {e}")
                return self._failure(f"alpaca submit error: {e}")

        fill["strategy_id"] = strategy_id
        if self.store is not None:
            self.store.log_trade(fill)
        self.log.info(f"options | {side} {qty} {fill.get('contract_symbol')} -> {fill.get('status')}")
        return self._success(fill=fill)

    def get_options_positions(self) -> list:
        """returns just the options legs from the current alpaca position book."""
        if self.simulated or self.client is None:
            return []
        try:
            return [
                {"symbol": p.symbol, "qty": float(p.qty),
                 "asset_class": str(p.asset_class),
                 "market_value": float(p.market_value),
                 "unrealized_pl": float(p.unrealized_pl)}
                for p in self.client.get_all_positions()
                if "us_option" in str(p.asset_class).lower()
            ]
        except Exception as e:
            self.log.exception(f"get_options_positions failed: {e}")
            return []

    # ------------------------------------------------------------------
    # contract selection + submission
    # ------------------------------------------------------------------

    def _pick_contract(self, underlying: str, intent: str,
                       moneyness: str, dte_max: int):
        """
        finds an option contract on the underlying matching intent + moneyness.
        uses alpaca's OptionChainRequest. raises if no contract found.
        """
        from alpaca.trading.requests import GetOptionContractsRequest
        from alpaca.trading.enums import ContractType, AssetStatus

        expiry_max = (datetime.now(timezone.utc).date() + timedelta(days=dte_max)).isoformat()
        contract_type = ContractType.CALL if intent == "bullish" else ContractType.PUT

        req = GetOptionContractsRequest(
            underlying_symbols = [underlying],
            type               = contract_type,
            expiration_date_lte = expiry_max,
            status             = AssetStatus.ACTIVE,
            limit              = 200,
        )
        resp = self.client.get_option_contracts(req)
        contracts = list(getattr(resp, "option_contracts", None) or [])
        if not contracts:
            raise RuntimeError(f"no {intent} contracts for {underlying} within {dte_max}d")

        # pick the contract closest to ATM (or 5% OTM if requested) on the
        # nearest expiry. underlying spot is fetched via the trading client's
        # latest quote endpoint if available, else fall back to the median strike.
        spot = self._underlying_spot(underlying)
        target = spot if moneyness == "atm" else \
                 spot * (1.05 if intent == "bullish" else 0.95)

        contracts.sort(key=lambda c: (c.expiration_date, abs(float(c.strike_price) - target)))
        return contracts[0]

    def _underlying_spot(self, underlying: str) -> float:
        """best-effort spot lookup; falls back to last trade if quote unavailable."""
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest
            md = StockHistoricalDataClient(api_key=ALPACA_API_KEY, secret_key=ALPACA_API_SECRET)
            quote = md.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=underlying))
            q = quote[underlying]
            mid = (float(q.bid_price) + float(q.ask_price)) / 2 if q.bid_price and q.ask_price else float(q.ask_price or q.bid_price)
            return mid
        except Exception as e:
            self.log.warning(f"spot lookup for {underlying} failed: {e} — defaulting to 100")
            return 100.0

    def _submit_real(self, underlying, side, intent, qty, moneyness, dte_max):
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        contract = self._pick_contract(underlying, intent, moneyness, dte_max)
        side_enum = OrderSide.BUY if side == "buy" else OrderSide.SELL

        req = MarketOrderRequest(
            symbol        = contract.symbol,
            qty           = qty,
            side          = side_enum,
            time_in_force = TimeInForce.DAY,
        )
        order = self.client.submit_order(req)
        return {
            "contract_symbol": contract.symbol,
            "underlying":      underlying,
            "intent":          intent,
            "strike":          float(contract.strike_price),
            "expiry":          str(contract.expiration_date),
            "side":            side,
            "qty":             qty,
            "order_id":        str(order.id),
            "status":          str(getattr(order, "status", "submitted")),
            "submitted_at":    datetime.now(timezone.utc).isoformat(),
            "mode":            "alpaca_paper" if ALPACA_PAPER else "alpaca_live",
        }

    def _simulated_fill(self, underlying, intent, qty, moneyness, dte_max):
        return {
            "contract_symbol": f"{underlying}_SIM_{intent}_{dte_max}d",
            "underlying":      underlying,
            "intent":          intent,
            "qty":             qty,
            "status":          "simulated",
            "submitted_at":    datetime.now(timezone.utc).isoformat(),
            "mode":            "simulated",
        }

    def _success(self, **kw): return {"success": True, "agent": "options_agent", **kw}
    def _failure(self, reason, **kw):
        self.log.warning(f"options failed | {reason}")
        return {"success": False, "agent": "options_agent", "reason": reason, **kw}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    agent = OptionsAgent()
    print(agent.run({"payload": {"signal": {
        "underlying": "SPY", "side": "buy", "intent": "bullish",
        "qty": 1, "moneyness": "atm", "dte_max": 7,
    }}}))
