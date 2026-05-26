"""
agents/monitor_agent.py
-----------------------
watches live paper trading positions and equity. flags drawdown breaches,
position concentration, and abnormal fills. tells orchestrator to pause or
kill a strategy if a threshold is broken.

implementation note:
  - first cut is a snapshot/polling design (one call = one health check)
  - websocket streaming is a future upgrade — alpaca-py supports it via
    TradingStream, but a streaming agent needs a separate event loop and
    background process, which is orchestration scope, not agent scope
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_PAPER, RISK

log = logging.getLogger("monitor_agent")


class MonitorAgent:
    """matches BaseAgent.run(task) contract used by orchestrator."""

    # alert thresholds — orchestrator can override per-strategy via task payload
    DEFAULT_THRESHOLDS = {
        "max_position_pct":      RISK["max_position_pct"],
        "max_unrealized_dd_pct": -0.10,   # single-position unrealized -10% triggers alert
        "max_portfolio_dd_pct":  -0.05,   # account-level day drawdown -5% triggers pause
    }

    def __init__(self, store=None):
        self.store    = store
        self.client   = None
        self.simulated = False
        self.log      = logging.getLogger("monitor_agent")

        if not ALPACA_API_KEY or not ALPACA_API_SECRET:
            self.log.warning("ALPACA creds missing — monitor in SIMULATED mode (returns all-clear)")
            self.simulated = True
            return

        try:
            from alpaca.trading.client import TradingClient
            self.client = TradingClient(
                api_key    = ALPACA_API_KEY,
                secret_key = ALPACA_API_SECRET,
                paper      = ALPACA_PAPER,
            )
        except Exception as e:
            self.log.exception(f"alpaca init failed — SIMULATED: {e}")
            self.simulated = True

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def run(self, task: dict) -> dict:
        thresholds = {**self.DEFAULT_THRESHOLDS, **(task.get("payload", {}).get("thresholds", {}))}

        if self.simulated:
            return self._success(monitor_status={
                "mode":              "simulated",
                "active_positions":  0,
                "portfolio_value":   0.0,
                "unrealized_pnl":    0.0,
                "alerts":            [],
                "actions_requested": [],
                "all_clear":         True,
                "checked_at":        datetime.now(timezone.utc).isoformat(),
            })

        try:
            positions = self.client.get_all_positions()
            account   = self.client.get_account()
        except Exception as e:
            return self._failure(f"alpaca query failed: {e}")

        portfolio_value = float(account.portfolio_value or 0)
        last_equity     = float(account.last_equity or portfolio_value or 1)
        day_dd_pct      = (portfolio_value - last_equity) / last_equity if last_equity else 0.0
        total_unrealized = sum(float(p.unrealized_pl) for p in positions)

        alerts = []
        actions = []

        # portfolio-level drawdown
        if day_dd_pct < thresholds["max_portfolio_dd_pct"]:
            alerts.append(f"day drawdown {day_dd_pct:.2%} breached {thresholds['max_portfolio_dd_pct']:.2%}")
            actions.append({"action": "pause_all", "reason": "portfolio drawdown"})

        # per-position checks
        for p in positions:
            mv  = float(p.market_value)
            upl = float(p.unrealized_pl)
            cost = float(p.cost_basis or mv)
            pct_of_portfolio = abs(mv) / portfolio_value if portfolio_value else 0
            pos_dd_pct       = upl / cost if cost else 0

            if pct_of_portfolio > thresholds["max_position_pct"]:
                alerts.append(
                    f"{p.symbol} is {pct_of_portfolio:.1%} of portfolio "
                    f"(limit {thresholds['max_position_pct']:.1%})"
                )
                actions.append({"action": "trim", "symbol": p.symbol})

            if pos_dd_pct < thresholds["max_unrealized_dd_pct"]:
                alerts.append(
                    f"{p.symbol} unrealized {pos_dd_pct:.2%} "
                    f"breached {thresholds['max_unrealized_dd_pct']:.2%}"
                )
                actions.append({"action": "close", "symbol": p.symbol})

        status = {
            "mode":              "alpaca_paper" if ALPACA_PAPER else "alpaca_live",
            "active_positions":  len(positions),
            "portfolio_value":   portfolio_value,
            "last_equity":       last_equity,
            "day_drawdown_pct":  day_dd_pct,
            "unrealized_pnl":    total_unrealized,
            "alerts":            alerts,
            "actions_requested": actions,
            "all_clear":         not alerts,
            "checked_at":        datetime.now(timezone.utc).isoformat(),
        }

        if alerts:
            self.log.warning(f"monitor alerts: {alerts}")
        return self._success(monitor_status=status)

    def _success(self, **kwargs):
        return {"success": True, "agent": "monitor_agent", **kwargs}

    def _failure(self, reason: str, **kwargs):
        self.log.warning(f"monitor failed | {reason}")
        return {"success": False, "agent": "monitor_agent", "reason": reason, **kwargs}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    agent = MonitorAgent()
    print(agent.run({"payload": {}}))
