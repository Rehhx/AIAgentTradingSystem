"""
agents/risk_agent.py
--------------------
gates strategies between backtesting and paper_trading in the orchestrator
lifecycle. reads thresholds from config.RISK so they're tunable centrally.

also checks for "soft" red flags that a strategy passed the numbers but
might still be unsafe (e.g. <30 trades in any single regime — backtest
sharpe is reliable in aggregate but unreliable per-regime).

orchestrator integration:
    risk = RiskAgent(store)
    result = risk.run({"strategy_id": sid})
    # result["passed"] -> bool
    # result["failures"] / result["warnings"] -> lists
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RISK

log = logging.getLogger("risk_agent")


class RiskAgent:
    """
    matches BaseAgent.run(task) contract used by orchestrator.

    failure (hard) conditions reject the strategy outright.
    warning (soft) conditions allow promotion but flag for monitoring.
    """

    def __init__(self, store=None):
        self.store     = store
        self.thresholds = dict(RISK)   # copy so callers can override
        self.log        = logging.getLogger("risk_agent")

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def run(self, task: dict) -> dict:
        strategy_id = task.get("strategy_id")
        if not strategy_id:
            return self._failure("no strategy_id in task")
        if self.store is None:
            return self._failure("risk agent has no store — cannot resolve strategy")

        strategy = self.store.get_strategy(strategy_id)
        if not strategy:
            return self._failure(f"strategy {strategy_id} not found")

        results = strategy.get("backtest_results") or {}
        if not results:
            return self._failure(f"strategy {strategy_id} has no backtest_results")

        verdict = self.evaluate(results)
        verdict["strategy_id"] = strategy_id

        # write the risk verdict back to the strategy record
        from orchestrator import StrategyStatus  # local import — avoid cycle
        new_status = StrategyStatus.APPROVED if verdict["passed"] else StrategyStatus.REJECTED
        self.store.update_strategy(
            strategy_id,
            status       = new_status,
            risk_results = verdict,
        )

        if verdict["passed"]:
            return self._success(**verdict)
        return self._failure(f"risk check failed: {verdict['failures']}", **verdict)

    def evaluate(self, results: dict) -> dict:
        """
        pure function — no store side effects. takes a backtest results dict
        (with sharpe, max_drawdown, win_rate, total_trades, optionally
        regime_breakdown / per_ticker) and returns pass/fail + reasons.
        """
        t = self.thresholds
        failures, warnings = [], []

        sharpe   = results.get("sharpe", 0.0)
        max_dd   = results.get("max_drawdown", 0.0)
        win_rate = results.get("win_rate", 0.0)
        trades   = results.get("total_trades", 0)

        if sharpe   < t["min_sharpe"]:
            failures.append(f"sharpe {sharpe:.3f} < {t['min_sharpe']}")
        if max_dd   < t["max_drawdown"]:
            failures.append(f"max drawdown {max_dd:.2%} < {t['max_drawdown']:.2%}")
        if win_rate < t["min_win_rate"]:
            failures.append(f"win rate {win_rate:.2%} < {t['min_win_rate']:.2%}")
        if trades   < t["min_trades"]:
            failures.append(f"only {trades} trades, need {t['min_trades']}")

        # soft checks — these inform monitoring, not gate the decision
        per_ticker = results.get("per_ticker", {})
        if per_ticker:
            ticker_sharpes = [r.get("sharpe", 0) for r in per_ticker.values()]
            if ticker_sharpes:
                low = min(ticker_sharpes)
                if low < 0 and sharpe >= t["min_sharpe"]:
                    warnings.append(
                        f"at least one ticker has negative sharpe ({low:.2f}) "
                        f"despite passing aggregate — concentration risk"
                    )

        regime_breakdown = results.get("regime_breakdown", {})
        for regime, r in regime_breakdown.items():
            if r.get("total_trades", 0) < 30 and r.get("total_trades", 0) > 0:
                warnings.append(
                    f"regime '{regime}' only has {r['total_trades']} trades — "
                    f"per-regime stats unreliable"
                )

        passed = not failures
        return {
            "passed":            passed,
            "failures":          failures,
            "warnings":          warnings,
            "thresholds_used":   t,
            "checked_fields":    {
                "sharpe": sharpe, "max_drawdown": max_dd,
                "win_rate": win_rate, "total_trades": trades,
            },
        }

    # ------------------------------------------------------------------
    # standard return shapes
    # ------------------------------------------------------------------

    def _success(self, **kwargs) -> dict:
        return {"success": True, "agent": "risk_agent", **kwargs}

    def _failure(self, reason: str, **kwargs) -> dict:
        self.log.warning(f"risk task failed | {reason}")
        return {"success": False, "agent": "risk_agent", "reason": reason, **kwargs}


if __name__ == "__main__":
    # standalone smoke test — no store, just evaluate() against a fake result
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    agent = RiskAgent()
    fake_pass = {"sharpe": 1.2, "max_drawdown": -0.08, "win_rate": 0.52, "total_trades": 200}
    fake_fail = {"sharpe": -0.1, "max_drawdown": -0.40, "win_rate": 0.30, "total_trades": 5}
    print("pass case:", agent.evaluate(fake_pass))
    print("fail case:", agent.evaluate(fake_fail))
