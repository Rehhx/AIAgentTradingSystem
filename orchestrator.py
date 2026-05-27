"""
orchestrator.py
---------------
the brain of the quant agent system. routes tasks between agents,
maintains shared state, tracks strategy lifecycle, and runs the feedback loop.

agent registry:
  - research_agent       : scans papers/blogs for known strategies
  - autonomous_agent     : generates novel strategy ideas from first principles
  - ml_research_agent    : trains and evaluates ML/DL models on 1m bar data
  - backtesting_agent    : tests strategies on historical parquet data
  - risk_agent           : validates risk metrics before any strategy proceeds
  - code_agent           : writes python implementation of validated strategies
  - monitor_agent        : watches live paper trading PnL and signals
  - execution_agent      : sends orders to alpaca paper account
"""

import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

# real agent implementations — orchestrator wrappers below delegate to these.
# imports are top-level; the agents that import StrategyStatus back from this
# module do so lazily inside their run() methods to avoid a circular import.
from agents.research_agent     import ResearchAgent     as _RealResearchAgent
from agents.autonomous_agent   import AutonomousAgent   as _RealAutonomousAgent
from agents.ml_research_agent  import MLResearchAgent   as _RealMLResearchAgent
from agents.backtesting_agent  import BacktestingAgent  as _RealBacktestingAgent
from agents.risk_agent         import RiskAgent         as _RealRiskAgent
from agents.code_agent         import CodeAgent         as _RealCodeAgent
from agents.monitor_agent      import MonitorAgent      as _RealMonitorAgent
from agents.execution_agent    import ExecutionAgent    as _RealExecutionAgent
from agents.options_agent      import OptionsAgent      as _RealOptionsAgent
from config                    import DEFAULT_TICKERS

# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/orchestrator.log"),
    ],
)
log = logging.getLogger("orchestrator")


# ---------------------------------------------------------------------------
# enums
# ---------------------------------------------------------------------------
class AgentName(str, Enum):
    RESEARCH       = "research_agent"
    AUTONOMOUS     = "autonomous_agent"
    ML_RESEARCH    = "ml_research_agent"
    BACKTESTING    = "backtesting_agent"
    RISK           = "risk_agent"
    CODE           = "code_agent"
    MONITOR        = "monitor_agent"
    EXECUTION      = "execution_agent"
    OPTIONS        = "options_agent"


class StrategyStatus(str, Enum):
    PROPOSED       = "proposed"       # just an idea, not yet backtested
    BACKTESTING    = "backtesting"    # currently being tested
    RISK_REVIEW    = "risk_review"    # passed backtest, checking risk
    APPROVED       = "approved"       # passed risk, ready to implement
    IMPLEMENTING   = "implementing"   # code agent writing it
    PAPER_TRADING  = "paper_trading"  # live in alpaca paper account
    PAUSED         = "paused"         # monitor flagged an issue
    REJECTED       = "rejected"       # failed backtest or risk checks
    RETIRED        = "retired"        # ran, collected results, archived


class TaskStatus(str, Enum):
    PENDING    = "pending"
    RUNNING    = "running"
    DONE       = "done"
    FAILED     = "failed"


# ---------------------------------------------------------------------------
# data classes (plain dicts for now, swap for pydantic later)
# ---------------------------------------------------------------------------
def new_strategy(
    name: str,
    source_agent: AgentName,
    description: str,
    params: dict,
) -> dict:
    return {
        "id": str(uuid.uuid4())[:8],
        "name": name,
        "source_agent": source_agent,
        "description": description,
        "params": params,
        "status": StrategyStatus.PROPOSED,
        "backtest_results": None,
        "risk_results": None,
        "code_path": None,
        "paper_results": None,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "history": [],
    }


def new_task(
    agent: AgentName,
    action: str,
    payload: dict,
    strategy_id: Optional[str] = None,
) -> dict:
    return {
        "task_id": str(uuid.uuid4())[:8],
        "agent": agent,
        "action": action,
        "payload": payload,
        "strategy_id": strategy_id,
        "status": TaskStatus.PENDING,
        "result": None,
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": None,
    }


# ---------------------------------------------------------------------------
# results store  (in-memory + json file for persistence)
# ---------------------------------------------------------------------------
class ResultsStore:
    """
    persists strategies, tasks, and model scores to disk.
    agents read/write through here — no direct file access elsewhere.
    """

    def __init__(self, path: str = "results/store.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {
            "strategies": {},
            "tasks": {},
            "model_scores": {},
            "trade_log": [],
            "performance_summary": {},
        }

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    # strategies
    def add_strategy(self, strategy: dict) -> str:
        sid = strategy["id"]
        self._data["strategies"][sid] = strategy
        self.save()
        log.info(f"strategy added | id={sid} name={strategy['name']}")
        return sid

    def update_strategy(self, sid: str, **kwargs):
        if sid not in self._data["strategies"]:
            raise KeyError(f"strategy {sid} not found")
        s = self._data["strategies"][sid]
        for k, v in kwargs.items():
            s[k] = v
        s["updated_at"] = datetime.utcnow().isoformat()
        s["history"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "changes": kwargs,
        })
        self.save()
        log.info(f"strategy updated | id={sid} changes={list(kwargs.keys())}")

    def get_strategy(self, sid: str) -> dict:
        return self._data["strategies"].get(sid)

    def get_strategies_by_status(self, status: StrategyStatus) -> list:
        return [
            s for s in self._data["strategies"].values()
            if s["status"] == status
        ]

    # tasks
    def add_task(self, task: dict) -> str:
        tid = task["task_id"]
        self._data["tasks"][tid] = task
        self.save()
        return tid

    def complete_task(self, tid: str, result: dict, status: TaskStatus = TaskStatus.DONE):
        if tid not in self._data["tasks"]:
            raise KeyError(f"task {tid} not found")
        t = self._data["tasks"][tid]
        t["status"] = status
        t["result"] = result
        t["completed_at"] = datetime.utcnow().isoformat()
        self.save()

    # model scores
    def log_model_score(self, model_name: str, ticker: str, metrics: dict):
        key = f"{model_name}_{ticker}"
        self._data["model_scores"][key] = {
            "model": model_name,
            "ticker": ticker,
            "metrics": metrics,
            "logged_at": datetime.utcnow().isoformat(),
        }
        self.save()

    # trade log
    def log_trade(self, trade: dict):
        trade["logged_at"] = datetime.utcnow().isoformat()
        self._data["trade_log"].append(trade)
        self.save()

    def summary(self) -> dict:
        strategies = self._data["strategies"]
        return {
            "total_strategies": len(strategies),
            "by_status": {
                status: len([s for s in strategies.values() if s["status"] == status])
                for status in StrategyStatus
            },
            "total_tasks": len(self._data["tasks"]),
            "total_trades": len(self._data["trade_log"]),
            "model_scores_logged": len(self._data["model_scores"]),
        }


# ---------------------------------------------------------------------------
# agent base class
# ---------------------------------------------------------------------------
class BaseAgent:
    """
    all agents inherit this. gives them access to the store and
    a standard interface the orchestrator can call.
    """

    def __init__(self, name: AgentName, store: ResultsStore):
        self.name = name
        self.store = store
        self.log = logging.getLogger(name)

    def run(self, task: dict) -> dict:
        """
        override this in each agent subclass.
        must return a dict with at least {"success": bool, ...}
        """
        raise NotImplementedError(f"{self.name} must implement run()")

    def _success(self, **kwargs) -> dict:
        return {"success": True, "agent": self.name, **kwargs}

    def _failure(self, reason: str, **kwargs) -> dict:
        self.log.error(f"task failed | reason={reason}")
        return {"success": False, "agent": self.name, "reason": reason, **kwargs}


# ---------------------------------------------------------------------------
# stub agents (will be replaced with real implementations)
# ---------------------------------------------------------------------------
class ResearchAgent(BaseAgent):
    """delegates to agents.research_agent.ResearchAgent (Claude Agent SDK + web_search)."""
    def __init__(self, store):
        super().__init__(AgentName.RESEARCH, store)
        self._impl = _RealResearchAgent(store=store)

    def run(self, task: dict) -> dict:
        return self._impl.run(task)


class AutonomousAgent(BaseAgent):
    """delegates to agents.autonomous_agent.AutonomousAgent (Claude Agent SDK, no tools)."""
    def __init__(self, store):
        super().__init__(AgentName.AUTONOMOUS, store)
        self._impl = _RealAutonomousAgent(store=store)

    def run(self, task: dict) -> dict:
        return self._impl.run(task)


class MLResearchAgent(BaseAgent):
    """delegates to agents.ml_research_agent.MLResearchAgent (XGBoost + walk-forward)."""
    def __init__(self, store):
        super().__init__(AgentName.ML_RESEARCH, store)
        self._impl = _RealMLResearchAgent(store=store)

    def run(self, task: dict) -> dict:
        return self._impl.run(task)


class BacktestingAgent(BaseAgent):
    """
    delegates to agents.backtesting_agent.BacktestingAgent. when called with a
    strategy_id, resolves the spec from the store, runs the backtest, and
    writes the aggregate metrics back as `backtest_results` so risk_agent can
    read them.
    """
    def __init__(self, store):
        super().__init__(AgentName.BACKTESTING, store)
        self._impl = _RealBacktestingAgent()

    def run(self, task: dict) -> dict:
        strategy_id = task.get("strategy_id")
        payload     = dict(task.get("payload") or {})

        # if a strategy_id is present and the caller didn't override the spec,
        # fill it in from the store so the real impl has name + params to work with.
        if strategy_id and "name" not in payload:
            spec = self.store.get_strategy(strategy_id) or {}
            payload.setdefault("name",    spec.get("name", "rsi_reversion"))
            payload.setdefault("params",  spec.get("params", {}))
            payload.setdefault("tickers", payload.get("tickers", DEFAULT_TICKERS[:5]))

        result = self._impl.run({"payload": payload})

        if strategy_id and result.get("success"):
            self.store.update_strategy(
                strategy_id,
                backtest_results = result.get("aggregate"),
            )
        return self._success(**result) if result.get("success") else \
               self._failure(result.get("reason", "backtest failed"), raw=result)


class RiskAgent(BaseAgent):
    """delegates to agents.risk_agent.RiskAgent — thresholds live in config.RISK."""
    def __init__(self, store):
        super().__init__(AgentName.RISK, store)
        self._impl = _RealRiskAgent(store=store)

    def run(self, task: dict) -> dict:
        return self._impl.run(task)


class CodeAgent(BaseAgent):
    """delegates to agents.code_agent.CodeAgent (Claude Agent SDK + validation)."""
    def __init__(self, store):
        super().__init__(AgentName.CODE, store)
        self._impl = _RealCodeAgent(store=store)

    def run(self, task: dict) -> dict:
        return self._impl.run(task)


class MonitorAgent(BaseAgent):
    """delegates to agents.monitor_agent.MonitorAgent (alpaca polling, drawdown alerts)."""
    def __init__(self, store):
        super().__init__(AgentName.MONITOR, store)
        self._impl = _RealMonitorAgent(store=store)

    def run(self, task: dict) -> dict:
        return self._impl.run(task)


class ExecutionAgent(BaseAgent):
    """delegates to agents.execution_agent.ExecutionAgent (alpaca-py paper trading)."""
    def __init__(self, store):
        super().__init__(AgentName.EXECUTION, store)
        self._impl = _RealExecutionAgent(store=store)

    def run(self, task: dict) -> dict:
        return self._impl.run(task)


class OptionsAgent(BaseAgent):
    """delegates to agents.options_agent.OptionsAgent (alpaca options paper trading)."""
    def __init__(self, store):
        super().__init__(AgentName.OPTIONS, store)
        self._impl = _RealOptionsAgent(store=store)

    def run(self, task: dict) -> dict:
        return self._impl.run(task)


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
class Orchestrator:
    """
    routes work between agents, manages strategy lifecycle,
    and runs the research → backtest → risk → code → paper loop.
    """

    def __init__(self, data_dir: str = r"C:\Users\pcagm\Downloads\StockData"):
        self.store    = ResultsStore()
        self.data_dir = Path(data_dir)

        # re-register any generated strategies sitting in strategies/ so the
        # registry survives restarts. built-in strategies are already present
        # at module load time; this picks up anything code_agent wrote.
        from agents.backtesting_agent import load_generated_strategies
        load_generated_strategies()

        self.agents: dict[AgentName, BaseAgent] = {
            AgentName.RESEARCH:    ResearchAgent(self.store),
            AgentName.AUTONOMOUS:  AutonomousAgent(self.store),
            AgentName.ML_RESEARCH: MLResearchAgent(self.store),
            AgentName.BACKTESTING: BacktestingAgent(self.store),
            AgentName.RISK:        RiskAgent(self.store),
            AgentName.CODE:        CodeAgent(self.store),
            AgentName.MONITOR:     MonitorAgent(self.store),
            AgentName.EXECUTION:   ExecutionAgent(self.store),
            AgentName.OPTIONS:     OptionsAgent(self.store),
        }
        log.info(f"orchestrator initialized | data_dir={self.data_dir} | agents={len(self.agents)}")

    # ------------------------------------------------------------------
    # core dispatcher
    # ------------------------------------------------------------------
    def dispatch(self, agent: AgentName, action: str, payload: dict,
                 strategy_id: str = None) -> dict:
        """send a task to a specific agent and record it"""
        task   = new_task(agent, action, payload, strategy_id)
        tid    = self.store.add_task(task)
        log.info(f"dispatching | agent={agent} action={action} task={tid}")

        try:
            result = self.agents[agent].run(task)
            status = TaskStatus.DONE if result.get("success") else TaskStatus.FAILED
            self.store.complete_task(tid, result, status)
            return result
        except Exception as e:
            log.exception(f"agent crashed | agent={agent} task={tid} error={e}")
            self.store.complete_task(tid, {"error": str(e)}, TaskStatus.FAILED)
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # strategy lifecycle pipelines
    # ------------------------------------------------------------------
    def run_research_pipeline(self) -> list[str]:
        """
        step 1 — research + autonomous agent generate strategy ideas
        step 2 — each idea goes to backtesting
        step 3 — passed backtests go to risk validation
        step 4 — approved strategies go to code agent
        returns list of strategy ids that made it to paper trading
        """
        approved_ids = []
        log.info("=== starting research pipeline ===")

        # gather ideas from all three research agents in parallel (TODO: asyncio)
        sources = [
            (AgentName.RESEARCH,    "scan_strategies",  {}),
            (AgentName.AUTONOMOUS,  "generate_ideas",   {}),
        ]

        all_ideas = []
        for agent, action, payload in sources:
            result = self.dispatch(agent, action, payload)
            if result.get("success"):
                ideas = result.get("strategies_found", []) + result.get("ideas", [])
                all_ideas.extend(ideas)

        log.info(f"collected {len(all_ideas)} strategy ideas")

        for idea in all_ideas:
            sid = self._run_strategy_lifecycle(idea)
            if sid:
                approved_ids.append(sid)

        log.info(f"pipeline complete | {len(approved_ids)} strategies approved for paper trading")
        return approved_ids

    def _run_strategy_lifecycle(self, idea: dict) -> Optional[str]:
        """
        runs a single idea through (code → if needed) → backtest → risk → paper.

        novel ideas (no signal function in STRATEGIES yet) go to code_agent
        first so a python implementation exists before backtest. ideas whose
        name matches an existing registry entry skip code generation and go
        straight to backtest.
        """
        from agents.backtesting_agent import is_registered  # local — runtime lookup

        # register strategy
        strategy = new_strategy(
            name         = idea.get("name", "unnamed"),
            source_agent = idea.get("source_agent", AgentName.AUTONOMOUS),
            description  = idea.get("description", idea.get("hypothesis", "")),
            params       = idea.get("params", {}),
        )
        sid = self.store.add_strategy(strategy)
        log.info(f"strategy registered | id={sid} name={strategy['name']}")

        # code generation FIRST for novel ideas — code_agent writes a signals()
        # function, validates it on synthetic data, and registers it into the
        # runtime STRATEGIES dict so the backtest dispatch below can find it.
        if not is_registered(strategy["name"]):
            log.info(f"strategy '{strategy['name']}' not in registry — dispatching code_agent first")
            self.store.update_strategy(sid, status=StrategyStatus.IMPLEMENTING)
            code_result = self.dispatch(AgentName.CODE, "implement", {}, sid)
            if not code_result.get("success"):
                log.warning(f"code generation failed | id={sid}")
                self.store.update_strategy(sid, status=StrategyStatus.REJECTED)
                return None

        # backtest
        self.store.update_strategy(sid, status=StrategyStatus.BACKTESTING)
        bt_result = self.dispatch(AgentName.BACKTESTING, "run_backtest", {}, sid)
        if not bt_result.get("success"):
            log.warning(f"backtest failed | id={sid}")
            self.store.update_strategy(sid, status=StrategyStatus.REJECTED)
            return None

        # risk check
        self.store.update_strategy(sid, status=StrategyStatus.RISK_REVIEW)
        risk_result = self.dispatch(AgentName.RISK, "validate", {}, sid)
        if not risk_result.get("success"):
            log.warning(f"risk check failed | id={sid}")
            return None

        # at this point risk_agent has set status to APPROVED. mark paper.
        self.store.update_strategy(sid, status=StrategyStatus.PAPER_TRADING)
        log.info(f"strategy approved and ready for paper trading | id={sid}")
        return sid

    def run_ml_pipeline(self, tickers: list[str] = None,
                        models: list[str] = None,
                        window_days: int = 365):
        """
        runs ML research agent across tickers and model types.
        results logged to store for the autonomous agent to use as signals.
        """
        tickers = tickers or self._available_tickers()
        models  = models  or ["xgboost", "lstm", "transformer"]
        log.info(f"=== starting ml pipeline | {len(tickers)} tickers | {len(models)} models ===")

        for ticker in tickers:
            for model in models:
                self.dispatch(AgentName.ML_RESEARCH, "train_model", {
                    "ticker":      ticker,
                    "model":       model,
                    "window_days": window_days,
                    "data_dir":    str(self.data_dir),
                }, strategy_id=None)

    def run_monitor_cycle(self):
        """call this on a schedule (e.g. every minute) to check live positions"""
        result = self.dispatch(AgentName.MONITOR, "check_positions", {})
        if result.get("monitor_status", {}).get("alerts"):
            log.warning(f"monitor alerts: {result['monitor_status']['alerts']}")
        return result

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------
    def _available_tickers(self) -> list[str]:
        """reads which tickers we have parquet files for"""
        if not self.data_dir.exists():
            log.warning(f"data_dir not found: {self.data_dir}")
            return ["SPY", "QQQ"]   # fallback
        tickers = [p.stem for p in self.data_dir.glob("*.parquet")]
        log.info(f"found {len(tickers)} parquet files: {tickers}")
        return tickers

    def status_report(self) -> dict:
        summary = self.store.summary()
        tickers = self._available_tickers()
        return {
            "orchestrator": "running",
            "data_dir": str(self.data_dir),
            "available_tickers": tickers,
            "registered_agents": [a for a in self.agents],
            "store_summary": summary,
        }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  quant agent system — orchestrator")
    print("=" * 60 + "\n")

    orc = Orchestrator(data_dir=r"C:\Users\pcagm\Downloads\StockData")

    # status check
    report = orc.status_report()
    print(f"available tickers : {report['available_tickers']}")
    print(f"registered agents : {len(report['registered_agents'])}")
    print()

    # run the full research pipeline (stubs for now)
    print("[1] running research pipeline...")
    approved = orc.run_research_pipeline()
    print(f"    strategies approved: {approved}")
    print()

    # run ml pipeline on available data
    print("[2] running ml research pipeline...")
    orc.run_ml_pipeline(window_days=365)
    print()

    # monitor check
    print("[3] running monitor cycle...")
    orc.run_monitor_cycle()
    print()

    # final summary
    print("[4] results store summary:")
    summary = orc.store.summary()
    for k, v in summary.items():
        print(f"    {k}: {v}")
    print()
    print("orchestrator run complete. results saved to results/store.json")
    print("next: implement individual agents starting with backtesting_agent.py")
