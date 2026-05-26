"""
embeddings/__init__.py
-----------------------
unified interface for the full embedding layer.
agents import from here — one import covers all three stores.

usage:
    from vector_stores import EmbeddingLayer

    emb = EmbeddingLayer()
    emb.seed()                          # run once on setup

    # regime detection
    result = emb.regime.find_similar(ticker, current_window_df)
    print(result["regime"], result["forward_return_stats"])

    # strategy memory
    check = emb.strategy.find_similar(new_strategy_dict)
    if check["recommendation"] == "skip":
        pass  # already tried something like this

    # research lookup
    docs = emb.research.search("momentum strategy with volume filter")
"""

import logging
from pathlib import Path

from vector_stores.client   import EmbeddingClient, LARGE, SMALL
from vector_stores.regime_store   import RegimeStore
from vector_stores.strategy_store import StrategyStore
from vector_stores.research_store import ResearchStore

log = logging.getLogger("vector_stores")

CHROMA_DIR = Path("vector_stores/chroma_db")


class EmbeddingLayer:
    """
    top-level interface to all three vector stores.
    the orchestrator and all agents go through this.
    """

    def __init__(self, chroma_dir: Path = CHROMA_DIR, openai_api_key: str = None):
        self.regime   = RegimeStore(chroma_dir)
        self.strategy = StrategyStore(chroma_dir)
        self.research = ResearchStore(chroma_dir)
        log.info("embedding layer initialized | 3 stores ready")

    def seed(self, force: bool = False) -> dict:
        """
        run once on first setup.
        seeds the research knowledge base with built-in strategy docs.
        regime and strategy stores get populated as the system runs.
        """
        n = self.research.seed_known_strategies(force=force)
        log.info(f"embedding layer seeded | {n} research docs added")
        return {"research_docs_seeded": n}

    def index_ticker_regimes(
        self,
        ticker: str,
        df,
        step: int = 15,
        start: str = None,
        end: str = None,
    ) -> int:
        """
        index all candle windows for a ticker into the regime store.
        call this for each parquet file during initial setup.
        takes a few minutes per ticker due to embedding api calls.
        """
        return self.regime.index_ticker(ticker, df, step=step, start=start, end=end)

    def check_strategy(self, strategy: dict) -> dict:
        """
        before backtesting a new strategy, check:
          1. is it similar to something we've tried before? (strategy store)
          2. what does the research base say about this type of strategy?
        returns combined intelligence to inform the orchestrator's decision.
        """
        memory_check = self.strategy.find_similar(strategy)
        research     = self.research.find_related_to_strategy(strategy, k=3)

        return {
            "memory":   memory_check,
            "research": research,
            "proceed":  memory_check["recommendation"] != "skip",
        }

    def stats(self) -> dict:
        return {
            "regime_store":   self.regime.stats(),
            "strategy_store": self.strategy.stats(),
            "research_store": self.research.stats(),
        }


__all__ = [
    "EmbeddingLayer",
    "RegimeStore",
    "StrategyStore",
    "ResearchStore",
    "EmbeddingClient",
    "LARGE",
    "SMALL",
]
