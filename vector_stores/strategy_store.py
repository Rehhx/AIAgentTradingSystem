"""
embeddings/strategy_store.py
-----------------------------
vector memory for strategies — prevents the system from reinventing the
same strategy twice and lets it learn from past results.

use cases:
  1. deduplication  — before backtesting a new idea, find similar past
                      strategies and their outcomes
  2. outcome lookup — "what happened last time we tried momentum with
                      a 20-bar window on high-vol days?"
  3. strategy search — research agent can query "find me all trend-following
                       strategies with sharpe > 1.5"

text-embedding-3-small used because:
  - strategy descriptions are short, structured text
  - we do high-frequency lookups (every new idea triggers a search)
  - small model is 5x cheaper and still captures semantic meaning well
"""

import logging
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np

from vector_stores.client import SMALL, embed, embed_batch

log = logging.getLogger("vector_stores.strategy_store")

CHROMA_DIR = Path("vector_stores/chroma_db")
COLLECTION = "strategy_memory"


class StrategyStore:
    """
    stores embedded strategy specs + their backtest/paper results.
    the orchestrator queries this before dispatching to the backtest agent
    to avoid redundant work and surface historical patterns.
    """

    def __init__(self, chroma_dir: Path = CHROMA_DIR):
        chroma_dir.mkdir(parents=True, exist_ok=True)
        self.chroma = chromadb.PersistentClient(path=str(chroma_dir))
        self.col    = self.chroma.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(f"strategy store ready | docs={self.col.count()}")

    # ------------------------------------------------------------------
    # storing strategies
    # ------------------------------------------------------------------

    def add_strategy(self, strategy: dict) -> str:
        """
        embed and store a strategy.

        strategy dict should have:
          id, name, description, params, source_agent
          optionally: backtest_results, risk_results, paper_results, status
        """
        text = self._strategy_to_text(strategy)
        vec  = embed(text, model=SMALL)

        meta = {
            "strategy_id":   strategy["id"],
            "name":          strategy["name"],
            "source_agent":  str(strategy.get("source_agent", "")),
            "status":        str(strategy.get("status", "proposed")),
            "sharpe":        float(strategy.get("backtest_results", {}).get("sharpe", 0.0) or 0.0),
            "max_drawdown":  float(strategy.get("backtest_results", {}).get("max_drawdown", 0.0) or 0.0),
            "win_rate":      float(strategy.get("backtest_results", {}).get("win_rate", 0.0) or 0.0),
            "passed_risk":   bool(strategy.get("risk_results", {}).get("passed", False)),
        }

        self.col.upsert(
            ids        = [strategy["id"]],
            embeddings = [vec],
            documents  = [text],
            metadatas  = [meta],
        )
        log.info(f"strategy indexed | id={strategy['id']} name={strategy['name']}")
        return strategy["id"]

    def update_results(self, strategy_id: str, backtest_results: dict = None,
                       risk_results: dict = None, paper_results: dict = None,
                       status: str = None):
        """update stored metadata when results come in"""
        existing = self.col.get(ids=[strategy_id], include=["metadatas", "documents", "embeddings"])
        if not existing["ids"]:
            log.warning(f"strategy {strategy_id} not found in store")
            return

        meta = existing["metadatas"][0].copy()
        if backtest_results:
            meta["sharpe"]       = float(backtest_results.get("sharpe", 0.0) or 0.0)
            meta["max_drawdown"] = float(backtest_results.get("max_drawdown", 0.0) or 0.0)
            meta["win_rate"]     = float(backtest_results.get("win_rate", 0.0) or 0.0)
        if risk_results:
            meta["passed_risk"]  = bool(risk_results.get("passed", False))
        if status:
            meta["status"]       = status

        self.col.upsert(
            ids        = [strategy_id],
            embeddings = existing["embeddings"][0],
            documents  = existing["documents"][0],
            metadatas  = [meta],
        )

    # ------------------------------------------------------------------
    # querying
    # ------------------------------------------------------------------

    def find_similar(
        self,
        strategy: dict,
        k: int = 5,
        min_similarity: float = 0.85,
        only_with_results: bool = False,
    ) -> dict:
        """
        find strategies similar to a new idea before backtesting.

        returns:
          {
            "is_duplicate": bool,         # True if very similar strategy exists
            "similar": [...],             # top-k similar strategies
            "best_similar_sharpe": float, # best sharpe among similar
            "recommendation": str,        # "skip" | "proceed" | "tweak_params"
          }
        """
        text   = self._strategy_to_text(strategy)
        vector = embed(text, model=SMALL)

        where  = {"sharpe": {"$gt": 0}} if only_with_results else None
        results = self.col.query(
            query_embeddings = [vector],
            n_results        = min(k, max(1, self.col.count())),
            where            = where,
            include          = ["metadatas", "distances", "documents"],
        )

        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        if not metas:
            return {
                "is_duplicate":         False,
                "similar":              [],
                "best_similar_sharpe":  0.0,
                "recommendation":       "proceed",
            }

        similarities  = [1 - d for d in distances]
        is_duplicate  = similarities[0] >= min_similarity

        similar_list = [
            {**m, "similarity": round(s, 4)}
            for m, s in zip(metas, similarities)
        ]

        sharpes       = [m["sharpe"] for m in metas if m["sharpe"] > 0]
        best_sharpe   = max(sharpes) if sharpes else 0.0

        if is_duplicate and best_sharpe > 1.0:
            rec = "skip"        # similar strategy already worked well
        elif is_duplicate and best_sharpe < 0.5:
            rec = "skip"        # similar strategy already failed
        elif similarities[0] > 0.7:
            rec = "tweak_params"   # related but not identical — adjust params
        else:
            rec = "proceed"        # novel enough to test

        return {
            "is_duplicate":        is_duplicate,
            "similar":             similar_list,
            "best_similar_sharpe": round(best_sharpe, 4),
            "recommendation":      rec,
        }

    def search(self, query: str, k: int = 10, status_filter: str = None) -> list:
        """
        free-text search over strategy memory.
        useful for research agent: "find all momentum strategies with sharpe > 1"
        """
        vector = embed(query, model=SMALL)
        where  = {"status": status_filter} if status_filter else None

        results = self.col.query(
            query_embeddings = [vector],
            n_results        = min(k, max(1, self.col.count())),
            where            = where,
            include          = ["metadatas", "distances", "documents"],
        )
        return [
            {**m, "similarity": round(1 - d, 4)}
            for m, d in zip(results["metadatas"][0], results["distances"][0])
        ]

    def get_top_performers(self, n: int = 10, min_sharpe: float = 0.8) -> list:
        """returns best strategies by sharpe ratio"""
        all_docs = self.col.get(include=["metadatas"])
        metas    = all_docs["metadatas"]
        filtered = [m for m in metas if m.get("sharpe", 0) >= min_sharpe]
        return sorted(filtered, key=lambda x: x.get("sharpe", 0), reverse=True)[:n]

    # ------------------------------------------------------------------
    # text representation
    # ------------------------------------------------------------------

    @staticmethod
    def _strategy_to_text(strategy: dict) -> str:
        """
        converts a strategy dict to descriptive text for embedding.
        more detail = better semantic matching.
        """
        params = strategy.get("params", {})
        param_str = ", ".join(f"{k}={v}" for k, v in params.items()) if params else "default"

        bt = strategy.get("backtest_results") or {}
        result_str = (
            f"sharpe {bt.get('sharpe', 'N/A')}, "
            f"max drawdown {bt.get('max_drawdown', 'N/A')}, "
            f"win rate {bt.get('win_rate', 'N/A')}, "
            f"{bt.get('total_trades', 'N/A')} trades"
        ) if bt else "not yet backtested"

        return (
            f"Strategy name: {strategy.get('name', 'unnamed')}. "
            f"Source: {strategy.get('source_agent', 'unknown')}. "
            f"Description: {strategy.get('description', '')}. "
            f"Parameters: {param_str}. "
            f"Status: {strategy.get('status', 'proposed')}. "
            f"Backtest results: {result_str}."
        )

    def stats(self) -> dict:
        all_docs = self.col.get(include=["metadatas"])
        metas    = all_docs["metadatas"]
        return {
            "total_strategies":  self.col.count(),
            "approved":          sum(1 for m in metas if m.get("status") == "approved"),
            "rejected":          sum(1 for m in metas if m.get("status") == "rejected"),
            "paper_trading":     sum(1 for m in metas if m.get("status") == "paper_trading"),
            "avg_sharpe":        round(float(np.mean([m.get("sharpe", 0) for m in metas])), 4) if metas else 0.0,
            "collection":        COLLECTION,
            "model":             SMALL,
        }
