"""
embeddings/research_store.py
-----------------------------
knowledge base for the research agent.
stores embedded documents: arxiv papers, quant blogs, strategy writeups,
market microstructure research, anything the research agent finds.

when the research agent discovers a new strategy idea, it:
  1. queries this store to find related prior art
  2. surfaces relevant papers/posts to inform parameter choices
  3. avoids proposing strategies already well-documented (without an edge)

text-embedding-3-large used because:
  - academic papers are dense with specialized vocabulary
  - we need to match concepts across different phrasings
    (e.g. "momentum" vs "trend following" vs "price continuation")
  - quality of retrieval directly affects strategy quality
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb

from vector_stores.client import LARGE, embed, embed_batch

log = logging.getLogger("vector_stores.research_store")

CHROMA_DIR = Path("vector_stores/chroma_db")
COLLECTION = "research_knowledge"


# ------------------------------------------------------------------
# document type templates
# the research agent feeds these in from different sources
# ------------------------------------------------------------------

KNOWN_STRATEGIES = [
    {
        "title": "RSI Mean Reversion",
        "category": "mean_reversion",
        "summary": (
            "Buy when RSI drops below 30 (oversold), sell when RSI rises above 70 (overbought). "
            "Works best in range-bound, low-trend markets. Fails badly in strong trending regimes. "
            "Key parameters: RSI period (14 default), oversold threshold, overbought threshold. "
            "Intraday variant: 1m RSI with tighter bands (20/80) and stop-loss based on ATR."
        ),
        "edge": "mean reversion after extreme short-term price moves",
        "regimes": ["chop", "mean_reversion"],
        "timeframes": ["1m", "5m", "15m"],
        "known_sharpe": "0.8 - 1.4 depending on market regime",
    },
    {
        "title": "VWAP Reversion",
        "category": "mean_reversion",
        "summary": (
            "Price tends to revert to VWAP (volume-weighted average price) during regular sessions. "
            "Long when price is 0.3%+ below VWAP, short when 0.3%+ above. "
            "Exit at VWAP touch or EOD. Most effective on liquid large-cap stocks. "
            "Volume confirmation critical — avoid trading against strong volume trends."
        ),
        "edge": "institutional order flow anchoring around VWAP",
        "regimes": ["mean_reversion", "chop"],
        "timeframes": ["1m", "5m"],
        "known_sharpe": "1.0 - 1.8",
    },
    {
        "title": "Opening Range Breakout (ORB)",
        "category": "breakout",
        "summary": (
            "Define the opening range as the high/low of the first N minutes (5, 15, or 30). "
            "Long on breakout above range high, short on breakdown below range low. "
            "Targets: 1x, 2x, 3x the opening range size. Stop: opposite side of range. "
            "Best on high-gap days and earnings. Filter with pre-market volume."
        ),
        "edge": "institutional position building drives price through range boundaries",
        "regimes": ["breakout", "trending"],
        "timeframes": ["1m", "5m"],
        "known_sharpe": "1.2 - 2.0 on gap days",
    },
    {
        "title": "Momentum (Price Continuation)",
        "category": "momentum",
        "summary": (
            "Buy stocks showing strong upward momentum over N bars, expecting continuation. "
            "Signal: close > N-bar high, volume confirmation, ADX > 25. "
            "Exit: trailing stop or momentum reversal signal. "
            "Cross-sectional momentum: rank universe by N-day return, long top decile. "
            "Time-series momentum: long when own N-day return is positive."
        ),
        "edge": "price trends persist due to investor underreaction and institutional flow",
        "regimes": ["trending", "breakout"],
        "timeframes": ["1m", "5m", "1d"],
        "known_sharpe": "0.8 - 1.6",
    },
    {
        "title": "Pairs Trading / Statistical Arbitrage",
        "category": "stat_arb",
        "summary": (
            "Identify cointegrated pairs (e.g. SPY/QQQ, GS/JPM). "
            "When spread deviates >2 std devs from mean, short the outperformer, long underperformer. "
            "Close when spread reverts to mean. Hedge ratio from Johansen or OLS. "
            "Key risk: cointegration breakdown during regime changes."
        ),
        "edge": "temporary dislocations between economically linked assets",
        "regimes": ["chop", "mean_reversion"],
        "timeframes": ["1m", "5m", "1d"],
        "known_sharpe": "1.5 - 2.5 when cointegration holds",
    },
    {
        "title": "Bollinger Band Squeeze",
        "category": "volatility",
        "summary": (
            "When Bollinger Bands contract significantly (squeeze), a large move is imminent. "
            "Trade the breakout direction when bands start expanding after squeeze. "
            "Combine with Keltner Channels: BB inside KC = high squeeze. "
            "Direction filter: use momentum oscillator or volume to bias long/short."
        ),
        "edge": "volatility clustering — low vol periods precede high vol expansions",
        "regimes": ["breakout"],
        "timeframes": ["5m", "15m", "1h"],
        "known_sharpe": "1.0 - 1.5",
    },
    {
        "title": "Market Microstructure: Bid-Ask Imbalance",
        "category": "microstructure",
        "summary": (
            "Order book imbalance (OBI) predicts short-term price direction. "
            "OBI = (bid_size - ask_size) / (bid_size + ask_size). "
            "High positive OBI → short-term upward pressure. Negative → downward. "
            "Works on sub-minute to 5-minute horizons. Requires L2 data."
        ),
        "edge": "order book imbalance directly measures supply/demand pressure",
        "regimes": ["all"],
        "timeframes": ["1m"],
        "known_sharpe": "2.0+ with L2 data",
    },
    {
        "title": "Time-of-Day Seasonality",
        "category": "seasonality",
        "summary": (
            "Markets exhibit consistent intraday patterns: "
            "9:30-10:00 high volatility open, 10:00-11:30 trend establishment, "
            "11:30-14:00 low-volume doldrums, 14:00-16:00 institutional activity resumes. "
            "Strategy: trade only during high-probability windows. "
            "Filter: avoid 11:30-13:30 for momentum strategies, prefer open and close."
        ),
        "edge": "institutional trading schedules create predictable volume and volatility patterns",
        "regimes": ["all"],
        "timeframes": ["1m", "5m"],
        "known_sharpe": "regime-dependent filter rather than standalone strategy",
    },
    {
        "title": "LSTM Price Prediction",
        "category": "ml",
        "summary": (
            "Long Short-Term Memory networks capture sequential dependencies in price data. "
            "Input: sequence of N bars with OHLCV + technical indicators (RSI, MACD, volume ratio). "
            "Output: probability of price increase in next M bars. "
            "Training: rolling window with walk-forward validation. "
            "Key challenge: overfitting — use dropout, L2 regularization, large training sets."
        ),
        "edge": "non-linear pattern recognition across multiple timeframes simultaneously",
        "regimes": ["all"],
        "timeframes": ["1m", "5m"],
        "known_sharpe": "varies widely — 0.5 to 2.0+ depending on feature engineering",
    },
    {
        "title": "XGBoost Classification for Direction Prediction",
        "category": "ml",
        "summary": (
            "Gradient boosted trees classify next-bar direction as up/down/flat. "
            "Features: rolling returns, RSI, MACD, volume z-score, time-of-day, ATR. "
            "Label: binary (up=1 if close > open + threshold, else 0). "
            "Walk-forward validation essential. Feature importance analysis reveals which signals matter. "
            "Ensemble with other models to improve robustness."
        ),
        "edge": "captures non-linear interactions between technical features",
        "regimes": ["all"],
        "timeframes": ["1m", "5m"],
        "known_sharpe": "0.8 - 1.6 with good feature engineering",
    },
    {
        "title": "Transformer Attention for Time Series",
        "category": "ml",
        "summary": (
            "Self-attention mechanism identifies which past bars are most relevant to current prediction. "
            "Unlike LSTM, can attend to any point in the sequence equally. "
            "Input: multi-variate time series (OHLCV + features). "
            "Architecture: encoder-only (BERT-style) or full encoder-decoder. "
            "Key advantage: parallelizable training, captures long-range dependencies."
        ),
        "edge": "attention mechanism selectively weights historically relevant patterns",
        "regimes": ["all"],
        "timeframes": ["1m", "5m"],
        "known_sharpe": "0.9 - 2.0 on liquid markets with sufficient training data",
    },
    {
        "title": "Volatility Regime Switching",
        "category": "regime",
        "summary": (
            "Markets alternate between low-vol (trending or choppy) and high-vol (breakout) regimes. "
            "Hidden Markov Models or realized vol ratios identify current regime. "
            "Strategy: deploy momentum in trending, mean-reversion in choppy, reduce size in high-vol. "
            "Regime indicator: 5-day realized vol / 20-day realized vol."
        ),
        "edge": "different strategies perform optimally in different volatility regimes",
        "regimes": ["all"],
        "timeframes": ["1m", "5m", "1d"],
        "known_sharpe": "meta-strategy — improves Sharpe of any underlying strategy by 20-40%",
    },
]


class ResearchStore:
    """
    knowledge base for the research agent.
    pre-seeded with known strategies and market research.
    research agent adds new documents as it discovers them.
    """

    def __init__(self, chroma_dir: Path = CHROMA_DIR):
        chroma_dir.mkdir(parents=True, exist_ok=True)
        self.chroma = chromadb.PersistentClient(path=str(chroma_dir))
        self.col    = self.chroma.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(f"research store ready | docs={self.col.count()}")

    # ------------------------------------------------------------------
    # seeding
    # ------------------------------------------------------------------

    def seed_known_strategies(self, force: bool = False) -> int:
        """
        embed and store the built-in strategy knowledge base.
        call once on setup. skips if already seeded unless force=True.
        """
        if self.col.count() >= len(KNOWN_STRATEGIES) and not force:
            log.info(f"research store already seeded ({self.col.count()} docs)")
            return 0

        texts = [self._doc_to_text(s) for s in KNOWN_STRATEGIES]
        ids   = [f"strategy_{i}" for i in range(len(KNOWN_STRATEGIES))]
        metas = [
            {
                "title":      s["title"],
                "category":   s["category"],
                "edge":       s["edge"],
                "regimes":    ",".join(s.get("regimes", [])),
                "timeframes": ",".join(s.get("timeframes", [])),
                "source":     "builtin",
                "added_at":   datetime.utcnow().isoformat(),
            }
            for s in KNOWN_STRATEGIES
        ]

        log.info(f"seeding {len(texts)} known strategies...")
        vectors = embed_batch(texts, model=LARGE)

        self.col.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
        log.info(f"seeded {len(texts)} strategy documents")
        return len(texts)

    # ------------------------------------------------------------------
    # adding new documents
    # ------------------------------------------------------------------

    def add_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        category: str = "general",
        source: str = "research_agent",
        extra_meta: dict = None,
    ) -> str:
        """
        add a new research document (paper, blog post, strategy writeup).
        called by the research agent when it finds something relevant.
        """
        text = f"Title: {title}.\n\n{content}"
        vec  = embed(text, model=LARGE)

        meta = {
            "title":    title,
            "category": category,
            "source":   source,
            "added_at": datetime.utcnow().isoformat(),
        }
        if extra_meta:
            meta.update(extra_meta)

        self.col.upsert(ids=[doc_id], embeddings=[vec], documents=[text], metadatas=[meta])
        log.info(f"document added | id={doc_id} title={title}")
        return doc_id

    def add_batch(self, documents: list[dict]) -> int:
        """
        bulk add documents. each dict needs: id, title, content, category.
        """
        texts  = [f"Title: {d['title']}.\n\n{d['content']}" for d in documents]
        ids    = [d["id"] for d in documents]
        metas  = [
            {"title": d["title"], "category": d.get("category",""), "source": d.get("source","research_agent"),
             "added_at": datetime.utcnow().isoformat()}
            for d in documents
        ]
        vectors = embed_batch(texts, model=LARGE)
        self.col.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
        log.info(f"bulk added {len(documents)} documents")
        return len(documents)

    # ------------------------------------------------------------------
    # querying
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 5, category: str = None) -> list:
        """
        semantic search over research knowledge base.
        the research agent uses this to find relevant prior art.
        """
        vector = embed(query, model=LARGE)
        where  = {"category": category} if category else None

        results = self.col.query(
            query_embeddings = [vector],
            n_results        = min(k, max(1, self.col.count())),
            where            = where,
            include          = ["metadatas", "distances", "documents"],
        )

        return [
            {
                "title":      m.get("title", ""),
                "category":   m.get("category", ""),
                "similarity": round(1 - d, 4),
                "source":     m.get("source", ""),
                "snippet":    doc[:300] + "..." if len(doc) > 300 else doc,
            }
            for m, d, doc in zip(
                results["metadatas"][0],
                results["distances"][0],
                results["documents"][0],
            )
        ]

    def find_related_to_strategy(self, strategy: dict, k: int = 5) -> list:
        """find research docs related to a strategy idea"""
        query = (
            f"{strategy.get('name', '')} "
            f"{strategy.get('description', '')} "
            f"{' '.join(str(v) for v in strategy.get('params', {}).values())}"
        )
        return self.search(query, k=k)

    # ------------------------------------------------------------------
    # text representation
    # ------------------------------------------------------------------

    @staticmethod
    def _doc_to_text(doc: dict) -> str:
        return (
            f"Title: {doc['title']}. "
            f"Category: {doc['category']}. "
            f"Summary: {doc['summary']} "
            f"Edge: {doc.get('edge', '')}. "
            f"Best regimes: {', '.join(doc.get('regimes', []))}. "
            f"Timeframes: {', '.join(doc.get('timeframes', []))}. "
            f"Known Sharpe: {doc.get('known_sharpe', 'unknown')}."
        )

    def stats(self) -> dict:
        return {
            "total_documents": self.col.count(),
            "collection":      COLLECTION,
            "model":           LARGE,
        }
