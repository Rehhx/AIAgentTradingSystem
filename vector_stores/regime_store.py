"""
embeddings/regime_store.py
--------------------------
embeds rolling windows of 1m candle data to detect market regimes.

the idea:
  - take a 60-bar window of OHLCV data (= 1 hour of 1m bars)
  - convert it to a rich text description of what the market is doing
  - embed it with text-embedding-3-large
  - store in chromadb
  - at inference time: embed the CURRENT window, find the k most similar
    historical windows, look at what happened in the next N bars

this gives us:
  1. regime labels  — "this looks like a breakout / mean reversion / chop"
  2. forward return distributions — "after this pattern, 60% of the time
     price was up >0.3% in the next 15 bars"
  3. regime-aware signal filtering — only trade certain strategies in
     certain regimes

text-embedding-3-large used because we need high resolution to distinguish
subtle differences in bar sequences (e.g. tight consolidation before a
breakout vs. tight consolidation before a breakdown looks similar in text).
"""

import logging
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np
import pandas as pd

from vector_stores.client import LARGE, embed, embed_batch

log = logging.getLogger("vector_stores.regime_store")

CHROMA_DIR   = Path("vector_stores/chroma_db")
COLLECTION   = "market_regimes"
WINDOW_BARS  = 60     # 60 x 1m bars = 1 hour lookback window
FORWARD_BARS = 15     # predict what happens in next 15 bars


class RegimeStore:
    """
    stores embedded candle windows and supports:
      - bulk indexing of historical data
      - real-time regime lookup for live trading
      - forward return analysis for any current pattern
    """

    def __init__(self, chroma_dir: Path = CHROMA_DIR):
        chroma_dir.mkdir(parents=True, exist_ok=True)
        self.chroma  = chromadb.PersistentClient(path=str(chroma_dir))
        self.col     = self.chroma.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},  # cosine similarity for patterns
        )
        log.info(f"regime store ready | collection={COLLECTION} | docs={self.col.count()}")

    # ------------------------------------------------------------------
    # indexing
    # ------------------------------------------------------------------

    def index_ticker(
        self,
        ticker: str,
        df: pd.DataFrame,
        step: int = 15,          # slide window every 15 bars (not every 1)
        batch_size: int = 50,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> int:
        """
        index all windows from a ticker's 1m dataframe.

        args:
            df         : clean OHLCV dataframe from data.loader
            step       : how many bars to slide between windows (15 = every 15min)
            batch_size : how many windows to embed per openai api call
            start/end  : optional date range to index (useful for incremental updates)

        returns: number of windows indexed
        """
        if start:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if end:
            df = df[df.index <= pd.Timestamp(end, tz="UTC")]

        if len(df) < WINDOW_BARS + FORWARD_BARS:
            log.warning(f"{ticker}: not enough data to index (need {WINDOW_BARS + FORWARD_BARS} bars)")
            return 0

        windows   = []
        meta_list = []

        for i in range(0, len(df) - WINDOW_BARS - FORWARD_BARS, step):
            window  = df.iloc[i : i + WINDOW_BARS]
            forward = df.iloc[i + WINDOW_BARS : i + WINDOW_BARS + FORWARD_BARS]

            # skip if window spans multiple days (don't want overnight gaps)
            if window.index[0].date() != window.index[-1].date():
                continue

            fwd_return = (forward["close"].iloc[-1] - window["close"].iloc[-1]) / window["close"].iloc[-1]

            windows.append(self._window_to_text(ticker, window))
            meta_list.append({
                "ticker":       ticker,
                "window_start": str(window.index[0]),
                "window_end":   str(window.index[-1]),
                "fwd_return":   round(float(fwd_return), 6),
                "fwd_bars":     FORWARD_BARS,
                "close_at_end": round(float(window["close"].iloc[-1]), 4),
                "volume_zscore": round(float(self._volume_zscore(window)), 4),
                "atr_pct":      round(float(self._atr_pct(window)), 6),
                "regime_label": self._label_regime(window, fwd_return),
            })

        if not windows:
            log.warning(f"{ticker}: no valid intraday windows found")
            return 0

        log.info(f"indexing {len(windows)} windows for {ticker}...")

        # batch embed and upsert
        indexed = 0
        for b in range(0, len(windows), batch_size):
            batch_texts = windows[b : b + batch_size]
            batch_meta  = meta_list[b : b + batch_size]
            batch_ids   = [
                f"{ticker}_{m['window_start'].replace(' ', 'T').replace(':', '').replace('+', 'p')}"
                for m in batch_meta
            ]

            vectors = embed_batch(batch_texts, model=LARGE)

            self.col.upsert(
                ids        = batch_ids,
                embeddings = vectors,
                documents  = batch_texts,
                metadatas  = batch_meta,
            )
            indexed += len(batch_texts)
            log.info(f"  {ticker}: {indexed}/{len(windows)} windows indexed")

        log.info(f"done indexing {ticker} | {indexed} windows stored")
        return indexed

    # ------------------------------------------------------------------
    # querying
    # ------------------------------------------------------------------

    def find_similar(
        self,
        ticker: str,
        current_window: pd.DataFrame,
        k: int = 20,
        same_ticker_only: bool = False,
    ) -> dict:
        """
        given the current 60-bar window, find k most similar historical patterns.

        returns:
          {
            "similar_windows": [...],
            "forward_return_stats": {mean, std, pct_positive, p25, p75},
            "regime": "breakout" | "mean_reversion" | "trending" | "chop",
            "confidence": 0.0-1.0,
          }
        """
        if len(current_window) < WINDOW_BARS:
            raise ValueError(f"need {WINDOW_BARS} bars, got {len(current_window)}")

        text   = self._window_to_text(ticker, current_window.iloc[-WINDOW_BARS:])
        vector = embed(text, model=LARGE)

        where = {"ticker": ticker} if same_ticker_only else None
        results = self.col.query(
            query_embeddings = [vector],
            n_results        = k,
            where            = where,
            include          = ["metadatas", "distances", "documents"],
        )

        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        if not metas:
            return {"similar_windows": [], "regime": "unknown", "confidence": 0.0}

        fwd_returns = [m["fwd_return"] for m in metas]
        arr         = np.array(fwd_returns)
        similarities = [1 - d for d in distances]   # cosine: 1=identical

        # vote on regime label from top-k neighbors
        labels  = [m["regime_label"] for m in metas]
        regime  = max(set(labels), key=labels.count)

        # confidence = avg similarity of top k
        confidence = float(np.mean(similarities))

        return {
            "similar_windows": [
                {**m, "similarity": round(s, 4)}
                for m, s in zip(metas, similarities)
            ],
            "forward_return_stats": {
                "mean":         round(float(arr.mean()), 6),
                "std":          round(float(arr.std()), 6),
                "pct_positive": round(float((arr > 0).mean()), 4),
                "p25":          round(float(np.percentile(arr, 25)), 6),
                "p75":          round(float(np.percentile(arr, 75)), 6),
            },
            "regime":     regime,
            "confidence": round(confidence, 4),
            "k":          len(metas),
        }

    # ------------------------------------------------------------------
    # text representation of a candle window
    # ------------------------------------------------------------------

    def _window_to_text(self, ticker: str, window: pd.DataFrame) -> str:
        """
        converts a 60-bar OHLCV window into a rich text description.
        the richer the text, the better the embedding captures the pattern.

        includes: price action, volume profile, momentum, volatility,
                  time-of-day, bar-level structure.
        """
        c      = window["close"]
        h      = window["high"]
        l      = window["low"]
        v      = window["volume"]
        o      = window["open"]

        pct_chg     = (c.iloc[-1] - c.iloc[0]) / c.iloc[0] * 100
        high_pct    = (h.max() - c.iloc[0]) / c.iloc[0] * 100
        low_pct     = (l.min() - c.iloc[0]) / c.iloc[0] * 100
        vol_trend   = "increasing" if v.iloc[-20:].mean() > v.iloc[:20].mean() else "decreasing"
        price_trend = "upward" if c.iloc[-1] > c.iloc[len(c)//2] > c.iloc[0] else (
                      "downward" if c.iloc[-1] < c.iloc[len(c)//2] < c.iloc[0] else "mixed")

        # bar structure
        body_sizes  = abs(c - o)
        wick_sizes  = (h - l) - body_sizes
        avg_body    = body_sizes.mean()
        avg_wick    = wick_sizes.mean()
        bar_type    = "doji-heavy" if avg_wick > avg_body * 2 else (
                      "strong-body" if avg_body > avg_wick else "balanced")

        # momentum
        c_arr   = c.values
        upper   = np.mean(c_arr[-10:])
        mid     = np.mean(c_arr[25:35])
        lower   = np.mean(c_arr[:10])
        momentum = "accelerating" if upper > mid > lower else (
                   "decelerating" if upper < mid < lower else "oscillating")

        # volatility
        returns  = np.diff(np.log(c_arr))
        vol_ann  = np.std(returns) * np.sqrt(252 * 390) * 100
        vol_desc = "low" if vol_ann < 20 else "moderate" if vol_ann < 50 else "high"

        # time of day
        hour = window.index[-1].tz_convert("America/New_York").hour
        tod  = "open" if hour < 10 else "midday" if hour < 14 else "close"

        return (
            f"Ticker: {ticker}. "
            f"60-minute window ending at market {tod}. "
            f"Price moved {pct_chg:+.3f}% overall, "
            f"reaching {high_pct:+.3f}% high and {low_pct:+.3f}% low from start. "
            f"Trend direction: {price_trend}. "
            f"Momentum is {momentum}. "
            f"Volume is {vol_trend} with {vol_desc} volatility ({vol_ann:.1f}% annualized). "
            f"Bars show {bar_type} candle structure. "
            f"Starting close: {c.iloc[0]:.4f}, ending close: {c.iloc[-1]:.4f}. "
            f"High: {h.max():.4f}, low: {l.min():.4f}. "
            f"Average volume per bar: {int(v.mean())}."
        )

    # ------------------------------------------------------------------
    # feature helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _volume_zscore(window: pd.DataFrame) -> float:
        v    = window["volume"]
        mean = v.mean()
        std  = v.std()
        return float((v.iloc[-1] - mean) / std) if std > 0 else 0.0

    @staticmethod
    def _atr_pct(window: pd.DataFrame) -> float:
        tr = pd.concat([
            window["high"] - window["low"],
            abs(window["high"] - window["close"].shift()),
            abs(window["low"]  - window["close"].shift()),
        ], axis=1).max(axis=1)
        return float(tr.mean() / window["close"].mean())

    @staticmethod
    def _label_regime(window: pd.DataFrame, fwd_return: float) -> str:
        """heuristic regime label based on bar structure + return"""
        c        = window["close"]
        returns  = c.pct_change().dropna()
        vol      = returns.std()
        trend    = (c.iloc[-1] - c.iloc[0]) / c.iloc[0]

        if abs(trend) > 0.005 and vol < 0.002:
            return "trending"
        elif abs(trend) < 0.001 and vol < 0.001:
            return "chop"
        elif vol > 0.003:
            return "breakout"
        else:
            return "mean_reversion"

    def stats(self) -> dict:
        return {
            "total_windows": self.col.count(),
            "collection":    COLLECTION,
            "window_bars":   WINDOW_BARS,
            "forward_bars":  FORWARD_BARS,
            "model":         LARGE,
        }
