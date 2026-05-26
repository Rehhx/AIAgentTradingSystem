"""
embeddings/client.py
--------------------
central openai embedding client used by all three embedding stores.
handles batching, rate limiting, and a local disk cache so we never
re-embed the same text twice — saves credits on repeated runs.

models:
  text-embedding-3-large  (3072-dim) → regime detection + research kb
  text-embedding-3-small  (1536-dim) → strategy memory (fast lookups)
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

log = logging.getLogger("vector_stores.client")

# model constants
LARGE = "text-embedding-3-large"   # 3072-dim  — regime + research
SMALL = "text-embedding-3-small"   # 1536-dim  — strategy memory

DIMS = {LARGE: 3072, SMALL: 1536}

# cache lives at vector_stores/.cache/ so we never re-embed identical text
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


class EmbeddingClient:
    """
    thin wrapper around openai embeddings api.
    - batches up to 100 texts per api call
    - disk-caches every embedding keyed by (model, sha256(text))
    - respects rate limits with configurable retry backoff
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache: bool = True,
        max_retries: int = 3,
    ):
        self.client     = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.use_cache  = cache
        self.max_retries = max_retries
        self._cache: dict = {}  # in-memory layer on top of disk cache

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def embed(self, text: str, model: str = LARGE) -> list[float]:
        """embed a single string. returns list of floats."""
        return self.embed_batch([text], model)[0]

    def embed_batch(
        self,
        texts: list[str],
        model: str = LARGE,
        batch_size: int = 100,
    ) -> list[list[float]]:
        """
        embed a list of strings. automatically batches and caches.
        returns list of embedding vectors in same order as input.
        """
        results  = [None] * len(texts)
        to_fetch = []   # (original_idx, text) pairs not in cache

        for i, text in enumerate(texts):
            cached = self._get_cache(text, model)
            if cached is not None:
                results[i] = cached
            else:
                to_fetch.append((i, text))

        if to_fetch:
            log.info(f"embedding {len(to_fetch)} texts via openai | model={model}")
            # batch the api calls
            for batch_start in range(0, len(to_fetch), batch_size):
                batch      = to_fetch[batch_start : batch_start + batch_size]
                idxs       = [b[0] for b in batch]
                batch_texts = [b[1] for b in batch]

                vectors = self._call_api(batch_texts, model)

                for idx, text, vector in zip(idxs, batch_texts, vectors):
                    results[idx] = vector
                    self._set_cache(text, model, vector)

        return results

    # ------------------------------------------------------------------
    # api call with retry
    # ------------------------------------------------------------------

    def _call_api(self, texts: list[str], model: str) -> list[list[float]]:
        """calls openai embeddings api with exponential backoff on rate limits"""
        for attempt in range(self.max_retries):
            try:
                # openai api expects non-empty strings
                cleaned = [t.strip() or " " for t in texts]
                resp    = self.client.embeddings.create(
                    input=cleaned,
                    model=model,
                )
                return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    log.warning(f"rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    log.error(f"embedding api error: {e}")
                    raise
        raise RuntimeError("max retries exceeded")

    # ------------------------------------------------------------------
    # disk cache (keyed by sha256 of model+text)
    # ------------------------------------------------------------------

    def _cache_key(self, text: str, model: str) -> str:
        h = hashlib.sha256(f"{model}:{text}".encode()).hexdigest()
        return h

    def _cache_path(self, key: str) -> Path:
        # shard into subdirs so we don't get millions of files in one dir
        return CACHE_DIR / key[:2] / f"{key}.json"

    def _get_cache(self, text: str, model: str) -> Optional[list[float]]:
        if not self.use_cache:
            return None
        key = self._cache_key(text, model)
        if key in self._cache:
            return self._cache[key]
        path = self._cache_path(key)
        if path.exists():
            v = json.loads(path.read_text())
            self._cache[key] = v
            return v
        return None

    def _set_cache(self, text: str, model: str, vector: list[float]):
        if not self.use_cache:
            return
        key  = self._cache_key(text, model)
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(vector))
        self._cache[key] = vector

    def cache_stats(self) -> dict:
        total = sum(1 for _ in CACHE_DIR.rglob("*.json"))
        return {"cached_embeddings": total, "cache_dir": str(CACHE_DIR)}


# module-level singleton — import and use directly
_client: Optional[EmbeddingClient] = None


def get_client() -> EmbeddingClient:
    global _client
    if _client is None:
        _client = EmbeddingClient()
    return _client


def embed(text: str, model: str = LARGE) -> list[float]:
    return get_client().embed(text, model)


def embed_batch(texts: list[str], model: str = LARGE) -> list[list[float]]:
    return get_client().embed_batch(texts, model)
