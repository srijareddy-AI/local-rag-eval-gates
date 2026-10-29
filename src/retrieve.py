"""Retrieval step: thin wrapper over the store, kept separate so the
retrieval policy (top-k, reranking) can change without touching the
pipeline or the eval code.
"""

from __future__ import annotations

from src.ingest import Chunk
from src.store import LocalStore


def retrieve(store: LocalStore, query: str, top_k: int) -> list[Chunk]:
    """Return the top-k chunks for a query."""
    return store.query(query, top_k)
