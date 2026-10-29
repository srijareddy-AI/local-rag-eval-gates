"""Local vector store.

The default store is an in-memory list of chunks plus their vectors,
persisted as JSON under ``.rag_store/`` if requested. If ChromaDB is
installed, ``ChromaStore`` (same interface) can be used instead — the
pipeline code does not depend on which store is behind it, and no data
ever leaves the local machine.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.embeddings import TfidfEmbedder, cosine, ollama_embed, ollama_available
from src.ingest import Chunk


class LocalStore:
    """Chunks + vectors, with an Ollama-first, TF-IDF-fallback embedder."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.backend = "tfidf"
        self._tfidf = TfidfEmbedder().fit([c.text for c in chunks])
        self.vectors = [self._tfidf.embed(c.text) for c in chunks]
        if ollama_available():
            ollama_vectors = ollama_embed([c.text for c in chunks])
            if ollama_vectors is not None:
                self.backend = "ollama"
                self.vectors = ollama_vectors

    def query(self, query_text: str, top_k: int) -> list[Chunk]:
        """Return the top-k chunks by cosine similarity to the query."""
        if self.backend == "ollama":
            query_vector = ollama_embed([query_text])
            if query_vector is None:
                # Ollama went away mid-run: fall back, do not crash.
                self.backend = "tfidf"
                query_vector = [self._tfidf.embed(query_text)]
        else:
            query_vector = [self._tfidf.embed(query_text)]
        scored = sorted(
            zip(self.chunks, self.vectors),
            key=lambda pair: cosine(query_vector[0], pair[1]),
            reverse=True,
        )
        return [chunk for chunk, _ in scored[:top_k]]

    def save(self, path: Path) -> None:
        """Persist chunks (vectors are rebuilt on load, keeping this small)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                [
                    {
                        "chunk_id": c.chunk_id,
                        "source": c.source,
                        "text": c.text,
                    }
                    for c in self.chunks
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
