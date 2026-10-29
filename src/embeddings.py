"""Embeddings: Ollama first, dependency-free TF-IDF fallback.

Primary path: POST to a local Ollama server
(``/api/embeddings``, model ``nomic-embed-text`` by default).
Fallback path: TF-IDF vectors built from the corpus itself, using only
the Python standard library. The fallback is deliberately explicit —
it lets the retrieval and eval code run and be measured on any machine,
even before Ollama is installed, while the production path stays local.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import requests

from src.config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "for", "with", "and", "or", "by", "at",
    "from", "as", "it", "its", "this", "that", "what", "why", "how",
    "does", "do", "did", "can", "should", "not", "no",
}


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, minus a small stopword list."""
    return [
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in _STOPWORDS
    ]


def ollama_available(timeout: float = 2.0) -> bool:
    """Return True only if a local Ollama server answers."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False


def ollama_embed(texts: list[str]) -> list[list[float]] | None:
    """Embed texts with local Ollama, or return None if unavailable."""
    vectors: list[list[float]] = []
    for text in texts:
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
                timeout=30,
            )
            response.raise_for_status()
            vectors.append(response.json()["embedding"])
        except (requests.RequestException, KeyError):
            return None
    return vectors


class TfidfEmbedder:
    """Small TF-IDF embedder (standard library only).

    Fit on the chunk corpus, then embed queries with the same vocabulary
    and IDF weights. Vectors are L2-normalised, so cosine similarity is
    just a dot product.
    """

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: list[float] = []

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        doc_counts: Counter[str] = Counter()
        for text in texts:
            doc_counts.update(set(tokenize(text)))
        self.vocab = {word: i for i, word in enumerate(sorted(doc_counts))}
        n_docs = max(len(texts), 1)
        self.idf = [
            math.log((1 + n_docs) / (1 + doc_counts[word])) + 1.0
            for word in self.vocab
        ]
        return self

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * len(self.vocab)
        for word, count in Counter(tokenize(text)).items():
            if word in self.vocab:
                idx = self.vocab[word]
                vector[idx] = count * self.idf[idx]
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity for (already normalised) vectors."""
    return sum(x * y for x, y in zip(a, b))
