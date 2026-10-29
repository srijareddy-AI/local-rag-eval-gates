"""Answer generation: Ollama first, extractive fallback.

Primary path: local Ollama chat (``llama3`` by default) with the
retrieved chunks as its only context, instructed to answer only from
that context. Fallback path: an extractive answer — the top retrieved
chunk, cited by source — so the pipeline never invents an answer when
no local model is running. The fallback is labelled as extractive in
the pipeline output, and the eval gate scores it the same way.
"""

from __future__ import annotations

import requests

from src.config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL
from src.ingest import Chunk

_SYSTEM_PROMPT = (
    "Answer only from the context below. If the context does not "
    "contain the answer, say so instead of guessing."
)


def ollama_generate(query: str, context_chunks: list[Chunk]) -> str | None:
    """Generate with local Ollama, or return None if unavailable."""
    context = "\n\n".join(
        f"[{c.chunk_id}] {c.text}" for c in context_chunks
    )
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_CHAT_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {query}",
                    },
                ],
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
    except (requests.RequestException, KeyError):
        return None


def extractive_answer(query: str, context_chunks: list[Chunk]) -> str:
    """Answer from the top chunk only, with an explicit citation."""
    if not context_chunks:
        return "I don't know — no retrieved context covers this question."
    top = context_chunks[0]
    return (
        f"[Extractive answer from {top.chunk_id}] {top.text} "
        f"(Sources: {', '.join(c.chunk_id for c in context_chunks)})"
    )


def generate(query: str, context_chunks: list[Chunk]) -> tuple[str, str]:
    """Return (answer, method), where method is 'ollama' or 'extractive'."""
    answer = ollama_generate(query, context_chunks)
    if answer:
        return answer, "ollama"
    return extractive_answer(query, context_chunks), "extractive"
