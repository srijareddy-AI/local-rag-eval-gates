"""Query pipeline: Query -> Retrieval -> Context -> Generation -> Evaluation.

Usage:
    python3 -m src.pipeline --query "What is an eval gate?"
    python3 -m src.pipeline --query "What is an eval gate?" --top-k 5

The pipeline prints the retrieved sources, the answer, and the method
used (ollama or extractive). Evaluation of a whole test set lives in
src/evaluate.py, not here — this module answers one question.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import DEFAULT_TOP_K, ROOT
from src.generate import generate
from src.ingest import load_documents
from src.retrieve import retrieve
from src.store import LocalStore


def answer_question(
    store: LocalStore, query: str, top_k: int = DEFAULT_TOP_K
) -> dict:
    """Run one question through retrieval and generation."""
    context = retrieve(store, query, top_k)
    answer, method = generate(query, context)
    return {
        "query": query,
        "method": method,
        "backend": store.backend,
        "sources": [c.chunk_id for c in context],
        "context": [c.text for c in context],
        "answer": answer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the local RAG pipeline.")
    parser.add_argument("--query", required=True, help="Question to answer")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "data", help="Sample docs dir"
    )
    args = parser.parse_args()

    store = LocalStore(load_documents(args.data_dir))
    result = answer_question(store, args.query, args.top_k)
    print(f"Backend: {result['backend']} | Generation: {result['method']}")
    print(f"Sources: {', '.join(result['sources'])}")
    print()
    print(result["answer"])


if __name__ == "__main__":
    main()
