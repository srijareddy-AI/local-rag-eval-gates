"""Eval gate: measure the pipeline on a fixed test set, then pass/fail.

Metrics (both are simple, auditable, and need no external judge):
- Retrieval accuracy: fraction of test questions where at least one
  retrieved chunk comes from the expected source document.
- Groundedness: fraction of the expected key facts that appear
  (case-insensitively, as substrings) in the generated answer.

The gate passes only if both metrics meet the thresholds in
evals/gate_config.json. A prompt or chunking change is therefore
measurable: re-run this script and compare the table, instead of
claiming an improvement.

Usage:
    python3 -m src.evaluate
    python3 -m src.evaluate --top-k 5 --chunk-size 1000 --overlap 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    ROOT,
)
from src.generate import generate
from src.ingest import load_documents_with_params
from src.retrieve import retrieve
from src.store import LocalStore


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def run_eval(
    test_set: list[dict],
    store: LocalStore,
    top_k: int,
) -> dict:
    """Run every test question and return per-question + summary rows."""
    rows: list[dict] = []
    retrieved_hits = 0
    groundedness_sum = 0.0
    for item in test_set:
        context = retrieve(store, item["question"], top_k)
        answer, method = generate(item["question"], context)
        retrieved = any(c.source == item["expected_source"] for c in context)
        retrieved_hits += int(retrieved)
        answer_norm = _normalise(answer)
        facts_found = sum(
            1
            for fact in item["key_facts"]
            if _normalise(fact) in answer_norm
        )
        groundedness = facts_found / max(len(item["key_facts"]), 1)
        groundedness_sum += groundedness
        rows.append(
            {
                "question": item["question"],
                "expected_source": item["expected_source"],
                "retrieved": retrieved,
                "retrieved_sources": [c.source for c in context],
                "method": method,
                "key_facts_found": f"{facts_found}/{len(item['key_facts'])}",
                "groundedness": round(groundedness, 3),
                "answer": answer,
            }
        )
    n = max(len(test_set), 1)
    return {
        "n_questions": len(test_set),
        "retrieval_accuracy": round(retrieved_hits / n, 3),
        "groundedness": round(groundedness_sum / n, 3),
        "backend": store.backend,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG eval gate.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--test-set", type=Path, default=ROOT / "evals" / "test_set.json")
    parser.add_argument("--gate-config", type=Path, default=ROOT / "evals" / "gate_config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "eval_results.json")
    args = parser.parse_args()

    gate = json.loads(args.gate_config.read_text(encoding="utf-8"))
    test_set = json.loads(args.test_set.read_text(encoding="utf-8"))
    store = LocalStore(
        load_documents_with_params(args.data_dir, args.chunk_size, args.overlap)
    )
    result = run_eval(test_set, store, args.top_k)
    result.update(
        {
            "top_k": args.top_k,
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "thresholds": gate,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Backend: {result['backend']} | n={result['n_questions']}")
    print(f"Retrieval accuracy: {result['retrieval_accuracy']} "
          f"(gate >= {gate['min_retrieval_accuracy']})")
    print(f"Groundedness:       {result['groundedness']} "
          f"(gate >= {gate['min_groundedness']})")
    print()
    print(f"{'Q':<4}{'Retrieved':<11}{'Grounded':<10}Question")
    for i, row in enumerate(result["rows"], 1):
        print(f"{i:<4}{str(row['retrieved']):<11}{row['groundedness']:<10}"
              f"{row['question']}")
    passed = (
        result["retrieval_accuracy"] >= gate["min_retrieval_accuracy"]
        and result["groundedness"] >= gate["min_groundedness"]
    )
    print()
    print("GATE: PASS" if passed else "GATE: FAIL")
    if not passed:
        print("Do not ship the change that produced this result — "
              "fix retrieval or grounding first.")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
