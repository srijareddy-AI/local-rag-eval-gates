"""Chunking / top-k experiment.

Compares chunk sizes and top-k values on the fixed test set, measuring
retrieval accuracy only (generation is skipped, so this runs fast and
deterministically). Results are written to results/experiment_results.md
as a table — every number in that file comes from an actual run of
this script, never from a guess.

Usage:
    python3 -m src.experiment
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import ROOT
from src.evaluate import run_eval
from src.ingest import load_documents_with_params
from src.store import LocalStore

CHUNK_SIZES = [400, 1000]
TOP_KS = [1, 3]


def main() -> None:
    test_set = json.loads((ROOT / "evals" / "test_set.json").read_text())
    lines = [
        "# Chunking / top-k experiment",
        "",
        "Retrieval accuracy on the fixed test set in evals/test_set.json. "
        "Generation is skipped in this experiment, so these numbers measure "
        "retrieval only.",
        "",
        "| Chunk size (chars) | Overlap (chars) | Top-k | Retrieval accuracy |",
        "|--------------------|-----------------|-------|--------------------|",
    ]
    for chunk_size in CHUNK_SIZES:
        for top_k in TOP_KS:
            store = LocalStore(
                load_documents_with_params(ROOT / "data", chunk_size, 100)
            )
            result = run_eval(test_set, store, top_k)
            lines.append(
                f"| {chunk_size} | 100 | {top_k} | "
                f"{result['retrieval_accuracy']:.3f} |"
            )
            print(lines[-1])
    output = ROOT / "results" / "experiment_results.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
