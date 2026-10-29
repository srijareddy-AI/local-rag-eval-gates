"""Document ingestion and chunking.

Reads .txt and .md files from a directory, splits each document into
overlapping character chunks, and keeps the source filename on every
chunk so answers can be traced back to a document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE


@dataclass
class Chunk:
    """One retrievable piece of a document."""

    chunk_id: str
    source: str
    text: str


def _clean(text: str) -> str:
    """Collapse whitespace (including Markdown headings) to single spaces."""
    text = re.sub(r"#+\s*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks.

    A chunk size that is too large dilutes relevance (more noise per
    chunk); a size that is too small splits a fact across chunks.
    That trade-off is exactly what src/experiment.py measures.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    text = _clean(text)
    if not text:
        return []
    step = chunk_size - overlap
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]


def load_documents(data_dir: Path) -> list[Chunk]:
    """Load and chunk every .txt / .md file in ``data_dir``."""
    return load_documents_with_params(
        data_dir, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
    )


def load_documents_with_params(
    data_dir: Path, chunk_size: int, overlap: int
) -> list[Chunk]:
    """Load and chunk with explicit chunking parameters (for experiments)."""
    chunks: list[Chunk] = []
    for path in sorted(data_dir.glob("*")):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        for i, piece in enumerate(
            chunk_text(path.read_text(encoding="utf-8"), chunk_size, overlap)
        ):
            chunks.append(
                Chunk(chunk_id=f"{path.name}#{i}", source=path.name, text=piece)
            )
    return chunks
