"""Configuration for the local RAG pipeline.

All defaults are local-only. No paid API is used at any point.
Ollama is the primary LLM / embedding backend when it is running;
otherwise the pipeline falls back to a dependency-free TF-IDF
retriever and an extractive answer, so the eval gate still runs.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Ollama (local only)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_MODEL = "llama3"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# Chunking (character-based; roughly 4 characters per token for English text)
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

# Retrieval
DEFAULT_TOP_K = 3

# Eval gate thresholds (see evals/gate_config.json for the same values)
GATE_MIN_RETRIEVAL_ACCURACY = 0.70
GATE_MIN_GROUNDEDNESS = 0.60
