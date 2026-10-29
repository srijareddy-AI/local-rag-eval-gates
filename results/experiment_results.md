# Chunking / top-k experiment

Retrieval accuracy on the fixed test set in evals/test_set.json. Generation is skipped in this experiment, so these numbers measure retrieval only.

| Chunk size (chars) | Overlap (chars) | Top-k | Retrieval accuracy |
|--------------------|-----------------|-------|--------------------|
| 400 | 100 | 1 | 0.833 |
| 400 | 100 | 3 | 1.000 |
| 1000 | 100 | 1 | 0.833 |
| 1000 | 100 | 3 | 1.000 |
