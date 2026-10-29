# Retrieval: Top-k, Similarity, and Reranking

Retrieval ranks document chunks by similarity to the query. The most
common similarity measure for embedding vectors is cosine similarity,
which compares the direction of two vectors rather than their length.
The retriever returns the top-k chunks, where k is a parameter: top-1
returns only the single best chunk, while top-3 or top-5 returns more
context at the cost of more noise.

Top-k is a trade-off. A small top-k keeps the context focused but
risks missing the chunk that actually answers the question. A large
top-k improves recall but can bury the relevant chunk among irrelevant
ones, and the generator may then answer from the wrong chunk.

Reranking is an optional second stage: a first retriever returns a
broad candidate set, and a reranker scores those candidates more
carefully, for example by comparing the query and each chunk directly,
before the top few are passed to generation. Reranking costs extra
time, so it is worth it only when retrieval accuracy measurably
improves on the eval set.
