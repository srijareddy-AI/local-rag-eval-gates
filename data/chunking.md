# Chunking Trade-offs

Chunking splits long documents into smaller pieces so a retriever can
return the most relevant piece instead of a whole document. Two
parameters matter most: chunk size and chunk overlap.

A larger chunk preserves more surrounding context, but it dilutes
relevance, because a chunk can contain several unrelated facts and the
retriever cannot tell which part matched the query. A larger chunk also
uses more of the model's context window, which increases cost and
latency. A smaller chunk is more precise, but it can split a single
fact across two chunks, so neither chunk alone answers the question.

Overlap helps with that split-fact problem: each chunk repeats a small
tail of the previous chunk, so a fact near a boundary still appears
whole in at least one chunk. Overlap of about one eighth of the chunk
size is a reasonable starting point. The right chunk size is not a
guess: it should be measured on a fixed test set, which is what the
chunking experiment in this project does.
