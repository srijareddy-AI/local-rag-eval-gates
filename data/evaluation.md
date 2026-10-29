# Evaluating RAG: Retrieval Accuracy and Groundedness

Two metrics are enough to start evaluating a RAG system honestly.
Retrieval accuracy is the fraction of test questions where at least
one retrieved chunk comes from the document that actually answers the
question. Groundedness is the fraction of expected key facts that
appear in the generated answer. A system can retrieve well and still
fail on groundedness if the generator invents facts that were not in
the retrieved context.

An eval gate turns those metrics into a deploy rule. Before a change
ships — a new prompt, a different chunk size, a new embedding model —
the fixed test set is run and both metrics must meet their thresholds.
If either metric falls below its threshold, the change does not ship
until retrieval or grounding is fixed. This is the same idea as a
continuous integration test suite, applied to a language model system.

A common failure mode is prompt drift: a small prompt tweak improves
fluency but quietly drops groundedness, because the model starts
answering from memory instead of from context. An eval gate catches
that regression, because groundedness is measured on every run, not
judged by reading a few outputs by eye.
