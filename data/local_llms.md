# Running Language Models Locally with Ollama

Ollama runs open language models on your own machine, so prompts and
documents do not need to leave the local environment. A typical local
setup uses Llama 3 for generation and a small embedding model such as
nomic-embed-text for retrieval embeddings, both served by Ollama on
localhost.

The trade-offs of local models are cost, latency, and privacy against
raw capability. Local models cost nothing per query beyond electricity
and hardware, because there are zero paid APIs, and they keep data
private, but they are generally smaller and slower per token than the
largest hosted models. For a retrieval system, that trade-off is often
worth it: retrieval does most of the factual work, and the generator
mainly needs to summarise the retrieved context faithfully.

When no local model is running, this project falls back to a
dependency-free TF-IDF retriever and an extractive answer, so the
pipeline and its eval gate can still run and be measured. The fallback
is labelled as extractive in the output, so it is never mistaken for
a model-generated answer.
