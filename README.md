# Local Production RAG with Eval Gates

A small, local Retrieval-Augmented Generation (RAG) pipeline with a
measurement gate that decides whether a change ships. Zero paid APIs.
Everything runs on your own machine.

## The problem

The obvious fix in a RAG system is usually the wrong lever. Teams
upgrade the model, rewrite prompts, or raise retrieval scores, and the
answer does not get better — because the real bottleneck was
elsewhere: the wrong chunk was retrieved, a fact was split across
chunks, or a prompt tweak quietly dropped groundedness.

This project makes that bottleneck measurable instead of arguable.

## Pipeline

```
Query → Retrieval → Context → Generation → Evaluation
```

1. **Query** — a question, asked on the command line.
2. **Retrieval** — the top-k chunks by cosine similarity to the query.
3. **Context** — the retrieved chunks, cited by source and chunk id.
4. **Generation** — a local Ollama model (`llama3`) answering only
   from that context, or an extractive answer (labelled as such) when
   no local model is running.
5. **Evaluation** — a fixed test set, scored on every run (see below).

## How to run locally

```bash
pip install -r requirements.txt

# With Ollama running (recommended, local only):
#   ollama pull llama3
#   ollama pull nomic-embed-text
#   ollama serve

# Ask one question:
python3 -m src.pipeline --query "What is an eval gate?"

# Run the eval gate (retrieval accuracy + groundedness):
python3 -m src.evaluate

# Run the chunking / top-k experiment:
python3 -m src.experiment
```

Without Ollama, the pipeline still runs: it falls back to a
dependency-free TF-IDF retriever (standard library only) and an
extractive answer. The fallback is labelled in the output, so it is
never mistaken for a model-generated answer. ChromaDB is optional —
the built-in store is in-memory, and no data ever leaves the machine.

## The eval gate

`src/evaluate.py` runs a fixed test set of 12 questions
(`evals/test_set.json`, each with an expected source document and
expected key facts) and reports two metrics:

- **Retrieval accuracy** — fraction of questions where at least one
  retrieved chunk comes from the expected source document.
- **Groundedness** — fraction of expected key facts that appear in
  the generated answer.

The gate passes only if both metrics meet the thresholds in
`evals/gate_config.json` (retrieval accuracy >= 0.70, groundedness
>= 0.60). If either falls below its threshold, the change does not
ship. A prompt or chunking change is therefore measurable: re-run the
gate and compare the table, instead of claiming an improvement.

## Results — measured, not claimed

These numbers were produced by running `src.evaluate` and
`src.experiment` in this repository's environment, without Ollama
running (TF-IDF backend, extractive generation). Re-run them locally
with Ollama for the model-backed numbers; they will differ, and that
difference is itself worth reporting.

**Eval gate (default: chunk size 800 chars, overlap 100, top-k 3):**

| Metric | Result | Gate |
|---|---|---|
| Retrieval accuracy | 0.917 (11/12) | >= 0.70 — pass |
| Groundedness | 0.792 | >= 0.60 — pass |
| **Gate** | **PASS** | |

**Chunking / top-k experiment** (`results/experiment_results.md`,
retrieval accuracy only, generation skipped):

| Chunk size (chars) | Overlap (chars) | Top-k | Retrieval accuracy |
|---|---|---|---|
| 400 | 100 | 1 | 0.833 |
| 400 | 100 | 3 | 1.000 |
| 1000 | 100 | 1 | 0.833 |
| 1000 | 100 | 3 | 1.000 |

One takeaway: on this small test set, top-k mattered more than chunk
size — top-3 retrieved the right document for every question, while
top-1 missed two, at both chunk sizes. That is a finding about this
test set, not a universal rule; a larger test set could change it.

## Trade-offs

- **Local vs hosted models:** local models cost nothing per query
  and keep data private, but are smaller and slower per token than
  the largest hosted models. Retrieval does most of the factual work
  here, so the generator mainly needs to summarise context faithfully.
- **Chunk size:** larger chunks preserve context but dilute
  relevance and use more context window; smaller chunks are more
  precise but can split a fact. Overlap mitigates the split-fact
  problem at the cost of duplicated text.
- **Top-k:** larger top-k improves recall but adds noise and can
  bury the relevant chunk; smaller top-k is focused but risks
  missing the answer entirely.
- **Extractive fallback:** always grounded in a retrieved chunk, but
  it cannot synthesise across chunks the way a model can.

## Limitations (honest)

- The test set is small (12 questions) and written for the sample
  documents in `data/` — it measures this system, not RAG in general.
- Two questions scored 0.0 groundedness in the measured run, because
  the extractive answer came from the top chunk while the expected
  facts sat in a lower-ranked chunk. That is a real limitation of
  extractive generation, and the reason the Ollama path matters.
- Groundedness here is key-fact substring overlap, not a human or
  LLM judge — it is auditable and cheap, but it will miss paraphrases.
- The sample documents are short, general explainers written for this
  project. Bring your own documents to `data/` to test a real corpus.

## Structure

```
README.md
requirements.txt
src/
  config.py       # local-only defaults (Ollama, chunking, top-k, gates)
  ingest.py       # load .txt/.md docs, split into overlapping chunks
  embeddings.py   # Ollama embeddings, TF-IDF fallback (stdlib only)
  store.py        # local chunk + vector store
  retrieve.py     # top-k retrieval step
  generate.py     # Ollama generation, extractive fallback
  pipeline.py     # Query -> Retrieval -> Context -> Generation
  evaluate.py     # eval gate: retrieval accuracy + groundedness
  experiment.py   # chunk size x top-k experiment
data/             # sample documents (5 short explainers)
evals/
  test_set.json   # 12 Q&A pairs with expected source + key facts
  gate_config.json# gate thresholds
results/
  eval_results.json       # full per-question eval output
  experiment_results.md   # chunking / top-k results table
```

## Next questions I'm exploring

- Does reranking actually improve retrieval accuracy on a larger,
  real corpus, or does it just add latency?
- How does groundedness change when the generator is a local model
  versus the extractive fallback, on the same test set?
- What test-set size is needed before a chunk-size difference is
  trustworthy rather than noise?
