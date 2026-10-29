# Retrieval-Augmented Generation Basics

Retrieval-Augmented Generation, or RAG, answers a question by first
retrieving relevant documents and then generating an answer from that
retrieved context, instead of relying only on what a language model
memorised during training.

This project follows five stages, in order: Query, Retrieval, Context,
Generation, Evaluation. The Query is the user's question. Retrieval
finds the most relevant chunks from the local document store. Context
is the retrieved text passed to the model. Generation produces the
answer from that context. Evaluation measures whether the answer was
retrieved well and grounded in the context.

The main benefit of RAG is grounding: an answer that cites retrieved
documents is easier to check than an answer produced from memory alone.
The main risk is the same as any pipeline: an error in an early stage,
such as retrieving the wrong chunk, compounds into a wrong final answer.
