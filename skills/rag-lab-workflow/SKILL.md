---
name: rag-lab-workflow
description: Build or modify the RAG Method Benchmark, including retrieval experiments, MLflow evaluation, Colab execution, and safe public-repository handling. Use for this repository's RAG lab work, not generic RAG questions.
---

# RAG Lab Workflow

Maintain this project as a reproducible comparison lab, not as a single fixed RAG application.

## Core workflow

- Identify the experimental layer being changed: ingestion/chunking, embeddings, vector store, retrieval method, orchestration, or evaluation. Avoid silently changing more than one layer in a comparison.
- Use `--profile local`, `--profile colab`, or `--profile cloud` to choose an execution environment. Read [execution profiles](references/execution-profiles.md) before changing profile behavior.
- Keep a no-key local path usable. Use the Colab profile for `multilingual-e5-small` and GPU work; it must degrade to CPU only when a GPU is unavailable.
- Add a deterministic test for changes to retrievers, profiles, or metrics. Use `uv` and run the unit suite before handoff.

## Data and publication safety

- Never add corpus contents, Chroma indexes, MLflow runs, `.env`, or credentials to Git. The repository is public.
- In Colab, persist raw/processed data and results in the configured Google Drive lab directory. Recreate Chroma from source documents on a new runtime.
- A ChatGPT, Claude, or Gemini app subscription is not an API credential. Do not add or assume an API key unless the user explicitly provides and authorizes it.

## Documentation

When a user-facing runtime choice changes, update the README and, when relevant, the Colab notebook. State which computation is local, Colab, or cloud, and which files are intentionally not tracked.
