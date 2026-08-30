# RAG Method Benchmark — Agent Guide

For work that changes RAG ingestion, retrieval, evaluation, execution environments, or the Colab workflow, first read [`skills/rag-lab-workflow/SKILL.md`](skills/rag-lab-workflow/SKILL.md).

## Project boundaries

- Keep experiment variables separate: corpus/chunking, embedding model, vector store, retrieval method, orchestration, and evaluation are independently selectable.
- Use the execution profile intentionally: `local` for CPU-only learning, `colab` for GPU E5 experiments, and `cloud` for mounted persistent storage. Do not make a GPU or an API key mandatory for the local baseline.
- Treat `data/raw/`, `data/processed/`, `data/evaluation/`, `data/chroma/`, `mlruns/`, `.env`, and API keys as private. Do not commit or publish their contents.
- Rebuild Chroma from corpus and configuration on ephemeral environments. Persist inputs and evaluation outputs instead of copying a live index between machines.

## Verification and delivery

- Use `uv` for Python environments and dependencies. Keep the supported Python version in `pyproject.toml` aligned with the documented runtime.
- Add or update deterministic tests when retrieval behavior or profile routing changes. Run `python -m unittest discover -s tests -v` in the project environment.
- When changing the Colab path, keep [`notebooks/rag_lab_colab.ipynb`](notebooks/rag_lab_colab.ipynb) runnable from a fresh runtime and ensure it writes only to the designated Google Drive lab folder.
- Before a public GitHub push, inspect the staged diff for secrets and private datasets. External publication still requires the user's authorization.
