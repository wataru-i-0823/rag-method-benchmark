from __future__ import annotations

import os


def traced_search(retriever, method: str):
    """Wrap retrieval in a LangSmith trace only when explicitly requested."""
    if not os.getenv("LANGSMITH_API_KEY"):
        raise RuntimeError("LANGSMITH_API_KEY is required with --langsmith; add it to .env, never to Git.")
    from langsmith import traceable

    @traceable(name=f"retrieve:{method}", run_type="retriever")
    def search(query: str, k: int):
        results = retriever.search(query, k)
        return [{"document_id": result.document.id, "score": result.score} for result in results]

    return search
