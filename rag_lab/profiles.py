"""Execution profiles keep deployment choices separate from retrieval methods."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionProfile:
    name: str
    description: str
    default_methods: str
    chroma_path: str
    embedding_device: str | None


PROFILES = {
    "local": ExecutionProfile(
        name="local",
        description="Laptop baseline: no API key or GPU; uses Chroma with hash embeddings.",
        default_methods="bm25,dense,chroma_hash,hyde,reverse_hyde,hybrid,advanced,agentic,langgraph_agentic,graph,corpus2skill",
        chroma_path="data/chroma",
        embedding_device="cpu",
    ),
    "colab": ExecutionProfile(
        name="colab",
        description="Google Colab: multilingual-e5-small on an available GPU, with an ephemeral Chroma index.",
        default_methods="bm25,dense,chroma_e5,hyde,reverse_hyde,hybrid,advanced,agentic,langgraph_agentic,graph,corpus2skill",
        chroma_path="data/chroma",
        embedding_device="auto",
    ),
    "cloud": ExecutionProfile(
        name="cloud",
        description="Cloud VM/container: E5 on CPU/GPU; set --chroma-path to a mounted persistent volume.",
        default_methods="bm25,dense,chroma_e5,hybrid,advanced,agentic,langgraph_agentic,graph,corpus2skill",
        chroma_path="data/chroma",
        embedding_device="auto",
    ),
}


def get_profile(name: str) -> ExecutionProfile:
    try:
        return PROFILES[name]
    except KeyError as error:
        raise ValueError(f"Unknown execution profile: {name}") from error
