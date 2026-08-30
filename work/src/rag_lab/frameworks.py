"""Reference implementations using common RAG libraries without an LLM/API key."""
from __future__ import annotations

from typing import TypedDict

from .types import Document, SearchResult


def split_with_langchain(documents: list[Document], chunk_size: int = 500, overlap: int = 80) -> list[Document]:
    """Chunk source documents with LangChain's recursive splitter."""
    from langchain_core.documents import Document as LCDocument
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    inputs = [LCDocument(page_content=d.text, metadata={"source_id": d.id, "title": d.title}) for d in documents]
    return [Document(f"{part.metadata['source_id']}:{i}", part.page_content, part.metadata.get("title", "")) for i, part in enumerate(splitter.split_documents(inputs))]


def nodes_with_llamaindex(documents: list[Document]) -> list[Document]:
    """Create LlamaIndex nodes; useful for inspecting its ingestion semantics."""
    from llama_index.core import Document as LlamaDocument
    from llama_index.core.node_parser import SentenceSplitter

    nodes = SentenceSplitter(chunk_size=500, chunk_overlap=80).get_nodes_from_documents(
        [LlamaDocument(text=d.text, doc_id=d.id, metadata={"title": d.title}) for d in documents]
    )
    return [Document(node.node_id, node.get_content(), node.metadata.get("title", "")) for node in nodes]


class AgentState(TypedDict):
    query: str
    results: list[SearchResult]


def agentic_search_with_langgraph(retriever, query: str, k: int = 5) -> list[SearchResult]:
    """A minimal LangGraph retrieval graph; extend with grade/rewrite nodes later."""
    from langgraph.graph import END, START, StateGraph

    def retrieve(state: AgentState):
        return {"results": retriever.search(state["query"], k)}

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", END)
    return graph.compile().invoke({"query": query, "results": []})["results"]


def framework_documents(documents: list[Document], framework: str) -> list[Document]:
    if framework == "langchain":
        return split_with_langchain(documents)
    if framework == "llamaindex":
        return nodes_with_llamaindex(documents)
    raise ValueError(f"Unknown framework: {framework}")
