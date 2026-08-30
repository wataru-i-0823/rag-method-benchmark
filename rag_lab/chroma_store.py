from __future__ import annotations

from pathlib import Path
import hashlib

from .types import Document, SearchResult
from .text import tokenize


class ChromaE5Retriever:
    """Persistent Chroma retrieval with a dependency-free local embedding."""

    name = "chroma_local"

    def __init__(
        self,
        documents: list[Document],
        path: str = "data/chroma",
        collection_name: str = "rag_documents",
        dimensions: int = 384,
    ):
        import chromadb
        from chromadb.config import Settings
        self.documents = {document.id: document for document in documents}
        self.dimensions = dimensions
        self.client = chromadb.PersistentClient(path=str(Path(path)), settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})
        self._upsert(documents)

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        values = [0.0] * self.dimensions
        for token in tokenize(text):
            index = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.dimensions
            values[index] += 1.0
        norm = sum(value * value for value in values) ** 0.5 or 1.0
        return [value / norm for value in values]

    def _upsert(self, documents: list[Document]) -> None:
        if not documents:
            return
        self.collection.upsert(
            ids=[document.id for document in documents],
            documents=[document.text for document in documents],
            metadatas=[{"title": document.title} for document in documents],
            embeddings=self._embed_documents([document.text for document in documents]),
        )

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        vector = [self._embed(query)]
        response = self.collection.query(query_embeddings=vector, n_results=min(k, len(self.documents)), include=["distances"])
        ids, distances = response["ids"][0], response["distances"][0]
        return [SearchResult(self.documents[id_], 1 - distance) for id_, distance in zip(ids, distances)]
