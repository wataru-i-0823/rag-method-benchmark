from __future__ import annotations

import hashlib
from pathlib import Path

from .types import Document, SearchResult
from .text import tokenize


class ChromaHashRetriever:
    """Chroma persistent storage with a deterministic, dependency-free vector baseline."""

    name = "chroma_hash"

    def __init__(
        self, documents: list[Document], path: str = "data/chroma", dimensions: int = 384
    ):
        import chromadb
        from chromadb.config import Settings

        self.documents = {document.id: document for document in documents}
        self.dimensions = dimensions
        self.client = chromadb.PersistentClient(
            path=str(Path(path)), settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            "rag_documents_hash", metadata={"hnsw:space": "cosine"}
        )
        self._upsert(documents)

    def _embed(self, text: str) -> list[float]:
        values = [0.0] * self.dimensions
        for token in tokenize(text):
            values[int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.dimensions] += 1.0
        norm = sum(value * value for value in values) ** 0.5 or 1.0
        return [value / norm for value in values]

    def _upsert(self, documents: list[Document]) -> None:
        if documents:
            self.collection.upsert(
                ids=[document.id for document in documents],
                documents=[document.text for document in documents],
                metadatas=[{"title": document.title} for document in documents],
                embeddings=[self._embed(document.text) for document in documents],
            )

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        response = self.collection.query(
            query_embeddings=[self._embed(query)],
            n_results=min(k, len(self.documents)),
            include=["distances"],
        )
        return [
            SearchResult(self.documents[id_], 1 - distance)
            for id_, distance in zip(response["ids"][0], response["distances"][0])
        ]


class ChromaE5Retriever:
    """Persistent Chroma retrieval using a selectable local embedding model."""

    name = "chroma_e5"

    def __init__(
        self,
        documents: list[Document],
        path: str = "data/chroma",
        embedding_model: str = "e5-small",
        device: str | None = "auto",
    ):
        import chromadb
        from chromadb.config import Settings
        model_configs = {
            "e5-small": ("intfloat/multilingual-e5-small", "rag_documents_e5_small", "query: ", "passage: "),
            "bge-m3": ("BAAI/bge-m3", "rag_documents_bge_m3", "", ""),
        }
        try:
            model_name, collection_name, self.query_prefix, self.document_prefix = model_configs[embedding_model]
        except KeyError as error:
            raise ValueError("embedding_model must be one of: e5-small, bge-m3") from error
        self.documents = {document.id: document for document in documents}
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError as error:
            raise RuntimeError(
                "chroma_e5 requires the Colab extra. Run `uv sync --extra colab` "
                "on Linux/Google Colab."
            ) from error

        resolved_device = device if device not in {None, "auto"} else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=resolved_device)
        self.client = chromadb.PersistentClient(path=str(Path(path)), settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})
        self._upsert(documents)

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            [f"{self.document_prefix}{text}" for text in texts],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def _embed(self, text: str) -> list[float]:
        return self.model.encode(
            f"{self.query_prefix}{text}", normalize_embeddings=True, show_progress_bar=False
        ).tolist()

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
