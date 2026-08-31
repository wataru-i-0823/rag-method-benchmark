"""A free, persistent semantic graph built from document embeddings."""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from .text import cosine, tokenize
from .types import Document


def hash_embedding(text: str) -> dict[str, float]:
    """Dependency-free sparse embedding used for local graph smoke tests."""
    return {token: 1.0 for token in tokenize(text)}


def build_semantic_graph(documents: list[Document], path: str | Path, threshold: float = 0.35, neighbors: int = 5, embedding_model: str = "hash") -> int:
    """Persist undirected similarity edges. Callers may supply E5/BGE similarities later."""
    if not 0 <= threshold <= 1 or neighbors < 1:
        raise ValueError("threshold must be between 0 and 1 and neighbors must be positive")
    texts = [document.title + " " + document.text for document in documents]
    if embedding_model == "hash":
        vectors = [hash_embedding(text) for text in texts]
        similarity = lambda left, right: cosine(left, right)
    else:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError("E5/BGE-M3には `uv sync --extra colab` が必要です。") from error
        models = {"e5-small": ("intfloat/multilingual-e5-small", "passage: "), "bge-m3": ("BAAI/bge-m3", "")}
        try:
            model_name, prefix = models[embedding_model]
        except KeyError as error:
            raise ValueError("embedding_model must be hash, e5-small, or bge-m3") from error
        vectors = SentenceTransformer(model_name).encode([prefix + text for text in texts], normalize_embeddings=True)
        similarity = lambda left, right: float(left @ right)
    candidates: dict[str, list[tuple[str, float]]] = {document.id: [] for document in documents}
    for index, left in enumerate(documents):
        for right_index, right in enumerate(documents[index + 1 :], start=index + 1):
            score = similarity(vectors[index], vectors[right_index])
            if score >= threshold:
                candidates[left.id].append((right.id, score))
                candidates[right.id].append((left.id, score))
    selected = {
        tuple(sorted((source, target))): score
        for source, entries in candidates.items()
        for target, score in sorted(entries, key=lambda item: item[1], reverse=True)[:neighbors]
    }
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(location) as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS semantic_nodes (id TEXT PRIMARY KEY, title TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS semantic_edges (
                source_id TEXT NOT NULL, target_id TEXT NOT NULL, similarity REAL NOT NULL,
                PRIMARY KEY (source_id, target_id)
            );
            DELETE FROM semantic_nodes;
            DELETE FROM semantic_edges;
        """)
        connection.executemany("INSERT INTO semantic_nodes VALUES (?, ?)", [(d.id, d.title) for d in documents])
        connection.executemany("INSERT INTO semantic_edges VALUES (?, ?, ?)", [(source, target, score) for (source, target), score in selected.items()])
    return len(selected)


def graph_neighbors(path: str | Path, document_id: str) -> list[tuple[str, float]]:
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT CASE WHEN source_id = ? THEN target_id ELSE source_id END, similarity "
            "FROM semantic_edges WHERE source_id = ? OR target_id = ? ORDER BY similarity DESC",
            (document_id, document_id, document_id),
        ).fetchall()
    return [(str(id_), float(score)) for id_, score in rows]
