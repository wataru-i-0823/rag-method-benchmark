from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    title: str = ""


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    relevant_ids: tuple[str, ...]


@dataclass(frozen=True)
class SearchResult:
    document: Document
    score: float
