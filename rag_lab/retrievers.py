from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

from .text import cosine, normalise, tokenize
from .types import Document, SearchResult


class Retriever:
    name = "base"

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        raise NotImplementedError


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75):
        self.documents, self.k1, self.b = documents, k1, b
        self.terms = [Counter(tokenize(f"{d.title} {d.text}")) for d in documents]
        self.lengths = [sum(term.values()) for term in self.terms]
        self.average_length = sum(self.lengths) / len(documents) if documents else 1
        document_frequency: Counter[str] = Counter()
        for terms in self.terms:
            document_frequency.update(terms.keys())
        count = len(documents)
        self.idf = {term: math.log(1 + (count - freq + 0.5) / (freq + 0.5)) for term, freq in document_frequency.items()}

    def scores(self, query: str) -> list[float]:
        query_terms = tokenize(query)
        result: list[float] = []
        for terms, length in zip(self.terms, self.lengths):
            score = 0.0
            for term in query_terms:
                freq = terms.get(term, 0)
                if freq:
                    denom = freq + self.k1 * (1 - self.b + self.b * length / self.average_length)
                    score += self.idf.get(term, 0.0) * freq * (self.k1 + 1) / denom
            result.append(score)
        return result

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        scores = self.scores(query)
        return _top(self.documents, scores, k)


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, documents: list[Document]):
        self.documents = documents
        raw = [Counter(tokenize(f"{d.title} {d.text}")) for d in documents]
        df: Counter[str] = Counter()
        for vector in raw:
            df.update(vector.keys())
        total = len(documents)
        self.idf = {term: math.log((total + 1) / (freq + 1)) + 1 for term, freq in df.items()}
        self.vectors = [Counter({term: count * self.idf[term] for term, count in vector.items()}) for vector in raw]

    def scores(self, query: str) -> list[float]:
        q = Counter(tokenize(query))
        weighted = Counter({term: count * self.idf.get(term, 0.0) for term, count in q.items()})
        return [cosine(weighted, vector) for vector in self.vectors]

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        return _top(self.documents, self.scores(query), k)


def local_hypothetical_document(question: str) -> str:
    """Deterministic offline fallback; replace with an LLM generator in production."""
    return f"回答の根拠となる文書。質問: {question}"


def local_hypothetical_question(document: Document) -> str:
    """Index-time fallback for Reverse HyDE, retaining document vocabulary."""
    return f"{document.title}について何が定められていますか。{document.text}"


class HyDERetriever(DenseRetriever):
    """Generate a document-shaped query at retrieval time, then run dense search."""
    name = "hyde"

    def __init__(self, documents: list[Document], generator: Callable[[str], str] = local_hypothetical_document):
        super().__init__(documents)
        self.generator = generator

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        return super().search(self.generator(query), k)


class ReverseHyDERetriever(DenseRetriever):
    """Index hypothetical questions for every document, then retrieve from them."""
    name = "reverse_hyde"

    def __init__(self, documents: list[Document], generator: Callable[[Document], str] = local_hypothetical_question):
        super().__init__(documents)
        self.virtual_questions = [Counter(tokenize(generator(document))) for document in documents]
        self.question_vectors = [Counter({term: count * self.idf.get(term, 0.0) for term, count in vector.items()}) for vector in self.virtual_questions]

    def scores(self, query: str) -> list[float]:
        weighted = Counter({term: count * self.idf.get(term, 0.0) for term, count in Counter(tokenize(query)).items()})
        return [cosine(weighted, vector) for vector in self.question_vectors]


class HybridRetriever(Retriever):
    name = "hybrid"

    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.bm25, self.dense = BM25Retriever(documents), DenseRetriever(documents)

    def scores(self, query: str) -> list[float]:
        lexical, semantic = normalise(self.bm25.scores(query)), normalise(self.dense.scores(query))
        return [0.5 * left + 0.5 * right for left, right in zip(lexical, semantic)]

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        return _top(self.documents, self.scores(query), k)


class AdvancedRetriever(HybridRetriever):
    name = "advanced"

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        candidates = super().search(query, max(k * 3, 10))
        query_terms = set(tokenize(query))
        reranked = []
        for candidate in candidates:
            terms = set(tokenize(f"{candidate.document.title} {candidate.document.text}"))
            coverage = len(query_terms & terms) / max(1, len(query_terms))
            reranked.append(SearchResult(candidate.document, 0.7 * candidate.score + 0.3 * coverage))
        return sorted(reranked, key=lambda item: item.score, reverse=True)[:k]


class AgenticRetriever(HybridRetriever):
    name = "agentic"

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        # Lightweight query planning: interrogative clauses and conjunctions each become retrieval actions.
        subqueries = [query]
        for separator in ("と", "及び", "および", "または", "、", "?", "？"):
            pieces = [piece.strip() for piece in query.split(separator) if len(piece.strip()) >= 2]
            if len(pieces) > 1:
                subqueries.extend(pieces)
        accumulated: defaultdict[str, float] = defaultdict(float)
        for subquery in dict.fromkeys(subqueries):
            for rank, result in enumerate(super().search(subquery, max(k, 3)), start=1):
                accumulated[result.document.id] += result.score / rank
        lookup = {document.id: document for document in self.documents}
        results = [SearchResult(lookup[id_], score) for id_, score in accumulated.items()]
        return sorted(results, key=lambda item: item.score, reverse=True)[:k]


class LangGraphAgenticRetriever(AgenticRetriever):
    """Agentic retrieval orchestrated with a compiled LangGraph StateGraph."""
    name = "langgraph_agentic"

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        from .frameworks import agentic_search_with_langgraph
        return agentic_search_with_langgraph(super(), query, k)


class GraphRetriever(HybridRetriever):
    name = "graph"

    def __init__(self, documents: list[Document]):
        super().__init__(documents)
        self.entities = {document.id: {token for token in tokenize(document.title + " " + document.text) if len(token) >= 3} for document in documents}

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        seeds = super().search(query, max(1, k))
        seed_entities = set().union(*(self.entities[result.document.id] for result in seeds))
        base = {result.document.id: result.score for result in super().search(query, len(self.documents))}
        expanded = []
        for document in self.documents:
            overlap = len(seed_entities & self.entities[document.id]) / max(1, len(seed_entities))
            expanded.append(SearchResult(document, 0.8 * base.get(document.id, 0.0) + 0.2 * overlap))
        return sorted(expanded, key=lambda item: item.score, reverse=True)[:k]


@dataclass
class SkillNode:
    label: str
    documents: list[Document]
    children: list["SkillNode"]


class Corpus2SkillRetriever(DenseRetriever):
    name = "corpus2skill"

    def __init__(self, documents: list[Document], branch_size: int = 4):
        super().__init__(documents)
        self.branch_size = branch_size
        self.root = self._build(documents)

    def _build(self, documents: list[Document]) -> SkillNode:
        label = " / ".join(self._keywords(documents)[:3]) or "knowledge"
        if len(documents) <= self.branch_size:
            return SkillNode(label, documents, [])
        groups: defaultdict[str, list[Document]] = defaultdict(list)
        for document in documents:
            terms = self._keywords([document])
            groups[terms[0] if terms else document.id].append(document)
        ordered = sorted(groups.values(), key=len, reverse=True)
        # Merge small groups deterministically to bound the branching factor.
        buckets = [[] for _ in range(self.branch_size)]
        for index, group in enumerate(ordered):
            buckets[index % self.branch_size].extend(group)
        children = [self._build(bucket) for bucket in buckets if bucket and len(bucket) < len(documents)]
        return SkillNode(label, documents, children)

    def _keywords(self, documents: list[Document]) -> list[str]:
        terms = Counter()
        for document in documents:
            terms.update(token for token in tokenize(document.title + " " + document.text) if len(token) >= 2)
        return [term for term, _ in terms.most_common(8)]

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        node = self.root
        # Navigate SKILL/INDEX-like topic nodes before retrieving documents at a leaf.
        while node.children:
            query_terms = set(tokenize(query))
            node = max(node.children, key=lambda child: len(query_terms & set(tokenize(child.label))))
        allowed = {document.id for document in node.documents}
        all_results = super().search(query, len(self.documents))
        chosen = [result for result in all_results if result.document.id in allowed][:k]
        return chosen or all_results[:k]


def build_retriever(name: str, documents: list[Document]) -> Retriever:
    if name == "chroma_local":
        from .chroma_store import ChromaE5Retriever
        return ChromaE5Retriever(documents)  # type: ignore[return-value]
    options = {cls.name: cls for cls in (BM25Retriever, DenseRetriever, HyDERetriever, ReverseHyDERetriever, HybridRetriever, AdvancedRetriever, AgenticRetriever, LangGraphAgenticRetriever, GraphRetriever, Corpus2SkillRetriever)}
    try:
        return options[name](documents)
    except KeyError as error:
        raise ValueError(f"Unknown method: {name}. Choose from {', '.join(options)}") from error


def _top(documents: list[Document], scores: Iterable[float], k: int) -> list[SearchResult]:
    results = [SearchResult(document, score) for document, score in zip(documents, scores)]
    return sorted(results, key=lambda item: item.score, reverse=True)[:k]
