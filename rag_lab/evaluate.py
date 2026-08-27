from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from .retrievers import build_retriever
from .types import Document, Question


def evaluate(documents: list[Document], questions: list[Question], methods: list[str], k: int) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    summary: dict[str, dict] = {}
    for method in methods:
        retriever = build_retriever(method, documents)
        totals: defaultdict[str, float] = defaultdict(float)
        for question in questions:
            started = time.perf_counter()
            results = retriever.search(question.question, k)
            elapsed = (time.perf_counter() - started) * 1000
            retrieved = [result.document.id for result in results]
            relevant = set(question.relevant_ids)
            hit_ranks = [index for index, doc_id in enumerate(retrieved, start=1) if doc_id in relevant]
            recall = len(set(retrieved) & relevant) / len(relevant)
            reciprocal_rank = 1 / hit_ranks[0] if hit_ranks else 0.0
            dcg = sum(1 / math_log2(rank + 1) for rank in hit_ranks)
            ideal = sum(1 / math_log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
            row = {"method": method, "question_id": question.id, "recall_at_k": recall, "mrr": reciprocal_rank, "ndcg_at_k": dcg / ideal if ideal else 0.0, "latency_ms": elapsed, "retrieved_ids": retrieved}
            rows.append(row)
            for key in ("recall_at_k", "mrr", "ndcg_at_k", "latency_ms"):
                totals[key] += row[key]
        count = len(questions) or 1
        summary[method] = {key: value / count for key, value in totals.items()} | {"questions": len(questions), "k": k}
    return rows, summary


def math_log2(value: float) -> float:
    import math
    return math.log2(value)


def write_results(rows: list[dict], summary: dict, output: Path) -> None:
    import csv
    import json
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_question.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys() if rows else ["method"])
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
