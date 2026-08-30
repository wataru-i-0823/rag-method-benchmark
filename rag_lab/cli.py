from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import evaluate, write_results
from .retrievers import build_retriever
from .types import Document, Question

DEFAULT_METHODS = "bm25,dense,chroma_local,hyde,reverse_hyde,hybrid,advanced,agentic,langgraph_agentic,graph,corpus2skill"


def load_documents(path: str) -> list[Document]:
    return [Document(id=row["id"], text=row["text"], title=row.get("title", "")) for row in _jsonl(path)]


def load_questions(path: str) -> list[Question]:
    return [Question(id=row["id"], question=row["question"], relevant_ids=tuple(row["relevant_ids"])) for row in _jsonl(path)]


def _jsonl(path: str):
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare retrieval strategies for RAG")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--corpus", required=True)
    evaluate_parser.add_argument("--qa", required=True)
    evaluate_parser.add_argument("--methods", default=DEFAULT_METHODS)
    evaluate_parser.add_argument("--k", type=int, default=3)
    evaluate_parser.add_argument("--output", default="results")
    evaluate_parser.add_argument("--mlflow", action="store_true", help="Log one MLflow run per method")
    evaluate_parser.add_argument("--tracking-uri", default="mlruns")
    evaluate_parser.add_argument("--experiment", default="rag-method-benchmark")
    evaluate_parser.add_argument("--langsmith", action="store_true", help="Trace retrieval calls to LangSmith")
    evaluate_parser.add_argument("--framework", choices=["langchain", "llamaindex"], help="Chunk documents with the selected RAG framework before indexing")
    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("--corpus", required=True)
    inspect_parser.add_argument("--query", required=True)
    inspect_parser.add_argument("--method", default="hybrid")
    inspect_parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    documents = load_documents(args.corpus)
    if args.command == "evaluate":
        if args.framework:
            from .frameworks import framework_documents
            documents = framework_documents(documents, args.framework)
        methods = [method.strip() for method in args.methods.split(",") if method.strip()]
        rows, summary = evaluate(documents, load_questions(args.qa), methods, args.k, langsmith=args.langsmith)
        write_results(rows, summary, Path(args.output))
        if args.mlflow:
            from .mlflow_tracking import log_experiment
            log_experiment(rows=rows, summary=summary, corpus_path=args.corpus, qa_path=args.qa, tracking_uri=args.tracking_uri, experiment_name=args.experiment)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        results = build_retriever(args.method, documents).search(args.query, args.k)
        print(json.dumps([{"id": result.document.id, "title": result.document.title, "score": round(result.score, 4), "text": result.document.text} for result in results], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
