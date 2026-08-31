from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import evaluate, write_results
from .profiles import PROFILES, get_profile
from .retrievers import build_retriever
from .types import Document, Question

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
    evaluate_parser.add_argument("--profile", choices=sorted(PROFILES), default="local", help="Execution environment profile (default: local)")
    evaluate_parser.add_argument("--chroma-path", help="Chroma index path; overrides the selected profile")
    evaluate_parser.add_argument("--embedding-model", choices=["e5-small", "bge-m3"], help="Embedding model for chroma_e5; defaults to the selected profile")
    evaluate_parser.add_argument("--context-scope", choices=["chunk", "parent"], default="chunk", help="Return a hit child chunk or parent context")
    evaluate_parser.add_argument("--parent-strategy", choices=["neighbors", "source"], default="neighbors", help="Construct parent context from neighboring chunks or the source document")
    evaluate_parser.add_argument("--neighbor-window", type=int, default=1, help="Number of chunks on each side for --parent-strategy neighbors")
    evaluate_parser.add_argument("--methods", help="Comma-separated methods; defaults to the selected profile")
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
    inspect_parser.add_argument("--profile", choices=sorted(PROFILES), default="local")
    inspect_parser.add_argument("--chroma-path", help="Chroma index path; overrides the selected profile")
    inspect_parser.add_argument("--embedding-model", choices=["e5-small", "bge-m3"], help="Embedding model for chroma_e5; defaults to the selected profile")
    inspect_parser.add_argument("--context-scope", choices=["chunk", "parent"], default="chunk")
    inspect_parser.add_argument("--parent-strategy", choices=["neighbors", "source"], default="neighbors")
    inspect_parser.add_argument("--neighbor-window", type=int, default=1)
    inspect_parser.add_argument("--framework", choices=["langchain", "llamaindex"], help="Chunk documents before retrieval")
    inspect_parser.add_argument("--method")
    inspect_parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    documents = load_documents(args.corpus)
    parent_documents = documents
    profile = get_profile(args.profile)
    retriever_options = {
        "chroma_path": args.chroma_path or profile.chroma_path,
        "embedding_device": profile.embedding_device,
        "embedding_model": args.embedding_model or profile.embedding_model,
        "context_scope": args.context_scope,
        "parent_strategy": args.parent_strategy,
        "neighbor_window": args.neighbor_window,
        "parent_documents": parent_documents,
    }
    if args.command == "evaluate":
        if args.framework:
            from .frameworks import framework_documents
            documents = framework_documents(documents, args.framework)
        selected_methods = args.methods or profile.default_methods
        methods = [method.strip() for method in selected_methods.split(",") if method.strip()]
        rows, summary = evaluate(documents, load_questions(args.qa), methods, args.k, langsmith=args.langsmith, retriever_options=retriever_options)
        write_results(rows, summary, Path(args.output))
        if args.mlflow:
            from .mlflow_tracking import log_experiment
            log_experiment(rows=rows, summary=summary, corpus_path=args.corpus, qa_path=args.qa, tracking_uri=args.tracking_uri, experiment_name=args.experiment)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        if args.framework:
            from .frameworks import framework_documents
            documents = framework_documents(documents, args.framework)
        method = args.method or ("chroma_e5" if args.profile in {"colab", "cloud"} else "hybrid")
        results = build_retriever(method, documents, **retriever_options).search(args.query, args.k)
        print(json.dumps([{"id": result.document.id, "title": result.document.title, "score": round(result.score, 4), "text": result.document.text} for result in results], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
