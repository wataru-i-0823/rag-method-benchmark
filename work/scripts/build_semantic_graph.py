"""Build a free persistent semantic graph from a JSONL corpus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_lab.semantic_graph import build_semantic_graph
from rag_lab.types import Document

parser = argparse.ArgumentParser(description="Build a free SQLite semantic graph")
parser.add_argument("--corpus", required=True)
parser.add_argument("--graph-path", default="data/graph/semantic_graph.sqlite")
parser.add_argument("--threshold", type=float, default=0.35)
parser.add_argument("--neighbors", type=int, default=5)
parser.add_argument("--embedding-model", choices=["hash", "e5-small", "bge-m3"], default="hash")
args = parser.parse_args()
documents = [Document(row["id"], row["text"], row.get("title", "")) for row in (json.loads(line) for line in Path(args.corpus).read_text(encoding="utf-8").splitlines() if line.strip())]
edges = build_semantic_graph(documents, args.graph_path, args.threshold, args.neighbors, args.embedding_model)
print(json.dumps({"documents": len(documents), "edges": edges, "embedding_model": args.embedding_model, "graph_path": args.graph_path}, ensure_ascii=False))
