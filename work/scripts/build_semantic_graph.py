"""Build a free persistent semantic graph from a JSONL corpus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_lab.semantic_graph import build_semantic_graph
from rag_lab.types import Document

parser = argparse.ArgumentParser(description="Build a free SQLite semantic graph")
parser.add_argument("--corpus", required=True)
parser.add_argument("--config", type=Path, help="Graph backend configuration JSON")
parser.add_argument("--backend", help="Backend name in --config; defaults to active_backend")
parser.add_argument("--graph-path")
parser.add_argument("--threshold", type=float)
parser.add_argument("--neighbors", type=int)
parser.add_argument("--embedding-model", choices=["hash", "e5-small", "bge-m3"])
args = parser.parse_args()
settings: dict[str, object] = {}
backend_name = None
if args.config:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    backend_name = args.backend or config["active_backend"]
    settings = config["backends"][backend_name]
    if settings["kind"] != "semantic":
        raise RuntimeError(f"{backend_name} はAPI版の将来バックエンドです。詳細は docs/graph-construction.md を参照してください。")
graph_path = args.graph_path or str(settings.get("graph_path", "data/graph/semantic_graph.sqlite"))
threshold = args.threshold if args.threshold is not None else float(settings.get("similarity_threshold", 0.35))
neighbors = args.neighbors if args.neighbors is not None else int(settings.get("max_neighbors_per_chunk", 5))
embedding_model = args.embedding_model or str(settings.get("embedding_model", "hash"))
documents = [Document(row["id"], row["text"], row.get("title", "")) for row in (json.loads(line) for line in Path(args.corpus).read_text(encoding="utf-8").splitlines() if line.strip())]
edges = build_semantic_graph(documents, graph_path, threshold, neighbors, embedding_model)
print(json.dumps({"backend": backend_name or "inline", "documents": len(documents), "edges": edges, "embedding_model": embedding_model, "graph_path": graph_path}, ensure_ascii=False))
