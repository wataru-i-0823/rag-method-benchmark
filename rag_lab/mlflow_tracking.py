from __future__ import annotations

from pathlib import Path


def log_experiment(
    summary: dict[str, dict],
    rows: list[dict],
    corpus_path: str,
    qa_path: str,
    tracking_uri: str,
    experiment_name: str,
) -> None:
    """Log one MLflow run per retrieval method without logging corpus contents."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    for method, metrics in summary.items():
        method_rows = [row for row in rows if row["method"] == method]
        with mlflow.start_run(run_name=method):
            mlflow.set_tags({"rag.method": method, "evaluation.type": "retrieval"})
            mlflow.log_params({"corpus_file": Path(corpus_path).name, "qa_file": Path(qa_path).name, "k": metrics["k"]})
            mlflow.log_metrics({key: float(value) for key, value in metrics.items() if key in {"recall_at_k", "mrr", "ndcg_at_k", "latency_ms"}})
            # Retrieved document IDs are useful for diagnosis but source contents are never uploaded.
            Path("/tmp/rag_lab_retrievals.json").write_text(__import__("json").dumps(method_rows, ensure_ascii=False, indent=2), encoding="utf-8")
            mlflow.log_artifact("/tmp/rag_lab_retrievals.json", artifact_path="retrievals")
