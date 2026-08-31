import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rag_lab.semantic_graph import build_semantic_graph, graph_neighbors
from rag_lab.types import Document


class SemanticGraphTests(unittest.TestCase):
    def test_persists_similarity_edges(self):
        documents = [Document("a", "日本銀行は金融政策を運営します"), Document("b", "日本銀行の金融政策と物価安定"), Document("c", "休暇の申請手続き")]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "graph.sqlite"
            self.assertGreater(build_semantic_graph(documents, path, threshold=0.1), 0)
            self.assertIn("b", [id_ for id_, _ in graph_neighbors(path, "a")])
