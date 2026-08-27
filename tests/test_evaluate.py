import unittest

from rag_lab.evaluate import evaluate
from rag_lab.retrievers import build_retriever
from rag_lab.types import Document, Question


DOCUMENTS = [
    Document("leave", "休暇は上長の承認が必要です。", "休暇規程"),
    Document("expense", "経費精算には領収書が必要です。", "経費規程"),
]


class RetrievalTests(unittest.TestCase):
    def test_all_methods_return_documents(self):
        for name in "bm25 dense hybrid advanced agentic graph corpus2skill".split():
            with self.subTest(method=name):
                self.assertEqual(build_retriever(name, DOCUMENTS).search("休暇の承認", 1)[0].document.id, "leave")

    def test_evaluation_metrics_are_perfect_for_obvious_question(self):
        _, summary = evaluate(DOCUMENTS, [Question("q", "休暇の承認", ("leave",))], ["hybrid"], 1)
        self.assertEqual(summary["hybrid"]["recall_at_k"], 1.0)
        self.assertEqual(summary["hybrid"]["mrr"], 1.0)
