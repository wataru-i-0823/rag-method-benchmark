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
        for name in "bm25 dense hyde reverse_hyde hybrid advanced agentic langgraph_agentic graph corpus2skill".split():
            with self.subTest(method=name):
                self.assertEqual(build_retriever(name, DOCUMENTS).search("休暇の承認", 1)[0].document.id, "leave")

    def test_evaluation_metrics_are_perfect_for_obvious_question(self):
        _, summary = evaluate(DOCUMENTS, [Question("q", "休暇の承認", ("leave",))], ["hybrid"], 1)
        self.assertEqual(summary["hybrid"]["recall_at_k"], 1.0)
        self.assertEqual(summary["hybrid"]["mrr"], 1.0)

    def test_local_chroma_profile_uses_persistent_vector_store(self):
        results = build_retriever("chroma_hash", DOCUMENTS, chroma_path="/tmp/rag_lab_test_chroma").search("休暇の承認", 1)
        self.assertEqual(results[0].document.id, "leave")

    def test_parent_context_returns_the_source_document(self):
        parent = Document("policy", "第3章全体の本文", "規程")
        chunks = [
            Document("policy:0", "第3章 3-1章 休暇は上長の承認が必要です。", "規程"),
            Document("policy:1", "第3章 3-2章 経費の申請方法。", "規程"),
        ]
        result = build_retriever("hybrid", chunks, context_scope="parent", parent_documents=[parent]).search("休暇の承認", 1)
        self.assertEqual(result[0].document.id, "policy")
