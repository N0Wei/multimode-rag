#!/usr/bin/env python
# coding: utf-8
"""Tests for the Agent-compatible multimodal RAG tool."""

import unittest

from agent.tools.multimodal_rag_tool import MultimodalRAGTool
from agent.tools.rag_tool import RAGTool


class FakeRetriever:
    last_search_info = {"vector_status": "ok", "bm25_status": "ok", "errors": []}

    def search(self, query, **kwargs):
        return [
            {
                "chunk_id": "image-1",
                "content": "胎压标签位于驾驶员侧车门区域。",
                "fused_score": 0.03,
                "vector_rank": 1,
                "bm25_rank": 2,
                "vector_distance": 0.2,
                "metadata": {
                    "chunk_id": "image-1",
                    "topic_id": "tire",
                    "topic_title": "轮胎养护",
                    "doc_type": "image",
                    "image_path": "data/assets/tire.png",
                    "pages": [21],
                },
            },
            {
                "chunk_id": "text-1",
                "content": "请按照胎压标签标注的标准胎压充气。",
                "fused_score": 0.02,
                "vector_rank": 2,
                "bm25_rank": 1,
                "vector_distance": 0.3,
                "metadata": {
                    "chunk_id": "text-1",
                    "topic_id": "tire",
                    "topic_title": "轮胎养护",
                    "doc_type": "text",
                    "pages": [22],
                },
            },
        ]


class FakeRerank:
    def predict(self, query, documents):
        return documents


class FakeLLM:
    def infer(self, prompts):
        return ["胎压标签位于驾驶员侧车门区域。"]


class MultimodalRAGToolTests(unittest.TestCase):
    def test_reuses_original_prompt_and_result_contract(self):
        tool = MultimodalRAGTool(FakeRetriever(), FakeRerank(), FakeLLM())

        result = tool.search_and_generate("胎压标签在哪里")

        self.assertIs(tool._build_prompt.__func__, RAGTool._build_prompt)
        self.assertEqual(result["source"], "rag")
        self.assertEqual(result["retrieval_mode"], "multimodal_hybrid")
        self.assertEqual(result["image_paths"], ["data/assets/tire.png"])
        self.assertIn("轮胎养护", result["context"])
        self.assertIn("documents", result)
        self.assertIn("confidence", result)
        self.assertIn("is_no_answer", result)

    def test_topic_limit_runs_after_rerank(self):
        class ReorderedRerank:
            def predict(self, query, documents):
                return list(reversed(documents))

        tool = MultimodalRAGTool(FakeRetriever(), ReorderedRerank(), FakeLLM())

        result = tool.search_and_generate("胎压标签在哪里", top_k=1)

        self.assertEqual(result["documents"][0].metadata["chunk_id"], "text-1")

    def test_text_does_not_hide_image_from_same_topic(self):
        tool = MultimodalRAGTool(FakeRetriever(), FakeRerank(), FakeLLM())

        result = tool.search_and_generate("胎压标签在哪里？", top_k=2)

        self.assertEqual(
            [document.metadata["doc_type"] for document in result["documents"]],
            ["image", "text"],
        )

    def test_high_ranked_image_is_restored_when_reranker_returns_only_text(self):
        class TextOnlyRerank:
            def predict(self, query, documents):
                return [doc for doc in documents if doc.metadata["doc_type"] == "text"]

        tool = MultimodalRAGTool(FakeRetriever(), TextOnlyRerank(), FakeLLM())
        result = tool.search_and_generate("胎压标签在哪里？", top_k=2)

        self.assertIn("image", [document.metadata["doc_type"] for document in result["documents"]])


    def test_uses_wider_candidate_pool_before_reranking(self):
        class TrackingRetriever(FakeRetriever):
            def search(self, query, **kwargs):
                self.kwargs = kwargs
                return super().search(query, **kwargs)

        retriever = TrackingRetriever()
        tool = MultimodalRAGTool(retriever, FakeRerank(), FakeLLM())

        tool.search_and_generate("test query")

        self.assertEqual(retriever.kwargs["top_k"], 20)
        self.assertEqual(retriever.kwargs["candidate_k"], 40)


if __name__ == "__main__":
    unittest.main()
