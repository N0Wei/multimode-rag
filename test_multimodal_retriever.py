#!/usr/bin/env python
# coding: utf-8
"""Dependency-light checks for multimodal corpus normalization and BM25."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from multimodal_retriever import MultimodalBM25, MultimodalRetriever, load_chunk_documents


class MultimodalRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "chunk_id": "text-1",
                "doc_type": "text",
                "chunk_type": "topic_child",
                "page_content": "胎压不足时请及时充气。",
                "metadata": {"topic_title": "轮胎养护", "pages": [20], "parent_id": "tire"},
            },
            {
                "chunk_id": "image-1",
                "doc_type": "image",
                "chunk_type": "image_block",
                "page_content": "图片证据，展示胎压标签。",
                "metadata": {
                    "topic_title": "轮胎养护",
                    "pages": [21],
                    "parent_id": "tire",
                    "image_path": "data/assets/tire.png",
                    "nearby_text": "标准胎压见标签",
                },
            },
        ]

    def test_normalize_and_bm25(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunks.json"
            path.write_text(json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8")
            documents = load_chunk_documents(path)

        self.assertEqual(len(documents), 2)
        self.assertIn("轮胎养护", documents[0]["index_text"])
        results = MultimodalBM25(documents).GetBM25TopK("标准标签", topk=2)
        self.assertTrue(results)
        self.assertEqual(results[0].metadata["chunk_id"], "image-1")

    def test_context_is_bounded_and_keeps_image_path(self):
        results = [
            {
                "content": self.chunks[1]["page_content"],
                "metadata": {
                    **self.chunks[1]["metadata"],
                    "doc_type": "image",
                },
            }
        ]
        context = MultimodalRetriever.build_context(results, max_chars=300)
        self.assertLessEqual(len(context), 300)
        self.assertIn("data/assets/tire.png", context)

    def test_hybrid_search_uses_equal_candidate_counts(self):
        calls = {}

        class FakeVectorStore:
            def similarity_search_with_score(self, query, k):
                calls["vector_k"] = k
                return [(SimpleNamespace(metadata={"chunk_id": "text-1"}), 0.25)]

        class FakeBM25:
            def GetBM25TopK(self, query, topk):
                calls["bm25_k"] = topk
                return [SimpleNamespace(metadata={"chunk_id": "text-1"})]

        retriever = MultimodalRetriever.__new__(MultimodalRetriever)
        retriever.vector_store = FakeVectorStore()
        retriever.bm25 = FakeBM25()
        retriever.by_id = {
            "text-1": {
                "chunk_id": "text-1",
                "index_text": "胎压不足时请及时充气。",
                "metadata": {
                    "chunk_id": "text-1",
                    "doc_type": "text",
                    "raw_content": "胎压不足时请及时充气。",
                },
            }
        }

        retriever.search("胎压", top_k=1, candidate_k=12)

        self.assertEqual(calls["vector_k"], 12)
        self.assertEqual(calls["bm25_k"], 12)

    def test_vector_failure_falls_back_to_bm25(self):
        class FailingVectorStore:
            def similarity_search_with_score(self, query, k):
                raise RuntimeError("embedding API unavailable")

        class FakeBM25:
            def GetBM25TopK(self, query, topk):
                return [SimpleNamespace(metadata={"chunk_id": "text-1"})]

        retriever = MultimodalRetriever.__new__(MultimodalRetriever)
        retriever.vector_store = FailingVectorStore()
        retriever.bm25 = FakeBM25()
        retriever.by_id = {
            "text-1": {
                "chunk_id": "text-1",
                "index_text": "胎压不足时请及时充气。",
                "metadata": {
                    "chunk_id": "text-1",
                    "doc_type": "text",
                    "raw_content": "胎压不足时请及时充气。",
                },
            }
        }

        results = retriever.search("胎压", top_k=1, candidate_k=5)

        self.assertEqual(results[0]["chunk_id"], "text-1")
        self.assertEqual(retriever.last_search_info["vector_status"], "failed")

    def test_visual_retrieval_participates_in_layered_image_fusion(self):
        class FakeVectorStore:
            def similarity_search_with_score(self, query, k):
                return [(SimpleNamespace(metadata={"chunk_id": "text-1"}), 0.3)]

        class FakeVisualStore:
            def similarity_search_with_score(self, query, k):
                return [(SimpleNamespace(metadata={"chunk_id": "image-1"}), 0.12)]

        class FakeBM25:
            def GetBM25TopK(self, query, topk):
                return [SimpleNamespace(metadata={"chunk_id": "text-1"})]

        retriever = MultimodalRetriever.__new__(MultimodalRetriever)
        retriever.vector_store = FakeVectorStore()
        retriever.visual_store = FakeVisualStore()
        retriever.bm25 = FakeBM25()
        retriever.by_id = {
            "text-1": {
                "chunk_id": "text-1",
                "index_text": "胎压不足时请及时充气。",
                "metadata": {
                    "chunk_id": "text-1",
                    "doc_type": "text",
                    "raw_content": "胎压不足时请及时充气。",
                },
            },
            "image-1": {
                "chunk_id": "image-1",
                "index_text": "图片证据，展示胎压标签。",
                "metadata": {
                    "chunk_id": "image-1",
                    "doc_type": "image",
                    "raw_content": "图片证据，展示胎压标签。",
                    "image_path": "data/assets/tire.png",
                },
            },
        }

        results = retriever.search("胎压标签图片", top_k=2, candidate_k=5, max_per_topic=0)

        by_id = {item["chunk_id"]: item for item in results}
        self.assertEqual(by_id["image-1"]["visual_rank"], 1)
        self.assertEqual(by_id["image-1"]["visual_distance"], 0.12)
        self.assertEqual(by_id["image-1"]["fusion_group"], "image")
        self.assertIsNone(by_id["image-1"]["image_text_rank"])
        self.assertEqual(retriever.last_search_info["visual_status"], "ok")

    def test_image_text_channels_become_one_vote_before_visual_fusion(self):
        class FakeVectorStore:
            def similarity_search_with_score(self, query, k):
                return [
                    (SimpleNamespace(metadata={"chunk_id": "image-1"}), 0.1),
                    (SimpleNamespace(metadata={"chunk_id": "text-1"}), 0.2),
                ]

        class FakeVisualStore:
            def similarity_search_with_score(self, query, k):
                return [(SimpleNamespace(metadata={"chunk_id": "image-1"}), 0.05)]

        class FakeBM25:
            def GetBM25TopK(self, query, topk):
                return [
                    SimpleNamespace(metadata={"chunk_id": "image-1"}),
                    SimpleNamespace(metadata={"chunk_id": "text-1"}),
                ]

        retriever = MultimodalRetriever.__new__(MultimodalRetriever)
        retriever.vector_store = FakeVectorStore()
        retriever.visual_store = FakeVisualStore()
        retriever.bm25 = FakeBM25()
        retriever.by_id = {
            "text-1": {
                "chunk_id": "text-1",
                "index_text": "胎压说明",
                "metadata": {
                    "chunk_id": "text-1",
                    "doc_type": "text",
                    "raw_content": "胎压说明",
                },
            },
            "image-1": {
                "chunk_id": "image-1",
                "index_text": "胎压标签图片",
                "metadata": {
                    "chunk_id": "image-1",
                    "doc_type": "image",
                    "raw_content": "胎压标签图片",
                    "image_path": "data/assets/tire.png",
                },
            },
        }

        results = retriever.search("胎压标签", top_k=2, candidate_k=5, max_per_topic=0)
        by_id = {item["chunk_id"]: item for item in results}

        self.assertEqual(by_id["image-1"]["image_text_rank"], 1)
        self.assertEqual(by_id["image-1"]["visual_rank"], 1)
        self.assertEqual(
            by_id["image-1"]["fused_score"],
            round(2.0 / 61.0, 6),
        )
        self.assertEqual(by_id["text-1"]["fusion_group"], "text")

    def test_final_candidate_pool_interleaves_three_text_then_one_image(self):
        candidates = []
        for index in range(1, 6):
            candidates.append(
                SimpleNamespace(
                    chunk_id=f"text-{index}",
                    fusion_group="text",
                    group_rank=index,
                )
            )
        for index in range(1, 4):
            candidates.append(
                SimpleNamespace(
                    chunk_id=f"image-{index}",
                    fusion_group="image",
                    group_rank=index,
                )
            )

        merged = MultimodalRetriever._merge_evidence_groups(candidates)

        self.assertEqual(
            [item.chunk_id for item in merged[:4]],
            ["text-1", "text-2", "text-3", "image-1"],
        )

    def test_table_row_hit_expands_to_parent_table(self):
        class FakeVectorStore:
            def similarity_search_with_score(self, query, k):
                return [(SimpleNamespace(metadata={"chunk_id": "table-row-1"}), 0.1)]

        class FakeBM25:
            def GetBM25TopK(self, query, topk):
                return [SimpleNamespace(metadata={"chunk_id": "table-row-1"})]

        retriever = MultimodalRetriever.__new__(MultimodalRetriever)
        retriever.vector_store = FakeVectorStore()
        retriever.visual_store = None
        retriever.bm25 = FakeBM25()
        retriever.by_id = {
            "table-parent": {
                "chunk_id": "table-parent",
                "index_text": "完整表格",
                "metadata": {
                    "chunk_id": "table-parent",
                    "doc_type": "table",
                    "chunk_type": "table_parent",
                    "raw_content": "| 项目 | 数值 |\n| --- | --- |\n| 胎压 | 230 kPa |",
                    "indexable": False,
                },
            },
            "table-row-1": {
                "chunk_id": "table-row-1",
                "index_text": "胎压：230 kPa",
                "metadata": {
                    "chunk_id": "table-row-1",
                    "doc_type": "table",
                    "chunk_type": "table_row_child",
                    "raw_content": "项目：胎压\n数值：230 kPa",
                    "parent_id": "table-parent",
                },
            },
        }

        result = retriever.search("胎压", top_k=1, candidate_k=3)[0]

        self.assertIn("命中表格行", result["content"])
        self.assertIn("完整表格", result["content"])
        self.assertEqual(result["metadata"]["parent_chunk_id"], "table-parent")


if __name__ == "__main__":
    unittest.main()
