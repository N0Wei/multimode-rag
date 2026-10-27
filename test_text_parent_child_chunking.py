#!/usr/bin/env python
# coding: utf-8
"""Tests for topic-based text parent/child chunking."""

import unittest

from langchain_core.documents import Document

from multimodal_pdf_parse import LayoutElement, ManualPDFSplitter, TopicSection
from multimodal_retriever import MultimodalRetriever


class TextParentChildChunkingTests(unittest.TestCase):
    @staticmethod
    def element(index: int, text: str, page: int = 1) -> LayoutElement:
        return LayoutElement(
            element_id=f"text-{index}",
            element_type="text",
            text=text,
            page_id=page,
            module="测试模块",
            topic_id="topic-1",
            topic_title="测试主题",
            bbox=[10, index * 20, 280, index * 20 + 15],
            metadata={"page_width": 600, "column": 0},
        )

    def section(self, texts) -> TopicSection:
        elements = [self.element(index, text) for index, text in enumerate(texts)]
        return TopicSection(
            topic_id="topic-1",
            title="测试主题",
            module="测试模块",
            pages=[1],
            elements=elements,
        )

    def test_short_topic_becomes_one_parent_with_searchable_children(self):
        splitter = ManualPDFSplitter(
            "manual.pdf",
            parent_chunk_size=1500,
            child_chunk_size=80,
            child_overlap=20,
        )
        chunks = splitter.build_text_chunks(
            self.section(["功能介绍", "车辆支持多种驾驶模式。" * 12])
        )

        parents = [chunk for chunk in chunks if chunk.chunk_type == "text_parent"]
        children = [chunk for chunk in chunks if chunk.chunk_type == "text_child"]

        self.assertEqual(len(parents), 1)
        self.assertFalse(parents[0].metadata["indexable"])
        self.assertGreater(len(children), 1)
        self.assertTrue(all(child.metadata["parent_id"] == parents[0].chunk_id for child in children))

    def test_long_topic_splits_only_at_semantic_unit_boundaries(self):
        splitter = ManualPDFSplitter(
            "manual.pdf",
            parent_chunk_size=120,
            child_chunk_size=60,
            child_overlap=10,
        )
        chunks = splitter.build_text_chunks(
            self.section(
                [
                    "功能介绍",
                    "功能说明。" * 12,
                    "警告！",
                    "■请始终安全驾驶。" * 8,
                    "操作方法",
                    "1 打开车辆设置。" * 8,
                ]
            )
        )
        parents = [chunk for chunk in chunks if chunk.chunk_type == "text_parent"]

        self.assertGreater(len(parents), 1)
        self.assertTrue(any("警告" in parent.page_content for parent in parents))
        self.assertTrue(all(parent.metadata["indexable"] is False for parent in parents))

    def test_reranked_text_child_expands_to_parent_once(self):
        retriever = MultimodalRetriever.__new__(MultimodalRetriever)
        retriever.by_id = {
            "parent-1": {
                "chunk_id": "parent-1",
                "index_text": "完整父块",
                "metadata": {
                    "chunk_id": "parent-1",
                    "chunk_type": "text_parent",
                    "doc_type": "text",
                    "raw_content": "完整父块正文",
                },
            }
        }
        children = [
            Document(
                page_content="命中子块一",
                metadata={
                    "chunk_id": "child-1",
                    "chunk_type": "text_child",
                    "doc_type": "text",
                    "parent_id": "parent-1",
                    "vector_rank": 1,
                },
            ),
            Document(
                page_content="命中子块二",
                metadata={
                    "chunk_id": "child-2",
                    "chunk_type": "text_child",
                    "doc_type": "text",
                    "parent_id": "parent-1",
                    "vector_rank": 2,
                },
            ),
        ]

        expanded = retriever.expand_text_parent_documents(children)

        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0].page_content, "完整父块正文")
        self.assertEqual(expanded[0].metadata["matched_child_id"], "child-1")


if __name__ == "__main__":
    unittest.main()
