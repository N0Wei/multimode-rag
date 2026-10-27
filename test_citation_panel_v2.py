#!/usr/bin/env python
# coding: utf-8

import unittest

from langchain_core.documents import Document

from app_gradio import (
    _build_citation_panel,
    _build_citation_text,
    _collect_citations,
    _render_citations_html,
)


class CitationPanelV2Tests(unittest.TestCase):
    def test_collects_rag_documents_from_preserved_snapshot(self):
        result = {
            "answer_source": "rag",
            "rag_result": {},
            "metadata": {
                "evidence_rag_result": {
                    "documents": [
                        Document(
                            page_content="左侧信息显示区域显示车辆当前车速。",
                            metadata={
                                "chunk_id": "instrument-1",
                                "topic_title": "组合仪表",
                                "pages": [69],
                                "doc_type": "text",
                                "vector_rank": 1,
                            },
                        )
                    ]
                }
            },
        }

        citations = _collect_citations(result)

        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["title"], "组合仪表")
        self.assertEqual(citations[0]["pages"], [69])

    def test_renders_direct_html_cards(self):
        rendered = _render_citations_html(
            [
                {
                    "source_type": "rag",
                    "title": "组合仪表",
                    "pages": [69],
                    "doc_type": "text",
                    "excerpt": "左侧信息显示区域。",
                    "ranks": ["向量 #1"],
                }
            ]
        )

        self.assertIn('class="citation-card citation-rag"', rendered)
        self.assertIn("组合仪表", rendered)
        self.assertIn("页码 [69]", rendered)
        self.assertNotIn("<pre>", rendered)

    def test_builds_panel_from_serialized_document(self):
        result = {
            "answer_source": "rag",
            "rag_result": {
                "documents": [
                    {
                        "page_content": "右侧信息显示区域显示多媒体和胎压。",
                        "metadata": {
                            "chunk_id": "instrument-2",
                            "topic_title": "组合仪表",
                            "pages": [69],
                            "doc_type": "text",
                        },
                    }
                ]
            },
        }

        rendered = _build_citation_panel(result)

        self.assertIn("右侧信息显示区域", rendered)
        self.assertIn("组合仪表", rendered)

    def test_final_web_answer_does_not_show_stale_rag_evidence(self):
        result = {
            "answer_source": "web_search",
            "metadata": {
                "evidence_rag_result": {
                    "documents": [
                        Document(page_content="旧 RAG 证据", metadata={"topic_title": "旧主题"})
                    ]
                }
            },
            "web_search_result": {
                "results": [
                    {
                        "title": "联网来源",
                        "url": "https://example.com",
                        "content": "联网答案摘要",
                    }
                ]
            },
        }

        rendered = _build_citation_panel(result)

        self.assertIn("联网来源", rendered)
        self.assertNotIn("旧 RAG 证据", rendered)

    def test_long_rag_reference_is_plain_text_for_native_textbox(self):
        result = {
            "answer_source": "rag",
            "rag_result": {
                "documents": [
                    Document(
                        page_content="组合仪表显示区域。" * 200,
                        metadata={
                            "chunk_id": "long-rag",
                            "topic_title": "组合仪表",
                            "pages": [69],
                            "doc_type": "text",
                        },
                    )
                ]
            },
        }

        text = _build_citation_text(result)

        self.assertIn("[1] RAG · 组合仪表", text)
        self.assertIn("页码 [69]", text)
        self.assertNotIn("<article", text)
        self.assertNotIn("<div", text)


if __name__ == "__main__":
    unittest.main()
