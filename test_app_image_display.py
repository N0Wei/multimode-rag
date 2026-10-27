#!/usr/bin/env python
# coding: utf-8
"""Tests for rendering retrieved image evidence in the Gradio UI."""

import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from app_gradio import (
    _browser_image_path,
    _extract_retrieved_images,
    _format_rag_citations,
    _render_citation_panel,
)


class AppImageDisplayTests(unittest.TestCase):
    def test_renders_citations_as_visible_wrapped_html(self):
        rendered = _render_citation_panel("来源 <第314页>\n胎压标签位于驾驶员侧B柱")

        self.assertIn('class="citation-output"', rendered)
        self.assertIn("&lt;第314页&gt;", rendered)
        self.assertIn("胎压标签位于驾驶员侧B柱", rendered)

    def test_extracts_existing_supported_images_with_captions(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "evidence.png"
            image_path.write_bytes(b"image")
            result = {
                "rag_result": {
                    "documents": [
                        Document(
                            page_content="图片证据",
                            metadata={
                                "image_path": str(image_path),
                                "topic_title": "组合仪表",
                                "pages": [88],
                            },
                        )
                    ]
                }
            }

            images = _extract_retrieved_images(result)

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0][0], str(image_path.resolve()))
        self.assertIn("组合仪表", images[0][1])

    def test_skips_missing_and_unsupported_images(self):
        result = {
            "rag_result": {
                "documents": [
                    Document(page_content="", metadata={"image_path": "missing.png"}),
                    Document(page_content="", metadata={"image_path": "missing.jpx"}),
                ]
            }
        }
        self.assertEqual(_extract_retrieved_images(result), [])

    def test_converts_browser_unsupported_image_format(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.bmp"
            Image.new("RGB", (20, 20), "red").save(path)
            result = {
                "rag_result": {
                    "documents": [
                        Document(
                            page_content="unsupported image evidence",
                            metadata={
                                "image_path": str(path),
                                "topic_title": "Image evidence",
                                "pages": [1],
                            },
                        )
                    ]
                }
            }
            images = _extract_retrieved_images(result)

        self.assertEqual(len(images), 1)
        converted = Path(images[0][0])
        self.assertEqual(converted.suffix, ".png")
        self.assertTrue(converted.exists())

    def test_formats_final_rag_documents_as_citations(self):
        result = {
            "rag_result": {
                "documents": [
                    Document(
                        page_content="组合仪表可以显示车辆状态和驾驶辅助信息。",
                        metadata={
                            "topic_title": "查看组合仪表信息",
                            "pages": [79],
                            "doc_type": "text",
                            "vector_rank": 2,
                            "bm25_rank": 1,
                            "fused_score": 0.03,
                        },
                    )
                ]
            }
        }

        citations = _format_rag_citations(result)

        self.assertIn("查看组合仪表信息", citations)
        self.assertIn("页码 `[79]`", citations)
        self.assertIn("BM25 #1", citations)
        self.assertIn("组合仪表可以显示", citations)

    def test_formats_faq_result_when_rag_documents_are_absent(self):
        result = {
            "rag_result": {},
            "faq_result": {
                "answer": "建议冷车状态下胎压保持在 2.3-2.5 bar 之间。",
                "confidence": 0.92,
                "source": "FAQ#FAQ004",
                "matched_question": "车辆胎压多少是正常的？",
            },
        }

        citations = _format_rag_citations(result)

        self.assertIn("车辆胎压多少是正常的", citations)
        self.assertIn("FAQ#FAQ004", citations)
        self.assertIn("FAQ 匹配置信度 0.92", citations)

    def test_formats_web_search_results_when_local_sources_are_absent(self):
        result = {
            "rag_result": {},
            "faq_result": None,
            "web_search_result": {
                "success": True,
                "results": [
                    {
                        "title": "官方说明",
                        "url": "https://example.com/manual",
                        "content": "联网搜索得到的来源摘要。",
                    }
                ],
            },
        }

        citations = _format_rag_citations(result)

        self.assertIn("官方说明", citations)
        self.assertIn("https://example.com/manual", citations)
        self.assertIn("联网搜索得到的来源摘要", citations)

    def test_formats_rag_context_when_document_objects_are_absent(self):
        result = {
            "answer_source": "rag",
            "rag_result": {
                "documents": [],
                "context": "[证据1 | text | 车辆识别代码 | 页码42]\nVIN 位于车辆铭牌处。",
            },
        }

        citations = _format_rag_citations(result)

        self.assertIn("RAG 检索上下文", citations)
        self.assertIn("车辆识别代码", citations)
        self.assertIn("页码42", citations)


    def test_uses_rag_evidence_preserved_across_agent_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "evidence.png"
            image_path.write_bytes(b"image")
            document = Document(
                page_content="保留的检索证据",
                metadata={
                    "image_path": str(image_path),
                    "topic_title": "紧急救援",
                    "pages": [330],
                    "doc_type": "image",
                },
            )
            result = {
                "rag_result": {},
                "metadata": {"evidence_rag_result": {"documents": [document]}},
            }

            images = _extract_retrieved_images(result)
            citations = _format_rag_citations(result)

        self.assertEqual(len(images), 1)
        self.assertIn("紧急救援", citations)


    def test_formats_serialized_document_dict_as_citation(self):
        result = {
            "answer_source": "rag",
            "rag_result": {
                "documents": [
                    {
                        "page_content": "胎压标签位于驾驶员侧车门区域。",
                        "metadata": {
                            "topic_title": "轮胎养护",
                            "pages": [351],
                            "doc_type": "text",
                        },
                    }
                ]
            },
        }

        citations = _format_rag_citations(result)

        self.assertIn("轮胎养护", citations)
        self.assertIn("胎压标签位于驾驶员侧车门区域", citations)


if __name__ == "__main__":
    unittest.main()
