#!/usr/bin/env python
# coding: utf-8
"""Offline tests for the Bocha MCP search adapter."""

import json
import unittest
from unittest.mock import patch

from agent.tools.bocha_search_mcp_tool import BochaSearchMCPTool


class TextContent:
    def __init__(self, text):
        self.text = text


class BochaSearchMCPToolTest(unittest.TestCase):
    def setUp(self):
        with patch.object(BochaSearchMCPTool, "_check_configuration", return_value=True):
            self.tool = BochaSearchMCPTool(max_results=2)

    def test_normalizes_bocha_web_pages(self):
        payload = {
            "data": {
                "webPages": {
                    "value": [
                        {"name": "结果一", "url": "https://a.example", "summary": "摘要一"},
                        {"name": "结果二", "url": "https://b.example", "snippet": "摘要二"},
                        {"name": "结果三", "url": "https://c.example", "snippet": "摘要三"},
                    ]
                }
            }
        }
        results = self.tool._normalize_results([TextContent(json.dumps(payload, ensure_ascii=False))])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["title"], "结果一")
        self.assertEqual(results[1]["content"], "摘要二")

    def test_search_preserves_original_result_contract(self):
        content = [TextContent(json.dumps({"results": [{"title": "标题", "content": "内容", "url": ""}]}))]
        def return_content(coroutine):
            coroutine.close()
            return content

        with patch.object(self.tool, "_run_async", side_effect=return_content), patch.object(
            self.tool, "_synthesize_answer", return_value="联网答案"
        ):
            result = self.tool.search_query("测试问题")

        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "联网答案")
        self.assertEqual(result["source"], "bocha_search_mcp")
        self.assertEqual(result["query"], "测试问题")

    def test_parses_official_mcp_text_format(self):
        content = [
            TextContent(
                "Title: 示例标题\n"
                "URL: https://example.com\n"
                "Description: 示例摘要\n"
                "Published date: 2026-01-01\n"
                "Site name: 示例站点"
            )
        ]
        results = self.tool._normalize_results(content)
        self.assertEqual(results[0]["title"], "示例标题")
        self.assertEqual(results[0]["url"], "https://example.com")
        self.assertEqual(results[0]["content"], "示例摘要")

    def test_preserves_multiline_description(self):
        content = [
            TextContent(
                "Title: 示例标题\n"
                "URL: https://example.com\n"
                "Description: 第一行摘要\n"
                "第二行摘要\n"
                "第三行摘要\n"
                "Published date: 2026-01-01\n"
                "Site name: 示例站点"
            )
        ]
        results = self.tool._normalize_results(content)
        self.assertEqual(results[0]["content"], "第一行摘要\n第二行摘要\n第三行摘要")

    def test_direct_web_search_does_not_require_rag_result(self):
        self.assertTrue(self.tool.should_search({}))
        self.assertTrue(self.tool.should_search({"is_no_answer": True}))
        self.assertFalse(self.tool.should_search({"is_no_answer": False}))

    def test_preserves_official_mcp_error(self):
        content = [TextContent("Error: Bocha API key is not configured.")]

        def return_content(coroutine):
            coroutine.close()
            return content

        with patch.object(self.tool, "_run_async", side_effect=return_content):
            result = self.tool.search_query("测试问题")
        self.assertFalse(result["success"])
        self.assertIn("API key", result["answer"])


if __name__ == "__main__":
    unittest.main()
