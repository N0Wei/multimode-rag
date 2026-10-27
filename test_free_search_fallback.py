#!/usr/bin/env python
# coding: utf-8

import unittest
from unittest.mock import Mock, patch

from agent.tools.ddgs_search_tool import DDGSSearchTool
from agent.tools.fallback_search_tool import FallbackSearchTool


class FakeSearch:
    def __init__(self, result, available=True):
        self.result = result
        self.available = available
        self.calls = []

    def search_query(self, query):
        self.calls.append(query)
        return dict(self.result)


class FreeSearchFallbackTests(unittest.TestCase):
    @patch.object(DDGSSearchTool, "__init__", lambda self, max_results=3, timeout=12: setattr(self, "max_results", max_results))
    def test_normalizes_ddgs_fields(self):
        tool = DDGSSearchTool(max_results=2)
        results = tool._normalize_results([
            {"title": "标题", "href": "https://example.com", "body": "摘要"},
        ])
        self.assertEqual("https://example.com", results[0]["url"])
        self.assertEqual("摘要", results[0]["content"])

    def test_quota_failure_switches_to_free_search(self):
        primary = FakeSearch({
            "success": False, "answer": "403 quota", "error": "403 quota", "results": []
        })
        fallback = FakeSearch({
            "success": True, "answer": "免费结果", "results": [{"title": "t"}], "source": "ddgs_free_search"
        })
        result = FallbackSearchTool(primary, fallback).search_query("银河 E8")
        self.assertTrue(result["success"])
        self.assertEqual("免费结果", result["answer"])
        self.assertEqual("403 quota", result["primary_error"])
        self.assertEqual(["银河 E8"], fallback.calls)
        self.assertFalse(primary.available)

    def test_primary_success_does_not_call_fallback(self):
        primary = FakeSearch({"success": True, "answer": "博查结果", "results": [], "source": "bocha"})
        fallback = FakeSearch({"success": True, "answer": "免费结果", "results": []})
        result = FallbackSearchTool(primary, fallback).search_query("query")
        self.assertEqual("博查结果", result["answer"])
        self.assertEqual([], fallback.calls)


if __name__ == "__main__":
    unittest.main()
