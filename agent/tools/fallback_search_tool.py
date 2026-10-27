#!/usr/bin/env python
# coding: utf-8
"""Primary/fallback web-search chain."""


class FallbackSearchTool:
    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
        self.available = bool(
            getattr(primary, "available", False) or getattr(fallback, "available", False)
        )

    def search_query(self, query: str) -> dict:
        primary_result = None
        if getattr(self.primary, "available", False):
            primary_result = self.primary.search_query(query)
            if primary_result.get("success"):
                primary_result["search_provider"] = "primary"
                return primary_result
            print(f"  [Web Search] 主搜索失败: {primary_result.get('answer', '未知错误')}")
            error_text = str(primary_result.get("error") or primary_result.get("answer") or "").lower()
            if "403" in error_text or "enough money" in error_text or "package quota" in error_text:
                self.primary.available = False
                print("  [Web Search] 检测到主搜索额度不足，本进程后续将直接使用免费搜索")

        if getattr(self.fallback, "available", False):
            print("  [Web Search] 自动切换 DDGS 免费搜索...")
            fallback_result = self.fallback.search_query(query)
            fallback_result["search_provider"] = "fallback"
            if primary_result:
                fallback_result["primary_error"] = primary_result.get("error") or primary_result.get("answer")
            return fallback_result

        return primary_result or {
            "success": False,
            "answer": "没有可用的联网搜索服务",
            "results": [],
            "source": "fallback_search",
            "query": query,
        }
