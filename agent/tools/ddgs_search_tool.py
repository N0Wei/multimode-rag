#!/usr/bin/env python
# coding: utf-8
"""Free, keyless metasearch fallback powered by the DDGS package."""

import os
from typing import Dict, List

from agent.config.model_config import ModelConfig
from llm_streaming import generate_text_streaming


class DDGSSearchTool:
    """Search the public web without an API key; intended for low-volume fallback use."""

    def __init__(self, max_results: int = 3, timeout: int = 12):
        self.max_results = max_results
        self.timeout = timeout
        self.backend = os.getenv("DDGS_BACKEND", "auto").strip() or "auto"
        self.region = os.getenv("DDGS_REGION", "cn-zh").strip() or "cn-zh"
        try:
            from ddgs import DDGS  # noqa: F401
            self.available = True
            print("[OK] DDGS 免费搜索工具已初始化（无需 API Key）")
        except ImportError:
            self.available = False
            print("[Warning] ddgs 未安装，免费搜索不可用；请执行: pip install ddgs")

    def search_query(self, query: str) -> Dict:
        if not self.available:
            return self._failure(query, "DDGS 搜索依赖未安装")
        try:
            from ddgs import DDGS

            raw_results = DDGS(timeout=self.timeout).text(
                query,
                region=self.region,
                safesearch="moderate",
                max_results=self.max_results,
                backend=self.backend,
            )
            results = self._normalize_results(raw_results)
            if not results:
                return self._failure(query, "DDGS 免费搜索未找到相关信息")
            return {
                "success": True,
                "answer": self._synthesize_answer(results, query),
                "results": results,
                "source": "ddgs_free_search",
                "query": query,
            }
        except Exception as exc:
            print(f"  [DDGS] 免费搜索失败: {exc}")
            return self._failure(query, f"DDGS 免费搜索失败: {exc}", str(exc))

    def _normalize_results(self, raw_results) -> List[Dict]:
        results = []
        for item in raw_results or []:
            if not isinstance(item, dict):
                continue
            title = item.get("title") or item.get("name") or ""
            url = item.get("href") or item.get("url") or item.get("link") or ""
            content = item.get("body") or item.get("content") or item.get("snippet") or ""
            if title or url or content:
                results.append({"title": str(title), "url": str(url), "content": str(content)})
            if len(results) >= self.max_results:
                break
        return results

    def _synthesize_answer(self, results: List[Dict], query: str) -> str:
        materials = "\n\n".join(
            f"来源 {index}: {item['title']}\n{item['content'][:2000]}"
            for index, item in enumerate(results, start=1)
        )
        try:
            answer = generate_text_streaming(ModelConfig.get_synthesize_llm(), f"""请仅根据免费联网搜索材料回答用户问题。

用户问题：{query}

搜索材料：
{materials}

要求：使用中文直接回答；不编造材料之外的事实；信息冲突时明确说明；正文不输出 URL。""")
            if not answer:
                raise ValueError("答案为空")
        except Exception as exc:
            print(f"  [DDGS] 搜索结果提炼失败: {exc}")
            answer = results[0]["content"][:500]

        sources = ["\n\n参考来源："]
        for index, item in enumerate(results, start=1):
            sources.append(f"\n{index}. {item['title'] or item['url']}")
            if item["url"]:
                sources.append(f"\n   {item['url']}")
        return "【以下为免费联网搜索结果。】\n\n" + answer + "".join(sources)

    @staticmethod
    def _failure(query: str, answer: str, error: str = "") -> Dict:
        result = {
            "success": False,
            "answer": answer,
            "results": [],
            "source": "ddgs_free_search",
            "query": query,
        }
        if error:
            result["error"] = error
        return result
