#!/usr/bin/env python
# coding: utf-8
"""Bocha Search MCP web-search tool."""

import asyncio
import json
import threading
from typing import Any, Dict, List

from agent.config.model_config import ModelConfig
from llm_streaming import generate_text_streaming


class BochaSearchMCPTool:
    """Use Bocha's official MCP server as the Agent web-search fallback."""

    def __init__(self, max_results: int = 3):
        self.max_results = max_results
        self.available = self._check_configuration()

    def _check_configuration(self) -> bool:
        if not ModelConfig.ENABLE_BOCHA_SEARCH:
            print("[Info] 博查搜索已关闭，将使用免费搜索 fallback")
            return False
        if not ModelConfig.BOCHA_API_KEY:
            print("[Warning] BOCHA_API_KEY 未配置，联网搜索将不可用")
            return False
        try:
            import mcp  # noqa: F401
        except ImportError:
            print("[Warning] MCP Python SDK 未安装，请执行: pip install mcp")
            return False
        print("[OK] 博查搜索 MCP 工具已初始化")
        return True

    def should_search(self, rag_result: Dict) -> bool:
        return self.available and (
            not rag_result or rag_result.get("is_no_answer", False)
        )

    def search_query(self, query: str) -> Dict:
        """Execute Bocha MCP search and preserve the original tool contract."""
        if not self.available:
            return self._failure(query, "博查搜索MCP未配置或依赖未安装")
        try:
            raw_result = self._run_async(self._call_mcp(query))
            error = self._extract_mcp_error(raw_result)
            if error:
                return self._failure(query, error, error)
            results = self._normalize_results(raw_result)
            if not results:
                return self._failure(query, "联网搜索未找到相关信息")
            return {
                "success": True,
                "answer": self._synthesize_answer(results, query),
                "results": results,
                "source": "bocha_search_mcp",
                "query": query,
            }
        except Exception as exc:
            print(f"[Error] 博查搜索 MCP 调用失败: {exc}")
            return self._failure(query, f"联网搜索过程出错: {exc}", str(exc))

    async def _call_mcp(self, query: str) -> Any:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server = StdioServerParameters(
            command=ModelConfig.BOCHA_MCP_COMMAND,
            args=ModelConfig.BOCHA_MCP_ARGS,
            env={"BOCHA_API_KEY": ModelConfig.BOCHA_API_KEY},
        )
        async with stdio_client(server) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    ModelConfig.BOCHA_MCP_TOOL,
                    arguments={"query": query, "freshness": "noLimit", "count": self.max_results},
                )
                if getattr(result, "isError", False):
                    raise RuntimeError(self._content_to_text(result.content))
                return result.content

    @staticmethod
    def _run_async(coroutine):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                asyncio.wait_for(coroutine, timeout=ModelConfig.BOCHA_MCP_TIMEOUT)
            )

        output: Dict[str, Any] = {}

        def runner():
            try:
                output["result"] = asyncio.run(
                    asyncio.wait_for(coroutine, timeout=ModelConfig.BOCHA_MCP_TIMEOUT)
                )
            except Exception as exc:
                output["error"] = exc

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join(timeout=ModelConfig.BOCHA_MCP_TIMEOUT)
        if thread.is_alive():
            raise TimeoutError(f"博查搜索 MCP 超时（{ModelConfig.BOCHA_MCP_TIMEOUT}s）")
        if "error" in output:
            raise output["error"]
        return output.get("result")

    def _normalize_results(self, content: Any) -> List[Dict]:
        payload = self._decode_content(content)
        candidates = self._find_result_list(payload)
        results = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            title = item.get("title") or item.get("name") or item.get("siteName") or ""
            url = item.get("url") or item.get("link") or ""
            text = item.get("summary") or item.get("snippet") or item.get("content") or ""
            if title or url or text:
                results.append({"title": str(title), "url": str(url), "content": str(text)})
            if len(results) >= self.max_results:
                break
        return results

    def _decode_content(self, content: Any) -> Any:
        texts = []
        for item in content if isinstance(content, list) else [content]:
            text = getattr(item, "text", None)
            if text is None and isinstance(item, dict):
                text = item.get("text")
            if text:
                texts.append(text)
        joined = "\n".join(texts)
        try:
            return json.loads(joined)
        except (TypeError, json.JSONDecodeError):
            return {"results": self._parse_formatted_text(joined)}

    @staticmethod
    def _extract_mcp_error(content: Any) -> str:
        texts = []
        for item in content if isinstance(content, list) else [content]:
            text = getattr(item, "text", None)
            if text:
                texts.append(str(text))
        joined = "\n".join(texts).strip()
        error_prefixes = ("Error:", "Search error", "Bocha Web Search API HTTP error",
                          "Bocha AI Search API HTTP error", "Unexpected error")
        return joined if joined.startswith(error_prefixes) else ""

    @staticmethod
    def _parse_formatted_text(text: str) -> List[Dict]:
        """Parse the official Bocha MCP server's formatted text response."""
        if not text or text.startswith(("Error:", "Search error", "No results", "Bocha ")):
            return []

        results = []
        for block in text.split("\n\n"):
            fields = {}
            current_field = ""
            for line in block.splitlines():
                key, separator, value = line.partition(": ")
                if separator and key in {"Title", "URL", "Description", "Published date", "Site name"}:
                    fields[key] = value
                    current_field = key
                elif current_field:
                    fields[current_field] = f"{fields[current_field]}\n{line}".strip()
            if fields.get("Title") or fields.get("URL") or fields.get("Description"):
                results.append(
                    {
                        "title": fields.get("Title", ""),
                        "url": fields.get("URL", ""),
                        "content": fields.get("Description", ""),
                    }
                )
        return results

    def _find_result_list(self, payload: Any) -> List[Dict]:
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            return []
        for key in ("results", "value", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        for key in ("data", "webPages", "news", "images"):
            found = self._find_result_list(payload.get(key))
            if found:
                return found
        return []

    def _synthesize_answer(self, results: List[Dict], query: str) -> str:
        source_text = "\n\n".join(
            f"来源 {index}: {item['title']}\n{item['content'][:2000]}"
            for index, item in enumerate(results, start=1)
        )
        try:
            llm = ModelConfig.get_synthesize_llm()
            prompt = f"""你是汽车客服系统的信息提炼助手。请仅根据联网搜索材料回答问题。

用户问题：{query}

联网搜索材料：
{source_text}

要求：
1. 使用简洁、自然的中文，直接回答问题。
2. 不编造材料中没有的信息；信息冲突时明确说明。
3. 不在正文中输出 URL，参考来源会在正文后单独列出。
"""
            answer = generate_text_streaming(llm, prompt, replace=True)
        except Exception as exc:
            print(f"  [Warning] 联网搜索结果提炼失败: {exc}")
            answer = results[0]["content"][:500]

        sources = ["\n\n参考来源："]
        for index, item in enumerate(results, start=1):
            sources.append(f"\n{index}. {item['title'] or item['url']}")
            if item["url"]:
                sources.append(f"\n   {item['url']}")
        return "【本地知识库未找到合适答案，以下为联网搜索结果。】\n\n" + answer + "".join(sources)

    @staticmethod
    def _content_to_text(content: Any) -> str:
        return " ".join(
            str(getattr(item, "text", item))
            for item in (content if isinstance(content, list) else [content])
        )

    @staticmethod
    def _failure(query: str, answer: str, error: str = "") -> Dict:
        result = {
            "success": False,
            "answer": answer,
            "results": [],
            "source": "bocha_search_mcp",
            "query": query,
        }
        if error:
            result["error"] = error
        return result
