#!/usr/bin/env python
# coding: utf-8
"""Call Bocha Search MCP directly and inspect every response layer.

Examples:
    python test_bocha_mcp_call.py "领克汽车最新新闻"
    python test_bocha_mcp_call.py "杭州今天天气" --tool bocha_ai_search --count 3
    python test_bocha_mcp_call.py "领克汽车最新新闻" --save data/bocha_mcp_result.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent.config.model_config import ModelConfig
from agent.tools.bocha_search_mcp_tool import BochaSearchMCPTool


def to_jsonable(value: Any) -> Any:
    """Convert MCP/Pydantic response objects into printable JSON values."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def content_text(content: List[Any]) -> str:
    return "\n".join(
        str(getattr(item, "text", item))
        for item in content
    )


async def call_bocha_mcp(query: str, tool_name: str, count: int, freshness: str) -> Dict[str, Any]:
    if not ModelConfig.BOCHA_API_KEY:
        raise RuntimeError("请先在 .env 中配置 BOCHA_API_KEY")

    server = StdioServerParameters(
        command=ModelConfig.BOCHA_MCP_COMMAND,
        args=ModelConfig.BOCHA_MCP_ARGS,
        env={"BOCHA_API_KEY": ModelConfig.BOCHA_API_KEY},
    )

    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            initialize_result = await session.initialize()
            tools_result = await session.list_tools()
            call_result = await session.call_tool(
                tool_name,
                arguments={
                    "query": query,
                    "freshness": freshness,
                    "count": count,
                },
            )

    adapter = BochaSearchMCPTool(max_results=count)
    normalized_results = adapter._normalize_results(call_result.content)
    raw_text = content_text(call_result.content)

    return {
        "request": {
            "query": query,
            "tool": tool_name,
            "freshness": freshness,
            "count": count,
        },
        "mcp_initialize_result": to_jsonable(initialize_result),
        "available_tools": to_jsonable(tools_result.tools),
        "raw_call_result": to_jsonable(call_result),
        "raw_text_content": raw_text,
        "normalized_results": normalized_results,
    }


def print_section(title: str, value: Any) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)
    if isinstance(value, str):
        print(value)
    else:
        print(json.dumps(value, ensure_ascii=False, indent=2))


def configure_stdout() -> None:
    """Allow raw web content to print safely on Windows GBK terminals."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="直接调用博查搜索 MCP 并查看返回格式")
    parser.add_argument("query", nargs="?", default="领克汽车最新新闻", help="搜索问题")
    parser.add_argument(
        "--tool",
        choices=["bocha_web_search", "bocha_ai_search"],
        default="bocha_web_search",
        help="需要调用的博查 MCP 工具",
    )
    parser.add_argument("--count", type=int, default=3, help="返回结果数量，范围 1-50")
    parser.add_argument(
        "--freshness",
        default="noLimit",
        help="noLimit、oneDay、oneWeek、oneMonth、oneYear 或日期范围",
    )
    parser.add_argument("--save", default="", help="可选：将完整结果保存为 JSON 文件")
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()
    count = max(1, min(50, args.count))
    result = asyncio.run(
        asyncio.wait_for(
            call_bocha_mcp(args.query, args.tool, count, args.freshness),
            timeout=ModelConfig.BOCHA_MCP_TIMEOUT,
        )
    )

    if args.save:
        output_path = Path(args.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n完整结果已保存：{output_path.resolve()}")

    print_section("1. 请求参数", result["request"])
    print_section("2. MCP 初始化返回", result["mcp_initialize_result"])
    print_section("3. MCP 可用工具列表", result["available_tools"])
    print_section("4. call_tool 原始返回对象", result["raw_call_result"])
    print_section("5. 博查 MCP 原始文本内容", result["raw_text_content"])
    print_section("6. 当前项目适配器解析后的统一格式", result["normalized_results"])


if __name__ == "__main__":
    main()
