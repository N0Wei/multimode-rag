#!/usr/bin/env python
# coding: utf-8
"""Tool execution with request-scoped caching, deduplication and budgets."""

import copy
import json
import os
import re
import time

from agent.state import AgentState


rag_tool = None
multi_manual_tool = None
faq_tool = None
web_search_tool = None

MAX_TOOL_CALLS = int(os.getenv("AGENT_MAX_TOOL_CALLS", "8"))
MAX_RAG_CALLS = int(os.getenv("AGENT_MAX_RAG_CALLS", "5"))
MAX_WEB_SEARCH_CALLS = int(os.getenv("AGENT_MAX_WEB_SEARCH_CALLS", "2"))
MAX_TOOL_SECONDS = float(os.getenv("AGENT_MAX_TOOL_SECONDS", "120"))


def set_tools(rag, faq, web_search):
    """Inject the tools used by the Agent graph."""
    global rag_tool, multi_manual_tool, faq_tool, web_search_tool
    rag_tool = rag
    multi_manual_tool = None
    faq_tool = faq
    web_search_tool = web_search


def set_multi_manual_tool(rag) -> None:
    """Register the immutable multi-manual tool alongside the legacy tool."""
    global multi_manual_tool
    multi_manual_tool = rag


def _preserve_rag_evidence(state: AgentState, rag_result: dict) -> None:
    """Keep the latest non-empty RAG evidence across Reflection retries."""
    documents = rag_result.get("documents") or []
    context = rag_result.get("context") or ""
    if not documents and not context:
        return

    metadata = state.setdefault("metadata", {})
    previous = metadata.get("evidence_rag_result") or {}
    previous_documents = previous.get("documents") or []
    previous_context = previous.get("context") or ""

    useful_answer = not rag_result.get("is_no_answer", False)
    if (documents and (useful_answer or not previous_documents)) or (
        not previous_documents and context and not previous_context
    ):
        metadata["evidence_rag_result"] = rag_result


def tool_execution_node(state: AgentState) -> AgentState:
    """Execute validated plan steps in the exact order selected by the model."""
    steps = _with_local_fallbacks(_get_executable_steps(state), state)
    # `need_human` describes the result of the current execution round.  A
    # failed early attempt must not poison a later successful RAG/FAQ result.
    state["need_human"] = False
    state["faq_result"] = None
    state.setdefault("rag_result", {})
    state["web_search_result"] = None
    state["last_executed_tool"] = ""
    state["answer_source"] = "no_answer"
    budget = _ensure_budget(state)
    cache = state.setdefault("execution_cache", {})

    for step in steps:
        tool = step["tool"]
        scoped_models = state.get("metadata", {}).get("manual_models") or []
        if scoped_models and tool in {"faq", "web_search", "direct_response"}:
            # An unscoped FAQ/web result must never be presented as evidence
            # for a selected vehicle manual. Keep Agentic query planning and
            # grading, but enforce the local manual retrieval boundary.
            tool = "rag"
        state["last_executed_tool"] = tool
        parameters = step.get("parameters", {})
        # Do not use ``dict.get(..., state[...])`` here: Python evaluates the
        # default expression eagerly, so valid plans with an explicit query
        # still crashed when a subgraph did not carry ``query_rewrite``.
        query = parameters.get("query") or state.get("query_rewrite") or state.get("user_question", "")
        cache_key = _cache_key(tool, parameters, query)
        trace = {
            "event_type": "tool",
            "name": tool,
            "tool": tool,
            "reason": step.get("reason", ""),
            "parameters": parameters,
            "cache_key": cache_key,
            "cache_hit": False,
            "started_ms": _workflow_elapsed_ms(state),
        }
        state.setdefault("metadata", {}).setdefault("tool_trace", []).append(trace)
        state.setdefault("execution_trace", []).append(trace)

        cached = cache.get(cache_key)
        if cached and cached.get("reusable"):
            trace.update({"cache_hit": True, "success": True, "duration_ms": 0})
            budget["cache_hits"] += 1
            state.setdefault("metadata", {})["active_tool_cache_key"] = cache_key
            _restore_cached_result(state, tool, cached.get("result"))
            print(f"  [Execution Cache] 复用 {tool}: {query}")
            return state
        if cached and cached.get("attempted"):
            trace.update({"deduplicated": True, "success": False, "duration_ms": 0,
                          "error": "相同调用已执行但结果不可复用"})
            budget["deduplicated_calls"] += 1
            print(f"  [Execution Cache] 跳过无效重复调用 {tool}: {query}")
            continue

        limit_reason = _budget_limit_reason(budget, tool)
        if limit_reason:
            trace.update({"budget_blocked": True, "success": False, "duration_ms": 0,
                          "error": limit_reason})
            budget["exhausted"] = True
            budget["stop_reason"] = limit_reason
            state.setdefault("metadata", {})["execution_budget_exhausted"] = limit_reason
            print(f"  [Execution Budget] {limit_reason}")
            break

        started = time.perf_counter()
        _record_call(budget, tool)

        if tool == "direct_response":
            state["answer_source"] = "direct_response"
            _finish_trace(trace, started, True)
            return state

        if tool == "human_handoff":
            state["answer_source"] = "no_answer"
            state["need_human"] = True
            _finish_trace(trace, started, True)
            return state

        if tool == "faq" and faq_tool:
            print("  [FAQ] 匹配中...")
            faq_result = faq_tool.match(query, threshold=parameters.get("threshold", 0.85))
            state["faq_result"] = faq_result
            cache[cache_key] = {"attempted": True, "reusable": bool(faq_result),
                                "result": copy.deepcopy(faq_result)}
            _finish_trace(trace, started, bool(faq_result))
            if faq_result:
                state["answer_source"] = "faq"
                return state

        elif tool == "rag" and (rag_tool or (scoped_models and multi_manual_tool)):
            print("  [RAG] 检索中...")
            state.setdefault("retrieval_query_history", []).append(query)
            state["active_retrieval_query"] = query
            state.setdefault("metadata", {})["active_tool_cache_key"] = cache_key
            selected_rag = multi_manual_tool if scoped_models else rag_tool
            rag_kwargs = {"model_names": scoped_models} if scoped_models else {}
            rag_result = selected_rag.search_and_generate(
                query, top_k=parameters.get("top_k", 6),
                max_context_length=parameters.get("max_context_length", 4000), **rag_kwargs,
            )
            state["rag_result"] = rag_result
            cache[cache_key] = {"attempted": True, "reusable": not rag_result.get("is_no_answer", False),
                                "result": copy.deepcopy(rag_result)}
            _preserve_rag_evidence(state, rag_result)
            _finish_trace(trace, started, not rag_result.get("is_no_answer", False))
            if not rag_result["is_no_answer"]:
                state["answer_source"] = "rag"
                return state

        elif tool == "web_search" and web_search_tool:
            if getattr(web_search_tool, "available", True) is False:
                state.setdefault("metadata", {})["web_search_available"] = False
                cache[cache_key] = {"attempted": True, "reusable": False, "result": None}
                _finish_trace(trace, started, False, "联网搜索不可用")
                continue
            print("  [Web Search] 启动博查搜索 MCP...")
            search_result = web_search_tool.search_query(query)
            state["web_search_result"] = search_result
            cache[cache_key] = {"attempted": True, "reusable": bool(search_result.get("success")),
                                "result": copy.deepcopy(search_result)}
            trace["source"] = search_result.get("source", "")
            _finish_trace(trace, started, bool(search_result.get("success")))
            if search_result["success"]:
                state["answer_source"] = "web_search"
                return state
            error = search_result.get("error") or search_result.get("answer") or "未知错误"
            trace["error"] = error
            print(f"  [Web Search] 搜索失败: {error}")

    state["answer_source"] = "no_answer"
    if any(step["tool"] == "web_search" for step in steps) and not web_search_tool:
        state.setdefault("metadata", {})["web_search_available"] = False

    return state


def _ensure_budget(state: AgentState) -> dict:
    budget = state.setdefault("execution_budget", {})
    budget.setdefault("started_at", time.monotonic())
    budget.setdefault("tool_calls", 0)
    budget.setdefault("rag_calls", 0)
    budget.setdefault("web_search_calls", 0)
    budget.setdefault("faq_calls", 0)
    budget.setdefault("cache_hits", 0)
    budget.setdefault("deduplicated_calls", 0)
    budget.setdefault("exhausted", False)
    return budget


def _budget_limit_reason(budget: dict, tool: str) -> str:
    if time.monotonic() - budget["started_at"] >= MAX_TOOL_SECONDS:
        return f"工具执行时间达到上限 ({MAX_TOOL_SECONDS:g}s)"
    if budget["tool_calls"] >= MAX_TOOL_CALLS:
        return f"工具调用达到上限 ({MAX_TOOL_CALLS})"
    if tool == "rag" and budget["rag_calls"] >= MAX_RAG_CALLS:
        return f"RAG 调用达到上限 ({MAX_RAG_CALLS})"
    if tool == "web_search" and budget["web_search_calls"] >= MAX_WEB_SEARCH_CALLS:
        return f"联网搜索调用达到上限 ({MAX_WEB_SEARCH_CALLS})"
    return ""


def _record_call(budget: dict, tool: str) -> None:
    budget["tool_calls"] += 1
    key = {"rag": "rag_calls", "web_search": "web_search_calls", "faq": "faq_calls"}.get(tool)
    if key:
        budget[key] += 1


def _cache_key(tool: str, parameters: dict, query: str) -> str:
    normalized_query = re.sub(r"\s+", "", str(query)).lower()
    effective = {}
    if tool == "rag":
        effective = {"top_k": parameters.get("top_k", 6),
                     "max_context_length": parameters.get("max_context_length", 4000)}
    elif tool == "faq":
        effective = {"threshold": parameters.get("threshold", 0.85)}
    return f"{tool}:{normalized_query}:{json.dumps(effective, sort_keys=True, ensure_ascii=False)}"


def _restore_cached_result(state: AgentState, tool: str, result) -> None:
    value = copy.deepcopy(result)
    if tool == "faq":
        state["faq_result"], state["answer_source"] = value, "faq"
    elif tool == "rag":
        state["rag_result"], state["answer_source"] = value, "rag"
        _preserve_rag_evidence(state, value)
    elif tool == "web_search":
        state["web_search_result"], state["answer_source"] = value, "web_search"
    else:
        state["answer_source"] = tool


def _finish_trace(trace: dict, started: float, success: bool, error: str = "") -> None:
    trace["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
    trace["success"] = success
    if error:
        trace["error"] = error


def _workflow_elapsed_ms(state: AgentState) -> float:
    started = state.setdefault("metrics", {}).setdefault("workflow_started_at", time.monotonic())
    return round((time.monotonic() - started) * 1000, 2)


def _get_executable_steps(state: AgentState) -> list:
    """Prefer the validated structured plan and support legacy callers."""
    structured = state.get("structured_plan") or {}
    steps = structured.get("steps") if isinstance(structured, dict) else None
    plan = state.get("plan", [])
    if isinstance(steps, list) and [step.get("tool") for step in steps] == plan:
        return steps
    query = state.get("query_rewrite", state.get("user_question", ""))
    return [{
        "tool": tool,
        "reason": "legacy plan compatibility",
        "parameters": {"query": query} if tool in {"faq", "rag", "web_search"} else {},
    } for tool in plan]


def _with_local_fallbacks(steps: list, state: AgentState) -> list:
    """Add a bounded local fallback when a narrow lookup can legitimately miss.

    FAQ is an exact/semantic shortcut, not an authoritative proof that the
    knowledge base has no answer.  A miss must fall through to local RAG before
    the workflow declares no-answer or considers the web.
    """
    tools = [step.get("tool") for step in steps]
    if "faq" in tools and "rag" not in tools:
        query = state.get("query_rewrite") or state.get("current_subquery") or state["user_question"]
        steps = list(steps) + [{
            "tool": "rag",
            "reason": "FAQ 未命中时回退到本地知识库",
            "parameters": {"query": query},
        }]
        state.setdefault("metadata", {})["faq_rag_fallback_injected"] = True
    return steps
