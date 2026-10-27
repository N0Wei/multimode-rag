#!/usr/bin/env python
# coding: utf-8
"""Finite capability routing with explicit tool and fallback boundaries."""

from agent.state import AgentState


_TEMPORAL_MARKERS = (
    "今天", "今日", "现在", "当前", "最新", "实时", "刚刚", "近期", "本周", "本月", "今年",
)
_WEB_REQUEST_MARKERS = ("联网", "网上搜索", "搜索网页", "查一下官网", "官网最新")


def bounded_planning_node(state: AgentState) -> AgentState:
    """Choose from a closed routing policy; intent labels do not select tools."""
    question = state.get("current_subquery") or state["user_question"]
    intent = state.get("intent", "general")

    if intent == "complaint":
        steps = [_step("human_handoff", "用户明确投诉或请求人工处理")]
        route = "human_handoff"
    elif intent == "chitchat":
        steps = [_step("direct_response", "无需外部证据的会话响应")]
        route = "direct_response"
    elif requires_fresh_web(question):
        steps = [_step("web_search", "问题包含明确时效约束，需要外部最新信息", query=question)]
        route = "temporal_web"
    else:
        # FAQ is a cheap, curated fast path. A miss falls through to the local
        # manual; external web is deliberately outside this route's boundary.
        steps = [
            _step("faq", "优先匹配人工审核标准答案", query=question),
            _step("rag", "FAQ 未命中时检索本地知识库", query=question),
        ]
        route = "local_knowledge"

    state["plan"] = [step["tool"] for step in steps]
    state["structured_plan"] = {
        "reasoning": f"有限能力路由: {route}",
        "steps": steps,
        "fallback_used": False,
        "output_valid": True,
    }
    metadata = state.setdefault("metadata", {})
    metadata["capability_route"] = route
    metadata["allow_web_fallback"] = route == "temporal_web"
    metadata["planner"] = {
        "reasoning": state["structured_plan"]["reasoning"],
        "fallback_used": False,
        "output_valid": True,
        "error_type": "",
        "validation_error": "",
        "method": "bounded_policy",
    }
    print(f"  [Bounded Router] {route}: {state['plan']}")
    return state


def requires_fresh_web(question: str) -> bool:
    normalized = str(question).lower()
    return any(marker in normalized for marker in _TEMPORAL_MARKERS + _WEB_REQUEST_MARKERS)


def _step(tool: str, reason: str, query: str = "") -> dict:
    return {
        "tool": tool,
        "reason": reason,
        "parameters": {"query": query} if tool in {"faq", "rag", "web_search"} else {},
    }
