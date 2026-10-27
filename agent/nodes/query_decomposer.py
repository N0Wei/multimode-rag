#!/usr/bin/env python
# coding: utf-8
"""Detect complex questions and decompose them into bounded sub-questions."""

import json
import os
import re
from typing import Any, Dict, List

from agent.config.model_config import ModelConfig
from agent.nodes.bounded_router import requires_fresh_web
from agent.state import AgentState, format_conversation_history


MAX_SUBQUERIES = int(os.getenv("AGENT_MAX_SUBQUERIES", "5"))


def query_decomposer_node(state: AgentState) -> AgentState:
    """Initialize a single-query or multi-hop run."""
    question = state["user_question"]
    intent = state.get("intent", "general")
    history = state.get("conversation_history", [])

    if intent in {"complaint", "chitchat"}:
        decision = _single_query_decision(question, "该意图不需要多跳检索")
    elif requires_fresh_web(question):
        # Freshness queries must preserve the user's exact time scope. Letting
        # an LLM fan them out by guessed products both widens the request and
        # burns the bounded web-search budget.
        decision = _single_query_decision(question, "时效性问题保留原查询直接联网")
    else:
        decision = _decompose_with_llm(question, intent, history)

    state["is_complex_query"] = decision["is_complex"]
    state["decomposition_reason"] = decision["reason"]
    state["sub_queries"] = decision["sub_queries"]
    state["current_subquery_index"] = 0
    state["current_subquery"] = decision["sub_queries"][0]
    state["subquery_results"] = []
    state["retrieval_retry_count"] = 0
    state.setdefault("metadata", {})["decomposition"] = decision

    print(f"  [Decomposer] complex={decision['is_complex']}: {decision['reason']}")
    if decision["is_complex"]:
        print(f"  [Decomposer] 子问题: {decision['sub_queries']}")
    return state


def _decompose_with_llm(question: str, intent: str, history: List[dict]) -> Dict[str, Any]:
    history_text = format_conversation_history(history, max_turns=3) or "无"
    prompt = f"""你是 Agentic RAG 的问题分解器。判断问题是否必须通过多个独立检索任务才能完整回答。

用户意图：{intent}
对话历史：
{history_text}
当前问题：{question}

只输出 JSON：
{{
  "is_complex": true,
  "reason": "需要分别检索两个车型的多个指标",
  "sub_queries": ["独立完整的子问题1", "独立完整的子问题2"]
}}

规则：
1. 单一事实、单一对象介绍、闲聊不是复杂问题，sub_queries 只保留原问题。
2. 比较多个对象、同时询问多个独立属性、需要先查 A 再基于 A 查 B 时才拆分。
3. 复杂问题拆成 2~{MAX_SUBQUERIES} 个互不重复、可以独立检索的中文问题。
4. 每个子问题必须补全实体和限定条件，不使用“它”“前者”等指代词。
5. 不回答问题，不增加原问题没有要求的调查方向。"""
    try:
        response = ModelConfig.get_planner_llm().invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        return _validate_decomposition(_extract_json(text), question)
    except Exception as exc:
        print(f"  [Decomposer] 模型分解失败: {exc}，使用规则降级")
        return _fallback_decomposition(question)


def _extract_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("问题分解器没有返回 JSON")
    value = json.loads(match.group())
    if not isinstance(value, dict):
        raise ValueError("分解结果必须是 JSON 对象")
    return value


def _validate_decomposition(candidate: Dict[str, Any], original_question: str) -> Dict[str, Any]:
    is_complex = candidate.get("is_complex")
    reason = candidate.get("reason")
    sub_queries = candidate.get("sub_queries")
    if not isinstance(is_complex, bool):
        raise ValueError("is_complex 必须是布尔值")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason 必须是非空字符串")
    if not isinstance(sub_queries, list) or not all(isinstance(q, str) for q in sub_queries):
        raise ValueError("sub_queries 必须是字符串列表")

    cleaned = []
    for query in sub_queries:
        query = query.strip()
        if not 2 <= len(query) <= 200:
            raise ValueError("子问题长度必须为 2~200 个字符")
        if query not in cleaned:
            cleaned.append(query)
    if is_complex and not 2 <= len(cleaned) <= MAX_SUBQUERIES:
        raise ValueError(f"复杂问题必须包含 2~{MAX_SUBQUERIES} 个子问题")
    if not is_complex:
        cleaned = [original_question]
    return {"is_complex": is_complex, "reason": reason.strip(), "sub_queries": cleaned, "method": "llm"}


def _single_query_decision(question: str, reason: str) -> Dict[str, Any]:
    return {"is_complex": False, "reason": reason, "sub_queries": [question], "method": "rule"}


def _fallback_decomposition(question: str) -> Dict[str, Any]:
    # The fallback is deliberately conservative: it never invents entities.
    separators = ["并且", "以及", "同时", "分别"]
    is_likely_complex = any(word in question for word in ["对比", "比较", "区别"]) or sum(
        word in question for word in separators
    ) >= 1
    return _single_query_decision(
        question,
        "规则检测到复杂表达，但无法安全生成独立子问题，保留原问题"
        if is_likely_complex
        else "规则判断为单一检索问题",
    )
