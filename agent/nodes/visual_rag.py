#!/usr/bin/env python
# coding: utf-8
"""Deterministic plain-RAG lane for explicit visual retrieval requests."""

from typing import Literal

from agent.nodes.synthesizer import synthesize_answer_node
from agent.state import AgentState
from agent.tools.multimodal_rag_tool import MultimodalRAGTool


def route_after_intent(state: AgentState) -> Literal["visual_rag", "agentic_rag"]:
    """Keep visual retrieval out of the Agentic planning/retry loop."""
    return "visual_rag" if MultimodalRAGTool._is_visual_request(state["user_question"]) else "agentic_rag"


def visual_rag_plan_node(state: AgentState) -> AgentState:
    """Create exactly one local multimodal RAG call, without LLM planning."""
    question = state["user_question"]
    state["is_complex_query"] = False
    state["sub_queries"] = [question]
    state["current_subquery"] = question
    state["query_rewrite"] = question
    state["plan"] = ["rag"]
    state["structured_plan"] = {
        "reasoning": "显式图片请求走普通多模态 RAG 单次检索",
        "steps": [{
            "tool": "rag",
            "reason": "检索本地知识库中的图片证据",
            "parameters": {"query": question, "top_k": 6, "max_context_length": 4000},
        }],
    }
    state.setdefault("metadata", {})["retrieval_lane"] = "plain_visual_rag"
    print("  [Visual RAG] 使用普通多模态 RAG 单次检索，不进入 Agentic 循环")
    return state


def route_after_tool_execution(state: AgentState) -> Literal["visual_synthesize", "agentic_grade"]:
    if state.get("metadata", {}).get("retrieval_lane") == "plain_visual_rag":
        return "visual_synthesize"
    return "agentic_grade"


def visual_rag_synthesize_node(state: AgentState) -> AgentState:
    """Synthesize once and mark direct image evidence without LLM reflection."""
    state = synthesize_answer_node(state)
    rag_result = state.get("rag_result") or {}
    image_paths = rag_result.get("image_paths") or []
    if state.get("answer_source") == "rag" and image_paths:
        state["citation_coverage"] = 1.0
        state["grounding_status"] = "grounded"
    else:
        state["citation_coverage"] = 0.0
        state["grounding_status"] = "skipped"
    state["reflection_count"] = 0
    return state
