#!/usr/bin/env python
# coding: utf-8
"""Collect sub-query evidence and drive the bounded multi-hop loop."""

from typing import Literal

from agent.state import AgentState


def collect_subquery_result_node(state: AgentState) -> AgentState:
    """Snapshot the current result, then advance without re-running completed work."""
    if not state.get("is_complex_query"):
        return state

    question = state["current_subquery"]
    source = state.get("answer_source", "no_answer")
    result = {
        "index": state.get("current_subquery_index", 0),
        "question": question,
        "status": "success" if source not in {"", "no_answer"} else "failed",
        "source": source or "no_answer",
        "answer": _current_answer(state),
        "confidence": float(state.get("confidence", 0.0) or _current_confidence(state)),
        "evidence": _current_evidence(state),
        "retrieval_grade": state.get("retrieval_grade", "not_applicable"),
        "retrieval_grade_reason": state.get("retrieval_grade_reason", ""),
    }
    state.setdefault("subquery_results", []).append(result)
    print(f"  [Subquery] {result['index'] + 1}/{len(state['sub_queries'])} {result['status']}: {question}")

    next_index = result["index"] + 1
    state["current_subquery_index"] = next_index
    state["retrieval_retry_count"] = 0
    state["active_retrieval_query"] = ""
    if next_index < len(state["sub_queries"]):
        state["current_subquery"] = state["sub_queries"][next_index]
        state["improvement_actions"] = []
    else:
        state["current_subquery"] = ""
    return state


def route_after_collection(
    state: AgentState,
) -> Literal["next_subquery", "multi_synthesize", "single_synthesize"]:
    if not state.get("is_complex_query"):
        return "single_synthesize"
    if state.get("current_subquery_index", 0) < len(state.get("sub_queries", [])):
        return "next_subquery"
    return "multi_synthesize"


def _current_answer(state: AgentState) -> str:
    source = state.get("answer_source")
    if source == "faq":
        return (state.get("faq_result") or {}).get("answer", "")
    if source == "rag":
        return (state.get("rag_result") or {}).get("answer", "")
    if source == "web_search":
        return (state.get("web_search_result") or {}).get("answer", "")
    return ""


def _current_confidence(state: AgentState) -> float:
    source = state.get("answer_source")
    if source == "faq":
        return float((state.get("faq_result") or {}).get("confidence", 0.0))
    if source == "rag":
        return float((state.get("rag_result") or {}).get("confidence", 0.0))
    return 0.0


def _current_evidence(state: AgentState):
    source = state.get("answer_source")
    if source == "faq":
        result = state.get("faq_result") or {}
        return {"matched_question": result.get("matched_question", ""), "source": result.get("source", "")}
    if source == "rag":
        result = state.get("rag_result") or {}
        return {"context": result.get("context", "")[:6000], "documents": result.get("documents", [])}
    if source == "web_search":
        return {"results": (state.get("web_search_result") or {}).get("results", [])}
    return {}
