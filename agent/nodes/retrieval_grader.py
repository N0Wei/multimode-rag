#!/usr/bin/env python
# coding: utf-8
"""Grade retrieved evidence and decide whether the Agent should search again."""

import json
import os
import re
from typing import Literal

from agent.config.model_config import ModelConfig
from agent.state import AgentState


MAX_RETRIEVAL_RETRIES = int(os.getenv("AGENT_MAX_RETRIEVAL_RETRIES", "2"))


def retrieval_grader_node(state: AgentState) -> AgentState:
    """Check whether RAG evidence can support an answer to the user's question."""
    rag_was_last_tool = state.get("last_executed_tool") == "rag"
    if state.get("answer_source") != "rag" and not rag_was_last_tool:
        state["retrieval_grade"] = "not_applicable"
        state["retrieval_grade_reason"] = "当前答案未使用 RAG，无需评估检索证据"
        return state

    rag_result = state.get("rag_result") or {}
    documents = rag_result.get("documents") or []
    context = rag_result.get("context") or ""
    # Grade evidence against the user's actual information need, not a query
    # produced by a previous retry.  Otherwise an over-expanded rewrite can
    # silently turn an already answered question into a new one.
    question = state.get("current_subquery") or state["user_question"]
    grade = _grade_evidence(question, documents, context, rag_result.get("confidence", 0.0))

    state["retrieval_grade"] = "sufficient" if grade["sufficient"] else "insufficient"
    state["retrieval_grade_reason"] = grade["reason"]
    state.setdefault("metadata", {})["retrieval_grader"] = grade

    print(f"  [Retrieval Grader] {state['retrieval_grade']}: {grade['reason']}")
    if grade["sufficient"]:
        # Evidence grading is the authoritative decision about whether the
        # retrieved material can answer the question.  Recover from a false
        # no-answer produced by generation keywords or a coarse distance
        # threshold, and make the current round internally consistent.
        rag_result["is_no_answer"] = False
        if _looks_like_no_answer(str(rag_result.get("answer") or "")):
            rag_result["answer"] = _regenerate_from_accepted_evidence(question, context)
        state["answer_source"] = "rag"
        state["need_human"] = False
        cache_key = state.get("metadata", {}).get("active_tool_cache_key")
        cached = state.get("execution_cache", {}).get(cache_key) if cache_key else None
        if cached:
            cached["reusable"] = True
            cached["result"] = rag_result
        return state

    # The tool returned data, but the grader rejected its evidence. Keep the
    # attempt for deduplication while preventing it from being reused as truth.
    cache_key = state.get("metadata", {}).get("active_tool_cache_key")
    cached = state.get("execution_cache", {}).get(cache_key) if cache_key else None
    if cached:
        cached["reusable"] = False
        cached["rejection_reason"] = grade["reason"]

    retry_count = state.get("retrieval_retry_count", 0)
    # Mark the current result unusable so the executor can fall back to web search.
    rag_result["is_no_answer"] = True
    state["answer_source"] = "no_answer"

    if retry_count < MAX_RETRIEVAL_RETRIES:
        state["retrieval_retry_count"] = retry_count + 1
        state["query_rewrite"] = _rewrite_for_retrieval(
            original_question=state.get("current_subquery") or state["user_question"],
            previous_query=question,
            reason=grade["reason"],
            retry_number=retry_count + 1,
        )
        state["plan"] = ["rag"]
        state["structured_plan"] = {
            "reasoning": "检索证据不足，使用改写后的查询重新检索",
            "steps": [{"tool": "rag", "reason": grade["reason"], "parameters": {"query": state["query_rewrite"]}}],
        }
        print(f"  [Retrieval Grader] 改写查询: {state['query_rewrite']}")
    else:
        if state.get("metadata", {}).get("allow_web_fallback", False):
            state["plan"] = ["web_search"]
            state["structured_plan"] = {
                "reasoning": "允许外部时效检索，本地知识库重检达到上限后切换联网搜索",
                "steps": [{"tool": "web_search", "reason": grade["reason"], "parameters": {"query": question}}],
            }
            print("  [Retrieval Grader] 重检次数已达上限，按路由边界切换联网搜索")
        else:
            state["plan"] = []
            state["structured_plan"] = {"reasoning": "本地知识边界内无充分证据", "steps": []}
            state.setdefault("metadata", {})["web_search_available"] = False
            print("  [Retrieval Grader] 本地证据不足且路由不允许联网，返回无答案")

    return state


def route_after_retrieval_grade(
    state: AgentState,
) -> Literal["synthesize", "retry_retrieval"]:
    """Route accepted evidence to generation; otherwise execute the new plan."""
    if state.get("retrieval_grade") in {"sufficient", "not_applicable"}:
        return "synthesize"
    return "retry_retrieval"


def _grade_evidence(question: str, documents: list, context: str, confidence: float) -> dict:
    """Use an LLM grader, with a deterministic fallback when the model is unavailable."""
    if not documents or not context.strip():
        return {"sufficient": False, "reason": "没有检索到可用证据", "method": "rule"}

    evidence = context[:6000]
    visual_request = any(marker in question for marker in ("图片", "示意图", "图示", "布局图", "照片", "图中"))
    visual_evidence = [
        document for document in documents
        if getattr(document, "metadata", {}).get("doc_type") == "image"
        and getattr(document, "metadata", {}).get("image_path")
    ]
    visual_contract = (
        "该问题要求视觉内容。证据中标记为 image 且包含 image_path 的条目代表实际可交付的图片资产；"
        "不要因为提示词中只展示了图片路径和文字描述就判定‘没有图片’，只需判断图片主题/描述是否与问题相关。"
        if visual_request and visual_evidence else ""
    )
    prompt = f"""你是 RAG 检索质量评估器。只判断检索证据能否支持回答问题，不评价文风，也不要补充外部知识。

问题：{question}
检索置信度：{confidence:.2f}
证据：
{evidence}

业务域约束：用户正在当前知识库对应的产品/业务系统中提问。对于省略业务对象的表达，应在当前知识库语境内理解，不得擅自改写成跨领域的通用科普问题。
视觉证据契约：{visual_contract or '本题不适用'}

严格输出 JSON：
{{"sufficient": true, "reason": "一句话说明理由"}}

判定标准：证据必须与问题直接相关，并包含回答核心事实所需的信息；仅主题相似但缺少答案时判为 false。"""
    try:
        response = ModelConfig.get_reflection_llm().invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        match = re.search(r"\{[\s\S]*?\}", text)
        if not match:
            raise ValueError("grader did not return JSON")
        parsed = json.loads(match.group())
        sufficient = parsed.get("sufficient") is True
        reason = str(parsed.get("reason") or "模型未提供理由")
        return {"sufficient": sufficient, "reason": reason, "method": "llm"}
    except Exception as exc:
        # Keep the workflow usable if the grader model or JSON parsing fails.
        sufficient = confidence >= 0.65 and bool(documents)
        reason = (
            "评分模型不可用，按检索置信度规则判定为可用"
            if sufficient
            else "评分模型不可用，且检索置信度不足"
        )
        return {"sufficient": sufficient, "reason": reason, "method": "fallback", "error": str(exc)}


def _looks_like_no_answer(answer: str) -> bool:
    return not answer.strip() or any(marker in answer for marker in ("无答案", "无法回答", "没有找到", "不知道"))


def _regenerate_from_accepted_evidence(question: str, context: str) -> str:
    """Resolve the rare conflict: evidence is sufficient but generation said no-answer."""
    prompt = f"""请只根据已通过充分性校验的证据回答问题。

问题：{question}
证据：
{context[:6000]}

要求：直接回答原问题；不添加证据之外的信息；保留必要单位和操作条件；不要解释评测过程。"""
    try:
        response = ModelConfig.get_synthesize_llm().invoke(prompt)
        answer = response.content.strip() if hasattr(response, "content") else str(response).strip()
        if answer and not _looks_like_no_answer(answer):
            return answer
    except Exception as exc:
        print(f"  [Retrieval Grader] 证据答案再生成失败: {exc}")
    return "已检索到能够回答该问题的本地证据，请参考下方来源与图片。"


def _rewrite_for_retrieval(
    original_question: str,
    previous_query: str,
    reason: str,
    retry_number: int,
) -> str:
    """Create a search-focused query after the evidence grader rejects a result."""
    prompt = f"""你是 RAG 查询改写器。请根据失败原因生成一个更适合知识库检索的中文查询。

原始问题：{original_question}
上次查询：{previous_query}
失败原因：{reason}
当前重试轮次：{retry_number}

要求：
1. 保持原始问题的任务目标、实体关系和约束条件不变；
2. 只允许同义改写、指代消解和已有关键词提炼，不得添加原问题无法推出的新实体、新约束或新信息需求；
3. 改写后的查询必须仍能由原问题独立推出；若无法安全改写，原样返回原始问题；
4. 不要回答问题，只输出一条查询。"""
    try:
        response = ModelConfig.get_planner_llm().invoke(prompt)
        rewritten = response.content.strip() if hasattr(response, "content") else str(response).strip()
        if 2 <= len(rewritten) <= 300 and rewritten != previous_query:
            return rewritten
    except Exception as exc:
        print(f"  [Retrieval Grader] 查询改写失败: {exc}")
    return original_question
