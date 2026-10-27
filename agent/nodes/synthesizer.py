#!/usr/bin/env python
# coding: utf-8
"""
Answer synthesis node.

This node combines tool results into a draft answer. It uses only the bounded
current-session context prepared by the short-term memory layer.
"""

import re
from typing import List

from agent.config.model_config import ModelConfig
from agent.state import AgentState, format_conversation_history
from llm_streaming import generate_text_streaming


def synthesize_answer_node(state: AgentState) -> AgentState:
    # A newly synthesized answer starts a fresh bounded grounding pass.
    state["grounding_retry_count"] = 0
    answer_source = state.get("answer_source", "no_answer")
    history = state.get("conversation_history", [])

    if answer_source == "direct_response":
        draft_answer = _generate_chitchat_response(state["user_question"], history)
        confidence = 0.9

    elif answer_source == "faq":
        faq_result = state["faq_result"]
        draft_answer = _contextualize_faq_answer(
            faq_result["answer"],
            state["user_question"],
            history,
        )
        confidence = faq_result["confidence"]

    elif answer_source == "rag":
        rag_result = state["rag_result"]
        draft_answer = _enhance_answer_with_context(
            rag_result["answer"],
            state["user_question"],
            history,
        )
        draft_answer = _preserve_requested_unit(draft_answer, state["user_question"])
        confidence = rag_result["confidence"]

    elif answer_source == "web_search":
        web_search_result = state["web_search_result"]
        draft_answer = web_search_result["answer"]
        confidence = None

    else:
        if state.get("need_human"):
            draft_answer = "已记录您的诉求，正在为您转接人工客服。"
        else:
            draft_answer = "抱歉，当前可用证据不足，我暂时无法可靠回答这个问题。"
        confidence = 0.0
        # Evidence exhaustion is not a handoff decision. Only an explicit
        # human_handoff route (or a safety policy) may set this flag.
        state["need_human"] = bool(state.get("need_human"))

    state["draft_answer"] = draft_answer
    state["confidence"] = confidence if confidence is not None else 0.0

    print(f"  [Synthesize] answer source: {answer_source}")
    if confidence is not None:
        print(f"  [Synthesize] confidence: {confidence:.2f}")
    else:
        print("  [Synthesize] confidence: N/A (web search)")

    return state


def _preserve_requested_unit(answer: str, question: str) -> str:
    """Restore an explicitly requested unit when generation returns only a number."""
    requested_units = ("毫米", "厘米", "米", "公里", "千瓦", "千瓦时", "秒", "分钟", "小时")
    if not re.fullmatch(r"\s*[-+]?\d+(?:\.\d+)?\s*", answer):
        return answer
    for unit in requested_units:
        if unit in question:
            return f"{answer.strip()}{unit}"
    return answer


def _contextualize_faq_answer(faq_answer: str, question: str, history: List[dict]) -> str:
    if not history or len(history) < 2:
        return faq_answer
    return faq_answer


def _enhance_answer_with_context(answer: str, question: str, history: List[dict]) -> str:
    if len(answer) > 100 or len(history) < 2:
        return answer
    return answer


def _generate_chitchat_response(question: str, history: List[dict]) -> str:
    try:
        llm = ModelConfig.get_synthesize_llm()

        context = ""
        if history:
            context = format_conversation_history(history, max_turns=3)
            context = f"\n对话历史:\n{context}\n"

        prompt = f"""你是一个友好的智能客服助手。请对用户的问候或闲聊做出简短、自然、友好的回复。
{context}
用户: {question}

要求:
1. 回复简短、自然、友好
2. 如果用户问候，礼貌回应并询问有什么可以帮助的
3. 如果是闲聊，简短回应即可
4. 不要超过50字

助手:"""

        answer = generate_text_streaming(llm, prompt, replace=True)

        if len(answer) < 2 or len(answer) > 100:
            return "您好！我是智能客服助手，有什么可以帮您的吗？"

        return answer

    except Exception as e:
        print(f"  [Warning] chitchat generation failed: {e}")
        return "您好！我是智能客服助手，有什么可以帮您的吗？"
