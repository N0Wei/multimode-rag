#!/usr/bin/env python
# coding: utf-8
"""Bounded scope and safety guard before the text Agentic-RAG workflow."""

import json
import re
from typing import Literal

from agent.config.model_config import ModelConfig
from agent.state import AgentState


_DANGEROUS_ACTIONS = ("拆除", "拆掉", "禁用", "屏蔽", "绕过", "破解", "短接", "伪造", "篡改")
_SAFETY_TARGETS = (
    "安全气囊", "制动", "刹车", "安全带", "报警", "碰撞", "转向", "电池保护", "高压系统",
)


def scope_guard_node(state: AgentState) -> AgentState:
    question = state["user_question"]
    decision = _rule_decision(question)
    if decision is None and state.get("intent") == "general":
        decision = _classify_scope_with_llm(question)
    if decision is None:
        decision = {"route": "in_scope", "reason": "已识别为汽车客服业务问题", "method": "intent"}

    state.setdefault("metadata", {})["scope_guard"] = decision
    state["scope_route"] = decision["route"]
    print(f"  [Scope Guard] {decision['route']}: {decision['reason']}")
    return state


def route_after_scope_guard(state: AgentState) -> Literal["continue_agentic", "guard_finalize"]:
    return "continue_agentic" if state.get("scope_route") == "in_scope" else "guard_finalize"


def guard_response_node(state: AgentState) -> AgentState:
    route = state.get("scope_route")
    if route == "unsafe":
        state["draft_answer"] = (
            "抱歉，我不能提供拆除、绕过或破坏车辆安全系统的操作指导。"
            "这类操作可能导致严重人身风险，请停止自行处理并联系授权维修机构进行检查。"
        )
        state["need_human"] = True
    else:
        state["draft_answer"] = (
            "这个问题超出了当前汽车用户手册与客服知识库的服务范围，"
            "我无法基于现有资料给出可靠答案。"
        )
        state["need_human"] = False
    state["answer_source"] = "no_answer"
    state["confidence"] = 1.0
    state["citation_coverage"] = 1.0
    state["grounding_status"] = "skipped"
    state["reflection_count"] = 0
    return state


def _rule_decision(question: str):
    if any(action in question for action in _DANGEROUS_ACTIONS) and any(
        target in question for target in _SAFETY_TARGETS
    ):
        return {
            "route": "unsafe",
            "reason": "请求包含对安全关键系统的拆除、绕过或破坏性操作",
            "method": "rule",
        }
    return None


def _classify_scope_with_llm(question: str) -> dict:
    prompt = f"""你是汽车智能客服的业务范围判定器。业务范围包括车辆功能、操作、参数、维护、安全说明、售前售后、车型发布、官方价格与促销政策。

这里只判断业务领域，不判断本地知识库是否已经收录答案，也不决定使用本地检索还是联网搜索。汽车相关但具有“今天、当前、最新、实时”等时效要求的问题仍属于 in_scope，后续路由会负责联网。

用户问题：{question}

判断：
- in_scope：能够由车辆用户手册或汽车客服知识回答；
- unsafe：要求实施危险、违法、规避安全保护或破坏车辆安全系统的操作；
- out_of_scope：与车辆、用车或汽车客户服务明显无关。

只输出 JSON：{{"route":"in_scope|unsafe|out_of_scope","reason":"一句话理由"}}"""
    try:
        response = ModelConfig.get_intent_llm().invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        match = re.search(r"\{[\s\S]*?\}", text)
        if not match:
            raise ValueError("范围分类器没有返回 JSON")
        value = json.loads(match.group())
        route = value.get("route")
        if route not in {"in_scope", "unsafe", "out_of_scope"}:
            raise ValueError("范围分类器 route 不合法")
        return {"route": route, "reason": str(value.get("reason") or "模型范围判断"), "method": "llm"}
    except Exception as exc:
        # Fail open to the existing bounded Agent workflow; never reject a
        # legitimate customer request merely because the guard model failed.
        return {"route": "in_scope", "reason": f"范围分类不可用，保守进入受限工作流: {exc}", "method": "fallback"}
