#!/usr/bin/env python
# coding: utf-8
"""Synthesize a complex answer from independently collected sub-query evidence."""

import json

from agent.config.model_config import ModelConfig
from agent.state import AgentState
from llm_streaming import generate_text_streaming


def multi_hop_synthesize_node(state: AgentState) -> AgentState:
    # A newly synthesized answer starts a fresh bounded grounding pass.
    state["grounding_retry_count"] = 0
    results = state.get("subquery_results", [])
    successful = [item for item in results if item.get("status") == "success"]
    if not successful:
        state["draft_answer"] = "抱歉，多个子问题均未检索到足够证据，暂时无法可靠回答。"
        state["answer_source"] = "no_answer"
        state["confidence"] = 0.0
        state["need_human"] = False
        return state

    evidence_payload = [{
        "question": item["question"],
        "answer": item["answer"],
        "source": item["source"],
        "evidence": item["evidence"],
    } for item in successful]
    prompt = f"""你是多跳 RAG 答案合成器。只能使用下列子问题结果回答原始问题。

原始问题：{state['user_question']}
子问题结果：
{json.dumps(evidence_payload, ensure_ascii=False, default=str)}

要求：
1. 直接回答原始问题，并把多个子问题的结论组织成清晰结构。
2. 比较问题要使用一致维度进行对比。
3. 不得补充证据中不存在的数字或事实。
4. 若部分子问题失败，明确说明缺失部分，不要猜测。
5. 使用中文。"""
    try:
        answer = generate_text_streaming(
            ModelConfig.get_synthesize_llm(), prompt, replace=True
        )
        if not answer:
            raise ValueError("多跳合成模型返回空答案")
    except Exception as exc:
        print(f"  [Multi-hop] 合成失败: {exc}，使用结果拼接降级")
        answer = "\n\n".join(f"### {item['question']}\n{item['answer']}" for item in successful)

    failed = [item["question"] for item in results if item.get("status") != "success"]
    if failed:
        answer += "\n\n未获得充分证据的部分：" + "；".join(failed)
    state["draft_answer"] = answer
    state["answer_source"] = "multi_hop"
    confidences = [item["confidence"] for item in successful if item.get("confidence", 0) > 0]
    state["confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
    print(f"  [Multi-hop] 已汇总 {len(successful)}/{len(results)} 个子问题")
    return state
