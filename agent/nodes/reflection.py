#!/usr/bin/env python
# coding: utf-8
"""
Reflection 反思节点
评估答案质量并决定是否重试
"""

from typing import Literal
from agent.state import AgentState
from agent.config.model_config import ModelConfig


def reflection_node(state: AgentState) -> AgentState:
    """
    Reflection 节点
    
    使用 LLM 评估答案质量:
    - 相关性 (Relevance)
    - 完整性 (Completeness)
    - 准确性 (Accuracy)
    - 流畅性 (Fluency)
    
    并在质量不足时自动生成改进建议
    """
    draft_answer = state["draft_answer"]
    question = state["user_question"]
    answer_source = state.get("answer_source", "no_answer")
    
    # 增加反思计数
    reflection_count = state.get("reflection_count", 0) + 1
    state["reflection_count"] = reflection_count

    # Terminal routes are control decisions, not open-ended answer-quality
    # tasks.  Calling an LLM judge here wastes latency and can send an explicit
    # handoff back into planning.
    if state.get("need_human") or answer_source in {"direct_response", "no_answer"}:
        state["quality_score"] = 10.0 if state.get("need_human") else (8.5 if answer_source == "direct_response" else 0.0)
        if state.get("need_human"):
            note = "已明确转人工，终止自动重试"
        elif answer_source == "direct_response":
            note = "闲聊已直接响应"
        else:
            note = "允许的证据路径已耗尽，终止自动重试"
        state["reflection_notes"] = state.get("reflection_notes", []) + [note]
        state["improvement_actions"] = []
        print(f"  [Reflection] {note}")
        return state
    
    # 使用 LLM 评估质量
    evaluation_result = _evaluate_with_llm(
        question=question,
        draft_answer=draft_answer,
        answer_source=answer_source,
        confidence=state.get("confidence", 0.0),
        context=state
    )
    
    quality_score = evaluation_result["score"]
    notes = evaluation_result["notes"]
    improvement_suggestions = evaluation_result.get("improvements", [])
    
    state["quality_score"] = quality_score
    state["reflection_notes"] = state.get("reflection_notes", []) + notes
    
    # 如果有改进建议，保存到状态
    if improvement_suggestions:
        state["improvement_actions"] = improvement_suggestions
    
    print(f"  [Reflection] 轮次: {reflection_count}/3")
    print(f"  [Reflection] 质量分数: {quality_score:.1f}/10")
    if notes:
        print(f"  [Reflection] 评估意见: {', '.join(notes[:2])}")
    if improvement_suggestions:
        print(f"  [Reflection] 改进建议: {', '.join(improvement_suggestions[:2])}")
    
    return state


def should_continue(state: AgentState) -> Literal["finalize", "retry"]:
    """
    决定是否继续 Reflection
    
    Returns:
        "finalize": 结束，输出最终答案
        "retry": 重新规划和执行
    """
    quality_score = state.get("quality_score", 0.0)
    reflection_count = state.get("reflection_count", 0)
    answer_source = state.get("answer_source", "")

    if state.get("need_human") or answer_source in {"direct_response", "no_answer"}:
        print("  [Decision] 已进入终态，结束 Reflection")
        return "finalize"

    if state.get("execution_budget", {}).get("exhausted"):
        reason = state["execution_budget"].get("stop_reason", "执行预算已用尽")
        print(f"  [Decision] {reason}，使用当前已有结果结束")
        return "finalize"

    # A retry cannot improve a web answer while the execution cache holds the
    # same search response. Re-evaluating it only spends LLM calls and latency;
    # a fresh-search policy, if added later, must use an explicit new query.
    if answer_source == "web_search":
        print("  [Decision] 联网搜索已完成，避免对同一缓存结果进行无效重试")
        state["improvement_actions"] = []
        return "finalize"

    if answer_source == "no_answer" and state.get("metadata", {}).get("web_search_available") is False:
        print("  [Decision] 无可用的后续检索工具，结束 Reflection")
        return "finalize"

    # Evidence checks have higher authority than a stylistic LLM judge.  When
    # retrieval and grounding both pass, do not let Reflection invent extra
    # requirements and widen the user's question on a retry.
    if (
        answer_source in {"rag", "multi_hop"}
        and state.get("retrieval_grade") == "sufficient"
        and state.get("grounding_status") == "grounded"
        and state.get("citation_coverage", 0.0) >= 0.8
    ):
        print("  [Decision] 检索证据充分且答案已落到证据，结束 Reflection")
        state["improvement_actions"] = []
        return "finalize"
    
    # 条件 1: 质量达标
    if quality_score >= 7.0:
        print(f"  [Decision] 质量达标 ({quality_score:.1f}/10)，结束 Reflection")
        return "finalize"
    
    # 条件 2: 达到最大重试次数
    if reflection_count >= 3:
        print("  [Decision] 达到最大重试次数 (3/3)，强制结束")
        return "finalize"
    
    # 条件 3: 来源为 FAQ 且分数不太低
    if answer_source == "faq" and quality_score >= 6.0:
        print(f"  [Decision] 答案来源可靠 ({answer_source})，结束 Reflection")
        return "finalize"
    
    # 条件 4: 质量不足，且还有重试机会
    if quality_score < 7.0 and reflection_count < 3:
        print(f"  [Decision] 质量不足 ({quality_score:.1f}/10)，第 {reflection_count} 次重试")
        
        # 改进建议已由 LLM 在 reflection_node 中生成
        improvement_actions = state.get("improvement_actions", [])
        if improvement_actions:
            print(f"  [Improvement] 改进措施: {', '.join(improvement_actions[:2])}")
        
        return "retry"
    
    # 其他情况: 质量一般但可接受
    print(f"  [Decision] 质量可接受 ({quality_score:.1f}/10)，结束 Reflection")
    return "finalize"


def _evaluate_with_llm(
    question: str,
    draft_answer: str,
    answer_source: str,
    confidence: float,
    context: AgentState
) -> dict:
    """
    使用 LLM 评估答案质量并生成改进建议
    
    Args:
        question: 用户问题
        draft_answer: 草稿答案
        answer_source: 答案来源
        confidence: 置信度
        context: 完整状态上下文
        
    Returns:
        {
            "score": float,           # 质量分数 0-10
            "notes": List[str],       # 评估意见
            "improvements": List[str] # 改进建议（仅在分数<7时返回）
        }
    """
    try:
        llm = ModelConfig.get_reflection_llm()
        
        # 构建评估提示词
        confidence_str = f"{confidence:.2f}" if confidence is not None else "N/A (网络搜索)"
        
        prompt = f"""你是一个严格的答案质量评估专家。请从以下维度评估答案质量：

**用户问题**: {question}

**草稿答案**: {draft_answer}

**答案来源**: {answer_source}
**检索置信度**: {confidence_str}

---

**评估维度**（每项0-10分）：

1. **相关性 (Relevance)**: 答案是否直接回答了用户问题？
2. **完整性 (Completeness)**: 信息是否充分？是否遗漏关键内容？
3. **准确性 (Accuracy)**: 事实陈述是否准确？是否有矛盾？
4. **流畅性 (Fluency)**: 表达是否自然、专业、易懂？

---

**评估要求**：

- 只评估答案是否回答了用户明确提出的问题，以及回答是否受已有证据支持
- 先从用户原问题中提取明确的信息需求与约束；完整性只在这些需求与约束内判断
- 不得把答案或证据中没有覆盖的新信息需求，当成答案必须补充的内容
- 如果答案已经覆盖原问题的核心信息，不得因缺少延伸背景而扣分
- 改进建议只能修正相关性、证据支持、表达或原问题内的遗漏，不得改变任务目标或扩大语义边界
- 如果答案包含"无答案"、"无法回答"等，分数≤5分
- 如果答案过短（<15字）且不是问候语，扣1分
- 如果答案来源是 FAQ，基础分10分
- 如果答案来源是 RAG 且置信度<0.6，基础分≤6.8分
- 如果答案来源是 no_answer，分数≤6分

**输出格式（严格JSON）**：
```json
{{
  "score": 7.5,
  "notes": [
    "相关性强，直接回答问题",
    "完整性中等，缺少充电时间说明",
    "准确性高，数据可信",
    "表达流畅自然"
  ],
  "improvements": [
    "补充快充和慢充的具体时间",
    "添加续航的NEDC和WLTP工况对比",
    "说明不同驾驶模式对续航的影响"
  ]
}}
```

**注意**：
- `score` 必须是 0-10 的数字
- `notes` 是评估意见列表，至少2条
- `improvements` 是具体的改进建议列表（仅当 score < 7.0 时需要提供，否则为空列表）
- 改进建议要具体、可操作，指明缺少什么信息或如何优化表达

请开始评估："""

        # 调用 LLM（LangChain 使用 invoke 方法）
        response = llm.invoke(prompt)
        
        # 提取文本内容
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        # 解析 JSON 响应
        import json
        import re
        
        # 尝试提取 JSON
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            result = json.loads(json_match.group())
            
            # 验证必需字段
            score = float(result.get("score", 5.0))
            notes = result.get("notes", ["LLM 评估完成"])
            improvements = result.get("improvements", [])
            
            # 限制分数范围
            score = max(0.0, min(10.0, score))
            
            return {
                "score": score,
                "notes": notes,
                "improvements": improvements
            }
        else:
            # 解析失败，使用基础规则
            print("  [Warning] LLM 响应解析失败，使用基础规则评估")
            return _fallback_evaluation(question, draft_answer, answer_source, confidence)
            
    except Exception as e:
        print(f"  [Error] Reflection LLM 调用失败: {e}")
        # 降级到基础规则
        return _fallback_evaluation(question, draft_answer, answer_source, confidence)


def _fallback_evaluation(
    question: str,
    answer: str,
    source: str,
    confidence: float
) -> dict:
    """
    降级评估方法（当 LLM 失败时使用）
    
    Returns:
        {"score": float, "notes": List[str], "improvements": List[str]}
    """
    notes = []
    improvements = []
    
    # 基础分数 (基于来源)
    if source == "faq":
        base_score = 10.0
        notes.append("FAQ 精确匹配，质量高")
    elif source == "rag":
        base_score = 8.0 + confidence * 2  # 7-10分
        if confidence > 0.8:
            notes.append("RAG 置信度高")
        else:
            notes.append("RAG 置信度中等")
            improvements.append("提高检索相关性")
    elif source == "web_search":
        base_score = 7.0
        notes.append("联网搜索结果，已标注来源")
    elif source == "direct_response":
        base_score = 8.5
        notes.append("直接响应，友好回复")
    elif source == "multi_hop":
        base_score = 7.5
        notes.append("多跳检索结果已完成汇总")
    else:
        base_score = 3.0
        notes.append("无法生成有效答案")
        improvements.append("尝试其他检索策略")
        improvements.append("考虑联网搜索")
    
    # 检查答案长度
    if len(answer) < 20:
        base_score -= 2
        notes.append("答案过短")
        improvements.append("补充更详细的信息")
    
    # 检查是否包含"无答案"
    if "无答案" in answer or "无法回答" in answer:
        base_score = min(base_score, 4.0)
        notes.append("答案为无答案")
        improvements.append("尝试不同的工具或数据源")
    
    # 限制分数范围
    quality_score = max(0, min(10, base_score))
    
    return {
        "score": quality_score,
        "notes": notes,
        "improvements": improvements if quality_score < 7.0 else []
    }
