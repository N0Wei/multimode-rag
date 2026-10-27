#!/usr/bin/env python
# coding: utf-8
"""
规划节点
制定工具调用计划
"""

import json
import re
from typing import Any, Dict, List
from agent.state import AgentState, format_conversation_history
from agent.config.model_config import ModelConfig


def planning_node(state: AgentState) -> AgentState:
    """
    规划节点
    
    根据意图和对话历史决定调用哪些工具
    支持指代消解（理解"它"、"那个"等代词）
    支持基于 Reflection 的重新规划
    """
    intent = state["intent"]
    question = state.get("current_subquery") or state["user_question"]
    history = state.get("conversation_history", [])
    memory_summary = state.get("metadata", {}).get("short_term_summary", "")
    reflection_count = state.get("reflection_count", 0)
    improvement_actions = state.get("improvement_actions", [])
    
    # 检查是否是重试
    is_retry = reflection_count > 0
    
    if is_retry:
        print(f"  [Planning] 第 {reflection_count} 次重新规划")
        print(f"  [Planning] 改进措施: {improvement_actions}")
    
    # 如果有对话历史，进行查询重写（指代消解）
    if not state.get("is_complex_query") and (history or memory_summary):
        query_rewrite = _rewrite_query_with_context(question, history, memory_summary)
        if not is_retry:  # 只在首次打印
            print(f"  [指代消解] 原问题: {question}")
            print(f"  [指代消解] 重写为: {query_rewrite}")
    else:
        query_rewrite = question
    
    structured_plan = _create_structured_plan(
        intent=intent,
        question=query_rewrite,
        improvement_actions=improvement_actions if is_retry else [],
        previous_source=state.get("answer_source", ""),
    )
    state["structured_plan"] = structured_plan
    state["plan"] = [step["tool"] for step in structured_plan["steps"]]
    state["query_rewrite"] = query_rewrite
    state.setdefault("metadata", {})["planner"] = {
        "reasoning": structured_plan["reasoning"],
        "fallback_used": structured_plan.get("fallback_used", False),
        "output_valid": structured_plan.get("output_valid", True),
        "error_type": structured_plan.get("error_type", ""),
        "validation_error": structured_plan.get("validation_error", ""),
    }
    print(f"  [Plan] 规划理由: {structured_plan['reasoning']}")
    print(f"  [Plan] 执行计划: {state['plan']}")
    
    return state


def _rewrite_query_with_context(question: str, history: List[dict], memory_summary: str = "") -> str:
    """
    利用对话历史重写查询（指代消解 + 上下文补全）
    
    完全依赖 LLM 判断：
    - 是否需要指代消解
    - 是否需要补充上下文信息
    - 是否需要重写查询
    
    示例1 - 指代消解:
    历史: "吉利银河E8的续航是多少？" -> "620公里"
    当前: "它的价格呢？"
    重写: "吉利银河E8的价格是多少？"
    
    示例2 - 上下文补全:
    历史: "特斯拉 Model 3 怎么样？" -> "..."
    当前: "充电速度如何？"
    重写: "特斯拉 Model 3 的充电速度如何？"
    
    示例3 - 无需重写:
    历史: "..."
    当前: "吉利银河E8的续航是多少？"（已经完整）
    重写: "吉利银河E8的续航是多少？"（原样返回）
    """
    try:
        llm = ModelConfig.get_planner_llm()
        
        history_text = format_conversation_history(history, max_turns=3)
        if memory_summary:
            history_text = f"Session summary:\n{memory_summary}\n\nRecent messages:\n{history_text}"
        
        prompt = f"""你是一个对话理解专家。请根据对话历史，判断当前问题是否需要重写以便更好地理解。

**对话历史**:
{history_text}

**当前问题**: {question}

---

**你的任务**:

1. **判断是否需要重写**：
   - 如果问题包含指代词（它、他、她、这个、那个等），需要替换为具体对象
   - 如果问题缺少主语但从上下文可以推断，需要补充
   - 如果问题本身已经完整清晰，无需重写

2. **重写原则**：
   - 保持问题的原意和语气
   - 只补充必要的信息，不要过度扩展
   - 确保重写后的问题是一个独立、完整的问题
   - 不要改变问题的时态或语境

3. **输出格式**：
   - 直接输出重写后的问题
   - 不要添加任何解释或前缀
   - 如果无需重写，原样返回问题

---

**示例**:

对话历史:
用户: 吉利银河E8的续航是多少？
助手: 吉利银河E8在CLTC工况下纯电续航里程为620公里。

当前问题: 它的价格呢？
重写后: 吉利银河E8的价格是多少？

---

对话历史:
用户: 介绍一下特斯拉 Model 3
助手: 特斯拉 Model 3 是一款紧凑型纯电动轿车...

当前问题: 充电速度如何？
重写后: 特斯拉 Model 3 的充电速度如何？

---

对话历史:
用户: 你好
助手: 你好！有什么可以帮你的？

当前问题: 吉利银河E8的续航是多少？
重写后: 吉利银河E8的续航是多少？

---

**现在请处理上面的问题，直接输出重写结果**:"""

        response = llm.invoke(prompt)
        rewritten = response.content.strip()
        
        # 验证重写结果的合理性
        if not rewritten or len(rewritten) < 2:
            # 重写结果太短，使用原问题
            return question
        
        if len(rewritten) > len(question) * 4:
            # 重写结果过长（可能 LLM 输出了解释），使用原问题
            print(f"  [警告] 重写结果过长，使用原问题")
            return question
        
        # 检查是否包含不应该出现的内容
        invalid_patterns = ["重写后", "原样返回", "无需重写", "解释", "因为"]
        if any(pattern in rewritten for pattern in invalid_patterns):
            # LLM 可能输出了解释而不是问题本身
            print(f"  [警告] 重写结果包含解释内容，使用原问题")
            return question
        
        return rewritten
        
    except Exception as e:
        print(f"  [警告] 查询重写失败: {e}")
        return question  # 失败时返回原问题


# Runtime planning uses this strict tool schema instead of intent-based if/else routing.
TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "faq": {"parameters": {"query": str, "threshold": (int, float)}},
    "rag": {"parameters": {"query": str, "top_k": int, "max_context_length": int}},
    "web_search": {"parameters": {"query": str}},
    "direct_response": {"parameters": {}},
    "human_handoff": {"parameters": {}},
}


def _create_structured_plan(intent, question, improvement_actions, previous_source):
    """Ask the planner model for an executable plan, then validate it locally."""
    prompt = f"""你是 Agentic RAG 工具规划器。自主选择最少且必要的工具及其执行顺序。

用户意图：{intent}
独立问题：{question}
上一轮来源：{previous_source or '无'}
改进建议：{json.dumps(improvement_actions, ensure_ascii=False)}

可用工具：
- faq：人工审核的标准 FAQ；参数 query、可选 threshold(0~1)
- rag：本地知识库；参数 query、可选 top_k(1~12)、max_context_length(500~12000)
- web_search：时效性互联网信息或知识库失败后的补充；参数 query
- direct_response：无需外部证据的闲聊；无参数
- human_handoff：投诉、退款、赔偿等人工请求；无参数

只输出 JSON，不要 Markdown：
{{"reasoning":"选择依据","steps":[{{"tool":"rag","reason":"需要本地证据","parameters":{{"query":"检索查询","top_k":6}}}}]}}

约束：steps 为 1~3 步且工具不可重复；后一步是前一步失败时的降级方案；direct_response 或 human_handoff 必须单独使用；不要默认选择所有工具；只能使用上述工具和参数。"""
    try:
        response = ModelConfig.get_planner_llm().invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        plan = _validate_plan(_extract_json_object(raw), question)
        plan["output_valid"] = True
        return plan
    except Exception as exc:
        print(f"  [警告] 动态规划失败: {exc}，使用安全降级计划")
        fallback = _fallback_structured_plan(intent, question)
        fallback["fallback_used"] = True
        fallback["output_valid"] = False
        if isinstance(exc, json.JSONDecodeError):
            fallback["error_type"] = "json_parse_error"
        elif isinstance(exc, ValueError):
            message = str(exc)
            fallback["error_type"] = "unknown_tool" if "未知工具" in message else "schema_validation_error"
        else:
            fallback["error_type"] = "model_call_error"
        fallback["validation_error"] = str(exc)
        return fallback


def _extract_json_object(text: str) -> Dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("规划模型没有返回 JSON 对象")
    value = json.loads(match.group())
    if not isinstance(value, dict):
        raise ValueError("执行计划必须是 JSON 对象")
    return value


def _validate_plan(candidate: Dict[str, Any], default_query: str) -> Dict[str, Any]:
    """Validate model output against a strict local allowlist."""
    reasoning, steps = candidate.get("reasoning"), candidate.get("steps")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("reasoning 必须是非空字符串")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 3:
        raise ValueError("steps 数量必须为 1~3")

    validated, seen = [], set()
    for index, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            raise ValueError(f"第 {index} 步必须是对象")
        tool = step.get("tool")
        if tool not in TOOL_SCHEMAS:
            raise ValueError(f"第 {index} 步包含未知工具: {tool}")
        if tool in seen:
            raise ValueError(f"工具不可重复: {tool}")
        seen.add(tool)
        reason, parameters = step.get("reason"), step.get("parameters", {})
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"第 {index} 步缺少 reason")
        if not isinstance(parameters, dict):
            raise ValueError(f"第 {index} 步 parameters 必须是对象")
        allowed = TOOL_SCHEMAS[tool]["parameters"]
        unknown = set(parameters) - set(allowed)
        if unknown:
            raise ValueError(f"工具 {tool} 包含未知参数: {sorted(unknown)}")
        for name, value in parameters.items():
            if isinstance(value, bool) or not isinstance(value, allowed[name]):
                raise ValueError(f"工具 {tool} 的参数 {name} 类型错误")

        clean = dict(parameters)
        if tool in {"faq", "rag", "web_search"}:
            clean.setdefault("query", default_query)
            if not clean["query"].strip() or len(clean["query"]) > 300:
                raise ValueError(f"工具 {tool} 的 query 无效")
        if "threshold" in clean and not 0 <= clean["threshold"] <= 1:
            raise ValueError("FAQ threshold 必须在 0~1")
        if "top_k" in clean and not 1 <= clean["top_k"] <= 12:
            raise ValueError("RAG top_k 必须在 1~12")
        if "max_context_length" in clean and not 500 <= clean["max_context_length"] <= 12000:
            raise ValueError("RAG max_context_length 必须在 500~12000")
        validated.append({"tool": tool, "reason": reason.strip(), "parameters": clean})

    if seen & {"direct_response", "human_handoff"} and len(validated) != 1:
        raise ValueError("direct_response 和 human_handoff 必须单独使用")
    return {"reasoning": reasoning.strip(), "steps": validated, "fallback_used": False}


def _fallback_structured_plan(intent: str, question: str) -> Dict[str, Any]:
    if intent == "complaint":
        tools = ["human_handoff"]
    elif intent == "chitchat":
        tools = ["direct_response"]
    elif intent == "faq":
        tools = ["faq", "rag"]
    else:
        tools = ["rag", "web_search"]
    return {
        "reasoning": "规划模型不可用或输出不合法，采用安全降级计划",
        "steps": [{
            "tool": tool,
            "reason": "安全降级",
            "parameters": {"query": question} if tool in {"faq", "rag", "web_search"} else {},
        } for tool in tools],
    }
