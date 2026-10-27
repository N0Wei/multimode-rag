#!/usr/bin/env python
# coding: utf-8
"""
Agent 状态定义
使用 TypedDict 定义 LangGraph 状态流转
"""

from typing import TypedDict, List, Dict, Optional, Any
from langchain_core.documents import Document


class AgentState(TypedDict):
    """
    Agent 工作流状态
    """
    # === 输入 ===
    user_question: str              # 用户原始问题
    conversation_history: List[Dict]  # 对话历史: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    
    # === 意图与规划 ===
    intent: str                     # 意图分类: product_inquiry, faq, general, complaint
    query_rewrite: str              # 查询重写后的问题
    plan: List[str]                 # 执行计划
    structured_plan: Dict[str, Any] # 经校验的模型执行计划
    scope_route: str               # in_scope / unsafe / out_of_scope

    # === Query Decomposition / Multi-hop ===
    is_complex_query: bool          # 是否需要拆分为多个子问题
    decomposition_reason: str       # 复杂度判定及拆分理由
    sub_queries: List[str]          # 经过校验的子问题列表
    current_subquery_index: int     # 当前执行到的子问题索引
    current_subquery: str           # 当前正在处理的子问题
    subquery_results: List[Dict[str, Any]]  # 每个子问题的工具结果与证据
    
    # === 工具调用结果 ===
    rag_result: Dict[str, Any]      # RAG 检索结果
    faq_result: Optional[Dict]      # FAQ 匹配结果
    web_search_result: Optional[Dict]  # 联网搜索结果
    last_executed_tool: str         # 最近实际执行的工具，用于失败后的正确路由
    
    # === 答案生成 ===
    draft_answer: str               # 草稿答案

    # === Answer Grounding / Citation Coverage ===
    claims: List[Dict[str, Any]]     # 从草稿中抽取的可验证事实声明
    evidence_catalog: List[Dict[str, Any]]  # 统一编号后的检索证据
    claim_verdicts: List[Dict[str, Any]]    # 每条声明的证据判定与引用
    citation_coverage: float         # 被证据支持的声明比例 (0-1)
    grounding_status: str            # grounded / needs_rewrite / skipped
    grounding_retry_count: int       # 基于现有证据改写的次数
    
    # === Reflection 反思 ===
    reflection_count: int           # 反思轮数
    quality_score: float            # 答案质量分数 (0-10)
    reflection_notes: List[str]     # 反思意见
    improvement_actions: List[str]  # 改进建议

    # === Retrieval Grading ===
    retrieval_grade: str            # sufficient / insufficient / not_applicable
    retrieval_grade_reason: str     # 检索证据判定理由
    retrieval_retry_count: int      # 因证据不足触发的重检次数
    retrieval_query_history: List[str]  # 每轮实际使用的检索查询
    active_retrieval_query: str     # 最近一次 RAG 工具实际使用的查询
    
    # === 输出 ===
    final_answer: str               # 最终答案
    answer_source: str              # 答案来源: rag, faq, web_search, no_answer
    confidence: float               # 置信度
    need_human: bool                # 是否需要转人工
    
    # === 元数据 ===
    metadata: Dict[str, Any]        # 其他元信息
    metrics: Dict[str, Any]         # 节点、模型、工具、token 与总耗时指标
    execution_trace: List[Dict[str, Any]]  # 按发生顺序记录节点/模型/工具事件
    execution_cache: Dict[str, Any] # 当前请求内的工具结果缓存
    execution_budget: Dict[str, Any] # 工具调用预算与使用量


def format_conversation_history(history: List[Dict], max_turns: int = 5) -> str:
    """
    格式化对话历史为文本形式
    
    Args:
        history: 对话历史列表
        max_turns: 最多保留的对话轮数
    
    Returns:
        格式化后的对话历史文本
    """
    if not history:
        return ""
    
    # 只保留最近 N 轮对话
    recent_history = history[-(max_turns * 2):]  # 每轮包含 user + assistant
    
    formatted = []
    for msg in recent_history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        if role == "user":
            formatted.append(f"用户: {content}")
        elif role == "assistant":
            formatted.append(f"助手: {content}")
    
    return "\n".join(formatted)
