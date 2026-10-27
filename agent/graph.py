#!/usr/bin/env python
# coding: utf-8
"""
LangGraph 工作流定义
"""

from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import (
    intent_classify_node,
    planning_node,
    tool_execution_node,
    synthesize_answer_node,
    reflection_node,
    should_continue,
    retrieval_grader_node,
    route_after_retrieval_grade,
    query_decomposer_node,
    collect_subquery_result_node,
    route_after_collection,
    multi_hop_synthesize_node,
    claim_extractor_node,
    evidence_validator_node,
    grounded_rewriter_node,
    route_after_evidence_validation,
    route_after_intent,
    visual_rag_plan_node,
    route_after_tool_execution,
    visual_rag_synthesize_node,
    scope_guard_node,
    route_after_scope_guard,
    guard_response_node,
    bounded_planning_node,
)
from agent.nodes.finalizer import finalize_answer_node
from agent.observability import instrument_node


def build_agent_graph() -> StateGraph:
    """
    构建 Agent 工作流图
    
    流程:
    START → Intent → Planning → Tool Execution → Synthesize 
          → Reflection → Should Continue?
                           ├─ finalize → Finalize → END
                           └─ retry → Planning (重新规划，最多3轮)
    
    Reflection 重试机制:
    - 质量分数 < 6.0: 触发重试
    - 最多重试 3 轮
    - 每次重试会根据改进建议调整执行计划
    """
    # 创建状态图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("intent_classify", instrument_node("intent_classify", intent_classify_node))
    workflow.add_node("query_decomposer", instrument_node("query_decomposer", query_decomposer_node))
    workflow.add_node("visual_rag_plan", instrument_node("visual_rag_plan", visual_rag_plan_node))
    workflow.add_node("scope_guard", instrument_node("scope_guard", scope_guard_node))
    workflow.add_node("guard_response", instrument_node("guard_response", guard_response_node))
    workflow.add_node("planning", instrument_node("planning", bounded_planning_node))
    workflow.add_node("tool_execution", instrument_node("tool_execution", tool_execution_node))
    workflow.add_node("retrieval_grader", instrument_node("retrieval_grader", retrieval_grader_node))
    workflow.add_node("collect_subquery_result", instrument_node("collect_subquery_result", collect_subquery_result_node))
    workflow.add_node("synthesize", instrument_node("synthesize", synthesize_answer_node))
    workflow.add_node("visual_rag_synthesize", instrument_node("visual_rag_synthesize", visual_rag_synthesize_node))
    workflow.add_node("multi_hop_synthesize", instrument_node("multi_hop_synthesize", multi_hop_synthesize_node))
    workflow.add_node("claim_extractor", instrument_node("claim_extractor", claim_extractor_node))
    workflow.add_node("evidence_validator", instrument_node("evidence_validator", evidence_validator_node))
    workflow.add_node("grounded_rewriter", instrument_node("grounded_rewriter", grounded_rewriter_node))
    workflow.add_node("reflection", instrument_node("reflection", reflection_node))
    workflow.add_node("finalize", instrument_node("finalize", finalize_answer_node))
    
    # 设置入口点
    workflow.set_entry_point("intent_classify")
    
    # 定义边
    workflow.add_conditional_edges(
        "intent_classify",
        route_after_intent,
        {"visual_rag": "visual_rag_plan", "agentic_rag": "scope_guard"},
    )
    workflow.add_conditional_edges(
        "scope_guard",
        route_after_scope_guard,
        {"continue_agentic": "query_decomposer", "guard_finalize": "guard_response"},
    )
    workflow.add_edge("guard_response", "finalize")
    workflow.add_edge("visual_rag_plan", "tool_execution")
    workflow.add_edge("query_decomposer", "planning")
    workflow.add_edge("planning", "tool_execution")
    workflow.add_conditional_edges(
        "tool_execution",
        route_after_tool_execution,
        {"visual_synthesize": "visual_rag_synthesize", "agentic_grade": "retrieval_grader"},
    )
    workflow.add_edge("visual_rag_synthesize", "finalize")
    workflow.add_conditional_edges(
        "retrieval_grader",
        route_after_retrieval_grade,
        {
            "synthesize": "collect_subquery_result",
            "retry_retrieval": "tool_execution",
        },
    )
    workflow.add_conditional_edges(
        "collect_subquery_result",
        route_after_collection,
        {
            "next_subquery": "planning",
            "multi_synthesize": "multi_hop_synthesize",
            "single_synthesize": "synthesize",
        },
    )
    workflow.add_edge("synthesize", "claim_extractor")
    workflow.add_edge("multi_hop_synthesize", "claim_extractor")
    workflow.add_edge("claim_extractor", "evidence_validator")
    workflow.add_conditional_edges(
        "evidence_validator",
        route_after_evidence_validation,
        {"reflection": "reflection", "rewrite": "grounded_rewriter"},
    )
    workflow.add_edge("grounded_rewriter", "claim_extractor")
    
    # 条件分支: 是否继续 Reflection
    workflow.add_conditional_edges(
        "reflection",
        should_continue,
        {
            "finalize": "finalize",
            "retry": "query_decomposer"  # 重置子任务后重新分解与规划
        }
    )
    
    workflow.add_edge("finalize", END)
    
    # 编译
    return workflow.compile()


def save_graph_visualization(graph, output_path: str = "./images/agent_workflow.png"):
    """
    保存工作流可视化图
    
    Args:
        graph: 编译后的 LangGraph
        output_path: 输出路径
    """
    from pathlib import Path
    
    print("\n" + "=" * 70)
    print("📊 生成 Agent 工作流图")
    print("=" * 70)
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        with open(output_file, "wb") as f:
            f.write(png_data)
        print(f"✓ 工作流图已保存: {output_file}")
        return True
    except Exception as e:
        print(f"❌ 生成工作流图失败: {e}")
        print(f"   提示: 需要安装 pygraphviz 或 graphviz")
        return False
    finally:
        print("=" * 70)
