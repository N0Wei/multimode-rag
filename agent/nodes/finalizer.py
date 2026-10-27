#!/usr/bin/env python
# coding: utf-8
"""
最终化节点
输出最终答案
"""

from agent.state import AgentState
from agent.observability import finalize_metrics


def finalize_answer_node(state: AgentState) -> AgentState:
    """
    最终化节点
    
    将 draft_answer 设置为 final_answer
    """
    state["final_answer"] = state["draft_answer"]
    finalize_metrics(state)
    
    print("  [Finalize] 最终答案已生成")
    print(f"  [Finalize] 来源: {state.get('answer_source', 'unknown')}")
    
    return state
