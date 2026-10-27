#!/usr/bin/env python
# coding: utf-8
"""
智能客服 Agent 系统 - 主入口
基于 LangGraph + Reflection 机制
"""

import json
import time
from pathlib import Path

# 现有模块
from config import LLM_BACKEND
from pdf_parse import DataProcess
from faiss_retriever import FaissRetriever
from bm25_retriever import BM25
from rerank_model import reRankLLM

# Agent 模块
from agent.config.model_config import ModelConfig
from agent.tools import BochaSearchMCPTool, DDGSSearchTool, FallbackSearchTool, RAGTool, FAQTool
from agent.graph import build_agent_graph, save_graph_visualization
from agent.state import AgentState
from agent.nodes.tool_executor import set_tools


def build_llm(backend, model_path):
    """
    构建 LLM（统一接口）
    """
    from llm_model import get_llm_model
    return get_llm_model(model_path=model_path, backend=backend)


def main():
    """主函数"""
    print("=" * 70)
    print("智能客服 Agent 系统 - LangGraph + Reflection")
    print("=" * 70)
    
    start_time = time.time()
    base = "."
    
    # ============================================================
    # 1. 加载现有 RAG 组件
    # ============================================================
    print("\n[1/6] 📄 解析 PDF 文档...")
    qwen7 = base + "/pre_train_model/Qwen-7B-Chat"
    # 模型路径不再直接传递，改为通过配置读取
    # 保留本地路径仅用于向后兼容
    
    dp = DataProcess(pdf_path=base + "/data/train_a.pdf")
    dp.ParseBlock(max_seq=1024)
    dp.ParseBlock(max_seq=512)
    dp.ParseAllPage(max_seq=256)
    dp.ParseAllPage(max_seq=512)
    dp.ParseOnePageWithRule(max_seq=256)
    dp.ParseOnePageWithRule(max_seq=512)
    data = dp.data
    print(f"  ✓ 共解析 {len(data)} 个文档块")
    
    print("\n[2/6] 🔍 初始化 FAISS 检索器...")
    # 不传 model_path，自动从配置读取（支持本地/API切换）
    faiss_retriever = FaissRetriever(data=data)
    print("  ✓ FAISS 检索器已就绪")
    
    print("\n[3/6] 📊 初始化 BM25 检索器...")
    bm25_retriever = BM25(data)
    print("  ✓ BM25 检索器已就绪")
    
    print(f"\n[4/6] 🤖 初始化 LLM ({LLM_BACKEND})...")
    llm = build_llm(LLM_BACKEND, qwen7)
    print(f"  ✓ LLM 后端 {LLM_BACKEND} 已就绪")
    
    print("\n[5/6] 🎯 初始化重排序模型...")
    # 不传 model_path，自动从配置读取（支持本地/API切换）
    rerank = reRankLLM()
    print("  ✓ 重排序模型已就绪")
    
    # ============================================================
    # 2. 初始化 Agent 工具
    # ============================================================
    print("\n[6/6] 🛠️  初始化 Agent 工具...")
    
    # RAG 工具
    rag_tool = RAGTool(
        faiss_retriever=faiss_retriever,
        bm25_retriever=bm25_retriever,
        rerank_model=rerank,
        llm=llm
    )
    print("  ✓ RAG 工具已封装")
    
    # FAQ 工具
    faq_tool = FAQTool(faq_path=base + "/data/faq_database.json")
    print("  ✓ FAQ 工具已初始化")
    
    # 博查搜索 MCP 工具
    web_search_tool = FallbackSearchTool(
        BochaSearchMCPTool(max_results=3),
        DDGSSearchTool(max_results=3),
    )
    print("  ✓ 博查搜索 MCP 工具已初始化")
    
    # 设置全局工具
    set_tools(rag_tool, faq_tool, web_search_tool)
    
    # ============================================================
    # 3. 构建 LangGraph 工作流
    # ============================================================
    print("\n" + "=" * 70)
    print("构建 LangGraph 工作流")
    print("=" * 70)
    
    agent_graph = build_agent_graph()
    print("✓ Agent 工作流已构建")
    
    # 保存可视化图
    save_graph_visualization(agent_graph, output_path=base + "/images/agent_workflow.png")
    
    # ============================================================
    # 4. 处理测试问题
    # ============================================================
    print("\n" + "=" * 70)
    print("处理测试问题")
    print("=" * 70)
    
    test_file = base + "/data/test_question.json"
    output_file = base + "/data/result-agent.json"
    
    with open(test_file, "r", encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"\n共 {len(test_data)} 个测试问题\n")
    
    # 只处理前几个问题进行测试
    test_limit = 5  # 可以修改这个数字，或设为 None 处理全部
    test_questions = test_data[:test_limit] if test_limit else test_data
    
    for idx, item in enumerate(test_questions):
        question = item["question"]
        print(f"\n{'=' * 70}")
        print(f"问题 {idx + 1}/{len(test_questions)}: {question}")
        print('-' * 70)
        
        # 初始化状态
        initial_state: AgentState = {
            "user_question": question,
            "conversation_history": [],
            "intent": "",
            "query_rewrite": "",
            "plan": [],
            "rag_result": {},
            "faq_result": None,
            "web_search_result": None,
            "draft_answer": "",
            "reflection_count": 0,
            "quality_score": 0.0,
            "reflection_notes": [],
            "improvement_actions": [],
            "claims": [],
            "evidence_catalog": [],
            "claim_verdicts": [],
            "citation_coverage": 0.0,
            "grounding_status": "",
            "grounding_retry_count": 0,
            "execution_cache": {},
            "execution_budget": {},
            "metrics": {},
            "execution_trace": [],
            "final_answer": "",
            "answer_source": "",
            "confidence": 0.0,
            "need_human": False,
            "metadata": {}
        }
        
        try:
            # 执行 Agent 工作流
            result = agent_graph.invoke(initial_state, config={"recursion_limit": 80})
            
            # 保存结果（转换为 Python 原生类型以支持 JSON 序列化）
            item["agent_answer"] = result["final_answer"]
            item["answer_source"] = result.get("answer_source", "unknown")
            item["confidence"] = float(result.get("confidence", 0.0))
            item["quality_score"] = float(result.get("quality_score", 0.0))
            item["reflection_count"] = int(result.get("reflection_count", 0))
            item["need_human"] = bool(result.get("need_human", False))
            item["metrics"] = result.get("metrics", {})
            item["execution_trace"] = result.get("execution_trace", [])
            
            # 显示结果
            print(f"\n✅ 答案: {result['final_answer'][:200]}...")
            print(f"   来源: {result['answer_source']}")
            print(f"   置信度: {result['confidence']:.2f}")
            
        except Exception as e:
            print(f"\n❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            item["agent_answer"] = f"处理失败: {str(e)}"
            item["answer_source"] = "error"
    
    # ============================================================
    # 5. 保存结果
    # ============================================================
    print("\n" + "=" * 70)
    
    # 保存所有结果（包括未处理的）
    for item in test_data:
        if "agent_answer" not in item:
            item["agent_answer"] = ""
            item["answer_source"] = "not_processed"
    
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    end_time = time.time()
    elapsed = (end_time - start_time) / 60
    
    print(f"✓ 结果已保存: {output_file}")
    print(f"✓ 总耗时: {elapsed:.2f} 分钟")
    print("=" * 70)


if __name__ == "__main__":
    main()
