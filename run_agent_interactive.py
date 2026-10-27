#!/usr/bin/env python
# coding: utf-8
"""
智能客服 Agent 系统 - 交互式多轮对话
支持命令行交互，保持对话上下文
"""

import json
import time
from pathlib import Path
from typing import List, Dict

# 现有模块
from config import LLM_BACKEND
from pdf_parse import DataProcess
from faiss_retriever import FaissRetriever
from bm25_retriever import BM25
from rerank_model import reRankLLM

# Agent 模块
from agent.config.model_config import ModelConfig
from agent.tools import BochaSearchMCPTool, DDGSSearchTool, FallbackSearchTool, RAGTool, FAQTool
from agent.graph import build_agent_graph
from agent.state import AgentState
from agent.nodes.tool_executor import set_tools


def build_llm(backend, model_path):
    """
    构建 LLM（统一接口）
    """
    from llm_model import get_llm_model
    return get_llm_model(model_path=model_path, backend=backend)


class InteractiveAgent:
    """交互式 Agent 会话管理"""
    
    def __init__(self, agent_graph):
        self.agent_graph = agent_graph
        self.conversation_history: List[Dict] = []
        self.session_start = time.time()
        
    def chat(self, user_question: str) -> Dict:
        """
        处理单次用户输入
        
        Args:
            user_question: 用户问题
            
        Returns:
            Agent 返回结果
        """
        # 构建初始状态
        initial_state: AgentState = {
            "user_question": user_question,
            "conversation_history": self.conversation_history.copy(),
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
        
        # 运行 Agent
        start_time = time.time()
        final_state = self.agent_graph.invoke(initial_state, config={"recursion_limit": 80})
        elapsed_time = time.time() - start_time
        
        # 更新对话历史
        self.conversation_history.append({
            "role": "user",
            "content": user_question
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": final_state["final_answer"]
        })
        
        # 返回结果
        return {
            "question": user_question,
            "answer": final_state["final_answer"],
            "source": final_state["answer_source"],
            "confidence": final_state["confidence"],
            "quality_score": final_state["quality_score"],
            "need_human": final_state["need_human"],
            "elapsed_time": elapsed_time,
            "metrics": final_state.get("metrics", {}),
            "execution_trace": final_state.get("execution_trace", []),
            "conversation_turns": len(self.conversation_history) // 2
        }
    
    def reset_session(self):
        """重置会话"""
        self.conversation_history = []
        self.session_start = time.time()
        print("\n🔄 会话已重置")
    
    def get_session_info(self):
        """获取会话信息"""
        duration = time.time() - self.session_start
        turns = len(self.conversation_history) // 2
        return {
            "duration_seconds": duration,
            "total_turns": turns,
            "history_length": len(self.conversation_history)
        }


def print_welcome():
    """打印欢迎信息"""
    print("\n" + "=" * 70)
    print("🤖 智能客服 Agent - 交互式多轮对话")
    print("=" * 70)
    print("\n功能特性:")
    print("  ✓ 多轮对话支持 - 自动记忆上下文")
    print("  ✓ 指代消解 - 理解 '它'、'那个' 等代词")
    print("  ✓ FAQ + RAG + 博查搜索 MCP")
    print("  ✓ Reflection 质量评估")
    print("\n命令:")
    print("  /help   - 显示帮助")
    print("  /reset  - 重置会话")
    print("  /info   - 会话信息")
    print("  /quit   - 退出系统")
    print("=" * 70 + "\n")


def print_help():
    """打印帮助信息"""
    print("\n" + "-" * 70)
    print("📖 帮助信息")
    print("-" * 70)
    print("\n可用命令:")
    print("  /help   - 显示此帮助信息")
    print("  /reset  - 重置对话历史，开始新会话")
    print("  /info   - 查看当前会话统计信息")
    print("  /quit   - 退出交互式系统")
    print("\n使用示例:")
    print("  用户: 吉利银河E8的续航是多少？")
    print("  助手: 吉利银河E8在CLTC工况下的纯电续航里程为620公里。")
    print("  用户: 它的价格呢？  ← 自动理解'它'指的是银河E8")
    print("-" * 70 + "\n")


def initialize_system():
    """初始化系统组件"""
    print("\n[系统初始化]")
    print("-" * 70)
    
    start_time = time.time()
    base = "."
    
    # 1. 加载 RAG 组件
    print("[1/7] 📄 解析 PDF 文档...")
    qwen7 = base + "/pre_train_model/Qwen-7B-Chat"
    # 模型路径改为从配置读取
    
    dp = DataProcess(pdf_path=base + "/data/train_a.pdf")
    dp.ParseBlock(max_seq=1024)
    dp.ParseBlock(max_seq=512)
    dp.ParseAllPage(max_seq=256)
    dp.ParseAllPage(max_seq=512)
    dp.ParseOnePageWithRule(max_seq=256)
    dp.ParseOnePageWithRule(max_seq=512)
    data = dp.data
    print(f"  ✓ 共解析 {len(data)} 个文档块")
    
    print("\n[2/7] 🔍 初始化 FAISS 检索器...")
    # 自动从配置读取嵌入模型
    faiss_retriever = FaissRetriever(data=data)
    print("  ✓ FAISS 检索器已就绪")
    
    print("\n[3/7] 📊 初始化 BM25 检索器...")
    bm25_retriever = BM25(data)
    print("  ✓ BM25 检索器已就绪")
    
    print(f"\n[4/7] 🤖 初始化 LLM ({LLM_BACKEND})...")
    llm = build_llm(LLM_BACKEND, qwen7)
    print(f"  ✓ LLM 后端 {LLM_BACKEND} 已就绪")
    
    print("\n[5/7] 🎯 初始化重排序模型...")
    # 自动从配置读取 Rerank 模型
    rerank = reRankLLM()
    print("  ✓ 重排序模型已就绪")
    
    # 2. 初始化 Agent 工具
    print("\n[6/7] 🛠️  初始化 Agent 工具...")
    
    rag_tool = RAGTool(
        faiss_retriever=faiss_retriever,
        bm25_retriever=bm25_retriever,
        rerank_model=rerank,
        llm=llm
    )
    print("  ✓ RAG Tool 已就绪")
    
    faq_tool = FAQTool(faq_path=base + "/data/faq_database.json")
    print("  ✓ FAQ Tool 已就绪")
    
    web_search_tool = FallbackSearchTool(
        BochaSearchMCPTool(),
        DDGSSearchTool(max_results=3),
    )
    print("  ✓ 博查搜索 MCP Tool 已就绪")
    
    # 注入工具到 tool_executor
    set_tools(rag_tool, faq_tool, web_search_tool)
    
    # 3. 构建 LangGraph
    print("\n[7/7] 🔗 构建 Agent 工作流...")
    agent_graph = build_agent_graph()
    print("  ✓ LangGraph 已编译")
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ 系统初始化完成 (耗时: {elapsed_time:.2f}s)")
    print("-" * 70 + "\n")
    
    return agent_graph


def main():
    """主函数 - 交互式循环"""
    
    # 打印欢迎信息
    print_welcome()
    
    # 初始化系统
    try:
        agent_graph = initialize_system()
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        return
    
    # 创建交互式 Agent
    interactive_agent = InteractiveAgent(agent_graph)
    
    # 主循环
    print("💬 开始对话（输入问题或命令）:\n")
    
    while True:
        try:
            # 获取用户输入
            user_input = input("用户: ").strip()
            
            if not user_input:
                continue
            
            # 处理命令
            if user_input.startswith("/"):
                command = user_input.lower()
                
                if command == "/quit" or command == "/exit":
                    print("\n👋 感谢使用，再见！\n")
                    break
                    
                elif command == "/help":
                    print_help()
                    continue
                    
                elif command == "/reset":
                    interactive_agent.reset_session()
                    continue
                    
                elif command == "/info":
                    info = interactive_agent.get_session_info()
                    print(f"\n📊 会话信息:")
                    print(f"  - 对话轮数: {info['total_turns']}")
                    print(f"  - 会话时长: {info['duration_seconds']:.1f}秒")
                    print(f"  - 历史记录: {info['history_length']}条\n")
                    continue
                    
                else:
                    print(f"❌ 未知命令: {user_input}")
                    print("   输入 /help 查看可用命令\n")
                    continue
            
            # 处理用户问题
            print()  # 换行
            result = interactive_agent.chat(user_input)
            
            # 打印答案
            print(f"助手: {result['answer']}")
            
            # 打印元信息（可选，调试用）
            print(f"\n  ℹ️  来源: {result['source']} | "
                  f"置信度: {result['confidence']:.2f} | "
                  f"质量: {result['quality_score']:.1f} | "
                  f"耗时: {result['elapsed_time']:.2f}s")
            
            if result['need_human']:
                print("  ⚠️  建议转人工客服")
            
            print()  # 换行
            
        except KeyboardInterrupt:
            print("\n\n👋 检测到 Ctrl+C，退出系统\n")
            break
            
        except Exception as e:
            print(f"\n❌ 处理出错: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
