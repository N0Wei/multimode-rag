#!/usr/bin/env python
# coding: utf-8
"""
优化版本：复用现有模块的 LangGraph 实现

这个版本展示了如何在使用 LangGraph 的同时，
复用项目中已有的 faiss_retriever.py 和 bm25_retriever.py

优势：
1. 避免代码重复
2. 保持项目一致性
3. 更易维护
4. 自动生成工作流可视化图片
"""

import json
import time
from typing import TypedDict, List, Dict
from pathlib import Path

# LangChain 1.0 导入
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# LangGraph
from langgraph.graph import StateGraph, END

# 项目模块 - 复用现有的检索器！
from config import LLM_BACKEND
from rerank_model import reRankLLM
from pdf_parse import DataProcess
from faiss_retriever import FaissRetriever  
from bm25_retriever import BM25             


# ============================================================================
# LangGraph 状态定义
# ============================================================================

class RAGState(TypedDict):
    """RAG 工作流的状态类型"""
    question: str
    faiss_docs: List[tuple]
    bm25_docs: List[Document]
    reranked_docs: List[Document]
    contexts: Dict[str, str]
    answers: Dict[str, str]
    final_answer: str
    faiss_min_score: float


# ============================================================================
# LangGraph 节点函数 - 使用现有的检索器类
# ============================================================================

def faiss_retrieve_node(state: RAGState) -> RAGState:
    """FAISS 向量检索节点 - 调用现有的 FaissRetriever"""
    question = state["question"]
    # 使用现有类的 GetTopK 方法（保持原有命名）
    faiss_docs = faiss_retriever.GetTopK(question, 15)
    
    state["faiss_docs"] = faiss_docs
    state["faiss_min_score"] = faiss_docs[0][1] if faiss_docs else float('inf')
    
    print(f"  [FAISS] 检索到 {len(faiss_docs)} 个文档，最小距离: {state['faiss_min_score']:.2f}")
    return state


def bm25_retrieve_node(state: RAGState) -> RAGState:
    """BM25 关键词检索节点 - 调用现有的 BM25"""
    question = state["question"]
    # 使用现有类的 GetBM25TopK 方法（保持原有命名）
    bm25_docs = bm25_retriever.GetBM25TopK(question, 15)
    
    state["bm25_docs"] = bm25_docs
    
    print(f"  [BM25] 检索到 {len(bm25_docs)} 个文档")
    return state


def rerank_node(state: RAGState) -> RAGState:
    """重排序节点"""
    question = state["question"]
    faiss_docs = state["faiss_docs"]
    bm25_docs = state["bm25_docs"]
    
    # 合并候选文档
    items = []
    for doc, score in faiss_docs:
        items.append(doc)
    items.extend(bm25_docs)
    
    # 重排序
    reranked = rerank.predict(question, items)
    state["reranked_docs"] = reranked[:6]
    
    print(f"  [Rerank] 重排序后保留 {len(state['reranked_docs'])} 个文档")
    return state


def build_context_node(state: RAGState) -> RAGState:
    """构建上下文节点"""
    max_length = 4000
    contexts = {}
    
    # 1. FAISS 上下文
    emb_ans = ""
    cnt = 0
    for doc, score in state["faiss_docs"]:
        cnt += 1
        if cnt > 6 or len(emb_ans + doc.page_content) > max_length:
            break
        emb_ans += doc.page_content
    contexts["faiss"] = emb_ans
    
    # 2. BM25 上下文
    bm25_ans = ""
    cnt = 0
    for doc in state["bm25_docs"]:
        cnt += 1
        if cnt > 6 or len(bm25_ans + doc.page_content) > max_length:
            break
        bm25_ans += doc.page_content
    contexts["bm25"] = bm25_ans
    
    # 3. 重排序上下文
    rerank_ans = ""
    for doc in state["reranked_docs"]:
        if len(rerank_ans + doc.page_content) > max_length:
            break
        rerank_ans += doc.page_content
    contexts["rerank"] = rerank_ans
    
    # 4. 合并上下文
    contexts["merged"] = emb_ans + bm25_ans
    
    state["contexts"] = contexts
    
    print(f"  [Context] 构建了 {len(contexts)} 种上下文")
    return state


def generate_answer_node(state: RAGState) -> RAGState:
    """生成答案节点（批量推理）"""
    question = state["question"]
    contexts = state["contexts"]
    
    # 使用 langchain-core 的 PromptTemplate
    prompt_template = PromptTemplate.from_template(
        """基于以下已知信息，简洁和专业的来回答用户的问题。
如果无法从中得到答案，请说 "无答案"，不允许在答案中添加编造成分，答案请使用中文。
已知内容为吉利控股集团汽车销售有限公司的吉利用户手册:
{context}

问题: {question}"""
    )
    
    # 原始顺序：merged, bm25, faiss, rerank
    # 这样 batch_output[0] 对应 merged，batch_output[3] 对应 rerank
    context_order = ["merged", "bm25", "faiss", "rerank"]
    
    # 准备批量输入
    batch_input = []
    
    for ctx_type in context_order:
        ctx_content = contexts.get(ctx_type, "")
        prompt = prompt_template.format(
            context=ctx_content,
            question=question
        )
        batch_input.append(prompt)
    
    # 批量推理
    print(f"  [Generate] 批量推理 {len(batch_input)} 个提示...")
    batch_output = llm.infer(batch_input)
    
    # 保存答案（按固定顺序映射）
    answers = {}
    for ctx_type, answer in zip(context_order, batch_output):
        answers[ctx_type] = answer
    
    state["answers"] = answers
    
    # 选择最终答案（优先使用重排序结果）
    if state["faiss_min_score"] > 500:
        state["final_answer"] = "无答案"
    else:
        state["final_answer"] = answers.get("rerank", answers.get("merged", "无答案"))
    
    print(f"  [Generate] 生成完成")
    return state


# ============================================================================
# LangGraph 工作流构建和可视化
# ============================================================================

def build_rag_workflow() -> StateGraph:
    """构建 RAG 工作流图"""
    
    # 创建状态图
    workflow = StateGraph(RAGState)
    
    # 添加节点
    workflow.add_node("faiss_retrieve", faiss_retrieve_node)
    workflow.add_node("bm25_retrieve", bm25_retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("build_context", build_context_node)
    workflow.add_node("generate", generate_answer_node)
    
    # 定义执行流程
    workflow.set_entry_point("faiss_retrieve")
    workflow.add_edge("faiss_retrieve", "bm25_retrieve")
    workflow.add_edge("bm25_retrieve", "rerank")
    workflow.add_edge("rerank", "build_context")
    workflow.add_edge("build_context", "generate")
    workflow.add_edge("generate", END)
    
    # 编译
    return workflow.compile()


def save_workflow_graph(workflow, output_path: str = "./images/langgraph_workflow.png"):
    """
    保存 LangGraph 工作流图为 PNG 图片
    
    Args:
        workflow: 编译后的工作流
        output_path: 输出图片路径（默认保存到 images 文件夹）
    
    注意：需要安装 pygraphviz 或设置 PIL/matplotlib 后端
    conda环境下推荐conda install -c conda-forge pygraphviz
    """
    print("\n" + "=" * 70)
    print("📊 生成 LangGraph 工作流图")
    print("=" * 70)
    
    # 确保输出目录存在
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:      
        # 使用 draw_png (需要 pygraphviz)
        png_data = workflow.get_graph().draw_png()
        with open(output_file, "wb") as f:
            f.write(png_data)
        print(f"✓ 工作流图已保存: {output_file}")
        print(f"  方法: Graphviz PNG")
        return True
    
    except Exception as e:
        print(f"❌ 生成工作流图失败: {e}")
        return False
    
    finally:
        print("=" * 70)


# ============================================================================
# 全局变量
# ============================================================================

llm = None
faiss_retriever = None  # 类型: FaissRetriever (来自 faiss_retriever.py)
bm25_retriever = None   # 类型: BM25 (来自 bm25_retriever.py)
rerank = None


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数 - 复用现有模块的 LangGraph 实现"""
    global llm, faiss_retriever, bm25_retriever, rerank
    
    print("=" * 70)
    print("RAG 系统 - LangGraph + 复用现有模块")
    print("=" * 70)
    
    start = time.time()
    
    base = "."
    qwen7 = base + "/pre_train_model/Qwen-7B-Chat"
    # 模型路径改为从配置读取
    
    # 1. 解析 PDF
    print("\n[1/5] 📄 解析 PDF 文档...")
    dp = DataProcess(pdf_path=base + "/data/train_a.pdf")
    dp.ParseBlock(max_seq=1024)
    dp.ParseBlock(max_seq=512)
    dp.ParseAllPage(max_seq=256)
    dp.ParseAllPage(max_seq=512)
    dp.ParseOnePageWithRule(max_seq=256)
    dp.ParseOnePageWithRule(max_seq=512)
    data = dp.data
    print(f"  ✓ 共解析 {len(data)} 个文档块")
    
    # 2. 初始化 FAISS - 使用现有的 FaissRetriever 类
    print("\n[2/5] 🔍 初始化 FAISS 检索器...")
    faiss_retriever = FaissRetriever(data=data)  # ✅ 自动从配置读取嵌入模型
    print("  ✓ FAISS 检索器已就绪")
    
    # 3. 初始化 BM25 - 使用现有的 BM25 类
    print("\n[3/5] 📊 初始化 BM25 检索器...")
    bm25_retriever = BM25(data)  # ✅ 来自 bm25_retriever.py
    print("  ✓ BM25 检索器已就绪")
    
    # 4. 初始化 LLM
    print(f"\n[4/5] 🤖 初始化 LLM ({LLM_BACKEND})...")
    from llm_model import get_llm_model
    llm = get_llm_model(model_path=qwen7, backend=LLM_BACKEND)
    print(f"  ✓ LLM 后端 {LLM_BACKEND} 已就绪")
    
    # 5. 初始化 Rerank
    print("\n[5/5] 🎯 初始化重排序模型...")
    rerank = reRankLLM()  # ✅ 自动从配置读取 Rerank 模型
    print("  ✓ 重排序模型已就绪")
    
    # 构建 LangGraph 工作流
    print("\n" + "=" * 70)
    print("构建 LangGraph 工作流")
    print("=" * 70)
    rag_app = build_rag_workflow()
    print("✓ RAG 工作流已构建")
    
    # 生成并保存工作流可视化图
    save_workflow_graph(rag_app, output_path=base + "/images/langgraph_workflow.png")
    
    # 处理测试问题
    print("\n" + "=" * 70)
    print("处理测试问题")
    print("=" * 70)

    with open(base + "/data/test_question.json", "r", encoding='utf-8') as f:
        jdata = json.loads(f.read())

        print(f"\n共 {len(jdata)} 个测试问题\n")
        
        for idx, line in enumerate(jdata):
            print(f"\n问题 {idx + 1}/{len(jdata)}: {line['question']}")
            print("-" * 70)
            
            # 初始化状态
            initial_state: RAGState = {
                "question": line["question"],
                "faiss_docs": [],
                "bm25_docs": [],
                "reranked_docs": [],
                "contexts": {},
                "answers": {},
                "final_answer": "",
                "faiss_min_score": 0.0
            }
            
            # 执行 LangGraph 工作流
            result = rag_app.invoke(initial_state)
            
            # 保存结果
            line["answer_langgraph"] = result["final_answer"]
            line["all_answers"] = result["answers"]
            line["faiss_score"] = str(result["faiss_min_score"])
            
            # 兼容旧格式
            line["answer_1"] = result["answers"].get("merged", "")
            line["answer_2"] = result["answers"].get("bm25", "")
            line["answer_3"] = result["answers"].get("faiss", "")
            line["answer_4"] = result["answers"].get("rerank", "")
            
            print(f"  ✓ 答案: {line['answer_langgraph'][:80]}...")
        
        # 保存结果
        output_file = base + "/data/result-langgraph-refactored.json"
        with open(output_file, "w", encoding='utf-8') as out:
            json.dump(jdata, out, ensure_ascii=False, indent=2)
        
        print("\n" + "=" * 70)
        print(f"✓ 结果已保存: {output_file}")
        
        end = time.time()
        print(f"✓ 总耗时: {(end - start) / 60:.2f} 分钟")
        print("=" * 70)


if __name__ == "__main__":
    main()
