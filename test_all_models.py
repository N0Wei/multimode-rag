#!/usr/bin/env python
# coding: utf-8
"""
综合测试脚本 - 测试所有模型是否可以成功调用
包括: 嵌入模型、Rerank 模型、LLM 模型、Agent 模型、评分模型
"""

import sys
from langchain_core.documents import Document


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_embedding_model():
    """测试嵌入模型"""
    print_section("1. 测试嵌入模型")
    
    try:
        from config import EMBEDDING_BACKEND, OLLAMA_EMBEDDING_MODEL
        from embedding_model import EmbeddingAdapter
        
        print(f"\n当前配置: EMBEDDING_BACKEND = {EMBEDDING_BACKEND}")
        if EMBEDDING_BACKEND == "ollama":
            print(f"           OLLAMA_EMBEDDING_MODEL = {OLLAMA_EMBEDDING_MODEL}")
        
        # 创建嵌入模型
        adapter = EmbeddingAdapter()
        
        # 测试查询嵌入
        query = "吉利银河E8的续航是多少？"
        print(f"\n测试查询: {query}")
        
        query_embedding = adapter.embed_query(query)
        print(f"✅ 查询嵌入成功! 维度: {len(query_embedding)}")
        
        # 测试文档嵌入
        docs = ["文档1: 续航里程达到665公里", "文档2: 座椅加热功能"]
        doc_embeddings = adapter.embed_documents(docs)
        print(f"✅ 文档嵌入成功! 数量: {len(doc_embeddings)}, 维度: {len(doc_embeddings[0])}")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ 依赖缺失: {e}")
        return False
    except FileNotFoundError as e:
        print(f"\n❌ 模型文件未找到: {e}")
        return False
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rerank_model():
    """测试 Rerank 模型"""
    print_section("2. 测试 Rerank 模型")
    
    try:
        from config import RERANK_BACKEND, RERANK_API_PROVIDER, RERANK_API_BASE
        from rerank_model import reRankLLM
        
        print(f"\n当前配置:")
        print(f"  RERANK_BACKEND = {RERANK_BACKEND}")
        if RERANK_BACKEND == "api":
            print(f"  RERANK_API_PROVIDER = {RERANK_API_PROVIDER}")
            print(f"  RERANK_API_BASE = {RERANK_API_BASE}")
        
        # 创建测试文档
        docs = [
            Document(page_content="吉利银河E8是一款纯电动SUV，续航里程达到665公里"),
            Document(page_content="这款车配备了前排座椅加热和通风功能"),
            Document(page_content="车辆支持语音助手唤醒功能"),
            Document(page_content="后排座椅支持电动调节"),
        ]
        
        query = "座椅加热功能"
        print(f"\n测试查询: {query}")
        print(f"文档数量: {len(docs)}")
        
        # 测试 Rerank
        rerank = reRankLLM()
        result = rerank.predict(query, docs)
        
        print(f"\n✅ 重排序成功! 返回文档数: {len(result)}")
        print(f"\n排序结果 (Top 3):")
        for i, doc in enumerate(result[:3], 1):
            content = doc.page_content[:50] + "..." if len(doc.page_content) > 50 else doc.page_content
            print(f"  {i}. {content}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_model():
    """测试基础 LLM 模型"""
    print_section("3. 测试基础 LLM 模型")
    
    try:
        from config import (
            LLM_BACKEND, 
            OLLAMA_HOST, 
            OLLAMA_MODEL,
            OPENAI_API_BASE,
            OPENAI_MODEL_NAME
        )
        
        print(f"\n当前配置: LLM_BACKEND = {LLM_BACKEND}")
        
        if LLM_BACKEND == "ollama":
            print(f"  OLLAMA_HOST = {OLLAMA_HOST}")
            print(f"  OLLAMA_MODEL = {OLLAMA_MODEL}")
        elif LLM_BACKEND == "openai":
            print(f"  OPENAI_API_BASE = {OPENAI_API_BASE}")
            print(f"  OPENAI_MODEL_NAME = {OPENAI_MODEL_NAME}")
        
        from llm_model import get_llm_model
        
        llm = get_llm_model()
        prompts = ["你好，请简单介绍一下自己"]
        
        print(f"\n测试提示: {prompts[0]}")
        result = llm.infer(prompts)
        
        print(f"\n✅ LLM 调用成功!")
        print(f"响应: {result[0][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_llm():
    """测试 Agent LLM 模型"""
    print_section("4. 测试 Agent LLM 模型")
    
    try:
        from agent.config.model_config import ModelConfig
        
        print(f"\n当前配置:")
        print(f"  AGENT_LLM_BACKEND = {ModelConfig.BACKEND}")
        
        if ModelConfig.BACKEND == "ollama":
            print(f"  OLLAMA_HOST = {ModelConfig.OLLAMA_HOST}")
            print(f"  OLLAMA_MODEL = {ModelConfig.OLLAMA_MODEL}")
        else:
            print(f"  OPENAI_API_BASE = {ModelConfig.OPENAI_API_BASE}")
            print(f"  OPENAI_MODEL_NAME = {ModelConfig.OPENAI_MODEL_NAME}")
        
        print(f"\nAgent 专用模型:")
        print(f"  Intent: {ModelConfig.INTENT_MODEL}")
        print(f"  Planner: {ModelConfig.PLANNER_MODEL}")
        print(f"  Reflection: {ModelConfig.REFLECTION_MODEL}")
        print(f"  Synthesize: {ModelConfig.SYNTHESIZE_MODEL}")
        
        # 测试获取 LLM
        llm = ModelConfig.get_llm()
        print(f"\n✅ Agent LLM 初始化成功!")
        
        # 简单测试
        test_prompt = "你好"
        response = llm.invoke(test_prompt)
        print(f"✅ Agent LLM 调用成功! 响应长度: {len(response.content)} 字符")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_faiss_retriever():
    """测试 FAISS 检索器"""
    print_section("5. 测试 FAISS 检索器 (可选)")
    
    try:
        from faiss_retriever import FaissRetriever
        from pdf_parse import DataProcess
        import os
        
        pdf_path = "./data/train_a.pdf"
        if not os.path.exists(pdf_path):
            print(f"\n⏭️  跳过（PDF 文件不存在: {pdf_path}）")
            return True
        
        print(f"\n解析 PDF: {pdf_path}")
        dp = DataProcess(pdf_path=pdf_path)
        dp.ParseBlock(max_seq=512)
        
        print(f"文档数量: {len(dp.data)}")
        
        # 创建 FAISS 检索器（自动从配置读取嵌入模型）
        print(f"\n初始化 FAISS 检索器...")
        faiss = FaissRetriever(data=dp.data)
        
        # 测试检索
        query = "座椅加热"
        print(f"\n测试查询: {query}")
        results = faiss.GetTopK(query, 3)
        
        print(f"\n✅ FAISS 检索成功! 返回 {len(results)} 个结果")
        print(f"\n检索结果 (Top 3):")
        for i, (doc, score) in enumerate(results, 1):
            content = doc.page_content[:50] + "..." if len(doc.page_content) > 50 else doc.page_content
            print(f"  {i}. (分数: {score:.4f}) {content}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n⏭️  跳过（文件不存在: {e}）")
        return True
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bm25_retriever():
    """测试 BM25 检索器"""
    print_section("6. 测试 BM25 检索器 (可选)")
    
    try:
        from bm25_retriever import BM25
        from pdf_parse import DataProcess
        import os
        
        pdf_path = "./data/train_a.pdf"
        if not os.path.exists(pdf_path):
            print(f"\n⏭️  跳过（PDF 文件不存在: {pdf_path}）")
            return True
        
        print(f"\n解析 PDF: {pdf_path}")
        dp = DataProcess(pdf_path=pdf_path)
        dp.ParseBlock(max_seq=512)
        
        print(f"文档数量: {len(dp.data)}")
        
        # 创建 BM25 检索器
        print(f"\n初始化 BM25 检索器...")
        bm25 = BM25(dp.data)
        
        # 测试检索
        query = "座椅加热"
        print(f"\n测试查询: {query}")
        results = bm25.GetBM25TopK(query, 3)
        
        print(f"\n✅ BM25 检索成功! 返回 {len(results)} 个结果")
        print(f"\n检索结果 (Top 3):")
        for i, doc in enumerate(results, 1):
            content = doc.page_content[:50] + "..." if len(doc.page_content) > 50 else doc.page_content
            print(f"  {i}. {content}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n⏭️  跳过（文件不存在: {e}）")
        return True
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scoring_model():
    """测试评分模型"""
    print_section("7. 测试评分模型")
    
    try:
        from config import SCORING_BACKEND
        from scoring_model import get_scoring_model
        
        print(f"\n当前配置: SCORING_BACKEND = {SCORING_BACKEND}")
        
        # 初始化评分模型
        print(f"\n初始化评分模型...")
        scorer = get_scoring_model()
        
        # 测试用例
        text1 = "这款车的续航里程是多少？"
        text2 = "续航达到665公里"
        
        print(f"\n测试查询:")
        print(f"  文本1: {text1}")
        print(f"  文本2: {text2}")
        
        # 计算相似度
        score = scorer.calc_semantic_similarity(text1, text2)
        
        print(f"\n✅ 相似度计算成功! 分数: {score:.4f}")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ 依赖缺失: {e}")
        print(f"   提示: pip install scikit-learn")
        return False
    except FileNotFoundError as e:
        print(f"\n❌ 模型文件未找到: {e}")
        return False
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "🔍" * 40)
    print("  RAG 系统 - 所有模型综合测试")
    print("🔍" * 40)
    
    results = {
        "嵌入模型": test_embedding_model(),
        "Rerank 模型": test_rerank_model(),
        "基础 LLM": test_llm_model(),
        "Agent LLM": test_agent_llm(),
        "FAISS 检索器": test_faiss_retriever(),
        "BM25 检索器": test_bm25_retriever(),
        "评分模型": test_scoring_model(),
    }
    
    # 汇总结果
    print_section("测试结果汇总")
    
    print("\n")
    success_count = 0
    fail_count = 0
    
    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name:15} {status}")
        if result:
            success_count += 1
        else:
            fail_count += 1
    
    print("\n" + "-" * 80)
    print(f"  总计: {len(results)} 项测试")
    print(f"  成功: {success_count} 项")
    print(f"  失败: {fail_count} 项")
    print("-" * 80)
    
    if fail_count > 0:
        print("\n⚠️  部分测试失败，请检查配置和依赖")
        sys.exit(1)
    else:
        print("\n🎉 所有测试通过！系统配置正常")
        print("\n📝 提示:")
        print("  - 配置文件: .env")
        print("  - 嵌入模型配置: EMBEDDING_BACKEND, EMBEDDING_MODEL_PATH")
        print("  - Rerank 配置: RERANK_BACKEND, RERANK_API_BASE")
        print("  - LLM 配置: LLM_BACKEND, OLLAMA_HOST")
        print("  - Agent 配置: AGENT_LLM_BACKEND, OPENAI_API_BASE")
        print("  - 评分模型配置: SCORING_BACKEND, SCORING_MODEL_PATH")
        print("\n详细配置请参考: MODEL_CONFIG_GUIDE.md, SCORING_CONFIG_GUIDE.md\n")


if __name__ == "__main__":
    main()
