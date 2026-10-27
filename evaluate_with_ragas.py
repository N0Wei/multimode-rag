#!/usr/bin/env python
# coding: utf-8
"""
基于 RAGAS 框架的 RAG 系统性能评估脚本
评估指标：Faithfulness, Answer Relevance, Context Recall
"""
import os
import json
import sys
import pandas as pd
from datasets import Dataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目核心组件
from pdf_parse import DataProcess
from faiss_retriever import FaissRetriever
from llm_model import get_llm_model
from config import LLM_BACKEND, OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL_NAME

# 设置 RAGAS 需要的 OpenAI 环境变量
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE

def prepare_ragas_dataset(eval_path="./data/eval_cases.json", gold_path="./data/golden_gold.json", top_k=5):
    print("=" * 70)
    print("🚀 正在准备 RAGAS 评估数据集...")
    print("=" * 70)
    
    # 1. 初始化组件
    dp = DataProcess(pdf_path="./data/train_a.pdf")
    dp.ParseBlock(max_seq=512)
    data = dp.data
    
    retriever = FaissRetriever(data=data)
    llm = get_llm_model(backend=LLM_BACKEND)
    
    with open(eval_path, "r", encoding='utf-8') as f:
        eval_cases = json.load(f)
    
    # 加载黄金标准答案
    gold_map = {}
    if os.path.exists(gold_path):
        with open(gold_path, "r", encoding='utf-8') as f:
            gold_list = json.load(f)
            for item in gold_list:
                gold_map[item['question']] = item['answer']
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    # 为了演示，我们只评估前 10 条（RAGAS 打分非常消耗 Token）
    sample_size = min(10, len(eval_cases))
    
    for i in range(sample_size):
        case = eval_cases[i]
        query = case['question']
        
        # 2. 执行检索
        retrieved_docs = [doc for doc, _ in retriever.GetTopK(query, top_k)]
        context_list = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in retrieved_docs]
        
        # 3. 生成答案
        context_str = "\n".join(context_list[:3]) # 只取前3个文档作为生成参考，避免超出上下文
        prompt = f"请根据以下参考信息回答问题：{query}\n\n参考信息：\n{context_str}\n\n回答："
        
        try:
            response = llm.invoke(prompt)
            answer = response.content.strip()
        except Exception as e:
            answer = "API 调用失败"
            
        # 4. 获取标准答案 (必须是字符串)
        gt = gold_map.get(query, "暂无标准答案")
        
        questions.append(query)
        answers.append(answer)
        contexts.append(context_list)
        ground_truths.append(gt)
        
        print(f"  [{i+1}/{sample_size}] 已完成: {query[:30]}...")
        
    # 5. 构建 HuggingFace Dataset
    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    return Dataset.from_dict(dataset_dict)

def run_evaluation():
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall
    from langchain_openai import ChatOpenAI
    
    print("\n📊 开始 RAGAS 自动化评估...")
    print("   (注：此过程需要调用 LLM 进行裁判打分，请耐心等待)")
    
    # 配置裁判模型 (使用与生成相同的 OpenAI 配置)
    evaluator_llm = ChatOpenAI(
        model=OPENAI_MODEL_NAME,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_API_BASE,
        temperature=0
    )
    
    # 准备数据
    dataset = prepare_ragas_dataset()
    
    # 执行评估
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,      # 忠实度：答案是否源自上下文
            answer_relevancy,  # 答案相关性：答案是否解决了问题
            context_recall     # 上下文召回率：检索到的内容是否包含标准答案信息
        ],
        llm=evaluator_llm
    )
    
    # 打印结果
    print("\n" + "=" * 70)
    print("🏆 RAGAS 评估报告")
    print("=" * 70)
    print(result.to_pandas())
    
    # 保存结果
    df = result.to_pandas()
    df.to_csv("./data/ragas_evaluation_result.csv", index=False, encoding='utf-8-sig')
    print(f"\n💾 详细报告已保存至: ./data/ragas_evaluation_result.csv")

if __name__ == "__main__":
    run_evaluation()
