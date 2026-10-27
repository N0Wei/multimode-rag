#!/usr/bin/env python
# coding: utf-8
"""
RAG 性能评估结果可视化脚本
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_evaluation_results(file_path):
    """加载评估结果"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def plot_comprehensive_comparison(results_data):
    """绘制综合对比图"""
    # 准备数据
    strategies = ['单路召回 (FAISS)', '多路召回 (FAISS+BM25)', '重排序多路召回']
    strategy_keys = ['faiss_only', 'multi_recall', 'rerank_multi']
    
    # 提取核心指标
    top1_acc = [results_data['summary'][key]['Top-1 Accuracy'] for key in strategy_keys]
    mrr = [results_data['summary'][key]['MRR'] for key in strategy_keys]
    recall_10 = [results_data['summary'][key]['Recall@10'] for key in strategy_keys]
    ndcg_10 = [results_data['summary'][key]['NDCG@10'] for key in strategy_keys]
    avg_time = [results_data['summary'][key]['avg_time_per_query'] for key in strategy_keys]
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('RAG 性能评估结果对比', fontsize=16, fontweight='bold')
    
    # 1. Top-1 Accuracy 对比
    axes[0, 0].bar(strategies, top1_acc, color=['skyblue', 'lightgreen', 'orange'])
    axes[0, 0].set_title('Top-1 Accuracy 对比')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(top1_acc):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 2. MRR 对比
    axes[0, 1].bar(strategies, mrr, color=['skyblue', 'lightgreen', 'orange'])
    axes[0, 1].set_title('MRR (Mean Reciprocal Rank) 对比')
    axes[0, 1].set_ylabel('MRR')
    axes[0, 1].set_ylim(0, 1)
    for i, v in enumerate(mrr):
        axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 3. Recall@10 对比
    axes[1, 0].bar(strategies, recall_10, color=['skyblue', 'lightgreen', 'orange'])
    axes[1, 0].set_title('Recall@10 对比')
    axes[1, 0].set_ylabel('Recall@10')
    axes[1, 0].set_ylim(0, 1)
    for i, v in enumerate(recall_10):
        axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 4. 性能时间对比
    axes[1, 1].bar(strategies, avg_time, color=['red', 'lightcoral', 'pink'])
    axes[1, 1].set_title('平均查询时间对比')
    axes[1, 1].set_ylabel('时间 (秒)')
    for i, v in enumerate(avg_time):
        axes[1, 1].text(i, v + 0.01, f'{v:.3f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('RAG_性能对比图.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_recall_ndcg_curves(results_data):
    """绘制 Recall 和 NDCG 曲线"""
    strategies = ['单路召回 (FAISS)', '多路召回 (FAISS+BM25)', '重排序多路召回']
    strategy_keys = ['faiss_only', 'multi_recall', 'rerank_multi']
    
    # 准备 K 值和对应指标
    k_values = [1, 3, 5, 10]
    
    # Recall@K
    recall_data = []
    for key in strategy_keys:
        recalls = [results_data['detailed_by_strategy'][key]['metrics'][f'Recall@{k}'] for k in k_values]
        recall_data.append(recalls)
    
    # NDCG@K
    ndcg_data = []
    for key in strategy_keys:
        ndcgs = [results_data['detailed_by_strategy'][key]['metrics'][f'NDCG@{k}'] for k in k_values]
        ndcg_data.append(ndcgs)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Recall 曲线
    for i, (recall, strategy) in enumerate(zip(recall_data, strategies)):
        ax1.plot(k_values, recall, marker='o', label=strategy, linewidth=2)
    ax1.set_xlabel('K 值')
    ax1.set_ylabel('Recall')
    ax1.set_title('Recall@K 曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values)
    
    # NDCG 曲线
    for i, (ndcg, strategy) in enumerate(zip(ndcg_data, strategies)):
        ax2.plot(k_values, ndcg, marker='s', label=strategy, linewidth=2)
    ax2.set_xlabel('K 值')
    ax2.set_ylabel('NDCG')
    ax2.set_title('NDCG@K 曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_values)
    
    plt.tight_layout()
    plt.savefig('RAG_Recall和NDCG曲线图.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_table(results_data):
    """打印汇总表格"""
    print("="*80)
    print("📊 RAG 性能评估结果汇总表")
    print("="*80)
    
    strategies = ['单路召回 (FAISS)', '多路召回 (FAISS+BM25)', '重排序多路召回']
    strategy_keys = ['faiss_only', 'multi_recall', 'rerank_multi']
    
    # 表头
    print(f"{'指标':<20} {'单路召回 (FAISS)':<18} {'多路召回 (FAISS+BM25)':<22} {'重排序多路召回':<18}")
    print("-"*80)
    
    # 核心指标
    metrics = [
        ('Top-1 Accuracy', 'Top-1 Accuracy'),
        ('MRR', 'MRR'), 
        ('Recall@10', 'Recall@10'),
        ('NDCG@10', 'NDCG@10'),
        ('Avg Query Time(s)', 'avg_time_per_query')
    ]
    
    for display_name, key_name in metrics:
        row = f"{display_name:<20}"
        for strategy_key in strategy_keys:
            value = results_data['summary'][strategy_key][key_name]
            if key_name == 'avg_time_per_query':
                formatted_value = f"{value:.3f}s"
            else:
                formatted_value = f"{value:.4f}"
            row += f" {formatted_value:<20}"
        print(row)
    
    print("-"*80)
    
    # 计算改进百分比
    faiss_mrr = results_data['summary']['faiss_only']['MRR']
    multi_mrr = results_data['summary']['multi_recall']['MRR']
    rerank_mrr = results_data['summary']['rerank_multi']['MRR']
    
    faiss_recall10 = results_data['summary']['faiss_only']['Recall@10']
    multi_recall10 = results_data['summary']['multi_recall']['Recall@10']
    rerank_recall10 = results_data['summary']['rerank_multi']['Recall@10']
    
    print("\n📈 相对于单路召回(FAISS)的改进:")
    print(f"  多路召回改进 MRR:     +{((multi_mrr-faiss_mrr)/faiss_mrr)*100:.2f}%")
    print(f"  重排序多路召回改进 MRR: +{((rerank_mrr-faiss_mrr)/faiss_mrr)*100:.2f}%")
    print(f"  多路召回改进 Recall@10: +{((multi_recall10-faiss_recall10)/faiss_recall10)*100:.2f}%")
    print(f"  重排序多路召回改进 Recall@10: +{((rerank_recall10-faiss_recall10)/faiss_recall10)*100:.2f}%")

def main():
    # 加载结果
    results_path = "data/rag_evaluation_result.json"
    results_data = load_evaluation_results(results_path)
    
    # 打印汇总表格
    print_summary_table(results_data)
    
    # 绘制图表
    plot_comprehensive_comparison(results_data)
    plot_recall_ndcg_curves(results_data)
    
    print(f"\n✅ 图表已保存为 'RAG_性能对比图.png' 和 'RAG_Recall和NDCG曲线图.png'")
    print(f"✅ 完整评估报告见 'RAG_性能评估报告.md'")

if __name__ == "__main__":
    main()