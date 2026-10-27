#!/usr/bin/env python
# coding: utf-8
"""
RAG 性能评估脚本
对比三种检索策略:
1. 单路召回 (仅 FAISS)
2. 多路召回 (FAISS + BM25 合并)
3. 重排序多路召回 (FAISS + BM25 + Rerank)

评估指标:
- Recall@K (命中率)
- MRR (平均倒数排名)
- NDCG@K (归一化折损累计增益)
- Top-1 Accuracy
"""

import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from faiss_retriever import FaissRetriever
from bm25_retriever import BM25
from rerank_model import reRankLLM
from langchain_core.documents import Document


class RAGEvaluator:
    """RAG 性能评估器"""
    
    def __init__(self, documents_path: str, eval_cases_path: str):
        """
        Args:
            documents_path: 文档JSON文件路径
            eval_cases_path: 评估用例JSON文件路径
        """
        print("=" * 80)
        print("🚀 初始化 RAG 评估系统")
        print("=" * 80)
        
        # 加载数据
        with open(documents_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        
        with open(eval_cases_path, 'r', encoding='utf-8') as f:
            self.eval_cases = json.load(f)
        
        print(f"✓ 加载了 {len(self.documents)} 个文档")
        print(f"✓ 加载了 {len(self.eval_cases)} 个评估用例")
        
        # 构建文档ID映射
        self.doc_id_to_content = {doc['doc_id']: doc['content'] for doc in self.documents}
        
        # 初始化检索组件
        print("\n📦 初始化检索组件...")
        self._init_retrievers()
    
    def _init_retrievers(self):
        """初始化各种检索器"""
        # 准备文档列表 (用于FAISS和BM25)
        doc_texts = [doc['content'] for doc in self.documents]
        
        # 1. FAISS 检索器
        print("  [1/3] 初始化 FAISS 检索器...")
        self.faiss_retriever = FaissRetriever(data=doc_texts, force_rebuild=False)
        
        # 2. BM25 检索器
        print("  [2/3] 初始化 BM25 检索器...")
        self.bm25_retriever = BM25(doc_texts)
        
        # 3. Rerank 模型
        print("  [3/3] 初始化 Rerank 模型...")
        self.rerank_model = reRankLLM()
        
        print("✓ 所有检索组件初始化完成\n")
    
    def search_faiss_only(self, query: str, top_k: int = 10) -> List[int]:
        """
        策略1: 单路召回 - 仅使用 FAISS
        
        Returns:
            返回文档ID列表 (按相关性排序)
        """
        results = self.faiss_retriever.GetTopK(query, top_k)
        doc_ids = []
        for doc, score in results:
            # 从metadata中提取原始文档ID
            if hasattr(doc, 'metadata') and 'id' in doc.metadata:
                doc_ids.append(doc.metadata['id'])
        return doc_ids
    
    def search_multi_recall(self, query: str, top_k: int = 10) -> List[int]:
        """
        策略2: 多路召回 - FAISS + BM25 合并 (交替融合)
        
        使用交替融合策略: 从两个列表中轮流取文档
        """
        # 获取两路结果 (各取top_k*2以保证融合后足够)
        faiss_results = self.faiss_retriever.GetTopK(query, top_k * 2)
        bm25_results = self.bm25_retriever.GetBM25TopK(query, top_k * 2)
        
        # 提取文档ID
        faiss_ids = []
        for doc, score in faiss_results:
            if hasattr(doc, 'metadata') and 'id' in doc.metadata:
                faiss_ids.append(doc.metadata['id'])
        
        bm25_ids = [doc.metadata['id'] for doc in bm25_results if hasattr(doc, 'metadata') and 'id' in doc.metadata]
        
        # 交替融合 (Round-Robin)
        merged_ids = []
        seen_ids = set()
        
        max_len = max(len(faiss_ids), len(bm25_ids))
        for i in range(max_len):
            # 先取FAISS
            if i < len(faiss_ids) and faiss_ids[i] not in seen_ids:
                merged_ids.append(faiss_ids[i])
                seen_ids.add(faiss_ids[i])
            
            # 再取BM25
            if i < len(bm25_ids) and bm25_ids[i] not in seen_ids:
                merged_ids.append(bm25_ids[i])
                seen_ids.add(bm25_ids[i])
            
            if len(merged_ids) >= top_k:
                break
        
        return merged_ids[:top_k]
    
    def search_rerank_multi_recall(self, query: str, top_k: int = 10) -> List[int]:
        """
        策略3: 重排序多路召回 - FAISS + BM25 + Rerank
        
        先用两路召回粗排,再用Rerank精排
        """
        # 1. 多路召回 (获取更多候选)
        faiss_results = self.faiss_retriever.GetTopK(query, 15)
        bm25_results = self.bm25_retriever.GetBM25TopK(query, 15)
        
        # 2. 合并去重
        merged_docs = []
        seen_content = set()
        
        # 添加FAISS结果
        for doc, score in faiss_results:
            if doc.page_content not in seen_content:
                merged_docs.append(doc)
                seen_content.add(doc.page_content)
        
        # 添加BM25结果
        for doc in bm25_results:
            if doc.page_content not in seen_content:
                merged_docs.append(doc)
                seen_content.add(doc.page_content)
        
        # 3. Rerank 重排序
        reranked_docs = self.rerank_model.predict(query, merged_docs)
        
        # 4. 提取top_k的文档ID
        doc_ids = []
        for doc in reranked_docs[:top_k]:
            if hasattr(doc, 'metadata') and 'id' in doc.metadata:
                doc_ids.append(doc.metadata['id'])
        
        return doc_ids
    
    def calculate_metrics(
        self, 
        retrieved_ids: List[int], 
        relevant_ids: List[int],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            retrieved_ids: 检索到的文档ID列表 (按相关性排序)
            relevant_ids: 相关文档ID列表 (ground truth)
            k_values: 需要计算的K值
        
        Returns:
            包含各种指标的字典
        """
        metrics = {}
        
        # 1. Recall@K
        for k in k_values:
            retrieved_at_k = retrieved_ids[:k]
            hits = len(set(retrieved_at_k) & set(relevant_ids))
            recall = hits / len(relevant_ids) if relevant_ids else 0
            metrics[f'Recall@{k}'] = recall
        
        # 2. Top-1 Accuracy
        if retrieved_ids:
            metrics['Top-1 Accuracy'] = 1.0 if retrieved_ids[0] in relevant_ids else 0.0
        else:
            metrics['Top-1 Accuracy'] = 0.0
        
        # 3. MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                mrr = 1.0 / rank
                break
        metrics['MRR'] = mrr
        
        # 4. NDCG@K
        for k in k_values:
            ndcg = self._calculate_ndcg(retrieved_ids[:k], relevant_ids, k)
            metrics[f'NDCG@{k}'] = ndcg
        
        return metrics
    
    def _calculate_ndcg(self, retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
        """计算 NDCG@K"""
        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                rel = 1.0
                dcg += rel / np.log2(i + 2)  # i+2 因为i从0开始
        
        # IDCG (理想情况下的DCG)
        ideal_retrieved = relevant_ids[:k]
        idcg = 0.0
        for i, doc_id in enumerate(ideal_retrieved):
            rel = 1.0
            idcg += rel / np.log2(i + 2)
        
        # NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return ndcg
    
    def evaluate_strategy(
        self, 
        strategy_name: str, 
        search_func,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        评估一个检索策略
        
        Args:
            strategy_name: 策略名称
            search_func: 搜索函数 (query, top_k) -> List[int]
            verbose: 是否打印详细进度
        
        Returns:
            评估结果字典
        """
        print(f"\n{'=' * 80}")
        print(f"📊 评估策略: {strategy_name}")
        print(f"{'=' * 80}")
        
        all_metrics = []
        start_time = time.time()
        
        for i, case in enumerate(self.eval_cases):
            if verbose and (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                print(f"  进度: {i + 1}/{len(self.eval_cases)} | "
                      f"平均耗时: {avg_time:.2f}s/案例 | "
                      f"预计剩余: {avg_time * (len(self.eval_cases) - i - 1):.1f}s")
            
            query = case['question']
            relevant_ids = case['reference_doc_ids']
            
            # 执行检索
            retrieved_ids = search_func(query, top_k=10)
            
            # 计算指标
            metrics = self.calculate_metrics(retrieved_ids, relevant_ids)
            metrics['case_id'] = case['case_id']
            metrics['category'] = case.get('category', 'unknown')
            metrics['retrieved_count'] = len(retrieved_ids)
            metrics['relevant_count'] = len(relevant_ids)
            
            all_metrics.append(metrics)
        
        total_time = time.time() - start_time
        
        # 聚合统计
        aggregated = self._aggregate_metrics(all_metrics)
        aggregated['total_time'] = total_time
        aggregated['avg_time_per_query'] = total_time / len(self.eval_cases)
        aggregated['detailed_results'] = all_metrics
        
        # 打印汇总
        self._print_summary(strategy_name, aggregated)
        
        return aggregated
    
    def _aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, float]:
        """聚合所有案例的指标"""
        aggregated = {}
        
        metric_keys = ['Recall@1', 'Recall@3', 'Recall@5', 'Recall@10', 
                       'Top-1 Accuracy', 'MRR', 'NDCG@1', 'NDCG@3', 'NDCG@5', 'NDCG@10']
        
        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                aggregated[key] = np.mean(values)
        
        # 按类别统计
        categories = {}
        for m in all_metrics:
            cat = m['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(m)
        
        category_stats = {}
        for cat, metrics_list in categories.items():
            cat_agg = {}
            for key in metric_keys:
                values = [m[key] for m in metrics_list if key in m]
                if values:
                    cat_agg[key] = np.mean(values)
            cat_agg['count'] = len(metrics_list)
            category_stats[cat] = cat_agg
        
        aggregated['category_stats'] = category_stats
        
        return aggregated
    
    def _print_summary(self, strategy_name: str, aggregated: Dict):
        """打印评估摘要"""
        print(f"\n{'=' * 80}")
        print(f"📈 {strategy_name} - 评估结果汇总")
        print(f"{'=' * 80}")
        
        print(f"\n⏱️  性能:")
        print(f"  总耗时: {aggregated['total_time']:.2f}s")
        print(f"  平均每查询: {aggregated['avg_time_per_query']:.3f}s")
        
        print(f"\n🎯 核心指标:")
        print(f"  Top-1 Accuracy: {aggregated.get('Top-1 Accuracy', 0):.4f}")
        print(f"  MRR:            {aggregated.get('MRR', 0):.4f}")
        
        print(f"\n📊 Recall@K:")
        for k in [1, 3, 5, 10]:
            print(f"  Recall@{k}: {aggregated.get(f'Recall@{k}', 0):.4f}")
        
        print(f"\n📈 NDCG@K:")
        for k in [1, 3, 5, 10]:
            print(f"  NDCG@{k}:   {aggregated.get(f'NDCG@{k}', 0):.4f}")
    
    def compare_strategies(self, output_path: str = None):
        """
        对比所有策略并生成报告
        
        Args:
            output_path: 输出报告路径 (可选)
        """
        print("\n" + "=" * 80)
        print("🔬 开始全面对比评估")
        print("=" * 80)
        
        results = {}
        
        # 策略1: 单路召回 (FAISS)
        results['faiss_only'] = self.evaluate_strategy(
            "策略1: 单路召回 (FAISS)",
            self.search_faiss_only
        )
        
        # 策略2: 多路召回 (FAISS + BM25)
        results['multi_recall'] = self.evaluate_strategy(
            "策略2: 多路召回 (FAISS + BM25)",
            self.search_multi_recall
        )
        
        # 策略3: 重排序多路召回
        results['rerank_multi'] = self.evaluate_strategy(
            "策略3: 重排序多路召回 (FAISS + BM25 + Rerank)",
            self.search_rerank_multi_recall
        )
        
        # 生成对比报告
        self._generate_comparison_report(results, output_path)
        
        return results
    
    def _generate_comparison_report(self, results: Dict, output_path: str = None):
        """生成对比报告"""
        print("\n" + "=" * 80)
        print("📋 策略对比报告")
        print("=" * 80)
        
        strategies = {
            'faiss_only': '单路召回 (FAISS)',
            'multi_recall': '多路召回 (FAISS+BM25)',
            'rerank_multi': '重排序多路召回'
        }
        
        metric_keys = ['Top-1 Accuracy', 'MRR', 'Recall@1', 'Recall@3', 'Recall@5', 'Recall@10',
                       'NDCG@1', 'NDCG@3', 'NDCG@5', 'NDCG@10']
        
        # 表头
        print(f"\n{'指标':<20}", end='')
        for key, name in strategies.items():
            print(f"{name:<30}", end='')
        print()
        print("-" * 110)
        
        # 表格内容
        for metric in metric_keys:
            print(f"{metric:<20}", end='')
            for key in strategies.keys():
                value = results[key].get(metric, 0)
                print(f"{value:<30.4f}", end='')
            print()
        
        # 性能对比
        print(f"\n{'性能指标':<20}", end='')
        for key in strategies.keys():
            avg_time = results[key].get('avg_time_per_query', 0)
            print(f"{avg_time:<30.3f}s", end='')
        print()
        
        # 找出最佳策略
        print(f"\n🏆 最佳策略分析:")
        for metric in ['Top-1 Accuracy', 'MRR', 'Recall@10', 'NDCG@10']:
            best_key = max(strategies.keys(), key=lambda k: results[k].get(metric, 0))
            best_value = results[best_key][metric]
            print(f"  {metric:<20}: {strategies[best_key]} ({best_value:.4f})")
        
        # 保存详细结果
        if output_path:
            self._save_detailed_results(results, output_path)
    
    def _save_detailed_results(self, results: Dict, output_path: str):
        """保存详细结果到文件"""
        output = {
            'summary': {},
            'detailed_by_strategy': {}
        }
        
        strategies = {
            'faiss_only': '单路召回 (FAISS)',
            'multi_recall': '多路召回 (FAISS+BM25)',
            'rerank_multi': '重排序多路召回'
        }
        
        for key, name in strategies.items():
            result = results[key]
            output['detailed_by_strategy'][key] = {
                'name': name,
                'metrics': {k: v for k, v in result.items() if k != 'detailed_results'},
                'category_stats': result.get('category_stats', {})
            }
            
            output['summary'][key] = {
                'Top-1 Accuracy': result.get('Top-1 Accuracy', 0),
                'MRR': result.get('MRR', 0),
                'Recall@10': result.get('Recall@10', 0),
                'NDCG@10': result.get('NDCG@10', 0),
                'avg_time_per_query': result.get('avg_time_per_query', 0)
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 详细结果已保存到: {output_path}")


def main():
    """主函数"""
    # 配置路径
    base_dir = Path(__file__).parent
    documents_path = base_dir / "data" / "documents.json"
    eval_cases_path = base_dir / "data" / "eval_cases.json"
    output_path = base_dir / "data" / "rag_evaluation_result.json"
    
    # 检查文件是否存在
    if not documents_path.exists():
        print(f"❌ 错误: 找不到文档文件 {documents_path}")
        return
    
    if not eval_cases_path.exists():
        print(f"❌ 错误: 找不到评估用例文件 {eval_cases_path}")
        return
    
    # 创建评估器
    evaluator = RAGEvaluator(
        documents_path=str(documents_path),
        eval_cases_path=str(eval_cases_path)
    )
    
    # 执行评估
    results = evaluator.compare_strategies(output_path=str(output_path))
    
    print("\n" + "=" * 80)
    print("✅ 评估完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
