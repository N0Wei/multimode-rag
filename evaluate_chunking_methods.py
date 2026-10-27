#!/usr/bin/env python
# coding: utf-8
"""
文档切分方法对比评估脚本
对比3种PDF切分策略的效果
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from pdf_parse import DataProcess
from faiss_retriever import FaissRetriever
from bm25_retriever import BM25
from rerank_model import reRankLLM


class ChunkingEvaluator:
    """文档切分方法评估器"""
    
    def __init__(self, pdf_path: str, eval_cases_path: str):
        """
        Args:
            pdf_path: PDF文件路径
            eval_cases_path: 评估用例JSON路径
        """
        print("=" * 80)
        print("📄 文档切分方法对比评估系统")
        print("=" * 80)
        
        self.pdf_path = pdf_path
        with open(eval_cases_path, 'r', encoding='utf-8') as f:
            self.eval_cases = json.load(f)
        
        print(f"✓ 加载了 {len(self.eval_cases)} 个评估用例\n")
    
    def generate_chunks_method1(self, max_seq: int = 512) -> List[str]:
        """
        方法1: ParseBlock - 基于视觉布局的块提取
        
        Args:
            max_seq: 最大序列长度
        
        Returns:
            文档片段列表
        """
        print(f"  🔨 方法1: ParseBlock (max_seq={max_seq})...")
        dp = DataProcess(pdf_path=self.pdf_path)
        dp.ParseBlock(max_seq=max_seq)
        chunks = dp.data.copy()
        dp.data.clear()  # 清空以便下次使用
        print(f"     ✓ 生成 {len(chunks)} 个文档片段")
        return chunks
    
    def generate_chunks_method2(self, max_seq: int = 512, min_len: int = 6) -> List[str]:
        """
        方法2: ParseOnePageWithRule - 按页切分 + 句号分割
        
        Args:
            max_seq: 最大序列长度
            min_len: 最小长度
        
        Returns:
            文档片段列表
        """
        print(f"  🔨 方法2: ParseOnePageWithRule (max_seq={max_seq}, min_len={min_len})...")
        dp = DataProcess(pdf_path=self.pdf_path)
        dp.ParseOnePageWithRule(max_seq=max_seq, min_len=min_len)
        chunks = dp.data.copy()
        dp.data.clear()
        print(f"     ✓ 生成 {len(chunks)} 个文档片段")
        return chunks
    
    def generate_chunks_method3(self, max_seq: int = 512, min_len: int = 6) -> List[str]:
        """
        方法3: ParseAllPage + SlidingWindow - 全局滑动窗口
        
        Args:
            max_seq: 窗口大小
            min_len: 最小长度
        
        Returns:
            文档片段列表
        """
        print(f"  🔨 方法3: ParseAllPage + SlidingWindow (max_seq={max_seq})...")
        dp = DataProcess(pdf_path=self.pdf_path)
        dp.ParseAllPage(max_seq=max_seq, min_len=min_len)
        chunks = dp.data.copy()
        dp.data.clear()
        print(f"     ✓ 生成 {len(chunks)} 个文档片段")
        return chunks
    
    def analyze_chunk_quality(self, chunks: List[str], method_name: str) -> Dict[str, Any]:
        """
        分析文档片段质量
        
        Returns:
            质量指标字典
        """
        if not chunks:
            return {}
        
        lengths = [len(chunk) for chunk in chunks]
        
        # 计算统计信息
        stats = {
            'method': method_name,
            'total_chunks': len(chunks),
            'avg_length': np.mean(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'std_length': np.std(lengths),
            'median_length': np.median(lengths),
            'total_chars': sum(lengths),
        }
        
        # 计算长度分布
        length_bins = {
            '0-100': sum(1 for l in lengths if l <= 100),
            '100-300': sum(1 for l in lengths if 100 < l <= 300),
            '300-500': sum(1 for l in lengths if 300 < l <= 500),
            '500-800': sum(1 for l in lengths if 500 < l <= 800),
            '800+': sum(1 for l in lengths if l > 800),
        }
        stats['length_distribution'] = length_bins
        
        return stats
    
    def evaluate_chunking_method(
        self, 
        method_name: str, 
        chunks: List[str],
        use_rerank: bool = False,
        skip_retrieval_eval: bool = True
    ) -> Dict[str, Any]:
        """
        评估一种切分方法的检索效果
        
        Args:
            method_name: 方法名称
            chunks: 文档片段列表
            use_rerank: 是否使用重排序
        
        Returns:
            评估结果
        """
        print(f"\n{'=' * 80}")
        print(f"📊 评估: {method_name}")
        print(f"{'=' * 80}")
        
        # 1. 分析文档质量
        quality_stats = self.analyze_chunk_quality(chunks, method_name)
        print(f"\n📈 文档质量分析:")
        print(f"  总片段数: {quality_stats['total_chunks']}")
        print(f"  平均长度: {quality_stats['avg_length']:.1f} 字符")
        print(f"  长度范围: [{quality_stats['min_length']}, {quality_stats['max_length']}]")
        print(f"  标准差: {quality_stats['std_length']:.1f}")
        
        # 2. 构建检索索引
        print(f"\n🔧 构建检索索引...")
        start_time = time.time()
        
        faiss_retriever = FaissRetriever(data=chunks, force_rebuild=True)
        bm25_retriever = BM25(chunks)
        
        if use_rerank:
            rerank_model = reRankLLM()
        
        index_time = time.time() - start_time
        print(f"  ✓ 索引构建完成 (耗时: {index_time:.2f}s)")
        
        # 3. 执行检索评估（可选）
        aggregated = {}
        if not skip_retrieval_eval:
            print(f"\n🔍 执行检索评估...")
            all_metrics = []
            start_eval = time.time()
            
            for i, case in enumerate(self.eval_cases):
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - start_eval
                    avg_time = elapsed / (i + 1)
                    print(f"  进度: {i + 1}/{len(self.eval_cases)} | "
                          f"平均: {avg_time:.2f}s/案例")
                
                query = case['question']
                relevant_ids = case['reference_doc_ids']
                
                # 执行检索 (这里简化为只使用FAISS)
                results = faiss_retriever.GetTopK(query, 10)
                retrieved_ids = []
                for doc, score in results:
                    if hasattr(doc, 'metadata') and 'id' in doc.metadata:
                        retrieved_ids.append(doc.metadata['id'])
                
                # 计算指标
                metrics = self._calculate_metrics(retrieved_ids, relevant_ids)
                metrics['case_id'] = case['case_id']
                all_metrics.append(metrics)
            
            eval_time = time.time() - start_eval
            
            # 聚合结果
            aggregated = self._aggregate_metrics(all_metrics)
            aggregated['eval_time'] = eval_time
            aggregated['avg_query_time'] = eval_time / len(self.eval_cases)
        else:
            print(f"\n⏭️  跳过检索评估（需要重新标注测试集）")
            aggregated['avg_query_time'] = 0
            aggregated['eval_time'] = 0
        
        # 4. 聚合结果
        aggregated['quality_stats'] = quality_stats
        aggregated['index_time'] = index_time
        
        # 打印汇总
        self._print_evaluation_summary(method_name, aggregated)
        
        return aggregated
    
    def _calculate_metrics(self, retrieved_ids: List[int], relevant_ids: List[int]) -> Dict[str, float]:
        """计算单个查询的评估指标"""
        metrics = {}
        
        # Recall@K
        for k in [1, 3, 5, 10]:
            retrieved_at_k = retrieved_ids[:k]
            hits = len(set(retrieved_at_k) & set(relevant_ids))
            recall = hits / len(relevant_ids) if relevant_ids else 0
            metrics[f'Recall@{k}'] = recall
        
        # Top-1 Accuracy
        if retrieved_ids:
            metrics['Top-1 Accuracy'] = 1.0 if retrieved_ids[0] in relevant_ids else 0.0
        else:
            metrics['Top-1 Accuracy'] = 0.0
        
        # MRR
        mrr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                mrr = 1.0 / rank
                break
        metrics['MRR'] = mrr
        
        # NDCG@K
        for k in [1, 3, 5, 10]:
            ndcg = self._calculate_ndcg(retrieved_ids[:k], relevant_ids, k)
            metrics[f'NDCG@{k}'] = ndcg
        
        return metrics
    
    def _calculate_ndcg(self, retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
        """计算 NDCG@K"""
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                dcg += 1.0 / np.log2(i + 2)
        
        ideal_retrieved = relevant_ids[:k]
        idcg = sum(1.0 / np.log2(i + 2) for i in range(len(ideal_retrieved)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, float]:
        """聚合所有查询的指标"""
        aggregated = {}
        
        metric_keys = ['Recall@1', 'Recall@3', 'Recall@5', 'Recall@10',
                       'Top-1 Accuracy', 'MRR', 'NDCG@1', 'NDCG@3', 'NDCG@5', 'NDCG@10']
        
        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                aggregated[key] = np.mean(values)
        
        return aggregated
    
    def _print_evaluation_summary(self, method_name: str, aggregated: Dict):
        """打印评估摘要"""
        print(f"\n{'=' * 80}")
        print(f"📈 {method_name} - 评估结果")
        print(f"{'=' * 80}")
        
        print(f"\n⏱️  性能:")
        print(f"  索引构建时间: {aggregated['index_time']:.2f}s")
        print(f"  评估总时间: {aggregated['eval_time']:.2f}s")
        print(f"  平均每查询: {aggregated['avg_query_time']:.3f}s")
        
        print(f"\n🎯 核心指标:")
        print(f"  Top-1 Accuracy: {aggregated.get('Top-1 Accuracy', 0):.4f}")
        print(f"  MRR:            {aggregated.get('MRR', 0):.4f}")
        
        print(f"\n📊 Recall@K:")
        for k in [1, 3, 5, 10]:
            print(f"  Recall@{k}: {aggregated.get(f'Recall@{k}', 0):.4f}")
        
        print(f"\n📈 NDCG@K:")
        for k in [1, 3, 5, 10]:
            print(f"  NDCG@{k}:   {aggregated.get(f'NDCG@{k}', 0):.4f}")
    
    def compare_all_methods(self, output_path: str = None):
        """
        对比所有切分方法
        
        Args:
            output_path: 输出结果路径
        """
        print("\n" + "=" * 80)
        print("🚀 开始全面对比评估")
        print("=" * 80)
        
        results = {}
        
        # 定义要测试的配置
        configs = [
            ('ParseBlock_512', lambda: self.generate_chunks_method1(max_seq=512)),
            ('ParseBlock_1024', lambda: self.generate_chunks_method1(max_seq=1024)),
            ('ParseOnePage_512', lambda: self.generate_chunks_method2(max_seq=512)),
            ('ParseOnePage_1024', lambda: self.generate_chunks_method2(max_seq=1024)),
            ('SlidingWindow_512', lambda: self.generate_chunks_method3(max_seq=512)),
            ('SlidingWindow_256', lambda: self.generate_chunks_method3(max_seq=256)),
        ]
        
        for method_name, generate_func in configs:
            try:
                chunks = generate_func()
                result = self.evaluate_chunking_method(method_name, chunks)
                results[method_name] = result
            except Exception as e:
                print(f"  ❌ {method_name} 评估失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 生成对比报告
        self._generate_comparison_report(results, output_path)
        
        return results
    
    def _generate_comparison_report(self, results: Dict, output_path: str = None):
        """生成对比报告"""
        print("\n" + "=" * 80)
        print("📋 切分方法对比报告")
        print("=" * 80)
        
        # 表头
        print(f"\n{'方法':<25} {'片段数':>6} {'平均长度':>8} {'Top-1':>8} {'MRR':>8} {'R@10':>8} {'N@10':>8} {'耗时(s)':>8}")
        print("-" * 85)
        
        # 表格内容
        for method_name, result in results.items():
            quality = result['quality_stats']
            top1 = result.get('Top-1 Accuracy', 0)
            mrr = result.get('MRR', 0)
            r10 = result.get('Recall@10', 0)
            n10 = result.get('NDCG@10', 0)
            avg_time = result.get('avg_query_time', 0)
            
            print(f"{method_name:<25} {quality['total_chunks']:>6} {quality['avg_length']:>8.1f} "
                  f"{top1:>8.4f} {mrr:>8.4f} {r10:>8.4f} {n10:>8.4f} {avg_time:>8.3f}")
        
        # 找出最佳方法
        print(f"\n🏆 最佳方法分析:")
        
        best_top1 = max(results.items(), key=lambda x: x[1].get('Top-1 Accuracy', 0))
        best_mrr = max(results.items(), key=lambda x: x[1].get('MRR', 0))
        best_recall = max(results.items(), key=lambda x: x[1].get('Recall@10', 0))
        fastest = min(results.items(), key=lambda x: x[1].get('avg_query_time', float('inf')))
        
        print(f"  Top-1 Accuracy 最高: {best_top1[0]} ({best_top1[1]['Top-1 Accuracy']:.4f})")
        print(f"  MRR 最高:           {best_mrr[0]} ({best_mrr[1]['MRR']:.4f})")
        print(f"  Recall@10 最高:     {best_recall[0]} ({best_recall[1]['Recall@10']:.4f})")
        print(f"  速度最快:           {fastest[0]} ({fastest[1]['avg_query_time']:.3f}s)")
        
        # 保存结果
        if output_path:
            # 简化输出，移除详细数据
            simplified_results = {}
            for method_name, result in results.items():
                simplified_results[method_name] = {
                    'quality_stats': result['quality_stats'],
                    'metrics': {k: v for k, v in result.items() if k != 'quality_stats'},
                }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(simplified_results, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 详细结果已保存到: {output_path}")


def main():
    """主函数"""
    base_dir = Path(__file__).parent
    pdf_path = base_dir / "data" / "train_a.pdf"
    eval_cases_path = base_dir / "data" / "eval_cases.json"
    output_path = base_dir / "data" / "chunking_evaluation_result.json"
    
    # 检查文件
    if not pdf_path.exists():
        print(f"❌ 错误: 找不到PDF文件 {pdf_path}")
        return
    
    if not eval_cases_path.exists():
        print(f"❌ 错误: 找不到评估用例 {eval_cases_path}")
        return
    
    # 创建评估器
    evaluator = ChunkingEvaluator(
        pdf_path=str(pdf_path),
        eval_cases_path=str(eval_cases_path)
    )
    
    # 执行评估
    results = evaluator.compare_all_methods(output_path=str(output_path))
    
    print("\n" + "=" * 80)
    print("✅ 评估完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
