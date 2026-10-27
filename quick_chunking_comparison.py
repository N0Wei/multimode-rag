#!/usr/bin/env python
# coding: utf-8
"""
文档切分方法快速对比脚本（无需测试集标注）
仅评估文档质量和系统性能
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from pdf_parse import DataProcess
from faiss_retriever import FaissRetriever


def analyze_chunks(chunks: List[str], method_name: str) -> Dict[str, Any]:
    """分析文档片段质量"""
    if not chunks:
        return {}
    
    lengths = [len(chunk) for chunk in chunks]
    
    stats = {
        'method': method_name,
        'total_chunks': len(chunks),
        'avg_length': float(np.mean(lengths)),
        'min_length': int(np.min(lengths)),
        'max_length': int(np.max(lengths)),
        'std_length': float(np.std(lengths)),
        'median_length': float(np.median(lengths)),
        'total_chars': sum(lengths),
    }
    
    # 长度分布
    length_bins = {
        '0-100': sum(1 for l in lengths if l <= 100),
        '100-300': sum(1 for l in lengths if 100 < l <= 300),
        '300-500': sum(1 for l in lengths if 300 < l <= 500),
        '500-800': sum(1 for l in lengths if 500 < l <= 800),
        '800+': sum(1 for l in lengths if l > 800),
    }
    stats['length_distribution'] = length_bins
    
    # 示例片段
    stats['sample_chunks'] = chunks[:3]  # 前3个作为示例
    
    return stats


def test_indexing_performance(chunks: List[str]) -> Dict[str, float]:
    """测试索引构建性能"""
    start_time = time.time()
    faiss_retriever = FaissRetriever(data=chunks, force_rebuild=True)
    index_time = time.time() - start_time
    
    # 测试查询速度
    query_times = []
    test_queries = ["续航", "充电", "保养"]
    for query in test_queries:
        start = time.time()
        results = faiss_retriever.GetTopK(query, 5)
        query_times.append(time.time() - start)
    
    return {
        'index_time': index_time,
        'avg_query_time': np.mean(query_times),
        'index_size_estimate': len(chunks) * 768 * 4 / (1024*1024),  # 估算FAISS索引大小(MB)
    }


def main():
    print("=" * 80)
    print("📄 文档切分方法快速对比")
    print("=" * 80)
    
    base_dir = Path(__file__).parent
    pdf_path = base_dir / "data" / "train_a.pdf"
    
    if not pdf_path.exists():
        print(f"❌ 错误: 找不到PDF文件 {pdf_path}")
        return
    
    dp = DataProcess(pdf_path=str(pdf_path))
    
    # 定义要测试的配置
    configs = [
        ('ParseBlock_512', lambda: dp.ParseBlock(max_seq=512)),
        ('ParseBlock_1024', lambda: dp.ParseBlock(max_seq=1024)),
        ('ParseOnePage_512', lambda: dp.ParseOnePageWithRule(max_seq=512)),
        ('ParseOnePage_1024', lambda: dp.ParseOnePageWithRule(max_seq=1024)),
        ('SlidingWindow_512', lambda: dp.ParseAllPage(max_seq=512)),
        ('SlidingWindow_256', lambda: dp.ParseAllPage(max_seq=256)),
    ]
    
    results = {}
    
    for method_name, generate_func in configs:
        print(f"\n{'=' * 80}")
        print(f"🔨 测试: {method_name}")
        print(f"{'=' * 80}")
        
        # 生成文档片段
        dp.data.clear()
        generate_func()
        chunks = dp.data.copy()
        
        if not chunks:
            print(f"  ⚠️  未生成任何片段")
            continue
        
        # 分析质量
        quality_stats = analyze_chunks(chunks, method_name)
        
        print(f"\n📊 文档质量:")
        print(f"  总片段数: {quality_stats['total_chunks']}")
        print(f"  平均长度: {quality_stats['avg_length']:.1f} 字符")
        print(f"  长度范围: [{quality_stats['min_length']}, {quality_stats['max_length']}]")
        print(f"  标准差: {quality_stats['std_length']:.1f}")
        print(f"  总字符数: {quality_stats['total_chars']}")
        
        print(f"\n📈 长度分布:")
        for bin_range, count in quality_stats['length_distribution'].items():
            bar = '█' * (count // 2)
            print(f"  {bin_range:>8}: {count:>4} {bar}")
        
        print(f"\n📝 示例片段:")
        for i, sample in enumerate(quality_stats['sample_chunks'], 1):
            preview = sample[:100] + "..." if len(sample) > 100 else sample
            print(f"  [{i}] ({len(sample)}字) {preview}")
        
        # 测试索引性能
        print(f"\n⚡ 索引性能测试...")
        perf_stats = test_indexing_performance(chunks)
        
        print(f"  索引构建时间: {perf_stats['index_time']:.2f}s")
        print(f"  平均查询时间: {perf_stats['avg_query_time']:.4f}s")
        print(f"  估算索引大小: {perf_stats['index_size_estimate']:.2f} MB")
        
        results[method_name] = {
            'quality': quality_stats,
            'performance': perf_stats,
        }
    
    # 生成对比报告
    print("\n" + "=" * 80)
    print("📋 对比报告")
    print("=" * 80)
    
    print(f"\n{'方法':<25} {'片段数':>6} {'平均长度':>8} {'中位数':>8} {'标准差':>8} {'索引时间':>8} {'查询时间':>10}")
    print("-" * 85)
    
    for method_name, result in results.items():
        q = result['quality']
        p = result['performance']
        print(f"{method_name:<25} {q['total_chunks']:>6} {q['avg_length']:>8.1f} "
              f"{q['median_length']:>8.1f} {q['std_length']:>8.1f} "
              f"{p['index_time']:>8.2f}s {p['avg_query_time']:>10.4f}s")
    
    # 找出最佳配置
    print(f"\n💡 推荐建议:")
    
    # 最少片段数（节省存储）
    min_chunks = min(results.items(), key=lambda x: x[1]['quality']['total_chunks'])
    print(f"  📦 最节省存储: {min_chunks[0]} ({min_chunks[1]['quality']['total_chunks']}个片段)")
    
    # 最快索引
    fastest_index = min(results.items(), key=lambda x: x[1]['performance']['index_time'])
    print(f"  ⚡ 最快索引: {fastest_index[0]} ({fastest_index[1]['performance']['index_time']:.2f}s)")
    
    # 最快查询
    fastest_query = min(results.items(), key=lambda x: x[1]['performance']['avg_query_time'])
    print(f"  🔍 最快查询: {fastest_query[0]} ({fastest_query[1]['performance']['avg_query_time']:.4f}s)")
    
    # 最均匀的长度分布（标准差最小）
    most_uniform = min(results.items(), key=lambda x: x[1]['quality']['std_length'])
    print(f"  📏 最均匀分布: {most_uniform[0]} (标准差={most_uniform[1]['quality']['std_length']:.1f})")
    
    # 保存结果
    output_path = base_dir / "data" / "chunking_quick_comparison.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 详细结果已保存到: {output_path}")
    print("\n" + "=" * 80)
    print("✅ 快速对比完成!")
    print("=" * 80)
    print("\n📌 下一步建议:")
    print("  1. 根据以上数据选择2-3种候选方法")
    print("  2. 人工标注少量测试集(20-30个案例)")
    print("  3. 运行完整的检索评估 (evaluate_chunking_methods.py)")


if __name__ == "__main__":
    main()
