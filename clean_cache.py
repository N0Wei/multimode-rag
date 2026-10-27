#!/usr/bin/env python
# coding: utf-8
"""
清理 FAISS 缓存

使用场景：
1. 切换嵌入模型后，清除旧缓存
2. 调试时强制重建索引
3. 释放磁盘空间
"""

import shutil
from pathlib import Path


def clean_vector_cache():
    """清理向量缓存"""
    cache_dir = Path("./vector_cache")
    
    if not cache_dir.exists():
        print("✓ 缓存目录不存在，无需清理")
        return
    
    # 列出所有缓存文件
    cache_files = list(cache_dir.glob("faiss_*"))
    
    if not cache_files:
        print("✓ 缓存目录为空")
        return
    
    print(f"发现 {len(cache_files)} 个缓存文件:")
    for f in cache_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  - {f.name} ({size_mb:.2f} MB)")
    
    # 确认删除
    confirm = input("\n是否删除所有缓存？[y/N] ").strip().lower()
    
    if confirm == 'y':
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        print("\n✓ 缓存已清理")
    else:
        print("\n✗ 取消清理")


if __name__ == "__main__":
    print("=" * 60)
    print("FAISS 缓存清理工具")
    print("=" * 60)
    print()
    
    clean_vector_cache()
