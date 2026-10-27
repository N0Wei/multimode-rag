#!/usr/bin/env python
# coding: utf-8
"""
评分模型统一接口
支持本地模型和 API 调用
"""

import os
import numpy as np
from typing import Union, List


# 从环境变量读取配置
SCORING_BACKEND = os.environ.get("SCORING_BACKEND", "local").strip().lower()
SCORING_MODEL_PATH = os.environ.get("SCORING_MODEL_PATH", "./pre_train_model/text2vec-base-chinese")
SCORING_DEVICE = os.environ.get("SCORING_DEVICE", "cuda:0")

# API 配置
SCORING_API_TYPE = os.environ.get("SCORING_API_TYPE", "openai").strip().lower()
SCORING_API_KEY = os.environ.get("SCORING_API_KEY", "")
SCORING_API_BASE = os.environ.get("SCORING_API_BASE", "https://api.openai.com/v1")
SCORING_MODEL_NAME = os.environ.get("SCORING_MODEL_NAME", "text-embedding-3-small")


class ScoringModel:
    """
    评分模型适配器
    支持本地和 API 两种后端
    """
    
    def __init__(self, backend=None):
        """
        初始化评分模型
        
        Args:
            backend: 'local' (本地 text2vec) 或 'api' (OpenAI 等 API)
        """
        self.backend = backend or SCORING_BACKEND
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化底层模型"""
        if self.backend == "local":
            self._initialize_local_model()
        elif self.backend == "api":
            self._initialize_api_model()
        else:
            raise ValueError(f"不支持的后端: {self.backend}，请使用 'local' 或 'api'")
    
    def _initialize_local_model(self):
        """初始化本地 text2vec 模型"""
        try:
            from text2vec import SentenceModel
            print(f"  [Scoring] 加载本地模型: {SCORING_MODEL_PATH}")
            print(f"  [Scoring] 设备: {SCORING_DEVICE}")
            self.model = SentenceModel(
                model_name_or_path=SCORING_MODEL_PATH,
                device=SCORING_DEVICE
            )
        except ImportError:
            raise ImportError(
                "本地模式需要安装 text2vec: pip install text2vec"
            )
    
    def _initialize_api_model(self):
        """初始化 API 嵌入模型"""
        print(f"  [Scoring] 使用 API: {SCORING_API_TYPE}")
        print(f"  [Scoring] 模型: {SCORING_MODEL_NAME}")
        
        if SCORING_API_TYPE == "openai":
            try:
                from langchain_openai import OpenAIEmbeddings
                self.model = OpenAIEmbeddings(
                    model=SCORING_MODEL_NAME,
                    openai_api_key=SCORING_API_KEY,
                    openai_api_base=SCORING_API_BASE
                )
            except ImportError:
                raise ImportError(
                    "OpenAI API 需要安装: pip install langchain-openai"
                )
        
        elif SCORING_API_TYPE == "jina":
            try:
                from langchain_community.embeddings import JinaEmbeddings
                self.model = JinaEmbeddings(
                    jina_api_key=SCORING_API_KEY,
                    model_name=SCORING_MODEL_NAME
                )
            except ImportError:
                raise ImportError(
                    "Jina API 需要安装: pip install langchain-community"
                )
        
        else:
            raise ValueError(f"不支持的 API 类型: {SCORING_API_TYPE}")
    
    def calc_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的语义相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
        
        Returns:
            相似度分数 (0-1)
        """
        if self.backend == "local":
            return self._calc_similarity_local(text1, text2)
        else:
            return self._calc_similarity_api(text1, text2)
    
    def _calc_similarity_local(self, text1: str, text2: str) -> float:
        """使用本地 text2vec 计算相似度（一步完成）"""
        from text2vec import semantic_search
        
        # text2vec 的 semantic_search 直接返回相似度分数
        vec1 = self.model.encode([text1])
        vec2 = self.model.encode(text2)
        
        results = semantic_search(vec1, vec2, top_k=1)
        score = results[0][0]['score']
        
        return score
    
    def _calc_similarity_api(self, text1: str, text2: str) -> float:
        """使用 API 计算相似度（需要手动计算余弦相似度）"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # 生成嵌入向量
        vec1 = np.array(self.model.embed_query(text1))
        vec2 = np.array(self.model.embed_query(text2))
        
        # 手动计算余弦相似度
        similarity = cosine_similarity(
            vec1.reshape(1, -1),
            vec2.reshape(1, -1)
        )[0][0]
        
        # 确保返回值在 [0, 1] 范围内
        # 余弦相似度范围是 [-1, 1]，但嵌入向量通常都是正相关
        # 归一化到 [0, 1]
        normalized_score = (similarity + 1) / 2
        
        return float(normalized_score)


def get_scoring_model(backend=None):
    """
    获取评分模型实例
    
    Args:
        backend: 'local' 或 'api'，默认从环境变量读取
    
    Returns:
        ScoringModel 实例
    """
    return ScoringModel(backend=backend)


# 为了向后兼容，提供简单的函数接口
def calc_semantic_similarity(text1: str, text2: str, backend=None) -> float:
    """
    计算两个文本的语义相似度（便捷函数）
    
    Args:
        text1: 第一个文本
        text2: 第二个文本
        backend: 'local' 或 'api'
    
    Returns:
        相似度分数 (0-1)
    """
    model = get_scoring_model(backend=backend)
    return model.calc_semantic_similarity(text1, text2)


if __name__ == "__main__":
    """测试评分模型"""
    print("=" * 60)
    print("测试评分模型")
    print("=" * 60)
    
    # 测试文本
    text1 = "这款车的续航里程是多少？"
    text2 = "续航达到665公里"
    text3 = "座椅支持加热功能"
    
    print(f"\n当前配置: SCORING_BACKEND = {SCORING_BACKEND}")
    
    try:
        # 初始化模型
        scorer = get_scoring_model()
        
        # 计算相似度
        print(f"\n文本1: {text1}")
        print(f"文本2: {text2}")
        score1 = scorer.calc_semantic_similarity(text1, text2)
        print(f"✅ 相似度: {score1:.4f}")
        
        print(f"\n文本1: {text1}")
        print(f"文本3: {text3}")
        score2 = scorer.calc_semantic_similarity(text1, text3)
        print(f"✅ 相似度: {score2:.4f}")
        
        print("\n" + "=" * 60)
        print("✅ 评分模型测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
