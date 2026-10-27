#!/usr/bin/env python
# coding: utf-8
"""
Rerank 模型统一接口
支持本地模型 (transformers) 和 API 服务 (LangChain)
"""

import os
import torch
from typing import List, Optional

from langchain_core.documents import Document

# LangChain Rerankers (仅用于 API)
try:
    from langchain_community.document_compressors import CohereRerank, JinaRerank
except ImportError:
    CohereRerank = None
    JinaRerank = None

# transformers (用于本地模型)
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

from config import (
    RERANK_BACKEND,
    RERANK_API_PROVIDER,
    RERANK_MODEL_PATH,
    RERANK_API_KEY,
    RERANK_API_BASE,
    RERANK_MODEL_NAME,
    RERANK_TOP_N,
    LLM_DEVICE,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_rerank_model(
    model_path: Optional[str] = None,
    backend: Optional[str] = None,
    provider: Optional[str] = None,
):
    """
    获取 Rerank 模型
    
    根据配置返回相应的 Reranker:
    - backend="local": 本地 transformers 模型
    - backend="api" + provider="cohere": Cohere API (LangChain)
    - backend="api" + provider="dashscope": 阿里云百炼 Rerank API
    - backend="api" + provider="jina": Jina AI API (自定义适配器)
    
    Args:
        model_path: 本地模型路径（可选，如果提供则强制使用本地模型）
        backend: 后端类型 "local" 或 "api"（可选，默认从配置读取）
        provider: API 服务商 "cohere", "jina"（可选，默认从配置读取）
    
    Returns:
        Reranker 实例
    """
    # 如果显式提供 model_path，强制使用本地模型
    if model_path:
        backend = "local"
    
    backend = (backend or RERANK_BACKEND).lower()
    provider = (provider or RERANK_API_PROVIDER).lower()
    
    if backend == "api":
        return _get_api_reranker(provider)
    else:
        return _get_local_reranker(model_path or RERANK_MODEL_PATH)


def _get_local_reranker(model_path: str):
    """获取本地 Reranker (使用 transformers)"""
    if AutoModelForSequenceClassification is None or AutoTokenizer is None:
        raise ImportError(
            "transformers 需要安装: pip install transformers torch"
        )
    
    print(f"  [Rerank] 加载本地模型: {model_path}")
    
    return _LocalBGEReranker(
        model_path=model_path,
        top_n=RERANK_TOP_N,
    )


def _get_api_reranker(provider: str):
    """获取 API Reranker"""
    print(f"  [Rerank] 使用 API: {provider}")
    print(f"  [Rerank] 模型: {RERANK_MODEL_NAME}")
    
    if provider in {"dashscope", "aliyun", "bailian"}:
        return _DashScopeRerankAdapter(
            api_key=RERANK_API_KEY,
            api_base=RERANK_API_BASE,
            model=RERANK_MODEL_NAME,
            top_n=RERANK_TOP_N,
        )

    elif provider == "cohere":
        if CohereRerank is None:
            raise ImportError(
                "CohereRerank 需要安装: pip install langchain-community cohere"
            )
        
        return CohereRerank(
            cohere_api_key=RERANK_API_KEY,
            model=RERANK_MODEL_NAME,
            top_n=RERANK_TOP_N,
        )
    
    elif provider == "jina":
        # Jina 使用自定义实现（不依赖 LangChain）
        return _JinaRerankAdapter(
            api_key=RERANK_API_KEY,
            api_base=RERANK_API_BASE or "https://api.jina.ai",
            model=RERANK_MODEL_NAME,
            top_n=RERANK_TOP_N,
        )
    
    else:
        raise ValueError(f"不支持的 API 服务商: {provider}")


class _LocalBGEReranker:
    """本地 BGE Rerank 模型 (使用 transformers)"""
    
    def __init__(self, model_path: str, top_n: int = 10, max_length: int = 512):
        self.model_path = model_path
        self.top_n = top_n
        self.max_length = max_length
        
        # 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.model.half()
        
        # 设置设备
        self.device = "cuda" if LLM_DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def compress_documents(
        self,
        documents: List[Document],
        query: str,
    ) -> List[Document]:
        """
        压缩（重排序）文档
        
        Args:
            documents: 文档列表
            query: 查询文本
            
        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []
        
        # 准备输入
        pairs = [(query, doc.page_content) for doc in documents]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=self.max_length
        ).to(self.device)
        
        # 推理
        with torch.no_grad():
            scores = self.model(**inputs).logits
        
        # 排序
        scores = scores.detach().cpu().clone().numpy().flatten()
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        # 返回 top_n
        result = [documents[i] for i in sorted_indices[:self.top_n]]
        
        # 释放显存
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        return result


class _JinaRerankAdapter:
    """Jina AI Rerank 适配器"""
    
    def __init__(self, api_key: str, api_base: str, model: str, top_n: int = 10):
        self.api_key = api_key
        self.api_base = api_base.rstrip('/')
        self.model = model
        self.top_n = top_n
    
    def compress_documents(
        self,
        documents: List[Document],
        query: str,
    ) -> List[Document]:
        """
        压缩（重排序）文档
        
        Args:
            documents: 文档列表
            query: 查询文本
            
        Returns:
            重排序后的文档列表
        """
        import requests
        
        doc_texts = [doc.page_content for doc in documents]
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        payload = {
            "model": self.model,
            "query": query,
            "documents": doc_texts,
            "top_n": str(min(self.top_n, len(doc_texts))),  # Jina 需要字符串
            "return_documents": False,
        }
        
        try:
            response = requests.post(
                self.api_base,  # 直接使用配置的完整 URL
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            
            result = response.json()
            ranked_indices = [item["index"] for item in result.get("results", [])]
            
            return [documents[idx] for idx in ranked_indices]
            
        except Exception as e:
            print(f"  [警告] Jina Rerank API 调用失败: {e}，使用原顺序")
            return documents[:self.top_n]


class _DashScopeRerankAdapter:
    """Adapter for DashScope's qwen3-rerank compatible API."""

    def __init__(self, api_key: str, api_base: str, model: str, top_n: int = 10):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.top_n = top_n

    def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
        if not documents:
            return []
        import requests

        payload = {
            "model": self.model,
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "top_n": min(self.top_n, len(documents)),
            "return_documents": False,
        }
        try:
            response = requests.post(
                self.api_base,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            ranked_indices = [item["index"] for item in response.json().get("results", [])]
            return [documents[index] for index in ranked_indices]
        except Exception as exc:
            print(f"  [警告] DashScope Rerank API 调用失败: {exc}，使用原顺序")
            return documents[:self.top_n]


class reRankLLM:
    """
    向后兼容的 Rerank 类
    保持原有的 predict() 接口，内部使用 LangChain Reranker
    """
    
    def __init__(self, model_path: Optional[str] = None, max_length: int = 512):
        """
        Args:
            model_path: 模型路径（可选，如果提供则强制使用本地模型）
            max_length: 最大序列长度（保留参数以兼容旧代码，但不使用）
        """
        self.reranker = get_rerank_model(model_path=model_path)
        self.max_length = max_length  # 保留以兼容旧代码
    
    def predict(self, query: str, docs: List[Document]) -> List[Document]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            docs: 文档列表
            
        Returns:
            按相关性排序的文档列表
        """
        if not docs:
            return []
        
        return self.reranker.compress_documents(
            documents=docs,
            query=query,
        )


if __name__ == "__main__":
    # 测试
    print("测试 Rerank 模型...")
    
    from pdf_parse import DataProcess
    from bm25_retriever import BM25
    
    # 准备测试数据
    dp = DataProcess(pdf_path="./data/train_a.pdf")
    dp.ParseBlock(max_seq=512)
    
    bm25 = BM25(dp.data)
    
    query = "吉利银河E8的续航"
    docs = bm25.GetBM25TopK(query, 10)
    
    # 测试 Rerank
    rerank = reRankLLM()
    reranked_docs = rerank.predict(query, docs)
    
    print(f"\n重排序完成，返回 {len(reranked_docs)} 个文档")
