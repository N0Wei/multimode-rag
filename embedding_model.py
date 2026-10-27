#!/usr/bin/env python
# coding: utf-8
"""
嵌入模型统一接口
支持本地模型和 API 调用
"""

from typing import List
import os

from config import (
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL_PATH,
    EMBEDDING_API_TYPE,
    EMBEDDING_API_KEY,
    EMBEDDING_API_BASE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_API_BATCH_SIZE,
    EMBEDDING_DEVICE,
    API_TIMEOUT,
    API_MAX_RETRIES,
    OLLAMA_EMBEDDING_BASE_URL,
    OLLAMA_EMBEDDING_MODEL
)


def get_embedding_model():
    """
    获取嵌入模型
    
    根据 EMBEDDING_BACKEND 配置返回相应的模型:
    - "local": 本地 HuggingFace 模型
    - "api": OpenAI 兼容的 API
    - "ollama": Ollama 本地服务
    
    Returns:
        LangChain Embeddings 对象
    """
    backend = EMBEDDING_BACKEND.lower()
    
    if backend == "api":
        return _get_api_embeddings()
    elif backend == "ollama":
        return _get_ollama_embeddings()
    else:
        # 默认使用本地模型
        return _get_local_embeddings()


def _get_local_embeddings():
    """获取本地嵌入模型"""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    
    print(f"  [Embedding] 加载本地模型: {EMBEDDING_MODEL_PATH}")
    
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={"device": EMBEDDING_DEVICE}
    )


def _get_api_embeddings():
    """获取 API 嵌入模型"""
    api_type = EMBEDDING_API_TYPE.lower()
    
    if api_type == "jina":
        return _get_jina_embeddings()
    else:
        # 默认使用 OpenAI
        return _get_openai_embeddings()


def _get_openai_embeddings():
    """获取 OpenAI 嵌入模型"""
    from langchain_openai import OpenAIEmbeddings
    
    print(f"  [Embedding] 使用 OpenAI API: {EMBEDDING_API_BASE}")
    print(f"  [Embedding] 模型: {EMBEDDING_MODEL_NAME}")
    
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        openai_api_key=EMBEDDING_API_KEY,
        openai_api_base=EMBEDDING_API_BASE,
        # DashScope's OpenAI-compatible endpoint accepts strings, not the
        # token-id arrays produced by LangChain's length-safe tokenizer.
        check_embedding_ctx_length=False,
        chunk_size=EMBEDDING_API_BATCH_SIZE,
        timeout=API_TIMEOUT,
        max_retries=API_MAX_RETRIES,
    )


def _get_jina_embeddings():
    """获取 Jina AI 嵌入模型"""
    from langchain_community.embeddings import JinaEmbeddings
    
    print(f"  [Embedding] 使用 Jina AI: {EMBEDDING_API_BASE}")
    print(f"  [Embedding] 模型: {EMBEDDING_MODEL_NAME}")
    
    return JinaEmbeddings(
        jina_api_key=EMBEDDING_API_KEY,
        model_name=EMBEDDING_MODEL_NAME
    )


def _get_ollama_embeddings():
    """获取 Ollama 嵌入模型"""
    from langchain_ollama import OllamaEmbeddings
    
    print(f"  [Embedding] 使用 Ollama: {OLLAMA_EMBEDDING_BASE_URL}")
    print(f"  [Embedding] 模型: {OLLAMA_EMBEDDING_MODEL}")
    
    return OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_EMBEDDING_BASE_URL
    )



class EmbeddingAdapter:
    """
    嵌入模型适配器
    提供统一接口，支持批量嵌入
    """
    
    def __init__(self, embedding_model=None):
        """
        Args:
            embedding_model: LangChain Embeddings 对象
                           如果为 None，自动从配置加载
        """
        self.embedding_model = embedding_model or get_embedding_model()
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        return self.embedding_model.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多个文档"""
        return self.embedding_model.embed_documents(texts)


if __name__ == "__main__":
    # 测试
    print("测试嵌入模型...")
    
    adapter = EmbeddingAdapter()
    
    # 测试查询嵌入
    query = "吉利银河E8的续航是多少？"
    query_embedding = adapter.embed_query(query)
    print(f"\n查询嵌入维度: {len(query_embedding)}")
    
    # 测试文档嵌入
    docs = ["文档1", "文档2"]
    doc_embeddings = adapter.embed_documents(docs)
    print(f"文档嵌入数量: {len(doc_embeddings)}")
    print(f"每个文档嵌入维度: {len(doc_embeddings[0])}")
