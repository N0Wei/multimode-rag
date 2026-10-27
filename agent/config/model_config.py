#!/usr/bin/env python
# coding: utf-8
"""
多模型配置管理
支持 Ollama、OpenAI、阿里云、DeepSeek 等
"""

import json
import os
from pathlib import Path
from typing import Literal, Optional
from dotenv import load_dotenv

load_dotenv()

# LLM 后端类型
LLMBackend = Literal["ollama", "openai"]


class ModelConfig:
    """统一的模型配置管理"""
    
    # ============================================================
    # 后端选择 (在 .env 中修改)
    # Agent 可以使用独立的后端，如果未设置则使用 LLM_BACKEND
    # ============================================================
    BACKEND: LLMBackend = os.getenv("AGENT_LLM_BACKEND", os.getenv("LLM_BACKEND", "ollama"))
    
    # ============================================================
    # Ollama 本地模型配置
    # ============================================================
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    OLLAMA_MAX_WORKERS = int(os.getenv("OLLAMA_MAX_WORKERS", "1"))
    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "900"))
    
    # ============================================================
    # API 供应商。默认使用阿里云百炼 OpenAI 兼容接口。
    # ============================================================
    PROVIDER = os.getenv("AGENT_LLM_PROVIDER", os.getenv("LLM_PROVIDER", "dashscope")).strip().lower()
    _LEGACY_API_BASE = os.getenv("OPENAI_API_BASE", "").strip()
    _LEGACY_MODEL = os.getenv("OPENAI_MODEL_NAME", "").strip()
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
    DASHSCOPE_API_BASE = os.getenv("DASHSCOPE_API_BASE", "").strip() or (
        _LEGACY_API_BASE if "aliyuncs.com" in _LEGACY_API_BASE
        else "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    DASHSCOPE_MODEL = os.getenv("DASHSCOPE_MODEL_NAME", "").strip() or (
        _LEGACY_MODEL if _LEGACY_MODEL.startswith("qwen") else "qwen-plus"
    )
    if PROVIDER in {"dashscope", "aliyun", "bailian"}:
        OPENAI_API_KEY = DASHSCOPE_API_KEY
        OPENAI_API_BASE = DASHSCOPE_API_BASE
        OPENAI_MODEL_NAME = DASHSCOPE_MODEL
    else:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
        OPENAI_API_BASE = _LEGACY_API_BASE or "https://api.openai.com/v1"
        OPENAI_MODEL_NAME = _LEGACY_MODEL or "gpt-4o-mini"
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
    API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "1"))
    
    # ============================================================
    # Agent 专用模型 (可在 .env 中单独配置)
    # 修改位置: 在 .env 中添加以下变量
    # AGENT_INTENT_MODEL=qwen-plus
    # AGENT_PLANNER_MODEL=qwen-plus
    # AGENT_REFLECTION_MODEL=qwen-max
    # AGENT_SYNTHESIZE_MODEL=qwen-plus
    # ============================================================
    INTENT_MODEL = os.getenv("AGENT_INTENT_MODEL", OPENAI_MODEL_NAME)
    PLANNER_MODEL = os.getenv("AGENT_PLANNER_MODEL", OPENAI_MODEL_NAME)
    REFLECTION_MODEL = os.getenv("AGENT_REFLECTION_MODEL", OPENAI_MODEL_NAME)
    SYNTHESIZE_MODEL = os.getenv("AGENT_SYNTHESIZE_MODEL", OPENAI_MODEL_NAME)
    
    # ============================================================
    # 博查搜索 MCP
    # ============================================================
    BOCHA_API_KEY = os.getenv("BOCHA_API_KEY", "")
    ENABLE_BOCHA_SEARCH = os.getenv("ENABLE_BOCHA_SEARCH", "true").strip().lower() == "true"
    BOCHA_MCP_COMMAND = os.getenv("BOCHA_MCP_COMMAND", "uv")
    BOCHA_MCP_TOOL = os.getenv("BOCHA_MCP_TOOL", "bocha_web_search")
    BOCHA_MCP_TIMEOUT = int(os.getenv("BOCHA_MCP_TIMEOUT", "60"))
    _BOCHA_MCP_DIR = str(
        Path(__file__).resolve().parents[2] / "integrations" / "bocha-search-mcp-master"
    )
    BOCHA_MCP_ARGS = json.loads(os.getenv("BOCHA_MCP_ARGS")) if os.getenv("BOCHA_MCP_ARGS") else [
        "--directory",
        _BOCHA_MCP_DIR,
        "run",
        "bocha-search-mcp",
    ]
    
    # ============================================================
    # LangSmith 配置
    # ============================================================
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "customer-service-agent")
    
    @classmethod
    def get_llm(cls, model_name: Optional[str] = None, temperature: float = 0.0, purpose: str = "general"):
        """
        工厂方法: 根据配置返回对应的 LLM 实例
        
        Args:
            model_name: 模型名称，如果为 None 则使用默认配置
            temperature: 温度参数
            
        Returns:
            LLM 实例 (ChatOllama 或 ChatOpenAI)
        """
        if cls.BACKEND == "ollama":
            from langchain_ollama import ChatOllama
            llm = ChatOllama(
                base_url=cls.OLLAMA_HOST,
                model=model_name or cls.OLLAMA_MODEL,
                temperature=temperature,
            )
        else:
            # OpenAI 兼容接口 (支持阿里云、DeepSeek 等)
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=model_name or cls.OPENAI_MODEL_NAME,
                openai_api_key=cls.OPENAI_API_KEY,
                openai_api_base=cls.OPENAI_API_BASE,
                temperature=temperature,
                timeout=cls.API_TIMEOUT,
                max_retries=cls.API_MAX_RETRIES,
            )
        from agent.observability import InstrumentedLLM
        return InstrumentedLLM(llm, purpose=purpose)
    
    @classmethod
    def get_intent_llm(cls):
        """意图分类模型"""
        return cls.get_llm(cls.INTENT_MODEL, temperature=0.0, purpose="intent")
    
    @classmethod
    def get_planner_llm(cls):
        """规划模型"""
        return cls.get_llm(cls.PLANNER_MODEL, temperature=0.0, purpose="planner")
    
    @classmethod
    def get_reflection_llm(cls):
        """反思评估模型"""
        return cls.get_llm(cls.REFLECTION_MODEL, temperature=0.0, purpose="reflection")
    
    @classmethod
    def get_synthesize_llm(cls):
        """答案合成模型"""
        return cls.get_llm(cls.SYNTHESIZE_MODEL, temperature=0.3, purpose="synthesize")
