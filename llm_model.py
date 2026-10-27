#!/usr/bin/env python
# coding: utf-8
"""
基础 LLM 模型统一接口
支持本地模型 (Ollama/vLLM) 和 API 服务 (OpenAI 兼容)
"""

from typing import List, Optional, Dict, Any

from config import (
    LLM_BACKEND,
    OLLAMA_MODEL,
    OLLAMA_HOST,
    OLLAMA_OPTIONS,
    OLLAMA_MAX_WORKERS,
    OLLAMA_TIMEOUT,
    OPENAI_API_KEY,
    OPENAI_API_BASE,
    OPENAI_MODEL_NAME,
    API_TIMEOUT,
    API_MAX_RETRIES,
)


def get_llm_model(model_path: Optional[str] = None, backend: Optional[str] = None):
    """
    获取基础 LLM 模型
    
    根据配置返回相应的 LLM:
    - "ollama": Ollama 本地服务
    - "vllm": vLLM 本地部署
    - "openai": OpenAI 兼容 API
    
    Args:
        model_path: 模型路径（可选，用于 vLLM）
        backend: 后端类型（可选，默认从配置读取）
    
    Returns:
        LLM 实例（统一 infer() 接口）
    """
    backend = (backend or LLM_BACKEND).lower()
    
    if backend == "ollama":
        return _get_ollama_llm()
    elif backend == "openai":
        return _get_openai_llm()
    elif backend == "vllm":
        return _get_vllm_llm(model_path)
    else:
        raise ValueError(f"不支持的 LLM 后端: {backend}")


def _get_ollama_llm():
    """获取 Ollama LLM"""
    from ollama_model import ChatLLM as OllamaChatLLM
    
    print(f"  [LLM] 使用 Ollama: {OLLAMA_HOST}")
    print(f"  [LLM] 模型: {OLLAMA_MODEL}")
    
    return OllamaChatLLM(
        model=OLLAMA_MODEL,
        host=OLLAMA_HOST,
        options=OLLAMA_OPTIONS,
        max_workers=OLLAMA_MAX_WORKERS,
        timeout=OLLAMA_TIMEOUT,
    )


def _get_vllm_llm(model_path: Optional[str]):
    """获取 vLLM LLM"""
    if model_path is None:
        raise ValueError("vLLM 后端需要提供 model_path 参数")
    
    from vllm_model import ChatLLM as VLLMChatLLM
    
    print(f"  [LLM] 使用 vLLM")
    print(f"  [LLM] 模型路径: {model_path}")
    
    return VLLMChatLLM(model_path)


def _get_openai_llm():
    """获取 OpenAI 兼容 LLM"""
    print(f"  [LLM] 使用 OpenAI API: {OPENAI_API_BASE}")
    print(f"  [LLM] 模型: {OPENAI_MODEL_NAME}")
    
    return _OpenAILLMAdapter(
        api_key=OPENAI_API_KEY,
        api_base=OPENAI_API_BASE,
        model=OPENAI_MODEL_NAME,
    )


class _OpenAILLMAdapter:
    """
    OpenAI 兼容 API 适配器
    提供与 Ollama/vLLM 相同的 infer() 接口
    """
    
    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ):
        self.api_key = api_key
        self.api_base = api_base.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 初始化 OpenAI 客户端
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai 包需要安装: pip install openai"
            )
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=API_TIMEOUT,
            max_retries=API_MAX_RETRIES,
        )
    
    def infer(self, prompts: List[str]) -> List[str]:
        """
        批量推理
        
        Args:
            prompts: 提示词列表
            
        Returns:
            生成的文本列表
        """
        if not prompts:
            return []
        
        outputs = []
        for prompt in prompts:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                answer = response.choices[0].message.content
                outputs.append(answer)
                
            except Exception as e:
                print(f"  [Warning] API 调用失败: {e}")
                outputs.append("无答案")
        
        return outputs

    def stream(self, prompt: str):
        """Yield answer text as it arrives from an OpenAI-compatible API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in response:
            if not chunk.choices:
                continue
            text = chunk.choices[0].delta.content
            if isinstance(text, str) and text:
                yield text


# 向后兼容：保留原有的 build_llm 函数
def build_llm(backend: Optional[str], model_path: Optional[str] = None):
    """
    构建 LLM（向后兼容接口）
    
    Args:
        backend: 后端类型
        model_path: 模型路径（用于 vLLM）
    
    Returns:
        LLM 实例
    """
    return get_llm_model(model_path=model_path, backend=backend)


if __name__ == "__main__":
    # 测试
    print("测试 LLM 模型...")
    
    # 测试 OpenAI API
    import config
    original_backend = config.LLM_BACKEND
    config.LLM_BACKEND = "openai"
    
    llm = get_llm_model()
    
    prompts = ["你好，请简单介绍一下自己"]
    outputs = llm.infer(prompts)
    
    print(f"\n输入: {prompts[0]}")
    print(f"输出: {outputs[0]}")
    
    config.LLM_BACKEND = original_backend
