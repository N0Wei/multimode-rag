from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

try:
    import ollama
except ImportError as exc:  # pragma: no cover - explicit guidance for missing deps
    raise ImportError(
        "The ollama package is required when LLM_BACKEND=ollama. Install it via `pip install ollama`."
    ) from exc

from config import (
    OLLAMA_HOST,
    OLLAMA_MAX_WORKERS,
    OLLAMA_MODEL,
    OLLAMA_OPTIONS,
    OLLAMA_TIMEOUT,
)


class ChatLLM:
    """Ollama-backed adapter exposing the same infer interface as the vLLM wrapper."""

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        max_workers: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.model = model or OLLAMA_MODEL
        self.host = host or OLLAMA_HOST
        self.options: Dict[str, Any] = options or OLLAMA_OPTIONS or {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_ctx": 4096,
        }
        self.max_workers = max_workers or OLLAMA_MAX_WORKERS
        self.timeout = timeout or OLLAMA_TIMEOUT
        self.client = ollama.Client(host=self.host)

    def _generate_one(self, prompt: str) -> str:
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            options=self.options,
            stream=False,
        )
        return response.get("response", "")

    def infer(self, prompts: List[str]) -> List[str]:
        """
        并发批量推理，保证输出顺序与输入顺序严格一致
        
        使用 executor.map() 而不是 as_completed()，这样可以:
        1. 并发执行提高性能
        2. 保证输出顺序与输入顺序一致
        """
        if not prompts:
            return []

        workers = max(1, min(self.max_workers, len(prompts)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # executor.map() 保证返回顺序与输入顺序一致
            # 即使任务按不同顺序完成，结果也会按提交顺序返回
            try:
                outputs = list(executor.map(
                    self._generate_one, 
                    prompts,
                    timeout=self.timeout
                ))
            except Exception as e:
                # 如果整体超时，回退到逐个处理
                print(f"⚠️  批量推理超时，回退到逐个处理: {e}")
                outputs = []
                for prompt in prompts:
                    try:
                        outputs.append(self._generate_one(prompt))
                    except Exception:
                        outputs.append("")
        
        return outputs

    def stream(self, prompt: str):
        """Yield response fragments from Ollama's streaming generate endpoint."""
        for chunk in self.client.generate(
            model=self.model,
            prompt=prompt,
            options=self.options,
            stream=True,
        ):
            text = (chunk.get("response", "") if isinstance(chunk, dict)
                    else getattr(chunk, "response", ""))
            if text:
                yield text
