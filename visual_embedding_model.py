#!/usr/bin/env python
# coding: utf-8
"""CLIP visual embeddings for optional image-vector retrieval."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from langchain_core.embeddings import Embeddings

from config import (
    VISUAL_EMBEDDING_API_BASE,
    VISUAL_EMBEDDING_API_INPUT_FIELD,
    VISUAL_EMBEDDING_API_KEY,
    VISUAL_EMBEDDING_API_MODEL,
    VISUAL_EMBEDDING_API_STYLE,
    VISUAL_EMBEDDING_API_TIMEOUT,
    VISUAL_EMBEDDING_BACKEND,
    VISUAL_EMBEDDING_DEVICE,
    VISUAL_EMBEDDING_MODEL_PATH,
)


class CLIPVisualEmbeddings(Embeddings):
    """LangChain-compatible CLIP embedding wrapper.

    Codex-MM-ADD: This class is the new visual retrieval adapter. LangChain's
    FAISS store calls ``embed_documents`` when building an index and
    ``embed_query`` at search time, so we map document strings to image paths
    and query strings to CLIP text embeddings.
    """

    def __init__(self, model_path: str | None = None, device: str | None = None):
        from transformers import AutoModel, AutoProcessor

        self.model_path = model_path or VISUAL_EMBEDDING_MODEL_PATH
        self.device = device or VISUAL_EMBEDDING_DEVICE
        print(f"  [VisualEmbedding] 加载 CLIP 模型: {self.model_path}")
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
        self.model.eval()

    def embed_query(self, text: str) -> List[float]:
        """Embed a text query into CLIP's shared image/text space."""
        import torch

        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            embedding = self.model.get_text_features(**inputs)
        return self._normalize(embedding.detach().cpu().numpy())[0].tolist()

    def __call__(self, text: str) -> List[float]:
        """Compatibility for older LangChain FAISS call paths."""
        return self.embed_query(text)

    def embed_documents(self, image_paths: List[str]) -> List[List[float]]:
        """Embed image paths into CLIP's visual space."""
        import torch

        embeddings: List[List[float]] = []
        for image_path in image_paths:
            path = Path(image_path)
            with Image.open(path) as image:
                image = image.convert("RGB")
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embedding = self.model.get_image_features(**inputs)
            embeddings.append(self._normalize(embedding.detach().cpu().numpy())[0].tolist())
        return embeddings

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return values / norms


class APIVisualEmbeddings(Embeddings):
    """OpenAI/Jina-style multimodal embedding API wrapper.

    Codex-MM-ADD: API 版视觉检索适配器。文本查询和图片 data URL 必须发给
    同一个多模态 embedding 模型，确保两者处在同一向量空间。
    """

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        input_field: str | None = None,
        api_style: str | None = None,
        timeout: int | None = None,
    ):
        self.api_base = (api_base or VISUAL_EMBEDDING_API_BASE).rstrip("/")
        self.api_key = api_key if api_key is not None else VISUAL_EMBEDDING_API_KEY
        self.model = model or VISUAL_EMBEDDING_API_MODEL
        self.input_field = input_field or VISUAL_EMBEDDING_API_INPUT_FIELD
        self.api_style = self._resolve_api_style(api_style or VISUAL_EMBEDDING_API_STYLE, self.api_base)
        self.timeout = timeout or VISUAL_EMBEDDING_API_TIMEOUT
        if not self.api_base:
            raise ValueError("VISUAL_EMBEDDING_API_BASE is required when VISUAL_EMBEDDING_BACKEND=api")
        if not self.model:
            raise ValueError("VISUAL_EMBEDDING_API_MODEL is required when VISUAL_EMBEDDING_BACKEND=api")
        self._validate_api_key()
        print(f"  [VisualEmbedding] 使用多模态 API: {self.api_base}")
        print(f"  [VisualEmbedding] 模型: {self.model}")

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)

    def __call__(self, text: str) -> List[float]:
        """Compatibility for older LangChain FAISS call paths."""
        return self.embed_query(text)

    def embed_documents(self, image_paths: List[str]) -> List[List[float]]:
        return [self._embed_one(self._image_to_data_url(path)) for path in image_paths]

    def _embed_one(self, value: str) -> List[float]:
        import requests

        payload = self._build_payload(value)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            self.api_base,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        embedding = self._extract_embedding(response.json())
        return CLIPVisualEmbeddings._normalize(np.array([embedding], dtype="float32"))[0].tolist()

    def _build_payload(self, value: str) -> Dict[str, Any]:
        if self.api_style == "dashscope":
            content_key = "image" if value.startswith("data:image/") else "text"
            return {
                "model": self.model,
                "input": {
                    "contents": [
                        {
                            content_key: value,
                        }
                    ]
                },
            }
        return {"model": self.model, self.input_field: value}

    @staticmethod
    def _extract_embedding(payload: Dict[str, Any]) -> List[float]:
        output = payload.get("output")
        if isinstance(output, dict):
            embeddings = output.get("embeddings")
            if isinstance(embeddings, list) and embeddings:
                first = embeddings[0]
                if isinstance(first, dict) and isinstance(first.get("embedding"), list):
                    return first["embedding"]
                if isinstance(first, list):
                    return first
        if isinstance(payload.get("data"), list) and payload["data"]:
            first = payload["data"][0]
            if isinstance(first, dict) and isinstance(first.get("embedding"), list):
                return first["embedding"]
        if isinstance(payload.get("embedding"), list):
            return payload["embedding"]
        if isinstance(payload.get("embeddings"), list) and payload["embeddings"]:
            first = payload["embeddings"][0]
            if isinstance(first, list):
                return first
            if isinstance(first, dict) and isinstance(first.get("embedding"), list):
                return first["embedding"]
        raise ValueError("Visual embedding API response does not contain an embedding vector")

    @staticmethod
    def _resolve_api_style(style: str, api_base: str) -> str:
        if style and style != "auto":
            return style
        if "dashscope.aliyuncs.com" in api_base:
            return "dashscope"
        return "openai"

    def _validate_api_key(self) -> None:
        if not self.api_key:
            raise ValueError("VISUAL_EMBEDDING_API_KEY is required when VISUAL_EMBEDDING_BACKEND=api")
        try:
            self.api_key.encode("latin-1")
        except UnicodeEncodeError as exc:
            raise ValueError(
                "VISUAL_EMBEDDING_API_KEY contains non-ASCII characters. "
                "Please replace the Chinese placeholder with your real DashScope API key."
            ) from exc

    @staticmethod
    def _image_to_data_url(image_path: str) -> str:
        path = Path(image_path)
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"


def get_visual_embedding_model():
    """Load the optional visual embedding provider."""
    if VISUAL_EMBEDDING_BACKEND == "api":
        return APIVisualEmbeddings()
    return CLIPVisualEmbeddings()
