import os

import json
import torch
from dotenv import load_dotenv

load_dotenv()

# device config
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"
LLM_DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"
num_gpus = torch.cuda.device_count()

# model cache config
MODEL_CACHE_PATH = os.path.join(os.path.dirname(__file__), 'model_cache')


# vector storage config
VECTOR_STORE_PATH='./vector_store'
COLLECTION_NAME='my_collection'


# llm backend config
DEF_BACKEND = "vllm"
LLM_BACKEND = os.environ.get("LLM_BACKEND", DEF_BACKEND).strip().lower() or DEF_BACKEND

def _safe_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

API_TIMEOUT = _safe_int(os.environ.get("API_TIMEOUT", "30"), 30)
API_MAX_RETRIES = _safe_int(os.environ.get("API_MAX_RETRIES", "1"), 1)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_MAX_WORKERS = _safe_int(os.environ.get("OLLAMA_MAX_WORKERS", "1"), 1)
OLLAMA_TIMEOUT = _safe_int(os.environ.get("OLLAMA_TIMEOUT", "900"), 900)


def _parse_options(options_env: str):
    if not options_env:
        return None
    try:
        return json.loads(options_env)
    except json.JSONDecodeError:
        return None


OLLAMA_OPTIONS = _parse_options(os.environ.get("OLLAMA_OPTIONS"))

# ============================================================
# LLM API 配置。默认通过阿里云百炼的 OpenAI 兼容接口调用。
# 保留 OPENAI_* 作为非阿里供应商的兼容配置，但不再包含任何硬编码 Key。
# ============================================================
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "dashscope").strip().lower()
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "").strip()
_legacy_api_base = os.environ.get("OPENAI_API_BASE", "").strip()
_legacy_model_name = os.environ.get("OPENAI_MODEL_NAME", "").strip()
DASHSCOPE_API_BASE = os.environ.get("DASHSCOPE_API_BASE", "").strip() or (
    _legacy_api_base if "aliyuncs.com" in _legacy_api_base
    else "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
DASHSCOPE_MODEL_NAME = os.environ.get("DASHSCOPE_MODEL_NAME", "").strip() or (
    _legacy_model_name if _legacy_model_name.startswith("qwen") else "qwen-plus"
)

if LLM_PROVIDER in {"dashscope", "aliyun", "bailian"}:
    OPENAI_API_KEY = DASHSCOPE_API_KEY
    OPENAI_API_BASE = DASHSCOPE_API_BASE
    OPENAI_MODEL_NAME = DASHSCOPE_MODEL_NAME
else:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
    OPENAI_API_BASE = _legacy_api_base or "https://api.openai.com/v1"
    OPENAI_MODEL_NAME = _legacy_model_name or "gpt-4o-mini"

# ============================================================
# 嵌入模型配置
# ============================================================
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "local").strip().lower()
EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", "./pre_train_model/m3e-large")

# API 嵌入模型配置
EMBEDDING_API_PROVIDER = os.environ.get("EMBEDDING_API_PROVIDER", "dashscope").strip().lower()
EMBEDDING_API_TYPE = os.environ.get("EMBEDDING_API_TYPE", "openai").strip().lower()  # 协议类型
if EMBEDDING_API_PROVIDER in {"dashscope", "aliyun", "bailian"}:
    EMBEDDING_API_KEY = os.environ.get("DASHSCOPE_EMBEDDING_API_KEY", DASHSCOPE_API_KEY).strip()
    EMBEDDING_API_BASE = os.environ.get("DASHSCOPE_EMBEDDING_API_BASE", DASHSCOPE_API_BASE).strip()
    EMBEDDING_MODEL_NAME = os.environ.get("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4").strip()
    EMBEDDING_API_BATCH_SIZE = _safe_int(os.environ.get("DASHSCOPE_EMBEDDING_BATCH_SIZE", "10"), 10)
else:
    EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "").strip()
    EMBEDDING_API_BASE = os.environ.get("EMBEDDING_API_BASE", "").strip()
    EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-v1").strip()
    EMBEDDING_API_BATCH_SIZE = _safe_int(os.environ.get("EMBEDDING_API_BATCH_SIZE", "100"), 100)

# Ollama 嵌入模型配置
OLLAMA_EMBEDDING_BASE_URL = os.environ.get("OLLAMA_EMBEDDING_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# ============================================================
# Rerank 模型配置
# ============================================================
RERANK_BACKEND = os.environ.get("RERANK_BACKEND", "local").strip().lower()
RERANK_API_PROVIDER = (
    os.environ.get("DASHSCOPE_RERANK_PROVIDER", "dashscope")
    if LLM_PROVIDER in {"dashscope", "aliyun", "bailian"}
    else os.environ.get("RERANK_API_PROVIDER", "jina")
).strip().lower()
RERANK_MODEL_PATH = os.environ.get("RERANK_MODEL_PATH", "./pre_train_model/bge-reranker-large")
if RERANK_API_PROVIDER in {"dashscope", "aliyun", "bailian"}:
    RERANK_API_KEY = os.environ.get("DASHSCOPE_RERANK_API_KEY", DASHSCOPE_API_KEY).strip()
    RERANK_API_BASE = os.environ.get(
        "DASHSCOPE_RERANK_API_BASE",
        "https://dashscope.aliyuncs.com/compatible-api/v1/reranks",
    ).strip()
    RERANK_MODEL_NAME = os.environ.get("DASHSCOPE_RERANK_MODEL", "qwen3-rerank").strip()
else:
    RERANK_API_KEY = os.environ.get("RERANK_API_KEY", "").strip()
    RERANK_API_BASE = os.environ.get("RERANK_API_BASE", "").strip()
    RERANK_MODEL_NAME = os.environ.get("RERANK_MODEL_NAME", "bge-reranker-v2-m3").strip()
RERANK_TOP_N = _safe_int(os.environ.get("RERANK_TOP_N", "10"), 10)

# ============================================================
# 评分模型配置 (test_score.py 使用)
# ============================================================
SCORING_BACKEND = os.environ.get("SCORING_BACKEND", "local").strip().lower()
SCORING_MODEL_PATH = os.environ.get("SCORING_MODEL_PATH", "./pre_train_model/text2vec-base-chinese")
SCORING_DEVICE = os.environ.get("SCORING_DEVICE", "cuda:0")
SCORING_API_TYPE = os.environ.get("SCORING_API_TYPE", "openai").strip().lower()
SCORING_API_KEY = os.environ.get("SCORING_API_KEY", "")
SCORING_API_BASE = os.environ.get("SCORING_API_BASE", "https://api.openai.com/v1")
SCORING_MODEL_NAME = os.environ.get("SCORING_MODEL_NAME", "text-embedding-3-small")

# ============================================================
# 实验模式配置 (用于 AB Test)
# ============================================================
# 是否启用 FAQ 工具
ENABLE_FAQ_TOOL = os.environ.get("ENABLE_FAQ_TOOL", "true").strip().lower() == "true"
# 是否启用联网搜索
ENABLE_WEB_SEARCH_TOOL = os.environ.get("ENABLE_WEB_SEARCH_TOOL", "true").strip().lower() == "true"
# 是否启用 Rerank 重排序
ENABLE_RERANK = os.environ.get("ENABLE_RERANK", "true").strip().lower() == "true"

# ============================================================
# 多模态视觉检索配置
# ============================================================
# Codex-MM-ADD: 新增图片视觉向量检索开关。关闭或模型不可用时，系统仍会退回
# 到“图片附近文本 + BM25/文本向量”的原有多模态检索链路。
ENABLE_VISUAL_RETRIEVAL = os.environ.get("ENABLE_VISUAL_RETRIEVAL", "false").strip().lower() == "true"
VISUAL_EMBEDDING_BACKEND = os.environ.get("VISUAL_EMBEDDING_BACKEND", "local").strip().lower()
VISUAL_EMBEDDING_MODEL_PATH = os.environ.get(
    "VISUAL_EMBEDDING_MODEL_PATH",
    "./pre_train_model/clip-vit-base-patch32",
)
VISUAL_EMBEDDING_DEVICE = os.environ.get("VISUAL_EMBEDDING_DEVICE", EMBEDDING_DEVICE)
VISUAL_RETRIEVAL_CANDIDATE_K = _safe_int(os.environ.get("VISUAL_RETRIEVAL_CANDIDATE_K", "12"), 12)
VISUAL_EMBEDDING_API_KEY = os.environ.get("VISUAL_EMBEDDING_API_KEY", DASHSCOPE_API_KEY)
VISUAL_EMBEDDING_API_BASE = os.environ.get("VISUAL_EMBEDDING_API_BASE", "")
VISUAL_EMBEDDING_API_MODEL = os.environ.get("VISUAL_EMBEDDING_API_MODEL", "")
VISUAL_EMBEDDING_API_TIMEOUT = _safe_int(os.environ.get("VISUAL_EMBEDDING_API_TIMEOUT", "60"), 60)
VISUAL_EMBEDDING_API_INPUT_FIELD = os.environ.get("VISUAL_EMBEDDING_API_INPUT_FIELD", "input")
VISUAL_EMBEDDING_API_STYLE = os.environ.get("VISUAL_EMBEDDING_API_STYLE", "auto").strip().lower()
