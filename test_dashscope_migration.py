#!/usr/bin/env python
# coding: utf-8

import unittest
from unittest.mock import Mock, patch

import config
from embedding_model import get_embedding_model
from agent.config.model_config import ModelConfig
from langchain_core.documents import Document
from rerank_model import _DashScopeRerankAdapter


class DashScopeMigrationTests(unittest.TestCase):
    def test_llm_and_agent_resolve_to_dashscope(self):
        self.assertEqual("dashscope", config.LLM_PROVIDER)
        self.assertIn("aliyuncs.com", config.OPENAI_API_BASE)
        self.assertTrue(config.OPENAI_MODEL_NAME.startswith("qwen"))
        self.assertEqual("dashscope", ModelConfig.PROVIDER)
        self.assertIn("aliyuncs.com", ModelConfig.OPENAI_API_BASE)
        self.assertTrue(ModelConfig.OPENAI_MODEL_NAME.startswith("qwen"))

    def test_embedding_and_rerank_use_dashscope_models(self):
        self.assertEqual("dashscope", config.EMBEDDING_API_PROVIDER)
        self.assertEqual("text-embedding-v4", config.EMBEDDING_MODEL_NAME)
        self.assertEqual("dashscope", config.RERANK_API_PROVIDER)
        self.assertEqual("qwen3-rerank", config.RERANK_MODEL_NAME)

    @patch("langchain_openai.OpenAIEmbeddings")
    def test_dashscope_embedding_keeps_string_inputs(self, embeddings):
        get_embedding_model()
        kwargs = embeddings.call_args.kwargs
        self.assertFalse(kwargs["check_embedding_ctx_length"])
        self.assertEqual(10, kwargs["chunk_size"])

    @patch("requests.post")
    def test_dashscope_rerank_request_and_order(self, post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"results": [{"index": 1}, {"index": 0}]}
        post.return_value = response
        adapter = _DashScopeRerankAdapter(
            api_key="test-key",
            api_base="https://dashscope.aliyuncs.com/compatible-api/v1/reranks",
            model="qwen3-rerank",
            top_n=2,
        )
        docs = [Document(page_content="A"), Document(page_content="B")]
        result = adapter.compress_documents(docs, "query")
        self.assertEqual(["B", "A"], [doc.page_content for doc in result])
        payload = post.call_args.kwargs["json"]
        self.assertEqual("qwen3-rerank", payload["model"])
        self.assertEqual(2, payload["top_n"])


if __name__ == "__main__":
    unittest.main()
