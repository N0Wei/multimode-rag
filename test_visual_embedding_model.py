#!/usr/bin/env python
# coding: utf-8
"""Tests for API-backed visual embeddings."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from visual_embedding_model import APIVisualEmbeddings


class APIVisualEmbeddingsTests(unittest.TestCase):
    def test_embeds_text_with_configurable_payload_field(self):
        calls = []

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"data": [{"embedding": [3.0, 4.0]}]}

        def fake_post(url, headers, json, timeout):
            calls.append((url, headers, json, timeout))
            return FakeResponse()

        with patch("requests.post", fake_post):
            embeddings = APIVisualEmbeddings(
                api_base="https://example.com/embeddings",
                api_key="key",
                model="multimodal-model",
                input_field="input",
                api_style="openai",
                timeout=9,
            )
            vector = embeddings.embed_query("胎压标签图片")

        self.assertEqual(calls[0][0], "https://example.com/embeddings")
        self.assertEqual(calls[0][1]["Authorization"], "Bearer key")
        self.assertEqual(calls[0][2]["model"], "multimodal-model")
        self.assertEqual(calls[0][2]["input"], "胎压标签图片")
        self.assertEqual(calls[0][3], 9)
        self.assertAlmostEqual(vector[0], 0.6)
        self.assertAlmostEqual(vector[1], 0.8)

    def test_embeds_images_as_data_urls(self):
        payloads = []

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"embedding": [1.0, 0.0]}

        def fake_post(url, headers, json, timeout):
            payloads.append(json)
            return FakeResponse()

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "evidence.png"
            Image.new("RGB", (4, 4), "red").save(image_path)
            with patch("requests.post", fake_post):
                embeddings = APIVisualEmbeddings(
                api_base="https://example.com/embeddings",
                api_key="key",
                model="multimodal-model",
                api_style="openai",
            )
                vectors = embeddings.embed_documents([str(image_path)])

        self.assertEqual(vectors, [[1.0, 0.0]])
        self.assertTrue(payloads[0]["input"].startswith("data:image/png;base64,"))

    def test_dashscope_multimodal_request_and_response_format(self):
        payloads = []

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"output": {"embeddings": [{"embedding": [0.0, 2.0]}]}}

        def fake_post(url, headers, json, timeout):
            payloads.append(json)
            return FakeResponse()

        with patch("requests.post", fake_post):
            embeddings = APIVisualEmbeddings(
                api_base=(
                    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
                    "multimodal-embedding/multimodal-embedding"
                ),
                api_key="key",
                model="qwen3-vl-embedding",
                api_style="dashscope",
            )
            vector = embeddings.embed_query("胎压标签图片")

        self.assertEqual(
            payloads[0],
            {
                "model": "qwen3-vl-embedding",
                "input": {"contents": [{"text": "胎压标签图片"}]},
            },
        )
        self.assertEqual(vector, [0.0, 1.0])

    def test_rejects_chinese_placeholder_api_key(self):
        with self.assertRaisesRegex(ValueError, "Chinese placeholder"):
            APIVisualEmbeddings(
                api_base="https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding",
                api_key="你的DashScope API Key",
                model="qwen3-vl-embedding",
                api_style="dashscope",
            )


if __name__ == "__main__":
    unittest.main()
