#!/usr/bin/env python
# coding: utf-8
"""Tests for the upload-to-knowledge-base workflow without embedding API calls."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import knowledge_base_service as service


class KnowledgeBaseServiceTests(unittest.TestCase):
    def test_name_is_sanitized(self):
        self.assertEqual(service.normalize_knowledge_base_name("../测试/知识库"), "测试_知识库")

    def test_active_knowledge_base_is_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "knowledge_bases"
            store = root / "demo" / "vector_store"
            store.mkdir(parents=True)
            (store / "manifest.json").write_text("{}", encoding="utf-8")
            with (
                patch.object(service, "KNOWLEDGE_BASE_ROOT", root),
                patch.object(service, "ACTIVE_KNOWLEDGE_BASE_FILE", root / "active.json"),
            ):
                service.set_active_knowledge_base("demo")
                self.assertEqual(service.get_active_knowledge_base(), "demo")

    def test_build_workflow_persists_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "manual.pdf"
            source.write_bytes(b"%PDF-1.4 test")

            def save_outputs(_splitter, _chunks, out_path, preview_path):
                Path(out_path).write_text("[]", encoding="utf-8")
                Path(preview_path).write_text("# preview", encoding="utf-8")

            fake_chunks = [
                SimpleNamespace(doc_type="text"),
                SimpleNamespace(doc_type="image"),
                SimpleNamespace(doc_type="image"),
            ]
            with (
                patch.object(service, "KNOWLEDGE_BASE_ROOT", root / "knowledge_bases"),
                patch.object(service.ManualPDFSplitter, "split", return_value=fake_chunks),
                patch.object(service.ManualPDFSplitter, "save_outputs", new=save_outputs),
                patch.object(
                    service.MultimodalIndexBuilder,
                    "build",
                    return_value={"document_count": 3, "embedding": "test"},
                ),
            ):
                result = service.build_knowledge_base(str(source), "测试/知识库")

            self.assertEqual(result["name"], "测试_知识库")
            self.assertEqual(result["doc_type_counts"], {"image": 2, "text": 1})
            metadata = json.loads(
                (root / "knowledge_bases" / "测试_知识库" / "knowledge_base.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["chunk_count"], 3)


if __name__ == "__main__":
    unittest.main()
