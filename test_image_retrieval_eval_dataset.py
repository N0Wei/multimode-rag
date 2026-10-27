#!/usr/bin/env python
# coding: utf-8

import json
import unittest
from collections import Counter
from pathlib import Path


class ImageRetrievalEvalDatasetTests(unittest.TestCase):
    def test_dataset_has_100_valid_image_cases(self):
        chunks = json.loads(
            Path("data/knowledge_bases/train_a/chunks.json").read_text(encoding="utf-8")
        )
        dataset = json.loads(
            Path("data/evaluation/image_retrieval_eval_100.json").read_text(encoding="utf-8")
        )
        by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

        self.assertEqual(dataset["metadata"]["case_count"], 100)
        self.assertEqual(len(dataset["cases"]), 100)
        self.assertEqual(
            set(dataset["metadata"]["comparison_modes"]),
            {"visual_only", "visual_plus_image_text"},
        )
        self.assertEqual(
            Counter(case["query_type"] for case in dataset["cases"]),
            Counter({"visual_object": 50, "semantic_context": 50}),
        )

        queries = set()
        for case in dataset["cases"]:
            self.assertNotIn(case["query"], queries, case["case_id"])
            queries.add(case["query"])
            self.assertTrue(case["expected_image_ids"], case["case_id"])
            self.assertLessEqual(len(case["expected_image_ids"]), 3, case["case_id"])
            self.assertNotIn("expected_child_ids", case)
            self.assertNotIn("expected_parent_ids", case)
            for image_id in case["expected_image_ids"]:
                self.assertIn(image_id, by_id, case["case_id"])
                self.assertEqual(by_id[image_id]["doc_type"], "image", case["case_id"])
                metadata = by_id[image_id].get("metadata") or {}
                self.assertTrue(metadata.get("image_path"), case["case_id"])
                self.assertTrue(Path(metadata["image_path"]).exists(), case["case_id"])
                self.assertEqual(metadata.get("topic_title"), case["expected_topic"], case["case_id"])


if __name__ == "__main__":
    unittest.main()
