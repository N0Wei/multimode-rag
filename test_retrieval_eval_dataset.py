#!/usr/bin/env python
# coding: utf-8

import json
import unittest
from pathlib import Path


class RetrievalEvalDatasetTests(unittest.TestCase):
    def test_dataset_stores_only_parent_or_image_ground_truth(self):
        chunks = json.loads(
            Path("data/knowledge_bases/train_a/chunks.json").read_text(encoding="utf-8")
        )
        dataset = json.loads(
            Path("data/evaluation/multimodal_retrieval_eval.json").read_text(encoding="utf-8")
        )
        by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

        self.assertGreaterEqual(len(dataset["cases"]), 50)
        self.assertLessEqual(len(dataset["cases"]), 100)
        for case in dataset["cases"]:
            expected_ids = []
            for field in ("expected_parent_ids", "expected_image_ids"):
                expected_ids.extend(case[field])
            self.assertNotIn("expected_child_ids", case)
            self.assertNotIn("expected_table_row_ids", case)
            self.assertNotIn("expected_table_parent_ids", case)
            if case["is_answerable"]:
                self.assertTrue(expected_ids, case["case_id"])
            else:
                self.assertFalse(expected_ids, case["case_id"])
            for chunk_id in expected_ids:
                self.assertIn(chunk_id, by_id, case["case_id"])
            for parent_id in case["expected_parent_ids"]:
                self.assertIn(
                    by_id[parent_id]["chunk_type"],
                    {"text_parent", "table_parent"},
                    case["case_id"],
                )


if __name__ == "__main__":
    unittest.main()
