#!/usr/bin/env python
# coding: utf-8

import json
import unittest
from pathlib import Path


class TextTableRetrievalEvalDatasetTests(unittest.TestCase):
    def test_dataset_has_100_valid_child_and_parent_ground_truth_cases(self):
        chunks = json.loads(
            Path("data/knowledge_bases/train_a/chunks.json").read_text(encoding="utf-8")
        )
        dataset = json.loads(
            Path("data/evaluation/text_table_retrieval_eval_100.json").read_text(encoding="utf-8")
        )
        by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

        self.assertEqual(dataset["metadata"]["case_count"], 100)
        self.assertEqual(len(dataset["cases"]), 100)
        self.assertEqual(
            dataset["metadata"]["ground_truth_fields"],
            ["expected_child_ids", "expected_parent_ids"],
        )

        queries = set()
        for case in dataset["cases"]:
            self.assertNotIn(case["query"], queries, case["case_id"])
            queries.add(case["query"])
            self.assertTrue(case["expected_child_ids"], case["case_id"])
            self.assertTrue(case["expected_parent_ids"], case["case_id"])
            self.assertNotIn("expected_image_ids", case)

            child_parent_ids = set()
            for child_id in case["expected_child_ids"]:
                self.assertIn(child_id, by_id, case["case_id"])
                child = by_id[child_id]
                self.assertIn(
                    child["chunk_type"],
                    {"text_child", "table_row_child"},
                    case["case_id"],
                )
                child_parent_ids.add(child["metadata"]["parent_id"])

            self.assertEqual(set(case["expected_parent_ids"]), child_parent_ids, case["case_id"])
            for parent_id in case["expected_parent_ids"]:
                self.assertIn(parent_id, by_id, case["case_id"])
                self.assertIn(
                    by_id[parent_id]["chunk_type"],
                    {"text_parent", "table_parent"},
                    case["case_id"],
                )


if __name__ == "__main__":
    unittest.main()
