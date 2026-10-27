#!/usr/bin/env python
# coding: utf-8

import unittest

from evaluate_text_table_retrieval import aggregate, evaluate_case, recall_at_k, reciprocal_rank


class RetrievalMetricTests(unittest.TestCase):
    def test_recall_and_reciprocal_rank(self):
        self.assertEqual(recall_at_k(["a", "b", "c"], ["b", "c"], 2), 0.5)
        self.assertEqual(recall_at_k(["a", "b", "c"], ["b", "c"], 3), 1.0)
        self.assertEqual(reciprocal_rank(["a", "b", "c"], ["c"]), 1 / 3)

    def test_case_maps_child_results_to_parent_metrics(self):
        case = {
            "case_id": "TT-001",
            "query": "query",
            "category": "text_fact",
            "expected_topic": "topic",
            "expected_child_ids": ["child-b"],
            "expected_parent_ids": ["parent-b"],
        }
        results = [
            {"chunk_id": "child-a", "metadata": {"parent_id": "parent-a"}},
            {"chunk_id": "child-b", "metadata": {"parent_id": "parent-b"}},
        ]
        detail = evaluate_case(case, results, 12.5, [1, 3, 5])
        self.assertEqual(detail["child_recall"]["@1"], 0.0)
        self.assertEqual(detail["child_recall"]["@3"], 1.0)
        self.assertEqual(detail["parent_recall"]["@3"], 1.0)
        self.assertEqual(detail["child_mrr"], 0.5)
        self.assertEqual(detail["parent_mrr"], 0.5)

    def test_aggregate(self):
        details = [
            {
                "child_recall": {"@1": 1.0},
                "parent_recall": {"@1": 0.0},
                "child_mrr": 1.0,
                "parent_mrr": 0.5,
                "latency_ms": 10.0,
                "error": None,
                "search_info": {"vector_status": "ok", "bm25_status": "ok"},
            },
            {
                "child_recall": {"@1": 0.0},
                "parent_recall": {"@1": 1.0},
                "child_mrr": 0.0,
                "parent_mrr": 1.0,
                "latency_ms": 20.0,
                "error": "failed",
                "search_info": {"vector_status": "failed", "bm25_status": "ok"},
            },
        ]
        metrics = aggregate(details, [1])
        self.assertEqual(metrics["child_recall"]["@1"], 0.5)
        self.assertEqual(metrics["parent_recall"]["@1"], 0.5)
        self.assertEqual(metrics["latency_ms"]["average"], 15.0)
        self.assertEqual(metrics["error_count"], 1)
        self.assertEqual(metrics["degraded_case_count"], 1)


if __name__ == "__main__":
    unittest.main()
