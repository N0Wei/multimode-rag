#!/usr/bin/env python
# coding: utf-8

import unittest

from evaluate_image_retrieval import (
    aggregate,
    evaluate_ranking,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)


class ImageRetrievalMetricTests(unittest.TestCase):
    def test_core_metrics_with_multiple_correct_images(self):
        retrieved = ["a", "b", "c", "d"]
        expected = ["b", "d"]
        self.assertEqual(hit_at_k(retrieved, expected, 1), 0.0)
        self.assertEqual(hit_at_k(retrieved, expected, 3), 1.0)
        self.assertEqual(recall_at_k(retrieved, expected, 3), 0.5)
        self.assertEqual(recall_at_k(retrieved, expected, 5), 1.0)
        self.assertEqual(reciprocal_rank(retrieved, expected), 0.5)

    def test_ranking_evaluates_exact_image_and_topic(self):
        case = {
            "expected_image_ids": ["img-2"],
            "expected_topic": "topic-b",
        }
        ranking = [
            {"chunk_id": "img-1", "topic_title": "topic-b"},
            {"chunk_id": "img-2", "topic_title": "topic-b"},
        ]
        detail = evaluate_ranking(case, ranking, 10.0, [1, 3, 5])
        self.assertEqual(detail["image_hit"]["@1"], 0.0)
        self.assertEqual(detail["image_hit"]["@3"], 1.0)
        self.assertEqual(detail["topic_hit"]["@1"], 1.0)
        self.assertEqual(detail["image_mrr"], 0.5)
        self.assertEqual(detail["expected_image_count"], 1)
        self.assertEqual(detail["matched_image_count"]["@3"], 1)

    def test_aggregate(self):
        details = [
            {
                "image_hit": {"@1": 1.0},
                "image_recall": {"@1": 0.5},
                "expected_image_count": 2,
                "matched_image_count": {"@1": 1},
                "image_mrr": 1.0,
                "topic_hit": {"@1": 1.0},
                "latency_ms": 10.0,
                "error": None,
            },
            {
                "image_hit": {"@1": 0.0},
                "image_recall": {"@1": 0.0},
                "expected_image_count": 1,
                "matched_image_count": {"@1": 0},
                "image_mrr": 0.0,
                "topic_hit": {"@1": 0.0},
                "latency_ms": 20.0,
                "error": "failed",
            },
        ]
        metrics = aggregate(details, [1])
        self.assertEqual(metrics["image_hit"]["@1"], 0.5)
        self.assertEqual(metrics["image_recall"]["@1"], 0.25)
        self.assertAlmostEqual(metrics["micro_image_recall"]["@1"], 1 / 3, places=6)
        self.assertEqual(metrics["latency_ms"]["average"], 15.0)
        self.assertEqual(metrics["error_count"], 1)


if __name__ == "__main__":
    unittest.main()
