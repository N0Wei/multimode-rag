#!/usr/bin/env python
# coding: utf-8

import unittest
from unittest.mock import Mock, patch

from agent.nodes.multi_hop_synthesizer import multi_hop_synthesize_node
from agent.nodes.subquery_coordinator import collect_subquery_result_node, route_after_collection


class SubqueryCoordinatorTests(unittest.TestCase):
    def _state(self):
        return {
            "user_question": "比较两个指标",
            "is_complex_query": True,
            "sub_queries": ["问题一？", "问题二？"],
            "current_subquery_index": 0,
            "current_subquery": "问题一？",
            "subquery_results": [],
            "answer_source": "rag",
            "rag_result": {"answer": "答案一", "context": "证据一", "documents": [], "confidence": 0.8},
            "confidence": 0.0,
            "retrieval_grade": "sufficient",
            "retrieval_grade_reason": "证据充分",
        }

    def test_collects_result_and_advances_only_one_step(self):
        state = collect_subquery_result_node(self._state())
        self.assertEqual(1, len(state["subquery_results"]))
        self.assertEqual("问题二？", state["current_subquery"])
        self.assertEqual("next_subquery", route_after_collection(state))

    def test_routes_to_multi_synthesis_when_all_done(self):
        state = self._state()
        state["current_subquery_index"] = 1
        state["current_subquery"] = "问题二？"
        state = collect_subquery_result_node(state)
        self.assertEqual("multi_synthesize", route_after_collection(state))

    @patch("agent.nodes.multi_hop_synthesizer.ModelConfig.get_synthesize_llm")
    def test_synthesizes_successes_and_reports_failed_parts(self, get_llm):
        get_llm.return_value.invoke.return_value = Mock(content="综合答案")
        state = {
            "user_question": "综合问题",
            "subquery_results": [
                {"question": "问题一", "status": "success", "answer": "答案一", "source": "rag", "evidence": {"context": "证据"}, "confidence": 0.8},
                {"question": "问题二", "status": "failed", "answer": "", "source": "no_answer", "evidence": {}, "confidence": 0.0},
            ],
        }
        result = multi_hop_synthesize_node(state)
        self.assertEqual("multi_hop", result["answer_source"])
        self.assertIn("问题二", result["draft_answer"])
        self.assertEqual(0.8, result["confidence"])


if __name__ == "__main__":
    unittest.main()
