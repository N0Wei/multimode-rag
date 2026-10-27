#!/usr/bin/env python
# coding: utf-8

import unittest
from unittest.mock import patch

from agent.nodes.retrieval_grader import retrieval_grader_node, route_after_retrieval_grade


class RetrievalGraderTests(unittest.TestCase):
    def _state(self):
        return {
            "user_question": "银河 E8 的续航是多少？",
            "query_rewrite": "银河 E8 的续航是多少？",
            "answer_source": "rag",
            "rag_result": {
                "documents": [object()],
                "context": "银河 E8 的 CLTC 续航为 620 公里。",
                "confidence": 0.8,
                "is_no_answer": False,
            },
            "metadata": {},
            "plan": ["rag"],
        }

    @patch("agent.nodes.retrieval_grader._grade_evidence")
    def test_sufficient_evidence_routes_to_synthesis(self, grade):
        grade.return_value = {"sufficient": True, "reason": "包含核心事实", "method": "llm"}
        state = retrieval_grader_node(self._state())
        self.assertEqual("sufficient", state["retrieval_grade"])
        self.assertEqual("synthesize", route_after_retrieval_grade(state))

    @patch("agent.nodes.retrieval_grader._regenerate_from_accepted_evidence")
    @patch("agent.nodes.retrieval_grader._grade_evidence")
    def test_sufficient_evidence_recovers_false_no_answer_state(self, grade, regenerate):
        grade.return_value = {"sufficient": True, "reason": "图片证据直接相关", "method": "llm"}
        regenerate.return_value = "已找到正确的图片证据"
        state = self._state()
        state["answer_source"] = "no_answer"
        state["last_executed_tool"] = "rag"
        state["need_human"] = True
        state["rag_result"]["is_no_answer"] = True
        state = retrieval_grader_node(state)
        self.assertEqual("rag", state["answer_source"])
        self.assertFalse(state["need_human"])
        self.assertFalse(state["rag_result"]["is_no_answer"])
        self.assertEqual("已找到正确的图片证据", state["rag_result"]["answer"])

    @patch("agent.nodes.retrieval_grader._grade_evidence")
    def test_grades_against_original_question_not_expanded_retry_query(self, grade):
        state = self._state()
        state["active_retrieval_query"] = "银河 E8 车长以及不存在的 GB/T 标准"
        retrieval_grader_node(state)
        self.assertEqual("银河 E8 的续航是多少？", grade.call_args.args[0])

    @patch("agent.nodes.retrieval_grader._rewrite_for_retrieval")
    @patch("agent.nodes.retrieval_grader._grade_evidence")
    def test_insufficient_evidence_rewrites_and_retries(self, grade, rewrite):
        grade.return_value = {"sufficient": False, "reason": "缺少续航数据", "method": "llm"}
        rewrite.return_value = "银河 E8 CLTC 纯电续航里程"
        state = retrieval_grader_node(self._state())
        self.assertEqual(["rag"], state["plan"])
        self.assertEqual(1, state["retrieval_retry_count"])
        self.assertTrue(state["rag_result"]["is_no_answer"])
        self.assertEqual("retry_retrieval", route_after_retrieval_grade(state))

    @patch("agent.nodes.retrieval_grader._grade_evidence")
    def test_retry_limit_stops_at_local_boundary_by_default(self, grade):
        grade.return_value = {"sufficient": False, "reason": "证据仍不足", "method": "llm"}
        state = self._state()
        state["retrieval_retry_count"] = 2
        state = retrieval_grader_node(state)
        self.assertEqual([], state["plan"])

    @patch("agent.nodes.retrieval_grader._grade_evidence")
    def test_retry_limit_uses_web_only_when_route_authorized_it(self, grade):
        grade.return_value = {"sufficient": False, "reason": "证据仍不足", "method": "llm"}
        state = self._state()
        state["retrieval_retry_count"] = 2
        state["metadata"]["allow_web_fallback"] = True
        state = retrieval_grader_node(state)
        self.assertEqual(["web_search"], state["plan"])

    def test_non_rag_source_skips_grading(self):
        state = self._state()
        state["answer_source"] = "faq"
        state = retrieval_grader_node(state)
        self.assertEqual("not_applicable", state["retrieval_grade"])

    @patch("agent.nodes.retrieval_grader._rewrite_for_retrieval")
    def test_rag_no_answer_still_enters_retry_loop(self, rewrite):
        rewrite.return_value = "改写后的查询"
        state = self._state()
        state["answer_source"] = "no_answer"
        state["last_executed_tool"] = "rag"
        state["rag_result"] = {
            "documents": [], "context": "", "confidence": 0.0, "is_no_answer": True
        }
        state = retrieval_grader_node(state)
        self.assertEqual("insufficient", state["retrieval_grade"])
        self.assertEqual(["rag"], state["plan"])
        self.assertEqual(1, state["retrieval_retry_count"])

    def test_rag_no_answer_does_not_cross_local_boundary(self):
        state = self._state()
        state["answer_source"] = "no_answer"
        state["last_executed_tool"] = "rag"
        state["retrieval_retry_count"] = 2
        state["rag_result"] = {
            "documents": [], "context": "", "confidence": 0.0, "is_no_answer": True
        }
        state = retrieval_grader_node(state)
        self.assertEqual("insufficient", state["retrieval_grade"])
        self.assertEqual([], state["plan"])
        self.assertEqual([], state["structured_plan"]["steps"])


if __name__ == "__main__":
    unittest.main()
