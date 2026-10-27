#!/usr/bin/env python
# coding: utf-8

import unittest
from unittest.mock import patch

from agent.nodes.reflection import reflection_node, should_continue
from agent.nodes.synthesizer import _preserve_requested_unit
from agent.nodes.synthesizer import synthesize_answer_node


class ReflectionControlTests(unittest.TestCase):
    def test_requested_unit_is_kept_for_numeric_rag_answer(self):
        self.assertEqual("4549毫米", _preserve_requested_unit("4549", "车辆长度是多少毫米？"))

    def test_grounded_rag_is_not_retried_for_judge_scope_expansion(self):
        state = {
            "answer_source": "rag",
            "quality_score": 5.0,
            "reflection_count": 1,
            "retrieval_grade": "sufficient",
            "grounding_status": "grounded",
            "citation_coverage": 1.0,
            "improvement_actions": ["补充用户未询问的法规标准"],
            "execution_budget": {},
        }
        self.assertEqual("finalize", should_continue(state))
        self.assertEqual([], state["improvement_actions"])

    def test_web_result_is_not_retried_with_same_cached_evidence(self):
        state = {
            "answer_source": "web_search",
            "quality_score": 4.0,
            "reflection_count": 1,
            "improvement_actions": ["重新搜索"],
            "execution_budget": {},
        }
        self.assertEqual("finalize", should_continue(state))
        self.assertEqual([], state["improvement_actions"])

    @patch("agent.nodes.reflection._evaluate_with_llm")
    def test_human_handoff_skips_llm_judge(self, evaluate):
        state = {
            "user_question": "我要投诉",
            "draft_answer": "正在为您转接人工客服。",
            "answer_source": "no_answer",
            "need_human": True,
            "reflection_count": 0,
        }
        state = reflection_node(state)
        evaluate.assert_not_called()
        self.assertEqual(10.0, state["quality_score"])
        self.assertEqual("finalize", should_continue(state))

    @patch("agent.nodes.reflection._evaluate_with_llm")
    def test_direct_response_skips_llm_judge(self, evaluate):
        state = {
            "user_question": "你好",
            "draft_answer": "您好！",
            "answer_source": "direct_response",
            "need_human": False,
            "reflection_count": 0,
        }
        state = reflection_node(state)
        evaluate.assert_not_called()
        self.assertEqual("finalize", should_continue(state))

    @patch("agent.nodes.reflection._evaluate_with_llm")
    def test_no_answer_is_terminal_without_implying_handoff(self, evaluate):
        state = {
            "user_question": "本地资料里没有答案吗？",
            "draft_answer": "当前可用证据不足。",
            "answer_source": "no_answer",
            "need_human": False,
            "reflection_count": 0,
        }
        state = reflection_node(state)
        evaluate.assert_not_called()
        self.assertFalse(state["need_human"])
        self.assertEqual("finalize", should_continue(state))

    def test_explicit_handoff_survives_no_answer_synthesis(self):
        state = synthesize_answer_node({
            "user_question": "请转人工",
            "answer_source": "no_answer",
            "need_human": True,
            "conversation_history": [],
        })
        self.assertTrue(state["need_human"])
        self.assertIn("转接人工", state["draft_answer"])


if __name__ == "__main__":
    unittest.main()
