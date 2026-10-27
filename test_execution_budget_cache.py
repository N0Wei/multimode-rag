#!/usr/bin/env python
# coding: utf-8

import unittest
from unittest.mock import patch

import agent.nodes.tool_executor as executor
from agent.nodes.reflection import should_continue
from agent.nodes.retrieval_grader import retrieval_grader_node


class FakeRAG:
    def __init__(self):
        self.calls = 0

    def search_and_generate(self, query, top_k=6, max_context_length=4000):
        self.calls += 1
        return {"answer": "续航 665 公里", "documents": [object()], "context": "续航 665 公里",
                "confidence": 0.9, "is_no_answer": False}


class MissingFAQ:
    def match(self, query, threshold=0.85):
        return None


class ExecutionBudgetCacheTests(unittest.TestCase):
    def setUp(self):
        self.old_tools = (executor.rag_tool, executor.faq_tool, executor.web_search_tool)
        self.rag = FakeRAG()
        executor.set_tools(self.rag, None, None)

    def tearDown(self):
        executor.set_tools(*self.old_tools)

    def _state(self, query="银河 E8 续航"):
        return {
            "user_question": query,
            "query_rewrite": query,
            "plan": ["rag"],
            "structured_plan": {"steps": [{"tool": "rag", "reason": "测试", "parameters": {"query": query}}]},
            "metadata": {}, "execution_cache": {}, "execution_budget": {},
        }

    def test_successful_call_is_reused_without_second_tool_call(self):
        state = executor.tool_execution_node(self._state())
        state = executor.tool_execution_node(state)
        self.assertEqual(1, self.rag.calls)
        self.assertEqual(1, state["execution_budget"]["tool_calls"])
        self.assertEqual(1, state["execution_budget"]["cache_hits"])
        self.assertTrue(state["metadata"]["tool_trace"][-1]["cache_hit"])
        self.assertEqual("rag", state["answer_source"])

    def test_successful_retry_clears_stale_human_handoff_flag(self):
        state = self._state()
        state["need_human"] = True
        state = executor.tool_execution_node(state)
        self.assertFalse(state["need_human"])
        self.assertEqual("rag", state["answer_source"])

    def test_faq_miss_falls_back_to_local_rag(self):
        executor.set_tools(self.rag, MissingFAQ(), None)
        state = self._state("如何联系领克客服？")
        state["plan"] = ["faq"]
        state["structured_plan"] = {
            "steps": [{"tool": "faq", "reason": "标准问答", "parameters": {"query": state["user_question"]}}]
        }
        state = executor.tool_execution_node(state)
        self.assertEqual("rag", state["answer_source"])
        self.assertEqual(1, self.rag.calls)
        self.assertTrue(state["metadata"]["faq_rag_fallback_injected"])

    @patch("agent.nodes.retrieval_grader._grade_evidence")
    def test_rejected_rag_result_is_not_reused(self, grade):
        grade.return_value = {"sufficient": False, "reason": "缺少车型限定", "method": "test"}
        state = self._state()
        state["retrieval_retry_count"] = 2
        state = executor.tool_execution_node(state)
        state = retrieval_grader_node(state)
        state["structured_plan"] = {"steps": [{"tool": "rag", "reason": "重复", "parameters": {"query": "银河 E8 续航"}}]}
        state["plan"] = ["rag"]
        state = executor.tool_execution_node(state)
        self.assertEqual(1, self.rag.calls)
        self.assertEqual(1, state["execution_budget"]["deduplicated_calls"])
        self.assertEqual("no_answer", state["answer_source"])

    def test_budget_blocks_new_call_and_stops_reflection_retry(self):
        state = self._state()
        state["execution_budget"] = {"tool_calls": 8, "rag_calls": 5, "web_search_calls": 0,
                                     "faq_calls": 0, "cache_hits": 0, "deduplicated_calls": 0}
        state = executor.tool_execution_node(state)
        self.assertEqual(0, self.rag.calls)
        self.assertTrue(state["execution_budget"]["exhausted"])
        state.update({"quality_score": 2.0, "reflection_count": 1, "answer_source": "no_answer"})
        self.assertEqual("finalize", should_continue(state))

    def test_cache_key_includes_effective_rag_parameters(self):
        first = self._state()
        first = executor.tool_execution_node(first)
        first["structured_plan"]["steps"][0]["parameters"]["top_k"] = 10
        executor.tool_execution_node(first)
        self.assertEqual(2, self.rag.calls)


if __name__ == "__main__":
    unittest.main()
