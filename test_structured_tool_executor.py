#!/usr/bin/env python
# coding: utf-8

import unittest

import agent.nodes.tool_executor as executor


class FakeFAQ:
    def __init__(self, calls): self.calls = calls
    def match(self, query, threshold=0.85):
        self.calls.append(("faq", query, threshold))
        return None


class FakeRAG:
    def __init__(self, calls): self.calls = calls
    def search_and_generate(self, query, top_k=6, max_context_length=4000):
        self.calls.append(("rag", query, top_k, max_context_length))
        return {"answer": "ok", "documents": [object()], "context": "evidence", "confidence": 0.9, "is_no_answer": False}


class StructuredExecutorTests(unittest.TestCase):
    def test_executes_model_order_and_parameters(self):
        calls = []
        executor.set_tools(FakeRAG(calls), FakeFAQ(calls), None)
        state = {
            "user_question": "q", "query_rewrite": "q", "plan": ["faq", "rag"],
            "structured_plan": {"steps": [
                {"tool": "faq", "reason": "standard", "parameters": {"query": "faq q", "threshold": 0.9}},
                {"tool": "rag", "reason": "fallback", "parameters": {"query": "rag q", "top_k": 4, "max_context_length": 2000}},
            ]},
            "metadata": {},
        }
        result = executor.tool_execution_node(state)
        self.assertEqual([("faq", "faq q", 0.9), ("rag", "rag q", 4, 2000)], calls)
        self.assertEqual("rag", result["answer_source"])


if __name__ == "__main__":
    unittest.main()
