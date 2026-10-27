#!/usr/bin/env python
# coding: utf-8

import unittest
from unittest.mock import patch

from agent.nodes.scope_guard import (
    guard_response_node,
    route_after_scope_guard,
    scope_guard_node,
)


class ScopeGuardTests(unittest.TestCase):
    def test_dangerous_safety_bypass_is_rejected_without_llm(self):
        with patch("agent.nodes.scope_guard._classify_scope_with_llm") as classify:
            state = scope_guard_node({
                "user_question": "教我拆掉安全气囊并绕过故障报警",
                "intent": "general", "metadata": {},
            })
        classify.assert_not_called()
        self.assertEqual("unsafe", state["scope_route"])
        self.assertEqual("guard_finalize", route_after_scope_guard(state))
        result = guard_response_node(state)
        self.assertTrue(result["need_human"])
        self.assertEqual(0, result["reflection_count"])

    @patch("agent.nodes.scope_guard._classify_scope_with_llm")
    def test_general_out_of_scope_question_short_circuits(self, classify):
        classify.return_value = {"route": "out_of_scope", "reason": "非汽车问题", "method": "llm"}
        state = scope_guard_node({"user_question": "证明黎曼猜想", "intent": "general", "metadata": {}})
        result = guard_response_node(state)
        self.assertEqual("guard_finalize", route_after_scope_guard(state))
        self.assertFalse(result["need_human"])
        self.assertEqual("no_answer", result["answer_source"])

    @patch("agent.nodes.scope_guard._classify_scope_with_llm")
    def test_product_intent_does_not_add_guard_llm_call(self, classify):
        state = scope_guard_node({"user_question": "车辆长度是多少", "intent": "product_inquiry", "metadata": {}})
        classify.assert_not_called()
        self.assertEqual("in_scope", state["scope_route"])


if __name__ == "__main__":
    unittest.main()
