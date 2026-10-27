#!/usr/bin/env python
# coding: utf-8

import unittest

from agent.nodes.bounded_router import bounded_planning_node


class BoundedRouterTests(unittest.TestCase):
    def route(self, question, intent="general"):
        return bounded_planning_node({
            "user_question": question, "current_subquery": question,
            "intent": intent, "metadata": {},
        })

    def test_temporal_constraint_is_route_not_new_intent(self):
        state = self.route("现在领克官方最新优惠政策是什么？")
        self.assertEqual(["web_search"], state["plan"])
        self.assertEqual("temporal_web", state["metadata"]["capability_route"])

    def test_local_question_has_faq_then_rag_and_no_web(self):
        state = self.route("如何创建和删除蓝牙钥匙？", "product_inquiry")
        self.assertEqual(["faq", "rag"], state["plan"])
        self.assertFalse(state["metadata"]["allow_web_fallback"])

    def test_complaint_and_chitchat_have_terminal_routes(self):
        self.assertEqual(["human_handoff"], self.route("我要投诉", "complaint")["plan"])
        self.assertEqual(["direct_response"], self.route("你好", "chitchat")["plan"])


if __name__ == "__main__":
    unittest.main()
