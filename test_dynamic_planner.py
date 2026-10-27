#!/usr/bin/env python
# coding: utf-8

import unittest
from unittest.mock import Mock, patch

from agent.nodes.planner import _create_structured_plan, _validate_plan, planning_node


class DynamicPlannerTests(unittest.TestCase):
    def test_validates_model_selected_web_plan(self):
        plan = _validate_plan({
            "reasoning": "问题具有时效性",
            "steps": [{
                "tool": "web_search",
                "reason": "需要最新信息",
                "parameters": {"query": "银河 E8 最新售价"},
            }],
        }, "银河 E8 多少钱")
        self.assertEqual("web_search", plan["steps"][0]["tool"])
        self.assertFalse(plan["fallback_used"])

    def test_rejects_unknown_tool_and_parameters(self):
        with self.assertRaises(ValueError):
            _validate_plan({
                "reasoning": "test",
                "steps": [{"tool": "shell", "reason": "test", "parameters": {}}],
            }, "question")
        with self.assertRaises(ValueError):
            _validate_plan({
                "reasoning": "test",
                "steps": [{"tool": "rag", "reason": "test", "parameters": {"danger": True}}],
            }, "question")

    @patch("agent.nodes.planner.ModelConfig.get_planner_llm")
    def test_invalid_model_output_uses_safe_fallback(self, get_llm):
        get_llm.return_value.invoke.return_value = Mock(content="not json")
        plan = _create_structured_plan("product_inquiry", "续航是多少", [], "")
        self.assertTrue(plan["fallback_used"])
        self.assertEqual(["rag", "web_search"], [s["tool"] for s in plan["steps"]])

    @patch("agent.nodes.planner.ModelConfig.get_planner_llm")
    def test_planning_node_uses_model_plan_not_intent_if_else(self, get_llm):
        get_llm.return_value.invoke.return_value = Mock(content='''{
          "reasoning": "当前价格会变化，直接搜索最新信息",
          "steps": [{"tool":"web_search","reason":"需要实时价格","parameters":{"query":"银河 E8 当前售价"}}]
        }''')
        state = {
            "user_question": "银河 E8 现在多少钱？",
            "conversation_history": [],
            "intent": "product_inquiry",
            "reflection_count": 0,
            "improvement_actions": [],
            "metadata": {},
        }
        result = planning_node(state)
        self.assertEqual(["web_search"], result["plan"])
        self.assertEqual("银河 E8 当前售价", result["structured_plan"]["steps"][0]["parameters"]["query"])


if __name__ == "__main__":
    unittest.main()
