#!/usr/bin/env python
# coding: utf-8

import unittest
from unittest.mock import Mock, patch

from agent.nodes.query_decomposer import _validate_decomposition, query_decomposer_node


class QueryDecomposerTests(unittest.TestCase):
    def test_valid_complex_decomposition(self):
        result = _validate_decomposition({
            "is_complex": True,
            "reason": "需要分别检索两个维度",
            "sub_queries": ["银河 E8 的续航是多少？", "银河 E8 的充电速度是多少？"],
        }, "比较续航和充电速度")
        self.assertTrue(result["is_complex"])
        self.assertEqual(2, len(result["sub_queries"]))

    def test_rejects_too_many_subqueries(self):
        with self.assertRaises(ValueError):
            _validate_decomposition({
                "is_complex": True,
                "reason": "too many",
                "sub_queries": [f"子问题 {i}" for i in range(6)],
            }, "question")

    def test_simple_decision_uses_original_question(self):
        result = _validate_decomposition({
            "is_complex": False,
            "reason": "单一事实",
            "sub_queries": ["模型错误改写的问题"],
        }, "银河 E8 的续航是多少？")
        self.assertEqual(["银河 E8 的续航是多少？"], result["sub_queries"])

    @patch("agent.nodes.query_decomposer.ModelConfig.get_planner_llm")
    def test_node_initializes_multi_hop_state(self, get_llm):
        get_llm.return_value.invoke.return_value = Mock(content='''{
          "is_complex": true,
          "reason": "两个独立属性",
          "sub_queries": ["银河 E8 的续航是多少？", "银河 E8 的安全配置有哪些？"]
        }''')
        state = {
            "user_question": "介绍银河 E8 的续航和安全配置",
            "intent": "product_inquiry",
            "conversation_history": [],
            "metadata": {},
            "reflection_count": 0,
        }
        result = query_decomposer_node(state)
        self.assertTrue(result["is_complex_query"])
        self.assertEqual(0, result["current_subquery_index"])
        self.assertEqual(result["sub_queries"][0], result["current_subquery"])
        self.assertEqual([], result["subquery_results"])

    @patch("agent.nodes.query_decomposer.ModelConfig.get_planner_llm")
    def test_temporal_question_bypasses_llm_decomposition(self, get_llm):
        question = "现在领克官方最新优惠政策是什么？"
        result = query_decomposer_node({
            "user_question": question,
            "intent": "general",
            "conversation_history": [],
            "metadata": {},
        })
        get_llm.assert_not_called()
        self.assertFalse(result["is_complex_query"])
        self.assertEqual([question], result["sub_queries"])


if __name__ == "__main__":
    unittest.main()
