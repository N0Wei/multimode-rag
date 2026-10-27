#!/usr/bin/env python
# coding: utf-8

import unittest

from agent.nodes.visual_rag import (
    route_after_intent,
    route_after_tool_execution,
    visual_rag_plan_node,
    visual_rag_synthesize_node,
)


class VisualRAGLaneTests(unittest.TestCase):
    def test_explicit_image_request_uses_plain_visual_lane(self):
        self.assertEqual("visual_rag", route_after_intent({"user_question": "查找方向盘布局图"}))
        self.assertEqual("agentic_rag", route_after_intent({"user_question": "方向盘有哪些按键"}))

    def test_visual_plan_contains_exactly_one_local_rag_call(self):
        state = visual_rag_plan_node({"user_question": "展示安全带使用示意图", "metadata": {}})
        self.assertEqual(["rag"], state["plan"])
        self.assertEqual(1, len(state["structured_plan"]["steps"]))
        self.assertEqual("plain_visual_rag", state["metadata"]["retrieval_lane"])
        self.assertEqual("visual_synthesize", route_after_tool_execution(state))

    def test_visual_synthesis_bypasses_reflection_and_marks_image_evidence(self):
        state = {
            "user_question": "展示安全带使用示意图",
            "answer_source": "rag",
            "rag_result": {
                "answer": "已找到安全带使用示意图",
                "confidence": 0.9,
                "image_paths": ["assets/seatbelt.png"],
            },
            "conversation_history": [],
            "reflection_count": 2,
        }
        result = visual_rag_synthesize_node(state)
        self.assertEqual(0, result["reflection_count"])
        self.assertEqual(1.0, result["citation_coverage"])
        self.assertEqual("grounded", result["grounding_status"])


if __name__ == "__main__":
    unittest.main()
