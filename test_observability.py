#!/usr/bin/env python
# coding: utf-8

import unittest
from types import SimpleNamespace

from agent.observability import InstrumentedLLM, instrument_node


class FakeLLM:
    model_name = "fake-model"

    def invoke(self, prompt):
        return SimpleNamespace(
            content="ok",
            usage_metadata={"input_tokens": 12, "output_tokens": 4},
            response_metadata={},
        )


class ObservabilityTests(unittest.TestCase):
    def test_node_and_llm_are_recorded_in_one_trace(self):
        def node(state):
            InstrumentedLLM(FakeLLM(), purpose="planner").invoke("hello")
            state["value"] = 1
            return state

        state = instrument_node("planning", node)({})
        self.assertEqual(1, state["metrics"]["node_calls"])
        self.assertEqual(1, state["metrics"]["llm_calls"])
        self.assertEqual(12, state["metrics"]["input_tokens"])
        self.assertEqual(4, state["metrics"]["output_tokens"])
        self.assertEqual(["node", "llm"], [x["event_type"] for x in state["execution_trace"]])
        self.assertEqual("success", state["execution_trace"][0]["status"])

    def test_llm_outside_graph_does_not_require_state(self):
        response = InstrumentedLLM(FakeLLM(), purpose="test").invoke("hello")
        self.assertEqual("ok", response.content)


if __name__ == "__main__":
    unittest.main()
