#!/usr/bin/env python
# coding: utf-8

import unittest
import os
from unittest.mock import Mock, patch

import agent.nodes.tool_executor as executor
from agent.graph import build_agent_graph


class FakeRAG:
    def __init__(self):
        self.queries = []

    def search_and_generate(self, query, top_k=6, max_context_length=4000):
        self.queries.append(query)
        return {
            "answer": f"{query}的答案",
            "documents": [object()],
            "context": f"{query}的证据",
            "confidence": 0.9,
            "is_no_answer": False,
        }


class MultiHopGraphTests(unittest.TestCase):
    @patch("agent.nodes.multi_hop_synthesizer.ModelConfig.get_synthesize_llm")
    @patch("agent.nodes.retrieval_grader.ModelConfig.get_reflection_llm")
    @patch("agent.nodes.query_decomposer.ModelConfig.get_planner_llm")
    @patch("agent.nodes.intent_classifier.ModelConfig.get_intent_llm")
    def test_two_subqueries_complete_end_to_end(
        self, intent_llm, planner_llm, reflection_llm, synth_llm,
    ):
        intent_llm.return_value.invoke.return_value = Mock(content="product_inquiry")
        planner_llm.return_value.invoke.side_effect = [
            Mock(content='''{
              "is_complex": true,
              "reason": "两个独立属性",
              "sub_queries": ["银河 E8 的续航是多少？", "银河 E8 的安全配置有哪些？"]
            }'''),
        ]
        reflection_llm.return_value.invoke.side_effect = [
            Mock(content='{"sufficient":true,"reason":"包含续航事实"}'),
            Mock(content='{"sufficient":true,"reason":"包含安全配置"}'),
            Mock(content='''{"claims":[
              {"claim_id":"C1","text":"银河 E8 的续航已有检索结果"},
              {"claim_id":"C2","text":"银河 E8 的安全配置已有检索结果"}
            ]}'''),
            Mock(content='''{"verdicts":[
              {"claim_id":"C1","verdict":"supported","evidence_ids":["E1"],"reason":"子问题证据支持"},
              {"claim_id":"C2","verdict":"supported","evidence_ids":["E2"],"reason":"子问题证据支持"}
            ]}'''),
            Mock(content='''{
              "score": 8.5, "notes": ["覆盖全部子问题", "证据充分"], "improvements": []
            }'''),
        ]
        synth_llm.return_value.invoke.return_value = Mock(content="续航与安全配置综合答案")

        old_tools = (executor.rag_tool, executor.faq_tool, executor.web_search_tool)
        rag = FakeRAG()
        executor.set_tools(rag, None, None)
        try:
            with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "false", "LANGSMITH_TRACING": "false"}):
                result = build_agent_graph().invoke({
                    "user_question": "介绍银河 E8 的续航和安全配置",
                    "conversation_history": [],
                    "reflection_count": 0,
                    "reflection_notes": [],
                    "improvement_actions": [],
                    "metadata": {},
                    "need_human": False,
                }, config={"recursion_limit": 80})
        finally:
            executor.set_tools(*old_tools)

        self.assertEqual(["银河 E8 的续航是多少？", "银河 E8 的安全配置有哪些？"], rag.queries)
        self.assertEqual(2, len(result["subquery_results"]))
        self.assertEqual("multi_hop", result["answer_source"])
        self.assertEqual("续航与安全配置综合答案", result["final_answer"])
        self.assertGreater(result["metrics"]["node_calls"], 0)
        self.assertGreaterEqual(result["metrics"]["total_duration_ms"], 0)
        self.assertTrue(any(item.get("event_type") == "tool" for item in result["execution_trace"]))
        self.assertTrue(any(item.get("name") == "finalize" for item in result["execution_trace"]))


if __name__ == "__main__":
    unittest.main()
