import unittest

from evaluate_agent_end_to_end import score_case, summarize


class AgentEndToEndEvaluatorTests(unittest.TestCase):
    def test_scores_route_source_answer_and_citations(self):
        case = {
            "expected_intents": ["product_inquiry"],
            "required_tools": ["rag"],
            "forbidden_tools": ["web_search"],
            "expected_sources": ["rag"],
            "expected_need_human": False,
            "expected_complex": False,
            "answer_keywords_any": ["防盗"],
            "min_citation_coverage": 0.8,
        }
        result = {
            "intent": "product_inquiry",
            "answer_source": "rag",
            "need_human": False,
            "is_complex_query": False,
            "final_answer": "闭锁后车辆防盗系统进入警戒状态。",
            "citation_coverage": 1.0,
            "execution_trace": [{"event_type": "tool", "tool": "rag", "success": True}],
            "metrics": {"total_duration_ms": 100, "tool_calls": 1, "llm_calls": 4},
        }
        scored = score_case(case, result)
        self.assertTrue(scored["passed"])
        self.assertEqual(1.0, scored["score"])

    def test_deduplicated_tool_event_is_not_counted_as_real_call(self):
        case = {
            "expected_intents": ["general"],
            "required_tools": ["rag"],
            "forbidden_tools": ["web_search"],
            "expected_sources": ["rag"],
            "expected_need_human": False,
        }
        result = {
            "intent": "general",
            "answer_source": "rag",
            "need_human": False,
            "execution_trace": [
                {"event_type": "tool", "tool": "rag", "success": True},
                {"event_type": "tool", "tool": "web_search", "deduplicated": True},
            ],
        }
        scored = score_case(case, result)
        self.assertEqual(["rag"], scored["actual"]["tools"])
        self.assertTrue(scored["checks"]["forbidden_tools"])

    def test_summary_reports_failed_dimensions(self):
        rows = [{
            "case_id": "A",
            "evaluation": {
                "passed": False,
                "score": 0.5,
                "checks": {"intent": True, "source": False},
                "actual": {"metrics": {"total_duration_ms": 200, "llm_calls": 3, "tool_calls": 1}},
            },
        }]
        summary = summarize(rows)
        self.assertEqual(0.0, summary["check_pass_rates"]["source"])
        self.assertEqual(["A"], summary["failed_case_ids"])


if __name__ == "__main__":
    unittest.main()
