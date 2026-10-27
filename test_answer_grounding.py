#!/usr/bin/env python
# coding: utf-8

import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from agent.nodes.grounding import (
    claim_extractor_node,
    evidence_validator_node,
    grounded_rewriter_node,
    route_after_evidence_validation,
)


class AnswerGroundingTests(unittest.TestCase):
    def _state(self):
        return {
            "user_question": "银河 E8 的续航是多少？",
            "draft_answer": "银河 E8 的 CLTC 续航为 665 公里。它支持飞行模式。",
            "answer_source": "rag",
            "rag_result": {
                "documents": [Document(
                    page_content="银河 E8 后驱版 CLTC 纯电续航里程为 665 公里。",
                    metadata={"page": 12, "chunk_id": "chunk-12"},
                )],
            },
            "metadata": {},
        }

    @patch("agent.nodes.grounding.ModelConfig.get_reflection_llm")
    def test_extracts_claims_and_builds_evidence_catalog(self, get_llm):
        get_llm.return_value.invoke.return_value = Mock(content='''{
          "claims": [
            {"claim_id":"ignored","text":"银河 E8 的 CLTC 续航为 665 公里"},
            {"claim_id":"ignored2","text":"它支持飞行模式"}
          ]
        }''')
        state = claim_extractor_node(self._state())
        self.assertEqual(["C1", "C2"], [item["claim_id"] for item in state["claims"]])
        self.assertEqual("E1", state["evidence_catalog"][0]["evidence_id"])
        self.assertEqual(12, state["evidence_catalog"][0]["metadata"]["page"])

    @patch("agent.nodes.grounding.ModelConfig.get_reflection_llm")
    def test_invalid_citation_cannot_count_as_supported(self, get_llm):
        state = self._state()
        state["claims"] = [{"claim_id": "C1", "text": "续航为 665 公里"}]
        state["evidence_catalog"] = [{"evidence_id": "E1", "source": "rag", "content": "续航 665 公里", "metadata": {}}]
        get_llm.return_value.invoke.return_value = Mock(content='''{
          "verdicts":[{"claim_id":"C1","verdict":"supported","evidence_ids":["E999"],"reason":"支持"}]
        }''')
        state = evidence_validator_node(state)
        self.assertEqual("unsupported", state["claim_verdicts"][0]["verdict"])
        self.assertEqual(0.0, state["citation_coverage"])
        self.assertEqual("rewrite", route_after_evidence_validation(state))

    @patch("agent.nodes.grounding.ModelConfig.get_reflection_llm")
    def test_supported_claim_reaches_reflection(self, get_llm):
        state = self._state()
        state["claims"] = [{"claim_id": "C1", "text": "续航为 665 公里"}]
        state["evidence_catalog"] = [{"evidence_id": "E1", "source": "rag", "content": "续航 665 公里", "metadata": {}}]
        get_llm.return_value.invoke.return_value = Mock(content='''{
          "verdicts":[{"claim_id":"C1","verdict":"supported","evidence_ids":["E1"],"reason":"直接支持"}]
        }''')
        state = evidence_validator_node(state)
        self.assertEqual(1.0, state["citation_coverage"])
        self.assertEqual("reflection", route_after_evidence_validation(state))

    @patch("agent.nodes.grounding.ModelConfig.get_synthesize_llm")
    def test_grounded_rewriter_is_bounded(self, get_llm):
        state = self._state()
        state["claims"] = [{"claim_id": "C2", "text": "支持飞行模式"}]
        state["claim_verdicts"] = [{"claim_id": "C2", "verdict": "unsupported", "evidence_ids": [], "reason": "未提及"}]
        state["evidence_catalog"] = []
        get_llm.return_value.invoke.return_value = Mock(content="现有资料仅能确认其 CLTC 续航信息。")
        state = grounded_rewriter_node(state)
        self.assertEqual(1, state["grounding_retry_count"])
        self.assertNotIn("飞行模式", state["draft_answer"])
        state["grounding_status"] = "needs_rewrite"
        self.assertEqual("reflection", route_after_evidence_validation(state))

    def test_direct_response_skips_grounding(self):
        state = self._state()
        state["answer_source"] = "direct_response"
        state = claim_extractor_node(state)
        self.assertEqual("skipped", state["grounding_status"])
        self.assertEqual(1.0, state["citation_coverage"])


if __name__ == "__main__":
    unittest.main()
