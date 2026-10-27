#!/usr/bin/env python
# coding: utf-8

import unittest

from agent.nodes.tool_executor import _preserve_rag_evidence


class ToolExecutorEvidenceTests(unittest.TestCase):
    def test_empty_retry_does_not_erase_existing_documents(self):
        original = {"documents": [{"page_content": "evidence"}], "context": "context"}
        state = {"metadata": {"evidence_rag_result": original}}

        _preserve_rag_evidence(state, {"documents": [], "context": ""})

        self.assertIs(state["metadata"]["evidence_rag_result"], original)

    def test_non_empty_retry_updates_evidence_snapshot(self):
        state = {"metadata": {}}
        latest = {"documents": [{"page_content": "latest"}], "context": "latest context"}

        _preserve_rag_evidence(state, latest)

        self.assertIs(state["metadata"]["evidence_rag_result"], latest)

    def test_no_answer_retry_with_documents_keeps_useful_evidence(self):
        original = {"documents": [{"page_content": "useful"}], "is_no_answer": False}
        state = {"metadata": {"evidence_rag_result": original}}

        _preserve_rag_evidence(
            state,
            {"documents": [{"page_content": "irrelevant"}], "is_no_answer": True},
        )

        self.assertIs(state["metadata"]["evidence_rag_result"], original)


if __name__ == "__main__":
    unittest.main()
