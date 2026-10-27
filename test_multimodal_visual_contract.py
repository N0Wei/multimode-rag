#!/usr/bin/env python
# coding: utf-8

import unittest

from langchain_core.documents import Document

from agent.tools.multimodal_rag_tool import MultimodalRAGTool


class MultimodalVisualContractTests(unittest.TestCase):
    def test_visual_request_detection_is_domain_independent(self):
        self.assertTrue(MultimodalRAGTool._is_visual_request("查找安装示意图"))
        self.assertTrue(MultimodalRAGTool._is_visual_request("请给我相关图片"))
        self.assertFalse(MultimodalRAGTool._is_visual_request("如何安装这个部件"))

    def test_visual_answer_exposes_retrieved_image_evidence(self):
        document = Document(
            page_content="方向盘左侧键盘与右侧键盘布局",
            metadata={"doc_type": "image", "topic_title": "方向盘介绍", "pages": [56], "image_path": "assets/p56.png"},
        )
        answer = MultimodalRAGTool._build_visual_evidence_answer([document])
        self.assertIn("方向盘介绍", answer)
        self.assertIn("方向盘左侧键盘", answer)
        self.assertIn("assets/p56.png", answer)


if __name__ == "__main__":
    unittest.main()
