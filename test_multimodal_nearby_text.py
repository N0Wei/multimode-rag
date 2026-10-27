#!/usr/bin/env python
# coding: utf-8
"""Tests for image-nearby text selection in the multimodal PDF splitter."""

import unittest

from multimodal_pdf_parse import LayoutElement, ManualPDFSplitter


class MultimodalNearbyTextTests(unittest.TestCase):
    def element(self, text, bbox, element_type="text", column=0):
        return LayoutElement(
            element_id=f"{element_type}-{column}-{bbox[1]}",
            page_id=1,
            element_type=element_type,
            text=text,
            module="module-1",
            bbox=bbox,
            topic_id="topic-1",
            topic_title="测试主题",
            metadata={"page_width": 600, "column": column},
        )

    def test_nearby_text_uses_only_downstream_same_column_content(self):
        splitter = ManualPDFSplitter("manual.pdf")
        image = LayoutElement(
            element_id="image-1",
            page_id=1,
            element_type="image",
            text="",
            module="module-1",
            bbox=[40, 120, 240, 220],
            topic_id="topic-1",
            topic_title="测试主题",
            metadata={"page_width": 600, "column": 0},
        )
        elements = [
            self.element("上方引导文字不应该进入图片检索文本", [40, 80, 240, 110], column=0),
            image,
            self.element("图片下方说明应该保留", [40, 235, 240, 270], column=0),
            self.element("右栏下方文字不应该混入", [330, 235, 540, 270], column=1),
        ]

        nearby_text = splitter._nearby_text(elements, image)

        self.assertIn("图片下方说明应该保留", nearby_text)
        self.assertNotIn("上方引导文字", nearby_text)
        self.assertNotIn("右栏下方文字", nearby_text)


if __name__ == "__main__":
    unittest.main()
