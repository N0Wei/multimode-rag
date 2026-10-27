#!/usr/bin/env python
# coding: utf-8

import inspect
import unittest

from app_gradio import create_gradio_interface
from pathlib import Path


class CitationLayoutTests(unittest.TestCase):
    def test_primary_ui_keeps_citations_inside_conversation_column(self):
        source = inspect.getsource(create_gradio_interface)

        self.assertIn("#customer-workspace-row {", source)
        self.assertIn("display: grid !important;", source)
        self.assertIn("grid-template-columns: minmax(220px", source)
        self.assertIn("#source-reference-panel", source)
        self.assertIn("#citation-panel textarea", source)
        self.assertIn("citation_panel = gr.Textbox(", source)
        self.assertNotIn("citation_panel = gr.HTML(", source)
        self.assertIn('with gr.Accordion("本轮来源引用"', source)
        self.assertIn("contain: layout paint;", source)
        self.assertNotIn("#source-reference-panel *,", source)

    def test_app_launch_enables_manifest_route(self):
        source = Path("app_gradio.py").read_text(encoding="utf-8")
        self.assertIn("pwa=True,", source)


if __name__ == "__main__":
    unittest.main()
