#!/usr/bin/env python
# coding: utf-8
"""Tests for image filtering and exact-deduplication in the PDF splitter."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from multimodal_pdf_parse import Chunk, ManualPDFSplitter


class MultimodalImageFilterTests(unittest.TestCase):
    @staticmethod
    def image_chunk(path: Path, page: int, topic: str, nearby: str) -> Chunk:
        return Chunk(
            chunk_id=f"image-{page}",
            doc_type="image",
            chunk_type="image_block",
            page_content="image",
            metadata={
                "image_path": str(path),
                "page_id": page,
                "pages": [page],
                "topic_id": topic,
                "topic_title": topic,
                "nearby_text": nearby,
                "bbox": [0, 0, 100, 100],
            },
        )

    def test_exact_duplicates_are_merged_with_all_contexts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.png"
            second = root / "second.png"
            image = Image.new("RGB", (200, 120), "white")
            ImageDraw.Draw(image).rectangle((20, 20, 180, 100), fill="red")
            image.save(first)
            second.write_bytes(first.read_bytes())

            splitter = ManualPDFSplitter("manual.pdf", asset_dir=str(root / "assets"))
            chunks = splitter.filter_and_deduplicate_image_chunks(
                [
                    self.image_chunk(first, 10, "主题一", "第一段说明"),
                    self.image_chunk(second, 20, "主题二", "第二段说明"),
                ]
            )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["pages"], [10, 20])
        self.assertEqual(chunks[0].metadata["duplicate_count"], 2)
        self.assertIn("第一段说明", chunks[0].metadata["nearby_text"])
        self.assertIn("第二段说明", chunks[0].metadata["nearby_text"])

    def test_low_value_small_image_is_filtered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "small.png"
            image = Image.new("RGB", (34, 34), "white")
            ImageDraw.Draw(image).line((2, 2, 30, 30), fill="black", width=2)
            image.save(path)

            splitter = ManualPDFSplitter("manual.pdf", asset_dir=str(root / "assets"))
            chunks = splitter.filter_and_deduplicate_image_chunks(
                [self.image_chunk(path, 10, "普通主题", "普通附近文字")]
            )

        self.assertEqual(chunks, [])

    def test_indicator_icon_is_kept_even_when_small(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "indicator.png"
            image = Image.new("RGB", (34, 34), "white")
            ImageDraw.Draw(image).ellipse((4, 4, 30, 30), outline="red", width=3)
            image.save(path)

            splitter = ManualPDFSplitter("manual.pdf", asset_dir=str(root / "assets"))
            chunks = splitter.filter_and_deduplicate_image_chunks(
                [self.image_chunk(path, 71, "指示灯和警告灯", "发动机故障警告灯")]
            )

        self.assertEqual(len(chunks), 1)

    def test_low_value_canonical_discards_all_duplicate_occurrences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.png"
            second = root / "second.png"
            image = Image.new("RGB", (34, 34), "white")
            ImageDraw.Draw(image).line((2, 2, 30, 30), fill="black", width=2)
            image.save(first)
            second.write_bytes(first.read_bytes())

            splitter = ManualPDFSplitter("manual.pdf", asset_dir=str(root / "assets"))
            chunks = splitter.filter_and_deduplicate_image_chunks(
                [
                    self.image_chunk(first, 10, "普通主题", "普通附近文字"),
                    self.image_chunk(second, 71, "指示灯和警告灯", "警告灯"),
                ]
            )

        self.assertEqual(chunks, [])


if __name__ == "__main__":
    unittest.main()
