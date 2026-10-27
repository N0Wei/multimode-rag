#!/usr/bin/env python
# coding: utf-8
"""
TOC-aware multimodal PDF splitter for the Lynk & Co user manual style.

The original project treats the PDF as plain text. This splitter follows the
actual structure of `train_a.pdf`:

1. Parse the table of contents pages and build topic spans.
2. Treat page headers such as "仪表和灯光" as module metadata, not section cuts.
3. Assign text/table/image elements to TOC topics.
4. Keep short topics as text parents and semantically split only long topics.
5. Build overlapping searchable children from each non-indexed text parent.
6. Save image blocks and bind them to nearby text in the same topic.

Example:
    python multimodal_pdf_parse.py --pdf data/train_a.pdf --out data/mm_chunks.json --preview data/mm_chunks.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageStat


TOC_LINE_RE = re.compile(r"^(.+?)[\.\u2026·\s]{3,}(\d{1,3})$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])")
NUMBERED_ITEM_RE = re.compile(r"^(0?\d{1,2}|[一二三四五六七八九十]+)[、.．\s]")
NOTICE_RE = re.compile(r"^(警告！|注意！|说明！|提示信息)$")
LIST_ITEM_RE = re.compile(r"^[■□●◆•·]")
TERMINAL_PUNCTUATION_RE = re.compile(r"[。！？!?；;：:]$")

SEMANTIC_SMALL_TOPICS = {"指示灯和警告灯", "查看组合仪表信息"}
SEMANTIC_SMALL_KEYWORDS = ("指示灯", "警告灯", "故障灯", "按钮", "按键", "图标", "开关")

KNOWN_MODULES = {
    "前言",
    "用车前准备",
    "安全出行",
    "上车和下车",
    "驾驶前的准备",
    "仪表和灯光",
    "空调",
    "装载货物",
    "启动和驾驶",
    "驾驶辅助",
    "泊车",
    "中央显示屏",
    "Lynk & Co App",
    "高压系统",
    "保养和维护",
    "紧急情况下",
    "技术资料",
    "OTA升级",
}


@dataclass
class TocEntry:
    toc_id: str
    title: str     #章节标题
    start_page: int
    end_page: int
    module: str = ""   #目录大主题


@dataclass
class LayoutElement:
    element_id: str
    element_type: str
    text: str
    page_id: int
    module: str
    topic_id: str
    topic_title: str
    bbox: Optional[List[float]] = None
    image_path: Optional[str] = None
    table: Optional[List[List[str]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopicSection:
    topic_id: str
    title: str
    module: str
    pages: List[int]
    elements: List[LayoutElement]


@dataclass
class Chunk:
    chunk_id: str
    doc_type: str
    chunk_type: str
    page_content: str
    metadata: Dict[str, Any]


@dataclass
class SemanticUnit:
    """A small ordered text unit that should not be split when building parents."""

    texts: List[str]
    pages: List[int]
    kind: str

    @property
    def content(self) -> str:
        return "\n".join(text for text in self.texts if text).strip()


class ManualPDFSplitter:
    def __init__(
        self,
        pdf_path: str,
        asset_dir: str = "data/multimodal_assets",
        toc_pages: str = "3-7",
        child_chunk_size: int = 450,
        child_overlap: int = 80,
        parent_chunk_size: int = 1500,
        nearby_window: int = 4,
    ):
        self.pdf_path = Path(pdf_path)
        self.asset_dir = Path(asset_dir)
        self.toc_page_range = self._parse_page_range(toc_pages)
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap
        self.parent_chunk_size = parent_chunk_size
        self.nearby_window = nearby_window
        self.asset_dir.mkdir(parents=True, exist_ok=True)

    def split(self) -> List[Chunk]:
        doc = self._open_doc()
        toc_entries = self.extract_toc_entries(doc)
        if not toc_entries:
            toc_entries = [
                TocEntry(
                    toc_id=self._stable_id(f"{self.pdf_path.stem}:1:{doc.page_count}"),
                    title=self.pdf_path.stem,
                    start_page=1,
                    end_page=doc.page_count,
                    module="",
                )
            ]
        elements = self.extract_elements(doc, toc_entries)
        sections = self.build_topic_sections(elements, toc_entries)

        chunks: List[Chunk] = []
        for section in sections:
            chunks.extend(self.build_text_chunks(section))
            chunks.extend(self.build_table_chunks(section))
            chunks.extend(self.build_image_chunks(section))
        return self.filter_and_deduplicate_image_chunks(chunks)

    def extract_toc_entries(self, doc) -> List[TocEntry]:
        raw_entries: List[Tuple[int, str, int]] = []
        order = 0
        for page_id in self.toc_page_range:
            if page_id < 1 or page_id > doc.page_count:
                continue
            lines = self._page_lines(doc[page_id - 1])
            for line in lines:
                match = TOC_LINE_RE.match(line)
                if not match:
                    continue
                title = self._normalize_text(match.group(1))
                page = int(match.group(2))
                if title and title != "目录" and page > 0:
                    raw_entries.append((order, title, page))
                    order += 1

        # Keep duplicate titles if they point to different pages; the manual has
        # repeated topics such as "指示灯和警告灯".
        deduped: List[Tuple[int, str, int]] = []
        seen = set()
        for item_order, title, page in raw_entries:
            key = (title, page)
            if key not in seen:
                seen.add(key)
                deduped.append((item_order, title, page))

        deduped.sort(key=lambda item: (item[2], item[0]))
        entries: List[TocEntry] = []
        for idx, (_, title, start_page) in enumerate(deduped):
            next_start = deduped[idx + 1][2] if idx + 1 < len(deduped) else doc.page_count + 1
            end_page = max(start_page, next_start - 1)
            toc_id = self._stable_id(f"{title}:{start_page}:{idx}")
            entries.append(TocEntry(toc_id=toc_id, title=title, start_page=start_page, end_page=end_page))
        return entries

    def extract_elements(self, doc, toc_entries: Sequence[TocEntry]) -> List[LayoutElement]:
        elements: List[LayoutElement] = []
        for page_index in range(doc.page_count):
            page_id = page_index + 1
            page = doc[page_index]
            lines = self._page_lines(page)
            module = self._detect_module(lines)
            topic = self._topic_for_page(page_id, toc_entries)
            if topic is None:
                continue

            current_topic = topic
            local_topic_titles = {entry.title: entry for entry in toc_entries if entry.start_page == page_id}
            # Codex-TABLE-ADD: locate real tables before text extraction so
            # table cells are not indexed again as ordinary text blocks.
            table_elements = self._extract_table_elements(
                page, page_id, module, current_topic, local_topic_titles
            )
            elements.extend(
                self._extract_text_elements(
                    page,
                    page_id,
                    module,
                    current_topic,
                    local_topic_titles,
                    excluded_bboxes=[item.bbox for item in table_elements if item.bbox],
                )
            )
            elements.extend(table_elements)
            elements.extend(self._extract_image_elements(page, page_id, module, current_topic))

        elements.sort(key=self._reading_order_key)
        return elements

    def build_topic_sections(self, elements: Sequence[LayoutElement], toc_entries: Sequence[TocEntry]) -> List[TopicSection]:
        by_topic: Dict[str, List[LayoutElement]] = {}
        for element in elements:
            by_topic.setdefault(element.topic_id, []).append(element)

        entry_map = {entry.toc_id: entry for entry in toc_entries}
        sections: List[TopicSection] = []
        for topic_id, topic_elements in by_topic.items():
            entry = entry_map.get(topic_id)
            if entry is None:
                continue
            pages = sorted({item.page_id for item in topic_elements})
            module = self._first_non_empty([item.module for item in topic_elements])
            sections.append(
                TopicSection(
                    topic_id=topic_id,
                    title=entry.title,
                    module=module,
                    pages=pages,
                    elements=topic_elements,
                )
            )
        sections.sort(key=lambda section: (section.pages[0] if section.pages else 9999, section.title))
        return sections

    def build_text_chunks(self, section: TopicSection) -> List[Chunk]:
        text_elements = [item for item in section.elements if item.element_type == "text"]
        if not text_elements:
            return []

        semantic_units = self._build_semantic_units(text_elements)
        parent_units = self._build_text_parent_units(semantic_units)
        chunks: List[Chunk] = []
        for parent_index, units in enumerate(parent_units):
            # Preserve semantic-unit line breaks in the parent context shown to
            # the LLM; searchable children are normalized separately.
            parent_content = "\n".join(
                unit.content for unit in units if unit.content
            ).strip()
            if not parent_content:
                continue
            parent_id = f"{section.topic_id}_text_parent_{parent_index:03d}"
            parent_pages = sorted({page for unit in units for page in unit.pages})
            chunks.append(
                Chunk(
                    chunk_id=parent_id,
                    doc_type="text",
                    chunk_type="text_parent",
                    page_content=parent_content,
                    metadata={
                        "source_pdf": str(self.pdf_path),
                        "module": section.module,
                        "topic_id": section.topic_id,
                        "topic_title": section.title,
                        "pages": parent_pages,
                        "indexable": False,
                    },
                )
            )

            children = self._split_text(
                parent_content,
                self.child_chunk_size,
                self.child_overlap,
            )
            for child_index, text in enumerate(children):
                chunks.append(
                    Chunk(
                        chunk_id=f"{parent_id}_child_{child_index:03d}",
                        doc_type="text",
                        chunk_type="text_child",
                        page_content=text,
                        metadata={
                            "source_pdf": str(self.pdf_path),
                            "module": section.module,
                            "topic_id": section.topic_id,
                            "topic_title": section.title,
                            "pages": parent_pages,
                            "parent_id": parent_id,
                            "parent_index": parent_index,
                            "child_index": child_index,
                        },
                    )
                )
        return chunks

    def build_table_chunks(self, section: TopicSection) -> List[Chunk]:
        chunks: List[Chunk] = []
        table_elements = [item for item in section.elements if item.element_type == "table"]
        for idx, element in enumerate(table_elements):
            rows = element.table or []
            if len(rows) < 2:
                continue
            parent_id = f"{section.topic_id}_table_{idx:03d}_parent"
            parent_content = self._table_to_markdown(rows)
            if not parent_content:
                continue

            # Codex-TABLE-ADD: the complete table is a non-indexed parent.
            chunks.append(
                Chunk(
                    chunk_id=parent_id,
                    doc_type="table",
                    chunk_type="table_parent",
                    page_content=parent_content,
                    metadata={
                        "source_pdf": str(self.pdf_path),
                        "module": section.module,
                        "topic_id": section.topic_id,
                        "topic_title": section.title,
                        "pages": [element.page_id],
                        "page_id": element.page_id,
                        "bbox": element.bbox,
                        "indexable": False,
                        "table_strategy": element.metadata.get("table_strategy", ""),
                    },
                )
            )

            # Codex-TABLE-ADD: one data row becomes one searchable child.
            headers, data_rows = self._table_headers_and_data_rows(rows)
            for row_index, row in enumerate(data_rows, start=1):
                row_content = self._table_row_to_text(section.title, headers, row)
                if not row_content:
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=f"{section.topic_id}_table_{idx:03d}_row_{row_index:03d}",
                        doc_type="table",
                        chunk_type="table_row_child",
                        page_content=row_content,
                        metadata={
                            "source_pdf": str(self.pdf_path),
                            "module": section.module,
                            "topic_id": section.topic_id,
                            "topic_title": section.title,
                            "pages": [element.page_id],
                            "page_id": element.page_id,
                            "bbox": element.bbox,
                            "parent_id": parent_id,
                            "row_index": row_index,
                            "table_strategy": element.metadata.get("table_strategy", ""),
                        },
                    )
                )
        return chunks

    def build_image_chunks(self, section: TopicSection) -> List[Chunk]:
        chunks: List[Chunk] = []
        images = [item for item in section.elements if item.element_type == "image"]
        for idx, image in enumerate(images):
            nearby_text = self._nearby_text(section.elements, image)
            content = self._normalize_text(
                "\n".join(
                    [
                        f"图片证据，位于《{section.title}》主题，第{image.page_id}页。",
                        f"附近文本：{nearby_text}" if nearby_text else "附近文本：未提取到。",
                    ]
                )
            )
            chunks.append(
                Chunk(
                    chunk_id=f"{section.topic_id}_image_{idx:03d}",
                    doc_type="image",
                    chunk_type="image_block",
                    page_content=content,
                    metadata={
                        "source_pdf": str(self.pdf_path),
                        "module": section.module,
                        "topic_id": section.topic_id,
                        "topic_title": section.title,
                        "pages": [image.page_id],
                        "page_id": image.page_id,
                        "bbox": image.bbox,
                        "image_path": image.image_path,
                        "nearby_text": nearby_text,
                    },
                )
            )
        return chunks

    def filter_and_deduplicate_image_chunks(self, chunks: Sequence[Chunk]) -> List[Chunk]:
        """Remove low-value images and merge exact duplicates into one evidence chunk."""
        non_images = [chunk for chunk in chunks if chunk.doc_type != "image"]
        image_chunks = [chunk for chunk in chunks if chunk.doc_type == "image"]
        inspected = []
        hash_counts: Dict[str, int] = {}

        for chunk in image_chunks:
            info = self._inspect_image_chunk(chunk)
            if not info:
                continue
            inspected.append((chunk, info))
            image_hash = info["sha256"]
            hash_counts[image_hash] = hash_counts.get(image_hash, 0) + 1

        raw_groups: Dict[str, List[Tuple[Chunk, Dict[str, Any]]]] = {}
        for chunk, info in inspected:
            raw_groups.setdefault(info["sha256"], []).append((chunk, info))

        grouped: Dict[str, List[Tuple[Chunk, Dict[str, Any]]]] = {}
        filtered_counts: Dict[str, int] = {}
        for image_hash, items in raw_groups.items():
            canonical_chunk, canonical_info = items[0]
            reason = self._image_discard_reason(
                canonical_chunk,
                canonical_info,
                hash_counts[image_hash],
            )
            if reason:
                filtered_counts[reason] = filtered_counts.get(reason, 0) + len(items)
                continue
            grouped[image_hash] = items

        deduplicated = [self._merge_duplicate_image_chunks(items) for items in grouped.values()]
        deduplicated.sort(
            key=lambda chunk: (
                min(chunk.metadata.get("pages") or [9999]),
                chunk.metadata.get("topic_title", ""),
                chunk.chunk_id,
            )
        )
        print(
            "  [images] "
            f"extracted={len(image_chunks)}, kept_unique={len(deduplicated)}, "
            f"duplicates_merged={sum(max(0, len(items) - 1) for items in grouped.values())}, "
            f"filtered={sum(filtered_counts.values())}"
        )
        return non_images + deduplicated

    def _inspect_image_chunk(self, chunk: Chunk) -> Optional[Dict[str, Any]]:
        image_path = Path(chunk.metadata.get("image_path") or "")
        if not image_path.exists():
            return None
        try:
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
                visual_range = max(
                    high - low for low, high in ImageStat.Stat(image.convert("RGB")).extrema
                )
                alpha_empty = image.mode == "RGBA" and image.getchannel("A").getbbox() is None
            return {
                "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "width": width,
                "height": height,
                "area": width * height,
                "bytes": image_path.stat().st_size,
                "visual_range": visual_range,
                "alpha_empty": alpha_empty,
            }
        except Exception:
            return None

    def _image_discard_reason(self, chunk: Chunk, info: Dict[str, Any], duplicate_count: int) -> str:
        if info["alpha_empty"] or info["visual_range"] <= 3:
            return "blank_or_nearly_blank"
        if (
            info["width"] == 54
            and info["height"] == 54
            and info["bytes"] <= 500
            and duplicate_count >= 3
        ):
            return "generic_notice_symbol"

        topic = str(chunk.metadata.get("topic_title", ""))
        nearby_text = str(chunk.metadata.get("nearby_text", ""))
        semantic_small = (
            topic in SEMANTIC_SMALL_TOPICS
            or any(keyword in nearby_text for keyword in SEMANTIC_SMALL_KEYWORDS)
        )
        if (
            not semantic_small
            and (info["width"] < 80 or info["height"] < 80 or info["area"] < 10000)
        ):
            return "low_value_small_icon"
        return ""

    def _merge_duplicate_image_chunks(
        self,
        items: Sequence[Tuple[Chunk, Dict[str, Any]]],
        max_context_chars: int = 2400,
    ) -> Chunk:
        canonical, image_info = items[0]
        pages = sorted({
            int(page)
            for chunk, _ in items
            for page in (chunk.metadata.get("pages") or [chunk.metadata.get("page_id")])
            if page is not None
        })
        topic_titles = list(dict.fromkeys(
            str(chunk.metadata.get("topic_title", ""))
            for chunk, _ in items
            if chunk.metadata.get("topic_title")
        ))
        nearby_texts = list(dict.fromkeys(
            str(chunk.metadata.get("nearby_text", "")).strip()
            for chunk, _ in items
            if str(chunk.metadata.get("nearby_text", "")).strip()
        ))
        merged_nearby = ""
        for text in nearby_texts:
            addition = ("\n" if merged_nearby else "") + text
            if len(merged_nearby) + len(addition) > max_context_chars:
                remaining = max_context_chars - len(merged_nearby)
                if remaining > 80:
                    merged_nearby += addition[:remaining]
                break
            merged_nearby += addition

        metadata = dict(canonical.metadata)
        metadata.update(
            {
                "pages": pages,
                "page_id": pages[0] if pages else metadata.get("page_id"),
                "topic_titles": topic_titles,
                "nearby_text": merged_nearby,
                "duplicate_count": len(items),
                "image_sha256": image_info["sha256"],
                "image_width": image_info["width"],
                "image_height": image_info["height"],
                "occurrences": [
                    {
                        "page_id": chunk.metadata.get("page_id"),
                        "topic_id": chunk.metadata.get("topic_id"),
                        "topic_title": chunk.metadata.get("topic_title"),
                        "bbox": chunk.metadata.get("bbox"),
                    }
                    for chunk, _ in items
                ],
            }
        )
        topics_text = "、".join(topic_titles)
        pages_text = "、".join(str(page) for page in pages)
        canonical.page_content = self._normalize_text(
            f"图片证据，出现于第{pages_text}页，关联主题：{topics_text}。"
            + (f"\n关联文本：{merged_nearby}" if merged_nearby else "")
        )
        canonical.metadata = metadata
        canonical.chunk_id = f"image_{image_info['sha256'][:16]}"
        canonical.chunk_type = "deduplicated_image_block"
        return canonical

    def save_outputs(self, chunks: Sequence[Chunk], out_path: str, preview_path: Optional[str] = None):
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("w", encoding="utf-8") as f:
            json.dump([asdict(chunk) for chunk in chunks], f, ensure_ascii=False, indent=2)

        if preview_path:
            preview_file = Path(preview_path)
            preview_file.parent.mkdir(parents=True, exist_ok=True)
            preview_file.write_text(self.render_markdown_preview(chunks), encoding="utf-8")

    def render_markdown_preview(self, chunks: Sequence[Chunk], max_chunks: int = 180) -> str:
        counts: Dict[str, int] = {}
        for chunk in chunks:
            key = f"{chunk.doc_type}/{chunk.chunk_type}"
            counts[key] = counts.get(key, 0) + 1

        lines = [
            "# TOC-Aware Multimodal Split Preview",
            "",
            f"- Source: `{self.pdf_path}`",
            f"- Total chunks: {len(chunks)}",
            "",
            "## Counts",
        ]
        for key, value in sorted(counts.items()):
            lines.append(f"- {key}: {value}")

        lines.extend(["", "## Samples"])
        for chunk in chunks[:max_chunks]:
            meta = chunk.metadata
            lines.extend(
                [
                    "",
                    f"### {chunk.chunk_id}",
                    "",
                    f"- type: `{chunk.doc_type}` / `{chunk.chunk_type}`",
                    f"- module: `{meta.get('module', '')}`",
                    f"- topic: `{meta.get('topic_title', '')}`",
                    f"- pages: `{meta.get('pages', '')}`",
                    f"- image: `{meta.get('image_path', '')}`" if chunk.doc_type == "image" else "",
                    "",
                    "```text",
                    chunk.page_content[:1200],
                    "```",
                ]
            )
        return "\n".join(line for line in lines if line is not None)

    def _open_doc(self):
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required. Install it with `pip install PyMuPDF`.") from exc
        return fitz.open(str(self.pdf_path))

    def _extract_text_elements(
        self,
        page,
        page_id: int,
        module: str,
        topic: TocEntry,
        local_topic_titles: Dict[str, TocEntry],
        excluded_bboxes: Optional[Sequence[Sequence[float]]] = None,
    ) -> List[LayoutElement]:
        elements: List[LayoutElement] = []
        current_topic = topic
        line_idx = 0
        page_width = float(page.rect.width)
        for block in page.get_text("blocks"):
            if len(block) < 5:
                continue
            x0, y0, x1, y1, raw_text = block[:5]
            block_type = block[6] if len(block) > 6 else 0
            if block_type != 0:
                continue
            text = self._normalize_text(raw_text)
            if self._skip_line(text, module):
                continue
            bbox = [float(x0), float(y0), float(x1), float(y1)]
            if any(self._bbox_overlap_ratio(bbox, excluded) >= 0.5 for excluded in (excluded_bboxes or [])):
                continue
            matched_topic = local_topic_titles.get(text)
            if matched_topic:
                current_topic = matched_topic
            elements.append(
                LayoutElement(
                    element_id=f"p{page_id:03d}_text_{line_idx:03d}",
                    element_type="text",
                    text=text,
                    page_id=page_id,
                    module=module,
                    topic_id=current_topic.toc_id,
                    topic_title=current_topic.title,
                    bbox=bbox,
                    metadata={
                        "page_width": page_width,
                        "column": self._column_for_bbox(bbox, page_width),
                    },
                )
            )
            line_idx += 1
        return elements

    def _extract_table_elements(
        self,
        page,
        page_id: int,
        module: str,
        topic: TocEntry,
        local_topic_titles: Dict[str, TocEntry],
    ) -> List[LayoutElement]:
        """Extract structurally valid tables instead of classifying a whole page."""
        candidates: List[Tuple[List[float], List[List[str]], str]] = []

        # Codex-TABLE-ADD: ruled tables are the reliable primary path.
        for table in page.find_tables(strategy="lines").tables:
            bbox = [float(value) for value in table.bbox]
            rows = self._clean_table_rows(table.extract())
            candidates.append((bbox, rows, "lines"))

        page_width = float(page.rect.width)

        elements: List[LayoutElement] = []
        for table_index, (bbox, rows, strategy) in enumerate(candidates):
            if not self._is_valid_table(rows):
                continue
            table_topic = self._topic_for_table_position(page, bbox, topic, local_topic_titles)
            elements.append(
                LayoutElement(
                    element_id=f"p{page_id:03d}_table_{table_index:03d}",
                    element_type="table",
                    text=self._table_to_markdown(rows),
                    table=rows,
                    page_id=page_id,
                    module=module,
                    topic_id=table_topic.toc_id,
                    topic_title=table_topic.title,
                    bbox=bbox,
                    metadata={
                        "page_width": page_width,
                        "column": self._column_for_bbox(bbox, page_width),
                        "table_strategy": strategy,
                    },
                )
            )
        return elements

        # PyMuPDF cannot reliably recover ruled tables here without additional
        # dependencies. This heuristic captures parameter pages where each value
        # is extracted as a short line, such as "长度 / 4549 / 高度 / 1689".
    def _extract_image_elements(self, page, page_id: int, module: str, topic: TocEntry) -> List[LayoutElement]:
        elements: List[LayoutElement] = []
        blocks = page.get_text("dict").get("blocks", [])
        page_width = float(page.rect.width)
        image_idx = 0
        for block in blocks:
            if block.get("type") != 1:
                continue
            image_path = self._save_image_block(block, page_id, image_idx)
            bbox = [float(v) for v in block.get("bbox", [])]
            elements.append(
                LayoutElement(
                    element_id=f"p{page_id:03d}_image_{image_idx:03d}",
                    element_type="image",
                    text=f"Image on page {page_id}",
                    page_id=page_id,
                    module=module,
                    topic_id=topic.toc_id,
                    topic_title=topic.title,
                    bbox=bbox,
                    image_path=image_path,
                    metadata={
                        "page_width": page_width,
                        "column": self._column_for_bbox(bbox, page_width),
                    },
                )
            )
            image_idx += 1
        return elements

    def _save_image_block(self, block: Dict[str, Any], page_id: int, image_idx: int) -> Optional[str]:
        image_bytes = block.get("image")
        ext = block.get("ext", "png")
        if not image_bytes:
            return None
        filename = f"{self.pdf_path.stem}_p{page_id:03d}_img_{image_idx:03d}.{ext}"
        path = self.asset_dir / filename
        try:
            path.write_bytes(image_bytes)
            return str(path)
        except Exception:
            return None

    def _topic_for_page(self, page_id: int, toc_entries: Sequence[TocEntry]) -> Optional[TocEntry]:
        candidates = [entry for entry in toc_entries if entry.start_page <= page_id <= entry.end_page]
        if candidates:
            return candidates[-1]
        before = [entry for entry in toc_entries if entry.start_page <= page_id]
        return before[-1] if before else None

    def _detect_module(self, lines: Sequence[str]) -> str:
        for line in lines[:6]:
            if line in KNOWN_MODULES:
                return line
        return ""

    def _page_lines(self, page) -> List[str]:
        return [self._normalize_text(line) for line in page.get_text("text").splitlines() if self._normalize_text(line)]

    def _skip_line(self, text: str, module: str) -> bool:
        if not text:
            return True
        if text == module:
            return True
        if text.isdigit():
            return True
        if text == "目录":
            return True
        if "................" in text:
            return True
        return False

    def _group_manual_text_units(self, elements: Sequence[LayoutElement]) -> List[str]:
        """Legacy-compatible view of the semantic units used for parent building."""
        return [unit.content for unit in self._build_semantic_units(elements) if unit.content]

    def _build_semantic_units(self, elements: Sequence[LayoutElement]) -> List[SemanticUnit]:
        """Group ordered PDF text blocks into indivisible parent-building units.

        The rules intentionally stay conservative: headings and notice blocks
        start units, while ordinary paragraphs and their following lists remain
        together. Misclassification therefore changes a parent boundary but
        never drops text.
        """
        units: List[SemanticUnit] = []
        current: Optional[SemanticUnit] = None
        for element in sorted(elements, key=self._reading_order_key):
            text = self._normalize_text(element.text)
            if not text:
                continue
            kind = self._semantic_kind(text)
            starts_new = self._starts_new_semantic_unit(kind, current)
            if starts_new and current and current.content:
                units.append(current)
                current = None
            if current is None:
                current = SemanticUnit(texts=[], pages=[], kind=kind)
            current.texts.append(text)
            if element.page_id not in current.pages:
                current.pages.append(element.page_id)
        if current and current.content:
            units.append(current)
        return units

    def _semantic_kind(self, text: str) -> str:
        if NOTICE_RE.match(text):
            return "notice"
        if NUMBERED_ITEM_RE.match(text):
            return "step"
        if LIST_ITEM_RE.match(text):
            return "list"
        if self._looks_like_subtitle(text):
            return "subtitle"
        return "paragraph"

    @staticmethod
    def _starts_new_semantic_unit(
        kind: str,
        current: Optional[SemanticUnit],
    ) -> bool:
        if current is None:
            return False
        if kind in {"subtitle", "notice"}:
            return True
        if kind == "step":
            return current.kind not in {"subtitle", "step"}
        if kind == "paragraph" and current.kind in {"notice", "step", "list"}:
            return True
        return False

    @staticmethod
    def _looks_like_subtitle(text: str) -> bool:
        """Recognize likely manual subtitles without making every short line a cut."""
        if len(text) > 28 or "\n" in text:
            return False
        if TERMINAL_PUNCTUATION_RE.search(text):
            return False
        if re.search(r"[，,。！？!?；;]", text):
            return False
        if re.search(r"\d", text) and len(text) > 16:
            return False
        return len(text) >= 2

    def _build_text_parent_units(
        self,
        semantic_units: Sequence[SemanticUnit],
    ) -> List[List[SemanticUnit]]:
        """Keep a short topic whole; split a long topic only at semantic boundaries."""
        usable_units = [unit for unit in semantic_units if unit.content]
        if not usable_units:
            return []
        total_length = sum(len(unit.content) for unit in usable_units)
        if total_length <= self.parent_chunk_size:
            return [usable_units]

        expanded_units: List[SemanticUnit] = []
        for unit in usable_units:
            expanded_units.extend(self._split_oversized_semantic_unit(unit))

        parents: List[List[SemanticUnit]] = []
        current: List[SemanticUnit] = []
        current_length = 0
        for unit in expanded_units:
            unit_length = len(unit.content)
            if current and current_length + unit_length > self.parent_chunk_size:
                parents.append(current)
                current = []
                current_length = 0
            current.append(unit)
            current_length += unit_length
        if current:
            parents.append(current)
        if len(parents) > 1:
            final_length = sum(len(unit.content) for unit in parents[-1])
            if final_length < 300:
                parents[-2].extend(parents[-1])
                parents.pop()
        return parents

    def _split_oversized_semantic_unit(self, unit: SemanticUnit) -> List[SemanticUnit]:
        """Fallback for a single very long paragraph/list while preserving sentences."""
        if len(unit.content) <= self.parent_chunk_size:
            return [unit]
        pieces = self._split_text(unit.content, self.parent_chunk_size, overlap=0)
        return [
            SemanticUnit(texts=[piece], pages=list(unit.pages), kind=unit.kind)
            for piece in pieces
            if piece
        ]

    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        text = self._normalize_text(text)
        if not text:
            return []
        if len(text) <= chunk_size:
            return [text]
        pieces = [piece for piece in SENTENCE_SPLIT_RE.split(text) if piece]
        if len(pieces) <= 1:
            pieces = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        #滑动窗口
        chunks: List[str] = []
        current = ""
        for piece in pieces:
            if current and len(current) + len(piece) > chunk_size:
                chunks.append(current.strip())
                overlap_text = current[-overlap:] if overlap > 0 else ""
                current = overlap_text + piece
            else:
                current += piece
        if current:
            chunks.append(current.strip())
        return chunks

    def _nearby_text(self, elements: Sequence[LayoutElement], image: LayoutElement) -> str:
        if not image.bbox:
            return ""

        image_column = image.metadata.get("column")
        if image_column is None:
            page_width = float(image.metadata.get("page_width", 0.0))
            image_column = self._column_for_bbox(image.bbox, page_width)

        candidates = [
            item
            for item in elements
            if item.page_id == image.page_id
            and item.topic_id == image.topic_id
            and item.element_type in {"text", "table"}
            and item.bbox
            and item.text.strip()
            and item.bbox[1] >= image.bbox[3]
            and int(item.metadata.get("column", self._column_for_bbox(item.bbox, float(item.metadata.get("page_width", 0.0)))))
            == int(image_column)
        ]
        scored = sorted(
            ((self._downstream_text_score(image, item), item) for item in candidates),
            key=lambda pair: pair[0],
        )
        nearby = [item for score, item in scored[: self.nearby_window] if score < 2000]
        nearby.sort(key=self._reading_order_key)
        return self._normalize_text("\n".join(item.text for item in nearby))[:1000]

    def _downstream_text_score(self, image: LayoutElement, candidate: LayoutElement) -> float:
        ix0, _, ix1, iy1 = image.bbox
        tx0, ty0, tx1, _ = candidate.bbox

        vertical_gap = max(0.0, ty0 - iy1)
        horizontal_overlap = max(0.0, min(ix1, tx1) - max(ix0, tx0))
        min_width = max(1.0, min(ix1 - ix0, tx1 - tx0))
        overlap_ratio = horizontal_overlap / min_width
        overlap_penalty = (1.0 - min(overlap_ratio, 1.0)) * 300.0

        image_center_x = (ix0 + ix1) / 2
        text_center_x = (tx0 + tx1) / 2
        horizontal_center_gap = abs(image_center_x - text_center_x) * 0.15

        return vertical_gap + overlap_penalty + horizontal_center_gap

    def _reading_order_key(self, element: LayoutElement) -> Tuple[int, int, float, float]:
        if not element.bbox:
            return element.page_id, 2, float("inf"), float("inf")
        page_width = float(element.metadata.get("page_width", 0.0))
        column = element.metadata.get("column")
        if column is None:
            column = self._column_for_bbox(element.bbox, page_width)
        return element.page_id, int(column), float(element.bbox[1]), float(element.bbox[0])

    def _column_for_bbox(self, bbox: Sequence[float], page_width: float) -> int:
        if len(bbox) < 4 or page_width <= 0:
            return 0
        x0, _, x1, _ = bbox
        center_x = (float(x0) + float(x1)) / 2
        return 0 if center_x < page_width / 2 else 1

    def _spatial_nearby_score(self, image: LayoutElement, candidate: LayoutElement) -> float:
        """Lower is better. Prefer same-column text immediately above/below the image."""
        ix0, iy0, ix1, iy1 = image.bbox
        tx0, ty0, tx1, ty1 = candidate.bbox

        image_column = image.metadata.get("column", 0)
        text_column = candidate.metadata.get("column", 0)
        column_penalty = 0.0 if image_column == text_column else 2000.0

        horizontal_overlap = max(0.0, min(ix1, tx1) - max(ix0, tx0))
        min_width = max(1.0, min(ix1 - ix0, tx1 - tx0))
        overlap_ratio = horizontal_overlap / min_width
        overlap_penalty = (1.0 - min(overlap_ratio, 1.0)) * 500.0

        if ty1 < iy0:
            vertical_gap = iy0 - ty1
        elif ty0 > iy1:
            vertical_gap = ty0 - iy1
        else:
            vertical_gap = 0.0

        image_center_x = (ix0 + ix1) / 2
        text_center_x = (tx0 + tx1) / 2
        horizontal_center_gap = abs(image_center_x - text_center_x) * 0.25

        return column_penalty + overlap_penalty + vertical_gap + horizontal_center_gap

    def _looks_like_parameter_page(self, lines: Sequence[str]) -> bool:
        if len(lines) < 8:
            return False
        short_lines = sum(1 for line in lines if len(line) <= 18)
        numeric_lines = sum(1 for line in lines if re.search(r"\d", line))
        keywords = sum(1 for line in lines if any(key in line for key in ["参数", "单位", "长度", "宽度", "高度", "质量", "车型", "术语", "说明"]))
        return short_lines / max(len(lines), 1) > 0.55 and numeric_lines >= 3 and keywords >= 2

    def _parameter_lines_to_markdown(self, lines: Sequence[str]) -> str:
        clean = [line for line in lines if not self._skip_line(line, "")]
        if not clean:
            return ""
        rows = ["| 项目 | 内容 |", "| --- | --- |"]
        pending_key = ""
        for line in clean:
            if pending_key:
                rows.append(f"| {pending_key} | {line} |")
                pending_key = ""
            elif len(line) <= 12 and not re.fullmatch(r"[\d（）().a-zA-Z/-]+", line):
                pending_key = line
            else:
                rows.append(f"| 说明 | {line} |")
        if pending_key:
            rows.append(f"| 说明 | {pending_key} |")
        return "\n".join(rows)

    def _clean_table_rows(self, rows: Sequence[Sequence[Any]]) -> List[List[str]]:
        cleaned: List[List[str]] = []
        for row in rows or []:
            values = [self._normalize_text(str(cell or "")) for cell in row]
            if any(values):
                cleaned.append(values)
        if not cleaned:
            return []
        width = max(len(row) for row in cleaned)
        return [row + [""] * (width - len(row)) for row in cleaned]

    def _is_valid_table(self, rows: Sequence[Sequence[str]]) -> bool:
        if len(rows) < 2:
            return False
        width = max((len(row) for row in rows), default=0)
        if width < 2:
            return False
        total_cells = len(rows) * width
        non_empty = sum(1 for row in rows for cell in row if str(cell).strip())
        multi_cell_rows = sum(
            1 for row in rows if sum(1 for cell in row if str(cell).strip()) >= 2
        )
        if non_empty / max(total_cells, 1) < 0.45:
            return False
        return multi_cell_rows >= 1

    def _normalize_table_headers(self, header_row: Sequence[str]) -> List[str]:
        headers: List[str] = []
        for index, value in enumerate(header_row):
            header = self._normalize_text(str(value or ""))
            headers.append(header or f"第{index + 1}列")
        return headers

    def _table_headers_and_data_rows(
        self, rows: Sequence[Sequence[str]]
    ) -> Tuple[List[str], List[Sequence[str]]]:
        first_row = rows[0]
        # Numbered instruction tables often have no explicit header. Preserve
        # their first row by assigning generic column names.
        if any(re.fullmatch(r"\d+[.)]?", str(cell).strip()) for cell in first_row):
            return [f"第{index + 1}列" for index in range(len(first_row))], list(rows)
        return self._normalize_table_headers(first_row), list(rows[1:])

    def _table_row_to_text(
        self, topic_title: str, headers: Sequence[str], row: Sequence[str]
    ) -> str:
        pairs = []
        for index, value in enumerate(row):
            value = self._normalize_text(str(value or ""))
            if not value:
                continue
            header = headers[index] if index < len(headers) else f"第{index + 1}列"
            pairs.append(f"{header}：{value}")
        if not pairs:
            return ""
        return "\n".join([f"表格：{topic_title}", *pairs])

    def _table_to_markdown(self, rows: Sequence[Sequence[str]]) -> str:
        if not rows:
            return ""
        width = max(len(row) for row in rows)

        def escaped(row: Sequence[str]) -> List[str]:
            values = list(row) + [""] * (width - len(row))
            return [self._normalize_text(str(value or "")).replace("|", "\\|") for value in values]

        header = escaped(rows[0])
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * width) + " |",
        ]
        lines.extend("| " + " | ".join(escaped(row)) + " |" for row in rows[1:])
        return "\n".join(lines)

    def _bbox_overlap_ratio(self, first: Sequence[float], second: Sequence[float]) -> float:
        if len(first) < 4 or len(second) < 4:
            return 0.0
        x0 = max(float(first[0]), float(second[0]))
        y0 = max(float(first[1]), float(second[1]))
        x1 = min(float(first[2]), float(second[2]))
        y1 = min(float(first[3]), float(second[3]))
        intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        first_area = max(
            1.0,
            (float(first[2]) - float(first[0])) * (float(first[3]) - float(first[1])),
        )
        return intersection / first_area

    def _topic_for_table_position(
        self,
        page,
        table_bbox: Sequence[float],
        default_topic: TocEntry,
        local_topic_titles: Dict[str, TocEntry],
    ) -> TocEntry:
        """Use the nearest preceding same-column local topic title when available."""
        if not local_topic_titles:
            return default_topic
        page_width = float(page.rect.width)
        table_column = self._column_for_bbox(table_bbox, page_width)
        preceding: List[Tuple[float, TocEntry]] = []
        for block in page.get_text("blocks"):
            if len(block) < 5:
                continue
            bbox = [float(value) for value in block[:4]]
            title = self._normalize_text(str(block[4]))
            matched = local_topic_titles.get(title)
            if (
                matched
                and self._column_for_bbox(bbox, page_width) == table_column
                and bbox[1] <= table_bbox[1]
            ):
                preceding.append((bbox[1], matched))
        return max(preceding, key=lambda item: item[0])[1] if preceding else default_topic

    def _first_non_empty(self, values: Sequence[str]) -> str:
        for value in values:
            if value:
                return value
        return ""

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()

    def _parse_page_range(self, value: str) -> List[int]:
        pages: List[int] = []
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = part.split("-", 1)
                pages.extend(range(int(start), int(end) + 1))
            else:
                pages.append(int(part))
        return sorted(set(pages))

    def _stable_id(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser(description="Preview TOC-aware multimodal PDF splitting.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file.")
    parser.add_argument("--out", default="data/multimodal_chunks.json", help="Output JSON path.")
    parser.add_argument("--preview", default="data/multimodal_chunks.md", help="Markdown preview path.")
    parser.add_argument("--assets", default="data/multimodal_assets", help="Directory for extracted image assets.")
    parser.add_argument("--toc-pages", default="3-7", help="TOC pages, e.g. 3-7 or 3,4,5,6,7.")
    parser.add_argument("--child-size", type=int, default=450, help="Text child chunk size in characters.")
    parser.add_argument("--overlap", type=int, default=80, help="Text child overlap in characters.")
    parser.add_argument("--parent-size", type=int, default=1500, help="Maximum topic text parent size.")
    args = parser.parse_args()

    splitter = ManualPDFSplitter(
        pdf_path=args.pdf,
        asset_dir=args.assets,
        toc_pages=args.toc_pages,
        child_chunk_size=args.child_size,
        child_overlap=args.overlap,
        parent_chunk_size=args.parent_size,
    )
    chunks = splitter.split()
    splitter.save_outputs(chunks, args.out, args.preview)
    print(f"Generated {len(chunks)} chunks")
    print(f"JSON: {args.out}")
    print(f"Preview: {args.preview}")
    print(f"Assets: {args.assets}")


if __name__ == "__main__":
    main()
