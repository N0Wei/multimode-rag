#!/usr/bin/env python
# coding: utf-8
"""Preview N text blocks before and after MinerU image blocks.

This is an ingestion experiment only. It reads MinerU's per-page middle_json
from parse ZIPs and does not modify the production knowledge base.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable


TEXT_TYPES = {"text", "paragraph_title", "list", "table"}


def parse_pages(value: str) -> set[int]:
    pages: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(piece) for piece in part.split("-", 1))
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    return pages


def flatten_content(value: Any) -> str:
    pieces: list[str] = []
    if isinstance(value, str):
        pieces.append(value)
    elif isinstance(value, list):
        for item in value:
            pieces.append(flatten_content(item))
    elif isinstance(value, dict):
        if isinstance(value.get("content"), str):
            pieces.append(value["content"])
        elif "content" in value:
            pieces.append(flatten_content(value["content"]))

    text = html.unescape(" ".join(piece for piece in pieces if piece))
    text = re.sub(r"</?(?:table|tbody|thead|tr|td|th|br|p)[^>]*>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def bbox_key(block: dict[str, Any]) -> tuple[float, float, int]:
    bbox = block.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    return float(bbox[1]), float(bbox[0]), int(block.get("index", 0))


def block_text(block: dict[str, Any]) -> str:
    return flatten_content(block.get("content", []))


def image_path(block: dict[str, Any]) -> str:
    for item in block.get("content", []):
        if isinstance(item, dict) and item.get("image_path"):
            return str(item["image_path"])
    return ""


def collect_page_blocks(page: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = []
    for raw in page.get("blocks", []):
        kind = raw.get("type", "")
        bbox = raw.get("bbox")
        if not bbox:
            continue
        item = {
            "type": kind,
            "bbox": bbox,
            "index": raw.get("index", 0),
            "text": block_text(raw) if kind in TEXT_TYPES else "",
            "image_path": image_path(raw) if kind == "image" else "",
        }
        if (kind == "image" and item["image_path"]) or (kind in TEXT_TYPES and item["text"]):
            blocks.append(item)
    return sorted(blocks, key=bbox_key)


def neighbors_for_image(blocks: list[dict[str, Any]], image_index: int, count: int):
    before = [block for block in blocks[:image_index] if block["type"] in TEXT_TYPES]
    after = [block for block in blocks[image_index + 1 :] if block["type"] in TEXT_TYPES]
    return before[-count:], after[:count]


def format_blocks(blocks: Iterable[dict[str, Any]]) -> str:
    rendered = []
    for block in blocks:
        rendered.append(f"[{block['type']}] {block['text']}")
    return "\n".join(rendered) if rendered else "（无）"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zips", nargs="+", required=True, help="MinerU result ZIP files")
    parser.add_argument("--pages", default="", help="1-based physical pages, e.g. 21,25-26; empty means all")
    parser.add_argument("--neighbors", type=int, default=1, help="Text blocks to take on each side")
    parser.add_argument("--out", required=True, help="Output Markdown preview")
    args = parser.parse_args()
    if args.neighbors < 1:
        parser.error("--neighbors must be at least 1")

    selected_pages = parse_pages(args.pages) if args.pages else set()
    lines = [
        "# MinerU 图片前后邻近正文试验",
        "",
        f"- 规则：按页面坐标从上到下、同一高度从左到右排序；每张图取前后各 {args.neighbors} 个正文块。",
        "- 注意：这是线性邻近基线，不跨页；本报告用于检查是否会错配，不代表已接入正式检索。",
    ]
    total_images = 0
    missing_before = 0
    missing_after = 0

    for zip_name in args.zips:
        zip_path = Path(zip_name)
        lines.extend(["", f"## {zip_path.name}"])
        with zipfile.ZipFile(zip_path) as archive:
            middle_json_name = next((name for name in archive.namelist() if name.endswith("middle_json.json")), None)
            if not middle_json_name:
                lines.extend(["", "未找到 middle_json.json。"])
                continue
            data = json.loads(archive.read(middle_json_name))

        for page in data.get("pages", []):
            physical_page = int(page.get("page_idx", 0)) + 1
            if selected_pages and physical_page not in selected_pages:
                continue
            blocks = collect_page_blocks(page)
            for index, block in enumerate(blocks):
                if block["type"] != "image":
                    continue
                total_images += 1
                before, after = neighbors_for_image(blocks, index, args.neighbors)
                missing_before += not bool(before)
                missing_after += not bool(after)
                lines.extend(
                    [
                        "",
                        f"### 第 {physical_page} 页 · {block['image_path']}",
                        f"- image bbox: `{block['bbox']}`",
                        "- 前置正文：",
                        "```text",
                        format_blocks(before),
                        "```",
                        "- 后置正文：",
                        "```text",
                        format_blocks(after),
                        "```",
                    ]
                )

    lines.extend(
        [
            "",
            "## 汇总",
            f"- 图片数：{total_images}",
            f"- 无前置正文：{missing_before}",
            f"- 无后置正文：{missing_after}",
        ]
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Preview written to {output} ({total_images} images)")


if __name__ == "__main__":
    main()
