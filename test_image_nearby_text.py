#!/usr/bin/env python
# coding: utf-8
"""Inspect image-to-nearby-text bindings produced by multimodal_pdf_parse.py."""

import argparse
import json
from pathlib import Path


def parse_pages(value: str):
    pages = []
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


def main():
    parser = argparse.ArgumentParser(description="Preview image nearby-text bindings.")
    parser.add_argument("--chunks", default="data/multimodal_chunks.json", help="Generated chunk JSON.")
    parser.add_argument("--pages", default="69,70,74,78", help="Pages to inspect.")
    parser.add_argument("--out", default="data/image_nearby_preview.md", help="Markdown output.")
    parser.add_argument("--limit-per-page", type=int, default=20, help="Maximum images per page.")
    args = parser.parse_args()

    chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    pages = parse_pages(args.pages)

    lines = [
        "# Image Nearby Text Preview",
        "",
        f"- Chunks: `{args.chunks}`",
        f"- Pages: `{pages}`",
    ]

    for page in pages:
        images = [
            chunk
            for chunk in chunks
            if chunk.get("doc_type") == "image"
            and chunk.get("metadata", {}).get("page_id") == page
        ][: args.limit_per_page]
        lines.extend(["", f"## Page {page}", "", f"Images: {len(images)}"])

        for image in images:
            metadata = image.get("metadata", {})
            image_path = metadata.get("image_path", "")
            lines.extend(
                [
                    "",
                    f"### {image.get('chunk_id', '')}",
                    "",
                    f"- Topic: `{metadata.get('topic_title', '')}`",
                    f"- Image: `{image_path}`",
                    f"- BBox: `{metadata.get('bbox', '')}`",
                    "",
                    "```text",
                    metadata.get("nearby_text", ""),
                    "```",
                ]
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Preview: {out}")


if __name__ == "__main__":
    main()
