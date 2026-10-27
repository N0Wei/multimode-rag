#!/usr/bin/env python
# coding: utf-8
"""
Preview the TOC extraction used by multimodal_pdf_parse.py.

Example:
    python test_toc_extraction.py --pdf data/train_a.pdf --out data/toc_preview.md --json-out data/toc_preview.json
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from multimodal_pdf_parse import ManualPDFSplitter


def render_markdown(entries, pdf_path: str, toc_pages: str) -> str:
    lines = [
        "# TOC Extraction Preview",
        "",
        f"- Source: `{pdf_path}`",
        f"- TOC pages: `{toc_pages}`",
        f"- Entries: {len(entries)}",
        "",
        "| # | Title | Start | End | TOC ID |",
        "|---:|---|---:|---:|---|",
    ]
    for idx, entry in enumerate(entries, start=1):
        title = entry.title.replace("|", "/")
        lines.append(f"| {idx} | {title} | {entry.start_page} | {entry.end_page} | `{entry.toc_id}` |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Preview TOC entries parsed from a PDF manual.")
    parser.add_argument("--pdf", default="data/train_a.pdf", help="Path to the PDF file.")
    parser.add_argument("--toc-pages", default="3-7", help="TOC pages, e.g. 3-7 or 3,4,5,6,7.")
    parser.add_argument("--out", default="data/toc_preview.md", help="Markdown preview output path.")
    parser.add_argument("--json-out", default="data/toc_preview.json", help="JSON output path.")
    args = parser.parse_args()

    splitter = ManualPDFSplitter(pdf_path=args.pdf, toc_pages=args.toc_pages)
    doc = splitter._open_doc()
    entries = splitter.extract_toc_entries(doc)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(entries, args.pdf, args.toc_pages), encoding="utf-8")

    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps([asdict(entry) for entry in entries], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Extracted {len(entries)} TOC entries")
    print(f"Markdown: {out_path}")
    print(f"JSON: {json_path}")
    for idx, entry in enumerate(entries[:20], start=1):
        print(f"{idx:03d}. {entry.title} [{entry.start_page}-{entry.end_page}]")


if __name__ == "__main__":
    main()
