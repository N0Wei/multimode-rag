#!/usr/bin/env python
# coding: utf-8
"""Create a visual audit of PDF images that may be discarded.

This script does not modify the PDF, chunks, assets, or vector store. It copies
candidate images into a separate audit directory so the filtering policy can be
reviewed before it is applied to the production parser.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image, ImageDraw, ImageFont, ImageStat


SEMANTIC_SMALL_TOPICS = {
    "指示灯和警告灯",
    "查看组合仪表信息",
}
SEMANTIC_SMALL_KEYWORDS = (
    "指示灯",
    "警告灯",
    "故障灯",
    "按钮",
    "按键",
    "图标",
    "开关",
)


def load_image_metadata(documents_path: Path) -> Dict[str, Dict]:
    metadata = {}
    if not documents_path.exists():
        return metadata
    for line in documents_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        document = json.loads(line)
        item = document.get("metadata") or {}
        image_path = item.get("image_path")
        if image_path:
            metadata[Path(image_path).name] = item
    return metadata


def inspect_image(path: Path, metadata: Dict) -> Dict:
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        rgb = image.convert("RGB")
        stat = ImageStat.Stat(rgb)
        extrema = stat.extrema
        visual_range = max(high - low for low, high in extrema)
        alpha_empty = image.mode == "RGBA" and image.getchannel("A").getbbox() is None

    nearby_text = str(metadata.get("nearby_text", ""))
    topic = str(metadata.get("topic_title", ""))
    return {
        "filename": path.name,
        "source_path": str(path.resolve()),
        "page": metadata.get("page_id", ""),
        "topic": topic,
        "width": width,
        "height": height,
        "area": width * height,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "nearby_text": nearby_text,
        "semantic_small_context": (
            topic in SEMANTIC_SMALL_TOPICS
            or any(keyword in nearby_text for keyword in SEMANTIC_SMALL_KEYWORDS)
        ),
        "visual_range": visual_range,
        "alpha_empty": alpha_empty,
    }


def classify(records: List[Dict]) -> None:
    hash_counts = Counter(record["sha256"] for record in records)
    first_by_hash = {}

    for record in records:
        image_hash = record["sha256"]
        canonical = first_by_hash.setdefault(image_hash, record["filename"])
        record["duplicate_count"] = hash_counts[image_hash]
        record["canonical_file"] = canonical

        if record["alpha_empty"] or record["visual_range"] <= 3:
            record["decision"] = "discard"
            record["reason"] = "blank_or_nearly_blank"
        elif record["filename"] != canonical:
            record["decision"] = "discard"
            record["reason"] = "exact_duplicate_copy"
        elif (
            record["width"] == 54
            and record["height"] == 54
            and record["bytes"] <= 500
            and record["duplicate_count"] >= 3
        ):
            record["decision"] = "discard"
            record["reason"] = "generic_notice_symbol"
        elif record["semantic_small_context"]:
            record["decision"] = "keep"
            record["reason"] = "semantic_icon_or_indicator"
        elif record["width"] < 80 or record["height"] < 80 or record["area"] < 10000:
            record["decision"] = "discard"
            record["reason"] = "low_value_small_icon"
        else:
            record["decision"] = "keep"
            record["reason"] = "meaningful_visual_candidate"


def safe_preview(source: Path, destination: Path, max_size: int = 360) -> None:
    with Image.open(source) as image:
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, "white")
        background.alpha_composite(image)
        preview = background.convert("RGB")
        preview.thumbnail((max_size, max_size))
        destination.parent.mkdir(parents=True, exist_ok=True)
        preview.save(destination, format="PNG")


def copy_records(records: Iterable[Dict], output_dir: Path) -> None:
    for record in records:
        category_dir = output_dir / record["decision"] / record["reason"]
        category_dir.mkdir(parents=True, exist_ok=True)
        source = Path(record["source_path"])
        prefix = f"p{str(record['page']).zfill(3)}_" if record["page"] != "" else ""
        target_name = prefix + source.name
        shutil.copy2(source, category_dir / target_name)
        safe_preview(source, category_dir / (Path(target_name).stem + "_preview.png"))


def make_contact_sheet(records: List[Dict], output_path: Path, title: str) -> None:
    if not records:
        return
    cell_w, cell_h, columns = 320, 300, 4
    rows = (len(records) + columns - 1) // columns
    sheet = Image.new("RGB", (cell_w * columns, cell_h * rows + 50), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((12, 15), title, fill="black", font=font)

    for index, record in enumerate(records):
        x = (index % columns) * cell_w
        y = (index // columns) * cell_h + 50
        with Image.open(record["source_path"]) as image:
            image = image.convert("RGBA")
            background = Image.new("RGBA", image.size, "white")
            background.alpha_composite(image)
            preview = background.convert("RGB")
            preview.thumbnail((260, 210))
            sheet.paste(preview, (x + (cell_w - preview.width) // 2, y + 5))
        label = (
            f"p{record['page']} {record['filename'][:28]}\n"
            f"{record['width']}x{record['height']} {record['reason']}"
        )
        draw.multiline_text((x + 8, y + 222), label, fill="black", font=font, spacing=3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=88)


def make_paginated_contact_sheets(
    records: List[Dict],
    output_dir: Path,
    title: str,
    page_size: int = 60,
) -> None:
    for page_index, start in enumerate(range(0, len(records), page_size), start=1):
        make_contact_sheet(
            records[start : start + page_size],
            output_dir / f"{page_index:03d}.jpg",
            f"{title} - page {page_index}",
        )


def write_reports(records: List[Dict], output_dir: Path) -> None:
    fields = [
        "decision", "reason", "page", "topic", "filename", "width", "height",
        "area", "bytes", "duplicate_count", "canonical_file", "source_path",
    ]
    with (output_dir / "image_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    (output_dir / "image_audit.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    counts = Counter(record["decision"] for record in records)
    reasons = Counter(record["reason"] for record in records)
    summary = [
        "# PDF 图片过滤审计",
        "",
        "本目录仅用于检查候选过滤结果，未修改原始图片、切分结果和向量库。",
        "",
        f"- 总图片数：{len(records)}",
        f"- 建议舍弃：{counts['discard']}",
        f"- 建议保留：{counts['keep']}",
        f"- 需要人工复核：{counts['review']}",
        "",
        "## 分类原因",
        "",
    ]
    summary.extend(f"- `{reason}`：{count}" for reason, count in sorted(reasons.items()))
    summary.extend(
        [
            "",
            "## 目录说明",
            "",
            "- `discard/`：建议舍弃的图片及 PNG 预览。",
            "- `keep/`：建议保留图片，便于抽样对照。",
            "- `contact_sheets/`：各分类总览图。",
            "- `image_audit.csv`：可筛选的完整清单。",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(summary), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit extracted PDF images before filtering")
    parser.add_argument("--kb", default="data/knowledge_bases/train_a")
    parser.add_argument("--out", default="data/image_filter_audit/train_a")
    args = parser.parse_args()

    kb_dir = Path(args.kb)
    assets_dir = kb_dir / "assets"
    documents_path = kb_dir / "vector_store" / "documents.jsonl"
    output_dir = Path(args.out)
    if not assets_dir.exists():
        raise FileNotFoundError(f"Assets directory not found: {assets_dir}")

    metadata = load_image_metadata(documents_path)
    records = []
    for path in sorted(item for item in assets_dir.iterdir() if item.is_file()):
        try:
            records.append(inspect_image(path, metadata.get(path.name, {})))
        except Exception as exc:
            print(f"[skip] {path.name}: {exc}")

    classify(records)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    copy_records(records, output_dir)
    write_reports(records, output_dir)

    reasons = sorted(set(record["reason"] for record in records))
    for reason in reasons:
        selected = [record for record in records if record["reason"] == reason]
        make_paginated_contact_sheets(
            selected,
            output_dir / "contact_sheets" / reason,
            f"{reason.upper()} ({len(selected)})",
        )

    counts = Counter(record["decision"] for record in records)
    print(f"Audit complete: {output_dir.resolve()}")
    print(f"total={len(records)} discard={counts['discard']} review={counts['review']} keep={counts['keep']}")


if __name__ == "__main__":
    main()
