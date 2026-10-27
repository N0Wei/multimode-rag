"""Build auditable, directory-level parent/child records from MinerU archives.

This is deliberately separate from ``multimodal_pdf_parse.py``: the latter
extracts an older PDF directly and has different section boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import fitz


SCHEMA_VERSION = "multi-manual-v1"
BODY_TYPES = {"text", "paragraph_title", "page_footnote", "table", "image"}
SKIP_TYPES = {"header", "footer", "page_number"}
WARNING_LABELS = ("警告", "注意", "温馨提示", "提示", "危险")
MODEL_SPECS = {
    "song_ultra_ev": ("宋 Ultra EV", "宋Ultra EV用户手册20261016.pdf", "宋Ultra_1-202.zip"),
    "han_ev_flash": ("汉 EV 闪充版", "汉EV闪充版用户手册260827.pdf", "汉EV_1-195.zip"),
    "qin_max_ev": ("秦 MAX EV", "秦MAX EV用户手册260902.pdf", "秦MAX_1-200.zip"),
}
# The Qin manual promotes this whole topic to numbered section 1-3. Its
# lower-level printed TOC entries remain children, not independent parents.
PARENT_OVERRIDES = {"qin_max_ev": {"安全气囊触发条件及注意事项"}}


def _norm(value: str) -> str:
    return re.sub(r"[\s.．·…:：、，,()（）*＊-]+", "", value or "")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _items_text(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    return "\n".join(str(item.get("content") or "") for item in items if item.get("type") == "text").strip()


def _image_body_text(items: Any) -> str:
    """Read MinerU's OCR text attached to an image block, without treating it as a caption."""
    if not isinstance(items, list):
        return ""
    parts = []
    for item in items:
        if not isinstance(item, dict) or item.get("type") != "image_body":
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
        elif isinstance(content, list):
            text = _items_text(content)
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def _ocr_table_heading(text: str) -> str:
    lines = [line.strip() for line in text.replace("\u3000", " ").splitlines() if line.strip()]
    header_index = next((i for i, line in enumerate(lines)
                         if re.search(r"项目\s*(?:参数|名称|数值)|序号\s*(?:安培|规格|说明|被保护组件)|乘坐位置编号", line)),
                        len(lines))
    candidates = [line for line in lines[:header_index]
                  if len(line) <= 48 and re.search(r"参数|尺寸|规格|性能|位置|标牌|信息", line)]
    return candidates[-1] if candidates else "OCR 表格"


def _is_structured_ocr_table(text: str) -> bool:
    """Conservatively identify OCR tables in MinerU image_body blocks.

    Full-page/table scans can be represented as image blocks even when MinerU
    has produced readable OCR. Keep figure labels and ordinary image text in
    the image channel; route only content with a recognizable table header and
    repeated cell structure to the text index.
    """
    normalized = text.replace("\u3000", " ")
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if len(lines) < 4:
        return False
    has_header = any(re.search(
        r"项目\s*(?:参数|名称|数值)|序号\s*(?:安培|规格|说明|被保护组件)|"
        r"乘坐位置编号|产品型号|编号\s*(?:规格|安培|说明)", line
    ) for line in lines)
    if not has_header:
        return False
    cell_rows = [re.split(r"(?:\t+|\s{2,})", line) for line in lines]
    cell_rows = [row for row in cell_rows if len(row) >= 2]
    numeric_rows = sum(any(re.search(r"\d", cell) for cell in row[1:]) for row in cell_rows)
    return len(cell_rows) >= 3 and numeric_rows >= 2


def _split_ocr_table_sections(text: str) -> list[tuple[str, str]]:
    """Split a scanned parameter page into focused table chunks when headings permit."""
    lines = [line.strip() for line in text.replace("\u3000", " ").splitlines() if line.strip()]
    if not lines:
        return []
    heading_positions = []
    for i, line in enumerate(lines):
        compact = re.sub(r"\s+", "", line).rstrip("：:")
        has_single_cell = len(re.split(r"(?:\t+|\s{2,})", line)) == 1
        if (len(compact) <= 40 and has_single_cell and not re.search(r"\d", compact)
                and compact.endswith(("参数", "规格", "性能"))):
            heading_positions.append((i, compact))

    # A page-level title such as 整车参数 is context, not a separate table,
    # when a more specific table heading follows immediately.
    if (len(heading_positions) > 1 and heading_positions[0][0] == 0
            and heading_positions[0][1] in {"整车参数", "车辆参数"}):
        heading_positions = heading_positions[1:]
    if not heading_positions:
        heading = _ocr_table_heading(text)
        return [(heading, text.strip())]

    sections = []
    prefix = lines[:heading_positions[0][0]]
    for pos, (start, heading) in enumerate(heading_positions):
        end = heading_positions[pos + 1][0] if pos + 1 < len(heading_positions) else len(lines)
        section_lines = (prefix if pos == 0 else []) + lines[start:end]
        section_text = "\n".join(section_lines).strip()
        sections.append((heading, section_text))

    # Every line remains in exactly one section. A final table may continue on
    # the next page and be too short to pass the standalone structure check.
    return sections


def _asset_path(items: Any) -> str | None:
    if not isinstance(items, list):
        return None
    for item in items:
        if isinstance(item, dict) and item.get("image_path"):
            return str(item["image_path"])
    return None


class _TableText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"}:
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._row is not None and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(self._row):
                self.rows.append(self._row)
            self._row = None


def _table_text(html: str) -> tuple[str, str | None]:
    parser = _TableText()
    parser.feed(html)
    rows = parser.rows
    if not rows:
        return "", None
    label = next((x for x in WARNING_LABELS if rows[0] and rows[0][0].startswith(x)), None)
    return "\n".join(" | ".join(cell for cell in row if cell) for row in rows), label


@dataclass(frozen=True)
class ManualInput:
    key: str
    model_name: str
    pdf: Path
    mineru_zip: Path
    manual_version: str | None = None
    model_year: str | None = None


def default_inputs(root: Path) -> list[ManualInput]:
    return [
        ManualInput(key, model, root / "data" / pdf, root / "tmp" / "mineru_flash_txt_half" / archive)
        for key, (model, pdf, archive) in MODEL_SPECS.items()
    ]


def _printed_toc(doc: fitz.Document) -> str:
    toc = doc.get_toc()
    start = next((page for _, title, page in toc if _norm(title) == "目录"), None)
    end = next((page for _, title, page in toc if title == "图片索引" and start and page > start), None)
    if not start or not end:
        raise ValueError("Cannot identify printed table-of-contents pages")
    return _norm("".join(doc[page - 1].get_text() for page in range(start, end)))


def _parent_toc_entries(doc: fitz.Document, key: str, last_page: int) -> list[dict[str, Any]]:
    toc = doc.get_toc()
    printed = _printed_toc(doc)
    chapter_start = next((i for i, (level, title, page) in enumerate(toc) if level == 1 and title == "安全" and page > 0), None)
    if chapter_start is None:
        raise ValueError("Manual has no main chapter '安全'")
    entries = []
    for i in range(chapter_start, len(toc)):
        level, title, page = toc[i]
        if page < 1 or page > last_page:
            continue
        if level in {2, 3} and _norm(title) in printed:
            entries.append({"toc_index": i, "level": level, "title": title, "page": page})
    if not entries:
        raise ValueError("No parent candidates occur in printed TOC")
    overrides = PARENT_OVERRIDES.get(key, set())
    selected: list[dict[str, Any]] = []
    for candidate in entries:
        i, level, title = candidate["toc_index"], candidate["level"], candidate["title"]
        ancestors = [e for e in entries if e["toc_index"] < i and e["level"] == 2]
        parent_l2 = ancestors[-1] if ancestors else None
        # A new level-1 chapter resets ancestry; the scan below handles it.
        for j in range(i - 1, chapter_start - 1, -1):
            if toc[j][0] == 1:
                if parent_l2 and parent_l2["toc_index"] < j:
                    parent_l2 = None
                break
        if level == 3:
            if parent_l2 and parent_l2["title"] in overrides:
                continue
            selected.append(candidate)
        elif title in overrides:
            selected.append(candidate)
        else:
            descendants = [
                e for e in entries
                if e["level"] == 3 and e["toc_index"] > i
                and all(toc[j][0] > 2 for j in range(i + 1, e["toc_index"] + 1))
            ]
            if not descendants:
                selected.append(candidate)
    return sorted(selected, key=lambda e: e["toc_index"])


def _outline_path(toc: list[list[Any]], index: int) -> list[str]:
    level = toc[index][0]
    path = [toc[index][1]]
    for i in range(index - 1, -1, -1):
        if toc[i][0] < level:
            path.append(toc[i][1])
            level = toc[i][0]
            if level == 1:
                break
    return list(reversed(path))


def _block_text(block: dict[str, Any]) -> str:
    if block.get("type") == "table":
        html = next((str(x.get("content") or "") for x in block.get("content", []) if x.get("type") == "table_body"), "")
        return _table_text(html)[0]
    return _items_text(block.get("content"))


def _is_child_heading(kind: str, content: str, cues: set[str]) -> bool:
    if _norm(content) in cues:
        return True
    # MinerU sometimes labels numbered list *content* as paragraph_title.
    # Only accept its free-form title guess if it still looks like a heading.
    return (kind == "paragraph_title" and len(content) <= 40
            and not re.match(r"^[①-⑳\d]", content) and "。" not in content)


def _normalize_pages(raw: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    pages: dict[int, list[dict[str, Any]]] = {}
    for page in raw.get("pages", []):
        page_idx = int(page["page_idx"])
        blocks = []
        for ordinal, item in enumerate(page.get("blocks", [])):
            kind = item.get("type")
            if kind in SKIP_TYPES or kind not in BODY_TYPES:
                continue
            if kind == "page_footnote":
                kind = "text"  # MinerU uses this for real icon labels in Qin's instrument table.
            content = _block_text(item)
            asset = _asset_path(item.get("content"))
            bbox = item.get("bbox") or []
            if (kind in {"text", "paragraph_title"} and re.fullmatch(r"\d{1,2}", content)
                    and len(bbox) == 4 and (bbox[0] > 0.9 or bbox[2] < 0.1)):
                continue  # chapter number printed in the outer margin
            if kind != "image" and not content:
                continue
            if kind == "image" and not asset:
                continue
            normalized = {
                "block_id": f"p{page_idx:04d}_b{ordinal:03d}", "page_number": page_idx + 1,
                "ordinal": ordinal, "type": kind, "text": content,
                "bbox": item.get("bbox"), "asset": asset,
            }
            blocks.append(normalized)
            if kind == "image":
                ocr_text = _image_body_text(item.get("content"))
                if _is_structured_ocr_table(ocr_text):
                    table_sections = _split_ocr_table_sections(ocr_text)
                    heading = table_sections[0][0]
                    # Keep the source image for visual retrieval, and create a
                    # separate text-index record for each focused OCR table.
                    normalized["ocr_table_heading"] = heading
                    for table_index, (table_heading, table_text) in enumerate(table_sections):
                        blocks.append({
                            "block_id": f"{normalized['block_id']}_ocr_table_{table_index:02d}",
                            "page_number": page_idx + 1, "ordinal": ordinal,
                            "type": "table", "text": table_text,
                            "bbox": item.get("bbox"), "asset": None,
                            "table_label": table_heading, "ocr_source": "image_body",
                            "source_image_block_id": normalized["block_id"],
                        })
        pages[page_idx + 1] = blocks
    return pages


def _start_position(entry: dict[str, Any], toc: list[list[Any]], pages: dict[int, list[dict[str, Any]]]) -> tuple[int, int, str]:
    page = entry["page"]
    blocks = pages.get(page, [])
    title = _norm(entry["title"])
    matches = [b for b in blocks if b["type"] in {"paragraph_title", "text"} and _norm(b["text"]) == title]
    if matches:
        best = min(matches, key=lambda b: (b["type"] != "paragraph_title", b["ordinal"]))
        return page, best["ordinal"], "exact_heading"
    prefixes = [b for b in blocks if b["type"] in {"paragraph_title", "text"}
                and _norm(b["text"]).startswith(title) and len(_norm(b["text"])) > len(title)]
    if prefixes:
        return page, prefixes[0]["ordinal"], "prefix_heading"
    # Some numbered chapters (notably Qin 1-3) only appear in the running
    # header. Their first descendant heading is the first body block.
    i = entry["toc_index"]
    for j in range(i + 1, len(toc)):
        if toc[j][0] <= entry["level"]:
            break
        if toc[j][2] != page:
            continue
        descendant = _norm(toc[j][1])
        for b in blocks:
            if b["type"] in {"paragraph_title", "text"} and _norm(b["text"]) == descendant:
                return page, b["ordinal"], "descendant_heading"
    if blocks:
        return page, blocks[0]["ordinal"], "first_body_block"
    return page, 0, "missing_page_body"


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _extract_image(archive: zipfile.ZipFile, name: str, dest: Path) -> Path:
    rel = Path(name.replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts or not rel.parts or rel.parts[0] != "images":
        raise ValueError(f"Unsafe MinerU asset path: {name}")
    members = [n for n in archive.namelist() if n.endswith("/" + name) or n == name]
    if len(members) != 1:
        raise FileNotFoundError(f"Image asset not uniquely found: {name}")
    target = dest / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(archive.read(members[0]))
    return target


def parse_manual(spec: ManualInput, output_dir: Path, *, batch_id: str, ingested_at: str) -> dict[str, Any]:
    if not spec.pdf.is_file() or not spec.mineru_zip.is_file():
        raise FileNotFoundError(f"Missing PDF or MinerU archive for {spec.key}")
    source_sha256 = hashlib.sha256(spec.pdf.read_bytes()).hexdigest()
    doc_id = _digest(source_sha256)
    manual_version = (spec.manual_version or "").strip() or "未标注"
    model_year = (spec.model_year or "").strip() or None
    manual_revision_id = f"rev_{source_sha256[:20]}"
    with zipfile.ZipFile(spec.mineru_zip) as archive:
        middle = [name for name in archive.namelist() if name.endswith("middle_json.json")]
        if len(middle) != 1:
            raise ValueError(f"Expected one middle_json.json in {spec.mineru_zip}")
        raw = json.loads(archive.read(middle[0]))
        pages = _normalize_pages(raw)
        if not pages or min(pages) != 1 or sorted(pages) != list(range(1, max(pages) + 1)):
            raise ValueError("Only a contiguous PDF page range starting at page 1 is supported")
        max_page = max(pages)
        with fitz.open(spec.pdf) as doc:
            pdf_page_count = len(doc)
            toc = doc.get_toc()
            entries = _parent_toc_entries(doc, spec.key, max_page)
            for entry in entries:
                entry["start_page"], entry["start_ordinal"], entry["anchor_method"] = _start_position(entry, toc, pages)
                entry["section_path"] = _outline_path(toc, entry["toc_index"])
            if any((a["start_page"], a["start_ordinal"]) >= (b["start_page"], b["start_ordinal"])
                   for a, b in zip(entries, entries[1:])):
                raise ValueError("Non-increasing parent section anchors; inspect TOC/body alignment")

        root = output_dir / spec.key
        assets = root / "assets"
        root.mkdir(parents=True, exist_ok=True)
        parents: list[dict[str, Any]] = []
        children: list[dict[str, Any]] = []
        images: list[dict[str, Any]] = []
        image_by_hash: dict[str, dict[str, Any]] = {}
        image_ocr_table_count = 0
        anchor_counts = Counter(entry["anchor_method"] for entry in entries)

        for index, entry in enumerate(entries):
            begin = (entry["start_page"], entry["start_ordinal"])
            end = ((entries[index + 1]["start_page"], entries[index + 1]["start_ordinal"])
                   if index + 1 < len(entries) else (max_page + 1, 0))
            blocks = [b for page in range(begin[0], min(end[0], max_page) + 1)
                      for b in pages.get(page, []) if begin <= (b["page_number"], b["ordinal"]) < end]
            if not blocks:
                continue
            section_id = _digest(f"{doc_id}:{entry['toc_index']}:{'/'.join(entry['section_path'])}")
            parent_id = f"{spec.key}:parent:{section_id}"
            common = {
                "model_name": spec.model_name, "doc_id": doc_id,
                "source_pdf": str(spec.pdf.resolve()), "source_sha256": source_sha256,
                "manual_version": manual_version, "model_year": model_year,
                "manual_revision_id": manual_revision_id, "schema_version": SCHEMA_VERSION,
                "ingest_batch_id": batch_id, "ingested_at": ingested_at,
                "section_id": section_id, "section_title": entry["title"],
                "section_path": entry["section_path"],
                "section_number": next((x for x in reversed(entry["section_path"]) if re.match(r"^\d+-\d+", x)), None),
            }
            descendants = []
            for j in range(entry["toc_index"] + 1, len(toc)):
                if toc[j][0] <= entry["level"]:
                    break
                descendants.append(_norm(toc[j][1]))
            heading_cues = set(descendants)
            current_heading = ""
            pending: list[dict[str, Any]] = []
            parent_lines: list[str] = []
            parent_block_ids: list[str] = []

            def flush() -> None:
                if not pending:
                    return
                body = "\n".join(b["text"] for b in pending if b["text"]).strip()
                if body:
                    child_index = sum(1 for c in children if c["metadata"]["parent_id"] == parent_id)
                    content = (current_heading + "\n" + body).strip() if current_heading else body
                    child_id = f"{parent_id}:child:{child_index:03d}"
                    children.append({"chunk_id": child_id, "doc_type": "text", "chunk_type": "text_child",
                                     "page_content": content, "metadata": {
                                         **common, "chunk_id": child_id, "chunk_type": "text_child",
                                         "parent_id": parent_id, "child_heading": current_heading or None,
                                         "child_index": child_index,
                                         "page_start": pending[0]["page_number"],
                                         "page_end": pending[-1]["page_number"],
                                         "source_block_ids": [b["block_id"] for b in pending],
                                         "oversize": len(content) > 800,
                                     }})
                pending.clear()

            for pos, block in enumerate(blocks):
                kind, content = block["type"], block["text"]
                if kind != "image":
                    parent_block_ids.append(block["block_id"])
                    if content:
                        parent_lines.append(content)
                if kind == "image":
                    prev_text = next((b["text"] for b in reversed(blocks[:pos]) if b["page_number"] == block["page_number"] and b["type"] == "text"), "")
                    next_text = next((b["text"] for b in blocks[pos + 1:] if b["page_number"] == block["page_number"] and b["type"] == "text"), "")
                    image_heading = block.get("ocr_table_heading")
                    near = "\n".join(dict.fromkeys(
                        x for x in (image_heading, prev_text, next_text) if x
                    ))
                    path = _extract_image(archive, block["asset"], assets)
                    image_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                    occurrence = {"page_number": block["page_number"], "bbox": block["bbox"],
                                  "source_block_id": block["block_id"]}
                    if image_hash in image_by_hash:
                        image_by_hash[image_hash]["metadata"]["occurrences"].append(occurrence)
                        continue
                    image_id = f"{spec.key}:image:{image_hash[:20]}"
                    record = {"image_id": image_id, "nearby_text": near, "metadata": {
                        **common, "image_id": image_id, "parent_id": parent_id,
                        "image_path": str(path.resolve()), "image_sha256": image_hash,
                        "page_number": block["page_number"], "bbox": block["bbox"],
                        "bbox_coord_system": "mineru_normalized_0_1",
                        "occurrences": [occurrence],
                    }}
                    image_by_hash[image_hash] = record
                    images.append(record)
                    continue
                if kind == "table":
                    flush()
                    if block.get("ocr_source") == "image_body":
                        label = block.get("table_label")
                        image_ocr_table_count += 1
                    else:
                        html = next((str(x.get("content") or "") for x in raw["pages"][block["page_number"] - 1]["blocks"][block["ordinal"]].get("content", []) if x.get("type") == "table_body"), "")
                        _, label = _table_text(html)
                    child_index = sum(1 for c in children if c["metadata"]["parent_id"] == parent_id)
                    child_id = f"{parent_id}:child:{child_index:03d}"
                    table_record = {"chunk_id": child_id, "doc_type": "table", "chunk_type": "table_child",
                                    "page_content": content, "metadata": {
                                         **common, "chunk_id": child_id, "chunk_type": "table_child",
                                         "parent_id": parent_id, "child_heading": label,
                                         "child_index": child_index,
                                         "page_start": block["page_number"], "page_end": block["page_number"],
                                         "source_block_ids": [block.get("source_image_block_id") or block["block_id"]],
                                         "bbox": block["bbox"], "bbox_coord_system": "mineru_normalized_0_1",
                                         "text_extraction_method": (
                                             "mineru_image_body_ocr"
                                             if block.get("ocr_source") == "image_body"
                                             else "mineru_table_html"
                                         ),
                                     }}
                    previous = children[-1] if children else None
                    if (label and previous and previous["metadata"]["parent_id"] == parent_id
                            and previous["chunk_type"] == "table_child"
                            and previous["metadata"]["child_heading"] == label
                            and block["page_number"] <= previous["metadata"]["page_end"] + 1):
                        previous["page_content"] += "\n" + content
                        previous["metadata"]["page_end"] = block["page_number"]
                        previous["metadata"]["source_block_ids"].append(block["block_id"])
                    else:
                        children.append(table_record)
                    current_heading = ""  # A warning/table ends the preceding body subheading.
                    continue
                is_parent_heading = (block["page_number"], block["ordinal"]) == begin and content.startswith(entry["title"])
                if is_parent_heading:
                    content = content[len(entry["title"]):].strip()
                    if not content:
                        continue
                is_child_heading = _is_child_heading(kind, content, heading_cues)
                if is_child_heading:
                    flush()
                    current_heading = content
                    continue
                if pending and sum(len(b["text"]) for b in pending) + len(content) > 800 and not re.match(r"^[①-⑳]", content):
                    flush()
                pending.append(block)
            flush()
            parents.append({"chunk_id": parent_id, "doc_type": "text", "chunk_type": "text_parent",
                            "page_content": "\n".join(parent_lines), "metadata": {
                                **common, "chunk_id": parent_id, "chunk_type": "text_parent",
                                "page_start": blocks[0]["page_number"], "page_end": blocks[-1]["page_number"],
                                "source_block_ids": parent_block_ids,
                                "anchor_method": entry["anchor_method"],
                            }})

        _write_jsonl(root / "parents.jsonl", parents)
        _write_jsonl(root / "children.jsonl", children)
        _write_jsonl(root / "images.jsonl", images)
        manifest = {
            "model_name": spec.model_name, "doc_id": doc_id, "source_pdf": str(spec.pdf.resolve()),
            "source_sha256": source_sha256, "mineru_archive": str(spec.mineru_zip.resolve()),
            "manual_version": manual_version, "model_year": model_year,
            "manual_revision_id": manual_revision_id,
            "mineru_mode": "flash+txt", "schema_version": SCHEMA_VERSION,
            "ingest_batch_id": batch_id, "ingested_at": ingested_at,
            "coverage_page_start": 1, "coverage_page_end": max_page,
            "is_full_document": bool(raw.get("is_full_document")) and max_page >= pdf_page_count,
            "pdf_page_count": pdf_page_count, "parent_count": len(parents),
            "child_count": len(children), "image_count": len(images),
            "image_ocr_table_count": image_ocr_table_count,
            "anchor_methods": dict(anchor_counts),
        }
        (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multi-manual parent/child/image records from MinerU archives")
    parser.add_argument("--output", type=Path, default=Path("data/multi_manual"))
    parser.add_argument("--model", choices=[*MODEL_SPECS, "all"], default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    batch = now.replace(":", "").replace("+00:00", "Z")
    specs = default_inputs(root)
    for spec in specs:
        if args.model not in {"all", spec.key}:
            continue
        print(json.dumps(parse_manual(spec, args.output, batch_id=batch, ingested_at=now), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
