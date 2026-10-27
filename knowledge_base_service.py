#!/usr/bin/env python
# coding: utf-8
"""Knowledge-base lifecycle helpers used by the Gradio management page."""

from __future__ import annotations

import json
import re
import shutil
import threading
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional

from multimodal_pdf_parse import ManualPDFSplitter
from multimodal_retriever import MultimodalIndexBuilder, MultimodalRetriever, load_manifest


KNOWLEDGE_BASE_ROOT = Path("data/knowledge_bases")
ACTIVE_KNOWLEDGE_BASE_FILE = KNOWLEDGE_BASE_ROOT / "active.json"
_BUILD_LOCK = threading.Lock()


def normalize_knowledge_base_name(name: str, source_path: str = "") -> str:
    candidate = (name or "").strip() or Path(source_path).stem
    candidate = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", candidate, flags=re.UNICODE).strip("._-")
    if not candidate:
        raise ValueError("请输入有效的知识库名称")
    return candidate[:80]


def knowledge_base_paths(name: str) -> Dict[str, Path]:
    root = KNOWLEDGE_BASE_ROOT / name
    return {
        "root": root,
        "source": root / "source.pdf",
        "chunks": root / "chunks.json",
        "preview": root / "chunks_preview.md",
        "assets": root / "assets",
        "store": root / "vector_store",
    }


def build_knowledge_base(
    uploaded_pdf: str,
    name: str = "",
    toc_pages: str = "3-7",
    child_chunk_size: int = 450,
    child_overlap: int = 80,
    parent_chunk_size: int = 1500,
    batch_size: int = 128,
    status_callback: Optional[Callable[[str], None]] = None,
) -> Dict:
    source = Path(uploaded_pdf)
    if not source.exists() or source.suffix.lower() != ".pdf":
        raise ValueError("请上传有效的 PDF 文件")
    if child_overlap >= child_chunk_size:
        raise ValueError("切分重叠长度必须小于文本块长度")
    if parent_chunk_size < child_chunk_size:
        raise ValueError("文本父块最大长度必须大于或等于文本子块长度")

    kb_name = normalize_knowledge_base_name(name, uploaded_pdf)
    paths = knowledge_base_paths(kb_name)

    def report(message: str) -> None:
        if status_callback:
            status_callback(message)

    if not _BUILD_LOCK.acquire(blocking=False):
        raise RuntimeError("已有知识库正在构建，请等待当前任务完成")
    try:
        report("正在保存上传的 PDF...")
        paths["root"].mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, paths["source"])

        report("正在解析版面、目录、文本、表格和图片...")
        splitter = ManualPDFSplitter(
            pdf_path=str(paths["source"]),
            asset_dir=str(paths["assets"]),
            toc_pages=toc_pages,
            child_chunk_size=int(child_chunk_size),
            child_overlap=int(child_overlap),
            parent_chunk_size=int(parent_chunk_size),
        )
        chunks = splitter.split()
        if not chunks:
            raise ValueError("未切分出可入库内容，请检查目录页范围或 PDF 是否可提取文本")
        splitter.save_outputs(chunks, str(paths["chunks"]), str(paths["preview"]))

        counts = Counter(chunk.doc_type for chunk in chunks)
        report(f"切分完成，共 {len(chunks)} 条证据。正在生成向量索引...")

        manifest = MultimodalIndexBuilder(paths["chunks"], paths["store"]).build(
            batch_size=int(batch_size),
            progress_callback=lambda done, total: report(f"正在向量化入库：{done}/{total}"),
        )
        result = {
            "name": kb_name,
            "pdf": str(paths["source"]),
            "chunks": str(paths["chunks"]),
            "preview": str(paths["preview"]),
            "store": str(paths["store"]),
            "chunk_count": len(chunks),
            "doc_type_counts": dict(sorted(counts.items())),
            "manifest": manifest,
        }
        (paths["root"] / "knowledge_base.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report("知识库构建完成")
        return result
    finally:
        _BUILD_LOCK.release()


def list_knowledge_bases() -> List[Dict]:
    if not KNOWLEDGE_BASE_ROOT.exists():
        return []
    items: List[Dict] = []
    for root in sorted(path for path in KNOWLEDGE_BASE_ROOT.iterdir() if path.is_dir()):
        metadata_path = root / "knowledge_base.json"
        manifest_path = root / "vector_store" / "manifest.json"
        try:
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            elif manifest_path.exists():
                metadata = {"name": root.name, "manifest": load_manifest(root / "vector_store")}
            else:
                continue
            items.append(metadata)
        except (OSError, json.JSONDecodeError):
            continue
    return items


def set_active_knowledge_base(name: str) -> str:
    kb_name = normalize_knowledge_base_name(name)
    store = knowledge_base_paths(kb_name)["store"]
    if not (store / "manifest.json").exists():
        raise ValueError(f"知识库 `{kb_name}` 尚未构建完成")
    KNOWLEDGE_BASE_ROOT.mkdir(parents=True, exist_ok=True)
    ACTIVE_KNOWLEDGE_BASE_FILE.write_text(
        json.dumps({"name": kb_name}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return kb_name


def get_active_knowledge_base() -> Optional[str]:
    """Return persisted active KB, otherwise the most recently built KB."""
    if ACTIVE_KNOWLEDGE_BASE_FILE.exists():
        try:
            name = json.loads(ACTIVE_KNOWLEDGE_BASE_FILE.read_text(encoding="utf-8")).get("name", "")
            if name and (knowledge_base_paths(name)["store"] / "manifest.json").exists():
                return name
        except (OSError, json.JSONDecodeError):
            pass

    candidates = []
    for item in list_knowledge_bases():
        name = item.get("name")
        if not name:
            continue
        manifest = knowledge_base_paths(name)["store"] / "manifest.json"
        if manifest.exists():
            candidates.append((manifest.stat().st_mtime, name))
    return max(candidates)[1] if candidates else None


def search_knowledge_base(name: str, query: str, top_k: int = 6) -> List[Dict]:
    kb_name = normalize_knowledge_base_name(name)
    store = knowledge_base_paths(kb_name)["store"]
    if not (store / "manifest.json").exists():
        raise ValueError(f"知识库 `{kb_name}` 尚未构建完成")
    return MultimodalRetriever(store).search(query, top_k=int(top_k))
