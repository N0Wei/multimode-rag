"""Upload one MinerU-parsed manual and atomically activate a rebuilt index."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multi_manual_ingest import ManualInput, parse_manual
from multi_manual_mineru import parse_pdf_with_mineru
from multi_manual_store import MultiManualIndexBuilder, activate_index_version


_upload_lock = threading.Lock()


def _active_source_dirs(source_root: Path, store_root: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    pointer = store_root / "active.json"
    if not pointer.exists():
        return [], []
    active = json.loads(pointer.read_text(encoding="utf-8"))
    version_id = active.get("version_id") or active["batch_id"]
    manifest = json.loads((store_root / "versions" / version_id / "manifest.json").read_text(encoding="utf-8"))
    active_manuals = manifest["manuals"]
    by_identity: dict[tuple[str, str, str], Path] = {}
    if source_root.exists():
        for folder in source_root.iterdir():
            path = folder / "manifest.json"
            if folder.is_dir() and path.is_file():
                source = json.loads(path.read_text(encoding="utf-8"))
                by_identity[(source["doc_id"], source["model_name"], source["ingest_batch_id"])] = folder
    missing = [m["model_name"] for m in active_manuals
               if (m["doc_id"], m["model_name"], m["ingest_batch_id"]) not in by_identity]
    if missing:
        raise ValueError(f"活动索引的源切片缺失：{', '.join(missing)}")
    return [by_identity[(m["doc_id"], m["model_name"], m["ingest_batch_id"])]
            for m in active_manuals], active_manuals


def upload_and_activate_manual(model_name: str, pdf: str | Path,
                               *, manual_version: str | None = None,
                               model_year: str | None = None,
                               source_root: str | Path = "data/multi_manual",
                               store_root: str | Path = "data/multi_manual_store",
                               upload_root: str | Path = "data/manual_uploads",
                               text_embedding=None, visual_embedding=None) -> dict[str, Any]:
    """Keep the previous active index untouched unless parsing and indexing succeed.

    MinerU Flash+TXT runs locally to create the required structural archive.
    """
    model_name = (model_name or "").strip()
    manual_version = (manual_version or "").strip() or "未标注"
    model_year = (model_year or "").strip() or None
    pdf = Path(pdf)
    if not model_name or len(model_name) > 100:
        raise ValueError("请输入 1～100 字的车型名称")
    if len(manual_version) > 100:
        raise ValueError("手册版本不能超过 100 字")
    if model_year and not re.fullmatch(r"\d{4}", model_year):
        raise ValueError("车型年款请填写四位年份；没有明确依据可留空")
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        raise ValueError("请上传 PDF 手册")
    with _upload_lock:
        source_root, store_root, upload_root = Path(source_root), Path(store_root), Path(upload_root)
        current_dirs, active_manuals = _active_source_dirs(source_root, store_root)
        pdf_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()
        if any(m["source_sha256"] == pdf_hash for m in active_manuals):
            raise ValueError("该 PDF 已在活动索引中，不能重复上传")
        existing = next((m for m in active_manuals if m["model_name"] == model_name), None)
        if existing:
            old_year = existing.get("model_year")
            if old_year and model_year and old_year != model_year:
                raise ValueError("该车型的年款与当前版本不同，请用不同车型名称分别管理")
            model_year = model_year or old_year
        if source_root.is_dir():
            for manifest_path in source_root.glob("*/manifest.json"):
                previous = json.loads(manifest_path.read_text(encoding="utf-8"))
                if previous["source_sha256"] == pdf_hash and previous["model_name"] != model_name:
                    raise ValueError("该 PDF 曾作为另一车型的手册入库，不能改名重复使用")
                if (manual_version != "未标注" and previous["model_name"] == model_name
                        and previous.get("manual_version") == manual_version
                        and previous["source_sha256"] != pdf_hash):
                    raise ValueError("该车型的同名手册版本已有不同 PDF；请填写可区分的官方修订版次")
        key = f"upload_{uuid.uuid4().hex[:16]}"
        saved = upload_root / key
        saved.mkdir(parents=True)
        saved_pdf, saved_zip = saved / "manual.pdf", saved / "mineru.zip"
        shutil.copy2(pdf, saved_pdf)
        parse_pdf_with_mineru(saved_pdf, saved_zip)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        batch_id = f"{now.replace(':', '').replace('+00:00', 'Z')}-{key}"
        parsed = parse_manual(ManualInput(key, model_name, saved_pdf, saved_zip,
                                          manual_version=manual_version, model_year=model_year), source_root,
                              batch_id=batch_id, ingested_at=now)
        if not parsed["child_count"] or not parsed["image_count"]:
            raise ValueError("解析结果缺少文本子块或图片，未切换活动索引；请检查 MinerU 产物")
        index = MultiManualIndexBuilder(source_root, store_root).build(
            text_embedding=text_embedding, visual_embedding=visual_embedding,
            manual_dirs=[folder for folder, manual in zip(current_dirs, active_manuals)
                         if manual["model_name"] != model_name] + [source_root / key])
        return {"parsed": parsed, "index": index, "updated": existing is not None,
                "previous_doc_id": existing["doc_id"] if existing else None}


def rollback_to_index_version(version_id: str,
                              *, store_root: str | Path = "data/multi_manual_store") -> dict[str, Any]:
    """Serialize rollback against uploads in this app process."""
    with _upload_lock:
        return activate_index_version(store_root, version_id)
