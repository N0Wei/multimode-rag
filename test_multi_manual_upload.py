"""Upload activation and live version refresh regressions."""

from __future__ import annotations

import json
import hashlib
import shutil
import weakref
from pathlib import Path

import pytest

from multi_manual_ingest import ManualInput, default_inputs, parse_manual
from multi_manual_mineru import parse_pdf_with_mineru
from multi_manual_store import MultiManualRetriever, _point_id, list_active_manual_models, list_index_versions
from multi_manual_upload import rollback_to_index_version, upload_and_activate_manual


class TinyEmbedding:
    def embed_documents(self, texts):
        return [[float("气囊" in text), 1.0] for text in texts]

    def embed_query(self, text):
        return [float("气囊" in text), 1.0]


class TinyVisualEmbedding:
    def embed_documents(self, paths):
        return [[1.0, 1.0] for _ in paths]

    def embed_query(self, text):
        return [1.0, 1.0]


def test_upload_activate_and_keep_prior_index_on_duplicate(tmp_path: Path, monkeypatch) -> None:
    first_spec, second_spec, third_spec = default_inputs(Path(__file__).resolve().parent)
    if any(not s.pdf.is_file() or not s.mineru_zip.is_file()
           for s in (first_spec, second_spec, third_spec)):
        pytest.skip("Local test manual and MinerU archive are absent")
    archives = {hashlib.sha256(s.pdf.read_bytes()).hexdigest(): s.mineru_zip
                for s in (first_spec, second_spec, third_spec)}

    def fake_mineru(pdf: Path, archive: Path) -> Path:
        source = archives.get(hashlib.sha256(pdf.read_bytes()).hexdigest())
        if source is None:
            raise ValueError("MinerU 无法解析此 PDF")
        shutil.copy2(source, archive)
        return archive

    monkeypatch.setattr("multi_manual_upload.parse_pdf_with_mineru", fake_mineru)
    source, store, uploads = (tmp_path / name for name in ("source", "store", "uploads"))
    result = upload_and_activate_manual(
        "上传测试车型", first_spec.pdf, manual_version="2026-08版", model_year="2026",
        source_root=source, store_root=store,
        upload_root=uploads, text_embedding=TinyEmbedding(), visual_embedding=TinyVisualEmbedding())
    assert result["parsed"]["child_count"] > 0
    assert result["parsed"]["image_count"] > 0
    assert result["parsed"]["manual_version"] == "2026-08版"
    assert result["parsed"]["model_year"] == "2026"
    old_doc_id = result["parsed"]["doc_id"]
    assert list_active_manual_models(store) == ["上传测试车型"]
    second = upload_and_activate_manual(
        "第二款测试车型", second_spec.pdf,
        source_root=source, store_root=store, upload_root=uploads,
        text_embedding=TinyEmbedding(), visual_embedding=TinyVisualEmbedding())
    assert len(second["index"]["manuals"]) == 2
    assert list_active_manual_models(store) == ["上传测试车型", "第二款测试车型"]
    active_before = (store / "active.json").read_bytes()
    previous_version_id = json.loads(active_before)["version_id"]
    with pytest.raises(ValueError, match="已在活动索引"):
        upload_and_activate_manual("上传测试车型", first_spec.pdf,
                                   source_root=source, store_root=store, upload_root=uploads)
    assert (store / "active.json").read_bytes() == active_before
    with pytest.raises(ValueError, match="同名手册版本"):
        upload_and_activate_manual("上传测试车型", third_spec.pdf, manual_version="2026-08版",
                                   source_root=source, store_root=store, upload_root=uploads)
    with pytest.raises(ValueError, match="年款"):
        upload_and_activate_manual("上传测试车型", third_spec.pdf, model_year="2027",
                                   source_root=source, store_root=store, upload_root=uploads)
    assert (store / "active.json").read_bytes() == active_before
    updated = upload_and_activate_manual(
        "上传测试车型", third_spec.pdf, manual_version="2026-09版",
        source_root=source, store_root=store, upload_root=uploads,
        text_embedding=TinyEmbedding(), visual_embedding=TinyVisualEmbedding())
    assert updated["updated"] is True
    assert updated["previous_doc_id"] == old_doc_id
    assert updated["parsed"]["model_year"] == "2026"
    assert len(updated["index"]["manuals"]) == 2
    assert list_active_manual_models(store) == ["第二款测试车型", "上传测试车型"]
    with pytest.raises(ValueError, match="另一车型"):
        upload_and_activate_manual("冒名车型", first_spec.pdf,
                                   source_root=source, store_root=store, upload_root=uploads)
    retriever = MultiManualRetriever(store, TinyEmbedding(), TinyVisualEmbedding())
    try:
        assert retriever.models["上传测试车型"]["doc_id"] == updated["parsed"]["doc_id"]
        hit = retriever.search_text("安全气囊", "上传测试车型", top_k=1)[0]
        assert hit["metadata"]["doc_id"] == updated["parsed"]["doc_id"]
        assert hit["metadata"]["manual_version"] == "2026-09版"
        with pytest.raises(ValueError, match="doc_id"):
            retriever.search_text("安全气囊", "上传测试车型", doc_id=old_doc_id)
        point = retriever.client.retrieve(
            "manual_text", ids=[_point_id(hit["chunk_id"])],
            with_payload=True)[0]
        assert point.payload["manual_version"] == "2026-09版"
        assert point.payload["model_year"] == "2026"
        assert point.payload["manual_revision_id"] == updated["parsed"]["manual_revision_id"]
    finally:
        retriever.close()
    assert any(item["version_id"] == previous_version_id and not item["active"]
               for item in list_index_versions(store))
    rolled_back = rollback_to_index_version(previous_version_id, store_root=store)
    assert {m["model_name"]: m["doc_id"] for m in rolled_back["manuals"]}["上传测试车型"] == old_doc_id
    assert next(item for item in list_index_versions(store)
                if item["version_id"] == previous_version_id)["active"] is True
    active_before = (store / "active.json").read_bytes()
    with pytest.raises(ValueError, match="无效"):
        rollback_to_index_version("../outside", store_root=store)
    with pytest.raises(ValueError, match="不完整"):
        rollback_to_index_version("not-found", store_root=store)
    assert (store / "active.json").read_bytes() == active_before
    bad_pdf = tmp_path / "bad.pdf"
    bad_pdf.write_bytes(b"not a real manual")
    with pytest.raises(ValueError, match="无法解析"):
        upload_and_activate_manual("损坏的车型", bad_pdf,
                                   source_root=source, store_root=store, upload_root=uploads)
    assert (store / "active.json").read_bytes() == active_before


def test_local_mineru_zip_matches_parent_child_parser(tmp_path: Path) -> None:
    spec = default_inputs(Path(__file__).resolve().parent)[0]
    if not spec.pdf.is_file():
        pytest.skip("Local test manual is absent")
    archive = parse_pdf_with_mineru(spec.pdf, tmp_path / "auto.zip", page_range="1-32")
    manifest = parse_manual(ManualInput("auto_probe", "自动解析测试车型", spec.pdf, archive),
                            tmp_path / "parsed", batch_id="auto-test", ingested_at="2026-01-01T00:00:00Z")
    assert manifest["coverage_page_end"] == 32
    assert manifest["parent_count"] > 0
    assert manifest["child_count"] > 0
    assert manifest["image_count"] > 0


def test_retriever_hot_swap_on_active_version_change(tmp_path: Path, monkeypatch) -> None:
    import app_gradio

    store = tmp_path / "store"
    store.mkdir()
    (store / "active.json").write_text(json.dumps({"version_id": "v1"}), encoding="utf-8")
    loaded = []

    class FakeRetriever:
        def __init__(self, root):
            self.version = Path(json.loads((Path(root) / "active.json").read_text())["version_id"])
            loaded.append(self)

        def close(self):
            pass

    monkeypatch.setattr(app_gradio, "MULTI_MANUAL_STORE", store)
    monkeypatch.setattr(app_gradio, "MultiManualRetriever", FakeRetriever)
    monkeypatch.setattr(app_gradio, "plain_rag_llm", object())
    monkeypatch.setattr(app_gradio, "agent_graph", object())
    monkeypatch.setattr(app_gradio, "multi_manual_tool", None)
    monkeypatch.setattr(app_gradio, "_multi_manual_tool_cache", weakref.WeakValueDictionary())
    monkeypatch.setattr(app_gradio, "set_multi_manual_tool", lambda tool: None)
    first = app_gradio._get_multi_manual_tool()
    assert app_gradio._get_multi_manual_tool() is first
    (store / "active.json").write_text(json.dumps({"version_id": "v2"}), encoding="utf-8")
    second = app_gradio._get_multi_manual_tool()
    assert second is not first
    assert second.retriever.version.name == "v2"
    (store / "active.json").write_text(json.dumps({"version_id": "v1"}), encoding="utf-8")
    assert app_gradio._get_multi_manual_tool() is first
    assert len(loaded) == 2
