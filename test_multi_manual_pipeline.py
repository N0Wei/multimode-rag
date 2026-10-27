"""Structural tests for the new MinerU -> two-collection retrieval path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from multi_manual_ingest import MODEL_SPECS, default_inputs, parse_manual
from multi_manual_store import MultiManualIndexBuilder, MultiManualRetriever


class TinyEmbedding:
    def embed_query(self, text: str) -> list[float]:
        return [float("气囊" in text), float("座椅" in text), 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class TinyVisualEmbedding:
    def embed_query(self, text: str) -> list[float]:
        return [float("图片" in text), 1.0]

    def embed_documents(self, paths: list[str]) -> list[list[float]]:
        return [[1.0, 1.0] for _ in paths]


def _write_lines(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def test_qdrant_two_collections_filter_before_topk(tmp_path: Path) -> None:
    source = tmp_path / "source"
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"test-image")
    for index, model in enumerate(["宋 Ultra EV", "汉 EV 闪充版"]):
        folder = source / f"model-{index}"
        folder.mkdir(parents=True)
        common = {"model_name": model, "doc_id": f"doc-{index}", "section_id": f"section-{index}",
                  "section_title": "安全气囊", "section_path": ["安全", "安全气囊"],
                  "schema_version": "multi-manual-v1", "ingest_batch_id": "test-batch"}
        parent_id = f"parent-{index}"
        _write_lines(folder / "parents.jsonl", [{"chunk_id": parent_id, "page_content": "安全气囊的说明",
                                                 "metadata": {**common, "chunk_id": parent_id}}])
        _write_lines(folder / "children.jsonl", [{
            "chunk_id": f"child-{index}", "page_content": f"{model}的安全气囊说明",
            "metadata": {**common, "chunk_id": f"child-{index}", "parent_id": parent_id,
                         "page_start": 25, "page_end": 25},
        }])
        _write_lines(folder / "images.jsonl", [{
            "image_id": f"image-{index}", "nearby_text": f"{model}安全气囊图片",
            "metadata": {**common, "image_id": f"image-{index}", "image_path": str(image_path),
                         "page_number": 25},
        }])
        (folder / "manifest.json").write_text(json.dumps({
            **common, "coverage_page_start": 1, "coverage_page_end": 50,
        }, ensure_ascii=False), encoding="utf-8")

    store = tmp_path / "store"
    built = MultiManualIndexBuilder(source, store).build(TinyEmbedding(), TinyVisualEmbedding(), batch_size=1)
    assert built["child_count"] == 2
    assert built["image_count"] == 2
    retriever = MultiManualRetriever(store, TinyEmbedding(), TinyVisualEmbedding())
    try:
        for model in ("宋 Ultra EV", "汉 EV 闪充版"):
            texts = retriever.search_text("安全气囊", model, top_k=1, candidate_k=1)
            images = retriever.search_images("安全气囊图片", model, top_k=1, candidate_k=1)
            assert len(texts) == len(images) == 1
            assert texts[0]["metadata"]["model_name"] == model
            assert texts[0]["metadata"]["manual_version"] == "未标注"
            assert images[0]["metadata"]["model_name"] == model
        assert set(retriever.compare_text("安全气囊", ["宋 Ultra EV", "汉 EV 闪充版"])) == {
            "宋 Ultra EV", "汉 EV 闪充版",
        }
        with pytest.raises(ValueError):
            retriever.search_text("安全气囊", "")
        with pytest.raises(ValueError):
            retriever.search_text("安全气囊", "宋 Ultra EV", doc_id="doc-1")
    finally:
        retriever.close()
    rebuilt = MultiManualIndexBuilder(source, store).build(TinyEmbedding(), TinyVisualEmbedding(), batch_size=1)
    assert rebuilt["batch_id"] == built["batch_id"]
    assert rebuilt["version_id"] != built["version_id"]
    active_before_failure = (store / "active.json").read_text(encoding="utf-8")

    class FailingEmbedding(TinyEmbedding):
        def embed_documents(self, texts):
            raise RuntimeError("embedding service unavailable")

    with pytest.raises(RuntimeError, match="embedding service unavailable"):
        MultiManualIndexBuilder(source, store).build(FailingEmbedding(), TinyVisualEmbedding(), batch_size=1)
    assert (store / "active.json").read_text(encoding="utf-8") == active_before_failure


@pytest.mark.parametrize("model_key", ["song_ultra_ev", "han_ev_flash", "qin_max_ev"])
def test_real_manual_airbag_parent_boundary(tmp_path: Path, model_key: str) -> None:
    root = Path(__file__).resolve().parent
    spec = next(s for s in default_inputs(root) if s.key == model_key)
    if not spec.pdf.exists() or not spec.mineru_zip.exists():
        pytest.skip("Local manuals or MinerU artifacts are absent")
    manifest = parse_manual(spec, tmp_path, batch_id="test-batch", ingested_at="2026-01-01T00:00:00Z")
    folder = tmp_path / model_key
    parents = [json.loads(line) for line in (folder / "parents.jsonl").read_text(encoding="utf-8").splitlines()]
    children = [json.loads(line) for line in (folder / "children.jsonl").read_text(encoding="utf-8").splitlines()]
    images = [json.loads(line) for line in (folder / "images.jsonl").read_text(encoding="utf-8").splitlines()]
    parent = next(p for p in parents if p["metadata"]["section_title"] == "安全气囊触发条件及注意事项")
    assert not any(p["metadata"]["section_title"] == "安全气囊可能不会展开的情况" for p in parents)
    airbag_children = [c for c in children if c["metadata"]["parent_id"] == parent["chunk_id"]]
    eight_items = [c for c in airbag_children if c["metadata"]["child_heading"] == "安全气囊可能不会展开的情况"
                   and "①" in c["page_content"] and "⑧" in c["page_content"]]
    assert len(eight_items) == 1
    assert all(Path(image["metadata"]["image_path"]).is_file() for image in images)
    assert manifest["is_full_document"] is False
    assert manifest["parent_count"] == len(parents)
