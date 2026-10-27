"""Two-collection Qdrant index and model-scoped retrieval for MinerU manuals."""

from __future__ import annotations

import json
import hashlib
import math
import re
import uuid
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import jieba
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi


TEXT_COLLECTION = "manual_text"
IMAGE_COLLECTION = "manual_images"
IMAGE_TEXT_VECTOR = "nearby_text"
IMAGE_VISUAL_VECTOR = "visual"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _point_id(record_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, record_id))


def _batched(items: list[Any], size: int) -> Iterable[list[Any]]:
    for offset in range(0, len(items), size):
        yield items[offset:offset + size]


def _validate_vectors(vectors: list[list[float]], count: int, label: str) -> int:
    if len(vectors) != count or not vectors or not vectors[0]:
        raise ValueError(f"{label}: embedding returned the wrong number of vectors")
    dimension = len(vectors[0])
    if any(len(v) != dimension or not all(math.isfinite(float(x)) for x in v) for v in vectors):
        raise ValueError(f"{label}: inconsistent dimensions or non-finite embedding values")
    return dimension


def _payload(metadata: dict[str, Any], record_id: str) -> dict[str, Any]:
    keys = (
        "model_name", "doc_id", "section_id", "section_title", "section_path",
        "parent_id", "child_heading", "page_start", "page_end", "page_number",
        "image_path", "schema_version", "ingest_batch_id", "ingested_at",
        "manual_version", "model_year", "manual_revision_id", "source_sha256",
    )
    return {**{key: metadata[key] for key in keys if metadata.get(key) is not None}, "record_id": record_id}


def _embedding_identity(embedding: Any) -> str:
    model = (getattr(embedding, "model_name", None) or getattr(embedding, "model_path", None)
             or getattr(embedding, "model", None) or "")
    if not isinstance(model, (str, int, float)):
        model = type(model).__name__
    endpoint = (getattr(embedding, "api_base", None) or getattr(embedding, "openai_api_base", None)
                or getattr(embedding, "base_url", None) or "")
    return f"{type(embedding).__module__}.{type(embedding).__name__}:{model}:{endpoint}"


def _cached_embed_documents(items: list[str], embedding: Any, cache_root: Path,
                            batch_size: int, label: str, *, image_paths: bool = False
                            ) -> tuple[list[list[float]], int]:
    """Cache per-item vectors so an API failure can be retried without rebilling earlier items."""
    backend_key = hashlib.sha256(_embedding_identity(embedding).encode("utf-8")).hexdigest()[:20]
    folder = cache_root / label / backend_key
    folder.mkdir(parents=True, exist_ok=True)
    results: list[list[float] | None] = [None] * len(items)
    missing: list[tuple[int, str, Path]] = []
    for index, item in enumerate(items):
        digest = hashlib.sha256(Path(item).read_bytes() if image_paths else item.encode("utf-8")).hexdigest()
        path = folder / f"{digest}.json"
        if path.exists():
            try:
                vector = json.loads(path.read_text(encoding="utf-8"))
                _validate_vectors([vector], 1, label)
                results[index] = vector
                continue
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        missing.append((index, item, path))
    # The visual adapter currently sends one request per image internally.
    # Calling it one at a time allows progress to survive a mid-batch failure.
    effective_batch = 1 if image_paths else batch_size
    for group in _batched(missing, effective_batch):
        vectors = embedding.embed_documents([item for _, item, _ in group])
        _validate_vectors(vectors, len(group), label)
        for (index, _, path), vector in zip(group, vectors):
            values = [float(value) for value in vector]
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(values), encoding="utf-8")
            temporary.replace(path)
            results[index] = values
        print(f"  [{label}] {sum(v is not None for v in results)}/{len(items)} ready", flush=True)
    if any(vector is None for vector in results):
        raise ValueError(f"{label}: missing embedding after cache fill")
    return [vector for vector in results if vector is not None], len(items) - len(missing)


def _filter(model_name: str, doc_id: str | None = None) -> models.Filter:
    if not model_name:
        raise ValueError("model_name is required; unscoped multi-manual search is forbidden")
    conditions = [models.FieldCondition(key="model_name", match=models.MatchValue(value=model_name))]
    if doc_id:
        conditions.append(models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)))
    return models.Filter(must=conditions)


def _record_text(record: dict[str, Any]) -> str:
    metadata = record["metadata"]
    return "\n".join(filter(None, [metadata.get("section_title"), metadata.get("child_heading"), record["page_content"]]))


def _image_text(record: dict[str, Any]) -> str:
    return "\n".join(filter(None, [record["metadata"].get("section_title"), record.get("nearby_text")]))


def _load_corpus(source_root: Path, manual_dirs: list[Path] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    parents: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    folders = manual_dirs if manual_dirs is not None else sorted(path for path in source_root.iterdir() if path.is_dir())
    for folder in folders:
        if not (folder / "manifest.json").exists():
            if manual_dirs is not None:
                raise ValueError(f"Parsed manual is incomplete: {folder}")
            continue
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        # Existing parsed manuals predate version metadata. Enrich them in
        # memory when rebuilding, without rewriting their auditable source.
        manifest["manual_version"] = manifest.get("manual_version") or "未标注"
        manifest.setdefault("model_year", None)
        manifest["manual_revision_id"] = (manifest.get("manual_revision_id") or
                                          f"rev_{manifest.get('source_sha256', manifest['doc_id'])[:20]}")
        manifests.append(manifest)
        for records, filename in ((parents, "parents.jsonl"), (children, "children.jsonl"),
                                  (images, "images.jsonl")):
            for record in _read_jsonl(folder / filename):
                metadata = record["metadata"]
                for key in ("manual_version", "model_year", "manual_revision_id"):
                    metadata[key] = manifest[key]
                records.append(record)
    if not manifests:
        raise ValueError(f"No parsed manuals in {source_root}")
    schema_versions = {m["schema_version"] for m in manifests}
    if len(schema_versions) != 1:
        raise ValueError("Manuals must use one schema version")
    if len({m["model_name"] for m in manifests}) != len(manifests):
        raise ValueError("Duplicate model names")
    if len({m["doc_id"] for m in manifests}) != len(manifests):
        raise ValueError("Duplicate source PDFs")
    parent_ids = {record["chunk_id"] for record in parents}
    if len(parent_ids) != len(parents):
        raise ValueError("Duplicate parent IDs")
    if any(record["metadata"]["parent_id"] not in parent_ids for record in children):
        raise ValueError("Child without a directory-level parent")
    if any(not Path(record["metadata"]["image_path"]).is_file() for record in images):
        raise ValueError("Image record points to a missing file")
    if len({r["metadata"]["model_name"] for r in children}) != len(manifests):
        raise ValueError("One or more manuals have no searchable child chunks")
    return parents, children, images, manifests


def list_active_manual_models(store_root: str | Path) -> list[str]:
    """List selectable models from the active, fully built index manifest.

    Parsing a source PDF does not make it selectable; only an activated index
    does. The UI does not need to initialize embedding clients or scan Qdrant.
    """
    root = Path(store_root)
    pointer = root / "active.json"
    if not pointer.is_file():
        return []
    active = json.loads(pointer.read_text(encoding="utf-8"))
    version_id = active.get("version_id") or active["batch_id"]
    manifest = json.loads((root / "versions" / version_id / "manifest.json").read_text(encoding="utf-8"))
    names = [manual["model_name"] for manual in manifest["manuals"]]
    return list(dict.fromkeys(names))


def list_index_versions(store_root: str | Path) -> list[dict[str, Any]]:
    """List completed immutable index versions for audit and rollback."""
    root = Path(store_root)
    versions_root = root / "versions"
    if not versions_root.is_dir():
        return []
    active_path = root / "active.json"
    active_id = None
    if active_path.is_file():
        pointer = json.loads(active_path.read_text(encoding="utf-8"))
        active_id = pointer.get("version_id") or pointer["batch_id"]
    result = []
    for folder in versions_root.iterdir():
        manifest_path = folder / "manifest.json"
        if not folder.is_dir() or not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result.append({
            "version_id": folder.name,
            "built_at": manifest.get("built_at") or datetime.fromtimestamp(
                manifest_path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
            "active": folder.name == active_id,
            "manuals": [{"model_name": m["model_name"],
                         "manual_version": m.get("manual_version") or "未标注",
                         "doc_id": m["doc_id"]} for m in manifest["manuals"]],
        })
    return sorted(result, key=lambda item: (item["built_at"], item["version_id"]), reverse=True)


def activate_index_version(store_root: str | Path, version_id: str) -> dict[str, Any]:
    """Rollback by switching the pointer to a previously completed index."""
    if version_id in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.:+-]+", version_id or ""):
        raise ValueError("无效的索引版本 ID")
    root = Path(store_root)
    version = root / "versions" / version_id
    required = ("manifest.json", "parents.jsonl", "children.jsonl", "images.jsonl", "qdrant/meta.json")
    if any(not (version / name).is_file() for name in required):
        raise ValueError("索引版本不完整，不能切换")
    manifest = json.loads((version / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version_id") != version_id or not manifest.get("manuals"):
        raise ValueError("索引版本清单不匹配，不能切换")
    if len({m["model_name"] for m in manifest["manuals"]}) != len(manifest["manuals"]):
        raise ValueError("索引版本存在重复车型，不能切换")
    pending = root / f"active.{uuid.uuid4().hex}.tmp"
    pending.write_text(json.dumps({"batch_id": manifest["batch_id"], "version_id": version_id},
                                  ensure_ascii=False), encoding="utf-8")
    pending.replace(root / "active.json")
    return manifest


class MultiManualIndexBuilder:
    def __init__(self, source_root: str | Path, store_root: str | Path):
        self.source_root = Path(source_root)
        self.store_root = Path(store_root)

    def build(self, text_embedding=None, visual_embedding=None, batch_size: int = 64,
              manual_dirs: list[Path] | None = None) -> dict[str, Any]:
        """Build a new immutable version; activate it only after all checks pass."""
        parents, children, images, source_manifests = _load_corpus(self.source_root, manual_dirs)
        if text_embedding is None:
            from embedding_model import get_embedding_model
            text_embedding = get_embedding_model()
        if visual_embedding is None:
            from visual_embedding_model import get_visual_embedding_model
            visual_embedding = get_visual_embedding_model()
        source_batches = {m["ingest_batch_id"] for m in source_manifests}
        batch_id = (next(iter(source_batches)) if len(source_batches) == 1 else
                    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        # A failed attempt must not block retrying the same parsed batch or
        # replace a previously active index. Each build gets its own version.
        version_id = f"{batch_id}-{uuid.uuid4().hex[:8]}"
        version = self.store_root / "versions" / version_id
        version.mkdir(parents=True)
        client = None
        try:
            client = QdrantClient(path=str(version / "qdrant"))
            text_vectors, text_cache_hits = _cached_embed_documents(
                [_record_text(record) for record in children], text_embedding,
                self.store_root / "embedding_cache", batch_size, "text")
            text_dim = _validate_vectors(text_vectors, len(children), "text")
            client.create_collection(TEXT_COLLECTION, vectors_config=models.VectorParams(size=text_dim, distance=models.Distance.COSINE))
            for start in range(0, len(children), batch_size):
                group = children[start:start + batch_size]
                vectors = text_vectors[start:start + batch_size]
                client.upsert(TEXT_COLLECTION, points=[models.PointStruct(
                    id=_point_id(record["chunk_id"]), vector=vector,
                    payload=_payload(record["metadata"], record["chunk_id"]),
                ) for record, vector in zip(group, vectors)], wait=True)

            image_text_vectors, image_text_cache_hits = _cached_embed_documents(
                [_image_text(record) for record in images], text_embedding,
                self.store_root / "embedding_cache", batch_size, "image_text")
            image_dim = _validate_vectors(image_text_vectors, len(images), "image text")
            if image_dim != text_dim:
                raise ValueError("Image-nearby text and text chunks use different embedding dimensions")
            visual_vectors, visual_cache_hits = _cached_embed_documents(
                [record["metadata"]["image_path"] for record in images], visual_embedding,
                self.store_root / "embedding_cache", batch_size, "image_visual", image_paths=True)
            visual_dim = _validate_vectors(visual_vectors, len(images), "image visual")
            client.create_collection(IMAGE_COLLECTION, vectors_config={
                IMAGE_TEXT_VECTOR: models.VectorParams(size=text_dim, distance=models.Distance.COSINE),
                IMAGE_VISUAL_VECTOR: models.VectorParams(size=visual_dim, distance=models.Distance.COSINE),
            })
            for start in range(0, len(images), batch_size):
                group = images[start:start + batch_size]
                client.upsert(IMAGE_COLLECTION, points=[models.PointStruct(
                    id=_point_id(record["image_id"]),
                    vector={IMAGE_TEXT_VECTOR: image_text_vectors[start + offset],
                            IMAGE_VISUAL_VECTOR: visual_vectors[start + offset]},
                    payload=_payload(record["metadata"], record["image_id"]),
                ) for offset, record in enumerate(group)], wait=True)
            local_mode = type(client._client).__name__ == "QdrantLocal"
            if not local_mode:
                for collection in (TEXT_COLLECTION, IMAGE_COLLECTION):
                    for field in ("model_name", "doc_id"):
                        client.create_payload_index(collection_name=collection, field_name=field,
                                                    field_schema=models.PayloadSchemaType.KEYWORD, wait=True)
            if client.count(TEXT_COLLECTION, exact=True).count != len(children):
                raise ValueError("Text collection count mismatch")
            if client.count(IMAGE_COLLECTION, exact=True).count != len(images):
                raise ValueError("Image collection count mismatch")
            client.close()
            client = None
            for name, records in (("parents", parents), ("children", children), ("images", images)):
                with (version / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
                    for record in records:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest = {
                "batch_id": batch_id, "version_id": version_id,
                "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "schema_version": source_manifests[0]["schema_version"],
                "manuals": source_manifests, "parent_count": len(parents),
                "child_count": len(children), "image_count": len(images),
                "text_dimension": text_dim, "visual_dimension": visual_dim,
                "collections": [TEXT_COLLECTION, IMAGE_COLLECTION],
                "embedding_backend": _embedding_identity(text_embedding),
                "visual_embedding_backend": _embedding_identity(visual_embedding),
                "qdrant_mode": "local" if local_mode else "server",
                "payload_indexes_effective": not local_mode,
                "embedding_cache_hits": {"text": text_cache_hits, "image_text": image_text_cache_hits,
                                         "image_visual": visual_cache_hits},
            }
            (version / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            self.store_root.mkdir(parents=True, exist_ok=True)
            pending = self.store_root / f"active.{uuid.uuid4().hex}.tmp"
            pending.write_text(json.dumps({"batch_id": batch_id, "version_id": version_id}, ensure_ascii=False), encoding="utf-8")
            pending.replace(self.store_root / "active.json")
            return manifest
        except Exception:
            if client is not None:
                client.close()
            # Keep the incomplete version for inspection, but never activate it.
            raise


class MultiManualRetriever:
    def __init__(self, store_root: str | Path, text_embedding=None, visual_embedding=None):
        self.store_root = Path(store_root)
        active = json.loads((self.store_root / "active.json").read_text(encoding="utf-8"))
        self.version = self.store_root / "versions" / active.get("version_id", active["batch_id"])
        self.manifest = json.loads((self.version / "manifest.json").read_text(encoding="utf-8"))
        if text_embedding is None:
            from embedding_model import get_embedding_model
            text_embedding = get_embedding_model()
        if visual_embedding is None:
            from visual_embedding_model import get_visual_embedding_model
            visual_embedding = get_visual_embedding_model()
        if _embedding_identity(text_embedding) != self.manifest["embedding_backend"]:
            raise ValueError("Text embedding backend differs from the active index")
        if _embedding_identity(visual_embedding) != self.manifest["visual_embedding_backend"]:
            raise ValueError("Visual embedding backend differs from the active index")
        self.text_embedding = text_embedding
        self.visual_embedding = visual_embedding
        self.client = QdrantClient(path=str(self.version / "qdrant"))
        self.parents = {r["chunk_id"]: r for r in _read_jsonl(self.version / "parents.jsonl")}
        self.children = {r["chunk_id"]: r for r in _read_jsonl(self.version / "children.jsonl")}
        self.images = {r["image_id"]: r for r in _read_jsonl(self.version / "images.jsonl")}
        self.models = {m["model_name"]: m for m in self.manifest["manuals"]}
        self._bm25: dict[tuple[str, str | None], tuple[list[str], BM25Okapi]] = {}

    def close(self) -> None:
        self.client.close()

    def _active_doc_id(self, model_name: str, doc_id: str | None) -> str:
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}; available: {list(self.models)}")
        active_doc_id = self.models[model_name]["doc_id"]
        if doc_id and doc_id != active_doc_id:
            raise ValueError("doc_id does not belong to the selected model")
        return active_doc_id

    def _bm25_search(self, query: str, model_name: str, doc_id: str | None, limit: int) -> list[str]:
        key = (model_name, doc_id)
        if key not in self._bm25:
            records = [r for r in self.children.values()
                       if r["metadata"]["model_name"] == model_name
                       and (not doc_id or r["metadata"]["doc_id"] == doc_id)]
            ids = [r["chunk_id"] for r in records]
            corpus = [list(jieba.cut_for_search(_record_text(r))) for r in records]
            self._bm25[key] = (ids, BM25Okapi(corpus))
        ids, index = self._bm25[key]
        scores = index.get_scores(list(jieba.cut_for_search(query)))
        return [ids[i] for i in sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:limit]]

    @staticmethod
    def _fuse(channels: dict[str, list[str]], limit: int) -> list[tuple[str, float, dict[str, int]]]:
        scores: dict[str, float] = defaultdict(float)
        ranks: dict[str, dict[str, int]] = defaultdict(dict)
        for channel, ids in channels.items():
            for rank, record_id in enumerate(ids, 1):
                scores[record_id] += 1.0 / (60 + rank)
                ranks[record_id][channel] = rank
        ordered = sorted(scores, key=lambda record_id: (-scores[record_id], record_id))[:limit]
        return [(record_id, scores[record_id], ranks[record_id]) for record_id in ordered]

    def search_text(self, query: str, model_name: str, *, doc_id: str | None = None,
                    top_k: int = 6, candidate_k: int = 24, max_per_parent: int = 2) -> list[dict[str, Any]]:
        doc_id = self._active_doc_id(model_name, doc_id)
        query_vector = self.text_embedding.embed_query(query)
        points = self.client.query_points(
            TEXT_COLLECTION, query=query_vector, query_filter=_filter(model_name, doc_id),
            limit=candidate_k, with_payload=True,
        ).points
        vector_ids = [p.payload["record_id"] for p in points]
        bm25_ids = self._bm25_search(query, model_name, doc_id, candidate_k)
        chosen: list[dict[str, Any]] = []
        parent_counts: dict[str, int] = defaultdict(int)
        for record_id, score, ranks in self._fuse({"vector": vector_ids, "bm25": bm25_ids}, candidate_k * 2):
            record = self.children[record_id]
            parent_id = record["metadata"]["parent_id"]
            if parent_counts[parent_id] >= max_per_parent:
                continue
            parent_counts[parent_id] += 1
            chosen.append({"chunk_id": record_id, "content": record["page_content"],
                           "metadata": record["metadata"], "parent_id": parent_id,
                           "parent_content": self.parents[parent_id]["page_content"],
                           "fused_score": score, "ranks": ranks})
            if len(chosen) >= top_k:
                break
        return chosen

    def search_images(self, query: str, model_name: str, *, doc_id: str | None = None,
                      top_k: int = 3, candidate_k: int = 12) -> list[dict[str, Any]]:
        doc_id = self._active_doc_id(model_name, doc_id)
        scope = _filter(model_name, doc_id)
        text_points = self.client.query_points(
            IMAGE_COLLECTION, query=self.text_embedding.embed_query(query), using=IMAGE_TEXT_VECTOR,
            query_filter=scope, limit=candidate_k, with_payload=True,
        ).points
        visual_points = self.client.query_points(
            IMAGE_COLLECTION, query=self.visual_embedding.embed_query(query), using=IMAGE_VISUAL_VECTOR,
            query_filter=scope, limit=candidate_k, with_payload=True,
        ).points
        channels = {"nearby_text": [p.payload["record_id"] for p in text_points],
                    "visual": [p.payload["record_id"] for p in visual_points]}
        return [{"image_id": image_id, "nearby_text": self.images[image_id]["nearby_text"],
                 "metadata": self.images[image_id]["metadata"], "fused_score": score, "ranks": ranks}
                for image_id, score, ranks in self._fuse(channels, top_k)]

    def compare_text(self, query: str, model_names: list[str], **kwargs: Any) -> dict[str, list[dict[str, Any]]]:
        if len(set(model_names)) != len(model_names) or len(model_names) < 2:
            raise ValueError("Comparison requires two or more distinct models")
        return {name: self.search_text(query, name, **kwargs) for name in model_names}
