#!/usr/bin/env python
# coding: utf-8
"""Persistent multimodal corpus index and hybrid retriever.

The current PDF parser produces text, table, and image evidence. Images are
retrieved through their topic/nearby text while the original image path is
kept in metadata for display or later vision-model processing.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import jieba

INDEX_NAME = "multimodal"
VISUAL_INDEX_NAME = "multimodal_visual"
DOCS_FILE = "documents.jsonl"
MANIFEST_FILE = "manifest.json"


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "、".join(str(item) for item in value if item is not None)
    return str(value).strip()


def _index_text(chunk: Dict[str, Any]) -> str:
    metadata = chunk.get("metadata") or {}
    fields = [
        f"模块：{_as_text(metadata.get('module'))}",
        f"主题：{_as_text(metadata.get('topic_title'))}",
        f"证据类型：{_as_text(chunk.get('doc_type'))}/{_as_text(chunk.get('chunk_type'))}",
        _as_text(chunk.get("page_content")),
    ]
    nearby_text = _as_text(metadata.get("nearby_text"))
    if nearby_text and nearby_text not in fields[-1]:
        fields.append(f"图片附近文本：{nearby_text}")
    return "\n".join(field for field in fields if field and not field.endswith("："))


def _embedding_identifier() -> str:
    from config import (
        EMBEDDING_API_TYPE,
        EMBEDDING_BACKEND,
        EMBEDDING_MODEL_NAME,
        EMBEDDING_MODEL_PATH,
        OLLAMA_EMBEDDING_MODEL,
    )

    backend = EMBEDDING_BACKEND.lower()
    if backend == "api":
        return f"api:{EMBEDDING_API_TYPE}:{EMBEDDING_MODEL_NAME}"
    if backend == "ollama":
        return f"ollama:{OLLAMA_EMBEDDING_MODEL}"
    return f"local:{EMBEDDING_MODEL_PATH}"


def load_chunk_documents(chunks_path: str | Path) -> List[Dict[str, Any]]:
    """Load parser output and normalize it into persistable index documents."""
    path = Path(chunks_path)
    chunks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(chunks, list):
        raise ValueError(f"{path} must contain a JSON list")

    documents: List[Dict[str, Any]] = []
    seen_ids = set()
    for position, chunk in enumerate(chunks):
        raw_content = _as_text(chunk.get("page_content"))
        metadata = dict(chunk.get("metadata") or {})
        chunk_id = _as_text(chunk.get("chunk_id")) or f"chunk-{position}"
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)

        doc_type = _as_text(chunk.get("doc_type")) or "text"
        if not raw_content and not _as_text(metadata.get("nearby_text")):
            continue

        metadata.update(
            {
                "chunk_id": chunk_id,
                "doc_type": doc_type,
                "chunk_type": _as_text(chunk.get("chunk_type")),
                "raw_content": raw_content,
                "indexable": metadata.get("indexable", True) is not False,
            }
        )
        documents.append(
            {
                "chunk_id": chunk_id,
                "index_text": _index_text(chunk),
                "metadata": metadata,
            }
        )
    return documents


def _write_jsonl(path: Path, documents: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class MultimodalIndexBuilder:
    def __init__(self, chunks_path: str | Path, store_dir: str | Path):
        self.chunks_path = Path(chunks_path)
        self.store_dir = Path(store_dir)

    def build(self, batch_size: int = 128, progress_callback=None) -> Dict[str, Any]:
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
        from embedding_model import get_embedding_model
        from config import ENABLE_VISUAL_RETRIEVAL

        documents = load_chunk_documents(self.chunks_path)
        if not documents:
            raise ValueError("No usable chunks were found")
        indexed_documents = [
            item for item in documents if item["metadata"].get("indexable", True)
        ]
        if not indexed_documents:
            raise ValueError("No indexable chunks were found")

        self.store_dir.mkdir(parents=True, exist_ok=True)
        embeddings = get_embedding_model()
        langchain_docs = [
            Document(page_content=item["index_text"], metadata=item["metadata"])
            for item in indexed_documents
        ]

        vector_store = None
        for start in range(0, len(langchain_docs), batch_size):
            batch = langchain_docs[start : start + batch_size]
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                vector_store.add_documents(batch)
            done = min(start + batch_size, len(langchain_docs))
            print(f"  [index] embedded {done}/{len(langchain_docs)}")
            if progress_callback:
                progress_callback(done, len(langchain_docs))

        vector_store.save_local(str(self.store_dir), index_name=INDEX_NAME)
        _write_jsonl(self.store_dir / DOCS_FILE, documents)

        # Codex-MM-ADD: 在文本索引之外，额外为图片块构建 CLIP 视觉向量索引。
        # 这一路解决“用户描述图片外观/图标形状”时，纯 nearby_text 难以召回的问题。
        visual_status = self._build_visual_index(documents) if ENABLE_VISUAL_RETRIEVAL else {
            "enabled": False,
            "status": "disabled",
            "document_count": 0,
        }

        source_hash = hashlib.sha256(self.chunks_path.read_bytes()).hexdigest()
        counts = Counter(item["metadata"]["doc_type"] for item in indexed_documents)
        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": str(self.chunks_path.resolve()),
            "source_sha256": source_hash,
            "embedding": _embedding_identifier(),
            "document_count": len(indexed_documents),
            "source_chunk_count": len(documents),
            "doc_type_counts": dict(sorted(counts.items())),
            "index_name": INDEX_NAME,
            "visual_retrieval": visual_status,
        }
        (self.store_dir / MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    def _build_visual_index(self, documents: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
        from visual_embedding_model import get_visual_embedding_model

        image_documents = []
        for item in documents:
            metadata = item["metadata"]
            image_path = metadata.get("image_path")
            if metadata.get("doc_type") == "image" and image_path and Path(image_path).exists():
                visual_metadata = dict(metadata)
                visual_metadata["visual_index_text"] = item["index_text"]
                image_documents.append(Document(page_content=str(image_path), metadata=visual_metadata))

        if not image_documents:
            return {"enabled": True, "status": "empty", "document_count": 0}

        try:
            visual_embeddings = get_visual_embedding_model()
            visual_store = FAISS.from_documents(image_documents, visual_embeddings)
            visual_store.save_local(str(self.store_dir), index_name=VISUAL_INDEX_NAME)
            print(f"  [visual-index] embedded images {len(image_documents)}")
            return {
                "enabled": True,
                "status": "ok",
                "document_count": len(image_documents),
                "index_name": VISUAL_INDEX_NAME,
            }
        except Exception as exc:
            print(f"  [visual-index] failed, text/image-text retrieval remains available: {exc}")
            return {
                "enabled": True,
                "status": "failed",
                "document_count": 0,
                "error": str(exc),
            }


class MultimodalBM25:
    """BM25 retriever following the same jieba + LangChain style as bm25_retriever.py."""

    def __init__(self, documents: Sequence[Dict[str, Any]]):
        from langchain_community.retrievers import BM25Retriever
        from langchain_core.documents import Document

        documents = [
            item for item in documents if item["metadata"].get("indexable", True)
        ]
        self.documents = [
            Document(
                page_content=item["index_text"],
                metadata=item["metadata"],
            )
            for item in documents
        ]
        tokenized_documents = [
            Document(
                page_content=" ".join(jieba.cut_for_search(item["index_text"])),
                metadata={"id": index},
            )
            for index, item in enumerate(documents)
        ]
        self.retriever = BM25Retriever.from_documents(tokenized_documents)

    def GetBM25TopK(self, query: str, topk: int):
        self.retriever.k = topk
        tokenized_query = " ".join(jieba.cut_for_search(query))
        result = self.retriever.invoke(tokenized_query)
        return [
            self.documents[document.metadata["id"]]
            for document in result
        ]


@dataclass
class SearchResult:
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    fused_score: float
    vector_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    visual_rank: Optional[int] = None
    vector_distance: Optional[float] = None
    visual_distance: Optional[float] = None
    fusion_group: str = "text"
    group_rank: Optional[int] = None
    image_text_rank: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "fused_score": round(self.fused_score, 6),
            "vector_rank": self.vector_rank,
            "bm25_rank": self.bm25_rank,
            "visual_rank": self.visual_rank,
            "vector_distance": self.vector_distance,
            "visual_distance": self.visual_distance,
            "fusion_group": self.fusion_group,
            "group_rank": self.group_rank,
            "image_text_rank": self.image_text_rank,
            "metadata": self.metadata,
        }


class MultimodalRetriever:
    def __init__(self, store_dir: str | Path, load_visual: bool = True):
        from langchain_community.vectorstores import FAISS
        from embedding_model import get_embedding_model

        self.store_dir = Path(store_dir)
        manifest = load_manifest(self.store_dir)
        configured_embedding = _embedding_identifier()
        if manifest.get("embedding") != configured_embedding:
            raise ValueError(
                "The index embedding model does not match the current configuration: "
                f"index={manifest.get('embedding')}, current={configured_embedding}. "
                "Run the build command again."
            )
        self.documents = _read_jsonl(self.store_dir / DOCS_FILE)
        self.by_id = {item["chunk_id"]: item for item in self.documents}
        self.bm25 = MultimodalBM25(self.documents)
        self.embeddings = get_embedding_model()
        self.vector_store = FAISS.load_local(
            str(self.store_dir),
            self.embeddings,
            index_name=INDEX_NAME,
            allow_dangerous_deserialization=True,
        )
        # Codex-MM-ADD: 视觉索引是可选能力。旧索引或未配置 CLIP 时保持 None，
        # search() 会自动退回文本向量 + BM25，不影响原项目运行。
        self.visual_store = None
        self.visual_embeddings = None
        visual_info = manifest.get("visual_retrieval") or {}
        if load_visual and visual_info.get("status") == "ok":
            try:
                from visual_embedding_model import get_visual_embedding_model

                self.visual_embeddings = get_visual_embedding_model()
                self.visual_store = FAISS.load_local(
                    str(self.store_dir),
                    self.visual_embeddings,
                    index_name=VISUAL_INDEX_NAME,
                    allow_dangerous_deserialization=True,
                )
            except Exception as exc:
                print(f"  [VisualRetrieval] failed to load visual index: {exc}")
        self.last_search_info: Dict[str, Any] = {}

    def search(
        self,
        query: str,
        top_k: int = 8,
        candidate_k: int = 24,
        doc_types: Optional[Iterable[str]] = None,
        use_rerank: bool = False,
        max_per_topic: int = 2,
    ) -> List[Dict[str, Any]]:
        allowed_types = set(doc_types or [])
        # Keep both retrieval channels equally sized before RRF fusion.
        self.last_search_info = {
            "vector_status": "ok",
            "bm25_status": "ok",
            "visual_status": "disabled" if getattr(self, "visual_store", None) is None else "ok",
            "fusion_strategy": "layered_text_image_rrf",
            "errors": [],
        }
        try:
            vector_results = self.vector_store.similarity_search_with_score(query, k=candidate_k)
        except Exception as exc:
            vector_results = []
            self.last_search_info["vector_status"] = "failed"
            self.last_search_info["errors"].append(f"vector retrieval failed: {exc}")
            print(f"  [RAG] Vector retrieval failed, falling back to BM25: {exc}")
        bm25_docs = self.bm25.GetBM25TopK(query, candidate_k)

        # Codex-MM-ADD: 第三路召回。用户用“图标长什么样/按钮图片/标识位置”
        # 这类视觉表达提问时，CLIP 文本查询可以直接召回图片向量。
        visual_results = []
        # Codex-EVAL-ADD: text/table-only evaluation must not call the visual
        # index, otherwise unrelated visual API latency pollutes its metrics.
        visual_is_allowed = not allowed_types or "image" in allowed_types
        if getattr(self, "visual_store", None) is not None and visual_is_allowed:
            try:
                # Codex-MM-ADD: 三路召回候选数保持一致，避免视觉召回或文本召回
                # 因数量不同在 RRF 前天然占优。
                visual_results = self.visual_store.similarity_search_with_score(query, k=candidate_k)
            except Exception as exc:
                self.last_search_info["visual_status"] = "failed"
                self.last_search_info["errors"].append(f"visual retrieval failed: {exc}")
                print(f"  [RAG] Visual retrieval failed, continuing with text retrieval: {exc}")

        ranks: Dict[str, Dict[str, int]] = defaultdict(dict)
        vector_distances: Dict[str, float] = {}
        visual_distances: Dict[str, float] = {}
        for rank, (document, distance) in enumerate(vector_results, start=1):
            chunk_id = document.metadata.get("chunk_id")
            if chunk_id:
                ranks[chunk_id]["vector"] = rank
                vector_distances[chunk_id] = float(distance)
        for rank, document in enumerate(bm25_docs, start=1):
            chunk_id = document.metadata.get("chunk_id")
            if chunk_id:
                ranks[chunk_id]["bm25"] = rank
        for rank, (document, distance) in enumerate(visual_results, start=1):
            chunk_id = document.metadata.get("chunk_id")
            if chunk_id:
                ranks[chunk_id]["visual"] = rank
                visual_distances[chunk_id] = float(distance)

        # Codex-MM-LAYERED-FUSION:
        # Text/table evidence and image evidence are ranked independently.
        # An image's vector + BM25 hits first become one image-text rank, then
        # that rank is fused with its visual rank. This prevents images from
        # receiving three independent RRF votes against text evidence.
        text_scores: Dict[str, float] = {}
        image_text_scores: Dict[str, float] = {}
        image_text_ranks: Dict[str, int] = {}
        image_scores: Dict[str, float] = {}
        for chunk_id, source_ranks in ranks.items():
            item = self.by_id.get(chunk_id)
            if not item:
                continue
            metadata = item["metadata"]
            if allowed_types and metadata.get("doc_type") not in allowed_types:
                continue
            text_channel_score = self._rrf_score(
                source_ranks.get("vector"),
                source_ranks.get("bm25"),
            )
            if metadata.get("doc_type") == "image":
                if text_channel_score:
                    image_text_scores[chunk_id] = text_channel_score
            elif text_channel_score:
                text_scores[chunk_id] = text_channel_score

        for rank, (chunk_id, _) in enumerate(
            sorted(image_text_scores.items(), key=lambda item: item[1], reverse=True),
            start=1,
        ):
            image_text_ranks[chunk_id] = rank

        image_ids = {
            chunk_id
            for chunk_id, source_ranks in ranks.items()
            if self.by_id.get(chunk_id, {}).get("metadata", {}).get("doc_type") == "image"
            and (
                not allowed_types
                or self.by_id.get(chunk_id, {}).get("metadata", {}).get("doc_type") in allowed_types
            )
            and (source_ranks.get("visual") is not None or chunk_id in image_text_ranks)
        }
        for chunk_id in image_ids:
            image_scores[chunk_id] = self._rrf_score(
                image_text_ranks.get(chunk_id),
                ranks[chunk_id].get("visual"),
            )

        candidates: List[SearchResult] = []
        group_scores = {**text_scores, **image_scores}
        group_ranks: Dict[str, int] = {}
        for scores in (text_scores, image_scores):
            for rank, (chunk_id, _) in enumerate(
                sorted(scores.items(), key=lambda item: item[1], reverse=True),
                start=1,
            ):
                group_ranks[chunk_id] = rank

        for chunk_id, score in group_scores.items():
            source_ranks = ranks[chunk_id]
            item = self.by_id.get(chunk_id)
            if not item:
                continue
            metadata = item["metadata"]
            content = metadata.get("raw_content") or item["index_text"]
            result_metadata = dict(metadata)
            fusion_group = "image" if metadata.get("doc_type") == "image" else "text"
            result_metadata.update(
                {
                    "fusion_group": fusion_group,
                    "group_rank": group_ranks.get(chunk_id),
                    "image_text_rank": image_text_ranks.get(chunk_id),
                }
            )
            parent_id = _as_text(metadata.get("parent_id"))
            parent = self.by_id.get(parent_id) if parent_id else None
            if parent and metadata.get("chunk_type") == "table_row_child":
                # Codex-TABLE-ADD: retrieve by the precise row child, then give
                # the Agent the complete parent table for answer generation.
                parent_content = parent["metadata"].get("raw_content") or parent["index_text"]
                result_metadata["matched_row_content"] = content
                result_metadata["parent_chunk_id"] = parent_id
                content = f"命中表格行：\n{content}\n\n完整表格：\n{parent_content}"
            candidates.append(
                SearchResult(
                    chunk_id=chunk_id,
                    content=content,
                    metadata=result_metadata,
                    fused_score=score,
                    vector_rank=source_ranks.get("vector"),
                    bm25_rank=source_ranks.get("bm25"),
                    visual_rank=source_ranks.get("visual"),
                    vector_distance=vector_distances.get(chunk_id),
                    visual_distance=visual_distances.get(chunk_id),
                    fusion_group=fusion_group,
                    group_rank=group_ranks.get(chunk_id),
                    image_text_rank=image_text_ranks.get(chunk_id),
                )
            )
        candidates = self._merge_evidence_groups(candidates)

        if use_rerank and candidates:
            candidates = self._rerank(query, candidates)

        if max_per_topic <= 0:
            return [result.as_dict() for result in candidates[:top_k]]

        selected: List[SearchResult] = []
        topic_counts: Counter[str] = Counter()
        for result in candidates:
            topic_id = _as_text(result.metadata.get("topic_id")) or result.chunk_id
            if topic_counts[topic_id] >= max_per_topic:
                continue
            selected.append(result)
            topic_counts[topic_id] += 1
            if len(selected) >= top_k:
                break
        return [result.as_dict() for result in selected]

    @staticmethod
    def _rrf_score(*ranks: Optional[int], constant: int = 60) -> float:
        return sum(1.0 / (constant + rank) for rank in ranks if rank is not None)

    @staticmethod
    def _merge_evidence_groups(
        candidates: Sequence[SearchResult],
        text_per_image: int = 3,
    ) -> List[SearchResult]:
        """Interleave independently ranked text and image evidence.

        The resulting pool reserves roughly one image slot per three text/table
        slots while allowing either group to fill all remaining positions.
        A downstream reranker can then compare evidence content without images
        gaining an extra vote merely because they have a visual index.
        """
        text_candidates = sorted(
            (item for item in candidates if item.fusion_group == "text"),
            key=lambda item: item.group_rank or 10**9,
        )
        image_candidates = sorted(
            (item for item in candidates if item.fusion_group == "image"),
            key=lambda item: item.group_rank or 10**9,
        )
        merged: List[SearchResult] = []
        text_index = 0
        image_index = 0
        while text_index < len(text_candidates) or image_index < len(image_candidates):
            for _ in range(text_per_image):
                if text_index >= len(text_candidates):
                    break
                merged.append(text_candidates[text_index])
                text_index += 1
            if image_index < len(image_candidates):
                merged.append(image_candidates[image_index])
                image_index += 1
            if text_index >= len(text_candidates):
                merged.extend(image_candidates[image_index:])
                break
            if image_index >= len(image_candidates):
                merged.extend(text_candidates[text_index:])
                break
        return merged

    @staticmethod
    def build_context(results: Sequence[Dict[str, Any]], max_chars: int = 5000) -> str:
        """Format retrieval results into bounded, source-aware Agent context."""
        sections: List[str] = []
        used = 0
        for position, result in enumerate(results, start=1):
            metadata = result["metadata"]
            source = (
                f"[证据{position} | {metadata.get('doc_type')} | "
                f"{metadata.get('topic_title') or '未命名主题'} | "
                f"页码{metadata.get('pages') or metadata.get('page_id')}]"
            )
            image_line = f"\n图片路径：{metadata['image_path']}" if metadata.get("image_path") else ""
            section = f"{source}\n{result['content']}{image_line}\n"
            if used + len(section) > max_chars:
                remaining = max_chars - used
                if remaining > len(source) + 20:
                    sections.append(section[:remaining])
                break
            sections.append(section)
            used += len(section)
        return "\n".join(sections)

    def expand_text_parent_documents(self, documents):
        """Replace reranked text children with their complete non-indexed parents."""
        from langchain_core.documents import Document

        expanded = []
        seen_parent_ids = set()
        for document in documents:
            metadata = dict(document.metadata)
            parent_id = _as_text(metadata.get("parent_id"))
            if metadata.get("chunk_type") != "text_child" or not parent_id:
                expanded.append(document)
                continue
            parent = self.by_id.get(parent_id)
            if not parent:
                expanded.append(document)
                continue
            if parent_id in seen_parent_ids:
                continue
            seen_parent_ids.add(parent_id)
            parent_metadata = dict(parent["metadata"])
            parent_metadata.update(
                {
                    "chunk_id": parent_id,
                    "matched_child_id": metadata.get("chunk_id"),
                    "matched_child_content": document.page_content,
                    "matched_child_vector_rank": metadata.get("vector_rank"),
                    "matched_child_bm25_rank": metadata.get("bm25_rank"),
                    "fused_score": metadata.get("fused_score"),
                    "vector_rank": metadata.get("vector_rank"),
                    "bm25_rank": metadata.get("bm25_rank"),
                    "vector_distance": metadata.get("vector_distance"),
                }
            )
            expanded.append(
                Document(
                    page_content=parent_metadata.get("raw_content") or parent["index_text"],
                    metadata=parent_metadata,
                )
            )
        return expanded

    @staticmethod
    def _rerank(query: str, candidates: List[SearchResult]) -> List[SearchResult]:
        from langchain_core.documents import Document
        from rerank_model import reRankLLM

        docs = [
            Document(page_content=item.content, metadata={"chunk_id": item.chunk_id})
            for item in candidates
        ]
        ranked_docs = reRankLLM().predict(query, docs)
        by_id = {item.chunk_id: item for item in candidates}
        return [by_id[doc.metadata["chunk_id"]] for doc in ranked_docs if doc.metadata["chunk_id"] in by_id]


def load_manifest(store_dir: str | Path) -> Dict[str, Any]:
    return json.loads((Path(store_dir) / MANIFEST_FILE).read_text(encoding="utf-8"))
