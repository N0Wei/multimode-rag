#!/usr/bin/env python
# coding: utf-8
"""Build and query a pure visual image-retrieval index.

This script intentionally does NOT use:
- BM25 keyword retrieval
- normal text-vector retrieval
- image nearby-text retrieval
- RRF fusion or reranking

Pipeline:
    image file -> multimodal embedding API -> visual-only FAISS index
    text query -> same multimodal embedding API -> visual-only FAISS search
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from visual_embedding_model import get_visual_embedding_model


INDEX_NAME = "visual_only"
MANIFEST_FILE = "visual_only_manifest.json"


def load_image_documents(chunks_path: Path, limit: int = 0):
    """Load only image paths and metadata; nearby text is never embedded."""
    from langchain_core.documents import Document

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    documents = []
    for chunk in chunks:
        if chunk.get("doc_type") != "image":
            continue
        metadata = dict(chunk.get("metadata") or {})
        image_path = Path(metadata.get("image_path") or "")
        if not image_path.exists():
            continue

        # Codex-MM-ADD: page_content is deliberately only the image path.
        # The visual embedding provider reads the image bytes from this path;
        # no topic title, nearby text, or page content participates in retrieval.
        metadata.update(
            {
                "chunk_id": chunk.get("chunk_id"),
                "doc_type": "image",
                "image_path": str(image_path),
            }
        )
        documents.append(Document(page_content=str(image_path), metadata=metadata))
        if limit > 0 and len(documents) >= limit:
            break
    return documents


def build_visual_only_index(chunks_path: Path, store_dir: Path, limit: int) -> None:
    from langchain_community.vectorstores import FAISS

    documents = load_image_documents(chunks_path, limit=limit)
    if not documents:
        raise ValueError("No valid image chunks were found")

    print("=" * 72)
    print("构建纯视觉图片索引")
    print(f"图片数量: {len(documents)}")
    print("不会使用文本向量、BM25、附近文本、RRF 或 rerank")
    print("=" * 72)

    embeddings = get_visual_embedding_model()
    vector_store = FAISS.from_documents(documents, embeddings)
    store_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(store_dir), index_name=INDEX_NAME)

    manifest = {
        "mode": "visual_only",
        "source_chunks": str(chunks_path.resolve()),
        "document_count": len(documents),
        "index_name": INDEX_NAME,
        "uses_nearby_text": False,
        "uses_bm25": False,
        "uses_text_vector_store": False,
        "uses_rrf": False,
        "uses_rerank": False,
    }
    (store_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n纯视觉索引已保存: {store_dir.resolve()}")


def search_visual_only_index(
    query: str,
    store_dir: Path,
    top_k: int,
    output_dir: Path | None,
) -> None:
    from langchain_community.vectorstores import FAISS

    embeddings = get_visual_embedding_model()
    vector_store = FAISS.load_local(
        str(store_dir),
        embeddings,
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True,
    )

    # Codex-MM-ADD: This is the only retrieval call in the script. The query
    # is embedded by the multimodal model and compared only with image vectors.
    results = vector_store.similarity_search_with_score(query, k=top_k)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"纯视觉检索 Query: {query}")
    print(f"Top K: {top_k}")
    print("=" * 72)
    for rank, (document, distance) in enumerate(results, start=1):
        metadata = document.metadata or {}
        image_path = Path(metadata.get("image_path") or document.page_content)
        print(f"\n[{rank}] visual_distance={float(distance):.6f}")
        print(f"    topic={metadata.get('topic_title') or '未命名主题'}")
        print(f"    pages={metadata.get('pages') or metadata.get('page_id') or '未知'}")
        print(f"    image={image_path}")

        if output_dir and image_path.exists():
            suffix = image_path.suffix or ".png"
            destination = output_dir / f"{rank:02d}_{image_path.stem}{suffix}"
            shutil.copy2(image_path, destination)

    if output_dir:
        print(f"\n命中图片已复制到: {output_dir.resolve()}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test pure multimodal visual image retrieval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build a visual-only image index")
    build.add_argument(
        "--chunks",
        default="data/knowledge_bases/train_a/chunks.json",
        help="Multimodal chunks JSON containing image_path metadata",
    )
    build.add_argument(
        "--store",
        default="data/knowledge_bases/train_a/visual_only_test_store",
        help="Independent output directory for the visual-only test index",
    )
    build.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Number of images to embed for a quick API test; use 0 for all images",
    )

    search = subparsers.add_parser("search", help="Search only the visual image index")
    search.add_argument("query", help="Text query embedded by the multimodal model")
    search.add_argument(
        "--store",
        default="data/knowledge_bases/train_a/visual_only_test_store",
    )
    search.add_argument("--top-k", type=int, default=6)
    search.add_argument(
        "--output-dir",
        default="data/visual_only_search_results",
        help="Copy matched images here; pass an empty string to disable",
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()
    if args.command == "build":
        build_visual_only_index(Path(args.chunks), Path(args.store), args.limit)
        return

    output_dir = Path(args.output_dir) if args.output_dir else None
    search_visual_only_index(args.query, Path(args.store), args.top_k, output_dir)


if __name__ == "__main__":
    main()
