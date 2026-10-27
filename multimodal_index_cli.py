#!/usr/bin/env python
# coding: utf-8
"""CLI for building and querying the multimodal PDF index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_retriever import (
    MultimodalIndexBuilder,
    MultimodalRetriever,
    load_chunk_documents,
    load_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and search the multimodal RAG index")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Vectorize chunks and create a persistent index")
    build.add_argument("--chunks", default="data/multimodal_chunks.json")
    build.add_argument("--store", default="multimodal_vector_store")
    build.add_argument("--batch-size", type=int, default=128)

    search = subparsers.add_parser("search", help="Run hybrid vector + BM25 retrieval")
    search.add_argument("query")
    search.add_argument("--store", default="multimodal_vector_store")
    search.add_argument("--top-k", type=int, default=8)
    search.add_argument("--candidate-k", type=int, default=24)
    search.add_argument("--doc-types", help="Comma-separated: text,image,table")
    search.add_argument("--rerank", action="store_true")
    search.add_argument("--json", action="store_true", dest="as_json")

    stats = subparsers.add_parser("stats", help="Show source or built-index statistics")
    stats.add_argument("--chunks", default="data/multimodal_chunks.json")
    stats.add_argument("--store", default="multimodal_vector_store")
    return parser


def _print_result(position: int, result: dict) -> None:
    metadata = result["metadata"]
    print(
        f"\n[{position}] {metadata.get('doc_type')}/{metadata.get('chunk_type')} "
        f"score={result['fused_score']} vector_rank={result['vector_rank']} "
        f"bm25_rank={result['bm25_rank']} visual_rank={result.get('visual_rank')}"
    )
    print(
        f"    topic={metadata.get('topic_title')} pages={metadata.get('pages') or metadata.get('page_id')}"
    )
    if metadata.get("image_path"):
        print(f"    image={metadata['image_path']}")
    content = result["content"].replace("\n", " ")
    print(f"    {content[:500]}")


def main() -> None:
    args = _parser().parse_args()
    if args.command == "build":
        manifest = MultimodalIndexBuilder(args.chunks, args.store).build(args.batch_size)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    if args.command == "search":
        doc_types = args.doc_types.split(",") if args.doc_types else None
        results = MultimodalRetriever(args.store).search(
            args.query,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            doc_types=doc_types,
            use_rerank=args.rerank,
        )
        if args.as_json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            for position, result in enumerate(results, start=1):
                _print_result(position, result)
        return

    store = Path(args.store)
    if (store / "manifest.json").exists():
        print(json.dumps(load_manifest(store), ensure_ascii=False, indent=2))
    else:
        documents = load_chunk_documents(args.chunks)
        counts = {}
        indexed_documents = [
            document for document in documents if document["metadata"].get("indexable", True)
        ]
        for document in indexed_documents:
            doc_type = document["metadata"]["doc_type"]
            counts[doc_type] = counts.get(doc_type, 0) + 1
        print(
            json.dumps(
                {
                    "document_count": len(indexed_documents),
                    "source_chunk_count": len(documents),
                    "doc_type_counts": counts,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
