#!/usr/bin/env python
# coding: utf-8
"""Compare visual-only retrieval with visual + image-text vector fusion."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from multimodal_retriever import INDEX_NAME, VISUAL_INDEX_NAME


DEFAULT_DATASET = "data/evaluation/image_retrieval_eval_100.json"
DEFAULT_STORE = "data/knowledge_bases/train_a/vector_store"
DEFAULT_OUTPUT = "data/evaluation/results/image_retrieval_comparison.json"
DEFAULT_REPORT = "data/evaluation/results/image_retrieval_comparison.md"


def unique_in_order(items: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(item for item in items if item))


def hit_at_k(retrieved: Sequence[str], expected: Iterable[str], k: int) -> float:
    return float(bool(set(retrieved[:k]) & set(expected)))


def recall_at_k(retrieved: Sequence[str], expected: Iterable[str], k: int) -> float:
    expected_set = set(expected)
    if not expected_set:
        return 0.0
    return len(set(retrieved[:k]) & expected_set) / len(expected_set)


def reciprocal_rank(retrieved: Sequence[str], expected: Iterable[str]) -> float:
    expected_set = set(expected)
    for rank, item_id in enumerate(retrieved, start=1):
        if item_id in expected_set:
            return 1.0 / rank
    return 0.0


def percentile(values: Sequence[float], percentage: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def document_id(document: Any) -> str:
    return str((document.metadata or {}).get("chunk_id") or "")


def document_topic(document: Any) -> str:
    return str((document.metadata or {}).get("topic_title") or "")


def ranked_items(results: Sequence[Tuple[Any, float]]) -> List[Dict[str, Any]]:
    return [
        {
            "chunk_id": document_id(document),
            "topic_title": document_topic(document),
            "distance": round(float(distance), 6),
        }
        for document, distance in results
        if document_id(document)
    ]


def rrf_fuse(
    visual_results: Sequence[Tuple[Any, float]],
    image_text_results: Sequence[Tuple[Any, float]],
    constant: int = 60,
) -> List[Dict[str, Any]]:
    ranks: Dict[str, Dict[str, int]] = defaultdict(dict)
    documents: Dict[str, Any] = {}
    distances: Dict[str, Dict[str, float]] = defaultdict(dict)
    for source, results in (("visual", visual_results), ("image_text", image_text_results)):
        for rank, (document, distance) in enumerate(results, start=1):
            chunk_id = document_id(document)
            if not chunk_id:
                continue
            ranks[chunk_id][source] = rank
            documents[chunk_id] = document
            distances[chunk_id][source] = float(distance)

    fused = []
    for chunk_id, source_ranks in ranks.items():
        score = sum(1.0 / (constant + rank) for rank in source_ranks.values())
        fused.append(
            {
                "chunk_id": chunk_id,
                "topic_title": document_topic(documents[chunk_id]),
                "fused_score": score,
                "visual_rank": source_ranks.get("visual"),
                "image_text_rank": source_ranks.get("image_text"),
                "visual_distance": distances[chunk_id].get("visual"),
                "image_text_distance": distances[chunk_id].get("image_text"),
            }
        )
    return sorted(fused, key=lambda item: (-item["fused_score"], item["chunk_id"]))


def evaluate_ranking(
    case: Dict[str, Any],
    ranking: Sequence[Dict[str, Any]],
    latency_ms: float,
    ks: Sequence[int],
    error: str | None = None,
) -> Dict[str, Any]:
    image_ids = unique_in_order(item["chunk_id"] for item in ranking)
    topics = [item.get("topic_title", "") for item in ranking]
    expected_images = case["expected_image_ids"]
    expected_topic = case["expected_topic"]
    return {
        "latency_ms": round(latency_ms, 3),
        "error": error,
        "retrieved_image_ids": image_ids,
        "retrieved_topics": topics,
        "expected_image_count": len(set(expected_images)),
        "matched_image_count": {
            f"@{k}": len(set(image_ids[:k]) & set(expected_images))
            for k in ks
        },
        "image_hit": {f"@{k}": hit_at_k(image_ids, expected_images, k) for k in ks},
        "image_recall": {f"@{k}": recall_at_k(image_ids, expected_images, k) for k in ks},
        "image_mrr": reciprocal_rank(image_ids, expected_images),
        "topic_hit": {
            f"@{k}": float(expected_topic in topics[:k])
            for k in ks
        },
    }


def aggregate(details: Sequence[Dict[str, Any]], ks: Sequence[int]) -> Dict[str, Any]:
    def mean(values: Iterable[float]) -> float:
        values = list(values)
        return statistics.fmean(values) if values else 0.0

    latencies = [detail["latency_ms"] for detail in details]
    total_expected_images = sum(detail.get("expected_image_count", 0) for detail in details)
    return {
        "case_count": len(details),
        "error_count": sum(bool(detail.get("error")) for detail in details),
        "image_hit": {
            f"@{k}": round(mean(detail["image_hit"][f"@{k}"] for detail in details), 6)
            for k in ks
        },
        "image_recall": {
            f"@{k}": round(mean(detail["image_recall"][f"@{k}"] for detail in details), 6)
            for k in ks
        },
        "micro_image_recall": {
            f"@{k}": round(
                (
                    sum(detail.get("matched_image_count", {}).get(f"@{k}", 0) for detail in details)
                    / total_expected_images
                )
                if total_expected_images
                else 0.0,
                6,
            )
            for k in ks
        },
        "image_mrr": round(mean(detail["image_mrr"] for detail in details), 6),
        "topic_hit": {
            f"@{k}": round(mean(detail["topic_hit"][f"@{k}"] for detail in details), 6)
            for k in ks
        },
        "latency_ms": {
            "average": round(mean(latencies), 3),
            "median": round(statistics.median(latencies), 3) if latencies else 0.0,
            "p95": round(percentile(latencies, 0.95), 3),
        },
    }


def metric_delta(combined: Dict[str, Any], visual: Dict[str, Any], ks: Sequence[int]) -> Dict[str, Any]:
    return {
        "image_hit": {
            f"@{k}": round(combined["image_hit"][f"@{k}"] - visual["image_hit"][f"@{k}"], 6)
            for k in ks
        },
        "image_recall": {
            f"@{k}": round(
                combined["image_recall"][f"@{k}"] - visual["image_recall"][f"@{k}"], 6
            )
            for k in ks
        },
        "micro_image_recall": {
            f"@{k}": round(
                combined["micro_image_recall"][f"@{k}"]
                - visual["micro_image_recall"][f"@{k}"],
                6,
            )
            for k in ks
        },
        "image_mrr": round(combined["image_mrr"] - visual["image_mrr"], 6),
        "topic_hit": {
            f"@{k}": round(combined["topic_hit"][f"@{k}"] - visual["topic_hit"][f"@{k}"], 6)
            for k in ks
        },
        "average_latency_ms": round(
            combined["latency_ms"]["average"] - visual["latency_ms"]["average"], 3
        ),
    }


def report_table(title: str, metrics: Dict[str, Any], ks: Sequence[int]) -> List[str]:
    header = "| Metric | " + " | ".join(f"@{k}" for k in ks) + " |"
    separator = "|---|" + "|".join("---:" for _ in ks) + "|"
    lines = [f"## {title}", "", header, separator]
    for label, key in (
        ("Image Hit", "image_hit"),
        ("Image Recall (Macro)", "image_recall"),
        ("Image Recall (Micro)", "micro_image_recall"),
        ("Topic Hit", "topic_hit"),
    ):
        values = " | ".join(f"{metrics[key][f'@{k}']:.4f}" for k in ks)
        lines.append(f"| {label} | {values} |")
    lines.extend(
        [
            "",
            f"- Image MRR: **{metrics['image_mrr']:.4f}**",
            f"- Average latency: **{metrics['latency_ms']['average']:.2f} ms**",
            f"- P95 latency: **{metrics['latency_ms']['p95']:.2f} ms**",
            f"- Errors: **{metrics['error_count']}**",
            "",
        ]
    )
    return lines


def render_report(output: Dict[str, Any]) -> str:
    ks = output["config"]["ks"]
    lines = [
        "# Image Retrieval Comparison Report",
        "",
        f"- Dataset: `{output['config']['dataset']}`",
        f"- Vector store: `{output['config']['store']}`",
        f"- Candidate K per channel: {output['config']['candidate_k']}",
        "",
    ]
    lines.extend(report_table("Visual Only", output["metrics"]["visual_only"], ks))
    lines.extend(
        report_table(
            "Visual + Image Text Vector RRF",
            output["metrics"]["visual_plus_image_text"],
            ks,
        )
    )
    lines.extend(["## Delta: Combined - Visual Only", ""])
    delta = output["metrics"]["delta"]
    for metric in ("image_hit", "image_recall", "micro_image_recall", "topic_hit"):
        values = ", ".join(f"@{k}={delta[metric][f'@{k}']:+.4f}" for k in ks)
        lines.append(f"- {metric}: {values}")
    lines.append(f"- image_mrr: {delta['image_mrr']:+.4f}")
    lines.append(f"- average_latency_ms: {delta['average_latency_ms']:+.2f}")
    lines.extend(["", "## Metrics By Query Type", ""])
    for query_type, metrics in output["metrics_by_query_type"].items():
        lines.append(f"### {query_type}")
        lines.append("")
        lines.append(
            f"- Visual only: Hit@1={metrics['visual_only']['image_hit']['@1']:.4f}, "
            f"MRR={metrics['visual_only']['image_mrr']:.4f}"
        )
        lines.append(
            f"- Combined: Hit@1={metrics['visual_plus_image_text']['image_hit']['@1']:.4f}, "
            f"MRR={metrics['visual_plus_image_text']['image_mrr']:.4f}"
        )
        lines.append("")
    return "\n".join(lines)


def parse_ks(value: str) -> List[int]:
    ks = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not ks or any(k <= 0 for k in ks):
        raise argparse.ArgumentTypeError("ks must contain positive integers")
    return ks


def main() -> None:
    from embedding_model import get_embedding_model
    from langchain_community.vectorstores import FAISS
    from visual_embedding_model import get_visual_embedding_model

    parser = argparse.ArgumentParser(description="Compare two image-layer retrieval modes.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--store", default=DEFAULT_STORE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--ks", type=parse_ks, default=parse_ks("1,3,5"))
    parser.add_argument("--candidate-k", type=int, default=15)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    cases = dataset["cases"][: args.limit or None]
    max_k = max(args.ks)
    fetch_k = max(args.candidate_k * 10, 150)

    visual_embeddings = get_visual_embedding_model()
    text_embeddings = get_embedding_model()
    visual_store = FAISS.load_local(
        args.store,
        visual_embeddings,
        index_name=VISUAL_INDEX_NAME,
        allow_dangerous_deserialization=True,
    )
    text_store = FAISS.load_local(
        args.store,
        text_embeddings,
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True,
    )

    details = []
    print(f"[image-eval] cases={len(cases)} candidate_k={args.candidate_k}")
    for index, case in enumerate(cases, start=1):
        visual_error = None
        text_error = None
        visual_results: List[Tuple[Any, float]] = []
        image_text_results: List[Tuple[Any, float]] = []

        started = time.perf_counter()
        try:
            visual_results = visual_store.similarity_search_with_score(
                case["query"], k=args.candidate_k
            )
        except Exception as exc:
            visual_error = f"{type(exc).__name__}: {exc}"
        visual_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        try:
            image_text_results = text_store.similarity_search_with_score(
                case["query"],
                k=args.candidate_k,
                filter={"doc_type": "image"},
                fetch_k=fetch_k,
            )
        except Exception as exc:
            text_error = f"{type(exc).__name__}: {exc}"
        text_ms = (time.perf_counter() - started) * 1000

        fusion_started = time.perf_counter()
        visual_ranking = ranked_items(visual_results)[:max_k]
        combined_ranking = rrf_fuse(visual_results, image_text_results)[:max_k]
        fusion_ms = (time.perf_counter() - fusion_started) * 1000

        visual_detail = evaluate_ranking(
            case, visual_ranking, visual_ms, args.ks, error=visual_error
        )
        combined_error = "; ".join(error for error in (visual_error, text_error) if error) or None
        combined_detail = evaluate_ranking(
            case,
            combined_ranking,
            visual_ms + text_ms + fusion_ms,
            args.ks,
            error=combined_error,
        )
        details.append(
            {
                "case_id": case["case_id"],
                "query": case["query"],
                "query_type": case["query_type"],
                "expected_topic": case["expected_topic"],
                "expected_image_ids": case["expected_image_ids"],
                "visual_only": visual_detail,
                "visual_plus_image_text": combined_detail,
            }
        )
        print(
            f"[{index:03d}/{len(cases):03d}] {case['case_id']} "
            f"visual_H@{max_k}={visual_detail['image_hit'][f'@{max_k}']:.0f} "
            f"combined_H@{max_k}={combined_detail['image_hit'][f'@{max_k}']:.0f}"
        )

    visual_details = [detail["visual_only"] for detail in details]
    combined_details = [detail["visual_plus_image_text"] for detail in details]
    visual_metrics = aggregate(visual_details, args.ks)
    combined_metrics = aggregate(combined_details, args.ks)
    by_type: Dict[str, Dict[str, Any]] = {}
    for query_type in sorted({detail["query_type"] for detail in details}):
        selected = [detail for detail in details if detail["query_type"] == query_type]
        visual_type = aggregate([detail["visual_only"] for detail in selected], args.ks)
        combined_type = aggregate(
            [detail["visual_plus_image_text"] for detail in selected], args.ks
        )
        by_type[query_type] = {
            "visual_only": visual_type,
            "visual_plus_image_text": combined_type,
            "delta": metric_delta(combined_type, visual_type, args.ks),
        }

    output = {
        "config": {
            "dataset": args.dataset,
            "store": args.store,
            "ks": args.ks,
            "candidate_k": args.candidate_k,
            "fusion": "RRF(visual_rank, image_text_vector_rank)",
            "bm25_used": False,
        },
        "metrics": {
            "visual_only": visual_metrics,
            "visual_plus_image_text": combined_metrics,
            "delta": metric_delta(combined_metrics, visual_metrics, args.ks),
        },
        "metrics_by_query_type": by_type,
        "details": details,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(output), encoding="utf-8")
    print("\n" + render_report({**output, "details": []}))
    print(f"JSON: {output_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
