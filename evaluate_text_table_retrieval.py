#!/usr/bin/env python
# coding: utf-8
"""Evaluate unified text/table child retrieval against parent-child ground truth."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

# A retrieval benchmark should not create 100 LangSmith traces by default.
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from multimodal_retriever import MultimodalRetriever


DEFAULT_DATASET = "data/evaluation/text_table_retrieval_eval_100.json"
DEFAULT_STORE = "data/knowledge_bases/train_a/vector_store"
DEFAULT_OUTPUT = "data/evaluation/results/text_table_retrieval_metrics.json"
DEFAULT_REPORT = "data/evaluation/results/text_table_retrieval_report.md"


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


def unique_in_order(items: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(item for item in items if item))


def result_child_id(result: Dict[str, Any]) -> str:
    return str(result.get("chunk_id") or result.get("metadata", {}).get("chunk_id") or "")


def result_parent_id(result: Dict[str, Any]) -> str:
    metadata = result.get("metadata") or {}
    return str(metadata.get("parent_id") or metadata.get("parent_chunk_id") or "")


def percentile(values: Sequence[float], percentage: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def evaluate_case(
    case: Dict[str, Any],
    results: Sequence[Dict[str, Any]],
    latency_ms: float,
    ks: Sequence[int],
    search_info: Dict[str, Any] | None = None,
    error: str | None = None,
) -> Dict[str, Any]:
    child_ids = unique_in_order(result_child_id(result) for result in results)
    # Keep one parent ID per retrieved child. Duplicate parents must retain
    # their child ranks, otherwise Parent Recall@K/MRR would be overestimated.
    parent_ids = [result_parent_id(result) for result in results]
    expected_children = case["expected_child_ids"]
    expected_parents = case["expected_parent_ids"]
    return {
        "case_id": case["case_id"],
        "query": case["query"],
        "category": case["category"],
        "expected_topic": case["expected_topic"],
        "latency_ms": round(latency_ms, 3),
        "error": error,
        "retrieved_child_ids": child_ids,
        "retrieved_parent_ids": parent_ids,
        "retrieved_unique_parent_ids": unique_in_order(parent_ids),
        "expected_child_ids": expected_children,
        "expected_parent_ids": expected_parents,
        "child_recall": {
            f"@{k}": recall_at_k(child_ids, expected_children, k) for k in ks
        },
        "parent_recall": {
            f"@{k}": recall_at_k(parent_ids, expected_parents, k) for k in ks
        },
        "child_mrr": reciprocal_rank(child_ids, expected_children),
        "parent_mrr": reciprocal_rank(parent_ids, expected_parents),
        "search_info": search_info or {},
    }


def aggregate(details: Sequence[Dict[str, Any]], ks: Sequence[int]) -> Dict[str, Any]:
    count = len(details)

    def mean(values: Iterable[float]) -> float:
        values = list(values)
        return statistics.fmean(values) if values else 0.0

    latencies = [detail["latency_ms"] for detail in details]
    errors = [detail for detail in details if detail.get("error")]
    degraded = [
        detail
        for detail in details
        if any(
            detail.get("search_info", {}).get(channel) == "failed"
            for channel in ("vector_status", "bm25_status")
        )
    ]
    return {
        "case_count": count,
        "error_count": len(errors),
        "degraded_case_count": len(degraded),
        "child_recall": {
            f"@{k}": round(mean(detail["child_recall"][f"@{k}"] for detail in details), 6)
            for k in ks
        },
        "parent_recall": {
            f"@{k}": round(mean(detail["parent_recall"][f"@{k}"] for detail in details), 6)
            for k in ks
        },
        "child_mrr": round(mean(detail["child_mrr"] for detail in details), 6),
        "parent_mrr": round(mean(detail["parent_mrr"] for detail in details), 6),
        "latency_ms": {
            "average": round(mean(latencies), 3),
            "median": round(statistics.median(latencies), 3) if latencies else 0.0,
            "p95": round(percentile(latencies, 0.95), 3),
            "min": round(min(latencies), 3) if latencies else 0.0,
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
    }


def render_report(output: Dict[str, Any]) -> str:
    metrics = output["metrics"]
    config = output["config"]
    ks = config["ks"]
    header = "| Metric | " + " | ".join(f"@{k}" for k in ks) + " |"
    separator = "|---|" + "|".join("---:" for _ in ks) + "|"
    child_values = " | ".join(f"{metrics['child_recall'][f'@{k}']:.4f}" for k in ks)
    parent_values = " | ".join(f"{metrics['parent_recall'][f'@{k}']:.4f}" for k in ks)
    max_k = max(ks)
    lines = [
        "# Text And Table Retrieval Evaluation Report",
        "",
        f"- Dataset: `{config['dataset']}`",
        f"- Vector store: `{config['store']}`",
        f"- Cases: {metrics['case_count']}",
        f"- Errors: {metrics['error_count']}",
        f"- Degraded retrieval cases: {metrics['degraded_case_count']}",
        f"- Rerank: {config['use_rerank']}",
        f"- Candidate K: {config['candidate_k']}",
        "",
        header,
        separator,
        f"| Child Recall | {child_values} |",
        f"| Parent Recall | {parent_values} |",
        "",
        f"- Child MRR: **{metrics['child_mrr']:.4f}**",
        f"- Parent MRR: **{metrics['parent_mrr']:.4f}**",
        f"- Average latency: **{metrics['latency_ms']['average']:.2f} ms**",
        f"- P95 latency: **{metrics['latency_ms']['p95']:.2f} ms**",
        "",
        "## Failed Or Missed Cases",
        "",
        f"| ID | Query | Child R@{max_k} | Parent R@{max_k} | Error |",
        "|---|---|---:|---:|---|",
    ]
    missed = [
        detail
        for detail in output["details"]
        if detail["child_recall"][f"@{max_k}"] < 1.0
        or detail["parent_recall"][f"@{max_k}"] < 1.0
        or detail.get("error")
    ]
    if not missed:
        lines.append("| - | No misses | - | - | - |")
    for detail in missed:
        error = (detail.get("error") or "").replace("|", "/")
        lines.append(
            f"| {detail['case_id']} | {detail['query']} | "
            f"{detail['child_recall'][f'@{max_k}']:.2f} | "
            f"{detail['parent_recall'][f'@{max_k}']:.2f} | "
            f"{error} |"
        )
    return "\n".join(lines)


def parse_ks(value: str) -> List[int]:
    ks = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not ks or any(k <= 0 for k in ks):
        raise argparse.ArgumentTypeError("ks must contain positive integers, for example: 1,3,5")
    return ks


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate current hybrid text/table retrieval.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--store", default=DEFAULT_STORE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--ks", type=parse_ks, default=parse_ks("1,3,5"))
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=15,
        help="Candidates retrieved from each of FAISS and BM25 before RRF (default: 15).",
    )
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N cases.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = dataset["cases"][: args.limit or None]
    max_k = max(args.ks)
    retriever = MultimodalRetriever(args.store, load_visual=False)
    details = []

    print(f"[eval] cases={len(cases)} candidate_k={args.candidate_k} rerank={args.rerank}")
    for index, case in enumerate(cases, start=1):
        started = time.perf_counter()
        error = None
        results: List[Dict[str, Any]] = []
        try:
            results = retriever.search(
                case["query"],
                top_k=max_k,
                candidate_k=max(args.candidate_k, max_k),
                doc_types={"text", "table"},
                use_rerank=args.rerank,
                max_per_topic=0,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - started) * 1000
        detail = evaluate_case(
            case,
            results,
            latency_ms,
            args.ks,
            search_info=dict(getattr(retriever, "last_search_info", {})),
            error=error,
        )
        details.append(detail)
        print(
            f"[{index:03d}/{len(cases):03d}] {case['case_id']} "
            f"child_R@{max_k}={detail['child_recall'][f'@{max_k}']:.2f} "
            f"parent_R@{max_k}={detail['parent_recall'][f'@{max_k}']:.2f} "
            f"{latency_ms:.1f}ms"
        )

    output = {
        "config": {
            "dataset": str(dataset_path),
            "store": str(Path(args.store)),
            "ks": args.ks,
            "candidate_k": args.candidate_k,
            "use_rerank": args.rerank,
            "doc_types": ["text", "table"],
        },
        "metrics": aggregate(details, args.ks),
        "details": details,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(output), encoding="utf-8")

    print("\n" + render_report({**output, "details": []}).split("## Failed Or Missed Cases")[0])
    print(f"JSON: {output_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
