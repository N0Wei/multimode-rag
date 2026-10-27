"""Evaluate a *real* active multi-manual index against manually checked source IDs.

The small bundled set is a smoke benchmark, not a statistically representative
quality claim. Queries use one selected model and gold IDs from the MinerU
corpus. No answer-quality judge or invented citation score is implied.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from multi_manual_store import MultiManualRetriever


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Evaluation cases must have distinct IDs")
    for case in cases:
        if case["lane"] not in {"text", "image"} or not case["query"] or not case["model_name"]:
            raise ValueError(f"Invalid evaluation case: {case.get('id')}")
        if not case.get("gold_ids"):
            raise ValueError(f"Case has no verified gold IDs: {case['id']}")
    return cases


def evaluate(retriever: MultiManualRetriever, cases: list[dict[str, Any]], top_k: int = 5) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        lane = case["lane"]
        known = retriever.children if lane == "text" else retriever.images
        missing = set(case["gold_ids"]) - known.keys()
        if missing:
            raise ValueError(f"Gold IDs missing from active index for {case['id']}: {sorted(missing)}")
        if any(known[record_id]["metadata"]["model_name"] != case["model_name"] for record_id in case["gold_ids"]):
            raise ValueError(f"Gold label crosses model boundary in {case['id']}")
        started = time.perf_counter()
        hits = (retriever.search_text(case["query"], case["model_name"], top_k=top_k)
                if lane == "text" else retriever.search_images(case["query"], case["model_name"], top_k=top_k))
        elapsed_ms = (time.perf_counter() - started) * 1000
        ids = [hit["chunk_id"] if lane == "text" else hit["image_id"] for hit in hits]
        gold = set(case["gold_ids"])
        first_rank = next((index for index, record_id in enumerate(ids, 1) if record_id in gold), None)
        leakage = [record_id for record_id, hit in zip(ids, hits)
                   if hit["metadata"]["model_name"] != case["model_name"]]
        # Parent recall is distinct from exact child recall: another child in
        # the correct directory section counts only for the former.
        parent_ids = {known[record_id]["metadata"]["parent_id"] for record_id in gold} if lane == "text" else set()
        parent_hit = any(hit["parent_id"] in parent_ids for hit in hits) if lane == "text" else None
        rows.append({
            "id": case["id"], "lane": lane, "model_name": case["model_name"],
            "hit_ids": ids, "gold_ids": case["gold_ids"], "first_gold_rank": first_rank,
            "recall_at_k": len(gold.intersection(ids)) / len(gold),
            "parent_recall_at_k": bool(parent_hit) if lane == "text" else None,
            "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
            "cross_model_leakage": leakage, "latency_ms": round(elapsed_ms, 2),
        })
    summary: dict[str, Any] = {"top_k": top_k, "case_count": len(rows), "lanes": {}}
    for lane in ("text", "image"):
        subset = [row for row in rows if row["lane"] == lane]
        if not subset:
            continue
        summary["lanes"][lane] = {
            "count": len(subset),
            "recall_at_k": statistics.mean(row["recall_at_k"] for row in subset),
            "mrr_at_k": statistics.mean(row["reciprocal_rank"] for row in subset),
            "latency_p50_ms": statistics.median(row["latency_ms"] for row in subset),
            "latency_p95_ms": sorted(row["latency_ms"] for row in subset)[max(0, int(len(subset) * 0.95 + 0.9999) - 1)],
            "cross_model_leakage_count": sum(len(row["cross_model_leakage"]) for row in subset),
        }
        if lane == "text":
            summary["lanes"][lane]["parent_recall_at_k"] = statistics.mean(
                row["parent_recall_at_k"] for row in subset
            )
    return {"summary": summary, "cases": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=Path("data/multi_manual_store"))
    parser.add_argument("--cases", type=Path, default=Path("data/eval/multi_manual_smoke.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    retriever = MultiManualRetriever(args.store_root)
    try:
        report = evaluate(retriever, load_cases(args.cases), args.top_k)
        report["index_batch_id"] = retriever.manifest["batch_id"]
        report["benchmark_kind"] = "small manually checked retrieval smoke set; no answer correctness judge"
    finally:
        retriever.close()
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
