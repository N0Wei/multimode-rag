#!/usr/bin/env python
"""Run and score the real LangGraph Agent on a versioned end-to-end dataset."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

# Evaluation should not block on external trace export. The project's local
# execution_trace remains available in every result.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from agent.runtime import AgentRuntime, new_agent_state


def _unique_tools(trace: Iterable[Dict[str, Any]]) -> List[str]:
    tools = []
    for event in trace:
        if event.get("event_type") != "tool":
            continue
        tool = event.get("tool") or event.get("name")
        if tool and tool not in tools and not event.get("deduplicated"):
            tools.append(tool)
    return tools


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    normalized = text.lower()
    return any(str(keyword).lower() in normalized for keyword in keywords)


def score_case(case: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    answer = str(result.get("final_answer") or "")
    tools = _unique_tools(result.get("execution_trace") or [])
    required = set(case.get("required_tools") or [])
    forbidden = set(case.get("forbidden_tools") or [])
    planner = (result.get("metadata") or {}).get("planner") or {}
    checks = {
        "intent": result.get("intent") in case.get("expected_intents", [result.get("intent")]),
        "required_tools": required.issubset(set(tools)),
        "forbidden_tools": not bool(forbidden & set(tools)),
        "source": result.get("answer_source") in case.get("expected_sources", [result.get("answer_source")]),
        "need_human": bool(result.get("need_human", False)) == bool(case.get("expected_need_human", False)),
        "planner_output_valid": bool(planner.get("output_valid", True)),
    }
    if "expected_complex" in case:
        checks["complexity"] = bool(result.get("is_complex_query")) == bool(case["expected_complex"])
    if case.get("answer_keywords_any"):
        checks["answer_keyword"] = _contains_any(answer, case["answer_keywords_any"])
    if case.get("forbidden_answer_keywords"):
        checks["answer_safety"] = not _contains_any(answer, case["forbidden_answer_keywords"])
    if "min_citation_coverage" in case:
        checks["citation_coverage"] = float(result.get("citation_coverage", 0.0) or 0.0) >= float(
            case["min_citation_coverage"]
        )
    passed = sum(bool(value) for value in checks.values())
    return {
        "checks": checks,
        "passed_checks": passed,
        "check_count": len(checks),
        "score": round(passed / len(checks), 4) if checks else 0.0,
        "passed": passed == len(checks),
        "actual": {
            "intent": result.get("intent"),
            "tools": tools,
            "answer_source": result.get("answer_source"),
            "is_complex_query": bool(result.get("is_complex_query")),
            "need_human": bool(result.get("need_human", False)),
            "citation_coverage": float(result.get("citation_coverage", 0.0) or 0.0),
            "retrieval_retry_count": int(result.get("retrieval_retry_count", 0) or 0),
            "reflection_count": int(result.get("reflection_count", 0) or 0),
            "answer": answer,
            "planner": planner,
            "metrics": result.get("metrics", {}),
        },
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    completed = [row for row in rows if not row.get("error")]
    latencies = [
        float(row["evaluation"]["actual"]["metrics"].get("total_duration_ms", 0) or 0)
        for row in completed
    ]
    check_names = sorted({
        name for row in completed for name in row["evaluation"]["checks"]
    })
    check_rates = {
        name: round(
            sum(bool(row["evaluation"]["checks"].get(name)) for row in completed if name in row["evaluation"]["checks"])
            / max(1, sum(name in row["evaluation"]["checks"] for row in completed)),
            4,
        )
        for name in check_names
    }
    planner_errors = {}
    for row in completed:
        planner = row["evaluation"]["actual"].get("planner") or {}
        error_type = planner.get("error_type")
        if error_type:
            planner_errors[error_type] = planner_errors.get(error_type, 0) + 1
    return {
        "case_count": len(rows),
        "completed_count": len(completed),
        "error_count": len(rows) - len(completed),
        "case_pass_rate": round(sum(row["evaluation"]["passed"] for row in completed) / max(1, len(completed)), 4),
        "average_score": round(statistics.mean(row["evaluation"]["score"] for row in completed), 4) if completed else 0.0,
        "check_pass_rates": check_rates,
        "latency_ms": {
            "average": round(statistics.mean(latencies), 2) if latencies else 0.0,
            "median": round(statistics.median(latencies), 2) if latencies else 0.0,
            "max": round(max(latencies), 2) if latencies else 0.0,
        },
        "total_llm_calls": sum(
            int(row["evaluation"]["actual"]["metrics"].get("llm_calls", 0) or 0) for row in completed
        ),
        "total_tool_calls": sum(
            int(row["evaluation"]["actual"]["metrics"].get("tool_calls", 0) or 0) for row in completed
        ),
        "planner_output_valid_rate": round(
            sum(bool(row["evaluation"]["checks"].get("planner_output_valid")) for row in completed)
            / max(1, len(completed)), 4
        ),
        "planner_error_counts": planner_errors,
        "failed_case_ids": [row["case_id"] for row in completed if not row["evaluation"]["passed"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/evaluation/agent_e2e_v1.json")
    parser.add_argument("--output", default="data/evaluation/results/agent_e2e_baseline.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--resume", action="store_true", help="continue from an existing output file")
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    cases = dataset["cases"]
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in wanted]
    if args.limit:
        cases = cases[: args.limit]

    output = Path(args.output)
    existing_rows = []
    if args.resume and output.exists():
        existing_rows = json.loads(output.read_text(encoding="utf-8")).get("results", [])
    completed_ids = {row.get("case_id") for row in existing_rows if row.get("case_id")}
    cases = [case for case in cases if case["case_id"] not in completed_ids]

    runtime = AgentRuntime()
    runtime.initialize()
    rows = list(existing_rows)
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['case_id']} {case['question']}", flush=True)
        started = time.perf_counter()
        try:
            result = runtime.invoke(new_agent_state(case["question"], metadata={"eval_case_id": case["case_id"]}))
            evaluation = score_case(case, result)
            rows.append({"case_id": case["case_id"], "category": case["category"], "question": case["question"], "evaluation": evaluation})
            print(f"  score={evaluation['score']:.2f} passed={evaluation['passed']} actual={evaluation['actual']['tools']}", flush=True)
        except Exception as exc:
            rows.append({
                "case_id": case["case_id"], "category": case["category"], "question": case["question"],
                "error": str(exc), "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            })
            print(f"  ERROR: {exc}", flush=True)

        # Persist every completed case so a long evaluation can resume after
        # process, network, or user interruption without repeating API calls.
        partial = {
            "dataset_version": dataset["version"],
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "knowledge_base": runtime.knowledge_base,
            "summary": summarize(rows),
            "results": rows,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = {
        "dataset_version": dataset["version"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "knowledge_base": runtime.knowledge_base,
        "summary": summarize(rows),
        "results": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
