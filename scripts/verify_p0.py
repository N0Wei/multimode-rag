#!/usr/bin/env python
"""Run the deterministic P0 acceptance checks without external model APIs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_TESTS = [
    "test_dynamic_planner.py",
    "test_structured_tool_executor.py",
    "test_retrieval_grader.py",
    "test_multi_hop_graph.py",
    "test_answer_grounding.py",
    "test_execution_budget_cache.py",
    "test_observability.py",
    "test_api_service.py",
]
RESULT_FILES = [
    ROOT / "data/evaluation/results/text_table_retrieval_metrics.json",
    ROOT / "data/evaluation/results/image_retrieval_comparison.json",
]


def validate_evaluation_artifacts() -> None:
    for path in RESULT_FILES:
        if not path.is_file():
            raise SystemExit(f"missing evaluation artifact: {path.relative_to(ROOT)}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload.get("metrics"), dict):
            raise SystemExit(f"invalid metrics object: {path.relative_to(ROOT)}")
        print(f"[ok] evaluation artifact: {path.relative_to(ROOT)}")


def run_core_tests() -> None:
    command = [sys.executable, "-m", "pytest", "-q", *CORE_TESTS]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    validate_evaluation_artifacts()
    run_core_tests()
    print("[ok] P0 offline acceptance checks passed")
