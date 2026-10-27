#!/usr/bin/env python
"""Audit Agent retrieval cases against text extracted from train_a.pdf."""

import json
import re
from collections import Counter
from pathlib import Path


def normalize(value):
    return re.sub(r"\s+", "", str(value)).lower()


def main():
    root = Path("data/evaluation")
    agent = json.loads((root / "agent_e2e_100_v2.json").read_text(encoding="utf-8"))["cases"]
    sources = {}
    for name in ("text_table_retrieval_eval_100.json", "image_retrieval_eval_100.json"):
        payload = json.loads((root / name).read_text(encoding="utf-8"))
        sources.update({case["case_id"]: case for case in payload["cases"]})
    pages = Path("tmp/pdfs/train_a_audit/train_a.txt").read_text(encoding="utf-8").split("\f")

    rows = []
    for case in agent:
        provenance = case.get("provenance", {})
        source_id = provenance.get("source_case_id", "")
        if source_id not in sources:
            continue
        source = sources[source_id]
        expected_pages = source.get("expected_pages") or []
        keywords = source.get("expected_keywords") or []
        page_indexes = set()
        for logical_page in expected_pages:
            # The PDF contains cover/front-matter offsets. Search a bounded
            # neighborhood and record the actual PDF page that supports it.
            for index in range(max(0, logical_page - 3), min(len(pages), logical_page + 4)):
                page_indexes.add(index)
        neighborhood = "\n".join(pages[index] for index in sorted(page_indexes))
        normalized = normalize(neighborhood)
        matched = [keyword for keyword in keywords if normalize(keyword) in normalized]
        topic = source.get("expected_topic", "")
        topic_matched = bool(topic and normalize(topic) in normalized)
        rows.append({
            "case_id": case["case_id"], "source_case_id": source_id,
            "expected_pages": expected_pages, "keywords": keywords,
            "matched_keywords": matched, "topic": topic, "topic_matched": topic_matched,
            "pdf_supported": bool(matched or topic_matched),
        })

    unsupported = [row for row in rows if not row["pdf_supported"]]
    report = {
        "pdf": "data/train_a.pdf", "pdf_page_count": len(pages),
        "retrieval_case_count": len(rows), "supported_count": len(rows) - len(unsupported),
        "unsupported_count": len(unsupported),
        "support_rate": round((len(rows) - len(unsupported)) / max(1, len(rows)), 4),
        "unsupported_case_ids": [row["case_id"] for row in unsupported],
        "rows": rows,
    }
    output = root / "results/agent_e2e_100_pdf_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False, indent=2))
    print("source groups:", dict(Counter(row["source_case_id"].split("-")[0] for row in rows)))
    print("saved:", output)


if __name__ == "__main__":
    main()
