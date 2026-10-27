#!/usr/bin/env python
# coding: utf-8
"""Generate a 100-case text/table retrieval evaluation set.

This dataset is intentionally separate from the existing multimodal evaluation
set. Text children and table-row children share the same evaluation schema.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from generate_retrieval_eval_dataset import CASES as EXISTING_CASES


CaseSpec = Tuple[str, str, List[str], str]


ADDITIONAL_CASES: List[CaseSpec] = [
    ("车辆上的远程监控系统有什么作用？", "远程监控系统", ["远程监控", "车辆"], "text_fact"),
    ("安装非原厂附件可能带来哪些风险？", "原厂精装附件、选装装备和改装", ["非原厂", "安全"], "safety"),
    ("在车辆上使用无线电设备需要遵守哪些规定？", "无线电设备", ["无线电设备", "切勿"], "safety"),
    ("车辆达到报废条件后应该如何处理？", "车辆报废", ["报废", "回收企业"], "operation"),
    ("动力电池报废后应该交给谁回收？", "隐私告知", ["动力电池", "回收"], "operation"),
    ("前排有哪些可用的储物空间？", "前排储物空间", ["手套箱", "储物"], "text_fact"),
    ("第二排中央扶手和杯托如何使用？", "第二排储物空间", ["中央扶手", "杯托"], "operation"),
    ("如何折叠后排座椅以扩大后备厢空间？", "后备厢储物空间", ["折叠", "后排座椅"], "operation"),
    ("如何启用手套箱密码保护功能？", "后备厢载物", ["手套箱", "密码保护"], "operation"),
    ("如何从车外打开尾门？", "车外打开/关闭尾门", ["开启按键", "自动打开"], "operation"),
    ("如何设置尾门的开启角度？", "设置尾门开启角度", ["尾门", "开启角度"], "operation"),
    ("怎样通过转向指示灯判断车辆是否锁止？", "车辆锁止/解锁状态", ["转向指示灯", "锁止"], "text_fact"),
    ("如何使用Lynk & Co App解锁和闭锁车门？", "使用Lynk & Co App解锁和闭锁", ["App", "解锁"], "operation"),
    ("如何从车内打开后车门？", "车内打开/关闭车门", ["后门内把手", "打开后门"], "operation"),
    ("开门预警系统会在什么车速条件下工作？", "开门预警系统", ["5 km/h", "开门预警"], "text_fact"),
    ("驾驶员位置记忆功能如何保存座椅位置？", "位置记忆功能", ["保存", "位置"], "operation"),
    ("方向盘左侧按键可以执行哪些操作？", "调整方向盘", ["向左按键", "增加音量"], "operation"),
    ("如何使用车门上的控制按钮调节外后视镜？", "调节外后视镜", ["后视镜控制按钮", "调节"], "operation"),
    ("内后视镜自动防眩目功能有什么作用？", "调节内后视镜", ["自动防眩目", "后车灯光"], "text_fact"),
    ("如何启动后挡风玻璃雨刮和洗涤器？", "后雨刮和洗涤器", ["后挡风玻璃", "洗涤器"], "operation"),
    ("自动大灯根据什么条件自动开关近光灯？", "打开/关闭自动大灯", ["环境光照", "近光灯"], "text_fact"),
    ("智能远近光控制系统在什么条件下开启？", "智能远近光控制系统", ["40 km/h", "智能远近光"], "text_fact"),
    ("大灯随动转向功能需要满足哪些启用条件？", "大灯随动转向功能", ["10 km/h", "近光灯"], "text_fact"),
    ("如何开启后雾灯？", "打开/关闭后雾灯", ["后雾灯"], "operation"),
    ("如何操作转向灯使其只闪烁三次？", "打开/关闭转向指示灯", ["闪烁三次", "拨动拨杆"], "operation"),
    ("车辆遇到紧急情况时如何开启危险警告灯？", "使用危险警告灯", ["危险警告灯按键", "紧急情况"], "operation"),
    ("超车时如何用远光灯提醒前方车辆？", "使用超车灯", ["远光灯", "提醒"], "operation"),
    ("如何调节车辆开关和显示屏的背光亮度？", "调节背光亮度", ["调光旋钮", "亮度"], "operation"),
    ("如何在中央显示屏中设置车内氛围灯？", "设置车内氛围灯", ["座舱体验", "氛围灯"], "operation"),
    ("伴我回家灯功能有什么效果？", "使用伴我回家灯", ["伴我回家", "灯"], "text_fact"),
    ("8AT这一缩略语代表什么？", "缩略语和术语", ["8AT", "8 速自动变速器"], "table"),
    ("泊车辅助系统后部两侧的检测范围是多少？", "泊车辅助系统", ["后部两侧", "0.3"], "table"),
    ("车辆常规保养中发动机机油多久更换一次？", "保养项目", ["发动机机油", "15,000公里"], "table"),
    ("车辆长度的参数是多少毫米？", "车辆参数", ["长度", "4549"], "table"),
    ("推荐儿童安全座椅规格表包含哪些座位位置？", "推荐的儿童安全座椅规格", ["前排乘客座", "第二排外侧"], "table"),
]


def contains_any(text: str, keywords: Iterable[str]) -> int:
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword.lower() in lowered)


def topic_matches(metadata: Dict[str, Any], topic: str) -> bool:
    return metadata.get("topic_title") == topic or topic in (metadata.get("topic_titles") or [])


def select_best(
    chunks: Sequence[Dict[str, Any]], keywords: List[str], limit: int = 4
) -> List[Dict[str, Any]]:
    scored = [(contains_any(chunk.get("page_content", ""), keywords), chunk) for chunk in chunks]
    scored = [item for item in scored if item[0] > 0]
    scored.sort(key=lambda item: (-item[0], len(item[1].get("page_content", ""))))
    return [chunk for _, chunk in scored[:limit]]


def bind_case(
    index: int,
    spec: CaseSpec,
    chunks: List[Dict[str, Any]],
    chunks_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    query, topic, keywords, category = spec
    topic_chunks = [chunk for chunk in chunks if topic_matches(chunk.get("metadata") or {}, topic)]
    children = [
        chunk
        for chunk in topic_chunks
        if chunk.get("chunk_type") in {"text_child", "table_row_child"}
    ]
    selected_children = select_best(children, keywords)
    if not selected_children:
        raise ValueError(f"Unable to bind case TT-{index:03d}: {query} / {topic} / {keywords}")

    parent_ids = list(
        dict.fromkeys(
            child.get("metadata", {}).get("parent_id")
            for child in selected_children
            if child.get("metadata", {}).get("parent_id")
        )
    )
    missing_parents = [parent_id for parent_id in parent_ids if parent_id not in chunks_by_id]
    if missing_parents:
        raise ValueError(f"Missing parent chunks for TT-{index:03d}: {missing_parents}")

    pages = {
        page
        for child in selected_children
        for page in (
            child.get("metadata", {}).get("pages")
            or [child.get("metadata", {}).get("page_id")]
        )
        if page is not None
    }
    return {
        "case_id": f"TT-{index:03d}",
        "query": query,
        "category": category,
        "expected_topic": topic,
        "expected_keywords": keywords,
        "expected_child_ids": [child["chunk_id"] for child in selected_children],
        "expected_parent_ids": parent_ids,
        "expected_pages": sorted(pages),
    }


def render_markdown(dataset: Dict[str, Any]) -> str:
    lines = [
        "# Text And Table Retrieval Evaluation Dataset",
        "",
        f"- Cases: {dataset['metadata']['case_count']}",
        f"- Source: `{dataset['metadata']['source_chunks']}`",
        "",
        "| ID | Category | Query | Topic | Child GT | Parent GT | Pages |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for case in dataset["cases"]:
        lines.append(
            f"| {case['case_id']} | {case['category']} | {case['query']} | "
            f"{case['expected_topic']} | {len(case['expected_child_ids'])} | "
            f"{len(case['expected_parent_ids'])} | {case['expected_pages']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a separate 100-case text/table eval set.")
    parser.add_argument("--chunks", default="data/knowledge_bases/train_a/chunks.json")
    parser.add_argument("--out", default="data/evaluation/text_table_retrieval_eval_100.json")
    parser.add_argument("--preview", default="data/evaluation/text_table_retrieval_eval_100.md")
    args = parser.parse_args()

    base_cases = [
        spec
        for spec in EXISTING_CASES
        if spec[3] in {"text_fact", "operation", "safety", "table"}
    ]
    specs: List[CaseSpec] = base_cases + ADDITIONAL_CASES
    if len(specs) != 100:
        raise ValueError(f"Expected exactly 100 cases, got {len(specs)}")

    chunks_path = Path(args.chunks)
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    cases = [
        bind_case(index, spec, chunks, chunks_by_id)
        for index, spec in enumerate(specs, start=1)
    ]
    dataset = {
        "metadata": {
            "version": "1.0",
            "description": "Separate 100-case ground truth for unified text and table retrieval.",
            "source_chunks": str(chunks_path),
            "case_count": len(cases),
            "category_counts": dict(sorted(Counter(case["category"] for case in cases).items())),
            "ground_truth_fields": ["expected_child_ids", "expected_parent_ids"],
            "recommended_metrics": [
                "Child Recall@1/3/5",
                "Parent Recall@1/3/5",
                "MRR",
                "Average retrieval latency",
            ],
            "notes": [
                "Text children and table-row children are evaluated together.",
                "Child Recall compares retrieved child chunk IDs directly.",
                "Parent Recall maps each retrieved child through metadata.parent_id before comparison.",
                "Image retrieval cases are intentionally excluded.",
                "The existing multimodal evaluation dataset is not modified.",
            ],
        },
        "cases": cases,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    preview = Path(args.preview)
    preview.parent.mkdir(parents=True, exist_ok=True)
    preview.write_text(render_markdown(dataset), encoding="utf-8")
    print(json.dumps(dataset["metadata"], ensure_ascii=False, indent=2))
    print(f"JSON: {out}")
    print(f"Preview: {preview}")


if __name__ == "__main__":
    main()
