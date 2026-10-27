#!/usr/bin/env python
# coding: utf-8
"""Generate a curated retrieval evaluation set bound to real chunk IDs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


CASES = [
    # Text / operation / safety questions
    ("如何联系领克客户联络中心？", "联系Lynk & Co领克", ["4006-010101"], "text_fact"),
    ("事件数据记录系统会记录哪些碰撞信息？", "事件数据记录系统", ["行驶速度", "制动状态"], "text_fact"),
    ("每次驾车前应该检查哪些项目？", "安全检查", ["车灯", "轮胎压力"], "text_fact"),
    ("在后备厢装载货物时需要注意什么？", "车辆装载", ["后备厢", "货物"], "safety"),
    ("如何使用遥控钥匙解锁和闭锁车辆？", "使用遥控钥匙解锁和闭锁", ["遥控钥匙", "解锁"], "operation"),
    ("无钥匙进入系统在什么情况下可能无法工作？", "无钥匙进入系统", ["无钥匙进入"], "text_fact"),
    ("如何开启或关闭车辆防盗系统？", "开启/关闭防盗系统", ["防盗系统"], "operation"),
    ("怎样调节驾驶员座椅？", "调节驾驶员座椅", ["驾驶员座椅"], "operation"),
    ("方向盘上有哪些主要控制按键？", "方向盘介绍", ["方向盘"], "text_fact"),
    ("如何正确调节车内后视镜？", "调节内后视镜", ["调节内后视镜"], "operation"),
    ("如何操作前雨刮和洗涤器？", "前雨刮和洗涤器", ["雨刮", "洗涤"], "operation"),
    ("组合仪表包含哪些显示区域？", "组合仪表", ["左侧信息显示区域", "右侧信息显示区域"], "text_fact"),
    ("车辆的指示灯和警告灯分别表示什么？", "指示灯和警告灯", ["指示灯", "警告灯"], "text_fact"),
    ("如何打开和关闭远近光灯？", "打开/关闭远近光灯", ["近光灯", "远光灯"], "operation"),
    ("如何通过组合仪表查看行程信息？", "查看组合仪表信息", ["行程信息", "RESET"], "operation"),
    ("安全带使用时有哪些注意事项？", "安全带", ["安全带"], "safety"),
    ("安全气囊有哪些安全警告？", "安全气囊", ["安全气囊", "警告"], "safety"),
    ("如何打开和关闭车窗？", "打开/关闭车窗", ["车窗"], "operation"),
    ("如何打开和关闭全景天窗？", "打开/关闭全景天窗", ["全景天窗"], "operation"),
    ("儿童锁应该如何使用？", "儿童锁", ["儿童锁"], "operation"),
    ("如何唤醒并使用车载语音助手？", "语音助手", ["语音助手"], "operation"),
    ("车载12V电源使用时需要注意什么？", "车载12 V电源", ["12 V", "电源"], "safety"),
    ("如何给智能设备进行无线充电？", "智能设备充电", ["无线充电"], "operation"),
    ("车辆点火模式有哪些状态？", "点火模式", ["点火模式"], "text_fact"),
    ("如何通过手机App启动车辆？", "通过手机APP启动车辆", ["手机APP", "启动车辆"], "operation"),
    ("车辆应该如何进行换挡操作？", "换挡", ["换挡"], "operation"),
    ("车辆提供哪些驾驶模式？", "驾驶模式", ["驾驶模式"], "text_fact"),
    ("抬头显示系统可以显示哪些信息？", "抬头显示", ["抬头显示"], "text_fact"),
    ("能量回收系统有哪些等级？", "能量回收系统", ["能量回收"], "text_fact"),
    ("如何使用电子驻车制动？", "电子驻车制动（EPB）", ["电子驻车制动"], "operation"),
    ("自动驻车系统如何启用和退出？", "自动驻车系统", ["自动驻车"], "operation"),
    ("给车辆加油时需要注意什么？", "加油", ["加油"], "safety"),
    ("冬季驾驶车辆需要注意哪些事项？", "冬季驾驶", ["冬季驾驶"], "safety"),
    ("自适应巡航系统有什么作用？", "自适应巡航系统", ["自适应巡航"], "text_fact"),
    ("使用高级智能驾驶时驾驶员需要承担什么责任？", "高级智能驾驶", ["安全驾驶", "接管"], "safety"),
    ("交通标志识别系统有哪些使用限制？", "交通标志识别系统", ["交通标志"], "text_fact"),
    ("驾驶员状态监测系统会检测什么？", "驾驶员状态监测系统", ["驾驶员状态"], "text_fact"),
    ("车道辅助系统在什么情况下可能无法正常工作？", "车道辅助系统", ["车道辅助"], "text_fact"),
    ("前向碰撞减缓系统有什么作用？", "前向碰撞减缓系统", ["碰撞"], "text_fact"),
    ("泊车辅助传感器的工作原理是什么？", "泊车辅助传感器", ["泊车辅助传感器"], "text_fact"),
    ("360度全景影像如何使用？", "360°全景影像", ["全景影像"], "operation"),
    ("如何使用全自动泊车功能？", "全自动泊车", ["全自动泊车"], "operation"),
    ("如何使用遥控泊车功能？", "遥控泊车", ["遥控泊车"], "operation"),
    ("如何开启或关闭空调？", "开启/关闭空调", ["空调"], "operation"),
    ("如何使用空调进行除霜和除雾？", "空调除霜/除雾", ["除霜", "除雾"], "operation"),
    ("如何创建和删除蓝牙钥匙？", "创建和删除蓝牙钥匙", ["蓝牙钥匙"], "operation"),
    ("如何使用车载充电设备给车辆充电？", "车载充电设备充电", ["充电"], "operation"),
    ("如何更换遥控钥匙电池？", "更换遥控钥匙电池", ["遥控钥匙", "电池"], "operation"),
    ("胎压标签位于车辆什么位置？", "胎压标签", ["驾驶员侧B柱"], "text_fact"),
    ("应急情况下如何解锁充电枪？", "应急解锁充电枪", ["应急", "充电枪"], "operation"),

    # Table questions
    ("左前阅读灯开关如何调节亮度？", "使用阅读灯", ["左前阅读灯", "长按"], "table"),
    ("前排安全气囊在什么碰撞条件下触发？", "安全气囊", ["前排安全气囊", "正面碰撞"], "table"),
    ("泊车辅助系统后部中央的检测范围是多少？", "泊车辅助系统", ["后部中央", "1.5"], "table"),
    ("发动机机油和机油滤芯的保养周期是多少？", "保养项目", ["发动机机油", "15,000"], "table"),
    ("车辆的长度、宽度和高度分别是多少？", "车辆参数", ["4549", "1860", "1689"], "table"),
    ("车辆整备质量和最大允许总重量是多少？", "车辆参数", ["1890", "2330"], "table"),
    ("发动机最大净功率和最大扭矩是多少？", "车辆参数", ["108/5500", "225"], "table"),
    ("车辆的最大设计车速是多少？", "车辆参数", ["200"], "table"),
    ("车辆油箱容积和燃油等级是多少？", "车辆参数", ["42 L", "92"], "table"),
    ("235/50 R19轮胎推荐胎压是多少？", "车辆参数", ["235/50 R19", "230", "280"], "table"),
    ("制动液的规格是什么？", "车辆参数", ["DOT 4"], "table"),
    ("ACC代表什么系统？", "缩略语和术语", ["ACC", "自适应巡航"], "table"),
    ("EPB代表什么？", "缩略语和术语", ["EPB", "电子驻车制动"], "table"),
    ("VIN代表什么？", "缩略语和术语", ["VIN", "车辆识别代码"], "table"),
    ("kPa是什么单位？", "缩略语和术语", ["kPa", "千帕"], "table"),

    # Image-text / visual questions
    ("找到抬头显示系统的示意图片。", "抬头显示", ["抬头显示"], "image_joint"),
    ("找到遥控钥匙按键说明图片。", "使用遥控钥匙解锁和闭锁", ["遥控钥匙"], "image_joint"),
    ("找到前雨刮控制杆的示意图片。", "前雨刮和洗涤器", ["雨刮"], "image_joint"),
    ("找到全景天窗操作界面的图片。", "打开/关闭全景天窗", ["全景天窗"], "image_joint"),
    ("找到空调出风方向调节的示意图片。", "调节空调出风方向", ["出风"], "image_joint"),
    ("找到蓝牙钥匙创建流程的界面图片。", "创建和删除蓝牙钥匙", ["蓝牙钥匙"], "image_joint"),
    ("找到应急解锁充电枪的操作图片。", "应急解锁充电枪", ["充电枪"], "image_joint"),
    ("找到胎压标签的图片。", "胎压标签", ["胎压"], "image_visual"),
    ("找到车辆识别代码位置的示意图片。", "车辆标识", ["车辆识别代码", "VIN"], "image_visual"),
    ("找到指示灯和警告灯图标图片。", "指示灯和警告灯", ["指示灯", "警告灯"], "image_visual"),

    # Negative / no-answer cases
    ("这辆车支持在月球表面自动驾驶吗？", "", [], "no_answer"),
    ("车辆能否变形成机器人？", "", [], "no_answer"),
    ("如何连接家里的智能冰箱？", "", [], "no_answer"),
    ("这辆车的2028年二手市场价格是多少？", "", [], "no_answer"),
    ("请告诉我明天所在城市的天气。", "", [], "no_answer"),
]


def contains_any(text: str, keywords: Iterable[str]) -> int:
    return sum(1 for keyword in keywords if keyword.lower() in text.lower())


def topic_matches(metadata: Dict[str, Any], topic: str) -> bool:
    return metadata.get("topic_title") == topic or topic in (metadata.get("topic_titles") or [])


def select_best(chunks: List[Dict[str, Any]], keywords: List[str], limit: int = 4) -> List[Dict[str, Any]]:
    scored = [(contains_any(chunk.get("page_content", ""), keywords), chunk) for chunk in chunks]
    if keywords:
        scored = [item for item in scored if item[0] > 0]
    scored.sort(key=lambda item: (-item[0], len(item[1].get("page_content", ""))))
    return [chunk for _, chunk in scored[:limit]]


def bind_case(index: int, spec, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    query, topic, keywords, category = spec
    case = {
        "case_id": f"MM-{index:03d}",
        "query": query,
        "category": category,
        "expected_topic": topic or None,
        "expected_keywords": keywords,
        "expected_parent_ids": [],
        "expected_image_ids": [],
        "expected_pages": [],
        "is_answerable": category != "no_answer",
    }
    if category == "no_answer":
        return case

    topic_chunks = [chunk for chunk in chunks if topic_matches(chunk.get("metadata") or {}, topic)]
    if category in {"text_fact", "operation", "safety"}:
        parents = select_best(
            [chunk for chunk in topic_chunks if chunk.get("chunk_type") == "text_parent"],
            keywords,
        )
        case["expected_parent_ids"] = [chunk["chunk_id"] for chunk in parents]
        evidence = parents
    elif category == "table":
        parents = select_best(
            [chunk for chunk in topic_chunks if chunk.get("chunk_type") == "table_parent"],
            keywords,
        )
        case["expected_parent_ids"] = [chunk["chunk_id"] for chunk in parents]
        evidence = parents
    else:
        images = select_best(
            [chunk for chunk in topic_chunks if chunk.get("doc_type") == "image"],
            keywords,
            limit=6,
        )
        if not images:
            images = [chunk for chunk in topic_chunks if chunk.get("doc_type") == "image"][:6]
        case["expected_image_ids"] = [chunk["chunk_id"] for chunk in images]
        evidence = images

    pages = {
        page
        for chunk in evidence
        for page in (chunk.get("metadata", {}).get("pages") or [chunk.get("metadata", {}).get("page_id")])
        if page is not None
    }
    case["expected_pages"] = sorted(pages)
    expected_ids = (
        case["expected_parent_ids"]
        + case["expected_image_ids"]
    )
    if not expected_ids:
        raise ValueError(f"Unable to bind case {case['case_id']}: {query} / {topic} / {keywords}")
    return case


def render_markdown(dataset: Dict[str, Any]) -> str:
    lines = [
        "# Multimodal Retrieval Evaluation Dataset",
        "",
        f"- Cases: {dataset['metadata']['case_count']}",
        f"- Source: `{dataset['metadata']['source_chunks']}`",
        "",
        "| ID | Category | Query | Topic | Pages |",
        "|---|---|---|---|---|",
    ]
    for case in dataset["cases"]:
        lines.append(
            f"| {case['case_id']} | {case['category']} | {case['query']} | "
            f"{case['expected_topic'] or '-'} | {case['expected_pages']} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate the curated multimodal retrieval eval set.")
    parser.add_argument("--chunks", default="data/knowledge_bases/train_a/chunks.json")
    parser.add_argument("--out", default="data/evaluation/multimodal_retrieval_eval.json")
    parser.add_argument("--preview", default="data/evaluation/multimodal_retrieval_eval.md")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    cases = [bind_case(index, spec, chunks) for index, spec in enumerate(CASES, start=1)]
    category_counts = dict(sorted(Counter(case["category"] for case in cases).items()))
    dataset = {
        "metadata": {
            "version": "1.0",
            "description": "Curated retrieval ground truth for the train_a automotive manual.",
            "source_chunks": str(chunks_path),
            "case_count": len(cases),
            "category_counts": category_counts,
            "notes": [
                "Text/table retrieval results must be mapped from child chunks to parent_id before scoring.",
                "Text and table cases store only expected parent IDs.",
                "Image cases evaluate image-text and visual retrieval.",
                "No-answer cases should not match any local evidence.",
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
