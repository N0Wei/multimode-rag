#!/usr/bin/env python
# coding: utf-8
"""Generate a separate 100-case image retrieval evaluation dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


CaseSpec = Tuple[str, str, List[str], str]


# Visual-object queries describe the visible object/interface/layout.
# Semantic-context queries describe an operation or meaning normally found in
# the image's nearby text. Both modes share the same image ground-truth format.
CASES: List[CaseSpec] = [
    # Visual object / layout queries (50)
    ("查找方向盘左右两侧按键布局图", "方向盘介绍", ["方向盘左侧键盘", "方向盘右侧键盘"], "visual_object"),
    ("查找驾驶员座椅侧面调节按钮示意图", "调节驾驶员座椅", ["腰部支撑", "座垫角度"], "visual_object"),
    ("查找副驾驶座椅调节按钮示意图", "调节副驾驶座椅", ["座椅前后位置", "座椅靠背角度"], "visual_object"),
    ("查找方向盘锁定杆调节示意图", "调整方向盘", ["锁定杆", "方向盘"], "visual_object"),
    ("查找车门外后视镜控制按钮示意图", "调节外后视镜", ["控制杆", "L和R按键"], "visual_object"),
    ("查找遥控钥匙按钮示意图", "使用遥控钥匙解锁和闭锁", ["解锁按钮", "尾门按钮"], "visual_object"),
    ("查找车门中控锁按键示意图", "车内解锁和闭锁", ["中控锁按键", "锁止"], "visual_object"),
    ("查找手套箱打开方式示意图", "前排储物空间", ["手套箱自动打开", "解锁按钮"], "visual_object"),
    ("查找后排中央扶手打开示意图", "第二排储物空间", ["扶手向前拉动", "向下打开"], "visual_object"),
    ("查找尾门开启按键示意图", "车外打开/关闭尾门", ["开启", "尾门"], "visual_object"),
    ("查找全景天窗控制按钮示意图", "打开/关闭全景天窗", ["全景天窗", "控制按钮"], "visual_object"),
    ("查找车窗升降按钮示意图", "打开/关闭车窗", ["车窗", "按钮"], "visual_object"),
    ("查找前雨刮拨杆和滚轮示意图", "前雨刮和洗涤器", ["雨刮拨杆", "滚轮"], "visual_object"),
    ("查找后雨刮拨杆操作示意图", "后雨刮和洗涤器", ["拨杆", "后挡风玻璃"], "visual_object"),
    ("查找灯光调节滚轮示意图", "打开/关闭远近光灯", ["滚轮", "车灯"], "visual_object"),
    ("查找转向灯拨杆操作示意图", "打开/关闭转向指示灯", ["拨动拨杆", "闪烁三次"], "visual_object"),
    ("查找危险警告灯三角形按键图片", "使用危险警告灯", ["危险警告灯按键"], "visual_object"),
    ("查找后雾灯控制界面图片", "打开/关闭后雾灯", ["后雾灯"], "visual_object"),
    ("查找组合仪表信息区域布局图", "查看组合仪表信息", ["组合仪表设置项", "二级菜单"], "visual_object"),
    ("查找抬头显示HUD调节界面图片", "抬头显示", ["HUD", "高度", "亮度"], "visual_object"),
    ("查找换挡杆挡位示意图", "换挡", ["倒挡", "空挡", "前进挡"], "visual_object"),
    ("查找电子驻车制动EPB控制界面", "电子驻车制动（EPB）", ["EPB AUTO"], "visual_object"),
    ("查找自动驻车Auto Hold指示图标", "自动驻车系统", ["Auto Hold", "指示灯"], "visual_object"),
    ("查找陡坡缓降系统控制界面", "陡坡缓降系统", ["陡坡缓降", "HDC"], "visual_object"),
    ("查找车身稳定控制系统开关界面", "车身稳定控制系统", ["车身稳定控制系统"], "visual_object"),
    ("查找驾驶模式动力分配显示图", "驾驶模式", ["电机图标", "发动机图标"], "visual_object"),
    ("查找能量回收等级显示图", "能量回收系统", ["能量回收图标", "能量回收等级"], "visual_object"),
    ("查找高级智能驾驶仪表显示图", "高级智能驾驶", ["目标车辆", "跟车距离"], "visual_object"),
    ("查找变道辅助监控区域示意图", "变道辅助系统", ["BSD监控区域", "CVW/LCW"], "visual_object"),
    ("查找驾驶辅助摄像头和雷达位置图", "超速报警", ["风挡摄像头", "前雷达"], "visual_object"),
    ("查找泊车辅助传感器位置示意图", "泊车辅助传感器", [], "visual_object"),
    ("查找360度全景影像按钮图标", "360°全景影像", ["360°全景"], "visual_object"),
    ("查找全自动泊车APA界面图片", "全自动泊车", ["自动泊车", "APA"], "visual_object"),
    ("查找遥控泊车手机界面图片", "遥控泊车", ["Lynk & Co App", "泊车完成"], "visual_object"),
    ("查找中央显示屏主界面布局图", "设置中央显示屏显示状态", ["快捷卡片", "功能显示区域"], "visual_object"),
    ("查找应用程序列表界面图片", "应用程序", ["应用程序列表", "历史记录区域"], "visual_object"),
    ("查找车辆功能快捷开关界面", "车辆功能界面", ["功能栏", "快捷开关"], "visual_object"),
    ("查找空调温度调节旋钮图片", "调节空调温度", ["温度调节旋钮"], "visual_object"),
    ("查找空调风量调节界面图片", "调节空调风量", ["风量", "9个速度等级"], "visual_object"),
    ("查找前除霜除雾按钮图片", "空调除霜/除雾", ["最大除霜", "按钮"], "visual_object"),
    ("查找座椅加热控制界面图片", "前排座椅加热", ["座椅加热强度"], "visual_object"),
    ("查找座椅通风控制界面图片", "前排座椅通风", ["座椅通风强度"], "visual_object"),
    ("查找方向盘加热控制界面图片", "方向盘加热", ["方向盘加热强度"], "visual_object"),
    ("查找ISOFIX儿童座椅固定装置图片", "儿童座椅固定装置", ["ISOFIX固定装置"], "visual_object"),
    ("查找胎压标签在B柱上的图片", "胎压标签", ["胎压标签", "驾驶员侧B柱"], "visual_object"),
    ("查找12V电源插座位置图片", "车载12 V电源", ["12 V电源插座"], "visual_object"),
    ("查找前机舱盖开启拉手图片", "打开前机舱盖", ["开启拉手"], "visual_object"),
    ("查找保险丝盒位置示意图", "更换保险丝", ["保险丝位置", "保险丝盒"], "visual_object"),
    ("查找车辆被平板拖车牵引的示意图", "牵引车辆", ["平板拖车", "推荐方式"], "visual_object"),
    ("查找SOS紧急救援按键图片", "紧急救援", ["SOS", "按键"], "visual_object"),

    # Semantic context / operation queries (50)
    ("查找展示如何更换遥控钥匙电池的图片", "更换遥控钥匙电池", ["安装遥控钥匙", "电池"], "semantic_context"),
    ("查找展示如何应急解锁充电枪的操作图", "应急解锁充电枪", ["应急解锁拉线", "充电枪解锁按钮"], "semantic_context"),
    ("查找展示如何从车内应急打开尾门的图片", "应急打开尾门", ["机械方式", "后备厢"], "semantic_context"),
    ("查找展示如何设置尾门开启角度的界面", "设置尾门开启角度", ["电动尾门开启角度"], "semantic_context"),
    ("查找展示靠近车辆自动开启尾门的图片", "车外打开/关闭尾门", ["靠近自动开启", "遥控钥匙"], "semantic_context"),
    ("查找展示如何折叠后排座椅的图片", "折叠后排座椅", [], "semantic_context"),
    ("查找展示如何正确使用安全带的图片", "使用安全带", ["安全带高度", "调节"], "semantic_context"),
    ("查找展示孕妇如何使用安全带的图片", "安全气囊", ["孕妇", "安全带"], "semantic_context"),
    ("查找展示儿童座椅顶部系带安装方式的图片", "推荐的儿童安全座椅规格", ["顶部系带", "固定装置"], "semantic_context"),
    ("查找展示头枕保护作用的图片", "头枕", ["头枕", "保护"], "semantic_context"),
    ("查找展示自动大灯开启条件的图片", "打开/关闭自动大灯", ["自动开启", "环境光照"], "semantic_context"),
    ("查找展示智能远近光控制系统的图片", "智能远近光控制系统", ["智能远近光", "40 km/h"], "semantic_context"),
    ("查找展示伴我回家灯设置方式的图片", "使用伴我回家灯", ["伴我回家", "外部照明"], "semantic_context"),
    ("查找展示车内氛围灯提醒设置的图片", "设置车内氛围灯", ["来电提醒", "低燃油提醒"], "semantic_context"),
    ("查找展示如何调节背光亮度的图片", "调节背光亮度", ["背光联动"], "semantic_context"),
    ("查找展示如何开启低速行驶提示音的图片", "低速行驶提示音", ["低速行驶提示音"], "semantic_context"),
    ("查找展示最高限速辅助系统操作的图片", "最高限速辅助系统", ["最高限速辅助系统"], "semantic_context"),
    ("查找展示自适应巡航系统操作的图片", "自适应巡航系统", ["自适应巡航", "ACC"], "semantic_context"),
    ("查找展示交通标志识别系统设置的图片", "交通标志识别系统", ["交通标志识别", "限速"], "semantic_context"),
    ("查找展示驾驶员状态监测系统的图片", "驾驶员状态监测系统", ["驾驶员状态", "摄像头"], "semantic_context"),
    ("查找展示车道辅助系统纠偏功能的图片", "车道辅助系统", ["偏离当前行车道", "转向系统"], "semantic_context"),
    ("查找展示后方横向来车预警的图片", "后方横向来车预警系统", ["倒挡", "后方横向来车"], "semantic_context"),
    ("查找展示后方碰撞预警信号的图片", "后方碰撞预警系统", ["声音警告信号", "可视警告信号"], "semantic_context"),
    ("查找展示泊车紧急制动开关的图片", "泊车紧急制动", ["PEB"], "semantic_context"),
    ("查找展示泊车雷达开启方式的图片", "泊车辅助系统", ["泊车辅助系统", "倒挡"], "semantic_context"),
    ("查找展示生命体检测系统作用的图片", "生命体检测系统", ["后排", "雷达传感器"], "semantic_context"),
    ("查找展示外后视镜倒车自动调节的图片", "外后视镜倒车自动调节", [], "semantic_context"),
    ("查找展示如何使用加油漏斗的图片", "加油", ["加油漏斗", "燃油加注口"], "semantic_context"),
    ("查找展示随车设备快速充电设置的图片", "随车设备快速充电", ["充电外部灯效"], "semantic_context"),
    ("查找展示车辆对外供电设置的图片", "车辆供电", ["驻车发电", "启用"], "semantic_context"),
    ("查找展示车载充电故障处理的图片", "车载充电设备充电", ["充电故障", "红色指示灯"], "semantic_context"),
    ("查找展示混合动力电池位置的图片", "混合动力电池", [], "semantic_context"),
    ("查找展示低压蓄电池维护注意事项的图片", "保养和维护低压蓄电池", ["低压蓄电池", "回收"], "semantic_context"),
    ("查找展示车辆电量较低时跨接启动的图片", "电池电量较低", ["跨接电缆", "救援车辆"], "semantic_context"),
    ("查找展示发动机机油检查方法的图片", "检查发动机机油", ["机油", "MAX"], "semantic_context"),
    ("查找展示制动液液位检查方法的图片", "检查制动液", ["制动液", "MIN"], "semantic_context"),
    ("查找展示冷却液液位检查方法的图片", "检查冷却液", ["冷却液", "MIN"], "semantic_context"),
    ("查找展示洗涤液加注位置的图片", "添加洗涤液", ["洗涤液", "加注口"], "semantic_context"),
    ("查找展示如何更换雨刮片的图片", "更换雨刮片", ["更换雨刮片"], "semantic_context"),
    ("查找展示补胎套装和三角警示牌使用的图片", "补胎套装", ["三角警示牌", "支架"], "semantic_context"),
    ("查找展示轮胎前后换位方式的图片", "保养轮胎", ["前后互换"], "semantic_context"),
    ("查找展示车辆故障检测入口的图片", "处理车辆故障", ["车辆检测", "故障部位"], "semantic_context"),
    ("查找展示车辆OTA更新入口的图片", "车辆远程升级（OTA)", ["更新", "OTA"], "semantic_context"),
    ("查找展示蓝牙连接设置界面的图片", "连接设置", ["蓝牙", "Wi-Fi"], "semantic_context"),
    ("查找展示Wi-Fi开关设置界面的图片", "系统设置", ["Wi-Fi功能"], "semantic_context"),
    ("查找展示创建蓝牙钥匙流程的图片", "创建和删除蓝牙钥匙", ["蓝牙钥匙"], "semantic_context"),
    ("查找展示行车记录仪设置界面的图片", "查看行车记录仪视频", ["行车记录仪分辨率", "循环录像"], "semantic_context"),
    ("查找展示相机拍照模式操作的图片", "相机", ["延时拍照", "拍摄照片"], "semantic_context"),
    ("查找展示香氛浓度调节界面的图片", "香氛系统", ["香氛浓度"], "semantic_context"),
    ("查找展示空气质量智能除味功能的图片", "空气质量管理系统", ["智能除味", "倒计时"], "semantic_context"),
]


def contains_any(text: str, keywords: Iterable[str]) -> int:
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword.lower() in lowered)


def bind_case(index: int, spec: CaseSpec, images: List[Dict[str, Any]]) -> Dict[str, Any]:
    query, topic, keywords, query_type = spec
    topic_images = [
        image
        for image in images
        # Use only the canonical topic. Historical topic_titles can contain
        # unrelated occurrences of a deduplicated generic icon.
        if (image.get("metadata") or {}).get("topic_title") == topic
    ]
    if not topic_images:
        raise ValueError(f"Unable to find image topic for IMG-{index:03d}: {topic}")

    scored = [
        (
            contains_any(
                " ".join(
                    [
                        image.get("page_content", ""),
                        str((image.get("metadata") or {}).get("nearby_text", "")),
                    ]
                ),
                keywords,
            ),
            image,
        )
        for image in topic_images
    ]
    if keywords:
        matched = [item for item in scored if item[0] > 0]
        if not matched:
            raise ValueError(f"Unable to bind IMG-{index:03d}: {query} / {topic} / {keywords}")
        scored = matched
    scored.sort(
        key=lambda item: (
            -item[0],
            -len(str((item[1].get("metadata") or {}).get("nearby_text", ""))),
            item[1]["chunk_id"],
        )
    )
    # Multiple images can be valid for one operation, but limiting the ground
    # truth avoids treating every vaguely related image in a large topic as correct.
    selected = [image for _, image in scored[:3]]
    pages = sorted(
        {
            page
            for image in selected
            for page in (
                (image.get("metadata") or {}).get("pages")
                or [(image.get("metadata") or {}).get("page_id")]
            )
            if page is not None
        }
    )
    return {
        "case_id": f"IMG-{index:03d}",
        "query": query,
        "query_type": query_type,
        "expected_topic": topic,
        "expected_keywords": keywords,
        "expected_image_ids": [image["chunk_id"] for image in selected],
        "expected_pages": pages,
    }


def render_markdown(dataset: Dict[str, Any]) -> str:
    lines = [
        "# Image Retrieval Evaluation Dataset",
        "",
        f"- Cases: {dataset['metadata']['case_count']}",
        f"- Source: `{dataset['metadata']['source_chunks']}`",
        "",
        "| ID | Query Type | Query | Topic | Image GT | Pages |",
        "|---|---|---|---|---:|---|",
    ]
    for case in dataset["cases"]:
        lines.append(
            f"| {case['case_id']} | {case['query_type']} | {case['query']} | "
            f"{case['expected_topic']} | {len(case['expected_image_ids'])} | "
            f"{case['expected_pages']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the 100-case image retrieval eval set.")
    parser.add_argument("--chunks", default="data/knowledge_bases/train_a/chunks.json")
    parser.add_argument("--out", default="data/evaluation/image_retrieval_eval_100.json")
    parser.add_argument("--preview", default="data/evaluation/image_retrieval_eval_100.md")
    args = parser.parse_args()
    if len(CASES) != 100:
        raise ValueError(f"Expected exactly 100 cases, got {len(CASES)}")

    chunks_path = Path(args.chunks)
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    images = [chunk for chunk in chunks if chunk.get("doc_type") == "image"]
    cases = [bind_case(index, spec, images) for index, spec in enumerate(CASES, start=1)]
    dataset = {
        "metadata": {
            "version": "1.0",
            "description": "Shared image ground truth for visual-only and visual-plus-image-text retrieval.",
            "source_chunks": str(chunks_path),
            "case_count": len(cases),
            "query_type_counts": dict(sorted(Counter(case["query_type"] for case in cases).items())),
            "comparison_modes": ["visual_only", "visual_plus_image_text"],
            "recommended_metrics": [
                "Image Hit@1/3/5",
                "Image Recall@1/3/5 (Macro)",
                "Image Recall@1/3/5 (Micro)",
                "Image MRR",
                "Topic Hit@1/3/5",
                "Average retrieval latency",
                "P95 retrieval latency",
            ],
            "notes": [
                "Both retrieval modes must use this exact same dataset and expected_image_ids.",
                "visual_only searches only the image visual-vector index.",
                "visual_plus_image_text fuses visual-vector and image-text-vector ranks.",
                "Text/table retrieval and BM25 are excluded from this image-layer benchmark.",
                "Metrics should also be reported separately by query_type.",
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
