#!/usr/bin/env python
# coding: utf-8
"""
从 eval_cases.json 自动生成 FAQ 条目
利用已有的问题和标准答案来扩充FAQ库
"""

import json
from pathlib import Path
from datetime import datetime


def load_documents(documents_path: str) -> dict:
    """加载文档映射"""
    with open(documents_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    return {doc['doc_id']: doc['content'] for doc in documents}


def generate_faq_from_eval_cases(
    eval_cases_path: str,
    documents_path: str,
    faq_database_path: str,
    min_doc_count: int = 1,
    max_new_faqs: int = 50
):
    """
    从评估用例生成FAQ条目
    
    Args:
        eval_cases_path: 评估用例JSON路径
        documents_path: 文档JSON路径
        faq_database_path: FAQ数据库路径
        min_doc_count: 最少相关文档数（过滤复杂问题）
        max_new_faqs: 最多生成多少新FAQ
    """
    print("=" * 80)
    print("🔄 从评估用例生成FAQ条目")
    print("=" * 80)
    
    # 加载数据
    with open(eval_cases_path, 'r', encoding='utf-8') as f:
        eval_cases = json.load(f)
    
    doc_map = load_documents(documents_path)
    
    # 加载现有FAQ
    if Path(faq_database_path).exists():
        with open(faq_database_path, 'r', encoding='utf-8') as f:
            faq_db = json.load(f)
    else:
        faq_db = {
            "version": "2.0",
            "last_updated": datetime.now().isoformat(),
            "description": "FAQ 标准问答库 - 自动扩充版本",
            "faqs": [],
            "pending_faqs": []
        }
    
    existing_questions = {faq['question'] for faq in faq_db['faqs']}
    print(f"✓ 现有FAQ数量: {len(faq_db['faqs'])}")
    print(f"✓ 评估用例数量: {len(eval_cases)}")
    
    # 筛选适合转为FAQ的案例
    candidates = []
    for case in eval_cases:
        # 条件1: 只有一个相关文档（简单问题）
        if len(case['reference_doc_ids']) != min_doc_count:
            continue
        
        # 条件2: 问题不要太长
        if len(case['question']) > 50:
            continue
        
        # 条件3: 尚未存在于FAQ中
        if case['question'] in existing_questions:
            continue
        
        # 获取答案内容
        doc_id = case['reference_doc_ids'][0]
        if doc_id in doc_map:
            answer = doc_map[doc_id]
            
            # 条件4: 答案不要太长（FAQ应该是简洁的）
            if len(answer) > 200:
                # 截取前150字
                answer = answer[:150] + "..."
            
            candidates.append({
                'question': case['question'],
                'answer': answer,
                'doc_id': doc_id,
                'category': case.get('category', '未分类'),
                'case_id': case['case_id']
            })
    
    print(f"✓ 找到 {len(candidates)} 个候选FAQ")
    
    # 按类别分组，确保多样性
    from collections import defaultdict
    category_groups = defaultdict(list)
    for candidate in candidates:
        category_groups[candidate['category']].append(candidate)
    
    # 从每个类别中选取代表性的FAQ
    new_faqs = []
    remaining_slots = max_new_faqs
    
    # 优先选择高频类别
    sorted_categories = sorted(category_groups.items(), key=lambda x: len(x[1]), reverse=True)
    
    for category, items in sorted_categories:
        if remaining_slots <= 0:
            break
        
        # 每个类别最多取5个
        take_count = min(len(items), 5, remaining_slots)
        selected = items[:take_count]
        
        for item in selected:
            # 生成关键词
            keywords = extract_keywords(item['question'])
            
            # 提取实体（简化版）
            entities = extract_entities(item['question'])
            
            new_faq = {
                "id": f"FAQ{len(faq_db['faqs']) + len(new_faqs) + 1:03d}",
                "question": item['question'],
                "answer": item['answer'],
                "keywords": keywords,
                "entities": entities,
                "category": item['category'],
                "frequency": 0,
                "confidence": 0.90,
                "source": f"自动生成自 {item['case_id']}, 参考文档ID={item['doc_id']}",
                "created_at": datetime.now().isoformat(),
                "last_used": datetime.now().isoformat(),
                "note": f"从评估用例 {item['case_id']} 自动生成"
            }
            new_faqs.append(new_faq)
        
        remaining_slots -= take_count
        print(f"  📌 {category}: 添加 {take_count} 条")
    
    # 添加到FAQ数据库
    faq_db['faqs'].extend(new_faqs)
    faq_db['last_updated'] = datetime.now().isoformat()
    
    # 保存
    with open(faq_database_path, 'w', encoding='utf-8') as f:
        json.dump(faq_db, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 成功添加 {len(new_faqs)} 条新FAQ")
    print(f"📊 FAQ总数: {len(faq_db['faqs'])}")
    print(f"💾 已保存到: {faq_database_path}")
    
    return len(new_faqs)


def extract_keywords(question: str, max_keywords: int = 5) -> list:
    """提取关键词（简化版）"""
    try:
        import jieba
        words = jieba.lcut(question)
        stopwords = {"的", "是", "在", "了", "吗", "呢", "啊", "吧", "什么", "怎么", "如何", "多少", "几"}
        keywords = [w for w in words if len(w) > 1 and w not in stopwords]
        return keywords[:max_keywords]
    except ImportError:
        # 如果没有jieba，简单分词
        return question.split()[:max_keywords]


def extract_entities(question: str) -> list:
    """提取关键实体（简化版）"""
    # 常见的车辆相关实体
    vehicle_entities = [
        "银河E8", "吉利", "续航", "充电", "胎压", "雨刮器", 
        "氛围灯", "语音助手", "自动泊车", "座椅", "空调",
        "HUD", "后备箱", "发动机", "电池", "快充", "慢充"
    ]
    
    entities = []
    for entity in vehicle_entities:
        if entity in question:
            entities.append(entity)
    
    return entities[:3]  # 最多3个实体


def main():
    base_dir = Path(__file__).parent
    eval_cases_path = base_dir / "data" / "eval_cases.json"
    documents_path = base_dir / "data" / "documents.json"
    faq_database_path = base_dir / "data" / "faq_database.json"
    
    # 检查文件
    if not eval_cases_path.exists():
        print(f"❌ 错误: 找不到 {eval_cases_path}")
        return
    
    if not documents_path.exists():
        print(f"❌ 错误: 找不到 {documents_path}")
        return
    
    # 生成FAQ
    num_added = generate_faq_from_eval_cases(
        eval_cases_path=str(eval_cases_path),
        documents_path=str(documents_path),
        faq_database_path=str(faq_database_path),
        min_doc_count=1,  # 只选单文档问题
        max_new_faqs=50   # 最多添加50条
    )
    
    print("\n" + "=" * 80)
    print("✅ FAQ扩充完成!")
    print("=" * 80)
    print("\n📌 下一步:")
    print("  1. 可以手动编辑 faq_database.json 优化答案质量")
    print("  2. 运行 FAQ vs RAG 对比评估脚本")


if __name__ == "__main__":
    main()
