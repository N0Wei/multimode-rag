#!/usr/bin/env python
# coding: utf-8
"""
自动化 FAQ 生成器
利用 LLM 从原始文档中提取高频问答对，扩充 faq_database.json
"""
import json
import os
from pdf_parse import DataProcess
from llm_model import get_llm_model
from config import LLM_BACKEND

def generate_faqs_from_pdf(pdf_path, output_path="./data/faq_database.json", num_batches=5):
    print("=" * 70)
    print("🤖 自动化 FAQ 生成器启动")
    print("=" * 70)
    
    # 1. 解析 PDF 获取原始文本
    print("\n[1/3] 📄 正在解析 PDF 文档...")
    dp = DataProcess(pdf_path=pdf_path)
    dp.ParseBlock(max_seq=512) # 使用中等长度切片以保留上下文
    data = dp.data
    
    # 2. 初始化 LLM
    print("[2/3] 🧠 正在初始化 LLM...")
    llm = get_llm_model(backend=LLM_BACKEND)
    
    # 3. 批量提取 FAQ
    print(f"[3/3] ⚙️  正在分 {num_batches} 批次提取 FAQ...\n")
    
    new_faqs = []
    batch_size = len(data) // num_batches
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_text = "\n".join(data[start_idx:end_idx])
        
        prompt = f"""
        你是一个专业的汽车客服知识库专家。请阅读以下【吉利汽车用户手册片段】，
        从中提取出 5 个最具代表性、用户最常问的“问题-答案”对。
        
        【要求】：
        1. 问题要简洁明了，符合车主口语习惯（如：“怎么开空调？”而不是“空调系统的开启方式是什么？”）。
        2. 答案要准确、专业，直接引用手册内容。
        3. 必须严格按照 JSON 格式输出，不要包含 Markdown 标记。
        
        【输出格式示例】：
        [
            {{"question": "问题1", "answer": "答案1"}},
            {{"question": "问题2", "answer": "答案2"}}
        ]
        
        【文档片段】：
        {batch_text[:2000]} 
        """
        
        try:
            response = llm.invoke(prompt)
            content = response.content.strip()
            # 清理可能存在的 Markdown 代码块标记
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            batch_faqs = json.loads(content)
            new_faqs.extend(batch_faqs)
            print(f"  ✓ 第 {i+1} 批提取成功，获得 {len(batch_faqs)} 条 FAQ")
        except Exception as e:
            print(f"  ❌ 第 {i+1} 批提取失败: {e}")
    
    # 4. 合并并保存到 faq_database.json
    print("\n💾 正在保存至数据库...")
    if os.path.exists(output_path):
        with open(output_path, "r", encoding='utf-8') as f:
            db = json.load(f)
    else:
        db = {"version": "2.0", "faqs": [], "pending_faqs": []}
    
    # 去重逻辑：如果问题已经存在则跳过
    existing_questions = {item['question'] for item in db['faqs']}
    added_count = 0
    
    for item in new_faqs:
        if item['question'] not in existing_questions:
            # 补充标准字段
            faq_entry = {
                "id": f"FAQ_AUTO_{len(db['faqs']) + 1:03d}",
                "question": item['question'],
                "answer": item['answer'],
                "keywords": [], # 后续可由 FAQTool 自动提取
                "entities": [],
                "category": "自动生成",
                "frequency": 0,
                "confidence": 0.9,
                "source": "LLM 自动提取自 train_a.pdf"
            }
            db['faqs'].append(faq_entry)
            existing_questions.add(item['question'])
            added_count += 1
    
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ 处理完成！共新增 {added_count} 条高质量 FAQ。")
    print(f"   当前 FAQ 库总数: {len(db['faqs'])} 条")

if __name__ == "__main__":
    generate_faqs_from_pdf("./data/train_a.pdf")
