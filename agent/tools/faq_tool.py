#!/usr/bin/env python
# coding: utf-8
"""
FAQ 工具
混合匹配策略: 向量粗筛 + 关键词精排 + 自动学习

核心价值:
1. 标准答案: 人工审核的权威回复
2. 零成本: 无需 LLM 推理
3. 一致性: 同一问题永远返回相同答案
4. 合规性: 法务/客服审核的标准话术
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import numpy as np


class FAQTool:
    """
    FAQ 智能匹配 + 自动学习
    
    匹配策略:
    - 阶段1 (粗筛): 向量检索 Top-3 候选 (如果可用)
    - 阶段2 (精排): 关键实体匹配 + 高阈值
    - 只有双重验证通过才返回标准答案
    
    与 RAG 的区别:
    - FAQ: 问题↔问题匹配，返回固定答案
    - RAG: 问题↔文档匹配，LLM 动态生成
    """
    
    def __init__(
        self, 
        faq_path: str = "./data/faq_database.json",
        use_vector: bool = False  # 是否启用向量匹配
    ):
        self.faq_path = Path(faq_path)
        self.faqs = self._load_faqs()
        self.use_vector = use_vector
        self.encoder = None
        self.question_embeddings = None
        
        # 如果启用向量且有 FAQ，初始化向量索引
        if use_vector and len(self.faqs["faqs"]) > 0:
            self._init_vector_index()
    
    def _load_faqs(self) -> Dict:
        """加载 FAQ 数据库"""
        if self.faq_path.exists():
            with open(self.faq_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 初始化空数据库
            initial_data = {
                "version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "faqs": [],
                "pending_faqs": []
            }
            self._save_faqs(initial_data)
            return initial_data
    
    def _init_vector_index(self):
        """
        初始化向量索引 (可选) + 本地缓存
        使用轻量级模型，与 RAG 的大模型区分
        """
        try:
            from sentence_transformers import SentenceTransformer
            import pickle
            from pathlib import Path
            
            # 缓存路径
            cache_dir = Path("./vector_cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / "faq_embeddings.pkl"
            
            # 尝试加载缓存
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    # 检查 FAQ 是否变化
                    current_questions = [faq["question"] for faq in self.faqs["faqs"]]
                    if cache_data["questions"] == current_questions:
                        self.question_embeddings = cache_data["embeddings"]
                        print(f"  ✓ FAQ 向量索引已从缓存加载 ({len(current_questions)} 个问题)")
                        
                        # 仍需加载模型用于查询
                        self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                        return
                except Exception as e:
                    print(f"  ⚠️  FAQ 缓存加载失败: {e}")
            
            # 缓存不存在或失效，重新构建
            self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            questions = [faq["question"] for faq in self.faqs["faqs"]]
            self.question_embeddings = self.encoder.encode(questions, show_progress_bar=False)
            
            # 保存缓存
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        "questions": questions,
                        "embeddings": self.question_embeddings
                    }, f)
                print(f"  ✓ FAQ 向量索引已构建并缓存 ({len(questions)} 个问题)")
            except Exception as e:
                print(f"  ⚠️  FAQ 缓存保存失败: {e}")
                print(f"  ✓ FAQ 向量索引已构建 ({len(questions)} 个问题)")
                
        except ImportError:
            print("  ⚠️  sentence-transformers 未安装，使用关键词匹配")
            self.use_vector = False
    
    def match(self, query: str, threshold: float = 0.85) -> Optional[Dict]:
        """
        混合匹配策略: 向量粗筛 + 关键词精排
        
        Args:
            query: 用户问题
            threshold: 相似度阈值 (注意: 比 RAG 更高)
            
        Returns:
            匹配结果或 None
            
        核心逻辑:
        1. 如果启用向量: 先找 Top-3 候选
        2. 对候选进行关键实体验证
        3. 只有高置信度 (≥0.85) 才返回标准答案
        """
        if len(self.faqs["faqs"]) == 0:
            return None
        
        if self.use_vector and self.encoder is not None:
            # 向量匹配 (粗筛)
            candidates = self._vector_match(query, top_k=3)
        else:
            # 关键词匹配 (全量)
            candidates = [(faq, self._keyword_similarity(query, faq)) 
                         for faq in self.faqs["faqs"]]
            candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = candidates[:3]  # 也只取 Top-3
        
        # 精排: 关键实体验证
        best_match = None
        best_score = 0.0
        
        for faq, vector_score in candidates:
            # 关键词得分
            keyword_score = self._keyword_similarity(query, faq)
            
            # 综合得分: 向量 60% + 关键词 40%
            if self.use_vector:
                final_score = vector_score * 0.6 + keyword_score * 0.4
            else:
                final_score = keyword_score
            
            # 关键实体验证 (提高准确性)
            if not self._entity_match(query, faq):
                final_score *= 0.7  # 实体不匹配，大幅降分
            
            if final_score > best_score:
                best_score = final_score
                best_match = faq
        
        # 高阈值过滤 (确保准确性)
        if best_score >= threshold:
            # 更新使用统计
            best_match["frequency"] = best_match.get("frequency", 0) + 1
            best_match["last_used"] = datetime.now().isoformat()
            self._save_faqs(self.faqs)
            
            return {
                "answer": best_match["answer"],
                "confidence": float(best_score),
                "source": f"FAQ#{best_match['id']}",
                "matched_question": best_match["question"]
            }
        
        return None
    
    def _vector_match(self, query: str, top_k: int = 3) -> List[tuple]:
        """
        向量粗筛: 快速找到候选问题
        """
        query_embedding = self.encoder.encode([query], show_progress_bar=False)[0]
        similarities = np.dot(self.question_embeddings, query_embedding)
        
        # Top-K 候选
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        candidates = []
        for idx in top_indices:
            faq = self.faqs["faqs"][idx]
            score = float(similarities[idx])
            candidates.append((faq, score))
        
        return candidates
    
    def _keyword_similarity(self, query: str, faq: Dict) -> float:
        """
        关键词相似度计算
        """
        faq_question = faq["question"]
        keywords = faq.get("keywords", [])
        
        query_lower = query.lower()
        faq_lower = faq_question.lower()
        
        # 完全匹配
        if query_lower == faq_lower:
            return 1.0
        
        # 关键词匹配
        if keywords:
            matched_keywords = sum(1 for kw in keywords if kw.lower() in query_lower)
            keyword_score = matched_keywords / len(keywords)
            if keyword_score > 0:
                return 0.5 + keyword_score * 0.5
        
        # 部分匹配
        if faq_lower in query_lower or query_lower in faq_lower:
            return 0.6
        
        return 0.0
    
    def _entity_match(self, query: str, faq: Dict) -> bool:
        """
        关键实体匹配验证
        
        确保问题中的关键实体 (产品型号、功能名) 一致
        这是 FAQ 与 RAG 的关键区别
        
        示例:
        - "银河E8续航" vs "银河E8续航" → True (实体匹配)
        - "银河E8续航" vs "星越L续航" → False (产品不同)
        """
        # 提取 FAQ 问题中的关键实体
        entities = faq.get("entities", [])
        if not entities:
            return True  # 没有定义实体，跳过检查
        
        # 检查实体是否都出现在用户问题中
        query_lower = query.lower()
        matched_entities = sum(1 for entity in entities if entity.lower() in query_lower)
        
        # 至少匹配 80% 的关键实体
        return matched_entities >= len(entities) * 0.8
    

    
    def add_to_pending(self, question: str, answer: str):
        """
        添加候选 FAQ (需人工审核)
        
        触发条件: 高质量答案 + 重复出现
        """
        # 检查是否已存在
        for pending in self.faqs["pending_faqs"]:
            if pending["question"] == question:
                pending["occurrences"] += 1
                # 超过 3 次自动提升
                if pending["occurrences"] >= 3:
                    self.promote_to_faq(question)
                else:
                    self._save_faqs(self.faqs)
                return
        
        # 新增候选
        self.faqs["pending_faqs"].append({
            "question": question,
            "candidate_answer": answer,
            "occurrences": 1,
            "need_review": True,
            "created_at": datetime.now().isoformat()
        })
        self._save_faqs(self.faqs)
    
    def promote_to_faq(self, pending_question: str):
        """
        将审核通过的问题提升为正式 FAQ
        """
        for pending in self.faqs["pending_faqs"]:
            if pending["question"] == pending_question:
                new_faq = {
                    "id": f"FAQ{len(self.faqs['faqs']) + 1:03d}",
                    "question": pending["question"],
                    "answer": pending["candidate_answer"],
                    "keywords": self._extract_keywords(pending["question"]),
                    "category": "自动学习",
                    "frequency": pending["occurrences"],
                    "confidence": 0.95,
                    "source": "自动学习",
                    "created_at": datetime.now().isoformat(),
                    "last_used": datetime.now().isoformat()
                }
                self.faqs["faqs"].append(new_faq)
                self.faqs["pending_faqs"].remove(pending)
                self._save_faqs(self.faqs)
                print(f"✅ 新 FAQ 已添加: {pending['question']}")
                break
    
    def _extract_keywords(self, question: str) -> List[str]:
        """
        提取关键词 (简单版本)
        """
        # 简单的分词 + 过滤
        import jieba
        words = jieba.lcut(question)
        # 过滤停用词和短词
        stopwords = {"的", "是", "在", "了", "吗", "呢", "啊", "吧", "什么", "怎么", "如何"}
        keywords = [w for w in words if len(w) > 1 and w not in stopwords]
        return keywords[:5]  # 最多 5 个关键词
    
    def _save_faqs(self, data: Dict):
        """保存 FAQ 数据库"""
        data["last_updated"] = datetime.now().isoformat()
        self.faq_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.faq_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
