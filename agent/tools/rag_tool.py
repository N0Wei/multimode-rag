#!/usr/bin/env python
# coding: utf-8
"""
RAG 工具
封装现有的 FAISS + BM25 + Rerank 检索系统
"""

from typing import Dict, Any, List
from langchain_core.documents import Document
from llm_streaming import generate_text_streaming


class RAGTool:
    """
    封装现有 RAG 检索系统
    复用 faiss_retriever.py, bm25_retriever.py, rerank_model.py
    """
    
    def __init__(self, faiss_retriever, bm25_retriever, rerank_model, llm):
        """
        Args:
            faiss_retriever: FaissRetriever 实例
            bm25_retriever: BM25 实例
            rerank_model: reRankLLM 实例
            llm: 生成模型实例
        """
        self.faiss_retriever = faiss_retriever
        self.bm25_retriever = bm25_retriever
        self.rerank = rerank_model
        self.llm = llm
    
    def search_and_generate(
        self, 
        query: str, 
        top_k: int = 6,
        max_context_length: int = 4000
    ) -> Dict[str, Any]:
        """
        执行 RAG 检索 + 答案生成
        
        Returns:
            {
                "answer": str,              # 生成的答案
                "documents": List[Document], # 检索到的文档
                "confidence": float,         # 置信度
                "is_no_answer": bool,        # 是否为"无答案"
                "faiss_min_score": float,    # FAISS 最小距离
                "context": str               # 使用的上下文
            }
        """
        # 1. FAISS 检索
        faiss_docs = self.faiss_retriever.GetTopK(query, 15)
        faiss_min_score = faiss_docs[0][1] if faiss_docs else float('inf')
        
        # 2. BM25 检索
        bm25_docs = self.bm25_retriever.GetBM25TopK(query, 15)
        
        # 3. 重排序
        merged_items = []
        for doc, score in faiss_docs:
            merged_items.append(doc)
        merged_items.extend(bm25_docs)
        
        reranked_docs = self.rerank.predict(query, merged_items)[:top_k]
        
        # 4. 构建上下文
        context = ""
        for doc in reranked_docs:
            if len(context + doc.page_content) > max_context_length:
                break
            context += doc.page_content + "\n"
        
        # 5. 生成答案
        prompt = self._build_prompt(context, query)
        answer = self._generate_answer(prompt)
        
        # 6. 判断是否为"无答案"
        is_no_answer = self._is_no_answer(answer, faiss_min_score)
        
        # 7. 计算置信度
        confidence = self._calculate_confidence(faiss_min_score, len(reranked_docs))
        
        return {
            "answer": answer,
            "documents": reranked_docs,
            "confidence": confidence,
            "is_no_answer": is_no_answer,
            "faiss_min_score": faiss_min_score,
            "context": context,
            "source": "rag"
        }
    
    def _build_prompt(self, context: str, query: str) -> str:
        """构建生成提示"""
        prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
如果无法从中得到答案，请说 "无答案"，不允许在答案中添加编造成分，答案请使用中文。
已知内容为吉利控股集团汽车销售有限公司的吉利用户手册:

{context}

问题: {question}"""
        
        return prompt_template.format(context=context, question=query)
    
    def _generate_answer(self, prompt: str) -> str:
        """调用 LLM 生成答案"""
        answer = generate_text_streaming(self.llm, prompt, replace=True)
        return answer or "无答案"
    
    def _is_no_answer(self, answer: str, faiss_score: float) -> bool:
        """
        判断是否为"无答案"
        
        条件:
        1. 答案中包含"无答案"关键词
        2. FAISS 距离过大 (> 500)
        """
        no_answer_keywords = ["无答案", "无法回答", "没有找到", "不知道"]
        
        # 检查关键词
        if any(kw in answer for kw in no_answer_keywords):
            return True
        
        # 检查 FAISS 分数
        if faiss_score > 500:
            return True
        
        return False
    
    def _calculate_confidence(self, faiss_score: float, doc_count: int) -> float:
        """
        计算置信度 (0-1)
        
        基于:
        - FAISS 距离分数
        - 检索到的文档数量
        """
        # FAISS 分数越低越好，转换为置信度
        score_confidence = max(0, 1 - faiss_score / 1000)
        
        # 文档数量置信度
        doc_confidence = min(doc_count / 6, 1.0)
        
        # 综合置信度
        confidence = (score_confidence * 0.7 + doc_confidence * 0.3)
        
        return round(confidence, 2)
