#!/usr/bin/env python
# coding: utf-8
"""Multimodal RAG tool compatible with the original RAGTool contract."""

from collections import Counter
from typing import Any, Dict, List

from langchain_core.documents import Document

from .rag_tool import RAGTool


class MultimodalRAGTool(RAGTool):
    """Use multimodal hybrid retrieval while reusing the original RAG prompt."""

    def __init__(self, multimodal_retriever, rerank_model, llm):
        super().__init__(
            faiss_retriever=None,
            bm25_retriever=None,
            rerank_model=rerank_model,
            llm=llm,
        )
        self.multimodal_retriever = multimodal_retriever

    def search_and_generate(
        self,
        query: str,
        top_k: int = 6,
        max_context_length: int = 4000,
    ) -> Dict[str, Any]:
        # Codex-MM-ADD: 这里接入新的多模态混合检索。底层 retriever 会同时融合
        # 文本向量、BM25，以及可选的 CLIP 图片视觉向量召回。
        print("  [RAG] Stage 1/4: layered text/image retrieval fusion")
        fused_results = self.multimodal_retriever.search(
            query,
            top_k=20,
            # Codex-MM-LAYERED-FUSION: retrieve a wide pool for independent
            # text/image group fusion before the shared reranker inspects it.
            candidate_k=40,
            use_rerank=False,
            max_per_topic=0,
        )
        retrieval_info = dict(self.multimodal_retriever.last_search_info)
        print(f"  [RAG] Stage 1/4 complete: {len(fused_results)} fused candidates")
        documents = [self._to_document(result) for result in fused_results]
        print("  [RAG] Stage 2/4: reranking")
        reranked_documents = self.rerank.predict(query, documents)
        selected_documents = self._limit_documents_by_topic_and_type(reranked_documents, top_k)
        selected_documents = self._ensure_image_evidence(selected_documents, documents, top_k)
        # Codex-TEXT-PARENT-CHILD: rerank precise text children first, then
        # replace selected children with their complete parent context.
        expand_parents = getattr(self.multimodal_retriever, "expand_text_parent_documents", None)
        if expand_parents:
            selected_documents = expand_parents(selected_documents)
        print(f"  [RAG] Stage 2/4 complete: {len(selected_documents)} selected documents")

        context = self._build_multimodal_context(selected_documents, max_context_length)
        prompt = self._build_prompt(context, query)
        print("  [RAG] Stage 3/4: answer generation")
        image_documents = [
            document for document in selected_documents
            if document.metadata.get("doc_type") == "image" and document.metadata.get("image_path")
        ]
        if self._is_visual_request(query) and image_documents:
            answer = self._build_visual_evidence_answer(image_documents)
        else:
            answer = self._generate_answer(prompt)
        print("  [RAG] Stage 3/4 complete")

        vector_distances = [
            document.metadata["vector_distance"]
            for document in selected_documents
            if document.metadata.get("vector_distance") is not None
        ]
        # Preserve the original RAGTool threshold contract during BM25-only
        # degradation. Empty retrieval still remains a no-answer condition.
        faiss_min_score = min(vector_distances) if vector_distances else (500.0 if selected_documents else float("inf"))
        is_no_answer = self._is_no_answer(answer, faiss_min_score)
        if self._is_visual_request(query) and image_documents:
            is_no_answer = False
        confidence = self._calculate_confidence(faiss_min_score, len(selected_documents))
        print("  [RAG] Stage 4/4: result assembled")

        return {
            "answer": answer,
            "documents": selected_documents,
            "confidence": confidence,
            "is_no_answer": is_no_answer,
            "faiss_min_score": faiss_min_score,
            "context": context,
            "source": "rag",
            "retrieval_mode": "multimodal_hybrid",
            "retrieval_info": retrieval_info,
            "image_paths": [
                document.metadata["image_path"]
                for document in selected_documents
                if document.metadata.get("image_path")
            ],
        }

    @staticmethod
    def _is_visual_request(query: str) -> bool:
        normalized = str(query).lower()
        return any(marker in normalized for marker in ("图片", "示意图", "图示", "布局图", "照片", "图中"))

    @staticmethod
    def _build_visual_evidence_answer(documents: List[Document]) -> str:
        lines = ["已在用户手册中找到相关图片证据："]
        for index, document in enumerate(documents[:3], 1):
            metadata = document.metadata
            topic = metadata.get("topic_title") or "相关图示"
            pages = metadata.get("pages") or metadata.get("page_id") or "未知"
            description = " ".join(str(document.page_content).split())[:220]
            lines.append(
                f"{index}. {topic}（手册第{pages}页）"
                + (f"：{description}" if description else "")
                + f"\n图片路径：{metadata['image_path']}"
            )
        return "\n".join(lines)

    @staticmethod
    def _to_document(result: Dict[str, Any]) -> Document:
        metadata = dict(result["metadata"])
        metadata.update(
            {
                "fused_score": result["fused_score"],
                "vector_rank": result["vector_rank"],
                "bm25_rank": result["bm25_rank"],
                "visual_rank": result.get("visual_rank"),
                "vector_distance": result.get("vector_distance"),
                "visual_distance": result.get("visual_distance"),
                "fusion_group": result.get("fusion_group"),
                "group_rank": result.get("group_rank"),
                "image_text_rank": result.get("image_text_rank"),
            }
        )
        return Document(page_content=result["content"], metadata=metadata)

    @staticmethod
    def _limit_documents_by_topic_and_type(
        documents: List[Document],
        top_k: int,
        max_per_topic_type: int = 2,
    ):
        """Keep evidence diverse without letting text hide images from the same topic."""
        selected = []
        group_counts: Counter[tuple] = Counter()
        for document in documents:
            topic_id = document.metadata.get("topic_id") or document.metadata.get("chunk_id")
            doc_type = document.metadata.get("doc_type", "text")
            group_key = (topic_id, doc_type)
            if group_counts[group_key] >= max_per_topic_type:
                continue
            selected.append(document)
            group_counts[group_key] += 1
            if len(selected) >= top_k:
                break
        return selected

    @staticmethod
    def _ensure_image_evidence(
        selected_documents: List[Document],
        retrieved_documents: List[Document],
        top_k: int,
    ) -> List[Document]:
        """Reserve one evidence slot for a highly ranked retrieved image."""
        if any(document.metadata.get("doc_type") == "image" for document in selected_documents):
            return selected_documents
        if top_k < 2:
            return selected_documents
        image = next(
            (
                document
                for document in retrieved_documents[:12]
                if document.metadata.get("doc_type") == "image"
            ),
            None,
        )
        if image is None:
            return selected_documents
        if len(selected_documents) >= top_k and selected_documents:
            return selected_documents[:-1] + [image]
        return selected_documents + [image]

    @staticmethod
    def _build_multimodal_context(documents: List[Document], max_context_length: int) -> str:
        context = ""
        for index, document in enumerate(documents, start=1):
            metadata = document.metadata
            source = (
                f"[证据{index} | {metadata.get('doc_type', 'text')} | "
                f"{metadata.get('topic_title', '未命名主题')} | "
                f"页码{metadata.get('pages') or metadata.get('page_id')}]\n"
            )
            image_path = f"\n图片路径：{metadata['image_path']}" if metadata.get("image_path") else ""
            visual_line = (
                f"\n视觉召回排名：#{metadata['visual_rank']}"
                if metadata.get("visual_rank") is not None
                else ""
            )
            section = f"{source}{document.page_content}{image_path}{visual_line}\n"
            if len(context) + len(section) > max_context_length:
                break
            context += section
        return context
