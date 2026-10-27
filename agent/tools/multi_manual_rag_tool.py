"""Scoped BYD manual RAG tool shared by plain and Agentic text paths."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.documents import Document
from llm_streaming import generate_text_streaming

from .multimodal_rag_tool import MultimodalRAGTool


class MultiManualRAGTool:
    supports_model_scope = True

    QUERY_REWRITE_PROMPT = """你是汽车用户手册 RAG 系统中的检索查询改写器。你的唯一任务是把用户问题改写成适合检索用户手册的查询，不回答问题、不生成答案。

车型范围由前端下拉框确定，当前已选择：{selected_models}
程序会在每款已选择车型自己的手册范围内分别检索。你不能增删车型范围，也不能根据问题中的车型名称改变下拉框选择。

用户原问题：
{question}

改写要求：
1. 保持原问题的信息需求、提问对象和语气，只调整成更适合在手册中查找的表达。
2. 保留原问题要求的全部属性、步骤、比较维度和它们之间的关系，不得把多项问题缩成一项。
3. 原问题中的年份、年款、数字、单位、时间、位置、条件、范围、否定词和先后关系必须保留，不得猜测或改动。
4. 不得新增车型、品牌、年份、配置、数值、法规、标准、故障原因或用户没有提出的检索方向；不得把可能性改成确定事实。
5. 车型范围以已选择车型为准。若原问题中的车型名只是说明这些已选车型的检索范围，可以从检索查询中省略；若它可能表达其他对象或含义不确定，保留原文，不要自行映射别名。
6. 如果用户比较多款已选择车型，必须生成一条供所有已选车型共用的检索查询。提取并保留全部被比较的功能、属性和指标，删除只用于标识检索车型的车型名，以及“有什么区别”“分别”等纯比较表达；不得按车型拆成多个子句，不得输出“车型名＋该车型属性”的对应清单。
   示例：原问题“汉 EV 闪充版和秦 MAX EV 的充电方式和充电时间有什么区别？”应改写为“充电方式、充电时间”。
7. 如果问题含有“它、这个、前者”等无法仅凭本条问题确定的指代，或缺少关键对象，不要猜测，原样返回原问题。
8. 如果原问题已经清楚且适合检索，可以原样返回；多车型比较问题仍须遵守第 6 条，生成共用的属性查询。

只输出一个 JSON 对象，格式必须为：{{"query":"改写后的检索问题"}}
不要输出 Markdown、解释、前后缀或其他字段。"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def _rewrite_query(self, query: str, model_names: list[str]) -> str:
        """Rewrite only the retrieval query; fall back to the user's wording on any error."""
        prompt = self.QUERY_REWRITE_PROMPT.format(
            selected_models="、".join(model_names), question=query
        )
        try:
            output = self.llm.infer([prompt])
            raw = str(output[0]).strip() if output else ""
            match = re.search(r"\{[\s\S]*\}", raw)
            if not match:
                raise ValueError("改写模型没有返回 JSON 对象")
            value = json.loads(match.group())
            rewritten = value.get("query") if isinstance(value, dict) else None
            if not isinstance(rewritten, str):
                raise ValueError("改写结果缺少字符串 query 字段")
            rewritten = rewritten.strip()
            if not 2 <= len(rewritten) <= 500:
                raise ValueError("改写后的 query 长度必须为 2～500 个字符")
            return rewritten
        except Exception as exc:
            print(f"  [普通 RAG 查询改写] 失败，回退原问题: {exc}")
            return query

    @staticmethod
    def _source_line(metadata: dict[str, Any]) -> str:
        page = metadata.get("page_start") or metadata.get("page_number") or "未知"
        return (f"车型：{metadata['model_name']}；章节：{metadata['section_title']}；"
                f"PDF 物理页：{page}；手册 ID：{metadata['doc_id']}")

    def search_and_generate(self, query: str, top_k: int = 6, max_context_length: int = 4000,
                            model_names: list[str] | None = None,
                            rewrite_query: bool = False) -> dict[str, Any]:
        if not model_names:
            raise ValueError("必须明确选择至少一款车型；不能无车型范围检索三本手册")
        if len(set(model_names)) != len(model_names):
            raise ValueError("车型列表存在重复项")
        visual_request = MultimodalRAGTool._is_visual_request(query)
        retrieval_query = (
            self._rewrite_query(query, model_names)
            if rewrite_query and not visual_request else query
        )
        documents: list[Document] = []
        image_paths: list[str] = []
        per_model = 1 if visual_request else max(2, top_k // len(model_names))
        for model in model_names:
            for hit in self.retriever.search_text(retrieval_query, model, top_k=per_model):
                metadata = dict(hit["metadata"])
                metadata.update({
                    "topic_title": metadata["section_title"],
                    "pages": [metadata["page_start"], metadata["page_end"]],
                    "doc_type": "table" if metadata["chunk_type"] == "table_child" else "text",
                    "vector_rank": hit["ranks"].get("vector"),
                    "bm25_rank": hit["ranks"].get("bm25"),
                })
                documents.append(Document(page_content=hit["content"], metadata=metadata))
            if visual_request:
                for hit in self.retriever.search_images(query, model, top_k=2):
                    metadata = dict(hit["metadata"])
                    metadata.update({
                        "topic_title": metadata["section_title"], "pages": [metadata["page_number"]],
                        "doc_type": "image", "chunk_id": hit["image_id"],
                        "visual_rank": hit["ranks"].get("visual"),
                        "image_text_rank": hit["ranks"].get("nearby_text"),
                    })
                    documents.append(Document(page_content=hit["nearby_text"], metadata=metadata))

        if visual_request:
            # Images from every model get the first chance at the shared
            # context budget. Nearby text is only a locator, not a caption.
            documents.sort(key=lambda doc: doc.metadata["doc_type"] != "image")
        else:
            # Comparison questions must not let the first selected model use
            # the entire context before any evidence from later models.
            first_per_model: list[Document] = []
            remaining: list[Document] = []
            seen_models: set[str] = set()
            for doc in documents:
                model = doc.metadata["model_name"]
                if model not in seen_models:
                    first_per_model.append(doc)
                    seen_models.add(model)
                else:
                    remaining.append(doc)
            documents = first_per_model + remaining

        context_parts: list[str] = []
        used_documents: list[Document] = []
        for doc in documents:
            source = self._source_line(doc.metadata)
            if doc.metadata["doc_type"] == "image":
                passage = f"[{source}；图片候选，以下只是附近正文，并非图注]\n{doc.page_content}\n"
            else:
                passage = f"[{source}]\n{doc.page_content}\n"
            if sum(map(len, context_parts)) + len(passage) > max_context_length:
                continue  # do not let an oversized hit suppress all later models
            context_parts.append(passage)
            used_documents.append(doc)
            if doc.metadata["doc_type"] == "image":
                image_paths.append(doc.metadata["image_path"])
        context = "\n".join(context_parts)

        if not context:
            answer = "当前已解析的手册范围内没有足够证据回答。"
        elif visual_request and image_paths:
            # The configured generator accepts text only. Return the actual
            # image assets to the UI; never pretend nearby text is visual OCR.
            answer = ("已找到相关手册图片，请查看下方图片证据。"
                      "附近文字仅用于检索，不能据此断定图片中的全部细节。")
        elif visual_request:
            answer = "当前已解析的手册范围内没有足够的图片证据回答。"
        else:
            prompt = (
                "你是汽车手册问答助手。仅根据下列已检索的比亚迪用户手册证据回答。"
                "不同车型的证据不得混用；比较问题请逐车型回答并标注车型、PDF物理页。"
                "证据不足或手册尚未解析到该范围时明确说明，不编造车型配置、数字或操作建议。\n\n"
                f"证据：\n{context}\n问题：{query}"
            )
            answer = generate_text_streaming(self.llm, prompt, replace=True)
            if not answer:
                answer = "当前已解析的手册范围内没有足够证据回答。"
        no_answer = not context or (visual_request and not image_paths) or any(
            marker in answer for marker in ("无答案", "没有足够证据")
        )
        return {
            "answer": answer, "documents": used_documents, "confidence": 0.0,
            "is_no_answer": no_answer, "context": context, "source": "rag",
            "retrieval_mode": "multi_manual_scoped", "image_paths": image_paths,
            "selected_models": list(model_names), "retrieval_query": retrieval_query,
        }
