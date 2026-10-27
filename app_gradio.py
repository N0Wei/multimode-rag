#!/usr/bin/env python
# coding: utf-8
"""Gradio console for the RAG Agent customer-service demo."""

import os
import hashlib
import html
import inspect
import time
import threading
import queue
import json
import weakref
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple


def _ensure_localhost_no_proxy():
    """Keep Gradio's local startup checks away from system HTTP proxies."""
    local_hosts = ["localhost", "127.0.0.1", "::1"]
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        values = [item.strip() for item in current.split(",") if item.strip()]
        for host in local_hosts:
            if host not in values:
                values.append(host)
        os.environ[key] = ",".join(values)


_ensure_localhost_no_proxy()

import gradio as gr

_CHATBOT_MESSAGE_KWARGS = {"type": "messages"} if "type" in inspect.signature(gr.Chatbot).parameters else {}


def _make_chatbot(**kwargs):
    """Keep the existing message UI usable with Gradio 5 and 6."""
    accepted = inspect.signature(gr.Chatbot).parameters
    return gr.Chatbot(**{key: value for key, value in kwargs.items() if key in accepted})


def _make_gallery(**kwargs):
    accepted = inspect.signature(gr.Gallery).parameters
    return gr.Gallery(**{key: value for key, value in kwargs.items() if key in accepted})

from agent.graph import build_agent_graph
from agent.nodes.tool_executor import set_multi_manual_tool, set_tools
from agent.state import AgentState
from agent.tools import (
    BochaSearchMCPTool, DDGSSearchTool, FallbackSearchTool,
    FAQTool, MultimodalRAGTool, RAGTool,
)
from agent.tools.multi_manual_rag_tool import MultiManualRAGTool
from bm25_retriever import BM25
from config import LLM_BACKEND
from faiss_retriever import FaissRetriever
from llm_model import get_llm_model
from llm_streaming import reset_stream_callback, set_stream_callback
from pdf_parse import DataProcess
from rerank_model import reRankLLM
from short_term_memory import ShortTermMemory
from knowledge_base_service import (
    build_knowledge_base,
    get_active_knowledge_base,
    knowledge_base_paths,
    list_knowledge_bases,
    search_knowledge_base,
    set_active_knowledge_base,
)
from multimodal_retriever import MultimodalRetriever
from multi_manual_store import MultiManualRetriever, list_active_manual_models, list_index_versions
from multi_manual_upload import upload_and_activate_manual, rollback_to_index_version


agent_graph = None
plain_rag_tool = None
plain_rag_source_pdf: Path | None = None
plain_rag_llm = None
plain_rag_reranker = None
plain_rag_lock = threading.Lock()
multi_manual_tool = None
multi_manual_lock = threading.Lock()
_multi_manual_tool_cache = weakref.WeakValueDictionary()
MULTI_MANUAL_STORE = Path("data/multi_manual_store")
short_term_memory = ShortTermMemory()
latest_retrieved_images_by_session: Dict[str, List[Tuple[str, str]]] = {}
latest_citations_by_session: Dict[str, str] = {}


def _run_streaming_call(call, stream_state: Dict[str, Any] | None = None):
    """Run a blocking model/RAG call off-thread and forward its token events."""
    events: queue.Queue = queue.Queue()

    def worker():
        callback = lambda kind, text: events.put((kind, text))
        token = set_stream_callback(callback)
        metadata = stream_state.setdefault("metadata", {}) if stream_state is not None else None
        previous_callback = metadata.get("_stream_callback") if metadata is not None else None
        if metadata is not None:
            metadata["_stream_callback"] = callback
        try:
            result = call()
            if isinstance(result, dict):
                (result.get("metadata") or {}).pop("_stream_callback", None)
            events.put(("result", result))
        except BaseException as exc:
            events.put(("error", exc))
        finally:
            if metadata is not None:
                if previous_callback is None:
                    metadata.pop("_stream_callback", None)
                else:
                    metadata["_stream_callback"] = previous_callback
            reset_stream_callback(token)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        kind, payload = events.get()
        if kind == "error":
            raise payload
        yield kind, payload
        if kind == "result":
            return


def _format_execution_trace(events: List[Dict[str, Any]]) -> str:
    """Render a compact chronological trace for the Gradio status panel."""
    lines = []
    for index, event in enumerate(events, 1):
        kind = event.get("event_type", "event")
        name = event.get("name") or event.get("tool") or "unknown"
        duration = float(event.get("duration_ms", 0) or 0)
        status = event.get("status") or ("success" if event.get("success") else "failed")
        flags = []
        if event.get("cache_hit"):
            flags.append("缓存命中")
        if event.get("deduplicated"):
            flags.append("重复拦截")
        if event.get("budget_blocked"):
            flags.append("预算阻止")
        suffix = f" ({', '.join(flags)})" if flags else ""
        lines.append(f"{index}. [{kind}] {name} - {status} - {duration:.0f}ms{suffix}")
    return "\n".join(lines)


def _rag_documents(result: Dict[str, Any]) -> List[Any]:
    """Return final RAG evidence, including evidence preserved across retries."""
    multi_hop_documents = []
    for item in result.get("subquery_results") or []:
        evidence = item.get("evidence") or {}
        multi_hop_documents.extend(evidence.get("documents") or [])
    if multi_hop_documents:
        return multi_hop_documents
    documents = (result.get("rag_result") or {}).get("documents") or []
    if documents:
        return documents
    evidence = (result.get("metadata") or {}).get("evidence_rag_result") or {}
    return evidence.get("documents") or []


def _document_metadata(document: Any) -> Dict[str, Any]:
    """Read evidence metadata after either live or serialized graph state."""
    if isinstance(document, dict):
        return document.get("metadata") or {}
    return getattr(document, "metadata", {}) or {}


def _document_content(document: Any) -> str:
    if isinstance(document, dict):
        return str(document.get("page_content") or document.get("content") or "")
    return str(getattr(document, "page_content", ""))


def _collect_citations(result: Dict[str, Any], max_citations: int = 6) -> List[Dict[str, Any]]:
    """Convert the final Agent state into a stable, UI-only citation model."""
    citations: List[Dict[str, Any]] = []
    seen = set()
    answer_source = result.get("answer_source") or ""

    def add(citation: Dict[str, Any]) -> None:
        key = (
            citation.get("source_type"),
            citation.get("chunk_id") or citation.get("url") or citation.get("title"),
        )
        if key in seen or len(citations) >= max_citations:
            return
        seen.add(key)
        citations.append(citation)

    if answer_source == "multi_hop":
        for item in result.get("subquery_results") or []:
            evidence = item.get("evidence") or {}
            source = item.get("source")
            if source == "rag":
                documents = evidence.get("documents") or []
                if documents:
                    for document in documents:
                        metadata = _document_metadata(document)
                        content = " ".join(_document_content(document).split())
                        add({
                            "source_type": "rag",
                            "title": metadata.get("topic_title") or item.get("question") or "知识库证据",
                            "pages": metadata.get("pages") or metadata.get("page_id"),
                            "doc_type": metadata.get("doc_type") or "text",
                            "excerpt": content[:360] + ("..." if len(content) > 360 else ""),
                            "chunk_id": metadata.get("chunk_id") or f"subquery-{item.get('index')}-{len(citations)}",
                        })
                elif evidence.get("context"):
                    context = " ".join(str(evidence["context"]).split())
                    add({
                        "source_type": "rag",
                        "title": item.get("question") or "RAG 检索上下文",
                        "doc_type": "context",
                        "excerpt": context[:600] + ("..." if len(context) > 600 else ""),
                        "chunk_id": f"subquery-context-{item.get('index')}",
                    })
            elif source == "faq":
                add({
                    "source_type": "faq",
                    "title": evidence.get("matched_question") or item.get("question") or "FAQ 标准问答",
                    "doc_type": "faq",
                    "excerpt": str(item.get("answer") or "")[:360],
                    "chunk_id": evidence.get("source") or f"subquery-faq-{item.get('index')}",
                })
            elif source == "web_search":
                for web_item in evidence.get("results") or []:
                    content = " ".join(str(web_item.get("content") or "").split())
                    add({
                        "source_type": "web",
                        "title": web_item.get("title") or web_item.get("url") or item.get("question"),
                        "doc_type": "web_search",
                        "excerpt": content[:360] + ("..." if len(content) > 360 else ""),
                        "url": web_item.get("url") or "",
                    })

    current_rag = result.get("rag_result") or {}
    preserved_rag = (result.get("metadata") or {}).get("evidence_rag_result") or {}
    rag_documents = []
    if answer_source not in {"faq", "web_search", "direct_response", "multi_hop"}:
        if current_rag.get("documents") and not current_rag.get("is_no_answer", False):
            rag_documents = current_rag["documents"]
        elif preserved_rag.get("documents"):
            rag_documents = preserved_rag["documents"]
        else:
            rag_documents = current_rag.get("documents") or []

    for document in rag_documents:
        metadata = _document_metadata(document)
        content = " ".join(_document_content(document).split())
        ranks = []
        for key, label in (
            ("vector_rank", "向量"),
            ("bm25_rank", "BM25"),
            ("visual_rank", "视觉"),
        ):
            if metadata.get(key) is not None:
                ranks.append(f"{label} #{metadata[key]}")
        add(
            {
                "source_type": "rag",
                "title": metadata.get("topic_title") or "知识库证据",
                "pages": metadata.get("pages") or metadata.get("page_id"),
                "doc_type": metadata.get("doc_type") or "text",
                "excerpt": content[:360] + ("..." if len(content) > 360 else ""),
                "ranks": ranks,
                "chunk_id": metadata.get("chunk_id"),
            }
        )

    if not citations and answer_source not in {"faq", "web_search", "direct_response", "multi_hop"}:
        rag_result = preserved_rag or current_rag
        context = " ".join(str(rag_result.get("context") or "").split())
        if context:
            add(
                {
                    "source_type": "rag",
                    "title": "RAG 检索上下文",
                    "doc_type": "context",
                    "excerpt": context[:600] + ("..." if len(context) > 600 else ""),
                }
            )

    faq_result = result.get("faq_result") or {}
    if faq_result and (answer_source == "faq" or not citations):
        confidence = faq_result.get("confidence")
        add(
            {
                "source_type": "faq",
                "title": faq_result.get("matched_question") or "FAQ 标准问答",
                "doc_type": "faq",
                "excerpt": " ".join(str(faq_result.get("answer") or "").split())[:360],
                "ranks": [f"置信度 {float(confidence):.2f}"]
                if isinstance(confidence, (int, float))
                else [],
                "chunk_id": faq_result.get("source"),
            }
        )

    web_result = result.get("web_search_result") or {}
    if web_result.get("results") and (answer_source == "web_search" or not citations):
        for item in web_result["results"]:
            content = " ".join(str(item.get("content") or "").split())
            add(
                {
                    "source_type": "web",
                    "title": item.get("title") or item.get("url") or "联网搜索结果",
                    "doc_type": "web_search",
                    "excerpt": content[:360] + ("..." if len(content) > 360 else ""),
                    "url": item.get("url") or "",
                }
            )
    return citations


def _render_citations_html(
    citations: List[Dict[str, Any]],
    empty_message: str = "本轮回答没有可展示的来源引用。",
) -> str:
    """Render citation cards directly; no Markdown or Document objects reach Gradio."""
    if not citations:
        return (
            '<div class="citation-output citation-empty">'
            f"<p>{html.escape(empty_message)}</p>"
            "</div>"
        )

    cards = []
    for index, citation in enumerate(citations, start=1):
        source_type = str(citation.get("source_type") or "source")
        title = html.escape(str(citation.get("title") or "未命名来源"))
        excerpt = html.escape(str(citation.get("excerpt") or "无可展示摘要"))
        pages = citation.get("pages")
        page_text = f"页码 {pages}" if pages not in (None, "", []) else ""
        meta_parts = [
            str(citation.get("doc_type") or source_type),
            page_text,
            *[str(item) for item in citation.get("ranks") or []],
        ]
        meta = " · ".join(html.escape(item) for item in meta_parts if item)
        url = str(citation.get("url") or "")
        title_html = (
            f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{title}</a>'
            if url
            else title
        )
        cards.append(
            f'<article class="citation-card citation-{html.escape(source_type)}">'
            f'<div class="citation-card-top"><span class="citation-index">{index}</span>'
            f'<span class="citation-type">{html.escape(source_type.upper())}</span></div>'
            f'<h4>{title_html}</h4><div class="citation-meta">{meta}</div>'
            f'<p>{excerpt}</p></article>'
        )
    return '<div class="citation-output">' + "".join(cards) + "</div>"


def _format_citations_text(
    citations: List[Dict[str, Any]],
    empty_message: str = "本轮回答没有可展示的来源引用。",
) -> str:
    """Format citations for the native Gradio Textbox evidence panel."""
    if not citations:
        return empty_message

    sections = []
    for index, citation in enumerate(citations, start=1):
        source_type = str(citation.get("source_type") or "source").upper()
        title = str(citation.get("title") or "未命名来源")
        pages = citation.get("pages")
        meta = [str(citation.get("doc_type") or source_type)]
        if pages not in (None, "", []):
            meta.append(f"页码 {pages}")
        meta.extend(str(item) for item in citation.get("ranks") or [])
        url = str(citation.get("url") or "").strip()
        excerpt = str(citation.get("excerpt") or "无可展示摘要").strip()
        lines = [
            f"[{index}] {source_type} · {title}",
            " · ".join(meta),
        ]
        if url:
            lines.append(url)
        lines.append(excerpt)
        sections.append("\n".join(lines))
    return "\n\n" + ("\n\n" + "-" * 48 + "\n\n").join(sections)


def _build_citation_text(result: Dict[str, Any], max_citations: int = 6) -> str:
    return _format_citations_text(_collect_citations(result, max_citations=max_citations))


def _build_citation_panel(result: Dict[str, Any], max_citations: int = 6) -> str:
    return _render_citations_html(_collect_citations(result, max_citations=max_citations))


def _browser_image_path(path: Path) -> Path | None:
    """Return an existing browser-supported image, converting formats such as JPX."""
    resolved = path.resolve()
    if not resolved.exists():
        return None
    if resolved.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        return resolved

    try:
        from PIL import Image

        cache_dir = Path(".gradio") / "evidence_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_name = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:20] + ".png"
        cached = (cache_dir / cache_name).resolve()
        if not cached.exists() or cached.stat().st_mtime < resolved.stat().st_mtime:
            with Image.open(resolved) as image:
                image.convert("RGB").save(cached, format="PNG")
        return cached
    except Exception as exc:
        print(f"  [Warning] 无法转换图片用于前端展示: {resolved} ({exc})")
        return None


def _initialize_multimodal_agent(knowledge_base_name: str):
    """Build the original Agent graph with a multimodal RAG tool."""
    global plain_rag_llm, plain_rag_reranker

    store_path = knowledge_base_paths(knowledge_base_name)["store"]
    print(f"Loading multimodal knowledge base: {knowledge_base_name}")
    retriever = MultimodalRetriever(store_path)
    llm = get_llm_model(model_path="./pre_train_model/Qwen-7B-Chat", backend=LLM_BACKEND)
    rerank = reRankLLM()
    plain_rag_llm = llm
    plain_rag_reranker = rerank
    rag_tool = MultimodalRAGTool(
        multimodal_retriever=retriever,
        rerank_model=rerank,
        llm=llm,
    )
    faq_tool = FAQTool(faq_path="./data/faq_database.json")
    web_search_tool = FallbackSearchTool(
        BochaSearchMCPTool(max_results=3),
        DDGSSearchTool(max_results=3),
    )
    set_tools(rag_tool, faq_tool, web_search_tool)
    return build_agent_graph()


def _messages_to_agent_history(history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    conversation_history = []
    for msg in history or []:
        role = msg.get("role")
        content = msg.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content:
            conversation_history.append({"role": role, "content": content})
    return conversation_history


def _extract_retrieved_images(result: Dict[str, Any], max_images: int = 6) -> List[Tuple[str, str]]:
    """Return existing browser-friendly images from the final RAG evidence."""
    documents = _rag_documents(result)
    images: List[Tuple[str, str]] = []
    seen_paths = set()
    for document in documents:
        metadata = _document_metadata(document)
        image_path = metadata.get("image_path")
        if not image_path:
            continue
        path = Path(image_path)
        resolved_path = _browser_image_path(path)
        if resolved_path is None or resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)
        topic = metadata.get("topic_title") or "图片证据"
        pages = metadata.get("pages") or metadata.get("page_id") or "未知"
        images.append((str(resolved_path), f"{topic} · 页码 {pages}"))
        if len(images) >= max_images:
            break
    return images


def _format_rag_citations(result: Dict[str, Any], max_citations: int = 6) -> str:
    """Format the final evidence used to generate the answer."""
    documents = _rag_documents(result)
    if documents:
        return _format_document_citations(documents, max_citations=max_citations)

    rag_result = result.get("rag_result") or (result.get("metadata") or {}).get("evidence_rag_result") or {}
    if rag_result.get("context"):
        return _format_rag_context_citation(rag_result["context"])

    faq_result = result.get("faq_result")
    if faq_result:
        return _format_faq_citation(faq_result)

    web_search_result = result.get("web_search_result")
    if web_search_result and web_search_result.get("results"):
        return _format_web_search_citations(web_search_result, max_citations=max_citations)

    return "本轮回答没有使用知识库引用。"


def _render_citation_panel(citations: str) -> str:
    text = str(citations or "本轮回答没有使用知识库引用。").strip()
    if not text:
        text = "本轮回答没有使用知识库引用。"
    return (
        '<div class="citation-output">'
        f"<pre>{html.escape(text)}</pre>"
        "</div>"
    )


def _format_rag_context_citation(context: str, max_chars: int = 2400) -> str:
    """Fallback citation display when LangGraph preserved RAG context but not Document objects."""
    normalized = str(context).strip()
    if not normalized:
        return "本轮回答没有使用知识库引用。"
    excerpt = normalized[:max_chars] + ("..." if len(normalized) > max_chars else "")
    return "### RAG 检索上下文\n\n" + excerpt


def _format_document_citations(documents: List[Any], max_citations: int = 6) -> str:
    lines = []
    for index, document in enumerate(documents[:max_citations], start=1):
        metadata = _document_metadata(document)
        topic = metadata.get("topic_title") or "未命名主题"
        pages = metadata.get("pages") or metadata.get("page_id") or "未知"
        doc_type = metadata.get("doc_type") or "text"
        content = " ".join(_document_content(document).split())
        excerpt = content[:260] + ("..." if len(content) > 260 else "")

        ranks = []
        if metadata.get("vector_rank") is not None:
            ranks.append(f"向量 #{metadata['vector_rank']}")
        if metadata.get("bm25_rank") is not None:
            ranks.append(f"BM25 #{metadata['bm25_rank']}")
        # Codex-MM-ADD: 前端来源引用展示视觉召回排名，便于判断图片是否
        # 来自真正的 CLIP 视觉向量检索，而不是仅由附近文本召回。
        if metadata.get("visual_rank") is not None:
            ranks.append(f"视觉 #{metadata['visual_rank']}")
        if metadata.get("fused_score") is not None:
            ranks.append(f"RRF {metadata['fused_score']:.4f}")
        rank_text = " · ".join(ranks) or "原项目检索结果"
        if metadata.get("model_name"):
            source_line = (f"`{doc_type}` · 车型 `{metadata['model_name']}` · "
                           f"手册 `{metadata.get('doc_id', '未知')}` · PDF 物理页 `{pages}`")
        else:
            source_line = f"`{doc_type}` · 页码 `{pages}`"

        lines.extend(
            [
                f"### [{index}] {topic}",
                source_line,
                excerpt or "无文本摘录",
                f"<small>{rank_text}</small>",
                "",
            ]
        )
    return "\n".join(lines)


def _format_faq_citation(faq_result: Dict[str, Any]) -> str:
    source = faq_result.get("source") or "FAQ"
    matched_question = faq_result.get("matched_question") or "标准问答"
    confidence = faq_result.get("confidence")
    answer = " ".join(str(faq_result.get("answer", "")).split())
    excerpt = answer[:260] + ("..." if len(answer) > 260 else "")
    confidence_text = f"{float(confidence):.2f}" if isinstance(confidence, (int, float)) else "未知"
    return "\n".join(
        [
            f"### [FAQ] {matched_question}",
            f"`faq` · 来源 `{source}`",
            excerpt or "无文本摘录",
            f"<small>FAQ 匹配置信度 {confidence_text}</small>",
        ]
    )


def _format_web_search_citations(web_search_result: Dict[str, Any], max_citations: int = 6) -> str:
    lines = []
    for index, item in enumerate(web_search_result.get("results", [])[:max_citations], start=1):
        title = item.get("title") or item.get("url") or "联网搜索结果"
        url = item.get("url") or ""
        content = " ".join(str(item.get("content", "")).split())
        excerpt = content[:260] + ("..." if len(content) > 260 else "")
        lines.extend(
            [
                f"### [{index}] {title}",
                f"`web_search` · {url}" if url else "`web_search`",
                excerpt or "无文本摘录",
                "",
            ]
        )
    return "\n".join(lines) or "本轮回答没有使用知识库引用。"


def initialize_system():
    """Initialize the RAG Agent once."""
    global agent_graph, plain_rag_tool, plain_rag_source_pdf, plain_rag_llm, plain_rag_reranker

    knowledge_base_name = os.environ.get("KNOWLEDGE_BASE_NAME", "").strip() or get_active_knowledge_base()
    if agent_graph is None and knowledge_base_name:
        os.environ["KNOWLEDGE_BASE_NAME"] = knowledge_base_name
        plain_rag_tool = None
        plain_rag_source_pdf = knowledge_base_paths(knowledge_base_name)["source"]
        agent_graph = _initialize_multimodal_agent(knowledge_base_name)
        return f"多模态知识库 `{knowledge_base_name}` 已接入 Agent"

    if agent_graph is not None:
        return "系统已初始化"

    print("=" * 70)
    print("初始化 RAG Agent 系统...")
    print("=" * 70)

    base = "."

    print("\n[1/6] 解析 PDF 文档...")
    dp = DataProcess(pdf_path=base + "/data/train_a.pdf")
    dp.ParseBlock(max_seq=1024)
    dp.ParseBlock(max_seq=512)
    dp.ParseAllPage(max_seq=256)
    dp.ParseAllPage(max_seq=512)
    dp.ParseOnePageWithRule(max_seq=256)
    dp.ParseOnePageWithRule(max_seq=512)
    data = dp.data
    print(f"  parsed chunks: {len(data)}")

    print("\n[2/6] 初始化 FAISS 检索器...")
    faiss_retriever = FaissRetriever(data=data)

    print("\n[3/6] 初始化 BM25 检索器...")
    bm25_retriever = BM25(data)

    print(f"\n[4/6] 初始化 LLM ({LLM_BACKEND})...")
    qwen7 = base + "/pre_train_model/Qwen-7B-Chat"
    llm = get_llm_model(model_path=qwen7, backend=LLM_BACKEND)

    print("\n[5/6] 初始化 Rerank 模型...")
    rerank = reRankLLM()

    print("\n[6/6] 初始化 Agent 工具和工作流...")
    rag_tool = RAGTool(
        faiss_retriever=faiss_retriever,
        bm25_retriever=bm25_retriever,
        rerank_model=rerank,
        llm=llm,
    )
    # The legacy startup path already constructs the pre-Agentic RAG stack.
    # Reuse it for both modes instead of building a duplicate FAISS index.
    plain_rag_tool = rag_tool
    plain_rag_source_pdf = Path(base) / "data" / "train_a.pdf"
    plain_rag_llm = llm
    plain_rag_reranker = rerank
    faq_tool = FAQTool(faq_path=base + "/data/faq_database.json")
    web_search_tool = FallbackSearchTool(
        BochaSearchMCPTool(max_results=3),
        DDGSSearchTool(max_results=3),
    )

    set_tools(rag_tool, faq_tool, web_search_tool)
    agent_graph = build_agent_graph()

    print("\n" + "=" * 70)
    print("系统初始化完成")
    print("=" * 70)
    return "系统初始化成功，可以开始对话。"


def _get_plain_rag_tool():
    """Lazily build the original one-shot RAG chain for the active PDF."""
    global plain_rag_tool, plain_rag_source_pdf, plain_rag_llm, plain_rag_reranker

    if plain_rag_tool is not None:
        return plain_rag_tool

    with plain_rag_lock:
        if plain_rag_tool is not None:
            return plain_rag_tool

        source_pdf = plain_rag_source_pdf or (Path("data") / "train_a.pdf")
        if not source_pdf.exists():
            raise FileNotFoundError(f"普通 RAG 的知识库 PDF 不存在：{source_pdf}")

        print(f"初始化普通 RAG：{source_pdf}")
        data_process = DataProcess(pdf_path=str(source_pdf))
        data_process.ParseBlock(max_seq=1024)
        data_process.ParseBlock(max_seq=512)
        data_process.ParseAllPage(max_seq=256)
        data_process.ParseAllPage(max_seq=512)
        data_process.ParseOnePageWithRule(max_seq=256)
        data_process.ParseOnePageWithRule(max_seq=512)
        data = data_process.data

        if plain_rag_llm is None:
            plain_rag_llm = get_llm_model(
                model_path="./pre_train_model/Qwen-7B-Chat", backend=LLM_BACKEND
            )
        if plain_rag_reranker is None:
            plain_rag_reranker = reRankLLM()

        plain_rag_tool = RAGTool(
            faiss_retriever=FaissRetriever(data=data),
            bm25_retriever=BM25(data),
            rerank_model=plain_rag_reranker,
            llm=plain_rag_llm,
        )
        print(f"普通 RAG 已就绪：{len(data)} 个原始文本块")
        return plain_rag_tool


def _get_multi_manual_tool():
    """Hot-swap the tool when the active index pointer changes."""
    global multi_manual_tool, plain_rag_llm, agent_graph
    pointer = MULTI_MANUAL_STORE / "active.json"
    if not pointer.is_file():
        raise RuntimeError("多车型索引尚未构建；请先上传并构建手册")
    active = json.loads(pointer.read_text(encoding="utf-8"))
    version_id = active.get("version_id") or active["batch_id"]
    if multi_manual_tool is not None and multi_manual_tool.retriever.version.name == version_id:
        return multi_manual_tool
    with multi_manual_lock:
        active = json.loads(pointer.read_text(encoding="utf-8"))
        version_id = active.get("version_id") or active["batch_id"]
        if multi_manual_tool is not None and multi_manual_tool.retriever.version.name == version_id:
            return multi_manual_tool
        if plain_rag_llm is None:
            plain_rag_llm = get_llm_model(model_path="./pre_train_model/Qwen-7B-Chat", backend=LLM_BACKEND)
        cache_key = (str(MULTI_MANUAL_STORE.resolve()), version_id)
        next_tool = _multi_manual_tool_cache.get(cache_key)
        if next_tool is None:
            retriever = MultiManualRetriever(MULTI_MANUAL_STORE)
            next_tool = MultiManualRAGTool(retriever, plain_rag_llm)
            # The old tool remains alive while an in-flight request holds it;
            # close its Qdrant client only after the last reference disappears.
            weakref.finalize(next_tool, retriever.close)
            _multi_manual_tool_cache[cache_key] = next_tool
        if agent_graph is None:
            set_tools(None, None, None)
            agent_graph = build_agent_graph()
        multi_manual_tool = next_tool
        set_multi_manual_tool(multi_manual_tool)
        return multi_manual_tool


def _refresh_manual_choices(selected: list[str] | None = None):
    choices = list_active_manual_models(MULTI_MANUAL_STORE)
    return gr.update(choices=choices, value=[name for name in (selected or []) if name in choices])


def _refresh_index_history():
    versions = list_index_versions(MULTI_MANUAL_STORE)
    choices = [(f"{'当前' if item['active'] else '历史'} | {item['built_at']} | "
                f"{len(item['manuals'])} 款车型 | {item['version_id']}", item["version_id"])
               for item in versions]
    return gr.update(choices=choices, value=None)


def _upload_multi_manual(model_name: str, pdf_path: str | None,
                         manual_version: str | None, model_year: str | None):
    if not pdf_path:
        return "请上传车型手册 PDF。", _refresh_manual_choices(), _refresh_index_history()
    try:
        result = upload_and_activate_manual(
            model_name, pdf_path, manual_version=manual_version, model_year=model_year,
            store_root=MULTI_MANUAL_STORE)
        # New requests reload the active Qdrant version on demand. Do not close
        # the old retriever here: a concurrent request may still be using it.
        parsed, index = result["parsed"], result["index"]
        action = "已更新并激活" if result["updated"] else "已新增并激活"
        return (f"{action}：{model_name.strip()}（手册版本：{parsed['manual_version']}）；"
                f"本次解析 {parsed['parent_count']} 个父块、"
                f"{parsed['child_count']} 个子块、{parsed['image_count']} 张图片。"
                f"活动索引共 {len(index['manuals'])} 款车型。"), _refresh_manual_choices(), _refresh_index_history()
    except Exception as exc:
        return (f"上传/构建失败，活动索引未切换：{exc}",
                _refresh_manual_choices(), _refresh_index_history())


def _rollback_multi_manual(version_id: str | None):
    if not version_id:
        return "请先选择一个历史索引版本。", _refresh_manual_choices(), _refresh_index_history()
    try:
        manifest = rollback_to_index_version(version_id, store_root=MULTI_MANUAL_STORE)
        return (f"已切换到索引 {manifest['version_id']}；包含 {len(manifest['manuals'])} 款车型。",
                _refresh_manual_choices(), _refresh_index_history())
    except Exception as exc:
        return (f"切换失败，活动索引未改变：{exc}",
                _refresh_manual_choices(), _refresh_index_history())


def _check_selected_manuals(message: str, model_names: list[str]) -> list[str]:
    """Use only the UI selection as the multi-manual retrieval scope."""
    del message  # Question-text vehicle extraction is intentionally deferred.
    selected = list(dict.fromkeys(model_names or []))
    known = set(list_active_manual_models(MULTI_MANUAL_STORE))
    if not known:
        raise ValueError("多车型手册索引尚未构建，当前没有可选车型")
    if not selected:
        raise ValueError("请先在下拉框中选择至少一款车型；对比时选择两款")
    if any(name not in known for name in selected):
        raise ValueError("下拉框中包含不属于当前活动索引的车型，请重新打开页面后选择")
    return selected


def _chat_with_agent_text(
    message: str,
    history: list,
    session_id: str = "default",
    manual_models: list[str] | None = None,
) -> Generator[Tuple[str, str, List[Tuple[str, str]], str], None, None]:
    """Run one user message through the Agent."""
    retrieved_images: List[Tuple[str, str]] = []
    citation_text = "正在检索来源..."
    latest_retrieved_images_by_session[session_id] = retrieved_images
    latest_citations_by_session[session_id] = citation_text
    if manual_models:
        _get_multi_manual_tool()
    if agent_graph is None:
        yield ("系统尚未初始化，请稍候...", "System not ready", retrieved_images, citation_text)
        initialize_system()
        yield ("系统初始化完成，请重新发送您的问题。", "System ready", retrieved_images, citation_text)
        return

    memory_context = short_term_memory.build_context(session_id)
    redis_history = memory_context["messages"]
    ui_history = _messages_to_agent_history(history)
    conversation_history = redis_history or ui_history

    initial_state: AgentState = {
        "user_question": message,
        "conversation_history": conversation_history,
        "intent": "",
        "query_rewrite": "",
        "plan": [],
        "rag_result": {},
        "faq_result": None,
        "web_search_result": None,
        "draft_answer": "",
        "claims": [],
        "evidence_catalog": [],
        "claim_verdicts": [],
        "citation_coverage": 0.0,
        "grounding_status": "",
        "grounding_retry_count": 0,
        "execution_cache": {},
        "execution_budget": {},
        "metrics": {},
        "execution_trace": [],
        "reflection_count": 0,
        "quality_score": 0.0,
        "reflection_notes": [],
        "improvement_actions": [],
        "final_answer": "",
        "answer_source": "",
        "confidence": 0.0,
        "need_human": False,
        "metadata": {
            "session_id": session_id,
            "manual_models": manual_models or [],
            "short_term_summary": memory_context["summary"],
            "short_term_memory_tokens": memory_context["memory_tokens"],
            "short_term_redis_enabled": memory_context["redis_enabled"],
        },
    }

    yield ("正在处理您的问题...", "Processing...", retrieved_images, citation_text)

    try:
        start_time = time.time()
        result = None
        current_answer = ""
        for kind, payload in _run_streaming_call(
            lambda: agent_graph.invoke(initial_state, config={"recursion_limit": 80}),
            stream_state=initial_state,
        ):
            if kind == "token":
                current_answer += payload
                yield (current_answer, "Agent 正在生成并校验答案...", retrieved_images, citation_text)
            elif kind == "reset":
                current_answer = ""
                yield (current_answer, "答案需要修订，正在重新生成...", retrieved_images, citation_text)
            elif kind == "result":
                result = payload
        if result is None:
            raise RuntimeError("Agent 工作流未返回结果")
        retrieved_images = _extract_retrieved_images(result)
        citation_text = _build_citation_text(result)
        latest_retrieved_images_by_session[session_id] = retrieved_images
        latest_citations_by_session[session_id] = citation_text
        evidence_count = len(_rag_documents(result))
        image_count = len(retrieved_images)
        elapsed_time = time.time() - start_time

        final_answer = result["final_answer"]
        answer_source = result.get("answer_source", "unknown")
        confidence = result.get("confidence", 0.0)
        quality_score = result.get("quality_score", 0.0)
        reflection_count = result.get("reflection_count", 0)
        citation_coverage = result.get("citation_coverage", 0.0)
        grounding_status = result.get("grounding_status", "unknown")
        execution_budget = result.get("execution_budget", {})
        runtime_metrics = result.get("metrics", {})
        intent = result.get("intent", "unknown")

        short_term_memory.append_exchange(session_id, message, final_answer)

        memory_backend = "Redis" if memory_context["redis_enabled"] else "fallback"
        status_info = (
            f"处理完成\n"
            f"- 耗时: {elapsed_time:.2f}s\n"
            f"- 意图: {intent}\n"
            f"- 来源: {answer_source}\n"
            f"- 置信度: {confidence:.2f}\n"
            f"- 质量分: {quality_score:.1f}/10\n"
            f"- Reflection 次数: {reflection_count}\n"
            f"- 事实校验: {grounding_status}, 引用覆盖率 {citation_coverage:.0%}\n"
            f"- 工具调用: {execution_budget.get('tool_calls', 0)} 次 "
            f"(RAG {execution_budget.get('rag_calls', 0)}, 搜索 {execution_budget.get('web_search_calls', 0)})\n"
            f"- 结果复用: {execution_budget.get('cache_hits', 0)} 次, "
            f"重复拦截 {execution_budget.get('deduplicated_calls', 0)} 次\n"
            f"- 模型调用: {runtime_metrics.get('llm_calls', 0)} 次, "
            f"模型耗时 {runtime_metrics.get('llm_duration_ms', 0.0) / 1000:.2f}s\n"
            f"- Token: 输入 {runtime_metrics.get('input_tokens', 0)}, "
            f"输出 {runtime_metrics.get('output_tokens', 0)}\n"
            f"- 来源引用: {evidence_count} 条 RAG 证据, {image_count} 张图片\n"
            f"- Short memory: {memory_backend}, {memory_context['memory_tokens']} est. tokens"
        )
        trace_text = _format_execution_trace(result.get("execution_trace", []))
        if trace_text:
            status_info += "\n\n执行轨迹\n" + trace_text
        if result.get("need_human", False):
            status_info += "\n- 建议转人工客服"

        yield (final_answer, status_info, retrieved_images, citation_text)

    except Exception as e:
        import traceback

        traceback.print_exc()
        error_msg = f"处理出错: {e}"
        yield (f"抱歉，处理您的问题时出现错误：{e}", error_msg, retrieved_images, citation_text)


def chat_with_agent(
    message: str,
    history: list,
    session_id: str = "default",
    manual_models: list[str] | None = None,
) -> Generator[Tuple[str, str, List[Tuple[str, str]], str], None, None]:
    """Stream the answer and evidence produced by this exact Agent invocation."""
    if manual_models and MultimodalRAGTool._is_visual_request(message):
        # The image lane is intentionally a plain RAG path even when the
        # Agentic toggle is selected for text questions.
        for answer, status, images, citations in _chat_with_multi_manual_plain(
            message, session_id, manual_models
        ):
            yield answer, "图片链路: 普通 RAG（不经过 Agentic 编排）\n" + status, images, citations
        return
    for answer, status, images, citations in _chat_with_agent_text(message, history, session_id, manual_models):
        yield answer, "运行模式: Agentic RAG\n" + status, images, citations


def _chat_with_multi_manual_plain(message: str, session_id: str, manual_models: list[str]):
    images: List[Tuple[str, str]] = []
    yield "正在按车型检索手册...", "运行模式: 多车型普通 RAG\nProcessing...", images, ""
    try:
        started = time.time()
        result = None
        current_answer = ""
        for kind, payload in _run_streaming_call(
            lambda: _get_multi_manual_tool().search_and_generate(
                message, model_names=manual_models, rewrite_query=True,
            )
        ):
            if kind == "token":
                current_answer += payload
                yield current_answer, "运行模式: 多车型普通 RAG\n正在生成答案...", images, ""
            elif kind == "reset":
                current_answer = ""
                yield current_answer, "运行模式: 多车型普通 RAG\n正在准备答案...", images, ""
            elif kind == "result":
                result = payload
        if result is None:
            raise RuntimeError("多车型 RAG 未返回结果")
        answer = str(result["answer"])
        images = _extract_retrieved_images({"rag_result": result})
        citations = _format_rag_citations({"answer_source": "rag", "rag_result": result})
        latest_retrieved_images_by_session[session_id] = images
        latest_citations_by_session[session_id] = citations
        short_term_memory.append_exchange(session_id, message, answer)
        status = ("运行模式: 多车型普通 RAG\n"
                  f"车型: {', '.join(manual_models)}\n"
                  f"检索问题: {result.get('retrieval_query', message)}\n"
                  f"检索证据: {len(result['documents'])} 条\n"
                  f"耗时: {time.time() - started:.2f}s")
        yield answer, status, images, citations
    except Exception as exc:
        yield f"多车型检索失败：{exc}", "运行模式: 多车型普通 RAG\nError", images, ""


def chat_with_plain_rag(
    message: str,
    history: list,
    session_id: str = "default",
    manual_models: list[str] | None = None,
) -> Generator[Tuple[str, str, List[Tuple[str, str]], str], None, None]:
    """Run the original FAISS + BM25 + rerank + single-generation RAG chain."""
    del history  # The original RAG chain is single-turn and has no Agent memory.
    if manual_models:
        yield from _chat_with_multi_manual_plain(message, session_id, manual_models)
        return
    retrieved_images: List[Tuple[str, str]] = []
    latest_retrieved_images_by_session[session_id] = retrieved_images
    latest_citations_by_session[session_id] = "正在检索来源..."

    if agent_graph is None:
        yield ("系统尚未初始化，请稍候...", "System not ready", retrieved_images, "")
        initialize_system()
        yield ("系统初始化完成，请重新发送您的问题。", "System ready", retrieved_images, "")
        return

    yield ("正在使用普通 RAG 检索并生成答案...", "运行模式: 普通 RAG\nProcessing...", retrieved_images, "")
    try:
        start_time = time.time()
        result = None
        current_answer = ""
        for kind, payload in _run_streaming_call(
            lambda: _get_plain_rag_tool().search_and_generate(message, top_k=6)
        ):
            if kind == "token":
                current_answer += payload
                yield current_answer, "运行模式: 普通 RAG\n正在生成答案...", retrieved_images, ""
            elif kind == "reset":
                current_answer = ""
                yield current_answer, "运行模式: 普通 RAG\n正在准备答案...", retrieved_images, ""
            elif kind == "result":
                result = payload
        if result is None:
            raise RuntimeError("普通 RAG 未返回结果")
        final_answer = str(result.get("answer") or "抱歉，当前知识库中没有找到可靠答案。")
        citation_text = _format_rag_citations({"answer_source": "rag", "rag_result": result})
        latest_citations_by_session[session_id] = citation_text
        elapsed_time = time.time() - start_time
        documents = result.get("documents") or []
        short_term_memory.append_exchange(session_id, message, final_answer)
        status = (
            "运行模式: 普通 RAG\n"
            f"处理完成\n- 耗时: {elapsed_time:.2f}s\n"
            f"- 来源: {result.get('source', 'rag')}\n"
            f"- 置信度: {float(result.get('confidence') or 0.0):.2f}\n"
            f"- 检索证据: {len(documents)} 条\n"
            f"- 检索策略: FAISS + BM25 + Rerank + 单次答案生成"
        )
        yield final_answer, status, retrieved_images, citation_text
    except Exception as exc:
        import traceback

        traceback.print_exc()
        yield (
            f"抱歉，普通 RAG 处理问题时出现错误：{exc}",
            f"运行模式: 普通 RAG\n处理出错: {exc}",
            retrieved_images,
            latest_citations_by_session.get(session_id, ""),
        )


def _format_knowledge_base_list() -> Tuple[str, List[str]]:
    items = list_knowledge_bases()
    if not items:
        return "暂无已构建的知识库。", []
    lines = ["| 知识库 | 文档块 | 文本 | 图片 | 表格 | 嵌入模型 |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    names = []
    for item in items:
        name = item.get("name", "unknown")
        manifest = item.get("manifest", {})
        counts = item.get("doc_type_counts") or manifest.get("doc_type_counts", {})
        names.append(name)
        lines.append(
            f"| {name} | {item.get('chunk_count', manifest.get('document_count', 0))} "
            f"| {counts.get('text', 0)} | {counts.get('image', 0)} | {counts.get('table', 0)} "
            f"| {manifest.get('embedding', '-')} |"
        )
    return "\n".join(lines), names


def refresh_knowledge_bases():
    markdown, names = _format_knowledge_base_list()
    return markdown, gr.update(choices=names, value=names[0] if names else None)


def build_uploaded_knowledge_base(
    uploaded_pdf,
    kb_name: str,
    toc_pages: str,
    child_size: int,
    overlap: int,
    parent_size: int,
    batch_size: int,
    progress=gr.Progress(),
):
    if not uploaded_pdf:
        return "请先上传 PDF 文件。", "", gr.update(), None
    progress(0.02, desc="准备构建知识库")
    try:
        result = build_knowledge_base(
            uploaded_pdf=uploaded_pdf,
            name=kb_name,
            toc_pages=toc_pages,
            child_chunk_size=int(child_size),
            child_overlap=int(overlap),
            parent_chunk_size=int(parent_size),
            batch_size=int(batch_size),
            status_callback=lambda message: progress(None, desc=message),
        )
        progress(1.0, desc="知识库构建完成")
        counts = result["doc_type_counts"]
        status = (
            f"构建完成：{result['name']}\n"
            f"总证据数：{result['chunk_count']}\n"
            f"文本：{counts.get('text', 0)}，图片：{counts.get('image', 0)}，表格：{counts.get('table', 0)}\n"
            f"索引目录：{result['store']}"
        )
        markdown, names = _format_knowledge_base_list()
        return status, markdown, gr.update(choices=names, value=result["name"]), result["preview"]
    except Exception as exc:
        import traceback

        traceback.print_exc()
        return f"构建失败：{exc}", _format_knowledge_base_list()[0], gr.update(), None


def search_selected_knowledge_base(kb_name: str, query: str, top_k: int):
    if not kb_name:
        return "请先构建或选择知识库。"
    if not query or not query.strip():
        return "请输入测试问题。"
    try:
        results = search_knowledge_base(kb_name, query.strip(), int(top_k))
    except Exception as exc:
        return f"检索失败：{exc}"
    if not results:
        return "没有检索到结果。"

    lines = []
    for index, result in enumerate(results, start=1):
        metadata = result["metadata"]
        lines.extend(
            [
                f"### {index}. {metadata.get('topic_title') or '未命名主题'}",
                f"`{metadata.get('doc_type')}/{metadata.get('chunk_type')}` "
                f"· 页码 `{metadata.get('pages') or metadata.get('page_id')}` "
                f"· 融合分数 `{result['fused_score']}`",
                result["content"][:700],
            ]
        )
        if metadata.get("image_path"):
            lines.append(f"图片路径：`{metadata['image_path']}`")
    return "\n\n".join(lines)


def activate_selected_knowledge_base(kb_name: str):
    """Switch the Agent's RAG tool to a built multimodal knowledge base."""
    global agent_graph

    if not kb_name:
        return "请先选择知识库。"
    kb_name = set_active_knowledge_base(kb_name)
    os.environ["KNOWLEDGE_BASE_NAME"] = kb_name
    agent_graph = None
    return initialize_system()


def _legacy_create_gradio_interface():
    """Create a compact customer-service console."""

    css = """
    :root {
        --app-border: #d7dde8;
        --app-panel: #f7f9fc;
        --app-text: #1f2937;
    }
    .gradio-container {
        color: var(--app-text);
        max-width: 1240px !important;
        margin: 0 auto !important;
    }
    #sidebar {
        background: var(--app-panel);
        border: 1px solid var(--app-border);
        border-radius: 8px;
        padding: 14px;
    }
    #chatbot {
        min-height: 520px;
        border: 1px solid var(--app-border);
        border-radius: 8px;
    }
    #status-panel textarea {
        font-size: 13px;
        line-height: 1.45;
    }
    #answer-workspace {
        display: grid !important;
        grid-template-columns: minmax(0, 3fr) minmax(280px, 1.25fr);
        gap: 12px;
        align-items: start;
    }
    #answer-workspace > #chatbot {
        grid-column: 1;
        grid-row: 1;
    }
    #answer-workspace > #citation-panel {
        grid-column: 2;
        grid-row: 1;
        max-height: 520px;
        overflow-y: auto;
    }
    #answer-workspace > :not(#chatbot):not(#citation-panel) {
        grid-column: 1 / -1;
    }
    @media (max-width: 900px) {
        #answer-workspace {
            display: flex !important;
        }
    }
    .example-row button {
        min-width: 0 !important;
        white-space: normal !important;
        text-align: left;
    }
    footer {
        display: none !important;
    }
    """

    active_knowledge_base = os.environ.get("KNOWLEDGE_BASE_NAME", "").strip() or get_active_knowledge_base()
    initial_status = (
        f"System: ready\n"
        f"Backend: {LLM_BACKEND}\n"
        "Memory: Redis when available, fallback otherwise\n"
        f"Knowledge base: {active_knowledge_base or 'legacy data/train_a.pdf'}"
    )

    def normalize_session_id(raw_session_id: str) -> str:
        return (raw_session_id or "default").strip() or "default"

    def submit_message(message: str, chat_history: list, session_id: str):
        if not message or not message.strip():
            yield chat_history or [], "", "请输入问题。"
            return

        session_id = normalize_session_id(session_id)
        chat_history = list(chat_history or [])
        chat_history.append({"role": "user", "content": message.strip()})
        chat_history.append({"role": "assistant", "content": "正在思考..."})

        final_status = initial_status
        for partial_answer, status, retrieved_images, citations in chat_with_agent(
            message.strip(), chat_history[:-1], session_id=session_id
        ):
            chat_history[-1] = {"role": "assistant", "content": partial_answer}
            final_status = status
            yield chat_history, "", final_status, retrieved_images, citations

    def clear_current_session(session_id: str):
        session_id = normalize_session_id(session_id)
        short_term_memory.clear(session_id)
        status = f"Session `{session_id}` 已清空。\nBackend: {LLM_BACKEND}"
        latest_retrieved_images_by_session.pop(session_id, None)
        latest_citations_by_session.pop(session_id, None)
        return [], status, [], "本轮回答没有使用知识库引用。"

    def fill_example(example: str):
        return example

    def submit_message_for_ui(message: str, chat_history: list, session_id: str):
        for output in submit_message(message, chat_history, session_id):
            if len(output) == 3:
                yield output[0], output[1], output[2], [], "本轮回答没有使用知识库引用。"
            else:
                yield output

    examples = [
        "组合仪表有哪些显示区域？",
        "READY 指示灯是什么意思？",
        "车辆识别代码 VIN 在哪里？",
        "车载热点怎么开启？",
    ]

    with gr.Blocks(
        title="RAG Agent Customer Service",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=css,
    ) as demo:
        gr.Markdown(
            f"## 智能客服 RAG Agent\n"
            f"基于 PDF 用户手册的多工具问答工作台，当前模型后端：`{LLM_BACKEND}`"
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=260, elem_id="sidebar"):
                gr.Markdown("### 会话")
                session_id = gr.Textbox(label="Session ID", value="default", lines=1)
                clear_btn = gr.Button("清空当前会话", variant="secondary")

                gr.Markdown("### 示例问题")
                with gr.Column(elem_classes=["example-row"]):
                    example_buttons = [gr.Button(text, variant="secondary") for text in examples]

                gr.Markdown("### 系统状态")
                status_box = gr.Textbox(
                    value=initial_status,
                    label="Status",
                    lines=12,
                    interactive=False,
                    elem_id="status-panel",
                )

            with gr.Column(scale=4, elem_id="answer-workspace"):
                chatbot = _make_chatbot(
                    label="Agent 对话",
                    **_CHATBOT_MESSAGE_KWARGS,
                    elem_id="chatbot",
                    show_copy_button=True,
                )
                citation_panel = gr.Markdown(
                    "本轮回答没有使用知识库引用。",
                    label="本轮来源引用",
                    container=True,
                    elem_id="citation-panel",
                )
                retrieved_gallery = _make_gallery(
                    label="命中的图片证据",
                    columns=3,
                    rows=2,
                    height=280,
                    object_fit="contain",
                    allow_preview=True,
                    show_download_button=True,
                )
                with gr.Row():
                    message = gr.Textbox(
                        label="输入问题",
                        placeholder="例如：指示灯和警告灯分别代表什么？",
                        lines=3,
                        scale=6,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)

        gr.Markdown("## 知识库管理")
        gr.Markdown("上传 PDF 后，系统会执行多模态切分、图片提取、向量化入库，并生成可立即测试的独立知识库。")
        with gr.Row():
            with gr.Column(scale=2, min_width=360):
                kb_pdf = gr.File(label="上传 PDF 文档", file_types=[".pdf"], type="filepath")
                kb_name = gr.Textbox(label="知识库名称", placeholder="留空时使用 PDF 文件名")
                with gr.Accordion("切分与入库参数", open=False):
                    toc_pages = gr.Textbox(label="目录页范围", value="3-7")
                    with gr.Row():
                        child_size = gr.Number(label="文本块长度", value=450, precision=0)
                        overlap = gr.Number(label="重叠长度", value=80, precision=0)
                    with gr.Row():
                        parent_size = gr.Number(label="文本父块最大长度", value=1500, precision=0)
                        batch_size = gr.Number(label="向量化批次大小", value=128, precision=0)
                build_kb_btn = gr.Button("构建知识库", variant="primary")
                kb_build_status = gr.Textbox(label="构建状态", lines=6, interactive=False)
                kb_preview_file = gr.File(label="切分结果预览", interactive=False)

            with gr.Column(scale=3, min_width=460):
                kb_list_markdown, kb_names = _format_knowledge_base_list()
                kb_list = gr.Markdown(kb_list_markdown)
                refresh_kb_btn = gr.Button("刷新知识库列表", variant="secondary")
                kb_selector = gr.Dropdown(
                    label="测试知识库",
                    choices=kb_names,
                    value=kb_names[0] if kb_names else None,
                )
                activate_kb_btn = gr.Button("接入客服 Agent", variant="primary")
                with gr.Row():
                    kb_query = gr.Textbox(
                        label="测试问题",
                        placeholder="例如：车辆胎压标签在哪里？",
                        scale=5,
                    )
                    kb_top_k = gr.Number(label="Top K", value=6, precision=0, scale=1)
                search_kb_btn = gr.Button("测试检索", variant="secondary")
                kb_search_results = gr.Markdown("检索结果将在这里显示。")

        message.submit(
            submit_message_for_ui,
            inputs=[message, chatbot, session_id],
            outputs=[chatbot, message, status_box, retrieved_gallery, citation_panel],
        )
        send_btn.click(
            submit_message_for_ui,
            inputs=[message, chatbot, session_id],
            outputs=[chatbot, message, status_box, retrieved_gallery, citation_panel],
        )
        clear_btn.click(
            clear_current_session,
            inputs=[session_id],
            outputs=[chatbot, status_box, retrieved_gallery, citation_panel],
        )
        for button, example in zip(example_buttons, examples):
            button.click(fill_example, inputs=[gr.State(example)], outputs=[message])
        build_kb_btn.click(
            build_uploaded_knowledge_base,
            inputs=[kb_pdf, kb_name, toc_pages, child_size, overlap, parent_size, batch_size],
            outputs=[kb_build_status, kb_list, kb_selector, kb_preview_file],
        )
        refresh_kb_btn.click(
            refresh_knowledge_bases,
            outputs=[kb_list, kb_selector],
        )
        search_kb_btn.click(
            search_selected_knowledge_base,
            inputs=[kb_selector, kb_query, kb_top_k],
            outputs=[kb_search_results],
        )
        activate_kb_btn.click(
            activate_selected_knowledge_base,
            inputs=[kb_selector],
            outputs=[status_box],
        )
        kb_query.submit(
            search_selected_knowledge_base,
            inputs=[kb_selector, kb_query, kb_top_k],
            outputs=[kb_search_results],
        )

    return demo


def create_gradio_interface():
    """Create the primary customer-service workspace."""
    css = """
    :root {
        --line: #d8e0ea;
        --surface: #ffffff;
        --surface-subtle: #f8fafc;
        --surface-muted: #eef4f8;
        --text: #020617;
        --muted: #64748b;
        --ink: #0f172a;
        --accent: #0369a1;
        --accent-soft: #e0f2fe;
        --ok: #047857;
        --shadow: 0 14px 40px rgba(15, 23, 42, 0.08);
    }
    .gradio-container {
        max-width: 1600px !important;
        width: 100% !important;
        margin: 0 auto !important;
        color: var(--text);
        overflow-x: clip;
        background: #eef3f7;
        font-family: "Inter", "Segoe UI", "Microsoft YaHei", system-ui, sans-serif;
        padding: 0 18px 24px !important;
    }
    .gradio-container * {
        box-sizing: border-box;
        letter-spacing: 0;
    }
    #app-header {
        margin: 0 -18px 12px;
        padding: 0;
        background: #0f172a;
        border-bottom: 3px solid #22c55e;
    }
    .product-bar {
        min-height: 68px;
        padding: 12px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        color: #f8fafc;
    }
    .product-identity {
        display: flex;
        align-items: center;
        gap: 12px;
        min-width: 0;
    }
    .product-mark {
        width: 38px;
        height: 38px;
        display: grid;
        place-items: center;
        flex: 0 0 auto;
        color: #052e16;
        background: #4ade80;
        border-radius: 6px;
        font-family: "Fira Code", Consolas, monospace;
        font-size: 17px;
        font-weight: 800;
    }
    .product-title {
        color: #f8fafc;
        font-size: 17px;
        font-weight: 750;
        line-height: 1.2;
    }
    .product-subtitle {
        color: #94a3b8;
        margin-top: 3px;
        font-size: 12px;
    }
    .product-meta {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 8px;
        flex-wrap: wrap;
    }
    .meta-item {
        padding: 6px 9px;
        border: 1px solid #334155;
        border-radius: 6px;
        color: #cbd5e1;
        background: #172033;
        font-family: "Fira Code", Consolas, monospace;
        font-size: 11px;
    }
    .meta-item.live {
        color: #bbf7d0;
        border-color: #166534;
        background: #052e16;
    }
    #metric-strip {
        margin: 0 0 12px;
    }
    .metric-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 8px;
    }
    .metric {
        min-width: 0;
        padding: 10px 12px;
        border: 1px solid var(--line);
        border-radius: 7px;
        background: var(--surface);
    }
    .metric-label {
        color: var(--muted);
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
    }
    .metric-value {
        margin-top: 3px;
        overflow: hidden;
        color: var(--ink);
        font-size: 13px;
        font-weight: 700;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    #control-panel, #source-reference-panel, #kb-build-panel, #kb-test-panel {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        min-width: 0 !important;
    }
    #conversation-panel {
        min-width: 0 !important;
    }
    .panel-heading {
        margin-bottom: 8px;
        padding-bottom: 9px;
        border-bottom: 1px solid var(--line);
    }
    .panel-kicker {
        color: var(--accent);
        font-family: "Fira Code", Consolas, monospace;
        font-size: 10px;
        font-weight: 750;
        text-transform: uppercase;
    }
    .panel-title {
        margin-top: 2px;
        color: var(--ink);
        font-size: 14px;
        font-weight: 760;
    }
    .panel-copy {
        margin-top: 2px;
        color: var(--muted);
        font-size: 11px;
        line-height: 1.4;
    }
    #control-panel h3, #source-reference-panel h3 {
        color: var(--ink);
        font-size: 13px;
        font-weight: 720;
        margin: 6px 0 8px;
    }
    #customer-workspace-row {
        display: grid !important;
        grid-template-columns: minmax(220px, 0.72fr) minmax(0, 3.28fr);
        align-items: flex-start;
        gap: 12px;
        width: 100%;
        min-width: 0;
    }
    #customer-workspace-row > div {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        flex: none !important;
    }
    #control-panel {
        grid-column: 1;
    }
    #conversation-panel {
        grid-column: 2;
    }
    #chatbot {
        min-height: 540px;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--surface);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }
    #chatbot .message {
        max-width: 78ch;
    }
    textarea, input, select {
        border-radius: 7px !important;
    }
    button {
        border-radius: 7px !important;
        transition: background-color 180ms ease, border-color 180ms ease, color 180ms ease;
        cursor: pointer !important;
    }
    button:hover {
        border-color: var(--accent) !important;
    }
    #source-reference-panel {
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
        margin-top: 8px;
        padding: 0;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--surface);
        overflow: hidden;
        contain: layout paint;
    }
    #source-reference-panel > div {
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
    }
    #citation-panel textarea {
        width: 100% !important;
        max-width: 100% !important;
        min-height: 240px !important;
        max-height: 360px !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        white-space: pre-wrap !important;
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
        resize: none !important;
        font-family: "Fira Code", "Microsoft YaHei", monospace;
        font-size: 12px;
        line-height: 1.6;
    }
    .citation-output {
        width: 100% !important;
        max-width: 100% !important;
        min-height: 500px;
        max-height: 555px;
        overflow-y: auto;
        overflow-x: hidden;
        padding: 4px;
        display: grid;
        gap: 10px;
    }
    .citation-empty {
        display: block;
        min-height: 120px;
        padding: 16px;
        border: 1px dashed var(--line);
        border-radius: 8px;
        color: #64748b;
        background: #f8fafc;
    }
    .citation-card {
        width: 100%;
        max-width: 100%;
        min-width: 0;
        contain: inline-size;
        padding: 12px;
        border: 1px solid var(--line);
        border-left: 3px solid #22c55e;
        border-radius: 8px;
        background: #fff;
    }
    .citation-card-top {
        display: flex;
        align-items: center;
        gap: 7px;
        margin-bottom: 7px;
    }
    .citation-index, .citation-type {
        font-size: 11px;
        line-height: 1;
        font-weight: 700;
        color: #166534;
    }
    .citation-card h4 {
        margin: 0 0 5px;
        font-size: 13px;
        line-height: 1.4;
        overflow-wrap: anywhere;
    }
    .citation-card h4 a {
        color: #0369a1;
        text-decoration: none;
    }
    .citation-meta {
        margin-bottom: 7px;
        color: #64748b;
        font-size: 11px;
        line-height: 1.4;
        overflow-wrap: anywhere;
    }
    .citation-card p {
        margin: 0;
        color: #334155;
        font-size: 12px;
        line-height: 1.55;
        white-space: normal;
        overflow-wrap: anywhere;
    }
    #source-reference-panel h3 {
        font-size: 14px;
        margin-top: 8px;
    }
    #status-panel textarea {
        font-size: 12px;
        line-height: 1.45;
        color: #334155;
        background: #f8fafc;
    }
    #evidence-gallery {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--surface);
    }
    #composer-row {
        margin-top: 8px;
        padding: 8px;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--surface);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
    }
    #composer-row textarea {
        min-height: 58px !important;
    }
    #send-button {
        min-height: 58px;
        background: #0369a1 !important;
        border-color: #0369a1 !important;
        color: #ffffff !important;
        font-weight: 720;
    }
    #send-button:hover {
        background: #075985 !important;
    }
    #clear-button {
        border-color: #cbd5e1 !important;
        color: #334155 !important;
    }
    #status-panel textarea {
        min-height: 170px !important;
        border-left: 3px solid #0369a1 !important;
    }
    .example-row button {
        justify-content: flex-start !important;
        padding: 8px 10px !important;
        background: #f8fafc !important;
        border-color: #e2e8f0 !important;
        color: #334155 !important;
        font-size: 12px !important;
    }
    .example-row button:hover {
        background: #eff6ff !important;
        border-color: #7dd3fc !important;
        color: #075985 !important;
    }
    .tab-nav button {
        font-weight: 700 !important;
    }
    .example-row button {
        min-width: 0 !important;
        white-space: normal !important;
        text-align: left;
        cursor: pointer;
    }
    @media (max-width: 760px) {
        .gradio-container {
            padding: 0 8px 16px !important;
        }
        #app-header {
            margin-left: -8px;
            margin-right: -8px;
        }
        .product-bar {
            align-items: flex-start;
            flex-direction: column;
            padding: 12px;
        }
        .product-meta {
            justify-content: flex-start;
        }
        .metric-strip {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        #customer-workspace-row {
            grid-template-columns: minmax(0, 1fr) !important;
        }
        #control-panel,
        #conversation-panel,
        #source-reference-panel {
            grid-column: 1 !important;
        }
        #chatbot {
            min-height: 460px;
        }
    }
    @media (prefers-reduced-motion: reduce) {
        * {
            transition: none !important;
            animation: none !important;
        }
    }
    footer { display: none !important; }
    """

    active_kb = os.environ.get("KNOWLEDGE_BASE_NAME", "").strip() or get_active_knowledge_base()
    initial_status = (
        f"System: ready\n"
        f"Backend: {LLM_BACKEND}\n"
        f"Knowledge base: {active_kb or 'legacy data/train_a.pdf'}\n"
        "Memory: Redis when available, fallback otherwise"
    )

    def normalize_session_id(value: str) -> str:
        return (value or "default").strip() or "default"

    def submit_message(message: str, history: list, session_id: str, use_agentic_rag: bool,
                       manual_models: list[str]):
        if not message or not message.strip():
            yield history or [], "", "请输入问题。", [], _format_citations_text([])
            return
        try:
            manual_models = _check_selected_manuals(message, manual_models)
        except ValueError as exc:
            yield history or [], message, str(exc), [], _format_citations_text([])
            return
        session_id = normalize_session_id(session_id)
        history = list(history or [])
        history.append({"role": "user", "content": message.strip()})
        history.append({"role": "assistant", "content": "正在处理..."})
        chat_handler = chat_with_agent if use_agentic_rag else chat_with_plain_rag
        for answer, status, images, citations in chat_handler(
            message.strip(), history[:-1], session_id, manual_models=manual_models,
        ):
            history[-1] = {"role": "assistant", "content": answer}
            yield history, "", status, images, citations

    def clear_session(session_id: str):
        session_id = normalize_session_id(session_id)
        short_term_memory.clear(session_id)
        latest_retrieved_images_by_session.pop(session_id, None)
        latest_citations_by_session.pop(session_id, None)
        return [], f"Session `{session_id}` 已清空。\nBackend: {LLM_BACKEND}", [], _format_citations_text([])

    def refresh_kb():
        markdown, names = _format_knowledge_base_list()
        return markdown, gr.update(choices=names, value=names[0] if names else None)

    examples = [
        "组合仪表有哪些显示区域？",
        "如何更换遥控钥匙电池？",
        "车辆识别代码 VIN 在哪里？",
        "胎压标签在哪里？",
    ]

    with gr.Blocks(
        title="RAG Agent Customer Service",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=css,
    ) as demo:
        gr.HTML(
            f"""
            <div class="product-bar">
                <div class="product-identity">
                    <div class="product-mark">RA</div>
                    <div>
                        <div class="product-title">汽车智能客服 RAG Console</div>
                        <div class="product-subtitle">普通 RAG 与 Agentic RAG 可切换的客服工作台</div>
                    </div>
                </div>
                <div class="product-meta">
                    <span class="meta-item live">RAG READY</span>
                    <span class="meta-item">LLM {html.escape(str(LLM_BACKEND))}</span>
                    <span class="meta-item">KB {html.escape(str(active_kb or 'legacy'))}</span>
                </div>
            </div>
            """,
            elem_id="app-header",
        )
        gr.HTML(
            f"""
            <div class="metric-strip">
                <div class="metric"><div class="metric-label">Workspace</div><div class="metric-value">Customer Service</div></div>
                <div class="metric"><div class="metric-label">Knowledge Base</div><div class="metric-value">{html.escape(str(active_kb or 'legacy data/train_a.pdf'))}</div></div>
                <div class="metric"><div class="metric-label">Evidence</div><div class="metric-value">RAG citations + images</div></div>
                <div class="metric"><div class="metric-label">Memory</div><div class="metric-value">Redis with fallback</div></div>
            </div>
            """,
            elem_id="metric-strip",
        )

        with gr.Tabs():
            with gr.Tab("客服工作台"):
                with gr.Row(equal_height=False, elem_id="customer-workspace-row"):
                    with gr.Column(scale=1, min_width=245, elem_id="control-panel"):
                        gr.HTML(
                            """
                            <div class="panel-heading">
                                <div class="panel-kicker">Control</div>
                                <div class="panel-title">会话控制</div>
                                <div class="panel-copy">管理当前咨询会话和短期记忆。</div>
                            </div>
                            """
                        )
                        session_id = gr.Textbox(label="Session ID", value="default", lines=1)
                        clear_btn = gr.Button("清空当前会话", variant="secondary", elem_id="clear-button")
                        gr.HTML(
                            """
                            <div class="panel-heading">
                                <div class="panel-kicker">Examples</div>
                                <div class="panel-title">示例问题</div>
                            </div>
                            """
                        )
                        with gr.Column(elem_classes=["example-row"]):
                            example_buttons = [gr.Button(item, variant="secondary") for item in examples]
                        gr.HTML(
                            """
                            <div class="panel-heading">
                                <div class="panel-kicker">Runtime</div>
                                <div class="panel-title">运行状态</div>
                            </div>
                            """
                        )
                        status_box = gr.Textbox(
                            label="Status",
                            value=initial_status,
                            lines=10,
                            interactive=False,
                            elem_id="status-panel",
                        )

                    with gr.Column(scale=3, min_width=520, elem_id="conversation-panel"):
                        gr.HTML(
                            """
                            <div class="panel-heading">
                                <div class="panel-kicker">Assistant</div>
                                <div class="panel-title">RAG 对话工作区</div>
                                <div class="panel-copy">默认使用普通 RAG；勾选输入区的开关后使用 Agentic RAG。</div>
                            </div>
                            """
                        )
                        chatbot = _make_chatbot(
                            label="Agent 对话",
                            **_CHATBOT_MESSAGE_KWARGS,
                            elem_id="chatbot",
                            show_copy_button=True,
                        )
                        use_agentic_rag = gr.Checkbox(
                            label="启用 Agentic RAG（默认关闭）",
                            value=False,
                            info="关闭：使用原始普通 RAG；开启：使用 Agentic RAG 工作流。",
                            elem_id="rag-mode-toggle",
                        )
                        manual_models = gr.Dropdown(
                            label="选择车型（必选；可多选对比）",
                            choices=list_active_manual_models(MULTI_MANUAL_STORE),
                            value=[], multiselect=True,
                            info="车型来自当前已激活的手册索引。仅以下拉框为准；问题中的车型名暂不用于选择或冲突判断。",
                        )
                        refresh_manuals_btn = gr.Button("刷新可选车型", variant="secondary")
                        with gr.Row(elem_id="composer-row"):
                            message = gr.Textbox(
                                label="输入问题",
                                placeholder="请输入关于车辆功能、操作或维护的问题",
                                lines=2,
                                scale=7,
                            )
                            send_btn = gr.Button("发送", variant="primary", scale=1, elem_id="send-button")
                        with gr.Accordion("命中的图片证据", open=True):
                            retrieved_gallery = _make_gallery(
                                columns=3,
                                rows=2,
                                height=260,
                                object_fit="contain",
                                allow_preview=True,
                                show_download_button=True,
                                elem_id="evidence-gallery",
                            )
                        # Codex-CITATION-LAYOUT-FIX: Long RAG citations live inside
                        # the main conversation column instead of a fragile third grid column.
                        with gr.Accordion("本轮来源引用", open=True, elem_id="source-reference-panel"):
                            citation_panel = gr.Textbox(
                                value=_format_citations_text([]),
                                label="回答所使用的证据",
                                lines=12,
                                max_lines=16,
                                interactive=False,
                                elem_id="citation-panel",
                            )

            with gr.Tab("多车型手册上传"):
                gr.Markdown(
                    "只需上传车型手册 PDF；后台自动使用本机 MinerU `flash + txt` 解析，"
                    "生成结构化内容和图片，再构建新索引。解析和向量化可能需要数分钟。"
                    "同名车型上传不同 PDF 时，会构建新版本并保留旧索引以便回滚。"
                    "目前的目录/父子切分器适配当前比亚迪手册结构；不兼容的排版会报错，旧索引保持可用。"
                )
                upload_model_name = gr.Textbox(label="车型名称", placeholder="例如：新车型名称")
                with gr.Row():
                    upload_manual_version = gr.Textbox(
                        label="厂家手册版本/版次（选填）", placeholder="按手册原文填写；未标注则留空")
                    upload_model_year = gr.Textbox(
                        label="车型年款（选填）", placeholder="仅有明确依据时填写四位年份")
                upload_manual_pdf = gr.File(label="车型手册 PDF", file_types=[".pdf"], type="filepath")
                upload_manual_btn = gr.Button("解析、建索引并激活", variant="primary")
                upload_manual_status = gr.Textbox(label="处理结果", lines=4, interactive=False)
                with gr.Accordion("历史索引与回滚", open=False):
                    gr.Markdown("回滚会切换整套活动索引（包括所有车型），不会删除上传文件或历史版本。")
                    history_choices = list_index_versions(MULTI_MANUAL_STORE)
                    index_history = gr.Dropdown(
                        label="选择历史索引版本",
                        choices=[(f"{'当前' if item['active'] else '历史'} | {item['built_at']} | "
                                  f"{len(item['manuals'])} 款车型 | {item['version_id']}",
                                  item["version_id"]) for item in history_choices],
                        value=None,
                    )
                    with gr.Row():
                        refresh_index_history_btn = gr.Button("刷新版本列表", variant="secondary")
                        rollback_index_btn = gr.Button("切换到所选版本", variant="secondary")

            with gr.Tab("知识库管理"):
                gr.HTML(
                    """
                    <div class="panel-heading">
                        <div class="panel-kicker">Knowledge Base</div>
                        <div class="panel-title">上传并构建知识库</div>
                        <div class="panel-copy">从 PDF 生成多模态切分、图片资产和向量索引。</div>
                    </div>
                    """
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2, min_width=380, elem_id="kb-build-panel"):
                        kb_pdf = gr.File(label="上传 PDF 文档", file_types=[".pdf"], type="filepath")
                        kb_name = gr.Textbox(label="知识库名称", placeholder="留空时使用 PDF 文件名")
                        with gr.Accordion("切分与入库参数", open=False):
                            toc_pages = gr.Textbox(label="目录页范围", value="3-7")
                            with gr.Row():
                                child_size = gr.Number(label="文本块长度", value=450, precision=0)
                                overlap = gr.Number(label="重叠长度", value=80, precision=0)
                            with gr.Row():
                                parent_size = gr.Number(label="文本父块最大长度", value=1500, precision=0)
                                batch_size = gr.Number(label="向量化批次大小", value=128, precision=0)
                        build_kb_btn = gr.Button("构建知识库", variant="primary")
                        kb_build_status = gr.Textbox(label="构建状态", lines=6, interactive=False)
                        kb_preview_file = gr.File(label="切分结果预览", interactive=False)

                    with gr.Column(scale=3, min_width=500, elem_id="kb-test-panel"):
                        kb_list_markdown, kb_names = _format_knowledge_base_list()
                        kb_list = gr.Markdown(kb_list_markdown)
                        with gr.Row():
                            refresh_kb_btn = gr.Button("刷新列表", variant="secondary")
                            kb_selector = gr.Dropdown(
                                label="选择知识库",
                                choices=kb_names,
                                value=kb_names[0] if kb_names else None,
                                scale=3,
                            )
                            activate_kb_btn = gr.Button("接入客服 Agent", variant="primary")
                        with gr.Row():
                            kb_query = gr.Textbox(label="测试问题", placeholder="输入问题测试检索效果", scale=5)
                            kb_top_k = gr.Number(label="Top K", value=6, precision=0, scale=1)
                        search_kb_btn = gr.Button("测试检索", variant="secondary")
                        kb_search_results = gr.Markdown("检索结果将在这里显示。")

        message.submit(
            submit_message,
            inputs=[message, chatbot, session_id, use_agentic_rag, manual_models],
            outputs=[chatbot, message, status_box, retrieved_gallery, citation_panel],
        )
        send_btn.click(
            submit_message,
            inputs=[message, chatbot, session_id, use_agentic_rag, manual_models],
            outputs=[chatbot, message, status_box, retrieved_gallery, citation_panel],
        )
        clear_btn.click(
            clear_session,
            inputs=[session_id],
            outputs=[chatbot, status_box, retrieved_gallery, citation_panel],
        )
        for button, example in zip(example_buttons, examples):
            button.click(lambda text: text, inputs=[gr.State(example)], outputs=[message])

        refresh_manuals_btn.click(_refresh_manual_choices, inputs=[manual_models], outputs=[manual_models])
        upload_manual_btn.click(
            _upload_multi_manual,
            inputs=[upload_model_name, upload_manual_pdf, upload_manual_version, upload_model_year],
            outputs=[upload_manual_status, manual_models, index_history],
        )
        demo.load(_refresh_manual_choices, inputs=[manual_models], outputs=[manual_models])
        demo.load(_refresh_index_history, outputs=[index_history])
        refresh_index_history_btn.click(_refresh_index_history, outputs=[index_history])
        rollback_index_btn.click(
            _rollback_multi_manual, inputs=[index_history],
            outputs=[upload_manual_status, manual_models, index_history],
        )

        build_kb_btn.click(
            build_uploaded_knowledge_base,
            inputs=[kb_pdf, kb_name, toc_pages, child_size, overlap, parent_size, batch_size],
            outputs=[kb_build_status, kb_list, kb_selector, kb_preview_file],
        )
        refresh_kb_btn.click(refresh_kb, outputs=[kb_list, kb_selector])
        activate_kb_btn.click(activate_selected_knowledge_base, inputs=[kb_selector], outputs=[status_box])
        search_kb_btn.click(
            search_selected_knowledge_base,
            inputs=[kb_selector, kb_query, kb_top_k],
            outputs=[kb_search_results],
        )
        kb_query.submit(
            search_selected_knowledge_base,
            inputs=[kb_selector, kb_query, kb_top_k],
            outputs=[kb_search_results],
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the RAG Agent Gradio interface.")
    parser.add_argument("--skip-agent-init", action="store_true", help="Open only the UI and knowledge-base manager.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("启动 RAG Agent Gradio 界面...")
    print("=" * 70 + "\n")

    if not args.skip_agent_init and os.environ.get("SKIP_AGENT_INIT", "0") != "1":
        initialize_system()
    else:
        print("Skipping Agent initialization; knowledge-base management remains available.")
    demo = create_gradio_interface()
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=args.port,
        share=False,
        show_error=True,
        quiet=False,
        prevent_thread_lock=False,
        pwa=True,
    )
