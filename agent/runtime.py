"""Reusable Agent runtime shared by API and future interactive frontends."""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional


def new_agent_state(
    question: str,
    conversation_history: Optional[list] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return one complete, request-isolated LangGraph state."""
    return {
        "user_question": question,
        "conversation_history": conversation_history or [],
        "intent": "",
        "query_rewrite": "",
        "plan": [],
        "structured_plan": {},
        "scope_route": "",
        "is_complex_query": False,
        "decomposition_reason": "",
        "sub_queries": [],
        "current_subquery_index": 0,
        "current_subquery": "",
        "subquery_results": [],
        "rag_result": {},
        "faq_result": None,
        "web_search_result": None,
        "last_executed_tool": "",
        "draft_answer": "",
        "claims": [],
        "evidence_catalog": [],
        "claim_verdicts": [],
        "citation_coverage": 0.0,
        "grounding_status": "",
        "grounding_retry_count": 0,
        "reflection_count": 0,
        "quality_score": 0.0,
        "reflection_notes": [],
        "improvement_actions": [],
        "retrieval_grade": "",
        "retrieval_grade_reason": "",
        "retrieval_retry_count": 0,
        "retrieval_query_history": [],
        "active_retrieval_query": "",
        "final_answer": "",
        "answer_source": "",
        "confidence": 0.0,
        "need_human": False,
        "metadata": metadata or {},
        "metrics": {},
        "execution_trace": [],
        "execution_cache": {},
        "execution_budget": {},
    }


class AgentRuntime:
    """Lazily initialize and invoke the single active knowledge-base Agent."""

    def __init__(self, graph=None):
        self._graph = graph
        self._knowledge_base = None
        self._lock = threading.Lock()
        self._initialization_error = ""

    @property
    def ready(self) -> bool:
        return self._graph is not None

    @property
    def knowledge_base(self) -> Optional[str]:
        return self._knowledge_base

    @property
    def initialization_error(self) -> str:
        return self._initialization_error

    def initialize(self, knowledge_base_name: Optional[str] = None) -> None:
        if self.ready:
            return
        with self._lock:
            if self.ready:
                return
            try:
                # Heavy model/retrieval dependencies are loaded only for a real
                # runtime initialization. API control-plane tests can inject a
                # fake graph without importing torch, jieba, FAISS or models.
                from agent.graph import build_agent_graph
                from agent.nodes.tool_executor import set_tools
                from agent.tools import (
                    BochaSearchMCPTool,
                    DDGSSearchTool,
                    FAQTool,
                    FallbackSearchTool,
                    MultimodalRAGTool,
                )
                from config import LLM_BACKEND
                from knowledge_base_service import get_active_knowledge_base, knowledge_base_paths
                from llm_model import get_llm_model
                from multimodal_retriever import MultimodalRetriever
                from rerank_model import reRankLLM

                name = (
                    knowledge_base_name
                    or os.getenv("KNOWLEDGE_BASE_NAME", "").strip()
                    or get_active_knowledge_base()
                )
                if not name:
                    raise RuntimeError("没有可用知识库，请先通过现有建库流程创建并激活知识库")
                store = knowledge_base_paths(name)["store"]
                if not (store / "manifest.json").is_file():
                    raise RuntimeError(f"知识库索引不完整: {store}")

                retriever = MultimodalRetriever(store)
                llm = get_llm_model(
                    model_path="./pre_train_model/Qwen-7B-Chat",
                    backend=LLM_BACKEND,
                )
                rag_tool = MultimodalRAGTool(
                    multimodal_retriever=retriever,
                    rerank_model=reRankLLM(),
                    llm=llm,
                )
                faq_tool = FAQTool(faq_path="./data/faq_database.json")
                web_search_tool = FallbackSearchTool(
                    BochaSearchMCPTool(max_results=3),
                    DDGSSearchTool(max_results=3),
                )
                set_tools(rag_tool, faq_tool, web_search_tool)
                self._graph = build_agent_graph()
                self._knowledge_base = name
                self._initialization_error = ""
            except Exception as exc:
                self._initialization_error = str(exc)
                raise

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not self.ready:
            self.initialize()
        return self._graph.invoke(state, config={"recursion_limit": 80})
