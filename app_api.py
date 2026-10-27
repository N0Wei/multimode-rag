"""FastAPI service for the automotive Agentic RAG runtime."""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent.run_store import RunStore
from agent.runtime import AgentRuntime, new_agent_state
from short_term_memory import ShortTermMemory


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(default="default", min_length=1, max_length=128)
    history: List[Message] = Field(default_factory=list)


class ChatResponse(BaseModel):
    run_id: str
    session_id: str
    answer: str
    answer_source: str
    confidence: float
    need_human: bool
    citation_coverage: float
    metrics: Dict[str, Any]


def create_app(
    runtime: Optional[AgentRuntime] = None,
    run_store: Optional[RunStore] = None,
    memory: Optional[ShortTermMemory] = None,
) -> FastAPI:
    app = FastAPI(title="Automotive Agentic RAG API", version="0.1.0")
    app.state.runtime = runtime or AgentRuntime()
    app.state.run_store = run_store or RunStore(os.getenv("AGENT_RUN_DB", "data/runtime/agent_runs.sqlite3"))
    app.state.memory = memory or ShortTermMemory()

    @app.get("/health")
    def health():
        active_runtime = app.state.runtime
        return {
            "status": "ready" if active_runtime.ready else "not_ready",
            "knowledge_base": active_runtime.knowledge_base,
            "initialization_error": active_runtime.initialization_error,
        }

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest):
        run_id = uuid.uuid4().hex
        app.state.run_store.create(run_id, request.session_id, request.question)
        context = app.state.memory.build_context(request.session_id)
        history = context["messages"] or [item.model_dump() for item in request.history]
        state = new_agent_state(
            request.question,
            conversation_history=history,
            metadata={
                "run_id": run_id,
                "session_id": request.session_id,
                "short_term_summary": context["summary"],
                "short_term_memory_tokens": context["memory_tokens"],
                "short_term_redis_enabled": context["redis_enabled"],
            },
        )
        try:
            result = app.state.runtime.invoke(state)
            app.state.run_store.complete(run_id, result)
            answer = str(result.get("final_answer", ""))
            if answer:
                app.state.memory.append_exchange(request.session_id, request.question, answer)
            return ChatResponse(
                run_id=run_id,
                session_id=request.session_id,
                answer=answer,
                answer_source=str(result.get("answer_source", "unknown")),
                confidence=float(result.get("confidence", 0.0) or 0.0),
                need_human=bool(result.get("need_human", False)),
                citation_coverage=float(result.get("citation_coverage", 0.0) or 0.0),
                metrics=result.get("metrics", {}),
            )
        except Exception as exc:
            app.state.run_store.fail(run_id, str(exc), state.get("execution_trace", []))
            raise HTTPException(status_code=503, detail={"run_id": run_id, "error": str(exc)}) from exc

    @app.get("/runs/{run_id}")
    def get_run(run_id: str):
        item = app.state.run_store.get(run_id)
        if item is None:
            raise HTTPException(status_code=404, detail="run not found")
        return item

    @app.delete("/sessions/{session_id}")
    def clear_session(session_id: str):
        app.state.memory.clear(session_id)
        return {"session_id": session_id, "cleared": True}

    return app


app = create_app()

