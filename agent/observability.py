#!/usr/bin/env python
# coding: utf-8
"""Request-scoped observability for graph nodes and model calls."""

import time
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable
from llm_streaming import reset_stream_callback, set_stream_callback


_CURRENT_STATE: ContextVar = ContextVar("agent_observability_state", default=None)


def instrument_node(name: str, func: Callable) -> Callable:
    """Wrap a LangGraph node and append a duration/error event to its state."""
    @wraps(func)
    def wrapped(state, *args, **kwargs):
        metrics = state.setdefault("metrics", {})
        metrics.setdefault("workflow_started_at", time.monotonic())
        metrics.setdefault("node_calls", 0)
        metrics.setdefault("llm_calls", 0)
        metrics.setdefault("llm_duration_ms", 0.0)
        metrics.setdefault("input_tokens", 0)
        metrics.setdefault("output_tokens", 0)
        trace = state.setdefault("execution_trace", [])
        event = {"event_type": "node", "name": name,
                 "started_ms": _elapsed_ms(metrics), "status": "running"}
        trace.append(event)
        token = _CURRENT_STATE.set(state)
        stream_callback = (state.get("metadata") or {}).get("_stream_callback")
        stream_token = (set_stream_callback(stream_callback)
                        if callable(stream_callback) else None)
        started = time.perf_counter()
        try:
            result = func(state, *args, **kwargs)
            event["status"] = "success"
            return result
        except Exception as exc:
            event.update({"status": "error", "error": str(exc)[:500]})
            raise
        finally:
            duration = round((time.perf_counter() - started) * 1000, 2)
            event["duration_ms"] = duration
            metrics["node_calls"] += 1
            node_durations = metrics.setdefault("node_durations_ms", {})
            node_durations[name] = round(node_durations.get(name, 0.0) + duration, 2)
            if name == "finalize":
                finalize_metrics(state)
            if stream_token is not None:
                reset_stream_callback(stream_token)
            _CURRENT_STATE.reset(token)
    return wrapped


class InstrumentedLLM:
    """Transparent proxy that records invoke latency and token usage."""

    def __init__(self, llm: Any, purpose: str = "general"):
        self._llm = llm
        self._purpose = purpose

    def invoke(self, *args, **kwargs):
        state = _CURRENT_STATE.get()
        if state is None:
            return self._llm.invoke(*args, **kwargs)
        metrics = state.setdefault("metrics", {})
        event = {"event_type": "llm", "name": self._purpose,
                 "model": _model_name(self._llm), "started_ms": _elapsed_ms(metrics),
                 "status": "running"}
        state.setdefault("execution_trace", []).append(event)
        started = time.perf_counter()
        try:
            response = self._llm.invoke(*args, **kwargs)
            usage = _token_usage(response)
            event.update({"status": "success", **usage})
            metrics["input_tokens"] = metrics.get("input_tokens", 0) + usage["input_tokens"]
            metrics["output_tokens"] = metrics.get("output_tokens", 0) + usage["output_tokens"]
            return response
        except Exception as exc:
            event.update({"status": "error", "error": str(exc)[:500]})
            raise
        finally:
            duration = round((time.perf_counter() - started) * 1000, 2)
            event["duration_ms"] = duration
            metrics["llm_calls"] = metrics.get("llm_calls", 0) + 1
            metrics["llm_duration_ms"] = round(metrics.get("llm_duration_ms", 0.0) + duration, 2)

    def stream(self, *args, **kwargs):
        """Stream model chunks while retaining per-request LLM metrics."""
        stream_method = getattr(self._llm, "stream", None)
        if not callable(stream_method):
            yield self.invoke(*args, **kwargs)
            return

        state = _CURRENT_STATE.get()
        if state is None:
            yield from stream_method(*args, **kwargs)
            return

        metrics = state.setdefault("metrics", {})
        event = {"event_type": "llm", "name": self._purpose,
                 "model": _model_name(self._llm), "started_ms": _elapsed_ms(metrics),
                 "status": "running", "streaming": True}
        state.setdefault("execution_trace", []).append(event)
        started = time.perf_counter()
        input_tokens = output_tokens = 0
        try:
            for chunk in stream_method(*args, **kwargs):
                usage = _token_usage(chunk)
                input_tokens = max(input_tokens, usage["input_tokens"])
                output_tokens = max(output_tokens, usage["output_tokens"])
                yield chunk
            event.update({"status": "success", "input_tokens": input_tokens,
                          "output_tokens": output_tokens})
            metrics["input_tokens"] = metrics.get("input_tokens", 0) + input_tokens
            metrics["output_tokens"] = metrics.get("output_tokens", 0) + output_tokens
        except Exception as exc:
            event.update({"status": "error", "error": str(exc)[:500]})
            raise
        finally:
            duration = round((time.perf_counter() - started) * 1000, 2)
            event["duration_ms"] = duration
            metrics["llm_calls"] = metrics.get("llm_calls", 0) + 1
            metrics["llm_duration_ms"] = round(metrics.get("llm_duration_ms", 0.0) + duration, 2)

    def __getattr__(self, name):
        return getattr(self._llm, name)


def finalize_metrics(state: dict) -> None:
    metrics = state.setdefault("metrics", {})
    started = metrics.get("workflow_started_at")
    if started is not None:
        metrics["total_duration_ms"] = round((time.monotonic() - started) * 1000, 2)
    budget = state.get("execution_budget", {})
    metrics["tool_calls"] = budget.get("tool_calls", 0)
    metrics["cache_hits"] = budget.get("cache_hits", 0)
    metrics["deduplicated_calls"] = budget.get("deduplicated_calls", 0)


def _elapsed_ms(metrics: dict) -> float:
    started = metrics.setdefault("workflow_started_at", time.monotonic())
    return round((time.monotonic() - started) * 1000, 2)


def _model_name(llm: Any) -> str:
    return str(getattr(llm, "model_name", None) or getattr(llm, "model", None) or type(llm).__name__)


def _token_usage(response: Any) -> dict:
    usage = getattr(response, "usage_metadata", None) or {}
    metadata = getattr(response, "response_metadata", None) or {}
    if not isinstance(usage, dict):
        usage = {}
    if not isinstance(metadata, dict):
        metadata = {}
    legacy = metadata.get("token_usage") or metadata.get("usage") or {}
    if not isinstance(legacy, dict):
        legacy = {}
    input_tokens = usage.get("input_tokens", legacy.get("prompt_tokens", legacy.get("input_tokens", 0)))
    output_tokens = usage.get("output_tokens", legacy.get("completion_tokens", legacy.get("output_tokens", 0)))
    return {"input_tokens": int(input_tokens or 0), "output_tokens": int(output_tokens or 0)}
