"""Request-scoped helpers for forwarding answer tokens to the chat UI."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any, Callable


StreamCallback = Callable[[str, str], None]
_STREAM_CALLBACK: ContextVar[StreamCallback | None] = ContextVar(
    "rag_stream_callback", default=None
)


def set_stream_callback(callback: StreamCallback | None) -> Token:
    """Set the token callback for the current request/thread."""
    return _STREAM_CALLBACK.set(callback)


def reset_stream_callback(token: Token) -> None:
    """Restore the callback that was active before this request."""
    _STREAM_CALLBACK.reset(token)


def publish_stream_event(kind: str, text: str = "") -> None:
    callback = _STREAM_CALLBACK.get()
    if callback is not None:
        callback(kind, text)


def _chunk_text(chunk: Any) -> str:
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


def _complete_generation(llm: Any, prompt: str) -> str:
    """Compatibility fallback for adapters that do not expose token streaming."""
    if callable(getattr(llm, "invoke", None)):
        response = llm.invoke(prompt)
        text = _chunk_text(response)
    elif callable(getattr(llm, "infer", None)):
        outputs = llm.infer([prompt])
        text = str(outputs[0]) if outputs else ""
    else:
        raise TypeError("LLM adapter must provide stream(), invoke(), or infer()")
    if text:
        publish_stream_event("token", text)
    return text.strip()


def generate_text_streaming(llm: Any, prompt: str, *, replace: bool = False) -> str:
    """Generate one answer, publishing real model chunks when supported.

    ``replace`` marks a new answer-generation pass (for example, a grounded
    rewrite) so the UI clears any earlier draft before displaying new tokens.
    """
    if replace:
        publish_stream_event("reset")

    stream = getattr(llm, "stream", None)
    if not callable(stream):
        return _complete_generation(llm, prompt)

    chunks: list[str] = []
    try:
        for chunk in stream(prompt):
            text = _chunk_text(chunk)
            if text:
                chunks.append(text)
                publish_stream_event("token", text)
    except NotImplementedError:
        # Some third-party adapters expose stream() but do not implement it.
        # Fall back only if no partial answer has reached the UI.
        if chunks:
            raise
        return _complete_generation(llm, prompt)
    return "".join(chunks).strip()
