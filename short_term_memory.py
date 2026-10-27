#!/usr/bin/env python
# coding: utf-8
"""
Redis-backed short-term session memory.

The memory layer keeps only a bounded conversation window in prompts:
- Redis stores recent messages, a compact session summary, and token metadata.
- Older messages are summarized when the window grows too large.
- Context returned to the Agent is budgeted so the prompt cannot grow forever.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


def estimate_tokens(text: str) -> int:
    """A small tokenizer-free estimate that works well enough for budgeting."""
    if not text:
        return 0
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return max(1, ascii_chars // 4 + non_ascii_chars)


def truncate_by_tokens(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    # Conservative character estimate: keep the tail because it is usually newer.
    keep_chars = max(200, max_tokens * 2)
    return text[-keep_chars:]


class ShortTermMemory:
    """Session-level memory backed by Redis with an in-process fallback."""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        max_messages: Optional[int] = None,
        max_message_tokens: Optional[int] = None,
        max_summary_tokens: Optional[int] = None,
        max_context_tokens: Optional[int] = None,
    ):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.ttl_seconds = ttl_seconds or int(os.getenv("SHORT_MEMORY_TTL_SECONDS", "86400"))
        self.max_messages = max_messages or int(os.getenv("SHORT_MEMORY_MAX_MESSAGES", "12"))
        self.max_message_tokens = max_message_tokens or int(os.getenv("SHORT_MEMORY_MAX_MESSAGE_TOKENS", "1800"))
        self.max_summary_tokens = max_summary_tokens or int(os.getenv("SHORT_MEMORY_MAX_SUMMARY_TOKENS", "500"))
        self.max_context_tokens = max_context_tokens or int(os.getenv("SHORT_MEMORY_MAX_CONTEXT_TOKENS", "2400"))
        self._redis = self._connect()
        self._fallback: Dict[str, Dict[str, Any]] = {}

    def _connect(self):
        try:
            import redis

            client = redis.Redis.from_url(self.redis_url, decode_responses=True)
            client.ping()
            print(f"  [ShortMemory] Redis connected: {self.redis_url}")
            return client
        except Exception as exc:
            print(f"  [ShortMemory] Redis unavailable, using in-process fallback: {exc}")
            return None

    def build_context(self, session_id: str) -> Dict[str, Any]:
        """Return a bounded prompt context for the current session."""
        summary = self.get_summary(session_id)
        messages = self.get_messages(session_id)
        selected = self._select_recent_messages(messages)
        memory_tokens = estimate_tokens(summary) + sum(estimate_tokens(m.get("content", "")) for m in selected)
        return {
            "session_id": session_id,
            "summary": summary,
            "messages": selected,
            "memory_tokens": memory_tokens,
            "redis_enabled": self._redis is not None,
        }

    def append_exchange(self, session_id: str, user_message: str, assistant_message: str):
        now = datetime.now().isoformat(timespec="seconds")
        self._append_message(session_id, {
            "role": "user",
            "content": user_message,
            "timestamp": now,
            "token_count": estimate_tokens(user_message),
        })
        self._append_message(session_id, {
            "role": "assistant",
            "content": assistant_message,
            "timestamp": now,
            "token_count": estimate_tokens(assistant_message),
        })
        self._compact_if_needed(session_id)

    def clear(self, session_id: str):
        if self._redis:
            self._redis.delete(self._messages_key(session_id), self._summary_key(session_id), self._meta_key(session_id))
        else:
            self._fallback.pop(session_id, None)

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        if self._redis:
            raw_items = self._redis.lrange(self._messages_key(session_id), 0, -1)
            return [json.loads(item) for item in raw_items]
        return list(self._fallback.get(session_id, {}).get("messages", []))

    def get_summary(self, session_id: str) -> str:
        if self._redis:
            return self._redis.get(self._summary_key(session_id)) or ""
        return self._fallback.get(session_id, {}).get("summary", "")

    def _append_message(self, session_id: str, message: Dict[str, Any]):
        if self._redis:
            key = self._messages_key(session_id)
            self._redis.rpush(key, json.dumps(message, ensure_ascii=False))
            self._redis.expire(key, self.ttl_seconds)
            self._redis.expire(self._summary_key(session_id), self.ttl_seconds)
            self._redis.hset(self._meta_key(session_id), mapping={
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "ttl_seconds": str(self.ttl_seconds),
            })
            self._redis.expire(self._meta_key(session_id), self.ttl_seconds)
            return

        bucket = self._fallback.setdefault(session_id, {"messages": [], "summary": ""})
        bucket["messages"].append(message)

    def _compact_if_needed(self, session_id: str):
        messages = self.get_messages(session_id)
        total_message_tokens = sum(int(m.get("token_count", estimate_tokens(m.get("content", "")))) for m in messages)
        if len(messages) <= self.max_messages and total_message_tokens <= self.max_message_tokens:
            return

        keep_count = max(4, self.max_messages // 2)
        old_messages = messages[:-keep_count]
        recent_messages = messages[-keep_count:]
        old_text = self._format_messages(old_messages)
        existing_summary = self.get_summary(session_id)
        new_summary = self._summarize(existing_summary, old_text)
        self._set_summary(session_id, new_summary)
        self._replace_messages(session_id, recent_messages)
        print(
            "  [ShortMemory] Compacted session memory: "
            f"{len(old_messages)} old messages -> summary, {len(recent_messages)} messages kept"
        )

    def _summarize(self, existing_summary: str, old_text: str) -> str:
        if not old_text.strip():
            return truncate_by_tokens(existing_summary, self.max_summary_tokens)

        try:
            from agent.config.model_config import ModelConfig

            llm = ModelConfig.get_synthesize_llm()
            prompt = f"""Summarize the conversation memory for a RAG customer-service Agent.
Keep only information useful for future turns, such as referenced products, user preferences,
open questions, decisions, and important constraints. Do not include temporary filler.
Return Chinese if the conversation is Chinese. Keep it concise.

Existing summary:
{existing_summary or "(none)"}

Older messages:
{old_text}

Updated summary:"""
            response = llm.invoke(prompt)
            summary = response.content.strip() if hasattr(response, "content") else str(response).strip()
            if summary:
                return truncate_by_tokens(summary, self.max_summary_tokens)
        except Exception as exc:
            print(f"  [ShortMemory] LLM summary failed, using fallback summary: {exc}")

        merged = (existing_summary + "\n" + old_text).strip()
        return truncate_by_tokens(merged, self.max_summary_tokens)

    def _select_recent_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        used_tokens = 0
        for message in reversed(messages):
            content = message.get("content", "")
            tokens = int(message.get("token_count", estimate_tokens(content)))
            if selected and used_tokens + tokens > self.max_context_tokens:
                break
            selected.append({"role": message.get("role", ""), "content": content})
            used_tokens += tokens
        return list(reversed(selected))

    def _set_summary(self, session_id: str, summary: str):
        summary = truncate_by_tokens(summary, self.max_summary_tokens)
        if self._redis:
            self._redis.set(self._summary_key(session_id), summary, ex=self.ttl_seconds)
        else:
            self._fallback.setdefault(session_id, {"messages": [], "summary": ""})["summary"] = summary

    def _replace_messages(self, session_id: str, messages: List[Dict[str, Any]]):
        if self._redis:
            key = self._messages_key(session_id)
            pipe = self._redis.pipeline()
            pipe.delete(key)
            for message in messages:
                pipe.rpush(key, json.dumps(message, ensure_ascii=False))
            pipe.expire(key, self.ttl_seconds)
            pipe.execute()
        else:
            self._fallback.setdefault(session_id, {"messages": [], "summary": ""})["messages"] = list(messages)

    def _format_messages(self, messages: List[Dict[str, Any]]) -> str:
        lines = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role and content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _messages_key(self, session_id: str) -> str:
        return f"session:{session_id}:messages"

    def _summary_key(self, session_id: str) -> str:
        return f"session:{session_id}:summary"

    def _meta_key(self, session_id: str) -> str:
        return f"session:{session_id}:metadata"

