"""SQLite persistence for Agent run status, metrics and trace."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_default(value: Any):
    if hasattr(value, "page_content"):
        return {
            "page_content": value.page_content,
            "metadata": getattr(value, "metadata", {}) or {},
        }
    if isinstance(value, Path):
        return str(value)
    return str(value)


class RunStore:
    def __init__(self, path: str = "data/runtime/agent_runs.sqlite3"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
                    run_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    status TEXT NOT NULL,
                    answer TEXT NOT NULL DEFAULT '',
                    answer_source TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    trace_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_runs_session ON agent_runs(session_id, created_at)"
            )

    def create(self, run_id: str, session_id: str, question: str) -> None:
        now = _utc_now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO agent_runs
                (run_id, session_id, question, status, created_at, updated_at)
                VALUES (?, ?, ?, 'running', ?, ?)""",
                (run_id, session_id, question, now, now),
            )

    def complete(self, run_id: str, result: Dict[str, Any]) -> None:
        self._update(
            run_id,
            status="completed",
            answer=str(result.get("final_answer", "")),
            answer_source=str(result.get("answer_source", "")),
            metrics=result.get("metrics", {}),
            trace=result.get("execution_trace", []),
        )

    def fail(self, run_id: str, error: str, trace=None) -> None:
        self._update(run_id, status="failed", error=error[:2000], trace=trace or [])

    def _update(
        self,
        run_id: str,
        status: str,
        answer: str = "",
        answer_source: str = "",
        error: str = "",
        metrics=None,
        trace=None,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """UPDATE agent_runs SET status=?, answer=?, answer_source=?, error=?,
                metrics_json=?, trace_json=?, updated_at=? WHERE run_id=?""",
                (
                    status,
                    answer,
                    answer_source,
                    error,
                    json.dumps(metrics or {}, ensure_ascii=False, default=_json_default),
                    json.dumps(trace or [], ensure_ascii=False, default=_json_default),
                    _utc_now(),
                    run_id,
                ),
            )

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["metrics"] = json.loads(item.pop("metrics_json"))
        item["trace"] = json.loads(item.pop("trace_json"))
        return item

