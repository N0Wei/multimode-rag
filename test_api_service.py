from fastapi.testclient import TestClient

from agent.run_store import RunStore
from agent.runtime import AgentRuntime
from app_api import create_app


class FakeGraph:
    def invoke(self, state, config=None):
        state["final_answer"] = f"已处理：{state['user_question']}"
        state["answer_source"] = "rag"
        state["confidence"] = 0.91
        state["citation_coverage"] = 1.0
        state["metrics"] = {"total_duration_ms": 12.5, "tool_calls": 1}
        state["execution_trace"] = [
            {"event_type": "tool", "name": "rag", "success": True, "duration_ms": 3.2}
        ]
        return state


class FakeMemory:
    def __init__(self):
        self.exchanges = []

    def build_context(self, session_id):
        return {
            "messages": [],
            "summary": "",
            "memory_tokens": 0,
            "redis_enabled": False,
        }

    def append_exchange(self, session_id, question, answer):
        self.exchanges.append((session_id, question, answer))

    def clear(self, session_id):
        self.exchanges = [item for item in self.exchanges if item[0] != session_id]


def test_chat_persists_completed_run(tmp_path):
    store = RunStore(str(tmp_path / "runs.sqlite3"))
    memory = FakeMemory()
    client = TestClient(
        create_app(runtime=AgentRuntime(graph=FakeGraph()), run_store=store, memory=memory)
    )

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ready"

    response = client.post(
        "/chat",
        json={"question": "如何开启前雨刮？", "session_id": "session-1"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer_source"] == "rag"
    assert payload["citation_coverage"] == 1.0
    assert memory.exchanges[0][0] == "session-1"

    persisted = client.get(f"/runs/{payload['run_id']}")
    assert persisted.status_code == 200
    run = persisted.json()
    assert run["status"] == "completed"
    assert run["metrics"]["tool_calls"] == 1
    assert run["trace"][0]["name"] == "rag"


def test_chat_persists_failure_and_returns_run_id(tmp_path):
    class FailingGraph:
        def invoke(self, state, config=None):
            raise TimeoutError("reranker timeout")

    store = RunStore(str(tmp_path / "runs.sqlite3"))
    client = TestClient(
        create_app(runtime=AgentRuntime(graph=FailingGraph()), run_store=store, memory=FakeMemory())
    )

    response = client.post("/chat", json={"question": "测试失败路径"})
    assert response.status_code == 503
    detail = response.json()["detail"]
    run = store.get(detail["run_id"])
    assert run["status"] == "failed"
    assert "reranker timeout" in run["error"]


def test_unknown_run_returns_404(tmp_path):
    client = TestClient(
        create_app(
            runtime=AgentRuntime(graph=FakeGraph()),
            run_store=RunStore(str(tmp_path / "runs.sqlite3")),
            memory=FakeMemory(),
        )
    )
    assert client.get("/runs/missing").status_code == 404
