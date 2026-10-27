from __future__ import annotations

import agent.nodes.tool_executor as executor
from agent.tools.multi_manual_rag_tool import MultiManualRAGTool


class FakeLLM:
    def infer(self, prompts):
        return ["宋 Ultra EV：根据第25页，安全气囊在特定碰撞条件下可能展开。"]


class FakeRetriever:
    def search_text(self, query, model_name, top_k=2):
        return [{"content": "安全气囊可能不会展开的情况：①撞到柱子。", "ranks": {"vector": 1},
                 "metadata": {"model_name": model_name, "doc_id": "manual-1",
                              "section_title": "安全气囊触发条件及注意事项", "chunk_type": "text_child",
                              "page_start": 25, "page_end": 25, "chunk_id": "child-1"}}]

    def search_images(self, query, model_name, top_k=2):
        return [{"image_id": "image-1", "nearby_text": "图旁说明", "ranks": {"visual": 1},
                 "metadata": {"model_name": model_name, "doc_id": "manual-1",
                              "section_title": "安全气囊", "page_number": 25,
                              "image_path": "test.png"}}]


def test_tool_uses_selected_model_and_never_invents_visual_description():
    tool = MultiManualRAGTool(FakeRetriever(), FakeLLM())
    text = tool.search_and_generate("安全气囊可能不会展开吗", model_names=["宋 Ultra EV"])
    assert text["selected_models"] == ["宋 Ultra EV"]
    assert "PDF 物理页：25" in text["context"]
    visual = tool.search_and_generate("安全气囊图片", model_names=["宋 Ultra EV"])
    assert visual["image_paths"] == ["test.png"]
    assert "不能据此断定图片中的全部细节" in visual["answer"]


def test_agentic_manual_scope_never_calls_unscoped_faq():
    calls = []

    class ScopedTool:
        def search_and_generate(self, query, top_k=6, max_context_length=4000, model_names=None):
            calls.append((query, model_names))
            return {"answer": "有证据", "documents": [object()], "context": "证据",
                    "confidence": 0.0, "is_no_answer": False}

    class FAQ:
        def match(self, query, threshold=0.85):
            raise AssertionError("Scoped manual queries must never call FAQ")

    executor.set_tools(None, FAQ(), None)
    executor.set_multi_manual_tool(ScopedTool())
    state = {"user_question": "安全气囊", "query_rewrite": "安全气囊", "plan": ["faq"],
             "structured_plan": {"steps": [{"tool": "faq", "parameters": {"query": "安全气囊"}}]},
             "metadata": {"manual_models": ["宋 Ultra EV"]}}
    result = executor.tool_execution_node(state)
    assert calls == [("安全气囊", ["宋 Ultra EV"])]
    assert result["answer_source"] == "rag"
    executor.set_tools(None, None, None)


def test_visual_request_bypasses_agentic_even_when_toggled(monkeypatch):
    import app_gradio

    def plain(message, session_id, manual_models):
        assert message == "查看安全气囊图片"
        assert manual_models == ["宋 Ultra EV"]
        yield "图片证据", "普通 RAG", [], ""

    def agentic(*args, **kwargs):
        raise AssertionError("Visual query must not enter Agentic execution")

    monkeypatch.setattr(app_gradio, "_chat_with_multi_manual_plain", plain)
    monkeypatch.setattr(app_gradio, "_chat_with_agent_text", agentic)
    result = list(app_gradio.chat_with_agent("查看安全气囊图片", [], manual_models=["宋 Ultra EV"]))
    assert result[0][0] == "图片证据"
    assert "不经过 Agentic 编排" in result[0][1]


def test_multi_manual_citation_displays_model_manual_and_physical_page():
    import app_gradio
    from langchain_core.documents import Document

    document = Document(page_content="动力电池位于车身地板下", metadata={
        "topic_title": "动力电池", "doc_type": "text", "pages": [126],
        "model_name": "宋 Ultra EV", "doc_id": "manual-song",
    })
    citation = app_gradio._format_rag_citations({"answer_source": "rag", "rag_result": {"documents": [document]}})
    assert "宋 Ultra EV" in citation
    assert "manual-song" in citation
    assert "PDF 物理页" in citation


def test_dropdown_selection_is_required_and_only_active_models_are_allowed(tmp_path, monkeypatch):
    import json
    import pytest
    import app_gradio
    from multi_manual_store import list_active_manual_models

    version = tmp_path / "versions" / "active-v1"
    version.mkdir(parents=True)
    (tmp_path / "active.json").write_text(json.dumps({"batch_id": "batch-1", "version_id": "active-v1"}), encoding="utf-8")
    (version / "manifest.json").write_text(json.dumps({"manuals": [
        {"model_name": "宋 Ultra EV"}, {"model_name": "秦 MAX EV"},
    ]}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(app_gradio, "MULTI_MANUAL_STORE", tmp_path)

    assert list_active_manual_models(tmp_path) == ["宋 Ultra EV", "秦 MAX EV"]
    with pytest.raises(ValueError, match="下拉框"):
        app_gradio._check_selected_manuals("比较宋 Ultra EV 和秦 MAX EV 的安全气囊", [])
    assert app_gradio._check_selected_manuals("钥匙类型有什么区别？", [
        "宋 Ultra EV", "秦 MAX EV",
    ]) == ["宋 Ultra EV", "秦 MAX EV"]
    # Question text is deliberately not parsed yet; the dropdown is the sole scope.
    assert app_gradio._check_selected_manuals("秦 MAX EV 的电池", ["宋 Ultra EV"]) == ["宋 Ultra EV"]
    with pytest.raises(ValueError, match="活动索引"):
        app_gradio._check_selected_manuals("电池", ["汉 EV 闪充版"])
