#!/usr/bin/env python
# coding: utf-8
"""Claim-level evidence validation and grounded answer rewriting."""

import json
import os
import re
from typing import Any, Dict, List, Literal

from agent.config.model_config import ModelConfig
from agent.state import AgentState
from llm_streaming import generate_text_streaming


VALID_VERDICTS = {"supported", "unsupported", "contradicted", "not_verifiable"}


def claim_extractor_node(state: AgentState) -> AgentState:
    """Extract atomic, externally verifiable claims from the current draft."""
    state["evidence_catalog"] = _build_evidence_catalog(state)
    if state.get("answer_source") in {"direct_response", "no_answer"}:
        state["claims"] = []
        state["claim_verdicts"] = []
        state["citation_coverage"] = 1.0
        state["grounding_status"] = "skipped"
        return state

    prompt = f"""请从答案中抽取最小粒度、可由外部资料验证的事实声明。
用户问题：{state.get('user_question', '')}
答案：{state.get('draft_answer', '')}

只输出 JSON：{{"claims":[{{"claim_id":"C1","text":"完整事实声明"}}]}}
要求：不要抽取标题、建议语、主观措辞或“证据不足”的说明；最多 12 条。"""
    try:
        result = _invoke_json(ModelConfig.get_reflection_llm(), prompt)
        state["claims"] = _normalize_claims(result.get("claims", []))
    except Exception as exc:
        print(f"  [Grounding] 声明抽取失败: {exc}，使用规则降级")
        state["claims"] = _fallback_claims(state.get("draft_answer", ""))
    print(f"  [Grounding] 抽取 {len(state['claims'])} 条可验证声明")
    return state


def evidence_validator_node(state: AgentState) -> AgentState:
    """Judge every claim against retrieved evidence and calculate coverage."""
    claims = state.get("claims", [])
    evidence = state.get("evidence_catalog", [])
    if state.get("grounding_status") == "skipped" or not claims:
        state["claim_verdicts"] = []
        state["citation_coverage"] = 1.0
        state["grounding_status"] = "skipped" if not claims else "grounded"
        return state
    if not evidence:
        verdicts = [_verdict(c["claim_id"], "unsupported", [], "没有可用证据") for c in claims]
    else:
        prompt = f"""你是严格的 RAG 事实校验器。只能使用给定证据判断每条声明。
声明：{json.dumps(claims, ensure_ascii=False)}
证据：{json.dumps(evidence, ensure_ascii=False, default=str)}

只输出 JSON：{{"verdicts":[{{"claim_id":"C1","verdict":"supported|unsupported|contradicted|not_verifiable","evidence_ids":["E1"],"reason":"简短原因"}}]}}
规则：supported 必须给出直接支持它的 evidence_ids；不得引用不存在的 ID；证据未提及即 unsupported；证据明确相反才是 contradicted。"""
        try:
            raw = _invoke_json(ModelConfig.get_reflection_llm(), prompt).get("verdicts", [])
            verdicts = _normalize_verdicts(claims, raw, {item["evidence_id"] for item in evidence})
        except Exception as exc:
            print(f"  [Grounding] 证据判定失败: {exc}，按不支持处理")
            verdicts = [_verdict(c["claim_id"], "not_verifiable", [], "校验模型不可用") for c in claims]

    supported = sum(item["verdict"] == "supported" for item in verdicts)
    coverage = supported / len(claims) if claims else 1.0
    state["claim_verdicts"] = verdicts
    state["citation_coverage"] = round(coverage, 4)
    state["grounding_status"] = "grounded" if supported == len(claims) else "needs_rewrite"
    state.setdefault("metadata", {})["claim_citations"] = verdicts
    print(f"  [Grounding] 引用覆盖率: {coverage:.0%} ({supported}/{len(claims)})")
    return state


def route_after_evidence_validation(state: AgentState) -> Literal["reflection", "rewrite"]:
    if state.get("grounding_status") != "needs_rewrite":
        return "reflection"
    maximum = int(os.getenv("AGENT_MAX_GROUNDING_RETRIES", "1"))
    return "rewrite" if state.get("grounding_retry_count", 0) < maximum else "reflection"


def grounded_rewriter_node(state: AgentState) -> AgentState:
    """Remove or qualify unsupported claims without fetching new information."""
    bad_ids = {v["claim_id"] for v in state.get("claim_verdicts", []) if v["verdict"] != "supported"}
    prompt = f"""请根据证据修订答案，删除或明确保留条件地表达所有无支持/冲突声明。
用户问题：{state.get('user_question', '')}
原答案：{state.get('draft_answer', '')}
需要修订的声明 ID：{sorted(bad_ids)}
声明判定：{json.dumps(state.get('claim_verdicts', []), ensure_ascii=False)}
可用证据：{json.dumps(state.get('evidence_catalog', []), ensure_ascii=False, default=str)}

要求：只输出修订后的中文答案；不得新增证据中没有的事实或数字；若证据不足，直接说明不足。"""
    try:
        rewritten = generate_text_streaming(
            ModelConfig.get_synthesize_llm(), prompt, replace=True
        )
        if rewritten:
            state["draft_answer"] = rewritten
    except Exception as exc:
        print(f"  [Grounding] 改写失败: {exc}，保留原答案")
    state["grounding_retry_count"] = state.get("grounding_retry_count", 0) + 1
    return state


def _build_evidence_catalog(state: AgentState) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    def add(source: str, content: Any, metadata: Dict[str, Any] = None):
        text = str(content or "").strip()
        if not text:
            return
        items.append({"evidence_id": f"E{len(items) + 1}", "source": source,
                      "content": text[:3000], "metadata": metadata or {}})

    if state.get("answer_source") == "multi_hop":
        for result in state.get("subquery_results", []):
            evidence = result.get("evidence") or {}
            add(result.get("source", "multi_hop"), result.get("answer", ""),
                {"subquery": result.get("question", "")})
            for document in evidence.get("documents", []) if isinstance(evidence, dict) else []:
                _add_document(add, document, "rag")
            for row in evidence.get("results", []) if isinstance(evidence, dict) else []:
                add("web_search", row.get("content") or row.get("snippet") or row.get("summary"),
                    {"title": row.get("title", ""), "url": row.get("url", "")})
        return items[:20]

    source = state.get("answer_source", "")
    if source == "faq":
        result = state.get("faq_result") or {}
        add("faq", result.get("answer"), {"question": result.get("matched_question", "")})
    elif source == "rag":
        result = state.get("rag_result") or state.get("metadata", {}).get("evidence_rag_result", {})
        for document in result.get("documents", []) or []:
            _add_document(add, document, "rag")
        if not items:
            add("rag", result.get("context", ""))
    elif source == "web_search":
        for row in (state.get("web_search_result") or {}).get("results", []):
            add("web_search", row.get("content") or row.get("snippet") or row.get("summary"),
                {"title": row.get("title", ""), "url": row.get("url", "")})
    return items[:20]


def _add_document(add, document: Any, source: str) -> None:
    if hasattr(document, "page_content"):
        add(source, document.page_content, dict(getattr(document, "metadata", {}) or {}))
    elif isinstance(document, dict):
        add(source, document.get("page_content") or document.get("content"), document.get("metadata", {}))


def _invoke_json(llm, prompt: str) -> Dict[str, Any]:
    response = llm.invoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("模型未返回 JSON 对象")
    value = json.loads(match.group())
    if not isinstance(value, dict):
        raise ValueError("模型 JSON 顶层不是对象")
    return value


def _normalize_claims(raw: Any) -> List[Dict[str, str]]:
    result = []
    for item in raw if isinstance(raw, list) else []:
        text = str(item.get("text", "")).strip() if isinstance(item, dict) else ""
        if text and text not in {x["text"] for x in result}:
            result.append({"claim_id": f"C{len(result) + 1}", "text": text[:500]})
    return result[:12]


def _fallback_claims(answer: str) -> List[Dict[str, str]]:
    parts = [x.strip(" -\n") for x in re.split(r"[。！？!?；;\n]+", answer) if len(x.strip()) >= 6]
    return [{"claim_id": f"C{i + 1}", "text": text[:500]} for i, text in enumerate(parts[:12])]


def _normalize_verdicts(claims, raw, valid_evidence_ids):
    by_id = {str(item.get("claim_id")): item for item in raw if isinstance(item, dict)}
    output = []
    for claim in claims:
        item = by_id.get(claim["claim_id"], {})
        verdict = item.get("verdict") if item.get("verdict") in VALID_VERDICTS else "not_verifiable"
        ids = [x for x in item.get("evidence_ids", []) if x in valid_evidence_ids]
        if verdict == "supported" and not ids:
            verdict = "unsupported"
        output.append(_verdict(claim["claim_id"], verdict, ids, str(item.get("reason", ""))[:300]))
    return output


def _verdict(claim_id, verdict, evidence_ids, reason):
    return {"claim_id": claim_id, "verdict": verdict, "evidence_ids": evidence_ids, "reason": reason}
