# Agentic RAG 实施记录

本文档记录当前主链路已经实现的能力、代码边界和验收方式。README和简历中的
能力描述应以本文档与自动化测试为准。

## 总体链路

```text
Intent → Query Decomposer → Validated Planner → Tool Executor
       → Retrieval Grader → Subquery Collector → Synthesizer
       → Claim Extractor → Evidence Validator → Reflection → Finalizer
```

已完成五个阶段：

1. 检索质量判断与有限步自动重检
2. 动态工具选择与结构化计划校验
3. 复杂问题分解与多跳证据合成
4. 声明级答案校验与引用覆盖
5. 执行预算、请求级去重和可观测性

当前仍是单机工程原型。Trace、缓存与预算保存在单次请求状态中；跨进程
Checkpoint、历史回放、异步ETL和索引版本发布尚未实现。

## 1. 检索质量判断与自动重检

RAG结果先进入`Retrieval Grader`，而不是直接生成答案：

- RAG证据必须覆盖当前问题的核心事实。
- LLM评分不可用时，使用文档存在性和置信度阈值保守降级。
- 证据不足时根据原问题、实际检索词和失败原因改写查询。
- 默认最多重检2次，达到上限后切换联网搜索。
- FAQ、闲聊和联网结果显式标记为`not_applicable`，不伪造RAG评分。

核心状态包括`retrieval_grade`、`retrieval_retry_count`、
`retrieval_query_history`和`active_retrieval_query`。

对应代码：`agent/nodes/retrieval_grader.py`、`agent/graph.py`。

## 2. 动态工具选择与结构化计划

Planner输出结构化计划，但模型输出不直接获得执行权限：

```json
{
  "reasoning": "问题需要本地手册证据",
  "steps": [
    {
      "tool": "rag",
      "reason": "检索车辆手册",
      "parameters": {"query": "车辆续航里程", "top_k": 6}
    }
  ]
}
```

- 工具限制为`faq`、`rag`、`web_search`、`direct_response`和
  `human_handoff`。
- 本地校验器拒绝未知工具、未知参数、错误类型、越界值、重复工具与非法组合。
- 校验失败时执行最小安全降级计划，不执行原始模型输出。
- Tool Executor严格按校验后的步骤和参数执行。

对应代码：`agent/nodes/planner.py`、`agent/nodes/tool_executor.py`。

## 3. 复杂问题分解与多跳检索

比较、多对象、多属性和依赖型问题会被拆成2～5个可独立回答的子问题：

- 子问题必须长度合法、去重且语义完整。
- 模型输出不合法时保留原问题，不由规则凭空创造实体。
- 每个子问题独立规划、检索、评分和有限次重检。
- 已完成结果立即写入`subquery_results`，后续步骤不会重跑。
- 部分子问题失败时仍合成已有证据，并显式说明缺失信息。

对应代码：`agent/nodes/query_decomposer.py`、
`agent/nodes/subquery_coordinator.py`、`agent/nodes/multi_hop_synthesizer.py`。

## 4. 声明级答案校验与引用覆盖

系统把“检索到文档”和“答案事实被证据支持”分开验证：

- RAG和多跳结果被整理成统一编号的`evidence_catalog`。
- `claim_extractor`从草稿中抽取可核验事实声明。
- `evidence_validator`输出逐声明判定、引用与`citation_coverage`。
- 覆盖不足时只基于已有证据改写，默认最多改写1次。
- 无需文档证据的回答标记为`skipped`，不生成虚假引用。

对应代码：`agent/nodes/grounding.py`。

## 5. 执行预算、去重与请求级可观测性

- 单次请求限制工具总次数、RAG次数、联网次数和工具执行时长。
- 使用规范化工具参数构造缓存键，复用成功结果并拦截无效重复调用。
- 预算耗尽后阻止继续重规划，由Reflection进入安全结束路径。
- 节点、LLM和工具统一写入`execution_trace`。
- 汇总节点耗时、LLM耗时、Token、工具次数、缓存命中和去重次数。

对应代码：`agent/nodes/tool_executor.py`、`agent/observability.py`。

## 验收

P0离线验收不调用模型API或联网搜索：

```bash
python scripts/verify_p0.py
```

验收覆盖动态规划、结构化工具执行、证据重检、多跳工作流、答案Grounding、
执行预算和可观测性，并校验两份正式评测结果文件。更完整的测试分层见
`docs/TESTING.md`。
