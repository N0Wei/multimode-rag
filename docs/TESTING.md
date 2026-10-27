# 测试与验收

项目将测试按运行依赖分为三层，避免把需要密钥或本地模型的用例误当成稳定的
离线回归测试。

## 1. P0 离线核心回归

无需模型 API、联网搜索或本地向量模型：

```bash
python scripts/verify_p0.py
```

该命令会校验两份正式评测结果文件，并执行动态规划、结构化工具调用、证据
评分、多跳工作流、答案 Grounding、执行预算和可观测性测试。

## 2. 本地集成测试

这类测试可能读取知识库、FAISS 索引或加载本地依赖：

```bash
python -m pytest -q \
  test_text_parent_child_chunking.py \
  test_multimodal_nearby_text.py \
  test_multimodal_retriever.py
```

运行前必须安装 `requirements-dev.txt` 中的完整依赖。测试使用的知识库与索引
版本应记录在评测结果的 `config` 字段中。

## 3. 外部服务测试

文件名包含 `api`、`bocha`、`all_models` 的测试可能产生费用或需要网络和密钥，
不包含在默认离线验收中。仅在明确配置测试账号后单独运行。

## 指标口径

- 文本/表格正式结果：
  `data/evaluation/results/text_table_retrieval_metrics.json`
- 图片对照实验：
  `data/evaluation/results/image_retrieval_comparison.json`
- 旧版 205 题实验仅用于历史对照，不能与当前 100 题父子块评测混写。

简历或 README 中出现的数值必须能够追溯到上述结果文件，并同时说明数据规模、
基线、指标绝对值和延迟代价。
