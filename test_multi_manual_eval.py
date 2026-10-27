from pathlib import Path

from multi_manual_eval import evaluate, load_cases


def test_bundled_gold_ids_exist_in_parsed_corpus():
    root = Path(__file__).resolve().parent
    cases = load_cases(root / "data/eval/multi_manual_smoke.jsonl")
    if not (root / "data/multi_manual").exists():
        return  # this check applies after the optional local MinerU parse
    import json

    available = {}
    for folder in (root / "data/multi_manual").iterdir():
        if not folder.is_dir():
            continue
        for lane, filename, key in (("text", "children.jsonl", "chunk_id"),
                                    ("image", "images.jsonl", "image_id")):
            for line in (folder / filename).read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                available[(lane, record[key])] = record
    assert len(cases) == 15
    for case in cases:
        for gold_id in case["gold_ids"]:
            assert available[(case["lane"], gold_id)]["metadata"]["model_name"] == case["model_name"]


def test_retrieval_metrics_distinguish_child_and_parent_hits():
    class FakeRetriever:
        children = {
            "gold": {"metadata": {"model_name": "车 A", "parent_id": "parent"}},
        }
        images = {}

        def search_text(self, query, model_name, top_k):
            return [{"chunk_id": "sibling", "parent_id": "parent", "metadata": {"model_name": "车 A"}}]

    case = [{"id": "one", "lane": "text", "query": "问题", "model_name": "车 A", "gold_ids": ["gold"]}]
    result = evaluate(FakeRetriever(), case, top_k=1)
    row = result["cases"][0]
    assert row["recall_at_k"] == 0.0
    assert row["parent_recall_at_k"] is True
    assert row["cross_model_leakage"] == []
