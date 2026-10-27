"""Commands for the new multi-manual MinerU corpus and Qdrant index."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from multi_manual_ingest import default_inputs, parse_manual
from multi_manual_store import MultiManualIndexBuilder, MultiManualRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    parse_cmd = sub.add_parser("parse", help="Build auditable parent/child/image JSONL from MinerU archives")
    parse_cmd.add_argument("--source-root", type=Path, default=Path("data/multi_manual"))
    build_cmd = sub.add_parser("build", help="Build and activate the two Qdrant collections")
    build_cmd.add_argument("--source-root", type=Path, default=Path("data/multi_manual"))
    build_cmd.add_argument("--store-root", type=Path, default=Path("data/multi_manual_store"))
    build_cmd.add_argument("--batch-size", type=int, default=64)
    search_cmd = sub.add_parser("search", help="Search one model without cross-model leakage")
    search_cmd.add_argument("query")
    search_cmd.add_argument("--model", required=True, help="Exact model_name from the manifest")
    search_cmd.add_argument("--store-root", type=Path, default=Path("data/multi_manual_store"))
    search_cmd.add_argument("--images", action="store_true")
    search_cmd.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.command == "parse":
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        batch = now.replace(":", "").replace("+00:00", "Z")
        root = Path(__file__).resolve().parent
        for spec in default_inputs(root):
            manifest = parse_manual(spec, args.source_root, batch_id=batch, ingested_at=now)
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
    elif args.command == "build":
        manifest = MultiManualIndexBuilder(args.source_root, args.store_root).build(batch_size=args.batch_size)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        retriever = MultiManualRetriever(args.store_root)
        try:
            result = (retriever.search_images(args.query, args.model, top_k=args.top_k)
                      if args.images else retriever.search_text(args.query, args.model, top_k=args.top_k))
            for item in result:
                item.pop("parent_content", None)  # keep CLI results inspectable
            print(json.dumps(result, ensure_ascii=False, indent=2))
        finally:
            retriever.close()


if __name__ == "__main__":
    main()
