from __future__ import annotations

import argparse
import json
from pathlib import Path

from dsg_spatialqa_lab import (
    compare_qa_dataset,
    generate_qa_cases,
    load_graph_json,
    load_qa_dataset,
    qa_dataset_digest,
    qa_dataset_summary,
    save_qa_dataset,
    validate_qa_dataset,
)
from dsg_spatialqa_lab.schema import SpatialQAError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate and verify deterministic oracle SpatialQA JSONL datasets."
    )
    parser.add_argument("--graph", type=Path, help="Explicit local oracle graph JSON path.")
    parser.add_argument("--scene-id", help="Scene id to store in generated QA cases.")
    parser.add_argument("--episode-id", help="Episode id to store in generated QA cases.")
    parser.add_argument("--max-cases", type=int, help="Optional maximum generated case count.")
    parser.add_argument("--output", type=Path, help="Explicit local QA dataset JSONL output path.")
    parser.add_argument("--tag", action="append", dest="tags", help="Optional generated case tag.")
    parser.add_argument("--validate", type=Path, help="Validate an explicit local QA JSONL dataset.")
    parser.add_argument(
        "--compare",
        type=Path,
        help="Compare an explicit local QA JSONL dataset with --graph.",
    )
    args = parser.parse_args(argv)

    if args.validate is not None:
        try:
            validation = validate_qa_dataset(load_qa_dataset(args.validate))
        except (OSError, ValueError, SpatialQAError) as exc:
            payload = {
                "action": "validate_qa_dataset",
                "path": str(args.validate),
                "valid": False,
                "error": str(exc),
            }
            print(json.dumps(payload, sort_keys=True))
            return 1
        payload = {"action": "validate_qa_dataset", "path": str(args.validate), **validation}
        print(json.dumps(payload, sort_keys=True))
        return 0 if validation["valid"] is True else 1

    if args.compare is not None:
        if args.graph is None:
            parser.error("--compare requires --graph")
        try:
            comparison = compare_qa_dataset(load_qa_dataset(args.compare), load_graph_json(args.graph))
        except (OSError, ValueError, SpatialQAError) as exc:
            payload = {
                "action": "compare_qa_dataset",
                "path": str(args.compare),
                "matches": False,
                "error": str(exc),
            }
            print(json.dumps(payload, sort_keys=True))
            return 1
        payload = {"action": "compare_qa_dataset", "path": str(args.compare), **comparison}
        print(json.dumps(payload, sort_keys=True))
        return 0 if comparison["matches"] is True else 1

    if args.graph is None:
        parser.error("--graph is required when generating")
    if args.scene_id is None:
        parser.error("--scene-id is required when generating")
    if args.episode_id is None:
        parser.error("--episode-id is required when generating")
    if args.output is None:
        parser.error("--output is required when generating")

    graph = load_graph_json(args.graph)
    cases = generate_qa_cases(
        graph,
        scene_id=args.scene_id,
        episode_id=args.episode_id,
        max_cases=args.max_cases,
        tags=tuple(args.tags or ()),
    )
    save_qa_dataset(cases, args.output)
    validation = validate_qa_dataset(cases)
    payload = {
        "action": "generate_qa_dataset",
        "path": str(args.output),
        "valid": validation["valid"],
        "digest": qa_dataset_digest(cases),
        "summary": qa_dataset_summary(cases),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if validation["valid"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
