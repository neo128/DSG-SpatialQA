from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    baseline_spec_to_dict,
    list_baselines,
    load_graph_json,
    load_qa_dataset,
    qa_predictions_digest,
    run_baseline_predictions,
    save_qa_predictions,
)
from dsg_spatialqa_lab.schema import SpatialQAError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Run deterministic local SpatialQA baselines over explicit QA JSONL files.",
    )
    parser.add_argument("--list-baselines", action="store_true", help="List available baselines.")
    parser.add_argument("--baseline", help="Baseline name to run.")
    parser.add_argument("--graph", type=Path, help="Explicit local graph JSON path.")
    parser.add_argument("--qa", type=Path, help="Explicit local QA JSONL path.")
    parser.add_argument("--pred", type=Path, help="Explicit local prediction JSONL output path.")
    args = parser.parse_args(argv)

    if args.list_baselines:
        _emit_json(
            {
                "action": "list_baselines",
                "baselines": [baseline_spec_to_dict(spec) for spec in list_baselines()],
            }
        )
        return 0

    if args.baseline is None:
        parser.error("--baseline is required")
    if args.graph is None:
        parser.error("--graph is required")
    if args.qa is None:
        parser.error("--qa is required")
    if args.pred is None:
        parser.error("--pred is required")

    try:
        predictions = run_baseline_predictions(
            args.baseline,
            graph=load_graph_json(args.graph),
            cases=load_qa_dataset(args.qa),
        )
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(
            {
                "action": "run_baseline",
                "baseline": args.baseline,
                "path": str(args.pred),
                "valid": False,
                "error": str(exc),
            }
        )
        return 1

    save_qa_predictions(predictions, args.pred)
    _emit_json(
        {
            "action": "run_baseline",
            "baseline": args.baseline,
            "path": str(args.pred),
            "prediction_count": len(predictions),
            "digest": qa_predictions_digest(predictions),
        }
    )
    return 0


def _emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
