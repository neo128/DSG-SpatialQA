from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    compare_graph_eval_report,
    graph_eval_report,
    graph_eval_report_digest,
    load_graph_eval_report,
    load_graph_json,
    save_graph_eval_report,
    validate_graph_eval_report,
)
from dsg_spatialqa_lab.schema import SpatialQAError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Evaluate deterministic predicted DSG graph JSON against oracle graph JSON.",
    )
    parser.add_argument("--oracle", type=Path, help="Explicit local oracle graph JSON path.")
    parser.add_argument("--predicted", type=Path, help="Explicit local predicted graph JSON path.")
    parser.add_argument("--report", type=Path, help="Explicit local graph eval report output path.")
    parser.add_argument(
        "--matching",
        choices=("exact", "label_center", "label_center_room"),
        default="exact",
        help="Object/relation matching strategy.",
    )
    parser.add_argument(
        "--center-distance-threshold",
        type=float,
        default=0.25,
        help="Maximum object center distance for label_center and label_center_room matching.",
    )
    parser.add_argument("--validate-report", type=Path, help="Validate an explicit graph eval report.")
    parser.add_argument(
        "--compare-report",
        type=Path,
        help="Compare an explicit graph eval report with current graph files.",
    )
    args = parser.parse_args(argv)

    if args.validate_report is not None:
        try:
            validation = validate_graph_eval_report(load_graph_eval_report(args.validate_report))
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(_error_payload("validate_graph_eval_report", args.validate_report, exc))
            return 1
        _emit_json(
            {
                "action": "validate_graph_eval_report",
                "path": str(args.validate_report),
                **validation,
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.compare_report is not None:
        try:
            comparison = compare_graph_eval_report(load_graph_eval_report(args.compare_report))
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            payload = {
                **_error_payload("compare_graph_eval_report", args.compare_report, exc),
                "matches": False,
            }
            _emit_json(payload)
            return 1
        _emit_json(
            {
                "action": "compare_graph_eval_report",
                "path": str(args.compare_report),
                **comparison,
            }
        )
        return 0 if comparison["matches"] is True else 1

    if args.oracle is None:
        parser.error("--oracle is required when generating a report")
    if args.predicted is None:
        parser.error("--predicted is required when generating a report")
    if args.report is None:
        parser.error("--report is required when generating a report")

    try:
        report = graph_eval_report(
            load_graph_json(args.oracle),
            load_graph_json(args.predicted),
            oracle_path=args.oracle,
            predicted_path=args.predicted,
            center_distance_threshold=args.center_distance_threshold,
            matching=args.matching,
        )
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(_error_payload("graph_eval_report", args.report, exc))
        return 1

    save_graph_eval_report(report, args.report)
    validation = validate_graph_eval_report(report)
    _emit_json(
        {
            "action": "graph_eval_report",
            "path": str(args.report),
            "valid": validation["valid"],
            "digest": graph_eval_report_digest(report),
            "summary": report["summary"],
            "metrics": report["metrics"],
        }
    )
    return 0 if validation["valid"] is True else 1


def _error_payload(action: str, path: Path, error: Exception) -> dict[str, Any]:
    return {
        "action": action,
        "path": str(path),
        "valid": False,
        "error": str(error),
    }


def _emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
