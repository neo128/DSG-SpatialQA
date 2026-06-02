from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    compare_error_attribution_report,
    error_attribution_report,
    error_attribution_report_digest,
    load_error_attribution_report,
    load_graph_json,
    load_qa_dataset,
    load_qa_predictions,
    save_error_attribution_report,
    validate_error_attribution_report,
)
from dsg_spatialqa_lab.schema import SpatialQAError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Attribute deterministic SpatialQA errors across oracle graph, predicted graph, and predictions.",
    )
    parser.add_argument("--gold", type=Path, help="Explicit local gold QA JSONL dataset.")
    parser.add_argument("--oracle-graph", type=Path, help="Explicit local oracle graph JSON path.")
    parser.add_argument(
        "--predicted-graph",
        type=Path,
        help="Explicit local predicted graph JSON path.",
    )
    parser.add_argument("--predictions", type=Path, help="Explicit local QA prediction JSONL path.")
    parser.add_argument("--report", type=Path, help="Explicit local attribution report output path.")
    parser.add_argument(
        "--validate-report",
        type=Path,
        help="Validate an explicit error attribution report.",
    )
    parser.add_argument(
        "--compare-report",
        type=Path,
        help="Compare an explicit error attribution report with current files.",
    )
    args = parser.parse_args(argv)

    if args.validate_report is not None:
        try:
            validation = validate_error_attribution_report(
                load_error_attribution_report(args.validate_report)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_error_attribution_report",
                    args.validate_report,
                    exc,
                )
            )
            return 1
        payload = {
            "action": "validate_error_attribution_report",
            "path": str(args.validate_report),
            **validation,
        }
        _emit_json(payload)
        return 0 if validation["valid"] is True else 1

    if args.compare_report is not None:
        try:
            comparison = compare_error_attribution_report(
                load_error_attribution_report(args.compare_report)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            payload = {
                **_error_payload(
                    "compare_error_attribution_report",
                    args.compare_report,
                    exc,
                ),
                "matches": False,
            }
            _emit_json(payload)
            return 1
        payload = {
            "action": "compare_error_attribution_report",
            "path": str(args.compare_report),
            **comparison,
        }
        _emit_json(payload)
        return 0 if comparison["matches"] is True else 1

    if args.gold is None:
        parser.error("--gold is required when generating a report")
    if args.oracle_graph is None:
        parser.error("--oracle-graph is required when generating a report")
    if args.predicted_graph is None:
        parser.error("--predicted-graph is required when generating a report")
    if args.predictions is None:
        parser.error("--predictions is required when generating a report")
    if args.report is None:
        parser.error("--report is required when generating a report")

    try:
        report = error_attribution_report(
            load_qa_dataset(args.gold),
            oracle_graph=load_graph_json(args.oracle_graph),
            predicted_graph=load_graph_json(args.predicted_graph),
            predictions=load_qa_predictions(args.predictions),
            gold_path=args.gold,
            oracle_graph_path=args.oracle_graph,
            predicted_graph_path=args.predicted_graph,
            prediction_path=args.predictions,
        )
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(_error_payload("error_attribution_report", args.report, exc))
        return 1

    save_error_attribution_report(report, args.report)
    validation = validate_error_attribution_report(report)
    payload = {
        "action": "error_attribution_report",
        "path": str(args.report),
        "valid": validation["valid"],
        "digest": error_attribution_report_digest(report),
        "summary": report["summary"],
    }
    _emit_json(payload)
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
