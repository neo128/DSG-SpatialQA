from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    dashboard_bundle,
    export_dashboard,
    load_dashboard_bundle,
    load_error_attribution_report,
    load_active_task_delta_report,
    load_active_task_report,
    load_graph_json,
    load_experiment_summary_report,
    load_qa_dataset,
    load_qa_eval_report,
    load_qa_predictions,
    validate_dashboard_bundle,
)
from dsg_spatialqa_lab.schema import SpatialQAError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Export a deterministic static dashboard bundle for SpatialQA review.",
    )
    parser.add_argument("--qa", type=Path, help="Explicit local QA JSONL path.")
    parser.add_argument(
        "--pred",
        type=Path,
        help="Explicit local QA prediction JSONL path.",
    )
    parser.add_argument(
        "--eval-report",
        type=Path,
        help="Explicit local QA eval report JSON path.",
    )
    parser.add_argument("--graph", type=Path, help="Explicit local graph JSON path.")
    parser.add_argument(
        "--error-attribution",
        type=Path,
        help="Optional explicit local error attribution report JSON path.",
    )
    parser.add_argument(
        "--active-task-report",
        type=Path,
        help="Optional explicit local active task report JSON path.",
    )
    parser.add_argument(
        "--active-task-delta-report",
        type=Path,
        help="Optional explicit local active task delta report JSON path.",
    )
    parser.add_argument(
        "--experiment-summary-report",
        type=Path,
        help="Optional explicit local experiment summary report JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Explicit local dashboard output directory.",
    )
    parser.add_argument(
        "--validate-bundle",
        type=Path,
        help="Validate an explicit dashboard bundle JSON path.",
    )
    args = parser.parse_args(argv)

    if args.validate_bundle is not None:
        try:
            bundle = load_dashboard_bundle(args.validate_bundle)
            validation = validate_dashboard_bundle(bundle)
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(_error_payload("validate_dashboard_bundle", args.validate_bundle, exc))
            return 1

        _emit_json(
            {
                "action": "validate_dashboard_bundle",
                "path": str(args.validate_bundle),
                "valid": validation["valid"],
                "digest": bundle.get("bundle_digest"),
            }
        )
        return 0 if validation["valid"] is True else 1

    missing_required = [
        flag
        for flag, value in (
            ("--qa", args.qa),
            ("--pred", args.pred),
            ("--eval-report", args.eval_report),
            ("--graph", args.graph),
            ("--output", args.output),
        )
        if value is None
    ]
    if missing_required:
        parser.error(
            "the following arguments are required: " + ", ".join(missing_required)
        )

    try:
        attribution_report = (
            load_error_attribution_report(args.error_attribution)
            if args.error_attribution is not None
            else None
        )
        active_task_report = (
            load_active_task_report(args.active_task_report)
            if args.active_task_report is not None
            else None
        )
        active_task_delta_report = (
            load_active_task_delta_report(args.active_task_delta_report)
            if args.active_task_delta_report is not None
            else None
        )
        experiment_summary_report = (
            load_experiment_summary_report(args.experiment_summary_report)
            if args.experiment_summary_report is not None
            else None
        )
        bundle = dashboard_bundle(
            load_qa_dataset(args.qa),
            predictions=load_qa_predictions(args.pred),
            qa_eval_report=load_qa_eval_report(args.eval_report),
            graph=load_graph_json(args.graph),
            error_attribution_report=attribution_report,
            active_task_report=active_task_report,
            active_task_delta_report=active_task_delta_report,
            experiment_summary_report=experiment_summary_report,
            qa_path=args.qa,
            prediction_path=args.pred,
            qa_eval_report_path=args.eval_report,
            graph_path=args.graph,
            error_attribution_report_path=args.error_attribution,
            active_task_report_path=args.active_task_report,
            active_task_delta_report_path=args.active_task_delta_report,
            experiment_summary_report_path=args.experiment_summary_report,
        )
        export_result = export_dashboard(bundle, args.output)
        validation = validate_dashboard_bundle(bundle)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(_error_payload("export_dashboard", args.output, exc))
        return 1

    payload = {
        "action": "export_dashboard",
        "path": str(args.output),
        "valid": validation["valid"],
        "digest": export_result["digest"],
        "summary": export_result["summary"],
        "bundle_path": export_result["bundle_path"],
        "index_path": export_result["index_path"],
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
