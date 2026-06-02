from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    SpatialQAError,
    compare_experiment_record,
    experiment_record,
    experiment_record_digest,
    load_dashboard_bundle,
    load_experiment_record,
    load_experiment_summary_report,
    save_experiment_record,
    validate_experiment_record,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Record deterministic final experiment readiness and verdict evidence.",
    )
    parser.add_argument(
        "--summary-report",
        type=Path,
        help="Explicit experiment summary report JSON path.",
    )
    parser.add_argument(
        "--dashboard-bundle",
        type=Path,
        help="Optional explicit dashboard bundle JSON path.",
    )
    parser.add_argument("--record", type=Path, help="Explicit experiment record output path.")
    parser.add_argument(
        "--validate-record",
        type=Path,
        help="Validate an experiment record.",
    )
    parser.add_argument(
        "--compare-record",
        type=Path,
        help="Compare an experiment record with current summary/dashboard artifacts.",
    )
    args = parser.parse_args(argv)

    if args.validate_record is not None:
        try:
            validation = validate_experiment_record(
                load_experiment_record(args.validate_record)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload("validate_experiment_record", args.validate_record, exc)
            )
            return 1
        _emit_json(
            {
                "action": "validate_experiment_record",
                "path": str(args.validate_record),
                **validation,
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.compare_record is not None:
        try:
            comparison = compare_experiment_record(
                load_experiment_record(args.compare_record)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                {
                    **_error_payload("compare_experiment_record", args.compare_record, exc),
                    "matches": False,
                }
            )
            return 1
        _emit_json(
            {
                "action": "compare_experiment_record",
                "path": str(args.compare_record),
                **comparison,
            }
        )
        return 0 if comparison["matches"] is True else 1

    if args.summary_report is None:
        parser.error("--summary-report is required")
    if args.record is None:
        parser.error("--record is required")

    try:
        summary_report = load_experiment_summary_report(args.summary_report)
        dashboard_bundle = (
            load_dashboard_bundle(args.dashboard_bundle)
            if args.dashboard_bundle is not None
            else None
        )
        record = experiment_record(
            summary_report,
            summary_report_path=args.summary_report,
            dashboard_bundle=dashboard_bundle,
            dashboard_bundle_path=args.dashboard_bundle,
        )
        save_experiment_record(record, args.record)
        validation = validate_experiment_record(record)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(_error_payload("experiment_record", args.record, exc))
        return 1

    _emit_json(
        {
            "action": "experiment_record",
            "path": str(args.record),
            "valid": validation["valid"],
            "digest": experiment_record_digest(record),
            "readiness_status": record["readiness_status"],
            "verdict_counts": record["verdict_counts"],
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
