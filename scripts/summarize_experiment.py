from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    SpatialQAError,
    compare_experiment_summary_report,
    experiment_summary_report,
    experiment_summary_report_digest,
    load_benchmark_manifest,
    load_experiment_summary_report,
    save_experiment_summary_report,
    validate_experiment_summary_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Summarize deterministic experiment artifacts recorded by a benchmark manifest.",
    )
    parser.add_argument("--manifest", type=Path, help="Explicit benchmark manifest path.")
    parser.add_argument("--report", type=Path, help="Explicit experiment summary output path.")
    parser.add_argument(
        "--validate-report",
        type=Path,
        help="Validate an experiment summary report.",
    )
    parser.add_argument(
        "--compare-report",
        type=Path,
        help="Compare an experiment summary report with current manifest artifacts.",
    )
    args = parser.parse_args(argv)

    if args.validate_report is not None:
        try:
            validation = validate_experiment_summary_report(
                load_experiment_summary_report(args.validate_report)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_experiment_summary_report",
                    args.validate_report,
                    exc,
                )
            )
            return 1
        _emit_json(
            {
                "action": "validate_experiment_summary_report",
                "path": str(args.validate_report),
                **validation,
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.compare_report is not None:
        try:
            comparison = compare_experiment_summary_report(
                load_experiment_summary_report(args.compare_report)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                {
                    **_error_payload(
                        "compare_experiment_summary_report",
                        args.compare_report,
                        exc,
                    ),
                    "matches": False,
                }
            )
            return 1
        _emit_json(
            {
                "action": "compare_experiment_summary_report",
                "path": str(args.compare_report),
                **comparison,
            }
        )
        return 0 if comparison["matches"] is True else 1

    if args.manifest is None:
        parser.error("--manifest is required")
    if args.report is None:
        parser.error("--report is required")

    try:
        report = experiment_summary_report(
            load_benchmark_manifest(args.manifest),
            manifest_path=args.manifest,
        )
        save_experiment_summary_report(report, args.report)
        validation = validate_experiment_summary_report(report)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(_error_payload("experiment_summary_report", args.report, exc))
        return 1

    _emit_json(
        {
            "action": "experiment_summary_report",
            "path": str(args.report),
            "valid": validation["valid"],
            "digest": experiment_summary_report_digest(report),
            "readiness": report["readiness"],
            "summary": report["summary"],
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
