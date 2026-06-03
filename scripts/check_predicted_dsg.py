from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    SpatialQAError,
    compare_predicted_dsg_evidence_report,
    load_predicted_dsg_evidence_report,
    load_predicted_graph_report,
    predicted_dsg_evidence_report,
    save_predicted_dsg_evidence_report,
    validate_predicted_dsg_evidence_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Check whether a predicted DSG report is backed by explicit "
            "RGB-D/detector observation evidence."
        ),
    )
    parser.add_argument("--predicted-report", type=Path)
    parser.add_argument("--observation-sequence", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--min-observation-count", type=int, default=2)
    parser.add_argument("--min-object-observation-count", type=int, default=2)
    parser.add_argument(
        "--required-evidence-kind",
        action="append",
        dest="required_evidence_kinds",
    )
    parser.add_argument("--validate-report", type=Path)
    parser.add_argument("--compare-report", type=Path)
    args = parser.parse_args(argv)

    if args.validate_report is not None:
        try:
            validation = validate_predicted_dsg_evidence_report(
                load_predicted_dsg_evidence_report(args.validate_report)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_predicted_dsg_evidence_report",
                    args.validate_report,
                    exc,
                )
            )
            return 1
        _emit_json(
            {
                "action": "validate_predicted_dsg_evidence_report",
                "path": str(args.validate_report),
                **validation,
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.compare_report is not None:
        try:
            comparison = compare_predicted_dsg_evidence_report(
                load_predicted_dsg_evidence_report(args.compare_report)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                {
                    **_error_payload(
                        "compare_predicted_dsg_evidence_report",
                        args.compare_report,
                        exc,
                    ),
                    "matches": False,
                }
            )
            return 1
        _emit_json(
            {
                "action": "compare_predicted_dsg_evidence_report",
                "path": str(args.compare_report),
                **comparison,
            }
        )
        return 0 if comparison["matches"] is True else 1

    if args.predicted_report is None:
        _emit_json(
            _missing_argument_payload(
                "--predicted-report",
                args.report,
            )
        )
        return 1
    if args.report is None:
        _emit_json(_missing_argument_payload("--report", None))
        return 1

    try:
        report = predicted_dsg_evidence_report(
            load_predicted_graph_report(args.predicted_report),
            predicted_graph_report_path=args.predicted_report,
            observation_sequence_path=args.observation_sequence,
            min_observation_count=args.min_observation_count,
            min_object_observation_count=args.min_object_observation_count,
            required_evidence_kinds=tuple(
                args.required_evidence_kinds or ("depth", "detector", "rgb")
            ),
        )
        save_predicted_dsg_evidence_report(report, args.report)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(_error_payload("predicted_dsg_evidence", args.report, exc))
        return 1

    _emit_json(
        {
            "action": "predicted_dsg_evidence",
            "path": str(args.report),
            "ready": report["readiness"]["ready"],
            "report_digest": report["report_digest"],
            "readiness": report["readiness"],
            "summary": report["evidence_summary"],
        }
    )
    return 0 if report["readiness"]["ready"] is True else 1


def _error_payload(action: str, path: Path, error: Exception) -> dict[str, Any]:
    return {
        "action": action,
        "path": str(path),
        "valid": False,
        "error": str(error),
    }


def _missing_argument_payload(argument: str, path: Path | None) -> dict[str, Any]:
    return {
        "action": "predicted_dsg_evidence",
        "path": str(path) if path is not None else "",
        "valid": False,
        "error": f"{argument} is required",
    }


def _emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
