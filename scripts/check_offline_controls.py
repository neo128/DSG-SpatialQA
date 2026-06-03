from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    SpatialQAError,
    compare_offline_control_artifact_contracts,
    compare_offline_control_import_run_ledger,
    compare_offline_control_matrix_report,
    compare_offline_control_result_report,
    load_offline_control_artifact_contracts,
    load_offline_control_import_run_ledger,
    load_offline_control_matrix_report,
    load_offline_control_result_report,
    load_offline_prediction_import_report,
    offline_control_artifact_launch_report,
    offline_control_matrix_report,
    save_offline_control_matrix_report,
    validate_offline_control_artifact_contracts,
    validate_offline_control_import_run_ledger,
    validate_offline_control_matrix_report,
    validate_offline_control_result_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Check whether local offline prediction imports cover the required "
            "real DSG-vs-control baseline matrix."
        ),
    )
    parser.add_argument(
        "--import-report",
        "--offline-prediction-import-report",
        action="append",
        type=Path,
        dest="import_reports",
        help="Explicit offline prediction import report path. May be repeated.",
    )
    parser.add_argument("--report", type=Path, help="Explicit matrix report output path.")
    parser.add_argument(
        "--required-source-kind",
        action="append",
        dest="required_source_kinds",
        help="Required offline control source kind. May be repeated.",
    )
    parser.add_argument("--validate-report", type=Path)
    parser.add_argument("--compare-report", type=Path)
    parser.add_argument("--validate-result-report", type=Path)
    parser.add_argument("--compare-result-report", type=Path)
    parser.add_argument("--validate-artifact-contracts", type=Path)
    parser.add_argument("--compare-artifact-contracts", type=Path)
    parser.add_argument("--artifact-launch-report", type=Path)
    parser.add_argument("--validate-run-ledger", type=Path)
    parser.add_argument("--compare-run-ledger", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Offline control import manifest used when comparing saved "
            "artifact contracts against current preflight output."
        ),
    )
    args = parser.parse_args(argv)

    if args.validate_run_ledger is not None:
        try:
            validation = validate_offline_control_import_run_ledger(
                load_offline_control_import_run_ledger(args.validate_run_ledger)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_offline_control_import_run_ledger",
                    args.validate_run_ledger,
                    exc,
                )
            )
            return 1
        _emit_json(
            {
                "action": "validate_offline_control_import_run_ledger",
                "path": str(args.validate_run_ledger),
                **validation,
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.compare_run_ledger is not None:
        try:
            comparison = compare_offline_control_import_run_ledger(
                load_offline_control_import_run_ledger(args.compare_run_ledger)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                {
                    **_error_payload(
                        "compare_offline_control_import_run_ledger",
                        args.compare_run_ledger,
                        exc,
                    ),
                    "matches": False,
                }
            )
            return 1
        _emit_json(
            {
                "action": "compare_offline_control_import_run_ledger",
                "path": str(args.compare_run_ledger),
                **comparison,
            }
        )
        return 0 if comparison["matches"] is True else 1

    if args.validate_artifact_contracts is not None:
        try:
            validation = validate_offline_control_artifact_contracts(
                load_offline_control_artifact_contracts(
                    args.validate_artifact_contracts
                )
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_offline_control_artifact_contracts",
                    args.validate_artifact_contracts,
                    exc,
                )
            )
            return 1
        _emit_json(
            {
                "action": "validate_offline_control_artifact_contracts",
                "path": str(args.validate_artifact_contracts),
                **validation,
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.compare_artifact_contracts is not None:
        if args.manifest is None:
            parser.error("--manifest is required with --compare-artifact-contracts")
        try:
            comparison = compare_offline_control_artifact_contracts(
                load_offline_control_artifact_contracts(
                    args.compare_artifact_contracts
                ),
                args.manifest,
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                {
                    **_error_payload(
                        "compare_offline_control_artifact_contracts",
                        args.compare_artifact_contracts,
                        exc,
                    ),
                    "matches": False,
                }
            )
            return 1
        _emit_json(
            {
                "action": "compare_offline_control_artifact_contracts",
                "path": str(args.compare_artifact_contracts),
                "manifest_path": str(args.manifest),
                **comparison,
            }
        )
        return 0 if comparison["matches"] is True else 1

    if args.artifact_launch_report is not None:
        if args.manifest is None:
            parser.error("--manifest is required with --artifact-launch-report")
        try:
            report = offline_control_artifact_launch_report(
                load_offline_control_artifact_contracts(args.artifact_launch_report),
                manifest_path=args.manifest,
                contracts_path=args.artifact_launch_report,
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "offline_control_artifact_launch_report",
                    args.artifact_launch_report,
                    exc,
                )
            )
            return 1
        _emit_json(report)
        return 0 if report["ready_to_import"] is True else 1

    if args.manifest is not None:
        parser.error(
            "--manifest requires --compare-artifact-contracts or "
            "--artifact-launch-report"
        )


    if args.validate_result_report is not None:
        try:
            validation = validate_offline_control_result_report(
                load_offline_control_result_report(args.validate_result_report)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_offline_control_result_report",
                    args.validate_result_report,
                    exc,
                )
            )
            return 1
        _emit_json(
            {
                "action": "validate_offline_control_result_report",
                "path": str(args.validate_result_report),
                **validation,
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.compare_result_report is not None:
        try:
            comparison = compare_offline_control_result_report(
                load_offline_control_result_report(args.compare_result_report)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                {
                    **_error_payload(
                        "compare_offline_control_result_report",
                        args.compare_result_report,
                        exc,
                    ),
                    "matches": False,
                }
            )
            return 1
        _emit_json(
            {
                "action": "compare_offline_control_result_report",
                "path": str(args.compare_result_report),
                **comparison,
            }
        )
        return 0 if comparison["matches"] is True else 1

    if args.validate_report is not None:
        try:
            validation = validate_offline_control_matrix_report(
                load_offline_control_matrix_report(args.validate_report)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_offline_control_matrix_report",
                    args.validate_report,
                    exc,
                )
            )
            return 1
        _emit_json(
            {
                "action": "validate_offline_control_matrix_report",
                "path": str(args.validate_report),
                **validation,
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.compare_report is not None:
        try:
            comparison = compare_offline_control_matrix_report(
                load_offline_control_matrix_report(args.compare_report)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                {
                    **_error_payload(
                        "compare_offline_control_matrix_report",
                        args.compare_report,
                        exc,
                    ),
                    "matches": False,
                }
            )
            return 1
        _emit_json(
            {
                "action": "compare_offline_control_matrix_report",
                "path": str(args.compare_report),
                **comparison,
            }
        )
        return 0 if comparison["matches"] is True else 1

    if not args.import_reports:
        parser.error("--import-report is required")
    if args.report is None:
        parser.error("--report is required")

    try:
        matrix_report = offline_control_matrix_report(
            tuple(
                load_offline_prediction_import_report(path)
                for path in args.import_reports
            ),
            report_paths=tuple(args.import_reports),
            required_source_kinds=tuple(
                args.required_source_kinds
                or ("caption_memory", "graph_text", "multi_frame_vlm", "vlm")
            ),
        )
        save_offline_control_matrix_report(matrix_report, args.report)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(_error_payload("offline_control_matrix", args.report, exc))
        return 1

    _emit_json(
        {
            "action": "offline_control_matrix",
            "path": str(args.report),
            "ready": matrix_report["readiness"]["ready"],
            "report_digest": matrix_report["report_digest"],
            "readiness": matrix_report["readiness"],
            "summary": matrix_report["summary"],
        }
    )
    return 0 if matrix_report["readiness"]["ready"] is True else 1


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
