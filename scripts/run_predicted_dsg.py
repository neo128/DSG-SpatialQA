from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    SpatialQAError,
    compare_predicted_dsg_detector_run_ledger,
    load_predicted_dsg_detector_artifact_contract,
    load_predicted_dsg_detector_receipt_bundle,
    load_predicted_dsg_detector_run_ledger,
    predicted_dsg_detector_artifact_launch_report,
    predicted_dsg_detector_receipt_bundle,
    predicted_dsg_detector_request_bundle,
    predicted_dsg_detector_run_ledger,
    predicted_dsg_detector_run_manifest_preflight,
    run_predicted_dsg_detector_run_manifest,
    run_predicted_dsg_from_detector_jsonl,
    save_predicted_dsg_detector_artifact_contract,
    save_predicted_dsg_detector_receipt_bundle,
    save_predicted_dsg_detector_request_bundle,
    save_predicted_dsg_detector_run_ledger,
    validate_predicted_dsg_detector_receipt_bundle,
    validate_predicted_dsg_detector_run_ledger,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Run the deterministic predicted-DSG handoff from explicit local "
            "detector/RGB-D JSONL records."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Explicit predicted DSG detector-run manifest. When supplied, "
            "the manifest provides all local input and output paths."
        ),
    )
    parser.add_argument(
        "--preflight-manifest",
        type=Path,
        help=(
            "Explicit predicted DSG detector-run manifest to audit before "
            "writing outputs. This reads detector JSONL and runs the graph/"
            "evidence readiness checks in memory without writing artifacts."
        ),
    )
    parser.add_argument(
        "--detector-request-bundle",
        type=Path,
        help=(
            "Explicit predicted DSG detector-run manifest to convert into "
            "an external detector/RGB-D request bundle."
        ),
    )
    parser.add_argument(
        "--request-bundle-output",
        type=Path,
        help="Optional output path for --detector-request-bundle JSON.",
    )
    parser.add_argument(
        "--detector-receipt-bundle",
        type=Path,
        help=(
            "Explicit predicted DSG detector-run manifest to convert into "
            "an external detector/RGB-D receipt bundle after files return."
        ),
    )
    parser.add_argument(
        "--receipt-bundle-output",
        type=Path,
        help="Optional output path for --detector-receipt-bundle JSON.",
    )
    parser.add_argument(
        "--validate-detector-receipt-bundle",
        type=Path,
        help="Validate a saved predicted-DSG detector receipt bundle JSON.",
    )
    parser.add_argument(
        "--artifact-contract",
        type=Path,
        help=(
            "Optional output path for the artifact_contract section produced by "
            "--preflight-manifest."
        ),
    )
    parser.add_argument(
        "--artifact-launch-report",
        type=Path,
        help=(
            "Saved predicted DSG detector artifact contract to summarize "
            "current detector/RGB-D launch readiness. Requires --manifest."
        ),
    )
    parser.add_argument(
        "--run-ledger",
        type=Path,
        help=(
            "Optional output path for a compact detector-run ledger. Only used "
            "with --manifest after the predicted DSG artifacts have been written."
        ),
    )
    parser.add_argument("--validate-run-ledger", type=Path)
    parser.add_argument("--compare-run-ledger", type=Path)
    parser.add_argument("--detector-jsonl", type=Path)
    parser.add_argument("--observation-sequence", type=Path)
    parser.add_argument("--output-graph", type=Path)
    parser.add_argument("--predicted-report", type=Path)
    parser.add_argument("--detector-import-report", type=Path)
    parser.add_argument("--predicted-dsg-evidence-report", type=Path)
    parser.add_argument(
        "--infer-relation",
        action="append",
        dest="infer_relations",
        help="Spatial relation name to infer from the observation sequence.",
    )
    parser.add_argument(
        "--reference-frame",
        action="append",
        dest="reference_frames",
        help="Reference frame for inferred relations.",
    )
    parser.add_argument("--min-observation-count", type=int, default=2)
    parser.add_argument("--min-object-observation-count", type=int, default=2)
    parser.add_argument(
        "--required-evidence-kind",
        action="append",
        dest="required_evidence_kinds",
    )
    args = parser.parse_args(argv)

    if args.detector_request_bundle is not None:
        try:
            bundle = predicted_dsg_detector_request_bundle(
                args.detector_request_bundle
            )
            request_bundle_path = None
            if args.request_bundle_output is not None:
                save_predicted_dsg_detector_request_bundle(
                    bundle,
                    args.request_bundle_output,
                )
                request_bundle_path = str(args.request_bundle_output)
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "predicted_dsg_detector_request_bundle",
                    args.detector_request_bundle,
                    exc,
                )
            )
            return 1

        _emit_json(
            {
                "action": "predicted_dsg_detector_request_bundle",
                "manifest_path": str(args.detector_request_bundle),
                "request_bundle_path": request_bundle_path,
                "bundle": bundle,
            }
        )
        return 0

    if args.request_bundle_output is not None:
        parser.error("--request-bundle-output requires --detector-request-bundle")

    if args.detector_receipt_bundle is not None:
        try:
            bundle = predicted_dsg_detector_receipt_bundle(
                args.detector_receipt_bundle
            )
            receipt_bundle_path = None
            if args.receipt_bundle_output is not None:
                save_predicted_dsg_detector_receipt_bundle(
                    bundle,
                    args.receipt_bundle_output,
                )
                receipt_bundle_path = str(args.receipt_bundle_output)
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "predicted_dsg_detector_receipt_bundle",
                    args.detector_receipt_bundle,
                    exc,
                )
            )
            return 1

        _emit_json(
            {
                "action": "predicted_dsg_detector_receipt_bundle",
                "manifest_path": str(args.detector_receipt_bundle),
                "receipt_bundle_path": receipt_bundle_path,
                "bundle": bundle,
            }
        )
        return 0 if bundle["ready_to_build"] is True else 1

    if args.receipt_bundle_output is not None:
        parser.error("--receipt-bundle-output requires --detector-receipt-bundle")

    if args.validate_detector_receipt_bundle is not None:
        try:
            validation = validate_predicted_dsg_detector_receipt_bundle(
                load_predicted_dsg_detector_receipt_bundle(
                    args.validate_detector_receipt_bundle
                )
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_predicted_dsg_detector_receipt_bundle",
                    args.validate_detector_receipt_bundle,
                    exc,
                )
            )
            return 1
        _emit_json(
            {
                "action": "validate_predicted_dsg_detector_receipt_bundle",
                "path": str(args.validate_detector_receipt_bundle),
                **validation,
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.validate_run_ledger is not None:
        try:
            validation = validate_predicted_dsg_detector_run_ledger(
                load_predicted_dsg_detector_run_ledger(args.validate_run_ledger)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_predicted_dsg_detector_run_ledger",
                    args.validate_run_ledger,
                    exc,
                )
            )
            return 1
        _emit_json(
            {
                "action": "validate_predicted_dsg_detector_run_ledger",
                "path": str(args.validate_run_ledger),
                **validation,
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.compare_run_ledger is not None:
        try:
            comparison = compare_predicted_dsg_detector_run_ledger(
                load_predicted_dsg_detector_run_ledger(args.compare_run_ledger)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                {
                    **_error_payload(
                        "compare_predicted_dsg_detector_run_ledger",
                        args.compare_run_ledger,
                        exc,
                    ),
                    "matches": False,
                }
            )
            return 1
        _emit_json(
            {
                "action": "compare_predicted_dsg_detector_run_ledger",
                "path": str(args.compare_run_ledger),
                **comparison,
            }
        )
        return 0 if comparison["matches"] is True else 1

    if args.preflight_manifest is not None:
        try:
            result = predicted_dsg_detector_run_manifest_preflight(
                args.preflight_manifest
            )
            if args.artifact_contract is not None:
                save_predicted_dsg_detector_artifact_contract(
                    result["artifact_contract"],
                    args.artifact_contract,
                )
                result["artifact_contract_path"] = str(args.artifact_contract)
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "predicted_dsg_detector_run_manifest_preflight",
                    args.preflight_manifest,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["ready_to_build"] is True else 1

    if args.artifact_contract is not None:
        parser.error("--artifact-contract requires --preflight-manifest")

    if args.artifact_launch_report is not None:
        if args.manifest is None:
            parser.error("--manifest is required with --artifact-launch-report")
        try:
            report = predicted_dsg_detector_artifact_launch_report(
                load_predicted_dsg_detector_artifact_contract(
                    args.artifact_launch_report
                ),
                manifest_path=args.manifest,
                contract_path=args.artifact_launch_report,
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "predicted_dsg_detector_artifact_launch_report",
                    args.artifact_launch_report,
                    exc,
                )
            )
            return 1
        _emit_json(report)
        return 0 if report["ready_to_build"] is True else 1

    if args.manifest is not None:
        try:
            result = run_predicted_dsg_detector_run_manifest(args.manifest)
            if args.run_ledger is not None:
                ledger = predicted_dsg_detector_run_ledger(result)
                save_predicted_dsg_detector_run_ledger(ledger, args.run_ledger)
                result["run_ledger_path"] = str(args.run_ledger)
                result["run_ledger_digest"] = ledger["ledger_digest"]
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "run_predicted_dsg_detector_run_manifest",
                    args.manifest,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["ready"] is True else 1

    if args.run_ledger is not None:
        parser.error("--run-ledger requires --manifest")

    for name, value in (
        ("--detector-jsonl", args.detector_jsonl),
        ("--observation-sequence", args.observation_sequence),
        ("--output-graph", args.output_graph),
        ("--predicted-report", args.predicted_report),
        ("--detector-import-report", args.detector_import_report),
        ("--predicted-dsg-evidence-report", args.predicted_dsg_evidence_report),
    ):
        if value is None:
            _emit_json(_missing_argument_payload(name))
            return 1

    try:
        result = run_predicted_dsg_from_detector_jsonl(
            detector_jsonl_path=args.detector_jsonl,
            output_sequence_path=args.observation_sequence,
            output_graph_path=args.output_graph,
            predicted_graph_report_path=args.predicted_report,
            detector_import_report_path=args.detector_import_report,
            predicted_dsg_evidence_report_path=args.predicted_dsg_evidence_report,
            infer_relations=tuple(
                args.infer_relations or ("LEFT_OF", "RIGHT_OF", "NEAR")
            ),
            reference_frames=tuple(args.reference_frames or ("world",)),
            min_observation_count=args.min_observation_count,
            min_object_observation_count=args.min_object_observation_count,
            required_evidence_kinds=tuple(
                args.required_evidence_kinds or ("depth", "detector", "rgb")
            ),
        )
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(
            _error_payload(
                "run_predicted_dsg_from_detector_jsonl",
                args.predicted_dsg_evidence_report or Path(""),
                exc,
            )
        )
        return 1

    _emit_json(result)
    return 0 if result["ready"] is True else 1


def _missing_argument_payload(argument: str) -> dict[str, Any]:
    return {
        "action": "run_predicted_dsg_from_detector_jsonl",
        "valid": False,
        "error": f"{argument} is required",
    }


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
