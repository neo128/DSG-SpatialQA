from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    SpatialQAError,
    offline_control_import_manifest_preflight,
    offline_control_import_run_ledger,
    offline_control_prediction_receipt_bundle,
    offline_control_prediction_request_bundle,
    load_offline_control_prediction_receipt_bundle,
    run_offline_control_import_manifest,
    run_offline_control_imports,
    save_offline_control_artifact_contracts,
    save_offline_control_import_run_ledger,
    save_offline_control_prediction_receipt_bundle,
    save_offline_control_prediction_request_bundle,
    validate_offline_control_prediction_receipt_bundle,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Import explicit local offline control prediction records and "
            "write a real baseline control matrix."
        ),
    )
    parser.add_argument("--qa", type=Path, help="Explicit local gold QA JSONL dataset.")
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Explicit local offline control import manifest. When supplied, "
            "the manifest provides QA, source, output, candidate, and matrix paths."
        ),
    )
    parser.add_argument(
        "--preflight-manifest",
        type=Path,
        help=(
            "Explicit local offline control import manifest to audit before "
            "writing outputs. This reads QA, source prediction, and candidate "
            "prediction inputs and reports coverage/readiness without writing "
            "imports, QA eval reports, deltas, or the matrix report."
        ),
    )
    parser.add_argument(
        "--prediction-request-bundle",
        type=Path,
        help=(
            "Explicit local offline control import manifest to convert into "
            "an external baseline prediction request bundle."
        ),
    )
    parser.add_argument(
        "--request-bundle-output",
        type=Path,
        help="Optional output path for --prediction-request-bundle JSON.",
    )
    parser.add_argument(
        "--prediction-receipt-bundle",
        type=Path,
        help=(
            "Explicit local offline control import manifest to convert into "
            "an external baseline prediction receipt bundle after files return."
        ),
    )
    parser.add_argument(
        "--receipt-bundle-output",
        type=Path,
        help="Optional output path for --prediction-receipt-bundle JSON.",
    )
    parser.add_argument(
        "--validate-prediction-receipt-bundle",
        type=Path,
        help="Validate a saved offline-control prediction receipt bundle JSON.",
    )
    parser.add_argument(
        "--artifact-contracts",
        type=Path,
        help=(
            "Optional output path for the artifact_contracts section produced by "
            "--preflight-manifest."
        ),
    )
    parser.add_argument("--output-dir", type=Path, help="Output directory for imports.")
    parser.add_argument(
        "--matrix-report",
        type=Path,
        help="Explicit offline control matrix report output path.",
    )
    parser.add_argument(
        "--source",
        action="append",
        nargs=3,
        metavar=("KIND", "NAME", "INPUT"),
        help=(
            "Offline source tuple: source kind, stable source name, and local "
            "OfflinePredictionRecord or QAPrediction JSONL path. May be repeated."
        ),
    )
    parser.add_argument(
        "--source-input-format",
        action="append",
        nargs=2,
        metavar=("NAME", "FORMAT"),
        help=(
            "Input format keyed by source name: offline_prediction_record "
            "or qa_prediction. May be repeated."
        ),
    )
    parser.add_argument(
        "--source-metadata",
        action="append",
        nargs=2,
        metavar=("NAME", "KEY=VALUE"),
        help="Source metadata keyed by source name. May be repeated.",
    )
    parser.add_argument(
        "--required-source-kind",
        action="append",
        dest="required_source_kinds",
        help="Required offline control source kind. May be repeated.",
    )
    parser.add_argument(
        "--candidate-prediction",
        type=Path,
        help=(
            "Optional local candidate DSG GraphTool QA prediction JSONL path. "
            "When supplied, QA eval and DSG-vs-control delta reports are written."
        ),
    )
    parser.add_argument(
        "--candidate-name",
        default="predicted_graph_tool",
        help="Stable candidate name used in generated QA delta reports.",
    )
    parser.add_argument(
        "--qa-eval-output-dir",
        type=Path,
        help="Optional output directory for generated QA eval and delta reports.",
    )
    parser.add_argument(
        "--result-report",
        type=Path,
        help=(
            "Optional output path for the generated offline control result report. "
            "Only used when --candidate-prediction is supplied."
        ),
    )
    parser.add_argument(
        "--run-ledger",
        type=Path,
        help=(
            "Optional output path for a compact manifest-import run ledger. "
            "Only used with --manifest after the import has written outputs."
        ),
    )
    args = parser.parse_args(argv)

    if args.prediction_request_bundle is not None:
        try:
            bundle = offline_control_prediction_request_bundle(
                args.prediction_request_bundle
            )
            request_bundle_path = None
            if args.request_bundle_output is not None:
                save_offline_control_prediction_request_bundle(
                    bundle,
                    args.request_bundle_output,
                )
                request_bundle_path = str(args.request_bundle_output)
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "offline_control_prediction_request_bundle",
                    args.prediction_request_bundle,
                    exc,
                )
            )
            return 1

        _emit_json(
            {
                "action": "offline_control_prediction_request_bundle",
                "manifest_path": str(args.prediction_request_bundle),
                "request_bundle_path": request_bundle_path,
                "bundle": bundle,
            }
        )
        return 0

    if args.request_bundle_output is not None:
        parser.error("--request-bundle-output requires --prediction-request-bundle")

    if args.prediction_receipt_bundle is not None:
        try:
            bundle = offline_control_prediction_receipt_bundle(
                args.prediction_receipt_bundle
            )
            receipt_bundle_path = None
            if args.receipt_bundle_output is not None:
                save_offline_control_prediction_receipt_bundle(
                    bundle,
                    args.receipt_bundle_output,
                )
                receipt_bundle_path = str(args.receipt_bundle_output)
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "offline_control_prediction_receipt_bundle",
                    args.prediction_receipt_bundle,
                    exc,
                )
            )
            return 1

        _emit_json(
            {
                "action": "offline_control_prediction_receipt_bundle",
                "manifest_path": str(args.prediction_receipt_bundle),
                "receipt_bundle_path": receipt_bundle_path,
                "bundle": bundle,
            }
        )
        return 0 if bundle["ready_to_import"] is True else 1

    if args.receipt_bundle_output is not None:
        parser.error("--receipt-bundle-output requires --prediction-receipt-bundle")

    if args.validate_prediction_receipt_bundle is not None:
        try:
            validation = validate_offline_control_prediction_receipt_bundle(
                load_offline_control_prediction_receipt_bundle(
                    args.validate_prediction_receipt_bundle
                )
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_offline_control_prediction_receipt_bundle",
                    args.validate_prediction_receipt_bundle,
                    exc,
                )
            )
            return 1
        _emit_json(
            {
                "action": "validate_offline_control_prediction_receipt_bundle",
                "path": str(args.validate_prediction_receipt_bundle),
                **validation,
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.preflight_manifest is not None:
        try:
            result = offline_control_import_manifest_preflight(
                args.preflight_manifest
            )
            if args.artifact_contracts is not None:
                save_offline_control_artifact_contracts(
                    result["artifact_contracts"],
                    args.artifact_contracts,
                )
                result["artifact_contracts_path"] = str(args.artifact_contracts)
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "offline_control_import_manifest_preflight",
                    args.preflight_manifest,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["ready_to_import"] is True else 1

    if args.artifact_contracts is not None:
        parser.error("--artifact-contracts requires --preflight-manifest")

    if args.manifest is not None:
        try:
            result = run_offline_control_import_manifest(args.manifest)
            if args.run_ledger is not None:
                ledger = offline_control_import_run_ledger(result)
                save_offline_control_import_run_ledger(ledger, args.run_ledger)
                result["run_ledger_path"] = str(args.run_ledger)
                result["run_ledger_digest"] = ledger["ledger_digest"]
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "run_offline_control_import_manifest",
                    args.manifest,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["ready"] is True else 1

    if args.run_ledger is not None:
        parser.error("--run-ledger requires --manifest")

    if args.qa is None:
        parser.error("--qa is required")
    if args.output_dir is None:
        parser.error("--output-dir is required")
    if args.matrix_report is None:
        parser.error("--matrix-report is required")
    if not args.source:
        parser.error("--source is required")

    try:
        result = run_offline_control_imports(
            qa_path=args.qa,
            source_specs=_source_specs(
                args.source,
                tuple(args.source_metadata or ()),
                tuple(args.source_input_format or ()),
            ),
            output_dir=args.output_dir,
            matrix_report_path=args.matrix_report,
            required_source_kinds=tuple(
                args.required_source_kinds
                or ("caption_memory", "graph_text", "multi_frame_vlm", "vlm")
            ),
            candidate_prediction_path=args.candidate_prediction,
            candidate_name=args.candidate_name,
            qa_eval_output_dir=args.qa_eval_output_dir,
            result_report_path=args.result_report,
        )
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(_error_payload("run_offline_control_imports", args.matrix_report, exc))
        return 1

    _emit_json(result)
    return 0 if result["ready"] is True else 1


def _source_specs(
    source_args: list[list[str]],
    metadata_args: tuple[list[str], ...],
    input_format_args: tuple[list[str], ...],
) -> tuple[dict[str, object], ...]:
    metadata_by_source = _metadata_by_source(metadata_args)
    input_format_by_source = _input_format_by_source(input_format_args)
    specs: list[dict[str, object]] = []
    for source_kind, source_name, input_path in source_args:
        specs.append(
            {
                "source_kind": source_kind,
                "source_name": source_name,
                "input_path": Path(input_path),
                "input_format": input_format_by_source.get(
                    source_name,
                    "offline_prediction_record",
                ),
                "metadata": metadata_by_source.get(source_name, {}),
            }
        )
    return tuple(specs)


def _input_format_by_source(
    input_format_args: tuple[list[str], ...],
) -> dict[str, str]:
    return {
        source_name: input_format
        for source_name, input_format in sorted(input_format_args)
    }


def _metadata_by_source(
    metadata_args: tuple[list[str], ...],
) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    for source_name, item in metadata_args:
        key, separator, value = item.partition("=")
        if separator == "" or key == "":
            raise SpatialQAError("Source metadata entries must use key=value")
        source_metadata = metadata.setdefault(source_name, {})
        source_metadata[key] = value
    return {
        source_name: {key: values[key] for key in sorted(values)}
        for source_name, values in sorted(metadata.items())
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
