from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    OFFLINE_PREDICTION_RECORD_INPUT_FORMAT,
    QA_PREDICTION_INPUT_FORMAT,
    compare_offline_prediction_import_report,
    import_offline_predictions,
    import_qa_prediction_inputs,
    load_offline_prediction_import_report,
    load_offline_prediction_records,
    load_qa_dataset,
    load_qa_predictions,
    offline_prediction_import_report_digest,
    qa_predictions_digest,
    save_offline_prediction_import_report,
    save_qa_predictions,
    validate_offline_prediction_import_report,
)
from dsg_spatialqa_lab.schema import SpatialQAError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Import local offline SpatialQA predictions into deterministic QAPrediction JSONL.",
    )
    parser.add_argument("--qa", type=Path, help="Explicit local gold QA JSONL dataset.")
    parser.add_argument("--input", type=Path, help="Explicit local offline prediction JSONL input.")
    parser.add_argument(
        "--input-format",
        choices=(OFFLINE_PREDICTION_RECORD_INPUT_FORMAT, QA_PREDICTION_INPUT_FORMAT),
        default=OFFLINE_PREDICTION_RECORD_INPUT_FORMAT,
        help="Prediction input format.",
    )
    parser.add_argument("--source-name", help="Stable source name for this prediction artifact.")
    parser.add_argument("--source-kind", default="offline", help="Stable source kind.")
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Stable source metadata as key=value. May be repeated.",
    )
    parser.add_argument("--pred", type=Path, help="Explicit local QAPrediction JSONL output path.")
    parser.add_argument("--report", type=Path, help="Explicit local import report output path.")
    parser.add_argument("--validate-report", type=Path, help="Validate an explicit import report.")
    parser.add_argument(
        "--compare-report",
        type=Path,
        help="Compare an explicit import report with current local artifacts.",
    )
    args = parser.parse_args(argv)

    if args.validate_report is not None:
        try:
            validation = validate_offline_prediction_import_report(
                load_offline_prediction_import_report(args.validate_report)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(_error_payload("validate_offline_prediction_import_report", args.validate_report, exc))
            return 1
        _emit_json(
            {
                "action": "validate_offline_prediction_import_report",
                "path": str(args.validate_report),
                **validation,
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.compare_report is not None:
        try:
            comparison = compare_offline_prediction_import_report(
                load_offline_prediction_import_report(args.compare_report)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            payload = {
                **_error_payload("compare_offline_prediction_import_report", args.compare_report, exc),
                "matches": False,
            }
            _emit_json(payload)
            return 1
        _emit_json(
            {
                "action": "compare_offline_prediction_import_report",
                "path": str(args.compare_report),
                **comparison,
            }
        )
        return 0 if comparison["matches"] is True else 1

    if args.qa is None:
        parser.error("--qa is required when importing predictions")
    if args.input is None:
        parser.error("--input is required when importing predictions")
    if args.source_name is None:
        parser.error("--source-name is required when importing predictions")
    if args.pred is None:
        parser.error("--pred is required when importing predictions")
    if args.report is None:
        parser.error("--report is required when importing predictions")

    try:
        cases = load_qa_dataset(args.qa)
        metadata = _metadata(args.metadata)
        if args.input_format == QA_PREDICTION_INPUT_FORMAT:
            predictions, report = import_qa_prediction_inputs(
                cases,
                load_qa_predictions(args.input),
                source_name=args.source_name,
                source_kind=args.source_kind,
                source_metadata=metadata,
                qa_path=args.qa,
                input_path=args.input,
                prediction_path=args.pred,
            )
        else:
            predictions, report = import_offline_predictions(
                cases,
                load_offline_prediction_records(args.input),
                source_name=args.source_name,
                source_kind=args.source_kind,
                source_metadata=metadata,
                qa_path=args.qa,
                input_path=args.input,
                prediction_path=args.pred,
            )
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(_error_payload("import_predictions", args.pred, exc))
        return 1

    save_qa_predictions(predictions, args.pred)
    save_offline_prediction_import_report(report, args.report)
    validation = validate_offline_prediction_import_report(report)
    payload = {
        "action": "import_predictions",
        "path": str(args.pred),
        "valid": validation["valid"],
        "digest": offline_prediction_import_report_digest(report),
        "input_format": report.get("input_format", OFFLINE_PREDICTION_RECORD_INPUT_FORMAT),
        "prediction_digest": qa_predictions_digest(predictions),
        "source_profile": report["source_profile"],
        "summary": report["summary"],
    }
    _emit_json(payload)
    return 0 if validation["valid"] is True else 1


def _metadata(items: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if separator == "" or key == "":
            raise SpatialQAError("Metadata entries must use key=value")
        metadata[key] = value
    return {key: metadata[key] for key in sorted(metadata)}


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
