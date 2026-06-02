from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    compare_qa_eval_report,
    load_qa_dataset,
    load_qa_eval_report,
    load_qa_predictions,
    qa_eval_report,
    qa_eval_report_digest,
    save_qa_eval_report,
    validate_qa_eval_report,
)
from dsg_spatialqa_lab.schema import SpatialQAError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Evaluate deterministic SpatialQA predictions against QA JSONL gold cases.",
    )
    parser.add_argument("--gold", type=Path, help="Explicit local gold QA JSONL dataset.")
    parser.add_argument("--pred", type=Path, help="Explicit local QA prediction JSONL dataset.")
    parser.add_argument("--report", type=Path, help="Explicit local QA eval report output path.")
    parser.add_argument("--validate-report", type=Path, help="Validate an explicit QA eval report.")
    parser.add_argument(
        "--compare-report",
        type=Path,
        help="Compare an explicit QA eval report with current gold and prediction files.",
    )
    args = parser.parse_args(argv)

    if args.validate_report is not None:
        try:
            validation = validate_qa_eval_report(load_qa_eval_report(args.validate_report))
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(_error_payload("validate_qa_eval_report", args.validate_report, exc))
            return 1
        payload = {
            "action": "validate_qa_eval_report",
            "path": str(args.validate_report),
            **validation,
        }
        _emit_json(payload)
        return 0 if validation["valid"] is True else 1

    if args.compare_report is not None:
        try:
            comparison = compare_qa_eval_report(load_qa_eval_report(args.compare_report))
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            payload = {
                **_error_payload("compare_qa_eval_report", args.compare_report, exc),
                "matches": False,
            }
            _emit_json(payload)
            return 1
        payload = {
            "action": "compare_qa_eval_report",
            "path": str(args.compare_report),
            **comparison,
        }
        _emit_json(payload)
        return 0 if comparison["matches"] is True else 1

    if args.gold is None:
        parser.error("--gold is required when generating a report")
    if args.pred is None:
        parser.error("--pred is required when generating a report")
    if args.report is None:
        parser.error("--report is required when generating a report")

    try:
        report = qa_eval_report(
            load_qa_dataset(args.gold),
            load_qa_predictions(args.pred),
            gold_path=args.gold,
            prediction_path=args.pred,
        )
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(_error_payload("qa_eval_report", args.report, exc))
        return 1

    save_qa_eval_report(report, args.report)
    validation = validate_qa_eval_report(report)
    payload = {
        "action": "qa_eval_report",
        "path": str(args.report),
        "valid": validation["valid"],
        "digest": qa_eval_report_digest(report),
        "summary": report["summary"],
        "metrics": report["metrics"],
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
