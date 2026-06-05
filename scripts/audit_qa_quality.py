from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    SpatialQAError,
    audit_vlm_request_bundle_for_gold_leakage,
    load_qa_dataset,
    load_qa_observability_report,
    load_qa_quality_report,
    qa_quality_report,
    qa_quality_report_digest,
    save_qa_quality_report,
    validate_qa_quality_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Audit QA dataset quality, splits, anti-shortcut risk, and VLM request leakage.",
    )
    parser.add_argument("--qa", type=Path, help="Explicit local QA JSONL dataset.")
    parser.add_argument("--report", type=Path, help="QA quality report output path.")
    parser.add_argument(
        "--observability-report",
        type=Path,
        help="Optional QA observability report used for split recommendations.",
    )
    parser.add_argument(
        "--validate-report",
        type=Path,
        help="Validate an explicit QA quality report.",
    )
    parser.add_argument(
        "--audit-vlm-request-bundle",
        type=Path,
        help="Check an explicit local VLM request bundle for gold/evidence leakage.",
    )
    args = parser.parse_args(argv)

    if args.validate_report is not None:
        try:
            validation = validate_qa_quality_report(
                load_qa_quality_report(args.validate_report)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(_error_payload("validate_qa_quality_report", args.validate_report, exc))
            return 1
        _emit_json(
            {
                "action": "validate_qa_quality_report",
                "path": str(args.validate_report),
                **validation,
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.audit_vlm_request_bundle is not None:
        try:
            payload = json.loads(args.audit_vlm_request_bundle.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise SpatialQAError("VLM request bundle JSON must be an object")
            audit = audit_vlm_request_bundle_for_gold_leakage(payload)
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(_error_payload("audit_vlm_request_bundle", args.audit_vlm_request_bundle, exc))
            return 1
        _emit_json(
            {
                "action": "audit_vlm_request_bundle",
                "path": str(args.audit_vlm_request_bundle),
                **audit,
            }
        )
        return 0 if audit["leak_free"] is True else 1

    if args.qa is None or args.report is None:
        parser.error("quality audit requires --qa and --report")

    try:
        cases = load_qa_dataset(args.qa)
        observability_report = (
            load_qa_observability_report(args.observability_report)
            if args.observability_report is not None
            else None
        )
        report = qa_quality_report(
            cases,
            observability_report=observability_report,
            qa_path=args.qa,
            observability_report_path=args.observability_report,
        )
        save_qa_quality_report(report, args.report)
        validation = validate_qa_quality_report(report)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(_error_payload("qa_quality_report", args.report, exc))
        return 1

    _emit_json(
        {
            "action": "qa_quality_report",
            "path": str(args.report),
            "valid": validation["valid"],
            "digest": qa_quality_report_digest(report),
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
