from __future__ import annotations

import argparse
import json
from pathlib import Path

from dsg_spatialqa_lab import (
    compare_evaluation_bundle,
    compare_evaluation_case_listing,
    compare_evaluation_manifest,
    compare_evaluation_report,
    evaluation_bundle,
    evaluation_bundle_json,
    evaluation_case_listing,
    evaluation_case_listing_json,
    evaluation_manifest,
    evaluation_manifest_json,
    evaluation_report,
    evaluation_report_json,
    load_evaluation_bundle,
    load_evaluation_case_listing,
    load_evaluation_manifest,
    load_evaluation_report,
    run_evaluation_suite,
    validate_evaluation_bundle,
    validate_evaluation_case_listing,
    validate_evaluation_manifest,
    validate_evaluation_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic DSG-SpatialQA evaluation cases and emit a report."
    )
    parser.add_argument(
        "--name",
        action="append",
        dest="names",
        help="Select an exact evaluation case name. May be repeated.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        help="Require an evaluation tag. May be repeated.",
    )
    parser.add_argument(
        "--kind",
        action="append",
        dest="kinds",
        help="Select an evaluation case kind. May be repeated.",
    )
    parser.add_argument(
        "--question-type",
        action="append",
        dest="question_types",
        help="Select a QA question type. May be repeated.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional explicit local path where the selected JSON output should be written.",
    )
    parser.add_argument(
        "--bundle",
        action="store_true",
        help="Emit a self-contained benchmark bundle instead of the compact report.",
    )
    parser.add_argument(
        "--manifest",
        action="store_true",
        help="Emit filtered case and scene metadata without running evaluation cases.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Emit filtered evaluation case metadata without running evaluation cases.",
    )
    parser.add_argument(
        "--validate-bundle",
        type=Path,
        help="Validate an explicit local benchmark bundle file instead of running cases.",
    )
    parser.add_argument(
        "--validate-listing",
        type=Path,
        help="Validate an explicit local case listing file instead of running cases.",
    )
    parser.add_argument(
        "--validate-manifest",
        type=Path,
        help="Validate an explicit local benchmark manifest file instead of running cases.",
    )
    parser.add_argument(
        "--validate-report",
        type=Path,
        help="Validate an explicit local evaluation report file instead of running cases.",
    )
    parser.add_argument(
        "--compare-bundle",
        type=Path,
        help="Compare an explicit local benchmark bundle with a current deterministic rerun.",
    )
    parser.add_argument(
        "--compare-listing",
        type=Path,
        help="Compare an explicit local case listing with current deterministic metadata.",
    )
    parser.add_argument(
        "--compare-manifest",
        type=Path,
        help="Compare an explicit local benchmark manifest with current deterministic metadata.",
    )
    parser.add_argument(
        "--compare-report",
        type=Path,
        help="Compare an explicit local report with a current deterministic rerun.",
    )
    args = parser.parse_args(argv)
    names = tuple(args.names) if args.names is not None else None
    tags = tuple(args.tags) if args.tags is not None else None
    kinds = tuple(args.kinds) if args.kinds is not None else None
    question_types = tuple(args.question_types) if args.question_types is not None else None
    return_code = 0
    if args.validate_report is not None:
        try:
            validation = validate_evaluation_report(
                load_evaluation_report(args.validate_report)
            )
        except (OSError, ValueError) as exc:
            payload = _artifact_error_json("validate_report", args.validate_report, exc)
            return_code = 1
        else:
            payload = evaluation_report_json(validation)
            return_code = 0 if validation["valid"] is True else 1
    elif args.compare_report is not None:
        try:
            comparison = compare_evaluation_report(load_evaluation_report(args.compare_report))
        except (OSError, ValueError) as exc:
            payload = _artifact_error_json(
                "compare_report",
                args.compare_report,
                exc,
                matches=False,
            )
            return_code = 1
        else:
            payload = evaluation_report_json(comparison)
            return_code = 0 if comparison["matches"] is True else 1
    elif args.compare_manifest is not None:
        try:
            comparison = compare_evaluation_manifest(
                load_evaluation_manifest(args.compare_manifest)
            )
        except (OSError, ValueError) as exc:
            payload = _artifact_error_json(
                "compare_manifest",
                args.compare_manifest,
                exc,
                matches=False,
            )
            return_code = 1
        else:
            payload = evaluation_manifest_json(comparison)
            return_code = 0 if comparison["matches"] is True else 1
    elif args.compare_listing is not None:
        try:
            comparison = compare_evaluation_case_listing(
                load_evaluation_case_listing(args.compare_listing)
            )
        except (OSError, ValueError) as exc:
            payload = _artifact_error_json(
                "compare_listing",
                args.compare_listing,
                exc,
                matches=False,
            )
            return_code = 1
        else:
            payload = evaluation_case_listing_json(comparison)
            return_code = 0 if comparison["matches"] is True else 1
    elif args.validate_listing is not None:
        try:
            validation = validate_evaluation_case_listing(
                load_evaluation_case_listing(args.validate_listing)
            )
        except (OSError, ValueError) as exc:
            payload = _artifact_error_json("validate_listing", args.validate_listing, exc)
            return_code = 1
        else:
            payload = evaluation_case_listing_json(validation)
            return_code = 0 if validation["valid"] is True else 1
    elif args.validate_manifest is not None:
        try:
            validation = validate_evaluation_manifest(
                load_evaluation_manifest(args.validate_manifest)
            )
        except (OSError, ValueError) as exc:
            payload = _artifact_error_json("validate_manifest", args.validate_manifest, exc)
            return_code = 1
        else:
            payload = evaluation_manifest_json(validation)
            return_code = 0 if validation["valid"] is True else 1
    elif args.compare_bundle is not None:
        try:
            comparison = compare_evaluation_bundle(load_evaluation_bundle(args.compare_bundle))
        except (OSError, ValueError) as exc:
            payload = _artifact_error_json(
                "compare_bundle",
                args.compare_bundle,
                exc,
                matches=False,
            )
            return_code = 1
        else:
            payload = evaluation_bundle_json(comparison)
            return_code = 0 if comparison["matches"] is True else 1
    elif args.validate_bundle is not None:
        try:
            validation = validate_evaluation_bundle(load_evaluation_bundle(args.validate_bundle))
        except (OSError, ValueError) as exc:
            payload = _artifact_error_json("validate_bundle", args.validate_bundle, exc)
            return_code = 1
        else:
            payload = evaluation_bundle_json(validation)
            return_code = 0 if validation["valid"] is True else 1
    elif args.manifest:
        payload = evaluation_manifest_json(
            evaluation_manifest(
                names=names,
                tags=tags,
                kinds=kinds,
                question_types=question_types,
            )
        )
    elif args.list_cases:
        payload = evaluation_case_listing_json(
            evaluation_case_listing(
                names=names,
                tags=tags,
                kinds=kinds,
                question_types=question_types,
            )
        )
    elif args.bundle:
        payload = evaluation_bundle_json(
            evaluation_bundle(
                names=names,
                tags=tags,
                kinds=kinds,
                question_types=question_types,
            )
        )
    else:
        suite = run_evaluation_suite(
            names=names,
            tags=tags,
            kinds=kinds,
            question_types=question_types,
        )
        payload = evaluation_report_json(evaluation_report(suite))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return return_code


def _artifact_error_json(
    action: str,
    path: Path,
    error: Exception,
    *,
    matches: bool | None = None,
) -> str:
    payload: dict[str, object] = {
        "action": action,
        "path": str(path),
        "valid": False,
        "error": str(error),
    }
    if matches is not None:
        payload["matches"] = matches
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
