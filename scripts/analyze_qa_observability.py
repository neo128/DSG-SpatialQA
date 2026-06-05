from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    SpatialQAError,
    compare_qa_observability_report,
    filter_qa_cases_by_ids,
    load_graph_json,
    load_qa_dataset,
    load_qa_observability_report,
    qa_observability_report,
    qa_observability_report_digest,
    qa_observability_split_ids,
    save_qa_dataset,
    save_qa_observability_report,
    validate_qa_observability_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Analyze QA observability against an explicit predicted DSG graph.",
    )
    parser.add_argument("--qa", type=Path, help="Explicit local QA JSONL dataset.")
    parser.add_argument("--graph", type=Path, help="Explicit local predicted graph JSON.")
    parser.add_argument("--report", type=Path, help="QA observability report output path.")
    parser.add_argument(
        "--evidence-observable-qa",
        type=Path,
        help="Optional QA JSONL output for cases with all required evidence present.",
    )
    parser.add_argument(
        "--target-observable-qa",
        type=Path,
        help="Optional QA JSONL output for cases whose target nodes are present.",
    )
    parser.add_argument(
        "--missing-evidence-qa",
        type=Path,
        help="Optional QA JSONL output for cases missing required predicted evidence.",
    )
    parser.add_argument(
        "--target-observable-relation-missing-qa",
        type=Path,
        help=(
            "Optional QA JSONL output for cases whose target nodes are present "
            "but required predicted evidence relations are missing."
        ),
    )
    parser.add_argument(
        "--validate-report",
        type=Path,
        help="Validate an explicit QA observability report.",
    )
    parser.add_argument(
        "--compare-report",
        type=Path,
        help="Compare an explicit QA observability report with current QA and graph files.",
    )
    args = parser.parse_args(argv)

    if args.validate_report is not None:
        try:
            validation = validate_qa_observability_report(
                load_qa_observability_report(args.validate_report)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_qa_observability_report",
                    args.validate_report,
                    exc,
                )
            )
            return 1
        payload = {
            "action": "validate_qa_observability_report",
            "path": str(args.validate_report),
            **validation,
        }
        _emit_json(payload)
        return 0 if validation["valid"] is True else 1

    if args.compare_report is not None:
        try:
            comparison = compare_qa_observability_report(
                load_qa_observability_report(args.compare_report)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            payload = {
                **_error_payload(
                    "compare_qa_observability_report",
                    args.compare_report,
                    exc,
                ),
                "matches": False,
            }
            _emit_json(payload)
            return 1
        payload = {
            "action": "compare_qa_observability_report",
            "path": str(args.compare_report),
            **comparison,
        }
        _emit_json(payload)
        return 0 if comparison["matches"] is True else 1

    if args.qa is None or args.graph is None or args.report is None:
        parser.error("analysis requires --qa, --graph, and --report")

    try:
        cases = load_qa_dataset(args.qa)
        graph = load_graph_json(args.graph)
        report = qa_observability_report(
            cases,
            graph,
            qa_path=args.qa,
            graph_path=args.graph,
        )
        save_qa_observability_report(report, args.report)
        _write_split(args.evidence_observable_qa, "evidence_observable", cases, report)
        _write_split(args.target_observable_qa, "target_observable", cases, report)
        _write_split(args.missing_evidence_qa, "missing_evidence", cases, report)
        _write_split(
            args.target_observable_relation_missing_qa,
            "target_observable_relation_missing",
            cases,
            report,
        )
        validation = validate_qa_observability_report(report)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(_error_payload("qa_observability_report", args.report, exc))
        return 1

    payload = {
        "action": "qa_observability_report",
        "path": str(args.report),
        "valid": validation["valid"],
        "digest": qa_observability_report_digest(report),
        "summary": report["summary"],
    }
    _emit_json(payload)
    return 0 if validation["valid"] is True else 1


def _write_split(
    path: Path | None,
    split_name: str,
    cases: list[Any],
    report: dict[str, Any],
) -> None:
    if path is None:
        return
    save_qa_dataset(
        filter_qa_cases_by_ids(cases, qa_observability_split_ids(report, split_name)),
        path,
    )


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
