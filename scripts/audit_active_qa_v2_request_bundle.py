#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "dsg-spatialqa-lab.active-qa-v2-request-quality-report.v1"
ACTIVE_QA_V2_REQUEST_BUNDLE_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.active-qa-v2-vlm-request-bundle.v1"
)
FORBIDDEN_KEYS = {
    "gold_answer",
    "gold_evidence",
    "required_edges",
    "required_nodes",
    "visible_object_ids",
    "visible_object_labels",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Audit active QA v2 VLM request bundle quality before external prediction.",
    )
    parser.add_argument("--request-bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        bundle = _load_bundle(args.request_bundle)
        report = _audit_bundle(bundle, request_bundle=args.request_bundle)
        _write_json(args.report, report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "blockers": ["request_bundle_quality_audit_error"],
            "error": str(exc),
            "ready": False,
        }
        report["report_digest"] = _digest_without(report, "report_digest")
        _write_json(args.report, report)
        _emit(report)
        return 1

    _emit(
        {
            "action": "audit_active_qa_v2_request_bundle",
            "blockers": report["blockers"],
            "ready": report["ready"],
            "report": str(args.report),
            "request_bundle": str(args.request_bundle),
        }
    )
    return 0 if report["ready"] is True else 1


def _load_bundle(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("request bundle must be a JSON object")
    if payload.get("schema_version") != ACTIVE_QA_V2_REQUEST_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported_request_bundle_schema")
    return payload


def _audit_bundle(bundle: dict[str, Any], *, request_bundle: Path) -> dict[str, Any]:
    cases = _prediction_cases(bundle)
    question_type_counts = Counter(_string_or_unknown(case.get("question_type")) for case in cases)
    missing_task_hint_case_ids = [
        _case_id(case)
        for case in cases
        if not _non_empty_string(case.get("question_task_hint"))
    ]
    missing_primary_frame_case_ids = [
        _case_id(case)
        for case in cases
        if not _primary_frame_exists(case)
    ]
    support_candidates_by_type = Counter(
        _string_or_unknown(case.get("question_type"))
        for case in cases
        if isinstance(case.get("support_candidates"), list)
        and len(case.get("support_candidates", [])) > 0
    )
    leak_paths = _forbidden_paths(bundle)
    blockers: list[str] = []
    if bundle.get("leak_free") is not True or bundle.get("leak_paths") or leak_paths:
        blockers.append("request_bundle_not_leak_free")
    if missing_task_hint_case_ids:
        blockers.append("question_task_hints_incomplete")
    if missing_primary_frame_case_ids:
        blockers.append("primary_frame_artifacts_missing")
    summary = {
        "missing_primary_frame_count": len(missing_primary_frame_case_ids),
        "missing_question_task_hint_count": len(missing_task_hint_case_ids),
        "primary_frame_exists_count": len(cases) - len(missing_primary_frame_case_ids),
        "question_task_hint_count": len(cases) - len(missing_task_hint_case_ids),
        "question_type_counts": dict(sorted(question_type_counts.items())),
        "request_count": len(cases),
        "support_candidates_by_question_type": dict(sorted(support_candidates_by_type.items())),
        "support_candidates_case_count": sum(
            1
            for case in cases
            if isinstance(case.get("support_candidates"), list)
            and len(case.get("support_candidates", [])) > 0
        ),
        "target_crop_case_count": sum(
            1 for case in cases if isinstance(case.get("target_crop"), dict)
        ),
        "target_visual_context_case_count": sum(
            1
            for case in cases
            if isinstance(case.get("target_visual_context"), dict)
        ),
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "blockers": sorted(blockers),
        "leak_paths": leak_paths,
        "missing_primary_frame_case_ids": missing_primary_frame_case_ids,
        "missing_question_task_hint_case_ids": missing_task_hint_case_ids,
        "ready": not blockers,
        "request_bundle": str(request_bundle),
        "summary": summary,
    }
    report["report_digest"] = _digest_without(report, "report_digest")
    return report


def _prediction_cases(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    cases = bundle.get("prediction_cases")
    if not isinstance(cases, list):
        raise ValueError("prediction_cases must be a list")
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"prediction case {index} must be an object")
        if not isinstance(case.get("case_id"), str):
            raise ValueError(f"prediction case {index} missing string case_id")
        rows.append(case)
    return rows


def _primary_frame_exists(case: dict[str, Any]) -> bool:
    primary_frame = case.get("primary_frame")
    if not isinstance(primary_frame, dict):
        return False
    rgb_path = primary_frame.get("rgb_path")
    return isinstance(rgb_path, str) and bool(rgb_path) and Path(rgb_path).exists()


def _forbidden_paths(value: object, *, prefix: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}"
            if key in FORBIDDEN_KEYS:
                paths.append(child_prefix)
            paths.extend(_forbidden_paths(child, prefix=child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_forbidden_paths(child, prefix=f"{prefix}[{index}]"))
    return paths


def _case_id(case: dict[str, Any]) -> str:
    value = case.get("case_id")
    return value if isinstance(value, str) else "unknown"


def _non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _string_or_unknown(value: object) -> str:
    return value if isinstance(value, str) and value else "unknown"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _digest_without(payload: dict[str, Any], key_to_omit: str) -> str:
    normalized = {key: value for key, value in payload.items() if key != key_to_omit}
    return hashlib.sha256(
        json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True) + "\n", end="")


if __name__ == "__main__":
    raise SystemExit(main())
