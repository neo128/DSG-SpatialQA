from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.eval.offline_control_matrix import (
    load_offline_control_matrix_report,
    validate_offline_control_matrix_report,
)
from dsg_spatialqa_lab.eval.qa_metrics import (
    load_qa_eval_delta_report,
    load_qa_eval_report,
    qa_eval_delta_report_digest,
    qa_eval_report_digest,
    validate_qa_eval_delta_report,
    validate_qa_eval_report,
)
from dsg_spatialqa_lab.schema import SpatialQAError


OFFLINE_CONTROL_RESULT_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.offline-control-result-report.v1"
)


def offline_control_result_report(
    matrix_report: Mapping[str, Any],
    *,
    matrix_report_path: str | Path,
    candidate_qa_eval_report_path: str | Path,
    qa_eval_delta_report_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    candidate_report = load_qa_eval_report(candidate_qa_eval_report_path)
    delta_paths = _path_mapping(qa_eval_delta_report_paths)
    rows = _source_result_matrix(
        matrix_report,
        candidate_report,
        delta_paths,
    )
    checks = _checks_from_rows(
        matrix_report=matrix_report,
        candidate_report=candidate_report,
        rows=rows,
    )
    report: dict[str, Any] = {
        "schema_version": OFFLINE_CONTROL_RESULT_REPORT_SCHEMA_VERSION,
        "candidate_name": _candidate_name(rows),
        "candidate_qa_eval_report_digest": qa_eval_report_digest(candidate_report),
        "candidate_qa_eval_report_path": str(candidate_qa_eval_report_path),
        "matrix_readiness": _mapping(matrix_report.get("readiness"), "readiness"),
        "matrix_report_digest": _string_or_none(matrix_report.get("report_digest")),
        "matrix_report_path": str(matrix_report_path),
        "readiness": _readiness_from_checks(checks),
        "required_source_kinds": _string_list(
            matrix_report.get("required_source_kinds")
        ),
        "source_kind_counts": _int_mapping(matrix_report.get("source_kind_counts")),
        "source_result_matrix": rows,
        "summary": _summary_from_rows(rows),
        "checks": checks,
    }
    report["report_digest"] = offline_control_result_report_digest(report)
    return report


def offline_control_result_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def offline_control_result_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_offline_control_result_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(offline_control_result_report_json(report), encoding="utf-8")
    return output_path


def load_offline_control_result_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Offline control result report JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_offline_control_result_report(report: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = report.get("schema_version")
    report_digest = _string_or_none(report.get("report_digest"))
    expected_digest = offline_control_result_report_digest(report)
    rows = _mapping_sequence(report.get("source_result_matrix"))
    expected_summary = _summary_from_rows(rows)
    expected_checks = _checks_from_saved_report(report, rows)
    expected_readiness = _readiness_from_checks(expected_checks)
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == OFFLINE_CONTROL_RESULT_REPORT_SCHEMA_VERSION,
            "expected": OFFLINE_CONTROL_RESULT_REPORT_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "report_digest",
            "passed": report_digest == expected_digest,
            "expected": expected_digest,
            "actual": report_digest,
        },
        _equality_check("summary", expected_summary, report.get("summary")),
        _equality_check("checks", expected_checks, report.get("checks")),
        _equality_check("readiness", expected_readiness, report.get("readiness")),
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "report_digest": report_digest,
        "checks": checks,
    }


def compare_offline_control_result_report(report: Mapping[str, Any]) -> dict[str, Any]:
    matrix_report_path = _required_report_path(report, "matrix_report_path")
    candidate_report_path = _required_report_path(
        report,
        "candidate_qa_eval_report_path",
    )
    delta_paths = {
        _required_str(row, "source_key"): _required_report_path(
            row,
            "qa_eval_delta_report_path",
        )
        for row in _mapping_sequence(report.get("source_result_matrix"))
    }
    current_report = offline_control_result_report(
        load_offline_control_matrix_report(matrix_report_path),
        matrix_report_path=matrix_report_path,
        candidate_qa_eval_report_path=candidate_report_path,
        qa_eval_delta_report_paths=delta_paths,
    )
    validation = validate_offline_control_result_report(report)
    saved_digest = _string_or_none(report.get("report_digest"))
    current_digest = _string_or_none(current_report.get("report_digest"))
    checks = [
        {
            "name": "saved_report_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        _equality_check(
            "source_result_matrix_matches_current",
            report.get("source_result_matrix"),
            current_report["source_result_matrix"],
        ),
        _equality_check(
            "summary_matches_current",
            report.get("summary"),
            current_report["summary"],
        ),
        _equality_check(
            "checks_match_current",
            report.get("checks"),
            current_report["checks"],
        ),
        _equality_check(
            "readiness_matches_current",
            report.get("readiness"),
            current_report["readiness"],
        ),
        _equality_check("report_digest_matches_current", saved_digest, current_digest),
    ]
    return {
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def _source_result_matrix(
    matrix_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
    delta_paths: Mapping[str, Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_row in _mapping_sequence(matrix_report.get("source_profile_matrix")):
        source_key = _required_str(source_row, "source_key")
        delta_path = delta_paths.get(source_key)
        delta_report = (
            load_qa_eval_delta_report(delta_path) if delta_path is not None else None
        )
        summary_delta = (
            _mapping(delta_report.get("summary_delta"), "summary_delta")
            if delta_report is not None
            else {}
        )
        rows.append(
            {
                "baseline_exact_match_rate": _float_or_none(
                    summary_delta.get("baseline_exact_match_rate")
                ),
                "baseline_name": (
                    _string_or_none(delta_report.get("baseline_name"))
                    if delta_report is not None
                    else None
                ),
                "baseline_qa_eval_report_digest": (
                    _string_or_none(delta_report.get("baseline_report_digest"))
                    if delta_report is not None
                    else None
                ),
                "baseline_qa_eval_report_path": (
                    _string_or_none(delta_report.get("baseline_report_path"))
                    if delta_report is not None
                    else None
                ),
                "candidate_exact_match_rate": _float_or_none(
                    summary_delta.get("candidate_exact_match_rate")
                ),
                "candidate_name": (
                    _string_or_none(delta_report.get("candidate_name"))
                    if delta_report is not None
                    else None
                ),
                "candidate_report_digest": (
                    _string_or_none(delta_report.get("candidate_report_digest"))
                    if delta_report is not None
                    else None
                ),
                "case_count_match": (
                    summary_delta.get("case_count_match") is True
                    if delta_report is not None
                    else False
                ),
                "dataset_id": _string_or_none(source_row.get("dataset_id")),
                "exact_match_count_delta": _int_or_none(
                    summary_delta.get("exact_match_count_delta")
                ),
                "exact_match_rate_delta": _float_or_none(
                    summary_delta.get("exact_match_rate_delta")
                ),
                "import_report_digest": _string_or_none(
                    source_row.get("report_digest")
                ),
                "import_report_path": _string_or_none(source_row.get("path")),
                "model_id": _string_or_none(source_row.get("model_id")),
                "prompt_id": _string_or_none(source_row.get("prompt_id")),
                "qa_eval_delta_report_digest": (
                    qa_eval_delta_report_digest(delta_report)
                    if delta_report is not None
                    else None
                ),
                "qa_eval_delta_report_path": (
                    str(delta_path) if delta_path is not None else None
                ),
                "source_key": source_key,
                "source_kind": _required_str(source_row, "source_kind"),
                "source_name": _required_str(source_row, "source_name"),
            }
        )
    return sorted(rows, key=lambda row: str(row["source_key"]))


def _checks_from_rows(
    *,
    matrix_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    matrix_validation = validate_offline_control_matrix_report(matrix_report)
    candidate_validation = validate_qa_eval_report(candidate_report)
    matrix_readiness = _mapping(matrix_report.get("readiness"), "readiness")
    candidate_digest = qa_eval_report_digest(candidate_report)
    delta_validation_failures = _delta_validation_failures(rows)
    return _checks_from_values(
        candidate_digest=candidate_digest,
        candidate_valid=candidate_validation["valid"] is True,
        delta_validation_failures=delta_validation_failures,
        matrix_ready=matrix_readiness.get("ready") is True,
        matrix_valid=matrix_validation["valid"] is True,
        required_source_kinds=_string_list(matrix_report.get("required_source_kinds")),
        rows=rows,
    )


def _checks_from_saved_report(
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return _checks_from_values(
        candidate_digest=_string_or_none(
            report.get("candidate_qa_eval_report_digest")
        ),
        candidate_valid=True,
        delta_validation_failures=[],
        matrix_ready=_mapping(report.get("matrix_readiness"), "matrix_readiness").get(
            "ready"
        )
        is True,
        matrix_valid=True,
        required_source_kinds=_string_list(report.get("required_source_kinds")),
        rows=rows,
    )


def _checks_from_values(
    *,
    candidate_digest: str | None,
    candidate_valid: bool,
    delta_validation_failures: Sequence[str],
    matrix_ready: bool,
    matrix_valid: bool,
    required_source_kinds: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    missing_delta_sources = sorted(
        _required_str(row, "source_key")
        for row in rows
        if _string_or_none(row.get("qa_eval_delta_report_path")) is None
    )
    kinds_with_delta = {
        _required_str(row, "source_kind")
        for row in rows
        if _string_or_none(row.get("qa_eval_delta_report_path")) is not None
    }
    missing_required_delta_kinds = [
        kind for kind in required_source_kinds if kind not in kinds_with_delta
    ]
    mismatched_candidate_sources = sorted(
        _required_str(row, "source_key")
        for row in rows
        if _string_or_none(row.get("candidate_report_digest")) != candidate_digest
    )
    mismatched_case_sources = sorted(
        _required_str(row, "source_key")
        for row in rows
        if row.get("case_count_match") is not True
    )
    return [
        {
            "name": "matrix_report_valid",
            "passed": matrix_valid,
        },
        {
            "name": "matrix_report_ready",
            "passed": matrix_ready,
        },
        {
            "name": "candidate_eval_report_valid",
            "passed": candidate_valid,
        },
        {
            "name": "all_sources_have_delta",
            "passed": len(missing_delta_sources) == 0,
            "actual": missing_delta_sources,
        },
        {
            "name": "required_source_kind_deltas_present",
            "passed": len(missing_required_delta_kinds) == 0,
            "required": list(required_source_kinds),
            "missing": missing_required_delta_kinds,
        },
        {
            "name": "delta_reports_valid",
            "passed": len(delta_validation_failures) == 0,
            "actual": list(delta_validation_failures),
        },
        {
            "name": "same_candidate_report_digest",
            "passed": len(mismatched_candidate_sources) == 0,
            "actual": mismatched_candidate_sources,
        },
        {
            "name": "case_counts_match",
            "passed": len(mismatched_case_sources) == 0,
            "actual": mismatched_case_sources,
        },
    ]


def _delta_validation_failures(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    failures: list[str] = []
    for row in rows:
        path = _string_or_none(row.get("qa_eval_delta_report_path"))
        if path is None:
            continue
        report = load_qa_eval_delta_report(path)
        if validate_qa_eval_delta_report(report)["valid"] is not True:
            failures.append(_required_str(row, "source_key"))
    return sorted(failures)


def _summary_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [
        _float_or_none(row.get("exact_match_rate_delta"))
        for row in rows
        if _string_or_none(row.get("qa_eval_delta_report_path")) is not None
    ]
    candidate_rates = sorted(
        {
            rate
            for rate in (
                _float_or_none(row.get("candidate_exact_match_rate")) for row in rows
            )
            if rate is not None
        }
    )
    return {
        "candidate_exact_match_rate": (
            candidate_rates[0] if len(candidate_rates) == 1 else None
        ),
        "delta_report_count": len(deltas),
        "improved_source_count": sum(1 for delta in deltas if delta is not None and delta > 0),
        "regressed_source_count": sum(1 for delta in deltas if delta is not None and delta < 0),
        "source_count": len(rows),
        "unchanged_source_count": sum(1 for delta in deltas if delta == 0),
    }


def _candidate_name(rows: Sequence[Mapping[str, Any]]) -> str | None:
    names = sorted(
        {
            name
            for name in (_string_or_none(row.get("candidate_name")) for row in rows)
            if name is not None
        }
    )
    if len(names) != 1:
        return None
    return names[0]


def _readiness_from_checks(checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed = [
        _required_str(check, "name")
        for check in checks
        if check.get("passed") is not True
    ]
    return {
        "ready": len(failed) == 0,
        "failed_check_count": len(failed),
        "failed_checks": failed,
    }


def _path_mapping(paths: Mapping[str, str | Path]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for key, value in sorted(paths.items()):
        if not isinstance(key, str) or key == "":
            raise SpatialQAError("Offline control result delta paths require source keys")
        result[key] = Path(value)
    return result


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpatialQAError(f"Offline control result field must be an object: {field}")
    return cast(Mapping[str, Any], value)


def _mapping_sequence(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return ()
    return tuple(cast(Mapping[str, Any], item) for item in value if isinstance(item, Mapping))


def _int_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): item
        for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        if isinstance(item, int) and not isinstance(item, bool)
    }


def _required_report_path(payload: Mapping[str, Any], key: str) -> Path:
    value = payload.get(key)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Offline control result report path is required: {key}")
    return Path(value)


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Offline control result field is required: {key}")
    return value


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str) and value != "":
        return value
    return None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return sorted(item for item in value if isinstance(item, str))


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value), 6)


def _equality_check(name: str, expected: object, actual: object) -> dict[str, Any]:
    return {
        "name": name,
        "passed": expected == actual,
        "expected": expected,
        "actual": actual,
    }
