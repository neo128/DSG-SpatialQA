from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.eval.offline_predictions import (
    load_offline_prediction_import_report,
    offline_prediction_import_report_digest,
    validate_offline_prediction_import_report,
)
from dsg_spatialqa_lab.schema import SpatialQAError


OFFLINE_CONTROL_MATRIX_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.offline-control-matrix-report.v1"
)
DEFAULT_REQUIRED_OFFLINE_CONTROL_SOURCE_KINDS: tuple[str, ...] = (
    "caption_memory",
    "graph_text",
    "multi_frame_vlm",
    "vlm",
)


def offline_control_matrix_report(
    import_reports: Sequence[Mapping[str, Any]],
    *,
    report_paths: Sequence[str | Path] | None = None,
    required_source_kinds: Sequence[str] = DEFAULT_REQUIRED_OFFLINE_CONTROL_SOURCE_KINDS,
) -> dict[str, Any]:
    paths = _optional_paths(report_paths, len(import_reports))
    required_kinds = _unique_strings(required_source_kinds, "required_source_kinds")
    rows = [
        _source_profile_row(import_report, paths[index])
        for index, import_report in enumerate(import_reports)
    ]
    rows = sorted(rows, key=lambda row: row["source_key"])
    source_kind_counts = _source_kind_counts(rows)
    summary = _summary(rows, required_kinds, source_kind_counts)
    checks = _checks(import_reports, rows, required_kinds, summary)
    report: dict[str, Any] = {
        "schema_version": OFFLINE_CONTROL_MATRIX_REPORT_SCHEMA_VERSION,
        "required_source_kinds": list(required_kinds),
        "report_count": len(import_reports),
        "source_kind_counts": source_kind_counts,
        "source_profile_matrix": rows,
        "summary": summary,
        "checks": checks,
        "readiness": _readiness_from_checks(checks),
    }
    report["report_digest"] = offline_control_matrix_report_digest(report)
    return report


def offline_control_matrix_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def offline_control_matrix_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_offline_control_matrix_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(offline_control_matrix_report_json(report), encoding="utf-8")
    return output_path


def load_offline_control_matrix_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Offline control matrix report JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_offline_control_matrix_report(report: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = report.get("schema_version")
    report_digest = _string_or_none(report.get("report_digest"))
    expected_digest = offline_control_matrix_report_digest(report)
    rows = _mapping_sequence(report.get("source_profile_matrix"))
    required_kinds = _string_tuple_from_value(report.get("required_source_kinds"))
    source_kind_counts = _source_kind_counts(rows)
    summary = _summary(rows, required_kinds, source_kind_counts)
    checks = _checks_from_rows(rows, required_kinds, summary)
    expected_readiness = _readiness_from_checks(checks)
    checks_out = [
        {
            "name": "schema_version",
            "passed": schema_version == OFFLINE_CONTROL_MATRIX_REPORT_SCHEMA_VERSION,
            "expected": OFFLINE_CONTROL_MATRIX_REPORT_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "report_digest",
            "passed": report_digest == expected_digest,
            "expected": expected_digest,
            "actual": report_digest,
        },
        _equality_check(
            "source_kind_counts",
            source_kind_counts,
            report.get("source_kind_counts"),
        ),
        _equality_check("summary", summary, report.get("summary")),
        _equality_check("checks", checks, report.get("checks")),
        _equality_check("readiness", expected_readiness, report.get("readiness")),
    ]
    return {
        "valid": all(check["passed"] is True for check in checks_out),
        "schema_version": schema_version,
        "report_digest": report_digest,
        "checks": checks_out,
    }


def compare_offline_control_matrix_report(report: Mapping[str, Any]) -> dict[str, Any]:
    report_paths = _required_row_paths(report)
    current_import_reports = tuple(
        load_offline_prediction_import_report(path) for path in report_paths
    )
    current_report = offline_control_matrix_report(
        current_import_reports,
        report_paths=report_paths,
        required_source_kinds=_string_tuple(report, "required_source_kinds"),
    )
    validation = validate_offline_control_matrix_report(report)
    saved_digest = _string_or_none(report.get("report_digest"))
    current_digest = _string_or_none(current_report.get("report_digest"))
    checks = [
        {"name": "saved_report_valid", "passed": validation["valid"] is True},
        _equality_check(
            "source_profile_matrix_matches_current",
            report.get("source_profile_matrix"),
            current_report["source_profile_matrix"],
        ),
        _equality_check(
            "source_kind_counts_match_current",
            report.get("source_kind_counts"),
            current_report["source_kind_counts"],
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
        _equality_check(
            "report_digest_matches_current",
            saved_digest,
            current_digest,
        ),
    ]
    return {
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def _source_profile_row(
    import_report: Mapping[str, Any],
    path: Path | None,
) -> dict[str, Any]:
    source_profile = _mapping(import_report.get("source_profile"), "source_profile")
    summary = _mapping(import_report.get("summary"), "summary")
    source_key = _required_str(source_profile, "source_key")
    source_kind = _required_str(source_profile, "kind")
    source_name = _required_str(source_profile, "name")
    return {
        "adapter": _optional_str(source_profile, "adapter"),
        "capability_axes": _string_list(source_profile.get("capability_axes")),
        "dataset_id": _optional_str(source_profile, "dataset_id"),
        "duplicate_case_count": _int_value(summary, "duplicate_case_count"),
        "error_prediction_count": _int_value(summary, "error_prediction_count"),
        "gold_case_count": _int_value(summary, "gold_case_count"),
        "imported_prediction_count": _int_value(summary, "imported_prediction_count"),
        "metadata_keys": _string_list(source_profile.get("metadata_keys")),
        "missing_case_count": _int_value(summary, "missing_case_count"),
        "model_id": _optional_str(source_profile, "model_id"),
        "path": str(path) if path is not None else None,
        "prediction_digest": _string_or_none(import_report.get("prediction_digest")),
        "prompt_id": _optional_str(source_profile, "prompt_id"),
        "qa_digest": _string_or_none(import_report.get("qa_digest")),
        "report_digest": offline_prediction_import_report_digest(import_report),
        "source_key": source_key,
        "source_kind": source_kind,
        "source_name": source_name,
        "unknown_case_count": _int_value(summary, "unknown_case_count"),
    }


def _checks(
    import_reports: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    required_source_kinds: Sequence[str],
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import_report_failures = [
        _string_or_none(row.get("source_key")) or f"row:{index + 1}"
        for index, (row, import_report) in enumerate(zip(rows, import_reports, strict=True))
        if validate_offline_prediction_import_report(import_report)["valid"] is not True
    ]
    checks = _checks_from_rows(rows, required_source_kinds, summary)
    checks[0] = {
        "name": "import_reports_valid",
        "passed": len(import_report_failures) == 0,
        "actual": import_report_failures,
    }
    return checks


def _checks_from_rows(
    rows: Sequence[Mapping[str, Any]],
    required_source_kinds: Sequence[str],
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source_keys = [_required_str(row, "source_key") for row in rows]
    duplicate_source_keys = sorted(
        {source_key for source_key in source_keys if source_keys.count(source_key) > 1}
    )
    qa_digests = sorted(
        {
            digest
            for digest in (_string_or_none(row.get("qa_digest")) for row in rows)
            if digest is not None
        }
    )
    incomplete_source_keys = _string_list(summary.get("incomplete_source_keys"))
    diagnostic_source_keys = _string_list(summary.get("diagnostic_source_keys"))
    missing_required = _string_list(summary.get("missing_required_source_kinds"))
    return [
        {
            "name": "import_reports_valid",
            "passed": True,
            "actual": [],
        },
        {
            "name": "required_source_kinds_present",
            "passed": len(missing_required) == 0,
            "required": list(required_source_kinds),
            "missing": missing_required,
        },
        {
            "name": "complete_prediction_coverage",
            "passed": len(incomplete_source_keys) == 0,
            "actual": incomplete_source_keys,
        },
        {
            "name": "clean_import_diagnostics",
            "passed": len(diagnostic_source_keys) == 0,
            "actual": diagnostic_source_keys,
        },
        {
            "name": "unique_source_keys",
            "passed": len(duplicate_source_keys) == 0,
            "actual": duplicate_source_keys,
        },
        {
            "name": "qa_digest_consistent",
            "passed": len(qa_digests) <= 1,
            "actual": qa_digests,
        },
    ]


def _summary(
    rows: Sequence[Mapping[str, Any]],
    required_source_kinds: Sequence[str],
    source_kind_counts: Mapping[str, int],
) -> dict[str, Any]:
    source_kinds = sorted(source_kind_counts)
    missing_required = [
        kind for kind in required_source_kinds if source_kind_counts.get(kind, 0) == 0
    ]
    incomplete_source_keys = sorted(
        _required_str(row, "source_key")
        for row in rows
        if _int_value(row, "missing_case_count") != 0
        or _int_value(row, "imported_prediction_count") != _int_value(
            row,
            "gold_case_count",
        )
    )
    diagnostic_source_keys = sorted(
        _required_str(row, "source_key")
        for row in rows
        if _int_value(row, "unknown_case_count") != 0
        or _int_value(row, "duplicate_case_count") != 0
        or _int_value(row, "error_prediction_count") != 0
    )
    return {
        "complete_source_count": len(rows) - len(incomplete_source_keys),
        "diagnostic_source_keys": diagnostic_source_keys,
        "incomplete_source_keys": incomplete_source_keys,
        "missing_required_source_kinds": missing_required,
        "required_source_kind_count": len(required_source_kinds),
        "source_count": len(rows),
        "source_kind_count": len(source_kinds),
        "source_kinds": source_kinds,
        "total_imported_prediction_count": sum(
            _int_value(row, "imported_prediction_count") for row in rows
        ),
    }


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


def _source_kind_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        source_kind = _required_str(row, "source_kind")
        counts[source_kind] = counts.get(source_kind, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def _required_row_paths(report: Mapping[str, Any]) -> tuple[Path, ...]:
    rows = _mapping_sequence(report.get("source_profile_matrix"))
    paths: list[Path] = []
    for row in rows:
        path = _string_or_none(row.get("path"))
        if path is None:
            raise SpatialQAError("Offline control matrix rows require paths for comparison")
        paths.append(Path(path))
    return tuple(paths)


def _optional_paths(
    report_paths: Sequence[str | Path] | None,
    expected_count: int,
) -> tuple[Path | None, ...]:
    if report_paths is None:
        return tuple(None for _ in range(expected_count))
    if len(report_paths) != expected_count:
        raise SpatialQAError("Offline control matrix report_paths length must match import_reports")
    return tuple(Path(path) for path in report_paths)


def _unique_strings(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    strings: list[str] = []
    for value in values:
        if not isinstance(value, str) or value == "":
            raise SpatialQAError(f"{field_name} entries must be non-empty strings")
        strings.append(value)
    return tuple(sorted(set(strings)))


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpatialQAError(f"Offline control matrix field must be an object: {field_name}")
    return cast(Mapping[str, Any], value)


def _mapping_sequence(value: object) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return ()
    return tuple(cast(Mapping[str, Any], item) for item in value if isinstance(item, Mapping))


def _string_tuple(report: Mapping[str, Any], field_name: str) -> tuple[str, ...]:
    values = _string_tuple_from_value(report.get(field_name))
    if not values:
        raise SpatialQAError(f"Offline control matrix field must contain strings: {field_name}")
    return values


def _string_tuple_from_value(value: object) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return ()
    strings: list[str] = []
    for item in value:
        if isinstance(item, str) and item != "":
            strings.append(item)
    return tuple(strings)


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Offline control matrix field must be a non-empty string: {key}")
    return value


def _optional_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value == "":
        return "unspecified"
    return value


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value != "" else None


def _string_list(value: object) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return []
    return sorted(str(item) for item in value if isinstance(item, str))


def _int_value(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _equality_check(name: str, expected: object, actual: object) -> dict[str, Any]:
    return {
        "name": name,
        "passed": expected == actual,
        "expected": expected,
        "actual": actual,
    }
