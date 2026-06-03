from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.benchmark.experiment_summary import (
    experiment_summary_report_digest,
    load_experiment_summary_report,
)
from dsg_spatialqa_lab.benchmark.readiness import (
    load_real_experiment_readiness_report,
    real_experiment_readiness_report_digest,
    validate_real_experiment_readiness_report,
)
from dsg_spatialqa_lab.schema import SpatialQAError
from dsg_spatialqa_lab.visualization.dashboard_export import (
    dashboard_bundle_digest,
    load_dashboard_bundle,
)


EXPERIMENT_RECORD_SCHEMA_VERSION = "dsg-spatialqa-lab.experiment-record.v1"
RESEARCH_QUESTION_KEYS = (
    "dynamic_memory",
    "graph_tool_query",
    "interactive_task",
    "spatial_qa",
)
VERDICT_KEYS = ("improved", "inconclusive", "regressed", "unchanged")


def experiment_record(
    summary_report: Mapping[str, Any],
    *,
    summary_report_path: str | Path | None = None,
    dashboard_bundle: Mapping[str, Any] | None = None,
    dashboard_bundle_path: str | Path | None = None,
    real_readiness_report: Mapping[str, Any] | None = None,
    real_readiness_report_path: str | Path | None = None,
) -> dict[str, Any]:
    research_question_verdicts = _research_question_verdicts(summary_report)
    readiness = _mapping_or_empty(summary_report.get("readiness"))
    summary = _mapping_or_empty(summary_report.get("summary"))
    record: dict[str, Any] = {
        "schema_version": EXPERIMENT_RECORD_SCHEMA_VERSION,
        "summary_report_path": (
            str(summary_report_path) if summary_report_path is not None else None
        ),
        "summary_report_digest": experiment_summary_report_digest(summary_report),
        "manifest_path": _string_or_none(summary_report.get("manifest_path")),
        "manifest_digest": _string_or_none(summary_report.get("manifest_digest")),
        "readiness_status": _string_or_none(readiness.get("status")),
        "readiness": _json_value(readiness),
        "verdict_counts": _verdict_counts(research_question_verdicts),
        "research_question_verdicts": research_question_verdicts,
        "research_question_matrix": _research_question_matrix(summary_report),
        "source_profile_matrix": _source_profile_matrix(summary_report),
        "source_profile_count": _source_profile_count(summary_report),
        "diagnostic_ledger": _diagnostic_ledger(summary_report),
        "source_artifact_digests": _json_mapping(
            summary_report.get("source_artifact_digests")
        ),
        "source_artifact_count": _int_or_none(summary.get("source_artifact_count")),
    }
    if dashboard_bundle is not None:
        record["dashboard_bundle"] = _dashboard_record(
            dashboard_bundle,
            dashboard_bundle_path=dashboard_bundle_path,
        )
    if real_readiness_report is not None:
        real_package_readiness = _real_package_readiness_record(
            real_readiness_report
        )
        record["real_readiness_report_path"] = (
            str(real_readiness_report_path)
            if real_readiness_report_path is not None
            else None
        )
        record["real_readiness_report_digest"] = real_package_readiness[
            "report_digest"
        ]
        record["real_package_status"] = (
            "ready"
            if real_package_readiness["valid"] is True
            and real_package_readiness["ready"] is True
            else "not_ready"
        )
        record["real_package_readiness"] = real_package_readiness
    record["record_digest"] = experiment_record_digest(record)
    return record


def experiment_record_digest(record: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "record_digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def experiment_record_json(record: Mapping[str, Any]) -> str:
    return json.dumps(record, indent=2, sort_keys=True) + "\n"


def save_experiment_record(record: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(experiment_record_json(record), encoding="utf-8")
    return output_path


def load_experiment_record(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Experiment record JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_experiment_record(record: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = record.get("schema_version")
    record_digest = _string_or_none(record.get("record_digest"))
    expected_record_digest = experiment_record_digest(record)
    research_question_verdicts = _mapping_or_empty(
        record.get("research_question_verdicts")
    )
    readiness = _mapping_or_empty(record.get("readiness"))
    research_question_matrix = _mapping_sequence(record.get("research_question_matrix"))
    source_profile_matrix = _mapping_sequence(record.get("source_profile_matrix"))
    diagnostic_ledger = _mapping_or_empty(record.get("diagnostic_ledger"))
    expected_verdict_counts = _verdict_counts(research_question_verdicts)
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == EXPERIMENT_RECORD_SCHEMA_VERSION,
            "expected": EXPERIMENT_RECORD_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "record_digest",
            "passed": record_digest == expected_record_digest,
            "expected": expected_record_digest,
            "actual": record_digest,
        },
        {
            "name": "readiness_status",
            "passed": record.get("readiness_status") == readiness.get("status"),
            "expected": readiness.get("status"),
            "actual": record.get("readiness_status"),
        },
        {
            "name": "research_question_verdict_count",
            "passed": len(research_question_verdicts) == len(RESEARCH_QUESTION_KEYS),
            "expected": len(RESEARCH_QUESTION_KEYS),
            "actual": len(research_question_verdicts),
        },
        {
            "name": "verdict_counts",
            "passed": record.get("verdict_counts") == expected_verdict_counts,
            "expected": expected_verdict_counts,
            "actual": record.get("verdict_counts"),
        },
        {
            "name": "research_question_matrix_count",
            "passed": len(research_question_matrix)
            == _research_question_measurement_count(research_question_verdicts),
            "expected": _research_question_measurement_count(
                research_question_verdicts
            ),
            "actual": len(research_question_matrix),
        },
        {
            "name": "diagnostic_ledger",
            "passed": _validate_diagnostic_ledger(diagnostic_ledger),
        },
        {
            "name": "source_profile_count",
            "passed": record.get("source_profile_count")
            == len(source_profile_matrix),
            "expected": len(source_profile_matrix),
            "actual": record.get("source_profile_count"),
        },
    ]
    dashboard = record.get("dashboard_bundle")
    if dashboard is not None:
        checks.append(
            {
                "name": "dashboard_bundle",
                "passed": _validate_dashboard_record(dashboard),
            }
        )
    real_package_readiness = record.get("real_package_readiness")
    if real_package_readiness is not None:
        real_package = _mapping_or_empty(real_package_readiness)
        expected_status = (
            "ready"
            if real_package.get("valid") is True and real_package.get("ready") is True
            else "not_ready"
        )
        checks.extend(
            [
                {
                    "name": "real_package_status",
                    "passed": record.get("real_package_status") == expected_status,
                    "expected": expected_status,
                    "actual": record.get("real_package_status"),
                },
                {
                    "name": "real_package_manifest_digest",
                    "passed": real_package.get("manifest_digest")
                    == record.get("manifest_digest"),
                    "expected": record.get("manifest_digest"),
                    "actual": real_package.get("manifest_digest"),
                },
                {
                    "name": "real_package_ready",
                    "passed": record.get("real_package_status") == "ready",
                    "expected": "ready",
                    "actual": record.get("real_package_status"),
                },
            ]
        )
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "record_digest": record_digest,
        "checks": checks,
    }


def compare_experiment_record(record: Mapping[str, Any]) -> dict[str, Any]:
    summary_report_path = _required_record_path(record, "summary_report_path")
    dashboard_path = _dashboard_path(record.get("dashboard_bundle"))
    real_readiness_path = _optional_record_path(
        record,
        "real_readiness_report_path",
    )
    current_record = experiment_record(
        load_experiment_summary_report(summary_report_path),
        summary_report_path=summary_report_path,
        dashboard_bundle=load_dashboard_bundle(dashboard_path)
        if dashboard_path is not None
        else None,
        dashboard_bundle_path=dashboard_path,
        real_readiness_report=load_real_experiment_readiness_report(
            real_readiness_path
        )
        if real_readiness_path is not None
        else None,
        real_readiness_report_path=real_readiness_path,
    )
    validation = validate_experiment_record(record)
    saved_digest = _string_or_none(record.get("record_digest"))
    current_digest = _string_or_none(current_record.get("record_digest"))
    checks = [
        {
            "name": "record_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        {
            "name": "record_digest_matches_current",
            "passed": saved_digest == current_digest,
            "expected": saved_digest,
            "actual": current_digest,
        },
        _equality_check(
            "summary_report_digest_matches_current",
            record.get("summary_report_digest"),
            current_record["summary_report_digest"],
        ),
        _equality_check(
            "readiness_matches_current",
            record.get("readiness"),
            current_record["readiness"],
        ),
        _equality_check(
            "verdict_counts_match_current",
            record.get("verdict_counts"),
            current_record["verdict_counts"],
        ),
        _equality_check(
            "research_question_verdicts_match_current",
            record.get("research_question_verdicts"),
            current_record["research_question_verdicts"],
        ),
        _equality_check(
            "research_question_matrix_match_current",
            record.get("research_question_matrix"),
            current_record["research_question_matrix"],
        ),
        _equality_check(
            "diagnostic_ledger_match_current",
            record.get("diagnostic_ledger"),
            current_record["diagnostic_ledger"],
        ),
        _equality_check(
            "source_profile_matrix_match_current",
            record.get("source_profile_matrix"),
            current_record["source_profile_matrix"],
        ),
        _equality_check(
            "source_artifact_digests_match_current",
            record.get("source_artifact_digests"),
            current_record["source_artifact_digests"],
        ),
    ]
    if dashboard_path is not None:
        checks.append(
            _equality_check(
                "dashboard_bundle_matches_current",
                record.get("dashboard_bundle"),
                current_record.get("dashboard_bundle"),
            )
        )
    if real_readiness_path is not None:
        checks.extend(
            [
                _equality_check(
                    "real_package_status_matches_current",
                    record.get("real_package_status"),
                    current_record.get("real_package_status"),
                ),
                _equality_check(
                    "real_package_readiness_matches_current",
                    record.get("real_package_readiness"),
                    current_record.get("real_package_readiness"),
                ),
            ]
        )
    return {
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def _research_question_verdicts(summary_report: Mapping[str, Any]) -> dict[str, Any]:
    research_questions = _mapping_or_empty(summary_report.get("research_questions"))
    return {
        key: _research_question_verdict(_mapping_or_empty(research_questions.get(key)))
        for key in RESEARCH_QUESTION_KEYS
    }


def _research_question_verdict(question: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "measurement_count": len(_mapping_sequence(question.get("measurements"))),
        "primary_metric": _json_value(question.get("primary_metric")),
        "source_artifact_type": _string_or_none(question.get("source_artifact_type")),
        "status": _string_or_none(question.get("status")),
        "verdict": _string_or_none(question.get("verdict")),
    }


def _research_question_matrix(summary_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    research_questions = _mapping_or_empty(summary_report.get("research_questions"))
    rows: list[dict[str, Any]] = []
    for question_name, question_value in sorted(
        research_questions.items(),
        key=lambda item: str(item[0]),
    ):
        if not isinstance(question_value, Mapping):
            continue
        measurements = question_value.get("measurements")
        if not isinstance(measurements, Sequence) or isinstance(measurements, str):
            continue
        for measurement in measurements:
            if not isinstance(measurement, Mapping):
                continue
            primary_metric = _json_mapping(measurement.get("primary_metric"))
            rows.append(
                {
                    "artifact_key": _string_or_none(measurement.get("artifact_key")),
                    "baseline_name": _string_or_none(
                        measurement.get("baseline_name")
                    ),
                    "candidate_name": _string_or_none(
                        measurement.get("candidate_name")
                    ),
                    "case_count_match": measurement.get("case_count_match")
                    if isinstance(measurement.get("case_count_match"), bool)
                    else None,
                    "label": _string_or_none(question_value.get("label")),
                    "measurement_verdict": _measurement_verdict(
                        primary_metric,
                        measurement.get("case_count_match"),
                    ),
                    "primary_metric": primary_metric,
                    "question_verdict": _string_or_none(question_value.get("verdict")),
                    "research_question": str(question_name),
                    "source_artifact_type": _string_or_none(
                        question_value.get("source_artifact_type")
                    ),
                    "status": _string_or_none(question_value.get("status")),
                    "supporting_metrics": _json_mapping(
                        measurement.get("supporting_metrics")
                    ),
                }
            )
    return rows


def _source_profile_matrix(summary_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        _json_mapping(row)
        for row in _mapping_sequence(summary_report.get("source_profile_matrix"))
    ]


def _source_profile_count(summary_report: Mapping[str, Any]) -> int:
    summary = _mapping_or_empty(summary_report.get("summary"))
    count = _int_or_none(summary.get("source_profile_count"))
    if count is not None:
        return count
    return len(_mapping_sequence(summary_report.get("source_profile_matrix")))


def _diagnostic_ledger(summary_report: Mapping[str, Any]) -> dict[str, Any]:
    qa_diagnostic_slices = _mapping_or_empty(
        summary_report.get("qa_diagnostic_slices")
    )
    graph_diagnostics = _mapping_or_empty(
        summary_report.get("graph_construction_diagnostics")
    )
    attribution_diagnostics = _mapping_or_empty(
        summary_report.get("error_attribution_diagnostics")
    )
    failure_linkages = _mapping_or_empty(
        summary_report.get("failure_linkage_diagnostics")
    )
    summary = _mapping_or_empty(summary_report.get("summary"))
    return {
        "error_attribution_artifact_keys": sorted(
            str(key) for key in attribution_diagnostics
        ),
        "error_attribution_diagnostic_count": _ledger_count(
            summary,
            "error_attribution_diagnostic_count",
            attribution_diagnostics,
        ),
        "failure_linkage_pair_count": len(_failure_linkage_pairs(failure_linkages)),
        "failure_linkage_pairs": _failure_linkage_pairs(failure_linkages),
        "graph_construction_artifact_keys": sorted(
            str(key) for key in graph_diagnostics
        ),
        "graph_construction_diagnostic_count": _ledger_count(
            summary,
            "graph_construction_diagnostic_count",
            graph_diagnostics,
        ),
        "qa_diagnostic_slice_count": _ledger_count(
            summary,
            "qa_diagnostic_slice_count",
            qa_diagnostic_slices,
        ),
        "qa_diagnostic_slice_keys": sorted(str(key) for key in qa_diagnostic_slices),
    }


def _failure_linkage_pairs(failure_linkages: Mapping[str, Any]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for artifact_key, row in sorted(failure_linkages.items(), key=lambda item: str(item[0])):
        if not isinstance(row, Mapping):
            continue
        error_key = row.get("error_attribution_artifact_key")
        pairs.append(
            {
                "error_attribution_artifact_key": (
                    error_key if isinstance(error_key, str) else str(artifact_key)
                ),
                "graph_eval_artifact_key": _string_or_none(
                    row.get("graph_eval_artifact_key")
                ),
                "linked_by": _string_or_none(row.get("linked_by")),
            }
        )
    return pairs


def _ledger_count(
    summary: Mapping[str, Any],
    key: str,
    diagnostics: Mapping[str, Any],
) -> int:
    value = _int_or_none(summary.get(key))
    return value if value is not None else len(diagnostics)


def _validate_diagnostic_ledger(ledger: Mapping[str, Any]) -> bool:
    if not ledger:
        return False
    qa_keys = _string_sequence(ledger.get("qa_diagnostic_slice_keys"))
    graph_keys = _string_sequence(ledger.get("graph_construction_artifact_keys"))
    attribution_keys = _string_sequence(
        ledger.get("error_attribution_artifact_keys")
    )
    failure_pairs = _mapping_sequence(ledger.get("failure_linkage_pairs"))
    qa_count = _int_or_none(ledger.get("qa_diagnostic_slice_count"))
    return (
        qa_count is not None
        and qa_count >= len(qa_keys)
        and _int_or_none(ledger.get("graph_construction_diagnostic_count"))
        == len(graph_keys)
        and _int_or_none(ledger.get("error_attribution_diagnostic_count"))
        == len(attribution_keys)
        and _int_or_none(ledger.get("failure_linkage_pair_count"))
        == len(failure_pairs)
        and all(_validate_failure_linkage_pair(pair) for pair in failure_pairs)
    )


def _validate_failure_linkage_pair(pair: Mapping[str, Any]) -> bool:
    return (
        isinstance(pair.get("error_attribution_artifact_key"), str)
        and (
            pair.get("graph_eval_artifact_key") is None
            or isinstance(pair.get("graph_eval_artifact_key"), str)
        )
        and (
            pair.get("linked_by") is None
            or isinstance(pair.get("linked_by"), str)
        )
    )


def _measurement_verdict(primary_metric: Mapping[str, Any], case_count_match: object) -> str:
    if case_count_match is False:
        return "inconclusive"
    value = primary_metric.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "inconclusive"
    if value > 0:
        return "improved"
    if value < 0:
        return "regressed"
    return "unchanged"


def _research_question_measurement_count(
    research_question_verdicts: Mapping[str, Any],
) -> int:
    count = 0
    for key in RESEARCH_QUESTION_KEYS:
        row = _mapping_or_empty(research_question_verdicts.get(key))
        measurement_count = _int_or_none(row.get("measurement_count"))
        if measurement_count is not None:
            count += measurement_count
    return count


def _dashboard_record(
    bundle: Mapping[str, Any],
    *,
    dashboard_bundle_path: str | Path | None,
) -> dict[str, Any]:
    summary = _mapping_or_empty(bundle.get("summary"))
    return {
        "path": str(dashboard_bundle_path) if dashboard_bundle_path is not None else None,
        "digest": dashboard_bundle_digest(bundle),
        "case_count": _int_or_none(summary.get("case_count")),
        "has_experiment_summary_review": isinstance(
            bundle.get("experiment_summary_review"),
            Mapping,
        ),
    }


def _real_package_readiness_record(report: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_real_experiment_readiness_report(report)
    readiness = _mapping_or_empty(report.get("readiness"))
    return {
        "declared_data_source_kind": _string_or_none(
            report.get("declared_data_source_kind")
        ),
        "failed_checks": _json_value(readiness.get("failed_checks", [])),
        "failed_count": _int_or_none(readiness.get("failed_count")),
        "manifest_digest": _string_or_none(report.get("manifest_digest")),
        "missing_groups": _json_value(readiness.get("missing_groups", [])),
        "ready": readiness.get("ready") is True,
        "report_digest": real_experiment_readiness_report_digest(report),
        "valid": validation["valid"] is True,
    }


def _validate_dashboard_record(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return (
        isinstance(value.get("digest"), str)
        and (
            value.get("path") is None
            or isinstance(value.get("path"), str)
        )
        and (
            value.get("case_count") is None
            or (
                isinstance(value.get("case_count"), int)
                and not isinstance(value.get("case_count"), bool)
            )
        )
        and isinstance(value.get("has_experiment_summary_review"), bool)
    )


def _dashboard_path(value: object) -> Path | None:
    if not isinstance(value, Mapping):
        return None
    path = value.get("path")
    if path is None:
        return None
    if not isinstance(path, str) or path == "":
        raise SpatialQAError("Experiment record dashboard path must be a string")
    return Path(path)


def _verdict_counts(research_question_verdicts: Mapping[str, Any]) -> dict[str, int]:
    counts = {key: 0 for key in VERDICT_KEYS}
    for key in RESEARCH_QUESTION_KEYS:
        verdict_row = _mapping_or_empty(research_question_verdicts.get(key))
        verdict = _string_or_none(verdict_row.get("verdict"))
        if verdict not in counts:
            verdict = "inconclusive"
        counts[verdict] += 1
    return counts


def _required_record_path(record: Mapping[str, Any], key: str) -> Path:
    value = record.get(key)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Experiment record missing required path: {key}")
    return Path(value)


def _optional_record_path(record: Mapping[str, Any], key: str) -> Path | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Experiment record path must be a string: {key}")
    return Path(value)


def _mapping_sequence(value: object) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return ()
    return tuple(cast(Mapping[str, Any], item) for item in value if isinstance(item, Mapping))


def _string_sequence(value: object) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _mapping_or_empty(value: object) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else {}


def _json_mapping(value: object) -> dict[str, Any]:
    return cast(dict[str, Any], _json_value(value)) if isinstance(value, Mapping) else {}


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_json_value(item) for item in value]
    return value


def _equality_check(name: str, saved: Any, current: Any) -> dict[str, Any]:
    check: dict[str, Any] = {
        "name": name,
        "passed": saved == current,
        "expected": saved,
        "actual": current,
    }
    if saved != current:
        check["differences"] = _differences(saved, current)
    return check


def _differences(saved: Any, current: Any, path: str = "") -> list[dict[str, Any]]:
    if saved == current:
        return []
    if isinstance(saved, Mapping) and isinstance(current, Mapping):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(saved) | set(current), key=str):
            child_path = f"{path}.{key}" if path else str(key)
            differences.extend(_differences(saved.get(key), current.get(key), child_path))
        return differences
    return [{"path": path or "value", "expected": saved, "actual": current}]
