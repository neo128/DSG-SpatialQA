from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.benchmark import QACase, qa_dataset_digest
from dsg_spatialqa_lab.memory import VALID_RELATIONS
from dsg_spatialqa_lab.schema import SpatialQAError


QA_QUALITY_REPORT_SCHEMA_VERSION = "dsg-spatialqa-lab.qa-quality-report.v1"
DEFAULT_MIN_QUESTION_TYPE_COUNT = 2
DEFAULT_MAX_IN_ROOM_RELATION_RATE = 0.4
TEMPORAL_QUESTION_TYPES = frozenset(
    {
        "agent_history",
        "agent_timeline",
        "object_history",
        "object_timeline",
        "recent_events",
        "relation_timeline",
        "reobserve_targets",
        "scene_delta",
        "state_change",
        "temporal_last_seen",
    }
)
SITUATED_QUESTION_TYPES = frozenset(
    {
        "egocentric_view",
        "relative_direction",
        "relative_relation",
    }
)
QA_V2_QUALITY_FIELDS = (
    "question_text",
    "split",
    "situation",
    "answer_options",
    "observability",
    "anti_shortcut",
)
VLM_REQUEST_BUNDLE_BANNED_FIELDS = (
    "gold_answer",
    "gold_evidence",
    "required_edges",
    "required_nodes",
    "visible_object_ids",
    "visible_object_labels",
)


def qa_quality_report(
    cases: Sequence[QACase],
    *,
    observability_report: Mapping[str, Any] | None = None,
    qa_path: str | Path | None = None,
    observability_report_path: str | Path | None = None,
    min_question_type_count: int = DEFAULT_MIN_QUESTION_TYPE_COUNT,
    max_in_room_relation_rate: float = DEFAULT_MAX_IN_ROOM_RELATION_RATE,
) -> dict[str, Any]:
    if min_question_type_count <= 0:
        raise SpatialQAError("min_question_type_count must be positive")
    if max_in_room_relation_rate < 0.0:
        raise SpatialQAError("max_in_room_relation_rate must be non-negative")
    obs_splits = _observability_splits(observability_report)
    obs_statuses = _observability_statuses(observability_report)
    case_rows = [
        _case_quality_row(case, obs_splits=obs_splits, obs_statuses=obs_statuses)
        for case in cases
    ]
    summary = _quality_summary(cases, case_rows)
    quality_gates = _quality_gates(
        summary,
        min_question_type_count=min_question_type_count,
        max_in_room_relation_rate=max_in_room_relation_rate,
    )
    report: dict[str, Any] = {
        "schema_version": QA_QUALITY_REPORT_SCHEMA_VERSION,
        "qa_path": str(qa_path) if qa_path is not None else None,
        "observability_report_path": (
            str(observability_report_path)
            if observability_report_path is not None
            else None
        ),
        "qa_digest": qa_dataset_digest(cases),
        "thresholds": {
            "max_in_room_relation_rate": max_in_room_relation_rate,
            "min_question_type_count": min_question_type_count,
        },
        "summary": summary,
        "quality_gates": quality_gates,
        "cases": case_rows,
    }
    report["report_digest"] = qa_quality_report_digest(report)
    return report


def qa_quality_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def qa_quality_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_qa_quality_report(report: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(qa_quality_report_json(report), encoding="utf-8")
    return output_path


def load_qa_quality_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("QA quality report JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_qa_quality_report(report: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = report.get("schema_version")
    report_digest = report.get("report_digest")
    expected_digest = qa_quality_report_digest(report)
    summary = report.get("summary")
    quality_gates = report.get("quality_gates")
    cases = report.get("cases")
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == QA_QUALITY_REPORT_SCHEMA_VERSION,
            "expected": QA_QUALITY_REPORT_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "report_digest",
            "passed": report_digest == expected_digest,
            "expected": expected_digest,
            "actual": report_digest,
        },
        {
            "name": "summary_shape",
            "passed": isinstance(summary, Mapping)
            and isinstance(summary.get("case_count"), int),
        },
        {
            "name": "quality_gates_shape",
            "passed": isinstance(quality_gates, Mapping)
            and all(isinstance(value, Mapping) for value in quality_gates.values()),
        },
        {
            "name": "cases_shape",
            "passed": isinstance(cases, Sequence) and not isinstance(cases, str),
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "report_digest": report_digest,
        "checks": checks,
    }


def audit_vlm_request_bundle_for_gold_leakage(
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    leaks: list[dict[str, Any]] = []
    _collect_banned_field_leaks(bundle, path=(), leaks=leaks)
    leaked_fields = [
        field
        for field in VLM_REQUEST_BUNDLE_BANNED_FIELDS
        if any(leak["field"] == field for leak in leaks)
    ]
    return {
        "leak_free": not leaks,
        "leak_count": len(leaks),
        "leaked_fields": leaked_fields,
        "leaks": leaks,
    }


def _case_quality_row(
    case: QACase,
    *,
    obs_splits: Mapping[str, set[str]],
    obs_statuses: Mapping[str, str],
) -> dict[str, Any]:
    relations = _case_relations(case)
    risk = _language_prior_risk(case, relations)
    recommended_splits = _recommended_splits(case, obs_splits)
    return {
        "case_id": case.id,
        "question_type": case.question_type,
        "target_label": _target_label(case),
        "required_edge_relations": relations,
        "observability_status": obs_statuses.get(case.id),
        "recommended_splits": recommended_splits,
        "language_prior_risk": risk,
        "missing_v2_fields": _missing_v2_fields(case),
        "has_duplicate_target_label": False,
    }


def _quality_summary(
    cases: Sequence[QACase],
    case_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    relation_counts = _sorted_counts(
        relation for row in case_rows for relation in _string_sequence(row, "required_edge_relations")
    )
    split_counts = _split_counts(case_rows, case_count=len(cases))
    question_type_counts = _sorted_counts(case.question_type for case in cases)
    return {
        "case_count": len(cases),
        "question_type_count": len(question_type_counts),
        "question_type_counts": question_type_counts,
        "relation_counts": relation_counts,
        "split_counts": split_counts,
        "language_prior_risk_counts": _sorted_counts(
            str(row["language_prior_risk"]) for row in case_rows
        ),
        "schema_gap_counts": {
            field: sum(1 for row in case_rows if field in _string_sequence(row, "missing_v2_fields"))
            for field in QA_V2_QUALITY_FIELDS
        },
    }


def _quality_gates(
    summary: Mapping[str, Any],
    *,
    min_question_type_count: int,
    max_in_room_relation_rate: float,
) -> dict[str, Any]:
    relation_counts = summary.get("relation_counts")
    relation_counts = relation_counts if isinstance(relation_counts, Mapping) else {}
    relation_total = sum(int(value) for value in relation_counts.values())
    in_room_count = int(relation_counts.get("IN_ROOM", 0))
    in_room_rate = round(in_room_count / relation_total, 6) if relation_total else 0.0
    question_type_count = int(summary.get("question_type_count", 0))
    return {
        "question_type_coverage": {
            "passed": question_type_count >= min_question_type_count,
            "expected": f">= {min_question_type_count}",
            "actual": question_type_count,
        },
        "relation_balance": {
            "passed": in_room_rate <= max_in_room_relation_rate,
            "expected": f"IN_ROOM relation rate <= {max_in_room_relation_rate}",
            "actual": in_room_rate,
        },
    }


def _observability_splits(
    observability_report: Mapping[str, Any] | None,
) -> dict[str, set[str]]:
    if observability_report is None:
        return {}
    splits = observability_report.get("splits")
    if not isinstance(splits, Mapping):
        return {}
    result: dict[str, set[str]] = {}
    for split_name, values in splits.items():
        if isinstance(values, Sequence) and not isinstance(values, str):
            result[str(split_name)] = {str(value) for value in values}
    return result


def _observability_statuses(
    observability_report: Mapping[str, Any] | None,
) -> dict[str, str]:
    if observability_report is None:
        return {}
    rows = observability_report.get("cases")
    if not isinstance(rows, Sequence) or isinstance(rows, str):
        return {}
    statuses: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        case_id = row.get("case_id")
        status = row.get("observability_status")
        if isinstance(case_id, str) and isinstance(status, str):
            statuses[case_id] = status
    return statuses


def _recommended_splits(
    case: QACase,
    obs_splits: Mapping[str, set[str]],
) -> list[str]:
    splits = ["full_oracle"]
    split_map = (
        ("evidence_observable", "observation_aware"),
        ("target_observable", "target_observable"),
        ("missing_evidence", "missing_evidence"),
        ("target_observable_relation_missing", "target_observable_relation_missing"),
        ("target_missing", "target_missing"),
    )
    for source_split, recommended_split in split_map:
        if case.id in obs_splits.get(source_split, set()):
            splits.append(recommended_split)
    if case.question_type in SITUATED_QUESTION_TYPES or case.reference_frame is not None:
        splits.append("situated")
    if case.question_type in TEMPORAL_QUESTION_TYPES:
        splits.append("temporal")
    return splits


def _split_counts(
    case_rows: Sequence[Mapping[str, Any]],
    *,
    case_count: int,
) -> dict[str, int]:
    split_counts = {
        "full_oracle": case_count,
        "observation_aware": 0,
        "target_observable": 0,
        "missing_evidence": 0,
        "target_missing": 0,
        "target_observable_relation_missing": 0,
        "situated": 0,
        "temporal": 0,
        "anti_shortcut_candidate": 0,
    }
    for row in case_rows:
        for split in _string_sequence(row, "recommended_splits"):
            if split != "full_oracle":
                split_counts[split] += 1
        if row.get("language_prior_risk") != "high":
            split_counts["anti_shortcut_candidate"] += 1
    return split_counts


def _case_relations(case: QACase) -> list[str]:
    relations: list[str] = []
    current_location = case.answer.get("current_location")
    if isinstance(current_location, Mapping):
        relation = current_location.get("relation")
        if isinstance(relation, str) and relation in VALID_RELATIONS:
            relations.append(relation)
    for edge_id in case.required_edges:
        relation = _edge_relation(edge_id)
        if relation is not None:
            relations.append(relation)
    return sorted(set(relations))


def _edge_relation(edge_id: str) -> str | None:
    matches = [
        relation
        for relation in sorted(VALID_RELATIONS, key=len, reverse=True)
        if f"-{relation}-" in edge_id
    ]
    return matches[0] if matches else None


def _language_prior_risk(case: QACase, relations: Sequence[str]) -> str:
    if "IN_ROOM" in relations:
        return "high"
    if case.question_type == "object_location":
        return "medium"
    return "low"


def _missing_v2_fields(case: QACase) -> list[str]:
    missing: list[str] = []
    for field in QA_V2_QUALITY_FIELDS:
        if not hasattr(case, field):
            missing.append(field)
    return missing


def _target_label(case: QACase) -> str | None:
    object_id = case.question.get("object_id")
    if not isinstance(object_id, str):
        object_id = case.answer.get("object_id")
    if not isinstance(object_id, str):
        return None
    return object_id.split("_")[0]


def _string_sequence(row: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = row.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str):
        return ()
    return tuple(str(item) for item in value)


def _sorted_counts(values: Sequence[str] | Any) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _collect_banned_field_leaks(
    value: Any,
    *,
    path: tuple[str, ...],
    leaks: list[dict[str, Any]],
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_str = str(key)
            child_path = (*path, key_str)
            if key_str in VLM_REQUEST_BUNDLE_BANNED_FIELDS:
                leaks.append(
                    {
                        "field": key_str,
                        "path": ".".join(child_path),
                    }
                )
            _collect_banned_field_leaks(child, path=child_path, leaks=leaks)
        return
    if isinstance(value, Sequence) and not isinstance(value, str):
        for index, child in enumerate(value):
            _collect_banned_field_leaks(
                child,
                path=(*path, str(index)),
                leaks=leaks,
            )
