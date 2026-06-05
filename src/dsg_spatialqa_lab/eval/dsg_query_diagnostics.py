from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.benchmark.qa_generator import QACase, qa_dataset_digest
from dsg_spatialqa_lab.graph_tool import GraphTool
from dsg_spatialqa_lab.memory import DynamicSceneGraph
from dsg_spatialqa_lab.qa import SpatialQAEngine
from dsg_spatialqa_lab.scene_io import graph_json_digest


OBJECT_LOCATION_QUERY_DIAGNOSTIC_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.object-location-query-diagnostic-report.v1"
)


def object_location_query_diagnostic_report(
    graph: DynamicSceneGraph,
    cases: Sequence[QACase],
    *,
    semantic_eval_report: Mapping[str, Any] | None = None,
    graph_path: str | Path | None = None,
    qa_path: str | Path | None = None,
    semantic_eval_path: str | Path | None = None,
) -> dict[str, Any]:
    semantic_cases = _semantic_cases_by_id(semantic_eval_report)
    qa_engine = SpatialQAEngine(GraphTool(graph))
    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    semantic_match_status_counts: Counter[str] = Counter()
    semantic_mismatch_status_counts: Counter[str] = Counter()
    support_candidate_histogram: Counter[str] = Counter()
    query_error_count = 0
    room_fallback_count = 0
    semantic_mismatch_count = 0

    for case in cases:
        if case.question_type != "object_location":
            continue
        question = dict(case.question)
        question.setdefault("scene_id", case.scene_id)
        question.setdefault("step", case.step)
        question["include_diagnostics"] = True
        response = qa_engine.answer(question)
        semantic_case = semantic_cases.get(case.id, {})
        semantic_match = _optional_bool(semantic_case.get("semantic_match"))
        object_id = _object_id_from_question(case.question)
        answer = response.answer
        current_location = _optional_mapping(answer.get("current_location"))
        diagnostics = _optional_mapping(answer.get("query_diagnostics"))
        if response.error is not None:
            status = "query_error"
            missing_evidence = ["target_object"] if response.error.startswith("Object not found:") else []
            query_error_count += 1
            support_candidate_count = 0
            room_fallback_applied = False
            support_fallback_applied = False
        else:
            status = _diagnostic_status(diagnostics)
            diagnostic_values = diagnostics if diagnostics is not None else {}
            missing_evidence = _string_list(diagnostic_values.get("missing_evidence"))
            support_candidate_count = _int_value(
                diagnostic_values.get("support_candidate_count")
            )
            room_fallback_applied = _bool_value(
                diagnostic_values.get("room_fallback_applied")
            )
            support_fallback_applied = _bool_value(
                diagnostic_values.get("support_fallback_applied")
            )
        if room_fallback_applied:
            room_fallback_count += 1
        status_counts[status] += 1
        if semantic_match is True:
            semantic_match_status_counts[status] += 1
        elif semantic_match is False:
            semantic_mismatch_count += 1
            semantic_mismatch_status_counts[status] += 1
        if response.error is None:
            support_candidate_histogram[str(support_candidate_count)] += 1
        rows.append(
            {
                "case_id": case.id,
                "current_location": current_location,
                "failure_reason": _optional_str(semantic_case.get("failure_reason")),
                "location_evidence_status": status,
                "missing_evidence": missing_evidence,
                "object_id": object_id,
                "prediction_error": response.error,
                "question_type": case.question_type,
                "room_fallback_applied": room_fallback_applied,
                "semantic_match": semantic_match,
                "support_candidate_count": support_candidate_count,
                "support_fallback_applied": support_fallback_applied,
            }
        )

    report: dict[str, Any] = {
        "schema_version": OBJECT_LOCATION_QUERY_DIAGNOSTIC_REPORT_SCHEMA_VERSION,
        "artifacts": {
            "graph_path": str(graph_path) if graph_path is not None else None,
            "qa_path": str(qa_path) if qa_path is not None else None,
            "semantic_eval_path": (
                str(semantic_eval_path) if semantic_eval_path is not None else None
            ),
        },
        "graph_digest": graph_json_digest(graph),
        "qa_digest": qa_dataset_digest(cases),
        "summary": {
            "object_location_case_count": len(rows),
            "query_error_count": query_error_count,
            "room_fallback_count": room_fallback_count,
            "semantic_mismatch_count": semantic_mismatch_count,
            "semantic_mismatch_status_counts": dict(
                sorted(semantic_mismatch_status_counts.items())
            ),
            "semantic_match_status_counts": dict(
                sorted(semantic_match_status_counts.items())
            ),
            "status_counts": dict(sorted(status_counts.items())),
            "support_candidate_count_histogram": dict(
                sorted(support_candidate_histogram.items())
            ),
        },
        "cases": rows,
    }
    report["report_digest"] = object_location_query_diagnostic_report_digest(report)
    return report


def object_location_query_diagnostic_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def object_location_query_diagnostic_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_object_location_query_diagnostic_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        object_location_query_diagnostic_report_json(report),
        encoding="utf-8",
    )
    return output_path


def load_object_location_query_diagnostic_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Object-location query diagnostic report must be an object")
    return cast(dict[str, Any], payload)


def _semantic_cases_by_id(
    semantic_eval_report: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    if semantic_eval_report is None:
        return {}
    cases = semantic_eval_report.get("cases")
    if not isinstance(cases, list):
        return {}
    rows: dict[str, Mapping[str, Any]] = {}
    for item in cases:
        if not isinstance(item, Mapping):
            continue
        case_id = item.get("case_id")
        if isinstance(case_id, str):
            rows[case_id] = item
    return rows


def _object_id_from_question(question: Mapping[str, Any]) -> str | None:
    object_id = question.get("object_id")
    return object_id if isinstance(object_id, str) else None


def _optional_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _diagnostic_status(diagnostics: Mapping[str, Any] | None) -> str:
    if diagnostics is None:
        return "missing_query_diagnostics"
    value = diagnostics.get("location_evidence_status")
    return value if isinstance(value, str) else "missing_query_diagnostics"


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _bool_value(value: Any) -> bool:
    return value if isinstance(value, bool) else False


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None
