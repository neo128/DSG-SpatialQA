from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.benchmark import (
    QACase,
    load_qa_dataset,
    qa_dataset_digest,
)
from dsg_spatialqa_lab.schema import SpatialQAError


QA_PREDICTION_SCHEMA_VERSION = "dsg-spatialqa-lab.qa-prediction.v1"
QA_EVAL_REPORT_SCHEMA_VERSION = "dsg-spatialqa-lab.qa-eval-report.v1"
QA_EVAL_DELTA_REPORT_SCHEMA_VERSION = "dsg-spatialqa-lab.qa-eval-delta-report.v1"
QA_RESEARCH_AXIS_NAMES = (
    "dynamic_memory",
    "graph_tool_query",
    "spatial_qa",
)
QA_DYNAMIC_MEMORY_TAGS = frozenset(
    (
        "dynamic",
        "memory",
        "move",
        "occlusion",
        "reobserve",
        "temporal",
    )
)
QA_GRAPH_TOOL_QUERY_TAGS = frozenset(("graph_query", "nearest", "query", "retrieval"))
QA_GRAPH_TOOL_QUERY_TYPES = frozenset(
    (
        "nearest_object",
        "object_location",
        "object_room",
        "object_status",
        "relative_relation",
        "scene_delta",
        "scene_snapshot",
    )
)


@dataclass
class QAPrediction:
    id: str
    answer: dict[str, Any] = field(default_factory=dict)
    evidence_nodes: tuple[str, ...] = field(default_factory=tuple)
    evidence_edges: tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 0.0
    error: str | None = None


def qa_prediction_to_dict(prediction: QAPrediction) -> dict[str, Any]:
    return {
        "schema_version": QA_PREDICTION_SCHEMA_VERSION,
        "id": prediction.id,
        "answer": _json_mapping(prediction.answer),
        "evidence_nodes": list(prediction.evidence_nodes),
        "evidence_edges": list(prediction.evidence_edges),
        "confidence": prediction.confidence,
        "error": prediction.error,
    }


def qa_prediction_from_dict(payload: Mapping[str, Any]) -> QAPrediction:
    schema_version = _required_str(payload, "schema_version")
    if schema_version != QA_PREDICTION_SCHEMA_VERSION:
        raise SpatialQAError(f"Unsupported QA prediction schema version: {schema_version}")
    return QAPrediction(
        id=_required_str(payload, "id"),
        answer=_required_mapping(payload, "answer"),
        evidence_nodes=_string_tuple(payload, "evidence_nodes"),
        evidence_edges=_string_tuple(payload, "evidence_edges"),
        confidence=_required_float(payload, "confidence"),
        error=_optional_str(payload, "error"),
    )


def qa_predictions_jsonl(predictions: Sequence[QAPrediction]) -> str:
    return "".join(
        json.dumps(qa_prediction_to_dict(prediction), separators=(",", ":"), sort_keys=True)
        + "\n"
        for prediction in predictions
    )


def qa_predictions_digest(predictions: Sequence[QAPrediction]) -> str:
    return hashlib.sha256(qa_predictions_jsonl(predictions).encode("utf-8")).hexdigest()


def save_qa_predictions(predictions: Sequence[QAPrediction], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(qa_predictions_jsonl(predictions), encoding="utf-8")
    return output_path


def load_qa_predictions(path: str | Path) -> list[QAPrediction]:
    predictions: list[QAPrediction] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if line == "":
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise SpatialQAError(f"QA prediction line {line_number} must be an object")
        predictions.append(qa_prediction_from_dict(cast(Mapping[str, Any], payload)))
    return predictions


def evaluate_qa_predictions(
    gold_cases: Sequence[QACase],
    predictions: Sequence[QAPrediction],
) -> dict[str, Any]:
    return qa_eval_report(gold_cases, predictions)


def qa_eval_report(
    gold_cases: Sequence[QACase],
    predictions: Sequence[QAPrediction],
    *,
    gold_path: str | Path | None = None,
    prediction_path: str | Path | None = None,
) -> dict[str, Any]:
    prediction_by_id = _prediction_mapping(predictions)
    case_results = [_case_result(case, prediction_by_id.get(case.id)) for case in gold_cases]
    summary = _summary(case_results, prediction_count=len(predictions))
    metrics = _metrics(case_results)
    report: dict[str, Any] = {
        "schema_version": QA_EVAL_REPORT_SCHEMA_VERSION,
        "gold_path": str(gold_path) if gold_path is not None else None,
        "prediction_path": str(prediction_path) if prediction_path is not None else None,
        "gold_digest": qa_dataset_digest(gold_cases),
        "prediction_digest": qa_predictions_digest(predictions),
        "summary": summary,
        "metrics": metrics,
        "breakdown": {
            "by_episode_id": _breakdown_by_episode_id(case_results),
            "by_question_type": _breakdown_by_question_type(case_results),
            "by_reference_frame": _breakdown_by_reference_frame(case_results),
            "by_research_axis": _breakdown_by_research_axis(case_results),
            "by_scene_id": _breakdown_by_scene_id(case_results),
            "by_tag": _breakdown_by_tag(case_results),
        },
        "cases": case_results,
    }
    report["report_digest"] = qa_eval_report_digest(report)
    return report


def qa_eval_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def qa_eval_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_qa_eval_report(report: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(qa_eval_report_json(report), encoding="utf-8")
    return output_path


def load_qa_eval_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("QA eval report JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_qa_eval_report(report: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = report.get("schema_version")
    report_digest = _string_or_none(report.get("report_digest"))
    expected_report_digest = qa_eval_report_digest(report)
    cases = report.get("cases")
    summary = report.get("summary")
    metrics = report.get("metrics")
    breakdown = report.get("breakdown")
    case_results = cast(Sequence[Mapping[str, Any]], cases) if isinstance(cases, Sequence) else ()
    expected_case_count = len(case_results) if not isinstance(cases, str) else None
    exact_match_count = _case_bool_count(case_results, "exact_match")
    expected_episode_breakdown = _breakdown_by_episode_id(case_results)
    expected_research_axis_breakdown = _breakdown_by_research_axis(case_results)
    expected_scene_breakdown = _breakdown_by_scene_id(case_results)
    actual_episode_breakdown = (
        breakdown.get("by_episode_id") if isinstance(breakdown, Mapping) else None
    )
    actual_research_axis_breakdown = (
        breakdown.get("by_research_axis") if isinstance(breakdown, Mapping) else None
    )
    actual_scene_breakdown = (
        breakdown.get("by_scene_id") if isinstance(breakdown, Mapping) else None
    )
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == QA_EVAL_REPORT_SCHEMA_VERSION,
            "expected": QA_EVAL_REPORT_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "report_digest",
            "passed": report_digest == expected_report_digest,
            "expected": expected_report_digest,
            "actual": report_digest,
        },
        {
            "name": "case_count",
            "passed": _summary_value(summary, "case_count") == expected_case_count,
            "expected": expected_case_count,
            "actual": _summary_value(summary, "case_count"),
        },
        {
            "name": "exact_match_count",
            "passed": _summary_value(summary, "exact_match_count") == exact_match_count,
            "expected": exact_match_count,
            "actual": _summary_value(summary, "exact_match_count"),
        },
        {
            "name": "exact_match_metric",
            "passed": _metric_value(metrics, "exact_match", "count") == exact_match_count,
            "expected": exact_match_count,
            "actual": _metric_value(metrics, "exact_match", "count"),
        },
        {
            "name": "episode_breakdown",
            "passed": actual_episode_breakdown == expected_episode_breakdown,
            "expected": expected_episode_breakdown,
            "actual": actual_episode_breakdown,
        },
        {
            "name": "research_axis_breakdown",
            "passed": actual_research_axis_breakdown == expected_research_axis_breakdown,
            "expected": expected_research_axis_breakdown,
            "actual": actual_research_axis_breakdown,
        },
        {
            "name": "scene_breakdown",
            "passed": actual_scene_breakdown == expected_scene_breakdown,
            "expected": expected_scene_breakdown,
            "actual": actual_scene_breakdown,
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "report_digest": report_digest,
        "checks": checks,
    }


def compare_qa_eval_report(report: Mapping[str, Any]) -> dict[str, Any]:
    gold_path = _required_report_path(report, "gold_path")
    prediction_path = _required_report_path(report, "prediction_path")
    current_report = qa_eval_report(
        load_qa_dataset(gold_path),
        load_qa_predictions(prediction_path),
        gold_path=gold_path,
        prediction_path=prediction_path,
    )
    validation = validate_qa_eval_report(report)
    saved_digest = _string_or_none(report.get("report_digest"))
    current_digest = _string_or_none(current_report.get("report_digest"))
    checks = [
        {
            "name": "report_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        {
            "name": "report_digest_matches_current",
            "passed": saved_digest == current_digest,
            "expected": saved_digest,
            "actual": current_digest,
        },
        _equality_check("summary_matches_current", report.get("summary"), current_report["summary"]),
        _equality_check("metrics_match_current", report.get("metrics"), current_report["metrics"]),
        _equality_check("breakdown_matches_current", report.get("breakdown"), current_report["breakdown"]),
        _equality_check("cases_match_current", report.get("cases"), current_report["cases"]),
    ]
    return {
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def qa_eval_delta_report(
    candidate_report: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
    *,
    candidate_name: str = "candidate",
    baseline_name: str = "baseline",
    candidate_report_path: str | Path | None = None,
    baseline_report_path: str | Path | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": QA_EVAL_DELTA_REPORT_SCHEMA_VERSION,
        "candidate_name": candidate_name,
        "baseline_name": baseline_name,
        "candidate_report_path": (
            str(candidate_report_path) if candidate_report_path is not None else None
        ),
        "baseline_report_path": (
            str(baseline_report_path) if baseline_report_path is not None else None
        ),
        "candidate_report_digest": _string_or_none(candidate_report.get("report_digest")),
        "baseline_report_digest": _string_or_none(baseline_report.get("report_digest")),
        "summary_delta": _summary_delta(
            candidate_report.get("summary"),
            baseline_report.get("summary"),
        ),
        "metrics_delta": _metrics_delta(
            candidate_report.get("metrics"),
            baseline_report.get("metrics"),
        ),
        "breakdown_delta": {
            "by_episode_id": _breakdown_group_delta(
                candidate_report.get("breakdown"),
                baseline_report.get("breakdown"),
                "by_episode_id",
            ),
            "by_question_type": _breakdown_group_delta(
                candidate_report.get("breakdown"),
                baseline_report.get("breakdown"),
                "by_question_type",
            ),
            "by_reference_frame": _breakdown_group_delta(
                candidate_report.get("breakdown"),
                baseline_report.get("breakdown"),
                "by_reference_frame",
            ),
            "by_research_axis": _research_axis_delta(
                candidate_report.get("breakdown"),
                baseline_report.get("breakdown"),
            ),
            "by_scene_id": _breakdown_group_delta(
                candidate_report.get("breakdown"),
                baseline_report.get("breakdown"),
                "by_scene_id",
            ),
            "by_tag": _breakdown_group_delta(
                candidate_report.get("breakdown"),
                baseline_report.get("breakdown"),
                "by_tag",
            ),
        },
    }
    report["report_digest"] = qa_eval_delta_report_digest(report)
    return report


def qa_eval_delta_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def qa_eval_delta_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_qa_eval_delta_report(report: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(qa_eval_delta_report_json(report), encoding="utf-8")
    return output_path


def load_qa_eval_delta_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("QA eval delta report JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_qa_eval_delta_report(report: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = report.get("schema_version")
    report_digest = _string_or_none(report.get("report_digest"))
    expected_report_digest = qa_eval_delta_report_digest(report)
    summary_delta = report.get("summary_delta")
    metrics_delta = report.get("metrics_delta")
    breakdown_delta = report.get("breakdown_delta")
    expected_summary_delta = _summary_delta_from_entry(summary_delta)
    expected_metrics_delta = _metrics_delta_from_entry(metrics_delta)
    expected_breakdown_delta = _breakdown_delta_from_entry(breakdown_delta)
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == QA_EVAL_DELTA_REPORT_SCHEMA_VERSION,
            "expected": QA_EVAL_DELTA_REPORT_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "report_digest",
            "passed": report_digest == expected_report_digest,
            "expected": expected_report_digest,
            "actual": report_digest,
        },
        {
            "name": "summary_delta",
            "passed": summary_delta == expected_summary_delta,
            "expected": expected_summary_delta,
            "actual": summary_delta,
        },
        {
            "name": "metrics_delta",
            "passed": metrics_delta == expected_metrics_delta,
            "expected": expected_metrics_delta,
            "actual": metrics_delta,
        },
        {
            "name": "breakdown_delta",
            "passed": breakdown_delta == expected_breakdown_delta,
            "expected": expected_breakdown_delta,
            "actual": breakdown_delta,
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "report_digest": report_digest,
        "checks": checks,
    }


def compare_qa_eval_delta_report(report: Mapping[str, Any]) -> dict[str, Any]:
    candidate_report_path = _required_delta_report_path(report, "candidate_report_path")
    baseline_report_path = _required_delta_report_path(report, "baseline_report_path")
    current_report = qa_eval_delta_report(
        load_qa_eval_report(candidate_report_path),
        load_qa_eval_report(baseline_report_path),
        candidate_name=_delta_report_name(report, "candidate_name"),
        baseline_name=_delta_report_name(report, "baseline_name"),
        candidate_report_path=candidate_report_path,
        baseline_report_path=baseline_report_path,
    )
    validation = validate_qa_eval_delta_report(report)
    saved_digest = _string_or_none(report.get("report_digest"))
    current_digest = _string_or_none(current_report.get("report_digest"))
    checks = [
        {
            "name": "report_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        {
            "name": "report_digest_matches_current",
            "passed": saved_digest == current_digest,
            "expected": saved_digest,
            "actual": current_digest,
        },
        _equality_check(
            "summary_delta_matches_current",
            report.get("summary_delta"),
            current_report["summary_delta"],
        ),
        _equality_check(
            "metrics_delta_matches_current",
            report.get("metrics_delta"),
            current_report["metrics_delta"],
        ),
        _equality_check(
            "breakdown_delta_matches_current",
            report.get("breakdown_delta"),
            current_report["breakdown_delta"],
        ),
    ]
    return {
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def _summary_delta(candidate_summary: object, baseline_summary: object) -> dict[str, Any]:
    candidate_case_count = _int_mapping_value(candidate_summary, "case_count")
    baseline_case_count = _int_mapping_value(baseline_summary, "case_count")
    candidate_exact_match_count = _int_mapping_value(
        candidate_summary,
        "exact_match_count",
    )
    baseline_exact_match_count = _int_mapping_value(
        baseline_summary,
        "exact_match_count",
    )
    candidate_exact_match_rate = _float_mapping_value(
        candidate_summary,
        "exact_match_rate",
    )
    baseline_exact_match_rate = _float_mapping_value(
        baseline_summary,
        "exact_match_rate",
    )
    return {
        "baseline_case_count": baseline_case_count,
        "baseline_exact_match_count": baseline_exact_match_count,
        "baseline_exact_match_rate": baseline_exact_match_rate,
        "candidate_case_count": candidate_case_count,
        "candidate_exact_match_count": candidate_exact_match_count,
        "candidate_exact_match_rate": candidate_exact_match_rate,
        "case_count_delta": _int_delta(candidate_case_count, baseline_case_count),
        "case_count_match": candidate_case_count == baseline_case_count,
        "exact_match_count_delta": _int_delta(
            candidate_exact_match_count,
            baseline_exact_match_count,
        ),
        "exact_match_rate_delta": _float_delta(
            candidate_exact_match_rate,
            baseline_exact_match_rate,
        ),
    }


def _summary_delta_from_entry(entry: object) -> dict[str, Any] | None:
    if not isinstance(entry, Mapping):
        return None
    return {
        "baseline_case_count": _int_mapping_value(entry, "baseline_case_count"),
        "baseline_exact_match_count": _int_mapping_value(
            entry,
            "baseline_exact_match_count",
        ),
        "baseline_exact_match_rate": _float_mapping_value(
            entry,
            "baseline_exact_match_rate",
        ),
        "candidate_case_count": _int_mapping_value(entry, "candidate_case_count"),
        "candidate_exact_match_count": _int_mapping_value(
            entry,
            "candidate_exact_match_count",
        ),
        "candidate_exact_match_rate": _float_mapping_value(
            entry,
            "candidate_exact_match_rate",
        ),
        "case_count_delta": _int_delta(
            _int_mapping_value(entry, "candidate_case_count"),
            _int_mapping_value(entry, "baseline_case_count"),
        ),
        "case_count_match": (
            _int_mapping_value(entry, "candidate_case_count")
            == _int_mapping_value(entry, "baseline_case_count")
        ),
        "exact_match_count_delta": _int_delta(
            _int_mapping_value(entry, "candidate_exact_match_count"),
            _int_mapping_value(entry, "baseline_exact_match_count"),
        ),
        "exact_match_rate_delta": _float_delta(
            _float_mapping_value(entry, "candidate_exact_match_rate"),
            _float_mapping_value(entry, "baseline_exact_match_rate"),
        ),
    }


def _metrics_delta(candidate_metrics: object, baseline_metrics: object) -> dict[str, Any]:
    return {
        "answer_graph_consistency": _count_rate_metric_delta(
            candidate_metrics,
            baseline_metrics,
            "answer_graph_consistency",
        ),
        "evidence_edge_recall": _average_metric_delta(
            candidate_metrics,
            baseline_metrics,
            "evidence_edge_recall",
        ),
        "evidence_node_recall": _average_metric_delta(
            candidate_metrics,
            baseline_metrics,
            "evidence_node_recall",
        ),
        "exact_match": _count_rate_metric_delta(
            candidate_metrics,
            baseline_metrics,
            "exact_match",
        ),
        "multiple_choice_accuracy": _count_rate_metric_delta(
            candidate_metrics,
            baseline_metrics,
            "multiple_choice_accuracy",
        ),
        "numeric_mae": _numeric_mae_metric_delta(candidate_metrics, baseline_metrics),
    }


def _metrics_delta_from_entry(entry: object) -> dict[str, Any] | None:
    if not isinstance(entry, Mapping):
        return None
    return {
        "answer_graph_consistency": _count_rate_metric_delta_from_entry(
            entry.get("answer_graph_consistency")
        ),
        "evidence_edge_recall": _average_metric_delta_from_entry(
            entry.get("evidence_edge_recall")
        ),
        "evidence_node_recall": _average_metric_delta_from_entry(
            entry.get("evidence_node_recall")
        ),
        "exact_match": _count_rate_metric_delta_from_entry(entry.get("exact_match")),
        "multiple_choice_accuracy": _count_rate_metric_delta_from_entry(
            entry.get("multiple_choice_accuracy")
        ),
        "numeric_mae": _numeric_mae_metric_delta_from_entry(entry.get("numeric_mae")),
    }


def _count_rate_metric_delta(
    candidate_metrics: object,
    baseline_metrics: object,
    metric_name: str,
) -> dict[str, Any]:
    candidate_metric = _metric_mapping(candidate_metrics, metric_name)
    baseline_metric = _metric_mapping(baseline_metrics, metric_name)
    candidate_count = _int_mapping_value(candidate_metric, "count")
    baseline_count = _int_mapping_value(baseline_metric, "count")
    candidate_rate = _float_mapping_value(candidate_metric, "rate")
    baseline_rate = _float_mapping_value(baseline_metric, "rate")
    return _count_rate_metric_delta_from_values(
        candidate_count=candidate_count,
        baseline_count=baseline_count,
        candidate_rate=candidate_rate,
        baseline_rate=baseline_rate,
    )


def _count_rate_metric_delta_from_entry(entry: object) -> dict[str, Any] | None:
    if not isinstance(entry, Mapping):
        return None
    return _count_rate_metric_delta_from_values(
        candidate_count=_int_mapping_value(entry, "candidate_count"),
        baseline_count=_int_mapping_value(entry, "baseline_count"),
        candidate_rate=_float_mapping_value(entry, "candidate_rate"),
        baseline_rate=_float_mapping_value(entry, "baseline_rate"),
    )


def _count_rate_metric_delta_from_values(
    *,
    candidate_count: int | None,
    baseline_count: int | None,
    candidate_rate: float | None,
    baseline_rate: float | None,
) -> dict[str, Any]:
    return {
        "baseline_count": baseline_count,
        "baseline_rate": baseline_rate,
        "candidate_count": candidate_count,
        "candidate_rate": candidate_rate,
        "count_delta": _int_delta(candidate_count, baseline_count),
        "rate_delta": _float_delta(candidate_rate, baseline_rate),
    }


def _average_metric_delta(
    candidate_metrics: object,
    baseline_metrics: object,
    metric_name: str,
) -> dict[str, Any]:
    candidate_average = _float_mapping_value(
        _metric_mapping(candidate_metrics, metric_name),
        "average",
    )
    baseline_average = _float_mapping_value(
        _metric_mapping(baseline_metrics, metric_name),
        "average",
    )
    return _average_metric_delta_from_values(
        candidate_average=candidate_average,
        baseline_average=baseline_average,
    )


def _average_metric_delta_from_entry(entry: object) -> dict[str, Any] | None:
    if not isinstance(entry, Mapping):
        return None
    return _average_metric_delta_from_values(
        candidate_average=_float_mapping_value(entry, "candidate_average"),
        baseline_average=_float_mapping_value(entry, "baseline_average"),
    )


def _average_metric_delta_from_values(
    *,
    candidate_average: float | None,
    baseline_average: float | None,
) -> dict[str, Any]:
    return {
        "average_delta": _float_delta(candidate_average, baseline_average),
        "baseline_average": baseline_average,
        "candidate_average": candidate_average,
    }


def _numeric_mae_metric_delta(
    candidate_metrics: object,
    baseline_metrics: object,
) -> dict[str, Any]:
    candidate_mae = _float_mapping_value(
        _metric_mapping(candidate_metrics, "numeric_mae"),
        "mean_absolute_error",
    )
    baseline_mae = _float_mapping_value(
        _metric_mapping(baseline_metrics, "numeric_mae"),
        "mean_absolute_error",
    )
    return _numeric_mae_metric_delta_from_values(
        candidate_mae=candidate_mae,
        baseline_mae=baseline_mae,
    )


def _numeric_mae_metric_delta_from_entry(entry: object) -> dict[str, Any] | None:
    if not isinstance(entry, Mapping):
        return None
    return _numeric_mae_metric_delta_from_values(
        candidate_mae=_float_mapping_value(entry, "candidate_mean_absolute_error"),
        baseline_mae=_float_mapping_value(entry, "baseline_mean_absolute_error"),
    )


def _numeric_mae_metric_delta_from_values(
    *,
    candidate_mae: float | None,
    baseline_mae: float | None,
) -> dict[str, Any]:
    return {
        "baseline_mean_absolute_error": baseline_mae,
        "candidate_mean_absolute_error": candidate_mae,
        "mean_absolute_error_delta": _float_delta(candidate_mae, baseline_mae),
    }


def _research_axis_delta(
    candidate_breakdown: object,
    baseline_breakdown: object,
) -> dict[str, Any]:
    candidate_axes = _research_axis_mapping(candidate_breakdown)
    baseline_axes = _research_axis_mapping(baseline_breakdown)
    axis_names = sorted(set(QA_RESEARCH_AXIS_NAMES) | set(candidate_axes) | set(baseline_axes))
    return {
        axis: _breakdown_delta(candidate_axes.get(axis), baseline_axes.get(axis))
        for axis in axis_names
    }


def _breakdown_group_delta(
    candidate_breakdown: object,
    baseline_breakdown: object,
    group_name: str,
) -> dict[str, Any]:
    candidate_group = _breakdown_group_mapping(candidate_breakdown, group_name)
    baseline_group = _breakdown_group_mapping(baseline_breakdown, group_name)
    group_keys = sorted(set(candidate_group) | set(baseline_group))
    return {
        str(key): _breakdown_delta(candidate_group.get(key), baseline_group.get(key))
        for key in group_keys
    }


def _breakdown_delta_from_entry(entry: object) -> dict[str, Any] | None:
    if not isinstance(entry, Mapping):
        return None
    return {
        "by_episode_id": _breakdown_group_delta_from_entry(
            entry.get("by_episode_id")
        ),
        "by_question_type": _breakdown_group_delta_from_entry(
            entry.get("by_question_type")
        ),
        "by_reference_frame": _breakdown_group_delta_from_entry(
            entry.get("by_reference_frame")
        ),
        "by_research_axis": _breakdown_group_delta_from_entry(
            entry.get("by_research_axis")
        ),
        "by_scene_id": _breakdown_group_delta_from_entry(entry.get("by_scene_id")),
        "by_tag": _breakdown_group_delta_from_entry(entry.get("by_tag")),
    }


def _breakdown_group_delta_from_entry(entry: object) -> dict[str, Any] | None:
    if not isinstance(entry, Mapping):
        return None
    return {
        str(key): _breakdown_delta_from_values(value)
        for key, value in sorted(entry.items(), key=lambda item: str(item[0]))
    }


def _breakdown_delta(candidate: object, baseline: object) -> dict[str, Any]:
    return _breakdown_delta_from_numbers(
        candidate_case_count=_int_mapping_value(candidate, "case_count"),
        baseline_case_count=_int_mapping_value(baseline, "case_count"),
        candidate_exact_match_count=_int_mapping_value(candidate, "exact_match_count"),
        baseline_exact_match_count=_int_mapping_value(baseline, "exact_match_count"),
        candidate_exact_match_rate=_float_mapping_value(candidate, "exact_match_rate"),
        baseline_exact_match_rate=_float_mapping_value(baseline, "exact_match_rate"),
        candidate_mean_edge_recall=_float_mapping_value(
            candidate,
            "mean_evidence_edge_recall",
        ),
        baseline_mean_edge_recall=_float_mapping_value(
            baseline,
            "mean_evidence_edge_recall",
        ),
        candidate_mean_node_recall=_float_mapping_value(
            candidate,
            "mean_evidence_node_recall",
        ),
        baseline_mean_node_recall=_float_mapping_value(
            baseline,
            "mean_evidence_node_recall",
        ),
    )


def _breakdown_delta_from_values(entry: object) -> dict[str, Any] | None:
    if not isinstance(entry, Mapping):
        return None
    return _breakdown_delta_from_numbers(
        candidate_case_count=_int_mapping_value(entry, "candidate_case_count"),
        baseline_case_count=_int_mapping_value(entry, "baseline_case_count"),
        candidate_exact_match_count=_int_mapping_value(
            entry,
            "candidate_exact_match_count",
        ),
        baseline_exact_match_count=_int_mapping_value(
            entry,
            "baseline_exact_match_count",
        ),
        candidate_exact_match_rate=_float_mapping_value(
            entry,
            "candidate_exact_match_rate",
        ),
        baseline_exact_match_rate=_float_mapping_value(
            entry,
            "baseline_exact_match_rate",
        ),
        candidate_mean_edge_recall=_float_mapping_value(
            entry,
            "candidate_mean_evidence_edge_recall",
        ),
        baseline_mean_edge_recall=_float_mapping_value(
            entry,
            "baseline_mean_evidence_edge_recall",
        ),
        candidate_mean_node_recall=_float_mapping_value(
            entry,
            "candidate_mean_evidence_node_recall",
        ),
        baseline_mean_node_recall=_float_mapping_value(
            entry,
            "baseline_mean_evidence_node_recall",
        ),
    )


def _breakdown_delta_from_numbers(
    *,
    candidate_case_count: int | None,
    baseline_case_count: int | None,
    candidate_exact_match_count: int | None,
    baseline_exact_match_count: int | None,
    candidate_exact_match_rate: float | None,
    baseline_exact_match_rate: float | None,
    candidate_mean_edge_recall: float | None,
    baseline_mean_edge_recall: float | None,
    candidate_mean_node_recall: float | None,
    baseline_mean_node_recall: float | None,
) -> dict[str, Any]:
    return {
        "baseline_case_count": baseline_case_count,
        "baseline_exact_match_count": baseline_exact_match_count,
        "baseline_exact_match_rate": baseline_exact_match_rate,
        "baseline_mean_evidence_edge_recall": baseline_mean_edge_recall,
        "baseline_mean_evidence_node_recall": baseline_mean_node_recall,
        "candidate_case_count": candidate_case_count,
        "candidate_exact_match_count": candidate_exact_match_count,
        "candidate_exact_match_rate": candidate_exact_match_rate,
        "candidate_mean_evidence_edge_recall": candidate_mean_edge_recall,
        "candidate_mean_evidence_node_recall": candidate_mean_node_recall,
        "case_count_delta": _int_delta(candidate_case_count, baseline_case_count),
        "case_count_match": candidate_case_count == baseline_case_count,
        "exact_match_count_delta": _int_delta(
            candidate_exact_match_count,
            baseline_exact_match_count,
        ),
        "exact_match_rate_delta": _float_delta(
            candidate_exact_match_rate,
            baseline_exact_match_rate,
        ),
        "mean_evidence_edge_recall_delta": _float_delta(
            candidate_mean_edge_recall,
            baseline_mean_edge_recall,
        ),
        "mean_evidence_node_recall_delta": _float_delta(
            candidate_mean_node_recall,
            baseline_mean_node_recall,
        ),
    }


def _metric_mapping(metrics: object, metric_name: str) -> Mapping[str, Any] | None:
    if not isinstance(metrics, Mapping):
        return None
    metric = metrics.get(metric_name)
    return cast(Mapping[str, Any], metric) if isinstance(metric, Mapping) else None


def _research_axis_mapping(breakdown: object) -> Mapping[str, Any]:
    if not isinstance(breakdown, Mapping):
        return {}
    axes = breakdown.get("by_research_axis")
    return cast(Mapping[str, Any], axes) if isinstance(axes, Mapping) else {}


def _breakdown_group_mapping(
    breakdown: object,
    group_name: str,
) -> Mapping[str, Any]:
    if not isinstance(breakdown, Mapping):
        return {}
    group = breakdown.get(group_name)
    return cast(Mapping[str, Any], group) if isinstance(group, Mapping) else {}


def _int_mapping_value(payload: object, key: str) -> int | None:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _float_mapping_value(payload: object, key: str) -> float | None:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _int_delta(candidate: int | None, baseline: int | None) -> int | None:
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def _float_delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return round(candidate - baseline, 6)


def _required_delta_report_path(report: Mapping[str, Any], key: str) -> Path:
    value = report.get(key)
    if not isinstance(value, str) or not value:
        raise SpatialQAError(f"QA eval delta report missing {key}")
    return Path(value)


def _delta_report_name(report: Mapping[str, Any], key: str) -> str:
    value = report.get(key)
    return value if isinstance(value, str) and value else key.replace("_name", "")


def _prediction_mapping(predictions: Sequence[QAPrediction]) -> dict[str, QAPrediction]:
    mapping: dict[str, QAPrediction] = {}
    for prediction in predictions:
        if prediction.id not in mapping:
            mapping[prediction.id] = prediction
    return mapping


def _case_result(case: QACase, prediction: QAPrediction | None) -> dict[str, Any]:
    if prediction is None:
        return {
            "case_id": case.id,
            "episode_id": case.episode_id,
            "scene_id": case.scene_id,
            "question_type": case.question_type,
            "tags": list(case.tags),
            "reference_frame": case.reference_frame,
            "prediction_id": None,
            "exact_match": False,
            "multiple_choice": bool(case.choices),
            "multiple_choice_correct": False if case.choices else None,
            "numeric_absolute_errors": [],
            "evidence_node_recall": 0.0,
            "evidence_edge_recall": 0.0,
            "answer_graph_consistent": False,
            "confidence": None,
            "error": "missing_prediction",
        }

    exact_match = prediction.error is None and prediction.answer == case.answer
    node_recall = _recall(prediction.evidence_nodes, case.required_nodes)
    edge_recall = _recall(prediction.evidence_edges, case.required_edges)
    return {
        "case_id": case.id,
        "episode_id": case.episode_id,
        "scene_id": case.scene_id,
        "question_type": case.question_type,
        "tags": list(case.tags),
        "reference_frame": case.reference_frame,
        "prediction_id": prediction.id,
        "exact_match": exact_match,
        "multiple_choice": bool(case.choices),
        "multiple_choice_correct": exact_match if case.choices else None,
        "numeric_absolute_errors": _numeric_absolute_errors(case, prediction),
        "evidence_node_recall": node_recall,
        "evidence_edge_recall": edge_recall,
        "answer_graph_consistent": exact_match,
        "confidence": prediction.confidence,
        "error": prediction.error,
    }


def _summary(case_results: Sequence[Mapping[str, Any]], *, prediction_count: int) -> dict[str, Any]:
    case_count = len(case_results)
    exact_match_count = _case_bool_count(case_results, "exact_match")
    missing_prediction_count = sum(1 for result in case_results if result.get("prediction_id") is None)
    return {
        "case_count": case_count,
        "prediction_count": prediction_count,
        "matched_prediction_count": case_count - missing_prediction_count,
        "missing_prediction_count": missing_prediction_count,
        "exact_match_count": exact_match_count,
        "exact_match_rate": _rate(exact_match_count, case_count),
    }


def _metrics(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    case_count = len(case_results)
    exact_match_count = _case_bool_count(case_results, "exact_match")
    consistent_count = _case_bool_count(case_results, "answer_graph_consistent")
    multiple_choice_results = [
        result for result in case_results if result.get("multiple_choice") is True
    ]
    multiple_choice_count = sum(
        1 for result in multiple_choice_results if result.get("multiple_choice_correct") is True
    )
    numeric_errors = [
        float(error)
        for result in case_results
        for error in cast(Sequence[float], result.get("numeric_absolute_errors", ()))
    ]
    total_numeric_error = round(sum(numeric_errors), 6)
    return {
        "answer_graph_consistency": {
            "count": consistent_count,
            "rate": _rate(consistent_count, case_count),
            "total": case_count,
        },
        "evidence_edge_recall": {
            "average": _average_float(result["evidence_edge_recall"] for result in case_results),
            "total": case_count,
        },
        "evidence_node_recall": {
            "average": _average_float(result["evidence_node_recall"] for result in case_results),
            "total": case_count,
        },
        "exact_match": {
            "count": exact_match_count,
            "rate": _rate(exact_match_count, case_count),
            "total": case_count,
        },
        "multiple_choice_accuracy": {
            "count": multiple_choice_count,
            "rate": _rate(multiple_choice_count, len(multiple_choice_results)),
            "total": len(multiple_choice_results),
        },
        "numeric_mae": {
            "count": len(numeric_errors),
            "mean_absolute_error": (
                round(total_numeric_error / len(numeric_errors), 6) if numeric_errors else None
            ),
            "total_absolute_error": total_numeric_error,
        },
    }


def _breakdown_by_question_type(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return _breakdown(case_results, lambda result: str(result["question_type"]))


def _breakdown_by_episode_id(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return _breakdown(case_results, lambda result: str(result["episode_id"]))


def _breakdown_by_scene_id(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return _breakdown(case_results, lambda result: str(result["scene_id"]))


def _breakdown_by_tag(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for result in case_results:
        for tag in cast(Sequence[str], result["tags"]):
            grouped.setdefault(tag, []).append(result)
    return {key: _breakdown_entry(grouped[key]) for key in sorted(grouped)}


def _breakdown_by_reference_frame(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return _breakdown(
        case_results,
        lambda result: str(result["reference_frame"])
        if result["reference_frame"] is not None
        else "none",
    )


def _breakdown_by_research_axis(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {
        axis: [] for axis in QA_RESEARCH_AXIS_NAMES
    }
    for result in case_results:
        for axis in qa_research_axes_for_case(result):
            grouped[axis].append(result)
    return {axis: _breakdown_entry(grouped[axis]) for axis in QA_RESEARCH_AXIS_NAMES}


def qa_research_axes_for_case(case_result: Mapping[str, Any]) -> tuple[str, ...]:
    tags = {
        tag
        for tag in case_result.get("tags", ())
        if isinstance(tag, str) and tag
    }
    question_type = str(case_result.get("question_type", ""))
    axes = {"spatial_qa"}
    if tags & QA_DYNAMIC_MEMORY_TAGS:
        axes.add("dynamic_memory")
    if question_type in QA_GRAPH_TOOL_QUERY_TYPES or tags & QA_GRAPH_TOOL_QUERY_TAGS:
        axes.add("graph_tool_query")
    return tuple(axis for axis in QA_RESEARCH_AXIS_NAMES if axis in axes)


def _breakdown(
    case_results: Sequence[Mapping[str, Any]],
    key_fn: Any,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for result in case_results:
        grouped.setdefault(str(key_fn(result)), []).append(result)
    return {key: _breakdown_entry(grouped[key]) for key in sorted(grouped)}


def _breakdown_entry(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    exact_match_count = _case_bool_count(case_results, "exact_match")
    return {
        "case_count": len(case_results),
        "exact_match_count": exact_match_count,
        "exact_match_rate": _rate(exact_match_count, len(case_results)),
        "mean_evidence_edge_recall": _average_float(
            result["evidence_edge_recall"] for result in case_results
        ),
        "mean_evidence_node_recall": _average_float(
            result["evidence_node_recall"] for result in case_results
        ),
    }


def _recall(predicted: Sequence[str], required: Sequence[str]) -> float:
    if not required:
        return 1.0
    return round(len(set(predicted) & set(required)) / len(set(required)), 6)


def _numeric_absolute_errors(case: QACase, prediction: QAPrediction) -> list[float]:
    if case.answer_type not in {"distance", "metric", "numeric"}:
        return []
    errors: list[float] = []
    for path, expected in _numeric_leaves(case.answer):
        actual = _value_at_path(prediction.answer, path)
        if isinstance(actual, (int, float)) and not isinstance(actual, bool):
            errors.append(round(abs(float(expected) - float(actual)), 6))
    return errors


def _numeric_leaves(payload: Mapping[str, Any], prefix: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], float]]:
    leaves: list[tuple[tuple[str, ...], float]] = []
    for key, value in payload.items():
        path = (*prefix, key)
        if isinstance(value, Mapping):
            leaves.extend(_numeric_leaves(cast(Mapping[str, Any], value), path))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            leaves.append((path, float(value)))
    return leaves


def _value_at_path(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(count / total, 6)


def _average_float(values: Iterable[Any]) -> float:
    numbers = [float(value) for value in values]
    if not numbers:
        return 0.0
    return round(sum(numbers) / len(numbers), 6)


def _case_bool_count(case_results: Sequence[Mapping[str, Any]], key: str) -> int:
    return sum(1 for result in case_results if result.get(key) is True)


def _required_report_path(report: Mapping[str, Any], key: str) -> Path:
    value = report.get(key)
    if not isinstance(value, str) or not value:
        raise SpatialQAError(f"QA eval report missing {key}")
    return Path(value)


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


def _summary_value(summary: object, key: str) -> object:
    if not isinstance(summary, Mapping):
        return None
    return summary.get(key)


def _metric_value(metrics: object, metric_name: str, key: str) -> object:
    if not isinstance(metrics, Mapping):
        return None
    metric = metrics.get(metric_name)
    if not isinstance(metric, Mapping):
        return None
    return metric.get(key)


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], _json_value(value))


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_json_value(item) for item in value]
    return value


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise SpatialQAError(f"QA prediction field must be a string: {key}")
    return value


def _optional_str(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SpatialQAError(f"QA prediction field must be a string: {key}")
    return value


def _required_float(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SpatialQAError(f"QA prediction field must be a number: {key}")
    return float(value)


def _required_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise SpatialQAError(f"QA prediction field must be an object: {key}")
    return _json_mapping(cast(Mapping[str, Any], value))


def _string_tuple(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key, [])
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise SpatialQAError(f"QA prediction field must be a string sequence: {key}")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise SpatialQAError(f"QA prediction field must be a string sequence: {key}")
        items.append(item)
    return tuple(items)
