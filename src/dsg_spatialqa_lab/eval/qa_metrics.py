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
            "by_question_type": _breakdown_by_question_type(case_results),
            "by_tag": _breakdown_by_tag(case_results),
            "by_reference_frame": _breakdown_by_reference_frame(case_results),
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
    case_results = cast(Sequence[Mapping[str, Any]], cases) if isinstance(cases, Sequence) else ()
    expected_case_count = len(case_results) if not isinstance(cases, str) else None
    exact_match_count = _case_bool_count(case_results, "exact_match")
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
