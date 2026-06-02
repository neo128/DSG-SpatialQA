from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.benchmark import QACase, load_qa_dataset, qa_dataset_digest
from dsg_spatialqa_lab.eval.qa_metrics import (
    QA_RESEARCH_AXIS_NAMES,
    QAPrediction,
    load_qa_predictions,
    qa_predictions_digest,
    qa_research_axes_for_case,
)
from dsg_spatialqa_lab.graph_tool import GraphTool
from dsg_spatialqa_lab.memory import DynamicSceneGraph
from dsg_spatialqa_lab.qa import SpatialQAEngine
from dsg_spatialqa_lab.scene_io import graph_json_digest, load_graph_json
from dsg_spatialqa_lab.schema import Edge, QAResponse, SpatialQAError


ERROR_ATTRIBUTION_REPORT_SCHEMA_VERSION = "dsg-spatialqa-lab.error-attribution-report.v1"


def attribute_qa_errors(
    gold_cases: Sequence[QACase],
    *,
    oracle_graph: DynamicSceneGraph,
    predicted_graph: DynamicSceneGraph,
    predictions: Sequence[QAPrediction],
) -> list[dict[str, Any]]:
    prediction_by_id = _prediction_mapping(predictions)
    rows: list[dict[str, Any]] = []
    for case in gold_cases:
        prediction = prediction_by_id.get(case.id)
        oracle_response = _answer_case(oracle_graph, case)
        predicted_response = _answer_case(predicted_graph, case)
        evidence_status = _evidence_status(case, oracle_graph, predicted_graph)
        answer_correct = (
            prediction is not None
            and prediction.error is None
            and prediction.answer == case.answer
        )
        oracle_correct = oracle_response.error is None and oracle_response.answer == case.answer
        predicted_correct = (
            predicted_response.error is None and predicted_response.answer == case.answer
        )
        error_category = _error_category(
            answer_correct=answer_correct,
            oracle_correct=oracle_correct,
            predicted_correct=predicted_correct,
            required_nodes_present=bool(evidence_status["required_nodes_present"]),
            required_edges_present=bool(evidence_status["required_edges_present"]),
        )
        rows.append(
            {
                "case_id": case.id,
                "question_type": case.question_type,
                "tags": list(case.tags),
                "answer_correct": answer_correct,
                "oracle_answer": _response_answer(oracle_response),
                "predicted_answer": _response_answer(predicted_response),
                "model_answer": prediction.answer if prediction is not None else {},
                "prediction_error": prediction.error if prediction is not None else "missing_prediction",
                "oracle_graph_tool_correct": oracle_correct,
                "predicted_graph_tool_correct": predicted_correct,
                "error_category": error_category,
                **evidence_status,
            }
        )
    return rows


def error_attribution_report(
    gold_cases: Sequence[QACase],
    *,
    oracle_graph: DynamicSceneGraph,
    predicted_graph: DynamicSceneGraph,
    predictions: Sequence[QAPrediction],
    gold_path: str | Path | None = None,
    oracle_graph_path: str | Path | None = None,
    predicted_graph_path: str | Path | None = None,
    prediction_path: str | Path | None = None,
) -> dict[str, Any]:
    cases = attribute_qa_errors(
        gold_cases,
        oracle_graph=oracle_graph,
        predicted_graph=predicted_graph,
        predictions=predictions,
    )
    report: dict[str, Any] = {
        "schema_version": ERROR_ATTRIBUTION_REPORT_SCHEMA_VERSION,
        "gold_path": str(gold_path) if gold_path is not None else None,
        "oracle_graph_path": (
            str(oracle_graph_path) if oracle_graph_path is not None else None
        ),
        "predicted_graph_path": (
            str(predicted_graph_path) if predicted_graph_path is not None else None
        ),
        "prediction_path": str(prediction_path) if prediction_path is not None else None,
        "gold_digest": qa_dataset_digest(gold_cases),
        "oracle_graph_digest": graph_json_digest(oracle_graph),
        "predicted_graph_digest": graph_json_digest(predicted_graph),
        "prediction_digest": qa_predictions_digest(predictions),
        "summary": _summary(cases),
        "cases": cases,
    }
    report["report_digest"] = error_attribution_report_digest(report)
    return report


def error_attribution_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def error_attribution_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_error_attribution_report(report: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(error_attribution_report_json(report), encoding="utf-8")
    return output_path


def load_error_attribution_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Error attribution report JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_error_attribution_report(report: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = report.get("schema_version")
    report_digest = _string_or_none(report.get("report_digest"))
    expected_report_digest = error_attribution_report_digest(report)
    cases = report.get("cases")
    case_rows = (
        cast(Sequence[Mapping[str, Any]], cases)
        if isinstance(cases, Sequence) and not isinstance(cases, str)
        else ()
    )
    expected_summary = _summary(case_rows)
    summary = report.get("summary")
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == ERROR_ATTRIBUTION_REPORT_SCHEMA_VERSION,
            "expected": ERROR_ATTRIBUTION_REPORT_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "report_digest",
            "passed": report_digest == expected_report_digest,
            "expected": expected_report_digest,
            "actual": report_digest,
        },
        {
            "name": "summary",
            "passed": summary == expected_summary,
            "expected": expected_summary,
            "actual": summary,
        },
    ]
    if summary != expected_summary:
        checks[-1]["differences"] = _differences(summary, expected_summary)
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "report_digest": report_digest,
        "checks": checks,
    }


def compare_error_attribution_report(report: Mapping[str, Any]) -> dict[str, Any]:
    gold_path = _required_report_path(report, "gold_path")
    oracle_graph_path = _required_report_path(report, "oracle_graph_path")
    predicted_graph_path = _required_report_path(report, "predicted_graph_path")
    prediction_path = _required_report_path(report, "prediction_path")
    current_report = error_attribution_report(
        load_qa_dataset(gold_path),
        oracle_graph=load_graph_json(oracle_graph_path),
        predicted_graph=load_graph_json(predicted_graph_path),
        predictions=load_qa_predictions(prediction_path),
        gold_path=gold_path,
        oracle_graph_path=oracle_graph_path,
        predicted_graph_path=predicted_graph_path,
        prediction_path=prediction_path,
    )
    validation = validate_error_attribution_report(report)
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
        _equality_check("cases_match_current", report.get("cases"), current_report["cases"]),
    ]
    return {
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def _answer_case(graph: DynamicSceneGraph, case: QACase) -> QAResponse:
    return SpatialQAEngine(GraphTool(graph)).answer(case.question)


def _response_answer(response: QAResponse) -> dict[str, Any]:
    if response.error is None:
        return _json_mapping(response.answer)
    return {"error": response.error}


def _error_category(
    *,
    answer_correct: bool,
    oracle_correct: bool,
    predicted_correct: bool,
    required_nodes_present: bool,
    required_edges_present: bool,
) -> str:
    if not oracle_correct:
        return "benchmark_or_engine_error"
    if answer_correct:
        return "correct"
    if not required_nodes_present or not required_edges_present:
        return "evidence_missing"
    if not predicted_correct:
        return "graph_construction"
    return "reasoning_or_tool_use"


def _evidence_status(
    case: QACase,
    oracle_graph: DynamicSceneGraph,
    predicted_graph: DynamicSceneGraph,
) -> dict[str, Any]:
    missing_nodes = [
        node_id for node_id in case.required_nodes if node_id not in predicted_graph.nodes
    ]
    missing_edges = [
        edge_id for edge_id in case.required_edges if edge_id not in _edge_ids(predicted_graph)
    ]
    wrong_label_nodes = [
        node_id
        for node_id in case.required_nodes
        if node_id in oracle_graph.object_states
        and node_id in predicted_graph.object_states
        and oracle_graph.object_states[node_id].label
        != predicted_graph.object_states[node_id].label
    ]
    evidence_error_category = _evidence_error_category(
        missing_nodes=missing_nodes,
        missing_edges=missing_edges,
        wrong_label_nodes=wrong_label_nodes,
        oracle_graph=oracle_graph,
        predicted_graph=predicted_graph,
    )
    return {
        "required_nodes_present": not missing_nodes,
        "required_edges_present": not missing_edges,
        "missing_required_nodes": missing_nodes,
        "missing_required_edges": missing_edges,
        "predicted_evidence_sources": _predicted_evidence_sources(
            case,
            predicted_graph,
        ),
        "wrong_label_nodes": wrong_label_nodes,
        "evidence_error_category": evidence_error_category,
    }


def _evidence_error_category(
    *,
    missing_nodes: Sequence[str],
    missing_edges: Sequence[str],
    wrong_label_nodes: Sequence[str],
    oracle_graph: DynamicSceneGraph,
    predicted_graph: DynamicSceneGraph,
) -> str:
    if any(node_id in oracle_graph.object_states for node_id in missing_nodes):
        return "missing_object"
    if any(node_id.startswith("state:") for node_id in missing_nodes):
        return "missing_state"
    if wrong_label_nodes:
        return "wrong_object_label"
    if any(_missing_edge_category(edge_id, oracle_graph, predicted_graph) == "missing_state" for edge_id in missing_edges):
        return "missing_state"
    if any(_missing_edge_category(edge_id, oracle_graph, predicted_graph) == "wrong_relation" for edge_id in missing_edges):
        return "wrong_relation"
    if missing_edges:
        return "missing_relation"
    return "none"


def _missing_edge_category(
    edge_id: str,
    oracle_graph: DynamicSceneGraph,
    predicted_graph: DynamicSceneGraph,
) -> str:
    oracle_edge = _edge_by_id(oracle_graph).get(edge_id)
    if oracle_edge is None:
        return "missing_relation"
    if oracle_edge.relation == "STATE_CHANGED":
        return "missing_state"
    if _same_endpoints_different_relation(oracle_edge, predicted_graph):
        return "wrong_relation"
    return "missing_relation"


def _same_endpoints_different_relation(
    oracle_edge: Edge,
    predicted_graph: DynamicSceneGraph,
) -> bool:
    contradictory_relations = _contradictory_relations(oracle_edge.relation)
    return any(
        edge.src == oracle_edge.src
        and edge.dst == oracle_edge.dst
        and edge.reference_frame == oracle_edge.reference_frame
        and edge.step == oracle_edge.step
        and edge.relation in contradictory_relations
        for edge in predicted_graph.edges
    )


def _contradictory_relations(relation: str) -> set[str]:
    opposites = {
        "LEFT_OF": {"RIGHT_OF"},
        "RIGHT_OF": {"LEFT_OF"},
        "FRONT_OF": {"BEHIND"},
        "BEHIND": {"FRONT_OF"},
    }
    return opposites.get(relation, set())


def _summary(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = _summary_core(cases)
    summary["by_research_axis"] = _research_axis_summary(cases)
    return summary


def _summary_core(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "answer_correct_count": _bool_count(cases, "answer_correct"),
        "by_error_category": _sorted_counts(str(case["error_category"]) for case in cases),
        "by_evidence_error_category": _sorted_counts(
            str(case["evidence_error_category"]) for case in cases
        ),
        "by_predicted_evidence_source": _source_summary(cases),
        "case_count": len(cases),
        "error_count": sum(1 for case in cases if case.get("error_category") != "correct"),
        "oracle_graph_tool_correct_count": _bool_count(cases, "oracle_graph_tool_correct"),
        "predicted_graph_tool_correct_count": _bool_count(
            cases,
            "predicted_graph_tool_correct",
        ),
    }


def _research_axis_summary(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {
        axis: [] for axis in QA_RESEARCH_AXIS_NAMES
    }
    for case in cases:
        for axis in qa_research_axes_for_case(case):
            grouped[axis].append(case)
    return {axis: _summary_core(grouped[axis]) for axis in QA_RESEARCH_AXIS_NAMES}


def _predicted_evidence_sources(
    case: QACase,
    predicted_graph: DynamicSceneGraph,
) -> list[str]:
    edge_by_id = _edge_by_id(predicted_graph)
    sources = {
        _node_source(predicted_graph, node_id)
        for node_id in case.required_nodes
        if node_id in predicted_graph.nodes
    }
    sources.update(
        _edge_source(predicted_graph, edge)
        for edge_id in case.required_edges
        if (edge := edge_by_id.get(edge_id)) is not None
    )
    normalized = sorted(source for source in sources if source is not None)
    return normalized or ["missing_predicted_evidence"]


def _node_source(
    graph: DynamicSceneGraph,
    node_id: str,
    visited: set[str] | None = None,
) -> str:
    visited_value = visited or set()
    if node_id in visited_value:
        return "unknown"
    visited_value.add(node_id)
    node = graph.nodes[node_id]
    source = _source_from_attributes(node.attributes)
    if source is not None:
        return source
    object_id = node.attributes.get("object_id")
    if isinstance(object_id, str) and object_id in graph.nodes:
        return _node_source(graph, object_id, visited_value)
    return "unknown"


def _edge_source(graph: DynamicSceneGraph, edge: Edge) -> str:
    source = _source_from_attributes(edge.attributes)
    if source is not None:
        return source
    if edge.src in graph.nodes:
        return _node_source(graph, edge.src)
    if edge.dst in graph.nodes:
        return _node_source(graph, edge.dst)
    return "unknown"


def _source_from_attributes(attributes: Mapping[str, Any]) -> str | None:
    for key in ("source", "source_name", "source_kind"):
        value = attributes.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _source_summary(cases: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for case in cases:
        sources = case.get("predicted_evidence_sources")
        if not isinstance(sources, Sequence) or isinstance(sources, str):
            sources = ("missing_predicted_evidence",)
        for source in sources:
            if not isinstance(source, str) or source == "":
                source = "unknown"
            grouped.setdefault(source, []).append(case)
    return {
        source: {
            "by_error_category": _sorted_counts(
                str(case["error_category"]) for case in source_cases
            ),
            "by_evidence_error_category": _sorted_counts(
                str(case["evidence_error_category"]) for case in source_cases
            ),
            "case_count": len(source_cases),
            "error_count": sum(
                1
                for case in source_cases
                if case.get("error_category") != "correct"
            ),
        }
        for source, source_cases in sorted(grouped.items())
    }


def _prediction_mapping(predictions: Sequence[QAPrediction]) -> dict[str, QAPrediction]:
    mapping: dict[str, QAPrediction] = {}
    for prediction in predictions:
        if prediction.id not in mapping:
            mapping[prediction.id] = prediction
    return mapping


def _edge_ids(graph: DynamicSceneGraph) -> set[str]:
    return {edge.id for edge in graph.edges}


def _edge_by_id(graph: DynamicSceneGraph) -> dict[str, Edge]:
    return {edge.id: edge for edge in graph.edges}


def _bool_count(cases: Sequence[Mapping[str, Any]], key: str) -> int:
    return sum(1 for case in cases if case.get(key) is True)


def _sorted_counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def _required_report_path(report: Mapping[str, Any], key: str) -> Path:
    value = report.get(key)
    if not isinstance(value, str) or not value:
        raise SpatialQAError(f"Error attribution report missing {key}")
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
