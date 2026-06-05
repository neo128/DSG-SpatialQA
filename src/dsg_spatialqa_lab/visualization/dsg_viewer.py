from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.benchmark import QACase, qa_case_to_dict
from dsg_spatialqa_lab.memory import CONTAINMENT_RELATIONS, DynamicSceneGraph
from dsg_spatialqa_lab.scene_io import graph_summary
from dsg_spatialqa_lab.schema import Edge, Node, SpatialQAError


DSG_VIEWER_PAYLOAD_SCHEMA_VERSION = "dsg-spatialqa-lab.dsg-viewer-payload.v1"
DSG_VIEWER_WORKSPACE_PRESET_PATHS: Mapping[str, tuple[str, ...]] = {
    "predicted_graph_path": ("outputs/predicted-dsg/predicted-graph.json",),
    "oracle_graph_path": (
        "outputs/benchmark/graphs/oracle-graph.json",
        "outputs/benchmark/graphs/graph.json",
    ),
    "qa_path": (
        "inputs/qa.jsonl",
        "inputs/qa-dataset.jsonl",
        "outputs/benchmark/qa.jsonl",
    ),
    "qa_eval_report_path": (
        "outputs/offline-controls/qa-eval-observation-aware-p4-target60/qa-eval-report.json",
        "outputs/offline-controls/qa-eval-report.json",
    ),
    "graph_eval_report_path": (
        "outputs/benchmark/graph-eval-report.json",
        "outputs/reports/graph-eval.json",
    ),
    "evidence_report_path": (
        "outputs/predicted-dsg/predicted-evidence-report.json",
        "outputs/reports/predicted-dsg-evidence-report.json",
    ),
}


def dsg_viewer_payload(
    *,
    predicted_graph: DynamicSceneGraph,
    qa_cases: Sequence[QACase] = (),
    qa_eval_report: Mapping[str, Any] | None = None,
    graph_eval_report: Mapping[str, Any] | None = None,
    oracle_graph: DynamicSceneGraph | None = None,
    predicted_graph_path: str | Path | None = None,
    evidence_report: Mapping[str, Any] | None = None,
    evidence_report_path: str | Path | None = None,
) -> dict[str, Any]:
    summary = graph_summary(predicted_graph)
    readiness = _mapping_or_empty(
        evidence_report.get("readiness") if evidence_report is not None else None
    )
    qa_eval_by_id = _qa_eval_by_case_id(qa_eval_report)
    qa_rows = [_qa_case_row(case, qa_eval_by_id.get(case.id)) for case in qa_cases]
    qa_case_ids_by_object_id = _qa_case_ids_by_object_id(qa_rows)
    oracle_delta = _oracle_delta(graph_eval_report, oracle_graph)
    graph_eval_metrics = _graph_eval_metrics(graph_eval_report)
    payload: dict[str, Any] = {
        "schema_version": DSG_VIEWER_PAYLOAD_SCHEMA_VERSION,
        "artifacts": _artifact_paths(
            predicted_graph_path=predicted_graph_path,
            evidence_report_path=evidence_report_path,
        ),
        "graph": {
            "summary": summary,
            "nodes": [
                _node_row(node)
                for node in sorted(predicted_graph.nodes.values(), key=lambda item: item.id)
            ],
            "edges": [
                _edge_row(edge)
                for edge in sorted(predicted_graph.edges, key=_viewer_edge_sort_key)
            ],
        },
        "metrics": {
            "object_count": summary["object_count"],
            "edge_count": summary["edge_count"],
            "evidence_ready": readiness.get("ready"),
            **graph_eval_metrics,
        },
        "diagnostics": {
            "failed_checks": _string_list(readiness.get("failed_checks")),
        },
        "evidence": {
            "report_digest": evidence_report.get("report_digest")
            if evidence_report is not None
            else None,
            "summary": evidence_report.get("evidence_summary")
            if evidence_report is not None
            else {},
        },
        "qa": {"cases": qa_rows},
        "oracle": oracle_delta,
        "indexes": {"qa_case_ids_by_object_id": qa_case_ids_by_object_id},
    }
    payload["payload_digest"] = dsg_viewer_payload_digest(payload)
    return payload


def dsg_viewer_payload_digest(payload: Mapping[str, Any]) -> str:
    digest_payload = {
        key: value for key, value in payload.items() if key != "payload_digest"
    }
    return hashlib.sha256(
        json.dumps(digest_payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def dsg_viewer_payload_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def save_dsg_viewer_payload(payload: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dsg_viewer_payload_json(payload), encoding="utf-8")
    return output_path


def load_dsg_viewer_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("DSG viewer payload JSON must be an object")
    return cast(dict[str, Any], payload)


def dsg_viewer_resolve_workspace_path(workspace: str | Path, path: str | Path) -> Path:
    workspace_path = Path(workspace).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = workspace_path / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(workspace_path)
    except ValueError as exc:
        raise SpatialQAError(f"Path is outside workspace: {path}") from exc
    return resolved


def dsg_viewer_workspace_preset(workspace: str | Path) -> dict[str, Any]:
    workspace_path = Path(workspace)
    paths: dict[str, str] = {}
    for key, candidate_paths in DSG_VIEWER_WORKSPACE_PRESET_PATHS.items():
        existing_path = _first_existing_workspace_path(workspace_path, candidate_paths)
        if existing_path is not None:
            paths[key] = str(existing_path)
    return {
        "workspace_path": str(workspace_path),
        "paths": paths,
    }


def _node_row(node: Node) -> dict[str, Any]:
    return {
        "id": node.id,
        "type": node.type,
        "label": node.label,
        "attributes": dict(node.attributes),
    }


def _edge_row(edge: Edge) -> dict[str, Any]:
    return {
        "id": edge.id,
        "src": edge.src,
        "relation": edge.relation,
        "dst": edge.dst,
        "reference_frame": edge.reference_frame,
        "confidence": edge.confidence,
        "step": edge.step,
        "evidence": list(edge.evidence),
        "attributes": dict(edge.attributes),
    }


def _qa_case_row(
    case: QACase,
    eval_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    case_payload = qa_case_to_dict(case)
    return {
        "case_id": case.id,
        "scene_id": case.scene_id,
        "episode_id": case.episode_id,
        "step": case.step,
        "question_type": case.question_type,
        "question": case_payload["question"],
        "answer": case_payload["answer"],
        "target_object_ids": _target_object_ids(case),
        "eval": dict(eval_result) if eval_result is not None else {},
        "case": case_payload,
    }


def _target_object_ids(case: QACase) -> list[str]:
    values: list[str] = []
    for payload in (case.question, case.answer):
        for key in ("object_id", "target_object_id", "reference_object_id"):
            _append_unique(values, payload.get(key))
        _append_many_unique(values, payload.get("object_ids"))
        _append_many_unique(values, payload.get("target_object_ids"))
    return values


def _qa_eval_by_case_id(
    qa_eval_report: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    if qa_eval_report is None:
        return {}
    rows = _mapping_sequence(qa_eval_report.get("cases"))
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str):
            case_id = row.get("id")
        if isinstance(case_id, str):
            by_id[case_id] = row
    return by_id


def _qa_case_ids_by_object_id(qa_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    by_object_id: dict[str, list[str]] = {}
    for row in qa_rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str):
            continue
        for object_id in _string_list(row.get("target_object_ids")):
            by_object_id.setdefault(object_id, []).append(case_id)
    return dict(sorted(by_object_id.items()))


def _oracle_delta(
    graph_eval_report: Mapping[str, Any] | None,
    oracle_graph: DynamicSceneGraph | None,
) -> dict[str, Any]:
    comparison = _mapping_or_empty(
        graph_eval_report.get("comparison") if graph_eval_report is not None else None
    )
    object_matches = _mapping_sequence(comparison.get("object_matches"))
    missing_relations = _mapping_sequence(comparison.get("missing_relations"))
    extra_relations = _mapping_sequence(comparison.get("extra_relations"))
    object_matches_by_predicted_id = {
        predicted_id: dict(match)
        for match in object_matches
        if isinstance((predicted_id := match.get("predicted_object_id")), str)
    }
    delta: dict[str, Any] = {
        "object_matches_by_predicted_id": dict(
            sorted(object_matches_by_predicted_id.items())
        ),
        "missing_relations": [dict(row) for row in missing_relations],
        "extra_relations": [dict(row) for row in extra_relations],
        "missing_relation_count": len(missing_relations),
        "extra_relation_count": len(extra_relations),
    }
    if oracle_graph is not None:
        delta["graph"] = {
            "summary": graph_summary(oracle_graph),
            "nodes": [
                _node_row(node)
                for node in sorted(oracle_graph.nodes.values(), key=lambda item: item.id)
            ],
            "edges": [
                _edge_row(edge)
                for edge in sorted(oracle_graph.edges, key=_viewer_edge_sort_key)
            ],
        }
    return delta


def _graph_eval_metrics(
    graph_eval_report: Mapping[str, Any] | None,
) -> dict[str, float]:
    if graph_eval_report is None:
        return {}
    metrics: dict[str, float] = {}
    for key in (
        "object_recall",
        "relation_precision",
        "relation_recall",
        "relation_f1",
    ):
        rate = _metric_rate(graph_eval_report, key)
        if rate is not None:
            metrics[key] = rate
    return metrics


def _metric_rate(report: Mapping[str, Any], key: str) -> float | None:
    for section_name in ("metrics", "summary"):
        section = _mapping_or_empty(report.get(section_name))
        value = section.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
        value_mapping = _mapping_or_empty(value)
        rate = value_mapping.get("rate")
        if isinstance(rate, int | float) and not isinstance(rate, bool):
            return float(rate)
    return None


def _viewer_edge_sort_key(edge: Edge) -> tuple[int, int, str, str, str, str]:
    relation_rank = 0 if edge.relation in CONTAINMENT_RELATIONS else 1
    return (
        relation_rank,
        edge.step,
        edge.src,
        edge.relation,
        edge.dst,
        edge.reference_frame,
    )


def _artifact_paths(**paths: str | Path | None) -> dict[str, str]:
    return {
        key: str(path)
        for key, path in sorted(paths.items())
        if path is not None
    }


def _first_existing_workspace_path(
    workspace: Path,
    candidate_paths: Sequence[str],
) -> Path | None:
    for candidate_path in candidate_paths:
        resolved = dsg_viewer_resolve_workspace_path(workspace, candidate_path)
        if resolved.exists():
            return resolved
    return None


def _mapping_or_empty(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    return {}


def _mapping_sequence(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    return [
        cast(Mapping[str, Any], item)
        for item in value
        if isinstance(item, Mapping)
    ]


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [item for item in value if isinstance(item, str)]


def _append_unique(values: list[str], value: object) -> None:
    if isinstance(value, str) and value not in values:
        values.append(value)


def _append_many_unique(values: list[str], value: object) -> None:
    for item in _string_list(value):
        _append_unique(values, item)
