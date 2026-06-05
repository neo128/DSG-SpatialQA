from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from importlib.resources import files
import json
from pathlib import Path
from typing import Any, TypeAlias, cast

from dsg_spatialqa_lab.benchmark import QACase, qa_case_to_dict
from dsg_spatialqa_lab.memory import CONTAINMENT_RELATIONS, DynamicSceneGraph
from dsg_spatialqa_lab.scene_io import graph_summary
from dsg_spatialqa_lab.schema import Edge, Node, SpatialQAError


DSG_VIEWER_PAYLOAD_SCHEMA_VERSION = "dsg-spatialqa-lab.dsg-viewer-payload.v1"
DSGViewerQACase: TypeAlias = QACase | Mapping[str, Any]
ACTIVE_QA_V2_SCHEMA_VERSION = "dsg-spatialqa-lab.active-qa-case.v2"
REACHABLE_NBV_TRAJECTORY_SCHEMA_VERSION = "dsg-spatialqa-lab.reachable-nbv-trajectory.v1"
LATEST_TRAJECTORY_GLOB = "outputs/navigation/reachable-nbv-real-ai2thor-trajectory-episode*.json"
LATEST_PREDICTED_GRAPH_GLOB = (
    "outputs/predicted-dsg/predicted-graph-real-ai2thor-reachable-nbv-episode*.json"
)
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
    qa_cases: Sequence[DSGViewerQACase] = (),
    qa_eval_report: Mapping[str, Any] | None = None,
    graph_eval_report: Mapping[str, Any] | None = None,
    oracle_graph: DynamicSceneGraph | None = None,
    trajectory_report: Mapping[str, Any] | None = None,
    predicted_graph_path: str | Path | None = None,
    evidence_report: Mapping[str, Any] | None = None,
    evidence_report_path: str | Path | None = None,
    trajectory_report_path: str | Path | None = None,
) -> dict[str, Any]:
    summary = graph_summary(predicted_graph)
    readiness = _mapping_or_empty(
        evidence_report.get("readiness") if evidence_report is not None else None
    )
    qa_eval_by_id = _qa_eval_by_case_id(qa_eval_report)
    qa_rows = [
        _qa_case_row(case, qa_eval_by_id.get(_qa_input_case_id(case)))
        for case in qa_cases
    ]
    qa_case_ids_by_object_id = _qa_case_ids_by_object_id(qa_rows)
    oracle_delta = _oracle_delta(graph_eval_report, oracle_graph)
    graph_eval_metrics = _graph_eval_metrics(graph_eval_report)
    trajectory = _trajectory_payload(trajectory_report)
    trajectory_summary = _mapping_or_empty(trajectory.get("summary"))
    payload: dict[str, Any] = {
        "schema_version": DSG_VIEWER_PAYLOAD_SCHEMA_VERSION,
        "artifacts": _artifact_paths(
            predicted_graph_path=predicted_graph_path,
            evidence_report_path=evidence_report_path,
            trajectory_report_path=trajectory_report_path,
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
            "qa_case_count": len(qa_rows),
            "evidence_ready": readiness.get("ready"),
            "trajectory_step_count": trajectory_summary.get("step_count"),
            "trajectory_station_count": trajectory_summary.get("station_count"),
            "navigation_validated": trajectory_summary.get("navigation_validated"),
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
        "trajectory": trajectory,
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


def dsg_viewer_html() -> str:
    return (
        files("dsg_spatialqa_lab.visualization")
        .joinpath("static", "dsg_viewer.html")
        .read_text(encoding="utf-8")
    )


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
    trajectory_path = _latest_existing_workspace_glob(
        workspace_path,
        LATEST_TRAJECTORY_GLOB,
    )
    trajectory_report = _load_optional_workspace_mapping(trajectory_path)
    if trajectory_path is not None:
        paths["trajectory_report_path"] = str(trajectory_path)
    if trajectory_report is not None:
        predicted_graph_path = _workspace_artifact_path(
            workspace_path,
            trajectory_report.get("predicted_graph_path"),
        )
        if predicted_graph_path is not None and predicted_graph_path.exists():
            paths["predicted_graph_path"] = str(predicted_graph_path)
        episode_id = _string_or_none(trajectory_report.get("episode_id"))
        if episode_id is not None:
            qa_path = dsg_viewer_resolve_workspace_path(
                workspace_path,
                Path("inputs") / "qa-v2-active" / episode_id,
            )
            if qa_path.exists():
                paths["qa_path"] = str(qa_path)
    if "predicted_graph_path" not in paths:
        predicted_graph_path = _latest_existing_workspace_glob(
            workspace_path,
            LATEST_PREDICTED_GRAPH_GLOB,
        )
        if predicted_graph_path is not None:
            paths["predicted_graph_path"] = str(predicted_graph_path)
    if "qa_path" not in paths:
        latest_qa_path = _latest_active_qa_workspace_path(workspace_path)
        if latest_qa_path is not None:
            paths["qa_path"] = str(latest_qa_path)
    for key, candidate_paths in DSG_VIEWER_WORKSPACE_PRESET_PATHS.items():
        if key in paths:
            continue
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
    case: DSGViewerQACase,
    eval_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(case, QACase):
        return _legacy_qa_case_row(case, eval_result)
    return _active_qa_case_row(case, eval_result)


def _legacy_qa_case_row(
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


def _active_qa_case_row(
    case: Mapping[str, Any],
    eval_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    situation = _mapping_or_empty(case.get("situation"))
    target = _mapping_or_empty(case.get("target"))
    question_text = _string_or_none(case.get("question_text")) or ""
    case_id = _qa_input_case_id(case)
    row = {
        "case_id": case_id,
        "scene_id": _string_or_none(case.get("scene_id")),
        "episode_id": _string_or_none(case.get("episode_id")),
        "step": _int_or_none(situation.get("step")) or _int_or_none(case.get("step")),
        "question_type": _string_or_none(case.get("question_type")),
        "question_text": question_text,
        "question": {"text": question_text},
        "answer": dict(_mapping_or_empty(case.get("answer"))),
        "answer_options": [dict(item) for item in _mapping_sequence(case.get("answer_options"))],
        "target": dict(target),
        "target_object_ids": _active_target_object_ids(case),
        "evidence_frames": _int_list(case.get("evidence_frames")),
        "required_evidence": dict(_mapping_or_empty(case.get("required_evidence"))),
        "situation": dict(situation),
        "observability": dict(_mapping_or_empty(case.get("observability"))),
        "anti_shortcut": dict(_mapping_or_empty(case.get("anti_shortcut"))),
        "trajectory_context": dict(_mapping_or_empty(case.get("trajectory_context"))),
        "split": _string_or_none(case.get("split")),
        "source_graph_digest": _string_or_none(case.get("source_graph_digest")),
        "eval": dict(eval_result) if eval_result is not None else {},
        "case": dict(case),
    }
    return row


def _qa_input_case_id(case: DSGViewerQACase) -> str:
    if isinstance(case, QACase):
        return case.id
    case_id = _string_or_none(case.get("id")) or _string_or_none(case.get("case_id"))
    if case_id is None:
        raise SpatialQAError("Active QA v2 case must include id or case_id")
    return case_id


def _target_object_ids(case: QACase) -> list[str]:
    values: list[str] = []
    for payload in (case.question, case.answer):
        for key in ("object_id", "target_object_id", "reference_object_id"):
            _append_unique(values, payload.get(key))
        _append_many_unique(values, payload.get("object_ids"))
        _append_many_unique(values, payload.get("target_object_ids"))
    return values


def _active_target_object_ids(case: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    target = _mapping_or_empty(case.get("target"))
    _append_unique(values, target.get("object_id"))
    answer = _mapping_or_empty(case.get("answer"))
    _append_unique(values, answer.get("src"))
    required_evidence = _mapping_or_empty(case.get("required_evidence"))
    target_nodes = required_evidence.get("target_nodes")
    _append_many_unique(values, target_nodes)
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


def _trajectory_payload(trajectory_report: Mapping[str, Any] | None) -> dict[str, Any]:
    if trajectory_report is None:
        return {}
    paths = {
        key: value
        for key in (
            "predicted_graph_path",
            "decision_trace_path",
            "reachable_positions_path",
            "observation_sequence_path",
            "topdown_path_png",
            "fixed_vs_nbv_overlay_png",
        )
        if isinstance((value := trajectory_report.get(key)), str)
    }
    steps = _mapping_sequence(trajectory_report.get("steps"))
    stations = _mapping_sequence(trajectory_report.get("stations"))
    rejected_candidates = _mapping_sequence(trajectory_report.get("rejected_candidates"))
    return {
        "schema_version": trajectory_report.get("schema_version"),
        "trajectory_id": trajectory_report.get("trajectory_id"),
        "episode_id": trajectory_report.get("episode_id"),
        "scene_id": trajectory_report.get("scene_id"),
        "collection_kind": trajectory_report.get("collection_kind"),
        "runtime_kind": trajectory_report.get("runtime_kind"),
        "summary": {
            "step_count": len(steps),
            "station_count": len(stations),
            "rejected_candidate_count": len(rejected_candidates),
            "navigation_validated": trajectory_report.get("navigation_validated"),
            "real_ai2thor_runtime": trajectory_report.get("real_ai2thor_runtime"),
            "closed_loop_memory_update": trajectory_report.get("closed_loop_memory_update"),
        },
        "paths": paths,
        "steps": [dict(step) for step in steps],
        "stations": [dict(station) for station in stations],
        "rejected_candidates": [dict(candidate) for candidate in rejected_candidates],
    }


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


def _latest_existing_workspace_glob(workspace: Path, pattern: str) -> Path | None:
    workspace_path = Path(workspace)
    matches = sorted(
        path.resolve()
        for path in workspace_path.glob(pattern)
        if path.exists()
    )
    if not matches:
        return None
    return matches[-1]


def _latest_active_qa_workspace_path(workspace: Path) -> Path | None:
    qa_root = dsg_viewer_resolve_workspace_path(workspace, Path("inputs") / "qa-v2-active")
    if not qa_root.exists():
        return None
    episode_dirs = sorted(path.resolve() for path in qa_root.iterdir() if path.is_dir())
    if not episode_dirs:
        return None
    return episode_dirs[-1]


def _load_optional_workspace_mapping(path: Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError(f"Workspace artifact JSON must be an object: {path}")
    return cast(Mapping[str, Any], payload)


def _workspace_artifact_path(workspace: Path, value: object) -> Path | None:
    path_text = _string_or_none(value)
    if path_text is None:
        return None
    workspace_path = workspace.resolve()
    candidate = Path(path_text)
    candidates = (
        candidate if candidate.is_absolute() else workspace_path / candidate,
        candidate if candidate.is_absolute() else candidate.resolve(),
    )
    for item in candidates:
        resolved = item.resolve()
        try:
            resolved.relative_to(workspace_path)
        except ValueError:
            continue
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


def _int_list(value: object) -> list[int]:
    if not isinstance(value, list | tuple):
        return []
    return [
        item
        for item in value
        if isinstance(item, int) and not isinstance(item, bool)
    ]


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _append_unique(values: list[str], value: object) -> None:
    if isinstance(value, str) and value not in values:
        values.append(value)


def _append_many_unique(values: list[str], value: object) -> None:
    for item in _string_list(value):
        _append_unique(values, item)
