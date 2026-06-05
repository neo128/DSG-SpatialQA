from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, cast

from dsg_spatialqa_lab.memory import CONTAINMENT_RELATIONS, DynamicSceneGraph
from dsg_spatialqa_lab.scene_io import graph_json_digest, load_graph_json
from dsg_spatialqa_lab.schema import Edge, ObjectState, SpatialQAError


GRAPH_EVAL_REPORT_SCHEMA_VERSION = "dsg-spatialqa-lab.graph-eval-report.v1"
GraphMatchingStrategy = Literal["exact", "label_center", "label_center_room"]
DEFAULT_CENTER_DISTANCE_THRESHOLD = 0.25


def compare_graphs(
    oracle_graph: DynamicSceneGraph,
    predicted_graph: DynamicSceneGraph,
    *,
    matching: GraphMatchingStrategy = "exact",
    center_distance_threshold: float = DEFAULT_CENTER_DISTANCE_THRESHOLD,
) -> dict[str, Any]:
    _validate_matching(matching, center_distance_threshold)
    oracle_object_ids = set(oracle_graph.object_states)
    predicted_object_ids = set(predicted_graph.object_states)
    object_matches = _object_matches(
        oracle_graph,
        predicted_graph,
        matching=matching,
        center_distance_threshold=center_distance_threshold,
    )
    matched_oracle_object_ids = [match["oracle_object_id"] for match in object_matches]
    matched_predicted_object_ids = [match["predicted_object_id"] for match in object_matches]
    missing_object_ids = sorted(oracle_object_ids - set(matched_oracle_object_ids))
    extra_object_ids = sorted(predicted_object_ids - set(matched_predicted_object_ids))
    oracle_relations = _relation_mapping(oracle_graph)
    predicted_relations = _relation_mapping(
        predicted_graph,
        predicted_to_oracle_id={
            match["predicted_object_id"]: match["oracle_object_id"] for match in object_matches
        },
    )
    matched_relations = sorted(set(oracle_relations) & set(predicted_relations))
    missing_relations = sorted(set(oracle_relations) - set(predicted_relations))
    extra_relations = sorted(set(predicted_relations) - set(oracle_relations))
    label_matches = [
        match
        for match in object_matches
        if oracle_graph.object_states[match["oracle_object_id"]].label
        == predicted_graph.object_states[match["predicted_object_id"]].label
    ]
    state_matches = [
        match
        for match in object_matches
        if _object_state_matches(
            oracle_graph.object_states[match["oracle_object_id"]],
            predicted_graph.object_states[match["predicted_object_id"]],
        )
    ]
    center_distances = [
        round(
            oracle_graph.object_states[match["oracle_object_id"]].pose.distance_to(
                predicted_graph.object_states[match["predicted_object_id"]].pose
            ),
            6,
        )
        for match in object_matches
    ]
    object_weighted_precision = _confidence_weighted_metric(
        (
            predicted_graph.object_states[predicted_object_id].confidence
            for predicted_object_id in matched_predicted_object_ids
        ),
        (state.confidence for state in predicted_graph.object_states.values()),
    )
    object_weighted_recall = _confidence_weighted_metric(
        (
            oracle_graph.object_states[oracle_object_id].confidence
            for oracle_object_id in matched_oracle_object_ids
        ),
        (state.confidence for state in oracle_graph.object_states.values()),
    )
    relation_weighted_precision = _confidence_weighted_metric(
        (predicted_relations[key].confidence for key in matched_relations),
        (edge.confidence for edge in predicted_relations.values()),
    )
    relation_weighted_recall = _confidence_weighted_metric(
        (oracle_relations[key].confidence for key in matched_relations),
        (edge.confidence for edge in oracle_relations.values()),
    )
    diagnostics = _matching_diagnostics(
        oracle_graph,
        predicted_graph,
        object_matches,
        matching=matching,
        center_distance_threshold=center_distance_threshold,
    )
    unlocated_object_ids = _unlocated_object_ids(predicted_graph)
    metrics = {
        "bbox_center_error": {
            "average": _average(center_distances),
            "count": len(center_distances),
            "total": round(sum(center_distances), 6),
        },
        "object_label_accuracy": {
            "count": len(label_matches),
            "rate": _rate(len(label_matches), len(object_matches)),
            "total": len(object_matches),
        },
        "object_confidence_weighted_f1": {
            "rate": _f1_from_rates(
                object_weighted_precision["rate"],
                object_weighted_recall["rate"],
            )
        },
        "object_confidence_weighted_precision": object_weighted_precision,
        "object_confidence_weighted_recall": object_weighted_recall,
        "object_precision": {
            "count": len(object_matches),
            "rate": _rate(len(object_matches), len(predicted_object_ids)),
            "total": len(predicted_object_ids),
        },
        "object_recall": {
            "count": len(object_matches),
            "rate": _rate(len(object_matches), len(oracle_object_ids)),
            "total": len(oracle_object_ids),
        },
        "unlocated_object_count": {
            "count": len(unlocated_object_ids),
            "total": len(predicted_object_ids),
        },
        "relation_f1": {
            "rate": _f1(
                len(matched_relations),
                len(predicted_relations),
                len(oracle_relations),
            )
        },
        "relation_confidence_weighted_f1": {
            "rate": _f1_from_rates(
                relation_weighted_precision["rate"],
                relation_weighted_recall["rate"],
            )
        },
        "relation_confidence_weighted_precision": relation_weighted_precision,
        "relation_confidence_weighted_recall": relation_weighted_recall,
        "relation_precision": {
            "count": len(matched_relations),
            "rate": _rate(len(matched_relations), len(predicted_relations)),
            "total": len(predicted_relations),
        },
        "relation_recall": {
            "count": len(matched_relations),
            "rate": _rate(len(matched_relations), len(oracle_relations)),
            "total": len(oracle_relations),
        },
        "state_accuracy": {
            "count": len(state_matches),
            "mode": "matched_object_state",
            "rate": _rate(len(state_matches), len(object_matches)),
            "total": len(object_matches),
        },
    }
    return {
        "schema_version": "dsg-spatialqa-lab.graph-comparison.v1",
        "matching": {
            "center_distance_threshold": center_distance_threshold,
            "strategy": matching,
        },
        "summary": {
            "oracle_object_count": len(oracle_object_ids),
            "predicted_object_count": len(predicted_object_ids),
            "matched_object_count": len(object_matches),
            "predicted_unlocated_object_count": len(unlocated_object_ids),
            "oracle_relation_count": len(oracle_relations),
            "predicted_relation_count": len(predicted_relations),
            "matched_relation_count": len(matched_relations),
        },
        "metrics": metrics,
        "diagnostics": diagnostics,
        "breakdown": {
            "by_object_label": _object_label_breakdown(
                oracle_graph,
                predicted_graph,
                object_matches,
            ),
            "by_prediction_source": _prediction_source_breakdown(
                predicted_graph,
                object_matches,
                predicted_relations,
                matched_relations,
            ),
            "by_relation": _relation_breakdown(
                oracle_relations,
                predicted_relations,
                matched_relations,
            ),
        },
        "differences": {
            "missing_object_ids": missing_object_ids,
            "extra_object_ids": extra_object_ids,
            "unlocated_object_ids": unlocated_object_ids,
            "object_matches": _object_match_dicts(oracle_graph, predicted_graph, object_matches),
            "missing_relations": [_relation_dict(oracle_relations[key]) for key in missing_relations],
            "extra_relations": [_relation_dict(predicted_relations[key]) for key in extra_relations],
        },
    }


def graph_eval_report(
    oracle_graph: DynamicSceneGraph,
    predicted_graph: DynamicSceneGraph,
    *,
    oracle_path: str | Path | None = None,
    predicted_path: str | Path | None = None,
    matching: GraphMatchingStrategy = "exact",
    center_distance_threshold: float = DEFAULT_CENTER_DISTANCE_THRESHOLD,
) -> dict[str, Any]:
    comparison = compare_graphs(
        oracle_graph,
        predicted_graph,
        matching=matching,
        center_distance_threshold=center_distance_threshold,
    )
    report: dict[str, Any] = {
        "schema_version": GRAPH_EVAL_REPORT_SCHEMA_VERSION,
        "oracle_path": str(oracle_path) if oracle_path is not None else None,
        "predicted_path": str(predicted_path) if predicted_path is not None else None,
        "oracle_digest": graph_json_digest(oracle_graph),
        "predicted_digest": graph_json_digest(predicted_graph),
        "summary": comparison["summary"],
        "metrics": comparison["metrics"],
        "diagnostics": comparison["diagnostics"],
        "breakdown": comparison["breakdown"],
        "differences": comparison["differences"],
    }
    if matching != "exact" or center_distance_threshold != DEFAULT_CENTER_DISTANCE_THRESHOLD:
        report["matching"] = comparison["matching"]
    report["report_digest"] = graph_eval_report_digest(report)
    return report


def graph_eval_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def graph_eval_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_graph_eval_report(report: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(graph_eval_report_json(report), encoding="utf-8")
    return output_path


def load_graph_eval_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Graph eval report JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_graph_eval_report(report: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = report.get("schema_version")
    report_digest = _string_or_none(report.get("report_digest"))
    expected_report_digest = graph_eval_report_digest(report)
    summary = report.get("summary")
    metrics = report.get("metrics")
    diagnostics = report.get("diagnostics")
    breakdown = report.get("breakdown")
    matched_object_count = _mapping_value(summary, "matched_object_count")
    matched_relation_count = _mapping_value(summary, "matched_relation_count")
    predicted_object_count = _mapping_value(summary, "predicted_object_count")
    predicted_relation_count = _mapping_value(summary, "predicted_relation_count")
    prediction_source_breakdown = _mapping_value(breakdown, "by_prediction_source")
    prediction_source_objects = _mapping_value(prediction_source_breakdown, "objects")
    prediction_source_relations = _mapping_value(prediction_source_breakdown, "relations")
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == GRAPH_EVAL_REPORT_SCHEMA_VERSION,
            "expected": GRAPH_EVAL_REPORT_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "report_digest",
            "passed": report_digest == expected_report_digest,
            "expected": expected_report_digest,
            "actual": report_digest,
        },
        {
            "name": "object_precision_count",
            "passed": _metric_value(metrics, "object_precision", "count") == matched_object_count,
            "expected": matched_object_count,
            "actual": _metric_value(metrics, "object_precision", "count"),
        },
        {
            "name": "relation_precision_count",
            "passed": _metric_value(metrics, "relation_precision", "count")
            == matched_relation_count,
            "expected": matched_relation_count,
            "actual": _metric_value(metrics, "relation_precision", "count"),
        },
        _weighted_f1_check(metrics, "object_confidence_weighted_f1"),
        _weighted_f1_check(metrics, "relation_confidence_weighted_f1"),
        {
            "name": "prediction_source_object_count",
            "passed": _source_breakdown_total(prediction_source_objects, "predicted_count")
            == predicted_object_count,
            "expected": predicted_object_count,
            "actual": _source_breakdown_total(prediction_source_objects, "predicted_count"),
        },
        {
            "name": "prediction_source_object_match_count",
            "passed": _source_breakdown_total(prediction_source_objects, "matched_count")
            == matched_object_count,
            "expected": matched_object_count,
            "actual": _source_breakdown_total(prediction_source_objects, "matched_count"),
        },
        {
            "name": "prediction_source_relation_count",
            "passed": _source_breakdown_total(prediction_source_relations, "predicted_count")
            == predicted_relation_count,
            "expected": predicted_relation_count,
            "actual": _source_breakdown_total(prediction_source_relations, "predicted_count"),
        },
        {
            "name": "prediction_source_relation_match_count",
            "passed": _source_breakdown_total(prediction_source_relations, "matched_count")
            == matched_relation_count,
            "expected": matched_relation_count,
            "actual": _source_breakdown_total(prediction_source_relations, "matched_count"),
        },
        {
            "name": "duplicate_track_count",
            "passed": _mapping_value(diagnostics, "duplicate_track_count")
            == _sequence_length(_mapping_value(diagnostics, "duplicate_tracks")),
            "expected": _sequence_length(_mapping_value(diagnostics, "duplicate_tracks")),
            "actual": _mapping_value(diagnostics, "duplicate_track_count"),
        },
        {
            "name": "id_fragmentation_count",
            "passed": _mapping_value(diagnostics, "id_fragmentation_count")
            == _sequence_length(_mapping_value(diagnostics, "id_fragmentation")),
            "expected": _sequence_length(_mapping_value(diagnostics, "id_fragmentation")),
            "actual": _mapping_value(diagnostics, "id_fragmentation_count"),
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "report_digest": report_digest,
        "checks": checks,
    }


def compare_graph_eval_report(report: Mapping[str, Any]) -> dict[str, Any]:
    oracle_path = _required_report_path(report, "oracle_path")
    predicted_path = _required_report_path(report, "predicted_path")
    current_report = graph_eval_report(
        load_graph_json(oracle_path),
        load_graph_json(predicted_path),
        oracle_path=oracle_path,
        predicted_path=predicted_path,
        **_report_matching_kwargs(report),
    )
    validation = validate_graph_eval_report(report)
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
        _equality_check(
            "diagnostics_match_current",
            report.get("diagnostics"),
            current_report["diagnostics"],
        ),
        _equality_check("breakdown_matches_current", report.get("breakdown"), current_report["breakdown"]),
        _equality_check(
            "differences_match_current",
            report.get("differences"),
            current_report["differences"],
        ),
    ]
    return {
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def _relation_mapping(
    graph: DynamicSceneGraph,
    *,
    predicted_to_oracle_id: Mapping[str, str] | None = None,
) -> dict[tuple[str, str, str, str, int], Edge]:
    object_id_mapping = predicted_to_oracle_id or {}
    return {
        _relation_key(edge, object_id_mapping=object_id_mapping): edge for edge in graph.edges
    }


def _relation_key(
    edge: Edge,
    *,
    object_id_mapping: Mapping[str, str] | None = None,
) -> tuple[str, str, str, str, int]:
    mapping = object_id_mapping or {}
    return (
        _mapped_node_id(edge.src, mapping),
        edge.relation,
        _mapped_node_id(edge.dst, mapping),
        edge.reference_frame,
        edge.step,
    )


def _unlocated_object_ids(graph: DynamicSceneGraph) -> list[str]:
    located_sources = {
        edge.src for edge in graph.edges if edge.relation in CONTAINMENT_RELATIONS
    }
    object_ids: list[str] = []
    for object_id in graph.object_states:
        node = graph.nodes.get(object_id)
        attributes = node.attributes if node is not None else {}
        has_location_attribute = bool(
            attributes.get("current_location_id")
            or attributes.get("current_room_id")
        )
        if object_id not in located_sources and not has_location_attribute:
            object_ids.append(object_id)
    return sorted(object_ids)


def _mapped_node_id(node_id: str, object_id_mapping: Mapping[str, str]) -> str:
    if node_id in object_id_mapping:
        return object_id_mapping[node_id]
    if node_id.startswith("state:"):
        parts = node_id.split(":")
        if len(parts) == 3 and parts[1] in object_id_mapping:
            return f"state:{object_id_mapping[parts[1]]}:{parts[2]}"
    return node_id


def _object_matches(
    oracle_graph: DynamicSceneGraph,
    predicted_graph: DynamicSceneGraph,
    *,
    matching: GraphMatchingStrategy,
    center_distance_threshold: float,
) -> list[dict[str, str]]:
    if matching == "exact":
        return [
            {"oracle_object_id": object_id, "predicted_object_id": object_id}
            for object_id in sorted(set(oracle_graph.object_states) & set(predicted_graph.object_states))
        ]

    unmatched_predicted_ids = set(predicted_graph.object_states)
    matches: list[dict[str, str]] = []
    for oracle_id in sorted(oracle_graph.object_states):
        oracle_state = oracle_graph.object_states[oracle_id]
        candidates: list[tuple[float, str]] = []
        for predicted_id in sorted(unmatched_predicted_ids):
            predicted_state = predicted_graph.object_states[predicted_id]
            if oracle_state.label != predicted_state.label:
                continue
            if matching == "label_center_room" and _object_room(oracle_graph, oracle_id) != _object_room(
                predicted_graph,
                predicted_id,
            ):
                continue
            distance = oracle_state.pose.distance_to(predicted_state.pose)
            if distance <= center_distance_threshold:
                candidates.append((distance, predicted_id))
        if not candidates:
            continue
        _, predicted_id = min(candidates, key=lambda item: (round(item[0], 12), item[1]))
        unmatched_predicted_ids.remove(predicted_id)
        matches.append({"oracle_object_id": oracle_id, "predicted_object_id": predicted_id})
    return matches


def _object_match_dicts(
    oracle_graph: DynamicSceneGraph,
    predicted_graph: DynamicSceneGraph,
    object_matches: list[dict[str, str]],
) -> list[dict[str, Any]]:
    return [
        {
            "center_distance": round(
                oracle_graph.object_states[match["oracle_object_id"]].pose.distance_to(
                    predicted_graph.object_states[match["predicted_object_id"]].pose
                ),
                6,
            ),
            "label": oracle_graph.object_states[match["oracle_object_id"]].label,
            "oracle_object_id": match["oracle_object_id"],
            "predicted_object_id": match["predicted_object_id"],
        }
        for match in object_matches
    ]


def _matching_diagnostics(
    oracle_graph: DynamicSceneGraph,
    predicted_graph: DynamicSceneGraph,
    object_matches: list[dict[str, str]],
    *,
    matching: GraphMatchingStrategy,
    center_distance_threshold: float,
) -> dict[str, Any]:
    if matching not in {"label_center", "label_center_room"}:
        return {
            "duplicate_track_count": 0,
            "duplicate_tracks": [],
            "id_fragmentation": [],
            "id_fragmentation_count": 0,
        }
    matched_by_oracle = {
        match["oracle_object_id"]: match["predicted_object_id"] for match in object_matches
    }
    duplicate_tracks: list[dict[str, Any]] = []
    fragmented_objects: list[dict[str, Any]] = []
    for oracle_id in sorted(oracle_graph.object_states):
        oracle_state = oracle_graph.object_states[oracle_id]
        candidates: list[dict[str, Any]] = []
        for predicted_id in sorted(predicted_graph.object_states):
            predicted_state = predicted_graph.object_states[predicted_id]
            if oracle_state.label != predicted_state.label:
                continue
            if matching == "label_center_room" and _object_room(oracle_graph, oracle_id) != _object_room(
                predicted_graph,
                predicted_id,
            ):
                continue
            center_distance = round(oracle_state.pose.distance_to(predicted_state.pose), 6)
            if center_distance <= center_distance_threshold:
                candidates.append(
                    {
                        "center_distance": center_distance,
                        "predicted_object_id": predicted_id,
                    }
                )
        if len(candidates) <= 1:
            continue
        matched_predicted_id = matched_by_oracle.get(oracle_id)
        extra_predicted_ids = [
            candidate["predicted_object_id"]
            for candidate in candidates
            if candidate["predicted_object_id"] != matched_predicted_id
        ]
        fragmented_objects.append(
            {
                "candidate_count": len(candidates),
                "candidate_predicted_object_ids": [
                    candidate["predicted_object_id"] for candidate in candidates
                ],
                "extra_predicted_object_ids": extra_predicted_ids,
                "label": oracle_state.label,
                "matched_predicted_object_id": matched_predicted_id,
                "oracle_object_id": oracle_id,
            }
        )
        for candidate in candidates:
            predicted_id = candidate["predicted_object_id"]
            if predicted_id == matched_predicted_id:
                continue
            duplicate_tracks.append(
                {
                    "center_distance": candidate["center_distance"],
                    "label": oracle_state.label,
                    "matched_predicted_object_id": matched_predicted_id,
                    "oracle_object_id": oracle_id,
                    "predicted_object_id": predicted_id,
                }
            )
    duplicate_tracks.sort(
        key=lambda item: (
            str(item["oracle_object_id"]),
            str(item["predicted_object_id"]),
        )
    )
    return {
        "duplicate_track_count": len(duplicate_tracks),
        "duplicate_tracks": duplicate_tracks,
        "id_fragmentation": fragmented_objects,
        "id_fragmentation_count": len(fragmented_objects),
    }


def _object_room(graph: DynamicSceneGraph, object_id: str) -> str | None:
    containment_edges = _latest_containment_edges(graph)
    visited: set[str] = set()
    current_id = object_id
    while current_id not in visited:
        visited.add(current_id)
        edge = containment_edges.get(current_id)
        if edge is None:
            return None
        destination = graph.nodes.get(edge.dst)
        if destination is not None and destination.type == "room":
            return edge.dst
        current_id = edge.dst
    return None


def _latest_containment_edges(graph: DynamicSceneGraph) -> dict[str, Edge]:
    locations: dict[str, Edge] = {}
    for edge in graph.edges:
        if edge.relation not in CONTAINMENT_RELATIONS:
            continue
        current = locations.get(edge.src)
        if current is None or _edge_sort_key(current) < _edge_sort_key(edge):
            locations[edge.src] = edge
    return locations


def _edge_sort_key(edge: Edge) -> tuple[int, str, str, str, str]:
    return (edge.step, edge.src, edge.relation, edge.dst, edge.reference_frame)


def _object_state_matches(oracle: ObjectState, predicted: ObjectState) -> bool:
    return (
        oracle.label == predicted.label
        and oracle.visible is predicted.visible
        and oracle.pose.almost_equals(predicted.pose)
    )


def _object_label_breakdown(
    oracle_graph: DynamicSceneGraph,
    predicted_graph: DynamicSceneGraph,
    object_matches: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    labels = sorted(
        {state.label for state in oracle_graph.object_states.values()}
        | {state.label for state in predicted_graph.object_states.values()}
    )
    breakdown: dict[str, dict[str, Any]] = {}
    for label in labels:
        oracle_count = sum(1 for state in oracle_graph.object_states.values() if state.label == label)
        predicted_count = sum(
            1 for state in predicted_graph.object_states.values() if state.label == label
        )
        matched_count = sum(
            1
            for match in object_matches
            if oracle_graph.object_states[match["oracle_object_id"]].label == label
            and predicted_graph.object_states[match["predicted_object_id"]].label == label
        )
        breakdown[label] = {
            "matched_count": matched_count,
            "oracle_count": oracle_count,
            "predicted_count": predicted_count,
            "precision": _rate(matched_count, predicted_count),
            "recall": _rate(matched_count, oracle_count),
        }
    return breakdown


def _relation_breakdown(
    oracle_relations: Mapping[tuple[str, str, str, str, int], Edge],
    predicted_relations: Mapping[tuple[str, str, str, str, int], Edge],
    matched_relation_keys: list[tuple[str, str, str, str, int]],
) -> dict[str, dict[str, Any]]:
    relation_names = sorted(
        {edge.relation for edge in oracle_relations.values()}
        | {edge.relation for edge in predicted_relations.values()}
    )
    matched_key_set = set(matched_relation_keys)
    breakdown: dict[str, dict[str, Any]] = {}
    for relation in relation_names:
        oracle_count = sum(1 for edge in oracle_relations.values() if edge.relation == relation)
        predicted_count = sum(
            1 for edge in predicted_relations.values() if edge.relation == relation
        )
        matched_count = sum(
            1
            for key in matched_key_set
            if oracle_relations[key].relation == relation
        )
        breakdown[relation] = {
            "f1": _f1(matched_count, predicted_count, oracle_count),
            "matched_count": matched_count,
            "oracle_count": oracle_count,
            "predicted_count": predicted_count,
            "precision": _rate(matched_count, predicted_count),
            "recall": _rate(matched_count, oracle_count),
        }
    return breakdown


def _prediction_source_breakdown(
    predicted_graph: DynamicSceneGraph,
    object_matches: list[dict[str, str]],
    predicted_relations: Mapping[tuple[str, str, str, str, int], Edge],
    matched_relation_keys: list[tuple[str, str, str, str, int]],
) -> dict[str, dict[str, dict[str, Any]]]:
    matched_predicted_object_ids = {
        match["predicted_object_id"] for match in object_matches
    }
    matched_relation_key_set = set(matched_relation_keys)
    return {
        "objects": _prediction_object_source_breakdown(
            predicted_graph,
            matched_predicted_object_ids,
        ),
        "relations": _prediction_relation_source_breakdown(
            predicted_relations,
            matched_relation_key_set,
        ),
    }


def _prediction_object_source_breakdown(
    predicted_graph: DynamicSceneGraph,
    matched_predicted_object_ids: set[str],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, float | int]] = {}
    for object_id in sorted(predicted_graph.object_states):
        state = predicted_graph.object_states[object_id]
        source = _object_prediction_source(predicted_graph, object_id)
        entry = _source_entry(grouped, source)
        entry["predicted_count"] += 1
        entry["total_weight"] += state.confidence
        if object_id in matched_predicted_object_ids:
            entry["matched_count"] += 1
            entry["matched_weight"] += state.confidence
    return _finalize_source_breakdown(grouped)


def _prediction_relation_source_breakdown(
    predicted_relations: Mapping[tuple[str, str, str, str, int], Edge],
    matched_relation_keys: set[tuple[str, str, str, str, int]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, float | int]] = {}
    for key in sorted(predicted_relations):
        edge = predicted_relations[key]
        source = _prediction_source(edge.attributes)
        entry = _source_entry(grouped, source)
        entry["predicted_count"] += 1
        entry["total_weight"] += edge.confidence
        if key in matched_relation_keys:
            entry["matched_count"] += 1
            entry["matched_weight"] += edge.confidence
    return _finalize_source_breakdown(grouped)


def _source_entry(
    grouped: dict[str, dict[str, float | int]],
    source: str,
) -> dict[str, float | int]:
    entry = grouped.get(source)
    if entry is None:
        entry = {
            "matched_count": 0,
            "matched_weight": 0.0,
            "predicted_count": 0,
            "total_weight": 0.0,
        }
        grouped[source] = entry
    return entry


def _finalize_source_breakdown(
    grouped: Mapping[str, Mapping[str, float | int]],
) -> dict[str, dict[str, Any]]:
    finalized: dict[str, dict[str, Any]] = {}
    for source in sorted(grouped):
        entry = grouped[source]
        matched_count = int(entry["matched_count"])
        predicted_count = int(entry["predicted_count"])
        matched_weight = round(float(entry["matched_weight"]), 6)
        total_weight = round(float(entry["total_weight"]), 6)
        finalized[source] = {
            "confidence_weighted_precision": _weighted_rate(
                matched_weight,
                total_weight,
            ),
            "matched_count": matched_count,
            "matched_weight": matched_weight,
            "precision": _rate(matched_count, predicted_count),
            "predicted_count": predicted_count,
            "total_weight": total_weight,
        }
    return finalized


def _object_prediction_source(graph: DynamicSceneGraph, object_id: str) -> str:
    node = graph.nodes.get(object_id)
    if node is None:
        return "unknown"
    return _prediction_source(node.attributes)


def _prediction_source(attributes: Mapping[str, Any]) -> str:
    for key in ("source", "source_name", "source_kind"):
        value = attributes.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _relation_dict(edge: Edge) -> dict[str, Any]:
    return {
        "src": edge.src,
        "relation": edge.relation,
        "dst": edge.dst,
        "reference_frame": edge.reference_frame,
        "step": edge.step,
    }


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(count / total, 6)


def _f1(matched_count: int, predicted_total: int, oracle_total: int) -> float:
    precision = _rate(matched_count, predicted_total)
    recall = _rate(matched_count, oracle_total)
    return _f1_from_rates(precision, recall)


def _f1_from_rates(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return round((2.0 * precision * recall) / (precision + recall), 6)


def _confidence_weighted_metric(
    matched_weights: Iterable[float],
    total_weights: Iterable[float],
) -> dict[str, float]:
    matched_weight = round(sum(matched_weights), 6)
    total_weight = round(sum(total_weights), 6)
    return {
        "matched_weight": matched_weight,
        "rate": _weighted_rate(matched_weight, total_weight),
        "total_weight": total_weight,
    }


def _weighted_rate(matched_weight: float, total_weight: float) -> float:
    if total_weight == 0.0:
        return 0.0
    return round(matched_weight / total_weight, 6)


def _average(values: Iterable[float]) -> float:
    numbers = list(values)
    if not numbers:
        return 0.0
    return round(sum(numbers) / len(numbers), 6)


def _required_report_path(report: Mapping[str, Any], key: str) -> Path:
    value = report.get(key)
    if not isinstance(value, str) or not value:
        raise SpatialQAError(f"Graph eval report missing {key}")
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


def _mapping_value(payload: object, key: str) -> object:
    if not isinstance(payload, Mapping):
        return None
    return payload.get(key)


def _metric_value(metrics: object, metric_name: str, key: str) -> object:
    if not isinstance(metrics, Mapping):
        return None
    metric = metrics.get(metric_name)
    if not isinstance(metric, Mapping):
        return None
    return metric.get(key)


def _weighted_f1_check(metrics: object, f1_metric_name: str) -> dict[str, Any]:
    prefix = f1_metric_name.removesuffix("_f1")
    precision = _number_or_none(_metric_value(metrics, f"{prefix}_precision", "rate"))
    recall = _number_or_none(_metric_value(metrics, f"{prefix}_recall", "rate"))
    expected = (
        _f1_from_rates(precision, recall)
        if precision is not None and recall is not None
        else None
    )
    actual = _metric_value(metrics, f1_metric_name, "rate")
    return {
        "name": f1_metric_name,
        "passed": actual == expected,
        "expected": expected,
        "actual": actual,
    }


def _source_breakdown_total(breakdown: object, key: str) -> int | None:
    if not isinstance(breakdown, Mapping):
        return None
    total = 0
    for source_breakdown in breakdown.values():
        if not isinstance(source_breakdown, Mapping):
            return None
        value = source_breakdown.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        total += value
    return total


def _number_or_none(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _sequence_length(value: object) -> int | None:
    if not isinstance(value, list):
        return None
    return len(value)


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _validate_matching(
    matching: str,
    center_distance_threshold: float,
) -> None:
    if matching not in {"exact", "label_center", "label_center_room"}:
        raise SpatialQAError(f"Unsupported graph matching strategy: {matching}")
    if (
        not isinstance(center_distance_threshold, (int, float))
        or isinstance(center_distance_threshold, bool)
        or center_distance_threshold < 0.0
    ):
        raise SpatialQAError("center_distance_threshold must be non-negative")


def _report_matching_kwargs(report: Mapping[str, Any]) -> dict[str, Any]:
    matching_payload = report.get("matching")
    if matching_payload is None:
        return {}
    if not isinstance(matching_payload, Mapping):
        raise SpatialQAError("Graph eval report matching must be an object")
    strategy = matching_payload.get("strategy", "exact")
    if strategy not in {"exact", "label_center", "label_center_room"}:
        raise SpatialQAError(f"Unsupported graph matching strategy: {strategy}")
    threshold = matching_payload.get(
        "center_distance_threshold",
        DEFAULT_CENTER_DISTANCE_THRESHOLD,
    )
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise SpatialQAError("center_distance_threshold must be non-negative")
    return {
        "center_distance_threshold": float(threshold),
        "matching": cast(GraphMatchingStrategy, strategy),
    }
