from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.episodes import (
    EpisodeFrame,
    episode_sequence_digest,
    episode_sequence_summary,
    load_episode_sequence,
    validate_episode_sequence,
)
from dsg_spatialqa_lab.memory import CONTAINMENT_RELATIONS, DynamicSceneGraph
from dsg_spatialqa_lab.scene_io import (
    graph_json_digest,
    graph_report,
    graph_summary,
    load_graph_json,
    validate_graph_report,
)
from dsg_spatialqa_lab.schema import BBox3D, Edge, Pose3D, SpatialQAError


ORACLE_GRAPH_REPORT_SCHEMA_VERSION = "dsg-spatialqa-lab.oracle-graph-report.v1"


@dataclass(frozen=True)
class OracleObjectRecord:
    object_id: str
    label: str
    pose: Pose3D
    bbox: BBox3D
    confidence: float
    visible: bool
    room_id: str | None = None
    region_id: str | None = None
    states: Mapping[str, Any] = field(default_factory=dict)
    attributes: Mapping[str, Any] = field(default_factory=dict)


def build_oracle_graph_from_episode(frames: Sequence[EpisodeFrame]) -> DynamicSceneGraph:
    validation = validate_episode_sequence(frames)
    if validation["valid"] is not True:
        raise SpatialQAError("Episode sequence must be ordered and contain unique episode steps")

    graph = DynamicSceneGraph()
    for frame in frames:
        action_id = _add_action_node(graph, frame)
        graph.set_agent_pose(frame.agent_id, frame.agent_pose, step=frame.step)
        _add_rooms(graph, frame)
        _add_regions(graph, frame)
        _add_objects(graph, frame, action_id)
        _add_explicit_relations(graph, frame)
    return graph


def oracle_graph_summary(
    graph: DynamicSceneGraph,
    frames: Sequence[EpisodeFrame],
) -> dict[str, Any]:
    object_records = [
        record
        for frame in frames
        for record in _object_records_from_frame(frame)
    ]
    room_record_count = sum(
        len(_metadata_sequence(frame.metadata, "rooms")) for frame in frames
    )
    region_record_count = sum(
        len(_metadata_sequence(frame.metadata, "regions")) for frame in frames
    )
    explicit_relation_count = sum(
        len(_metadata_sequence(frame.metadata, "relations")) for frame in frames
    )
    return {
        "graph_summary": graph_summary(graph),
        "episode_summary": episode_sequence_summary(frames),
        "oracle_summary": {
            "action_node_count": sum(
                1 for node in graph.nodes.values() if node.type == "action"
            ),
            "event_node_count": sum(
                1 for node in graph.nodes.values() if node.type == "event"
            ),
            "explicit_relation_count": explicit_relation_count,
            "frame_count": len(frames),
            "moved_edge_count": sum(
                1 for edge in graph.edges if edge.relation in {"MOVED_FROM", "MOVED_TO"}
            ),
            "object_record_count": len(object_records),
            "region_record_count": region_record_count,
            "room_record_count": room_record_count,
            "state_node_count": sum(
                1 for node in graph.nodes.values() if node.type == "state"
            ),
            "unique_object_count": len({record.object_id for record in object_records}),
            "by_object_state_key": _sorted_counts(
                key for record in object_records for key in record.states
            ),
        },
    }


def oracle_graph_report(
    *,
    input_path: str | Path,
    graph_path: str | Path,
    graph: DynamicSceneGraph,
    frames: Sequence[EpisodeFrame],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": ORACLE_GRAPH_REPORT_SCHEMA_VERSION,
        "action": "build_oracle_graph",
        "path": str(input_path),
        "graph_path": str(graph_path),
        "valid": True,
        "episode_digest": episode_sequence_digest(frames),
        "summary": oracle_graph_summary(graph, frames),
        "graph_report": graph_report(
            graph,
            action="build_oracle_graph",
            graph_path=graph_path,
        ),
    }
    report["digest"] = oracle_graph_report_digest(report)
    return report


def oracle_graph_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def oracle_graph_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_oracle_graph_report(report: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(oracle_graph_report_json(report), encoding="utf-8")
    return output_path


def load_oracle_graph_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SpatialQAError("Oracle graph report JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_oracle_graph_report(report: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = report.get("schema_version")
    action = report.get("action")
    valid = report.get("valid")
    digest = report.get("digest")
    expected_digest = oracle_graph_report_digest(report)
    input_path = report.get("path")
    graph_path = report.get("graph_path")
    graph_report_value = report.get("graph_report")
    graph_report_path = (
        graph_report_value.get("path") if isinstance(graph_report_value, Mapping) else None
    )
    graph_report_valid = (
        validate_graph_report(_as_mapping(graph_report_value, "graph_report"))["valid"]
        is True
        if isinstance(graph_report_value, Mapping)
        else False
    )
    summary_value = report.get("summary")
    summary = summary_value if isinstance(summary_value, Mapping) else {}
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == ORACLE_GRAPH_REPORT_SCHEMA_VERSION,
            "expected": ORACLE_GRAPH_REPORT_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "action",
            "passed": action == "build_oracle_graph",
            "expected": "build_oracle_graph",
            "actual": action,
        },
        {
            "name": "report_valid",
            "passed": valid is True,
            "expected": True,
            "actual": valid,
        },
        {
            "name": "report_digest",
            "passed": digest == expected_digest,
            "expected": expected_digest,
            "actual": digest,
        },
        {
            "name": "input_path_present",
            "passed": _is_non_empty_string(input_path),
            "expected": "non-empty explicit local path",
            "actual": input_path,
        },
        {
            "name": "graph_path_present",
            "passed": _is_non_empty_string(graph_path),
            "expected": "non-empty explicit local path",
            "actual": graph_path,
        },
        {
            "name": "episode_digest_format",
            "passed": _is_sha256_hexdigest(report.get("episode_digest")),
            "expected": "64 lowercase sha256 hex characters",
            "actual": report.get("episode_digest"),
        },
        {
            "name": "graph_report_path_matches",
            "passed": _is_non_empty_string(graph_path) and graph_path == graph_report_path,
            "expected": graph_path,
            "actual": graph_report_path,
        },
        {
            "name": "graph_report_valid",
            "passed": graph_report_valid,
        },
        {
            "name": "summary_shape",
            "passed": all(
                key in summary
                for key in ("graph_summary", "episode_summary", "oracle_summary")
            ),
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "digest": digest,
        "checks": checks,
    }


def compare_oracle_graph_report(report: Mapping[str, Any]) -> dict[str, Any]:
    input_path = _required_str(report, "path")
    graph_path = _required_str(report, "graph_path")
    frames = load_episode_sequence(input_path)
    current_graph = build_oracle_graph_from_episode(frames)
    saved_episode_digest = report.get("episode_digest")
    current_episode_digest = episode_sequence_digest(frames)
    saved_graph_report = _as_mapping(report.get("graph_report"), "graph_report")
    saved_graph_digest = saved_graph_report.get("digest")
    current_graph_digest = graph_json_digest(current_graph)
    saved_summary = report.get("summary")
    current_summary = oracle_graph_summary(current_graph, frames)
    graph_file = load_graph_json(graph_path)
    graph_file_digest = graph_json_digest(graph_file)
    graph_file_summary = graph_summary(graph_file)
    saved_graph_summary = saved_graph_report.get("summary")
    oracle_summary_differences = _nested_differences(saved_summary, current_summary)
    graph_file_summary_differences = _nested_differences(
        saved_graph_summary,
        graph_file_summary,
    )
    checks: list[dict[str, Any]] = [
        {
            "name": "saved_report_valid",
            "passed": validate_oracle_graph_report(report)["valid"] is True,
        },
        {
            "name": "episode_digest_matches_current",
            "passed": saved_episode_digest == current_episode_digest,
            "expected": saved_episode_digest,
            "actual": current_episode_digest,
        },
        {
            "name": "graph_digest_matches_current",
            "passed": saved_graph_digest == current_graph_digest,
            "expected": saved_graph_digest,
            "actual": current_graph_digest,
        },
        {
            "name": "oracle_summary_matches_current",
            "passed": saved_summary == current_summary,
            "expected": saved_summary,
            "actual": current_summary,
        },
        {
            "name": "graph_file_digest_matches_report",
            "passed": saved_graph_digest == graph_file_digest,
            "expected": saved_graph_digest,
            "actual": graph_file_digest,
        },
        {
            "name": "graph_file_summary_matches_report",
            "passed": saved_graph_summary == graph_file_summary,
            "expected": saved_graph_summary,
            "actual": graph_file_summary,
        },
    ]
    if oracle_summary_differences:
        checks[3]["differences"] = oracle_summary_differences
    if graph_file_summary_differences:
        checks[5]["differences"] = graph_file_summary_differences
    return {
        "matches": all(check["passed"] is True for check in checks),
        "episode_path": input_path,
        "graph_path": graph_path,
        "saved_episode_digest": saved_episode_digest,
        "current_episode_digest": current_episode_digest,
        "saved_digest": saved_graph_digest,
        "current_digest": current_graph_digest,
        "checks": checks,
    }


def _add_action_node(graph: DynamicSceneGraph, frame: EpisodeFrame) -> str | None:
    if frame.action is None:
        return None
    action_id = f"action:{frame.episode_id}:{frame.step}"
    graph.add_node(
        action_id,
        "action",
        label=frame.action,
        attributes={
            "episode_id": frame.episode_id,
            "scene_id": frame.scene_id,
            "agent_id": frame.agent_id,
            "step": frame.step,
        },
    )
    return action_id


def _add_rooms(graph: DynamicSceneGraph, frame: EpisodeFrame) -> None:
    for room in _metadata_sequence(frame.metadata, "rooms"):
        graph.add_room(
            _required_str(room, "room_id"),
            _required_str(room, "label"),
            step=frame.step,
            attributes=dict(_optional_mapping(room, "attributes")),
        )


def _add_regions(graph: DynamicSceneGraph, frame: EpisodeFrame) -> None:
    evidence = [_frame_evidence(frame)]
    for region in _metadata_sequence(frame.metadata, "regions"):
        region_id = _required_str(region, "region_id")
        graph.add_region(
            region_id,
            _required_str(region, "label"),
            step=frame.step,
            attributes=dict(_optional_mapping(region, "attributes")),
        )
        room_id = _optional_str(region, "room_id")
        if room_id is not None:
            graph.add_edge(
                region_id,
                "IN_ROOM",
                room_id,
                "world",
                1.0,
                step=frame.step,
                evidence=evidence,
            )


def _add_objects(
    graph: DynamicSceneGraph,
    frame: EpisodeFrame,
    action_id: str | None,
) -> None:
    for record in _object_records_from_frame(frame):
        previous_state = graph.object_states.get(record.object_id)
        previous_parent = _latest_containment_edge(graph, record.object_id)
        moved = (
            previous_state is not None
            and (
                not previous_state.pose.almost_equals(record.pose)
                or _parent_destination(previous_parent) != _record_destination(record)
            )
        )
        graph.upsert_object(
            record.object_id,
            record.label,
            record.pose,
            record.bbox,
            confidence=record.confidence,
            visible=record.visible,
            step=frame.step,
            attributes=_object_attributes(record),
        )
        if moved:
            _add_move_edges(graph, frame, record, previous_parent, action_id)
        evidence = [_frame_evidence(frame)]
        if record.region_id is not None:
            graph.add_edge(
                record.object_id,
                "IN_REGION",
                record.region_id,
                "world",
                record.confidence,
                step=frame.step,
                evidence=evidence,
            )
        if record.room_id is not None:
            graph.add_edge(
                record.object_id,
                "IN_ROOM",
                record.room_id,
                "world",
                record.confidence,
                step=frame.step,
                evidence=evidence,
            )


def _add_move_edges(
    graph: DynamicSceneGraph,
    frame: EpisodeFrame,
    record: OracleObjectRecord,
    previous_parent: Edge | None,
    action_id: str | None,
) -> None:
    event_id = f"event:{record.object_id}:move:{frame.step}"
    evidence = [_frame_evidence(frame), event_id]
    graph.add_node(
        event_id,
        "event",
        label="move_object",
        attributes={
            "object_id": record.object_id,
            "episode_id": frame.episode_id,
            "scene_id": frame.scene_id,
            "step": frame.step,
        },
    )
    if action_id is not None:
        graph.add_edge(
            action_id,
            "ACTION_CAUSED",
            event_id,
            "world",
            1.0,
            step=frame.step,
            evidence=evidence,
        )
    if previous_parent is not None:
        graph.add_edge(
            record.object_id,
            "MOVED_FROM",
            previous_parent.dst,
            previous_parent.reference_frame,
            previous_parent.confidence,
            step=frame.step,
            evidence=evidence,
            attributes={"previous_relation": previous_parent.relation},
        )
    destination = _record_destination(record)
    if destination is not None:
        graph.add_edge(
            record.object_id,
            "MOVED_TO",
            destination,
            "world",
            record.confidence,
            step=frame.step,
            evidence=evidence,
        )


def _add_explicit_relations(graph: DynamicSceneGraph, frame: EpisodeFrame) -> None:
    for relation in _metadata_sequence(frame.metadata, "relations"):
        graph.add_edge(
            _required_str(relation, "src"),
            _required_str(relation, "relation"),
            _required_str(relation, "dst"),
            _optional_str(relation, "reference_frame") or "world",
            _optional_float(relation, "confidence", default=1.0),
            step=frame.step,
            evidence=_relation_evidence(relation, frame),
            attributes=dict(_optional_mapping(relation, "attributes")),
        )


def _object_records_from_frame(frame: EpisodeFrame) -> tuple[OracleObjectRecord, ...]:
    return tuple(
        _object_record_from_mapping(item) for item in _metadata_sequence(frame.metadata, "objects")
    )


def _object_record_from_mapping(payload: Mapping[str, Any]) -> OracleObjectRecord:
    return OracleObjectRecord(
        object_id=_required_str(payload, "object_id"),
        label=_required_str(payload, "label"),
        pose=_pose_from_mapping(_as_mapping(payload.get("pose"), "pose")),
        bbox=_bbox_from_mapping(_as_mapping(payload.get("bbox"), "bbox")),
        confidence=_required_float(payload, "confidence"),
        visible=_required_bool(payload, "visible"),
        room_id=_optional_str(payload, "room_id"),
        region_id=_optional_str(payload, "region_id"),
        states=_stable_mapping(_optional_mapping(payload, "states")),
        attributes=_stable_mapping(_optional_mapping(payload, "attributes")),
    )


def _object_attributes(record: OracleObjectRecord) -> dict[str, Any]:
    attributes = dict(record.attributes)
    attributes["states"] = _stable_mapping(record.states)
    if record.room_id is not None:
        attributes["room_id"] = record.room_id
    if record.region_id is not None:
        attributes["region_id"] = record.region_id
    return attributes


def _relation_evidence(
    relation: Mapping[str, Any],
    frame: EpisodeFrame,
) -> list[str]:
    evidence = relation.get("evidence")
    if evidence is None:
        return [_frame_evidence(frame)]
    return list(_sequence_of_strings(relation, "evidence"))


def _latest_containment_edge(graph: DynamicSceneGraph, object_id: str) -> Edge | None:
    candidates = [
        edge
        for edge in graph.edges
        if edge.src == object_id and edge.relation in CONTAINMENT_RELATIONS
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda edge: (edge.step, edge.src, edge.relation, edge.dst))[-1]


def _parent_destination(edge: Edge | None) -> str | None:
    return edge.dst if edge is not None else None


def _record_destination(record: OracleObjectRecord) -> str | None:
    if record.region_id is not None:
        return record.region_id
    return record.room_id


def _frame_evidence(frame: EpisodeFrame) -> str:
    return f"episode:{frame.episode_id}:{frame.step}"


def _metadata_sequence(metadata: Mapping[str, Any], field_name: str) -> tuple[Mapping[str, Any], ...]:
    value = metadata.get(field_name, ())
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise SpatialQAError(f"{field_name} must be a sequence")
    items: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise SpatialQAError(f"{field_name} entries must be objects")
        items.append(cast(Mapping[str, Any], item))
    return tuple(items)


def _pose_from_mapping(payload: Mapping[str, Any]) -> Pose3D:
    return Pose3D(
        _required_float(payload, "x"),
        _required_float(payload, "y"),
        _required_float(payload, "z"),
        yaw=_required_float(payload, "yaw"),
    )


def _bbox_from_mapping(payload: Mapping[str, Any]) -> BBox3D:
    size = _required_sequence(payload, "size")
    if len(size) != 3:
        raise SpatialQAError("BBox size must contain exactly three values")
    return BBox3D(
        center=_pose_from_mapping(_as_mapping(payload.get("center"), "bbox center")),
        size=(
            _float_value(size[0], "size[0]"),
            _float_value(size[1], "size[1]"),
            _float_value(size[2], "size[2]"),
        ),
    )


def _as_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpatialQAError(f"{field_name} must be an object")
    return cast(Mapping[str, Any], value)


def _optional_mapping(payload: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    value = payload.get(field_name, {})
    if not isinstance(value, Mapping):
        raise SpatialQAError(f"{field_name} must be an object")
    return cast(Mapping[str, Any], value)


def _stable_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value[key] for key in sorted(value)}


def _required_str(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str):
        raise SpatialQAError(f"{field_name} must be a string")
    return value


def _optional_str(payload: Mapping[str, Any], field_name: str) -> str | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SpatialQAError(f"{field_name} must be a string or null")
    return value


def _required_float(payload: Mapping[str, Any], field_name: str) -> float:
    value = payload.get(field_name)
    return _float_value(value, field_name)


def _optional_float(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    default: float,
) -> float:
    value = payload.get(field_name)
    if value is None:
        return default
    return _float_value(value, field_name)


def _float_value(value: object, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SpatialQAError(f"{field_name} must be a number")
    return float(value)


def _required_bool(payload: Mapping[str, Any], field_name: str) -> bool:
    value = payload.get(field_name)
    if not isinstance(value, bool):
        raise SpatialQAError(f"{field_name} must be a boolean")
    return value


def _required_sequence(payload: Mapping[str, Any], field_name: str) -> Sequence[Any]:
    value = payload.get(field_name)
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise SpatialQAError(f"{field_name} must be a sequence")
    return value


def _sequence_of_strings(payload: Mapping[str, Any], field_name: str) -> tuple[str, ...]:
    value = payload.get(field_name)
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise SpatialQAError(f"{field_name} must be a sequence")
    if not all(isinstance(item, str) for item in value):
        raise SpatialQAError(f"{field_name} must contain only strings")
    return tuple(str(item) for item in value)


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and value != ""


def _is_sha256_hexdigest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sorted_counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def _nested_differences(
    expected: object,
    actual: object,
    path: str = "",
) -> list[dict[str, Any]]:
    if expected == actual:
        return []
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        differences: list[dict[str, Any]] = []
        keys = sorted(set(expected) | set(actual), key=str)
        for key in keys:
            child_path = str(key) if path == "" else f"{path}.{key}"
            if key not in expected:
                differences.append(
                    {"path": child_path, "expected": None, "actual": actual[key]},
                )
                continue
            if key not in actual:
                differences.append(
                    {"path": child_path, "expected": expected[key], "actual": None},
                )
                continue
            differences.extend(_nested_differences(expected[key], actual[key], child_path))
        return differences
    if isinstance(expected, Sequence) and not isinstance(expected, str) and isinstance(actual, Sequence) and not isinstance(actual, str):
        differences = []
        max_length = max(len(expected), len(actual))
        for index in range(max_length):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            if index >= len(expected):
                differences.append(
                    {"path": child_path, "expected": None, "actual": actual[index]},
                )
                continue
            if index >= len(actual):
                differences.append(
                    {"path": child_path, "expected": expected[index], "actual": None},
                )
                continue
            differences.extend(
                _nested_differences(expected[index], actual[index], child_path)
            )
        return differences
    return [{"path": path if path else "$", "expected": expected, "actual": actual}]
