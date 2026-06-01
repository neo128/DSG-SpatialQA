from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.memory import CONTAINMENT_RELATIONS, DynamicSceneGraph, VALID_NODE_TYPES
from dsg_spatialqa_lab.schema import (
    AgentPoseState,
    BBox3D,
    Edge,
    Node,
    NodeType,
    ObjectState,
    Pose3D,
    SpatialQAError,
)


SCHEMA_VERSION = 1
GRAPH_REPORT_SCHEMA_VERSION = "dsg-spatialqa-lab.graph-report.v1"
SUMMARY_LOW_CONFIDENCE_THRESHOLD = 0.5


def graph_to_dict(graph: DynamicSceneGraph) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "agent_poses": [
            {"agent_id": agent_id, "pose": _pose_to_dict(pose)}
            for agent_id, pose in sorted(graph.agent_poses.items())
        ],
        "agent_pose_history": {
            agent_id: [_agent_pose_state_to_dict(state) for state in states]
            for agent_id, states in sorted(graph.agent_pose_history.items())
        },
        "edges": [_edge_to_dict(edge) for edge in sorted(graph.edges, key=_edge_sort_key)],
        "nodes": [_node_to_dict(node) for node in sorted(graph.nodes.values(), key=lambda n: n.id)],
        "object_state_history": {
            object_id: [_object_state_to_dict(state) for state in states]
            for object_id, states in sorted(graph.object_state_history.items())
        },
        "object_states": [
            _object_state_to_dict(state)
            for state in sorted(graph.object_states.values(), key=lambda s: s.object_id)
        ],
    }


def graph_from_dict(data: Mapping[str, Any]) -> DynamicSceneGraph:
    schema_version = _required_int(data, "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise SpatialQAError(f"Unsupported scene schema version: {schema_version}")

    graph = DynamicSceneGraph()
    graph.nodes = {
        node.id: node
        for node in (
            _node_from_mapping(_as_mapping(item, "node"))
            for item in _required_sequence(data, "nodes")
        )
    }
    graph.edges = [
        _edge_from_mapping(_as_mapping(item, "edge")) for item in _required_sequence(data, "edges")
    ]
    graph.agent_poses = {
        _required_str(_as_mapping(item, "agent_pose"), "agent_id"): _pose_from_mapping(
            _as_mapping(_as_mapping(item, "agent_pose").get("pose"), "pose")
        )
        for item in _required_sequence(data, "agent_poses")
    }
    agent_history_mapping = _as_mapping(data.get("agent_pose_history", {}), "agent_pose_history")
    graph.agent_pose_history = defaultdict(
        list,
        {
            str(agent_id): [
                _agent_pose_state_from_mapping(_as_mapping(item, "agent_pose_history_item"))
                for item in _as_sequence(states, "agent_pose_history states")
            ]
            for agent_id, states in agent_history_mapping.items()
        },
    )
    graph.object_states = {
        state.object_id: state
        for state in (
            _object_state_from_mapping(_as_mapping(item, "object_state"))
            for item in _required_sequence(data, "object_states")
        )
    }
    history_mapping = _as_mapping(data.get("object_state_history"), "object_state_history")
    graph.object_state_history = defaultdict(
        list,
        {
            str(object_id): [
                _object_state_from_mapping(_as_mapping(item, "object_state_history_item"))
                for item in _as_sequence(states, "object_state_history states")
            ]
            for object_id, states in history_mapping.items()
        },
    )
    return graph


def graph_to_json(graph: DynamicSceneGraph) -> str:
    return json.dumps(graph_to_dict(graph), indent=2, sort_keys=True) + "\n"


def graph_json_digest(graph: DynamicSceneGraph) -> str:
    return hashlib.sha256(graph_to_json(graph).encode("utf-8")).hexdigest()


def graph_summary(graph: DynamicSceneGraph) -> dict[str, Any]:
    object_states = list(graph.object_states.values())
    current_locations = _current_object_locations(graph)
    current_rooms = _current_object_rooms(graph)
    return {
        "schema_version": SCHEMA_VERSION,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "object_count": len(graph.object_states),
        "agent_count": len(graph.agent_poses),
        "object_history_count": sum(
            len(states) for states in graph.object_state_history.values()
        ),
        "agent_history_count": sum(
            len(states) for states in graph.agent_pose_history.values()
        ),
        "visible_object_count": sum(1 for state in object_states if state.visible),
        "hidden_object_count": sum(1 for state in object_states if not state.visible),
        "low_confidence_object_count": sum(
            1
            for state in object_states
            if state.confidence < SUMMARY_LOW_CONFIDENCE_THRESHOLD
        ),
        "reobserve_candidate_count": sum(
            1
            for state in object_states
            if not state.visible and state.confidence < SUMMARY_LOW_CONFIDENCE_THRESHOLD
        ),
        "unlocated_object_count": len(graph.object_states) - len(current_locations),
        "unroomed_object_count": len(graph.object_states) - len(current_rooms),
        "by_node_type": _sorted_counts(node.type for node in graph.nodes.values()),
        "by_edge_relation": _sorted_counts(edge.relation for edge in graph.edges),
        "by_object_label": _sorted_counts(state.label for state in object_states),
        "by_current_location": _sorted_counts(current_locations.values()),
        "by_current_room": _sorted_counts(current_rooms.values()),
    }


def graph_report(
    graph: DynamicSceneGraph,
    *,
    action: str | None = None,
    graph_path: str | Path | None = None,
    fixture: str | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": GRAPH_REPORT_SCHEMA_VERSION,
        "digest": graph_json_digest(graph),
        "summary": graph_summary(graph),
    }
    if action is not None:
        report["action"] = action
    if graph_path is not None:
        report["path"] = str(graph_path)
    if fixture is not None:
        report["fixture"] = fixture
    report["report_digest"] = graph_report_digest(report)
    return report


def graph_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def graph_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_graph_report(
    graph: DynamicSceneGraph,
    path: str | Path,
    *,
    action: str | None = None,
    graph_path: str | Path | None = None,
    fixture: str | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        graph_report_json(
            graph_report(
                graph,
                action=action,
                graph_path=graph_path,
                fixture=fixture,
            )
        ),
        encoding="utf-8",
    )
    return output_path


def load_graph_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Graph report JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_graph_report(report: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = report.get("schema_version")
    digest = report.get("digest")
    report_digest = report.get("report_digest")
    expected_report_digest = graph_report_digest(report)
    summary = report.get("summary")
    summary_schema_version = (
        summary.get("schema_version") if isinstance(summary, Mapping) else None
    )
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == GRAPH_REPORT_SCHEMA_VERSION,
            "expected": GRAPH_REPORT_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "digest_format",
            "passed": _is_sha256_hexdigest(digest),
            "expected": "64 lowercase sha256 hex characters",
            "actual": digest,
        },
        {
            "name": "report_digest",
            "passed": report_digest == expected_report_digest,
            "expected": expected_report_digest,
            "actual": report_digest,
        },
        {
            "name": "summary_schema_version",
            "passed": summary_schema_version == SCHEMA_VERSION,
            "expected": SCHEMA_VERSION,
            "actual": summary_schema_version,
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "digest": digest,
        "report_digest": report_digest,
        "checks": checks,
    }


def compare_graph_report(
    report: Mapping[str, Any],
    fixture: str | None = None,
) -> dict[str, Any]:
    from dsg_spatialqa_lab.scenes import load_scene_fixture

    report_fixture = fixture
    if report_fixture is None and isinstance(report.get("fixture"), str):
        report_fixture = str(report["fixture"])

    saved_digest = report.get("digest")
    checks: list[dict[str, Any]] = [
        {
            "name": "saved_report_valid",
            "passed": validate_graph_report(report)["valid"] is True,
        }
    ]
    if report_fixture is None:
        checks.append(
            {
                "name": "fixture_available",
                "passed": False,
                "expected": "fixture",
                "actual": report.get("fixture"),
            }
        )
        return {
            "matches": False,
            "fixture": report_fixture,
            "saved_digest": saved_digest,
            "current_digest": None,
            "checks": checks,
        }

    current_graph = load_scene_fixture(report_fixture)
    current_digest = graph_json_digest(current_graph)
    saved_summary = report.get("summary")
    current_summary = graph_summary(current_graph)
    summary_differences = _nested_differences(saved_summary, current_summary)
    checks.extend(
        [
            {
                "name": "graph_digest_matches_current",
                "passed": saved_digest == current_digest,
                "expected": saved_digest,
                "actual": current_digest,
            },
            {
                "name": "summary_matches_current",
                "passed": saved_summary == current_summary,
                "expected": saved_summary,
                "actual": current_summary,
            },
        ]
    )
    if summary_differences:
        checks[-1]["differences"] = summary_differences
    return {
        "matches": all(check["passed"] is True for check in checks),
        "fixture": report_fixture,
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "checks": checks,
    }


def compare_graph_report_to_file(report: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    graph = load_graph_json(path)
    saved_digest = report.get("digest")
    graph_digest_value = graph_json_digest(graph)
    saved_summary = report.get("summary")
    graph_summary_value = graph_summary(graph)
    summary_differences = _nested_differences(saved_summary, graph_summary_value)
    checks: list[dict[str, Any]] = [
        {
            "name": "saved_report_valid",
            "passed": validate_graph_report(report)["valid"] is True,
        },
        {
            "name": "graph_digest_matches_report",
            "passed": saved_digest == graph_digest_value,
            "expected": saved_digest,
            "actual": graph_digest_value,
        },
        {
            "name": "summary_matches_report",
            "passed": saved_summary == graph_summary_value,
            "expected": saved_summary,
            "actual": graph_summary_value,
        },
    ]
    if summary_differences:
        checks[-1]["differences"] = summary_differences
    return {
        "matches": all(check["passed"] is True for check in checks),
        "path": str(path),
        "saved_digest": saved_digest,
        "graph_digest": graph_digest_value,
        "checks": checks,
    }


def compare_graph_to_fixture(graph: DynamicSceneGraph, fixture: str) -> dict[str, Any]:
    from dsg_spatialqa_lab.scenes import load_scene_fixture

    fixture_graph = load_scene_fixture(fixture)
    graph_digest = graph_json_digest(graph)
    fixture_digest = graph_json_digest(fixture_graph)
    graph_summary_value = graph_summary(graph)
    fixture_summary_value = graph_summary(fixture_graph)
    digest_matches = graph_digest == fixture_digest
    summary_matches = graph_summary_value == fixture_summary_value
    summary_differences = _nested_differences(fixture_summary_value, graph_summary_value)

    return {
        "matches": digest_matches and summary_matches,
        "fixture": fixture,
        "graph_digest": graph_digest,
        "fixture_digest": fixture_digest,
        "checks": [
            {
                "name": "graph_digest_matches_fixture",
                "passed": digest_matches,
                "expected": fixture_digest,
                "actual": graph_digest,
            },
            {
                "name": "summary_matches_fixture",
                "passed": summary_matches,
                "expected": fixture_summary_value,
                "actual": graph_summary_value,
                **({"differences": summary_differences} if summary_differences else {}),
            },
        ],
    }


def graph_from_json(payload: str) -> DynamicSceneGraph:
    parsed = json.loads(payload)
    return graph_from_dict(_as_mapping(parsed, "scene"))


def save_graph_json(graph: DynamicSceneGraph, path: str | Path) -> None:
    Path(path).write_text(graph_to_json(graph), encoding="utf-8")


def load_graph_json(path: str | Path) -> DynamicSceneGraph:
    return graph_from_json(Path(path).read_text(encoding="utf-8"))


def compare_graph_file_to_fixture(path: str | Path, fixture: str) -> dict[str, Any]:
    comparison = compare_graph_to_fixture(load_graph_json(path), fixture)
    return {"path": str(path), **comparison}


def _is_sha256_hexdigest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sorted_counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def _current_object_locations(graph: DynamicSceneGraph) -> dict[str, str]:
    locations = {
        object_id: edge
        for object_id, edge in _latest_containment_edges(graph).items()
        if object_id in graph.object_states
    }
    return {
        object_id: locations[object_id].dst
        for object_id in sorted(locations)
    }


def _current_object_rooms(graph: DynamicSceneGraph) -> dict[str, str]:
    containment_edges = _latest_containment_edges(graph)
    rooms: dict[str, str] = {}
    for object_id in sorted(graph.object_states):
        room = _current_room_for_node(graph, object_id, containment_edges)
        if room is not None:
            rooms[object_id] = room
    return rooms


def _current_room_for_node(
    graph: DynamicSceneGraph,
    node_id: str,
    containment_edges: Mapping[str, Edge],
) -> str | None:
    visited: set[str] = set()
    current_id = node_id
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
    return [{"path": path if path else "$", "expected": expected, "actual": actual}]


def _node_to_dict(node: Node) -> dict[str, Any]:
    return {
        "id": node.id,
        "type": node.type,
        "label": node.label,
        "attributes": node.attributes,
    }


def _node_from_mapping(data: Mapping[str, Any]) -> Node:
    node_type = _node_type(_required_str(data, "type"))
    label_value = data.get("label")
    label = label_value if isinstance(label_value, str) else None
    return Node(
        id=_required_str(data, "id"),
        type=node_type,
        label=label,
        attributes=dict(_as_mapping(data.get("attributes", {}), "node attributes")),
    )


def _edge_to_dict(edge: Edge) -> dict[str, Any]:
    return {
        "src": edge.src,
        "relation": edge.relation,
        "dst": edge.dst,
        "reference_frame": edge.reference_frame,
        "confidence": edge.confidence,
        "step": edge.step,
        "evidence": edge.evidence,
        "attributes": edge.attributes,
    }


def _edge_from_mapping(data: Mapping[str, Any]) -> Edge:
    return Edge(
        src=_required_str(data, "src"),
        relation=_required_str(data, "relation").upper(),
        dst=_required_str(data, "dst"),
        reference_frame=_required_str(data, "reference_frame"),
        confidence=_required_float(data, "confidence"),
        step=_required_int(data, "step"),
        evidence=[str(item) for item in _required_sequence(data, "evidence")],
        attributes=dict(_as_mapping(data.get("attributes", {}), "edge attributes")),
    )


def _object_state_to_dict(state: ObjectState) -> dict[str, Any]:
    return {
        "object_id": state.object_id,
        "label": state.label,
        "pose": _pose_to_dict(state.pose),
        "bbox": _bbox_to_dict(state.bbox),
        "confidence": state.confidence,
        "visible": state.visible,
        "step": state.step,
        "last_seen_step": state.last_seen_step,
        "last_seen_pose": _pose_to_dict(state.last_seen_pose) if state.last_seen_pose else None,
    }


def _agent_pose_state_to_dict(state: AgentPoseState) -> dict[str, Any]:
    return {
        "agent_id": state.agent_id,
        "pose": _pose_to_dict(state.pose),
        "step": state.step,
    }


def _agent_pose_state_from_mapping(data: Mapping[str, Any]) -> AgentPoseState:
    return AgentPoseState(
        agent_id=_required_str(data, "agent_id"),
        pose=_pose_from_mapping(_as_mapping(data.get("pose"), "agent pose")),
        step=_required_int(data, "step"),
    )


def _object_state_from_mapping(data: Mapping[str, Any]) -> ObjectState:
    last_seen_pose_data = data.get("last_seen_pose")
    last_seen_step_data = data.get("last_seen_step")
    return ObjectState(
        object_id=_required_str(data, "object_id"),
        label=_required_str(data, "label"),
        pose=_pose_from_mapping(_as_mapping(data.get("pose"), "object pose")),
        bbox=_bbox_from_mapping(_as_mapping(data.get("bbox"), "object bbox")),
        confidence=_required_float(data, "confidence"),
        visible=_required_bool(data, "visible"),
        step=_required_int(data, "step"),
        last_seen_step=(
            _required_int(data, "last_seen_step") if last_seen_step_data is not None else None
        ),
        last_seen_pose=(
            _pose_from_mapping(_as_mapping(last_seen_pose_data, "last_seen_pose"))
            if last_seen_pose_data is not None
            else None
        ),
    )


def _bbox_to_dict(bbox: BBox3D) -> dict[str, Any]:
    return {"center": _pose_to_dict(bbox.center), "size": list(bbox.size)}


def _bbox_from_mapping(data: Mapping[str, Any]) -> BBox3D:
    size = _required_sequence(data, "size")
    if len(size) != 3:
        raise SpatialQAError("BBox size must contain exactly three values")
    return BBox3D(
        center=_pose_from_mapping(_as_mapping(data.get("center"), "bbox center")),
        size=(_float_value(size[0], "size[0]"), _float_value(size[1], "size[1]"), _float_value(size[2], "size[2]")),
    )


def _pose_to_dict(pose: Pose3D) -> dict[str, float]:
    return pose.to_dict()


def _pose_from_mapping(data: Mapping[str, Any]) -> Pose3D:
    return Pose3D(
        x=_required_float(data, "x"),
        y=_required_float(data, "y"),
        z=_required_float(data, "z"),
        yaw=_required_float(data, "yaw"),
    )


def _node_type(value: str) -> NodeType:
    if value not in VALID_NODE_TYPES:
        raise SpatialQAError(f"Unsupported node type in scene: {value}")
    return cast(NodeType, value)


def _edge_sort_key(edge: Edge) -> tuple[int, str, str, str, str]:
    return (edge.step, edge.src, edge.relation, edge.dst, edge.reference_frame)


def _required_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise SpatialQAError(f"Expected string field: {key}")
    return value


def _required_int(data: Mapping[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise SpatialQAError(f"Expected integer field: {key}")
    return value


def _required_float(data: Mapping[str, Any], key: str) -> float:
    return _float_value(data.get(key), key)


def _float_value(value: Any, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise SpatialQAError(f"Expected numeric field: {label}")
    return float(value)


def _required_bool(data: Mapping[str, Any], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise SpatialQAError(f"Expected boolean field: {key}")
    return value


def _required_sequence(data: Mapping[str, Any], key: str) -> Sequence[Any]:
    return _as_sequence(data.get(key), key)


def _as_sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise SpatialQAError(f"Expected sequence: {label}")
    return value


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpatialQAError(f"Expected mapping: {label}")
    return value
