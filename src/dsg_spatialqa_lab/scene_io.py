from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.memory import DynamicSceneGraph, VALID_NODE_TYPES
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
        "by_node_type": _sorted_counts(node.type for node in graph.nodes.values()),
        "by_edge_relation": _sorted_counts(edge.relation for edge in graph.edges),
        "by_object_label": _sorted_counts(state.label for state in object_states),
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


def _sorted_counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
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
