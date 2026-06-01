from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.graph_tool import GraphTool
from dsg_spatialqa_lab.memory import DynamicSceneGraph
from dsg_spatialqa_lab.scene_io import (
    graph_json_digest,
    graph_report,
    graph_summary,
    load_graph_json,
    validate_graph_report,
)
from dsg_spatialqa_lab.schema import BBox3D, Pose3D, SpatialQAError


SCENE_OBSERVATION_SCHEMA_VERSION = "dsg-spatialqa-lab.scene-observation.v1"
SCENE_OBSERVATION_SEQUENCE_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.scene-observation-sequence.v1"
)
OBSERVATION_INGEST_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.observation-ingest-report.v1"
)


@dataclass(frozen=True)
class NodeObservation:
    node_id: str
    label: str
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectObservation:
    object_id: str
    label: str
    pose: Pose3D
    bbox: BBox3D
    confidence: float
    visible: bool
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneObservation:
    step: int
    agent_pose: Pose3D | None = None
    agent_id: str = "agent"
    rooms: tuple[NodeObservation, ...] = field(default_factory=tuple)
    regions: tuple[NodeObservation, ...] = field(default_factory=tuple)
    objects: tuple[ObjectObservation, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class IngestResult:
    step: int
    node_ids: tuple[str, ...]
    object_ids: tuple[str, ...]
    state_edge_ids: tuple[str, ...]
    inferred_edge_ids: tuple[str, ...]


def scene_observation_to_dict(observation: SceneObservation) -> dict[str, Any]:
    return {
        "schema_version": SCENE_OBSERVATION_SCHEMA_VERSION,
        "step": observation.step,
        "agent_id": observation.agent_id,
        "agent_pose": (
            _pose_to_dict(observation.agent_pose)
            if observation.agent_pose is not None
            else None
        ),
        "rooms": [
            _node_observation_to_dict(room)
            for room in sorted(observation.rooms, key=lambda item: item.node_id)
        ],
        "regions": [
            _node_observation_to_dict(region)
            for region in sorted(observation.regions, key=lambda item: item.node_id)
        ],
        "objects": [
            _object_observation_to_dict(obj)
            for obj in sorted(observation.objects, key=lambda item: item.object_id)
        ],
    }


def scene_observation_from_dict(payload: Mapping[str, Any]) -> SceneObservation:
    schema_version = payload.get("schema_version")
    if schema_version != SCENE_OBSERVATION_SCHEMA_VERSION:
        raise SpatialQAError(f"Unsupported scene observation schema version: {schema_version}")
    agent_pose = payload.get("agent_pose")
    return SceneObservation(
        step=_required_int(payload, "step"),
        agent_id=_required_str(payload, "agent_id"),
        agent_pose=(
            _pose_from_mapping(_as_mapping(agent_pose, "agent_pose"))
            if agent_pose is not None
            else None
        ),
        rooms=tuple(
            _node_observation_from_mapping(_as_mapping(item, "room"))
            for item in _required_sequence(payload, "rooms")
        ),
        regions=tuple(
            _node_observation_from_mapping(_as_mapping(item, "region"))
            for item in _required_sequence(payload, "regions")
        ),
        objects=tuple(
            _object_observation_from_mapping(_as_mapping(item, "object"))
            for item in _required_sequence(payload, "objects")
        ),
    )


def scene_observation_to_json(observation: SceneObservation) -> str:
    return json.dumps(scene_observation_to_dict(observation), indent=2, sort_keys=True) + "\n"


def scene_observation_from_json(payload: str) -> SceneObservation:
    parsed = json.loads(payload)
    return scene_observation_from_dict(_as_mapping(parsed, "scene_observation"))


def save_scene_observation(observation: SceneObservation, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(scene_observation_to_json(observation), encoding="utf-8")
    return output_path


def load_scene_observation(path: str | Path) -> SceneObservation:
    return scene_observation_from_json(Path(path).read_text(encoding="utf-8"))


def scene_observation_sequence_to_dict(
    observations: Sequence[SceneObservation],
) -> dict[str, Any]:
    return {
        "schema_version": SCENE_OBSERVATION_SEQUENCE_SCHEMA_VERSION,
        "observation_count": len(observations),
        "steps": [observation.step for observation in observations],
        "observations": [
            scene_observation_to_dict(observation)
            for observation in observations
        ],
    }


def scene_observation_sequence_from_dict(
    payload: Mapping[str, Any],
) -> tuple[SceneObservation, ...]:
    schema_version = payload.get("schema_version")
    if schema_version != SCENE_OBSERVATION_SEQUENCE_SCHEMA_VERSION:
        raise SpatialQAError(
            f"Unsupported scene observation sequence schema version: {schema_version}"
        )
    observations = tuple(
        scene_observation_from_dict(_as_mapping(item, "scene_observation"))
        for item in _required_sequence(payload, "observations")
    )
    observation_count = _required_int(payload, "observation_count")
    if observation_count != len(observations):
        raise SpatialQAError("observation_count must match observations length")
    steps = _required_sequence(payload, "steps")
    if list(steps) != [observation.step for observation in observations]:
        raise SpatialQAError("steps must match observation steps")
    return observations


def scene_observation_sequence_to_json(
    observations: Sequence[SceneObservation],
) -> str:
    return (
        json.dumps(
            scene_observation_sequence_to_dict(observations),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def scene_observation_sequence_digest(observations: Sequence[SceneObservation]) -> str:
    return hashlib.sha256(
        scene_observation_sequence_to_json(observations).encode("utf-8")
    ).hexdigest()


def scene_observation_sequence_from_json(payload: str) -> tuple[SceneObservation, ...]:
    parsed = json.loads(payload)
    return scene_observation_sequence_from_dict(
        _as_mapping(parsed, "scene_observation_sequence")
    )


def save_scene_observation_sequence(
    observations: Sequence[SceneObservation],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        scene_observation_sequence_to_json(observations),
        encoding="utf-8",
    )
    return output_path


def load_scene_observation_sequence(path: str | Path) -> tuple[SceneObservation, ...]:
    return scene_observation_sequence_from_json(Path(path).read_text(encoding="utf-8"))


def ingest_scene_observation_sequence(
    observations: Sequence[SceneObservation],
    *,
    source_path: str | Path | None = None,
    infer_relations: Sequence[str] = (),
    reference_frames: Sequence[str] = ("world",),
) -> tuple[DynamicSceneGraph, tuple[IngestResult, ...]]:
    graph = DynamicSceneGraph()
    ingestor = ObservationIngestor(graph)
    source_prefix = (
        f"observation_sequence:{source_path}" if source_path is not None else None
    )
    results = tuple(
        ingestor.ingest(
            observation,
            infer_relations=infer_relations,
            reference_frames=reference_frames,
            relation_evidence=(
                (f"{source_prefix}:{observation.step}",)
                if source_prefix is not None
                else None
            ),
        )
        for observation in observations
    )
    return graph, results


def observation_ingest_report(
    *,
    input_path: str | Path,
    graph_path: str | Path,
    graph: DynamicSceneGraph,
    ingest_results: Sequence[IngestResult],
    sequence_digest: str | None = None,
    infer_relations: Sequence[str] = (),
    reference_frames: Sequence[str] = ("world",),
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": OBSERVATION_INGEST_REPORT_SCHEMA_VERSION,
        "action": "ingest_observation_sequence",
        "path": str(input_path),
        "graph_path": str(graph_path),
        "valid": True,
        "options": {
            "infer_relations": list(infer_relations),
            "reference_frames": list(reference_frames),
        },
        "observation_count": len(ingest_results),
        "steps": [result.step for result in ingest_results],
        "ingest_results": [
            _ingest_result_to_dict(result) for result in ingest_results
        ],
        "graph_report": graph_report(
            graph,
            action="ingest_observation_sequence",
            graph_path=graph_path,
        ),
    }
    if sequence_digest is not None:
        report["sequence_digest"] = sequence_digest
    report["digest"] = observation_ingest_report_digest(report)
    return report


def observation_ingest_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def observation_ingest_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_observation_ingest_report(report: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(observation_ingest_report_json(report), encoding="utf-8")
    return output_path


def load_observation_ingest_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SpatialQAError("Observation ingest report JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_observation_ingest_report(report: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = report.get("schema_version")
    action = report.get("action")
    valid = report.get("valid")
    report_digest = report.get("digest")
    expected_report_digest = observation_ingest_report_digest(report)
    input_path = report.get("path")
    graph_path = report.get("graph_path")
    sequence_digest = report.get("sequence_digest")
    ingest_results = report.get("ingest_results")
    ingest_result_count = (
        len(ingest_results)
        if isinstance(ingest_results, Sequence) and not isinstance(ingest_results, str)
        else None
    )
    result_steps = (
        [
            _as_mapping(item, "ingest_result").get("step")
            for item in ingest_results
        ]
        if isinstance(ingest_results, Sequence) and not isinstance(ingest_results, str)
        else None
    )
    graph_report_value = report.get("graph_report")
    graph_report_path = (
        graph_report_value.get("path")
        if isinstance(graph_report_value, Mapping)
        else None
    )
    graph_report_valid = (
        validate_graph_report(_as_mapping(graph_report_value, "graph_report"))["valid"]
        is True
        if isinstance(graph_report_value, Mapping)
        else False
    )
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == OBSERVATION_INGEST_REPORT_SCHEMA_VERSION,
            "expected": OBSERVATION_INGEST_REPORT_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "action",
            "passed": action == "ingest_observation_sequence",
            "expected": "ingest_observation_sequence",
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
            "passed": report_digest == expected_report_digest,
            "expected": expected_report_digest,
            "actual": report_digest,
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
            "name": "graph_report_path_matches",
            "passed": _is_non_empty_string(graph_path) and graph_path == graph_report_path,
            "expected": graph_path,
            "actual": graph_report_path,
        },
        {
            "name": "sequence_digest_format",
            "passed": _is_sha256_hexdigest(sequence_digest),
            "expected": "64 lowercase sha256 hex characters",
            "actual": sequence_digest,
        },
        {
            "name": "observation_count_matches_results",
            "passed": report.get("observation_count") == ingest_result_count,
            "expected": report.get("observation_count"),
            "actual": ingest_result_count,
        },
        {
            "name": "steps_match_results",
            "passed": report.get("steps") == result_steps,
            "expected": report.get("steps"),
            "actual": result_steps,
        },
        {
            "name": "graph_report_valid",
            "passed": graph_report_valid,
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "digest": report_digest,
        "checks": checks,
    }


def compare_observation_ingest_report(report: Mapping[str, Any]) -> dict[str, Any]:
    options = _as_mapping(report.get("options", {}), "options")
    infer_relations = tuple(
        str(item) for item in _optional_sequence(options, "infer_relations")
    )
    reference_frames = tuple(
        str(item) for item in _optional_sequence(options, "reference_frames")
    ) or ("world",)
    sequence_path = _required_str(report, "path")
    graph_path = _required_str(report, "graph_path")
    observations = load_scene_observation_sequence(sequence_path)
    saved_sequence_digest = report.get("sequence_digest")
    current_sequence_digest = scene_observation_sequence_digest(observations)
    current_graph, current_results = ingest_scene_observation_sequence(
        observations,
        source_path=sequence_path,
        infer_relations=infer_relations,
        reference_frames=reference_frames,
    )
    saved_graph_report = _as_mapping(report.get("graph_report"), "graph_report")
    saved_digest = saved_graph_report.get("digest")
    current_digest = graph_json_digest(current_graph)
    saved_summary = saved_graph_report.get("summary")
    current_summary = graph_summary(current_graph)
    graph_file = load_graph_json(graph_path)
    graph_file_digest = graph_json_digest(graph_file)
    graph_file_summary = graph_summary(graph_file)
    saved_results = report.get("ingest_results")
    current_result_payload = [
        _ingest_result_to_dict(result) for result in current_results
    ]
    summary_differences = _nested_differences(saved_summary, current_summary)
    graph_file_summary_differences = _nested_differences(
        saved_summary,
        graph_file_summary,
    )
    ingest_result_differences = _nested_differences(
        saved_results,
        current_result_payload,
    )
    checks: list[dict[str, Any]] = [
        {
            "name": "saved_report_valid",
            "passed": validate_observation_ingest_report(report)["valid"] is True,
        },
        {
            "name": "sequence_digest_matches_current",
            "passed": saved_sequence_digest == current_sequence_digest,
            "expected": saved_sequence_digest,
            "actual": current_sequence_digest,
        },
        {
            "name": "graph_digest_matches_current",
            "passed": saved_digest == current_digest,
            "expected": saved_digest,
            "actual": current_digest,
        },
        {
            "name": "graph_summary_matches_current",
            "passed": saved_summary == current_summary,
            "expected": saved_summary,
            "actual": current_summary,
        },
        {
            "name": "ingest_results_match_current",
            "passed": saved_results == current_result_payload,
            "expected": saved_results,
            "actual": current_result_payload,
        },
        {
            "name": "graph_file_digest_matches_report",
            "passed": saved_digest == graph_file_digest,
            "expected": saved_digest,
            "actual": graph_file_digest,
        },
        {
            "name": "graph_file_summary_matches_report",
            "passed": saved_summary == graph_file_summary,
            "expected": saved_summary,
            "actual": graph_file_summary,
        },
    ]
    if summary_differences:
        checks[3]["differences"] = summary_differences
    if ingest_result_differences:
        checks[4]["differences"] = ingest_result_differences
    if graph_file_summary_differences:
        checks[6]["differences"] = graph_file_summary_differences
    return {
        "matches": all(check["passed"] is True for check in checks),
        "sequence_path": sequence_path,
        "graph_path": graph_path,
        "saved_sequence_digest": saved_sequence_digest,
        "current_sequence_digest": current_sequence_digest,
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "checks": checks,
    }


class ObservationIngestor:
    def __init__(self, graph: DynamicSceneGraph, *, graph_tool: GraphTool | None = None) -> None:
        self.graph = graph
        self.graph_tool = graph_tool or GraphTool(graph)

    def ingest(
        self,
        observation: SceneObservation,
        *,
        infer_relations: Sequence[str] = (),
        reference_frames: Sequence[str] = ("world", "agent"),
        relation_confidence: float = 1.0,
        relation_evidence: Sequence[str] | None = None,
    ) -> IngestResult:
        self._validate_step(observation.step)
        self._ensure_unique((room.node_id for room in observation.rooms), "rooms")
        self._ensure_unique((region.node_id for region in observation.regions), "regions")
        self._ensure_unique((obj.object_id for obj in observation.objects), "objects")

        node_ids: set[str] = set()
        if observation.agent_pose is not None:
            self.graph.set_agent_pose(
                observation.agent_id,
                observation.agent_pose,
                step=observation.step,
            )
            node_ids.add(observation.agent_id)

        for room in sorted(observation.rooms, key=lambda item: item.node_id):
            self.graph.add_room(
                room.node_id,
                room.label,
                step=observation.step,
                attributes=dict(room.attributes),
            )
            node_ids.add(room.node_id)

        for region in sorted(observation.regions, key=lambda item: item.node_id):
            self.graph.add_region(
                region.node_id,
                region.label,
                step=observation.step,
                attributes=dict(region.attributes),
            )
            node_ids.add(region.node_id)

        object_ids: list[str] = []
        state_edge_ids: list[str] = []
        for obj in sorted(observation.objects, key=lambda item: item.object_id):
            self.graph.upsert_object(
                obj.object_id,
                obj.label,
                obj.pose,
                obj.bbox,
                confidence=obj.confidence,
                visible=obj.visible,
                step=observation.step,
                attributes=dict(obj.attributes),
            )
            object_ids.append(obj.object_id)
            node_ids.add(obj.object_id)
            state_edge_ids.append(self._state_edge_id(obj.object_id, observation.step))

        inferred_edge_ids: tuple[str, ...] = ()
        if infer_relations and object_ids:
            inferred_edges = self.graph_tool.update_spatial_relations(
                step=observation.step,
                object_ids=tuple(object_ids),
                relations=infer_relations,
                reference_frames=reference_frames,
                agent_id=observation.agent_id,
                confidence=relation_confidence,
                evidence=relation_evidence,
            )
            inferred_edge_ids = tuple(edge.id for edge in inferred_edges)

        return IngestResult(
            step=observation.step,
            node_ids=tuple(sorted(node_ids)),
            object_ids=tuple(object_ids),
            state_edge_ids=tuple(state_edge_ids),
            inferred_edge_ids=inferred_edge_ids,
        )

    @staticmethod
    def _validate_step(step: int) -> None:
        if not isinstance(step, int) or isinstance(step, bool):
            raise SpatialQAError("observation step must be an integer")

    @staticmethod
    def _ensure_unique(ids: Iterable[str], label: str) -> None:
        seen: set[str] = set()
        for item_id in ids:
            if item_id in seen:
                raise SpatialQAError(f"Duplicate {label} observation id: {item_id}")
            seen.add(item_id)

    @staticmethod
    def _state_edge_id(object_id: str, step: int) -> str:
        return f"{object_id}-STATE_CHANGED-state:{object_id}:{step}-{step}"


def _ingest_result_to_dict(result: IngestResult) -> dict[str, Any]:
    return {
        "step": result.step,
        "node_ids": list(result.node_ids),
        "object_ids": list(result.object_ids),
        "state_edge_ids": list(result.state_edge_ids),
        "inferred_edge_ids": list(result.inferred_edge_ids),
    }


def _node_observation_to_dict(observation: NodeObservation) -> dict[str, Any]:
    return {
        "node_id": observation.node_id,
        "label": observation.label,
        "attributes": _stable_attributes(observation.attributes),
    }


def _node_observation_from_mapping(payload: Mapping[str, Any]) -> NodeObservation:
    return NodeObservation(
        _required_str(payload, "node_id"),
        _required_str(payload, "label"),
        attributes=_attributes_from_mapping(payload),
    )


def _object_observation_to_dict(observation: ObjectObservation) -> dict[str, Any]:
    return {
        "object_id": observation.object_id,
        "label": observation.label,
        "pose": _pose_to_dict(observation.pose),
        "bbox": _bbox_to_dict(observation.bbox),
        "confidence": observation.confidence,
        "visible": observation.visible,
        "attributes": _stable_attributes(observation.attributes),
    }


def _object_observation_from_mapping(payload: Mapping[str, Any]) -> ObjectObservation:
    return ObjectObservation(
        _required_str(payload, "object_id"),
        _required_str(payload, "label"),
        _pose_from_mapping(_as_mapping(payload.get("pose"), "pose")),
        _bbox_from_mapping(_as_mapping(payload.get("bbox"), "bbox")),
        confidence=_required_float(payload, "confidence"),
        visible=_required_bool(payload, "visible"),
        attributes=_attributes_from_mapping(payload),
    )


def _pose_to_dict(pose: Pose3D) -> dict[str, float]:
    return pose.to_dict()


def _pose_from_mapping(payload: Mapping[str, Any]) -> Pose3D:
    return Pose3D(
        _required_float(payload, "x"),
        _required_float(payload, "y"),
        _required_float(payload, "z"),
        yaw=_required_float(payload, "yaw"),
    )


def _bbox_to_dict(bbox: BBox3D) -> dict[str, Any]:
    return {
        "center": _pose_to_dict(bbox.center),
        "size": list(bbox.size),
    }


def _bbox_from_mapping(payload: Mapping[str, Any]) -> BBox3D:
    size = _required_sequence(payload, "size")
    if len(size) != 3:
        raise SpatialQAError("bbox size must contain exactly three numbers")
    size_tuple = (
        _number_from_value(size[0], "size"),
        _number_from_value(size[1], "size"),
        _number_from_value(size[2], "size"),
    )
    return BBox3D(
        center=_pose_from_mapping(_as_mapping(payload.get("center"), "center")),
        size=size_tuple,
    )


def _attributes_from_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    return dict(_as_mapping(payload.get("attributes", {}), "attributes"))


def _stable_attributes(attributes: Mapping[str, Any]) -> dict[str, Any]:
    return {key: attributes[key] for key in sorted(attributes)}


def _as_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpatialQAError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _required_sequence(payload: Mapping[str, Any], field_name: str) -> Sequence[Any]:
    value = payload.get(field_name)
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise SpatialQAError(f"{field_name} must be a sequence")
    return value


def _optional_sequence(payload: Mapping[str, Any], field_name: str) -> Sequence[Any]:
    value = payload.get(field_name, ())
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise SpatialQAError(f"{field_name} must be a sequence")
    return value


def _required_str(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str):
        raise SpatialQAError(f"{field_name} must be a string")
    return value


def _required_int(payload: Mapping[str, Any], field_name: str) -> int:
    value = payload.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise SpatialQAError(f"{field_name} must be an integer")
    return value


def _required_float(payload: Mapping[str, Any], field_name: str) -> float:
    return _number_from_value(payload.get(field_name), field_name)


def _number_from_value(value: object, field_name: str) -> float:
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        raise SpatialQAError(f"{field_name} must be a number")
    return float(value)


def _required_bool(payload: Mapping[str, Any], field_name: str) -> bool:
    value = payload.get(field_name)
    if not isinstance(value, bool):
        raise SpatialQAError(f"{field_name} must be a boolean")
    return value


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and value != ""


def _is_sha256_hexdigest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


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
    if (
        isinstance(expected, Sequence)
        and not isinstance(expected, str)
        and isinstance(actual, Sequence)
        and not isinstance(actual, str)
    ):
        if not _sequence_items_are_mappings(expected, actual):
            return [{"path": path if path else "$", "expected": expected, "actual": actual}]
        differences = []
        max_length = max(len(expected), len(actual))
        for index in range(max_length):
            child_path = str(index) if path == "" else f"{path}.{index}"
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


def _sequence_items_are_mappings(expected: Sequence[Any], actual: Sequence[Any]) -> bool:
    return all(isinstance(item, Mapping) for item in expected) and all(
        isinstance(item, Mapping) for item in actual
    )
