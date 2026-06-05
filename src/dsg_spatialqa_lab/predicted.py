from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
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
from dsg_spatialqa_lab.memory import DynamicSceneGraph
from dsg_spatialqa_lab.observations import (
    SceneObservation,
    ingest_scene_observation_sequence,
    load_scene_observation_sequence,
    scene_observation_sequence_digest,
    scene_observation_sequence_summary,
)
from dsg_spatialqa_lab.perception import (
    MockDepthProjector,
    MockSegmenter,
    SimpleObjectFusion,
    SimpleObjectTracker,
)
from dsg_spatialqa_lab.scene_io import (
    graph_json_digest,
    graph_report,
    graph_summary,
    load_graph_json,
    validate_graph_report,
)
from dsg_spatialqa_lab.schema import SpatialQAError


PREDICTED_GRAPH_REPORT_SCHEMA_VERSION = "dsg-spatialqa-lab.predicted-graph-report.v1"
OBSERVATION_PREDICTED_RELATIONS = ("LEFT_OF", "RIGHT_OF", "NEAR")
OBSERVATION_PREDICTED_REFERENCE_FRAMES = ("world",)


def build_predicted_graph_from_episode(
    frames: Sequence[EpisodeFrame],
    *,
    segmenter: MockSegmenter | None = None,
    projector: MockDepthProjector | None = None,
    tracker: SimpleObjectTracker | None = None,
    fusion: SimpleObjectFusion | None = None,
    infer_relations: bool = True,
) -> DynamicSceneGraph:
    validation = validate_episode_sequence(frames)
    if validation["valid"] is not True:
        raise SpatialQAError("Episode sequence must be ordered and contain unique episode steps")

    segmenter_value = segmenter or MockSegmenter()
    projector_value = projector or MockDepthProjector()
    tracker_value = tracker or SimpleObjectTracker()
    fusion_value = fusion or SimpleObjectFusion()
    graph = DynamicSceneGraph()
    for frame in frames:
        detections = segmenter_value.detect(frame)
        instances = projector_value.project_all(frame, detections)
        tracked_instances = tracker_value.track(instances)
        fusion_value.ingest_frame(
            graph,
            frame,
            tracked_instances,
            infer_relations=infer_relations,
        )
    return graph


def build_predicted_graph_from_observations(
    observations: Sequence[SceneObservation],
    *,
    source_path: str | Path | None = None,
    infer_relations: Sequence[str] = OBSERVATION_PREDICTED_RELATIONS,
    reference_frames: Sequence[str] = OBSERVATION_PREDICTED_REFERENCE_FRAMES,
    infer_containment: bool = False,
    containment_axis: str = "z",
    relation_top_k: int | None = None,
    require_detector_state_evidence: bool = False,
) -> DynamicSceneGraph:
    graph, _ = ingest_scene_observation_sequence(
        observations,
        source_path=source_path,
        infer_relations=infer_relations,
        reference_frames=reference_frames,
        infer_containment=infer_containment,
        containment_axis=containment_axis,
        relation_top_k=relation_top_k,
        require_detector_state_evidence=require_detector_state_evidence,
    )
    return graph


def predicted_graph_summary(
    graph: DynamicSceneGraph,
    frames: Sequence[EpisodeFrame],
) -> dict[str, Any]:
    detections, instance_ids = _perception_rollup(frames)
    inferred_relation_count = sum(
        1 for edge in graph.edges if edge.attributes.get("inferred") is True
    )
    hidden_update_count = sum(
        1
        for states in graph.object_state_history.values()
        for state in states
        if state.visible is False
    )
    return {
        "graph_summary": graph_summary(graph),
        "episode_summary": episode_sequence_summary(frames),
        "predicted_summary": {
            "detection_count": len(detections),
            "frame_count": len(frames),
            "hidden_update_count": hidden_update_count,
            "instance_count": len(instance_ids),
            "inferred_relation_count": inferred_relation_count,
            "unique_instance_count": len(set(instance_ids)),
            "by_detection_source": _sorted_counts(
                _detection_source(detection) for detection in detections
            ),
            "by_detection_label": _sorted_counts(detection.label for detection in detections),
        },
    }


def predicted_graph_observation_summary(
    graph: DynamicSceneGraph,
    observations: Sequence[SceneObservation],
) -> dict[str, Any]:
    objects = [obj for observation in observations for obj in observation.objects]
    inferred_relation_count = sum(
        1 for edge in graph.edges if edge.attributes.get("inferred") is True
    )
    return {
        "graph_summary": graph_summary(graph),
        "observation_summary": scene_observation_sequence_summary(observations),
        "predicted_summary": {
            "input_kind": "observation_sequence",
            "observation_count": len(observations),
            "object_observation_count": len(objects),
            "visible_object_observation_count": sum(1 for obj in objects if obj.visible),
            "hidden_object_observation_count": sum(1 for obj in objects if not obj.visible),
            "inferred_relation_count": inferred_relation_count,
            "by_observation_source": _sorted_counts(
                _observation_source(obj) for obj in objects
            ),
            "by_object_label": _sorted_counts(obj.label for obj in objects),
        },
    }


def predicted_graph_report(
    *,
    input_path: str | Path,
    graph_path: str | Path,
    graph: DynamicSceneGraph,
    frames: Sequence[EpisodeFrame],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": PREDICTED_GRAPH_REPORT_SCHEMA_VERSION,
        "action": "build_predicted_graph",
        "path": str(input_path),
        "graph_path": str(graph_path),
        "valid": True,
        "episode_digest": episode_sequence_digest(frames),
        "summary": predicted_graph_summary(graph, frames),
        "graph_report": graph_report(
            graph,
            action="build_predicted_graph",
            graph_path=graph_path,
        ),
    }
    report["digest"] = predicted_graph_report_digest(report)
    return report


def predicted_graph_report_from_observations(
    *,
    input_path: str | Path,
    graph_path: str | Path,
    graph: DynamicSceneGraph,
    observations: Sequence[SceneObservation],
    infer_relations: Sequence[str] = OBSERVATION_PREDICTED_RELATIONS,
    reference_frames: Sequence[str] = OBSERVATION_PREDICTED_REFERENCE_FRAMES,
    infer_containment: bool = False,
    containment_axis: str = "z",
    relation_top_k: int | None = None,
    require_detector_state_evidence: bool = False,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": PREDICTED_GRAPH_REPORT_SCHEMA_VERSION,
        "action": "build_predicted_graph",
        "input_kind": "observation_sequence",
        "path": str(input_path),
        "graph_path": str(graph_path),
        "valid": True,
        "observation_sequence_digest": scene_observation_sequence_digest(observations),
        "options": {
            "infer_relations": list(infer_relations),
            "reference_frames": list(reference_frames),
            "infer_containment": infer_containment,
            "containment_axis": containment_axis,
            "relation_top_k": relation_top_k,
            "require_detector_state_evidence": require_detector_state_evidence,
        },
        "summary": predicted_graph_observation_summary(graph, observations),
        "graph_report": graph_report(
            graph,
            action="build_predicted_graph",
            graph_path=graph_path,
        ),
    }
    report["digest"] = predicted_graph_report_digest(report)
    return report


def predicted_graph_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def predicted_graph_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_predicted_graph_report(report: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(predicted_graph_report_json(report), encoding="utf-8")
    return output_path


def load_predicted_graph_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SpatialQAError("Predicted graph report JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_predicted_graph_report(report: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = report.get("schema_version")
    action = report.get("action")
    input_kind = report.get("input_kind", "episode")
    valid = report.get("valid")
    digest = report.get("digest")
    expected_digest = predicted_graph_report_digest(report)
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
    options_value = report.get("options", {})
    options = options_value if isinstance(options_value, Mapping) else {}
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == PREDICTED_GRAPH_REPORT_SCHEMA_VERSION,
            "expected": PREDICTED_GRAPH_REPORT_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "action",
            "passed": action == "build_predicted_graph",
            "expected": "build_predicted_graph",
            "actual": action,
        },
        {
            "name": "input_kind",
            "passed": input_kind in ("episode", "observation_sequence"),
            "expected": "episode or observation_sequence",
            "actual": input_kind,
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
            "passed": (
                _is_sha256_hexdigest(report.get("episode_digest"))
                if input_kind == "episode"
                else _is_sha256_hexdigest(report.get("observation_sequence_digest"))
            ),
            "expected": "64 lowercase sha256 hex characters",
            "actual": (
                report.get("episode_digest")
                if input_kind == "episode"
                else report.get("observation_sequence_digest")
            ),
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
            "passed": _predicted_summary_shape_valid(summary, str(input_kind)),
        },
        {
            "name": "observation_options_shape",
            "passed": (
                _string_sequence(options.get("infer_relations"))
                and _string_sequence(options.get("reference_frames"))
                and isinstance(options.get("infer_containment", False), bool)
                and options.get("containment_axis", "z") in ("z", "y")
                and _optional_non_negative_int(options.get("relation_top_k"))
                and isinstance(
                    options.get("require_detector_state_evidence", False),
                    bool,
                )
                if input_kind == "observation_sequence"
                else True
            ),
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "digest": digest,
        "checks": checks,
    }


def compare_predicted_graph_report(report: Mapping[str, Any]) -> dict[str, Any]:
    input_kind = str(report.get("input_kind", "episode"))
    input_path = _required_str(report, "path")
    graph_path = _required_str(report, "graph_path")
    if input_kind == "observation_sequence":
        options = _as_mapping(report.get("options", {}), "options")
        infer_relations = tuple(str(item) for item in _optional_sequence(options, "infer_relations"))
        reference_frames = tuple(str(item) for item in _optional_sequence(options, "reference_frames"))
        infer_containment = options.get("infer_containment") is True
        containment_axis = str(options.get("containment_axis", "z"))
        relation_top_k = _optional_int_or_none(options.get("relation_top_k"))
        require_detector_state_evidence = (
            options.get("require_detector_state_evidence") is True
        )
        observations = load_scene_observation_sequence(input_path)
        current_graph = build_predicted_graph_from_observations(
            observations,
            source_path=input_path,
            infer_relations=infer_relations,
            reference_frames=reference_frames,
            infer_containment=infer_containment,
            containment_axis=containment_axis,
            relation_top_k=relation_top_k,
            require_detector_state_evidence=require_detector_state_evidence,
        )
        saved_input_digest = report.get("observation_sequence_digest")
        current_input_digest = scene_observation_sequence_digest(observations)
        current_summary = predicted_graph_observation_summary(current_graph, observations)
    else:
        frames = load_episode_sequence(input_path)
        current_graph = build_predicted_graph_from_episode(frames)
        saved_input_digest = report.get("episode_digest")
        current_input_digest = episode_sequence_digest(frames)
        current_summary = predicted_graph_summary(current_graph, frames)
    saved_graph_report = _as_mapping(report.get("graph_report"), "graph_report")
    saved_graph_digest = saved_graph_report.get("digest")
    current_graph_digest = graph_json_digest(current_graph)
    saved_summary = report.get("summary")
    graph_file = load_graph_json(graph_path)
    graph_file_digest = graph_json_digest(graph_file)
    graph_file_summary = graph_summary(graph_file)
    saved_graph_summary = saved_graph_report.get("summary")
    predicted_summary_differences = _nested_differences(saved_summary, current_summary)
    graph_file_summary_differences = _nested_differences(
        saved_graph_summary,
        graph_file_summary,
    )
    checks: list[dict[str, Any]] = [
        {
            "name": "saved_report_valid",
            "passed": validate_predicted_graph_report(report)["valid"] is True,
        },
        {
            "name": "input_digest_matches_current",
            "passed": saved_input_digest == current_input_digest,
            "expected": saved_input_digest,
            "actual": current_input_digest,
        },
        {
            "name": "graph_digest_matches_current",
            "passed": saved_graph_digest == current_graph_digest,
            "expected": saved_graph_digest,
            "actual": current_graph_digest,
        },
        {
            "name": "predicted_summary_matches_current",
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
    if predicted_summary_differences:
        checks[3]["differences"] = predicted_summary_differences
    if graph_file_summary_differences:
        checks[5]["differences"] = graph_file_summary_differences
    return {
        "matches": all(check["passed"] is True for check in checks),
        "input_kind": input_kind,
        "episode_path": input_path,
        "graph_path": graph_path,
        "saved_episode_digest": saved_input_digest,
        "current_episode_digest": current_input_digest,
        "saved_digest": saved_graph_digest,
        "current_digest": current_graph_digest,
        "checks": checks,
    }


def _perception_rollup(frames: Sequence[EpisodeFrame]) -> tuple[list[Any], list[str]]:
    segmenter = MockSegmenter()
    projector = MockDepthProjector()
    detections: list[Any] = []
    instance_ids: list[str] = []
    for frame in frames:
        frame_detections = segmenter.detect(frame)
        detections.extend(frame_detections)
        instance_ids.extend(
            instance.instance_id
            for instance in projector.project_all(frame, frame_detections)
        )
    return detections, instance_ids


def _as_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpatialQAError(f"{field_name} must be an object")
    return cast(Mapping[str, Any], value)


def _required_str(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str):
        raise SpatialQAError(f"{field_name} must be a string")
    return value


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


def _detection_source(detection: Any) -> str:
    attributes = getattr(detection, "attributes", {})
    if not isinstance(attributes, Mapping):
        return "mock_perception"
    for key in ("source", "source_name", "source_kind"):
        value = attributes.get(key)
        if isinstance(value, str) and value:
            return value
    return "mock_perception"


def _observation_source(observation: SceneObservation | Any) -> str:
    attributes = getattr(observation, "attributes", {})
    if not isinstance(attributes, Mapping):
        return "observation_sequence"
    for key in ("source", "source_name", "source_kind"):
        value = attributes.get(key)
        if isinstance(value, str) and value:
            return value
    return "observation_sequence"


def _predicted_summary_shape_valid(summary: Mapping[str, Any], input_kind: str) -> bool:
    common_keys = ("graph_summary", "predicted_summary")
    if not all(key in summary for key in common_keys):
        return False
    if input_kind == "episode":
        return "episode_summary" in summary
    if input_kind == "observation_sequence":
        return "observation_summary" in summary
    return False


def _optional_sequence(payload: Mapping[str, Any], field_name: str) -> Sequence[Any]:
    value = payload.get(field_name, ())
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise SpatialQAError(f"{field_name} must be a sequence")
    return value


def _string_sequence(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str)
        and all(isinstance(item, str) for item in value)
    )


def _optional_non_negative_int(value: object) -> bool:
    return value is None or (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


def _optional_int_or_none(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise SpatialQAError("relation_top_k must be an integer or null")
    if value < 0:
        raise SpatialQAError("relation_top_k must be non-negative")
    return value


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
