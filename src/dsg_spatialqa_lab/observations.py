from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.episodes import EpisodeFrame
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
DETECTOR_OBSERVATION_RECORD_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.detector-observation-record.v1"
)
EXTERNAL_DETECTOR_FRAME_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.external-detector-frame.v1"
)
DETECTOR_OBSERVATION_IMPORT_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.detector-observation-import-report.v1"
)
SCENE_OBSERVATION_SEQUENCE_SUMMARY_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.scene-observation-sequence-summary.v1"
)
OBSERVATION_SEQUENCE_LOW_CONFIDENCE_THRESHOLD = 0.5
OBSERVATION_CONTAINMENT_SOURCE = "containment_inference"
OBSERVATION_CONTAINMENT_AXES = ("z", "y")
OBSERVATION_ON_VERTICAL_MARGIN = 0.08
OBSERVATION_ON_OVERLAP_MARGIN = 0.05
OBSERVATION_ON_SUPPORT_OVERLAP_RATIO = 0.25
EPISODE_METADATA_COVERAGE_DETECTOR_NAME = "ai2thor_metadata_coverage_objects"
EPISODE_METADATA_COVERAGE_SOURCE_KIND = "ai2thor_metadata_coverage"


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


def validate_scene_observation_sequence_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = payload.get("schema_version")
    observations_value = payload.get("observations")
    observations_count = (
        len(observations_value)
        if isinstance(observations_value, Sequence)
        and not isinstance(observations_value, str)
        else None
    )
    observation_steps = _observation_steps_from_payload(observations_value)
    load_error: str | None = None
    observations: tuple[SceneObservation, ...] | None = None
    try:
        observations = scene_observation_sequence_from_dict(payload)
    except SpatialQAError as exc:
        load_error = str(exc)

    digest = (
        scene_observation_sequence_digest(observations)
        if observations is not None
        else None
    )
    summary = (
        scene_observation_sequence_summary(observations)
        if observations is not None
        else None
    )
    sequence_load_check: dict[str, Any] = {
        "name": "sequence_loads",
        "passed": observations is not None,
    }
    if load_error is not None:
        sequence_load_check["error"] = load_error
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == SCENE_OBSERVATION_SEQUENCE_SCHEMA_VERSION,
            "expected": SCENE_OBSERVATION_SEQUENCE_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "observations_sequence_present",
            "passed": observations_count is not None,
            "expected": "sequence",
            "actual": "sequence" if observations_count is not None else type(observations_value).__name__,
        },
        {
            "name": "observation_count_matches_observations",
            "passed": payload.get("observation_count") == observations_count,
            "expected": payload.get("observation_count"),
            "actual": observations_count,
        },
        {
            "name": "steps_match_observations",
            "passed": payload.get("steps") == observation_steps,
            "expected": payload.get("steps"),
            "actual": observation_steps,
        },
        sequence_load_check,
    ]
    validation: dict[str, Any] = {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "digest": digest,
        "summary": summary,
        "checks": checks,
    }
    if load_error is not None:
        validation["error"] = load_error
    return validation


def scene_observation_sequence_summary(
    observations: Sequence[SceneObservation],
    *,
    low_confidence_threshold: float = OBSERVATION_SEQUENCE_LOW_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    threshold = _number_from_value(low_confidence_threshold, "low_confidence_threshold")
    steps = [observation.step for observation in observations]
    objects = [obj for observation in observations for obj in observation.objects]
    summary: dict[str, Any] = {
        "schema_version": SCENE_OBSERVATION_SEQUENCE_SUMMARY_SCHEMA_VERSION,
        "sequence_digest": scene_observation_sequence_digest(observations),
        "low_confidence_threshold": threshold,
        "observation_count": len(observations),
        "steps": steps,
        "first_step": steps[0] if steps else None,
        "last_step": steps[-1] if steps else None,
        "agent_pose_count": sum(
            1 for observation in observations if observation.agent_pose is not None
        ),
        "room_observation_count": sum(len(observation.rooms) for observation in observations),
        "region_observation_count": sum(
            len(observation.regions) for observation in observations
        ),
        "object_observation_count": len(objects),
        "unique_object_count": len({obj.object_id for obj in objects}),
        "unique_object_ids": sorted({obj.object_id for obj in objects}),
        "visible_object_observation_count": sum(1 for obj in objects if obj.visible),
        "hidden_object_observation_count": sum(1 for obj in objects if not obj.visible),
        "low_confidence_object_observation_count": sum(
            1 for obj in objects if obj.confidence < threshold
        ),
        "reobserve_candidate_observation_count": sum(
            1 for obj in objects if not obj.visible and obj.confidence < threshold
        ),
        "by_object_label": _sorted_counts(obj.label for obj in objects),
        "by_step": [
            _scene_observation_step_summary(observation, threshold)
            for observation in observations
        ],
    }
    summary["digest"] = scene_observation_sequence_summary_digest(summary)
    return summary


def scene_observation_sequence_summary_digest(summary: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in _sequence_summary_projection(summary).items()
        if key != "digest"
    }
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def scene_observation_sequence_summary_json(summary: Mapping[str, Any]) -> str:
    return json.dumps(summary, indent=2, sort_keys=True) + "\n"


def save_scene_observation_sequence_summary(
    summary: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(scene_observation_sequence_summary_json(summary), encoding="utf-8")
    return output_path


def load_scene_observation_sequence_summary(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SpatialQAError("Scene observation sequence summary JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_scene_observation_sequence_summary(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = summary.get("schema_version")
    action = summary.get("action")
    valid = summary.get("valid")
    digest = summary.get("digest")
    expected_digest = scene_observation_sequence_summary_digest(summary)
    sequence_path = summary.get("path")
    sequence_digest = summary.get("sequence_digest")
    step_entries = _summary_step_entries(summary)
    step_values = _summary_step_values(step_entries)
    steps = _summary_steps(summary)
    first_step = steps[0] if steps else None
    last_step = steps[-1] if steps else None
    unique_object_ids = _summary_unique_object_ids(summary)
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == SCENE_OBSERVATION_SEQUENCE_SUMMARY_SCHEMA_VERSION,
            "expected": SCENE_OBSERVATION_SEQUENCE_SUMMARY_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "action",
            "passed": action == "summarize_observation_sequence",
            "expected": "summarize_observation_sequence",
            "actual": action,
        },
        {
            "name": "summary_valid",
            "passed": valid is True,
            "expected": True,
            "actual": valid,
        },
        {
            "name": "summary_digest",
            "passed": digest == expected_digest,
            "expected": expected_digest,
            "actual": digest,
        },
        {
            "name": "sequence_path_present",
            "passed": _is_non_empty_string(sequence_path),
            "expected": "non-empty explicit local path",
            "actual": sequence_path,
        },
        {
            "name": "sequence_digest_format",
            "passed": _is_sha256_hexdigest(sequence_digest),
            "expected": "64 lowercase sha256 hex characters",
            "actual": sequence_digest,
        },
        {
            "name": "by_step_entries_valid",
            "passed": step_entries is not None,
        },
        {
            "name": "observation_count_matches_by_step",
            "passed": summary.get("observation_count")
            == (len(step_entries) if step_entries is not None else None),
            "expected": summary.get("observation_count"),
            "actual": len(step_entries) if step_entries is not None else None,
        },
        {
            "name": "steps_match_by_step",
            "passed": steps == step_values,
            "expected": steps,
            "actual": step_values,
        },
        {
            "name": "first_step_matches_steps",
            "passed": summary.get("first_step") == first_step,
            "expected": summary.get("first_step"),
            "actual": first_step,
        },
        {
            "name": "last_step_matches_steps",
            "passed": summary.get("last_step") == last_step,
            "expected": summary.get("last_step"),
            "actual": last_step,
        },
        _summary_step_count_check(
            summary,
            step_entries,
            "object_observation_count",
            "object_count",
            "object_observation_count_matches_steps",
        ),
        _summary_step_count_check(
            summary,
            step_entries,
            "visible_object_observation_count",
            "visible_object_count",
            "visible_object_observation_count_matches_steps",
        ),
        _summary_step_count_check(
            summary,
            step_entries,
            "hidden_object_observation_count",
            "hidden_object_count",
            "hidden_object_observation_count_matches_steps",
        ),
        _summary_step_count_check(
            summary,
            step_entries,
            "low_confidence_object_observation_count",
            "low_confidence_object_count",
            "low_confidence_object_observation_count_matches_steps",
        ),
        _summary_step_count_check(
            summary,
            step_entries,
            "reobserve_candidate_observation_count",
            "reobserve_candidate_count",
            "reobserve_candidate_observation_count_matches_steps",
        ),
        {
            "name": "unique_object_count_matches_ids",
            "passed": summary.get("unique_object_count")
            == (len(unique_object_ids) if unique_object_ids is not None else None),
            "expected": summary.get("unique_object_count"),
            "actual": len(unique_object_ids) if unique_object_ids is not None else None,
        },
        {
            "name": "object_label_counts_valid",
            "passed": _summary_label_counts_valid(summary.get("by_object_label")),
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "digest": digest,
        "checks": checks,
    }


def compare_scene_observation_sequence_summary(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    sequence_path = _required_str(summary, "path")
    low_confidence_threshold = _number_from_value(
        summary.get("low_confidence_threshold"),
        "low_confidence_threshold",
    )
    observations = load_scene_observation_sequence(sequence_path)
    current_summary = scene_observation_sequence_summary(
        observations,
        low_confidence_threshold=low_confidence_threshold,
    )
    saved_summary = _sequence_summary_projection(summary)
    saved_sequence_digest = saved_summary.get("sequence_digest")
    current_sequence_digest = current_summary["sequence_digest"]
    saved_digest = saved_summary.get("digest")
    current_digest = current_summary["digest"]
    summary_differences = _nested_differences(saved_summary, current_summary)
    checks = [
        {
            "name": "saved_summary_valid",
            "passed": validate_scene_observation_sequence_summary(summary)["valid"] is True,
        },
        {
            "name": "sequence_digest_matches_current",
            "passed": saved_sequence_digest == current_sequence_digest,
            "expected": saved_sequence_digest,
            "actual": current_sequence_digest,
        },
        {
            "name": "sequence_summary_matches_current",
            "passed": saved_summary == current_summary,
            "expected": saved_summary,
            "actual": current_summary,
        },
    ]
    if summary_differences:
        checks[2]["differences"] = summary_differences
    return {
        "matches": all(check["passed"] is True for check in checks),
        "sequence_path": sequence_path,
        "saved_sequence_digest": saved_sequence_digest,
        "current_sequence_digest": current_sequence_digest,
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "checks": checks,
    }


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


def detector_observation_sequence_from_jsonl(payload: str) -> tuple[SceneObservation, ...]:
    observations: list[SceneObservation] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if line == "":
            continue
        item = json.loads(line)
        observation = _scene_observation_from_detector_record(
            _as_mapping(item, f"detector_observation_record line {line_number}"),
        )
        observations.append(observation)
    return tuple(sorted(_validate_unique_observation_steps(observations), key=lambda item: item.step))


def load_detector_observation_sequence(path: str | Path) -> tuple[SceneObservation, ...]:
    return detector_observation_sequence_from_jsonl(Path(path).read_text(encoding="utf-8"))


def episode_metadata_coverage_detector_jsonl(
    frames: Sequence[EpisodeFrame],
    *,
    include_hidden: bool = True,
    path_prefix: str | None = None,
) -> str:
    return "".join(
        json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        for record in episode_metadata_coverage_detector_records(
            frames,
            include_hidden=include_hidden,
            path_prefix=path_prefix,
        )
    )


def episode_metadata_coverage_detector_records(
    frames: Sequence[EpisodeFrame],
    *,
    include_hidden: bool = True,
    path_prefix: str | None = None,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        _episode_frame_to_coverage_detector_record(
            frame,
            include_hidden=include_hidden,
            path_prefix=path_prefix,
        )
        for frame in sorted(frames, key=lambda item: (item.episode_id, item.step))
    )


def detector_observation_records_digest(payload: str) -> str:
    return hashlib.sha256(_normalized_detector_jsonl(payload).encode("utf-8")).hexdigest()


def detector_observation_import_report(
    *,
    input_path: str | Path,
    output_sequence_path: str | Path,
    observations: Sequence[SceneObservation],
    input_payload: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": DETECTOR_OBSERVATION_IMPORT_REPORT_SCHEMA_VERSION,
        "action": "import_detector_observation_jsonl",
        "path": str(input_path),
        "output_sequence_path": str(output_sequence_path),
        "valid": True,
        "input_digest": detector_observation_records_digest(input_payload),
        "sequence_digest": scene_observation_sequence_digest(observations),
        "summary": scene_observation_sequence_summary(observations),
    }
    report["digest"] = detector_observation_import_report_digest(report)
    return report


def detector_observation_import_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def detector_observation_import_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_detector_observation_import_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(detector_observation_import_report_json(report), encoding="utf-8")
    return output_path


def load_detector_observation_import_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SpatialQAError("Detector observation import report JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_detector_observation_import_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = report.get("schema_version")
    action = report.get("action")
    valid = report.get("valid")
    digest = report.get("digest")
    expected_digest = detector_observation_import_report_digest(report)
    sequence_digest = report.get("sequence_digest")
    summary = report.get("summary")
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == DETECTOR_OBSERVATION_IMPORT_REPORT_SCHEMA_VERSION,
            "expected": DETECTOR_OBSERVATION_IMPORT_REPORT_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "action",
            "passed": action == "import_detector_observation_jsonl",
            "expected": "import_detector_observation_jsonl",
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
            "passed": _is_non_empty_string(report.get("path")),
            "expected": "non-empty explicit local path",
            "actual": report.get("path"),
        },
        {
            "name": "output_sequence_path_present",
            "passed": _is_non_empty_string(report.get("output_sequence_path")),
            "expected": "non-empty explicit local path",
            "actual": report.get("output_sequence_path"),
        },
        {
            "name": "input_digest_format",
            "passed": _is_sha256_hexdigest(report.get("input_digest")),
            "expected": "64 lowercase sha256 hex characters",
            "actual": report.get("input_digest"),
        },
        {
            "name": "sequence_digest_format",
            "passed": _is_sha256_hexdigest(sequence_digest),
            "expected": "64 lowercase sha256 hex characters",
            "actual": sequence_digest,
        },
        {
            "name": "summary_sequence_digest_matches",
            "passed": (
                isinstance(summary, Mapping)
                and summary.get("sequence_digest") == sequence_digest
            ),
            "expected": sequence_digest,
            "actual": summary.get("sequence_digest") if isinstance(summary, Mapping) else None,
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "digest": digest,
        "checks": checks,
    }


def compare_detector_observation_import_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    input_path = _required_str(report, "path")
    output_sequence_path = _required_str(report, "output_sequence_path")
    input_payload = Path(input_path).read_text(encoding="utf-8")
    current_observations = detector_observation_sequence_from_jsonl(input_payload)
    output_observations = load_scene_observation_sequence(output_sequence_path)
    current_report = detector_observation_import_report(
        input_path=input_path,
        output_sequence_path=output_sequence_path,
        observations=current_observations,
        input_payload=input_payload,
    )
    current_sequence_digest = scene_observation_sequence_digest(current_observations)
    output_sequence_digest = scene_observation_sequence_digest(output_observations)
    current_summary = scene_observation_sequence_summary(current_observations)
    saved_summary = report.get("summary")
    summary_differences = _nested_differences(saved_summary, current_summary)
    checks = [
        {
            "name": "saved_report_valid",
            "passed": validate_detector_observation_import_report(report)["valid"] is True,
        },
        {
            "name": "input_digest_matches_current",
            "passed": report.get("input_digest") == current_report["input_digest"],
            "expected": report.get("input_digest"),
            "actual": current_report["input_digest"],
        },
        {
            "name": "sequence_digest_matches_current",
            "passed": report.get("sequence_digest") == current_sequence_digest,
            "expected": report.get("sequence_digest"),
            "actual": current_sequence_digest,
        },
        {
            "name": "output_sequence_digest_matches_report",
            "passed": report.get("sequence_digest") == output_sequence_digest,
            "expected": report.get("sequence_digest"),
            "actual": output_sequence_digest,
        },
        {
            "name": "summary_matches_current",
            "passed": saved_summary == current_summary,
            "expected": saved_summary,
            "actual": current_summary,
        },
    ]
    if summary_differences:
        checks[4]["differences"] = summary_differences
    return {
        "matches": all(check["passed"] is True for check in checks),
        "input_path": input_path,
        "output_sequence_path": output_sequence_path,
        "saved_input_digest": report.get("input_digest"),
        "current_input_digest": current_report["input_digest"],
        "saved_sequence_digest": report.get("sequence_digest"),
        "current_sequence_digest": current_sequence_digest,
        "output_sequence_digest": output_sequence_digest,
        "checks": checks,
    }


def _scene_observation_from_detector_record(
    payload: Mapping[str, Any],
) -> SceneObservation:
    schema_version = payload.get("schema_version")
    if schema_version == EXTERNAL_DETECTOR_FRAME_SCHEMA_VERSION:
        return _scene_observation_from_external_detector_frame(payload)
    if schema_version != DETECTOR_OBSERVATION_RECORD_SCHEMA_VERSION:
        raise SpatialQAError(
            f"Unsupported detector observation record schema version: {schema_version}"
        )
    agent_pose = payload.get("agent_pose")
    frame_attributes = _detector_frame_attributes(payload)
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
            for item in _optional_sequence(payload, "rooms")
        ),
        regions=tuple(
            _node_observation_from_mapping(_as_mapping(item, "region"))
            for item in _optional_sequence(payload, "regions")
        ),
        objects=tuple(
            _object_observation_from_detector_mapping(
                _as_mapping(item, "detection"),
                frame_attributes,
            )
            for item in _required_sequence(payload, "detections")
        ),
    )


def _scene_observation_from_external_detector_frame(
    payload: Mapping[str, Any],
) -> SceneObservation:
    detector_name = _required_str(payload, "detector_name")
    camera_pose = payload.get("camera_pose")
    frame_attributes = _external_detector_frame_attributes(payload, detector_name)
    return SceneObservation(
        step=_required_int(payload, "step"),
        agent_id=_optional_string(payload, "agent_id", default="agent"),
        agent_pose=(
            _pose_from_mapping(_as_mapping(camera_pose, "camera_pose"))
            if camera_pose is not None
            else None
        ),
        objects=tuple(
            _object_observation_from_external_detection(
                _as_mapping(item, "detection"),
                frame_attributes,
            )
            for item in _required_sequence(payload, "detections")
        ),
    )


def _external_detector_frame_attributes(
    payload: Mapping[str, Any],
    detector_name: str,
) -> dict[str, Any]:
    attributes = {
        "detector": detector_name,
        "source": detector_name,
        "source_kind": "detector",
        "source_name": detector_name,
    }
    for field_name in ("episode_id", "scene_id", "rgb_path", "depth_path"):
        value = payload.get(field_name)
        if value is not None:
            if not isinstance(value, str):
                raise SpatialQAError(f"{field_name} must be a string")
            attributes[field_name] = value
    return attributes


def _object_observation_from_external_detection(
    payload: Mapping[str, Any],
    frame_attributes: Mapping[str, Any],
) -> ObjectObservation:
    detection_id = _required_str(payload, "detection_id")
    object_id_value = payload.get("object_id")
    source_warning = None
    if isinstance(object_id_value, str) and object_id_value:
        object_id = object_id_value
    else:
        object_id = detection_id
        source_warning = "object_id_missing_used_detection_id"
    center = _pose_from_mapping(_as_mapping(payload.get("bbox_3d_center"), "bbox_3d_center"))
    bbox = BBox3D(
        center=center,
        size=_size_tuple(_required_sequence(payload, "bbox_3d_size")),
    )
    attributes = {
        **dict(frame_attributes),
        "detection_id": detection_id,
        "evidence_kinds": _stable_string_list(payload.get("evidence_kinds")),
    }
    if source_warning is not None:
        attributes["source_warning"] = source_warning
    for field_name in ("mask_path",):
        value = payload.get(field_name)
        if value is not None:
            if not isinstance(value, str):
                raise SpatialQAError(f"{field_name} must be a string")
            attributes[field_name] = value
    bbox_2d = payload.get("bbox_2d_xyxy")
    if bbox_2d is not None:
        attributes["bbox_2d_xyxy"] = _stable_json_value(
            _required_sequence(payload, "bbox_2d_xyxy")
        )
    attributes.update(
        _stable_json_mapping(
            _as_mapping(payload.get("attributes", {}), "detection attributes")
        )
    )
    return ObjectObservation(
        object_id,
        _required_str(payload, "label"),
        center,
        bbox,
        confidence=_required_float(payload, "confidence"),
        visible=_required_bool(payload, "visible"),
        attributes=attributes,
    )


def _episode_frame_to_coverage_detector_record(
    frame: EpisodeFrame,
    *,
    include_hidden: bool,
    path_prefix: str | None,
) -> dict[str, Any]:
    metadata = _as_mapping(frame.metadata, "metadata")
    objects = [
        _episode_metadata_object_to_detection(_as_mapping(item, "metadata object"))
        for item in _optional_sequence(metadata, "objects")
    ]
    if not include_hidden:
        objects = [item for item in objects if item["visible"] is True]
    objects = _disambiguate_detector_detection_ids(
        sorted(objects, key=lambda item: item["object_id"])
    )
    rooms = [
        _node_observation_to_dict(_metadata_room_observation(item))
        for item in _optional_sequence(metadata, "rooms")
    ]
    regions = [
        _node_observation_to_dict(_metadata_region_observation(item))
        for item in _optional_sequence(metadata, "regions")
    ]
    return {
        "schema_version": DETECTOR_OBSERVATION_RECORD_SCHEMA_VERSION,
        "step": frame.step,
        "agent_id": frame.agent_id,
        "agent_pose": frame.agent_pose.to_dict(),
        "rgb_path": _coverage_asset_path(frame.rgb_path, path_prefix),
        "depth_path": _coverage_asset_path(frame.depth_path, path_prefix),
        "segmentation_path": _coverage_asset_path(
            frame.segmentation_path,
            path_prefix,
        ),
        "metadata": {
            "action": frame.action,
            "detector": EPISODE_METADATA_COVERAGE_DETECTOR_NAME,
            "episode_id": frame.episode_id,
            "local_step": _episode_local_step(frame),
            "scene_id": frame.scene_id,
            "source_kind": EPISODE_METADATA_COVERAGE_SOURCE_KIND,
        },
        "rooms": sorted(rooms, key=lambda item: item["node_id"]),
        "regions": sorted(regions, key=lambda item: item["node_id"]),
        "detections": objects,
    }


def _episode_metadata_object_to_detection(payload: Mapping[str, Any]) -> dict[str, Any]:
    attributes = dict(_as_mapping(payload.get("attributes", {}), "object attributes"))
    states = payload.get("states")
    if isinstance(states, Mapping):
        attributes["states"] = _stable_json_mapping(states)
    for field_name in ("region_id", "room_id"):
        value = payload.get(field_name)
        if isinstance(value, str):
            attributes[field_name] = value
    attributes["coverage_source"] = "episode_metadata"
    return {
        "object_id": _required_str(payload, "object_id"),
        "label": _required_str(payload, "label"),
        "pose": _pose_from_mapping(_as_mapping(payload.get("pose"), "object pose")).to_dict(),
        "bbox": _bbox_to_dict(
            _bbox_from_mapping(_as_mapping(payload.get("bbox"), "object bbox"))
        ),
        "confidence": _required_float(payload, "confidence"),
        "visible": payload.get("visible") is True,
        "attributes": attributes,
    }


def _disambiguate_detector_detection_ids(
    detections: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    disambiguated: list[dict[str, Any]] = []
    for detection in detections:
        object_id = _required_str(detection, "object_id")
        counts[object_id] = counts.get(object_id, 0) + 1
        record = dict(detection)
        if counts[object_id] > 1:
            attributes = dict(_as_mapping(record.get("attributes", {}), "attributes"))
            attributes["original_object_id"] = object_id
            record["object_id"] = f"{object_id}_dup{counts[object_id]}"
            record["attributes"] = attributes
        disambiguated.append(record)
    return disambiguated


def _metadata_room_observation(value: object) -> NodeObservation:
    payload = _as_mapping(value, "metadata room")
    node_id = payload.get("node_id", payload.get("room_id"))
    if not isinstance(node_id, str):
        raise SpatialQAError("metadata room must include node_id or room_id")
    label = payload.get("label", node_id)
    if not isinstance(label, str):
        raise SpatialQAError("metadata room label must be a string")
    attributes = _as_mapping(payload.get("attributes", {}), "metadata room attributes")
    return NodeObservation(node_id, label, attributes=_stable_json_mapping(attributes))


def _metadata_region_observation(value: object) -> NodeObservation:
    payload = _as_mapping(value, "metadata region")
    node_id = payload.get("node_id", payload.get("region_id"))
    if not isinstance(node_id, str):
        raise SpatialQAError("metadata region must include node_id or region_id")
    label = payload.get("label", node_id)
    if not isinstance(label, str):
        raise SpatialQAError("metadata region label must be a string")
    attributes = dict(
        _as_mapping(payload.get("attributes", {}), "metadata region attributes")
    )
    room_id = payload.get("room_id")
    if isinstance(room_id, str):
        attributes["room_id"] = room_id
    return NodeObservation(node_id, label, attributes=_stable_json_mapping(attributes))


def _coverage_asset_path(path: str | None, path_prefix: str | None) -> str | None:
    if path is None:
        return None
    normalized = path
    while normalized.startswith("../"):
        normalized = normalized[3:]
    if path_prefix is None or path_prefix == "":
        return normalized
    clean_prefix = path_prefix.strip("/")
    if normalized == clean_prefix or normalized.startswith(f"{clean_prefix}/"):
        return normalized
    return f"{clean_prefix}/{normalized.lstrip('/')}"


def _episode_local_step(frame: EpisodeFrame) -> int:
    return ((frame.step - 1) % 10) + 1


def _stable_json_mapping(payload: Mapping[Any, Any]) -> dict[str, Any]:
    return {
        str(key): _stable_json_value(payload[key])
        for key in sorted(payload, key=lambda item: str(item))
    }


def _stable_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _stable_json_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_stable_json_value(item) for item in value]
    return value


def _stable_string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise SpatialQAError("evidence_kinds must be a sequence")
    values: list[str] = []
    for item in value:
        if not isinstance(item, str) or item == "":
            raise SpatialQAError("evidence_kinds must contain non-empty strings")
        if item not in values:
            values.append(item)
    return sorted(values)


def _size_tuple(value: Sequence[Any]) -> tuple[float, float, float]:
    if len(value) != 3:
        raise SpatialQAError("bbox_3d_size must contain exactly three numbers")
    return (
        _number_from_value(value[0], "bbox_3d_size"),
        _number_from_value(value[1], "bbox_3d_size"),
        _number_from_value(value[2], "bbox_3d_size"),
    )


def _object_observation_from_detector_mapping(
    payload: Mapping[str, Any],
    frame_attributes: Mapping[str, Any],
) -> ObjectObservation:
    observation = _object_observation_from_mapping(payload)
    attributes = {
        **dict(frame_attributes),
        **dict(observation.attributes),
    }
    return ObjectObservation(
        observation.object_id,
        observation.label,
        observation.pose,
        observation.bbox,
        confidence=observation.confidence,
        visible=observation.visible,
        attributes=attributes,
    )


def _detector_frame_attributes(payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _as_mapping(payload.get("metadata", {}), "metadata")
    attributes: dict[str, Any] = {
        key: metadata[key]
        for key in sorted(metadata)
        if _json_attribute_value(metadata[key])
    }
    for field_name in ("rgb_path", "depth_path", "segmentation_path"):
        value = payload.get(field_name)
        if value is None:
            continue
        if not isinstance(value, str):
            raise SpatialQAError(f"{field_name} must be a string")
        attributes[field_name] = value
    return attributes


def _validate_unique_observation_steps(
    observations: Sequence[SceneObservation],
) -> Sequence[SceneObservation]:
    seen: set[int] = set()
    for observation in observations:
        if observation.step in seen:
            raise SpatialQAError(f"Duplicate detector observation step: {observation.step}")
        seen.add(observation.step)
    return observations


def _normalized_detector_jsonl(payload: str) -> str:
    lines: list[str] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if line == "":
            continue
        item = json.loads(line)
        if not isinstance(item, Mapping):
            raise SpatialQAError(
                f"detector observation line {line_number} must be an object"
            )
        lines.append(json.dumps(item, separators=(",", ":"), sort_keys=True))
    return "".join(f"{line}\n" for line in lines)


def _json_attribute_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, Sequence) and not isinstance(value, str):
        return all(_json_attribute_value(item) for item in value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _json_attribute_value(item)
            for key, item in value.items()
        )
    return False


def ingest_scene_observation_sequence(
    observations: Sequence[SceneObservation],
    *,
    source_path: str | Path | None = None,
    infer_relations: Sequence[str] = (),
    reference_frames: Sequence[str] = ("world",),
    infer_containment: bool = False,
    containment_axis: str = "z",
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
            infer_containment=infer_containment,
            containment_axis=containment_axis,
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
    infer_containment: bool = False,
    containment_axis: str = "z",
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "infer_relations": list(infer_relations),
        "reference_frames": list(reference_frames),
    }
    if infer_containment or containment_axis != "z":
        options["infer_containment"] = infer_containment
        options["containment_axis"] = containment_axis
    report: dict[str, Any] = {
        "schema_version": OBSERVATION_INGEST_REPORT_SCHEMA_VERSION,
        "action": "ingest_observation_sequence",
        "path": str(input_path),
        "graph_path": str(graph_path),
        "valid": True,
        "options": options,
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
    infer_containment = options.get("infer_containment") is True
    containment_axis = str(options.get("containment_axis", "z"))
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
        infer_containment=infer_containment,
        containment_axis=containment_axis,
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
        infer_containment: bool = False,
        containment_axis: str = "z",
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
                attributes=_prediction_source_attributes(obj.attributes),
            )
            object_ids.append(obj.object_id)
            node_ids.add(obj.object_id)
            state_edge_ids.append(self._state_edge_id(obj.object_id, observation.step))

        inferred_edges = []
        if infer_containment and object_ids:
            inferred_edges.extend(
                self._infer_containment_edges(
                    observation,
                    axis=containment_axis,
                    evidence=relation_evidence,
                )
            )

        if infer_relations and object_ids:
            inferred_edges.extend(
                self.graph_tool.update_spatial_relations(
                    step=observation.step,
                    object_ids=tuple(object_ids),
                    relations=infer_relations,
                    reference_frames=reference_frames,
                    agent_id=observation.agent_id,
                    confidence=relation_confidence,
                    evidence=relation_evidence,
                    attributes={"source": "geometry_inference"},
                )
            )
        inferred_edge_ids = tuple(
            edge.id for edge in sorted(inferred_edges, key=self.graph_tool._edge_sort_key)
        )

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

    def _infer_containment_edges(
        self,
        observation: SceneObservation,
        *,
        axis: str,
        evidence: Sequence[str] | None,
    ) -> list[Any]:
        normalized_axis = axis.lower()
        if normalized_axis not in OBSERVATION_CONTAINMENT_AXES:
            raise SpatialQAError(
                f"containment_axis must be one of: {', '.join(OBSERVATION_CONTAINMENT_AXES)}"
            )
        added: list[Any] = []
        evidence_list = list(evidence or [])
        attributes = {
            "axis": normalized_axis,
            "inferred": True,
            "source": OBSERVATION_CONTAINMENT_SOURCE,
        }
        region = self._preferred_region(observation.regions)
        room = self._preferred_room(observation.rooms, region)
        for obj in sorted(observation.objects, key=lambda item: item.object_id):
            if region is not None and not self._has_edge(
                obj.object_id,
                "IN_REGION",
                region.node_id,
                observation.step,
            ):
                added.append(
                    self.graph.add_edge(
                        obj.object_id,
                        "IN_REGION",
                        region.node_id,
                        "world",
                        obj.confidence,
                        step=observation.step,
                        evidence=evidence_list,
                        attributes=attributes,
                    )
                )
            if room is not None and not self._has_edge(
                obj.object_id,
                "IN_ROOM",
                room.node_id,
                observation.step,
            ):
                added.append(
                    self.graph.add_edge(
                        obj.object_id,
                        "IN_ROOM",
                        room.node_id,
                        "world",
                        obj.confidence,
                        step=observation.step,
                        evidence=evidence_list,
                        attributes=attributes,
                    )
                )

        objects = sorted(observation.objects, key=lambda item: item.object_id)
        for src in objects:
            for dst in objects:
                if src.object_id == dst.object_id:
                    continue
                if not _is_on_axis(src.bbox, dst.bbox, normalized_axis):
                    continue
                if self._has_edge(src.object_id, "ON", dst.object_id, observation.step):
                    continue
                added.append(
                    self.graph.add_edge(
                        src.object_id,
                        "ON",
                        dst.object_id,
                        "world",
                        min(src.confidence, dst.confidence),
                        step=observation.step,
                        evidence=evidence_list,
                        attributes=attributes,
                    )
                )
        return added

    @staticmethod
    def _preferred_region(regions: Sequence[NodeObservation]) -> NodeObservation | None:
        if not regions:
            return None
        return sorted(
            regions,
            key=lambda item: (
                0 if item.node_id == "visible_region" else 1,
                item.node_id,
            ),
        )[0]

    @staticmethod
    def _preferred_room(
        rooms: Sequence[NodeObservation],
        region: NodeObservation | None,
    ) -> NodeObservation | None:
        if not rooms:
            return None
        room_by_id = {room.node_id: room for room in rooms}
        if region is not None:
            region_room_id = region.attributes.get("room_id")
            if isinstance(region_room_id, str) and region_room_id in room_by_id:
                return room_by_id[region_room_id]
        return sorted(rooms, key=lambda item: item.node_id)[0]

    def _has_edge(self, src: str, relation: str, dst: str, step: int) -> bool:
        return any(
            edge.step == step
            for edge in self.graph.find_edges(src=src, relation=relation, dst=dst)
        )


def _ingest_result_to_dict(result: IngestResult) -> dict[str, Any]:
    return {
        "step": result.step,
        "node_ids": list(result.node_ids),
        "object_ids": list(result.object_ids),
        "state_edge_ids": list(result.state_edge_ids),
        "inferred_edge_ids": list(result.inferred_edge_ids),
    }


def _scene_observation_step_summary(
    observation: SceneObservation,
    low_confidence_threshold: float,
) -> dict[str, Any]:
    return {
        "step": observation.step,
        "agent_pose": observation.agent_pose is not None,
        "room_count": len(observation.rooms),
        "region_count": len(observation.regions),
        "object_count": len(observation.objects),
        "object_ids": sorted(obj.object_id for obj in observation.objects),
        "visible_object_count": sum(1 for obj in observation.objects if obj.visible),
        "hidden_object_count": sum(1 for obj in observation.objects if not obj.visible),
        "low_confidence_object_count": sum(
            1 for obj in observation.objects if obj.confidence < low_confidence_threshold
        ),
        "reobserve_candidate_count": sum(
            1
            for obj in observation.objects
            if not obj.visible and obj.confidence < low_confidence_threshold
        ),
    }


def _is_on_axis(src: BBox3D, dst: BBox3D, axis: str) -> bool:
    vertical_touch = (
        abs(_bbox_min(src, axis) - _bbox_max(dst, axis))
        <= OBSERVATION_ON_VERTICAL_MARGIN
    )
    support_area = _horizontal_overlap_area(
        src,
        dst,
        axis=axis,
        margin=OBSERVATION_ON_OVERLAP_MARGIN,
    )
    required_area = (
        _horizontal_area(src, axis=axis) * OBSERVATION_ON_SUPPORT_OVERLAP_RATIO
    )
    return vertical_touch and support_area >= required_area


def _bbox_min(bbox: BBox3D, axis: str) -> float:
    if axis == "x":
        return bbox.min_x
    if axis == "y":
        return bbox.min_y
    if axis == "z":
        return bbox.min_z
    raise SpatialQAError(f"Unsupported bbox axis: {axis}")


def _bbox_max(bbox: BBox3D, axis: str) -> float:
    if axis == "x":
        return bbox.max_x
    if axis == "y":
        return bbox.max_y
    if axis == "z":
        return bbox.max_z
    raise SpatialQAError(f"Unsupported bbox axis: {axis}")


def _horizontal_axes(axis: str) -> tuple[str, str]:
    if axis == "x":
        return ("y", "z")
    if axis == "y":
        return ("x", "z")
    if axis == "z":
        return ("x", "y")
    raise SpatialQAError(f"Unsupported bbox axis: {axis}")


def _horizontal_overlap_area(
    src: BBox3D,
    dst: BBox3D,
    *,
    axis: str,
    margin: float,
) -> float:
    first_axis, second_axis = _horizontal_axes(axis)
    first_overlap = max(
        0.0,
        min(_bbox_max(src, first_axis), _bbox_max(dst, first_axis) + margin)
        - max(_bbox_min(src, first_axis), _bbox_min(dst, first_axis) - margin),
    )
    second_overlap = max(
        0.0,
        min(_bbox_max(src, second_axis), _bbox_max(dst, second_axis) + margin)
        - max(_bbox_min(src, second_axis), _bbox_min(dst, second_axis) - margin),
    )
    return first_overlap * second_overlap


def _horizontal_area(bbox: BBox3D, *, axis: str) -> float:
    first_axis, second_axis = _horizontal_axes(axis)
    return (_bbox_max(bbox, first_axis) - _bbox_min(bbox, first_axis)) * (
        _bbox_max(bbox, second_axis) - _bbox_min(bbox, second_axis)
    )


def _observation_steps_from_payload(observations_value: object) -> list[Any] | None:
    if not isinstance(observations_value, Sequence) or isinstance(observations_value, str):
        return None
    steps: list[Any] = []
    for item in observations_value:
        if not isinstance(item, Mapping):
            return None
        steps.append(item.get("step"))
    return steps


def _sorted_counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def _sequence_summary_projection(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in summary.items()
        if key not in {"action", "path", "valid"}
    }


def _summary_step_entries(
    summary: Mapping[str, Any],
) -> list[Mapping[str, Any]] | None:
    by_step = summary.get("by_step")
    if not isinstance(by_step, Sequence) or isinstance(by_step, str):
        return None
    entries: list[Mapping[str, Any]] = []
    for item in by_step:
        if not isinstance(item, Mapping):
            return None
        entries.append(cast(Mapping[str, Any], item))
    return entries


def _summary_steps(summary: Mapping[str, Any]) -> list[Any] | None:
    steps = summary.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, str):
        return None
    return list(steps)


def _summary_step_values(entries: list[Mapping[str, Any]] | None) -> list[Any] | None:
    if entries is None:
        return None
    return [entry.get("step") for entry in entries]


def _summary_unique_object_ids(summary: Mapping[str, Any]) -> list[str] | None:
    object_ids = summary.get("unique_object_ids")
    if not isinstance(object_ids, Sequence) or isinstance(object_ids, str):
        return None
    if not all(isinstance(item, str) for item in object_ids):
        return None
    return list(object_ids)


def _summary_step_count_check(
    summary: Mapping[str, Any],
    entries: list[Mapping[str, Any]] | None,
    summary_field: str,
    step_field: str,
    check_name: str,
) -> dict[str, Any]:
    actual = _summary_step_int_sum(entries, step_field)
    return {
        "name": check_name,
        "passed": summary.get(summary_field) == actual,
        "expected": summary.get(summary_field),
        "actual": actual,
    }


def _summary_step_int_sum(
    entries: list[Mapping[str, Any]] | None,
    field_name: str,
) -> int | None:
    if entries is None:
        return None
    total = 0
    for entry in entries:
        value = entry.get(field_name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return None
        total += value
    return total


def _summary_label_counts_valid(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return all(
        isinstance(label, str)
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
        for label, count in value.items()
    )


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


def _prediction_source_attributes(attributes: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(attributes)
    if _source_value(normalized) is not None:
        normalized["source"] = _source_value(normalized)
    return normalized


def _source_value(attributes: Mapping[str, Any]) -> str | None:
    for key in ("source", "source_name", "source_kind"):
        value = attributes.get(key)
        if isinstance(value, str) and value:
            return value
    return None


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


def _optional_string(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    default: str,
) -> str:
    value = payload.get(field_name, default)
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
