from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.schema import Pose3D, SpatialQAError


EPISODE_FRAME_SCHEMA_VERSION = "dsg-spatialqa-lab.episode-frame.v1"
EPISODE_SEQUENCE_SUMMARY_SCHEMA_VERSION = "dsg-spatialqa-lab.episode-sequence-summary.v1"


@dataclass(frozen=True)
class EpisodeAction:
    step: int
    name: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeMetadata:
    episode_id: str
    scene_id: str
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeFrame:
    episode_id: str
    scene_id: str
    step: int
    rgb_path: str | None
    depth_path: str | None
    segmentation_path: str | None
    agent_id: str
    agent_pose: Pose3D
    action: str | None
    visible_object_ids: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeRecord:
    frame: EpisodeFrame


@dataclass(frozen=True)
class EpisodeSequence:
    frames: tuple[EpisodeFrame, ...]


def episode_frame_to_dict(frame: EpisodeFrame) -> dict[str, Any]:
    return {
        "schema_version": EPISODE_FRAME_SCHEMA_VERSION,
        "episode_id": frame.episode_id,
        "scene_id": frame.scene_id,
        "step": frame.step,
        "rgb_path": frame.rgb_path,
        "depth_path": frame.depth_path,
        "segmentation_path": frame.segmentation_path,
        "agent_id": frame.agent_id,
        "agent_pose": frame.agent_pose.to_dict(),
        "action": frame.action,
        "visible_object_ids": list(frame.visible_object_ids),
        "metadata": _stable_mapping(frame.metadata),
    }


def episode_frame_from_dict(payload: Mapping[str, Any]) -> EpisodeFrame:
    schema_version = payload.get("schema_version")
    if schema_version != EPISODE_FRAME_SCHEMA_VERSION:
        raise SpatialQAError(f"Unsupported episode frame schema version: {schema_version}")
    return EpisodeFrame(
        episode_id=_required_str(payload, "episode_id"),
        scene_id=_required_str(payload, "scene_id"),
        step=_required_int(payload, "step"),
        rgb_path=_optional_str(payload, "rgb_path"),
        depth_path=_optional_str(payload, "depth_path"),
        segmentation_path=_optional_str(payload, "segmentation_path"),
        agent_id=_required_str(payload, "agent_id"),
        agent_pose=_pose_from_mapping(_as_mapping(payload.get("agent_pose"), "agent_pose")),
        action=_optional_str(payload, "action"),
        visible_object_ids=tuple(
            _sequence_of_strings(payload, "visible_object_ids")
        ),
        metadata=_stable_mapping(_as_mapping(payload.get("metadata"), "metadata")),
    )


def episode_sequence_to_jsonl(frames: Sequence[EpisodeFrame]) -> str:
    return "".join(
        json.dumps(
            episode_frame_to_dict(frame),
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
        for frame in frames
    )


def episode_sequence_from_jsonl(payload: str) -> tuple[EpisodeFrame, ...]:
    frames: list[EpisodeFrame] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if line.strip() == "":
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, Mapping):
            raise SpatialQAError(f"episode JSONL line {line_number} must be an object")
        frames.append(episode_frame_from_dict(cast(Mapping[str, Any], parsed)))
    return tuple(frames)


def save_episode_sequence(frames: Sequence[EpisodeFrame], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(episode_sequence_to_jsonl(frames), encoding="utf-8")
    return output_path


def load_episode_sequence(path: str | Path) -> tuple[EpisodeFrame, ...]:
    return episode_sequence_from_jsonl(Path(path).read_text(encoding="utf-8"))


def episode_sequence_digest(frames: Sequence[EpisodeFrame]) -> str:
    return hashlib.sha256(episode_sequence_to_jsonl(frames).encode("utf-8")).hexdigest()


def episode_sequence_summary(frames: Sequence[EpisodeFrame]) -> dict[str, Any]:
    steps = [frame.step for frame in frames]
    action_names = [frame.action for frame in frames if frame.action is not None]
    visible_object_ids = [
        object_id for frame in frames for object_id in frame.visible_object_ids
    ]
    return {
        "schema_version": EPISODE_SEQUENCE_SUMMARY_SCHEMA_VERSION,
        "frame_count": len(frames),
        "episode_count": len({frame.episode_id for frame in frames}),
        "scene_count": len({frame.scene_id for frame in frames}),
        "first_step": steps[0] if steps else None,
        "last_step": steps[-1] if steps else None,
        "episode_ids": sorted({frame.episode_id for frame in frames}),
        "scene_ids": sorted({frame.scene_id for frame in frames}),
        "action_count": len(action_names),
        "visible_object_observation_count": len(visible_object_ids),
        "unique_visible_object_ids": sorted(set(visible_object_ids)),
        "by_episode": _sorted_counts(frame.episode_id for frame in frames),
        "by_scene": _sorted_counts(frame.scene_id for frame in frames),
        "by_action": _sorted_counts(action_names),
    }


def validate_episode_sequence(frames: Sequence[EpisodeFrame]) -> dict[str, Any]:
    actual_order = [(frame.episode_id, frame.step) for frame in frames]
    expected_order = sorted(actual_order, key=lambda item: (item[0], item[1]))
    checks = [
        {
            "name": "frame_count_positive",
            "passed": len(frames) > 0,
            "expected": "at least one frame",
            "actual": len(frames),
        },
        {
            "name": "ordered_by_episode_and_step",
            "passed": actual_order == expected_order,
            "expected": expected_order,
            "actual": actual_order,
        },
        {
            "name": "unique_episode_steps",
            "passed": len(set(actual_order)) == len(actual_order),
            "expected": len(actual_order),
            "actual": len(set(actual_order)),
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "digest": episode_sequence_digest(frames),
        "summary": episode_sequence_summary(frames),
        "checks": checks,
    }


def compare_episode_sequence(frames: Sequence[EpisodeFrame]) -> dict[str, Any]:
    saved_digest = episode_sequence_digest(frames)
    round_tripped_frames = episode_sequence_from_jsonl(episode_sequence_to_jsonl(frames))
    current_digest = episode_sequence_digest(round_tripped_frames)
    validation = validate_episode_sequence(frames)
    checks = [
        {"name": "sequence_valid", "passed": validation["valid"] is True},
        {
            "name": "jsonl_round_trip_matches",
            "passed": saved_digest == current_digest,
            "expected": saved_digest,
            "actual": current_digest,
        },
    ]
    return {
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "checks": checks,
    }


def _pose_from_mapping(payload: Mapping[str, Any]) -> Pose3D:
    return Pose3D(
        _required_float(payload, "x"),
        _required_float(payload, "y"),
        _required_float(payload, "z"),
        yaw=_required_float(payload, "yaw"),
    )


def _as_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpatialQAError(f"{field_name} must be an object")
    return cast(Mapping[str, Any], value)


def _stable_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in sorted(value)}


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


def _required_int(payload: Mapping[str, Any], field_name: str) -> int:
    value = payload.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise SpatialQAError(f"{field_name} must be an integer")
    return value


def _required_float(payload: Mapping[str, Any], field_name: str) -> float:
    value = payload.get(field_name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SpatialQAError(f"{field_name} must be a number")
    return float(value)


def _sequence_of_strings(payload: Mapping[str, Any], field_name: str) -> tuple[str, ...]:
    value = payload.get(field_name)
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise SpatialQAError(f"{field_name} must be a sequence")
    if not all(isinstance(item, str) for item in value):
        raise SpatialQAError(f"{field_name} must contain only strings")
    return tuple(str(item) for item in value)


def _sorted_counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return {key: counts[key] for key in sorted(counts)}
