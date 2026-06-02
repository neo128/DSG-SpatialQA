from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from dsg_spatialqa_lab.episodes import EpisodeFrame
from dsg_spatialqa_lab.schema import Pose3D, SpatialQAError


HABITAT_MISSING_DEPENDENCY_MESSAGE = (
    "Habitat optional dependency is not installed. Install with .[habitat]."
)
HABITAT_REAL_COLLECTION_OMITTED_MESSAGE = (
    "Habitat collection is intentionally omitted from the deterministic lab runtime"
)


@dataclass(frozen=True)
class HabitatAdapterConfig:
    scene_id: str
    episode_id: str
    steps: tuple[int, ...]
    actions: tuple[str | None, ...] = field(default_factory=tuple)
    agent_id: str = "agent"
    artifact_root: str | None = None


class HabitatEpisodeCollector:
    def __init__(
        self,
        config: HabitatAdapterConfig,
        *,
        habitat_available: bool = False,
    ) -> None:
        self.config = config
        self.habitat_available = habitat_available

    def collect_episode(self) -> tuple[EpisodeFrame, ...]:
        _validate_config(self.config)
        if not self.habitat_available:
            raise SpatialQAError(HABITAT_MISSING_DEPENDENCY_MESSAGE)
        raise SpatialQAError(HABITAT_REAL_COLLECTION_OMITTED_MESSAGE)


def build_mock_habitat_episode(
    config: HabitatAdapterConfig,
) -> tuple[EpisodeFrame, ...]:
    _validate_config(config)
    actions = _actions_for_config(config)
    return tuple(
        convert_habitat_observation_to_episode_frame(
            _mock_observation(index=index, step=step),
            config=config,
            step=step,
            action=actions[index],
        )
        for index, step in enumerate(config.steps)
    )


def convert_habitat_observation_to_episode_frame(
    observation: Mapping[str, Any],
    *,
    config: HabitatAdapterConfig,
    step: int,
    action: str | None,
) -> EpisodeFrame:
    _validate_step(step)
    agent_pose = _pose_from_mapping(_as_mapping(observation.get("agent_pose"), "agent_pose"))
    metadata = _stable_mapping(_optional_mapping(observation, "metadata"))
    metadata["adapter"] = "habitat"
    return EpisodeFrame(
        episode_id=config.episode_id,
        scene_id=config.scene_id,
        step=step,
        rgb_path=_optional_str(observation, "rgb_path") or _artifact_path(config, "rgb", step, "png"),
        depth_path=(
            _optional_str(observation, "depth_path") or _artifact_path(config, "depth", step, "npy")
        ),
        segmentation_path=(
            _optional_str(observation, "segmentation_path")
            or _artifact_path(config, "segmentation", step, "png")
        ),
        agent_id=config.agent_id,
        agent_pose=agent_pose,
        action=action,
        visible_object_ids=_sequence_of_strings(observation, "visible_object_ids"),
        metadata=metadata,
    )


def _mock_observation(*, index: int, step: int) -> dict[str, Any]:
    chair_x = 0.3 + (0.1 * index)
    objects = [
        _object_payload(
            object_id="chair_1",
            label="chair",
            x=chair_x,
            y=0.5,
            z=0.45,
            size=(0.45, 0.45, 0.9),
            states={"occupied": False},
        ),
    ]
    visible_object_ids = ["chair_1"]
    if index == 0:
        objects.append(
            _object_payload(
                object_id="table_1",
                label="table",
                x=0.0,
                y=0.5,
                z=0.42,
                size=(0.8, 0.6, 0.84),
            )
        )
        visible_object_ids.append("table_1")
    return {
        "agent_pose": {
            "x": 0.15 * index,
            "y": 0.0,
            "z": 0.05 * index,
            "yaw": 30.0 * index,
        },
        "visible_object_ids": visible_object_ids,
        "metadata": {
            "rooms": [{"label": "Living room", "room_id": "living_room"}],
            "regions": [
                {
                    "label": "Seating region",
                    "region_id": "seating_region",
                    "room_id": "living_room",
                }
            ],
            "objects": objects,
            "relations": [
                {
                    "src": "chair_1",
                    "relation": "NEAR",
                    "dst": "table_1",
                    "confidence": 0.92,
                    "reference_frame": "world",
                }
            ],
            "source_step": step,
        },
    }


def _object_payload(
    *,
    object_id: str,
    label: str,
    x: float,
    y: float,
    z: float,
    size: tuple[float, float, float],
    states: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "object_id": object_id,
        "label": label,
        "pose": {"x": x, "y": y, "z": z, "yaw": 0.0},
        "bbox": {
            "center": {"x": x, "y": y, "z": z, "yaw": 0.0},
            "size": list(size),
        },
        "confidence": 0.9,
        "visible": True,
        "room_id": "living_room",
        "region_id": "seating_region",
        "states": _stable_mapping(states or {}),
    }


def _validate_config(config: HabitatAdapterConfig) -> None:
    if config.scene_id == "":
        raise SpatialQAError("Habitat adapter scene_id must be non-empty")
    if config.episode_id == "":
        raise SpatialQAError("Habitat adapter episode_id must be non-empty")
    if config.agent_id == "":
        raise SpatialQAError("Habitat adapter agent_id must be non-empty")
    if not config.steps:
        raise SpatialQAError("Habitat adapter steps must be explicit")
    for step in config.steps:
        _validate_step(step)
    if tuple(sorted(config.steps)) != config.steps:
        raise SpatialQAError("Habitat adapter steps must be sorted")
    if len(set(config.steps)) != len(config.steps):
        raise SpatialQAError("Habitat adapter steps must be unique")
    if config.actions and len(config.actions) != len(config.steps):
        raise SpatialQAError("Habitat adapter actions length must match steps length")


def _validate_step(step: int) -> None:
    if not isinstance(step, int) or isinstance(step, bool):
        raise SpatialQAError("Habitat adapter step must be an integer")


def _actions_for_config(config: HabitatAdapterConfig) -> tuple[str | None, ...]:
    if config.actions:
        return config.actions
    return tuple(None for _ in config.steps)


def _artifact_path(
    config: HabitatAdapterConfig,
    stream: str,
    step: int,
    suffix: str,
) -> str | None:
    if config.artifact_root is None:
        return None
    root = config.artifact_root.rstrip("/")
    return f"{root}/{config.episode_id}/{stream}/{step:04d}.{suffix}"


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


def _optional_mapping(payload: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    value = payload.get(field_name, {})
    if not isinstance(value, Mapping):
        raise SpatialQAError(f"{field_name} must be an object")
    return cast(Mapping[str, Any], value)


def _stable_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value[key] for key in sorted(value)}


def _optional_str(payload: Mapping[str, Any], field_name: str) -> str | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SpatialQAError(f"{field_name} must be a string or null")
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
