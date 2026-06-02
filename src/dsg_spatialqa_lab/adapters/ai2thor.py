from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from dsg_spatialqa_lab.episodes import EpisodeFrame
from dsg_spatialqa_lab.schema import Pose3D, SpatialQAError


AI2THOR_MISSING_DEPENDENCY_MESSAGE = (
    "AI2-THOR optional dependency is not installed. Install with .[ai2thor]."
)
AI2THOR_REAL_COLLECTION_OMITTED_MESSAGE = (
    "AI2-THOR collection is intentionally omitted from the deterministic lab runtime"
)


@dataclass(frozen=True)
class AI2ThorAdapterConfig:
    scene_id: str
    episode_id: str
    steps: tuple[int, ...]
    actions: tuple[str | None, ...] = field(default_factory=tuple)
    agent_id: str = "agent"
    artifact_root: str | None = None


class AI2ThorEpisodeCollector:
    def __init__(
        self,
        config: AI2ThorAdapterConfig,
        *,
        ai2thor_available: bool = False,
    ) -> None:
        self.config = config
        self.ai2thor_available = ai2thor_available

    def collect_episode(self) -> tuple[EpisodeFrame, ...]:
        _validate_config(self.config)
        if not self.ai2thor_available:
            raise SpatialQAError(AI2THOR_MISSING_DEPENDENCY_MESSAGE)
        raise SpatialQAError(AI2THOR_REAL_COLLECTION_OMITTED_MESSAGE)


def build_mock_ai2thor_episode(
    config: AI2ThorAdapterConfig,
) -> tuple[EpisodeFrame, ...]:
    _validate_config(config)
    actions = _actions_for_config(config)
    return tuple(
        convert_ai2thor_event_to_episode_frame(
            _mock_event(index=index, step=step),
            config=config,
            step=step,
            action=actions[index],
        )
        for index, step in enumerate(config.steps)
    )


def convert_ai2thor_event_to_episode_frame(
    event: Mapping[str, Any],
    *,
    config: AI2ThorAdapterConfig,
    step: int,
    action: str | None,
) -> EpisodeFrame:
    _validate_step(step)
    agent_pose = _pose_from_mapping(_as_mapping(event.get("agent_pose"), "agent_pose"))
    metadata = _stable_mapping(_optional_mapping(event, "metadata"))
    metadata["adapter"] = "ai2thor"
    return EpisodeFrame(
        episode_id=config.episode_id,
        scene_id=config.scene_id,
        step=step,
        rgb_path=_optional_str(event, "rgb_path") or _artifact_path(config, "rgb", step, "png"),
        depth_path=(
            _optional_str(event, "depth_path") or _artifact_path(config, "depth", step, "npy")
        ),
        segmentation_path=(
            _optional_str(event, "segmentation_path")
            or _artifact_path(config, "segmentation", step, "png")
        ),
        agent_id=config.agent_id,
        agent_pose=agent_pose,
        action=action,
        visible_object_ids=_sequence_of_strings(event, "visible_object_ids"),
        metadata=metadata,
    )


def _mock_event(*, index: int, step: int) -> dict[str, Any]:
    mug_x = 0.1 + (0.15 * index)
    objects = [
        _object_payload(
            object_id="mug_1",
            label="mug",
            x=mug_x,
            y=1.0,
            z=0.78,
            size=(0.12, 0.12, 0.16),
            states={"clean": True},
        ),
    ]
    visible_object_ids = ["mug_1"]
    if index == 0:
        objects.append(
            _object_payload(
                object_id="table_1",
                label="table",
                x=0.0,
                y=1.0,
                z=0.7,
                size=(1.0, 0.8, 0.1),
            )
        )
        visible_object_ids.append("table_1")
    return {
        "agent_pose": {
            "x": 0.25 * index,
            "y": 0.0,
            "z": 0.0,
            "yaw": 0.0,
        },
        "visible_object_ids": visible_object_ids,
        "metadata": {
            "rooms": [{"label": "Kitchen", "room_id": "kitchen"}],
            "regions": [
                {
                    "label": "Counter region",
                    "region_id": "counter_region",
                    "room_id": "kitchen",
                }
            ],
            "objects": objects,
            "relations": [
                {
                    "src": "mug_1",
                    "relation": "ON",
                    "dst": "table_1",
                    "confidence": 0.95,
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
        "room_id": "kitchen",
        "region_id": "counter_region",
        "states": _stable_mapping(states or {}),
    }


def _validate_config(config: AI2ThorAdapterConfig) -> None:
    if config.scene_id == "":
        raise SpatialQAError("AI2-THOR adapter scene_id must be non-empty")
    if config.episode_id == "":
        raise SpatialQAError("AI2-THOR adapter episode_id must be non-empty")
    if config.agent_id == "":
        raise SpatialQAError("AI2-THOR adapter agent_id must be non-empty")
    if not config.steps:
        raise SpatialQAError("AI2-THOR adapter steps must be explicit")
    for step in config.steps:
        _validate_step(step)
    if tuple(sorted(config.steps)) != config.steps:
        raise SpatialQAError("AI2-THOR adapter steps must be sorted")
    if len(set(config.steps)) != len(config.steps):
        raise SpatialQAError("AI2-THOR adapter steps must be unique")
    if config.actions and len(config.actions) != len(config.steps):
        raise SpatialQAError("AI2-THOR adapter actions length must match steps length")


def _validate_step(step: int) -> None:
    if not isinstance(step, int) or isinstance(step, bool):
        raise SpatialQAError("AI2-THOR adapter step must be an integer")


def _actions_for_config(config: AI2ThorAdapterConfig) -> tuple[str | None, ...]:
    if config.actions:
        return config.actions
    return tuple(None for _ in config.steps)


def _artifact_path(
    config: AI2ThorAdapterConfig,
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
