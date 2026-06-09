from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import importlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.episodes import EpisodeFrame
from dsg_spatialqa_lab.schema import Pose3D, SpatialQAError


AI2THOR_MISSING_DEPENDENCY_MESSAGE = (
    "AI2-THOR optional dependency is not installed. Install with .[ai2thor]."
)
AI2THOR_REAL_COLLECTION_REQUIRES_ARTIFACT_ROOT_MESSAGE = (
    "AI2-THOR real collection requires artifact_root to save RGB/depth/segmentation "
    "artifacts"
)

ControllerFactory = Callable[..., object]


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
        controller_factory: ControllerFactory | None = None,
    ) -> None:
        self.config = config
        self.ai2thor_available = ai2thor_available
        self.controller_factory = controller_factory

    def collect_episode(self) -> tuple[EpisodeFrame, ...]:
        _validate_config(self.config)
        if not self.ai2thor_available:
            raise SpatialQAError(AI2THOR_MISSING_DEPENDENCY_MESSAGE)
        controller_factory = self.controller_factory or _ai2thor_controller_factory()
        if self.config.artifact_root is None or self.config.artifact_root == "":
            raise SpatialQAError(AI2THOR_REAL_COLLECTION_REQUIRES_ARTIFACT_ROOT_MESSAGE)
        actions = _real_actions_for_config(self.config)
        controller = controller_factory(
            scene=self.config.scene_id,
            renderDepthImage=True,
            renderInstanceSegmentation=True,
        )
        frames: list[EpisodeFrame] = []
        try:
            for index, step in enumerate(self.config.steps):
                action = actions[index]
                event = _controller_step(controller, action)
                frames.append(
                    convert_ai2thor_event_to_episode_frame(
                        _real_event_payload(event, config=self.config, step=step),
                        config=self.config,
                        step=step,
                        action=action,
                    )
                )
        finally:
            _stop_controller(controller)
        return tuple(frames)


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


def _real_actions_for_config(config: AI2ThorAdapterConfig) -> tuple[str, ...]:
    actions = _actions_for_config(config)
    if any(not isinstance(action, str) or action == "" for action in actions):
        raise SpatialQAError("AI2-THOR real collection requires one explicit action per step")
    return cast(tuple[str, ...], actions)


def _ai2thor_controller_factory() -> ControllerFactory:
    try:
        module = importlib.import_module("ai2thor.controller")
    except ImportError as exc:
        raise SpatialQAError(AI2THOR_MISSING_DEPENDENCY_MESSAGE) from exc
    controller = getattr(module, "Controller", None)
    if not callable(controller):
        raise SpatialQAError(AI2THOR_MISSING_DEPENDENCY_MESSAGE)
    return cast(ControllerFactory, controller)


def _controller_step(controller: object, action: str) -> object:
    step_fn = getattr(controller, "step", None)
    if not callable(step_fn):
        raise SpatialQAError("AI2-THOR controller must expose a callable step method")
    return cast(Callable[..., object], step_fn)(action=action)


def _stop_controller(controller: object) -> None:
    stop_fn = getattr(controller, "stop", None)
    if callable(stop_fn):
        cast(Callable[[], object], stop_fn)()


def _real_event_payload(
    event: object,
    *,
    config: AI2ThorAdapterConfig,
    step: int,
) -> dict[str, Any]:
    metadata = _event_metadata(event)
    agent = _as_mapping(metadata.get("agent"), "AI2-THOR event metadata.agent")
    position = _as_mapping(
        agent.get("position"),
        "AI2-THOR event metadata.agent.position",
    )
    rotation = _as_mapping(
        agent.get("rotation"),
        "AI2-THOR event metadata.agent.rotation",
    )
    objects = _objects_from_ai2thor_metadata(metadata)
    visible_object_ids = [
        _required_str(obj, "object_id") for obj in objects if obj.get("visible") is True
    ]
    segmentation_color_map = _segmentation_color_map(event)
    artifact_paths = _write_real_event_artifacts(event, config=config, step=step)
    segmentation_metadata: dict[str, Any] = {
        "segmentation_color_map": segmentation_color_map,
        "segmentation_color_map_available": bool(segmentation_color_map),
        "segmentation_source": "ai2thor_instance_segmentation_frame",
    }
    if not segmentation_color_map:
        segmentation_metadata["segmentation_color_map_unavailable_reason"] = (
            "ai2thor_event_color_maps_empty"
        )
    return {
        "agent_pose": {
            "x": _required_float(position, "x"),
            "y": _required_float(position, "y"),
            "z": _required_float(position, "z"),
            "yaw": _required_float(rotation, "y"),
        },
        "rgb_path": artifact_paths["rgb_path"],
        "depth_path": artifact_paths["depth_path"],
        "segmentation_path": artifact_paths["segmentation_path"],
        "visible_object_ids": visible_object_ids,
        "metadata": {
            "collection_kind": "real",
            "objects": objects,
            "simulator": "ai2thor",
            "source_kind": "real_simulator",
            "source_step": step,
            **segmentation_metadata,
        },
    }


def _event_metadata(event: object) -> Mapping[str, Any]:
    metadata = getattr(event, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise SpatialQAError("AI2-THOR event.metadata must be an object")
    return cast(Mapping[str, Any], metadata)


def _objects_from_ai2thor_metadata(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = metadata.get("objects")
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise SpatialQAError("AI2-THOR event metadata.objects must be a sequence")
    objects: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise SpatialQAError(
                f"AI2-THOR event metadata.objects[{index}] must be an object"
            )
        objects.append(_object_from_ai2thor_metadata(cast(Mapping[str, Any], item)))
    return objects


def _object_from_ai2thor_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    object_id = _required_str(payload, "objectId")
    label = _required_str(payload, "objectType")
    position = _as_mapping(payload.get("position"), "AI2-THOR object position")
    bbox = _as_mapping(
        payload.get("axisAlignedBoundingBox"),
        "AI2-THOR object axisAlignedBoundingBox",
    )
    bbox_center = _as_mapping(
        bbox.get("center"),
        "AI2-THOR object axisAlignedBoundingBox.center",
    )
    bbox_size = _as_mapping(
        bbox.get("size"),
        "AI2-THOR object axisAlignedBoundingBox.size",
    )
    visible = payload.get("visible")
    if not isinstance(visible, bool):
        raise SpatialQAError("AI2-THOR object visible must be a boolean")
    yaw = _optional_rotation_y(payload.get("rotation"))
    center_yaw = _optional_rotation_y(bbox_center.get("rotation"))
    return {
        "bbox": {
            "center": {
                "x": _required_float(bbox_center, "x"),
                "y": _required_float(bbox_center, "y"),
                "z": _required_float(bbox_center, "z"),
                "yaw": center_yaw,
            },
            "size": [
                _required_float(bbox_size, "x"),
                _required_float(bbox_size, "y"),
                _required_float(bbox_size, "z"),
            ],
        },
        "confidence": 1.0,
        "label": label,
        "object_id": object_id,
        "pose": {
            "x": _required_float(position, "x"),
            "y": _required_float(position, "y"),
            "z": _required_float(position, "z"),
            "yaw": yaw,
        },
        "states": _ai2thor_object_states(payload),
        "visible": visible,
    }


def _ai2thor_object_states(payload: Mapping[str, Any]) -> dict[str, Any]:
    state_keys = (
        "breakable",
        "canFillWithLiquid",
        "cookable",
        "dirtyable",
        "isBroken",
        "isCooked",
        "isDirty",
        "isFilledWithLiquid",
        "isOpen",
        "isPickedUp",
        "isSliced",
        "isToggled",
        "moveable",
        "openable",
        "pickupable",
        "receptacle",
        "sliceable",
        "toggleable",
    )
    states: dict[str, Any] = {}
    for key in state_keys:
        value = payload.get(key)
        if isinstance(value, (bool, int, float, str)):
            states[key] = value
    return _stable_mapping(states)


def _segmentation_color_map(event: object) -> list[dict[str, Any]]:
    color_to_object_id = getattr(event, "color_to_object_id", None)
    if isinstance(color_to_object_id, Mapping) and color_to_object_id:
        entries = [
            {
                "object_id": _segmentation_object_id(object_id),
                "rgb": list(_segmentation_rgb(color, "AI2-THOR color_to_object_id key")),
            }
            for color, object_id in color_to_object_id.items()
        ]
        return sorted(entries, key=lambda item: (item["rgb"], str(item["object_id"])))

    object_id_to_color = getattr(event, "object_id_to_color", None)
    if isinstance(object_id_to_color, Mapping) and object_id_to_color:
        entries = [
            {
                "object_id": _segmentation_object_id(object_id),
                "rgb": list(
                    _segmentation_rgb(color, "AI2-THOR object_id_to_color value")
                ),
            }
            for object_id, color in object_id_to_color.items()
        ]
        return sorted(entries, key=lambda item: (item["rgb"], str(item["object_id"])))

    return []


def _segmentation_object_id(value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise SpatialQAError("AI2-THOR segmentation color map object id must be a string")
    return value


def _segmentation_rgb(value: object, field_name: str) -> tuple[int, int, int]:
    if isinstance(value, str):
        pieces = value.replace("(", "").replace(")", "").split(",")
        if len(pieces) == 3:
            try:
                return tuple(_segmentation_channel(int(piece.strip()), field_name) for piece in pieces)  # type: ignore[return-value]
            except ValueError as exc:
                raise SpatialQAError(f"{field_name} must contain RGB channels") from exc
        raise SpatialQAError(f"{field_name} must contain RGB channels")
    if isinstance(value, Sequence):
        if len(value) < 3:
            raise SpatialQAError(f"{field_name} must contain RGB channels")
        channels = tuple(
            _segmentation_channel(channel, field_name) for channel in value[:3]
        )
        return cast(tuple[int, int, int], channels)
    raise SpatialQAError(f"{field_name} must contain RGB channels")


def _segmentation_channel(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SpatialQAError(f"{field_name} channels must be integers")
    if value < 0 or value > 255:
        raise SpatialQAError(f"{field_name} channels must be in 0..255")
    return value


def _optional_rotation_y(value: object) -> float:
    if value is None:
        return 0.0
    if not isinstance(value, Mapping):
        raise SpatialQAError("AI2-THOR rotation must be an object")
    rotation = cast(Mapping[str, Any], value)
    y_value = rotation.get("y")
    if y_value is None:
        return 0.0
    if not isinstance(y_value, (int, float)) or isinstance(y_value, bool):
        raise SpatialQAError("AI2-THOR rotation.y must be a number")
    return float(y_value)


def _write_real_event_artifacts(
    event: object,
    *,
    config: AI2ThorAdapterConfig,
    step: int,
) -> dict[str, str]:
    rgb_path = _artifact_path(config, "rgb", step, "ppm")
    depth_path = _artifact_path(config, "depth", step, "npy")
    segmentation_path = _artifact_path(config, "segmentation", step, "ppm")
    if rgb_path is None or depth_path is None or segmentation_path is None:
        raise SpatialQAError(AI2THOR_REAL_COLLECTION_REQUIRES_ARTIFACT_ROOT_MESSAGE)
    _write_ppm_artifact(
        _required_event_attr(event, "frame"),
        Path(rgb_path),
        "AI2-THOR event.frame",
    )
    _write_depth_artifact(
        _required_event_attr(event, "depth_frame"),
        Path(depth_path),
    )
    _write_ppm_artifact(
        _required_event_attr(event, "instance_segmentation_frame"),
        Path(segmentation_path),
        "AI2-THOR event.instance_segmentation_frame",
    )
    return {
        "depth_path": depth_path,
        "rgb_path": rgb_path,
        "segmentation_path": segmentation_path,
    }


def _required_event_attr(event: object, name: str) -> object:
    sentinel = object()
    value = getattr(event, name, sentinel)
    if value is sentinel:
        raise SpatialQAError(f"{name} is required for AI2-THOR real collection")
    return value


def _write_ppm_artifact(value: object, path: Path, field_name: str) -> None:
    rows = _pixel_rows(value, field_name)
    height = len(rows)
    width = len(rows[0]) if rows else 0
    if height == 0 or width == 0:
        raise SpatialQAError(f"{field_name} must contain at least one pixel")
    if any(len(row) != width for row in rows):
        raise SpatialQAError(f"{field_name} rows must have stable width")
    path.parent.mkdir(parents=True, exist_ok=True)
    channel_lines = [
        " ".join(str(channel) for pixel in row for channel in pixel)
        for row in rows
    ]
    path.write_text(
        f"P3\n{width} {height}\n255\n" + "\n".join(channel_lines) + "\n",
        encoding="utf-8",
    )


def _write_depth_artifact(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _is_array_like(value) and not isinstance(value, (list, tuple)):
        try:
            import numpy as np
        except ImportError:
            pass
        else:
            np.save(path, cast(Any, value))
            return
    path.write_text(
        json.dumps(_jsonable(value), separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _pixel_rows(value: object, field_name: str) -> list[list[tuple[int, int, int]]]:
    jsonable = _jsonable(value)
    if isinstance(jsonable, str) or not isinstance(jsonable, Sequence):
        raise SpatialQAError(f"{field_name} must be a pixel matrix")
    rows: list[list[tuple[int, int, int]]] = []
    for row_index, row in enumerate(jsonable):
        if isinstance(row, str) or not isinstance(row, Sequence):
            raise SpatialQAError(f"{field_name}[{row_index}] must be a pixel row")
        pixels: list[tuple[int, int, int]] = []
        for column_index, pixel in enumerate(row):
            if isinstance(pixel, str) or not isinstance(pixel, Sequence):
                raise SpatialQAError(
                    f"{field_name}[{row_index}][{column_index}] must be an RGB pixel"
                )
            if len(pixel) < 3:
                raise SpatialQAError(
                    f"{field_name}[{row_index}][{column_index}] must contain RGB channels"
                )
            channels = tuple(
                _pixel_channel(channel, field_name, row_index, column_index)
                for channel in pixel[:3]
            )
            pixels.append(cast(tuple[int, int, int], channels))
        rows.append(pixels)
    return rows


def _pixel_channel(
    value: object,
    field_name: str,
    row_index: int,
    column_index: int,
) -> int:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SpatialQAError(
            f"{field_name}[{row_index}][{column_index}] channels must be numbers"
        )
    channel = int(value)
    if channel < 0 or channel > 255:
        raise SpatialQAError(
            f"{field_name}[{row_index}][{column_index}] channels must be in 0..255"
        )
    return channel


def _jsonable(value: object) -> object:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return cast(Callable[[], object], tolist)()
    return value


def _is_array_like(value: object) -> bool:
    return hasattr(value, "__array_interface__") or hasattr(value, "shape")


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


def _required_str(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"{field_name} must be a non-empty string")
    return value


def _sequence_of_strings(payload: Mapping[str, Any], field_name: str) -> tuple[str, ...]:
    value = payload.get(field_name)
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise SpatialQAError(f"{field_name} must be a sequence")
    if not all(isinstance(item, str) for item in value):
        raise SpatialQAError(f"{field_name} must contain only strings")
    return tuple(str(item) for item in value)
