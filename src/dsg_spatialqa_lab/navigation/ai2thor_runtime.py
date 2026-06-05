from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.navigation.action_planner import (
    ReachablePosition,
    reachable_positions_from_mappings,
)
from dsg_spatialqa_lab.navigation.reachable_nbv import CandidateObservation
from dsg_spatialqa_lab.observations import (
    NodeObservation,
    ObjectObservation,
    SceneObservation,
)
from dsg_spatialqa_lab.schema import BBox3D, Pose3D, SpatialQAError


SUPPORT_LABELS = frozenset(
    {
        "bathtub",
        "bed",
        "cabinet",
        "chair",
        "coffeetable",
        "countertop",
        "desk",
        "diningtable",
        "dresser",
        "floor",
        "shelf",
        "sink",
        "sofa",
        "table",
    }
)
STATE_KEYS = (
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
INSIDE_RECEPTACLE_TYPES = frozenset(
    {"box", "cabinet", "drawer", "fridge", "garbagecan", "microwave", "safe"}
)
DETECTOR_NAME = "ai2thor_reachable_nbv_rgbd"


def create_ai2thor_controller(
    *,
    scene_id: str,
    width: int,
    height: int,
    grid_size: float = 0.25,
    rotate_step: float = 90.0,
    visibility_distance: float = 2.0,
    platform: str | None = "CloudRendering",
) -> object:
    try:
        from ai2thor.controller import Controller  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SpatialQAError(
            "AI2-THOR optional dependency is not installed. Run with an environment "
            "that provides ai2thor, for example SimTools .venv-ai2thor."
        ) from exc

    kwargs: dict[str, Any] = {
        "gridSize": grid_size,
        "height": height,
        "renderDepthImage": True,
        "renderInstanceSegmentation": True,
        "rotateStep": rotate_step,
        "scene": scene_id,
        "visibilityDistance": visibility_distance,
        "width": width,
    }
    if platform == "CloudRendering":
        try:
            from ai2thor.platform import CloudRendering  # type: ignore[import-not-found]
        except ImportError:
            pass
        else:
            kwargs["platform"] = CloudRendering
    return cast(object, Controller(**kwargs))


def stop_ai2thor_controller(controller: object) -> None:
    stop = getattr(controller, "stop", None)
    if callable(stop):
        cast(Any, stop)()


def get_ai2thor_reachable_positions(
    controller: object,
) -> tuple[tuple[ReachablePosition, ...], object]:
    event = _controller_step(controller, action="GetReachablePositions")
    metadata = _metadata(event)
    if metadata.get("lastActionSuccess") is not True:
        raise SpatialQAError(
            "GetReachablePositions failed: "
            f"{metadata.get('errorMessage', 'missing errorMessage')}"
        )
    positions = reachable_positions_from_mappings(
        item
        for item in _sequence(metadata.get("actionReturn"))
        if isinstance(item, Mapping)
    )
    if not positions:
        raise SpatialQAError("GetReachablePositions returned no reachable positions")
    return positions, event


def execute_ai2thor_actions(
    controller: object,
    actions: Sequence[str],
) -> tuple[tuple[dict[str, Any], ...], tuple[object, ...]]:
    records: list[dict[str, Any]] = []
    events: list[object] = []
    for action in actions:
        if action == "TeleportFull":
            raise SpatialQAError("TeleportFull is forbidden in formal reachable NBV")
        event = _controller_step(controller, action=action)
        events.append(event)
        metadata = _metadata(event)
        success = metadata.get("lastActionSuccess") is True
        record: dict[str, Any] = {
            "action": action,
            "lastActionSuccess": success,
            "success": success,
        }
        error = metadata.get("errorMessage")
        if isinstance(error, str) and error:
            record["errorMessage"] = error
        records.append(record)
        if not success:
            break
    return tuple(records), tuple(events)


def real_ai2thor_candidate_priors(
    positions: Sequence[ReachablePosition],
) -> dict[str, CandidateObservation]:
    return {
        f"{position.x:.2f}:{position.z:.2f}": CandidateObservation(
            unseen_region_ids=frozenset({f"{position.x:.2f}:{position.z:.2f}"}),
        )
        for position in positions
    }


def ai2thor_agent_pose(event: object) -> Pose3D:
    metadata = _metadata(event)
    agent = _mapping(metadata.get("agent"))
    position = _mapping(agent.get("position"))
    rotation = _mapping(agent.get("rotation"))
    return Pose3D(
        _number(position.get("x")),
        _number(position.get("y")),
        _number(position.get("z")),
        yaw=_number(rotation.get("y", 0.0)),
    )


def ai2thor_event_to_observation(
    event: object,
    *,
    scene_id: str,
    episode_id: str,
    step: int,
    artifact_root: Path,
) -> SceneObservation:
    metadata = _metadata(event)
    agent_pose = ai2thor_agent_pose(event)
    frame_paths = _write_event_artifacts(
        event,
        artifact_root=artifact_root,
        episode_id=episode_id,
        step=step,
    )
    visible_objects = _dedupe_visible_objects(
        cast(Mapping[str, Any], item)
        for item in _sequence(metadata.get("objects"))
        if isinstance(item, Mapping) and item.get("visible") is True
    )
    visible_ids = {
        stable_ai2thor_object_id(_required_str(item, "objectId"))
        for item in visible_objects
    }
    objects = tuple(
        _object_observation_from_ai2thor(
            item,
            frame_paths=frame_paths,
            visible_object_ids=visible_ids,
        )
        for item in visible_objects
    )
    return SceneObservation(
        step=step,
        agent_pose=agent_pose,
        rooms=(
            NodeObservation(
                "ai2thor_room",
                scene_id,
                {
                    "episode_id": episode_id,
                    "scene_id": scene_id,
                    "source_kind": "detector",
                    "source_name": DETECTOR_NAME,
                },
            ),
        ),
        regions=(
            NodeObservation(
                f"visible_frame_region:{episode_id}:{step:06d}",
                "visible frame region",
                {
                    "episode_id": episode_id,
                    "scene_id": scene_id,
                    "source_kind": "detector",
                    "source_name": DETECTOR_NAME,
                },
            ),
        ),
        objects=objects,
    )


def candidate_observation_from_observations(
    observations: Sequence[SceneObservation],
) -> CandidateObservation:
    object_ids: set[str] = set()
    support_ids: set[str] = set()
    labels: dict[str, str] = {}
    locations: dict[str, str] = {}
    relation_keys: set[str] = set()
    for observation in observations:
        visible_ids = {obj.object_id for obj in observation.objects}
        for obj in observation.objects:
            object_ids.add(obj.object_id)
            labels[obj.object_id] = obj.label
            if obj.label.lower() in SUPPORT_LABELS:
                support_ids.add(obj.object_id)
        for obj in observation.objects:
            relation = obj.attributes.get("current_location_relation")
            location_id = obj.attributes.get("current_location_id")
            if (
                isinstance(relation, str)
                and relation in {"INSIDE", "ON"}
                and isinstance(location_id, str)
                and location_id in visible_ids
            ):
                key = f"{obj.object_id}-{relation}-{location_id}"
                relation_keys.add(key)
                locations[obj.object_id] = location_id
    return CandidateObservation(
        object_ids=frozenset(object_ids),
        support_ids=frozenset(support_ids),
        same_frame_relations=frozenset(relation_keys),
        current_location_edges=frozenset(relation_keys),
        state_object_ids=frozenset(object_ids),
        object_labels=labels,
        object_locations=locations,
    )


def stable_ai2thor_object_id(object_id: str) -> str:
    if not isinstance(object_id, str) or object_id == "":
        raise SpatialQAError("AI2-THOR objectId must be a non-empty string")
    if "|" not in object_id:
        return _safe_token(object_id)
    parts = object_id.split("|")
    coordinates = [_coordinate_token(piece) for piece in parts[1:4]]
    if len(coordinates) != 3 or any(piece == "" for piece in coordinates):
        return _safe_token(object_id)
    suffixes = [_safe_token(piece) for piece in parts[4:] if piece != ""]
    return "_".join([_safe_token(parts[0]), *coordinates, *suffixes])


def _dedupe_visible_objects(
    objects: Sequence[Mapping[str, Any]] | Any,
) -> tuple[Mapping[str, Any], ...]:
    by_stable_id: dict[str, Mapping[str, Any]] = {}
    for obj in sorted(
        tuple(objects),
        key=lambda payload: (
            stable_ai2thor_object_id(_required_str(payload, "objectId")),
            _required_str(payload, "objectId"),
        ),
    ):
        stable_id = stable_ai2thor_object_id(_required_str(obj, "objectId"))
        by_stable_id.setdefault(stable_id, obj)
    return tuple(by_stable_id[key] for key in sorted(by_stable_id))


def _object_observation_from_ai2thor(
    payload: Mapping[str, Any],
    *,
    frame_paths: Mapping[str, str],
    visible_object_ids: set[str],
) -> ObjectObservation:
    raw_id = _required_str(payload, "objectId")
    object_id = stable_ai2thor_object_id(raw_id)
    label = _required_str(payload, "objectType").lower()
    bbox_payload = _mapping(payload.get("axisAlignedBoundingBox"))
    center = _mapping(bbox_payload.get("center"))
    size = _mapping(bbox_payload.get("size"))
    position = _mapping(payload.get("position"))
    pose = Pose3D(
        _number(position.get("x", center.get("x", 0.0))),
        _number(position.get("y", center.get("y", 0.0))),
        _number(position.get("z", center.get("z", 0.0))),
        yaw=_rotation_y(payload.get("rotation")),
    )
    center_pose = Pose3D(
        _number(center.get("x", position.get("x", 0.0))),
        _number(center.get("y", position.get("y", 0.0))),
        _number(center.get("z", position.get("z", 0.0))),
        yaw=_rotation_y(center.get("rotation")),
    )
    attributes: dict[str, Any] = {
        "ai2thor_object_id": raw_id,
        "collection_source": DETECTOR_NAME,
        "depth_path": frame_paths["depth_path"],
        "detector": DETECTOR_NAME,
        "evidence_kinds": ["depth", "detector", "rgb"],
        "rgb_path": frame_paths["rgb_path"],
        "segmentation_path": frame_paths["segmentation_path"],
        "source_kind": "detector",
        "source_name": DETECTOR_NAME,
        "states": {
            key: payload[key]
            for key in STATE_KEYS
            if isinstance(payload.get(key), bool | int | float | str)
        },
    }
    current_location = _current_location_from_parent(
        payload,
        visible_object_ids=visible_object_ids,
    )
    if current_location is not None:
        attributes.update(current_location)
    return ObjectObservation(
        object_id=object_id,
        label=label,
        pose=pose,
        bbox=BBox3D(
            center_pose,
            (
                _number(size.get("x", 0.01)),
                _number(size.get("y", 0.01)),
                _number(size.get("z", 0.01)),
            ),
        ),
        confidence=1.0,
        visible=True,
        attributes=attributes,
    )


def _current_location_from_parent(
    payload: Mapping[str, Any],
    *,
    visible_object_ids: set[str],
) -> dict[str, str]:
    parents = _sequence(payload.get("parentReceptacles"))
    parent = next((item for item in parents if isinstance(item, str) and item), None)
    if parent is None:
        return {
            "current_location_id": "ai2thor_room",
            "current_location_relation": "IN_ROOM",
        }
    parent_id = stable_ai2thor_object_id(parent)
    parent_label = parent.split("|", 1)[0].lower()
    if parent_label == "floor" or parent_id not in visible_object_ids:
        return {
            "current_location_id": "ai2thor_room",
            "current_location_relation": "IN_ROOM",
        }
    return {
        "current_location_id": parent_id,
        "current_location_relation": (
            "INSIDE" if parent_label in INSIDE_RECEPTACLE_TYPES else "ON"
        ),
    }


def _write_event_artifacts(
    event: object,
    *,
    artifact_root: Path,
    episode_id: str,
    step: int,
) -> dict[str, str]:
    frame_root = artifact_root / episode_id / f"{step:06d}"
    frame_root.mkdir(parents=True, exist_ok=True)
    rgb_path = frame_root / "rgb.ppm"
    depth_path = frame_root / "depth.npy"
    segmentation_path = frame_root / "segmentation.ppm"
    _write_ppm(_required_attr(event, "frame"), rgb_path)
    _write_depth(_required_attr(event, "depth_frame"), depth_path)
    _write_ppm(_required_attr(event, "instance_segmentation_frame"), segmentation_path)
    return {
        "depth_path": str(depth_path),
        "rgb_path": str(rgb_path),
        "segmentation_path": str(segmentation_path),
    }


def _controller_step(controller: object, *, action: str, **kwargs: Any) -> object:
    step = getattr(controller, "step", None)
    if not callable(step):
        raise SpatialQAError("AI2-THOR controller must expose step()")
    return cast(Any, step)(action=action, **kwargs)


def _metadata(event: object) -> Mapping[str, Any]:
    metadata = getattr(event, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise SpatialQAError("AI2-THOR event.metadata must be an object")
    return cast(Mapping[str, Any], metadata)


def _required_attr(event: object, name: str) -> object:
    sentinel = object()
    value = getattr(event, name, sentinel)
    if value is sentinel:
        raise SpatialQAError(f"AI2-THOR event.{name} is required")
    return value


def _write_ppm(image: object, path: Path) -> None:
    rows = _image_rows(image)
    height = len(rows)
    width = len(rows[0]) if rows else 0
    payload = bytearray()
    for row in rows:
        for red, green, blue in row:
            payload.extend((red, green, blue))
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + bytes(payload))


def _write_depth(depth: object, path: Path) -> None:
    try:
        import numpy as np
    except ImportError:
        path.write_text(
            json.dumps(_jsonable(depth), separators=(",", ":"), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    np.save(path, cast(Any, depth))


def _image_rows(image: object) -> list[list[tuple[int, int, int]]]:
    jsonable = _jsonable(image)
    if isinstance(jsonable, str) or not isinstance(jsonable, Sequence):
        raise SpatialQAError("AI2-THOR image must be an array-like RGB image")
    rows: list[list[tuple[int, int, int]]] = []
    for row in jsonable:
        if isinstance(row, str) or not isinstance(row, Sequence):
            raise SpatialQAError("AI2-THOR image rows must be sequences")
        rows.append([_rgb(pixel) for pixel in row])
    return rows


def _rgb(value: object) -> tuple[int, int, int]:
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) < 3:
        raise SpatialQAError("AI2-THOR RGB pixel must contain three channels")
    channels = tuple(int(channel) for channel in value[:3])
    if any(channel < 0 or channel > 255 for channel in channels):
        raise SpatialQAError("AI2-THOR RGB channels must be in 0..255")
    return cast(tuple[int, int, int], channels)


def _jsonable(value: object) -> object:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return cast(Any, tolist)()
    return value


def _coordinate_token(value: str) -> str:
    cleaned = value.strip().replace("+", "").replace("-", "")
    try:
        number = abs(float(cleaned))
    except ValueError:
        return _safe_token(value)
    integer = int(number)
    fraction = int(round((number - integer) * 100))
    if fraction == 100:
        integer += 1
        fraction = 0
    return f"{integer:02d}_{fraction:02d}"


def _safe_token(value: str) -> str:
    cleaned = "".join(
        char.lower() if char.isalnum() else "_" for char in value
    ).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned


def _mapping(value: object) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _rotation_y(value: object) -> float:
    if isinstance(value, Mapping):
        return _number(value.get("y", 0.0))
    if value is None:
        return 0.0
    return _number(value)


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SpatialQAError("expected numeric AI2-THOR field")
    number = float(value)
    if not math.isfinite(number):
        raise SpatialQAError("AI2-THOR numeric field must be finite")
    return number


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"missing required AI2-THOR string field: {key}")
    return value
