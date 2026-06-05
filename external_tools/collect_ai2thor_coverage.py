from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any, cast


DETECTOR_SCHEMA_VERSION = "dsg-spatialqa-lab.detector-observation-record.v1"
DETECTOR_NAME = "ai2thor_visible_segmentation_rgbd"
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
    {
        "box",
        "cabinet",
        "drawer",
        "fridge",
        "garbagecan",
        "microwave",
        "safe",
    }
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Execute coverage top-batch AI2-THOR viewpoints and export visible "
            "RGB-D/instance-segmentation detector observations. This opt-in tool "
            "requires an environment with ai2thor installed."
        ),
    )
    parser.add_argument("--top-batch-tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--max-yaws-per-batch", type=int)
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument(
        "--auto-target-viewpoints",
        action="store_true",
        help=(
            "Search nearby reachable AI2-THOR viewpoints for each requested "
            "target object before capturing visible RGB-D detections."
        ),
    )
    parser.add_argument("--max-reachable-candidates-per-target", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        tasks = _load_tasks(args.top_batch_tasks)
        batches = _batches_from_tasks(tasks)
        if args.max_batches is not None:
            batches = batches[: args.max_batches]
        if args.dry_run:
            _emit_json(
                {
                    "action": "collect_ai2thor_coverage",
                    "auto_target_viewpoints": args.auto_target_viewpoints,
                    "batch_count": len(batches),
                    "dry_run": True,
                    "step_offset": args.step_offset,
                    "target_task_count": sum(len(batch["tasks"]) for batch in batches),
                }
            )
            return 0
        records = _collect_batches(
            batches,
            artifact_root=args.artifact_root,
            width=args.width,
            height=args.height,
            max_yaws_per_batch=args.max_yaws_per_batch,
            auto_target_viewpoints=args.auto_target_viewpoints,
            max_reachable_candidates_per_target=(
                args.max_reachable_candidates_per_target
            ),
            max_targets=args.max_targets,
            step_offset=args.step_offset,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(
                json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        _emit_json(
            {
                "action": "collect_ai2thor_coverage",
                "ready": False,
                "error": str(exc),
            }
        )
        return 1
    _emit_json(
        {
            "action": "collect_ai2thor_coverage",
            "artifact_root": str(args.artifact_root),
            "auto_target_viewpoints": args.auto_target_viewpoints,
            "batch_count": len(batches),
            "output": str(args.output),
            "record_count": len(records),
            "ready": True,
            "step_offset": args.step_offset,
        }
    )
    return 0


def stable_ai2thor_object_id(object_id: str) -> str:
    if not isinstance(object_id, str) or object_id == "":
        raise ValueError("object_id must be a non-empty string")
    if "|" not in object_id:
        return _safe_token(object_id)
    parts = object_id.split("|")
    label = _safe_token(parts[0])
    coordinates = [
        _coordinate_token(piece)
        for piece in parts[1:4]
    ]
    if len(coordinates) != 3 or any(piece == "" for piece in coordinates):
        return _safe_token(object_id)
    suffixes = [_safe_token(piece) for piece in parts[4:] if piece != ""]
    return "_".join([label, *coordinates, *suffixes])


def yaw_to_target(
    position: Mapping[str, Any],
    target_position: Mapping[str, Any],
) -> float:
    dx = _number(target_position.get("x")) - _number(position.get("x"))
    dz = _number(target_position.get("z")) - _number(position.get("z"))
    return round((math.degrees(math.atan2(dx, dz)) + 360.0) % 360.0, 2)


def nearest_reachable_positions(
    positions: Sequence[Mapping[str, Any]],
    target_position: Mapping[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    target_x = _number(target_position.get("x"))
    target_z = _number(target_position.get("z"))
    return [
        dict(position)
        for position in sorted(
            positions,
            key=lambda position: (
                (_number(position.get("x")) - target_x) ** 2
                + (_number(position.get("z")) - target_z) ** 2,
                _number(position.get("x")),
                _number(position.get("z")),
            ),
        )[:limit]
    ]


def coverage_capture_step(capture_index: int, *, step_offset: int) -> int:
    if capture_index <= 0:
        raise ValueError("capture_index must be positive")
    if step_offset < 0:
        raise ValueError("step_offset must be non-negative")
    return step_offset + capture_index


def detection_from_ai2thor_object(
    payload: Mapping[str, Any],
    *,
    bbox_2d_xyxy: Sequence[int] | None,
    frame_paths: Mapping[str, str],
    target_ids: set[str],
    visible_object_ids: set[str] | None = None,
) -> dict[str, Any]:
    raw_object_id = _required_str(payload, "objectId")
    object_id = stable_ai2thor_object_id(raw_object_id)
    label = _required_str(payload, "objectType").lower()
    bbox = _mapping(payload.get("axisAlignedBoundingBox"))
    center = _mapping(bbox.get("center"))
    size = _mapping(bbox.get("size"))
    position = _mapping(payload.get("position"))
    attributes: dict[str, Any] = {
        "ai2thor_object_id": raw_object_id,
        "collection_source": "ai2thor_visible_segmentation_rgbd",
        "detector": DETECTOR_NAME,
        "depth_path": frame_paths["depth_path"],
        "evidence_kinds": ["depth", "detector", "rgb"],
        "rgb_path": frame_paths["rgb_path"],
        "segmentation_path": frame_paths["segmentation_path"],
        "source_kind": "detector",
        "source_name": DETECTOR_NAME,
        "states": _object_states(payload),
    }
    if object_id in target_ids:
        attributes["collection_target_object_id"] = object_id
    mask_path = frame_paths.get("mask_path")
    if mask_path is not None:
        attributes["mask_path"] = mask_path
    current_location = _current_location_from_parent(
        payload,
        visible_object_ids=visible_object_ids,
    )
    if current_location is not None:
        attributes.update(current_location)
    detection: dict[str, Any] = {
        "attributes": attributes,
        "bbox": {
            "center": {
                "x": _number(center.get("x", position.get("x", 0.0))),
                "y": _number(center.get("y", position.get("y", 0.0))),
                "z": _number(center.get("z", position.get("z", 0.0))),
                "yaw": _rotation_y(center.get("rotation")),
            },
            "size": [
                _number(size.get("x", 0.01)),
                _number(size.get("y", 0.01)),
                _number(size.get("z", 0.01)),
            ],
        },
        "confidence": 1.0,
        "evidence_kinds": ["depth", "detector", "rgb"],
        "label": label,
        "object_id": object_id,
        "pose": {
            "x": _number(position.get("x", center.get("x", 0.0))),
            "y": _number(position.get("y", center.get("y", 0.0))),
            "z": _number(position.get("z", center.get("z", 0.0))),
            "yaw": _rotation_y(payload.get("rotation")),
        },
        "visible": payload.get("visible") is True,
    }
    if bbox_2d_xyxy is not None:
        detection["bbox_2d_xyxy"] = list(bbox_2d_xyxy)
        attributes["bbox_2d_xyxy"] = list(bbox_2d_xyxy)
    return detection


def _collect_batches(
    batches: Sequence[Mapping[str, Any]],
    *,
    artifact_root: Path,
    width: int,
    height: int,
    max_yaws_per_batch: int | None,
    auto_target_viewpoints: bool,
    max_reachable_candidates_per_target: int,
    max_targets: int | None,
    step_offset: int,
) -> list[dict[str, Any]]:
    from ai2thor.controller import Controller  # type: ignore[import-not-found]

    records: list[dict[str, Any]] = []
    capture_index = 0
    for batch in batches:
        scene_id = _required_str(batch, "scene_id")
        controller = Controller(
            scene=scene_id,
            width=width,
            height=height,
            renderDepthImage=True,
            renderInstanceSegmentation=True,
        )
        try:
            if auto_target_viewpoints:
                target_records, capture_index = _collect_target_viewpoints(
                    controller,
                    batch,
                    capture_index=capture_index,
                    artifact_root=artifact_root,
                    max_reachable_candidates_per_target=(
                        max_reachable_candidates_per_target
                    ),
                    remaining_targets=(
                        None
                        if max_targets is None
                        else max(0, max_targets - _captured_target_record_count(records))
                    ),
                    step_offset=step_offset,
                )
                records.extend(target_records)
                if max_targets is not None and _captured_target_record_count(records) >= max_targets:
                    break
            else:
                plan = _mapping(batch.get("execution_plan"))
                _execute_teleport(controller, plan)
                yaws = _target_yaws(plan)
                if max_yaws_per_batch is not None:
                    yaws = yaws[:max_yaws_per_batch]
                for yaw in yaws:
                    capture_index += 1
                    event = controller.step(
                        action="TeleportFull",
                        x=_number(_mapping(batch.get("position")).get("x")),
                        y=_number(_mapping(batch.get("position")).get("y")),
                        z=_number(_mapping(batch.get("position")).get("z")),
                        rotation=yaw,
                        horizon=0.0,
                        standing=True,
                        forceAction=True,
                    )
                    _require_success(event)
                    records.append(
                        _detector_record_from_event(
                            event,
                            batch=batch,
                            capture_index=capture_index,
                            artifact_root=artifact_root,
                            step_offset=step_offset,
                        )
                    )
        finally:
            controller.stop()
    return records


def _collect_target_viewpoints(
    controller: object,
    batch: Mapping[str, Any],
    *,
    capture_index: int,
    artifact_root: Path,
    max_reachable_candidates_per_target: int,
    remaining_targets: int | None,
    step_offset: int,
) -> tuple[list[dict[str, Any]], int]:
    reachable_positions, metadata = _reachable_positions_with_metadata(controller)
    target_objects = _target_objects_from_metadata(
        metadata,
        set(_string_sequence(batch.get("target_ids"))),
    )
    if remaining_targets is not None:
        target_objects = target_objects[:remaining_targets]
    records: list[dict[str, Any]] = []
    for target_object in target_objects:
        event = _visible_event_for_target_object(
            controller,
            target_object,
            reachable_positions,
            max_reachable_candidates_per_target=max_reachable_candidates_per_target,
        )
        if event is None:
            continue
        capture_index += 1
        target_batch = dict(batch)
        target_batch["target_ids"] = [
            stable_ai2thor_object_id(_required_str(target_object, "objectId"))
        ]
        records.append(
            _detector_record_from_event(
                event,
                batch=target_batch,
                capture_index=capture_index,
                artifact_root=artifact_root,
                step_offset=step_offset,
            )
        )
    return records, capture_index


def _reachable_positions_with_metadata(
    controller: object,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    event = cast(Any, controller).step(action="GetReachablePositions")
    _require_success(event)
    metadata = _mapping(getattr(event, "metadata", None))
    positions = [
        cast(Mapping[str, Any], item)
        for item in _sequence(metadata.get("actionReturn"))
        if isinstance(item, Mapping)
    ]
    return positions, metadata


def _target_objects_from_metadata(
    metadata: Mapping[str, Any],
    target_ids: set[str],
) -> list[Mapping[str, Any]]:
    targets: list[Mapping[str, Any]] = []
    for item in _sequence(metadata.get("objects")):
        if not isinstance(item, Mapping):
            continue
        stable_id = stable_ai2thor_object_id(_required_str(item, "objectId"))
        if stable_id in target_ids:
            targets.append(cast(Mapping[str, Any], item))
    return sorted(targets, key=lambda item: stable_ai2thor_object_id(_required_str(item, "objectId")))


def _visible_event_for_target_object(
    controller: object,
    target_object: Mapping[str, Any],
    reachable_positions: Sequence[Mapping[str, Any]],
    *,
    max_reachable_candidates_per_target: int,
) -> object | None:
    target_position = _mapping(target_object.get("position"))
    raw_object_id = _required_str(target_object, "objectId")
    for position in nearest_reachable_positions(
        reachable_positions,
        target_position,
        limit=max_reachable_candidates_per_target,
    ):
        base_yaw = yaw_to_target(position, target_position)
        for yaw in _target_search_yaws(base_yaw):
            for horizon in _target_search_horizons():
                event = cast(Any, controller).step(
                    action="TeleportFull",
                    x=_number(position.get("x")),
                    y=_number(position.get("y")),
                    z=_number(position.get("z")),
                    rotation=yaw,
                    horizon=horizon,
                    standing=True,
                    forceAction=True,
                )
                _require_success(event)
                if _event_object_visible(event, raw_object_id):
                    return cast(object, event)
    return None


def _target_search_yaws(base_yaw: float) -> tuple[float, ...]:
    return tuple(round((base_yaw + delta) % 360.0, 2) for delta in (0.0, -15.0, 15.0, -30.0, 30.0))


def _target_search_horizons() -> tuple[float, ...]:
    return (-30.0, 0.0, 15.0, 30.0, 45.0, 60.0)


def _event_object_visible(event: object, raw_object_id: str) -> bool:
    metadata = _mapping(getattr(event, "metadata", None))
    for item in _sequence(metadata.get("objects")):
        if (
            isinstance(item, Mapping)
            and item.get("objectId") == raw_object_id
            and item.get("visible") is True
        ):
            return True
    return False


def _captured_target_record_count(records: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for record in records if _string_sequence(_mapping(record.get("metadata")).get("target_ids")))


def _detector_record_from_event(
    event: object,
    *,
    batch: Mapping[str, Any],
    capture_index: int,
    artifact_root: Path,
    step_offset: int,
) -> dict[str, Any]:
    metadata = _mapping(getattr(event, "metadata", None))
    agent = _mapping(metadata.get("agent"))
    position = _mapping(agent.get("position"))
    rotation = _mapping(agent.get("rotation"))
    episode_id = _required_str(batch, "episode_id")
    scene_id = _required_str(batch, "scene_id")
    step = coverage_capture_step(capture_index, step_offset=step_offset)
    frame_dir = artifact_root / episode_id / f"{step:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = frame_dir / "rgb.ppm"
    depth_path = frame_dir / "depth.npy"
    segmentation_path = frame_dir / "segmentation.ppm"
    _write_ppm(getattr(event, "frame"), rgb_path)
    _write_depth_npy(getattr(event, "depth_frame"), depth_path)
    _write_ppm(getattr(event, "instance_segmentation_frame"), segmentation_path)
    color_bboxes = _color_bboxes(getattr(event, "instance_segmentation_frame"))
    color_to_object_id = _event_color_to_object_id(event)
    visible_objects = [
        cast(Mapping[str, Any], item)
        for item in _sequence(metadata.get("objects"))
        if isinstance(item, Mapping) and item.get("visible") is True
    ]
    visible_ids = {
        stable_ai2thor_object_id(_required_str(item, "objectId"))
        for item in visible_objects
    }
    target_ids = set(_string_sequence(batch.get("target_ids")))
    frame_paths = {
        "depth_path": str(depth_path),
        "rgb_path": str(rgb_path),
        "segmentation_path": str(segmentation_path),
    }
    detections = [
        detection_from_ai2thor_object(
            obj,
            bbox_2d_xyxy=_bbox_for_object(obj, color_bboxes, color_to_object_id),
            frame_paths=frame_paths,
            target_ids=target_ids,
            visible_object_ids=visible_ids,
        )
        for obj in sorted(visible_objects, key=lambda item: str(item.get("objectId")))
    ]
    return {
        "schema_version": DETECTOR_SCHEMA_VERSION,
        "agent_id": "agent",
        "agent_pose": {
            "x": _number(position.get("x")),
            "y": _number(position.get("y")),
            "z": _number(position.get("z")),
            "yaw": _number(rotation.get("y")),
        },
        "depth_path": str(depth_path),
        "detections": detections,
        "metadata": {
            "batch_id": batch.get("batch_id"),
            "capture_index": capture_index,
            "detector": DETECTOR_NAME,
            "episode_id": episode_id,
            "required_evidence_kinds": ["depth", "detector", "rgb"],
            "scene_id": scene_id,
            "source_kind": "detector",
            "source_name": DETECTOR_NAME,
            "step_offset": step_offset,
            "target_ids": sorted(target_ids),
        },
        "regions": [
            {
                "attributes": {
                    "episode_id": episode_id,
                    "scene_id": scene_id,
                    "source_kind": "detector",
                    "source_name": DETECTOR_NAME,
                },
                "label": "visible frame region",
                "node_id": f"visible_frame_region:{episode_id}:{step:06d}",
            }
        ],
        "rgb_path": str(rgb_path),
        "rooms": [
            {
                "attributes": {
                    "episode_id": episode_id,
                    "scene_id": scene_id,
                    "source_kind": "detector",
                    "source_name": DETECTOR_NAME,
                },
                "label": "AI2-THOR room",
                "node_id": "ai2thor_room",
            }
        ],
        "segmentation_path": str(segmentation_path),
        "step": step,
    }


def _execute_teleport(controller: object, plan: Mapping[str, Any]) -> None:
    action = _plan_action(plan, "TeleportFull")
    position = _mapping(action.get("position"))
    rotation = _mapping(action.get("rotation"))
    event = cast(Any, controller).step(
        action="TeleportFull",
        x=_number(position.get("x")),
        y=_number(position.get("y")),
        z=_number(position.get("z")),
        rotation=_number(rotation.get("y", 0.0)),
        horizon=0.0,
        standing=True,
        forceAction=True,
    )
    _require_success(event)


def _plan_action(plan: Mapping[str, Any], action_name: str) -> Mapping[str, Any]:
    for action in _sequence(plan.get("ai2thor_actions")):
        if isinstance(action, Mapping) and action.get("action") == action_name:
            return cast(Mapping[str, Any], action)
    raise ValueError(f"Missing AI2-THOR plan action: {action_name}")


def _target_yaws(plan: Mapping[str, Any]) -> list[float]:
    action = _plan_action(plan, "RotateToTargetYaw")
    yaws = [_number(value) for value in _sequence(action.get("target_yaws"))]
    return yaws or [0.0]


def _require_success(event: object) -> None:
    metadata = _mapping(getattr(event, "metadata", None))
    if metadata.get("lastActionSuccess") is not True:
        raise RuntimeError(str(metadata.get("errorMessage", "AI2-THOR action failed")))


def _batches_from_tasks(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_batch: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for task in tasks:
        batch_id = _required_str(task, "batch_id")
        plan = _mapping(task.get("batch_execution_plan"))
        batch = by_batch.setdefault(
            batch_id,
            {
                "batch_id": batch_id,
                "episode_id": _required_str(task, "episode_id"),
                "execution_plan": plan,
                "position": _mapping(_plan_action(plan, "TeleportFull").get("position")),
                "scene_id": _required_str(task, "scene_id"),
                "target_ids": [],
                "tasks": [],
            },
        )
        batch["tasks"].append(dict(task))
        target_id = _task_target_id(task)
        if target_id is not None and target_id not in batch["target_ids"]:
            batch["target_ids"].append(target_id)
    return list(by_batch.values())


def _task_target_id(task: Mapping[str, Any]) -> str | None:
    value = task.get("target_object_id")
    if isinstance(value, str) and value:
        return value
    task_id = _required_str(task, "task_id")
    pieces = task_id.split(":")
    return pieces[-1] if pieces else None


def _load_tasks(path: Path) -> list[Mapping[str, Any]]:
    tasks: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line == "":
            continue
        item = json.loads(line)
        if not isinstance(item, Mapping):
            raise ValueError(f"top-batch task line {line_number} must be an object")
        tasks.append(cast(Mapping[str, Any], item))
    return tasks


def _current_location_from_parent(
    payload: Mapping[str, Any],
    *,
    visible_object_ids: set[str] | None,
) -> dict[str, str] | None:
    parents = _sequence(payload.get("parentReceptacles"))
    parent = next((item for item in parents if isinstance(item, str) and item), None)
    if parent is None:
        return {
            "current_location_id": "ai2thor_room",
            "current_location_relation": "IN_ROOM",
        }
    parent_id = stable_ai2thor_object_id(parent)
    parent_label = parent.split("|", 1)[0].lower()
    if parent_label == "floor":
        return {
            "current_location_id": "ai2thor_room",
            "current_location_relation": "IN_ROOM",
        }
    if visible_object_ids is not None and parent_id not in visible_object_ids:
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


def _object_states(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in STATE_KEYS
        if isinstance(payload.get(key), (bool, int, float, str))
    }


def _event_color_to_object_id(event: object) -> dict[tuple[int, int, int], str]:
    color_to_object_id = getattr(event, "color_to_object_id", None)
    if not isinstance(color_to_object_id, Mapping):
        return {}
    result: dict[tuple[int, int, int], str] = {}
    for color, object_id in color_to_object_id.items():
        if isinstance(object_id, str):
            result[_rgb_tuple(color)] = stable_ai2thor_object_id(object_id)
    return result


def _bbox_for_object(
    obj: Mapping[str, Any],
    color_bboxes: Mapping[tuple[int, int, int], list[int]],
    color_to_object_id: Mapping[tuple[int, int, int], str],
) -> list[int] | None:
    object_id = stable_ai2thor_object_id(_required_str(obj, "objectId"))
    for color, mapped_object_id in color_to_object_id.items():
        if mapped_object_id == object_id:
            return color_bboxes.get(color)
    return None


def _color_bboxes(image: object) -> dict[tuple[int, int, int], list[int]]:
    rows = _image_rows(image)
    bboxes: dict[tuple[int, int, int], list[int]] = {}
    for y, row in enumerate(rows):
        for x, color in enumerate(row):
            current = bboxes.get(color)
            if current is None:
                bboxes[color] = [x, y, x, y]
            else:
                current[0] = min(current[0], x)
                current[1] = min(current[1], y)
                current[2] = max(current[2], x)
                current[3] = max(current[3], y)
    return bboxes


def _write_ppm(image: object, path: Path) -> None:
    rows = _image_rows(image)
    height = len(rows)
    width = len(rows[0]) if rows else 0
    payload = bytearray()
    for row in rows:
        for red, green, blue in row:
            payload.extend((red, green, blue))
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + bytes(payload))


def _write_depth_npy(depth: object, path: Path) -> None:
    import numpy as np

    np.save(path, cast(Any, depth))


def _image_rows(image: object) -> list[list[tuple[int, int, int]]]:
    if hasattr(image, "tolist"):
        image = image.tolist()
    if isinstance(image, str) or not isinstance(image, Sequence):
        raise ValueError("image must be an array-like RGB image")
    rows: list[list[tuple[int, int, int]]] = []
    for row in image:
        if isinstance(row, str) or not isinstance(row, Sequence):
            raise ValueError("image rows must be sequences")
        rows.append([_rgb_tuple(pixel) for pixel in row])
    return rows


def _rgb_tuple(value: object) -> tuple[int, int, int]:
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError("RGB value must contain three channels")
    channels = tuple(int(channel) for channel in value)
    if any(channel < 0 or channel > 255 for channel in channels):
        raise ValueError("RGB channels must be in 0..255")
    return cast(tuple[int, int, int], channels)


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
        char.lower() if char.isalnum() else "_"
        for char in value
    ).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned


def _rotation_y(value: object) -> float:
    if isinstance(value, Mapping):
        return _number(value.get("y", 0.0))
    if value is None:
        return 0.0
    return _number(value)


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a numeric value")
    if not math.isfinite(float(value)):
        raise ValueError("numeric value must be finite")
    return float(value)


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"missing required string field: {key}")
    return value


def _mapping(value: object) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _string_sequence(value: object) -> list[str]:
    return [item for item in _sequence(value) if isinstance(item, str) and item]


def _emit_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
