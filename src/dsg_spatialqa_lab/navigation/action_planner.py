from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab.schema import Pose3D


REACHABLE_POSITIONS_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.reachable-positions.v1"
)


@dataclass(frozen=True, order=True)
class ReachablePosition:
    x: float
    y: float
    z: float

    @property
    def key(self) -> tuple[int, int, int]:
        return (_quantize(self.x), _quantize(self.y), _quantize(self.z))

    def to_pose(self, *, yaw: float = 0.0) -> Pose3D:
        return Pose3D(self.x, self.y, self.z, yaw=yaw)

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass(frozen=True)
class PlannedPath:
    reachable: bool
    path_positions: tuple[ReachablePosition, ...]
    actions: tuple[str, ...]
    rejected_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reachable": self.reachable,
            "path_positions": [position.to_dict() for position in self.path_positions],
            "actions": list(self.actions),
            "rejected_reason": self.rejected_reason,
        }


def reachable_positions_from_mappings(
    payloads: Iterable[Mapping[str, Any]],
) -> tuple[ReachablePosition, ...]:
    positions = {
        ReachablePosition(
            _float_value(payload.get("x")),
            _float_value(payload.get("y", 0.0)),
            _float_value(payload.get("z")),
        )
        for payload in payloads
    }
    return tuple(sorted(positions, key=lambda item: item.key))


def reachable_positions_report(
    *,
    scene_id: str,
    episode_id: str,
    positions: Sequence[ReachablePosition],
    runtime_kind: str,
    real_ai2thor_runtime: bool,
) -> dict[str, Any]:
    return {
        "schema_version": REACHABLE_POSITIONS_SCHEMA_VERSION,
        "scene_id": scene_id,
        "episode_id": episode_id,
        "runtime_kind": runtime_kind,
        "real_ai2thor_runtime": real_ai2thor_runtime,
        "position_count": len(positions),
        "positions": [position.to_dict() for position in positions],
    }


def save_reachable_positions_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_reachable_positions_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("reachable positions report must be a JSON object")
    return payload


def plan_reachable_path(
    reachable_positions: Sequence[ReachablePosition],
    start_pose: Pose3D,
    target_pose: Pose3D,
    *,
    grid_step: float = 0.25,
) -> PlannedPath:
    position_by_key = {position.key: position for position in reachable_positions}
    start_key = _nearest_position_key(start_pose, position_by_key)
    target_key = _nearest_position_key(target_pose, position_by_key)
    if start_key is None:
        return PlannedPath(False, (), (), "start_not_in_reachable_positions")
    if target_key is None:
        return PlannedPath(False, (), (), "target_not_in_reachable_positions")

    path_keys = _bfs(position_by_key, start_key, target_key, grid_step=grid_step)
    if not path_keys:
        return PlannedPath(False, (), (), "no_reachable_path")
    positions = tuple(position_by_key[key] for key in path_keys)
    actions = _path_actions(positions, start_pose.yaw, target_pose.yaw)
    return PlannedPath(True, positions, actions)


def reject_unreachable_candidates(
    candidates: Sequence[Any],
    reachable_positions: Sequence[ReachablePosition],
) -> tuple[tuple[Any, ...], list[dict[str, str]]]:
    reachable_keys = {position.key for position in reachable_positions}
    accepted: list[Any] = []
    rejected: list[dict[str, str]] = []
    for candidate in candidates:
        pose = getattr(candidate, "pose")
        candidate_id = str(getattr(candidate, "candidate_id"))
        key = (_quantize(pose.x), _quantize(pose.y), _quantize(pose.z))
        if key not in reachable_keys:
            rejected.append(
                {
                    "candidate_id": candidate_id,
                    "reason": "not_in_reachable_positions",
                }
            )
            continue
        accepted.append(candidate)
    return tuple(accepted), rejected


def _bfs(
    position_by_key: Mapping[tuple[int, int, int], ReachablePosition],
    start_key: tuple[int, int, int],
    target_key: tuple[int, int, int],
    *,
    grid_step: float,
) -> tuple[tuple[int, int, int], ...]:
    queue: deque[tuple[int, int, int]] = deque([start_key])
    parent: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start_key: None}
    while queue:
        current = queue.popleft()
        if current == target_key:
            break
        for neighbor in _neighbors(current, position_by_key, grid_step=grid_step):
            if neighbor in parent:
                continue
            parent[neighbor] = current
            queue.append(neighbor)
    if target_key not in parent:
        return ()
    path: list[tuple[int, int, int]] = []
    current_key: tuple[int, int, int] | None = target_key
    while current_key is not None:
        path.append(current_key)
        current_key = parent[current_key]
    return tuple(reversed(path))


def _neighbors(
    key: tuple[int, int, int],
    position_by_key: Mapping[tuple[int, int, int], ReachablePosition],
    *,
    grid_step: float,
) -> tuple[tuple[int, int, int], ...]:
    delta = _quantize(grid_step)
    x, y, z = key
    candidates = (
        (x + delta, y, z),
        (x - delta, y, z),
        (x, y, z + delta),
        (x, y, z - delta),
    )
    return tuple(candidate for candidate in candidates if candidate in position_by_key)


def _nearest_position_key(
    pose: Pose3D,
    position_by_key: Mapping[tuple[int, int, int], ReachablePosition],
) -> tuple[int, int, int] | None:
    exact = (_quantize(pose.x), _quantize(pose.y), _quantize(pose.z))
    if exact in position_by_key:
        return exact
    if not position_by_key:
        return None
    nearest_key, nearest_distance = min(
        (
            (
                key,
                math.sqrt(
                    (pose.x - position.x) ** 2
                    + (pose.y - position.y) ** 2
                    + (pose.z - position.z) ** 2
                ),
            )
            for key, position in position_by_key.items()
        ),
        key=lambda item: (item[1], item[0]),
    )
    return nearest_key if nearest_distance <= 1e-6 else None


def _path_actions(
    positions: Sequence[ReachablePosition],
    start_yaw: float,
    target_yaw: float,
) -> tuple[str, ...]:
    actions: list[str] = []
    yaw = _normalize_yaw(start_yaw)
    for previous, current in zip(positions, positions[1:], strict=False):
        desired_yaw = _movement_yaw(previous, current)
        rotate_actions, yaw = _rotate_actions(yaw, desired_yaw)
        actions.extend(rotate_actions)
        actions.append("MoveAhead")
    rotate_actions, _ = _rotate_actions(yaw, target_yaw)
    actions.extend(rotate_actions)
    return tuple(actions)


def _movement_yaw(previous: ReachablePosition, current: ReachablePosition) -> float:
    dx = current.x - previous.x
    dz = current.z - previous.z
    if abs(dx) >= abs(dz):
        return 90.0 if dx > 0 else 270.0
    return 0.0 if dz > 0 else 180.0


def _rotate_actions(current_yaw: float, target_yaw: float) -> tuple[tuple[str, ...], float]:
    current = int(round(_normalize_yaw(current_yaw) / 90.0)) % 4
    target = int(round(_normalize_yaw(target_yaw) / 90.0)) % 4
    right_turns = (target - current) % 4
    left_turns = (current - target) % 4
    if right_turns <= left_turns:
        return (("RotateRight",) * right_turns, float(target * 90))
    return (("RotateLeft",) * left_turns, float(target * 90))


def _normalize_yaw(value: float) -> float:
    return value % 360.0


def _quantize(value: float) -> int:
    return int(round(value * 1000.0))


def _float_value(value: object) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError("reachable position coordinates must be numbers")
    return float(value)

