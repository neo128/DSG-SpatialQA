from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from dsg_spatialqa_lab.episodes import EpisodeFrame
from dsg_spatialqa_lab.perception.mock import Detection2D
from dsg_spatialqa_lab.schema import BBox3D, Pose3D, SpatialQAError


@dataclass(frozen=True)
class Instance3D:
    instance_id: str
    label: str
    pose: Pose3D
    bbox: BBox3D
    confidence: float
    visible: bool
    step: int
    source_detection_id: str
    attributes: Mapping[str, Any] = field(default_factory=dict)


class MockDepthProjector:
    def project(self, frame: EpisodeFrame, detection: Detection2D) -> Instance3D:
        pose = self._pose_for_detection(frame, detection)
        bbox = BBox3D(center=pose, size=self._bbox_size_for_detection(detection))
        return Instance3D(
            instance_id=_string_attribute(
                detection.attributes,
                "object_id",
                default=detection.detection_id,
            ),
            label=detection.label,
            pose=pose,
            bbox=bbox,
            confidence=detection.confidence,
            visible=detection.visible,
            step=frame.step,
            source_detection_id=detection.detection_id,
            attributes=_stable_mapping(detection.attributes),
        )

    def project_all(
        self,
        frame: EpisodeFrame,
        detections: Sequence[Detection2D],
    ) -> tuple[Instance3D, ...]:
        return tuple(self.project(frame, detection) for detection in detections)

    def _pose_for_detection(self, frame: EpisodeFrame, detection: Detection2D) -> Pose3D:
        pose_payload = detection.attributes.get("pose")
        if pose_payload is not None:
            return _pose_from_mapping(_as_mapping(pose_payload, "pose"))
        x1, y1, x2, y2 = detection.bbox_xyxy
        center_x = ((x1 + x2) / 2.0) / 100.0
        center_y = detection.depth
        center_z = ((y1 + y2) / 2.0) / 100.0
        return Pose3D(center_x, center_y, center_z, yaw=frame.agent_pose.yaw)

    def _bbox_size_for_detection(
        self,
        detection: Detection2D,
    ) -> tuple[float, float, float]:
        bbox_size = detection.attributes.get("bbox_size")
        if bbox_size is not None:
            return _float_tuple(bbox_size, "bbox_size", length=3)
        x1, y1, x2, y2 = detection.bbox_xyxy
        width = max((x2 - x1) / 100.0, 0.01)
        height = max((y2 - y1) / 100.0, 0.01)
        return (width, height, max(detection.depth / 10.0, 0.01))


def _pose_from_mapping(payload: Mapping[str, Any]) -> Pose3D:
    return Pose3D(
        _required_float(payload, "x"),
        _required_float(payload, "y"),
        _required_float(payload, "z"),
        yaw=_optional_float(payload, "yaw", default=0.0),
    )


def _as_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpatialQAError(f"{field_name} must be an object")
    return cast(Mapping[str, Any], value)


def _stable_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value[key] for key in sorted(value)}


def _string_attribute(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    default: str,
) -> str:
    value = payload.get(field_name)
    if value is None:
        return default
    if not isinstance(value, str):
        raise SpatialQAError(f"{field_name} must be a string")
    return value


def _required_float(payload: Mapping[str, Any], field_name: str) -> float:
    value = payload.get(field_name)
    return _float_value(value, field_name)


def _optional_float(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    default: float,
) -> float:
    value = payload.get(field_name)
    if value is None:
        return default
    return _float_value(value, field_name)


def _float_tuple(value: object, field_name: str, *, length: int) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise SpatialQAError(f"{field_name} must be a sequence")
    if len(value) != length:
        raise SpatialQAError(f"{field_name} must contain exactly {length} values")
    return (
        _float_value(value[0], f"{field_name}[0]"),
        _float_value(value[1], f"{field_name}[1]"),
        _float_value(value[2], f"{field_name}[2]"),
    )


def _float_value(value: object, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SpatialQAError(f"{field_name} must be a number")
    return float(value)
