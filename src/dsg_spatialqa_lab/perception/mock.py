from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from dsg_spatialqa_lab.episodes import EpisodeFrame
from dsg_spatialqa_lab.schema import SpatialQAError


@dataclass(frozen=True)
class Detection2D:
    detection_id: str
    label: str
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    depth: float
    visible: bool = True
    attributes: Mapping[str, Any] = field(default_factory=dict)


class MockSegmenter:
    def detect(self, frame: EpisodeFrame) -> tuple[Detection2D, ...]:
        detections = [
            self._detection_from_mapping(item)
            for item in _metadata_sequence(frame.metadata, "mock_detections")
        ]
        return tuple(
            sorted(detections, key=lambda item: (item.detection_id, item.label))
        )

    def _detection_from_mapping(self, payload: Mapping[str, Any]) -> Detection2D:
        bbox_xyxy = _required_bbox_xyxy(payload, "bbox_xyxy")
        attributes = dict(_optional_mapping(payload, "attributes"))
        object_id = _optional_str(payload, "object_id")
        if object_id is not None:
            attributes["object_id"] = object_id
        pose = payload.get("pose")
        if pose is not None:
            attributes["pose"] = _stable_mapping(_as_mapping(pose, "pose"))
        bbox_size = payload.get("bbox_size")
        if bbox_size is not None:
            attributes["bbox_size"] = _float_tuple(bbox_size, "bbox_size", length=3)
        return Detection2D(
            detection_id=_required_str(payload, "detection_id"),
            label=_required_str(payload, "label"),
            bbox_xyxy=bbox_xyxy,
            confidence=_optional_float(payload, "confidence", default=1.0),
            depth=_optional_float(payload, "depth", default=1.0),
            visible=_optional_bool(payload, "visible", default=True),
            attributes=_stable_mapping(attributes),
        )


def _metadata_sequence(
    metadata: Mapping[str, Any],
    field_name: str,
) -> tuple[Mapping[str, Any], ...]:
    value = metadata.get(field_name, ())
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise SpatialQAError(f"{field_name} must be a sequence")
    items: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise SpatialQAError(f"{field_name} entries must be objects")
        items.append(cast(Mapping[str, Any], item))
    return tuple(items)


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


def _optional_bool(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    default: bool,
) -> bool:
    value = payload.get(field_name)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise SpatialQAError(f"{field_name} must be a boolean")
    return value


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


def _required_bbox_xyxy(
    payload: Mapping[str, Any],
    field_name: str,
) -> tuple[float, float, float, float]:
    value = payload.get(field_name)
    values = _float_tuple(value, field_name, length=4)
    return (values[0], values[1], values[2], values[3])


def _float_tuple(value: object, field_name: str, *, length: int) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise SpatialQAError(f"{field_name} must be a sequence")
    if len(value) != length:
        raise SpatialQAError(f"{field_name} must contain exactly {length} values")
    return tuple(_float_value(item, f"{field_name}[{index}]") for index, item in enumerate(value))


def _float_value(value: object, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SpatialQAError(f"{field_name} must be a number")
    return float(value)
