from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import signal
import sys
from typing import Any, cast
import urllib.error
import urllib.request

from dsg_spatialqa_lab.observations import (
    detector_observation_records_digest,
    _read_depth_values,
)
from dsg_spatialqa_lab.schema import SpatialQAError


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_vlm_controls import (  # noqa: E402
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    _extract_structured_response,
    _image_data_url,
)


TRACE_SCHEMA_VERSION = "dsg-spatialqa-lab.vlm-rgbd-detector-trace.v1"
DEFAULT_DETECTOR_NAME = "qwen37_vlm_rgbd_detector"
EXTERNAL_DETECTOR_FRAME_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.external-detector-frame.v1"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Run an explicit external VLM detector pass over RGB frames and project "
            "2D boxes through local depth into external detector JSONL records."
        ),
    )
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--output-detector-jsonl", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--detector-name", default=DEFAULT_DETECTOR_NAME)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--request-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)

    if not args.allow_network:
        _emit_json(
            {
                "action": "run_vlm_detector_rgbd",
                "ready": False,
                "error": "network_not_allowed",
                "message": "Pass --allow-network to run external VLM detector requests.",
            }
        )
        return 1
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        _emit_json(
            {
                "action": "run_vlm_detector_rgbd",
                "ready": False,
                "api_key_env": args.api_key_env,
                "error": "api_key_env_unset",
                "message": f"Set {args.api_key_env} without modifying system keys.",
            }
        )
        return 1

    try:
        handoff = _load_handoff(args.handoff)
        frames = _limited_frames(
            _mapping_sequence(handoff.get("required_frames"), "required_frames"),
            args.limit,
        )
        records = _load_resume_records(args.output_detector_jsonl, args.resume)
        traces = _load_resume_traces(args.trace_output, args.resume)
        completed = {_record_key(record) for record in records}
        errors: list[dict[str, Any]] = []
        for frame in frames:
            if _frame_key(frame) in completed:
                continue
            request_payload = _chat_completion_payload(
                frame,
                detector_name=args.detector_name,
                model=args.model,
            )
            raw_response: Mapping[str, Any] | None = None
            try:
                raw_response = _send_chat_completion(
                    request_payload,
                    api_key=api_key,
                    base_url=args.base_url,
                    timeout_seconds=args.request_timeout_seconds,
                )
                structured_response = _extract_detector_response(raw_response)
                record = _detector_record_from_response(
                    frame,
                    structured_response,
                    detector_name=args.detector_name,
                    model=args.model,
                )
            except (
                OSError,
                SpatialQAError,
                ValueError,
                json.JSONDecodeError,
                urllib.error.URLError,
            ) as exc:
                if args.continue_on_error:
                    record = _empty_detector_record(
                        frame,
                        detector_name=args.detector_name,
                        error=str(exc),
                    )
                    records.append(record)
                    traces.append(
                        _trace_record(
                            frame,
                            detector_name=args.detector_name,
                            model=args.model,
                            request_payload=request_payload,
                            raw_response=raw_response,
                            structured_response=None,
                            detector_record=record,
                            error=str(exc),
                        )
                    )
                    errors.append(
                        {
                            "error": str(exc),
                            "frame": list(_frame_key(frame)),
                        }
                    )
                    _save_records(records, args.output_detector_jsonl)
                    _save_trace_records(traces, args.trace_output)
                    continue
                traces.append(
                    _trace_record(
                        frame,
                        detector_name=args.detector_name,
                        model=args.model,
                        request_payload=request_payload,
                        raw_response=raw_response,
                        structured_response=None,
                        detector_record=None,
                        error=str(exc),
                    )
                )
                _save_records(records, args.output_detector_jsonl)
                _save_trace_records(traces, args.trace_output)
                _emit_json(
                    {
                        "action": "run_vlm_detector_rgbd",
                        "checkpoint_frame_count": len(records),
                        "checkpoint_trace_count": len(traces),
                        "error": str(exc),
                        "failed_frame": _frame_key(frame),
                        "ready": False,
                    }
                )
                return 1
            records.append(record)
            traces.append(
                _trace_record(
                    frame,
                    detector_name=args.detector_name,
                    model=args.model,
                    request_payload=request_payload,
                    raw_response=raw_response,
                    structured_response=structured_response,
                    detector_record=record,
                    error=None,
                )
            )
            _save_records(records, args.output_detector_jsonl)
            _save_trace_records(traces, args.trace_output)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(
            {
                "action": "run_vlm_detector_rgbd",
                "ready": False,
                "error": str(exc),
            }
        )
        return 1

    payload = _records_jsonl(records)
    ready = len(errors) == 0
    _emit_json(
        {
            "action": "run_vlm_detector_rgbd",
            "detector_name": args.detector_name,
            "detection_count": sum(len(record["detections"]) for record in records),
            "frame_count": len(records),
            "input_digest": detector_observation_records_digest(payload),
            "model": args.model,
            "output_detector_jsonl": str(args.output_detector_jsonl),
            "ready": ready,
            "error_count": len(errors),
            "errors": errors,
            "trace_count": len(traces),
            "trace_output": str(args.trace_output),
        }
    )
    return 0 if ready else 1


def _load_handoff(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("VLM detector handoff must be a JSON object")
    return cast(dict[str, Any], dict(payload))


def _limited_frames(
    frames: Sequence[Mapping[str, Any]],
    limit: int | None,
) -> tuple[Mapping[str, Any], ...]:
    if limit is None:
        return tuple(frames)
    if limit < 0:
        raise SpatialQAError("--limit must be non-negative")
    return tuple(frames[:limit])


def _chat_completion_payload(
    frame: Mapping[str, Any],
    *,
    detector_name: str,
    model: str,
) -> dict[str, Any]:
    image_dimensions = _depth_dimensions(Path(_required_str(frame, "depth_path")))
    prompt = {
        "contract": {
            "bbox_rule": (
                "Return tight 2D bounding boxes in pixel coordinates [x1,y1,x2,y2]. "
                "Use only the image. Do not infer hidden objects. Keep coordinates "
                "inside image_dimensions."
            ),
            "id_rule": (
                "target_id is a request identifier. Echo it only when the requested "
                "target is visibly detected."
            ),
            "support_detection_rule": (
                "Also detect visible support surfaces/containers from support_labels. "
                "For support detections set target_id to null."
            ),
            "output_schema": {
                "detections": [
                    {
                        "bbox_2d_xyxy": [0, 0, 1, 1],
                        "confidence": 0.0,
                        "label": "object",
                        "target_id": "request id or null",
                        "visible": True,
                    }
                ],
                "error": None,
                "frame_id": _optional_str(frame.get("frame_id")),
            },
        },
        "detector_name": detector_name,
        "episode_id": _required_str(frame, "episode_id"),
        "frame_id": _optional_str(frame.get("frame_id")),
        "image_dimensions": image_dimensions,
        "scene_id": _required_str(frame, "scene_id"),
        "step": _required_int(frame.get("frame_step"), "frame_step"),
        "support_labels": _string_list(frame.get("support_labels")),
        "targets": [
            {"label": label, "target_id": target_id}
            for label, target_id in zip(
                _string_list(frame.get("target_labels")),
                _string_list(frame.get("target_object_ids")),
                strict=False,
            )
        ],
    }
    return {
        "messages": [
            {
                "content": (
                    "You are an external visual detector. Return only JSON. "
                    "Do not use scene metadata or gold answers."
                ),
                "role": "system",
            },
            {
                "content": [
                    {
                        "text": json.dumps(
                            prompt,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                        "type": "text",
                    },
                    {
                        "image_url": {"url": _image_data_url(_required_str(frame, "rgb_path"))},
                        "type": "image_url",
                    },
                ],
                "role": "user",
            },
        ],
        "model": model,
        "temperature": 0,
    }


def _depth_dimensions(path: Path) -> dict[str, int]:
    values = _read_depth_values(path)
    height = len(values)
    width = len(values[0]) if values else 0
    if height <= 0 or width <= 0:
        raise SpatialQAError("depth image must be non-empty")
    return {"height": height, "width": width}


def _detector_record_from_response(
    frame: Mapping[str, Any],
    response: Mapping[str, Any],
    *,
    detector_name: str,
    model: str,
) -> dict[str, Any]:
    depth_values = _read_depth_values(Path(_required_str(frame, "depth_path")))
    camera_pose = _pose_mapping(_mapping(frame.get("camera_pose"), "camera_pose"))
    detections = [
        _detection_from_response(
            frame,
            item,
            depth_values=depth_values,
            camera_pose=camera_pose,
            detector_name=detector_name,
            model=model,
            index=index,
        )
        for index, item in enumerate(
            _mapping_sequence(response.get("detections"), "detections"),
            start=1,
        )
        if item.get("visible") is True
    ]
    return {
        "schema_version": EXTERNAL_DETECTOR_FRAME_SCHEMA_VERSION,
        "camera_pose": camera_pose,
        "depth_path": _required_str(frame, "depth_path"),
        "detections": detections,
        "detector_name": detector_name,
        "episode_id": _required_str(frame, "episode_id"),
        "rgb_path": _required_str(frame, "rgb_path"),
        "scene_id": _required_str(frame, "scene_id"),
        "step": _required_int(frame.get("frame_step"), "frame_step"),
    }


def _empty_detector_record(
    frame: Mapping[str, Any],
    *,
    detector_name: str,
    error: str,
) -> dict[str, Any]:
    return {
        "schema_version": EXTERNAL_DETECTOR_FRAME_SCHEMA_VERSION,
        "camera_pose": _pose_mapping(_mapping(frame.get("camera_pose"), "camera_pose")),
        "depth_path": _required_str(frame, "depth_path"),
        "detections": [],
        "detector_name": detector_name,
        "episode_id": _required_str(frame, "episode_id"),
        "metadata": {
            "detector_error": error,
            "detector_method": "vlm_bbox_depth_projection_v1",
            "source_kind": "detector",
            "source_name": detector_name,
        },
        "rgb_path": _required_str(frame, "rgb_path"),
        "scene_id": _required_str(frame, "scene_id"),
        "step": _required_int(frame.get("frame_step"), "frame_step"),
    }


def _extract_detector_response(raw_response: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return _extract_structured_response(raw_response)
    except json.JSONDecodeError:
        content = _message_content(raw_response)
        parsed, _ = json.JSONDecoder().raw_decode(_strip_json_fence(content).lstrip())
        if not isinstance(parsed, Mapping):
            raise SpatialQAError("Structured detector response must be a JSON object")
        return cast(dict[str, Any], dict(parsed))


def _message_content(raw_response: Mapping[str, Any]) -> str:
    choices = raw_response.get("choices")
    if isinstance(choices, str) or not isinstance(choices, Sequence) or not choices:
        raise SpatialQAError("VLM response missing choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise SpatialQAError("VLM response choice must be an object")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise SpatialQAError("VLM response choice missing message")
    content = message.get("content")
    if not isinstance(content, str):
        raise SpatialQAError("VLM response content must be a JSON string")
    return content


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _send_chat_completion(
    payload: dict[str, Any],
    *,
    api_key: str,
    base_url: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise SpatialQAError("request timeout must be positive")
    endpoint = base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with _real_time_limit(timeout_seconds):
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    if not isinstance(parsed, Mapping):
        raise SpatialQAError("VLM detector response must be a JSON object")
    return cast(dict[str, Any], parsed)


class _real_time_limit:
    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds
        self.previous_handler: Any = None

    def __enter__(self) -> None:
        self.previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.timeout_seconds)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.previous_handler)

    def _handle_timeout(self, signum: int, frame: object) -> None:
        raise TimeoutError(
            f"VLM detector request exceeded {self.timeout_seconds:g} seconds"
        )


def _detection_from_response(
    frame: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    depth_values: Sequence[Sequence[float]],
    camera_pose: Mapping[str, float],
    detector_name: str,
    model: str,
    index: int,
) -> dict[str, Any]:
    label = _canonical_label(_required_str(payload, "label"))
    bbox_2d = _bbox_2d(payload.get("bbox_2d_xyxy"))
    depth = _median_depth(depth_values, bbox_2d)
    center = _project_bbox_center_to_world(camera_pose, bbox_2d, depth, depth_values)
    size = _bbox_size_estimate(bbox_2d, depth, depth_values)
    target_id = _optional_str(payload.get("target_id"))
    detection_id = _detection_id(frame, label, index)
    object_id = target_id or detection_id
    support_detection = target_id is None and label in set(
        _canonical_label(item) for item in _string_list(frame.get("support_labels"))
    )
    return {
        "attributes": {
            "detector_method": "vlm_bbox_depth_projection_v1",
            "model": model,
            "source_kind": "detector",
            "source_name": detector_name,
            "support_detection": support_detection,
            "targeted_detection": target_id is not None,
        },
        "bbox_2d_xyxy": bbox_2d,
        "bbox_3d_center": center,
        "bbox_3d_size": size,
        "confidence": _confidence(payload.get("confidence")),
        "detection_id": detection_id,
        "evidence_kinds": ["depth", "detector", "rgb"],
        "label": label,
        "object_id": object_id,
        "visible": True,
    }


def _median_depth(
    depth_values: Sequence[Sequence[float]],
    bbox: Sequence[int],
) -> float:
    height = len(depth_values)
    width = len(depth_values[0]) if height else 0
    if height <= 0 or width <= 0:
        raise SpatialQAError("depth image must be non-empty")
    x1, y1, x2, y2 = _clamped_bbox(bbox, width=width, height=height)
    values = [
        float(depth_values[y][x])
        for y in range(y1, y2 + 1)
        for x in range(x1, x2 + 1)
        if float(depth_values[y][x]) > 0
    ]
    if not values:
        raise SpatialQAError("bbox has no positive depth samples")
    values.sort()
    return values[len(values) // 2]


def _project_bbox_center_to_world(
    camera_pose: Mapping[str, float],
    bbox: Sequence[int],
    depth: float,
    depth_values: Sequence[Sequence[float]],
) -> dict[str, float]:
    height = len(depth_values)
    width = len(depth_values[0]) if height else 0
    if width <= 0:
        raise SpatialQAError("depth image width must be positive")
    x_center = (bbox[0] + bbox[2]) / 2.0
    normalized_x = 0.0 if width <= 1 else (x_center / (width - 1)) - 0.5
    lateral = normalized_x * depth
    yaw_radians = math.radians(camera_pose["yaw"])
    forward_x = math.sin(yaw_radians)
    forward_z = math.cos(yaw_radians)
    right_x = math.cos(yaw_radians)
    right_z = -math.sin(yaw_radians)
    return {
        "x": _rounded(camera_pose["x"] + (forward_x * depth) + (right_x * lateral)),
        "y": _rounded(camera_pose["y"]),
        "yaw": 0.0,
        "z": _rounded(camera_pose["z"] + (forward_z * depth) + (right_z * lateral)),
    }


def _bbox_size_estimate(
    bbox: Sequence[int],
    depth: float,
    depth_values: Sequence[Sequence[float]],
) -> list[float]:
    height = len(depth_values)
    width = len(depth_values[0]) if height else 0
    pixel_width = max(1, bbox[2] - bbox[0] + 1)
    pixel_height = max(1, bbox[3] - bbox[1] + 1)
    lateral = max(0.05, (pixel_width / max(width, 1)) * depth)
    vertical = max(0.05, (pixel_height / max(height, 1)) * depth)
    return [_rounded(lateral), _rounded(vertical), _rounded(max(0.05, lateral))]


def _clamped_bbox(
    bbox: Sequence[int],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    left = max(0, min(width - 1, min(x1, x2)))
    right = max(0, min(width - 1, max(x1, x2)))
    top = max(0, min(height - 1, min(y1, y2)))
    bottom = max(0, min(height - 1, max(y1, y2)))
    return left, top, right, bottom


def _bbox_2d(value: object) -> list[int]:
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) != 4:
        raise SpatialQAError("bbox_2d_xyxy must contain four numbers")
    return [_int_from_number(item, "bbox_2d_xyxy") for item in value]


def _pose_mapping(payload: Mapping[str, Any]) -> dict[str, float]:
    return {
        "x": _float_from_number(payload.get("x"), "camera_pose.x"),
        "y": _float_from_number(payload.get("y"), "camera_pose.y"),
        "yaw": _float_from_number(payload.get("yaw"), "camera_pose.yaw"),
        "z": _float_from_number(payload.get("z"), "camera_pose.z"),
    }


def _confidence(value: object) -> float:
    result = _float_from_number(value, "confidence")
    return max(0.0, min(1.0, result))


def _detection_id(frame: Mapping[str, Any], label: str, index: int) -> str:
    return (
        f"{_required_str(frame, 'episode_id')}:"
        f"{_required_str(frame, 'scene_id')}:"
        f"{_required_int(frame.get('frame_step'), 'frame_step'):06d}:"
        f"{label}:{index:03d}"
    )


def _record_key(record: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        _required_str(record, "episode_id"),
        _required_str(record, "scene_id"),
        _required_int(record.get("step"), "step"),
    )


def _frame_key(frame: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        _required_str(frame, "episode_id"),
        _required_str(frame, "scene_id"),
        _required_int(frame.get("frame_step"), "frame_step"),
    )


def _save_records(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_records_jsonl(records), encoding="utf-8")


def _records_jsonl(records: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        for record in sorted(records, key=_record_key)
    )


def _load_resume_records(path: Path, resume: bool) -> list[dict[str, Any]]:
    if not resume or not path.exists():
        return []
    return [
        cast(dict[str, Any], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _save_trace_records(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _load_resume_traces(path: Path, resume: bool) -> list[dict[str, Any]]:
    if not resume or not path.exists():
        return []
    return [
        cast(dict[str, Any], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _trace_record(
    frame: Mapping[str, Any],
    *,
    detector_name: str,
    model: str,
    request_payload: Mapping[str, Any],
    raw_response: Mapping[str, Any] | None,
    structured_response: Mapping[str, Any] | None,
    detector_record: Mapping[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "detector_name": detector_name,
        "detection_count": (
            len(detector_record.get("detections", []))
            if isinstance(detector_record, Mapping)
            else 0
        ),
        "error": error,
        "frame": {
            "depth_path": frame.get("depth_path"),
            "episode_id": frame.get("episode_id"),
            "frame_id": frame.get("frame_id"),
            "rgb_path": frame.get("rgb_path"),
            "scene_id": frame.get("scene_id"),
            "step": frame.get("frame_step"),
            "target_labels": frame.get("target_labels"),
            "target_object_ids": frame.get("target_object_ids"),
        },
        "model": model,
        "raw_response": raw_response,
        "request": {
            "model": request_payload.get("model"),
            "temperature": request_payload.get("temperature"),
        },
        "structured_response": structured_response,
    }


def _mapping_sequence(value: object, field_name: str) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise SpatialQAError(f"{field_name} must be a sequence")
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise SpatialQAError(f"{field_name}[{index}] must be an object")
        result.append(cast(Mapping[str, Any], item))
    return tuple(result)


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpatialQAError(f"{field_name} must be an object")
    return cast(Mapping[str, Any], value)


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"{key} must be a non-empty string")
    return value


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value != "":
        return value
    return None


def _required_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SpatialQAError(f"{field_name} must be an integer")
    return value


def _string_list(value: object) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _canonical_label(value: str) -> str:
    return "".join(ch.lower() for ch in value.strip() if ch.isalnum() or ch in {"_", "-"})


def _int_from_number(value: object, field_name: str) -> int:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SpatialQAError(f"{field_name} must be numeric")
    return int(round(float(value)))


def _float_from_number(value: object, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SpatialQAError(f"{field_name} must be numeric")
    return float(value)


def _rounded(value: float) -> float:
    rounded = round(float(value), 6)
    return 0.0 if abs(rounded) < 0.000001 else rounded


def _emit_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
