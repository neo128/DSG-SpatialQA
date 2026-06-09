#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any


SCHEMA_VERSION = "dsg-spatialqa-lab.active-qa-v2-target-crop-enrichment-report.v1"
ACTIVE_QA_V2_REQUEST_BUNDLE_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.active-qa-v2-vlm-request-bundle.v1"
)
FORBIDDEN_KEYS = {
    "gold_answer",
    "gold_evidence",
    "required_edges",
    "required_nodes",
    "visible_object_ids",
    "visible_object_labels",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Generate local target crops from RGB + AI2-THOR segmentation color "
            "evidence and inject them into an active QA v2 request bundle."
        ),
    )
    parser.add_argument("--request-bundle", type=Path, required=True)
    parser.add_argument("--observation-sequence", type=Path, action="append", required=True)
    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--padding-pixels", type=int, default=8)
    args = parser.parse_args(argv)

    try:
        if args.padding_pixels < 0:
            raise ValueError("--padding-pixels must be non-negative")
        bundle = _load_bundle(args.request_bundle)
        observations = _load_observation_objects(args.observation_sequence)
        enriched_bundle, report = _enrich_bundle(
            bundle,
            observations,
            request_bundle=args.request_bundle,
            observation_sequences=args.observation_sequence,
            crop_root=args.crop_root,
            padding_pixels=args.padding_pixels,
        )
        _write_json(args.output, enriched_bundle)
        _write_json(args.report, report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "blockers": ["target_crop_enrichment_error"],
            "error": str(exc),
            "ready": False,
        }
        report["report_digest"] = _digest_without(report, "report_digest")
        _write_json(args.report, report)
        _emit(report)
        return 1

    _emit(
        {
            "action": "enrich_active_qa_v2_request_bundle_with_target_crops",
            "blockers": report["blockers"],
            "enriched_target_crop_count": report["summary"]["cases_with_target_crop"],
            "infeasible_reasons": report["infeasible_reasons"],
            "output": str(args.output),
            "ready": report["ready"],
            "report": str(args.report),
        }
    )
    return 0 if report["ready"] is True else 1


def _load_bundle(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("request bundle must be a JSON object")
    if payload.get("schema_version") != ACTIVE_QA_V2_REQUEST_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported_request_bundle_schema")
    if payload.get("leak_free") is not True or payload.get("leak_paths"):
        raise ValueError("request_bundle_not_leak_free")
    leak_paths = _forbidden_paths(payload)
    if leak_paths:
        raise ValueError(f"request_bundle_has_forbidden_paths:{','.join(leak_paths)}")
    cases = payload.get("prediction_cases")
    if not isinstance(cases, list):
        raise ValueError("prediction_cases must be a list")
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"prediction case {index} must be an object")
        if not isinstance(case.get("case_id"), str):
            raise ValueError(f"prediction case {index} missing string case_id")
    return payload


def _load_observation_objects(paths: list[Path]) -> dict[tuple[int, str], dict[str, Any]]:
    objects: dict[tuple[int, str], dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"observation sequence must be a JSON object: {path}")
        observations = payload.get("observations")
        if not isinstance(observations, list):
            raise ValueError(f"observation sequence missing observations list: {path}")
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            step = _int_or_none(observation.get("step"))
            if step is None:
                continue
            rows = observation.get("objects")
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                object_id = row.get("object_id")
                if isinstance(object_id, str) and object_id:
                    objects[(step, object_id)] = row
    return objects


def _enrich_bundle(
    bundle: dict[str, Any],
    observations: dict[tuple[int, str], dict[str, Any]],
    *,
    request_bundle: Path,
    observation_sequences: list[Path],
    crop_root: Path,
    padding_pixels: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_cases = [case for case in bundle["prediction_cases"] if isinstance(case, dict)]
    crop_root.mkdir(parents=True, exist_ok=True)
    enriched_cases: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    summary: Counter[str] = Counter()
    enriched_case_ids: list[str] = []
    infeasible_cases: list[dict[str, str]] = []

    for case in source_cases:
        summary["request_count"] += 1
        case_copy = dict(case)
        case_id = _case_id(case_copy)
        crop_result = _target_crop_for_case(
            case_copy,
            observations,
            crop_root=crop_root,
            padding_pixels=padding_pixels,
        )
        if crop_result.ready:
            case_copy["target_crop"] = crop_result.crop
            summary["cases_with_target_crop"] += 1
            enriched_case_ids.append(case_id)
        else:
            reason = crop_result.reason
            context = _target_visual_context(case_copy, reason)
            if context is not None:
                case_copy["target_visual_context"] = context
                summary["cases_with_target_visual_context"] += 1
            reasons[reason] += 1
            infeasible_cases.append({"case_id": case_id, "reason": reason})
        enriched_cases.append(case_copy)

    blockers: list[str] = []
    if source_cases and summary["cases_with_target_crop"] == 0:
        blockers.append("target_crop_artifacts_missing")
    elif summary["cases_with_target_crop"] < len(source_cases):
        blockers.append("target_crop_artifacts_partial")

    enriched_bundle = dict(bundle)
    enriched_bundle["leak_free"] = True
    enriched_bundle["leak_paths"] = []
    enriched_bundle["prediction_cases"] = enriched_cases
    enriched_bundle["request_count"] = len(enriched_cases)
    enriched_bundle["target_crop_enrichment"] = {
        "blockers": blockers,
        "crop_root": str(crop_root),
        "enriched_case_ids_sample": enriched_case_ids[:50],
        "infeasible_cases_sample": infeasible_cases[:50],
        "infeasible_reasons": dict(sorted(reasons.items())),
        "observation_sequences": [str(path) for path in observation_sequences],
        "padding_pixels": padding_pixels,
        "request_bundle": str(request_bundle),
        "schema_version": SCHEMA_VERSION,
        "summary": {
            "cases_with_target_crop": summary["cases_with_target_crop"],
            "cases_with_target_visual_context": summary[
                "cases_with_target_visual_context"
            ],
            "infeasible_target_crop_case_count": len(source_cases)
            - summary["cases_with_target_crop"],
            "observation_object_count": len(observations),
            "request_count": len(source_cases),
        },
    }
    enriched_bundle["request_bundle_digest"] = _digest_without(
        enriched_bundle,
        "request_bundle_digest",
    )
    leak_paths = _forbidden_paths(enriched_bundle)
    if leak_paths:
        blockers.append("target_crop_enrichment_leaked_forbidden_fields")
        enriched_bundle["leak_free"] = False
        enriched_bundle["leak_paths"] = leak_paths

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "blockers": blockers,
        "crop_root": str(crop_root),
        "enriched_case_ids_sample": enriched_case_ids[:50],
        "infeasible_cases_sample": infeasible_cases[:50],
        "infeasible_reasons": dict(sorted(reasons.items())),
        "observation_sequences": [str(path) for path in observation_sequences],
        "ready": not blockers,
        "request_bundle": str(request_bundle),
        "summary": enriched_bundle["target_crop_enrichment"]["summary"],
    }
    report["report_digest"] = _digest_without(report, "report_digest")
    return enriched_bundle, report


class _CropResult:
    def __init__(
        self,
        *,
        crop: dict[str, Any] | None = None,
        reason: str = "",
        ready: bool,
    ) -> None:
        self.crop = crop
        self.reason = reason
        self.ready = ready


def _target_crop_for_case(
    case: dict[str, Any],
    observations: dict[tuple[int, str], dict[str, Any]],
    *,
    crop_root: Path,
    padding_pixels: int,
) -> _CropResult:
    step = _case_step(case)
    if step is None:
        return _CropResult(ready=False, reason="missing_step")
    target_object_id = _target_object_id(case)
    if target_object_id is None:
        return _CropResult(ready=False, reason="missing_target_id")
    observation_object = observations.get((step, target_object_id))
    if observation_object is None:
        return _CropResult(ready=False, reason="missing_matching_observation_object")
    attrs = _mapping_or_empty(observation_object.get("attributes"))
    rgb_path = _first_path(
        attrs.get("rgb_path"),
        observation_object.get("rgb_path"),
        _primary_rgb_path(case),
    )
    segmentation_path = _first_path(
        attrs.get("segmentation_path"),
        observation_object.get("segmentation_path"),
    )
    segmentation_color = _rgb(
        attrs.get("segmentation_color_rgb")
        or attrs.get("segmentation_color")
        or observation_object.get("segmentation_color_rgb")
        or observation_object.get("segmentation_color")
    )
    if rgb_path is None or not rgb_path.exists():
        return _CropResult(ready=False, reason="missing_rgb_path")
    if segmentation_path is None or not segmentation_path.exists():
        return _CropResult(ready=False, reason="missing_segmentation_path")
    if segmentation_color is None:
        return _CropResult(ready=False, reason="missing_segmentation_color")

    rgb = _read_ppm(rgb_path)
    segmentation = _read_ppm(segmentation_path)
    if (rgb.width, rgb.height) != (segmentation.width, segmentation.height):
        return _CropResult(ready=False, reason="rgb_segmentation_dimension_mismatch")
    unpadded_bbox = _color_bbox(segmentation, segmentation_color)
    if unpadded_bbox is None:
        return _CropResult(ready=False, reason="segmentation_color_not_found")
    padded_bbox = _pad_bbox(
        unpadded_bbox,
        width=rgb.width,
        height=rgb.height,
        padding_pixels=padding_pixels,
    )
    crop_path = crop_root / _safe_episode_id(case) / f"{_short_digest(_case_id(case))}.ppm"
    _write_ppm(crop_path, _crop_ppm(rgb, padded_bbox))
    source_frame_id = _source_frame_id(case, step)
    return _CropResult(
        ready=True,
        crop={
            "bbox_2d_xyxy": list(padded_bbox),
            "crop_kind": "segmentation_color_mask",
            "crop_path": str(crop_path),
            "object_id": target_object_id,
            "rgb_path": str(crop_path),
            "segmentation_color_rgb": segmentation_color,
            "source_frame_id": source_frame_id,
            "source_rgb_path": str(rgb_path),
            "source_segmentation_path": str(segmentation_path),
            "step": step,
            "target_object_id": target_object_id,
            "unpadded_bbox_2d_xyxy": list(unpadded_bbox),
        },
    )


def _target_visual_context(
    case: dict[str, Any],
    reason: str,
) -> dict[str, Any] | None:
    target = _mapping_or_empty(case.get("target"))
    label = _optional_string(target.get("label"))
    if label is None:
        return None
    return {
        "available": True,
        "context_kind": "primary_frame_without_target_crop",
        "instruction": (
            "No local target crop is available. Inspect only the primary RGB "
            "frame; if the target is not visually clear, return "
            "target_not_observed instead of guessing."
        ),
        "primary_frame_role": "primary_frame",
        "target_crop_available": False,
        "target_crop_unavailable_reason": reason,
        "target_label": label,
    }


class _PPMImage:
    def __init__(self, *, data: bytes, height: int, width: int) -> None:
        self.data = data
        self.height = height
        self.width = width


def _read_ppm(path: Path) -> _PPMImage:
    raw = path.read_bytes()
    magic, index = _ppm_token(raw, 0)
    width_token, index = _ppm_token(raw, index)
    height_token, index = _ppm_token(raw, index)
    maxval_token, index = _ppm_token(raw, index)
    if maxval_token != b"255":
        raise ValueError(f"unsupported PPM maxval in {path}")
    width = int(width_token)
    height = int(height_token)
    if magic == b"P6":
        index = _skip_ppm_ws_and_comments(raw, index)
        pixel_data = raw[index : index + width * height * 3]
        if len(pixel_data) != width * height * 3:
            raise ValueError(f"PPM raster size mismatch in {path}")
        return _PPMImage(data=bytes(pixel_data), height=height, width=width)
    if magic == b"P3":
        values: list[int] = []
        while len(values) < width * height * 3:
            token, index = _ppm_token(raw, index)
            values.append(int(token))
        return _PPMImage(data=bytes(values), height=height, width=width)
    raise ValueError(f"unsupported PPM magic in {path}: {magic!r}")


def _write_ppm(path: Path, image: _PPMImage) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"P6\n{image.width} {image.height}\n255\n".encode("ascii") + image.data)


def _ppm_token(data: bytes, index: int) -> tuple[bytes, int]:
    index = _skip_ppm_ws_and_comments(data, index)
    start = index
    while index < len(data) and data[index] not in b" \t\r\n":
        index += 1
    if start == index:
        raise ValueError("unexpected end of PPM header")
    return data[start:index], index


def _skip_ppm_ws_and_comments(data: bytes, index: int) -> int:
    while index < len(data):
        if data[index] in b" \t\r\n":
            index += 1
            continue
        if data[index] == ord("#"):
            while index < len(data) and data[index] not in b"\r\n":
                index += 1
            continue
        break
    return index


def _color_bbox(
    image: _PPMImage,
    color: list[int],
) -> tuple[int, int, int, int] | None:
    min_x = image.width
    min_y = image.height
    max_x = -1
    max_y = -1
    r, g, b = color
    for y in range(image.height):
        row_start = y * image.width * 3
        for x in range(image.width):
            index = row_start + x * 3
            if image.data[index] == r and image.data[index + 1] == g and image.data[index + 2] == b:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
    if max_x < min_x or max_y < min_y:
        return None
    return min_x, min_y, max_x + 1, max_y + 1


def _pad_bbox(
    bbox: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    padding_pixels: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (
        max(0, x0 - padding_pixels),
        max(0, y0 - padding_pixels),
        min(width, x1 + padding_pixels),
        min(height, y1 + padding_pixels),
    )


def _crop_ppm(image: _PPMImage, bbox: tuple[int, int, int, int]) -> _PPMImage:
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    rows = []
    for y in range(y0, y1):
        start = (y * image.width + x0) * 3
        end = start + width * 3
        rows.append(image.data[start:end])
    return _PPMImage(data=b"".join(rows), height=height, width=width)


def _case_step(case: dict[str, Any]) -> int | None:
    primary = case.get("primary_frame")
    if isinstance(primary, dict):
        step = _int_or_none(primary.get("step"))
        if step is not None:
            return step
    situation = case.get("situation")
    if isinstance(situation, dict):
        step = _int_or_none(situation.get("step"))
        if step is not None:
            return step
    parts = _case_id(case).split(":")
    if len(parts) >= 3:
        return _int_or_none(parts[2])
    return None


def _target_object_id(case: dict[str, Any]) -> str | None:
    target = case.get("target")
    if isinstance(target, dict):
        value = target.get("object_id")
        if isinstance(value, str) and value:
            return value
    parts = _case_id(case).split(":")
    if len(parts) >= 5 and parts[4]:
        return parts[4]
    return None


def _primary_rgb_path(case: dict[str, Any]) -> Path | None:
    primary = case.get("primary_frame")
    if isinstance(primary, dict):
        return _first_path(primary.get("rgb_path"))
    return None


def _source_frame_id(case: dict[str, Any], step: int) -> str:
    primary = case.get("primary_frame")
    if isinstance(primary, dict):
        frame_id = primary.get("frame_id")
        if isinstance(frame_id, str) and frame_id:
            return frame_id
    episode_id = _safe_episode_id(case)
    scene_id = _optional_string(case.get("scene_id")) or "scene"
    return f"{episode_id}:{scene_id}:{step}"


def _safe_episode_id(case: dict[str, Any]) -> str:
    value = case.get("episode_id")
    if isinstance(value, str) and value:
        return _slug(value)
    return _slug(_case_id(case).split(":")[0])


def _case_id(case: dict[str, Any]) -> str:
    value = case.get("case_id")
    return value if isinstance(value, str) else "unknown"


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _rgb(value: object) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return None
    if not all(0 <= item <= 255 for item in value):
        return None
    return list(value)


def _first_path(*values: object) -> Path | None:
    for value in values:
        if isinstance(value, Path):
            return value
        if isinstance(value, str) and value:
            return Path(value)
    return None


def _mapping_or_empty(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def _short_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _forbidden_paths(value: object, *, prefix: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}"
            if key in FORBIDDEN_KEYS:
                paths.append(child_prefix)
            paths.extend(_forbidden_paths(child, prefix=child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_forbidden_paths(child, prefix=f"{prefix}[{index}]"))
    return paths


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _digest_without(payload: dict[str, Any], key_to_omit: str) -> str:
    normalized = {key: value for key, value in payload.items() if key != key_to_omit}
    return hashlib.sha256(
        json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True) + "\n", end="")


if __name__ == "__main__":
    raise SystemExit(main())
