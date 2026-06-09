#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "dsg-spatialqa-lab.active-qa-v2-target-crop-feasibility-report.v1"
ACTIVE_QA_V2_REQUEST_BUNDLE_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.active-qa-v2-vlm-request-bundle.v1"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Audit whether active QA v2 VLM request cases have enough local "
            "observation evidence to build target crops."
        ),
    )
    parser.add_argument("--request-bundle", type=Path, required=True)
    parser.add_argument(
        "--observation-sequence",
        type=Path,
        action="append",
        required=True,
        help="SceneObservation sequence JSON. May be supplied more than once.",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        bundle = _load_bundle(args.request_bundle)
        observations = _load_observation_objects(args.observation_sequence)
        report = _audit_feasibility(
            bundle,
            observations,
            request_bundle=args.request_bundle,
            observation_sequences=args.observation_sequence,
        )
        _write_json(args.report, report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "blockers": ["target_crop_feasibility_audit_error"],
            "crop_generation_ready": False,
            "error": str(exc),
            "ready": False,
        }
        report["report_digest"] = _digest_without(report, "report_digest")
        _write_json(args.report, report)
        _emit(report)
        return 1

    _emit(
        {
            "action": "audit_active_qa_v2_target_crop_feasibility",
            "blockers": report["blockers"],
            "crop_generation_ready": report["crop_generation_ready"],
            "ready": report["ready"],
            "report": str(args.report),
            "request_bundle": str(args.request_bundle),
        }
    )
    return 0


def _load_bundle(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("request bundle must be a JSON object")
    if payload.get("schema_version") != ACTIVE_QA_V2_REQUEST_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported_request_bundle_schema")
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


def _audit_feasibility(
    bundle: dict[str, Any],
    observations: dict[tuple[int, str], dict[str, Any]],
    *,
    request_bundle: Path,
    observation_sequences: list[Path],
) -> dict[str, Any]:
    cases = [case for case in bundle["prediction_cases"] if isinstance(case, dict)]
    counters: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    feasible_case_ids: list[str] = []
    infeasible_cases: list[dict[str, str]] = []

    for case in cases:
        case_id = _case_id(case)
        step = _case_step(case)
        target_id = _target_object_id(case)
        if target_id is not None:
            counters["cases_with_target_id"] += 1
        else:
            reasons["missing_target_id"] += 1
            infeasible_cases.append({"case_id": case_id, "reason": "missing_target_id"})
            continue

        observation_object = observations.get((step, target_id)) if step is not None else None
        if observation_object is None:
            reasons["missing_matching_observation_object"] += 1
            infeasible_cases.append(
                {"case_id": case_id, "reason": "missing_matching_observation_object"}
            )
            continue
        counters["cases_with_matching_observation_object"] += 1

        features = _crop_features(case, observation_object)
        for key, value in features.items():
            if value is True:
                counters[key] += 1

        if features["cases_with_bbox_3d"] and not (
            features["cases_with_bbox_2d"]
            or features["cases_with_existing_mask_path"]
            or features["cases_with_segmentation_color"]
        ):
            counters["cases_with_bbox_3d_only"] += 1

        if features["cases_with_existing_rgb_path"] and (
            features["cases_with_bbox_2d"]
            or features["cases_with_existing_mask_path"]
            or (
                features["cases_with_segmentation_path"]
                and features["cases_with_segmentation_color"]
            )
        ):
            feasible_case_ids.append(case_id)
            continue

        if not features["cases_with_existing_rgb_path"]:
            reason = "missing_rgb_path"
        else:
            reason = "missing_bbox_2d_mask_or_segmentation_color"
        reasons[reason] += 1
        infeasible_cases.append({"case_id": case_id, "reason": reason})

    feasible_count = len(feasible_case_ids)
    request_count = len(cases)
    blockers: list[str] = []
    if request_count > 0 and feasible_count == 0:
        blockers.append("target_crop_artifacts_missing")
    elif feasible_count < request_count:
        blockers.append("target_crop_artifacts_partial")

    summary = {
        "cases_with_bbox_2d": counters["cases_with_bbox_2d"],
        "cases_with_bbox_3d": counters["cases_with_bbox_3d"],
        "cases_with_bbox_3d_only": counters["cases_with_bbox_3d_only"],
        "cases_with_existing_mask_path": counters["cases_with_existing_mask_path"],
        "cases_with_existing_rgb_path": counters["cases_with_existing_rgb_path"],
        "cases_with_matching_observation_object": counters[
            "cases_with_matching_observation_object"
        ],
        "cases_with_segmentation_color": counters["cases_with_segmentation_color"],
        "cases_with_segmentation_path": counters["cases_with_segmentation_path"],
        "cases_with_target_id": counters["cases_with_target_id"],
        "feasible_target_crop_case_count": feasible_count,
        "infeasible_target_crop_case_count": request_count - feasible_count,
        "observation_object_count": len(observations),
        "request_count": request_count,
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "blockers": blockers,
        "crop_generation_ready": request_count > 0 and feasible_count == request_count,
        "feasible_case_ids": feasible_case_ids,
        "infeasible_cases_sample": infeasible_cases[:50],
        "infeasible_reasons": dict(sorted(reasons.items())),
        "observation_sequences": [str(path) for path in observation_sequences],
        "ready": True,
        "request_bundle": str(request_bundle),
        "summary": summary,
    }
    report["report_digest"] = _digest_without(report, "report_digest")
    return report


def _crop_features(case: dict[str, Any], observation_object: dict[str, Any]) -> dict[str, bool]:
    attributes = observation_object.get("attributes")
    attrs = attributes if isinstance(attributes, dict) else {}
    rgb_path = _first_string(
        attrs.get("rgb_path"),
        observation_object.get("rgb_path"),
        _primary_rgb_path(case),
    )
    mask_path = _first_string(attrs.get("mask_path"), observation_object.get("mask_path"))
    segmentation_path = _first_string(
        attrs.get("segmentation_path"), observation_object.get("segmentation_path")
    )
    return {
        "cases_with_bbox_2d": _bbox2d(attrs.get("bbox_2d_xyxy"))
        or _bbox2d(observation_object.get("bbox_2d_xyxy")),
        "cases_with_bbox_3d": isinstance(observation_object.get("bbox"), dict),
        "cases_with_existing_mask_path": _path_exists(mask_path),
        "cases_with_existing_rgb_path": _path_exists(rgb_path),
        "cases_with_segmentation_color": _segmentation_color(attrs)
        or _segmentation_color(observation_object),
        "cases_with_segmentation_path": _path_exists(segmentation_path),
    }


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


def _primary_rgb_path(case: dict[str, Any]) -> str | None:
    primary = case.get("primary_frame")
    if isinstance(primary, dict):
        value = primary.get("rgb_path")
        if isinstance(value, str) and value:
            return value
    return None


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


def _bbox2d(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    return all(isinstance(item, int | float) and not isinstance(item, bool) for item in value)


def _segmentation_color(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    color = value.get("segmentation_color_rgb") or value.get("segmentation_color")
    if not isinstance(color, list) or len(color) != 3:
        return False
    return all(isinstance(item, int) and not isinstance(item, bool) for item in color)


def _first_string(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _path_exists(value: str | None) -> bool:
    return isinstance(value, str) and bool(value) and Path(value).exists()


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
