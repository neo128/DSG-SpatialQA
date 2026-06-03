from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.episodes import (
    EPISODE_FRAME_SCHEMA_VERSION,
    EpisodeFrame,
    episode_sequence_digest,
    load_episode_sequence,
    validate_episode_sequence,
)
from dsg_spatialqa_lab.schema import SpatialQAError


REAL_COLLECTION_REPORT_SCHEMA_VERSION = "dsg-spatialqa-lab.real-collection-report.v1"
REAL_COLLECTION_REQUEST_BUNDLE_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.real-collection-request-bundle.v1"
)
DEFAULT_REQUIRED_REAL_FRAME_EVIDENCE = ("depth", "rgb", "segmentation")
SUPPORTED_REAL_COLLECTION_SOURCE_KINDS = ("ai2thor", "habitat")
NON_REAL_COLLECTION_MARKERS = (
    "dummy",
    "fake",
    "mock",
    "placeholder",
    "synthetic",
)


def real_collection_request_bundle(
    *,
    dataset_name: str,
    episode_paths: Sequence[str | Path],
    source_kind: str,
    report_path: str | Path,
    min_episode_count: int = 3,
    min_scene_count: int = 1,
    min_frame_count: int = 30,
    required_frame_evidence: Sequence[str] = DEFAULT_REQUIRED_REAL_FRAME_EVIDENCE,
) -> dict[str, Any]:
    _validate_non_empty_str(dataset_name, "dataset_name")
    _validate_non_empty_str(source_kind, "source_kind")
    _validate_threshold(min_episode_count, "min_episode_count")
    _validate_threshold(min_scene_count, "min_scene_count")
    _validate_threshold(min_frame_count, "min_frame_count")
    required_evidence = _unique_strings(
        required_frame_evidence,
        "required_frame_evidence",
    )
    paths = tuple(Path(path) for path in episode_paths)
    if not paths:
        raise SpatialQAError("Real collection request bundle requires at least one episode path")
    output_report_path = Path(report_path)
    bundle: dict[str, Any] = {
        "schema_version": REAL_COLLECTION_REQUEST_BUNDLE_SCHEMA_VERSION,
        "action": "real_collection_request_bundle",
        "dataset_name": dataset_name,
        "source_kind": source_kind,
        "episode_paths": [str(path) for path in paths],
        "report_path": str(output_report_path),
        "thresholds": {
            "min_episode_count": min_episode_count,
            "min_frame_count": min_frame_count,
            "min_scene_count": min_scene_count,
        },
        "required_frame_evidence": list(required_evidence),
        "frame_asset_fields": [
            "rgb_path",
            "depth_path",
            "segmentation_path",
        ],
        "commands": {
            "collection_report": _real_collection_report_command(
                dataset_name=dataset_name,
                source_kind=source_kind,
                episode_paths=paths,
                report_path=output_report_path,
                min_episode_count=min_episode_count,
                min_scene_count=min_scene_count,
                min_frame_count=min_frame_count,
                required_frame_evidence=required_evidence,
            ),
            "compare_report": (
                "python scripts/check_real_collection.py "
                f"--compare-report {output_report_path}"
            ),
            "validate_report": (
                "python scripts/check_real_collection.py "
                f"--validate-report {output_report_path}"
            ),
        },
        "episode_record_template": _episode_record_template(source_kind),
    }
    bundle["request_bundle_digest"] = real_collection_request_bundle_digest(bundle)
    return bundle


def real_collection_request_bundle_digest(bundle: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in bundle.items()
        if key != "request_bundle_digest"
    }
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def real_collection_request_bundle_json(bundle: Mapping[str, Any]) -> str:
    return json.dumps(bundle, indent=2, sort_keys=True) + "\n"


def save_real_collection_request_bundle(
    bundle: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        real_collection_request_bundle_json(bundle),
        encoding="utf-8",
    )
    return output_path


def load_real_collection_request_bundle(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Real collection request bundle JSON must be an object")
    schema_version = payload.get("schema_version")
    if schema_version != REAL_COLLECTION_REQUEST_BUNDLE_SCHEMA_VERSION:
        raise SpatialQAError(
            f"Unsupported real collection request bundle schema version: {schema_version}"
        )
    return cast(dict[str, Any], payload)


def real_collection_report(
    *,
    dataset_name: str,
    episode_paths: Sequence[str | Path],
    source_kind: str,
    required_adapter: str | None = None,
    min_episode_count: int = 3,
    min_scene_count: int = 1,
    min_frame_count: int = 30,
    required_frame_evidence: Sequence[str] = DEFAULT_REQUIRED_REAL_FRAME_EVIDENCE,
) -> dict[str, Any]:
    _validate_non_empty_str(dataset_name, "dataset_name")
    _validate_non_empty_str(source_kind, "source_kind")
    if required_adapter is not None:
        _validate_non_empty_str(required_adapter, "required_adapter")
    _validate_threshold(min_episode_count, "min_episode_count")
    _validate_threshold(min_scene_count, "min_scene_count")
    _validate_threshold(min_frame_count, "min_frame_count")
    required_evidence = _unique_strings(
        required_frame_evidence,
        "required_frame_evidence",
    )
    paths = tuple(Path(path) for path in episode_paths)
    if not paths:
        raise SpatialQAError("Real collection report requires at least one episode path")
    frames_by_path = {path: load_episode_sequence(path) for path in paths}
    summary = _json_safe(_collection_summary(frames_by_path))
    checks = _json_safe(_checks(
        source_kind=source_kind,
        required_adapter=required_adapter,
        frames_by_path=frames_by_path,
        summary=summary,
        min_episode_count=min_episode_count,
        min_scene_count=min_scene_count,
        min_frame_count=min_frame_count,
        required_frame_evidence=required_evidence,
    ))
    report: dict[str, Any] = {
        "schema_version": REAL_COLLECTION_REPORT_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "source_kind": source_kind,
        "episode_paths": [str(path) for path in paths],
        "thresholds": {
            "min_episode_count": min_episode_count,
            "min_frame_count": min_frame_count,
            "min_scene_count": min_scene_count,
        },
        "required_frame_evidence": list(required_evidence),
        "collection_summary": summary,
        "checks": checks,
        "readiness": _readiness_from_checks(checks),
    }
    if required_adapter is not None:
        report["required_adapter"] = required_adapter
    report["report_digest"] = real_collection_report_digest(report)
    return report


def real_collection_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def real_collection_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_real_collection_report(report: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(real_collection_report_json(report), encoding="utf-8")
    return output_path


def load_real_collection_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Real collection report JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_real_collection_report(report: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = report.get("schema_version")
    report_digest = _string_or_none(report.get("report_digest"))
    expected_digest = real_collection_report_digest(report)
    checks = _mapping_sequence(report.get("checks"))
    expected_readiness = _readiness_from_checks(checks)
    checks_out = [
        {
            "name": "schema_version",
            "passed": schema_version == REAL_COLLECTION_REPORT_SCHEMA_VERSION,
            "expected": REAL_COLLECTION_REPORT_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "report_digest",
            "passed": report_digest == expected_digest,
            "expected": expected_digest,
            "actual": report_digest,
        },
        {
            "name": "readiness_matches_checks",
            "passed": report.get("readiness") == expected_readiness,
            "expected": expected_readiness,
            "actual": report.get("readiness"),
        },
        {
            "name": "thresholds_shape",
            "passed": _thresholds_shape_valid(report.get("thresholds")),
        },
        {
            "name": "required_frame_evidence_shape",
            "passed": _string_sequence(report.get("required_frame_evidence")),
        },
        {
            "name": "collection_summary_shape",
            "passed": _collection_summary_shape_valid(report.get("collection_summary")),
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks_out),
        "schema_version": schema_version,
        "report_digest": report_digest,
        "checks": checks_out,
    }


def compare_real_collection_report(report: Mapping[str, Any]) -> dict[str, Any]:
    thresholds = _mapping(report.get("thresholds"), "thresholds")
    current_report = real_collection_report(
        dataset_name=_required_str(report, "dataset_name"),
        episode_paths=_string_tuple(report, "episode_paths"),
        source_kind=_required_str(report, "source_kind"),
        required_adapter=_optional_non_empty_str(report.get("required_adapter")),
        min_episode_count=_required_int(thresholds, "min_episode_count"),
        min_scene_count=_required_int(thresholds, "min_scene_count"),
        min_frame_count=_required_int(thresholds, "min_frame_count"),
        required_frame_evidence=_string_tuple(report, "required_frame_evidence"),
    )
    validation = validate_real_collection_report(report)
    saved_digest = _string_or_none(report.get("report_digest"))
    current_digest = _string_or_none(current_report.get("report_digest"))
    checks = [
        {"name": "saved_report_valid", "passed": validation["valid"] is True},
        _equality_check(
            "collection_summary_matches_current",
            report.get("collection_summary"),
            current_report["collection_summary"],
        ),
        _equality_check(
            "checks_match_current",
            report.get("checks"),
            current_report["checks"],
        ),
        _equality_check(
            "readiness_matches_current",
            report.get("readiness"),
            current_report["readiness"],
        ),
        _equality_check(
            "report_digest_matches_current",
            saved_digest,
            current_digest,
        ),
    ]
    return {
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def validate_real_collection_request_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = bundle.get("schema_version")
    action = bundle.get("action")
    request_digest = _string_or_none(bundle.get("request_bundle_digest"))
    expected_digest = real_collection_request_bundle_digest(bundle)
    current_bundle, rebuild_error = _current_real_collection_request_bundle(bundle)
    current_commands = (
        current_bundle.get("commands") if current_bundle is not None else None
    )
    current_template = (
        current_bundle.get("episode_record_template")
        if current_bundle is not None
        else None
    )
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == REAL_COLLECTION_REQUEST_BUNDLE_SCHEMA_VERSION,
            "expected": REAL_COLLECTION_REQUEST_BUNDLE_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "action",
            "passed": action == "real_collection_request_bundle",
            "expected": "real_collection_request_bundle",
            "actual": action,
        },
        {
            "name": "request_bundle_digest",
            "passed": request_digest == expected_digest,
            "expected": expected_digest,
            "actual": request_digest,
        },
        {
            "name": "dataset_name_shape",
            "passed": _non_empty_string(bundle.get("dataset_name")),
        },
        {
            "name": "source_kind_supported",
            "passed": bundle.get("source_kind") in SUPPORTED_REAL_COLLECTION_SOURCE_KINDS,
            "expected": list(SUPPORTED_REAL_COLLECTION_SOURCE_KINDS),
            "actual": bundle.get("source_kind"),
        },
        {
            "name": "episode_paths_shape",
            "passed": _non_empty_string_sequence(bundle.get("episode_paths")),
        },
        {
            "name": "report_path_shape",
            "passed": _non_empty_string(bundle.get("report_path")),
        },
        {
            "name": "thresholds_shape",
            "passed": _thresholds_shape_valid(bundle.get("thresholds")),
        },
        {
            "name": "required_frame_evidence_shape",
            "passed": _non_empty_string_sequence(bundle.get("required_frame_evidence")),
        },
        {
            "name": "frame_asset_fields",
            "passed": bundle.get("frame_asset_fields")
            == ["rgb_path", "depth_path", "segmentation_path"],
            "expected": ["rgb_path", "depth_path", "segmentation_path"],
            "actual": bundle.get("frame_asset_fields"),
        },
        {
            "name": "episode_record_template",
            "passed": bundle.get("episode_record_template") == current_template,
            "expected": current_template,
            "actual": bundle.get("episode_record_template"),
        },
        {
            "name": "commands_match_bundle",
            "passed": bundle.get("commands") == current_commands,
            "expected": current_commands,
            "actual": bundle.get("commands"),
        },
    ]
    if rebuild_error is not None:
        checks.append(
            {
                "name": "bundle_fields_rebuildable",
                "passed": False,
                "error": rebuild_error,
            }
        )
    return {
        "action": "validate_real_collection_request_bundle",
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "request_bundle_digest": request_digest,
        "checks": checks,
    }


def compare_real_collection_request_bundle(
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_real_collection_request_bundle(bundle)
    current_bundle, rebuild_error = _current_real_collection_request_bundle(bundle)
    current_digest = (
        _string_or_none(current_bundle.get("request_bundle_digest"))
        if current_bundle is not None
        else None
    )
    saved_digest = _string_or_none(bundle.get("request_bundle_digest"))
    checks = [
        {"name": "saved_request_bundle_valid", "passed": validation["valid"] is True},
        _equality_check("request_bundle_matches_current", bundle, current_bundle),
        _equality_check(
            "request_bundle_digest_matches_current",
            saved_digest,
            current_digest,
        ),
    ]
    if rebuild_error is not None:
        checks.append(
            {
                "name": "current_request_bundle_rebuildable",
                "passed": False,
                "error": rebuild_error,
            }
        )
    return {
        "action": "compare_real_collection_request_bundle",
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def _episode_record_template(source_kind: str) -> dict[str, Any]:
    return {
        "schema_version": EPISODE_FRAME_SCHEMA_VERSION,
        "episode_id": "episode_001",
        "scene_id": "scene_001",
        "step": 1,
        "rgb_path": "frames/000001.rgb.png",
        "depth_path": "frames/000001.depth.png",
        "segmentation_path": "frames/000001.segmentation.png",
        "agent_id": "agent",
        "agent_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        "action": "Initialize",
        "visible_object_ids": [],
        "metadata": {
            "adapter": source_kind,
            "collection_kind": "real",
            "simulator": source_kind,
            "source_kind": "real_simulator",
        },
    }


def _current_real_collection_request_bundle(
    bundle: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        thresholds = _mapping(bundle.get("thresholds"), "thresholds")
        current = real_collection_request_bundle(
            dataset_name=_required_str(bundle, "dataset_name"),
            episode_paths=_string_tuple(bundle, "episode_paths"),
            source_kind=_required_str(bundle, "source_kind"),
            report_path=_required_str(bundle, "report_path"),
            min_episode_count=_required_int(thresholds, "min_episode_count"),
            min_scene_count=_required_int(thresholds, "min_scene_count"),
            min_frame_count=_required_int(thresholds, "min_frame_count"),
            required_frame_evidence=_string_tuple(
                bundle,
                "required_frame_evidence",
            ),
        )
    except (SpatialQAError, ValueError, TypeError) as exc:
        return None, str(exc)
    return current, None


def _real_collection_report_command(
    *,
    dataset_name: str,
    source_kind: str,
    episode_paths: Sequence[Path],
    report_path: Path,
    min_episode_count: int,
    min_scene_count: int,
    min_frame_count: int,
    required_frame_evidence: Sequence[str],
) -> str:
    parts = [
        "python scripts/check_real_collection.py",
        f"--dataset-name {dataset_name}",
        f"--source-kind {source_kind}",
    ]
    parts.extend(f"--episode {path}" for path in episode_paths)
    parts.extend(
        [
            f"--report {report_path}",
            f"--min-episode-count {min_episode_count}",
            f"--min-scene-count {min_scene_count}",
            f"--min-frame-count {min_frame_count}",
        ]
    )
    parts.extend(
        f"--required-frame-evidence {evidence}"
        for evidence in required_frame_evidence
    )
    return " ".join(parts)


def _collection_summary(
    frames_by_path: Mapping[Path, Sequence[EpisodeFrame]],
) -> dict[str, Any]:
    frames = [frame for frames in frames_by_path.values() for frame in frames]
    return {
        "action_counts": _sorted_counts(
            frame.action for frame in frames if isinstance(frame.action, str)
        ),
        "adapter_counts": _sorted_counts(_adapter(frame) for frame in frames),
        "agent_pose_frame_count": len(frames),
        "asset_summary": _frame_asset_summary(frames_by_path),
        "collection_kind_counts": _sorted_counts(
            _collection_kind(frame) for frame in frames
        ),
        "depth_frame_count": sum(1 for frame in frames if _has_path(frame.depth_path)),
        "episode_count": len({frame.episode_id for frame in frames}),
        "episode_digests": {
            str(path): episode_sequence_digest(frames_by_path[path])
            for path in sorted(frames_by_path, key=str)
        },
        "frame_count": len(frames),
        "frame_source_kind_counts": _sorted_counts(
            _frame_source_kind(frame) for frame in frames
        ),
        "rgb_frame_count": sum(1 for frame in frames if _has_path(frame.rgb_path)),
        "scene_count": len({frame.scene_id for frame in frames}),
        "segmentation_frame_count": sum(
            1 for frame in frames if _has_path(frame.segmentation_path)
        ),
        "simulator_counts": _sorted_counts(_simulator(frame) for frame in frames),
        "source_kind_counts": _sorted_counts(_source_kind(frame) for frame in frames),
        "visible_object_frame_count": sum(
            1 for frame in frames if len(frame.visible_object_ids) > 0
        ),
        "visible_object_nonempty_ratio": (
            sum(1 for frame in frames if len(frame.visible_object_ids) > 0) / len(frames)
            if frames
            else 0.0
        ),
    }


def _checks(
    *,
    source_kind: str,
    required_adapter: str | None,
    frames_by_path: Mapping[Path, Sequence[EpisodeFrame]],
    summary: Mapping[str, Any],
    min_episode_count: int,
    min_scene_count: int,
    min_frame_count: int,
    required_frame_evidence: Sequence[str],
) -> list[dict[str, Any]]:
    sequence_validations = {
        str(path): validate_episode_sequence(frames)
        for path, frames in sorted(frames_by_path.items(), key=lambda item: str(item[0]))
    }
    source_kind_counts = _int_mapping(summary.get("source_kind_counts"))
    adapter_counts = _int_mapping(summary.get("adapter_counts"))
    frame_source_kind_counts = _int_mapping(summary.get("frame_source_kind_counts"))
    simulator_counts = _int_mapping(summary.get("simulator_counts"))
    collection_kind_counts = _int_mapping(summary.get("collection_kind_counts"))
    asset_summary = _mapping(summary.get("asset_summary"), "asset_summary")
    missing_evidence = [
        evidence
        for evidence in required_frame_evidence
        if _evidence_count(summary, evidence) < _int_value(summary, "frame_count")
    ]
    mock_markers = _mock_markers(frames_by_path)
    non_real_markers = _non_real_markers(frames_by_path)
    expected_source_counts = {source_kind: _int_value(summary, "frame_count")}
    expected_required_adapter_counts = (
        {required_adapter: _int_value(summary, "frame_count")}
        if required_adapter is not None
        else None
    )
    checks = [
        {
            "name": "episode_sequences_valid",
            "passed": all(item["valid"] is True for item in sequence_validations.values()),
            "actual": sequence_validations,
        },
        {
            "name": "source_kind_supported",
            "passed": source_kind in SUPPORTED_REAL_COLLECTION_SOURCE_KINDS,
            "expected": list(SUPPORTED_REAL_COLLECTION_SOURCE_KINDS),
            "actual": source_kind,
        },
        {
            "name": "adapter_supported",
            "passed": _adapters_supported(adapter_counts),
            "expected": list(SUPPORTED_REAL_COLLECTION_SOURCE_KINDS),
            "actual": adapter_counts,
        },
        _minimum_check(
            "episode_count_minimum",
            _int_value(summary, "episode_count"),
            min_episode_count,
        ),
        _minimum_check(
            "scene_count_minimum",
            _int_value(summary, "scene_count"),
            min_scene_count,
        ),
        _minimum_check(
            "frame_count_minimum",
            _int_value(summary, "frame_count"),
            min_frame_count,
        ),
        {
            "name": "source_kind_matches_frames",
            "passed": source_kind_counts == expected_source_counts,
            "expected": expected_source_counts,
            "actual": source_kind_counts,
        },
        {
            "name": "collection_kind_real",
            "passed": collection_kind_counts == {"real": _int_value(summary, "frame_count")},
            "expected": {"real": _int_value(summary, "frame_count")},
            "actual": collection_kind_counts,
        },
        {
            "name": "required_frame_evidence_present",
            "passed": len(missing_evidence) == 0,
            "required": list(required_frame_evidence),
            "missing": missing_evidence,
            "actual": {
                evidence: _evidence_count(summary, evidence)
                for evidence in required_frame_evidence
            },
        },
        {
            "name": "frame_assets_present",
            "passed": _int_value(asset_summary, "missing_asset_count") == 0,
            "asset_path_count": _int_value(asset_summary, "asset_path_count"),
            "missing": _json_safe(asset_summary.get("missing_assets", [])),
            "missing_asset_count": _int_value(
                asset_summary,
                "missing_asset_count",
            ),
            "present_asset_count": _int_value(
                asset_summary,
                "present_asset_count",
            ),
        },
        {
            "name": "mock_markers_absent",
            "passed": len(mock_markers) == 0,
            "actual": mock_markers,
        },
        {
            "name": "non_real_markers_absent",
            "passed": len(non_real_markers) == 0,
            "actual": non_real_markers,
        },
        {
            "name": "visible_object_ids_observed",
            "passed": _int_value(summary, "visible_object_frame_count") > 0,
            "actual": _int_value(summary, "visible_object_frame_count"),
        },
        {
            "name": "agent_pose_present",
            "passed": _int_value(summary, "agent_pose_frame_count")
            == _int_value(summary, "frame_count"),
            "expected": _int_value(summary, "frame_count"),
            "actual": _int_value(summary, "agent_pose_frame_count"),
        },
        {
            "name": "action_coverage",
            "passed": sum(_int_mapping(summary.get("action_counts")).values())
            == _int_value(summary, "frame_count"),
            "expected": _int_value(summary, "frame_count"),
            "actual": sum(_int_mapping(summary.get("action_counts")).values()),
        },
    ]
    if required_adapter is not None:
        checks.extend(
            [
                {
                    "name": "required_adapter_supported",
                    "passed": required_adapter in SUPPORTED_REAL_COLLECTION_SOURCE_KINDS,
                    "expected": list(SUPPORTED_REAL_COLLECTION_SOURCE_KINDS),
                    "actual": required_adapter,
                },
                {
                    "name": "required_adapter_matches_frames",
                    "passed": adapter_counts == expected_required_adapter_counts,
                    "expected": expected_required_adapter_counts,
                    "actual": adapter_counts,
                },
                {
                    "name": "frame_source_kind_real_simulator",
                    "passed": frame_source_kind_counts
                    == {"real_simulator": _int_value(summary, "frame_count")},
                    "expected": {"real_simulator": _int_value(summary, "frame_count")},
                    "actual": frame_source_kind_counts,
                },
                {
                    "name": "simulator_matches_required_adapter",
                    "passed": simulator_counts == expected_required_adapter_counts,
                    "expected": expected_required_adapter_counts,
                    "actual": simulator_counts,
                },
            ]
        )
    return checks


def _frame_asset_summary(
    frames_by_path: Mapping[Path, Sequence[EpisodeFrame]],
) -> dict[str, Any]:
    field_kinds = (
        ("depth_path", "depth"),
        ("rgb_path", "rgb"),
        ("segmentation_path", "segmentation"),
    )
    asset_kind_counts = {"depth": 0, "rgb": 0, "segmentation": 0}
    asset_path_count = 0
    missing_assets: list[dict[str, Any]] = []
    for episode_path, frames in sorted(frames_by_path.items(), key=lambda item: str(item[0])):
        for frame in frames:
            for field_name, kind in field_kinds:
                value = getattr(frame, field_name)
                if not _has_path(value):
                    continue
                asset_path_count += 1
                asset_kind_counts[kind] += 1
                path_text = cast(str, value)
                local_path = Path(path_text)
                resolved_path = (
                    local_path if local_path.is_absolute() else episode_path.parent / local_path
                )
                if not resolved_path.exists():
                    missing_assets.append(
                        {
                            "episode_path": str(episode_path),
                            "kind": kind,
                            "path": path_text,
                            "resolved_path": str(resolved_path),
                            "step": frame.step,
                        }
                    )
    return {
        "asset_kind_counts": {
            key: asset_kind_counts[key] for key in sorted(asset_kind_counts)
        },
        "asset_path_count": asset_path_count,
        "missing_asset_count": len(missing_assets),
        "missing_assets": missing_assets,
        "present_asset_count": asset_path_count - len(missing_assets),
    }


def _collection_kind(frame: EpisodeFrame) -> str:
    value = frame.metadata.get("collection_kind")
    if value is None and frame.metadata.get("source_kind") == "real_simulator":
        return "real"
    return value if isinstance(value, str) and value != "" else "unspecified"


def _source_kind(frame: EpisodeFrame) -> str:
    adapter = frame.metadata.get("adapter")
    if isinstance(adapter, str) and adapter != "":
        return adapter
    value = frame.metadata.get("source_kind")
    if isinstance(value, str) and value != "":
        return value
    return "unspecified"


def _adapter(frame: EpisodeFrame) -> str:
    value = frame.metadata.get("adapter")
    return value if isinstance(value, str) and value != "" else "unspecified"


def _frame_source_kind(frame: EpisodeFrame) -> str:
    value = frame.metadata.get("source_kind")
    return value if isinstance(value, str) and value != "" else "unspecified"


def _simulator(frame: EpisodeFrame) -> str:
    value = frame.metadata.get("simulator")
    return value if isinstance(value, str) and value != "" else "unspecified"


def _adapters_supported(adapter_counts: Mapping[str, int]) -> bool:
    if not adapter_counts:
        return False
    return all(adapter in SUPPORTED_REAL_COLLECTION_SOURCE_KINDS for adapter in adapter_counts)


def _mock_markers(frames_by_path: Mapping[Path, Sequence[EpisodeFrame]]) -> list[str]:
    markers: set[str] = set()
    for frames in frames_by_path.values():
        for frame in frames:
            if "mock" in frame.episode_id.lower():
                markers.add(f"episode_id:{frame.episode_id}")
            if "mock" in frame.scene_id.lower():
                markers.add(f"scene_id:{frame.scene_id}")
            for artifact_path in (
                frame.depth_path,
                frame.rgb_path,
                frame.segmentation_path,
            ):
                if (
                    isinstance(artifact_path, str)
                    and "mock" in artifact_path.lower()
                ):
                    markers.add(f"path:{artifact_path}")
            for key, value in frame.metadata.items():
                if isinstance(value, str) and "mock" in value.lower():
                    markers.add(f"metadata:{key}:{value}")
    return sorted(markers)


def _non_real_markers(
    frames_by_path: Mapping[Path, Sequence[EpisodeFrame]],
) -> list[str]:
    markers: set[str] = set()
    for frames in frames_by_path.values():
        for frame in frames:
            if _contains_non_real_marker(frame.episode_id):
                markers.add(f"episode_id:{frame.episode_id}")
            if _contains_non_real_marker(frame.scene_id):
                markers.add(f"scene_id:{frame.scene_id}")
            for artifact_path in (
                frame.depth_path,
                frame.rgb_path,
                frame.segmentation_path,
            ):
                if isinstance(artifact_path, str) and _contains_non_real_marker(
                    artifact_path
                ):
                    markers.add(f"path:{artifact_path}")
            markers.update(_metadata_non_real_markers(frame.metadata, "metadata"))
    return sorted(markers)


def _metadata_non_real_markers(value: object, prefix: str) -> set[str]:
    markers: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in sorted(value.items(), key=lambda item: str(item[0])):
            if isinstance(key, str) and key != "":
                markers.update(_metadata_non_real_markers(item, f"{prefix}:{key}"))
        return markers
    if isinstance(value, Sequence) and not isinstance(value, str):
        for index, item in enumerate(value):
            markers.update(_metadata_non_real_markers(item, f"{prefix}:{index}"))
        return markers
    if isinstance(value, str) and _contains_non_real_marker(value):
        markers.add(f"{prefix}:{value}")
    return markers


def _contains_non_real_marker(value: str) -> bool:
    normalized = value.lower()
    return any(marker in normalized for marker in NON_REAL_COLLECTION_MARKERS)


def _evidence_count(summary: Mapping[str, Any], evidence: str) -> int:
    if evidence == "depth":
        return _int_value(summary, "depth_frame_count")
    if evidence == "rgb":
        return _int_value(summary, "rgb_frame_count")
    if evidence == "segmentation":
        return _int_value(summary, "segmentation_frame_count")
    return 0


def _minimum_check(name: str, actual: int, expected: int) -> dict[str, Any]:
    return {
        "name": name,
        "passed": actual >= expected,
        "expected": expected,
        "actual": actual,
    }


def _readiness_from_checks(checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed = [
        _required_mapping_str(check, "name")
        for check in checks
        if check.get("passed") is not True
    ]
    return {
        "ready": len(failed) == 0,
        "failed_check_count": len(failed),
        "failed_checks": failed,
    }


def _validate_non_empty_str(value: object, field_name: str) -> None:
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"{field_name} must be a non-empty string")


def _validate_threshold(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SpatialQAError(f"{field_name} must be a positive integer")


def _unique_strings(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    strings: list[str] = []
    for value in values:
        if not isinstance(value, str) or value == "":
            raise SpatialQAError(f"{field_name} entries must be non-empty strings")
        strings.append(value)
    return tuple(sorted(set(strings)))


def _thresholds_shape_valid(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return (
        _int_value(value, "min_episode_count") > 0
        and _int_value(value, "min_frame_count") > 0
        and _int_value(value, "min_scene_count") > 0
    )


def _collection_summary_shape_valid(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return all(
        key in value
        for key in (
            "collection_kind_counts",
            "episode_count",
            "frame_count",
            "scene_count",
            "source_kind_counts",
        )
    )


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpatialQAError(f"Real collection report field must be an object: {field_name}")
    return cast(Mapping[str, Any], value)


def _mapping_sequence(value: object) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return ()
    return tuple(cast(Mapping[str, Any], item) for item in value if isinstance(item, Mapping))


def _string_tuple(report: Mapping[str, Any], field_name: str) -> tuple[str, ...]:
    value = report.get(field_name)
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise SpatialQAError(f"Real collection report field must contain strings: {field_name}")
    strings = tuple(item for item in value if isinstance(item, str) and item != "")
    if not strings:
        raise SpatialQAError(f"Real collection report field must contain strings: {field_name}")
    return strings


def _string_sequence(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str)
        and all(isinstance(item, str) for item in value)
    )


def _non_empty_string(value: object) -> bool:
    return isinstance(value, str) and value != ""


def _non_empty_string_sequence(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str)
        and len(value) > 0
        and all(isinstance(item, str) and item != "" for item in value)
    )


def _int_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    output: dict[str, int] = {}
    for key, item in value.items():
        if isinstance(key, str) and isinstance(item, int) and not isinstance(item, bool):
            output[key] = item
    return output


def _required_int(payload: Mapping[str, Any], field_name: str) -> int:
    value = payload.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpatialQAError(f"Real collection report field must be an integer: {field_name}")
    return value


def _int_value(payload: Mapping[str, Any], field_name: str) -> int:
    value = payload.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _required_str(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Real collection report field must be a string: {field_name}")
    return value


def _required_mapping_str(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Real collection report field must be a string: {field_name}")
    return value


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value != "" else None


def _optional_non_empty_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        raise SpatialQAError("Real collection report optional field must be a string")
    return value


def _has_path(value: object) -> bool:
    return isinstance(value, str) and value != ""


def _json_safe(value: object) -> Any:
    return json.loads(json.dumps(value, separators=(",", ":"), sort_keys=True))


def _sorted_counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def _equality_check(name: str, expected: object, actual: object) -> dict[str, Any]:
    return {
        "name": name,
        "passed": expected == actual,
        "expected": expected,
        "actual": actual,
    }
