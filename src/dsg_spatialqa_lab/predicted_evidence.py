from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.observations import (
    SceneObservation,
    load_scene_observation_sequence,
    scene_observation_sequence_digest,
)
from dsg_spatialqa_lab.predicted import (
    load_predicted_graph_report,
    predicted_graph_report_digest,
    validate_predicted_graph_report,
)
from dsg_spatialqa_lab.schema import SpatialQAError


PREDICTED_DSG_EVIDENCE_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.predicted-dsg-evidence-report.v1"
)
DEFAULT_REQUIRED_PREDICTED_DSG_EVIDENCE_KINDS = ("depth", "detector", "rgb")
NON_REAL_PREDICTED_SOURCE_MARKERS = (
    "ai2thor",
    "dummy",
    "fake",
    "mock",
    "placeholder",
    "synthetic",
)


def predicted_dsg_evidence_report(
    predicted_graph_report: Mapping[str, Any],
    *,
    predicted_graph_report_path: str | Path | None = None,
    observation_sequence_path: str | Path | None = None,
    min_observation_count: int = 2,
    min_object_observation_count: int = 2,
    required_evidence_kinds: Sequence[str] = DEFAULT_REQUIRED_PREDICTED_DSG_EVIDENCE_KINDS,
) -> dict[str, Any]:
    _validate_threshold(min_observation_count, "min_observation_count")
    _validate_threshold(min_object_observation_count, "min_object_observation_count")
    required_kinds = _unique_strings(required_evidence_kinds, "required_evidence_kinds")
    input_kind = _input_kind(predicted_graph_report)
    sequence_path = _observation_sequence_path(
        predicted_graph_report,
        observation_sequence_path,
    )
    observations, load_error = _load_observations(input_kind, sequence_path)
    evidence_summary = _evidence_summary(
        input_kind=input_kind,
        predicted_graph_report=predicted_graph_report,
        observations=observations,
    )
    checks = _checks(
        predicted_graph_report,
        input_kind=input_kind,
        observations=observations,
        load_error=load_error,
        evidence_summary=evidence_summary,
        min_observation_count=min_observation_count,
        min_object_observation_count=min_object_observation_count,
        required_evidence_kinds=required_kinds,
    )
    report: dict[str, Any] = {
        "schema_version": PREDICTED_DSG_EVIDENCE_REPORT_SCHEMA_VERSION,
        "predicted_graph_report_path": (
            str(predicted_graph_report_path)
            if predicted_graph_report_path is not None
            else None
        ),
        "predicted_graph_report_digest": predicted_graph_report_digest(
            predicted_graph_report
        ),
        "observation_sequence_path": (
            str(sequence_path) if sequence_path is not None else None
        ),
        "observation_sequence_digest": evidence_summary.get(
            "observation_sequence_digest"
        ),
        "thresholds": {
            "min_object_observation_count": min_object_observation_count,
            "min_observation_count": min_observation_count,
        },
        "required_evidence_kinds": list(required_kinds),
        "evidence_summary": evidence_summary,
        "checks": checks,
        "readiness": _readiness_from_checks(checks),
    }
    report["report_digest"] = predicted_dsg_evidence_report_digest(report)
    return report


def predicted_dsg_evidence_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def predicted_dsg_evidence_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_predicted_dsg_evidence_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(predicted_dsg_evidence_report_json(report), encoding="utf-8")
    return output_path


def load_predicted_dsg_evidence_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Predicted DSG evidence report JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_predicted_dsg_evidence_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = report.get("schema_version")
    report_digest = _string_or_none(report.get("report_digest"))
    expected_digest = predicted_dsg_evidence_report_digest(report)
    checks = _mapping_sequence(report.get("checks"))
    expected_readiness = _readiness_from_checks(checks)
    checks_out = [
        {
            "name": "schema_version",
            "passed": schema_version == PREDICTED_DSG_EVIDENCE_REPORT_SCHEMA_VERSION,
            "expected": PREDICTED_DSG_EVIDENCE_REPORT_SCHEMA_VERSION,
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
            "name": "required_evidence_kinds_shape",
            "passed": _string_sequence(report.get("required_evidence_kinds")),
        },
        {
            "name": "evidence_summary_shape",
            "passed": _evidence_summary_shape_valid(report.get("evidence_summary")),
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks_out),
        "schema_version": schema_version,
        "report_digest": report_digest,
        "checks": checks_out,
    }


def compare_predicted_dsg_evidence_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    predicted_report_path = _required_report_path(
        report,
        "predicted_graph_report_path",
    )
    thresholds = _mapping(report.get("thresholds"), "thresholds")
    current_report = predicted_dsg_evidence_report(
        load_predicted_graph_report(predicted_report_path),
        predicted_graph_report_path=predicted_report_path,
        observation_sequence_path=_optional_path(report.get("observation_sequence_path")),
        min_observation_count=_required_int(thresholds, "min_observation_count"),
        min_object_observation_count=_required_int(
            thresholds,
            "min_object_observation_count",
        ),
        required_evidence_kinds=_string_tuple(report, "required_evidence_kinds"),
    )
    validation = validate_predicted_dsg_evidence_report(report)
    saved_digest = _string_or_none(report.get("report_digest"))
    current_digest = _string_or_none(current_report.get("report_digest"))
    checks = [
        {"name": "saved_report_valid", "passed": validation["valid"] is True},
        _equality_check(
            "predicted_graph_report_digest_matches_current",
            report.get("predicted_graph_report_digest"),
            current_report["predicted_graph_report_digest"],
        ),
        _equality_check(
            "observation_sequence_digest_matches_current",
            report.get("observation_sequence_digest"),
            current_report["observation_sequence_digest"],
        ),
        _equality_check(
            "evidence_summary_matches_current",
            report.get("evidence_summary"),
            current_report["evidence_summary"],
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


def _input_kind(predicted_graph_report: Mapping[str, Any]) -> str:
    value = predicted_graph_report.get("input_kind", "episode")
    return value if isinstance(value, str) and value != "" else "episode"


def _observation_sequence_path(
    predicted_graph_report: Mapping[str, Any],
    observation_sequence_path: str | Path | None,
) -> Path | None:
    if observation_sequence_path is not None:
        return Path(observation_sequence_path)
    path = predicted_graph_report.get("path")
    if isinstance(path, str) and path != "":
        return Path(path)
    return None


def _load_observations(
    input_kind: str,
    path: Path | None,
) -> tuple[tuple[SceneObservation, ...], str | None]:
    if input_kind != "observation_sequence":
        return (), None
    if path is None:
        return (), "observation sequence path is missing"
    try:
        return load_scene_observation_sequence(path), None
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        return (), str(exc)


def _evidence_summary(
    *,
    input_kind: str,
    predicted_graph_report: Mapping[str, Any],
    observations: Sequence[SceneObservation],
) -> dict[str, Any]:
    objects = [obj for observation in observations for obj in observation.objects]
    evidence_kind_counts = _evidence_kind_counts(objects)
    state_evidence_objects = [obj for obj in objects if _has_state_evidence(obj)]
    invalid_state_evidence_object_ids = _invalid_state_evidence_object_ids(
        state_evidence_objects
    )
    detector_names = sorted(
        {
            str(obj.attributes["detector"])
            for obj in objects
            if isinstance(obj.attributes, Mapping)
            and isinstance(obj.attributes.get("detector"), str)
            and obj.attributes.get("detector") != ""
        }
    )
    return {
        "detector_names": detector_names,
        "evidence_kind_counts": evidence_kind_counts,
        "hidden_object_observation_count": sum(1 for obj in objects if not obj.visible),
        "input_kind": input_kind,
        "invalid_state_evidence_object_ids": invalid_state_evidence_object_ids,
        "object_observation_count": len(objects),
        "observation_count": len(observations),
        "observation_sequence_digest": (
            scene_observation_sequence_digest(observations)
            if observations
            else _string_or_none(predicted_graph_report.get("observation_sequence_digest"))
        ),
        "source_counts": _source_counts(objects),
        "state_evidence_object_observation_count": len(state_evidence_objects),
        "visible_object_observation_count": sum(1 for obj in objects if obj.visible),
    }


def _checks(
    predicted_graph_report: Mapping[str, Any],
    *,
    input_kind: str,
    observations: Sequence[SceneObservation],
    load_error: str | None,
    evidence_summary: Mapping[str, Any],
    min_observation_count: int,
    min_object_observation_count: int,
    required_evidence_kinds: Sequence[str],
) -> list[dict[str, Any]]:
    predicted_validation = validate_predicted_graph_report(predicted_graph_report)
    evidence_kind_counts = _int_mapping(evidence_summary.get("evidence_kind_counts"))
    missing_evidence_kinds = [
        kind for kind in required_evidence_kinds if evidence_kind_counts.get(kind, 0) <= 0
    ]
    source_counts = _mapping(evidence_summary.get("source_counts"), "source_counts")
    mock_sources = sorted(
        source
        for source in source_counts
        if "mock" in source.lower() or source == "observation_sequence"
    )
    non_real_sources = _non_real_sources(source_counts)
    invalid_state_evidence_object_ids = _string_tuple_from_value(
        evidence_summary.get("invalid_state_evidence_object_ids")
    )
    load_check: dict[str, Any] = {
        "name": "observation_sequence_loads",
        "passed": input_kind == "observation_sequence" and load_error is None,
    }
    if load_error is not None:
        load_check["error"] = load_error
    return [
        {
            "name": "predicted_graph_report_valid",
            "passed": predicted_validation["valid"] is True,
        },
        {
            "name": "input_kind_observation_sequence",
            "passed": input_kind == "observation_sequence",
            "expected": "observation_sequence",
            "actual": input_kind,
        },
        load_check,
        {
            "name": "observation_sequence_digest_matches_report",
            "passed": (
                bool(observations)
                and predicted_graph_report.get("observation_sequence_digest")
                == evidence_summary.get("observation_sequence_digest")
            ),
            "expected": predicted_graph_report.get("observation_sequence_digest"),
            "actual": evidence_summary.get("observation_sequence_digest"),
        },
        {
            "name": "observation_count_minimum",
            "passed": _int_value(evidence_summary, "observation_count")
            >= min_observation_count,
            "minimum": min_observation_count,
            "actual": _int_value(evidence_summary, "observation_count"),
        },
        {
            "name": "object_observation_count_minimum",
            "passed": _int_value(evidence_summary, "object_observation_count")
            >= min_object_observation_count,
            "minimum": min_object_observation_count,
            "actual": _int_value(evidence_summary, "object_observation_count"),
        },
        {
            "name": "required_evidence_kinds_present",
            "passed": len(missing_evidence_kinds) == 0,
            "required": list(required_evidence_kinds),
            "missing": missing_evidence_kinds,
            "actual": evidence_kind_counts,
        },
        {
            "name": "mock_sources_absent",
            "passed": len(mock_sources) == 0,
            "actual": mock_sources,
        },
        {
            "name": "non_real_sources_absent",
            "passed": len(non_real_sources) == 0,
            "actual": non_real_sources,
        },
        {
            "name": "detector_state_evidence_valid",
            "passed": len(invalid_state_evidence_object_ids) == 0,
            "invalid_object_ids": list(invalid_state_evidence_object_ids),
            "state_evidence_object_observation_count": _int_value(
                evidence_summary,
                "state_evidence_object_observation_count",
            ),
        },
    ]


def _evidence_kind_counts(objects: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {"depth": 0, "detector": 0, "rgb": 0}
    for obj in objects:
        attributes = _attributes(obj)
        source_text = _source(obj)
        if _has_string(attributes, ("depth_path", "depth_file", "depth_image")):
            counts["depth"] += 1
        if _has_string(attributes, ("detector", "detector_id", "detector_name")) or (
            "detector" in source_text
        ):
            counts["detector"] += 1
        if _has_string(attributes, ("rgb_path", "rgb_file", "rgb_image")):
            counts["rgb"] += 1
    return {key: counts[key] for key in sorted(counts)}


def _invalid_state_evidence_object_ids(objects: Sequence[Any]) -> list[str]:
    invalid: list[str] = []
    for obj in objects:
        if not _state_evidence_valid(obj):
            object_id = getattr(obj, "object_id", None)
            invalid.append(object_id if isinstance(object_id, str) else "unknown")
    return sorted(set(invalid))


def _has_state_evidence(obj: Any) -> bool:
    states = _attributes(obj).get("states")
    return isinstance(states, Mapping) and bool(states)


def _state_evidence_valid(obj: Any) -> bool:
    attributes = _attributes(obj)
    if getattr(obj, "visible", None) is not True:
        return False
    source_kind = attributes.get("source_kind")
    detector_name = attributes.get("detector")
    source_text = _source(obj).lower()
    if (
        source_kind != "detector"
        and not (isinstance(detector_name, str) and detector_name)
        and "detector" not in source_text
    ):
        return False
    evidence_kinds = set(_string_tuple_from_value(attributes.get("evidence_kinds")))
    return {"depth", "detector", "rgb"}.issubset(evidence_kinds)


def _source_counts(objects: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for obj in objects:
        source = _source(obj) or "unspecified"
        counts[source] = counts.get(source, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def _non_real_sources(source_counts: Mapping[str, Any]) -> list[str]:
    return sorted(
        source
        for source in source_counts
        if _contains_non_real_source_marker(source) or source == "observation_sequence"
    )


def _contains_non_real_source_marker(value: str) -> bool:
    normalized = value.lower()
    return any(marker in normalized for marker in NON_REAL_PREDICTED_SOURCE_MARKERS)


def _source(obj: Any) -> str:
    attributes = _attributes(obj)
    for key in ("source", "source_name", "source_kind"):
        value = attributes.get(key)
        if isinstance(value, str) and value != "":
            return value
    return "unspecified"


def _attributes(obj: Any) -> Mapping[str, Any]:
    value = getattr(obj, "attributes", {})
    return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else {}


def _has_string(attributes: Mapping[str, Any], keys: Sequence[str]) -> bool:
    return any(isinstance(attributes.get(key), str) and attributes.get(key) for key in keys)


def _readiness_from_checks(checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed = [
        _required_str(check, "name")
        for check in checks
        if check.get("passed") is not True
    ]
    return {
        "ready": len(failed) == 0,
        "failed_check_count": len(failed),
        "failed_checks": failed,
    }


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


def _required_report_path(report: Mapping[str, Any], key: str) -> Path:
    value = report.get(key)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Predicted DSG evidence report missing path: {key}")
    return Path(value)


def _optional_path(value: object) -> Path | None:
    return Path(value) if isinstance(value, str) and value != "" else None


def _thresholds_shape_valid(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return _int_value(value, "min_observation_count") > 0 and _int_value(
        value,
        "min_object_observation_count",
    ) > 0


def _evidence_summary_shape_valid(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return all(
        key in value
        for key in (
            "evidence_kind_counts",
            "input_kind",
            "object_observation_count",
            "observation_count",
            "source_counts",
        )
    )


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpatialQAError(f"Predicted DSG evidence field must be an object: {field_name}")
    return cast(Mapping[str, Any], value)


def _mapping_sequence(value: object) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return ()
    return tuple(cast(Mapping[str, Any], item) for item in value if isinstance(item, Mapping))


def _string_tuple(report: Mapping[str, Any], field_name: str) -> tuple[str, ...]:
    values = _string_tuple_from_value(report.get(field_name))
    if not values:
        raise SpatialQAError(f"Predicted DSG evidence field must contain strings: {field_name}")
    return values


def _string_tuple_from_value(value: object) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return ()
    strings: list[str] = []
    for item in value:
        if isinstance(item, str) and item != "":
            strings.append(item)
    return tuple(strings)


def _string_sequence(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str)
        and all(isinstance(item, str) for item in value)
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
        raise SpatialQAError(f"Predicted DSG evidence field must be an integer: {field_name}")
    return value


def _int_value(payload: Mapping[str, Any], field_name: str) -> int:
    value = payload.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _required_str(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Predicted DSG evidence field must be a string: {field_name}")
    return value


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value != "" else None


def _equality_check(name: str, expected: object, actual: object) -> dict[str, Any]:
    return {
        "name": name,
        "passed": expected == actual,
        "expected": expected,
        "actual": actual,
    }
