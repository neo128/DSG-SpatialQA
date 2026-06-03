from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.benchmark.manifest import (
    _load_experiment_artifact,
    benchmark_manifest_digest,
    load_benchmark_manifest,
)
from dsg_spatialqa_lab.eval.offline_predictions import (
    compare_offline_prediction_import_report,
    load_offline_prediction_import_report,
    validate_offline_prediction_import_report,
)
from dsg_spatialqa_lab.eval.offline_control_matrix import (
    compare_offline_control_matrix_report,
    load_offline_control_matrix_report,
    validate_offline_control_matrix_report,
)
from dsg_spatialqa_lab.eval.offline_control_result import (
    compare_offline_control_result_report,
    load_offline_control_result_report,
    validate_offline_control_result_report,
)
from dsg_spatialqa_lab.eval.error_attribution import (
    compare_error_attribution_report,
    load_error_attribution_report,
    validate_error_attribution_report,
)
from dsg_spatialqa_lab.eval.graph_metrics import (
    compare_graph_eval_report,
    load_graph_eval_report,
    validate_graph_eval_report,
)
from dsg_spatialqa_lab.eval.task_metrics import (
    compare_active_task_delta_report,
    load_active_task_delta_report,
    validate_active_task_delta_report,
)
from dsg_spatialqa_lab.eval.qa_metrics import (
    compare_qa_eval_delta_report,
    load_qa_eval_delta_report,
    validate_qa_eval_delta_report,
)
from dsg_spatialqa_lab.benchmark.real_collection import (
    compare_real_collection_report,
    load_real_collection_report,
    validate_real_collection_report,
)
from dsg_spatialqa_lab.predicted_evidence import (
    compare_predicted_dsg_evidence_report,
    load_predicted_dsg_evidence_report,
    validate_predicted_dsg_evidence_report,
)
from dsg_spatialqa_lab.predicted import (
    compare_predicted_graph_report,
    load_predicted_graph_report,
    validate_predicted_graph_report,
)
from dsg_spatialqa_lab.schema import SpatialQAError
from dsg_spatialqa_lab.visualization.dashboard_export import (
    compare_dashboard_bundle,
    load_dashboard_bundle,
    validate_dashboard_bundle,
)


REAL_EXPERIMENT_READINESS_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.real-experiment-readiness.v1"
)
DEFAULT_REQUIRED_CONTROL_KINDS = (
    "caption_memory",
    "graph_text",
    "multi_frame_vlm",
    "vlm",
)
DEFAULT_REQUIRED_PREDICTED_INPUT_KINDS = ("observation_sequence",)
SPATIAL_QA_QUESTION_TYPES = frozenset(
    {
        "nearest_object",
        "object_location",
        "object_room",
        "relative_relation",
    }
)
DYNAMIC_MEMORY_QUESTION_TYPES = frozenset(
    {
        "agent_history",
        "agent_timeline",
        "object_history",
        "object_timeline",
        "recent_events",
        "relation_timeline",
        "reobserve_targets",
        "scene_delta",
    }
)
GRAPH_TOOL_QUERY_QUESTION_TYPES = frozenset(
    {
        "graph_query",
        "nearest_object",
        "relation_timeline",
        "relative_relation",
        "retrieve_subgraph",
    }
)
OFFLINE_CONTROL_REAL_METADATA_FIELDS = ("model_id", "prompt_id", "dataset_id")
OFFLINE_CONTROL_PLACEHOLDER_MARKERS = (
    "fixture",
    "mock",
    "placeholder",
    "synthetic",
    "unspecified",
)
QA_DELTA_PLACEHOLDER_NAME_MARKERS = (
    *OFFLINE_CONTROL_PLACEHOLDER_MARKERS,
    "baseline",
    "candidate",
)
ACTIVE_DELTA_PLACEHOLDER_NAME_MARKERS = QA_DELTA_PLACEHOLDER_NAME_MARKERS


def real_experiment_readiness_report(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path | None = None,
    declared_data_source_kind: str = "unspecified",
    min_episode_count: int = 3,
    min_scene_count: int = 1,
    min_qa_count: int = 30,
    required_control_kinds: Sequence[str] = DEFAULT_REQUIRED_CONTROL_KINDS,
    required_predicted_input_kinds: Sequence[str] = DEFAULT_REQUIRED_PREDICTED_INPUT_KINDS,
) -> dict[str, Any]:
    _validate_threshold(min_episode_count, "min_episode_count")
    _validate_threshold(min_scene_count, "min_scene_count")
    _validate_threshold(min_qa_count, "min_qa_count")
    control_kinds = _unique_strings(required_control_kinds, "required_control_kinds")
    predicted_input_kinds = _unique_strings(
        required_predicted_input_kinds,
        "required_predicted_input_kinds",
    )
    artifact_summary = _artifact_summary(manifest)
    checks = _readiness_checks(
        manifest,
        declared_data_source_kind=declared_data_source_kind,
        min_episode_count=min_episode_count,
        min_scene_count=min_scene_count,
        min_qa_count=min_qa_count,
        required_control_kinds=control_kinds,
        required_predicted_input_kinds=predicted_input_kinds,
        artifact_summary=artifact_summary,
    )
    report: dict[str, Any] = {
        "schema_version": REAL_EXPERIMENT_READINESS_REPORT_SCHEMA_VERSION,
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "manifest_digest": benchmark_manifest_digest(manifest),
        "declared_data_source_kind": declared_data_source_kind,
        "thresholds": {
            "min_episode_count": min_episode_count,
            "min_qa_count": min_qa_count,
            "min_scene_count": min_scene_count,
        },
        "required_control_kinds": list(control_kinds),
        "required_predicted_input_kinds": list(predicted_input_kinds),
        "artifact_summary": artifact_summary,
        "checks": checks,
        "readiness": _readiness_from_checks(checks),
    }
    report["report_digest"] = real_experiment_readiness_report_digest(report)
    return report


def real_experiment_readiness_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def real_experiment_readiness_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_real_experiment_readiness_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(real_experiment_readiness_report_json(report), encoding="utf-8")
    return output_path


def load_real_experiment_readiness_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Real experiment readiness report JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_real_experiment_readiness_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = report.get("schema_version")
    report_digest = _string_or_none(report.get("report_digest"))
    expected_digest = real_experiment_readiness_report_digest(report)
    checks = _mapping_sequence(report.get("checks"))
    expected_readiness = _readiness_from_checks(checks)
    readiness = report.get("readiness")
    checks_out = [
        {
            "name": "schema_version",
            "passed": schema_version == REAL_EXPERIMENT_READINESS_REPORT_SCHEMA_VERSION,
            "expected": REAL_EXPERIMENT_READINESS_REPORT_SCHEMA_VERSION,
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
            "passed": readiness == expected_readiness,
            "expected": expected_readiness,
            "actual": readiness,
        },
        {
            "name": "thresholds_shape",
            "passed": _thresholds_shape_valid(report.get("thresholds")),
        },
        {
            "name": "required_control_kinds_shape",
            "passed": _string_sequence(report.get("required_control_kinds")),
        },
        {
            "name": "required_predicted_input_kinds_shape",
            "passed": _string_sequence(report.get("required_predicted_input_kinds")),
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks_out),
        "schema_version": schema_version,
        "report_digest": report_digest,
        "checks": checks_out,
    }


def compare_real_experiment_readiness_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = _required_str(report, "manifest_path")
    thresholds = _as_mapping(report.get("thresholds"), "thresholds")
    current_report = real_experiment_readiness_report(
        load_benchmark_manifest(manifest_path),
        manifest_path=manifest_path,
        declared_data_source_kind=_required_str(report, "declared_data_source_kind"),
        min_episode_count=_required_int(thresholds, "min_episode_count"),
        min_scene_count=_required_int(thresholds, "min_scene_count"),
        min_qa_count=_required_int(thresholds, "min_qa_count"),
        required_control_kinds=_string_tuple(report, "required_control_kinds"),
        required_predicted_input_kinds=_string_tuple(
            report,
            "required_predicted_input_kinds",
        ),
    )
    validation = validate_real_experiment_readiness_report(report)
    saved_digest = _string_or_none(report.get("report_digest"))
    current_digest = _string_or_none(current_report.get("report_digest"))
    checks = [
        {"name": "saved_report_valid", "passed": validation["valid"] is True},
        _equality_check(
            "manifest_digest_matches_current",
            report.get("manifest_digest"),
            current_report["manifest_digest"],
        ),
        _equality_check(
            "artifact_summary_matches_current",
            report.get("artifact_summary"),
            current_report["artifact_summary"],
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
        "manifest_path": manifest_path,
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def _readiness_checks(
    manifest: Mapping[str, Any],
    *,
    declared_data_source_kind: str,
    min_episode_count: int,
    min_scene_count: int,
    min_qa_count: int,
    required_control_kinds: Sequence[str],
    required_predicted_input_kinds: Sequence[str],
    artifact_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    coverage = _as_mapping(manifest.get("coverage", {}), "coverage")
    question_type_counts = _int_mapping(coverage.get("by_question_type"))
    dynamic_static = _int_mapping(coverage.get("dynamic_static"))
    artifact_counts = _int_mapping(artifact_summary.get("artifact_counts"))
    offline_control_matrix_ready = artifact_summary.get("offline_control_matrix_ready")
    offline_control_matrix_required_kinds = _string_tuple_from_value(
        artifact_summary.get("offline_control_matrix_required_kinds")
    )
    offline_control_result_source_kinds = _string_tuple_from_value(
        artifact_summary.get("offline_control_result_source_kinds")
    )
    offline_control_result_not_ready_paths = _string_tuple_from_value(
        artifact_summary.get("offline_control_result_not_ready_paths")
    )
    offline_control_kinds = _string_tuple_from_value(
        artifact_summary.get("offline_control_kinds")
    )
    offline_control_qa_digests = _string_tuple_from_value(
        artifact_summary.get("offline_control_qa_digests")
    )
    offline_control_incomplete_source_keys = _string_tuple_from_value(
        artifact_summary.get("offline_control_incomplete_source_keys")
    )
    offline_control_diagnostic_source_keys = _string_tuple_from_value(
        artifact_summary.get("offline_control_diagnostic_source_keys")
    )
    offline_control_invalid_source_keys = _string_tuple_from_value(
        artifact_summary.get("offline_control_invalid_source_keys")
    )
    offline_control_missing_metadata_source_keys = _string_tuple_from_value(
        artifact_summary.get("offline_control_missing_metadata_source_keys")
    )
    offline_control_placeholder_source_keys = _string_tuple_from_value(
        artifact_summary.get("offline_control_placeholder_source_keys")
    )
    manifest_qa_digest = _string_or_none(artifact_summary.get("manifest_qa_digest"))
    qa_delta_baseline_names = _string_tuple_from_value(
        artifact_summary.get("qa_delta_baseline_names")
    )
    qa_delta_case_mismatch_paths = _string_tuple_from_value(
        artifact_summary.get("qa_delta_case_mismatch_paths")
    )
    qa_delta_not_ready_paths = _string_tuple_from_value(
        artifact_summary.get("qa_delta_not_ready_paths")
    )
    qa_delta_placeholder_name_paths = _string_tuple_from_value(
        artifact_summary.get("qa_delta_placeholder_name_paths")
    )
    active_delta_task_mismatch_paths = _string_tuple_from_value(
        artifact_summary.get("active_delta_task_mismatch_paths")
    )
    active_delta_not_ready_paths = _string_tuple_from_value(
        artifact_summary.get("active_delta_not_ready_paths")
    )
    active_delta_placeholder_name_paths = _string_tuple_from_value(
        artifact_summary.get("active_delta_placeholder_name_paths")
    )
    graph_eval_predicted_digests = _string_tuple_from_value(
        artifact_summary.get("graph_eval_predicted_digests")
    )
    graph_eval_not_ready_paths = _string_tuple_from_value(
        artifact_summary.get("graph_eval_not_ready_paths")
    )
    error_attribution_predicted_graph_digests = _string_tuple_from_value(
        artifact_summary.get("error_attribution_predicted_graph_digests")
    )
    error_attribution_not_ready_paths = _string_tuple_from_value(
        artifact_summary.get("error_attribution_not_ready_paths")
    )
    predicted_graph_graph_digests = _string_tuple_from_value(
        artifact_summary.get("predicted_graph_graph_digests")
    )
    predicted_graph_not_ready_paths = _string_tuple_from_value(
        artifact_summary.get("predicted_graph_not_ready_paths")
    )
    predicted_graph_report_digests = _string_tuple_from_value(
        artifact_summary.get("predicted_graph_report_digests")
    )
    predicted_dsg_evidence_predicted_report_digests = _string_tuple_from_value(
        artifact_summary.get("predicted_dsg_evidence_predicted_report_digests")
    )
    dashboard_bundle_not_ready_paths = _string_tuple_from_value(
        artifact_summary.get("dashboard_bundle_not_ready_paths")
    )
    manifest_artifact_digest_invalid_paths = _string_tuple_from_value(
        artifact_summary.get("benchmark_manifest_artifact_digest_invalid_paths")
    )
    manifest_artifact_digest_mismatch_paths = _string_tuple_from_value(
        artifact_summary.get("benchmark_manifest_artifact_digest_mismatch_paths")
    )
    predicted_input_kinds = _string_tuple_from_value(
        artifact_summary.get("predicted_input_kinds")
    )
    checks = [
        {
            "name": "benchmark_manifest_artifact_digests_current",
            "group": "manifest",
            "passed": artifact_summary.get(
                "benchmark_manifest_artifact_digests_current"
            )
            is True,
            "expected": [],
            "actual": sorted(
                (
                    *manifest_artifact_digest_invalid_paths,
                    *manifest_artifact_digest_mismatch_paths,
                )
            ),
            "invalid_paths": list(manifest_artifact_digest_invalid_paths),
            "mismatch_paths": list(manifest_artifact_digest_mismatch_paths),
        },
        _minimum_check(
            "episode_count_minimum",
            "real_data",
            _manifest_int(manifest, "episode_count"),
            min_episode_count,
        ),
        _minimum_check(
            "scene_count_minimum",
            "real_data",
            _manifest_int(manifest, "scene_count"),
            min_scene_count,
        ),
        _minimum_check(
            "qa_count_minimum",
            "real_data",
            _manifest_int(manifest, "qa_count"),
            min_qa_count,
        ),
        {
            "name": "data_source_kind_real",
            "group": "real_data",
            "passed": declared_data_source_kind == "real",
            "expected": "real",
            "actual": declared_data_source_kind,
        },
        _artifact_present_check(
            "real_collection_report_present",
            "real_data",
            artifact_counts,
            "real_collection_report",
        ),
        {
            "name": "real_collection_ready",
            "group": "real_data",
            "passed": artifact_summary.get("real_collection_ready") is True,
            "expected": True,
            "actual": artifact_summary.get("real_collection_ready"),
            "not_ready_paths": artifact_summary.get(
                "real_collection_not_ready_paths",
                [],
            ),
        },
        _coverage_check(
            "spatial_qa_coverage",
            "real_data",
            question_type_counts,
            SPATIAL_QA_QUESTION_TYPES,
        ),
        {
            "name": "dynamic_memory_coverage",
            "group": "real_data",
            "passed": dynamic_static.get("dynamic", 0) > 0,
            "expected": "at least one dynamic QA case",
            "actual": dynamic_static.get("dynamic", 0),
        },
        _coverage_check(
            "graph_tool_query_coverage",
            "real_data",
            question_type_counts,
            GRAPH_TOOL_QUERY_QUESTION_TYPES,
        ),
        _artifact_present_check(
            "qa_delta_report_present",
            "real_controls",
            artifact_counts,
            "qa_eval_delta_report",
        ),
        {
            "name": "qa_delta_reports_ready",
            "group": "real_controls",
            "passed": artifact_summary.get("qa_delta_ready") is True,
            "expected": True,
            "actual": artifact_summary.get("qa_delta_ready"),
            "not_ready_paths": list(qa_delta_not_ready_paths),
        },
        _required_values_check(
            "qa_delta_baselines_cover_controls",
            "real_controls",
            qa_delta_baseline_names,
            required_control_kinds,
        ),
        {
            "name": "qa_delta_case_counts_match",
            "group": "real_controls",
            "passed": len(qa_delta_case_mismatch_paths) == 0,
            "expected": [],
            "actual": list(qa_delta_case_mismatch_paths),
        },
        {
            "name": "qa_delta_non_placeholder_names",
            "group": "real_controls",
            "passed": len(qa_delta_placeholder_name_paths) == 0,
            "expected": [],
            "actual": list(qa_delta_placeholder_name_paths),
        },
        _artifact_present_check(
            "offline_control_matrix_report_present",
            "real_controls",
            artifact_counts,
            "offline_control_matrix_report",
        ),
        {
            "name": "offline_control_matrix_ready",
            "group": "real_controls",
            "passed": offline_control_matrix_ready is True,
            "expected": True,
            "actual": offline_control_matrix_ready,
            "not_ready_paths": artifact_summary.get(
                "offline_control_matrix_not_ready_paths",
                [],
            ),
        },
        _required_values_check(
            "offline_control_matrix_required_kinds_cover_controls",
            "real_controls",
            offline_control_matrix_required_kinds,
            required_control_kinds,
        ),
        _artifact_present_check(
            "offline_control_result_report_present",
            "real_controls",
            artifact_counts,
            "offline_control_result_report",
        ),
        {
            "name": "offline_control_result_ready",
            "group": "real_controls",
            "passed": artifact_summary.get("offline_control_result_ready") is True,
            "expected": True,
            "actual": artifact_summary.get("offline_control_result_ready"),
            "not_ready_paths": list(offline_control_result_not_ready_paths),
        },
        _required_values_check(
            "offline_control_result_deltas_cover_controls",
            "real_controls",
            offline_control_result_source_kinds,
            required_control_kinds,
        ),
        _required_values_check(
            "offline_controls_present",
            "real_controls",
            offline_control_kinds,
            required_control_kinds,
        ),
        {
            "name": "offline_control_import_reports_valid",
            "group": "real_controls",
            "passed": len(offline_control_invalid_source_keys) == 0,
            "expected": [],
            "actual": list(offline_control_invalid_source_keys),
        },
        {
            "name": "offline_control_complete_prediction_coverage",
            "group": "real_controls",
            "passed": len(offline_control_incomplete_source_keys) == 0,
            "expected": [],
            "actual": list(offline_control_incomplete_source_keys),
        },
        {
            "name": "offline_control_clean_import_diagnostics",
            "group": "real_controls",
            "passed": len(offline_control_diagnostic_source_keys) == 0,
            "expected": [],
            "actual": list(offline_control_diagnostic_source_keys),
        },
        {
            "name": "offline_control_metadata_present",
            "group": "real_controls",
            "passed": len(offline_control_missing_metadata_source_keys) == 0,
            "expected": [],
            "actual": list(offline_control_missing_metadata_source_keys),
        },
        {
            "name": "offline_control_no_placeholder_sources",
            "group": "real_controls",
            "passed": len(offline_control_placeholder_source_keys) == 0,
            "expected": [],
            "actual": list(offline_control_placeholder_source_keys),
        },
        {
            "name": "offline_control_qa_digest_matches_manifest",
            "group": "real_controls",
            "passed": manifest_qa_digest is not None
            and list(offline_control_qa_digests) == [manifest_qa_digest],
            "expected": manifest_qa_digest,
            "actual": list(offline_control_qa_digests),
        },
        _required_values_check(
            "predicted_observation_graph_present",
            "real_predicted_dsg",
            predicted_input_kinds,
            required_predicted_input_kinds,
        ),
        _artifact_present_check(
            "predicted_graph_report_present",
            "real_predicted_dsg",
            artifact_counts,
            "predicted_graph_report",
        ),
        {
            "name": "predicted_graph_reports_ready",
            "group": "real_predicted_dsg",
            "passed": artifact_summary.get("predicted_graph_ready") is True,
            "expected": True,
            "actual": artifact_summary.get("predicted_graph_ready"),
            "not_ready_paths": list(predicted_graph_not_ready_paths),
        },
        _artifact_present_check(
            "predicted_dsg_evidence_report_present",
            "real_predicted_dsg",
            artifact_counts,
            "predicted_dsg_evidence_report",
        ),
        {
            "name": "predicted_dsg_evidence_ready",
            "group": "real_predicted_dsg",
            "passed": artifact_summary.get("predicted_dsg_evidence_ready") is True,
            "expected": True,
            "actual": artifact_summary.get("predicted_dsg_evidence_ready"),
            "not_ready_paths": artifact_summary.get(
                "predicted_dsg_evidence_not_ready_paths",
                [],
            ),
        },
        _equality_check(
            "predicted_dsg_evidence_report_digest_alignment",
            list(predicted_graph_report_digests),
            list(predicted_dsg_evidence_predicted_report_digests),
        )
        | {"group": "real_predicted_dsg"},
        _artifact_present_check(
            "graph_eval_report_present",
            "real_predicted_dsg",
            artifact_counts,
            "graph_eval_report",
        ),
        {
            "name": "graph_eval_reports_ready",
            "group": "real_predicted_dsg",
            "passed": artifact_summary.get("graph_eval_ready") is True,
            "expected": True,
            "actual": artifact_summary.get("graph_eval_ready"),
            "not_ready_paths": list(graph_eval_not_ready_paths),
        },
        _artifact_present_check(
            "error_attribution_report_present",
            "real_predicted_dsg",
            artifact_counts,
            "error_attribution_report",
        ),
        {
            "name": "error_attribution_reports_ready",
            "group": "real_predicted_dsg",
            "passed": artifact_summary.get("error_attribution_ready") is True,
            "expected": True,
            "actual": artifact_summary.get("error_attribution_ready"),
            "not_ready_paths": list(error_attribution_not_ready_paths),
        },
        _equality_check(
            "graph_error_predicted_digest_alignment",
            list(graph_eval_predicted_digests),
            list(error_attribution_predicted_graph_digests),
        )
        | {"group": "real_predicted_dsg"},
        _equality_check(
            "predicted_graph_eval_digest_alignment",
            list(predicted_graph_graph_digests),
            list(graph_eval_predicted_digests),
        )
        | {"group": "real_predicted_dsg"},
        _equality_check(
            "predicted_graph_error_digest_alignment",
            list(predicted_graph_graph_digests),
            list(error_attribution_predicted_graph_digests),
        )
        | {"group": "real_predicted_dsg"},
        _artifact_present_check(
            "active_task_delta_report_present",
            "review_artifacts",
            artifact_counts,
            "active_task_delta_report",
        ),
        {
            "name": "active_delta_reports_ready",
            "group": "review_artifacts",
            "passed": artifact_summary.get("active_delta_ready") is True,
            "expected": True,
            "actual": artifact_summary.get("active_delta_ready"),
            "not_ready_paths": list(active_delta_not_ready_paths),
        },
        {
            "name": "active_delta_task_counts_match",
            "group": "review_artifacts",
            "passed": len(active_delta_task_mismatch_paths) == 0,
            "expected": [],
            "actual": list(active_delta_task_mismatch_paths),
        },
        {
            "name": "active_delta_non_placeholder_names",
            "group": "review_artifacts",
            "passed": len(active_delta_placeholder_name_paths) == 0,
            "expected": [],
            "actual": list(active_delta_placeholder_name_paths),
        },
        _artifact_present_check(
            "dashboard_bundle_present",
            "review_artifacts",
            artifact_counts,
            "dashboard_bundle",
        ),
        {
            "name": "dashboard_bundle_ready",
            "group": "review_artifacts",
            "passed": artifact_summary.get("dashboard_bundle_ready") is True,
            "expected": True,
            "actual": artifact_summary.get("dashboard_bundle_ready"),
            "not_ready_paths": list(dashboard_bundle_not_ready_paths),
        },
    ]
    return checks


def _artifact_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = _mapping_sequence(manifest.get("experiment_artifacts"))
    artifact_counts = _sorted_counts(
        _required_mapping_str(artifact, "artifact_type") for artifact in artifacts
    )
    manifest_artifact_digest_status = _manifest_artifact_digest_status(artifacts)
    offline_control_kinds = sorted(
        {
            kind
            for artifact in artifacts
            if artifact.get("artifact_type") == "offline_prediction_import_report"
            for kind in [_offline_control_kind(artifact)]
            if kind is not None
        }
    )
    offline_control_qa_digests = sorted(
        {
            digest
            for artifact in artifacts
            if artifact.get("artifact_type") == "offline_prediction_import_report"
            for digest in [_offline_control_qa_digest(artifact)]
            if digest is not None
        }
    )
    predicted_graph_status = _predicted_graph_status(artifacts)
    predicted_dsg_evidence_status = _predicted_dsg_evidence_status(artifacts)
    real_collection_status = _real_collection_status(artifacts)
    offline_control_matrix_status = _offline_control_matrix_status(artifacts)
    offline_control_result_status = _offline_control_result_status(artifacts)
    offline_control_import_status = _offline_control_import_status(artifacts)
    qa_delta_status = _qa_delta_status(artifacts)
    active_delta_status = _active_delta_status(artifacts)
    graph_eval_status = _graph_eval_status(artifacts)
    error_attribution_status = _error_attribution_status(artifacts)
    dashboard_status = _dashboard_bundle_status(artifacts)
    coverage = _as_mapping(manifest.get("coverage", {}), "coverage")
    return {
        "active_delta_baseline_names": active_delta_status["baseline_names"],
        "active_delta_candidate_names": active_delta_status["candidate_names"],
        "active_delta_not_ready_paths": active_delta_status["not_ready_paths"],
        "active_delta_placeholder_name_paths": active_delta_status[
            "placeholder_name_paths"
        ],
        "active_delta_ready": active_delta_status["ready"],
        "active_delta_stale_paths": active_delta_status["stale_paths"],
        "active_delta_task_mismatch_paths": active_delta_status[
            "task_mismatch_paths"
        ],
        "artifact_counts": artifact_counts,
        "benchmark_manifest_artifact_digest_invalid_paths": (
            manifest_artifact_digest_status["invalid_paths"]
        ),
        "benchmark_manifest_artifact_digest_mismatch_paths": (
            manifest_artifact_digest_status["mismatch_paths"]
        ),
        "benchmark_manifest_artifact_digests_current": (
            manifest_artifact_digest_status["current"]
        ),
        "dashboard_bundle_digests": dashboard_status["bundle_digests"],
        "dashboard_bundle_not_ready_paths": dashboard_status["not_ready_paths"],
        "dashboard_bundle_ready": dashboard_status["ready"],
        "dashboard_bundle_stale_paths": dashboard_status["stale_paths"],
        "dashboard_case_counts": dashboard_status["case_counts"],
        "error_attribution_gold_digests": error_attribution_status["gold_digests"],
        "error_attribution_not_ready_paths": error_attribution_status[
            "not_ready_paths"
        ],
        "error_attribution_oracle_graph_digests": error_attribution_status[
            "oracle_graph_digests"
        ],
        "error_attribution_predicted_graph_digests": error_attribution_status[
            "predicted_graph_digests"
        ],
        "error_attribution_prediction_digests": error_attribution_status[
            "prediction_digests"
        ],
        "error_attribution_ready": error_attribution_status["ready"],
        "error_attribution_stale_paths": error_attribution_status["stale_paths"],
        "graph_eval_not_ready_paths": graph_eval_status["not_ready_paths"],
        "graph_eval_oracle_digests": graph_eval_status["oracle_digests"],
        "graph_eval_predicted_digests": graph_eval_status["predicted_digests"],
        "graph_eval_ready": graph_eval_status["ready"],
        "graph_eval_stale_paths": graph_eval_status["stale_paths"],
        "manifest_qa_digest": _string_or_none(manifest.get("qa_digest")),
        "offline_control_matrix_not_ready_paths": offline_control_matrix_status[
            "not_ready_paths"
        ],
        "offline_control_matrix_ready": offline_control_matrix_status["ready"],
        "offline_control_matrix_required_kinds": offline_control_matrix_status[
            "required_source_kinds"
        ],
        "offline_control_matrix_stale_paths": offline_control_matrix_status[
            "stale_paths"
        ],
        "offline_control_result_candidate_names": offline_control_result_status[
            "candidate_names"
        ],
        "offline_control_result_not_ready_paths": offline_control_result_status[
            "not_ready_paths"
        ],
        "offline_control_result_ready": offline_control_result_status["ready"],
        "offline_control_result_source_kinds": offline_control_result_status[
            "source_kinds"
        ],
        "offline_control_result_stale_paths": offline_control_result_status[
            "stale_paths"
        ],
        "offline_control_diagnostic_source_keys": offline_control_import_status[
            "diagnostic_source_keys"
        ],
        "offline_control_incomplete_source_keys": offline_control_import_status[
            "incomplete_source_keys"
        ],
        "offline_control_invalid_source_keys": offline_control_import_status[
            "invalid_source_keys"
        ],
        "offline_control_kinds": offline_control_kinds,
        "offline_control_missing_metadata_source_keys": offline_control_import_status[
            "missing_metadata_source_keys"
        ],
        "offline_control_placeholder_source_keys": offline_control_import_status[
            "placeholder_source_keys"
        ],
        "offline_control_qa_digests": offline_control_qa_digests,
        "offline_control_stale_source_keys": offline_control_import_status[
            "stale_source_keys"
        ],
        "predicted_dsg_evidence_not_ready_paths": predicted_dsg_evidence_status[
            "not_ready_paths"
        ],
        "predicted_dsg_evidence_predicted_report_digests": (
            predicted_dsg_evidence_status["predicted_report_digests"]
        ),
        "predicted_dsg_evidence_ready": predicted_dsg_evidence_status["ready"],
        "predicted_graph_graph_digests": predicted_graph_status["graph_digests"],
        "predicted_graph_not_ready_paths": predicted_graph_status["not_ready_paths"],
        "predicted_graph_ready": predicted_graph_status["ready"],
        "predicted_graph_report_digests": predicted_graph_status["report_digests"],
        "predicted_graph_stale_paths": predicted_graph_status["stale_paths"],
        "predicted_input_kinds": predicted_graph_status["input_kinds"],
        "qa_delta_baseline_names": qa_delta_status["baseline_names"],
        "qa_delta_candidate_names": qa_delta_status["candidate_names"],
        "qa_delta_case_mismatch_paths": qa_delta_status["case_mismatch_paths"],
        "qa_delta_not_ready_paths": qa_delta_status["not_ready_paths"],
        "qa_delta_placeholder_name_paths": qa_delta_status["placeholder_name_paths"],
        "qa_delta_ready": qa_delta_status["ready"],
        "qa_delta_stale_paths": qa_delta_status["stale_paths"],
        "question_type_counts": _int_mapping(coverage.get("by_question_type")),
        "real_collection_not_ready_paths": real_collection_status["not_ready_paths"],
        "real_collection_ready": real_collection_status["ready"],
        "real_collection_source_kinds": real_collection_status["source_kinds"],
        "real_collection_stale_paths": real_collection_status["stale_paths"],
    }


def _manifest_artifact_digest_status(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    invalid_paths: set[str] = set()
    mismatch_paths: set[str] = set()
    for artifact in artifacts:
        artifact_type = artifact.get("artifact_type")
        path = artifact.get("path")
        saved_digest = _string_or_none(artifact.get("digest"))
        if not isinstance(artifact_type, str) or artifact_type == "":
            invalid_paths.add(path if isinstance(path, str) else "")
            continue
        if not isinstance(path, str) or path == "":
            invalid_paths.add("")
            continue
        try:
            _, current_digest = _load_experiment_artifact(artifact_type, Path(path))
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError):
            invalid_paths.add(path)
            continue
        if saved_digest != current_digest:
            mismatch_paths.add(path)
    return {
        "current": not invalid_paths and not mismatch_paths,
        "invalid_paths": sorted(invalid_paths),
        "mismatch_paths": sorted(mismatch_paths),
    }


def _offline_control_kind(artifact: Mapping[str, Any]) -> str | None:
    path = artifact.get("path")
    if not isinstance(path, str) or path == "":
        return None
    report = load_offline_prediction_import_report(path)
    source = report.get("source")
    if not isinstance(source, Mapping):
        return None
    kind = source.get("kind")
    return kind if isinstance(kind, str) and kind else None


def _offline_control_qa_digest(artifact: Mapping[str, Any]) -> str | None:
    path = artifact.get("path")
    if not isinstance(path, str) or path == "":
        return None
    report = load_offline_prediction_import_report(path)
    return _string_or_none(report.get("qa_digest"))


def _predicted_graph_status(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    graph_artifacts = [
        artifact
        for artifact in artifacts
        if artifact.get("artifact_type") == "predicted_graph_report"
    ]
    graph_digests: set[str] = set()
    input_kinds: set[str] = set()
    not_ready_paths: set[str] = set()
    report_digests: set[str] = set()
    stale_paths: set[str] = set()
    for artifact in graph_artifacts:
        path = artifact.get("path")
        if not isinstance(path, str) or path == "":
            not_ready_paths.add("")
            continue
        try:
            report = load_predicted_graph_report(path)
            validation = validate_predicted_graph_report(report)
            comparison_matches = (
                compare_predicted_graph_report(report).get("matches") is True
                if validation.get("valid") is True
                else False
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError):
            not_ready_paths.add(path)
            continue
        input_kind = report.get("input_kind", "episode")
        if isinstance(input_kind, str) and input_kind != "":
            input_kinds.add(input_kind)
        report_digest = _string_or_none(report.get("digest"))
        if report_digest is not None:
            report_digests.add(report_digest)
        graph_report = report.get("graph_report")
        graph_digest = (
            _string_or_none(graph_report.get("digest"))
            if isinstance(graph_report, Mapping)
            else None
        )
        if graph_digest is not None:
            graph_digests.add(graph_digest)
        if (
            validation.get("valid") is not True
            or comparison_matches is not True
            or graph_digest is None
            or report_digest is None
        ):
            not_ready_paths.add(path)
            if validation.get("valid") is True and comparison_matches is not True:
                stale_paths.add(path)
    return {
        "graph_digests": sorted(graph_digests),
        "input_kinds": sorted(input_kinds),
        "not_ready_paths": sorted(not_ready_paths),
        "ready": bool(graph_artifacts) and not not_ready_paths,
        "report_digests": sorted(report_digests),
        "stale_paths": sorted(stale_paths),
    }


def _graph_eval_status(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    graph_eval_artifacts = [
        artifact
        for artifact in artifacts
        if artifact.get("artifact_type") == "graph_eval_report"
    ]
    not_ready_paths: set[str] = set()
    oracle_digests: set[str] = set()
    predicted_digests: set[str] = set()
    stale_paths: set[str] = set()
    for artifact in graph_eval_artifacts:
        path = artifact.get("path")
        if not isinstance(path, str) or path == "":
            not_ready_paths.add("")
            continue
        try:
            report = load_graph_eval_report(path)
            validation = validate_graph_eval_report(report)
            comparison_matches = (
                compare_graph_eval_report(report).get("matches") is True
                if validation.get("valid") is True
                else False
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError):
            not_ready_paths.add(path)
            continue
        oracle_digest = _string_or_none(report.get("oracle_digest"))
        predicted_digest = _string_or_none(report.get("predicted_digest"))
        if oracle_digest is not None:
            oracle_digests.add(oracle_digest)
        if predicted_digest is not None:
            predicted_digests.add(predicted_digest)
        if (
            validation.get("valid") is not True
            or comparison_matches is not True
            or oracle_digest is None
            or predicted_digest is None
        ):
            not_ready_paths.add(path)
            if validation.get("valid") is True and comparison_matches is not True:
                stale_paths.add(path)
    return {
        "not_ready_paths": sorted(not_ready_paths),
        "oracle_digests": sorted(oracle_digests),
        "predicted_digests": sorted(predicted_digests),
        "ready": bool(graph_eval_artifacts) and not not_ready_paths,
        "stale_paths": sorted(stale_paths),
    }


def _error_attribution_status(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    attribution_artifacts = [
        artifact
        for artifact in artifacts
        if artifact.get("artifact_type") == "error_attribution_report"
    ]
    gold_digests: set[str] = set()
    not_ready_paths: set[str] = set()
    oracle_graph_digests: set[str] = set()
    predicted_graph_digests: set[str] = set()
    prediction_digests: set[str] = set()
    stale_paths: set[str] = set()
    for artifact in attribution_artifacts:
        path = artifact.get("path")
        if not isinstance(path, str) or path == "":
            not_ready_paths.add("")
            continue
        try:
            report = load_error_attribution_report(path)
            validation = validate_error_attribution_report(report)
            comparison_matches = (
                compare_error_attribution_report(report).get("matches") is True
                if validation.get("valid") is True
                else False
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError):
            not_ready_paths.add(path)
            continue
        gold_digest = _string_or_none(report.get("gold_digest"))
        oracle_graph_digest = _string_or_none(report.get("oracle_graph_digest"))
        predicted_graph_digest = _string_or_none(report.get("predicted_graph_digest"))
        prediction_digest = _string_or_none(report.get("prediction_digest"))
        if gold_digest is not None:
            gold_digests.add(gold_digest)
        if oracle_graph_digest is not None:
            oracle_graph_digests.add(oracle_graph_digest)
        if predicted_graph_digest is not None:
            predicted_graph_digests.add(predicted_graph_digest)
        if prediction_digest is not None:
            prediction_digests.add(prediction_digest)
        if (
            validation.get("valid") is not True
            or comparison_matches is not True
            or gold_digest is None
            or oracle_graph_digest is None
            or predicted_graph_digest is None
            or prediction_digest is None
        ):
            not_ready_paths.add(path)
            if validation.get("valid") is True and comparison_matches is not True:
                stale_paths.add(path)
    return {
        "gold_digests": sorted(gold_digests),
        "not_ready_paths": sorted(not_ready_paths),
        "oracle_graph_digests": sorted(oracle_graph_digests),
        "predicted_graph_digests": sorted(predicted_graph_digests),
        "prediction_digests": sorted(prediction_digests),
        "ready": bool(attribution_artifacts) and not not_ready_paths,
        "stale_paths": sorted(stale_paths),
    }


def _dashboard_bundle_status(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    dashboard_artifacts = [
        artifact
        for artifact in artifacts
        if artifact.get("artifact_type") == "dashboard_bundle"
    ]
    bundle_digests: set[str] = set()
    case_counts: set[int] = set()
    not_ready_paths: set[str] = set()
    stale_paths: set[str] = set()
    for artifact in dashboard_artifacts:
        path = artifact.get("path")
        if not isinstance(path, str) or path == "":
            not_ready_paths.add("")
            continue
        try:
            bundle = load_dashboard_bundle(path)
            validation = validate_dashboard_bundle(bundle)
            comparison = (
                compare_dashboard_bundle(bundle)
                if validation.get("valid") is True
                else {"matches": False, "comparable": False}
            )
            comparison_matches = comparison.get("matches") is True
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError):
            not_ready_paths.add(path)
            continue
        bundle_digest = _string_or_none(bundle.get("bundle_digest"))
        if bundle_digest is not None:
            bundle_digests.add(bundle_digest)
        summary = bundle.get("summary")
        case_count = (
            summary.get("case_count") if isinstance(summary, Mapping) else None
        )
        if isinstance(case_count, int) and not isinstance(case_count, bool):
            case_counts.add(case_count)
        if (
            validation.get("valid") is not True
            or comparison_matches is not True
            or bundle_digest is None
            or case_count is None
            or isinstance(case_count, bool)
        ):
            not_ready_paths.add(path)
            if (
                validation.get("valid") is True
                and comparison.get("comparable") is True
                and comparison_matches is not True
            ):
                stale_paths.add(path)
    return {
        "bundle_digests": sorted(bundle_digests),
        "case_counts": sorted(case_counts),
        "not_ready_paths": sorted(not_ready_paths),
        "ready": bool(dashboard_artifacts) and not not_ready_paths,
        "stale_paths": sorted(stale_paths),
    }


def _offline_control_matrix_status(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    matrix_artifacts = [
        artifact
        for artifact in artifacts
        if artifact.get("artifact_type") == "offline_control_matrix_report"
    ]
    not_ready_paths: list[str] = []
    required_source_kinds: set[str] = set()
    stale_paths: list[str] = []
    for artifact in matrix_artifacts:
        path = artifact.get("path")
        if not isinstance(path, str) or path == "":
            not_ready_paths.append("")
            continue
        try:
            report = load_offline_control_matrix_report(path)
            validation = validate_offline_control_matrix_report(report)
            comparison_matches = (
                compare_offline_control_matrix_report(report).get("matches") is True
                if validation.get("valid") is True
                else False
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError):
            not_ready_paths.append(path)
            continue
        required_source_kinds.update(
            _string_tuple_from_value(report.get("required_source_kinds"))
        )
        readiness = report.get("readiness")
        if (
            validation.get("valid") is not True
            or comparison_matches is not True
            or not isinstance(readiness, Mapping)
            or readiness.get("ready") is not True
        ):
            not_ready_paths.append(path)
            if validation.get("valid") is True and comparison_matches is not True:
                stale_paths.append(path)
    return {
        "ready": bool(matrix_artifacts) and not not_ready_paths,
        "not_ready_paths": sorted(not_ready_paths),
        "required_source_kinds": sorted(required_source_kinds),
        "stale_paths": sorted(stale_paths),
    }


def _offline_control_result_status(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result_artifacts = [
        artifact
        for artifact in artifacts
        if artifact.get("artifact_type") == "offline_control_result_report"
    ]
    candidate_names: set[str] = set()
    not_ready_paths: list[str] = []
    source_kinds: set[str] = set()
    stale_paths: list[str] = []
    for artifact in result_artifacts:
        path = artifact.get("path")
        if not isinstance(path, str) or path == "":
            not_ready_paths.append("")
            continue
        try:
            report = load_offline_control_result_report(path)
            validation = validate_offline_control_result_report(report)
            comparison_matches = (
                compare_offline_control_result_report(report).get("matches") is True
                if validation.get("valid") is True
                else False
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError):
            not_ready_paths.append(path)
            continue
        candidate_name = _string_or_none(report.get("candidate_name"))
        if candidate_name is not None:
            candidate_names.add(candidate_name)
        for row in _mapping_sequence(report.get("source_result_matrix")):
            source_kind = _string_or_none(row.get("source_kind"))
            if source_kind is not None:
                source_kinds.add(source_kind)
        readiness = report.get("readiness")
        if (
            validation.get("valid") is not True
            or comparison_matches is not True
            or not isinstance(readiness, Mapping)
            or readiness.get("ready") is not True
        ):
            not_ready_paths.append(path)
            if validation.get("valid") is True and comparison_matches is not True:
                stale_paths.append(path)
    return {
        "candidate_names": sorted(candidate_names),
        "not_ready_paths": sorted(not_ready_paths),
        "ready": bool(result_artifacts) and not not_ready_paths,
        "source_kinds": sorted(source_kinds),
        "stale_paths": sorted(stale_paths),
    }


def _qa_delta_status(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    delta_artifacts = [
        artifact
        for artifact in artifacts
        if artifact.get("artifact_type") == "qa_eval_delta_report"
    ]
    baseline_names: set[str] = set()
    candidate_names: set[str] = set()
    case_mismatch_paths: list[str] = []
    not_ready_paths: list[str] = []
    placeholder_name_paths: list[str] = []
    stale_paths: list[str] = []
    for artifact in delta_artifacts:
        path = artifact.get("path")
        if not isinstance(path, str) or path == "":
            not_ready_paths.append("")
            placeholder_name_paths.append("")
            continue
        try:
            report = load_qa_eval_delta_report(path)
            validation = validate_qa_eval_delta_report(report)
            comparison_matches = (
                compare_qa_eval_delta_report(report).get("matches") is True
                if validation.get("valid") is True
                else False
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError):
            not_ready_paths.append(path)
            continue
        candidate_name = _string_or_none(report.get("candidate_name"))
        baseline_name = _string_or_none(report.get("baseline_name"))
        if candidate_name is not None:
            candidate_names.add(candidate_name)
        if baseline_name is not None:
            baseline_names.add(baseline_name)
        if validation.get("valid") is not True:
            not_ready_paths.append(path)
        elif comparison_matches is not True:
            not_ready_paths.append(path)
            stale_paths.append(path)
        if _qa_delta_case_count_mismatch(report):
            case_mismatch_paths.append(path)
        if _qa_delta_has_placeholder_names(candidate_name, baseline_name):
            placeholder_name_paths.append(path)
    return {
        "baseline_names": sorted(baseline_names),
        "candidate_names": sorted(candidate_names),
        "case_mismatch_paths": sorted(case_mismatch_paths),
        "not_ready_paths": sorted(not_ready_paths),
        "placeholder_name_paths": sorted(placeholder_name_paths),
        "ready": bool(delta_artifacts)
        and not not_ready_paths
        and not case_mismatch_paths
        and not placeholder_name_paths,
        "stale_paths": sorted(stale_paths),
    }


def _qa_delta_case_count_mismatch(report: Mapping[str, Any]) -> bool:
    summary_delta = report.get("summary_delta")
    if not isinstance(summary_delta, Mapping):
        return True
    return summary_delta.get("case_count_match") is not True


def _qa_delta_has_placeholder_names(
    candidate_name: str | None,
    baseline_name: str | None,
) -> bool:
    if candidate_name is None or baseline_name is None:
        return True
    return _qa_delta_name_has_placeholder(candidate_name) or _qa_delta_name_has_placeholder(
        baseline_name
    )


def _qa_delta_name_has_placeholder(value: str) -> bool:
    normalized = value.lower()
    return any(marker in normalized for marker in QA_DELTA_PLACEHOLDER_NAME_MARKERS)


def _active_delta_status(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    delta_artifacts = [
        artifact
        for artifact in artifacts
        if artifact.get("artifact_type") == "active_task_delta_report"
    ]
    baseline_names: set[str] = set()
    candidate_names: set[str] = set()
    task_mismatch_paths: list[str] = []
    not_ready_paths: list[str] = []
    placeholder_name_paths: list[str] = []
    stale_paths: list[str] = []
    for artifact in delta_artifacts:
        path = artifact.get("path")
        if not isinstance(path, str) or path == "":
            not_ready_paths.append("")
            placeholder_name_paths.append("")
            continue
        try:
            report = load_active_task_delta_report(path)
            validation = validate_active_task_delta_report(report)
            comparison_matches = (
                compare_active_task_delta_report(report).get("matches") is True
                if validation.get("valid") is True
                else False
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError):
            not_ready_paths.append(path)
            continue
        candidate_name = _string_or_none(report.get("candidate_name"))
        baseline_name = _string_or_none(report.get("baseline_name"))
        if candidate_name is not None:
            candidate_names.add(candidate_name)
        if baseline_name is not None:
            baseline_names.add(baseline_name)
        if validation.get("valid") is not True:
            not_ready_paths.append(path)
        elif comparison_matches is not True:
            not_ready_paths.append(path)
            stale_paths.append(path)
        if _active_delta_task_count_mismatch(report):
            task_mismatch_paths.append(path)
        if _active_delta_has_placeholder_names(candidate_name, baseline_name):
            placeholder_name_paths.append(path)
    return {
        "baseline_names": sorted(baseline_names),
        "candidate_names": sorted(candidate_names),
        "not_ready_paths": sorted(not_ready_paths),
        "placeholder_name_paths": sorted(placeholder_name_paths),
        "ready": bool(delta_artifacts)
        and not not_ready_paths
        and not task_mismatch_paths
        and not placeholder_name_paths,
        "stale_paths": sorted(stale_paths),
        "task_mismatch_paths": sorted(task_mismatch_paths),
    }


def _active_delta_task_count_mismatch(report: Mapping[str, Any]) -> bool:
    summary_delta = report.get("summary_delta")
    if not isinstance(summary_delta, Mapping):
        return True
    return summary_delta.get("task_count_match") is not True


def _active_delta_has_placeholder_names(
    candidate_name: str | None,
    baseline_name: str | None,
) -> bool:
    if candidate_name is None or baseline_name is None:
        return True
    return _active_delta_name_has_placeholder(
        candidate_name
    ) or _active_delta_name_has_placeholder(baseline_name)


def _active_delta_name_has_placeholder(value: str) -> bool:
    normalized = value.lower()
    return any(marker in normalized for marker in ACTIVE_DELTA_PLACEHOLDER_NAME_MARKERS)


def _offline_control_import_status(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    incomplete_source_keys: set[str] = set()
    diagnostic_source_keys: set[str] = set()
    invalid_source_keys: set[str] = set()
    missing_metadata_source_keys: set[str] = set()
    placeholder_source_keys: set[str] = set()
    stale_source_keys: set[str] = set()
    for artifact in artifacts:
        if artifact.get("artifact_type") != "offline_prediction_import_report":
            continue
        path = artifact.get("path")
        if not isinstance(path, str) or path == "":
            incomplete_source_keys.add("")
            diagnostic_source_keys.add("")
            invalid_source_keys.add("")
            missing_metadata_source_keys.add("")
            continue
        try:
            report = load_offline_prediction_import_report(path)
            validation = validate_offline_prediction_import_report(report)
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError):
            incomplete_source_keys.add(path)
            diagnostic_source_keys.add(path)
            invalid_source_keys.add(path)
            missing_metadata_source_keys.add(path)
            continue
        source_key = _offline_control_source_key(report, fallback=path)
        if validation.get("valid") is not True:
            incomplete_source_keys.add(source_key)
            diagnostic_source_keys.add(source_key)
            invalid_source_keys.add(source_key)
        else:
            try:
                comparison = compare_offline_prediction_import_report(report)
            except (OSError, SpatialQAError, ValueError, json.JSONDecodeError):
                stale_source_keys.add(source_key)
                invalid_source_keys.add(source_key)
            else:
                if comparison.get("matches") is not True:
                    stale_source_keys.add(source_key)
                    invalid_source_keys.add(source_key)
        source_profile = report.get("source_profile")
        if _offline_control_missing_real_metadata(source_profile):
            missing_metadata_source_keys.add(source_key)
        if _offline_control_has_placeholder_source(report, source_key):
            placeholder_source_keys.add(source_key)
        summary = report.get("summary")
        if not isinstance(summary, Mapping):
            incomplete_source_keys.add(source_key)
            diagnostic_source_keys.add(source_key)
            continue
        imported_count = _summary_int(summary, "imported_prediction_count")
        gold_count = _summary_int(summary, "gold_case_count")
        if (
            _summary_int(summary, "missing_case_count") != 0
            or imported_count != gold_count
        ):
            incomplete_source_keys.add(source_key)
        if (
            _summary_int(summary, "unknown_case_count") != 0
            or _summary_int(summary, "duplicate_case_count") != 0
            or _summary_int(summary, "error_prediction_count") != 0
        ):
            diagnostic_source_keys.add(source_key)
    return {
        "diagnostic_source_keys": sorted(diagnostic_source_keys),
        "incomplete_source_keys": sorted(incomplete_source_keys),
        "invalid_source_keys": sorted(invalid_source_keys),
        "missing_metadata_source_keys": sorted(missing_metadata_source_keys),
        "placeholder_source_keys": sorted(placeholder_source_keys),
        "stale_source_keys": sorted(stale_source_keys),
    }


def _offline_control_missing_real_metadata(source_profile: object) -> bool:
    if not isinstance(source_profile, Mapping):
        return True
    return any(
        _source_profile_real_text(source_profile, field) is None
        for field in OFFLINE_CONTROL_REAL_METADATA_FIELDS
    )


def _source_profile_real_text(
    source_profile: Mapping[str, Any],
    field: str,
) -> str | None:
    value = source_profile.get(field)
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if normalized == "" or normalized.lower() == "unspecified":
        return None
    return normalized


def _offline_control_has_placeholder_source(
    report: Mapping[str, Any],
    source_key: str,
) -> bool:
    return any(
        _contains_placeholder_marker(value)
        for value in _offline_control_source_text_values(report, source_key)
    )


def _offline_control_source_text_values(
    report: Mapping[str, Any],
    source_key: str,
) -> tuple[str, ...]:
    values: list[str] = [source_key]
    source_profile = report.get("source_profile")
    if isinstance(source_profile, Mapping):
        for field in (
            "adapter",
            "dataset_id",
            "kind",
            "model_id",
            "name",
            "prompt_id",
            "source_key",
        ):
            value = source_profile.get(field)
            if isinstance(value, str):
                values.append(value)
    source = report.get("source")
    if isinstance(source, Mapping):
        for field in ("kind", "name"):
            value = source.get(field)
            if isinstance(value, str):
                values.append(value)
    return tuple(values)


def _contains_placeholder_marker(value: str) -> bool:
    normalized = value.lower()
    return any(marker in normalized for marker in OFFLINE_CONTROL_PLACEHOLDER_MARKERS)


def _offline_control_source_key(
    report: Mapping[str, Any],
    *,
    fallback: str,
) -> str:
    source_profile = report.get("source_profile")
    if isinstance(source_profile, Mapping):
        source_key = source_profile.get("source_key")
        if isinstance(source_key, str) and source_key != "":
            return source_key
    source = report.get("source")
    if isinstance(source, Mapping):
        kind = source.get("kind")
        name = source.get("name")
        if isinstance(kind, str) and kind != "" and isinstance(name, str) and name != "":
            return f"{kind}:{name}"
    return fallback


def _summary_int(summary: Mapping[str, Any], key: str) -> int:
    value = summary.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _predicted_input_kind(artifact: Mapping[str, Any]) -> str | None:
    path = artifact.get("path")
    if not isinstance(path, str) or path == "":
        return None
    report = load_predicted_graph_report(path)
    kind = report.get("input_kind", "episode")
    return kind if isinstance(kind, str) and kind else None


def _predicted_dsg_evidence_status(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    evidence_artifacts = [
        artifact
        for artifact in artifacts
        if artifact.get("artifact_type") == "predicted_dsg_evidence_report"
    ]
    not_ready_paths: list[str] = []
    predicted_report_digests: set[str] = set()
    for artifact in evidence_artifacts:
        path = artifact.get("path")
        if not isinstance(path, str) or path == "":
            not_ready_paths.append("")
            continue
        try:
            report = load_predicted_dsg_evidence_report(path)
            validation = validate_predicted_dsg_evidence_report(report)
            comparison_matches = (
                compare_predicted_dsg_evidence_report(report).get("matches") is True
                if validation.get("valid") is True
                else False
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError):
            not_ready_paths.append(path)
            continue
        predicted_report_digest = _string_or_none(
            report.get("predicted_graph_report_digest")
        )
        if predicted_report_digest is not None:
            predicted_report_digests.add(predicted_report_digest)
        readiness = report.get("readiness")
        if (
            validation.get("valid") is not True
            or comparison_matches is not True
            or predicted_report_digest is None
            or not isinstance(readiness, Mapping)
            or readiness.get("ready") is not True
        ):
            not_ready_paths.append(path)
    return {
        "ready": bool(evidence_artifacts) and not not_ready_paths,
        "not_ready_paths": sorted(not_ready_paths),
        "predicted_report_digests": sorted(predicted_report_digests),
    }


def _real_collection_status(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    collection_artifacts = [
        artifact
        for artifact in artifacts
        if artifact.get("artifact_type") == "real_collection_report"
    ]
    not_ready_paths: list[str] = []
    source_kinds: set[str] = set()
    stale_paths: list[str] = []
    for artifact in collection_artifacts:
        path = artifact.get("path")
        if not isinstance(path, str) or path == "":
            not_ready_paths.append("")
            continue
        try:
            report = load_real_collection_report(path)
            validation = validate_real_collection_report(report)
            comparison_matches = (
                compare_real_collection_report(report).get("matches") is True
                if validation.get("valid") is True
                else False
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError):
            not_ready_paths.append(path)
            continue
        source_kind = report.get("source_kind")
        if isinstance(source_kind, str) and source_kind != "":
            source_kinds.add(source_kind)
        readiness = report.get("readiness")
        if (
            validation.get("valid") is not True
            or comparison_matches is not True
            or not isinstance(readiness, Mapping)
            or readiness.get("ready") is not True
        ):
            not_ready_paths.append(path)
            if validation.get("valid") is True and comparison_matches is not True:
                stale_paths.append(path)
    return {
        "ready": bool(collection_artifacts) and not not_ready_paths,
        "not_ready_paths": sorted(not_ready_paths),
        "source_kinds": sorted(source_kinds),
        "stale_paths": sorted(stale_paths),
    }


def _readiness_from_checks(checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed_checks = [
        _required_mapping_str(check, "name")
        for check in checks
        if check.get("passed") is not True
    ]
    missing_groups = sorted(
        {
            _required_mapping_str(check, "group")
            for check in checks
            if check.get("passed") is not True
        }
    )
    return {
        "ready": len(failed_checks) == 0,
        "passed_count": len(checks) - len(failed_checks),
        "failed_count": len(failed_checks),
        "missing_groups": missing_groups,
        "failed_checks": failed_checks,
    }


def _minimum_check(
    name: str,
    group: str,
    actual: int,
    expected: int,
) -> dict[str, Any]:
    return {
        "name": name,
        "group": group,
        "passed": actual >= expected,
        "expected": expected,
        "actual": actual,
    }


def _coverage_check(
    name: str,
    group: str,
    counts: Mapping[str, int],
    question_types: Iterable[str],
) -> dict[str, Any]:
    matched_count = sum(counts.get(question_type, 0) for question_type in question_types)
    return {
        "name": name,
        "group": group,
        "passed": matched_count > 0,
        "expected": "at least one matching QA case",
        "actual": matched_count,
    }


def _artifact_present_check(
    name: str,
    group: str,
    artifact_counts: Mapping[str, int],
    artifact_type: str,
) -> dict[str, Any]:
    actual = artifact_counts.get(artifact_type, 0)
    return {
        "name": name,
        "group": group,
        "passed": actual > 0,
        "expected": f"at least one {artifact_type}",
        "actual": actual,
    }


def _required_values_check(
    name: str,
    group: str,
    actual_values: Sequence[str],
    required_values: Sequence[str],
) -> dict[str, Any]:
    actual = sorted(set(actual_values))
    required = sorted(set(required_values))
    missing = [value for value in required if value not in actual]
    return {
        "name": name,
        "group": group,
        "passed": not missing,
        "expected": required,
        "actual": actual,
        "missing": missing,
    }


def _manifest_int(manifest: Mapping[str, Any], key: str) -> int:
    value = manifest.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _validate_threshold(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SpatialQAError(f"Readiness threshold must be a non-negative integer: {field_name}")


def _unique_strings(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or value == "":
            raise SpatialQAError(f"Readiness field must be a string sequence: {field_name}")
        if value not in result:
            result.append(value)
    return tuple(sorted(result))


def _mapping_sequence(value: object) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return ()
    result: list[Mapping[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            result.append(cast(Mapping[str, Any], item))
    return tuple(result)


def _as_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpatialQAError(f"Readiness field must be an object: {field_name}")
    return cast(Mapping[str, Any], value)


def _int_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int] = {}
    for key, item in value.items():
        if isinstance(item, int) and not isinstance(item, bool):
            result[str(key)] = item
    return {key: result[key] for key in sorted(result)}


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Readiness field must be a non-empty string: {key}")
    return value


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise SpatialQAError(f"Readiness field must be an integer: {key}")
    return value


def _required_mapping_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Readiness mapping field must be a non-empty string: {key}")
    return value


def _string_tuple(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise SpatialQAError(f"Readiness field must be a string sequence: {key}")
    return tuple(_string_value(item, key) for item in value)


def _string_tuple_from_value(value: object) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _string_value(value: object, key: str) -> str:
    if not isinstance(value, str):
        raise SpatialQAError(f"Readiness field must be a string sequence: {key}")
    return value


def _string_sequence(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str)
        and all(isinstance(item, str) for item in value)
    )


def _thresholds_shape_valid(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return all(
        isinstance(value.get(key), int) and not isinstance(value.get(key), bool)
        for key in ("min_episode_count", "min_qa_count", "min_scene_count")
    )


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _sorted_counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def _equality_check(name: str, expected: Any, actual: Any) -> dict[str, Any]:
    check: dict[str, Any] = {
        "name": name,
        "passed": expected == actual,
        "expected": expected,
        "actual": actual,
    }
    if expected != actual:
        check["differences"] = _differences(expected, actual)
    return check


def _differences(expected: Any, actual: Any, path: str = "") -> list[dict[str, Any]]:
    if expected == actual:
        return []
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(expected) | set(actual), key=str):
            child_path = f"{path}.{key}" if path else str(key)
            differences.extend(_differences(expected.get(key), actual.get(key), child_path))
        return differences
    if (
        isinstance(expected, Sequence)
        and not isinstance(expected, str)
        and isinstance(actual, Sequence)
        and not isinstance(actual, str)
    ):
        differences = []
        max_length = max(len(expected), len(actual))
        for index in range(max_length):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            if index >= len(expected):
                differences.append(
                    {"path": child_path, "expected": None, "actual": actual[index]},
                )
                continue
            if index >= len(actual):
                differences.append(
                    {"path": child_path, "expected": expected[index], "actual": None},
                )
                continue
            differences.extend(
                _differences(expected[index], actual[index], child_path)
            )
        return differences
    return [{"path": path or "$", "expected": expected, "actual": actual}]
