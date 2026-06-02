from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.benchmark.manifest import (
    benchmark_manifest_digest,
    load_benchmark_manifest,
)
from dsg_spatialqa_lab.eval.error_attribution import (
    error_attribution_report_digest,
    load_error_attribution_report,
)
from dsg_spatialqa_lab.eval.graph_metrics import (
    graph_eval_report_digest,
    load_graph_eval_report,
)
from dsg_spatialqa_lab.eval.qa_metrics import (
    load_qa_eval_delta_report,
    load_qa_eval_report,
    qa_eval_delta_report_digest,
    qa_eval_report_digest,
)
from dsg_spatialqa_lab.eval.task_metrics import (
    active_task_delta_report_digest,
    active_task_report_digest,
    load_active_task_delta_report,
    load_active_task_report,
)
from dsg_spatialqa_lab.predicted import (
    load_predicted_graph_report,
    predicted_graph_report_digest,
)
from dsg_spatialqa_lab.schema import SpatialQAError
from dsg_spatialqa_lab.visualization.dashboard_export import (
    dashboard_bundle_digest,
    load_dashboard_bundle,
)


EXPERIMENT_SUMMARY_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.experiment-summary-report.v1"
)
QA_RESEARCH_QUESTION_AXES = {
    "spatial_qa": "Does the Dynamic Scene Graph improve spatial QA?",
    "dynamic_memory": "Does the Dynamic Scene Graph improve dynamic memory?",
    "graph_tool_query": "Does GraphTool improve graph-backed queries?",
}
REQUIRED_RESEARCH_QUESTION_KEYS = (
    "dynamic_memory",
    "graph_tool_query",
    "interactive_task",
    "spatial_qa",
)
REQUIRED_SOURCE_ARTIFACT_TYPES = (
    "active_task_delta_report",
    "qa_eval_delta_report",
)
QA_DIAGNOSTIC_SLICE_GROUPS = (
    "by_episode_id",
    "by_question_type",
    "by_reference_frame",
    "by_scene_id",
    "by_tag",
)


def experiment_summary_report(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    source_artifacts = _source_artifacts(manifest.get("experiment_artifacts"))
    qa_delta_comparisons = [
        _qa_delta_comparison(artifact)
        for artifact in source_artifacts
        if artifact["artifact_type"] == "qa_eval_delta_report"
    ]
    active_delta_comparisons = [
        _active_delta_comparison(artifact)
        for artifact in source_artifacts
        if artifact["artifact_type"] == "active_task_delta_report"
    ]
    graph_eval_summaries = [
        _graph_eval_summary(artifact)
        for artifact in source_artifacts
        if artifact["artifact_type"] == "graph_eval_report"
    ]
    error_attribution_summaries = [
        _error_attribution_summary(artifact)
        for artifact in source_artifacts
        if artifact["artifact_type"] == "error_attribution_report"
    ]
    qa_diagnostic_slices = _qa_diagnostic_slices(qa_delta_comparisons)
    graph_construction_diagnostics = _graph_construction_diagnostics(
        graph_eval_summaries
    )
    error_attribution_diagnostics = _error_attribution_diagnostics(
        error_attribution_summaries
    )
    failure_linkage_diagnostics = _failure_linkage_diagnostics(
        error_attribution_diagnostics,
        graph_construction_diagnostics,
    )
    research_questions = _research_questions(
        qa_delta_comparisons,
        active_delta_comparisons,
    )
    readiness = _readiness(research_questions, source_artifacts)
    summary = _summary(
        research_questions=research_questions,
        readiness=readiness,
        source_artifacts=source_artifacts,
        qa_delta_comparisons=qa_delta_comparisons,
        active_delta_comparisons=active_delta_comparisons,
        qa_diagnostic_slices=qa_diagnostic_slices,
        graph_construction_diagnostics=graph_construction_diagnostics,
        error_attribution_diagnostics=error_attribution_diagnostics,
        failure_linkage_diagnostics=failure_linkage_diagnostics,
    )
    report: dict[str, Any] = {
        "schema_version": EXPERIMENT_SUMMARY_REPORT_SCHEMA_VERSION,
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "manifest_digest": _manifest_digest(manifest),
        "source_artifact_digests": {
            artifact["artifact_key"]: artifact["digest"]
            for artifact in source_artifacts
        },
        "source_artifacts": source_artifacts,
        "qa_delta_comparisons": qa_delta_comparisons,
        "qa_diagnostic_slices": qa_diagnostic_slices,
        "active_task_delta_comparisons": active_delta_comparisons,
        "graph_eval_summaries": graph_eval_summaries,
        "graph_construction_diagnostics": graph_construction_diagnostics,
        "error_attribution_summaries": error_attribution_summaries,
        "error_attribution_diagnostics": error_attribution_diagnostics,
        "failure_linkage_diagnostics": failure_linkage_diagnostics,
        "research_questions": research_questions,
        "readiness": readiness,
        "summary": summary,
    }
    report["report_digest"] = experiment_summary_report_digest(report)
    return report


def experiment_summary_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def experiment_summary_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_experiment_summary_report(report: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(experiment_summary_report_json(report), encoding="utf-8")
    return output_path


def load_experiment_summary_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Experiment summary report JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_experiment_summary_report(report: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = report.get("schema_version")
    report_digest = _string_or_none(report.get("report_digest"))
    expected_report_digest = experiment_summary_report_digest(report)
    source_artifacts = _mapping_sequence(report.get("source_artifacts"))
    qa_delta_comparisons = _mapping_sequence(report.get("qa_delta_comparisons"))
    active_delta_comparisons = _mapping_sequence(
        report.get("active_task_delta_comparisons")
    )
    graph_eval_summaries = _mapping_sequence(report.get("graph_eval_summaries"))
    error_attribution_summaries = _mapping_sequence(
        report.get("error_attribution_summaries")
    )
    qa_diagnostic_slices = _mapping_or_empty(report.get("qa_diagnostic_slices"))
    graph_construction_diagnostics = _mapping_or_empty(
        report.get("graph_construction_diagnostics")
    )
    error_attribution_diagnostics = _mapping_or_empty(
        report.get("error_attribution_diagnostics")
    )
    failure_linkage_diagnostics = _mapping_or_empty(
        report.get("failure_linkage_diagnostics")
    )
    research_questions = _mapping_or_empty(report.get("research_questions"))
    summary = _mapping_or_empty(report.get("summary"))
    expected_qa_diagnostic_slices = _qa_diagnostic_slices(qa_delta_comparisons)
    expected_graph_construction_diagnostics = _graph_construction_diagnostics(
        graph_eval_summaries
    )
    expected_error_attribution_diagnostics = _error_attribution_diagnostics(
        error_attribution_summaries
    )
    expected_failure_linkage_diagnostics = _failure_linkage_diagnostics(
        expected_error_attribution_diagnostics,
        expected_graph_construction_diagnostics,
    )
    expected_research_questions = _research_questions(
        qa_delta_comparisons,
        active_delta_comparisons,
    )
    expected_readiness = _readiness(expected_research_questions, source_artifacts)
    expected_summary = _summary(
        research_questions=expected_research_questions,
        readiness=expected_readiness,
        source_artifacts=source_artifacts,
        qa_delta_comparisons=qa_delta_comparisons,
        active_delta_comparisons=active_delta_comparisons,
        qa_diagnostic_slices=expected_qa_diagnostic_slices,
        graph_construction_diagnostics=expected_graph_construction_diagnostics,
        error_attribution_diagnostics=expected_error_attribution_diagnostics,
        failure_linkage_diagnostics=expected_failure_linkage_diagnostics,
    )
    expected_artifact_digests = {
        _required_str(artifact, "artifact_key"): artifact.get("digest")
        for artifact in source_artifacts
    }
    expected_source_comparisons = _source_comparison_map(source_artifacts)
    actual_source_comparisons = _comparison_source_map(
        qa_delta_comparisons,
        active_delta_comparisons,
        graph_eval_summaries,
        error_attribution_summaries,
    )
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == EXPERIMENT_SUMMARY_REPORT_SCHEMA_VERSION,
            "expected": EXPERIMENT_SUMMARY_REPORT_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "report_digest",
            "passed": report_digest == expected_report_digest,
            "expected": expected_report_digest,
            "actual": report_digest,
        },
        {
            "name": "source_artifact_digests",
            "passed": report.get("source_artifact_digests") == expected_artifact_digests,
            "expected": expected_artifact_digests,
            "actual": report.get("source_artifact_digests"),
        },
        {
            "name": "source_artifact_count",
            "passed": summary.get("source_artifact_count") == len(source_artifacts),
            "expected": len(source_artifacts),
            "actual": summary.get("source_artifact_count"),
        },
        {
            "name": "source_artifact_comparisons",
            "passed": actual_source_comparisons == expected_source_comparisons,
            "expected": expected_source_comparisons,
            "actual": actual_source_comparisons,
        },
        {
            "name": "qa_diagnostic_slices",
            "passed": qa_diagnostic_slices == expected_qa_diagnostic_slices,
            "expected": expected_qa_diagnostic_slices,
            "actual": qa_diagnostic_slices,
        },
        {
            "name": "graph_construction_diagnostics",
            "passed": graph_construction_diagnostics
            == expected_graph_construction_diagnostics,
            "expected": expected_graph_construction_diagnostics,
            "actual": graph_construction_diagnostics,
        },
        {
            "name": "error_attribution_diagnostics",
            "passed": error_attribution_diagnostics
            == expected_error_attribution_diagnostics,
            "expected": expected_error_attribution_diagnostics,
            "actual": error_attribution_diagnostics,
        },
        {
            "name": "failure_linkage_diagnostics",
            "passed": failure_linkage_diagnostics
            == expected_failure_linkage_diagnostics,
            "expected": expected_failure_linkage_diagnostics,
            "actual": failure_linkage_diagnostics,
        },
        {
            "name": "research_questions",
            "passed": report.get("research_questions") == expected_research_questions,
            "expected": expected_research_questions,
            "actual": report.get("research_questions"),
        },
        {
            "name": "readiness",
            "passed": report.get("readiness") == expected_readiness,
            "expected": expected_readiness,
            "actual": report.get("readiness"),
        },
        {
            "name": "summary",
            "passed": report.get("summary") == expected_summary,
            "expected": expected_summary,
            "actual": report.get("summary"),
        },
        {
            "name": "available_research_question_count",
            "passed": summary.get("available_research_question_count")
            == _available_research_question_count(research_questions),
            "expected": _available_research_question_count(research_questions),
            "actual": summary.get("available_research_question_count"),
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "report_digest": report_digest,
        "checks": checks,
    }


def compare_experiment_summary_report(report: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = _required_report_path(report, "manifest_path")
    current_report = experiment_summary_report(
        load_benchmark_manifest(manifest_path),
        manifest_path=manifest_path,
    )
    validation = validate_experiment_summary_report(report)
    saved_digest = _string_or_none(report.get("report_digest"))
    current_digest = _string_or_none(current_report.get("report_digest"))
    checks = [
        {
            "name": "report_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        {
            "name": "report_digest_matches_current",
            "passed": saved_digest == current_digest,
            "expected": saved_digest,
            "actual": current_digest,
        },
        _equality_check(
            "source_artifact_digests_match_current",
            report.get("source_artifact_digests"),
            current_report["source_artifact_digests"],
        ),
        _equality_check(
            "source_artifacts_match_current",
            report.get("source_artifacts"),
            current_report["source_artifacts"],
        ),
        _equality_check(
            "qa_diagnostic_slices_match_current",
            report.get("qa_diagnostic_slices"),
            current_report["qa_diagnostic_slices"],
        ),
        _equality_check(
            "graph_eval_summaries_match_current",
            report.get("graph_eval_summaries"),
            current_report["graph_eval_summaries"],
        ),
        _equality_check(
            "graph_construction_diagnostics_match_current",
            report.get("graph_construction_diagnostics"),
            current_report["graph_construction_diagnostics"],
        ),
        _equality_check(
            "error_attribution_summaries_match_current",
            report.get("error_attribution_summaries"),
            current_report["error_attribution_summaries"],
        ),
        _equality_check(
            "error_attribution_diagnostics_match_current",
            report.get("error_attribution_diagnostics"),
            current_report["error_attribution_diagnostics"],
        ),
        _equality_check(
            "failure_linkage_diagnostics_match_current",
            report.get("failure_linkage_diagnostics"),
            current_report["failure_linkage_diagnostics"],
        ),
        _equality_check(
            "research_questions_match_current",
            report.get("research_questions"),
            current_report["research_questions"],
        ),
        _equality_check(
            "readiness_matches_current",
            report.get("readiness"),
            current_report["readiness"],
        ),
        _equality_check(
            "summary_matches_current",
            report.get("summary"),
            current_report["summary"],
        ),
    ]
    return {
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def _source_artifacts(value: object) -> list[dict[str, Any]]:
    artifacts = []
    for artifact in _mapping_sequence(value):
        artifact_type = _required_str(artifact, "artifact_type")
        path = Path(_required_str(artifact, "path"))
        payload, digest = _load_source_artifact(artifact_type, path)
        artifacts.append(
            {
                "artifact_key": _experiment_artifact_key(artifact_type, path),
                "artifact_type": artifact_type,
                "path": str(path),
                "schema_version": _string_or_none(payload.get("schema_version")),
                "digest": digest,
                "recorded_digest": _string_or_none(artifact.get("digest")),
                "digest_matches_manifest": artifact.get("digest") == digest,
            }
        )
    return sorted(
        artifacts,
        key=lambda item: (str(item["artifact_type"]), str(item["path"])),
    )


def _load_source_artifact(
    artifact_type: str,
    path: Path,
) -> tuple[Mapping[str, Any], str]:
    if artifact_type == "active_task_delta_report":
        payload = load_active_task_delta_report(path)
        return payload, active_task_delta_report_digest(payload)
    if artifact_type == "active_task_report":
        payload = load_active_task_report(path)
        return payload, active_task_report_digest(payload)
    if artifact_type == "dashboard_bundle":
        payload = load_dashboard_bundle(path)
        return payload, dashboard_bundle_digest(payload)
    if artifact_type == "error_attribution_report":
        payload = load_error_attribution_report(path)
        return payload, error_attribution_report_digest(payload)
    if artifact_type == "graph_eval_report":
        payload = load_graph_eval_report(path)
        return payload, graph_eval_report_digest(payload)
    if artifact_type == "predicted_graph_report":
        payload = load_predicted_graph_report(path)
        return payload, predicted_graph_report_digest(payload)
    if artifact_type == "qa_eval_delta_report":
        payload = load_qa_eval_delta_report(path)
        return payload, qa_eval_delta_report_digest(payload)
    if artifact_type == "qa_eval_report":
        payload = load_qa_eval_report(path)
        return payload, qa_eval_report_digest(payload)
    raise SpatialQAError(f"Unsupported experiment artifact type: {artifact_type}")


def _qa_delta_comparison(source_artifact: Mapping[str, Any]) -> dict[str, Any]:
    report = load_qa_eval_delta_report(_required_str(source_artifact, "path"))
    breakdown_delta = _mapping_or_empty(report.get("breakdown_delta"))
    axes = _mapping_or_empty(
        breakdown_delta.get("by_research_axis")
    )
    return {
        "artifact_key": _required_str(source_artifact, "artifact_key"),
        "path": _required_str(source_artifact, "path"),
        "digest": _required_str(source_artifact, "digest"),
        "candidate_name": _string_or_none(report.get("candidate_name")),
        "baseline_name": _string_or_none(report.get("baseline_name")),
        "candidate_report_digest": _string_or_none(
            report.get("candidate_report_digest")
        ),
        "baseline_report_digest": _string_or_none(report.get("baseline_report_digest")),
        "summary_delta": _json_value(report.get("summary_delta")),
        "metrics_delta": _json_value(report.get("metrics_delta")),
        "research_axis_delta": {
            str(axis): _json_value(value)
            for axis, value in sorted(axes.items(), key=lambda item: str(item[0]))
        },
        "diagnostic_slice_delta": _qa_diagnostic_slice_delta(breakdown_delta),
    }


def _active_delta_comparison(source_artifact: Mapping[str, Any]) -> dict[str, Any]:
    report = load_active_task_delta_report(_required_str(source_artifact, "path"))
    return {
        "artifact_key": _required_str(source_artifact, "artifact_key"),
        "path": _required_str(source_artifact, "path"),
        "digest": _required_str(source_artifact, "digest"),
        "candidate_name": _string_or_none(report.get("candidate_name")),
        "baseline_name": _string_or_none(report.get("baseline_name")),
        "candidate_report_digest": _string_or_none(
            report.get("candidate_report_digest")
        ),
        "baseline_report_digest": _string_or_none(report.get("baseline_report_digest")),
        "summary_delta": _json_value(report.get("summary_delta")),
        "metrics_delta": _json_value(report.get("metrics_delta")),
        "budget_delta": _json_value(report.get("budget_delta")),
    }


def _graph_eval_summary(source_artifact: Mapping[str, Any]) -> dict[str, Any]:
    report = load_graph_eval_report(_required_str(source_artifact, "path"))
    metrics = _mapping_or_empty(report.get("metrics"))
    diagnostics = _mapping_or_empty(report.get("diagnostics"))
    breakdown = _mapping_or_empty(report.get("breakdown"))
    return {
        "artifact_key": _required_str(source_artifact, "artifact_key"),
        "path": _required_str(source_artifact, "path"),
        "digest": _required_str(source_artifact, "digest"),
        "oracle_digest": _string_or_none(report.get("oracle_digest")),
        "predicted_digest": _string_or_none(report.get("predicted_digest")),
        "summary": _json_value(report.get("summary")),
        "primary_metrics": {
            "object_recall_rate": _metric_rate(metrics, "object_recall"),
            "relation_f1_rate": _metric_rate(metrics, "relation_f1"),
            "state_accuracy_rate": _metric_rate(metrics, "state_accuracy"),
        },
        "diagnostics": {
            "duplicate_track_count": _int_or_none(
                diagnostics.get("duplicate_track_count")
            ),
            "id_fragmentation_count": _int_or_none(
                diagnostics.get("id_fragmentation_count")
            ),
        },
        "source_breakdown": _json_value(
            _mapping_or_empty(breakdown.get("by_prediction_source"))
        ),
    }


def _error_attribution_summary(source_artifact: Mapping[str, Any]) -> dict[str, Any]:
    report = load_error_attribution_report(_required_str(source_artifact, "path"))
    return {
        "artifact_key": _required_str(source_artifact, "artifact_key"),
        "path": _required_str(source_artifact, "path"),
        "digest": _required_str(source_artifact, "digest"),
        "gold_digest": _string_or_none(report.get("gold_digest")),
        "oracle_graph_digest": _string_or_none(report.get("oracle_graph_digest")),
        "predicted_graph_digest": _string_or_none(
            report.get("predicted_graph_digest")
        ),
        "prediction_digest": _string_or_none(report.get("prediction_digest")),
        "summary": _json_value(report.get("summary")),
    }


def _qa_research_question(
    axis: str,
    comparisons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    measurements = []
    for comparison in comparisons:
        axis_delta = _mapping_or_empty(
            _mapping_or_empty(comparison.get("research_axis_delta")).get(axis)
        )
        if not axis_delta:
            continue
        measurements.append(
            {
                "artifact_key": _required_str(comparison, "artifact_key"),
                "candidate_name": _string_or_none(comparison.get("candidate_name")),
                "baseline_name": _string_or_none(comparison.get("baseline_name")),
                "case_count_match": _bool_or_none(axis_delta.get("case_count_match")),
                "primary_metric": {
                    "name": "exact_match_rate_delta",
                    "value": _float_or_none(axis_delta.get("exact_match_rate_delta")),
                },
                "supporting_metrics": {
                    "baseline_case_count": _int_or_none(
                        axis_delta.get("baseline_case_count")
                    ),
                    "candidate_case_count": _int_or_none(
                        axis_delta.get("candidate_case_count")
                    ),
                    "exact_match_count_delta": _int_or_none(
                        axis_delta.get("exact_match_count_delta")
                    ),
                    "mean_evidence_edge_recall_delta": _float_or_none(
                        axis_delta.get("mean_evidence_edge_recall_delta")
                    ),
                    "mean_evidence_node_recall_delta": _float_or_none(
                        axis_delta.get("mean_evidence_node_recall_delta")
                    ),
                },
            }
        )
    return _research_question_entry(
        label=QA_RESEARCH_QUESTION_AXES[axis],
        source_artifact_type="qa_eval_delta_report",
        measurements=measurements,
    )


def _interactive_task_research_question(
    comparisons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    measurements = []
    for comparison in comparisons:
        metrics_delta = _mapping_or_empty(comparison.get("metrics_delta"))
        task_success = _mapping_or_empty(metrics_delta.get("task_success"))
        answer_accuracy = _mapping_or_empty(metrics_delta.get("answer_accuracy"))
        evidence_coverage = _mapping_or_empty(metrics_delta.get("evidence_coverage"))
        action_count = _mapping_or_empty(metrics_delta.get("action_count"))
        measurements.append(
            {
                "artifact_key": _required_str(comparison, "artifact_key"),
                "candidate_name": _string_or_none(comparison.get("candidate_name")),
                "baseline_name": _string_or_none(comparison.get("baseline_name")),
                "case_count_match": _bool_or_none(task_success.get("total_match")),
                "primary_metric": {
                    "name": "task_success_rate_delta",
                    "value": _float_or_none(task_success.get("rate_delta")),
                },
                "supporting_metrics": {
                    "answer_accuracy_rate_delta": _float_or_none(
                        answer_accuracy.get("rate_delta")
                    ),
                    "evidence_coverage_average_delta": _float_or_none(
                        evidence_coverage.get("average_delta")
                    ),
                    "action_count_average_delta": _float_or_none(
                        action_count.get("average_delta")
                    ),
                    "success_count_delta": _int_or_none(
                        task_success.get("count_delta")
                    ),
                },
            }
        )
    return _research_question_entry(
        label="Does the Dynamic Scene Graph improve interactive task ability?",
        source_artifact_type="active_task_delta_report",
        measurements=measurements,
    )


def _research_questions(
    qa_delta_comparisons: Sequence[Mapping[str, Any]],
    active_delta_comparisons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "dynamic_memory": _qa_research_question(
            "dynamic_memory",
            qa_delta_comparisons,
        ),
        "graph_tool_query": _qa_research_question(
            "graph_tool_query",
            qa_delta_comparisons,
        ),
        "interactive_task": _interactive_task_research_question(
            active_delta_comparisons,
        ),
        "spatial_qa": _qa_research_question("spatial_qa", qa_delta_comparisons),
    }


def _research_question_entry(
    *,
    label: str,
    source_artifact_type: str,
    measurements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not measurements:
        return {
            "label": label,
            "status": "missing",
            "source_artifact_type": source_artifact_type,
            "primary_metric": None,
            "supporting_metrics": {},
            "verdict": "inconclusive",
            "measurements": [],
        }
    primary = _json_value(measurements[0]["primary_metric"])
    supporting = _json_value(measurements[0]["supporting_metrics"])
    return {
        "label": label,
        "status": "available",
        "source_artifact_type": source_artifact_type,
        "primary_metric": primary,
        "supporting_metrics": supporting,
        "verdict": _verdict(
            primary,
            case_count_match=_bool_or_none(measurements[0].get("case_count_match")),
        ),
        "measurements": [_json_value(measurement) for measurement in measurements],
    }


def _summary(
    *,
    research_questions: Mapping[str, Any],
    readiness: Mapping[str, Any],
    source_artifacts: Sequence[Mapping[str, Any]],
    qa_delta_comparisons: Sequence[Mapping[str, Any]],
    active_delta_comparisons: Sequence[Mapping[str, Any]],
    qa_diagnostic_slices: Mapping[str, Any],
    graph_construction_diagnostics: Mapping[str, Any],
    error_attribution_diagnostics: Mapping[str, Any],
    failure_linkage_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "active_task_delta_report_count": len(active_delta_comparisons),
        "available_research_question_count": _available_research_question_count(
            research_questions
        ),
        "error_attribution_diagnostic_count": len(
            error_attribution_diagnostics
        ),
        "failure_linkage_diagnostic_count": len(failure_linkage_diagnostics),
        "graph_construction_diagnostic_count": len(
            graph_construction_diagnostics
        ),
        "qa_eval_delta_report_count": len(qa_delta_comparisons),
        "qa_diagnostic_slice_count": _qa_diagnostic_slice_count(
            qa_diagnostic_slices
        ),
        "readiness_status": _string_or_none(readiness.get("status")),
        "research_question_count": len(research_questions),
        "source_artifact_count": len(source_artifacts),
        "verdict_counts": _verdict_counts(research_questions),
    }


def _verdict(primary_metric: object, *, case_count_match: bool | None) -> str:
    if case_count_match is False:
        return "inconclusive"
    metric = _mapping_or_empty(primary_metric)
    value = _float_or_none(metric.get("value"))
    if value is None:
        return "inconclusive"
    if value > 0:
        return "improved"
    if value < 0:
        return "regressed"
    return "unchanged"


def _verdict_counts(research_questions: Mapping[str, Any]) -> dict[str, int]:
    counts = {
        "improved": 0,
        "inconclusive": 0,
        "regressed": 0,
        "unchanged": 0,
    }
    for key in REQUIRED_RESEARCH_QUESTION_KEYS:
        question = _mapping_or_empty(research_questions.get(key))
        verdict = _string_or_none(question.get("verdict"))
        if verdict not in counts:
            verdict = "inconclusive"
        counts[verdict] += 1
    return counts


def _readiness(
    research_questions: Mapping[str, Any],
    source_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    available_source_artifact_types = sorted(
        {
            _required_str(artifact, "artifact_type")
            for artifact in source_artifacts
            if artifact.get("artifact_type") in REQUIRED_SOURCE_ARTIFACT_TYPES
        }
    )
    checks = [
        _readiness_check(key, research_questions, available_source_artifact_types)
        for key in REQUIRED_RESEARCH_QUESTION_KEYS
    ]
    available_research_questions = [
        check["name"] for check in checks if check["passed"] is True
    ]
    missing_research_questions = [
        check["name"] for check in checks if check["passed"] is not True
    ]
    missing_source_artifact_types = [
        artifact_type
        for artifact_type in REQUIRED_SOURCE_ARTIFACT_TYPES
        if artifact_type not in available_source_artifact_types
    ]
    return {
        "status": "ready" if not missing_research_questions else "incomplete",
        "required_research_questions": list(REQUIRED_RESEARCH_QUESTION_KEYS),
        "available_research_questions": available_research_questions,
        "missing_research_questions": missing_research_questions,
        "required_source_artifact_types": list(REQUIRED_SOURCE_ARTIFACT_TYPES),
        "available_source_artifact_types": available_source_artifact_types,
        "missing_source_artifact_types": missing_source_artifact_types,
        "checks": checks,
    }


def _readiness_check(
    key: str,
    research_questions: Mapping[str, Any],
    available_source_artifact_types: Sequence[str],
) -> dict[str, Any]:
    entry = _mapping_or_empty(research_questions.get(key))
    source_artifact_type = _string_or_none(entry.get("source_artifact_type"))
    if source_artifact_type is None:
        source_artifact_type = _expected_source_artifact_type(key)
    measurement_count = len(_mapping_sequence(entry.get("measurements")))
    passed = entry.get("status") == "available" and measurement_count > 0
    return {
        "name": key,
        "passed": passed,
        "source_artifact_type": source_artifact_type,
        "measurement_count": measurement_count,
        "missing_reason": None
        if passed
        else _readiness_missing_reason(
            key,
            source_artifact_type,
            available_source_artifact_types,
        ),
    }


def _expected_source_artifact_type(key: str) -> str:
    if key == "interactive_task":
        return "active_task_delta_report"
    return "qa_eval_delta_report"


def _readiness_missing_reason(
    key: str,
    source_artifact_type: str,
    available_source_artifact_types: Sequence[str],
) -> str:
    if source_artifact_type not in available_source_artifact_types:
        return f"missing_{source_artifact_type}"
    if key == "interactive_task":
        return "missing_interactive_task_measurement"
    return f"missing_{key}_research_axis_delta"


def _source_comparison_map(
    source_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        _required_str(artifact, "artifact_key"): {
            "artifact_type": _required_str(artifact, "artifact_type"),
            "digest": _required_str(artifact, "digest"),
            "path": _required_str(artifact, "path"),
        }
        for artifact in source_artifacts
        if artifact.get("artifact_type")
        in {
            "active_task_delta_report",
            "error_attribution_report",
            "graph_eval_report",
            "qa_eval_delta_report",
        }
    }


def _comparison_source_map(
    qa_delta_comparisons: Sequence[Mapping[str, Any]],
    active_delta_comparisons: Sequence[Mapping[str, Any]],
    graph_eval_summaries: Sequence[Mapping[str, Any]],
    error_attribution_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    comparison_items = [
        ("qa_eval_delta_report", comparison)
        for comparison in qa_delta_comparisons
    ]
    comparison_items.extend(
        ("active_task_delta_report", comparison)
        for comparison in active_delta_comparisons
    )
    comparison_items.extend(
        ("graph_eval_report", summary) for summary in graph_eval_summaries
    )
    comparison_items.extend(
        ("error_attribution_report", summary)
        for summary in error_attribution_summaries
    )
    return {
        _required_str(comparison, "artifact_key"): {
            "artifact_type": artifact_type,
            "digest": _required_str(comparison, "digest"),
            "path": _required_str(comparison, "path"),
        }
        for artifact_type, comparison in comparison_items
    }


def _available_research_question_count(research_questions: Mapping[str, Any]) -> int:
    count = 0
    for value in research_questions.values():
        if isinstance(value, Mapping) and value.get("status") == "available":
            count += 1
    return count


def _qa_diagnostic_slice_delta(
    breakdown_delta: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        group_name: {
            str(name): _json_value(value)
            for name, value in sorted(
                _mapping_or_empty(breakdown_delta.get(group_name)).items(),
                key=lambda item: str(item[0]),
            )
        }
        for group_name in QA_DIAGNOSTIC_SLICE_GROUPS
    }


def _qa_diagnostic_slices(
    qa_delta_comparisons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        _required_str(comparison, "artifact_key"): _json_value(
            comparison.get("diagnostic_slice_delta")
        )
        for comparison in sorted(
            qa_delta_comparisons,
            key=lambda item: _required_str(item, "artifact_key"),
        )
    }


def _qa_diagnostic_slice_count(qa_diagnostic_slices: Mapping[str, Any]) -> int:
    count = 0
    for artifact_value in qa_diagnostic_slices.values():
        artifact = _mapping_or_empty(artifact_value)
        for group_name in QA_DIAGNOSTIC_SLICE_GROUPS:
            count += len(_mapping_or_empty(artifact.get(group_name)))
    return count


def _graph_construction_diagnostics(
    graph_eval_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        _required_str(summary, "artifact_key"): {
            "digest": _required_str(summary, "digest"),
            "path": _required_str(summary, "path"),
            "oracle_digest": _string_or_none(summary.get("oracle_digest")),
            "predicted_digest": _string_or_none(summary.get("predicted_digest")),
            "summary": _json_value(summary.get("summary")),
            "primary_metrics": _json_value(
                _mapping_or_empty(summary.get("primary_metrics"))
            ),
            "diagnostics": _json_value(
                _mapping_or_empty(summary.get("diagnostics"))
            ),
            "source_breakdown": _json_value(
                _mapping_or_empty(summary.get("source_breakdown"))
            ),
        }
        for summary in sorted(
            graph_eval_summaries,
            key=lambda item: _required_str(item, "artifact_key"),
        )
    }


def _error_attribution_diagnostics(
    error_attribution_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        _required_str(summary, "artifact_key"): {
            "digest": _required_str(summary, "digest"),
            "path": _required_str(summary, "path"),
            "gold_digest": _string_or_none(summary.get("gold_digest")),
            "oracle_graph_digest": _string_or_none(
                summary.get("oracle_graph_digest")
            ),
            "predicted_graph_digest": _string_or_none(
                summary.get("predicted_graph_digest")
            ),
            "prediction_digest": _string_or_none(
                summary.get("prediction_digest")
            ),
            "summary": _json_value(_mapping_or_empty(summary.get("summary"))),
        }
        for summary in sorted(
            error_attribution_summaries,
            key=lambda item: _required_str(item, "artifact_key"),
        )
    }


def _failure_linkage_diagnostics(
    error_attribution_diagnostics: Mapping[str, Any],
    graph_construction_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    graph_by_digest = _graph_diagnostics_by_digest(graph_construction_diagnostics)
    linked: dict[str, Any] = {}
    for attribution_key, attribution_value in sorted(
        error_attribution_diagnostics.items(),
        key=lambda item: str(item[0]),
    ):
        attribution = _mapping_or_empty(attribution_value)
        oracle_digest = _string_or_none(attribution.get("oracle_graph_digest"))
        predicted_digest = _string_or_none(
            attribution.get("predicted_graph_digest")
        )
        graph_key, graph = graph_by_digest.get(
            (oracle_digest, predicted_digest),
            (None, {}),
        )
        linked[str(attribution_key)] = {
            "attribution_summary": _json_value(
                _mapping_or_empty(attribution.get("summary"))
            ),
            "error_attribution_artifact_key": str(attribution_key),
            "graph_diagnostics": _json_value(
                _mapping_or_empty(graph.get("diagnostics"))
            ),
            "graph_eval_artifact_key": graph_key,
            "graph_primary_metrics": _json_value(
                _mapping_or_empty(graph.get("primary_metrics"))
            ),
            "linked_by": "oracle_and_predicted_graph_digest"
            if graph_key is not None
            else "unmatched_graph_digest",
            "oracle_graph_digest": oracle_digest,
            "predicted_graph_digest": predicted_digest,
        }
    return linked


def _graph_diagnostics_by_digest(
    graph_construction_diagnostics: Mapping[str, Any],
) -> dict[tuple[str | None, str | None], tuple[str, Mapping[str, Any]]]:
    index: dict[tuple[str | None, str | None], tuple[str, Mapping[str, Any]]] = {}
    for graph_key, graph_value in sorted(
        graph_construction_diagnostics.items(),
        key=lambda item: str(item[0]),
    ):
        graph = _mapping_or_empty(graph_value)
        digest_key = (
            _string_or_none(graph.get("oracle_digest")),
            _string_or_none(graph.get("predicted_digest")),
        )
        if digest_key not in index:
            index[digest_key] = (str(graph_key), graph)
    return index


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    manifest_digest = _string_or_none(manifest.get("manifest_digest"))
    return manifest_digest if manifest_digest is not None else benchmark_manifest_digest(manifest)


def _experiment_artifact_key(artifact_type: str, path: Path) -> str:
    return f"{artifact_type}:{path.name}"


def _mapping_sequence(value: object) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return ()
    items: list[Mapping[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            items.append(cast(Mapping[str, Any], item))
    return tuple(items)


def _mapping_or_empty(value: object) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else {}


def _required_report_path(report: Mapping[str, Any], key: str) -> Path:
    value = report.get(key)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Experiment summary report missing required path: {key}")
    return Path(value)


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Experiment summary field must be a non-empty string: {key}")
    return value


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _bool_or_none(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _metric_rate(metrics: Mapping[str, Any], metric_name: str) -> float | None:
    return _float_or_none(_mapping_or_empty(metrics.get(metric_name)).get("rate"))


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_json_value(item) for item in value]
    return value


def _equality_check(name: str, saved: Any, current: Any) -> dict[str, Any]:
    check: dict[str, Any] = {
        "name": name,
        "passed": saved == current,
        "expected": saved,
        "actual": current,
    }
    if saved != current:
        check["differences"] = _differences(saved, current)
    return check


def _differences(saved: Any, current: Any, path: str = "") -> list[dict[str, Any]]:
    if saved == current:
        return []
    if isinstance(saved, Mapping) and isinstance(current, Mapping):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(saved) | set(current), key=str):
            child_path = f"{path}.{key}" if path else str(key)
            differences.extend(_differences(saved.get(key), current.get(key), child_path))
        return differences
    return [{"path": path or "value", "expected": saved, "actual": current}]
