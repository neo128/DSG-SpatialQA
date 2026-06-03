from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.benchmark.real_run import (
    REAL_EXPERIMENT_RUN_MANIFEST_SCHEMA_VERSION,
    compare_real_experiment_run_ledger,
    load_real_experiment_run_manifest,
    load_real_experiment_run_ledger,
    real_experiment_run_manifest_digest,
    real_experiment_run_manifest_preflight,
    real_experiment_run_ledger_digest,
    validate_real_experiment_run_ledger,
)
from dsg_spatialqa_lab.benchmark.manifest import (
    benchmark_manifest_digest,
    load_benchmark_manifest,
    validate_benchmark_manifest,
)
from dsg_spatialqa_lab.benchmark.experiment_record import (
    experiment_record_digest,
    load_experiment_record,
    validate_experiment_record,
)
from dsg_spatialqa_lab.benchmark.experiment_summary import (
    experiment_summary_report_digest,
    load_experiment_summary_report,
    validate_experiment_summary_report,
)
from dsg_spatialqa_lab.benchmark.readiness import (
    load_real_experiment_readiness_report,
    real_experiment_readiness_report_digest,
    validate_real_experiment_readiness_report,
)
from dsg_spatialqa_lab.benchmark.real_collection import (
    SUPPORTED_REAL_COLLECTION_SOURCE_KINDS,
    load_real_collection_request_bundle,
    load_real_collection_report,
    real_collection_request_bundle,
    save_real_collection_request_bundle,
    validate_real_collection_request_bundle,
    validate_real_collection_report,
)
from dsg_spatialqa_lab.eval.offline_control_run import (
    OFFLINE_CONTROL_IMPORT_MANIFEST_SCHEMA_VERSION,
    OFFLINE_CONTROL_PREDICTION_REQUEST_BUNDLE_SCHEMA_VERSION,
    load_offline_control_import_run_ledger,
    load_offline_control_artifact_contracts,
    load_offline_control_import_manifest,
    load_offline_control_prediction_receipt_bundle,
    load_offline_control_prediction_request_bundle,
    offline_control_artifact_launch_report,
    offline_control_import_manifest_digest,
    offline_control_import_run_ledger_digest,
    offline_control_prediction_request_bundle_digest,
    offline_control_prediction_request_bundle,
    offline_control_prediction_receipt_bundle_digest,
    save_offline_control_prediction_request_bundle,
    validate_offline_control_prediction_receipt_bundle,
    validate_offline_control_import_run_ledger,
)
from dsg_spatialqa_lab.eval.offline_predictions import QA_PREDICTION_INPUT_FORMAT
from dsg_spatialqa_lab.predicted_run import (
    PREDICTED_DSG_DETECTOR_REQUEST_BUNDLE_SCHEMA_VERSION,
    PREDICTED_DSG_DETECTOR_RUN_MANIFEST_SCHEMA_VERSION,
    load_predicted_dsg_detector_artifact_contract,
    load_predicted_dsg_detector_receipt_bundle,
    load_predicted_dsg_detector_request_bundle,
    load_predicted_dsg_detector_run_ledger,
    load_predicted_dsg_detector_run_manifest,
    predicted_dsg_detector_request_bundle,
    predicted_dsg_detector_request_bundle_digest,
    predicted_dsg_detector_artifact_launch_report,
    predicted_dsg_detector_receipt_bundle_digest,
    predicted_dsg_detector_run_ledger_digest,
    predicted_dsg_detector_run_manifest_digest,
    save_predicted_dsg_detector_request_bundle,
    validate_predicted_dsg_detector_receipt_bundle,
    validate_predicted_dsg_detector_run_ledger,
)
from dsg_spatialqa_lab.schema import SpatialQAError


REAL_EXPERIMENT_HANDOFF_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.real-experiment-handoff.v1"
)
REAL_EXPERIMENT_ARTIFACT_CHECKLIST_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.real-experiment-artifact-checklist.v1"
)
REAL_EXPERIMENT_OPERATOR_CHECKLIST_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.real-experiment-operator-checklist.v1"
)
REAL_EXPERIMENT_OPERATOR_PROGRESS_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.real-experiment-operator-progress-report.v1"
)
REAL_EXPERIMENT_EXTERNAL_ARTIFACT_CONTRACTS_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.real-experiment-external-artifact-contracts.v1"
)
REAL_EXPERIMENT_LAUNCH_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.real-experiment-launch-report.v1"
)
REAL_EXPERIMENT_PRIMARY_EVIDENCE_STATUS_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.real-experiment-primary-evidence-status.v1"
)
REAL_EXPERIMENT_PRIMARY_EVIDENCE_REQUEST_PACKAGE_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.real-experiment-primary-evidence-request-package.v1"
)
REAL_EXPERIMENT_PRIMARY_EVIDENCE_RETURN_CHECKLIST_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.real-experiment-primary-evidence-return-checklist.v1"
)
REAL_EXPERIMENT_PRIMARY_EVIDENCE_RETURN_PROGRESS_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.real-experiment-primary-evidence-return-progress-report.v1"
)
REAL_EXPERIMENT_PRIMARY_EVIDENCE_ACCEPTANCE_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.real-experiment-primary-evidence-acceptance-report.v1"
)
REAL_EXPERIMENT_EXECUTION_PACKET_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.real-experiment-execution-packet.v1"
)
REAL_EXPERIMENT_EXECUTION_RECEIPT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.real-experiment-execution-receipt.v1"
)
REAL_EXPERIMENT_SMOKE_RUN_CHECKLIST_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.real-experiment-smoke-run-checklist.v1"
)
REAL_EXPERIMENT_SMOKE_RUN_RUNBOOK_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.real-experiment-smoke-run-runbook.v1"
)
REAL_EXPERIMENT_RESEARCH_REVIEW_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.real-experiment-research-review.v1"
)
REAL_EXPERIMENT_CLAIM_READINESS_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.real-experiment-claim-readiness.v1"
)
DEFAULT_CLAIM_MIN_EPISODE_COUNT = 3
DEFAULT_CLAIM_MIN_SCENE_COUNT = 1
DEFAULT_CLAIM_MIN_QA_COUNT = 30
DEFAULT_CLAIM_MIN_DYNAMIC_QA_COUNT = 1
CLAIM_SCALE_THRESHOLD_FIELDS = {
    "dynamic_qa_count": "min_dynamic_qa_count",
    "episode_count": "min_episode_count",
    "qa_count": "min_qa_count",
    "scene_count": "min_scene_count",
}
DEFAULT_REAL_HANDOFF_CONTROL_KINDS = (
    "caption_memory",
    "graph_text",
    "multi_frame_vlm",
    "vlm",
)
DEFAULT_REAL_HANDOFF_PREDICTED_INPUT_KINDS = ("observation_sequence",)
REAL_EXPERIMENT_ARTIFACT_TRACKS = (
    "real_data",
    "real_controls",
    "predicted_dsg",
    "review_artifacts",
    "run_outputs",
)
REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS = (
    "real_data",
    "real_controls",
    "predicted_dsg",
)
REAL_EXPERIMENT_RESEARCH_QUESTION_KEYS = (
    "dynamic_memory",
    "graph_tool_query",
    "interactive_task",
    "spatial_qa",
)
REAL_EXPERIMENT_CONCLUSIVE_VERDICTS = ("improved", "regressed", "unchanged")
CLAIM_RESEARCH_QUESTION_SOURCE_ARTIFACT_TYPES = {
    "dynamic_memory": "qa_eval_delta_report",
    "graph_tool_query": "qa_eval_delta_report",
    "interactive_task": "active_task_delta_report",
    "spatial_qa": "qa_eval_delta_report",
}
CLAIM_RESEARCH_QUESTION_TRACKS = {
    "dynamic_memory": ("real_controls", "predicted_dsg", "review_artifacts"),
    "graph_tool_query": ("real_controls", "predicted_dsg", "review_artifacts"),
    "interactive_task": ("predicted_dsg", "review_artifacts"),
    "spatial_qa": ("real_controls", "predicted_dsg", "review_artifacts"),
}
REAL_EXPERIMENT_ARTIFACT_GROUP_TRACKS = {
    "offline_controls": "real_controls",
    "predicted_dsg": "predicted_dsg",
    "real_collection": "real_data",
    "real_run": "run_outputs",
    "review_artifacts": "review_artifacts",
    "run_outputs": "run_outputs",
}


def real_experiment_external_artifact_contracts_digest(
    contracts: Mapping[str, Any],
) -> str:
    payload = {
        key: value for key, value in contracts.items() if key != "contracts_digest"
    }
    return hashlib.sha256(
        json.dumps(
            _json_value(payload),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def real_experiment_operator_checklist_digest(
    checklist: Mapping[str, Any],
) -> str:
    payload = {
        key: value
        for key, value in checklist.items()
        if key != "operator_checklist_digest"
    }
    return hashlib.sha256(
        json.dumps(
            _json_value(payload),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def real_experiment_operator_checklist_json(
    checklist: Mapping[str, Any],
) -> str:
    return json.dumps(_json_value(checklist), indent=2, sort_keys=True) + "\n"


def real_experiment_operator_progress_report_digest(
    report: Mapping[str, Any],
) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return hashlib.sha256(
        json.dumps(
            _json_value(payload),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def real_experiment_operator_progress_report_json(
    report: Mapping[str, Any],
) -> str:
    return json.dumps(_json_value(report), indent=2, sort_keys=True) + "\n"


def save_real_experiment_operator_progress_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        real_experiment_operator_progress_report_json(report),
        encoding="utf-8",
    )
    return output_path


def load_real_experiment_operator_progress_report(
    path: str | Path,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError(
            "Real experiment operator progress report JSON must be an object"
        )
    schema_version = payload.get("schema_version")
    if schema_version != REAL_EXPERIMENT_OPERATOR_PROGRESS_REPORT_SCHEMA_VERSION:
        raise SpatialQAError(
            "Unsupported real experiment operator progress report schema "
            f"version: {schema_version}"
        )
    return cast(dict[str, Any], payload)


def real_experiment_operator_progress_report(
    checklist: Mapping[str, Any],
    *,
    checklist_path: str | Path | None = None,
) -> dict[str, Any]:
    validation = validate_real_experiment_operator_checklist(checklist)
    target_paths = _operator_progress_target_paths(checklist)
    rows: list[dict[str, Any]] = []
    for step in _mapping_sequence(checklist.get("steps")):
        key = _text_field(step, "key")
        target_path = target_paths[key]
        target_state = _operator_progress_target_state(key, target_path)
        row = {
            "command": _text_field(step, "command"),
            "key": key,
            "order": _summary_int(step, "order") or 0,
            "phase": _text_field(step, "phase"),
            "target_path": str(target_path),
            "track": _text_field(step, "track"),
            **target_state,
        }
        rows.append(row)
    present_count = sum(1 for row in rows if row["target_exists"] is True)
    missing_count = len(rows) - present_count
    ready_count = sum(1 for row in rows if row["target_ready"] is True)
    not_ready_count = len(rows) - ready_count
    tracks = sorted({_text_field(row, "track") for row in rows})
    phase_order = _string_sequence_or_empty(checklist.get("phase_order"))
    next_missing_step = _operator_progress_next_missing_step(rows)
    next_not_ready_step = _operator_progress_next_not_ready_step(rows)
    report: dict[str, Any] = {
        "schema_version": REAL_EXPERIMENT_OPERATOR_PROGRESS_REPORT_SCHEMA_VERSION,
        "action": "real_experiment_operator_progress_report",
        "operator_checklist_path": (
            str(checklist_path) if checklist_path is not None else None
        ),
        "operator_checklist_digest": _text_or_none(
            checklist.get("operator_checklist_digest")
        ),
        "operator_checklist_validation": {
            "operator_checklist_digest": validation["operator_checklist_digest"],
            "valid": validation["valid"],
        },
        "next_missing_step": next_missing_step,
        "next_not_ready_step": next_not_ready_step,
        "summary": {
            "all_targets_present": missing_count == 0,
            "all_targets_ready": not_ready_count == 0,
            "missing_target_step_count": missing_count,
            "not_ready_target_step_count": not_ready_count,
            "phase_count": len(phase_order),
            "present_target_step_count": present_count,
            "ready_target_step_count": ready_count,
            "step_count": len(rows),
            "track_count": len(tracks),
        },
        "steps": rows,
        "track_summary": _operator_progress_track_summary(rows, tracks),
    }
    report["report_digest"] = real_experiment_operator_progress_report_digest(report)
    return report


def validate_real_experiment_operator_progress_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = report.get("schema_version")
    action = report.get("action")
    report_digest = _text_or_none(report.get("report_digest"))
    expected_digest = real_experiment_operator_progress_report_digest(report)
    operator_validation = _mapping(
        report.get("operator_checklist_validation"),
        "operator_checklist_validation",
    )
    summary = _mapping(report.get("summary"), "summary")
    steps = _mapping_sequence(report.get("steps"))
    track_summary = _mapping(report.get("track_summary"), "track_summary")
    present_count = sum(1 for row in steps if row.get("target_exists") is True)
    missing_count = len(steps) - present_count
    ready_count = sum(1 for row in steps if row.get("target_ready") is True)
    not_ready_count = len(steps) - ready_count
    tracks = sorted({_text_field(row, "track") for row in steps})
    phases = sorted({_text_field(row, "phase") for row in steps})
    expected_track_summary = _operator_progress_track_summary(steps, tracks)
    expected_next_missing_step = _operator_progress_next_missing_step(steps)
    expected_next_not_ready_step = _operator_progress_next_not_ready_step(steps)
    next_missing_step = report.get("next_missing_step")
    if next_missing_step is not None and not isinstance(next_missing_step, Mapping):
        raise SpatialQAError(
            "Real experiment operator progress next_missing_step must be an object"
        )
    next_not_ready_step = report.get("next_not_ready_step")
    if next_not_ready_step is not None and not isinstance(
        next_not_ready_step,
        Mapping,
    ):
        raise SpatialQAError(
            "Real experiment operator progress next_not_ready_step must be an object"
        )
    status_rows_consistent = all(
        _operator_progress_status_row_valid(row) for row in steps
    )
    checks = [
        {
            "name": "schema_version",
            "passed": (
                schema_version
                == REAL_EXPERIMENT_OPERATOR_PROGRESS_REPORT_SCHEMA_VERSION
            ),
            "expected": REAL_EXPERIMENT_OPERATOR_PROGRESS_REPORT_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "action",
            "passed": action == "real_experiment_operator_progress_report",
            "expected": "real_experiment_operator_progress_report",
            "actual": action,
        },
        {
            "name": "report_digest",
            "passed": report_digest == expected_digest,
            "expected": expected_digest,
            "actual": report_digest,
        },
        {
            "name": "operator_checklist_path_present",
            "passed": _text_or_none(report.get("operator_checklist_path")) is not None,
            "expected": True,
            "actual": (
                _text_or_none(report.get("operator_checklist_path")) is not None
            ),
        },
        {
            "name": "operator_checklist_validation_valid",
            "passed": operator_validation.get("valid") is True,
            "expected": True,
            "actual": operator_validation.get("valid"),
        },
        {
            "name": "step_count",
            "passed": _summary_int(summary, "step_count") == len(steps),
            "expected": len(steps),
            "actual": _summary_int(summary, "step_count"),
        },
        {
            "name": "target_counts",
            "passed": (
                _summary_int(summary, "present_target_step_count") == present_count
                and _summary_int(summary, "missing_target_step_count")
                == missing_count
            ),
            "expected": {
                "missing_target_step_count": missing_count,
                "present_target_step_count": present_count,
            },
            "actual": {
                "missing_target_step_count": _summary_int(
                    summary,
                    "missing_target_step_count",
                ),
                "present_target_step_count": _summary_int(
                    summary,
                    "present_target_step_count",
                ),
            },
        },
        {
            "name": "target_ready_counts",
            "passed": (
                _summary_int(summary, "ready_target_step_count") == ready_count
                and _summary_int(summary, "not_ready_target_step_count")
                == not_ready_count
            ),
            "expected": {
                "not_ready_target_step_count": not_ready_count,
                "ready_target_step_count": ready_count,
            },
            "actual": {
                "not_ready_target_step_count": _summary_int(
                    summary,
                    "not_ready_target_step_count",
                ),
                "ready_target_step_count": _summary_int(
                    summary,
                    "ready_target_step_count",
                ),
            },
        },
        {
            "name": "all_targets_present",
            "passed": summary.get("all_targets_present") is (missing_count == 0),
            "expected": missing_count == 0,
            "actual": summary.get("all_targets_present"),
        },
        {
            "name": "all_targets_ready",
            "passed": summary.get("all_targets_ready") is (not_ready_count == 0),
            "expected": not_ready_count == 0,
            "actual": summary.get("all_targets_ready"),
        },
        {
            "name": "phase_count",
            "passed": _summary_int(summary, "phase_count") == len(phases),
            "expected": len(phases),
            "actual": _summary_int(summary, "phase_count"),
        },
        {
            "name": "track_count",
            "passed": _summary_int(summary, "track_count") == len(tracks),
            "expected": len(tracks),
            "actual": _summary_int(summary, "track_count"),
        },
        {
            "name": "step_status_rows",
            "passed": status_rows_consistent,
            "expected": True,
            "actual": status_rows_consistent,
        },
        {
            "name": "next_missing_step",
            "passed": _json_value(next_missing_step)
            == _json_value(expected_next_missing_step),
            "expected": _json_value(expected_next_missing_step),
            "actual": _json_value(next_missing_step),
        },
        {
            "name": "next_not_ready_step",
            "passed": _json_value(next_not_ready_step)
            == _json_value(expected_next_not_ready_step),
            "expected": _json_value(expected_next_not_ready_step),
            "actual": _json_value(next_not_ready_step),
        },
        {
            "name": "track_summary",
            "passed": _json_value(track_summary)
            == _json_value(expected_track_summary),
            "expected": _json_value(expected_track_summary),
            "actual": _json_value(track_summary),
        },
    ]
    return {
        "action": "validate_real_experiment_operator_progress_report",
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "report_digest": report_digest,
        "checks": checks,
    }


def compare_real_experiment_operator_progress_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_real_experiment_operator_progress_report(report)
    operator_checklist_path = _text_field(report, "operator_checklist_path")
    current = real_experiment_operator_progress_report(
        load_real_experiment_operator_checklist(operator_checklist_path),
        checklist_path=operator_checklist_path,
    )
    saved_digest = _text_or_none(report.get("report_digest"))
    current_digest = _text_or_none(current.get("report_digest"))
    checks = [
        {
            "name": "operator_progress_report_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        {
            "name": "operator_progress_report_digest_matches_current",
            "passed": saved_digest == current_digest,
            "expected": saved_digest,
            "actual": current_digest,
        },
        {
            "name": "operator_progress_report_payload_matches_current",
            "passed": _json_value(report) == _json_value(current),
            "expected": _json_value(report),
            "actual": _json_value(current),
        },
    ]
    return {
        "action": "compare_real_experiment_operator_progress_report",
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def _operator_progress_target_state(
    key: str,
    target_path: Path,
) -> dict[str, Any]:
    if key == "write_primary_evidence_request_bundles":
        return _operator_progress_request_bundles_target_state(target_path.parent)
    target_exists = target_path.exists()
    if not target_exists:
        return {
            "target_exists": False,
            "target_ready": False,
            "target_status": "missing",
        }
    if key == "real_collection_report":
        return _operator_progress_validation_target_state(
            target_path,
            load_real_collection_report,
            validate_real_collection_report,
            "validate_real_collection_report",
        )
    if key == "offline_control_prediction_receipt_bundle":
        return _operator_progress_validation_target_state(
            target_path,
            load_offline_control_prediction_receipt_bundle,
            validate_offline_control_prediction_receipt_bundle,
            "validate_offline_control_prediction_receipt_bundle",
        )
    if key == "predicted_dsg_detector_receipt_bundle":
        return _operator_progress_validation_target_state(
            target_path,
            load_predicted_dsg_detector_receipt_bundle,
            validate_predicted_dsg_detector_receipt_bundle,
            "validate_predicted_dsg_detector_receipt_bundle",
        )
    if key == "validate_external_artifact_contracts":
        try:
            validation = validate_real_experiment_external_artifact_contracts(
                load_real_experiment_external_artifact_contracts(target_path)
            )
            valid = validation.get("valid") is True
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            return {
                "target_audit": {
                    "checked": True,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                "target_exists": True,
                "target_ready": False,
                "target_status": "invalid",
            }
        return {
            "target_audit": {
                "action": validation.get("action"),
                "checked": True,
                "valid": valid,
            },
            "target_exists": True,
            "target_ready": valid,
            "target_status": "ready" if valid else "invalid",
        }
    if key == "compare_external_artifact_contracts":
        try:
            comparison = compare_real_experiment_external_artifact_contracts(
                load_real_experiment_external_artifact_contracts(target_path)
            )
            matches = comparison.get("matches") is True
            comparison_validation = _mapping_or_empty(comparison.get("validation"))
            target_status = (
                "ready"
                if matches
                else "invalid"
                if comparison_validation.get("valid") is not True
                else "stale"
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            return {
                "target_audit": {
                    "checked": True,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                "target_exists": True,
                "target_ready": False,
                "target_status": "invalid",
            }
        return {
            "target_audit": {
                "action": comparison.get("action"),
                "checked": True,
                "matches": matches,
            },
            "target_exists": True,
            "target_ready": matches,
            "target_status": target_status,
        }
    if key == "validate_external_artifact_launch_report":
        try:
            validation = validate_real_experiment_external_artifact_launch_report(
                load_real_experiment_external_artifact_launch_report(target_path)
            )
            valid = validation.get("valid") is True
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            return {
                "target_audit": {
                    "checked": True,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                "target_exists": True,
                "target_ready": False,
                "target_status": "invalid",
            }
        return {
            "target_audit": {
                "action": validation.get("action"),
                "checked": True,
                "valid": valid,
            },
            "target_exists": True,
            "target_ready": valid,
            "target_status": "ready" if valid else "invalid",
        }
    if key == "compare_external_artifact_launch_report":
        try:
            comparison = compare_real_experiment_external_artifact_launch_report(
                load_real_experiment_external_artifact_launch_report(target_path)
            )
            matches = comparison.get("matches") is True
            comparison_validation = _mapping_or_empty(comparison.get("validation"))
            target_status = (
                "ready"
                if matches
                else "invalid"
                if comparison_validation.get("valid") is not True
                else "stale"
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            return {
                "target_audit": {
                    "checked": True,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                "target_exists": True,
                "target_ready": False,
                "target_status": "invalid",
            }
        return {
            "target_audit": {
                "action": comparison.get("action"),
                "checked": True,
                "matches": matches,
            },
            "target_exists": True,
            "target_ready": matches,
            "target_status": target_status,
        }
    if key == "validate_primary_evidence_status":
        try:
            validation = validate_real_experiment_primary_evidence_status(
                load_real_experiment_primary_evidence_status(target_path)
            )
            valid = validation.get("valid") is True
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            return {
                "target_audit": {
                    "checked": True,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                "target_exists": True,
                "target_ready": False,
                "target_status": "invalid",
            }
        return {
            "target_audit": {
                "action": validation.get("action"),
                "checked": True,
                "valid": valid,
            },
            "target_exists": True,
            "target_ready": valid,
            "target_status": "ready" if valid else "invalid",
        }
    if key == "compare_primary_evidence_status":
        try:
            comparison = compare_real_experiment_primary_evidence_status(
                load_real_experiment_primary_evidence_status(target_path)
            )
            matches = comparison.get("matches") is True
            comparison_validation = _mapping_or_empty(comparison.get("validation"))
            target_status = (
                "ready"
                if matches
                else "invalid"
                if comparison_validation.get("valid") is not True
                else "stale"
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            return {
                "target_audit": {
                    "checked": True,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                "target_exists": True,
                "target_ready": False,
                "target_status": "invalid",
            }
        return {
            "target_audit": {
                "action": comparison.get("action"),
                "checked": True,
                "matches": matches,
            },
            "target_exists": True,
            "target_ready": matches,
            "target_status": target_status,
        }
    if key == "validate_primary_evidence_request_package":
        try:
            validation = validate_real_experiment_primary_evidence_request_package(
                load_real_experiment_primary_evidence_request_package(target_path)
            )
            valid = validation.get("valid") is True
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            return {
                "target_audit": {
                    "checked": True,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                "target_exists": True,
                "target_ready": False,
                "target_status": "invalid",
            }
        return {
            "target_audit": {
                "action": validation.get("action"),
                "checked": True,
                "valid": valid,
            },
            "target_exists": True,
            "target_ready": valid,
            "target_status": "ready" if valid else "invalid",
        }
    if key == "compare_primary_evidence_request_package":
        try:
            comparison = compare_real_experiment_primary_evidence_request_package(
                load_real_experiment_primary_evidence_request_package(target_path)
            )
            matches = comparison.get("matches") is True
            comparison_validation = _mapping_or_empty(comparison.get("validation"))
            target_status = (
                "ready"
                if matches
                else "invalid"
                if comparison_validation.get("valid") is not True
                else "stale"
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            return {
                "target_audit": {
                    "checked": True,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                "target_exists": True,
                "target_ready": False,
                "target_status": "invalid",
            }
        return {
            "target_audit": {
                "action": comparison.get("action"),
                "checked": True,
                "matches": matches,
            },
            "target_exists": True,
            "target_ready": matches,
            "target_status": target_status,
        }
    if key == "validate_primary_evidence_return_checklist":
        try:
            validation = validate_real_experiment_primary_evidence_return_checklist(
                load_real_experiment_primary_evidence_return_checklist(target_path)
            )
            valid = validation.get("valid") is True
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            return {
                "target_audit": {
                    "checked": True,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                "target_exists": True,
                "target_ready": False,
                "target_status": "invalid",
            }
        return {
            "target_audit": {
                "action": validation.get("action"),
                "checked": True,
                "valid": valid,
            },
            "target_exists": True,
            "target_ready": valid,
            "target_status": "ready" if valid else "invalid",
        }
    if key == "compare_primary_evidence_return_checklist":
        try:
            comparison = compare_real_experiment_primary_evidence_return_checklist(
                load_real_experiment_primary_evidence_return_checklist(target_path)
            )
            matches = comparison.get("matches") is True
            comparison_validation = _mapping_or_empty(comparison.get("validation"))
            target_status = (
                "ready"
                if matches
                else "invalid"
                if comparison_validation.get("valid") is not True
                else "stale"
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            return {
                "target_audit": {
                    "checked": True,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                "target_exists": True,
                "target_ready": False,
                "target_status": "invalid",
            }
        return {
            "target_audit": {
                "action": comparison.get("action"),
                "checked": True,
                "matches": matches,
            },
            "target_exists": True,
            "target_ready": matches,
            "target_status": target_status,
        }
    if key == "validate_primary_evidence_return_progress_report":
        try:
            validation = (
                validate_real_experiment_primary_evidence_return_progress_report(
                    load_real_experiment_primary_evidence_return_progress_report(
                        target_path
                    )
                )
            )
            valid = validation.get("valid") is True
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            return {
                "target_audit": {
                    "checked": True,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                "target_exists": True,
                "target_ready": False,
                "target_status": "invalid",
            }
        return {
            "target_audit": {
                "action": validation.get("action"),
                "checked": True,
                "valid": valid,
            },
            "target_exists": True,
            "target_ready": valid,
            "target_status": "ready" if valid else "invalid",
        }
    if key == "compare_primary_evidence_return_progress_report":
        try:
            comparison = (
                compare_real_experiment_primary_evidence_return_progress_report(
                    load_real_experiment_primary_evidence_return_progress_report(
                        target_path
                    )
                )
            )
            matches = comparison.get("matches") is True
            comparison_validation = _mapping_or_empty(comparison.get("validation"))
            target_status = (
                "ready"
                if matches
                else "invalid"
                if comparison_validation.get("valid") is not True
                else "stale"
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            return {
                "target_audit": {
                    "checked": True,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                "target_exists": True,
                "target_ready": False,
                "target_status": "invalid",
            }
        return {
            "target_audit": {
                "action": comparison.get("action"),
                "checked": True,
                "matches": matches,
            },
            "target_exists": True,
            "target_ready": matches,
            "target_status": target_status,
        }
    if key == "validate_primary_evidence_acceptance_report":
        try:
            validation = validate_real_experiment_primary_evidence_acceptance_report(
                load_real_experiment_primary_evidence_acceptance_report(target_path)
            )
            valid = validation.get("valid") is True
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            return {
                "target_audit": {
                    "checked": True,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                "target_exists": True,
                "target_ready": False,
                "target_status": "invalid",
            }
        return {
            "target_audit": {
                "action": validation.get("action"),
                "checked": True,
                "valid": valid,
            },
            "target_exists": True,
            "target_ready": valid,
            "target_status": "ready" if valid else "invalid",
        }
    if key == "compare_primary_evidence_acceptance_report":
        try:
            comparison = compare_real_experiment_primary_evidence_acceptance_report(
                load_real_experiment_primary_evidence_acceptance_report(target_path)
            )
            matches = comparison.get("matches") is True
            comparison_validation = _mapping_or_empty(comparison.get("validation"))
            target_status = (
                "ready"
                if matches
                else "invalid"
                if comparison_validation.get("valid") is not True
                else "stale"
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            return {
                "target_audit": {
                    "checked": True,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                "target_exists": True,
                "target_ready": False,
                "target_status": "invalid",
            }
        return {
            "target_audit": {
                "action": comparison.get("action"),
                "checked": True,
                "matches": matches,
            },
            "target_exists": True,
            "target_ready": matches,
            "target_status": target_status,
        }
    if key == "validate_smoke_run_checklist":
        try:
            validation = validate_real_experiment_smoke_run_checklist(
                load_real_experiment_smoke_run_checklist(target_path)
            )
            valid = validation.get("valid") is True
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            return {
                "target_audit": {
                    "checked": True,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                "target_exists": True,
                "target_ready": False,
                "target_status": "invalid",
            }
        return {
            "target_audit": {
                "action": validation.get("action"),
                "checked": True,
                "valid": valid,
            },
            "target_exists": True,
            "target_ready": valid,
            "target_status": "ready" if valid else "invalid",
        }
    if key == "compare_smoke_run_checklist":
        try:
            comparison = compare_real_experiment_smoke_run_checklist(
                load_real_experiment_smoke_run_checklist(target_path)
            )
            matches = comparison.get("matches") is True
            comparison_validation = _mapping_or_empty(comparison.get("validation"))
            target_status = (
                "ready"
                if matches
                else "invalid"
                if comparison_validation.get("valid") is not True
                else "stale"
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            return {
                "target_audit": {
                    "checked": True,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                "target_exists": True,
                "target_ready": False,
                "target_status": "invalid",
            }
        return {
            "target_audit": {
                "action": comparison.get("action"),
                "checked": True,
                "matches": matches,
            },
            "target_exists": True,
            "target_ready": matches,
            "target_status": target_status,
        }
    if key == "validate_smoke_run_runbook":
        try:
            validation = validate_real_experiment_smoke_run_runbook(
                load_real_experiment_smoke_run_runbook(target_path)
            )
            valid = validation.get("valid") is True
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            return {
                "target_audit": {
                    "checked": True,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                "target_exists": True,
                "target_ready": False,
                "target_status": "invalid",
            }
        return {
            "target_audit": {
                "action": validation.get("action"),
                "checked": True,
                "valid": valid,
            },
            "target_exists": True,
            "target_ready": valid,
            "target_status": "ready" if valid else "invalid",
        }
    if key == "compare_smoke_run_runbook":
        try:
            comparison = compare_real_experiment_smoke_run_runbook(
                load_real_experiment_smoke_run_runbook(target_path)
            )
            matches = comparison.get("matches") is True
            comparison_validation = _mapping_or_empty(comparison.get("validation"))
            target_status = (
                "ready"
                if matches
                else "invalid"
                if comparison_validation.get("valid") is not True
                else "stale"
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            return {
                "target_audit": {
                    "checked": True,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                "target_exists": True,
                "target_ready": False,
                "target_status": "invalid",
            }
        return {
            "target_audit": {
                "action": comparison.get("action"),
                "checked": True,
                "matches": matches,
            },
            "target_exists": True,
            "target_ready": matches,
            "target_status": target_status,
        }
    if key == "validate_run_ledger":
        try:
            validation = validate_real_experiment_run_ledger(
                load_real_experiment_run_ledger(target_path)
            )
            valid = validation.get("valid") is True
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            return {
                "target_audit": {
                    "checked": True,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                "target_exists": True,
                "target_ready": False,
                "target_status": "invalid",
            }
        return {
            "target_audit": {
                "action": validation.get("action"),
                "checked": True,
                "valid": valid,
            },
            "target_exists": True,
            "target_ready": valid,
            "target_status": "ready" if valid else "invalid",
        }
    if key == "compare_run_ledger":
        try:
            comparison = compare_real_experiment_run_ledger(
                load_real_experiment_run_ledger(target_path)
            )
            matches = comparison.get("matches") is True
            comparison_validation = _mapping_or_empty(comparison.get("validation"))
            target_status = (
                "ready"
                if matches
                else "invalid"
                if comparison_validation.get("valid") is not True
                else "stale"
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            return {
                "target_audit": {
                    "checked": True,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                "target_exists": True,
                "target_ready": False,
                "target_status": "invalid",
            }
        return {
            "target_audit": {
                "action": comparison.get("action"),
                "checked": True,
                "matches": matches,
            },
            "target_exists": True,
            "target_ready": matches,
            "target_status": target_status,
        }
    if key == "validate_execution_receipt":
        try:
            validation = validate_real_experiment_execution_receipt(
                load_real_experiment_execution_receipt(target_path)
            )
            valid = validation.get("valid") is True
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            return {
                "target_audit": {
                    "checked": True,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                "target_exists": True,
                "target_ready": False,
                "target_status": "invalid",
            }
        return {
            "target_audit": {
                "action": validation.get("action"),
                "checked": True,
                "valid": valid,
            },
            "target_exists": True,
            "target_ready": valid,
            "target_status": "ready" if valid else "invalid",
        }
    if key == "compare_execution_receipt":
        try:
            comparison = compare_real_experiment_execution_receipt(
                load_real_experiment_execution_receipt(target_path)
            )
            matches = comparison.get("matches") is True
            comparison_validation = _mapping_or_empty(comparison.get("validation"))
            target_status = (
                "ready"
                if matches
                else "invalid"
                if comparison_validation.get("valid") is not True
                else "stale"
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            return {
                "target_audit": {
                    "checked": True,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                "target_exists": True,
                "target_ready": False,
                "target_status": "invalid",
            }
        return {
            "target_audit": {
                "action": comparison.get("action"),
                "checked": True,
                "matches": matches,
            },
            "target_exists": True,
            "target_ready": matches,
            "target_status": target_status,
        }
    if key == "validate_research_review":
        try:
            validation = validate_real_experiment_research_review(
                load_real_experiment_research_review(target_path)
            )
            valid = validation.get("valid") is True
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            return {
                "target_audit": {
                    "checked": True,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                "target_exists": True,
                "target_ready": False,
                "target_status": "invalid",
            }
        return {
            "target_audit": {
                "action": validation.get("action"),
                "checked": True,
                "valid": valid,
            },
            "target_exists": True,
            "target_ready": valid,
            "target_status": "ready" if valid else "invalid",
        }
    if key == "compare_research_review":
        try:
            comparison = compare_real_experiment_research_review(
                load_real_experiment_research_review(target_path)
            )
            matches = comparison.get("matches") is True
            comparison_validation = _mapping_or_empty(comparison.get("validation"))
            target_status = (
                "ready"
                if matches
                else "invalid"
                if comparison_validation.get("valid") is not True
                else "stale"
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            return {
                "target_audit": {
                    "checked": True,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                "target_exists": True,
                "target_ready": False,
                "target_status": "invalid",
            }
        return {
            "target_audit": {
                "action": comparison.get("action"),
                "checked": True,
                "matches": matches,
            },
            "target_exists": True,
            "target_ready": matches,
            "target_status": target_status,
        }
    if key == "validate_claim_readiness":
        try:
            validation = validate_real_experiment_claim_readiness(
                load_real_experiment_claim_readiness(target_path)
            )
            valid = validation.get("valid") is True
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            return {
                "target_audit": {
                    "checked": True,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                "target_exists": True,
                "target_ready": False,
                "target_status": "invalid",
            }
        return {
            "target_audit": {
                "action": validation.get("action"),
                "checked": True,
                "valid": valid,
            },
            "target_exists": True,
            "target_ready": valid,
            "target_status": "ready" if valid else "invalid",
        }
    if key == "compare_claim_readiness":
        try:
            comparison = compare_real_experiment_claim_readiness(
                load_real_experiment_claim_readiness(target_path)
            )
            matches = comparison.get("matches") is True
            comparison_validation = _mapping_or_empty(comparison.get("validation"))
            target_status = (
                "ready"
                if matches
                else "invalid"
                if comparison_validation.get("valid") is not True
                else "stale"
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            return {
                "target_audit": {
                    "checked": True,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                "target_exists": True,
                "target_ready": False,
                "target_status": "invalid",
            }
        return {
            "target_audit": {
                "action": comparison.get("action"),
                "checked": True,
                "matches": matches,
            },
            "target_exists": True,
            "target_ready": matches,
            "target_status": target_status,
        }
    return {
        "target_exists": True,
        "target_ready": True,
        "target_status": "present",
    }


def _operator_progress_validation_target_state(
    target_path: Path,
    load_artifact: Any,
    validate_artifact: Any,
    action: str,
) -> dict[str, Any]:
    try:
        validation = validate_artifact(load_artifact(target_path))
        valid = validation.get("valid") is True
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        return {
            "target_audit": {
                "checked": True,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
            "target_exists": True,
            "target_ready": False,
            "target_status": "invalid",
        }
    return {
        "target_audit": {
            "action": _text_or_none(validation.get("action")) or action,
            "checked": True,
            "valid": valid,
        },
        "target_exists": True,
        "target_ready": valid,
        "target_status": "ready" if valid else "invalid",
    }


def _operator_progress_request_bundles_target_state(root: Path) -> dict[str, Any]:
    specs = (
        (
            "real_data",
            root / "real-collection-request-bundle.json",
            load_real_collection_request_bundle,
        ),
        (
            "real_controls",
            root / "offline-control-prediction-request-bundle.json",
            load_offline_control_prediction_request_bundle,
        ),
        (
            "predicted_dsg",
            root / "predicted-dsg-detector-request-bundle.json",
            load_predicted_dsg_detector_request_bundle,
        ),
    )
    track_errors: dict[str, dict[str, str]] = {}
    track_statuses: dict[str, str] = {}
    for track, path, load_bundle in specs:
        if not path.exists():
            track_statuses[track] = "missing"
            continue
        try:
            validation = _primary_evidence_request_bundle_validation(
                track,
                load_bundle(path),
            )
            track_statuses[track] = (
                "ready" if validation.get("valid") is True else "invalid"
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            track_statuses[track] = "invalid"
            track_errors[track] = {
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
    all_present = all(status != "missing" for status in track_statuses.values())
    all_ready = all(status == "ready" for status in track_statuses.values())
    if not all_present:
        target_status = "missing"
    elif all_ready:
        target_status = "ready"
    else:
        target_status = "invalid"
    target_audit: dict[str, Any] = {
        "action": "validate_real_experiment_primary_evidence_request_bundles",
        "checked": True,
        "track_statuses": track_statuses,
        "valid": all_ready,
    }
    if track_errors:
        target_audit["track_errors"] = track_errors
    return {
        "target_audit": target_audit,
        "target_exists": all_present,
        "target_ready": all_ready,
        "target_status": target_status,
    }


def _operator_progress_status_row_valid(row: Mapping[str, Any]) -> bool:
    target_exists = row.get("target_exists")
    target_ready = row.get("target_ready")
    target_status = _text_or_none(row.get("target_status"))
    status_allowed = target_status in {"invalid", "missing", "present", "ready", "stale"}
    status_consistent = (
        isinstance(target_exists, bool)
        and isinstance(target_ready, bool)
        and status_allowed
        and (
            (
                target_exists is False
                and target_ready is False
                and target_status == "missing"
            )
            or (
                target_exists is True
                and (
                    (
                        target_ready is True
                        and target_status in {"present", "ready"}
                    )
                    or (
                        target_ready is False
                        and target_status in {"invalid", "stale"}
                    )
                )
            )
        )
    )
    audit = row.get("target_audit")
    audit_consistent = (
        audit is None
        or (
            isinstance(audit, Mapping)
            and audit.get("checked") is True
            and (
                _text_or_none(audit.get("action")) is not None
                or _text_or_none(audit.get("error")) is not None
            )
        )
    )
    return (
        status_consistent
        and audit_consistent
        and _text_or_none(row.get("command")) is not None
        and _text_or_none(row.get("key")) is not None
        and _text_or_none(row.get("target_path")) is not None
        and _text_or_none(row.get("phase")) is not None
        and _text_or_none(row.get("track")) is not None
    )


def load_real_experiment_operator_checklist(
    path: str | Path,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError(
            "Real experiment operator checklist JSON must be an object"
        )
    return cast(dict[str, Any], payload)


def validate_real_experiment_operator_checklist(
    checklist: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = checklist.get("schema_version")
    action = checklist.get("action")
    checklist_digest = _text_or_none(checklist.get("operator_checklist_digest"))
    expected_digest = real_experiment_operator_checklist_digest(checklist)
    summary = _mapping(checklist.get("summary"), "summary")
    phase_order = _string_sequence_or_empty(checklist.get("phase_order"))
    steps = _mapping_sequence(checklist.get("steps"))
    orders = [_summary_int(step, "order") for step in steps]
    step_keys = [_text_field(step, "key") for step in steps]
    tracks = {
        _text_field(step, "track")
        for step in steps
    }
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == REAL_EXPERIMENT_OPERATOR_CHECKLIST_SCHEMA_VERSION,
            "expected": REAL_EXPERIMENT_OPERATOR_CHECKLIST_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "action",
            "passed": action == "real_experiment_operator_checklist",
            "expected": "real_experiment_operator_checklist",
            "actual": action,
        },
        {
            "name": "operator_checklist_digest",
            "passed": checklist_digest == expected_digest,
            "expected": expected_digest,
            "actual": checklist_digest,
        },
        {
            "name": "path_fields_present",
            "passed": (
                _text_or_none(checklist.get("root")) is not None
                and _text_or_none(checklist.get("run_manifest_path")) is not None
                and _text_or_none(
                    checklist.get("external_artifact_contracts_path")
                )
                is not None
            ),
            "expected": True,
            "actual": {
                "external_artifact_contracts_path": _text_or_none(
                    checklist.get("external_artifact_contracts_path")
                ),
                "root": _text_or_none(checklist.get("root")),
                "run_manifest_path": _text_or_none(
                    checklist.get("run_manifest_path")
                ),
            },
        },
        {
            "name": "phase_count",
            "passed": _summary_int(summary, "phase_count") == len(phase_order),
            "expected": len(phase_order),
            "actual": _summary_int(summary, "phase_count"),
        },
        {
            "name": "step_count",
            "passed": (
                _summary_int(checklist, "step_count") == len(steps)
                and _summary_int(summary, "step_count") == len(steps)
            ),
            "expected": len(steps),
            "actual": {
                "root": _summary_int(checklist, "step_count"),
                "summary": _summary_int(summary, "step_count"),
            },
        },
        {
            "name": "track_count",
            "passed": _summary_int(summary, "track_count") == len(tracks),
            "expected": len(tracks),
            "actual": _summary_int(summary, "track_count"),
        },
        {
            "name": "orders_are_consecutive",
            "passed": orders == list(range(1, len(steps) + 1)),
            "expected": list(range(1, len(steps) + 1)),
            "actual": orders,
        },
        {
            "name": "first_step_key",
            "passed": step_keys[:1] == ["validate_external_artifact_contracts"],
            "expected": "validate_external_artifact_contracts",
            "actual": step_keys[:1],
        },
        {
            "name": "last_step_key",
            "passed": step_keys[-1:] == ["compare_claim_readiness"],
            "expected": "compare_claim_readiness",
            "actual": step_keys[-1:],
        },
    ]
    return {
        "action": "validate_real_experiment_operator_checklist",
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "operator_checklist_digest": checklist_digest,
        "checks": checks,
    }


def compare_real_experiment_operator_checklist(
    checklist: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_real_experiment_operator_checklist(checklist)
    current = _current_handoff_operator_checklist(checklist)
    saved_digest = _text_or_none(checklist.get("operator_checklist_digest"))
    current_digest = _text_or_none(current.get("operator_checklist_digest"))
    checks = [
        {
            "name": "operator_checklist_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        {
            "name": "operator_checklist_digest_matches_current",
            "passed": saved_digest == current_digest,
            "expected": saved_digest,
            "actual": current_digest,
        },
        {
            "name": "operator_checklist_payload_matches_current",
            "passed": _json_value(checklist) == _json_value(current),
            "expected": _json_value(checklist),
            "actual": _json_value(current),
        },
    ]
    return {
        "action": "compare_real_experiment_operator_checklist",
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def real_experiment_external_artifact_contracts_json(
    contracts: Mapping[str, Any],
) -> str:
    return json.dumps(_json_value(contracts), indent=2, sort_keys=True) + "\n"


def save_real_experiment_external_artifact_contracts(
    contracts: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        real_experiment_external_artifact_contracts_json(contracts),
        encoding="utf-8",
    )
    return output_path


def load_real_experiment_external_artifact_contracts(
    path: str | Path,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError(
            "Real experiment external artifact contracts JSON must be an object"
        )
    return cast(dict[str, Any], payload)


def validate_real_experiment_external_artifact_contracts(
    contracts: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = contracts.get("schema_version")
    contracts_digest = _text_or_none(contracts.get("contracts_digest"))
    expected_digest = real_experiment_external_artifact_contracts_digest(contracts)
    summary = _mapping(contracts.get("summary"), "summary")
    track_summary = _mapping(contracts.get("track_summary"), "track_summary")
    tracks = _mapping(contracts.get("tracks"), "tracks")
    real_controls = _mapping(tracks.get("real_controls"), "real_controls")
    sources = _mapping_sequence(real_controls.get("sources"))
    missing_input_count = _track_summary_count(
        track_summary,
        "missing_input_artifact_count",
    )
    planned_output_count = _track_summary_count(
        track_summary,
        "planned_output_artifact_count",
    )
    required_input_count = _track_summary_count(
        track_summary,
        "input_artifact_count",
    )
    checks = [
        {
            "name": "schema_version",
            "passed": (
                schema_version
                == REAL_EXPERIMENT_EXTERNAL_ARTIFACT_CONTRACTS_SCHEMA_VERSION
            ),
            "expected": REAL_EXPERIMENT_EXTERNAL_ARTIFACT_CONTRACTS_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "contracts_digest",
            "passed": contracts_digest == expected_digest,
            "expected": expected_digest,
            "actual": contracts_digest,
        },
        {
            "name": "required_tracks_present",
            "passed": set(tracks) == set(REAL_EXPERIMENT_ARTIFACT_TRACKS),
            "expected": sorted(REAL_EXPERIMENT_ARTIFACT_TRACKS),
            "actual": sorted(str(track) for track in tracks),
        },
        {
            "name": "track_count",
            "passed": _summary_int(summary, "track_count") == len(track_summary),
            "expected": len(track_summary),
            "actual": _summary_int(summary, "track_count"),
        },
        {
            "name": "real_control_source_count",
            "passed": _summary_int(summary, "real_control_source_count")
            == len(sources),
            "expected": len(sources),
            "actual": _summary_int(summary, "real_control_source_count"),
        },
        {
            "name": "required_input_artifact_count",
            "passed": _summary_int(summary, "required_input_artifact_count")
            == required_input_count,
            "expected": required_input_count,
            "actual": _summary_int(summary, "required_input_artifact_count"),
        },
        {
            "name": "missing_input_artifact_count",
            "passed": _summary_int(summary, "missing_input_artifact_count")
            == missing_input_count,
            "expected": missing_input_count,
            "actual": _summary_int(summary, "missing_input_artifact_count"),
        },
        {
            "name": "planned_output_artifact_count",
            "passed": _summary_int(summary, "planned_output_artifact_count")
            == planned_output_count,
            "expected": planned_output_count,
            "actual": _summary_int(summary, "planned_output_artifact_count"),
        },
    ]
    return {
        "action": "validate_real_experiment_external_artifact_contracts",
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "contracts_digest": contracts_digest,
        "checks": checks,
    }


def compare_real_experiment_external_artifact_contracts(
    contracts: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_real_experiment_external_artifact_contracts(contracts)
    current = _current_external_artifact_contracts(contracts)
    saved_digest = _text_or_none(contracts.get("contracts_digest"))
    current_digest = _text_or_none(current.get("contracts_digest"))
    checks = [
        {
            "name": "contracts_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        {
            "name": "contracts_digest_matches_current",
            "passed": saved_digest == current_digest,
            "expected": saved_digest,
            "actual": current_digest,
        },
        {
            "name": "contracts_payload_matches_current",
            "passed": _json_value(contracts) == _json_value(current),
            "expected": _json_value(contracts),
            "actual": _json_value(current),
        },
    ]
    return {
        "action": "compare_real_experiment_external_artifact_contracts",
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def real_experiment_external_artifact_launch_report_digest(
    report: Mapping[str, Any],
) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return hashlib.sha256(
        json.dumps(
            _json_value(payload),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def real_experiment_external_artifact_launch_report_json(
    report: Mapping[str, Any],
) -> str:
    return json.dumps(_json_value(report), indent=2, sort_keys=True) + "\n"


def save_real_experiment_external_artifact_launch_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        real_experiment_external_artifact_launch_report_json(report),
        encoding="utf-8",
    )
    return output_path


def load_real_experiment_external_artifact_launch_report(
    path: str | Path,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError(
            "Real experiment external artifact launch report JSON must be an object"
        )
    schema_version = payload.get("schema_version")
    if schema_version != REAL_EXPERIMENT_LAUNCH_REPORT_SCHEMA_VERSION:
        raise SpatialQAError(
            "Unsupported real experiment external artifact launch report schema "
            f"version: {schema_version}"
        )
    return cast(dict[str, Any], payload)


def validate_real_experiment_external_artifact_launch_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = report.get("schema_version")
    action = report.get("action")
    report_digest = _text_or_none(report.get("report_digest"))
    expected_digest = real_experiment_external_artifact_launch_report_digest(report)
    tracks = _mapping(report.get("tracks"), "tracks")
    summary = _mapping(report.get("summary"), "summary")
    primary_gate = _mapping(
        report.get("primary_evidence_receipt_gate"),
        "primary_evidence_receipt_gate",
    )
    ready_to_run = report.get("ready_to_run")
    preflight_ready = report.get("preflight_ready_to_run")
    primary_ready = primary_gate.get("ready")
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == REAL_EXPERIMENT_LAUNCH_REPORT_SCHEMA_VERSION,
            "expected": REAL_EXPERIMENT_LAUNCH_REPORT_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "action",
            "passed": action == "real_experiment_external_artifact_launch_report",
            "expected": "real_experiment_external_artifact_launch_report",
            "actual": action,
        },
        {
            "name": "report_digest",
            "passed": report_digest == expected_digest,
            "expected": expected_digest,
            "actual": report_digest,
        },
        {
            "name": "contracts_path_present",
            "passed": _text_or_none(report.get("contracts_path")) is not None,
            "expected": True,
            "actual": _text_or_none(report.get("contracts_path")) is not None,
        },
        {
            "name": "run_manifest_path_present",
            "passed": _text_or_none(report.get("run_manifest_path")) is not None,
            "expected": True,
            "actual": _text_or_none(report.get("run_manifest_path")) is not None,
        },
        {
            "name": "track_count",
            "passed": _summary_int(summary, "track_count") == len(tracks),
            "expected": len(tracks),
            "actual": _summary_int(summary, "track_count"),
        },
        {
            "name": "ready_to_run_bool",
            "passed": isinstance(ready_to_run, bool),
            "expected": "bool",
            "actual": type(ready_to_run).__name__,
        },
        {
            "name": "preflight_ready_to_run_bool",
            "passed": isinstance(preflight_ready, bool),
            "expected": "bool",
            "actual": type(preflight_ready).__name__,
        },
        {
            "name": "primary_evidence_receipt_gate_bool",
            "passed": isinstance(primary_ready, bool),
            "expected": "bool",
            "actual": type(primary_ready).__name__,
        },
        {
            "name": "ready_to_run_receipt_gate_consistent",
            "passed": (
                ready_to_run is False
                or (preflight_ready is True and primary_ready is True)
            ),
            "expected": True,
            "actual": {
                "preflight_ready_to_run": preflight_ready,
                "primary_evidence_receipt_gate_ready": primary_ready,
                "ready_to_run": ready_to_run,
            },
        },
    ]
    return {
        "action": "validate_real_experiment_external_artifact_launch_report",
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "report_digest": report_digest,
        "checks": checks,
    }


def compare_real_experiment_external_artifact_launch_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_real_experiment_external_artifact_launch_report(report)
    contracts_path = _text_field(report, "contracts_path")
    current = real_experiment_external_artifact_launch_report(
        load_real_experiment_external_artifact_contracts(contracts_path),
        contracts_path=contracts_path,
    )
    saved_digest = _text_or_none(report.get("report_digest"))
    current_digest = _text_or_none(current.get("report_digest"))
    checks = [
        {
            "name": "launch_report_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        {
            "name": "launch_report_digest_matches_current",
            "passed": saved_digest == current_digest,
            "expected": saved_digest,
            "actual": current_digest,
        },
        {
            "name": "launch_report_payload_matches_current",
            "passed": _json_value(report) == _json_value(current),
            "expected": _json_value(report),
            "actual": _json_value(current),
        },
    ]
    return {
        "action": "compare_real_experiment_external_artifact_launch_report",
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def real_experiment_primary_evidence_status_digest(
    status: Mapping[str, Any],
) -> str:
    payload = {key: value for key, value in status.items() if key != "status_digest"}
    return hashlib.sha256(
        json.dumps(
            _json_value(payload),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def real_experiment_primary_evidence_status_json(
    status: Mapping[str, Any],
) -> str:
    return json.dumps(_json_value(status), indent=2, sort_keys=True) + "\n"


def save_real_experiment_primary_evidence_status(
    status: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        real_experiment_primary_evidence_status_json(status),
        encoding="utf-8",
    )
    return output_path


def load_real_experiment_primary_evidence_status(
    path: str | Path,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError(
            "Real experiment primary evidence status JSON must be an object"
        )
    schema_version = payload.get("schema_version")
    if schema_version != REAL_EXPERIMENT_PRIMARY_EVIDENCE_STATUS_SCHEMA_VERSION:
        raise SpatialQAError(
            "Unsupported real experiment primary evidence status schema "
            f"version: {schema_version}"
        )
    return cast(dict[str, Any], payload)


def real_experiment_primary_evidence_status(
    launch_report: Mapping[str, Any],
    *,
    launch_report_path: str | Path | None = None,
) -> dict[str, Any]:
    validation = validate_real_experiment_external_artifact_launch_report(
        launch_report
    )
    gate = _mapping(
        launch_report.get("primary_evidence_receipt_gate"),
        "primary_evidence_receipt_gate",
    )
    intake_plan = _mapping(
        launch_report.get("primary_evidence_intake_plan"),
        "primary_evidence_intake_plan",
    )
    steps_by_track = {
        _text_field(step, "track"): step
        for step in _mapping_sequence(intake_plan.get("steps"))
    }
    rows = [
        _primary_evidence_status_track_row(
            track,
            launch_report=launch_report,
            step=_mapping(steps_by_track.get(track), track),
        )
        for track in REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS
    ]
    ready_count = sum(1 for row in rows if row["ready"] is True)
    blocked_count = len(rows) - ready_count
    status: dict[str, Any] = {
        "schema_version": REAL_EXPERIMENT_PRIMARY_EVIDENCE_STATUS_SCHEMA_VERSION,
        "action": "real_experiment_primary_evidence_status",
        "launch_report_path": (
            str(launch_report_path) if launch_report_path is not None else None
        ),
        "launch_report_digest": _text_or_none(launch_report.get("report_digest")),
        "launch_report_validation": {
            "report_digest": validation["report_digest"],
            "valid": validation["valid"],
        },
        "summary": {
            "blocked_track_count": blocked_count,
            "preflight_ready_to_run": (
                launch_report.get("preflight_ready_to_run") is True
            ),
            "ready": gate.get("ready") is True,
            "ready_to_run": launch_report.get("ready_to_run") is True,
            "ready_track_count": ready_count,
            "track_count": len(rows),
        },
        "next_blocked_track": _primary_evidence_status_next_blocked_track(rows),
        "track_order": list(REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS),
        "tracks": rows,
    }
    status["status_digest"] = real_experiment_primary_evidence_status_digest(status)
    return status


def validate_real_experiment_primary_evidence_status(
    status: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = status.get("schema_version")
    action = status.get("action")
    status_digest = _text_or_none(status.get("status_digest"))
    expected_digest = real_experiment_primary_evidence_status_digest(status)
    launch_validation = _mapping(
        status.get("launch_report_validation"),
        "launch_report_validation",
    )
    summary = _mapping(status.get("summary"), "summary")
    tracks = _mapping_sequence(status.get("tracks"))
    track_order = _string_sequence_or_empty(status.get("track_order"))
    ready_count = sum(1 for row in tracks if row.get("ready") is True)
    blocked_count = len(tracks) - ready_count
    next_blocked_track = status.get("next_blocked_track")
    if next_blocked_track is not None and not isinstance(
        next_blocked_track,
        Mapping,
    ):
        raise SpatialQAError(
            "Real experiment primary evidence next_blocked_track must be an object"
        )
    expected_next_blocked_track = _primary_evidence_status_next_blocked_track(
        tracks
    )
    row_shape_valid = all(
        _primary_evidence_status_track_row_valid(row) for row in tracks
    )
    checks = [
        {
            "name": "schema_version",
            "passed": (
                schema_version
                == REAL_EXPERIMENT_PRIMARY_EVIDENCE_STATUS_SCHEMA_VERSION
            ),
            "expected": REAL_EXPERIMENT_PRIMARY_EVIDENCE_STATUS_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "action",
            "passed": action == "real_experiment_primary_evidence_status",
            "expected": "real_experiment_primary_evidence_status",
            "actual": action,
        },
        {
            "name": "status_digest",
            "passed": status_digest == expected_digest,
            "expected": expected_digest,
            "actual": status_digest,
        },
        {
            "name": "launch_report_path_present",
            "passed": _text_or_none(status.get("launch_report_path")) is not None,
            "expected": True,
            "actual": _text_or_none(status.get("launch_report_path")) is not None,
        },
        {
            "name": "launch_report_validation_valid",
            "passed": launch_validation.get("valid") is True,
            "expected": True,
            "actual": launch_validation.get("valid"),
        },
        {
            "name": "track_order",
            "passed": track_order == list(REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS),
            "expected": list(REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS),
            "actual": track_order,
        },
        {
            "name": "track_count",
            "passed": _summary_int(summary, "track_count") == len(tracks),
            "expected": len(tracks),
            "actual": _summary_int(summary, "track_count"),
        },
        {
            "name": "track_counts",
            "passed": (
                _summary_int(summary, "ready_track_count") == ready_count
                and _summary_int(summary, "blocked_track_count") == blocked_count
            ),
            "expected": {
                "blocked_track_count": blocked_count,
                "ready_track_count": ready_count,
            },
            "actual": {
                "blocked_track_count": _summary_int(
                    summary,
                    "blocked_track_count",
                ),
                "ready_track_count": _summary_int(
                    summary,
                    "ready_track_count",
                ),
            },
        },
        {
            "name": "ready_consistent",
            "passed": summary.get("ready") is (blocked_count == 0),
            "expected": blocked_count == 0,
            "actual": summary.get("ready"),
        },
        {
            "name": "track_rows",
            "passed": row_shape_valid,
            "expected": True,
            "actual": row_shape_valid,
        },
        {
            "name": "next_blocked_track",
            "passed": _json_value(next_blocked_track)
            == _json_value(expected_next_blocked_track),
            "expected": _json_value(expected_next_blocked_track),
            "actual": _json_value(next_blocked_track),
        },
    ]
    return {
        "action": "validate_real_experiment_primary_evidence_status",
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "status_digest": status_digest,
        "checks": checks,
    }


def compare_real_experiment_primary_evidence_status(
    status: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_real_experiment_primary_evidence_status(status)
    launch_report_path = _text_field(status, "launch_report_path")
    saved_launch_report = load_real_experiment_external_artifact_launch_report(
        launch_report_path
    )
    contracts_path = _text_field(saved_launch_report, "contracts_path")
    current_launch_report = real_experiment_external_artifact_launch_report(
        load_real_experiment_external_artifact_contracts(contracts_path),
        contracts_path=contracts_path,
    )
    current = real_experiment_primary_evidence_status(
        current_launch_report,
        launch_report_path=launch_report_path,
    )
    saved_digest = _text_or_none(status.get("status_digest"))
    current_digest = _text_or_none(current.get("status_digest"))
    checks = [
        {
            "name": "primary_evidence_status_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        {
            "name": "primary_evidence_status_digest_matches_current",
            "passed": saved_digest == current_digest,
            "expected": saved_digest,
            "actual": current_digest,
        },
        {
            "name": "primary_evidence_status_payload_matches_current",
            "passed": _json_value(status) == _json_value(current),
            "expected": _json_value(status),
            "actual": _json_value(current),
        },
    ]
    return {
        "action": "compare_real_experiment_primary_evidence_status",
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def real_experiment_primary_evidence_request_package_digest(
    package: Mapping[str, Any],
) -> str:
    payload = {
        key: value for key, value in package.items() if key != "package_digest"
    }
    return hashlib.sha256(
        json.dumps(
            _json_value(payload),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def real_experiment_primary_evidence_request_package_json(
    package: Mapping[str, Any],
) -> str:
    return json.dumps(_json_value(package), indent=2, sort_keys=True) + "\n"


def save_real_experiment_primary_evidence_request_package(
    package: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        real_experiment_primary_evidence_request_package_json(package),
        encoding="utf-8",
    )
    return output_path


def load_real_experiment_primary_evidence_request_package(
    path: str | Path,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError(
            "Real experiment primary evidence request package JSON must be an object"
        )
    schema_version = payload.get("schema_version")
    if (
        schema_version
        != REAL_EXPERIMENT_PRIMARY_EVIDENCE_REQUEST_PACKAGE_SCHEMA_VERSION
    ):
        raise SpatialQAError(
            "Unsupported real experiment primary evidence request package schema "
            f"version: {schema_version}"
        )
    return cast(dict[str, Any], payload)


def real_experiment_primary_evidence_request_package(
    launch_report: Mapping[str, Any],
    *,
    launch_report_path: str | Path | None = None,
) -> dict[str, Any]:
    validation = validate_real_experiment_external_artifact_launch_report(
        launch_report
    )
    rows = [
        _primary_evidence_request_package_track_row(
            track,
            launch_report=launch_report,
        )
        for track in REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS
    ]
    ready_count = sum(1 for row in rows if row["status"] == "ready")
    blocked_count = len(rows) - ready_count
    package: dict[str, Any] = {
        "schema_version": (
            REAL_EXPERIMENT_PRIMARY_EVIDENCE_REQUEST_PACKAGE_SCHEMA_VERSION
        ),
        "action": "real_experiment_primary_evidence_request_package",
        "launch_report_path": (
            str(launch_report_path) if launch_report_path is not None else None
        ),
        "launch_report_digest": _text_or_none(launch_report.get("report_digest")),
        "launch_report_validation": {
            "report_digest": validation["report_digest"],
            "valid": validation["valid"],
        },
        "summary": {
            "all_request_tracks_ready": blocked_count == 0,
            "blocked_request_track_count": blocked_count,
            "ready_request_track_count": ready_count,
            "track_count": len(rows),
        },
        "track_order": list(REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS),
        "tracks": rows,
    }
    package["package_digest"] = (
        real_experiment_primary_evidence_request_package_digest(package)
    )
    return package


def validate_real_experiment_primary_evidence_request_package(
    package: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = package.get("schema_version")
    action = package.get("action")
    package_digest = _text_or_none(package.get("package_digest"))
    expected_digest = real_experiment_primary_evidence_request_package_digest(
        package
    )
    launch_validation = _mapping(
        package.get("launch_report_validation"),
        "launch_report_validation",
    )
    summary = _mapping(package.get("summary"), "summary")
    tracks = _mapping_sequence(package.get("tracks"))
    track_order = _string_sequence_or_empty(package.get("track_order"))
    ready_count = sum(1 for row in tracks if row.get("status") == "ready")
    blocked_count = len(tracks) - ready_count
    row_shape_valid = all(
        _primary_evidence_request_package_track_row_valid(row)
        for row in tracks
    )
    request_bundle_validations = [
        _primary_evidence_request_package_row_bundle_validation(row)
        for row in tracks
    ]
    request_bundle_validations_valid = all(
        validation["valid"] is True for validation in request_bundle_validations
    )
    checks = [
        {
            "name": "schema_version",
            "passed": (
                schema_version
                == REAL_EXPERIMENT_PRIMARY_EVIDENCE_REQUEST_PACKAGE_SCHEMA_VERSION
            ),
            "expected": (
                REAL_EXPERIMENT_PRIMARY_EVIDENCE_REQUEST_PACKAGE_SCHEMA_VERSION
            ),
            "actual": schema_version,
        },
        {
            "name": "action",
            "passed": action == "real_experiment_primary_evidence_request_package",
            "expected": "real_experiment_primary_evidence_request_package",
            "actual": action,
        },
        {
            "name": "package_digest",
            "passed": package_digest == expected_digest,
            "expected": expected_digest,
            "actual": package_digest,
        },
        {
            "name": "launch_report_path_present",
            "passed": _text_or_none(package.get("launch_report_path")) is not None,
            "expected": True,
            "actual": _text_or_none(package.get("launch_report_path")) is not None,
        },
        {
            "name": "launch_report_validation_valid",
            "passed": launch_validation.get("valid") is True,
            "expected": True,
            "actual": launch_validation.get("valid"),
        },
        {
            "name": "track_order",
            "passed": track_order == list(REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS),
            "expected": list(REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS),
            "actual": track_order,
        },
        {
            "name": "track_count",
            "passed": _summary_int(summary, "track_count") == len(tracks),
            "expected": len(tracks),
            "actual": _summary_int(summary, "track_count"),
        },
        {
            "name": "request_counts",
            "passed": (
                _summary_int(summary, "ready_request_track_count") == ready_count
                and _summary_int(summary, "blocked_request_track_count")
                == blocked_count
            ),
            "expected": {
                "blocked_request_track_count": blocked_count,
                "ready_request_track_count": ready_count,
            },
            "actual": {
                "blocked_request_track_count": _summary_int(
                    summary,
                    "blocked_request_track_count",
                ),
                "ready_request_track_count": _summary_int(
                    summary,
                    "ready_request_track_count",
                ),
            },
        },
        {
            "name": "all_request_tracks_ready",
            "passed": summary.get("all_request_tracks_ready")
            is (blocked_count == 0),
            "expected": blocked_count == 0,
            "actual": summary.get("all_request_tracks_ready"),
        },
        {
            "name": "track_rows",
            "passed": row_shape_valid,
            "expected": True,
            "actual": row_shape_valid,
        },
        {
            "name": "request_bundle_validations",
            "passed": request_bundle_validations_valid,
            "expected": True,
            "actual": request_bundle_validations,
        },
    ]
    return {
        "action": "validate_real_experiment_primary_evidence_request_package",
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "package_digest": package_digest,
        "checks": checks,
    }


def compare_real_experiment_primary_evidence_request_package(
    package: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_real_experiment_primary_evidence_request_package(package)
    launch_report_path = _text_field(package, "launch_report_path")
    saved_launch_report = load_real_experiment_external_artifact_launch_report(
        launch_report_path
    )
    contracts_path = _text_field(saved_launch_report, "contracts_path")
    current_launch_report = real_experiment_external_artifact_launch_report(
        load_real_experiment_external_artifact_contracts(contracts_path),
        contracts_path=contracts_path,
    )
    current = real_experiment_primary_evidence_request_package(
        current_launch_report,
        launch_report_path=launch_report_path,
    )
    saved_digest = _text_or_none(package.get("package_digest"))
    current_digest = _text_or_none(current.get("package_digest"))
    checks = [
        {
            "name": "primary_evidence_request_package_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        {
            "name": "primary_evidence_request_package_digest_matches_current",
            "passed": saved_digest == current_digest,
            "expected": saved_digest,
            "actual": current_digest,
        },
        {
            "name": "primary_evidence_request_package_payload_matches_current",
            "passed": _json_value(package) == _json_value(current),
            "expected": _json_value(package),
            "actual": _json_value(current),
        },
    ]
    return {
        "action": "compare_real_experiment_primary_evidence_request_package",
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def write_real_experiment_primary_evidence_request_bundles(
    package: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_real_experiment_primary_evidence_request_package(package)
    if validation["valid"] is not True:
        raise SpatialQAError(
            "Real experiment primary evidence request package must validate before "
            "writing child request bundles"
        )
    rows = _mapping_sequence(package.get("tracks"))
    output_rows = [
        _primary_evidence_request_bundle_write_row(row)
        for row in rows
    ]
    written_count = sum(1 for row in output_rows if row.get("status") == "written")
    skipped_count = sum(
        1 for row in output_rows if row.get("status") == "skipped_blocked"
    )
    ready_count = sum(1 for row in rows if row.get("status") == "ready")
    blocked_count = len(rows) - ready_count
    return {
        "action": "write_real_experiment_primary_evidence_request_bundles",
        "package_digest": _text_or_none(package.get("package_digest")),
        "package_validation": {
            "package_digest": validation["package_digest"],
            "valid": validation["valid"],
        },
        "summary": {
            "all_request_bundles_written": skipped_count == 0,
            "blocked_request_track_count": blocked_count,
            "ready_request_track_count": ready_count,
            "skipped_request_track_count": skipped_count,
            "track_count": len(rows),
            "written_request_bundle_count": written_count,
        },
        "tracks": output_rows,
    }


def real_experiment_primary_evidence_return_checklist_digest(
    checklist: Mapping[str, Any],
) -> str:
    payload = {
        key: value
        for key, value in checklist.items()
        if key != "checklist_digest"
    }
    return hashlib.sha256(
        json.dumps(
            _json_value(payload),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def real_experiment_primary_evidence_return_checklist_json(
    checklist: Mapping[str, Any],
) -> str:
    return json.dumps(_json_value(checklist), indent=2, sort_keys=True) + "\n"


def save_real_experiment_primary_evidence_return_checklist(
    checklist: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        real_experiment_primary_evidence_return_checklist_json(checklist),
        encoding="utf-8",
    )
    return output_path


def load_real_experiment_primary_evidence_return_checklist(
    path: str | Path,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError(
            "Real experiment primary evidence return checklist JSON must be an object"
        )
    schema_version = payload.get("schema_version")
    if (
        schema_version
        != REAL_EXPERIMENT_PRIMARY_EVIDENCE_RETURN_CHECKLIST_SCHEMA_VERSION
    ):
        raise SpatialQAError(
            "Unsupported real experiment primary evidence return checklist schema "
            f"version: {schema_version}"
        )
    return cast(dict[str, Any], payload)


def real_experiment_primary_evidence_return_checklist(
    package: Mapping[str, Any],
    *,
    request_package_path: str | Path | None = None,
) -> dict[str, Any]:
    validation = validate_real_experiment_primary_evidence_request_package(package)
    launch_report_path = _text_field(package, "launch_report_path")
    launch_report = load_real_experiment_external_artifact_launch_report(
        launch_report_path
    )
    rows = _mapping_sequence(package.get("tracks"))
    steps = [
        _primary_evidence_return_checklist_step(
            row,
            launch_report=launch_report,
        )
        for row in rows
    ]
    actionable_count = sum(1 for step in steps if step["status"] == "actionable")
    blocked_count = len(steps) - actionable_count
    checklist: dict[str, Any] = {
        "schema_version": (
            REAL_EXPERIMENT_PRIMARY_EVIDENCE_RETURN_CHECKLIST_SCHEMA_VERSION
        ),
        "action": "real_experiment_primary_evidence_return_checklist",
        "request_package_path": (
            str(request_package_path) if request_package_path is not None else None
        ),
        "request_package_digest": _text_or_none(package.get("package_digest")),
        "request_package_validation": {
            "package_digest": validation["package_digest"],
            "valid": validation["valid"],
        },
        "launch_report_path": launch_report_path,
        "launch_report_digest": _text_or_none(launch_report.get("report_digest")),
        "summary": {
            "actionable_return_track_count": actionable_count,
            "all_return_tracks_actionable": blocked_count == 0,
            "blocked_return_track_count": blocked_count,
            "track_count": len(steps),
        },
        "track_order": list(REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS),
        "steps": steps,
    }
    checklist["checklist_digest"] = (
        real_experiment_primary_evidence_return_checklist_digest(checklist)
    )
    return checklist


def validate_real_experiment_primary_evidence_return_checklist(
    checklist: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = checklist.get("schema_version")
    action = checklist.get("action")
    checklist_digest = _text_or_none(checklist.get("checklist_digest"))
    expected_digest = real_experiment_primary_evidence_return_checklist_digest(
        checklist
    )
    package_validation = _mapping(
        checklist.get("request_package_validation"),
        "request_package_validation",
    )
    summary = _mapping(checklist.get("summary"), "summary")
    steps = _mapping_sequence(checklist.get("steps"))
    track_order = _string_sequence_or_empty(checklist.get("track_order"))
    actionable_count = sum(1 for step in steps if step.get("status") == "actionable")
    blocked_count = len(steps) - actionable_count
    step_rows_valid = all(
        _primary_evidence_return_checklist_step_valid(step) for step in steps
    )
    checks = [
        {
            "name": "schema_version",
            "passed": (
                schema_version
                == REAL_EXPERIMENT_PRIMARY_EVIDENCE_RETURN_CHECKLIST_SCHEMA_VERSION
            ),
            "expected": (
                REAL_EXPERIMENT_PRIMARY_EVIDENCE_RETURN_CHECKLIST_SCHEMA_VERSION
            ),
            "actual": schema_version,
        },
        {
            "name": "action",
            "passed": action == "real_experiment_primary_evidence_return_checklist",
            "expected": "real_experiment_primary_evidence_return_checklist",
            "actual": action,
        },
        {
            "name": "checklist_digest",
            "passed": checklist_digest == expected_digest,
            "expected": expected_digest,
            "actual": checklist_digest,
        },
        {
            "name": "request_package_path_present",
            "passed": _text_or_none(checklist.get("request_package_path")) is not None,
            "expected": True,
            "actual": _text_or_none(checklist.get("request_package_path")) is not None,
        },
        {
            "name": "request_package_validation_valid",
            "passed": package_validation.get("valid") is True,
            "expected": True,
            "actual": package_validation.get("valid"),
        },
        {
            "name": "track_order",
            "passed": track_order == list(REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS),
            "expected": list(REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS),
            "actual": track_order,
        },
        {
            "name": "track_count",
            "passed": _summary_int(summary, "track_count") == len(steps),
            "expected": len(steps),
            "actual": _summary_int(summary, "track_count"),
        },
        {
            "name": "return_counts",
            "passed": (
                _summary_int(summary, "actionable_return_track_count")
                == actionable_count
                and _summary_int(summary, "blocked_return_track_count")
                == blocked_count
            ),
            "expected": {
                "actionable_return_track_count": actionable_count,
                "blocked_return_track_count": blocked_count,
            },
            "actual": {
                "actionable_return_track_count": _summary_int(
                    summary,
                    "actionable_return_track_count",
                ),
                "blocked_return_track_count": _summary_int(
                    summary,
                    "blocked_return_track_count",
                ),
            },
        },
        {
            "name": "all_return_tracks_actionable",
            "passed": summary.get("all_return_tracks_actionable")
            is (blocked_count == 0),
            "expected": blocked_count == 0,
            "actual": summary.get("all_return_tracks_actionable"),
        },
        {
            "name": "step_rows",
            "passed": step_rows_valid,
            "expected": True,
            "actual": step_rows_valid,
        },
    ]
    return {
        "action": "validate_real_experiment_primary_evidence_return_checklist",
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "checklist_digest": checklist_digest,
        "checks": checks,
    }


def compare_real_experiment_primary_evidence_return_checklist(
    checklist: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_real_experiment_primary_evidence_return_checklist(
        checklist
    )
    request_package_path = _text_field(checklist, "request_package_path")
    current = real_experiment_primary_evidence_return_checklist(
        load_real_experiment_primary_evidence_request_package(request_package_path),
        request_package_path=request_package_path,
    )
    saved_digest = _text_or_none(checklist.get("checklist_digest"))
    current_digest = _text_or_none(current.get("checklist_digest"))
    checks = [
        {
            "name": "primary_evidence_return_checklist_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        {
            "name": "primary_evidence_return_checklist_digest_matches_current",
            "passed": saved_digest == current_digest,
            "expected": saved_digest,
            "actual": current_digest,
        },
        {
            "name": "primary_evidence_return_checklist_payload_matches_current",
            "passed": _json_value(checklist) == _json_value(current),
            "expected": _json_value(checklist),
            "actual": _json_value(current),
        },
    ]
    return {
        "action": "compare_real_experiment_primary_evidence_return_checklist",
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def real_experiment_primary_evidence_return_progress_report_digest(
    report: Mapping[str, Any],
) -> str:
    payload = {
        key: value for key, value in report.items() if key != "report_digest"
    }
    return hashlib.sha256(
        json.dumps(
            _json_value(payload),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def real_experiment_primary_evidence_return_progress_report_json(
    report: Mapping[str, Any],
) -> str:
    return json.dumps(_json_value(report), indent=2, sort_keys=True) + "\n"


def save_real_experiment_primary_evidence_return_progress_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        real_experiment_primary_evidence_return_progress_report_json(report),
        encoding="utf-8",
    )
    return output_path


def load_real_experiment_primary_evidence_return_progress_report(
    path: str | Path,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError(
            "Real experiment primary evidence return progress report JSON must be an object"
        )
    schema_version = payload.get("schema_version")
    if (
        schema_version
        != REAL_EXPERIMENT_PRIMARY_EVIDENCE_RETURN_PROGRESS_REPORT_SCHEMA_VERSION
    ):
        raise SpatialQAError(
            "Unsupported real experiment primary evidence return progress report "
            f"schema version: {schema_version}"
        )
    return cast(dict[str, Any], payload)


def real_experiment_primary_evidence_return_progress_report(
    checklist: Mapping[str, Any],
    *,
    return_checklist_path: str | Path | None = None,
) -> dict[str, Any]:
    validation = validate_real_experiment_primary_evidence_return_checklist(
        checklist
    )
    rows = [
        _primary_evidence_return_progress_track(step)
        for step in _mapping_sequence(checklist.get("steps"))
    ]
    summary = _primary_evidence_return_progress_summary(rows)
    report: dict[str, Any] = {
        "schema_version": (
            REAL_EXPERIMENT_PRIMARY_EVIDENCE_RETURN_PROGRESS_REPORT_SCHEMA_VERSION
        ),
        "action": "real_experiment_primary_evidence_return_progress_report",
        "return_checklist_path": (
            str(return_checklist_path)
            if return_checklist_path is not None
            else None
        ),
        "return_checklist_digest": _text_or_none(
            checklist.get("checklist_digest")
        ),
        "return_checklist_validation": {
            "checklist_digest": validation["checklist_digest"],
            "valid": validation["valid"],
        },
        "next_missing_return": _primary_evidence_return_progress_next_missing(
            rows
        ),
        "summary": summary,
        "track_order": list(REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS),
        "tracks": rows,
    }
    report["report_digest"] = (
        real_experiment_primary_evidence_return_progress_report_digest(report)
    )
    return report


def validate_real_experiment_primary_evidence_return_progress_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = report.get("schema_version")
    action = report.get("action")
    report_digest = _text_or_none(report.get("report_digest"))
    expected_digest = real_experiment_primary_evidence_return_progress_report_digest(
        report
    )
    checklist_validation = _mapping(
        report.get("return_checklist_validation"),
        "return_checklist_validation",
    )
    summary = _mapping(report.get("summary"), "summary")
    rows = _mapping_sequence(report.get("tracks"))
    track_order = _string_sequence_or_empty(report.get("track_order"))
    expected_summary = _primary_evidence_return_progress_summary(rows)
    expected_next_missing = _primary_evidence_return_progress_next_missing(rows)
    next_missing = report.get("next_missing_return")
    if next_missing is not None and not isinstance(next_missing, Mapping):
        raise SpatialQAError(
            "Real experiment primary evidence return progress next_missing_return "
            "must be an object"
        )
    row_shape_valid = all(
        _primary_evidence_return_progress_track_valid(row) for row in rows
    )
    checks = [
        {
            "name": "schema_version",
            "passed": (
                schema_version
                == REAL_EXPERIMENT_PRIMARY_EVIDENCE_RETURN_PROGRESS_REPORT_SCHEMA_VERSION
            ),
            "expected": (
                REAL_EXPERIMENT_PRIMARY_EVIDENCE_RETURN_PROGRESS_REPORT_SCHEMA_VERSION
            ),
            "actual": schema_version,
        },
        {
            "name": "action",
            "passed": action
            == "real_experiment_primary_evidence_return_progress_report",
            "expected": "real_experiment_primary_evidence_return_progress_report",
            "actual": action,
        },
        {
            "name": "report_digest",
            "passed": report_digest == expected_digest,
            "expected": expected_digest,
            "actual": report_digest,
        },
        {
            "name": "return_checklist_path_present",
            "passed": _text_or_none(report.get("return_checklist_path")) is not None,
            "expected": True,
            "actual": _text_or_none(report.get("return_checklist_path")) is not None,
        },
        {
            "name": "return_checklist_validation_valid",
            "passed": checklist_validation.get("valid") is True,
            "expected": True,
            "actual": checklist_validation.get("valid"),
        },
        {
            "name": "track_order",
            "passed": track_order == list(REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS),
            "expected": list(REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS),
            "actual": track_order,
        },
        {
            "name": "summary",
            "passed": _json_value(summary) == _json_value(expected_summary),
            "expected": _json_value(expected_summary),
            "actual": _json_value(summary),
        },
        {
            "name": "next_missing_return",
            "passed": _json_value(next_missing) == _json_value(expected_next_missing),
            "expected": _json_value(expected_next_missing),
            "actual": _json_value(next_missing),
        },
        {
            "name": "track_rows",
            "passed": row_shape_valid,
            "expected": True,
            "actual": row_shape_valid,
        },
    ]
    return {
        "action": "validate_real_experiment_primary_evidence_return_progress_report",
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "report_digest": report_digest,
        "checks": checks,
    }


def compare_real_experiment_primary_evidence_return_progress_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_real_experiment_primary_evidence_return_progress_report(
        report
    )
    return_checklist_path = _text_field(report, "return_checklist_path")
    current = real_experiment_primary_evidence_return_progress_report(
        load_real_experiment_primary_evidence_return_checklist(
            return_checklist_path
        ),
        return_checklist_path=return_checklist_path,
    )
    saved_digest = _text_or_none(report.get("report_digest"))
    current_digest = _text_or_none(current.get("report_digest"))
    checks = [
        {
            "name": "primary_evidence_return_progress_report_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        {
            "name": "primary_evidence_return_progress_report_digest_matches_current",
            "passed": saved_digest == current_digest,
            "expected": saved_digest,
            "actual": current_digest,
        },
        {
            "name": "primary_evidence_return_progress_report_payload_matches_current",
            "passed": _json_value(report) == _json_value(current),
            "expected": _json_value(report),
            "actual": _json_value(current),
        },
    ]
    return {
        "action": "compare_real_experiment_primary_evidence_return_progress_report",
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def real_experiment_primary_evidence_acceptance_report_digest(
    report: Mapping[str, Any],
) -> str:
    payload = {
        key: value
        for key, value in report.items()
        if key != "acceptance_digest"
    }
    return hashlib.sha256(
        json.dumps(
            _json_value(payload),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def real_experiment_primary_evidence_acceptance_report_json(
    report: Mapping[str, Any],
) -> str:
    return json.dumps(_json_value(report), indent=2, sort_keys=True) + "\n"


def save_real_experiment_primary_evidence_acceptance_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        real_experiment_primary_evidence_acceptance_report_json(report),
        encoding="utf-8",
    )
    return output_path


def load_real_experiment_primary_evidence_acceptance_report(
    path: str | Path,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError(
            "Real experiment primary evidence acceptance report JSON must be an object"
        )
    schema_version = payload.get("schema_version")
    if (
        schema_version
        != REAL_EXPERIMENT_PRIMARY_EVIDENCE_ACCEPTANCE_REPORT_SCHEMA_VERSION
    ):
        raise SpatialQAError(
            "Unsupported real experiment primary evidence acceptance report "
            f"schema version: {schema_version}"
        )
    return cast(dict[str, Any], payload)


def real_experiment_primary_evidence_acceptance_report(
    return_progress: Mapping[str, Any],
    *,
    return_progress_path: str | Path | None = None,
) -> dict[str, Any]:
    progress_validation = (
        validate_real_experiment_primary_evidence_return_progress_report(
            return_progress
        )
    )
    progress_comparison = (
        compare_real_experiment_primary_evidence_return_progress_report(
            return_progress
        )
    )
    return_checklist_path = _text_field(
        return_progress,
        "return_checklist_path",
    )
    return_checklist = load_real_experiment_primary_evidence_return_checklist(
        return_checklist_path
    )
    request_package_path = _text_field(return_checklist, "request_package_path")
    request_package = load_real_experiment_primary_evidence_request_package(
        request_package_path
    )
    launch_report_path = _text_field(request_package, "launch_report_path")
    saved_launch_report = load_real_experiment_external_artifact_launch_report(
        launch_report_path
    )
    contracts_path = _text_field(saved_launch_report, "contracts_path")
    current_launch_report = real_experiment_external_artifact_launch_report(
        load_real_experiment_external_artifact_contracts(contracts_path),
        contracts_path=contracts_path,
    )
    steps_by_track = {
        _text_field(step, "track"): step
        for step in _mapping_sequence(return_checklist.get("steps"))
    }
    tracks = [
        _primary_evidence_acceptance_track(
            track,
            _mapping(steps_by_track.get(track), track),
            current_launch_report,
        )
        for track in REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS
    ]
    summary = _primary_evidence_acceptance_summary(tracks)
    report: dict[str, Any] = {
        "schema_version": (
            REAL_EXPERIMENT_PRIMARY_EVIDENCE_ACCEPTANCE_REPORT_SCHEMA_VERSION
        ),
        "action": "real_experiment_primary_evidence_acceptance_report",
        "return_progress_path": (
            str(return_progress_path)
            if return_progress_path is not None
            else None
        ),
        "return_progress_digest": _text_or_none(
            return_progress.get("report_digest")
        ),
        "return_progress_validation": {
            "report_digest": progress_validation["report_digest"],
            "valid": progress_validation["valid"],
        },
        "return_progress_comparison": {
            "current_digest": progress_comparison["current_digest"],
            "matches": progress_comparison["matches"],
            "saved_digest": progress_comparison["saved_digest"],
        },
        "return_checklist_path": return_checklist_path,
        "return_checklist_digest": _text_or_none(
            return_checklist.get("checklist_digest")
        ),
        "request_package_path": request_package_path,
        "request_package_digest": _text_or_none(
            request_package.get("package_digest")
        ),
        "launch_report_path": launch_report_path,
        "current_launch_report_digest": _text_or_none(
            current_launch_report.get("report_digest")
        ),
        "contracts_path": contracts_path,
        "summary": summary,
        "track_order": list(REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS),
        "next_unaccepted_track": _primary_evidence_acceptance_next_unaccepted(
            tracks
        ),
        "tracks": tracks,
    }
    report["acceptance_digest"] = (
        real_experiment_primary_evidence_acceptance_report_digest(report)
    )
    return report


def validate_real_experiment_primary_evidence_acceptance_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = report.get("schema_version")
    action = report.get("action")
    acceptance_digest = _text_or_none(report.get("acceptance_digest"))
    expected_digest = real_experiment_primary_evidence_acceptance_report_digest(
        report
    )
    progress_validation = _mapping(
        report.get("return_progress_validation"),
        "return_progress_validation",
    )
    progress_comparison = _mapping(
        report.get("return_progress_comparison"),
        "return_progress_comparison",
    )
    summary = _mapping(report.get("summary"), "summary")
    rows = _mapping_sequence(report.get("tracks"))
    track_order = _string_sequence_or_empty(report.get("track_order"))
    expected_summary = _primary_evidence_acceptance_summary(rows)
    expected_next = _primary_evidence_acceptance_next_unaccepted(rows)
    next_unaccepted = report.get("next_unaccepted_track")
    if next_unaccepted is not None and not isinstance(next_unaccepted, Mapping):
        raise SpatialQAError(
            "Real experiment primary evidence acceptance next_unaccepted_track "
            "must be an object"
        )
    rows_valid = all(_primary_evidence_acceptance_track_valid(row) for row in rows)
    checks = [
        {
            "name": "schema_version",
            "passed": (
                schema_version
                == REAL_EXPERIMENT_PRIMARY_EVIDENCE_ACCEPTANCE_REPORT_SCHEMA_VERSION
            ),
            "expected": (
                REAL_EXPERIMENT_PRIMARY_EVIDENCE_ACCEPTANCE_REPORT_SCHEMA_VERSION
            ),
            "actual": schema_version,
        },
        {
            "name": "action",
            "passed": action
            == "real_experiment_primary_evidence_acceptance_report",
            "expected": "real_experiment_primary_evidence_acceptance_report",
            "actual": action,
        },
        {
            "name": "acceptance_digest",
            "passed": acceptance_digest == expected_digest,
            "expected": expected_digest,
            "actual": acceptance_digest,
        },
        {
            "name": "return_progress_path_present",
            "passed": _text_or_none(report.get("return_progress_path")) is not None,
            "expected": True,
            "actual": _text_or_none(report.get("return_progress_path")) is not None,
        },
        {
            "name": "return_progress_validation_valid",
            "passed": progress_validation.get("valid") is True,
            "expected": True,
            "actual": progress_validation.get("valid"),
        },
        {
            "name": "return_progress_comparison_matches",
            "passed": progress_comparison.get("matches") is True,
            "expected": True,
            "actual": progress_comparison.get("matches"),
        },
        {
            "name": "track_order",
            "passed": track_order == list(REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS),
            "expected": list(REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS),
            "actual": track_order,
        },
        {
            "name": "summary",
            "passed": _json_value(summary) == _json_value(expected_summary),
            "expected": _json_value(expected_summary),
            "actual": _json_value(summary),
        },
        {
            "name": "next_unaccepted_track",
            "passed": _json_value(next_unaccepted) == _json_value(expected_next),
            "expected": _json_value(expected_next),
            "actual": _json_value(next_unaccepted),
        },
        {
            "name": "track_rows",
            "passed": rows_valid,
            "expected": True,
            "actual": rows_valid,
        },
    ]
    return {
        "action": "validate_real_experiment_primary_evidence_acceptance_report",
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "acceptance_digest": acceptance_digest,
        "checks": checks,
    }


def compare_real_experiment_primary_evidence_acceptance_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_real_experiment_primary_evidence_acceptance_report(
        report
    )
    return_progress_path = _text_field(report, "return_progress_path")
    current = real_experiment_primary_evidence_acceptance_report(
        load_real_experiment_primary_evidence_return_progress_report(
            return_progress_path
        ),
        return_progress_path=return_progress_path,
    )
    saved_digest = _text_or_none(report.get("acceptance_digest"))
    current_digest = _text_or_none(current.get("acceptance_digest"))
    checks = [
        {
            "name": "primary_evidence_acceptance_report_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        {
            "name": "primary_evidence_acceptance_report_digest_matches_current",
            "passed": saved_digest == current_digest,
            "expected": saved_digest,
            "actual": current_digest,
        },
        {
            "name": "primary_evidence_acceptance_report_payload_matches_current",
            "passed": _json_value(report) == _json_value(current),
            "expected": _json_value(report),
            "actual": _json_value(current),
        },
    ]
    return {
        "action": "compare_real_experiment_primary_evidence_acceptance_report",
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def real_experiment_execution_packet_digest(packet: Mapping[str, Any]) -> str:
    payload = {
        key: value for key, value in packet.items() if key != "packet_digest"
    }
    return hashlib.sha256(
        json.dumps(
            _json_value(payload),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def real_experiment_execution_packet_json(packet: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(packet), indent=2, sort_keys=True) + "\n"


def save_real_experiment_execution_packet(
    packet: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        real_experiment_execution_packet_json(packet),
        encoding="utf-8",
    )
    return output_path


def load_real_experiment_execution_packet(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Real experiment execution packet JSON must be an object")
    schema_version = payload.get("schema_version")
    if schema_version != REAL_EXPERIMENT_EXECUTION_PACKET_SCHEMA_VERSION:
        raise SpatialQAError(
            "Unsupported real experiment execution packet schema version: "
            f"{schema_version}"
        )
    return cast(dict[str, Any], payload)


def validate_real_experiment_execution_packet(
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = packet.get("schema_version")
    action = packet.get("action")
    packet_digest = _text_or_none(packet.get("packet_digest"))
    expected_digest = real_experiment_execution_packet_digest(packet)
    ready_to_execute = packet.get("ready_to_execute")
    execution_blocked = packet.get("execution_blocked")
    launch_report_path = _text_or_none(packet.get("launch_report_path"))
    launch_report_digest = _text_or_none(packet.get("launch_report_digest"))
    primary_acceptance_path = _text_or_none(
        packet.get("primary_evidence_acceptance_report_path")
    )
    primary_acceptance = _mapping(
        packet.get("primary_evidence_acceptance"),
        "primary_evidence_acceptance",
    )
    audit_commands = _mapping_sequence(packet.get("audit_commands"))
    execution_commands = _mapping_sequence(packet.get("execution_commands"))
    audit_command_keys = [_text_field(command, "key") for command in audit_commands]
    execution_command_keys = [
        _text_field(command, "key") for command in execution_commands
    ]
    primary_acceptance_ready = (
        primary_acceptance.get("present") is True
        and primary_acceptance.get("valid") is True
        and primary_acceptance.get("matches_current") is True
        and primary_acceptance.get("ready_for_launch_refresh") is True
        and primary_acceptance.get("all_tracks_accepted") is True
    )
    checks = [
        {
            "name": "schema_version",
            "passed": (
                schema_version == REAL_EXPERIMENT_EXECUTION_PACKET_SCHEMA_VERSION
            ),
            "expected": REAL_EXPERIMENT_EXECUTION_PACKET_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "action",
            "passed": action == "real_experiment_execution_packet",
            "expected": "real_experiment_execution_packet",
            "actual": action,
        },
        {
            "name": "packet_digest",
            "passed": packet_digest == expected_digest,
            "expected": expected_digest,
            "actual": packet_digest,
        },
        {
            "name": "launch_report_path_present",
            "passed": launch_report_path is not None,
            "expected": True,
            "actual": launch_report_path is not None,
        },
        {
            "name": "launch_report_digest_present",
            "passed": launch_report_digest is not None,
            "expected": True,
            "actual": launch_report_digest is not None,
        },
        {
            "name": "primary_evidence_acceptance_report_path_present",
            "passed": primary_acceptance_path is not None,
            "expected": True,
            "actual": primary_acceptance_path is not None,
        },
        {
            "name": "primary_evidence_acceptance_consistent",
            "passed": ready_to_execute is not True or primary_acceptance_ready,
            "expected": True,
            "actual": {
                "primary_evidence_acceptance_ready": primary_acceptance_ready,
                "ready_to_execute": ready_to_execute,
            },
        },
        {
            "name": "ready_to_execute_bool",
            "passed": isinstance(ready_to_execute, bool),
            "expected": "bool",
            "actual": type(ready_to_execute).__name__,
        },
        {
            "name": "execution_blocked_bool",
            "passed": isinstance(execution_blocked, bool),
            "expected": "bool",
            "actual": type(execution_blocked).__name__,
        },
        {
            "name": "blocked_consistent",
            "passed": (
                isinstance(ready_to_execute, bool)
                and isinstance(execution_blocked, bool)
                and execution_blocked is (not ready_to_execute)
            ),
            "expected": True,
            "actual": {
                "execution_blocked": execution_blocked,
                "ready_to_execute": ready_to_execute,
            },
        },
        {
            "name": "audit_commands_present",
            "passed": len(audit_commands) >= 2,
            "expected": "at least 2",
            "actual": len(audit_commands),
        },
        {
            "name": "audit_commands_include_primary_evidence_acceptance",
            "passed": {
                "compare_primary_evidence_acceptance_report",
                "validate_primary_evidence_acceptance_report",
            }.issubset(audit_command_keys),
            "expected": [
                "validate_primary_evidence_acceptance_report",
                "compare_primary_evidence_acceptance_report",
            ],
            "actual": audit_command_keys,
        },
        {
            "name": "execution_commands_match_readiness",
            "passed": (
                (
                    ready_to_execute is True
                    and execution_command_keys
                    == ["preflight_run_manifest", "run_real_experiment"]
                )
                or (ready_to_execute is False and not execution_commands)
            ),
            "expected": {
                "ready": ["preflight_run_manifest", "run_real_experiment"],
                "not_ready": [],
            },
            "actual": execution_command_keys,
        },
    ]
    return {
        "action": "validate_real_experiment_execution_packet",
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "packet_digest": packet_digest,
        "checks": checks,
    }


def compare_real_experiment_execution_packet(
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_real_experiment_execution_packet(packet)
    launch_report_path = _text_field(packet, "launch_report_path")
    primary_evidence_acceptance_report_path = _text_or_none(
        packet.get("primary_evidence_acceptance_report_path")
    )
    execution_packet_path = _text_or_none(packet.get("execution_packet_path"))
    current = real_experiment_execution_packet(
        load_real_experiment_external_artifact_launch_report(launch_report_path),
        launch_report_path=launch_report_path,
        primary_evidence_acceptance_report_path=(
            primary_evidence_acceptance_report_path
        ),
        execution_packet_path=execution_packet_path,
    )
    saved_digest = _text_or_none(packet.get("packet_digest"))
    current_digest = _text_or_none(current.get("packet_digest"))
    checks = [
        {
            "name": "execution_packet_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        {
            "name": "execution_packet_digest_matches_current",
            "passed": saved_digest == current_digest,
            "expected": saved_digest,
            "actual": current_digest,
        },
        {
            "name": "execution_packet_payload_matches_current",
            "passed": _json_value(packet) == _json_value(current),
            "expected": _json_value(packet),
            "actual": _json_value(current),
        },
    ]
    return {
        "action": "compare_real_experiment_execution_packet",
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def real_experiment_execution_packet(
    launch_report: Mapping[str, Any],
    *,
    launch_report_path: str | Path,
    primary_evidence_acceptance_report_path: str | Path | None = None,
    execution_packet_path: str | Path | None = None,
) -> dict[str, Any]:
    validation = validate_real_experiment_external_artifact_launch_report(
        launch_report
    )
    comparison = compare_real_experiment_external_artifact_launch_report(
        launch_report
    )
    launch_report_path_text = str(launch_report_path)
    packet_path_text = str(
        execution_packet_path
        if execution_packet_path is not None
        else Path(launch_report_path_text).with_name(
            "real-experiment-execution-packet.json"
        )
    )
    contracts_path = _text_or_none(launch_report.get("contracts_path"))
    run_manifest_path = _text_field(launch_report, "run_manifest_path")
    next_commands = _mapping(launch_report.get("next_commands"), "next_commands")
    acceptance_report_path = _execution_packet_primary_evidence_acceptance_path(
        launch_report_path=launch_report_path_text,
        primary_evidence_acceptance_report_path=(
            primary_evidence_acceptance_report_path
        ),
    )
    primary_acceptance = _execution_packet_primary_evidence_acceptance(
        acceptance_report_path
    )
    primary_gate = _mapping(
        launch_report.get("primary_evidence_receipt_gate"),
        "primary_evidence_receipt_gate",
    )
    ready_to_execute = (
        validation["valid"] is True
        and comparison["matches"] is True
        and launch_report.get("ready_to_run") is True
        and primary_acceptance["present"] is True
        and primary_acceptance["valid"] is True
        and primary_acceptance["matches_current"] is True
        and primary_acceptance["ready_for_launch_refresh"] is True
        and primary_acceptance["all_tracks_accepted"] is True
    )
    packet: dict[str, Any] = {
        "schema_version": REAL_EXPERIMENT_EXECUTION_PACKET_SCHEMA_VERSION,
        "action": "real_experiment_execution_packet",
        "launch_report_path": launch_report_path_text,
        "launch_report_digest": _text_or_none(launch_report.get("report_digest")),
        "contracts_path": contracts_path,
        "contracts_digest": _text_or_none(launch_report.get("contracts_digest")),
        "run_manifest_path": run_manifest_path,
        "execution_packet_path": packet_path_text,
        "primary_evidence_acceptance_report_path": str(acceptance_report_path),
        "primary_evidence_acceptance": primary_acceptance,
        "ready_to_execute": ready_to_execute,
        "execution_blocked": not ready_to_execute,
        "readiness": {
            "launch_report_valid": validation["valid"] is True,
            "launch_report_matches_current": comparison["matches"] is True,
            "preflight_ready_to_run": (
                launch_report.get("preflight_ready_to_run") is True
            ),
            "primary_evidence_acceptance_report_matches_current": (
                primary_acceptance["matches_current"] is True
            ),
            "primary_evidence_acceptance_report_present": (
                primary_acceptance["present"] is True
            ),
            "primary_evidence_acceptance_report_ready": (
                primary_acceptance["ready_for_launch_refresh"] is True
                and primary_acceptance["all_tracks_accepted"] is True
            ),
            "primary_evidence_acceptance_report_valid": (
                primary_acceptance["valid"] is True
            ),
            "primary_evidence_receipt_gate_ready": primary_gate.get("ready") is True,
            "ready_to_run": launch_report.get("ready_to_run") is True,
        },
        "blocker_summary": _json_value(primary_gate),
        "audit_commands": _execution_packet_audit_commands(
            launch_report_path=launch_report_path_text,
            contracts_path=contracts_path,
            primary_evidence_acceptance_report_path=str(acceptance_report_path),
        ),
        "execution_commands": _execution_packet_execution_commands(
            next_commands,
            ready_to_execute=ready_to_execute,
            approved_execution_packet_path=packet_path_text,
        ),
        "validation": {
            "valid": validation["valid"],
            "report_digest": validation["report_digest"],
        },
        "comparison": {
            "matches": comparison["matches"],
            "saved_digest": comparison["saved_digest"],
            "current_digest": comparison["current_digest"],
        },
    }
    packet["packet_digest"] = real_experiment_execution_packet_digest(packet)
    return packet


def real_experiment_execution_receipt_digest(receipt: Mapping[str, Any]) -> str:
    payload = {
        key: value for key, value in receipt.items() if key != "receipt_digest"
    }
    return hashlib.sha256(
        json.dumps(
            _json_value(payload),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def real_experiment_execution_receipt_json(receipt: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(receipt), indent=2, sort_keys=True) + "\n"


def save_real_experiment_execution_receipt(
    receipt: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        real_experiment_execution_receipt_json(receipt),
        encoding="utf-8",
    )
    return output_path


def load_real_experiment_execution_receipt(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Real experiment execution receipt JSON must be an object")
    schema_version = payload.get("schema_version")
    if schema_version != REAL_EXPERIMENT_EXECUTION_RECEIPT_SCHEMA_VERSION:
        raise SpatialQAError(
            "Unsupported real experiment execution receipt schema version: "
            f"{schema_version}"
        )
    return cast(dict[str, Any], payload)


def validate_real_experiment_execution_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = receipt.get("schema_version")
    action = receipt.get("action")
    receipt_digest = _text_or_none(receipt.get("receipt_digest"))
    expected_digest = real_experiment_execution_receipt_digest(receipt)
    ready_to_review = receipt.get("ready_to_review")
    packet_ready = receipt.get("execution_packet_ready_to_execute")
    summary = _mapping(receipt.get("artifact_summary"), "artifact_summary")
    artifacts = _mapping_sequence(receipt.get("artifacts"))
    artifact_count = _summary_int(summary, "artifact_count")
    ready_artifact_count = _summary_int(summary, "ready_artifact_count")
    missing_artifact_count = _summary_int(summary, "missing_artifact_count")
    invalid_artifact_count = _summary_int(summary, "invalid_artifact_count")
    expected_ready = (
        packet_ready is True
        and ready_artifact_count == len(artifacts)
        and missing_artifact_count == 0
        and invalid_artifact_count == 0
    )
    checks = [
        {
            "name": "schema_version",
            "passed": (
                schema_version == REAL_EXPERIMENT_EXECUTION_RECEIPT_SCHEMA_VERSION
            ),
            "expected": REAL_EXPERIMENT_EXECUTION_RECEIPT_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "action",
            "passed": action == "real_experiment_execution_receipt",
            "expected": "real_experiment_execution_receipt",
            "actual": action,
        },
        {
            "name": "receipt_digest",
            "passed": receipt_digest == expected_digest,
            "expected": expected_digest,
            "actual": receipt_digest,
        },
        {
            "name": "execution_packet_path_present",
            "passed": _text_or_none(receipt.get("execution_packet_path")) is not None,
            "expected": True,
            "actual": _text_or_none(receipt.get("execution_packet_path")) is not None,
        },
        {
            "name": "run_manifest_path_present",
            "passed": _text_or_none(receipt.get("run_manifest_path")) is not None,
            "expected": True,
            "actual": _text_or_none(receipt.get("run_manifest_path")) is not None,
        },
        {
            "name": "ready_to_review_bool",
            "passed": isinstance(ready_to_review, bool),
            "expected": "bool",
            "actual": type(ready_to_review).__name__,
        },
        {
            "name": "artifact_count",
            "passed": artifact_count == len(artifacts),
            "expected": len(artifacts),
            "actual": artifact_count,
        },
        {
            "name": "artifact_summary_counts",
            "passed": (
                ready_artifact_count is not None
                and missing_artifact_count is not None
                and invalid_artifact_count is not None
                and (
                    ready_artifact_count
                    + missing_artifact_count
                    + invalid_artifact_count
                    == len(artifacts)
                )
            ),
            "expected": len(artifacts),
            "actual": {
                "invalid_artifact_count": invalid_artifact_count,
                "missing_artifact_count": missing_artifact_count,
                "ready_artifact_count": ready_artifact_count,
            },
        },
        {
            "name": "ready_to_review_consistent",
            "passed": ready_to_review is expected_ready,
            "expected": expected_ready,
            "actual": ready_to_review,
        },
    ]
    return {
        "action": "validate_real_experiment_execution_receipt",
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "receipt_digest": receipt_digest,
        "checks": checks,
    }


def compare_real_experiment_execution_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_real_experiment_execution_receipt(receipt)
    packet_path = _text_field(receipt, "execution_packet_path")
    current = real_experiment_execution_receipt(
        load_real_experiment_execution_packet(packet_path),
        execution_packet_path=packet_path,
    )
    saved_digest = _text_or_none(receipt.get("receipt_digest"))
    current_digest = _text_or_none(current.get("receipt_digest"))
    checks = [
        {
            "name": "execution_receipt_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        {
            "name": "execution_receipt_digest_matches_current",
            "passed": saved_digest == current_digest,
            "expected": saved_digest,
            "actual": current_digest,
        },
        {
            "name": "execution_receipt_payload_matches_current",
            "passed": _json_value(receipt) == _json_value(current),
            "expected": _json_value(receipt),
            "actual": _json_value(current),
        },
    ]
    return {
        "action": "compare_real_experiment_execution_receipt",
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def real_experiment_execution_receipt(
    execution_packet: Mapping[str, Any],
    *,
    execution_packet_path: str | Path,
) -> dict[str, Any]:
    packet_validation = validate_real_experiment_execution_packet(execution_packet)
    run_manifest_path = _text_field(execution_packet, "run_manifest_path")
    run_manifest = load_real_experiment_run_manifest(run_manifest_path)
    artifacts = _execution_receipt_artifacts(run_manifest)
    ready_artifact_count = sum(
        1 for artifact in artifacts if artifact["status"] == "ready"
    )
    missing_artifact_count = sum(
        1 for artifact in artifacts if artifact["status"] == "missing"
    )
    invalid_artifact_count = sum(
        1 for artifact in artifacts if artifact["status"] == "invalid"
    )
    run_ledger_approval = _execution_receipt_run_ledger_approval(run_manifest)
    ready_to_review = (
        packet_validation["valid"] is True
        and execution_packet.get("ready_to_execute") is True
        and ready_artifact_count == len(artifacts)
        and run_ledger_approval["ready"] is True
    )
    receipt: dict[str, Any] = {
        "schema_version": REAL_EXPERIMENT_EXECUTION_RECEIPT_SCHEMA_VERSION,
        "action": "real_experiment_execution_receipt",
        "execution_packet_path": str(execution_packet_path),
        "execution_packet_digest": _text_or_none(
            execution_packet.get("packet_digest")
        ),
        "execution_packet_ready_to_execute": (
            execution_packet.get("ready_to_execute") is True
        ),
        "run_manifest_path": run_manifest_path,
        "run_manifest_digest": real_experiment_run_manifest_digest(run_manifest),
        "ready_to_review": ready_to_review,
        "artifact_summary": {
            "artifact_count": len(artifacts),
            "invalid_artifact_count": invalid_artifact_count,
            "missing_artifact_count": missing_artifact_count,
            "ready_artifact_count": ready_artifact_count,
        },
        "run_ledger_approval": run_ledger_approval,
        "artifacts": artifacts,
        "packet_validation": {
            "valid": packet_validation["valid"],
            "packet_digest": packet_validation["packet_digest"],
        },
    }
    receipt["receipt_digest"] = real_experiment_execution_receipt_digest(receipt)
    return receipt


def real_experiment_smoke_run_checklist_digest(
    checklist: Mapping[str, Any],
) -> str:
    payload = {
        key: value for key, value in checklist.items() if key != "checklist_digest"
    }
    return hashlib.sha256(
        json.dumps(
            _json_value(payload),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def real_experiment_smoke_run_checklist_json(
    checklist: Mapping[str, Any],
) -> str:
    return json.dumps(_json_value(checklist), indent=2, sort_keys=True) + "\n"


def save_real_experiment_smoke_run_checklist(
    checklist: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        real_experiment_smoke_run_checklist_json(checklist),
        encoding="utf-8",
    )
    return output_path


def load_real_experiment_smoke_run_checklist(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Real experiment smoke run checklist JSON must be an object")
    schema_version = payload.get("schema_version")
    if schema_version != REAL_EXPERIMENT_SMOKE_RUN_CHECKLIST_SCHEMA_VERSION:
        raise SpatialQAError(
            "Unsupported real experiment smoke run checklist schema version: "
            f"{schema_version}"
        )
    return cast(dict[str, Any], payload)


def validate_real_experiment_smoke_run_checklist(
    checklist: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = checklist.get("schema_version")
    action = checklist.get("action")
    checklist_digest = _text_or_none(checklist.get("checklist_digest"))
    expected_digest = real_experiment_smoke_run_checklist_digest(checklist)
    ready_to_start = checklist.get("ready_to_start")
    blocked = checklist.get("blocked")
    summary = _mapping(checklist.get("summary"), "summary")
    steps = _mapping_sequence(checklist.get("steps"))
    step_count = _summary_int(summary, "step_count")
    audit_step_count = _summary_int(summary, "audit_step_count")
    execute_step_count = _summary_int(summary, "execute_step_count")
    review_step_count = _summary_int(summary, "review_step_count")
    required_step_count = _summary_int(summary, "required_step_count")
    phase_counts = {
        "audit": sum(1 for step in steps if step.get("phase") == "audit"),
        "execute": sum(1 for step in steps if step.get("phase") == "execute"),
        "review": sum(1 for step in steps if step.get("phase") == "review"),
    }
    required_steps = sum(1 for step in steps if step.get("required") is True)
    step_orders = [step.get("order") for step in steps]
    expected_orders = list(range(1, len(steps) + 1))
    checks = [
        {
            "name": "schema_version",
            "passed": (
                schema_version == REAL_EXPERIMENT_SMOKE_RUN_CHECKLIST_SCHEMA_VERSION
            ),
            "expected": REAL_EXPERIMENT_SMOKE_RUN_CHECKLIST_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "action",
            "passed": action == "real_experiment_smoke_run_checklist",
            "expected": "real_experiment_smoke_run_checklist",
            "actual": action,
        },
        {
            "name": "checklist_digest",
            "passed": checklist_digest == expected_digest,
            "expected": expected_digest,
            "actual": checklist_digest,
        },
        {
            "name": "execution_packet_path_present",
            "passed": _text_or_none(checklist.get("execution_packet_path"))
            is not None,
            "expected": True,
            "actual": _text_or_none(checklist.get("execution_packet_path"))
            is not None,
        },
        {
            "name": "execution_receipt_output_path_present",
            "passed": _text_or_none(checklist.get("execution_receipt_output_path"))
            is not None,
            "expected": True,
            "actual": _text_or_none(checklist.get("execution_receipt_output_path"))
            is not None,
        },
        {
            "name": "ready_to_start_bool",
            "passed": isinstance(ready_to_start, bool),
            "expected": "bool",
            "actual": type(ready_to_start).__name__,
        },
        {
            "name": "blocked_bool",
            "passed": isinstance(blocked, bool),
            "expected": "bool",
            "actual": type(blocked).__name__,
        },
        {
            "name": "blocked_consistent",
            "passed": (
                isinstance(ready_to_start, bool)
                and isinstance(blocked, bool)
                and blocked is (not ready_to_start)
            ),
            "expected": True,
            "actual": {
                "blocked": blocked,
                "ready_to_start": ready_to_start,
            },
        },
        {
            "name": "step_count",
            "passed": step_count == len(steps),
            "expected": len(steps),
            "actual": step_count,
        },
        {
            "name": "phase_step_counts",
            "passed": (
                audit_step_count == phase_counts["audit"]
                and execute_step_count == phase_counts["execute"]
                and review_step_count == phase_counts["review"]
            ),
            "expected": phase_counts,
            "actual": {
                "audit": audit_step_count,
                "execute": execute_step_count,
                "review": review_step_count,
            },
        },
        {
            "name": "required_step_count",
            "passed": required_step_count == required_steps,
            "expected": required_steps,
            "actual": required_step_count,
        },
        {
            "name": "step_order",
            "passed": step_orders == expected_orders,
            "expected": expected_orders,
            "actual": step_orders,
        },
    ]
    return {
        "action": "validate_real_experiment_smoke_run_checklist",
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "checklist_digest": checklist_digest,
        "checks": checks,
    }


def compare_real_experiment_smoke_run_checklist(
    checklist: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_real_experiment_smoke_run_checklist(checklist)
    packet_path = _text_field(checklist, "execution_packet_path")
    receipt_output_path = _text_field(checklist, "execution_receipt_output_path")
    current = real_experiment_smoke_run_checklist(
        load_real_experiment_execution_packet(packet_path),
        execution_packet_path=packet_path,
        execution_receipt_output_path=receipt_output_path,
    )
    saved_digest = _text_or_none(checklist.get("checklist_digest"))
    current_digest = _text_or_none(current.get("checklist_digest"))
    checks = [
        {
            "name": "smoke_run_checklist_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        {
            "name": "smoke_run_checklist_digest_matches_current",
            "passed": saved_digest == current_digest,
            "expected": saved_digest,
            "actual": current_digest,
        },
        {
            "name": "smoke_run_checklist_payload_matches_current",
            "passed": _json_value(checklist) == _json_value(current),
            "expected": _json_value(checklist),
            "actual": _json_value(current),
        },
    ]
    return {
        "action": "compare_real_experiment_smoke_run_checklist",
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def real_experiment_smoke_run_runbook_digest(
    runbook: Mapping[str, Any],
) -> str:
    payload = {key: value for key, value in runbook.items() if key != "runbook_digest"}
    return hashlib.sha256(
        json.dumps(
            _json_value(payload),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def real_experiment_smoke_run_runbook_json(runbook: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(runbook), indent=2, sort_keys=True) + "\n"


def save_real_experiment_smoke_run_runbook(
    runbook: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        real_experiment_smoke_run_runbook_json(runbook),
        encoding="utf-8",
    )
    return output_path


def load_real_experiment_smoke_run_runbook(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Real experiment smoke run runbook JSON must be an object")
    schema_version = payload.get("schema_version")
    if schema_version != REAL_EXPERIMENT_SMOKE_RUN_RUNBOOK_SCHEMA_VERSION:
        raise SpatialQAError(
            "Unsupported real experiment smoke run runbook schema version: "
            f"{schema_version}"
        )
    return cast(dict[str, Any], payload)


def real_experiment_smoke_run_runbook(
    checklist: Mapping[str, Any],
    *,
    smoke_run_checklist_path: str | Path,
) -> dict[str, Any]:
    checklist_validation = validate_real_experiment_smoke_run_checklist(checklist)
    commands = _smoke_run_runbook_commands(checklist)
    runbook: dict[str, Any] = {
        "schema_version": REAL_EXPERIMENT_SMOKE_RUN_RUNBOOK_SCHEMA_VERSION,
        "action": "real_experiment_smoke_run_runbook",
        "smoke_run_checklist_path": str(smoke_run_checklist_path),
        "smoke_run_checklist_digest": _text_or_none(
            checklist.get("checklist_digest")
        ),
        "execution_packet_path": _text_field(checklist, "execution_packet_path"),
        "execution_packet_digest": _text_or_none(
            checklist.get("execution_packet_digest")
        ),
        "execution_receipt_output_path": _text_field(
            checklist,
            "execution_receipt_output_path",
        ),
        "run_manifest_path": _text_field(checklist, "run_manifest_path"),
        "run_manifest_digest": _text_or_none(checklist.get("run_manifest_digest")),
        "ready_to_start": checklist.get("ready_to_start") is True,
        "blocked": checklist.get("blocked") is True,
        "blocker_summary": _json_value(
            _mapping_or_empty(checklist.get("blocker_summary"))
        ),
        "summary": _smoke_run_runbook_summary(commands),
        "commands": commands,
        "planned_outputs": _json_value(
            _mapping_or_empty(checklist.get("planned_outputs"))
        ),
        "checklist_validation": {
            "valid": checklist_validation["valid"],
            "checklist_digest": checklist_validation["checklist_digest"],
        },
    }
    runbook["runbook_digest"] = real_experiment_smoke_run_runbook_digest(runbook)
    return runbook


def validate_real_experiment_smoke_run_runbook(
    runbook: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = runbook.get("schema_version")
    action = runbook.get("action")
    runbook_digest = _text_or_none(runbook.get("runbook_digest"))
    expected_digest = real_experiment_smoke_run_runbook_digest(runbook)
    ready_to_start = runbook.get("ready_to_start")
    blocked = runbook.get("blocked")
    summary = _mapping(runbook.get("summary"), "summary")
    commands = _mapping_sequence(runbook.get("commands"))
    phase_counts = {
        "audit": sum(1 for command in commands if command.get("phase") == "audit"),
        "execute": sum(
            1 for command in commands if command.get("phase") == "execute"
        ),
        "review": sum(
            1 for command in commands if command.get("phase") == "review"
        ),
    }
    required_count = sum(1 for command in commands if command.get("required") is True)
    command_orders = [command.get("order") for command in commands]
    expected_orders = list(range(1, len(commands) + 1))
    command_rows_valid = all(
        _text_or_none(command.get("key")) is not None
        and _text_or_none(command.get("command")) is not None
        and _text_or_none(command.get("phase")) in {"audit", "execute", "review"}
        for command in commands
    )
    checklist_validation = _mapping_or_empty(runbook.get("checklist_validation"))
    checks = [
        {
            "name": "schema_version",
            "passed": (
                schema_version == REAL_EXPERIMENT_SMOKE_RUN_RUNBOOK_SCHEMA_VERSION
            ),
            "expected": REAL_EXPERIMENT_SMOKE_RUN_RUNBOOK_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "action",
            "passed": action == "real_experiment_smoke_run_runbook",
            "expected": "real_experiment_smoke_run_runbook",
            "actual": action,
        },
        {
            "name": "runbook_digest",
            "passed": runbook_digest == expected_digest,
            "expected": expected_digest,
            "actual": runbook_digest,
        },
        {
            "name": "smoke_run_checklist_path_present",
            "passed": _text_or_none(runbook.get("smoke_run_checklist_path"))
            is not None,
            "expected": True,
            "actual": _text_or_none(runbook.get("smoke_run_checklist_path"))
            is not None,
        },
        {
            "name": "ready_to_start_bool",
            "passed": isinstance(ready_to_start, bool),
            "expected": "bool",
            "actual": type(ready_to_start).__name__,
        },
        {
            "name": "blocked_bool",
            "passed": isinstance(blocked, bool),
            "expected": "bool",
            "actual": type(blocked).__name__,
        },
        {
            "name": "blocked_consistent",
            "passed": (
                isinstance(ready_to_start, bool)
                and isinstance(blocked, bool)
                and blocked is (not ready_to_start)
            ),
            "expected": True,
            "actual": {
                "blocked": blocked,
                "ready_to_start": ready_to_start,
            },
        },
        {
            "name": "command_count",
            "passed": _summary_int(summary, "command_count") == len(commands),
            "expected": len(commands),
            "actual": _summary_int(summary, "command_count"),
        },
        {
            "name": "phase_command_counts",
            "passed": (
                _summary_int(summary, "audit_command_count")
                == phase_counts["audit"]
                and _summary_int(summary, "execute_command_count")
                == phase_counts["execute"]
                and _summary_int(summary, "review_command_count")
                == phase_counts["review"]
            ),
            "expected": phase_counts,
            "actual": {
                "audit": _summary_int(summary, "audit_command_count"),
                "execute": _summary_int(summary, "execute_command_count"),
                "review": _summary_int(summary, "review_command_count"),
            },
        },
        {
            "name": "required_command_count",
            "passed": _summary_int(summary, "required_command_count")
            == required_count,
            "expected": required_count,
            "actual": _summary_int(summary, "required_command_count"),
        },
        {
            "name": "command_order",
            "passed": command_orders == expected_orders,
            "expected": expected_orders,
            "actual": command_orders,
        },
        {
            "name": "command_rows",
            "passed": command_rows_valid,
            "expected": True,
            "actual": command_rows_valid,
        },
        {
            "name": "checklist_validation_valid",
            "passed": checklist_validation.get("valid") is True,
            "expected": True,
            "actual": checklist_validation.get("valid"),
        },
    ]
    return {
        "action": "validate_real_experiment_smoke_run_runbook",
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "runbook_digest": runbook_digest,
        "checks": checks,
    }


def compare_real_experiment_smoke_run_runbook(
    runbook: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_real_experiment_smoke_run_runbook(runbook)
    checklist_path = _text_field(runbook, "smoke_run_checklist_path")
    current = real_experiment_smoke_run_runbook(
        load_real_experiment_smoke_run_checklist(checklist_path),
        smoke_run_checklist_path=checklist_path,
    )
    saved_digest = _text_or_none(runbook.get("runbook_digest"))
    current_digest = _text_or_none(current.get("runbook_digest"))
    checks = [
        {
            "name": "smoke_run_runbook_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        {
            "name": "smoke_run_runbook_digest_matches_current",
            "passed": saved_digest == current_digest,
            "expected": saved_digest,
            "actual": current_digest,
        },
        {
            "name": "smoke_run_runbook_payload_matches_current",
            "passed": _json_value(runbook) == _json_value(current),
            "expected": _json_value(runbook),
            "actual": _json_value(current),
        },
    ]
    return {
        "action": "compare_real_experiment_smoke_run_runbook",
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def real_experiment_smoke_run_checklist(
    execution_packet: Mapping[str, Any],
    *,
    execution_packet_path: str | Path,
    execution_receipt_output_path: str | Path | None = None,
) -> dict[str, Any]:
    packet_validation = validate_real_experiment_execution_packet(execution_packet)
    packet_path_text = str(execution_packet_path)
    receipt_output_path = _smoke_run_receipt_output_path(
        execution_packet_path,
        execution_receipt_output_path=execution_receipt_output_path,
    )
    run_manifest_path = _text_field(execution_packet, "run_manifest_path")
    run_manifest = load_real_experiment_run_manifest(run_manifest_path)
    ready_to_start = (
        packet_validation["valid"] is True
        and execution_packet.get("ready_to_execute") is True
    )
    steps = _smoke_run_checklist_steps(
        execution_packet,
        execution_packet_path=packet_path_text,
        execution_receipt_output_path=receipt_output_path,
        run_ledger_path=_text_or_none(
            run_manifest.get("real_experiment_run_ledger_path")
        ),
        ready_to_start=ready_to_start,
    )
    checklist: dict[str, Any] = {
        "schema_version": REAL_EXPERIMENT_SMOKE_RUN_CHECKLIST_SCHEMA_VERSION,
        "action": "real_experiment_smoke_run_checklist",
        "execution_packet_path": packet_path_text,
        "execution_packet_digest": _text_or_none(
            execution_packet.get("packet_digest")
        ),
        "execution_receipt_output_path": receipt_output_path,
        "launch_report_path": _text_or_none(execution_packet.get("launch_report_path")),
        "run_manifest_path": run_manifest_path,
        "run_manifest_digest": real_experiment_run_manifest_digest(run_manifest),
        "ready_to_start": ready_to_start,
        "blocked": not ready_to_start,
        "blocker_summary": _json_value(
            _mapping(execution_packet.get("blocker_summary"), "blocker_summary")
        ),
        "summary": _smoke_run_checklist_summary(steps),
        "steps": steps,
        "planned_outputs": _smoke_run_planned_outputs(
            run_manifest,
            execution_receipt_output_path=receipt_output_path,
        ),
        "packet_validation": {
            "valid": packet_validation["valid"],
            "packet_digest": packet_validation["packet_digest"],
        },
    }
    checklist["checklist_digest"] = real_experiment_smoke_run_checklist_digest(
        checklist
    )
    return checklist


def real_experiment_research_review_digest(review: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in review.items() if key != "review_digest"}
    return hashlib.sha256(
        json.dumps(
            _json_value(payload),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def real_experiment_research_review_json(review: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(review), indent=2, sort_keys=True) + "\n"


def save_real_experiment_research_review(
    review: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        real_experiment_research_review_json(review),
        encoding="utf-8",
    )
    return output_path


def load_real_experiment_research_review(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Real experiment research review JSON must be an object")
    schema_version = payload.get("schema_version")
    if schema_version != REAL_EXPERIMENT_RESEARCH_REVIEW_SCHEMA_VERSION:
        raise SpatialQAError(
            "Unsupported real experiment research review schema version: "
            f"{schema_version}"
        )
    return cast(dict[str, Any], payload)


def validate_real_experiment_research_review(
    review: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = review.get("schema_version")
    action = review.get("action")
    review_digest = _text_or_none(review.get("review_digest"))
    expected_digest = real_experiment_research_review_digest(review)
    ready_for_review = review.get("ready_for_research_review")
    blocked = review.get("blocked")
    blockers = _string_sequence_or_empty(review.get("blockers"))
    research_questions = _mapping_or_empty(review.get("research_questions"))
    rq_summary = _mapping_or_empty(review.get("research_question_summary"))
    available_count = sum(
        1
        for row in research_questions.values()
        if isinstance(row, Mapping) and row.get("available") is True
    )
    conclusive_count = sum(
        1
        for row in research_questions.values()
        if isinstance(row, Mapping)
        and _text_or_none(row.get("verdict"))
        in REAL_EXPERIMENT_CONCLUSIVE_VERDICTS
    )
    inconclusive_count = len(research_questions) - conclusive_count
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == REAL_EXPERIMENT_RESEARCH_REVIEW_SCHEMA_VERSION,
            "expected": REAL_EXPERIMENT_RESEARCH_REVIEW_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "action",
            "passed": action == "real_experiment_research_review",
            "expected": "real_experiment_research_review",
            "actual": action,
        },
        {
            "name": "review_digest",
            "passed": review_digest == expected_digest,
            "expected": expected_digest,
            "actual": review_digest,
        },
        {
            "name": "execution_receipt_path_present",
            "passed": _text_or_none(review.get("execution_receipt_path")) is not None,
            "expected": True,
            "actual": _text_or_none(review.get("execution_receipt_path")) is not None,
        },
        {
            "name": "ready_for_research_review_bool",
            "passed": isinstance(ready_for_review, bool),
            "expected": "bool",
            "actual": type(ready_for_review).__name__,
        },
        {
            "name": "blocked_bool",
            "passed": isinstance(blocked, bool),
            "expected": "bool",
            "actual": type(blocked).__name__,
        },
        {
            "name": "blocked_consistent",
            "passed": (
                isinstance(ready_for_review, bool)
                and isinstance(blocked, bool)
                and blocked is (not ready_for_review)
                and ready_for_review is (len(blockers) == 0)
            ),
            "expected": True,
            "actual": {
                "blocked": blocked,
                "blocker_count": len(blockers),
                "ready_for_research_review": ready_for_review,
            },
        },
        {
            "name": "research_question_count",
            "passed": set(research_questions)
            == set(REAL_EXPERIMENT_RESEARCH_QUESTION_KEYS),
            "expected": sorted(REAL_EXPERIMENT_RESEARCH_QUESTION_KEYS),
            "actual": sorted(str(key) for key in research_questions),
        },
        {
            "name": "research_question_summary_counts",
            "passed": (
                _summary_int(rq_summary, "required_count")
                == len(REAL_EXPERIMENT_RESEARCH_QUESTION_KEYS)
                and _summary_int(rq_summary, "available_count") == available_count
                and _summary_int(rq_summary, "conclusive_count")
                == conclusive_count
                and _summary_int(rq_summary, "inconclusive_count")
                == inconclusive_count
            ),
            "expected": {
                "available_count": available_count,
                "conclusive_count": conclusive_count,
                "inconclusive_count": inconclusive_count,
                "required_count": len(REAL_EXPERIMENT_RESEARCH_QUESTION_KEYS),
            },
            "actual": _json_value(rq_summary),
        },
    ]
    return {
        "action": "validate_real_experiment_research_review",
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "review_digest": review_digest,
        "checks": checks,
    }


def compare_real_experiment_research_review(
    review: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_real_experiment_research_review(review)
    receipt_path = _text_field(review, "execution_receipt_path")
    current = real_experiment_research_review(
        load_real_experiment_execution_receipt(receipt_path),
        execution_receipt_path=receipt_path,
    )
    saved_digest = _text_or_none(review.get("review_digest"))
    current_digest = _text_or_none(current.get("review_digest"))
    checks = [
        {
            "name": "research_review_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        {
            "name": "research_review_digest_matches_current",
            "passed": saved_digest == current_digest,
            "expected": saved_digest,
            "actual": current_digest,
        },
        {
            "name": "research_review_payload_matches_current",
            "passed": _json_value(review) == _json_value(current),
            "expected": _json_value(review),
            "actual": _json_value(current),
        },
    ]
    return {
        "action": "compare_real_experiment_research_review",
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def real_experiment_research_review(
    execution_receipt: Mapping[str, Any],
    *,
    execution_receipt_path: str | Path,
) -> dict[str, Any]:
    receipt_validation = validate_real_experiment_execution_receipt(
        execution_receipt
    )
    summary_path = _execution_receipt_artifact_path(
        execution_receipt,
        "experiment_summary",
    )
    record_path = _execution_receipt_artifact_path(
        execution_receipt,
        "experiment_record",
    )
    summary_audit, summary_report = _research_review_json_artifact(
        role="experiment_summary",
        path=summary_path,
        digest_field="report_digest",
        digest_fn=experiment_summary_report_digest,
        load_fn=load_experiment_summary_report,
        validate_fn=validate_experiment_summary_report,
    )
    record_audit, record = _research_review_json_artifact(
        role="experiment_record",
        path=record_path,
        digest_field="record_digest",
        digest_fn=experiment_record_digest,
        load_fn=load_experiment_record,
        validate_fn=validate_experiment_record,
    )
    research_questions = _research_review_questions(summary_report, record)
    evidence_summary = _research_review_evidence_summary(summary_report, record)
    blockers = _research_review_blockers(
        execution_receipt,
        receipt_validation=receipt_validation,
        summary_audit=summary_audit,
        record_audit=record_audit,
        research_questions=research_questions,
        evidence_summary=evidence_summary,
        record=record,
        summary_report=summary_report,
    )
    ready_for_review = len(blockers) == 0
    review: dict[str, Any] = {
        "schema_version": REAL_EXPERIMENT_RESEARCH_REVIEW_SCHEMA_VERSION,
        "action": "real_experiment_research_review",
        "execution_receipt_path": str(execution_receipt_path),
        "execution_receipt_digest": _text_or_none(
            execution_receipt.get("receipt_digest")
        ),
        "execution_receipt_ready_to_review": (
            execution_receipt.get("ready_to_review") is True
        ),
        "ready_for_research_review": ready_for_review,
        "blocked": not ready_for_review,
        "blockers": blockers,
        "artifacts": [summary_audit, record_audit],
        "research_question_summary": _research_review_question_summary(
            research_questions
        ),
        "research_questions": research_questions,
        "verdict_counts": _json_value(
            _mapping_or_empty(
                record.get("verdict_counts") if record is not None else None
            )
        ),
        "evidence_summary": evidence_summary,
        "receipt_validation": {
            "valid": receipt_validation["valid"],
            "receipt_digest": receipt_validation["receipt_digest"],
        },
    }
    review["review_digest"] = real_experiment_research_review_digest(review)
    return review


def real_experiment_claim_readiness_digest(
    claim_readiness: Mapping[str, Any],
) -> str:
    payload = {
        key: value
        for key, value in claim_readiness.items()
        if key != "claim_readiness_digest"
    }
    return hashlib.sha256(
        json.dumps(
            _json_value(payload),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def real_experiment_claim_readiness_json(
    claim_readiness: Mapping[str, Any],
) -> str:
    return json.dumps(_json_value(claim_readiness), indent=2, sort_keys=True) + "\n"


def save_real_experiment_claim_readiness(
    claim_readiness: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        real_experiment_claim_readiness_json(claim_readiness),
        encoding="utf-8",
    )
    return output_path


def load_real_experiment_claim_readiness(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Real experiment claim readiness JSON must be an object")
    schema_version = payload.get("schema_version")
    if schema_version != REAL_EXPERIMENT_CLAIM_READINESS_SCHEMA_VERSION:
        raise SpatialQAError(
            "Unsupported real experiment claim readiness schema version: "
            f"{schema_version}"
        )
    return cast(dict[str, Any], payload)


def validate_real_experiment_claim_readiness(
    claim_readiness: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = claim_readiness.get("schema_version")
    action = claim_readiness.get("action")
    digest = _text_or_none(claim_readiness.get("claim_readiness_digest"))
    expected_digest = real_experiment_claim_readiness_digest(claim_readiness)
    claim_ready = claim_readiness.get("claim_ready")
    status = _text_or_none(claim_readiness.get("status"))
    blockers = _mapping_sequence(claim_readiness.get("blockers"))
    checks = _mapping_sequence(claim_readiness.get("checks"))
    thresholds = _mapping_or_empty(claim_readiness.get("thresholds"))
    scale_summary = _mapping_or_empty(claim_readiness.get("scale_summary"))
    claim_gap_summary = _mapping_or_empty(claim_readiness.get("claim_gap_summary"))
    claim_scope_assessment = _mapping_or_empty(
        claim_readiness.get("claim_scope_assessment")
    )
    claim_scope_next_actions = _mapping_sequence(
        claim_readiness.get("claim_scope_next_actions")
    )
    claim_scope_handoff_plan = _mapping_or_empty(
        claim_readiness.get("claim_scope_handoff_plan")
    )
    next_handoff_plan = _mapping_or_empty(
        claim_readiness.get("next_handoff_plan")
    )
    next_actions = _mapping_sequence(claim_readiness.get("next_actions"))
    episode_collection_plan = _mapping_or_empty(
        next_handoff_plan.get("episode_collection_plan")
    )
    external_artifact_slots = _mapping_or_empty(
        next_handoff_plan.get("external_artifact_slots")
    )
    offline_control_prediction_paths = _mapping_or_empty(
        external_artifact_slots.get("offline_control_prediction_paths")
    )
    next_handoff_required_predicted_input_kinds = _string_sequence_or_empty(
        next_handoff_plan.get("required_predicted_input_kinds")
    )
    next_handoff_commands = _mapping_or_empty(next_handoff_plan.get("commands"))
    next_handoff_write_command = _text_or_none(
        next_handoff_commands.get("write_handoff_manifests")
    )
    after_write_intake_plan = _mapping_or_empty(
        next_handoff_plan.get("after_write_intake_plan")
    )
    after_write_artifact_paths = _mapping_or_empty(
        after_write_intake_plan.get("artifact_paths")
    )
    after_write_commands = _mapping_or_empty(
        after_write_intake_plan.get("commands")
    )
    next_run_review_plan = _mapping_or_empty(
        next_handoff_plan.get("next_run_review_plan")
    )
    next_run_review_artifact_paths = _mapping_or_empty(
        next_run_review_plan.get("artifact_paths")
    )
    next_run_review_commands = _mapping_or_empty(
        next_run_review_plan.get("commands")
    )
    operator_checklist = _mapping_or_empty(
        next_handoff_plan.get("operator_checklist")
    )
    operator_steps = _mapping_sequence(operator_checklist.get("steps"))
    operator_step_keys = [_text_field(step, "key") for step in operator_steps]
    failed_checks = [check for check in checks if check.get("passed") is not True]
    scale_deficits = _mapping_or_empty(claim_gap_summary.get("scale_deficits"))
    research_question_gaps = _mapping_or_empty(
        claim_gap_summary.get("research_question_gaps")
    )
    research_question_verdicts = _mapping_or_empty(
        claim_readiness.get("research_question_verdicts")
    )
    claim_conclusion_summary = _mapping_or_empty(
        claim_readiness.get("claim_conclusion_summary")
    )
    claim_conclusion_evidence = _mapping_or_empty(
        claim_readiness.get("claim_conclusion_evidence")
    )
    claim_effect_matrix = _mapping_sequence(
        claim_readiness.get("claim_effect_matrix")
    )
    claim_effect_direction_summary = _mapping_or_empty(
        claim_readiness.get("claim_effect_direction_summary")
    )
    claim_hypothesis_assessment = _mapping_or_empty(
        claim_readiness.get("claim_hypothesis_assessment")
    )
    research_question_missing_keys = _string_sequence_or_empty(
        research_question_gaps.get("missing_keys")
    )
    research_question_inconclusive_keys = _string_sequence_or_empty(
        research_question_gaps.get("inconclusive_keys")
    )
    research_question_gap_count = len(research_question_missing_keys) + len(
        research_question_inconclusive_keys
    )
    expected_research_question_gap_action = (
        _claim_research_question_gap_action_matches(
            next_actions,
            research_question_gaps=research_question_gaps,
        )
    )
    expected_claim_scope_assessment = _claim_scope_assessment(
        scale_summary,
        thresholds,
        claim_ready=claim_ready is True,
    )
    expected_claim_scope_next_actions = _claim_scope_next_actions(
        expected_claim_scope_assessment,
        claim_ready=claim_ready is True,
        scale_summary=scale_summary,
    )
    expected_claim_scope_handoff_plan_valid = _claim_scope_handoff_plan_matches(
        claim_scope_handoff_plan,
        claim_scope_assessment=expected_claim_scope_assessment,
        claim_scope_next_actions=expected_claim_scope_next_actions,
        source_run_manifest_digest=_text_or_none(
            next_handoff_plan.get("source_run_manifest_digest")
        ),
        source_run_manifest_path=_text_or_none(
            next_handoff_plan.get("source_run_manifest_path")
        ),
        source_dataset_name=_text_or_none(next_handoff_plan.get("dataset_name")),
        source_current_handoff_thresholds=_mapping_or_empty(
            next_handoff_plan.get("current_handoff_thresholds")
        ),
        source_offline_control_prediction_paths=offline_control_prediction_paths,
        source_required_predicted_input_kinds=(
            next_handoff_required_predicted_input_kinds
        ),
    )
    expected_claim_conclusion_summary = _claim_conclusion_summary(
        research_question_verdicts,
        claim_ready=claim_ready is True,
        research_question_gaps=research_question_gaps,
    )
    expected_claim_effect_matrix = _claim_effect_matrix(
        claim_conclusion_evidence
    )
    expected_claim_effect_direction_summary = _claim_effect_direction_summary(
        claim_effect_matrix
    )
    expected_claim_hypothesis_assessment = _claim_hypothesis_assessment(
        expected_claim_conclusion_summary,
        expected_claim_effect_direction_summary,
    )
    checks_result = [
        {
            "name": "schema_version",
            "passed": schema_version == REAL_EXPERIMENT_CLAIM_READINESS_SCHEMA_VERSION,
            "expected": REAL_EXPERIMENT_CLAIM_READINESS_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "action",
            "passed": action == "real_experiment_claim_readiness",
            "expected": "real_experiment_claim_readiness",
            "actual": action,
        },
        {
            "name": "claim_readiness_digest",
            "passed": digest == expected_digest,
            "expected": expected_digest,
            "actual": digest,
        },
        {
            "name": "research_review_path_present",
            "passed": _text_or_none(claim_readiness.get("research_review_path"))
            is not None,
            "expected": True,
            "actual": _text_or_none(claim_readiness.get("research_review_path"))
            is not None,
        },
        {
            "name": "claim_ready_bool",
            "passed": isinstance(claim_ready, bool),
            "expected": "bool",
            "actual": type(claim_ready).__name__,
        },
        {
            "name": "status",
            "passed": status in {"claim_ready", "pilot_only"},
            "expected": ["claim_ready", "pilot_only"],
            "actual": status,
        },
        {
            "name": "status_consistent",
            "passed": (
                (claim_ready is True and status == "claim_ready" and not blockers)
                or (
                    claim_ready is False
                    and status == "pilot_only"
                    and len(blockers) == len(failed_checks)
                )
            ),
            "expected": True,
            "actual": {
                "blocker_count": len(blockers),
                "claim_ready": claim_ready,
                "failed_check_count": len(failed_checks),
                "status": status,
            },
        },
        {
            "name": "thresholds",
            "passed": (
                _summary_int(thresholds, "min_episode_count") is not None
                and _summary_int(thresholds, "min_scene_count") is not None
                and _summary_int(thresholds, "min_qa_count") is not None
                and _summary_int(thresholds, "min_dynamic_qa_count") is not None
            ),
            "expected": [
                "min_dynamic_qa_count",
                "min_episode_count",
                "min_qa_count",
                "min_scene_count",
            ],
            "actual": sorted(str(key) for key in thresholds),
        },
        {
            "name": "scale_summary",
            "passed": (
                _summary_int(scale_summary, "episode_count") is not None
                and _summary_int(scale_summary, "scene_count") is not None
                and _summary_int(scale_summary, "qa_count") is not None
                and _summary_int(scale_summary, "dynamic_qa_count") is not None
            ),
            "expected": [
                "dynamic_qa_count",
                "episode_count",
                "qa_count",
                "scene_count",
            ],
            "actual": sorted(str(key) for key in scale_summary),
        },
        {
            "name": "claim_gap_summary",
            "passed": (
                _summary_int(claim_gap_summary, "failed_check_count")
                == len(failed_checks)
                and _summary_int(claim_gap_summary, "scale_deficit_count")
                == len(scale_deficits)
                and _summary_int(
                    claim_gap_summary,
                    "research_question_gap_count",
                )
                == research_question_gap_count
                and _summary_int(claim_gap_summary, "evidence_gap_count")
                == len(failed_checks) - len(scale_deficits)
                and isinstance(claim_gap_summary.get("target_thresholds"), Mapping)
                and isinstance(
                    claim_gap_summary.get("research_question_gaps"),
                    Mapping,
                )
                and isinstance(research_question_gaps.get("verdicts"), Mapping)
            ),
            "expected": {
                "evidence_gap_count": len(failed_checks) - len(scale_deficits),
                "failed_check_count": len(failed_checks),
                "research_question_gap_count": research_question_gap_count,
                "scale_deficit_count": len(scale_deficits),
                "research_question_gaps": "object",
                "target_thresholds": "object",
            },
            "actual": {
                "evidence_gap_count": _summary_int(
                    claim_gap_summary,
                    "evidence_gap_count",
                ),
                "failed_check_count": _summary_int(
                    claim_gap_summary,
                    "failed_check_count",
                ),
                "research_question_gap_count": _summary_int(
                    claim_gap_summary,
                    "research_question_gap_count",
                ),
                "research_question_gaps": type(
                    claim_gap_summary.get("research_question_gaps")
                ).__name__,
                "scale_deficit_count": _summary_int(
                    claim_gap_summary,
                    "scale_deficit_count",
                ),
                "target_thresholds": type(
                    claim_gap_summary.get("target_thresholds")
                ).__name__,
            },
        },
        {
            "name": "claim_scope_assessment",
            "passed": _json_value(claim_scope_assessment)
            == _json_value(expected_claim_scope_assessment),
            "expected": _json_value(expected_claim_scope_assessment),
            "actual": _json_value(claim_scope_assessment),
        },
        {
            "name": "claim_scope_next_actions",
            "passed": _json_value(claim_scope_next_actions)
            == _json_value(expected_claim_scope_next_actions),
            "expected": _json_value(expected_claim_scope_next_actions),
            "actual": _json_value(claim_scope_next_actions),
        },
        {
            "name": "claim_scope_handoff_plan",
            "passed": expected_claim_scope_handoff_plan_valid,
            "expected": {
                "required": bool(expected_claim_scope_next_actions),
                "scale_deficits": _json_value(
                    expected_claim_scope_assessment["default_scale_deficits"]
                ),
                "source_claim_scope": expected_claim_scope_assessment["claim_scope"],
                "target_thresholds": _json_value(
                    expected_claim_scope_assessment["default_thresholds"]
                ),
                "tracks_to_expand": _claim_action_tracks(
                    expected_claim_scope_next_actions
                ),
            },
            "actual": {
                "command_keys": sorted(
                    str(key)
                    for key in _mapping_or_empty(
                        claim_scope_handoff_plan.get("commands")
                    )
                ),
                "required": claim_scope_handoff_plan.get("required"),
                "scale_deficits": _json_value(
                    _mapping_or_empty(
                        claim_scope_handoff_plan.get("scale_deficits")
                    )
                ),
                "source_claim_scope": _text_or_none(
                    claim_scope_handoff_plan.get("source_claim_scope")
                ),
                "target_thresholds": _json_value(
                    _mapping_or_empty(
                        claim_scope_handoff_plan.get("target_thresholds")
                    )
                ),
                "tracks_to_expand": _string_sequence_or_empty(
                    claim_scope_handoff_plan.get("tracks_to_expand")
                ),
            },
        },
        {
            "name": "claim_conclusion_summary",
            "passed": (
                set(research_question_verdicts)
                == set(REAL_EXPERIMENT_RESEARCH_QUESTION_KEYS)
                and _json_value(claim_conclusion_summary)
                == _json_value(expected_claim_conclusion_summary)
            ),
            "expected": {
                "claim_conclusion_summary": _json_value(
                    expected_claim_conclusion_summary
                ),
                "research_question_verdict_keys": sorted(
                    REAL_EXPERIMENT_RESEARCH_QUESTION_KEYS
                ),
            },
            "actual": {
                "claim_conclusion_summary": _json_value(
                    claim_conclusion_summary
                ),
                "research_question_verdict_keys": sorted(
                    str(key) for key in research_question_verdicts
                ),
            },
        },
        {
            "name": "claim_conclusion_evidence",
            "passed": (
                set(claim_conclusion_evidence)
                == set(REAL_EXPERIMENT_RESEARCH_QUESTION_KEYS)
                and all(
                    isinstance(
                        _mapping_or_empty(claim_conclusion_evidence.get(key)).get(
                            "primary_metric"
                        ),
                        Mapping,
                    )
                    for key in REAL_EXPERIMENT_RESEARCH_QUESTION_KEYS
                )
                and {
                    key: _text_or_none(
                        _mapping_or_empty(
                            claim_conclusion_evidence.get(key)
                        ).get("verdict")
                    )
                    for key in REAL_EXPERIMENT_RESEARCH_QUESTION_KEYS
                }
                == _json_value(research_question_verdicts)
            ),
            "expected": {
                "research_question_keys": sorted(
                    REAL_EXPERIMENT_RESEARCH_QUESTION_KEYS
                ),
                "verdicts": _json_value(research_question_verdicts),
            },
            "actual": {
                "research_question_keys": sorted(
                    str(key) for key in claim_conclusion_evidence
                ),
                "verdicts": {
                    key: _text_or_none(
                        _mapping_or_empty(
                            claim_conclusion_evidence.get(key)
                        ).get("verdict")
                    )
                    for key in REAL_EXPERIMENT_RESEARCH_QUESTION_KEYS
                },
            },
        },
        {
            "name": "claim_effect_matrix",
            "passed": _json_value(claim_effect_matrix)
            == _json_value(expected_claim_effect_matrix),
            "expected": _json_value(expected_claim_effect_matrix),
            "actual": _json_value(claim_effect_matrix),
        },
        {
            "name": "claim_effect_direction_summary",
            "passed": (
                _json_value(claim_effect_direction_summary)
                == _json_value(expected_claim_effect_direction_summary)
                and expected_claim_effect_direction_summary["consistent"] is True
            ),
            "expected": _json_value(expected_claim_effect_direction_summary),
            "actual": _json_value(claim_effect_direction_summary),
        },
        {
            "name": "claim_hypothesis_assessment",
            "passed": _json_value(claim_hypothesis_assessment)
            == _json_value(expected_claim_hypothesis_assessment),
            "expected": _json_value(expected_claim_hypothesis_assessment),
            "actual": _json_value(claim_hypothesis_assessment),
        },
        {
            "name": "next_actions",
            "passed": (
                (claim_ready is True and not next_actions)
                or (claim_ready is False and len(next_actions) > 0)
            ),
            "expected": "empty when claim_ready else non-empty",
            "actual": {
                "claim_ready": claim_ready,
                "next_action_count": len(next_actions),
            },
        },
        {
            "name": "next_actions_research_question_gap_guidance",
            "passed": (
                research_question_gap_count == 0
                or expected_research_question_gap_action is True
            ),
            "expected": "matching RQ evidence targets when RQ gaps exist",
            "actual": {
                "matching_action_present": expected_research_question_gap_action,
                "research_question_gap_count": research_question_gap_count,
            },
        },
        {
            "name": "next_handoff_plan",
            "passed": (
                (
                    claim_ready is True
                    and next_handoff_plan.get("required") is False
                    and not _mapping_or_empty(next_handoff_plan.get("commands"))
                )
                or (
                    claim_ready is False
                    and next_handoff_plan.get("required") is True
                    and _text_or_none(next_handoff_plan.get("handoff_root"))
                    is not None
                    and _text_or_none(next_handoff_plan.get("source_run_manifest_path"))
                    is not None
                    and next_handoff_write_command is not None
                    and bool(next_handoff_required_predicted_input_kinds)
                    and all(
                        (
                            f"--required-predicted-input-kind {input_kind}"
                            in next_handoff_write_command
                        )
                        for input_kind in next_handoff_required_predicted_input_kinds
                    )
                )
            ),
            "expected": (
                "no command when claim_ready else write_handoff_manifests with "
                "required predicted inputs"
            ),
            "actual": {
                "claim_ready": claim_ready,
                "command_keys": sorted(
                    str(key) for key in next_handoff_commands
                ),
                "handoff_root_present": _text_or_none(
                    next_handoff_plan.get("handoff_root")
                )
                is not None,
                "required_predicted_input_kinds": (
                    next_handoff_required_predicted_input_kinds
                ),
                "required": next_handoff_plan.get("required"),
                "source_run_manifest_path_present": _text_or_none(
                    next_handoff_plan.get("source_run_manifest_path")
                )
                is not None,
            },
        },
        {
            "name": "next_handoff_episode_collection_plan",
            "passed": (
                _summary_int(episode_collection_plan, "current_episode_count")
                is not None
                and _summary_int(episode_collection_plan, "episode_deficit")
                is not None
                and _summary_int(episode_collection_plan, "target_episode_count")
                is not None
                and len(
                    _string_sequence_or_empty(
                        episode_collection_plan.get("planned_episode_paths")
                    )
                )
                == (
                    _summary_int(episode_collection_plan, "episode_deficit")
                    or 0
                )
            ),
            "expected": "episode counts with planned paths for every deficit",
            "actual": {
                "current_episode_count": _summary_int(
                    episode_collection_plan,
                    "current_episode_count",
                ),
                "episode_deficit": _summary_int(
                    episode_collection_plan,
                    "episode_deficit",
                ),
                "planned_episode_path_count": len(
                    _string_sequence_or_empty(
                        episode_collection_plan.get("planned_episode_paths")
                    )
                ),
                "target_episode_count": _summary_int(
                    episode_collection_plan,
                    "target_episode_count",
                ),
            },
        },
        {
            "name": "next_handoff_external_artifact_slots",
            "passed": (
                _text_or_none(
                    external_artifact_slots.get("candidate_prediction_path")
                )
                is not None
                and _text_or_none(external_artifact_slots.get("detector_jsonl_path"))
                is not None
                and len(offline_control_prediction_paths) > 0
                and all(
                    _text_or_none(path) is not None
                    for path in offline_control_prediction_paths.values()
                )
            ),
            "expected": (
                "candidate prediction, detector JSONL, and offline control "
                "prediction input paths"
            ),
            "actual": {
                "candidate_prediction_path_present": _text_or_none(
                    external_artifact_slots.get("candidate_prediction_path")
                )
                is not None,
                "detector_jsonl_path_present": _text_or_none(
                    external_artifact_slots.get("detector_jsonl_path")
                )
                is not None,
                "offline_control_prediction_path_count": len(
                    offline_control_prediction_paths
                ),
            },
        },
        {
            "name": "next_handoff_after_write_intake_plan",
            "passed": (
                (
                    claim_ready is True
                    and after_write_intake_plan.get("required") is False
                    and not after_write_commands
                )
                or (
                    claim_ready is False
                    and after_write_intake_plan.get("required") is True
                    and len(after_write_artifact_paths) > 0
                    and all(
                        _text_or_none(path) is not None
                        for path in after_write_artifact_paths.values()
                    )
                    and {
                        "external_artifact_launch_report",
                        "offline_control_prediction_request_bundle",
                        "predicted_dsg_detector_request_bundle",
                        "real_collection_request_bundle",
                    }.issubset(after_write_commands)
                )
            ),
            "expected": (
                "empty when claim_ready else after-write intake commands for "
                "real data, controls, predicted DSG, and launch audit"
            ),
            "actual": {
                "artifact_path_count": len(after_write_artifact_paths),
                "claim_ready": claim_ready,
                "command_keys": sorted(str(key) for key in after_write_commands),
                "required": after_write_intake_plan.get("required"),
            },
        },
        {
            "name": "next_handoff_next_run_review_plan",
            "passed": (
                (
                    claim_ready is True
                    and next_run_review_plan.get("required") is False
                    and not next_run_review_commands
                )
                or (
                    claim_ready is False
                    and next_run_review_plan.get("required") is True
                    and len(next_run_review_artifact_paths) > 0
                    and all(
                        _text_or_none(path) is not None
                        for path in next_run_review_artifact_paths.values()
                    )
                    and {
                        "claim_readiness",
                        "execution_packet",
                        "execution_receipt",
                        "research_review",
                        "smoke_run_checklist",
                    }.issubset(next_run_review_commands)
                )
            ),
            "expected": (
                "empty when claim_ready else execution packet, smoke checklist, "
                "receipt, research review, and claim recheck commands"
            ),
            "actual": {
                "artifact_path_count": len(next_run_review_artifact_paths),
                "claim_ready": claim_ready,
                "command_keys": sorted(str(key) for key in next_run_review_commands),
                "required": next_run_review_plan.get("required"),
            },
        },
        {
            "name": "next_handoff_operator_checklist",
            "passed": (
                (
                    claim_ready is True
                    and operator_checklist.get("required") is False
                    and _summary_int(operator_checklist, "step_count") == 0
                    and not operator_steps
                )
                or (
                    claim_ready is False
                    and operator_checklist.get("required") is True
                    and _summary_int(operator_checklist, "step_count")
                    == len(operator_steps)
                    and operator_step_keys[:1] == ["write_handoff_manifests"]
                    and operator_step_keys[-1:] == ["compare_claim_readiness"]
                    and all(
                        _summary_int(step, "order") == index
                        for index, step in enumerate(operator_steps, start=1)
                    )
                )
            ),
            "expected": (
                "empty when claim_ready else ordered operator checklist from "
                "handoff writing through claim recheck"
            ),
            "actual": {
                "claim_ready": claim_ready,
                "first_key": operator_step_keys[:1],
                "last_key": operator_step_keys[-1:],
                "required": operator_checklist.get("required"),
                "step_count": _summary_int(operator_checklist, "step_count"),
            },
        },
    ]
    return {
        "action": "validate_real_experiment_claim_readiness",
        "valid": all(check["passed"] is True for check in checks_result),
        "schema_version": schema_version,
        "claim_readiness_digest": digest,
        "checks": checks_result,
    }


def compare_real_experiment_claim_readiness(
    claim_readiness: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_real_experiment_claim_readiness(claim_readiness)
    review_path = _text_field(claim_readiness, "research_review_path")
    thresholds = _mapping_or_empty(claim_readiness.get("thresholds"))
    current = real_experiment_claim_readiness(
        load_real_experiment_research_review(review_path),
        research_review_path=review_path,
        min_dynamic_qa_count=_int_threshold(
            thresholds,
            "min_dynamic_qa_count",
            DEFAULT_CLAIM_MIN_DYNAMIC_QA_COUNT,
        ),
        min_episode_count=_int_threshold(
            thresholds,
            "min_episode_count",
            DEFAULT_CLAIM_MIN_EPISODE_COUNT,
        ),
        min_qa_count=_int_threshold(
            thresholds,
            "min_qa_count",
            DEFAULT_CLAIM_MIN_QA_COUNT,
        ),
        min_scene_count=_int_threshold(
            thresholds,
            "min_scene_count",
            DEFAULT_CLAIM_MIN_SCENE_COUNT,
        ),
    )
    saved_digest = _text_or_none(claim_readiness.get("claim_readiness_digest"))
    current_digest = _text_or_none(current.get("claim_readiness_digest"))
    checks = [
        {
            "name": "claim_readiness_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        {
            "name": "claim_readiness_digest_matches_current",
            "passed": saved_digest == current_digest,
            "expected": saved_digest,
            "actual": current_digest,
        },
        {
            "name": "claim_readiness_payload_matches_current",
            "passed": _json_value(claim_readiness) == _json_value(current),
            "expected": _json_value(claim_readiness),
            "actual": _json_value(current),
        },
    ]
    return {
        "action": "compare_real_experiment_claim_readiness",
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def real_experiment_claim_readiness(
    research_review: Mapping[str, Any],
    *,
    research_review_path: str | Path,
    min_episode_count: int = DEFAULT_CLAIM_MIN_EPISODE_COUNT,
    min_scene_count: int = DEFAULT_CLAIM_MIN_SCENE_COUNT,
    min_qa_count: int = DEFAULT_CLAIM_MIN_QA_COUNT,
    min_dynamic_qa_count: int = DEFAULT_CLAIM_MIN_DYNAMIC_QA_COUNT,
) -> dict[str, Any]:
    _validate_non_negative_int(min_dynamic_qa_count, "min_dynamic_qa_count")
    _validate_non_negative_int(min_episode_count, "min_episode_count")
    _validate_non_negative_int(min_qa_count, "min_qa_count")
    _validate_non_negative_int(min_scene_count, "min_scene_count")
    review_validation = validate_real_experiment_research_review(research_review)
    execution_receipt = load_real_experiment_execution_receipt(
        _text_field(research_review, "execution_receipt_path")
    )
    run_manifest_path = _text_field(execution_receipt, "run_manifest_path")
    run_manifest = load_real_experiment_run_manifest(run_manifest_path)
    benchmark_path = _execution_receipt_artifact_path(
        execution_receipt,
        "benchmark_manifest",
    )
    benchmark_audit, benchmark_manifest = _research_review_json_artifact(
        role="benchmark_manifest",
        path=benchmark_path,
        digest_field="manifest_digest",
        digest_fn=benchmark_manifest_digest,
        load_fn=load_benchmark_manifest,
        validate_fn=validate_benchmark_manifest,
    )
    scale_summary = _claim_scale_summary(benchmark_manifest)
    thresholds = {
        "min_dynamic_qa_count": min_dynamic_qa_count,
        "min_episode_count": min_episode_count,
        "min_qa_count": min_qa_count,
        "min_scene_count": min_scene_count,
    }
    research_question_gaps = _claim_research_question_gaps(research_review)
    checks = _claim_readiness_checks(
        research_review,
        review_validation=review_validation,
        benchmark_audit=benchmark_audit,
        scale_summary=scale_summary,
        thresholds=thresholds,
    )
    blockers = [
        _claim_readiness_blocker(check)
        for check in checks
        if check.get("passed") is not True
    ]
    claim_gap_summary = _claim_gap_summary(
        blockers,
        research_question_gaps=research_question_gaps,
        scale_summary=scale_summary,
        thresholds=thresholds,
    )
    claim_ready = len(blockers) == 0
    next_actions = _claim_next_actions(
        blockers,
        claim_gap_summary=claim_gap_summary,
        scale_summary=scale_summary,
        thresholds=thresholds,
    )
    research_question_verdicts = _claim_research_question_verdicts(
        research_review
    )
    claim_conclusion_evidence = _claim_conclusion_evidence(research_review)
    claim_effect_matrix = _claim_effect_matrix(claim_conclusion_evidence)
    claim_conclusion_summary = _claim_conclusion_summary(
        research_question_verdicts,
        claim_ready=claim_ready,
        research_question_gaps=research_question_gaps,
    )
    claim_effect_direction_summary = _claim_effect_direction_summary(
        claim_effect_matrix
    )
    claim_scope_assessment = _claim_scope_assessment(
        scale_summary,
        thresholds,
        claim_ready=claim_ready,
    )
    claim_scope_next_actions = _claim_scope_next_actions(
        claim_scope_assessment,
        claim_ready=claim_ready,
        scale_summary=scale_summary,
    )
    claim_readiness: dict[str, Any] = {
        "schema_version": REAL_EXPERIMENT_CLAIM_READINESS_SCHEMA_VERSION,
        "action": "real_experiment_claim_readiness",
        "research_review_path": str(research_review_path),
        "research_review_digest": _text_or_none(research_review.get("review_digest")),
        "execution_receipt_path": _text_or_none(
            research_review.get("execution_receipt_path")
        ),
        "benchmark_manifest_path": benchmark_path,
        "benchmark_manifest_digest": _text_or_none(benchmark_audit.get("digest")),
        "claim_ready": claim_ready,
        "status": "claim_ready" if claim_ready else "pilot_only",
        "thresholds": thresholds,
        "scale_summary": scale_summary,
        "research_question_summary": _json_value(
            _mapping_or_empty(research_review.get("research_question_summary"))
        ),
        "claim_scope_assessment": claim_scope_assessment,
        "claim_scope_next_actions": claim_scope_next_actions,
        "claim_scope_handoff_plan": _claim_scope_handoff_plan(
            claim_scope_assessment,
            claim_scope_next_actions=claim_scope_next_actions,
            run_manifest=run_manifest,
            run_manifest_path=run_manifest_path,
        ),
        "research_question_verdicts": _json_value(research_question_verdicts),
        "claim_conclusion_summary": claim_conclusion_summary,
        "claim_conclusion_evidence": claim_conclusion_evidence,
        "claim_effect_matrix": claim_effect_matrix,
        "claim_effect_direction_summary": claim_effect_direction_summary,
        "claim_hypothesis_assessment": _claim_hypothesis_assessment(
            claim_conclusion_summary,
            claim_effect_direction_summary,
        ),
        "evidence_summary": _json_value(
            _mapping_or_empty(research_review.get("evidence_summary"))
        ),
        "checks": checks,
        "blockers": blockers,
        "claim_gap_summary": claim_gap_summary,
        "next_actions": next_actions,
        "next_handoff_plan": _claim_next_handoff_plan(
            claim_ready=claim_ready,
            claim_gap_summary=claim_gap_summary,
            next_actions=next_actions,
            run_manifest=run_manifest,
            run_manifest_path=run_manifest_path,
            thresholds=thresholds,
        ),
        "artifacts": [benchmark_audit],
        "review_validation": {
            "valid": review_validation["valid"],
            "review_digest": review_validation["review_digest"],
        },
    }
    claim_readiness["claim_readiness_digest"] = (
        real_experiment_claim_readiness_digest(claim_readiness)
    )
    return claim_readiness


def real_experiment_external_artifact_launch_report(
    contracts: Mapping[str, Any],
    *,
    contracts_path: str | Path | None = None,
) -> dict[str, Any]:
    validation = validate_real_experiment_external_artifact_contracts(contracts)
    comparison = compare_real_experiment_external_artifact_contracts(contracts)
    run_manifest_path = Path(_text_field(contracts, "run_manifest_path"))
    run_manifest = load_real_experiment_run_manifest(run_manifest_path)
    preflight = real_experiment_run_manifest_preflight(run_manifest_path)
    tracks = _launch_report_tracks(preflight)
    summary = _launch_report_summary(preflight, tracks)
    contracts_path_text = str(contracts_path) if contracts_path is not None else None
    child_launch_gates = _launch_report_child_gates(contracts)
    actionable_blockers = _launch_report_actionable_blockers(
        tracks,
        child_launch_gates,
    )
    next_commands = _launch_report_next_commands(
        contracts_path=contracts_path_text,
        run_manifest_path=str(run_manifest_path),
        run_ledger_path=_text_or_none(
            run_manifest.get("real_experiment_run_ledger_path")
        ),
    )
    real_data_collection_intake_plan = (
        _launch_report_real_data_collection_intake_plan(
            contracts,
            tracks,
            child_launch_gates,
        )
    )
    real_controls_prediction_intake_plan = (
        _launch_report_real_controls_prediction_intake_plan(
            tracks,
            child_launch_gates,
        )
    )
    predicted_dsg_detector_intake_plan = (
        _launch_report_predicted_dsg_detector_intake_plan(
            tracks,
            child_launch_gates,
        )
    )
    primary_evidence_plans = {
        "predicted_dsg": predicted_dsg_detector_intake_plan,
        "real_controls": real_controls_prediction_intake_plan,
        "real_data": real_data_collection_intake_plan,
    }
    primary_evidence_receipt_gate = _launch_report_primary_evidence_receipt_gate(
        primary_evidence_plans,
    )
    ready_to_run = (
        validation["valid"] is True
        and comparison["matches"] is True
        and preflight.get("ready_to_run") is True
        and primary_evidence_receipt_gate["ready"] is True
    )
    report: dict[str, Any] = {
        "schema_version": REAL_EXPERIMENT_LAUNCH_REPORT_SCHEMA_VERSION,
        "action": "real_experiment_external_artifact_launch_report",
        "contracts_path": contracts_path_text,
        "contracts_digest": _text_or_none(contracts.get("contracts_digest")),
        "run_manifest_path": str(run_manifest_path),
        "preflight_ready_to_run": preflight.get("ready_to_run") is True,
        "ready_to_run": ready_to_run,
        "child_launch_gates": child_launch_gates,
        "actionable_blockers": actionable_blockers,
        "external_artifact_intake_plan": _launch_report_external_artifact_intake_plan(
            tracks,
            child_launch_gates,
            actionable_blockers,
            next_commands,
        ),
        "real_data_collection_intake_plan": real_data_collection_intake_plan,
        "real_controls_prediction_intake_plan": real_controls_prediction_intake_plan,
        "predicted_dsg_detector_intake_plan": predicted_dsg_detector_intake_plan,
        "primary_evidence_receipt_gate": primary_evidence_receipt_gate,
        "primary_evidence_intake_plan": (
            _launch_report_primary_evidence_intake_plan(
                tracks,
                child_launch_gates,
                next_commands,
                primary_evidence_plans=primary_evidence_plans,
            )
        ),
        "summary": summary,
        "tracks": tracks,
        "validation": validation,
        "comparison": {
            "matches": comparison["matches"],
            "saved_digest": comparison["saved_digest"],
            "current_digest": comparison["current_digest"],
        },
        "next_commands": next_commands,
    }
    report["report_digest"] = (
        real_experiment_external_artifact_launch_report_digest(report)
    )
    return report


def write_real_experiment_handoff_manifests(
    *,
    root: str | Path,
    dataset_name: str = "real_experiment",
    episode_paths: Sequence[str | Path],
    offline_qa_path: str | Path = "inputs/qa.jsonl",
    candidate_prediction_path: str | Path = (
        "inputs/candidate/predicted-graph-tool.jsonl"
    ),
    detector_jsonl_path: str | Path = "inputs/predicted-dsg/detector-rgbd.jsonl",
    real_collection_report_path: str | Path = "inputs/real-collection-report.json",
    active_task_delta_report_path: str | Path = (
        "inputs/review/active-task-delta.json"
    ),
    dashboard_bundle_path: str | Path = "inputs/review/dashboard.json",
    error_attribution_report_path: str | Path = (
        "inputs/review/error-attribution.json"
    ),
    graph_eval_report_path: str | Path = "inputs/review/graph-eval.json",
    max_qa_per_episode: int | None = None,
    tags: Sequence[str] = ("benchmark", "real"),
    declared_data_source_kind: str = "real",
    real_collection_source_kind: str = "ai2thor",
    min_episode_count: int = 3,
    min_scene_count: int = 1,
    min_frame_count: int = 30,
    min_qa_count: int = 30,
    required_control_kinds: Sequence[str] = DEFAULT_REAL_HANDOFF_CONTROL_KINDS,
    required_predicted_input_kinds: Sequence[
        str
    ] = DEFAULT_REAL_HANDOFF_PREDICTED_INPUT_KINDS,
    offline_control_input_format: str = QA_PREDICTION_INPUT_FORMAT,
) -> dict[str, Any]:
    handoff_root = Path(root)
    _validate_non_empty_str(dataset_name, "dataset_name")
    _validate_non_empty_str(declared_data_source_kind, "declared_data_source_kind")
    _validate_real_collection_source_kind(real_collection_source_kind)
    _validate_positive_int(min_frame_count, "min_frame_count")
    episodes = _non_empty_path_sequence(episode_paths, "episode_paths")
    control_kinds = _non_empty_string_sequence(
        required_control_kinds,
        "required_control_kinds",
    )
    predicted_input_kinds = _non_empty_string_sequence(
        required_predicted_input_kinds,
        "required_predicted_input_kinds",
    )
    tag_values = _non_empty_string_sequence(tags, "tags")
    if offline_control_input_format != QA_PREDICTION_INPUT_FORMAT:
        raise SpatialQAError(
            "Real handoff template currently supports qa_prediction source inputs"
        )

    offline_manifest_path = handoff_root / "offline-control-import-manifest.json"
    predicted_manifest_path = (
        handoff_root / "predicted-dsg-detector-run-manifest.json"
    )
    offline_control_run_ledger_path = (
        handoff_root
        / "outputs/offline-controls/offline-control-import-run-ledger.json"
    )
    predicted_dsg_run_ledger_path = (
        handoff_root / "outputs/predicted-dsg/predicted-dsg-detector-run-ledger.json"
    )
    real_experiment_run_ledger_path = (
        handoff_root / "outputs/real-experiment-run-ledger.json"
    )
    run_manifest_path = handoff_root / "real-experiment-run-manifest.json"
    preflight_report_path = handoff_root / "real-experiment-preflight.json"
    artifact_checklist_path = handoff_root / "real-experiment-artifact-checklist.json"
    operator_checklist_path = handoff_root / "real-experiment-operator-checklist.json"
    smoke_runbook_path = handoff_root / "real-experiment-smoke-run-runbook.json"
    external_contracts_path = (
        handoff_root / "real-experiment-external-artifact-contracts.json"
    )
    primary_status_path = (
        handoff_root / "real-experiment-primary-evidence-status.json"
    )
    primary_request_package_path = (
        handoff_root / "real-experiment-primary-evidence-request-package.json"
    )
    primary_return_checklist_path = (
        handoff_root / "real-experiment-primary-evidence-return-checklist.json"
    )
    primary_return_progress_path = (
        handoff_root / "real-experiment-primary-evidence-return-progress.json"
    )
    primary_acceptance_report_path = (
        handoff_root / "real-experiment-primary-evidence-acceptance-report.json"
    )

    offline_manifest = _offline_control_manifest(
        root=handoff_root,
        dataset_name=dataset_name,
        qa_path=offline_qa_path,
        candidate_prediction_path=candidate_prediction_path,
        required_control_kinds=control_kinds,
        input_format=offline_control_input_format,
    )
    predicted_manifest = _predicted_dsg_manifest(
        root=handoff_root,
        detector_jsonl_path=detector_jsonl_path,
    )
    run_manifest = _run_manifest(
        root=handoff_root,
        dataset_name=dataset_name,
        episode_paths=episodes,
        offline_manifest_path=offline_manifest_path,
        offline_control_run_ledger_path=offline_control_run_ledger_path,
        predicted_manifest_path=predicted_manifest_path,
        predicted_dsg_run_ledger_path=predicted_dsg_run_ledger_path,
        real_experiment_run_ledger_path=real_experiment_run_ledger_path,
        real_collection_report_path=real_collection_report_path,
        active_task_delta_report_path=active_task_delta_report_path,
        dashboard_bundle_path=dashboard_bundle_path,
        error_attribution_report_path=error_attribution_report_path,
        graph_eval_report_path=graph_eval_report_path,
        max_qa_per_episode=max_qa_per_episode,
        tags=tag_values,
        declared_data_source_kind=declared_data_source_kind,
        real_collection_source_kind=real_collection_source_kind,
        min_episode_count=min_episode_count,
        min_scene_count=min_scene_count,
        min_frame_count=min_frame_count,
        min_qa_count=min_qa_count,
        required_control_kinds=control_kinds,
        required_predicted_input_kinds=predicted_input_kinds,
    )

    _write_json(offline_manifest_path, offline_manifest)
    _write_json(predicted_manifest_path, predicted_manifest)
    _write_json(run_manifest_path, run_manifest)
    preflight_report = real_experiment_run_manifest_preflight(run_manifest_path)
    _write_json(preflight_report_path, preflight_report)
    artifact_checklist = _artifact_checklist(
        preflight_report,
        run_manifest_path=run_manifest_path,
        preflight_report_path=preflight_report_path,
    )
    _write_json(artifact_checklist_path, artifact_checklist)
    external_contracts = _external_artifact_contracts(
        root=handoff_root,
        run_manifest=run_manifest,
        offline_manifest=offline_manifest,
        predicted_manifest=predicted_manifest,
        artifact_checklist=artifact_checklist,
        run_manifest_path=run_manifest_path,
        preflight_report_path=preflight_report_path,
        artifact_checklist_path=artifact_checklist_path,
    )
    save_real_experiment_external_artifact_contracts(
        external_contracts,
        external_contracts_path,
    )
    loaded_run_manifest = load_real_experiment_run_manifest(run_manifest_path)
    operator_checklist = _handoff_operator_checklist(
        root=handoff_root,
        run_manifest=loaded_run_manifest,
        run_manifest_path=run_manifest_path,
        external_contracts_path=external_contracts_path,
        preflight_report=preflight_report,
    )
    _write_json(operator_checklist_path, operator_checklist)

    loaded_offline_manifest = load_offline_control_import_manifest(
        offline_manifest_path
    )
    loaded_predicted_manifest = load_predicted_dsg_detector_run_manifest(
        predicted_manifest_path
    )
    return {
        "schema_version": REAL_EXPERIMENT_HANDOFF_SCHEMA_VERSION,
        "action": "write_real_experiment_handoff_manifests",
        "dataset_name": dataset_name,
        "root": str(handoff_root),
        "manifest_paths": {
            "offline_control_import_manifest": str(offline_manifest_path),
            "predicted_dsg_detector_run_manifest": str(predicted_manifest_path),
            "real_experiment_artifact_checklist": str(artifact_checklist_path),
            "real_experiment_external_artifact_contracts": str(
                external_contracts_path
            ),
            "real_experiment_operator_checklist": str(operator_checklist_path),
            "real_experiment_primary_evidence_acceptance_report": str(
                primary_acceptance_report_path
            ),
            "real_experiment_primary_evidence_request_package": str(
                primary_request_package_path
            ),
            "real_experiment_primary_evidence_return_checklist": str(
                primary_return_checklist_path
            ),
            "real_experiment_primary_evidence_return_progress": str(
                primary_return_progress_path
            ),
            "real_experiment_primary_evidence_status": str(primary_status_path),
            "real_experiment_preflight_report": str(preflight_report_path),
            "real_experiment_run_manifest": str(run_manifest_path),
            "real_experiment_smoke_run_runbook": str(smoke_runbook_path),
        },
        "artifact_checklist_path": str(artifact_checklist_path),
        "artifact_checklist_summary": artifact_checklist["summary"],
        "artifact_track_summary": artifact_checklist["track_summary"],
        "external_artifact_contracts_path": str(external_contracts_path),
        "external_artifact_contracts_summary": external_contracts["summary"],
        "operator_checklist_path": str(operator_checklist_path),
        "operator_checklist_summary": operator_checklist["summary"],
        "preflight_report_path": str(preflight_report_path),
        "preflight_ready_to_run": preflight_report["ready_to_run"],
        "preflight_summary": preflight_report["summary"],
        "run_manifest_digest": real_experiment_run_manifest_digest(
            loaded_run_manifest
        ),
        "offline_control_import_manifest_digest": (
            offline_control_import_manifest_digest(loaded_offline_manifest)
        ),
        "predicted_dsg_detector_run_manifest_digest": (
            predicted_dsg_detector_run_manifest_digest(loaded_predicted_manifest)
        ),
        "source_summary": {
            "offline_control_input_format": offline_control_input_format,
            "offline_control_source_kinds": sorted(control_kinds),
            "predicted_input_kinds": sorted(predicted_input_kinds),
        },
        "next_commands": {
            "preflight": (
                "python scripts/run_real_experiment.py --preflight-run-manifest "
                f"{run_manifest_path}"
            ),
            "run": (
                "python scripts/run_real_experiment.py --run-manifest "
                f"{run_manifest_path} --run-ledger-output "
                f"{real_experiment_run_ledger_path}"
            ),
        },
    }


def _handoff_operator_checklist(
    *,
    root: Path,
    run_manifest: Mapping[str, Any],
    run_manifest_path: Path,
    external_contracts_path: Path,
    preflight_report: Mapping[str, Any],
) -> dict[str, Any]:
    episode_paths = _string_sequence_or_empty(run_manifest.get("episode_paths"))
    episode_collection_plan = {
        "current_episode_count": len(episode_paths),
        "episode_deficit": 0,
        "existing_episode_paths": episode_paths,
        "planned_episode_paths": [],
        "target_episode_count": len(episode_paths),
    }
    target_thresholds = {
        "min_dynamic_qa_count": DEFAULT_CLAIM_MIN_DYNAMIC_QA_COUNT,
        "min_episode_count": _summary_int(run_manifest, "min_episode_count")
        or DEFAULT_CLAIM_MIN_EPISODE_COUNT,
        "min_qa_count": _summary_int(run_manifest, "min_qa_count")
        or DEFAULT_CLAIM_MIN_QA_COUNT,
        "min_scene_count": _summary_int(run_manifest, "min_scene_count")
        or DEFAULT_CLAIM_MIN_SCENE_COUNT,
    }
    after_write_intake_plan = _claim_after_write_intake_plan(
        run_manifest,
        episode_collection_plan=episode_collection_plan,
        handoff_required=True,
        handoff_root=str(root),
        target_thresholds=target_thresholds,
    )
    next_run_review_plan = _claim_next_run_review_plan(
        handoff_required=True,
        handoff_root=str(root),
        target_thresholds=target_thresholds,
    )
    checklist = _claim_operator_checklist(
        after_write_intake_plan=after_write_intake_plan,
        handoff_commands={},
        handoff_required=True,
        next_run_review_plan=next_run_review_plan,
    )
    tracks = sorted(
        {
            _text_field(step, "track")
            for step in _mapping_sequence(checklist.get("steps"))
        }
    )
    phase_order = _string_sequence_or_empty(checklist.get("phase_order"))
    checklist_payload = {
        "schema_version": REAL_EXPERIMENT_OPERATOR_CHECKLIST_SCHEMA_VERSION,
        "action": "real_experiment_operator_checklist",
        "root": str(root),
        "run_manifest_path": str(run_manifest_path),
        "external_artifact_contracts_path": str(external_contracts_path),
        "ready_to_run": preflight_report.get("ready_to_run") is True,
        "required": checklist["required"],
        "phase_order": phase_order,
        "step_count": checklist["step_count"],
        "steps": checklist["steps"],
        "summary": {
            "phase_count": len(phase_order),
            "ready_to_run": preflight_report.get("ready_to_run") is True,
            "step_count": checklist["step_count"],
            "track_count": len(tracks),
        },
        "track_order": tracks,
    }
    checklist_payload["operator_checklist_digest"] = (
        real_experiment_operator_checklist_digest(checklist_payload)
    )
    return checklist_payload


def _artifact_checklist(
    preflight_report: Mapping[str, Any],
    *,
    run_manifest_path: Path,
    preflight_report_path: Path,
) -> dict[str, Any]:
    input_rows = [
        _input_artifact_row(row)
        for row in _mapping_sequence(preflight_report.get("required_inputs"))
    ]
    planned_rows = [
        _planned_output_row(row)
        for row in _mapping_sequence(preflight_report.get("planned_outputs"))
    ]
    groups = _mapping(preflight_report.get("groups"), "groups")
    blocked_group_count = sum(
        1
        for group in groups.values()
        if isinstance(group, Mapping) and group.get("ready") is not True
    )
    missing_input_count = sum(1 for row in input_rows if row["status"] == "missing")
    present_input_count = sum(1 for row in input_rows if row["status"] == "present")
    existing_planned_output_count = sum(
        1 for row in planned_rows if row["status"] == "already_exists"
    )
    summary = {
        "blocked_group_count": blocked_group_count,
        "existing_planned_output_count": existing_planned_output_count,
        "input_artifact_count": len(input_rows),
        "missing_input_artifact_count": missing_input_count,
        "planned_output_artifact_count": len(planned_rows),
        "present_input_artifact_count": present_input_count,
        "ready_to_run": preflight_report.get("ready_to_run") is True,
    }
    track_summary = _artifact_track_summary(input_rows, planned_rows)
    return {
        "schema_version": REAL_EXPERIMENT_ARTIFACT_CHECKLIST_SCHEMA_VERSION,
        "action": "real_experiment_artifact_checklist",
        "run_manifest_path": str(run_manifest_path),
        "preflight_report_path": str(preflight_report_path),
        "ready_to_run": preflight_report.get("ready_to_run") is True,
        "summary": summary,
        "track_summary": track_summary,
        "input_artifacts": input_rows,
        "planned_output_artifacts": planned_rows,
    }


def _input_artifact_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = _artifact_row(row)
    result["status"] = "present" if row.get("exists") is True else "missing"
    return result


def _planned_output_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = _artifact_row(row)
    result["status"] = "already_exists" if row.get("exists") is True else "planned"
    return result


def _artifact_row(row: Mapping[str, Any]) -> dict[str, Any]:
    group = _text_field(row, "group")
    result: dict[str, Any] = {
        "exists": row.get("exists") is True,
        "group": group,
        "path": _text_field(row, "path"),
        "role": _text_field(row, "role"),
        "track": _artifact_track(group),
    }
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        result["metadata"] = {
            str(key): _json_value(value)
            for key, value in sorted(metadata.items(), key=lambda item: str(item[0]))
        }
    return result


def _artifact_track_summary(
    input_rows: Sequence[Mapping[str, Any]],
    planned_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    summary = {
        track: {
            "existing_planned_output_artifact_count": 0,
            "input_artifact_count": 0,
            "input_ready": True,
            "missing_input_artifact_count": 0,
            "planned_output_artifact_count": 0,
            "present_input_artifact_count": 0,
        }
        for track in REAL_EXPERIMENT_ARTIFACT_TRACKS
    }
    for row in input_rows:
        track = _text_field(row, "track")
        track_summary = summary[track]
        track_summary["input_artifact_count"] += 1
        if row.get("status") == "present":
            track_summary["present_input_artifact_count"] += 1
        else:
            track_summary["missing_input_artifact_count"] += 1
    for row in planned_rows:
        track = _text_field(row, "track")
        track_summary = summary[track]
        track_summary["planned_output_artifact_count"] += 1
        if row.get("status") == "already_exists":
            track_summary["existing_planned_output_artifact_count"] += 1
    for track_summary in summary.values():
        track_summary["input_ready"] = (
            track_summary["missing_input_artifact_count"] == 0
        )
    return summary


def _artifact_track(group: str) -> str:
    try:
        return REAL_EXPERIMENT_ARTIFACT_GROUP_TRACKS[group]
    except KeyError as exc:
        raise SpatialQAError(
            f"Unknown real experiment artifact checklist group: {group}"
        ) from exc


def _launch_report_tracks(preflight: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    required_inputs = _mapping_sequence(preflight.get("required_inputs"))
    missing_inputs = _mapping_sequence(preflight.get("missing_inputs"))
    invalid_inputs = _mapping_sequence(preflight.get("invalid_inputs"))
    missing_requirements = _mapping_sequence(preflight.get("missing_requirements"))
    planned_outputs = _mapping_sequence(preflight.get("planned_outputs"))
    return {
        track: _launch_report_track(
            track=track,
            required_inputs=required_inputs,
            missing_inputs=missing_inputs,
            invalid_inputs=invalid_inputs,
            missing_requirements=missing_requirements,
            planned_outputs=planned_outputs,
        )
        for track in REAL_EXPERIMENT_ARTIFACT_TRACKS
    }


def _launch_report_track(
    *,
    track: str,
    required_inputs: Sequence[Mapping[str, Any]],
    missing_inputs: Sequence[Mapping[str, Any]],
    invalid_inputs: Sequence[Mapping[str, Any]],
    missing_requirements: Sequence[Mapping[str, Any]],
    planned_outputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    groups = sorted(
        group
        for group, group_track in REAL_EXPERIMENT_ARTIFACT_GROUP_TRACKS.items()
        if group_track == track
    )
    track_required_inputs = _track_rows(required_inputs, track)
    track_missing_inputs = _track_rows(missing_inputs, track)
    track_invalid_inputs = _track_rows(invalid_inputs, track)
    track_missing_requirements = _track_rows(missing_requirements, track)
    track_planned_outputs = _track_rows(planned_outputs, track)
    blocking_roles = sorted(
        {
            _text_field(row, "role")
            for row in (
                *track_missing_inputs,
                *track_invalid_inputs,
                *track_missing_requirements,
            )
        }
    )
    return {
        "ready": not blocking_roles,
        "groups": groups,
        "required_input_count": len(track_required_inputs),
        "present_input_count": len(track_required_inputs) - len(track_missing_inputs),
        "missing_input_count": len(track_missing_inputs),
        "invalid_input_count": len(track_invalid_inputs),
        "missing_requirement_count": len(track_missing_requirements),
        "planned_output_count": len(track_planned_outputs),
        "blocking_roles": blocking_roles,
        "missing_inputs": [_launch_path_row(row) for row in track_missing_inputs],
        "invalid_inputs": [_launch_invalid_row(row) for row in track_invalid_inputs],
        "missing_requirements": [
            _launch_missing_requirement_row(row)
            for row in track_missing_requirements
        ],
    }


def _launch_report_summary(
    preflight: Mapping[str, Any],
    tracks: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    summary = _mapping(preflight.get("summary"), "summary")
    ready_track_count = sum(1 for row in tracks.values() if row.get("ready") is True)
    return {
        "blocked_track_count": len(tracks) - ready_track_count,
        "invalid_input_count": _int_field(summary, "invalid_input_count"),
        "missing_input_count": _int_field(summary, "missing_input_count"),
        "missing_requirement_count": _int_field(summary, "missing_requirement_count"),
        "planned_output_count": _int_field(summary, "planned_output_count"),
        "ready_track_count": ready_track_count,
        "required_input_count": _int_field(summary, "required_input_count"),
        "track_count": len(tracks),
    }


def _launch_report_child_gates(contracts: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(_text_field(contracts, "root"))
    tracks = _mapping(contracts.get("tracks"), "tracks")
    real_data = _mapping(tracks.get("real_data"), "real_data")
    real_controls = _mapping(tracks.get("real_controls"), "real_controls")
    predicted_dsg = _mapping(tracks.get("predicted_dsg"), "predicted_dsg")
    review_artifacts = _mapping(
        tracks.get("review_artifacts"),
        "review_artifacts",
    )
    episode_paths = [
        _anchored_path(path, root)
        for path in _string_list(real_data.get("episode_paths"))
    ]
    report_paths = [
        _anchored_path(path, root)
        for path in _string_list(real_data.get("real_collection_report_paths"))
    ]
    real_collection_report_path = report_paths[0] if report_paths else root / ""
    real_collection_request_bundle_path = root / "real-collection-request-bundle.json"
    real_collection_source_kind = _text_field_or_default(
        real_data,
        "source_kind",
        "ai2thor",
    )
    min_frame_count = _int_field_or_default(real_data, "min_frame_count", 30)
    offline_manifest_path = _anchored_path(
        _text_field(real_controls, "offline_control_import_manifest_path"),
        root,
    )
    predicted_manifest_path = _anchored_path(
        _text_field(predicted_dsg, "predicted_dsg_detector_run_manifest_path"),
        root,
    )
    offline_contracts_path = root / "offline-control-artifact-contracts.json"
    predicted_contract_path = root / "predicted-dsg-detector-artifact-contract.json"
    offline_request_bundle_path = root / "offline-control-prediction-request-bundle.json"
    offline_receipt_bundle_path = root / "offline-control-prediction-receipt-bundle.json"
    predicted_request_bundle_path = root / "predicted-dsg-detector-request-bundle.json"
    predicted_receipt_bundle_path = root / "predicted-dsg-detector-receipt-bundle.json"
    active_task_delta_report_paths = [
        _anchored_path(path, root)
        for path in _string_list(review_artifacts.get("active_task_delta_report_paths"))
    ]
    dashboard_bundle_paths = [
        _anchored_path(path, root)
        for path in _string_list(review_artifacts.get("dashboard_bundle_paths"))
    ]
    error_attribution_report_paths = [
        _anchored_path(path, root)
        for path in _string_list(review_artifacts.get("error_attribution_report_paths"))
    ]
    graph_eval_report_paths = [
        _anchored_path(path, root)
        for path in _string_list(review_artifacts.get("graph_eval_report_paths"))
    ]
    return {
        "real_data": {
            "collection_report_command": _real_collection_report_command(
                dataset_name=_text_field(real_data, "dataset_name"),
                source_kind=real_collection_source_kind,
                episode_paths=episode_paths,
                report_path=real_collection_report_path,
                min_episode_count=_int_field(real_data, "min_episode_count"),
                min_scene_count=_int_field(real_data, "min_scene_count"),
                min_frame_count=min_frame_count,
            ),
            "compare_report_command": (
                "python scripts/check_real_collection.py "
                f"--compare-report {real_collection_report_path}"
            ),
            "episode_paths": [str(path) for path in episode_paths],
            "real_collection_report_path": str(real_collection_report_path),
            "request_bundle_command": _real_collection_request_bundle_command(
                bundle_path=real_collection_request_bundle_path,
                dataset_name=_text_field(real_data, "dataset_name"),
                source_kind=real_collection_source_kind,
                episode_paths=episode_paths,
                report_path=real_collection_report_path,
                min_episode_count=_int_field(real_data, "min_episode_count"),
                min_scene_count=_int_field(real_data, "min_scene_count"),
                min_frame_count=min_frame_count,
            ),
            "request_bundle_path": str(real_collection_request_bundle_path),
            "source_kind": real_collection_source_kind,
            "track": "real_data",
            "validate_report_command": (
                "python scripts/check_real_collection.py "
                f"--validate-report {real_collection_report_path}"
            ),
        },
        "offline_controls": {
            "artifact_contract_path": str(offline_contracts_path),
            "artifact_launch_report_command": (
                "python scripts/check_offline_controls.py "
                f"--artifact-launch-report {offline_contracts_path} "
                f"--manifest {offline_manifest_path}"
            ),
            "manifest_path": str(offline_manifest_path),
            "preflight_contract_command": (
                "python scripts/run_offline_controls.py "
                f"--preflight-manifest {offline_manifest_path} "
                f"--artifact-contracts {offline_contracts_path}"
            ),
            "prediction_receipt_bundle_command": (
                "python scripts/run_offline_controls.py "
                f"--prediction-receipt-bundle {offline_manifest_path} "
                f"--receipt-bundle-output {offline_receipt_bundle_path}"
            ),
            "prediction_receipt_bundle_path": str(offline_receipt_bundle_path),
            "prediction_request_bundle_command": (
                "python scripts/run_offline_controls.py "
                f"--prediction-request-bundle {offline_manifest_path} "
                f"--request-bundle-output {offline_request_bundle_path}"
            ),
            "prediction_request_bundle_path": str(offline_request_bundle_path),
            "track": "real_controls",
        },
        "predicted_dsg": {
            "artifact_contract_path": str(predicted_contract_path),
            "artifact_launch_report_command": (
                "python scripts/run_predicted_dsg.py "
                f"--artifact-launch-report {predicted_contract_path} "
                f"--manifest {predicted_manifest_path}"
            ),
            "manifest_path": str(predicted_manifest_path),
            "preflight_contract_command": (
                "python scripts/run_predicted_dsg.py "
                f"--preflight-manifest {predicted_manifest_path} "
                f"--artifact-contract {predicted_contract_path}"
            ),
            "detector_receipt_bundle_command": (
                "python scripts/run_predicted_dsg.py "
                f"--detector-receipt-bundle {predicted_manifest_path} "
                f"--receipt-bundle-output {predicted_receipt_bundle_path}"
            ),
            "detector_receipt_bundle_path": str(predicted_receipt_bundle_path),
            "detector_request_bundle_command": (
                "python scripts/run_predicted_dsg.py "
                f"--detector-request-bundle {predicted_manifest_path} "
                f"--request-bundle-output {predicted_request_bundle_path}"
            ),
            "detector_request_bundle_path": str(predicted_request_bundle_path),
            "track": "predicted_dsg",
        },
        "review_artifacts": {
            "active_task_delta_report_commands": _review_report_command_rows(
                active_task_delta_report_paths,
                script="scripts/run_active_tasks.py",
                validate_flag="--validate-delta-report",
                compare_flag="--compare-delta-report",
            ),
            "active_task_delta_report_paths": [
                str(path) for path in active_task_delta_report_paths
            ],
            "dashboard_bundle_commands": _review_report_command_rows(
                dashboard_bundle_paths,
                script="scripts/export_dashboard.py",
                validate_flag="--validate-bundle",
            ),
            "dashboard_bundle_paths": [str(path) for path in dashboard_bundle_paths],
            "error_attribution_report_commands": _review_report_command_rows(
                error_attribution_report_paths,
                script="scripts/attribute_errors.py",
                validate_flag="--validate-report",
                compare_flag="--compare-report",
            ),
            "error_attribution_report_paths": [
                str(path) for path in error_attribution_report_paths
            ],
            "graph_eval_report_commands": _review_report_command_rows(
                graph_eval_report_paths,
                script="scripts/evaluate_graphs.py",
                validate_flag="--validate-report",
                compare_flag="--compare-report",
            ),
            "graph_eval_report_paths": [str(path) for path in graph_eval_report_paths],
            "track": "review_artifacts",
        },
    }


def _launch_report_actionable_blockers(
    tracks: Mapping[str, Mapping[str, Any]],
    child_launch_gates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    child_gate_by_track = {
        "predicted_dsg": "predicted_dsg",
        "real_controls": "offline_controls",
        "real_data": "real_data",
        "review_artifacts": "review_artifacts",
    }
    blockers: dict[str, Any] = {}
    for track in REAL_EXPERIMENT_ARTIFACT_TRACKS:
        row = _mapping(tracks.get(track), track)
        if row.get("ready") is True:
            continue
        blocker = {
            "track": track,
            "blocking_roles": _string_list(row.get("blocking_roles")),
            "invalid_input_count": _int_field(row, "invalid_input_count"),
            "invalid_inputs": _json_value(row.get("invalid_inputs", [])),
            "missing_input_count": _int_field(row, "missing_input_count"),
            "missing_inputs": _json_value(row.get("missing_inputs", [])),
            "missing_requirement_count": _int_field(
                row,
                "missing_requirement_count",
            ),
            "missing_requirements": _json_value(
                row.get("missing_requirements", [])
            ),
        }
        child_gate_key = child_gate_by_track.get(track)
        if child_gate_key is not None and child_gate_key in child_launch_gates:
            blocker["child_launch_gate"] = _json_value(
                child_launch_gates[child_gate_key]
            )
        blockers[track] = blocker
    return blockers


def _launch_report_external_artifact_intake_plan(
    tracks: Mapping[str, Mapping[str, Any]],
    child_launch_gates: Mapping[str, Mapping[str, Any]],
    actionable_blockers: Mapping[str, Mapping[str, Any]],
    next_commands: Mapping[str, str],
) -> dict[str, Any]:
    child_gate_by_track = {
        "predicted_dsg": "predicted_dsg",
        "real_controls": "offline_controls",
        "real_data": "real_data",
        "review_artifacts": "review_artifacts",
    }
    command_keys_by_track = {
        "predicted_dsg": [
            "detector_request_bundle_command",
            "detector_receipt_bundle_command",
            "preflight_contract_command",
            "artifact_launch_report_command",
        ],
        "real_controls": [
            "prediction_request_bundle_command",
            "prediction_receipt_bundle_command",
            "preflight_contract_command",
            "artifact_launch_report_command",
        ],
        "real_data": [
            "collection_report_command",
            "request_bundle_command",
            "validate_report_command",
            "compare_report_command",
        ],
        "review_artifacts": [
            "active_task_delta_report_commands",
            "dashboard_bundle_commands",
            "error_attribution_report_commands",
            "graph_eval_report_commands",
        ],
        "run_outputs": ["preflight", "run"],
    }
    steps: list[dict[str, Any]] = []
    for track in REAL_EXPERIMENT_ARTIFACT_TRACKS:
        blocker = actionable_blockers.get(track)
        if blocker is None:
            continue
        step = {
            "blocking_roles": _string_list(blocker.get("blocking_roles")),
            "order": len(steps) + 1,
            "recommended_command_keys": command_keys_by_track[track],
            "track": track,
        }
        child_gate_key = child_gate_by_track.get(track)
        if child_gate_key is not None and child_gate_key in child_launch_gates:
            step["child_launch_gate"] = _json_value(child_launch_gates[child_gate_key])
        steps.append(step)
    ready_tracks = [
        track
        for track in REAL_EXPERIMENT_ARTIFACT_TRACKS
        if _mapping(tracks.get(track), track).get("ready") is True
    ]
    return {
        "blocked_track_count": len(steps),
        "final_commands": _json_value(next_commands),
        "ready_track_count": len(ready_tracks),
        "ready_tracks": ready_tracks,
        "steps": steps,
        "track_order": list(REAL_EXPERIMENT_ARTIFACT_TRACKS),
    }


def _launch_report_primary_evidence_intake_plan(
    tracks: Mapping[str, Mapping[str, Any]],
    child_launch_gates: Mapping[str, Mapping[str, Any]],
    next_commands: Mapping[str, str],
    *,
    primary_evidence_plans: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    child_gate_by_track = {
        "predicted_dsg": "predicted_dsg",
        "real_controls": "offline_controls",
        "real_data": "real_data",
    }
    artifact_goal_by_track = {
        "predicted_dsg": "predicted_dsg_detector_inputs",
        "real_controls": "offline_control_predictions",
        "real_data": "real_collection",
    }
    command_keys_by_track = {
        "predicted_dsg": [
            "detector_request_bundle_command",
            "detector_receipt_bundle_command",
            "preflight_contract_command",
            "artifact_launch_report_command",
        ],
        "real_controls": [
            "prediction_request_bundle_command",
            "prediction_receipt_bundle_command",
            "preflight_contract_command",
            "artifact_launch_report_command",
        ],
        "real_data": [
            "collection_report_command",
            "request_bundle_command",
            "validate_report_command",
            "compare_report_command",
        ],
    }
    steps: list[dict[str, Any]] = []
    ready_track_count = 0
    for track in REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS:
        row = _mapping(tracks.get(track), track)
        intake_plan = (
            _mapping(primary_evidence_plans.get(track), track)
            if primary_evidence_plans is not None and track in primary_evidence_plans
            else None
        )
        ready = (
            intake_plan.get("ready") is True
            if intake_plan is not None
            else row.get("ready") is True
        )
        if ready:
            ready_track_count += 1
        step: dict[str, Any] = {
            "artifact_goal": artifact_goal_by_track[track],
            "blocking_roles": (
                _string_list(intake_plan.get("blocking_roles"))
                if intake_plan is not None
                else _string_list(row.get("blocking_roles"))
            ),
            "order": len(steps) + 1,
            "ready": ready,
            "recommended_command_keys": command_keys_by_track[track],
            "track": track,
        }
        child_gate_key = child_gate_by_track[track]
        if child_gate_key in child_launch_gates:
            step["child_launch_gate"] = _json_value(child_launch_gates[child_gate_key])
        steps.append(step)
    blocked_track_count = len(REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS) - ready_track_count
    return {
        "track": "primary_evidence",
        "ready": blocked_track_count == 0,
        "blocked_track_count": blocked_track_count,
        "ready_track_count": ready_track_count,
        "final_commands": _json_value(next_commands),
        "steps": steps,
        "track_order": list(REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS),
    }


def _launch_report_primary_evidence_receipt_gate(
    primary_evidence_plans: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    intake_plan_keys = {
        "predicted_dsg": "predicted_dsg_detector_intake_plan",
        "real_controls": "real_controls_prediction_intake_plan",
        "real_data": "real_data_collection_intake_plan",
    }
    blocked_tracks: list[dict[str, Any]] = []
    ready_tracks: list[str] = []
    for track in REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS:
        plan = _mapping(primary_evidence_plans.get(track), track)
        if plan.get("ready") is True:
            ready_tracks.append(track)
            continue
        blocked_tracks.append(
            {
                "blocking_roles": _string_list(plan.get("blocking_roles")),
                "intake_plan_key": intake_plan_keys[track],
                "receipt_status": _launch_report_primary_receipt_status(
                    track,
                    plan,
                ),
                "track": track,
            }
        )
    return {
        "track": "primary_evidence",
        "ready": not blocked_tracks,
        "blocked_track_count": len(blocked_tracks),
        "blocked_tracks": blocked_tracks,
        "ready_track_count": len(ready_tracks),
        "ready_tracks": ready_tracks,
        "track_order": list(REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS),
    }


def _launch_report_primary_receipt_status(
    track: str,
    plan: Mapping[str, Any],
) -> str:
    if track == "real_data":
        receipt = _mapping(plan.get("collection_report_receipt"), "receipt")
    elif track == "real_controls":
        receipt = _mapping(plan.get("prediction_receipt_bundle"), "receipt")
    elif track == "predicted_dsg":
        receipt = _mapping(plan.get("detector_receipt_bundle"), "receipt")
    else:
        receipt = _mapping(plan.get("artifact_contract_receipt"), "receipt")
    return _text_field(receipt, "status")


def _primary_evidence_status_track_row(
    track: str,
    *,
    launch_report: Mapping[str, Any],
    step: Mapping[str, Any],
) -> dict[str, Any]:
    plan_key = _primary_evidence_status_plan_key(track)
    plan = _mapping(launch_report.get(plan_key), plan_key)
    child_gate = _mapping(step.get("child_launch_gate"), "child_launch_gate")
    receipt = _primary_evidence_status_receipt(track, plan)
    recommended_commands = [
        {
            "command": _text_field(child_gate, command_key),
            "key": command_key,
        }
        for command_key in _string_list(step.get("recommended_command_keys"))
    ]
    ready = step.get("ready") is True
    next_command = None if ready or not recommended_commands else recommended_commands[0]
    return {
        "artifact_goal": _text_field(step, "artifact_goal"),
        "blocking_roles": _string_list(step.get("blocking_roles")),
        "next_command": (
            _text_field(next_command, "command")
            if isinstance(next_command, Mapping)
            else None
        ),
        "next_command_key": (
            _text_field(next_command, "key")
            if isinstance(next_command, Mapping)
            else None
        ),
        "ready": ready,
        "receipt_path": _text_or_none(receipt.get("path")),
        "receipt_status": _text_field(receipt, "status"),
        "recommended_commands": recommended_commands,
        "track": track,
    }


def _primary_evidence_status_plan_key(track: str) -> str:
    plan_keys = {
        "predicted_dsg": "predicted_dsg_detector_intake_plan",
        "real_controls": "real_controls_prediction_intake_plan",
        "real_data": "real_data_collection_intake_plan",
    }
    try:
        return plan_keys[track]
    except KeyError as exc:
        raise SpatialQAError(
            f"Unknown real experiment primary evidence track: {track}"
        ) from exc


def _primary_evidence_status_receipt(
    track: str,
    plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    if track == "real_data":
        return _mapping(plan.get("collection_report_receipt"), "receipt")
    if track == "real_controls":
        return _mapping(plan.get("prediction_receipt_bundle"), "receipt")
    if track == "predicted_dsg":
        return _mapping(plan.get("detector_receipt_bundle"), "receipt")
    raise SpatialQAError(
        f"Unknown real experiment primary evidence track: {track}"
    )


def _primary_evidence_status_next_blocked_track(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    for row in rows:
        if row.get("ready") is True:
            continue
        return {
            "blocking_roles": _string_list(row.get("blocking_roles")),
            "next_command": _text_or_none(row.get("next_command")),
            "next_command_key": _text_or_none(row.get("next_command_key")),
            "receipt_status": _text_field(row, "receipt_status"),
            "track": _text_field(row, "track"),
        }
    return None


def _primary_evidence_status_track_row_valid(row: Mapping[str, Any]) -> bool:
    recommended_commands = _mapping_sequence(row.get("recommended_commands"))
    ready = row.get("ready")
    return (
        _text_or_none(row.get("artifact_goal")) is not None
        and isinstance(ready, bool)
        and _text_or_none(row.get("receipt_status")) is not None
        and _text_or_none(row.get("track")) in REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS
        and len(recommended_commands) > 0
        and all(
            _text_or_none(command.get("command")) is not None
            and _text_or_none(command.get("key")) is not None
            for command in recommended_commands
        )
        and (
            (
                ready is True
                and row.get("next_command") is None
                and row.get("next_command_key") is None
            )
            or (
                ready is False
                and _text_or_none(row.get("next_command")) is not None
                and _text_or_none(row.get("next_command_key")) is not None
            )
        )
    )


def _primary_evidence_request_package_track_row(
    track: str,
    *,
    launch_report: Mapping[str, Any],
) -> dict[str, Any]:
    plan_key = _primary_evidence_status_plan_key(track)
    plan = _mapping(launch_report.get(plan_key), plan_key)
    try:
        bundle, path, command = _primary_evidence_request_bundle(
            track,
            plan,
            launch_report=launch_report,
        )
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        return {
            "track": track,
            "status": "blocked",
            "request_bundle": None,
            "request_bundle_digest": None,
            "request_bundle_validation": None,
            "request_bundle_path": _primary_evidence_request_bundle_path(
                track,
                launch_report=launch_report,
            ),
            "request_bundle_command": _primary_evidence_request_bundle_command(
                track,
                launch_report=launch_report,
            ),
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
    return {
        "track": track,
        "status": "ready",
        "request_bundle": bundle,
        "request_bundle_digest": _text_field(bundle, "request_bundle_digest"),
        "request_bundle_validation": _primary_evidence_request_bundle_validation(
            track,
            bundle,
        ),
        "request_bundle_path": path,
        "request_bundle_command": command,
        "error": None,
        "error_type": None,
    }


def _primary_evidence_request_bundle(
    track: str,
    plan: Mapping[str, Any],
    *,
    launch_report: Mapping[str, Any],
) -> tuple[dict[str, Any], str, str]:
    if track == "real_data":
        thresholds = _mapping(plan.get("thresholds"), "thresholds")
        bundle = real_collection_request_bundle(
            dataset_name=_text_field(plan, "dataset_name"),
            episode_paths=_string_list(plan.get("episode_paths")),
            source_kind=_text_field(plan, "source_kind"),
            report_path=_text_field(plan, "real_collection_report_path"),
            min_episode_count=_int_field(thresholds, "min_episode_count"),
            min_frame_count=_int_field(thresholds, "min_frame_count"),
            min_scene_count=_int_field(thresholds, "min_scene_count"),
        )
    elif track == "real_controls":
        bundle = offline_control_prediction_request_bundle(
            _text_field(plan, "manifest_path")
        )
    elif track == "predicted_dsg":
        bundle = predicted_dsg_detector_request_bundle(
            _text_field(plan, "manifest_path")
        )
    else:
        raise SpatialQAError(
            f"Unknown real experiment primary evidence track: {track}"
        )
    return (
        bundle,
        _primary_evidence_request_bundle_path(track, launch_report=launch_report),
        _primary_evidence_request_bundle_command(track, launch_report=launch_report),
    )


def _primary_evidence_request_bundle_path(
    track: str,
    *,
    launch_report: Mapping[str, Any],
) -> str:
    child_gates = _mapping(launch_report.get("child_launch_gates"), "child_gates")
    if track == "real_data":
        return _text_field(
            _mapping(child_gates.get("real_data"), "real_data"),
            "request_bundle_path",
        )
    if track == "real_controls":
        return _text_field(
            _mapping(child_gates.get("offline_controls"), "offline_controls"),
            "prediction_request_bundle_path",
        )
    if track == "predicted_dsg":
        return _text_field(
            _mapping(child_gates.get("predicted_dsg"), "predicted_dsg"),
            "detector_request_bundle_path",
        )
    raise SpatialQAError(
        f"Unknown real experiment primary evidence track: {track}"
    )


def _primary_evidence_request_bundle_command(
    track: str,
    *,
    launch_report: Mapping[str, Any],
) -> str:
    child_gates = _mapping(launch_report.get("child_launch_gates"), "child_gates")
    if track == "real_data":
        return _text_field(
            _mapping(child_gates.get("real_data"), "real_data"),
            "request_bundle_command",
        )
    if track == "real_controls":
        return _text_field(
            _mapping(child_gates.get("offline_controls"), "offline_controls"),
            "prediction_request_bundle_command",
        )
    if track == "predicted_dsg":
        return _text_field(
            _mapping(child_gates.get("predicted_dsg"), "predicted_dsg"),
            "detector_request_bundle_command",
        )
    raise SpatialQAError(
        f"Unknown real experiment primary evidence track: {track}"
    )


def _primary_evidence_request_package_track_row_valid(
    row: Mapping[str, Any],
) -> bool:
    status = _text_or_none(row.get("status"))
    bundle = row.get("request_bundle")
    validation = row.get("request_bundle_validation")
    return (
        _text_or_none(row.get("track")) in REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS
        and status in {"blocked", "ready"}
        and _text_or_none(row.get("request_bundle_path")) is not None
        and _text_or_none(row.get("request_bundle_command")) is not None
        and (
            (
                status == "ready"
                and isinstance(bundle, Mapping)
                and _text_or_none(row.get("request_bundle_digest")) is not None
                and _text_or_none(bundle.get("request_bundle_digest"))
                == _text_or_none(row.get("request_bundle_digest"))
                and _primary_evidence_request_bundle_validation_shape_valid(
                    validation
                )
                and _text_or_none(
                    _mapping(validation, "request_bundle_validation").get(
                        "request_bundle_digest"
                    )
                )
                == _text_or_none(row.get("request_bundle_digest"))
                and row.get("error") is None
                and row.get("error_type") is None
            )
            or (
                status == "blocked"
                and bundle is None
                and row.get("request_bundle_digest") is None
                and row.get("request_bundle_validation") is None
                and _text_or_none(row.get("error")) is not None
                and _text_or_none(row.get("error_type")) is not None
            )
        )
    )


def _primary_evidence_request_package_row_bundle_validation(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    track = _text_or_none(row.get("track"))
    status = _text_or_none(row.get("status"))
    saved_validation = row.get("request_bundle_validation")
    if status == "blocked":
        return {
            "matches_saved": saved_validation is None,
            "request_bundle_digest": None,
            "saved_valid": None,
            "track": track,
            "valid": saved_validation is None,
        }
    if track not in REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS:
        return {
            "matches_saved": False,
            "request_bundle_digest": None,
            "saved_valid": None,
            "track": track,
            "valid": False,
        }
    bundle = row.get("request_bundle")
    if not isinstance(bundle, Mapping):
        return {
            "matches_saved": False,
            "request_bundle_digest": None,
            "saved_valid": None,
            "track": track,
            "valid": False,
        }
    try:
        current_validation = _primary_evidence_request_bundle_validation(
            track,
            bundle,
        )
    except (SpatialQAError, ValueError, TypeError) as exc:
        return {
            "error": str(exc),
            "matches_saved": False,
            "request_bundle_digest": _text_or_none(
                bundle.get("request_bundle_digest")
            ),
            "saved_valid": (
                _mapping(saved_validation, "request_bundle_validation").get("valid")
                if isinstance(saved_validation, Mapping)
                else None
            ),
            "track": track,
            "valid": False,
        }
    matches_saved = _json_value(saved_validation) == _json_value(current_validation)
    return {
        "matches_saved": matches_saved,
        "request_bundle_digest": current_validation["request_bundle_digest"],
        "saved_valid": (
            _mapping(saved_validation, "request_bundle_validation").get("valid")
            if isinstance(saved_validation, Mapping)
            else None
        ),
        "track": track,
        "valid": current_validation["valid"] is True and matches_saved,
    }


def _primary_evidence_request_bundle_validation_shape_valid(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return (
        _text_or_none(value.get("action")) is not None
        and isinstance(value.get("valid"), bool)
        and _text_or_none(value.get("request_bundle_digest")) is not None
    )


def _primary_evidence_request_bundle_validation(
    track: str,
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    if track == "real_data":
        validation = validate_real_collection_request_bundle(bundle)
        return {
            "action": _text_field(validation, "action"),
            "request_bundle_digest": _text_field(
                validation,
                "request_bundle_digest",
            ),
            "valid": validation.get("valid") is True,
        }
    if track == "real_controls":
        return _primary_evidence_basic_request_bundle_validation(
            bundle,
            action="offline_control_prediction_request_bundle",
            digest=offline_control_prediction_request_bundle_digest(bundle),
            schema_version=OFFLINE_CONTROL_PREDICTION_REQUEST_BUNDLE_SCHEMA_VERSION,
        )
    if track == "predicted_dsg":
        return _primary_evidence_basic_request_bundle_validation(
            bundle,
            action="predicted_dsg_detector_request_bundle",
            digest=predicted_dsg_detector_request_bundle_digest(bundle),
            schema_version=PREDICTED_DSG_DETECTOR_REQUEST_BUNDLE_SCHEMA_VERSION,
        )
    raise SpatialQAError(
        f"Unknown real experiment primary evidence track: {track}"
    )


def _primary_evidence_basic_request_bundle_validation(
    bundle: Mapping[str, Any],
    *,
    action: str,
    digest: str,
    schema_version: str,
) -> dict[str, Any]:
    request_digest = _text_or_none(bundle.get("request_bundle_digest"))
    return {
        "action": f"validate_{action}",
        "request_bundle_digest": request_digest,
        "valid": (
            bundle.get("schema_version") == schema_version
            and bundle.get("action") == action
            and request_digest == digest
        ),
    }


def _primary_evidence_request_bundle_write_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    track = _text_field(row, "track")
    path = _text_field(row, "request_bundle_path")
    if _text_field(row, "status") != "ready":
        return {
            "error": _text_or_none(row.get("error")),
            "error_type": _text_or_none(row.get("error_type")),
            "request_bundle_digest": None,
            "request_bundle_path": path,
            "status": "skipped_blocked",
            "track": track,
        }
    bundle = _mapping(row.get("request_bundle"), "request_bundle")
    written_path = _save_primary_evidence_request_bundle(
        track,
        bundle=bundle,
        path=path,
    )
    return {
        "error": None,
        "error_type": None,
        "request_bundle_digest": _text_field(row, "request_bundle_digest"),
        "request_bundle_path": str(written_path),
        "status": "written",
        "track": track,
    }


def _save_primary_evidence_request_bundle(
    track: str,
    *,
    bundle: Mapping[str, Any],
    path: str,
) -> Path:
    if track == "real_data":
        return save_real_collection_request_bundle(bundle, path)
    if track == "real_controls":
        return save_offline_control_prediction_request_bundle(bundle, path)
    if track == "predicted_dsg":
        return save_predicted_dsg_detector_request_bundle(bundle, path)
    raise SpatialQAError(
        f"Unknown real experiment primary evidence track: {track}"
    )


def _primary_evidence_return_checklist_step(
    row: Mapping[str, Any],
    *,
    launch_report: Mapping[str, Any],
) -> dict[str, Any]:
    track = _text_field(row, "track")
    request_bundle_command = _text_field(row, "request_bundle_command")
    base = {
        "track": track,
        "request_bundle_path": _text_field(row, "request_bundle_path"),
        "request_bundle_digest": _text_or_none(row.get("request_bundle_digest")),
        "request_bundle_command": request_bundle_command,
    }
    if _text_field(row, "status") != "ready":
        return {
            **base,
            "status": "blocked",
            "expected_return_artifact_paths": [],
            "receipt_artifact_path": None,
            "return_commands": [],
            "next_return_command_key": "request_bundle",
            "next_return_command": request_bundle_command,
            "error": _text_or_none(row.get("error")),
            "error_type": _text_or_none(row.get("error_type")),
        }
    plan_key = _primary_evidence_status_plan_key(track)
    plan = _mapping(launch_report.get(plan_key), plan_key)
    bundle = _mapping(row.get("request_bundle"), "request_bundle")
    commands = _primary_evidence_return_commands(track, bundle=bundle, plan=plan)
    next_command = commands[0]
    return {
        **base,
        "status": "actionable",
        "expected_return_artifact_paths": (
            _primary_evidence_expected_return_artifact_paths(
                track,
                bundle=bundle,
                plan=plan,
            )
        ),
        "receipt_artifact_path": _primary_evidence_return_receipt_path(
            track,
            plan=plan,
        ),
        "return_commands": commands,
        "next_return_command_key": _text_field(next_command, "key"),
        "next_return_command": _text_field(next_command, "command"),
        "error": None,
        "error_type": None,
    }


def _primary_evidence_expected_return_artifact_paths(
    track: str,
    *,
    bundle: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> list[str]:
    if track == "real_data":
        return sorted(
            [
                *(_string_list(bundle.get("episode_paths"))),
                _text_field(bundle, "report_path"),
            ]
        )
    if track == "real_controls":
        receipt = _mapping(plan.get("prediction_receipt_bundle"), "receipt")
        paths = [
            _text_field(source, "prediction_output_path")
            for source in _mapping_sequence(bundle.get("sources"))
        ]
        paths.append(_text_field(receipt, "path"))
        return sorted(paths)
    if track == "predicted_dsg":
        detector_jsonl = _mapping(bundle.get("detector_jsonl"), "detector_jsonl")
        receipt = _mapping(plan.get("detector_receipt_bundle"), "receipt")
        return sorted(
            [
                _text_field(detector_jsonl, "path"),
                _text_field(receipt, "path"),
            ]
        )
    raise SpatialQAError(
        f"Unknown real experiment primary evidence track: {track}"
    )


def _primary_evidence_return_receipt_path(
    track: str,
    *,
    plan: Mapping[str, Any],
) -> str:
    if track == "real_data":
        return _text_field(plan, "real_collection_report_path")
    if track == "real_controls":
        receipt = _mapping(plan.get("prediction_receipt_bundle"), "receipt")
        return _text_field(receipt, "path")
    if track == "predicted_dsg":
        receipt = _mapping(plan.get("detector_receipt_bundle"), "receipt")
        return _text_field(receipt, "path")
    raise SpatialQAError(
        f"Unknown real experiment primary evidence track: {track}"
    )


def _primary_evidence_return_commands(
    track: str,
    *,
    bundle: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> list[dict[str, str]]:
    if track == "real_data":
        command_map = _mapping(bundle.get("commands"), "commands")
        keys = ("collection_report", "validate_report", "compare_report")
    elif track == "real_controls":
        command_map = _mapping(plan.get("commands"), "commands")
        keys = (
            "prediction_receipt_bundle",
            "preflight_contract",
            "artifact_launch_report",
        )
    elif track == "predicted_dsg":
        command_map = _mapping(plan.get("commands"), "commands")
        keys = (
            "detector_receipt_bundle",
            "preflight_contract",
            "artifact_launch_report",
        )
    else:
        raise SpatialQAError(
            f"Unknown real experiment primary evidence track: {track}"
        )
    return [
        {"key": key, "command": _text_field(command_map, key)}
        for key in keys
    ]


def _primary_evidence_return_checklist_step_valid(
    step: Mapping[str, Any],
) -> bool:
    status = _text_or_none(step.get("status"))
    commands = _mapping_sequence(step.get("return_commands"))
    expected_paths = _string_sequence_or_empty(
        step.get("expected_return_artifact_paths")
    )
    return (
        _text_or_none(step.get("track")) in REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS
        and status in {"actionable", "blocked"}
        and _text_or_none(step.get("request_bundle_path")) is not None
        and _text_or_none(step.get("request_bundle_command")) is not None
        and isinstance(expected_paths, list)
        and (
            (
                status == "actionable"
                and _text_or_none(step.get("request_bundle_digest")) is not None
                and _text_or_none(step.get("receipt_artifact_path")) is not None
                and len(commands) > 0
                and all(
                    _text_or_none(command.get("command")) is not None
                    and _text_or_none(command.get("key")) is not None
                    for command in commands
                )
                and _text_or_none(step.get("next_return_command_key")) is not None
                and _text_or_none(step.get("next_return_command")) is not None
                and step.get("error") is None
                and step.get("error_type") is None
            )
            or (
                status == "blocked"
                and step.get("request_bundle_digest") is None
                and step.get("receipt_artifact_path") is None
                and len(commands) == 0
                and step.get("next_return_command_key") == "request_bundle"
                and _text_or_none(step.get("next_return_command")) is not None
                and _text_or_none(step.get("error")) is not None
                and _text_or_none(step.get("error_type")) is not None
            )
        )
    )


def _primary_evidence_return_progress_track(
    step: Mapping[str, Any],
) -> dict[str, Any]:
    expected_paths = _string_sequence_or_empty(
        step.get("expected_return_artifact_paths")
    )
    present_paths = [path for path in expected_paths if Path(path).exists()]
    missing_paths = [path for path in expected_paths if not Path(path).exists()]
    checklist_status = _text_field(step, "status")
    if checklist_status == "blocked":
        return_status = "blocked"
    elif missing_paths:
        return_status = "missing"
    else:
        return_status = "complete"
    return {
        "track": _text_field(step, "track"),
        "checklist_status": checklist_status,
        "return_status": return_status,
        "expected_return_artifact_paths": expected_paths,
        "present_return_artifact_paths": present_paths,
        "missing_return_artifact_paths": missing_paths,
        "present_return_artifact_count": len(present_paths),
        "missing_return_artifact_count": len(missing_paths),
        "return_artifact_count": len(expected_paths),
        "request_bundle_path": _text_field(step, "request_bundle_path"),
        "receipt_artifact_path": _text_or_none(step.get("receipt_artifact_path")),
        "next_return_command_key": _text_or_none(
            step.get("next_return_command_key")
        ),
        "next_return_command": _text_or_none(step.get("next_return_command")),
        "error": _text_or_none(step.get("error")),
        "error_type": _text_or_none(step.get("error_type")),
    }


def _primary_evidence_return_progress_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    present_count = sum(
        _summary_int(row, "present_return_artifact_count") or 0 for row in rows
    )
    missing_count = sum(
        _summary_int(row, "missing_return_artifact_count") or 0 for row in rows
    )
    artifact_count = sum(
        _summary_int(row, "return_artifact_count") or 0 for row in rows
    )
    blocked_count = sum(1 for row in rows if row.get("return_status") == "blocked")
    complete_count = sum(1 for row in rows if row.get("return_status") == "complete")
    actionable_count = len(rows) - blocked_count
    ready = missing_count == 0 and blocked_count == 0
    return {
        "actionable_return_track_count": actionable_count,
        "all_return_artifacts_present": ready,
        "blocked_return_track_count": blocked_count,
        "complete_return_track_count": complete_count,
        "missing_return_artifact_count": missing_count,
        "present_return_artifact_count": present_count,
        "ready_for_launch_refresh": ready,
        "return_artifact_count": artifact_count,
        "track_count": len(rows),
    }


def _primary_evidence_return_progress_next_missing(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    for row in rows:
        return_status = _text_field(row, "return_status")
        if return_status != "missing":
            continue
        missing_paths = _string_sequence_or_empty(
            row.get("missing_return_artifact_paths")
        )
        return {
            "missing_return_artifact_path": missing_paths[0] if missing_paths else None,
            "next_return_command": _text_or_none(row.get("next_return_command")),
            "next_return_command_key": _text_or_none(
                row.get("next_return_command_key")
            ),
            "return_status": return_status,
            "track": _text_field(row, "track"),
        }
    for row in rows:
        return_status = _text_field(row, "return_status")
        if return_status != "blocked":
            continue
        return {
            "missing_return_artifact_path": None,
            "next_return_command": _text_or_none(row.get("next_return_command")),
            "next_return_command_key": _text_or_none(
                row.get("next_return_command_key")
            ),
            "return_status": return_status,
            "track": _text_field(row, "track"),
        }
    return None


def _primary_evidence_return_progress_track_valid(
    row: Mapping[str, Any],
) -> bool:
    expected_paths = _string_sequence_or_empty(
        row.get("expected_return_artifact_paths")
    )
    present_paths = _string_sequence_or_empty(
        row.get("present_return_artifact_paths")
    )
    missing_paths = _string_sequence_or_empty(
        row.get("missing_return_artifact_paths")
    )
    return_status = _text_or_none(row.get("return_status"))
    checklist_status = _text_or_none(row.get("checklist_status"))
    return (
        _text_or_none(row.get("track")) in REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS
        and checklist_status in {"actionable", "blocked"}
        and return_status in {"blocked", "complete", "missing"}
        and _text_or_none(row.get("request_bundle_path")) is not None
        and _summary_int(row, "present_return_artifact_count") == len(present_paths)
        and _summary_int(row, "missing_return_artifact_count") == len(missing_paths)
        and _summary_int(row, "return_artifact_count") == len(expected_paths)
        and sorted([*present_paths, *missing_paths]) == sorted(expected_paths)
        and (
            (
                return_status == "blocked"
                and checklist_status == "blocked"
                and len(expected_paths) == 0
                and _text_or_none(row.get("error")) is not None
                and _text_or_none(row.get("error_type")) is not None
            )
            or (
                return_status == "missing"
                and checklist_status == "actionable"
                and len(missing_paths) > 0
                and _text_or_none(row.get("receipt_artifact_path")) is not None
                and _text_or_none(row.get("next_return_command")) is not None
                and _text_or_none(row.get("next_return_command_key")) is not None
                and row.get("error") is None
                and row.get("error_type") is None
            )
            or (
                return_status == "complete"
                and checklist_status == "actionable"
                and len(missing_paths) == 0
                and _text_or_none(row.get("receipt_artifact_path")) is not None
                and _text_or_none(row.get("next_return_command")) is not None
                and _text_or_none(row.get("next_return_command_key")) is not None
                and row.get("error") is None
                and row.get("error_type") is None
            )
        )
    )


def _primary_evidence_acceptance_track(
    track: str,
    step: Mapping[str, Any],
    launch_report: Mapping[str, Any],
) -> dict[str, Any]:
    next_command_key = _text_or_none(step.get("next_return_command_key"))
    next_command = _text_or_none(step.get("next_return_command"))
    if _text_field(step, "status") == "blocked":
        return {
            "track": track,
            "status": "blocked",
            "receipt_kind": _primary_evidence_acceptance_receipt_kind(track),
            "receipt_path": _text_or_none(step.get("receipt_artifact_path")),
            "receipt_status": "blocked",
            "receipt_digest": None,
            "digest_valid": False,
            "validation_valid": False,
            "manifest_matches": None,
            "blocking_roles": ["request_bundle"],
            "next_return_command_key": next_command_key,
            "next_return_command": next_command,
            "error": _text_or_none(step.get("error")),
            "error_type": _text_or_none(step.get("error_type")),
        }
    plan_key = _primary_evidence_status_plan_key(track)
    plan = _mapping(launch_report.get(plan_key), plan_key)
    receipt = _primary_evidence_acceptance_receipt(track, plan=plan)
    receipt_status = _text_field(receipt, "status")
    status = _primary_evidence_acceptance_status(
        track,
        plan=plan,
        receipt=receipt,
    )
    return {
        "track": track,
        "status": status,
        "receipt_kind": _primary_evidence_acceptance_receipt_kind(track),
        "receipt_path": _text_or_none(receipt.get("path")),
        "receipt_status": receipt_status,
        "receipt_digest": _primary_evidence_acceptance_receipt_digest(
            track,
            receipt=receipt,
        ),
        "digest_valid": receipt.get("digest_valid") is True,
        "validation_valid": receipt.get("validation_valid") is True,
        "manifest_matches": (
            receipt.get("manifest_matches") is True
            if "manifest_matches" in receipt
            else None
        ),
        "blocking_roles": _string_list(plan.get("blocking_roles")),
        "next_return_command_key": next_command_key,
        "next_return_command": next_command,
        "error": _text_or_none(receipt.get("error")),
        "error_type": _text_or_none(receipt.get("error_type")),
    }


def _primary_evidence_acceptance_receipt(
    track: str,
    *,
    plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    if track == "real_data":
        return _mapping(plan.get("collection_report_receipt"), "receipt")
    if track == "real_controls":
        return _mapping(plan.get("prediction_receipt_bundle"), "receipt")
    if track == "predicted_dsg":
        return _mapping(plan.get("detector_receipt_bundle"), "receipt")
    raise SpatialQAError(
        f"Unknown real experiment primary evidence track: {track}"
    )


def _primary_evidence_acceptance_receipt_kind(track: str) -> str:
    if track == "real_data":
        return "real_collection_report"
    if track == "real_controls":
        return "offline_control_prediction_receipt_bundle"
    if track == "predicted_dsg":
        return "predicted_dsg_detector_receipt_bundle"
    raise SpatialQAError(
        f"Unknown real experiment primary evidence track: {track}"
    )


def _primary_evidence_acceptance_receipt_digest(
    track: str,
    *,
    receipt: Mapping[str, Any],
) -> str | None:
    if track == "real_data":
        return _text_or_none(receipt.get("report_digest"))
    if track in {"real_controls", "predicted_dsg"}:
        return _text_or_none(receipt.get("receipt_bundle_digest"))
    raise SpatialQAError(
        f"Unknown real experiment primary evidence track: {track}"
    )


def _primary_evidence_acceptance_status(
    track: str,
    *,
    plan: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> str:
    if plan.get("ready") is True:
        return "accepted"
    receipt_status = _text_field(receipt, "status")
    if receipt_status in {"missing", "invalid"}:
        return receipt_status
    if receipt_status == "ready" and track != "real_data":
        if receipt.get("manifest_matches") is not True:
            return "invalid"
    return "not_ready"


def _primary_evidence_acceptance_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    accepted_count = sum(1 for row in rows if row.get("status") == "accepted")
    blocked_count = sum(1 for row in rows if row.get("status") == "blocked")
    invalid_count = sum(1 for row in rows if row.get("status") == "invalid")
    missing_count = sum(1 for row in rows if row.get("status") == "missing")
    not_ready_count = sum(1 for row in rows if row.get("status") == "not_ready")
    all_accepted = accepted_count == len(rows) and len(rows) > 0
    return {
        "accepted_track_count": accepted_count,
        "all_tracks_accepted": all_accepted,
        "blocked_track_count": blocked_count,
        "invalid_track_count": invalid_count,
        "missing_track_count": missing_count,
        "not_ready_track_count": not_ready_count,
        "ready_for_launch_refresh": all_accepted,
        "track_count": len(rows),
    }


def _primary_evidence_acceptance_next_unaccepted(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    for row in rows:
        if row.get("status") == "accepted":
            continue
        return {
            "next_return_command": _text_or_none(row.get("next_return_command")),
            "next_return_command_key": _text_or_none(
                row.get("next_return_command_key")
            ),
            "receipt_path": _text_or_none(row.get("receipt_path")),
            "receipt_status": _text_or_none(row.get("receipt_status")),
            "status": _text_field(row, "status"),
            "track": _text_field(row, "track"),
        }
    return None


def _primary_evidence_acceptance_track_valid(row: Mapping[str, Any]) -> bool:
    status = _text_or_none(row.get("status"))
    track = _text_or_none(row.get("track"))
    manifest_matches = row.get("manifest_matches")
    return (
        track in REAL_EXPERIMENT_PRIMARY_EVIDENCE_TRACKS
        and status in {"accepted", "blocked", "invalid", "missing", "not_ready"}
        and _text_or_none(row.get("receipt_kind")) is not None
        and _text_or_none(row.get("receipt_status")) is not None
        and isinstance(row.get("digest_valid"), bool)
        and isinstance(row.get("validation_valid"), bool)
        and (manifest_matches is None or isinstance(manifest_matches, bool))
        and _text_or_none(row.get("next_return_command_key")) is not None
        and _text_or_none(row.get("next_return_command")) is not None
        and (
            status != "accepted"
            or (
                _text_or_none(row.get("receipt_path")) is not None
                and _text_or_none(row.get("receipt_digest")) is not None
                and row.get("digest_valid") is True
                and row.get("validation_valid") is True
                and (
                    track == "real_data"
                    or row.get("manifest_matches") is True
                )
                and row.get("error") is None
                and row.get("error_type") is None
            )
        )
    )


def _launch_report_real_data_collection_intake_plan(
    contracts: Mapping[str, Any],
    tracks: Mapping[str, Mapping[str, Any]],
    child_launch_gates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    contract_tracks = _mapping(contracts.get("tracks"), "tracks")
    real_data = _mapping(contract_tracks.get("real_data"), "real_data")
    track = _mapping(tracks.get("real_data"), "real_data")
    child_gate = _mapping(child_launch_gates.get("real_data"), "real_data")
    track_ready = track.get("ready") is True
    collection_report_receipt = _launch_report_real_collection_report_receipt(
        _text_field(child_gate, "real_collection_report_path")
    )
    ready = track_ready and collection_report_receipt["ready"] is True
    blocking_roles = _string_list(track.get("blocking_roles"))
    if not ready and "real_collection_report" not in blocking_roles:
        blocking_roles.append("real_collection_report")
    return {
        "track": "real_data",
        "blocked": not ready,
        "blocking_roles": blocking_roles,
        "commands": {
            "collection_report": _text_field(
                child_gate,
                "collection_report_command",
            ),
            "compare_report": _text_field(child_gate, "compare_report_command"),
            "request_bundle": _text_field(child_gate, "request_bundle_command"),
            "validate_report": _text_field(child_gate, "validate_report_command"),
        },
        "dataset_name": _text_field(real_data, "dataset_name"),
        "episode_paths": _string_list(child_gate.get("episode_paths")),
        "invalid_inputs": _json_value(track.get("invalid_inputs", [])),
        "missing_inputs": _json_value(track.get("missing_inputs", [])),
        "collection_report_receipt": collection_report_receipt,
        "real_collection_report_path": _text_field(
            child_gate,
            "real_collection_report_path",
        ),
        "ready": ready,
        "source_kind": _text_field(child_gate, "source_kind"),
        "thresholds": {
            "min_episode_count": _int_field(real_data, "min_episode_count"),
            "min_frame_count": _int_field(real_data, "min_frame_count"),
            "min_qa_count": _int_field(real_data, "min_qa_count"),
            "min_scene_count": _int_field(real_data, "min_scene_count"),
        },
    }


def _launch_report_real_controls_prediction_intake_plan(
    tracks: Mapping[str, Mapping[str, Any]],
    child_launch_gates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    track = _mapping(tracks.get("real_controls"), "real_controls")
    child_gate = _mapping(child_launch_gates.get("offline_controls"), "offline_controls")
    track_ready = track.get("ready") is True
    artifact_contract_receipt = _launch_report_offline_control_artifact_receipt(
        artifact_contract_path=_text_field(child_gate, "artifact_contract_path"),
        manifest_path=_text_field(child_gate, "manifest_path"),
    )
    prediction_receipt_bundle = _launch_report_offline_control_prediction_bundle_receipt(
        manifest_path=_text_field(child_gate, "manifest_path"),
        receipt_bundle_path=_text_field(
            child_gate,
            "prediction_receipt_bundle_path",
        ),
    )
    ready = (
        track_ready
        and artifact_contract_receipt["ready_to_import"] is True
        and prediction_receipt_bundle["ready_to_import"] is True
    )
    blocking_roles = _string_list(track.get("blocking_roles"))
    if not ready and "offline_control_source_input" not in blocking_roles:
        blocking_roles.append("offline_control_source_input")
    return {
        "track": "real_controls",
        "blocked": not ready,
        "blocking_roles": blocking_roles,
        "commands": {
            "artifact_launch_report": _text_field(
                child_gate,
                "artifact_launch_report_command",
            ),
            "preflight_contract": _text_field(
                child_gate,
                "preflight_contract_command",
            ),
            "prediction_receipt_bundle": _text_field(
                child_gate,
                "prediction_receipt_bundle_command",
            ),
            "prediction_request_bundle": _text_field(
                child_gate,
                "prediction_request_bundle_command",
            ),
        },
        "invalid_inputs": _json_value(track.get("invalid_inputs", [])),
        "missing_inputs": _json_value(track.get("missing_inputs", [])),
        "artifact_contract_path": _text_field(child_gate, "artifact_contract_path"),
        "artifact_contract_receipt": artifact_contract_receipt,
        "manifest_path": _text_field(child_gate, "manifest_path"),
        "prediction_receipt_bundle": prediction_receipt_bundle,
        "ready": ready,
    }


def _launch_report_predicted_dsg_detector_intake_plan(
    tracks: Mapping[str, Mapping[str, Any]],
    child_launch_gates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    track = _mapping(tracks.get("predicted_dsg"), "predicted_dsg")
    child_gate = _mapping(child_launch_gates.get("predicted_dsg"), "predicted_dsg")
    track_ready = track.get("ready") is True
    artifact_contract_receipt = _launch_report_predicted_dsg_artifact_receipt(
        artifact_contract_path=_text_field(child_gate, "artifact_contract_path"),
        manifest_path=_text_field(child_gate, "manifest_path"),
    )
    detector_receipt_bundle = _launch_report_predicted_dsg_detector_bundle_receipt(
        manifest_path=_text_field(child_gate, "manifest_path"),
        receipt_bundle_path=_text_field(child_gate, "detector_receipt_bundle_path"),
    )
    ready = (
        track_ready
        and artifact_contract_receipt["ready_to_build"] is True
        and detector_receipt_bundle["ready_to_build"] is True
    )
    blocking_roles = _string_list(track.get("blocking_roles"))
    if not ready and "detector_jsonl" not in blocking_roles:
        blocking_roles.append("detector_jsonl")
    return {
        "track": "predicted_dsg",
        "blocked": not ready,
        "blocking_roles": blocking_roles,
        "commands": {
            "artifact_launch_report": _text_field(
                child_gate,
                "artifact_launch_report_command",
            ),
            "detector_receipt_bundle": _text_field(
                child_gate,
                "detector_receipt_bundle_command",
            ),
            "detector_request_bundle": _text_field(
                child_gate,
                "detector_request_bundle_command",
            ),
            "preflight_contract": _text_field(
                child_gate,
                "preflight_contract_command",
            ),
        },
        "invalid_inputs": _json_value(track.get("invalid_inputs", [])),
        "missing_inputs": _json_value(track.get("missing_inputs", [])),
        "artifact_contract_path": _text_field(child_gate, "artifact_contract_path"),
        "artifact_contract_receipt": artifact_contract_receipt,
        "detector_receipt_bundle": detector_receipt_bundle,
        "manifest_path": _text_field(child_gate, "manifest_path"),
        "ready": ready,
    }


def _launch_report_offline_control_prediction_bundle_receipt(
    *,
    manifest_path: str,
    receipt_bundle_path: str,
) -> dict[str, Any]:
    bundle_path = Path(receipt_bundle_path)
    if not bundle_path.exists():
        return {
            "digest_valid": False,
            "manifest_matches": False,
            "manifest_path": manifest_path,
            "path": receipt_bundle_path,
            "ready_to_import": False,
            "receipt_bundle_digest": None,
            "status": "missing",
            "summary": None,
            "validation_valid": False,
        }
    try:
        bundle = load_offline_control_prediction_receipt_bundle(bundle_path)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        return {
            "digest_valid": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "manifest_matches": False,
            "manifest_path": manifest_path,
            "path": receipt_bundle_path,
            "ready_to_import": False,
            "receipt_bundle_digest": None,
            "status": "invalid",
            "summary": None,
            "validation_valid": False,
        }
    validation = validate_offline_control_prediction_receipt_bundle(bundle)
    receipt_digest = _text_or_none(bundle.get("receipt_bundle_digest"))
    digest_valid = (
        receipt_digest == offline_control_prediction_receipt_bundle_digest(bundle)
    )
    manifest_matches = _text_or_none(bundle.get("manifest_path")) == manifest_path
    validation_valid = validation.get("valid") is True
    ready_to_import = (
        bundle.get("ready_to_import") is True
        and digest_valid
        and manifest_matches
        and validation_valid
    )
    return {
        "digest_valid": digest_valid,
        "manifest_matches": manifest_matches,
        "manifest_path": manifest_path,
        "path": receipt_bundle_path,
        "ready_to_import": ready_to_import,
        "receipt_bundle_digest": receipt_digest,
        "status": (
            "ready"
            if ready_to_import
            else "invalid"
            if not validation_valid
            else "not_ready"
        ),
        "summary": _json_value(bundle.get("summary")),
        "validation_valid": validation_valid,
    }


def _launch_report_predicted_dsg_detector_bundle_receipt(
    *,
    manifest_path: str,
    receipt_bundle_path: str,
) -> dict[str, Any]:
    bundle_path = Path(receipt_bundle_path)
    if not bundle_path.exists():
        return {
            "digest_valid": False,
            "manifest_matches": False,
            "manifest_path": manifest_path,
            "path": receipt_bundle_path,
            "ready_to_build": False,
            "receipt_bundle_digest": None,
            "status": "missing",
            "summary": None,
            "validation_valid": False,
        }
    try:
        bundle = load_predicted_dsg_detector_receipt_bundle(bundle_path)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        return {
            "digest_valid": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "manifest_matches": False,
            "manifest_path": manifest_path,
            "path": receipt_bundle_path,
            "ready_to_build": False,
            "receipt_bundle_digest": None,
            "status": "invalid",
            "summary": None,
            "validation_valid": False,
        }
    validation = validate_predicted_dsg_detector_receipt_bundle(bundle)
    receipt_digest = _text_or_none(bundle.get("receipt_bundle_digest"))
    digest_valid = receipt_digest == predicted_dsg_detector_receipt_bundle_digest(
        bundle
    )
    manifest_matches = _text_or_none(bundle.get("manifest_path")) == manifest_path
    validation_valid = validation.get("valid") is True
    ready_to_build = (
        bundle.get("ready_to_build") is True
        and digest_valid
        and manifest_matches
        and validation_valid
    )
    return {
        "digest_valid": digest_valid,
        "manifest_matches": manifest_matches,
        "manifest_path": manifest_path,
        "path": receipt_bundle_path,
        "ready_to_build": ready_to_build,
        "receipt_bundle_digest": receipt_digest,
        "status": (
            "ready"
            if ready_to_build
            else "invalid"
            if not validation_valid
            else "not_ready"
        ),
        "summary": _json_value(bundle.get("summary")),
        "validation_valid": validation_valid,
    }


def _launch_report_predicted_dsg_artifact_receipt(
    *,
    artifact_contract_path: str,
    manifest_path: str,
) -> dict[str, Any]:
    contract_path = Path(artifact_contract_path)
    if not contract_path.exists():
        return {
            "actionable_blockers": None,
            "external_detector_intake_plan": None,
            "manifest_path": manifest_path,
            "path": artifact_contract_path,
            "ready_to_build": False,
            "report_digest": None,
            "status": "missing",
            "summary": None,
        }
    try:
        child_report = predicted_dsg_detector_artifact_launch_report(
            load_predicted_dsg_detector_artifact_contract(contract_path),
            manifest_path=manifest_path,
            contract_path=contract_path,
        )
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        return {
            "actionable_blockers": None,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "external_detector_intake_plan": None,
            "manifest_path": manifest_path,
            "path": artifact_contract_path,
            "ready_to_build": False,
            "report_digest": None,
            "status": "invalid",
            "summary": None,
        }
    ready_to_build = child_report.get("ready_to_build") is True
    return {
        "actionable_blockers": _json_value(
            child_report.get("actionable_blockers", {})
        ),
        "external_detector_intake_plan": _json_value(
            child_report.get("external_detector_intake_plan", {})
        ),
        "manifest_path": manifest_path,
        "path": artifact_contract_path,
        "ready_to_build": ready_to_build,
        "report_digest": _text_or_none(child_report.get("report_digest")),
        "status": "ready" if ready_to_build else "not_ready",
        "summary": _json_value(child_report.get("summary", {})),
    }


def _launch_report_offline_control_artifact_receipt(
    *,
    artifact_contract_path: str,
    manifest_path: str,
) -> dict[str, Any]:
    contract_path = Path(artifact_contract_path)
    if not contract_path.exists():
        return {
            "actionable_blockers": None,
            "external_prediction_intake_plan": None,
            "manifest_path": manifest_path,
            "path": artifact_contract_path,
            "ready_to_import": False,
            "report_digest": None,
            "status": "missing",
            "summary": None,
        }
    try:
        child_report = offline_control_artifact_launch_report(
            load_offline_control_artifact_contracts(contract_path),
            manifest_path=manifest_path,
            contracts_path=contract_path,
        )
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        return {
            "actionable_blockers": None,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "external_prediction_intake_plan": None,
            "manifest_path": manifest_path,
            "path": artifact_contract_path,
            "ready_to_import": False,
            "report_digest": None,
            "status": "invalid",
            "summary": None,
        }
    ready_to_import = child_report.get("ready_to_import") is True
    return {
        "actionable_blockers": _json_value(
            child_report.get("actionable_blockers", {})
        ),
        "external_prediction_intake_plan": _json_value(
            child_report.get("external_prediction_intake_plan", {})
        ),
        "manifest_path": manifest_path,
        "path": artifact_contract_path,
        "ready_to_import": ready_to_import,
        "report_digest": _text_or_none(child_report.get("report_digest")),
        "status": "ready" if ready_to_import else "not_ready",
        "summary": _json_value(child_report.get("summary", {})),
    }


def _launch_report_real_collection_report_receipt(path: str) -> dict[str, Any]:
    report_path = Path(path)
    if not report_path.exists():
        return {
            "asset_summary": None,
            "digest_valid": False,
            "failed_checks": ["real_collection_report_missing"],
            "path": path,
            "readiness": None,
            "ready": False,
            "report_digest": None,
            "status": "missing",
            "validation_valid": False,
        }
    try:
        report = load_real_collection_report(report_path)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        return {
            "asset_summary": None,
            "digest_valid": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "failed_checks": ["real_collection_report_invalid"],
            "path": path,
            "readiness": None,
            "ready": False,
            "report_digest": None,
            "status": "invalid",
            "validation_valid": False,
        }

    validation = validate_real_collection_report(report)
    digest_valid = any(
        check.get("name") == "report_digest" and check.get("passed") is True
        for check in _mapping_sequence(validation.get("checks"))
    )
    validation_valid = validation.get("valid") is True
    readiness_payload = report.get("readiness")
    readiness = (
        _json_value(readiness_payload)
        if isinstance(readiness_payload, Mapping)
        else None
    )
    failed_checks = (
        _string_list(readiness_payload.get("failed_checks"))
        if isinstance(readiness_payload, Mapping)
        else []
    )
    report_ready = (
        isinstance(readiness_payload, Mapping) and readiness_payload.get("ready") is True
    )
    ready = report_ready and validation_valid
    if not digest_valid and "real_collection_report_digest_invalid" not in failed_checks:
        failed_checks.append("real_collection_report_digest_invalid")
    if (
        not validation_valid
        and digest_valid
        and "real_collection_report_validation_failed" not in failed_checks
    ):
        failed_checks.append("real_collection_report_validation_failed")
    if not report_ready and not failed_checks:
        failed_checks = ["real_collection_report_not_ready"]

    collection_summary = report.get("collection_summary")
    asset_summary = None
    if isinstance(collection_summary, Mapping):
        asset_summary_payload = collection_summary.get("asset_summary")
        if isinstance(asset_summary_payload, Mapping):
            asset_summary = _json_value(asset_summary_payload)
    if ready:
        status = "ready"
    elif not validation_valid:
        status = "invalid"
    else:
        status = "not_ready"
    return {
        "asset_summary": asset_summary,
        "digest_valid": digest_valid,
        "failed_checks": failed_checks,
        "path": path,
        "readiness": readiness,
        "ready": ready,
        "report_digest": _text_or_none(report.get("report_digest")),
        "status": status,
        "validation_valid": validation_valid,
    }


def _real_collection_report_command(
    *,
    dataset_name: str,
    source_kind: str,
    episode_paths: Sequence[Path],
    report_path: Path,
    min_episode_count: int,
    min_scene_count: int,
    min_frame_count: int,
) -> str:
    episode_args = " ".join(f"--episode {path}" for path in episode_paths)
    return (
        "python scripts/check_real_collection.py "
        f"--dataset-name {dataset_name} "
        f"--source-kind {source_kind} "
        f"{episode_args} "
        f"--report {report_path} "
        f"--min-episode-count {min_episode_count} "
        f"--min-scene-count {min_scene_count} "
        f"--min-frame-count {min_frame_count}"
    )


def _real_collection_request_bundle_command(
    *,
    bundle_path: Path,
    dataset_name: str,
    source_kind: str,
    episode_paths: Sequence[Path],
    report_path: Path,
    min_episode_count: int,
    min_scene_count: int,
    min_frame_count: int,
) -> str:
    episode_args = " ".join(f"--episode {path}" for path in episode_paths)
    return (
        "python scripts/check_real_collection.py "
        f"--request-bundle {bundle_path} "
        f"--dataset-name {dataset_name} "
        f"--source-kind {source_kind} "
        f"{episode_args} "
        f"--report {report_path} "
        f"--min-episode-count {min_episode_count} "
        f"--min-scene-count {min_scene_count} "
        f"--min-frame-count {min_frame_count}"
    )


def _review_report_command_rows(
    paths: Sequence[Path],
    *,
    script: str,
    validate_flag: str,
    compare_flag: str | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        row = {
            "path": str(path),
            "validate_command": f"python {script} {validate_flag} {path}",
        }
        if compare_flag is not None:
            row["compare_command"] = f"python {script} {compare_flag} {path}"
        rows.append(row)
    return rows


def _launch_report_next_commands(
    *,
    contracts_path: str | None,
    run_manifest_path: str,
    run_ledger_path: str | None,
) -> dict[str, str]:
    run_command = (
        "python scripts/run_real_experiment.py --run-manifest "
        f"{run_manifest_path}"
    )
    if run_ledger_path is not None:
        run_command = f"{run_command} --run-ledger-output {run_ledger_path}"
    commands = {
        "preflight": (
            "python scripts/run_real_experiment.py --preflight-run-manifest "
            f"{run_manifest_path}"
        ),
        "run": run_command,
    }
    if contracts_path is not None:
        commands["compare_external_artifact_contracts"] = (
            "python scripts/run_real_experiment.py "
            f"--compare-external-artifact-contracts {contracts_path}"
        )
        commands["validate_external_artifact_contracts"] = (
            "python scripts/run_real_experiment.py "
            f"--validate-external-artifact-contracts {contracts_path}"
        )
    return {key: commands[key] for key in sorted(commands)}


def _execution_packet_primary_evidence_acceptance_path(
    *,
    launch_report_path: str,
    primary_evidence_acceptance_report_path: str | Path | None,
) -> Path:
    if primary_evidence_acceptance_report_path is not None:
        return Path(primary_evidence_acceptance_report_path)
    return Path(launch_report_path).with_name(
        "real-experiment-primary-evidence-acceptance-report.json"
    )


def _execution_packet_primary_evidence_acceptance(path: Path) -> dict[str, Any]:
    path_text = str(path)
    if not path.exists():
        return {
            "acceptance_digest": None,
            "all_tracks_accepted": False,
            "matches_current": False,
            "path": path_text,
            "present": False,
            "ready_for_launch_refresh": False,
            "summary": None,
            "valid": False,
        }
    try:
        report = load_real_experiment_primary_evidence_acceptance_report(path)
        validation = validate_real_experiment_primary_evidence_acceptance_report(
            report
        )
        comparison = compare_real_experiment_primary_evidence_acceptance_report(
            report
        )
        summary = _mapping(report.get("summary"), "summary")
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        return {
            "acceptance_digest": None,
            "all_tracks_accepted": False,
            "error": str(exc),
            "matches_current": False,
            "path": path_text,
            "present": True,
            "ready_for_launch_refresh": False,
            "summary": None,
            "valid": False,
        }
    return {
        "acceptance_digest": _text_or_none(report.get("acceptance_digest")),
        "all_tracks_accepted": summary.get("all_tracks_accepted") is True,
        "matches_current": comparison["matches"] is True,
        "path": path_text,
        "present": True,
        "ready_for_launch_refresh": summary.get("ready_for_launch_refresh") is True,
        "summary": _json_value(summary),
        "valid": validation["valid"] is True,
    }


def _execution_packet_audit_commands(
    *,
    launch_report_path: str,
    contracts_path: str | None,
    primary_evidence_acceptance_report_path: str,
) -> list[dict[str, Any]]:
    commands = [
        {
            "key": "validate_launch_report",
            "command": (
                "python scripts/run_real_experiment.py "
                f"--validate-external-artifact-launch-report {launch_report_path}"
            ),
            "order": 1,
            "phase": "audit",
            "required": True,
        },
        {
            "key": "compare_launch_report",
            "command": (
                "python scripts/run_real_experiment.py "
                f"--compare-external-artifact-launch-report {launch_report_path}"
            ),
            "order": 2,
            "phase": "audit",
            "required": True,
        },
    ]
    next_order = len(commands) + 1
    if contracts_path is not None:
        commands.append(
            {
                "key": "refresh_launch_report",
                "command": (
                    "python scripts/run_real_experiment.py "
                    f"--external-artifact-launch-report {contracts_path} "
                    f"--launch-report-output {launch_report_path}"
                ),
                "order": next_order,
                "phase": "audit",
                "required": False,
            }
        )
        next_order += 1
    commands.extend(
        [
            {
                "key": "validate_primary_evidence_acceptance_report",
                "command": (
                    "python scripts/run_real_experiment.py "
                    "--validate-primary-evidence-acceptance-report "
                    f"{primary_evidence_acceptance_report_path}"
                ),
                "order": next_order,
                "phase": "audit",
                "required": True,
            },
            {
                "key": "compare_primary_evidence_acceptance_report",
                "command": (
                    "python scripts/run_real_experiment.py "
                    "--compare-primary-evidence-acceptance-report "
                    f"{primary_evidence_acceptance_report_path}"
                ),
                "order": next_order + 1,
                "phase": "audit",
                "required": True,
            },
        ]
    )
    return commands


def _execution_packet_execution_commands(
    next_commands: Mapping[str, Any],
    *,
    ready_to_execute: bool,
    approved_execution_packet_path: str,
) -> list[dict[str, Any]]:
    if not ready_to_execute:
        return []
    run_command = (
        f"{_text_field(next_commands, 'run')} "
        f"--approved-execution-packet {approved_execution_packet_path}"
    )
    return [
        {
            "key": "preflight_run_manifest",
            "command": _text_field(next_commands, "preflight"),
            "order": 1,
            "phase": "execute",
            "required": True,
        },
        {
            "key": "run_real_experiment",
            "command": run_command,
            "order": 2,
            "phase": "execute",
            "required": True,
        },
    ]


def _execution_receipt_run_ledger_approval(
    run_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    run_ledger_path = _text_or_none(
        run_manifest.get("real_experiment_run_ledger_path")
    )
    if run_ledger_path is None:
        return {
            "path": None,
            "present": None,
            "ready": True,
            "required": False,
            "status": "not_required",
            "validation_valid": None,
        }
    if not Path(run_ledger_path).exists():
        return {
            "path": run_ledger_path,
            "present": False,
            "ready": False,
            "required": True,
            "status": "missing",
            "validation_valid": False,
        }
    try:
        ledger = load_real_experiment_run_ledger(run_ledger_path)
        validation = validate_real_experiment_run_ledger(ledger)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        return {
            "path": run_ledger_path,
            "present": True,
            "ready": False,
            "required": True,
            "status": "invalid",
            "validation_valid": False,
            "error": str(exc),
        }
    execution_approval = _mapping(
        ledger.get("execution_approval"),
        "execution_approval",
    )
    approval_ready = execution_approval.get("ready") is True
    approval_required = execution_approval.get("required") is True
    ledger_ready = ledger.get("ready") is True
    validation_valid = validation["valid"] is True
    ready = validation_valid and ledger_ready and approval_required and approval_ready
    return {
        "path": run_ledger_path,
        "present": True,
        "ready": ready,
        "required": True,
        "status": "approved" if ready else "not_approved",
        "validation_valid": validation_valid,
    }


def _execution_receipt_artifacts(
    run_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    artifacts = [
        _directory_execution_artifact_receipt(
            role="output_dir",
            path=_text_field(run_manifest, "output_dir"),
        ),
        _json_execution_artifact_receipt(
            role="benchmark_manifest",
            path=_text_field(run_manifest, "manifest_path"),
            digest_field="manifest_digest",
            digest_fn=benchmark_manifest_digest,
            load_fn=load_benchmark_manifest,
            validate_fn=validate_benchmark_manifest,
        ),
        _json_execution_artifact_receipt(
            role="real_readiness_report",
            path=_text_field(run_manifest, "readiness_report_path"),
            digest_field="report_digest",
            digest_fn=real_experiment_readiness_report_digest,
            load_fn=load_real_experiment_readiness_report,
            validate_fn=validate_real_experiment_readiness_report,
        ),
        _json_execution_artifact_receipt(
            role="experiment_summary",
            path=_text_field(run_manifest, "summary_report_path"),
            digest_field="report_digest",
            digest_fn=experiment_summary_report_digest,
            load_fn=load_experiment_summary_report,
            validate_fn=validate_experiment_summary_report,
        ),
        _json_execution_artifact_receipt(
            role="experiment_record",
            path=_text_field(run_manifest, "record_path"),
            digest_field="record_digest",
            digest_fn=experiment_record_digest,
            load_fn=load_experiment_record,
            validate_fn=validate_experiment_record,
        ),
    ]
    run_ledger_path = _text_or_none(
        run_manifest.get("real_experiment_run_ledger_path")
    )
    if run_ledger_path is not None:
        artifacts.append(
            _json_execution_artifact_receipt(
                role="real_experiment_run_ledger",
                path=run_ledger_path,
                digest_field="ledger_digest",
                digest_fn=real_experiment_run_ledger_digest,
                load_fn=load_real_experiment_run_ledger,
                validate_fn=validate_real_experiment_run_ledger,
            )
        )
    offline_ledger_path = _text_or_none(
        run_manifest.get("offline_control_import_run_ledger_path")
    )
    if offline_ledger_path is not None:
        artifacts.append(
            _json_execution_artifact_receipt(
                role="offline_control_import_run_ledger",
                path=offline_ledger_path,
                digest_field="ledger_digest",
                digest_fn=offline_control_import_run_ledger_digest,
                load_fn=load_offline_control_import_run_ledger,
                validate_fn=validate_offline_control_import_run_ledger,
            )
        )
    predicted_ledger_path = _text_or_none(
        run_manifest.get("predicted_dsg_detector_run_ledger_path")
    )
    if predicted_ledger_path is not None:
        artifacts.append(
            _json_execution_artifact_receipt(
                role="predicted_dsg_detector_run_ledger",
                path=predicted_ledger_path,
                digest_field="ledger_digest",
                digest_fn=predicted_dsg_detector_run_ledger_digest,
                load_fn=load_predicted_dsg_detector_run_ledger,
                validate_fn=validate_predicted_dsg_detector_run_ledger,
            )
        )
    return artifacts


def _smoke_run_receipt_output_path(
    execution_packet_path: str | Path,
    *,
    execution_receipt_output_path: str | Path | None,
) -> str:
    if execution_receipt_output_path is not None:
        return str(execution_receipt_output_path)
    return str(Path(execution_packet_path).with_name("real-experiment-execution-receipt.json"))


def _smoke_run_checklist_steps(
    execution_packet: Mapping[str, Any],
    *,
    execution_packet_path: str,
    execution_receipt_output_path: str,
    run_ledger_path: str | None,
    ready_to_start: bool,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = [
        {
            "key": "validate_execution_packet",
            "command": (
                "python scripts/run_real_experiment.py "
                f"--validate-execution-packet {execution_packet_path}"
            ),
            "phase": "audit",
            "required": True,
            "source": "smoke_run_checklist",
        },
        {
            "key": "compare_execution_packet",
            "command": (
                "python scripts/run_real_experiment.py "
                f"--compare-execution-packet {execution_packet_path}"
            ),
            "phase": "audit",
            "required": True,
            "source": "smoke_run_checklist",
        },
    ]
    for command in _mapping_sequence(execution_packet.get("audit_commands")):
        steps.append(
            {
                "key": _text_field(command, "key"),
                "command": _text_field(command, "command"),
                "phase": "audit",
                "required": command.get("required") is not False,
                "source": "execution_packet.audit_commands",
            }
        )
    if ready_to_start:
        for command in _mapping_sequence(execution_packet.get("execution_commands")):
            steps.append(
                {
                    "key": _text_field(command, "key"),
                    "command": _text_field(command, "command"),
                    "phase": "execute",
                    "required": command.get("required") is not False,
                    "source": "execution_packet.execution_commands",
                }
            )
        if run_ledger_path is not None:
            steps.extend(
                [
                    {
                        "key": "validate_run_ledger",
                        "command": (
                            "python scripts/run_real_experiment.py "
                            f"--validate-run-ledger {run_ledger_path}"
                        ),
                        "phase": "review",
                        "required": True,
                        "source": "smoke_run_checklist",
                    },
                    {
                        "key": "compare_run_ledger",
                        "command": (
                            "python scripts/run_real_experiment.py "
                            f"--compare-run-ledger {run_ledger_path}"
                        ),
                        "phase": "review",
                        "required": True,
                        "source": "smoke_run_checklist",
                    },
                ]
            )
        steps.extend(
            [
                {
                    "key": "write_execution_receipt",
                    "command": (
                        "python scripts/run_real_experiment.py "
                        f"--execution-receipt {execution_packet_path} "
                        f"--execution-receipt-output {execution_receipt_output_path}"
                    ),
                    "phase": "review",
                    "required": True,
                    "source": "smoke_run_checklist",
                },
                {
                    "key": "validate_execution_receipt",
                    "command": (
                        "python scripts/run_real_experiment.py "
                        f"--validate-execution-receipt {execution_receipt_output_path}"
                    ),
                    "phase": "review",
                    "required": True,
                    "source": "smoke_run_checklist",
                },
                {
                    "key": "compare_execution_receipt",
                    "command": (
                        "python scripts/run_real_experiment.py "
                        f"--compare-execution-receipt {execution_receipt_output_path}"
                    ),
                    "phase": "review",
                    "required": False,
                    "source": "smoke_run_checklist",
                },
            ]
        )
    return [
        {
            **step,
            "order": index,
        }
        for index, step in enumerate(steps, start=1)
    ]


def _smoke_run_checklist_summary(
    steps: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    return {
        "audit_step_count": sum(1 for step in steps if step.get("phase") == "audit"),
        "execute_step_count": sum(
            1 for step in steps if step.get("phase") == "execute"
        ),
        "required_step_count": sum(1 for step in steps if step.get("required") is True),
        "review_step_count": sum(1 for step in steps if step.get("phase") == "review"),
        "step_count": len(steps),
    }


def _smoke_run_runbook_commands(
    checklist: Mapping[str, Any],
) -> list[dict[str, Any]]:
    commands = [
        {
            "command": _text_field(step, "command"),
            "key": _text_field(step, "key"),
            "order": _summary_int(step, "order") or 0,
            "phase": _text_field(step, "phase"),
            "required": step.get("required") is True,
            "source": _text_field(step, "source"),
        }
        for step in _mapping_sequence(checklist.get("steps"))
    ]
    return sorted(commands, key=lambda command: _summary_int(command, "order") or 0)


def _smoke_run_runbook_summary(
    commands: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    return {
        "audit_command_count": sum(
            1 for command in commands if command.get("phase") == "audit"
        ),
        "command_count": len(commands),
        "execute_command_count": sum(
            1 for command in commands if command.get("phase") == "execute"
        ),
        "required_command_count": sum(
            1 for command in commands if command.get("required") is True
        ),
        "review_command_count": sum(
            1 for command in commands if command.get("phase") == "review"
        ),
    }


def _smoke_run_planned_outputs(
    run_manifest: Mapping[str, Any],
    *,
    execution_receipt_output_path: str,
) -> dict[str, str]:
    outputs = {
        "benchmark_manifest": _text_field(run_manifest, "manifest_path"),
        "execution_receipt": execution_receipt_output_path,
        "experiment_record": _text_field(run_manifest, "record_path"),
        "experiment_summary": _text_field(run_manifest, "summary_report_path"),
        "output_dir": _text_field(run_manifest, "output_dir"),
        "real_readiness_report": _text_field(run_manifest, "readiness_report_path"),
    }
    offline_ledger_path = _text_or_none(
        run_manifest.get("offline_control_import_run_ledger_path")
    )
    if offline_ledger_path is not None:
        outputs["offline_control_import_run_ledger"] = offline_ledger_path
    predicted_ledger_path = _text_or_none(
        run_manifest.get("predicted_dsg_detector_run_ledger_path")
    )
    if predicted_ledger_path is not None:
        outputs["predicted_dsg_detector_run_ledger"] = predicted_ledger_path
    run_ledger_path = _text_or_none(
        run_manifest.get("real_experiment_run_ledger_path")
    )
    if run_ledger_path is not None:
        outputs["real_experiment_run_ledger"] = run_ledger_path
    return {key: outputs[key] for key in sorted(outputs)}


def _execution_receipt_artifact_path(
    execution_receipt: Mapping[str, Any],
    role: str,
) -> str | None:
    for artifact in _mapping_sequence(execution_receipt.get("artifacts")):
        if artifact.get("role") == role:
            return _text_or_none(artifact.get("path"))
    return None


def _research_review_json_artifact(
    *,
    role: str,
    path: str | None,
    digest_field: str,
    digest_fn: Any,
    load_fn: Any,
    validate_fn: Any,
) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    if path is None:
        return (
            {
                "role": role,
                "path": None,
                "kind": "json",
                "exists": False,
                "status": "missing",
                "digest": None,
                "digest_valid": False,
                "valid": False,
                "validation": None,
            },
            None,
        )
    artifact_path = Path(path)
    if not artifact_path.exists():
        return (
            {
                "role": role,
                "path": path,
                "kind": "json",
                "exists": False,
                "status": "missing",
                "digest": None,
                "digest_valid": False,
                "valid": False,
                "validation": None,
            },
            None,
        )
    try:
        payload = load_fn(artifact_path)
        digest = _text_or_none(payload.get(digest_field))
        expected_digest = digest_fn(payload)
        digest_valid = digest == expected_digest
        validation = validate_fn(payload)
        valid = validation.get("valid") is True
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        return (
            {
                "role": role,
                "path": path,
                "kind": "json",
                "exists": True,
                "status": "invalid",
                "digest": None,
                "digest_valid": False,
                "valid": False,
                "validation": None,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
            None,
        )
    return (
        {
            "role": role,
            "path": path,
            "kind": "json",
            "exists": True,
            "status": "ready" if digest_valid and valid else "invalid",
            "digest": digest,
            "digest_valid": digest_valid,
            "expected_digest": expected_digest,
            "valid": valid,
            "validation": {
                "action": validation.get("action"),
                "valid": validation.get("valid"),
            },
        },
        payload if isinstance(payload, Mapping) else None,
    )


def _research_review_questions(
    summary_report: Mapping[str, Any] | None,
    record: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    summary_questions = _mapping_or_empty(
        summary_report.get("research_questions") if summary_report is not None else None
    )
    record_verdicts = _mapping_or_empty(
        record.get("research_question_verdicts") if record is not None else None
    )
    questions: dict[str, dict[str, Any]] = {}
    for key in REAL_EXPERIMENT_RESEARCH_QUESTION_KEYS:
        summary_entry = _mapping_or_empty(summary_questions.get(key))
        record_entry = _mapping_or_empty(record_verdicts.get(key))
        measurements = _mapping_sequence(summary_entry.get("measurements"))
        measurement_count = len(measurements)
        if measurement_count == 0:
            record_measurement_count = _summary_int(record_entry, "measurement_count")
            measurement_count = record_measurement_count or 0
        status = _text_or_none(summary_entry.get("status")) or _text_or_none(
            record_entry.get("status")
        )
        verdict = _text_or_none(summary_entry.get("verdict")) or _text_or_none(
            record_entry.get("verdict")
        )
        source_artifact_type = _text_or_none(
            summary_entry.get("source_artifact_type")
        ) or _text_or_none(record_entry.get("source_artifact_type"))
        primary_metric = summary_entry.get("primary_metric")
        if primary_metric is None:
            primary_metric = record_entry.get("primary_metric")
        questions[key] = {
            "available": status == "available" and measurement_count > 0,
            "measurement_count": measurement_count,
            "primary_metric": _json_value(primary_metric),
            "source_artifact_type": source_artifact_type,
            "status": status,
            "verdict": verdict,
        }
    return questions


def _research_review_question_summary(
    research_questions: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    conclusive_count = sum(
        1
        for row in research_questions.values()
        if _text_or_none(row.get("verdict"))
        in REAL_EXPERIMENT_CONCLUSIVE_VERDICTS
    )
    return {
        "available_count": sum(
            1 for row in research_questions.values() if row.get("available") is True
        ),
        "conclusive_count": conclusive_count,
        "inconclusive_count": len(research_questions) - conclusive_count,
        "required_count": len(REAL_EXPERIMENT_RESEARCH_QUESTION_KEYS),
    }


def _research_review_evidence_summary(
    summary_report: Mapping[str, Any] | None,
    record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    summary = _mapping_or_empty(
        summary_report.get("summary") if summary_report is not None else None
    )
    diagnostic_ledger = _mapping_or_empty(
        record.get("diagnostic_ledger") if record is not None else None
    )
    return {
        "error_attribution_diagnostic_count": _summary_int(
            summary,
            "error_attribution_diagnostic_count",
        )
        or _summary_int(diagnostic_ledger, "error_attribution_diagnostic_count")
        or 0,
        "failure_linkage_diagnostic_count": _summary_int(
            summary,
            "failure_linkage_diagnostic_count",
        )
        or _summary_int(diagnostic_ledger, "failure_linkage_pair_count")
        or 0,
        "graph_construction_diagnostic_count": _summary_int(
            summary,
            "graph_construction_diagnostic_count",
        )
        or _summary_int(diagnostic_ledger, "graph_construction_diagnostic_count")
        or 0,
        "qa_diagnostic_slice_count": _summary_int(
            summary,
            "qa_diagnostic_slice_count",
        )
        or _summary_int(diagnostic_ledger, "qa_diagnostic_slice_count")
        or 0,
        "source_artifact_count": _summary_int(summary, "source_artifact_count")
        or _summary_int(record or {}, "source_artifact_count")
        or 0,
        "source_profile_count": _summary_int(summary, "source_profile_count")
        or _summary_int(record or {}, "source_profile_count")
        or 0,
    }


def _research_review_blockers(
    execution_receipt: Mapping[str, Any],
    *,
    receipt_validation: Mapping[str, Any],
    summary_audit: Mapping[str, Any],
    record_audit: Mapping[str, Any],
    research_questions: Mapping[str, Mapping[str, Any]],
    evidence_summary: Mapping[str, Any],
    record: Mapping[str, Any] | None,
    summary_report: Mapping[str, Any] | None,
) -> list[str]:
    blockers: list[str] = []
    if receipt_validation.get("valid") is not True:
        blockers.append("execution_receipt_invalid")
    if execution_receipt.get("ready_to_review") is not True:
        blockers.append("execution_receipt_not_ready")
    if summary_audit.get("status") != "ready":
        blockers.append("experiment_summary_not_ready")
    if record_audit.get("status") != "ready":
        blockers.append("experiment_record_not_ready")
    if record is None or record.get("real_package_status") != "ready":
        blockers.append("real_package_not_ready")
    readiness = _mapping_or_empty(
        summary_report.get("readiness") if summary_report is not None else None
    )
    if readiness.get("status") != "ready":
        blockers.append("research_questions_incomplete")
    for key, row in sorted(research_questions.items()):
        if row.get("available") is not True:
            blockers.append(f"research_question_missing:{key}")
    if _summary_int(evidence_summary, "source_profile_count") == 0:
        blockers.append("source_profiles_missing")
    if _summary_int(evidence_summary, "graph_construction_diagnostic_count") == 0:
        blockers.append("graph_diagnostics_missing")
    if _summary_int(evidence_summary, "failure_linkage_diagnostic_count") == 0:
        blockers.append("failure_linkage_missing")
    return blockers


def _claim_scale_summary(
    benchmark_manifest: Mapping[str, Any] | None,
) -> dict[str, int]:
    if benchmark_manifest is None:
        return {
            "dynamic_qa_count": 0,
            "episode_count": 0,
            "qa_count": 0,
            "scene_count": 0,
        }
    summary = _mapping_or_empty(benchmark_manifest.get("summary"))
    coverage = _mapping_or_empty(benchmark_manifest.get("coverage"))
    dynamic_static = _mapping_or_empty(coverage.get("dynamic_static"))
    return {
        "dynamic_qa_count": _summary_int(dynamic_static, "dynamic") or 0,
        "episode_count": _summary_int(summary, "episode_count") or 0,
        "qa_count": _summary_int(summary, "qa_count") or 0,
        "scene_count": _summary_int(summary, "scene_count") or 0,
    }


def _claim_readiness_checks(
    research_review: Mapping[str, Any],
    *,
    review_validation: Mapping[str, Any],
    benchmark_audit: Mapping[str, Any],
    scale_summary: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rq_gaps = _claim_research_question_gaps(research_review)
    evidence_summary = _mapping_or_empty(research_review.get("evidence_summary"))
    required_count = _summary_int(rq_gaps, "required_count") or len(
        REAL_EXPERIMENT_RESEARCH_QUESTION_KEYS
    )
    return [
        {
            "name": "research_review_ready",
            "passed": (
                review_validation.get("valid") is True
                and research_review.get("ready_for_research_review") is True
            ),
            "expected": True,
            "actual": {
                "ready_for_research_review": research_review.get(
                    "ready_for_research_review"
                ),
                "valid": review_validation.get("valid"),
            },
        },
        {
            "name": "benchmark_manifest_ready",
            "passed": benchmark_audit.get("status") == "ready",
            "expected": "ready",
            "actual": benchmark_audit.get("status"),
        },
        _minimum_count_check(
            "episode_count",
            _summary_int(scale_summary, "episode_count") or 0,
            _int_threshold(
                thresholds,
                "min_episode_count",
                DEFAULT_CLAIM_MIN_EPISODE_COUNT,
            ),
        ),
        _minimum_count_check(
            "scene_count",
            _summary_int(scale_summary, "scene_count") or 0,
            _int_threshold(
                thresholds,
                "min_scene_count",
                DEFAULT_CLAIM_MIN_SCENE_COUNT,
            ),
        ),
        _minimum_count_check(
            "qa_count",
            _summary_int(scale_summary, "qa_count") or 0,
            _int_threshold(thresholds, "min_qa_count", DEFAULT_CLAIM_MIN_QA_COUNT),
        ),
        _minimum_count_check(
            "dynamic_qa_count",
            _summary_int(scale_summary, "dynamic_qa_count") or 0,
            _int_threshold(
                thresholds,
                "min_dynamic_qa_count",
                DEFAULT_CLAIM_MIN_DYNAMIC_QA_COUNT,
            ),
        ),
        {
            "name": "research_question_availability",
            "passed": rq_gaps.get("ready") is True,
            "expected": {
                "available_count": required_count,
                "conclusive_count": required_count,
                "inconclusive_keys": [],
                "missing_keys": [],
            },
            "actual": {
                "available_count": _summary_int(rq_gaps, "available_count"),
                "conclusive_count": _summary_int(rq_gaps, "conclusive_count"),
                "inconclusive_keys": _string_sequence_or_empty(
                    rq_gaps.get("inconclusive_keys")
                ),
                "missing_keys": _string_sequence_or_empty(
                    rq_gaps.get("missing_keys")
                ),
                "verdicts": _json_value(
                    _mapping_or_empty(rq_gaps.get("verdicts"))
                ),
            },
        },
        {
            "name": "source_profile_count",
            "passed": (_summary_int(evidence_summary, "source_profile_count") or 0)
            > 0,
            "expected": ">0",
            "actual": _summary_int(evidence_summary, "source_profile_count") or 0,
        },
        {
            "name": "graph_construction_diagnostic_count",
            "passed": (
                _summary_int(evidence_summary, "graph_construction_diagnostic_count")
                or 0
            )
            > 0,
            "expected": ">0",
            "actual": _summary_int(
                evidence_summary,
                "graph_construction_diagnostic_count",
            )
            or 0,
        },
        {
            "name": "failure_linkage_diagnostic_count",
            "passed": (
                _summary_int(evidence_summary, "failure_linkage_diagnostic_count")
                or 0
            )
            > 0,
            "expected": ">0",
            "actual": _summary_int(
                evidence_summary,
                "failure_linkage_diagnostic_count",
            )
            or 0,
        },
    ]


def _claim_research_question_gaps(
    research_review: Mapping[str, Any],
) -> dict[str, Any]:
    research_questions = _mapping_or_empty(research_review.get("research_questions"))
    missing_keys: list[str] = []
    inconclusive_keys: list[str] = []
    verdicts: dict[str, str | None] = {}
    conclusive_count = 0
    for key in REAL_EXPERIMENT_RESEARCH_QUESTION_KEYS:
        row_value = research_questions.get(key)
        row = _mapping_or_empty(row_value)
        verdict = _text_or_none(row.get("verdict"))
        verdicts[key] = verdict
        if row.get("available") is not True:
            missing_keys.append(key)
            continue
        if verdict in REAL_EXPERIMENT_CONCLUSIVE_VERDICTS:
            conclusive_count += 1
        else:
            inconclusive_keys.append(key)
    required_count = len(REAL_EXPERIMENT_RESEARCH_QUESTION_KEYS)
    return {
        "available_count": required_count - len(missing_keys),
        "conclusive_count": conclusive_count,
        "inconclusive_count": len(inconclusive_keys),
        "inconclusive_keys": inconclusive_keys,
        "missing_keys": missing_keys,
        "ready": not missing_keys and not inconclusive_keys,
        "required_count": required_count,
        "verdicts": verdicts,
    }


def _claim_research_question_verdicts(
    research_review: Mapping[str, Any],
) -> dict[str, str | None]:
    research_questions = _mapping_or_empty(research_review.get("research_questions"))
    return {
        key: _text_or_none(_mapping_or_empty(research_questions.get(key)).get("verdict"))
        for key in REAL_EXPERIMENT_RESEARCH_QUESTION_KEYS
    }


def _claim_conclusion_evidence(
    research_review: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    research_questions = _mapping_or_empty(research_review.get("research_questions"))
    evidence: dict[str, dict[str, Any]] = {}
    for key in REAL_EXPERIMENT_RESEARCH_QUESTION_KEYS:
        row = _mapping_or_empty(research_questions.get(key))
        evidence[key] = {
            "available": row.get("available") is True,
            "measurement_count": _summary_int(row, "measurement_count") or 0,
            "primary_metric": _json_value(
                _mapping_or_empty(row.get("primary_metric"))
            ),
            "source_artifact_type": _text_or_none(row.get("source_artifact_type")),
            "verdict": _text_or_none(row.get("verdict")),
        }
    return evidence


def _claim_effect_matrix(
    claim_conclusion_evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in REAL_EXPERIMENT_RESEARCH_QUESTION_KEYS:
        row = _mapping_or_empty(claim_conclusion_evidence.get(key))
        primary_metric = _mapping_or_empty(row.get("primary_metric"))
        metric_value = primary_metric.get("value")
        if isinstance(metric_value, bool) or not isinstance(
            metric_value,
            (int, float),
        ):
            metric_value = None
        rows.append(
            {
                "available": row.get("available") is True,
                "measurement_count": _summary_int(row, "measurement_count") or 0,
                "metric_name": _text_or_none(primary_metric.get("name")),
                "metric_value": metric_value,
                "research_question": key,
                "source_artifact_type": _text_or_none(
                    row.get("source_artifact_type")
                ),
                "verdict": _text_or_none(row.get("verdict")),
            }
        )
    return rows


def _claim_effect_direction_summary(
    claim_effect_matrix: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    positive_keys: list[str] = []
    zero_keys: list[str] = []
    negative_keys: list[str] = []
    missing_metric_keys: list[str] = []
    verdict_mismatch_keys: list[str] = []
    for row in claim_effect_matrix:
        key = _text_or_none(row.get("research_question"))
        if key is None:
            continue
        metric_value = _claim_effect_metric_value(row.get("metric_value"))
        verdict = _text_or_none(row.get("verdict"))
        if metric_value is None:
            missing_metric_keys.append(key)
        elif metric_value > 0:
            positive_keys.append(key)
        elif metric_value < 0:
            negative_keys.append(key)
        else:
            zero_keys.append(key)
        if not _claim_metric_value_matches_verdict(metric_value, verdict):
            verdict_mismatch_keys.append(key)
    return {
        "consistent": not verdict_mismatch_keys,
        "direction_counts": {
            "missing_metric": len(missing_metric_keys),
            "negative": len(negative_keys),
            "positive": len(positive_keys),
            "verdict_mismatch": len(verdict_mismatch_keys),
            "zero": len(zero_keys),
        },
        "missing_metric_keys": missing_metric_keys,
        "negative_keys": negative_keys,
        "positive_keys": positive_keys,
        "verdict_mismatch_keys": verdict_mismatch_keys,
        "zero_keys": zero_keys,
    }


def _claim_effect_metric_value(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _claim_metric_value_matches_verdict(
    metric_value: int | float | None,
    verdict: str | None,
) -> bool:
    if verdict == "improved":
        return metric_value is not None and metric_value > 0
    if verdict == "regressed":
        return metric_value is not None and metric_value < 0
    if verdict == "unchanged":
        return metric_value is not None and metric_value == 0
    return True


def _claim_hypothesis_assessment(
    claim_conclusion_summary: Mapping[str, Any],
    claim_effect_direction_summary: Mapping[str, Any],
) -> dict[str, Any]:
    ready_to_assess = claim_conclusion_summary.get("ready_to_conclude") is True
    positive_keys = _string_sequence_or_empty(
        claim_effect_direction_summary.get("positive_keys")
    )
    negative_keys = _string_sequence_or_empty(
        claim_effect_direction_summary.get("negative_keys")
    )
    neutral_keys = _string_sequence_or_empty(
        claim_effect_direction_summary.get("zero_keys")
    )
    missing_or_inconclusive_keys = _ordered_unique_strings(
        [
            *_string_sequence_or_empty(claim_conclusion_summary.get("missing_keys")),
            *_string_sequence_or_empty(
                claim_conclusion_summary.get("inconclusive_keys")
            ),
            *_string_sequence_or_empty(
                claim_effect_direction_summary.get("missing_metric_keys")
            ),
        ]
    )
    required_count = _summary_int(
        claim_conclusion_summary,
        "required_count",
    ) or len(REAL_EXPERIMENT_RESEARCH_QUESTION_KEYS)
    supports_full_hypothesis = (
        ready_to_assess
        and len(positive_keys) == required_count
        and not negative_keys
        and not neutral_keys
        and not missing_or_inconclusive_keys
    )
    return {
        "assessment": _claim_hypothesis_assessment_label(
            ready_to_assess=ready_to_assess,
            supports_full_hypothesis=supports_full_hypothesis,
            positive_keys=positive_keys,
            negative_keys=negative_keys,
            neutral_keys=neutral_keys,
        ),
        "hypothesis": "dynamic_scene_graph_improves_all_target_capabilities",
        "missing_or_inconclusive_keys": missing_or_inconclusive_keys,
        "negative_evidence_keys": negative_keys,
        "neutral_evidence_keys": neutral_keys,
        "no_change_observed": len(neutral_keys) > 0,
        "partial_improvement_observed": len(positive_keys) > 0,
        "positive_evidence_keys": positive_keys,
        "ready_to_assess": ready_to_assess,
        "regression_observed": len(negative_keys) > 0,
        "required_count": required_count,
        "supports_full_hypothesis": supports_full_hypothesis,
    }


def _claim_hypothesis_assessment_label(
    *,
    ready_to_assess: bool,
    supports_full_hypothesis: bool,
    positive_keys: Sequence[str],
    negative_keys: Sequence[str],
    neutral_keys: Sequence[str],
) -> str:
    if not ready_to_assess:
        return "pilot_only"
    if supports_full_hypothesis:
        return "supported_all_capabilities"
    if negative_keys:
        return "contradicted_by_regression"
    if positive_keys:
        return "partial_improvement_observed"
    if neutral_keys:
        return "no_improvement_observed"
    return "inconclusive"


def _ordered_unique_strings(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _claim_conclusion_summary(
    research_question_verdicts: Mapping[str, Any],
    *,
    claim_ready: bool,
    research_question_gaps: Mapping[str, Any],
) -> dict[str, Any]:
    missing_keys = _string_sequence_or_empty(research_question_gaps.get("missing_keys"))
    inconclusive_keys = _string_sequence_or_empty(
        research_question_gaps.get("inconclusive_keys")
    )
    improved_keys: list[str] = []
    regressed_keys: list[str] = []
    unchanged_keys: list[str] = []
    for key in REAL_EXPERIMENT_RESEARCH_QUESTION_KEYS:
        if key in missing_keys or key in inconclusive_keys:
            continue
        verdict = _text_or_none(research_question_verdicts.get(key))
        if verdict == "improved":
            improved_keys.append(key)
        elif verdict == "regressed":
            regressed_keys.append(key)
        elif verdict == "unchanged":
            unchanged_keys.append(key)
        elif verdict is None:
            missing_keys.append(key)
        else:
            inconclusive_keys.append(key)
    required_count = len(REAL_EXPERIMENT_RESEARCH_QUESTION_KEYS)
    conclusive_count = len(improved_keys) + len(regressed_keys) + len(unchanged_keys)
    ready_to_conclude = (
        claim_ready
        and not missing_keys
        and not inconclusive_keys
        and conclusive_count == required_count
    )
    return {
        "available_count": required_count - len(missing_keys),
        "claim_conclusion": _claim_conclusion(
            ready_to_conclude=ready_to_conclude,
            improved_count=len(improved_keys),
            regressed_count=len(regressed_keys),
            unchanged_count=len(unchanged_keys),
            required_count=required_count,
        ),
        "conclusive_count": conclusive_count,
        "improved_keys": improved_keys,
        "inconclusive_keys": inconclusive_keys,
        "missing_keys": missing_keys,
        "ready_to_conclude": ready_to_conclude,
        "regressed_keys": regressed_keys,
        "required_count": required_count,
        "unchanged_keys": unchanged_keys,
        "verdict_counts": {
            "improved": len(improved_keys),
            "inconclusive": len(inconclusive_keys),
            "missing": len(missing_keys),
            "regressed": len(regressed_keys),
            "unchanged": len(unchanged_keys),
        },
    }


def _claim_conclusion(
    *,
    ready_to_conclude: bool,
    improved_count: int,
    regressed_count: int,
    unchanged_count: int,
    required_count: int,
) -> str:
    if not ready_to_conclude:
        return "pilot_only"
    if regressed_count > 0:
        return "regression"
    if improved_count == required_count:
        return "all_improved"
    if improved_count > 0:
        return "mixed_improvement"
    if unchanged_count == required_count:
        return "no_change"
    return "inconclusive"


def _minimum_count_check(name: str, actual: int, minimum: int) -> dict[str, Any]:
    return {
        "name": name,
        "passed": actual >= minimum,
        "expected": {f"min_{name}": minimum},
        "actual": actual,
    }


def _claim_readiness_blocker(check: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "actual": _json_value(check.get("actual")),
        "expected": _json_value(check.get("expected")),
        "name": _text_field(check, "name"),
    }


def _claim_gap_summary(
    blockers: Sequence[Mapping[str, Any]],
    *,
    research_question_gaps: Mapping[str, Any],
    scale_summary: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    scale_deficits: dict[str, dict[str, int | str]] = {}
    for blocker in blockers:
        name = _text_field(blocker, "name")
        threshold_field = CLAIM_SCALE_THRESHOLD_FIELDS.get(name)
        if threshold_field is None:
            continue
        actual = _summary_int(scale_summary, name) or 0
        minimum = _int_threshold(thresholds, threshold_field, 0)
        deficit = max(0, minimum - actual)
        if deficit <= 0:
            continue
        scale_deficits[name] = {
            "actual": actual,
            "deficit": deficit,
            "minimum": minimum,
            "threshold_field": threshold_field,
        }
    missing_keys = _string_sequence_or_empty(research_question_gaps.get("missing_keys"))
    inconclusive_keys = _string_sequence_or_empty(
        research_question_gaps.get("inconclusive_keys")
    )
    return {
        "evidence_gap_count": len(blockers) - len(scale_deficits),
        "failed_check_count": len(blockers),
        "research_question_gap_count": len(missing_keys) + len(inconclusive_keys),
        "research_question_gaps": {
            "inconclusive_keys": inconclusive_keys,
            "missing_keys": missing_keys,
            "verdicts": _json_value(
                _mapping_or_empty(research_question_gaps.get("verdicts"))
            ),
        },
        "scale_deficit_count": len(scale_deficits),
        "scale_deficits": scale_deficits,
        "target_thresholds": _claim_target_thresholds(thresholds),
    }


def _claim_target_thresholds(thresholds: Mapping[str, Any]) -> dict[str, int]:
    return {
        "min_dynamic_qa_count": _int_threshold(
            thresholds,
            "min_dynamic_qa_count",
            DEFAULT_CLAIM_MIN_DYNAMIC_QA_COUNT,
        ),
        "min_episode_count": _int_threshold(
            thresholds,
            "min_episode_count",
            DEFAULT_CLAIM_MIN_EPISODE_COUNT,
        ),
        "min_qa_count": _int_threshold(
            thresholds,
            "min_qa_count",
            DEFAULT_CLAIM_MIN_QA_COUNT,
        ),
        "min_scene_count": _int_threshold(
            thresholds,
            "min_scene_count",
            DEFAULT_CLAIM_MIN_SCENE_COUNT,
        ),
    }


def _claim_scope_assessment(
    scale_summary: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    *,
    claim_ready: bool,
) -> dict[str, Any]:
    active_thresholds = _claim_target_thresholds(thresholds)
    default_thresholds = _claim_target_thresholds({})
    below_default_fields = [
        field
        for field, default in default_thresholds.items()
        if active_thresholds[field] < default
    ]
    above_default_fields = [
        field
        for field, default in default_thresholds.items()
        if active_thresholds[field] > default
    ]
    default_scale_deficits = _claim_scale_deficits(
        scale_summary,
        default_thresholds,
    )
    active_scale_ready = not _claim_scale_deficits(
        scale_summary,
        active_thresholds,
    )
    default_scale_ready = not default_scale_deficits
    full_scale_claim_permitted = claim_ready and default_scale_ready
    return {
        "active_scale_ready": active_scale_ready,
        "below_default_threshold_fields": below_default_fields,
        "claim_scope": _claim_scope_label(
            claim_ready=claim_ready,
            default_scale_ready=default_scale_ready,
            below_default_threshold_fields=below_default_fields,
        ),
        "default_scale_deficits": default_scale_deficits,
        "default_scale_ready": default_scale_ready,
        "default_thresholds": default_thresholds,
        "full_scale_claim_permitted": full_scale_claim_permitted,
        "threshold_profile": _claim_threshold_profile(
            below_default_fields=below_default_fields,
            above_default_fields=above_default_fields,
        ),
    }


def _claim_scope_next_actions(
    claim_scope_assessment: Mapping[str, Any],
    *,
    claim_ready: bool,
    scale_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not claim_ready:
        return []
    if claim_scope_assessment.get("full_scale_claim_permitted") is True:
        return []
    default_scale_deficits = _mapping_or_empty(
        claim_scope_assessment.get("default_scale_deficits")
    )
    if not default_scale_deficits:
        return []
    return [
        {
            "action": "expand_to_default_benchmark_scale",
            "claim_scope": _text_or_none(claim_scope_assessment.get("claim_scope")),
            "current_scale": _claim_current_scale(scale_summary),
            "default_scale_deficits": _json_value(default_scale_deficits),
            "default_thresholds": _json_value(
                _mapping_or_empty(claim_scope_assessment.get("default_thresholds"))
            ),
            "order": 1,
            "reason": "Saved claim thresholds are below the default benchmark policy.",
            "threshold_profile": _text_or_none(
                claim_scope_assessment.get("threshold_profile")
            ),
            "track": "real_data",
        }
    ]


def _claim_scope_handoff_plan(
    claim_scope_assessment: Mapping[str, Any],
    *,
    claim_scope_next_actions: Sequence[Mapping[str, Any]],
    run_manifest: Mapping[str, Any],
    run_manifest_path: str,
) -> dict[str, Any]:
    handoff_required = len(claim_scope_next_actions) > 0
    target_thresholds = _claim_target_thresholds(
        _mapping_or_empty(claim_scope_assessment.get("default_thresholds"))
    )
    scale_deficits = _mapping_or_empty(
        claim_scope_assessment.get("default_scale_deficits")
    )
    tracks_to_expand = _claim_action_tracks(claim_scope_next_actions)
    handoff_root = str(
        Path(run_manifest_path).with_name("next-full-scale-claim-handoff")
    )
    episode_collection_plan = _claim_episode_collection_plan(
        run_manifest,
        handoff_root=handoff_root,
        target_episode_count=target_thresholds["min_episode_count"],
    )
    external_artifact_slots = _claim_external_artifact_slots(
        run_manifest,
        handoff_root=handoff_root,
    )
    required_predicted_input_kinds = sorted(
        _string_sequence_or_empty(run_manifest.get("required_predicted_input_kinds"))
    )
    after_write_intake_plan = _claim_after_write_intake_plan(
        run_manifest,
        episode_collection_plan=episode_collection_plan,
        handoff_required=handoff_required,
        handoff_root=handoff_root,
        target_thresholds=target_thresholds,
    )
    next_run_review_plan = _claim_next_run_review_plan(
        handoff_required=handoff_required,
        handoff_root=handoff_root,
        target_thresholds=target_thresholds,
    )
    commands: dict[str, str] = {}
    if handoff_required:
        commands["write_handoff_manifests"] = _claim_write_handoff_command(
            run_manifest,
            episode_paths=_string_sequence_or_empty(
                episode_collection_plan.get("existing_episode_paths")
            )
            + _string_sequence_or_empty(
                episode_collection_plan.get("planned_episode_paths")
            ),
            handoff_root=handoff_root,
            target_thresholds=target_thresholds,
            candidate_prediction_path=_text_field(
                external_artifact_slots,
                "candidate_prediction_path",
            ),
            detector_jsonl_path=_text_field(
                external_artifact_slots,
                "detector_jsonl_path",
            ),
        )
    plan: dict[str, Any] = {
        "after_write_intake_plan": after_write_intake_plan,
        "required": handoff_required,
        "commands": commands,
        "current_handoff_thresholds": _claim_current_handoff_thresholds(
            run_manifest
        ),
        "dataset_name": _text_field(run_manifest, "dataset_name"),
        "episode_collection_plan": episode_collection_plan,
        "external_artifact_slots": external_artifact_slots,
        "handoff_kind": "full_scale_claim_scope",
        "handoff_root": handoff_root,
        "reason": _claim_scope_handoff_reason(handoff_required),
        "required_predicted_input_kinds": required_predicted_input_kinds,
        "scale_deficits": _json_value(scale_deficits),
        "source_claim_scope": _text_or_none(
            claim_scope_assessment.get("claim_scope")
        ),
        "source_run_manifest_digest": real_experiment_run_manifest_digest(
            cast(dict[str, Any], run_manifest)
        ),
        "source_run_manifest_path": run_manifest_path,
        "next_run_review_plan": next_run_review_plan,
        "target_thresholds": target_thresholds,
        "threshold_updates": _claim_threshold_updates(
            run_manifest,
            target_thresholds=target_thresholds,
        ),
        "tracks_to_expand": tracks_to_expand,
    }
    plan["operator_checklist"] = _claim_operator_checklist(
        after_write_intake_plan=after_write_intake_plan,
        handoff_commands=commands,
        handoff_required=handoff_required,
        next_run_review_plan=next_run_review_plan,
    )
    return plan


def _claim_scope_handoff_reason(handoff_required: bool) -> str:
    if handoff_required:
        return (
            "Smoke-threshold-ready claims require default benchmark scale before "
            "full-scale DSG benefit claims."
        )
    return "No separate full-scale claim-scope handoff is required."


def _claim_scope_handoff_plan_matches(
    claim_scope_handoff_plan: Mapping[str, Any],
    *,
    claim_scope_assessment: Mapping[str, Any],
    claim_scope_next_actions: Sequence[Mapping[str, Any]],
    source_run_manifest_digest: str | None,
    source_run_manifest_path: str | None,
    source_dataset_name: str | None,
    source_current_handoff_thresholds: Mapping[str, Any],
    source_offline_control_prediction_paths: Mapping[str, Any],
    source_required_predicted_input_kinds: Sequence[str],
) -> bool:
    required = len(claim_scope_next_actions) > 0
    commands = _mapping_or_empty(claim_scope_handoff_plan.get("commands"))
    after_write_intake_plan = _mapping_or_empty(
        claim_scope_handoff_plan.get("after_write_intake_plan")
    )
    next_run_review_plan = _mapping_or_empty(
        claim_scope_handoff_plan.get("next_run_review_plan")
    )
    operator_checklist = _mapping_or_empty(
        claim_scope_handoff_plan.get("operator_checklist")
    )
    operator_steps = _mapping_sequence(operator_checklist.get("steps"))
    target_thresholds = _claim_target_thresholds(
        _mapping_or_empty(claim_scope_assessment.get("default_thresholds"))
    )
    expected_scale_deficits = _mapping_or_empty(
        claim_scope_assessment.get("default_scale_deficits")
    )
    if (
        claim_scope_handoff_plan.get("required") is not required
        or _text_or_none(claim_scope_handoff_plan.get("handoff_kind"))
        != "full_scale_claim_scope"
        or _text_or_none(claim_scope_handoff_plan.get("reason"))
        != _claim_scope_handoff_reason(required)
        or _text_or_none(claim_scope_handoff_plan.get("source_claim_scope"))
        != _text_or_none(claim_scope_assessment.get("claim_scope"))
        or _text_or_none(claim_scope_handoff_plan.get("dataset_name"))
        != source_dataset_name
        or _json_value(
            _mapping_or_empty(claim_scope_handoff_plan.get("scale_deficits"))
        )
        != _json_value(expected_scale_deficits)
        or _json_value(
            _mapping_or_empty(claim_scope_handoff_plan.get("target_thresholds"))
        )
        != _json_value(target_thresholds)
        or _string_sequence_or_empty(claim_scope_handoff_plan.get("tracks_to_expand"))
        != _claim_action_tracks(claim_scope_next_actions)
        or sorted(
            _string_sequence_or_empty(
                claim_scope_handoff_plan.get("required_predicted_input_kinds")
            )
        )
        != sorted(source_required_predicted_input_kinds)
    ):
        return False
    if not required:
        return (
            not commands
            and after_write_intake_plan.get("required") is False
            and next_run_review_plan.get("required") is False
            and operator_checklist.get("required") is False
            and _summary_int(operator_checklist, "step_count") == 0
            and not operator_steps
        )
    return (
        _text_or_none(claim_scope_handoff_plan.get("handoff_root")) is not None
        and _text_or_none(commands.get("write_handoff_manifests")) is not None
        and _claim_scope_handoff_provenance_matches(
            claim_scope_handoff_plan,
            source_run_manifest_digest=source_run_manifest_digest,
            source_run_manifest_path=source_run_manifest_path,
        )
        and _claim_scope_handoff_episode_plan_matches(
            claim_scope_handoff_plan,
            scale_deficits=expected_scale_deficits,
            target_thresholds=target_thresholds,
        )
        and _claim_scope_handoff_external_artifact_slots_match(
            claim_scope_handoff_plan,
            source_offline_control_prediction_paths=(
                source_offline_control_prediction_paths
            ),
        )
        and _claim_scope_handoff_commands_match_plan(
            claim_scope_handoff_plan,
            target_thresholds=target_thresholds,
        )
        and _claim_scope_handoff_threshold_metadata_matches(
            claim_scope_handoff_plan,
            source_current_handoff_thresholds=source_current_handoff_thresholds,
            target_thresholds=target_thresholds,
        )
        and _claim_scope_handoff_downstream_plans_match(
            claim_scope_handoff_plan,
            target_thresholds=target_thresholds,
        )
        and after_write_intake_plan.get("required") is True
        and next_run_review_plan.get("required") is True
        and operator_checklist.get("required") is True
        and _summary_int(operator_checklist, "step_count") == len(operator_steps)
        and len(operator_steps) > 0
        and _text_field(operator_steps[0], "key") == "write_handoff_manifests"
        and _text_field(operator_steps[-1], "key") == "compare_claim_readiness"
    )


def _claim_scope_handoff_provenance_matches(
    claim_scope_handoff_plan: Mapping[str, Any],
    *,
    source_run_manifest_digest: str | None,
    source_run_manifest_path: str | None,
) -> bool:
    handoff_root = _text_or_none(claim_scope_handoff_plan.get("handoff_root"))
    scoped_source_run_manifest_path = _text_or_none(
        claim_scope_handoff_plan.get("source_run_manifest_path")
    )
    scoped_source_run_manifest_digest = _text_or_none(
        claim_scope_handoff_plan.get("source_run_manifest_digest")
    )
    if (
        handoff_root is None
        or scoped_source_run_manifest_path is None
        or scoped_source_run_manifest_digest is None
        or source_run_manifest_path is None
        or source_run_manifest_digest is None
    ):
        return False
    expected_handoff_root = str(
        Path(scoped_source_run_manifest_path).with_name(
            "next-full-scale-claim-handoff"
        )
    )
    return (
        handoff_root == expected_handoff_root
        and scoped_source_run_manifest_path == source_run_manifest_path
        and scoped_source_run_manifest_digest == source_run_manifest_digest
        and len(scoped_source_run_manifest_digest) == 64
        and all(
            character in "0123456789abcdef"
            for character in scoped_source_run_manifest_digest
        )
    )


def _claim_scope_handoff_threshold_metadata_matches(
    claim_scope_handoff_plan: Mapping[str, Any],
    *,
    source_current_handoff_thresholds: Mapping[str, Any],
    target_thresholds: Mapping[str, int],
) -> bool:
    current_thresholds = _mapping_or_empty(
        claim_scope_handoff_plan.get("current_handoff_thresholds")
    )
    expected_current_fields = (
        "min_episode_count",
        "min_frame_count",
        "min_qa_count",
        "min_scene_count",
    )
    expected_current_thresholds: dict[str, int] = {}
    for field in expected_current_fields:
        value = _summary_int(current_thresholds, field)
        if value is None:
            return False
        expected_current_thresholds[field] = value
    if (
        _json_value(current_thresholds) != _json_value(expected_current_thresholds)
        or _json_value(current_thresholds)
        != _json_value(source_current_handoff_thresholds)
    ):
        return False
    expected_threshold_updates: dict[str, dict[str, int]] = {}
    for field in ("min_episode_count", "min_scene_count", "min_qa_count"):
        current_value = expected_current_thresholds[field]
        target_value = target_thresholds[field]
        if target_value <= current_value:
            continue
        expected_threshold_updates[field] = {
            "current": current_value,
            "increase": target_value - current_value,
            "target": target_value,
        }
    return bool(
        _json_value(
            _mapping_or_empty(claim_scope_handoff_plan.get("threshold_updates"))
        )
        == _json_value(expected_threshold_updates)
    )


def _claim_scope_handoff_downstream_plans_match(
    claim_scope_handoff_plan: Mapping[str, Any],
    *,
    target_thresholds: Mapping[str, int],
) -> bool:
    handoff_root = _text_or_none(claim_scope_handoff_plan.get("handoff_root"))
    dataset_name = _text_or_none(claim_scope_handoff_plan.get("dataset_name"))
    if handoff_root is None or dataset_name is None:
        return False
    root = Path(handoff_root)
    after_write_intake_plan = _mapping_or_empty(
        claim_scope_handoff_plan.get("after_write_intake_plan")
    )
    next_run_review_plan = _mapping_or_empty(
        claim_scope_handoff_plan.get("next_run_review_plan")
    )
    after_write_artifact_paths = _mapping_or_empty(
        after_write_intake_plan.get("artifact_paths")
    )
    next_run_artifact_paths = _mapping_or_empty(
        next_run_review_plan.get("artifact_paths")
    )
    expected_after_write_paths = _claim_scope_after_write_artifact_paths(root)
    expected_next_run_paths = _claim_scope_next_run_artifact_paths(root)
    if (
        after_write_intake_plan.get("required") is not True
        or next_run_review_plan.get("required") is not True
        or _json_value(after_write_artifact_paths)
        != _json_value(expected_after_write_paths)
        or _json_value(next_run_artifact_paths) != _json_value(expected_next_run_paths)
        or _string_sequence_or_empty(after_write_intake_plan.get("track_order"))
        != [
            "real_data",
            "real_controls",
            "predicted_dsg",
            "primary_evidence",
            "launch_audit",
        ]
        or _string_sequence_or_empty(next_run_review_plan.get("phase_order"))
        != [
            "execution_packet",
            "smoke_run",
            "post_run_receipt",
            "research_review",
            "claim_recheck",
        ]
        or _json_value(_mapping_or_empty(next_run_review_plan.get("claim_thresholds")))
        != _json_value(target_thresholds)
    ):
        return False
    episode_collection_plan = _mapping_or_empty(
        claim_scope_handoff_plan.get("episode_collection_plan")
    )
    episode_paths = _string_sequence_or_empty(
        episode_collection_plan.get("existing_episode_paths")
    ) + _string_sequence_or_empty(
        episode_collection_plan.get("planned_episode_paths")
    )
    current_handoff_thresholds = _mapping_or_empty(
        claim_scope_handoff_plan.get("current_handoff_thresholds")
    )
    min_frame_count = _summary_int(
        current_handoff_thresholds,
        "min_frame_count",
    ) or 30
    after_write_commands = _mapping_or_empty(after_write_intake_plan.get("commands"))
    next_run_commands = _mapping_or_empty(next_run_review_plan.get("commands"))
    if not _claim_commands_match_fragments(
        after_write_commands,
        _claim_scope_after_write_command_fragments(
            dataset_name=dataset_name,
            episode_paths=episode_paths,
            min_frame_count=min_frame_count,
            paths=expected_after_write_paths,
            target_thresholds=target_thresholds,
        ),
    ):
        return False
    if not _claim_commands_match_fragments(
        next_run_commands,
        _claim_scope_next_run_command_fragments(
            paths=expected_next_run_paths,
            target_thresholds=target_thresholds,
        ),
    ):
        return False
    operator_checklist = _mapping_or_empty(
        claim_scope_handoff_plan.get("operator_checklist")
    )
    handoff_commands = _mapping_or_empty(claim_scope_handoff_plan.get("commands"))
    expected_operator_checklist = _claim_operator_checklist(
        after_write_intake_plan=after_write_intake_plan,
        handoff_commands=cast(Mapping[str, str], handoff_commands),
        handoff_required=True,
        next_run_review_plan=next_run_review_plan,
    )
    return bool(
        _json_value(operator_checklist)
        == _json_value(expected_operator_checklist)
    )


def _claim_scope_after_write_artifact_paths(root: Path) -> dict[str, str]:
    return {
        "external_artifact_contracts_path": str(
            root / "real-experiment-external-artifact-contracts.json"
        ),
        "external_artifact_launch_report_path": str(
            root / "real-experiment-external-artifact-launch-report.json"
        ),
        "offline_control_import_manifest_path": str(
            root / "offline-control-import-manifest.json"
        ),
        "offline_control_prediction_receipt_bundle_path": str(
            root / "offline-control-prediction-receipt-bundle.json"
        ),
        "offline_control_prediction_request_bundle_path": str(
            root / "offline-control-prediction-request-bundle.json"
        ),
        "predicted_dsg_detector_receipt_bundle_path": str(
            root / "predicted-dsg-detector-receipt-bundle.json"
        ),
        "predicted_dsg_detector_request_bundle_path": str(
            root / "predicted-dsg-detector-request-bundle.json"
        ),
        "predicted_dsg_detector_run_manifest_path": str(
            root / "predicted-dsg-detector-run-manifest.json"
        ),
        "primary_evidence_acceptance_report_path": str(
            root / "real-experiment-primary-evidence-acceptance-report.json"
        ),
        "primary_evidence_request_package_path": str(
            root / "real-experiment-primary-evidence-request-package.json"
        ),
        "primary_evidence_return_checklist_path": str(
            root / "real-experiment-primary-evidence-return-checklist.json"
        ),
        "primary_evidence_return_progress_path": str(
            root / "real-experiment-primary-evidence-return-progress.json"
        ),
        "primary_evidence_status_path": str(
            root / "real-experiment-primary-evidence-status.json"
        ),
        "real_collection_report_path": str(root / "inputs/real-collection-report.json"),
        "real_collection_request_bundle_path": str(
            root / "real-collection-request-bundle.json"
        ),
        "real_experiment_run_manifest_path": str(
            root / "real-experiment-run-manifest.json"
        ),
    }


def _claim_scope_next_run_artifact_paths(root: Path) -> dict[str, str]:
    return {
        "claim_readiness_path": str(root / "real-experiment-claim-readiness.json"),
        "execution_packet_path": str(root / "real-experiment-execution-packet.json"),
        "execution_receipt_path": str(root / "real-experiment-execution-receipt.json"),
        "external_artifact_launch_report_path": str(
            root / "real-experiment-external-artifact-launch-report.json"
        ),
        "primary_evidence_acceptance_report_path": str(
            root / "real-experiment-primary-evidence-acceptance-report.json"
        ),
        "real_experiment_run_ledger_path": str(
            root / "outputs/real-experiment-run-ledger.json"
        ),
        "research_review_path": str(root / "real-experiment-research-review.json"),
        "smoke_run_checklist_path": str(
            root / "real-experiment-smoke-run-checklist.json"
        ),
        "smoke_run_runbook_path": str(root / "real-experiment-smoke-run-runbook.json"),
    }


def _claim_scope_after_write_command_fragments(
    *,
    dataset_name: str,
    episode_paths: Sequence[str],
    min_frame_count: int,
    paths: Mapping[str, str],
    target_thresholds: Mapping[str, int],
) -> dict[str, list[str]]:
    episode_fragments = [f"--episode {path}" for path in episode_paths]
    collection_threshold_fragments = [
        f"--min-episode-count {target_thresholds['min_episode_count']}",
        f"--min-scene-count {target_thresholds['min_scene_count']}",
        f"--min-frame-count {min_frame_count}",
    ]
    return {
        "compare_external_artifact_contracts": [
            f"--compare-external-artifact-contracts {paths['external_artifact_contracts_path']}"
        ],
        "compare_external_artifact_launch_report": [
            f"--compare-external-artifact-launch-report {paths['external_artifact_launch_report_path']}"
        ],
        "compare_primary_evidence_acceptance_report": [
            "--compare-primary-evidence-acceptance-report "
            f"{paths['primary_evidence_acceptance_report_path']}"
        ],
        "compare_primary_evidence_request_package": [
            "--compare-primary-evidence-request-package "
            f"{paths['primary_evidence_request_package_path']}"
        ],
        "compare_primary_evidence_return_checklist": [
            "--compare-primary-evidence-return-checklist "
            f"{paths['primary_evidence_return_checklist_path']}"
        ],
        "compare_primary_evidence_return_progress_report": [
            "--compare-primary-evidence-return-progress-report "
            f"{paths['primary_evidence_return_progress_path']}"
        ],
        "compare_primary_evidence_status": [
            f"--compare-primary-evidence-status {paths['primary_evidence_status_path']}"
        ],
        "external_artifact_launch_report": [
            f"--external-artifact-launch-report {paths['external_artifact_contracts_path']}",
            f"--launch-report-output {paths['external_artifact_launch_report_path']}",
        ],
        "offline_control_prediction_receipt_bundle": [
            f"--prediction-receipt-bundle {paths['offline_control_import_manifest_path']}",
            "--receipt-bundle-output "
            f"{paths['offline_control_prediction_receipt_bundle_path']}",
        ],
        "offline_control_prediction_request_bundle": [
            f"--prediction-request-bundle {paths['offline_control_import_manifest_path']}",
            "--request-bundle-output "
            f"{paths['offline_control_prediction_request_bundle_path']}",
        ],
        "predicted_dsg_detector_receipt_bundle": [
            f"--detector-receipt-bundle {paths['predicted_dsg_detector_run_manifest_path']}",
            "--receipt-bundle-output "
            f"{paths['predicted_dsg_detector_receipt_bundle_path']}",
        ],
        "predicted_dsg_detector_request_bundle": [
            f"--detector-request-bundle {paths['predicted_dsg_detector_run_manifest_path']}",
            "--request-bundle-output "
            f"{paths['predicted_dsg_detector_request_bundle_path']}",
        ],
        "preflight_run_manifest": [
            f"--preflight-run-manifest {paths['real_experiment_run_manifest_path']}"
        ],
        "primary_evidence_acceptance_report": [
            "--primary-evidence-acceptance-report "
            f"{paths['primary_evidence_return_progress_path']}",
            "--primary-evidence-acceptance-output "
            f"{paths['primary_evidence_acceptance_report_path']}",
        ],
        "primary_evidence_request_package": [
            f"--primary-evidence-request-package {paths['external_artifact_launch_report_path']}",
            "--primary-evidence-request-package-output "
            f"{paths['primary_evidence_request_package_path']}",
        ],
        "primary_evidence_return_checklist": [
            "--primary-evidence-return-checklist "
            f"{paths['primary_evidence_request_package_path']}",
            "--primary-evidence-return-checklist-output "
            f"{paths['primary_evidence_return_checklist_path']}",
        ],
        "primary_evidence_return_progress_report": [
            "--primary-evidence-return-progress-report "
            f"{paths['primary_evidence_return_checklist_path']}",
            "--primary-evidence-return-progress-output "
            f"{paths['primary_evidence_return_progress_path']}",
        ],
        "primary_evidence_status": [
            f"--primary-evidence-status {paths['external_artifact_launch_report_path']}",
            f"--primary-evidence-status-output {paths['primary_evidence_status_path']}",
        ],
        "real_collection_report": [
            f"--dataset-name {dataset_name}",
            *episode_fragments,
            f"--report {paths['real_collection_report_path']}",
            *collection_threshold_fragments,
        ],
        "real_collection_request_bundle": [
            f"--request-bundle {paths['real_collection_request_bundle_path']}",
            f"--dataset-name {dataset_name}",
            *episode_fragments,
            f"--report {paths['real_collection_report_path']}",
            *collection_threshold_fragments,
        ],
        "validate_external_artifact_contracts": [
            f"--validate-external-artifact-contracts {paths['external_artifact_contracts_path']}"
        ],
        "validate_external_artifact_launch_report": [
            f"--validate-external-artifact-launch-report {paths['external_artifact_launch_report_path']}"
        ],
        "validate_primary_evidence_acceptance_report": [
            "--validate-primary-evidence-acceptance-report "
            f"{paths['primary_evidence_acceptance_report_path']}"
        ],
        "validate_primary_evidence_request_package": [
            "--validate-primary-evidence-request-package "
            f"{paths['primary_evidence_request_package_path']}"
        ],
        "validate_primary_evidence_return_checklist": [
            "--validate-primary-evidence-return-checklist "
            f"{paths['primary_evidence_return_checklist_path']}"
        ],
        "validate_primary_evidence_return_progress_report": [
            "--validate-primary-evidence-return-progress-report "
            f"{paths['primary_evidence_return_progress_path']}"
        ],
        "validate_primary_evidence_status": [
            f"--validate-primary-evidence-status {paths['primary_evidence_status_path']}"
        ],
        "write_primary_evidence_request_bundles": [
            "--write-primary-evidence-request-bundles "
            f"{paths['primary_evidence_request_package_path']}"
        ],
    }


def _claim_scope_next_run_command_fragments(
    *,
    paths: Mapping[str, str],
    target_thresholds: Mapping[str, int],
) -> dict[str, list[str]]:
    return {
        "claim_readiness": [
            f"--claim-readiness {paths['research_review_path']}",
            f"--claim-readiness-output {paths['claim_readiness_path']}",
            f"--claim-min-episode-count {target_thresholds['min_episode_count']}",
            f"--claim-min-scene-count {target_thresholds['min_scene_count']}",
            f"--claim-min-qa-count {target_thresholds['min_qa_count']}",
            "--claim-min-dynamic-qa-count "
            f"{target_thresholds['min_dynamic_qa_count']}",
        ],
        "compare_claim_readiness": [
            f"--compare-claim-readiness {paths['claim_readiness_path']}"
        ],
        "compare_execution_packet": [
            f"--compare-execution-packet {paths['execution_packet_path']}"
        ],
        "compare_execution_receipt": [
            f"--compare-execution-receipt {paths['execution_receipt_path']}"
        ],
        "compare_research_review": [
            f"--compare-research-review {paths['research_review_path']}"
        ],
        "compare_run_ledger": [
            f"--compare-run-ledger {paths['real_experiment_run_ledger_path']}"
        ],
        "compare_smoke_run_checklist": [
            f"--compare-smoke-run-checklist {paths['smoke_run_checklist_path']}"
        ],
        "compare_smoke_run_runbook": [
            f"--compare-smoke-run-runbook {paths['smoke_run_runbook_path']}"
        ],
        "execution_packet": [
            f"--execution-packet {paths['external_artifact_launch_report_path']}",
            "--execution-packet-primary-evidence-acceptance-report "
            f"{paths['primary_evidence_acceptance_report_path']}",
            f"--execution-packet-output {paths['execution_packet_path']}",
        ],
        "execution_receipt": [
            f"--execution-receipt {paths['execution_packet_path']}",
            f"--execution-receipt-output {paths['execution_receipt_path']}",
        ],
        "research_review": [
            f"--research-review {paths['execution_receipt_path']}",
            f"--research-review-output {paths['research_review_path']}",
        ],
        "smoke_run_checklist": [
            f"--smoke-run-checklist {paths['execution_packet_path']}",
            f"--smoke-run-checklist-output {paths['smoke_run_checklist_path']}",
            "--smoke-run-checklist-receipt-output "
            f"{paths['execution_receipt_path']}",
        ],
        "smoke_run_runbook": [
            f"--smoke-run-runbook {paths['smoke_run_checklist_path']}",
            f"--smoke-run-runbook-output {paths['smoke_run_runbook_path']}",
        ],
        "validate_claim_readiness": [
            f"--validate-claim-readiness {paths['claim_readiness_path']}"
        ],
        "validate_execution_packet": [
            f"--validate-execution-packet {paths['execution_packet_path']}"
        ],
        "validate_execution_receipt": [
            f"--validate-execution-receipt {paths['execution_receipt_path']}"
        ],
        "validate_research_review": [
            f"--validate-research-review {paths['research_review_path']}"
        ],
        "validate_run_ledger": [
            f"--validate-run-ledger {paths['real_experiment_run_ledger_path']}"
        ],
        "validate_smoke_run_checklist": [
            f"--validate-smoke-run-checklist {paths['smoke_run_checklist_path']}"
        ],
        "validate_smoke_run_runbook": [
            f"--validate-smoke-run-runbook {paths['smoke_run_runbook_path']}"
        ],
    }


def _claim_commands_match_fragments(
    commands: Mapping[str, Any],
    fragments_by_key: Mapping[str, Sequence[str]],
) -> bool:
    if set(str(key) for key in commands) != set(fragments_by_key):
        return False
    for key, fragments in fragments_by_key.items():
        command = _text_or_none(commands.get(key))
        if command is None or any(fragment not in command for fragment in fragments):
            return False
    return True


def _claim_scope_handoff_external_artifact_slots_match(
    claim_scope_handoff_plan: Mapping[str, Any],
    *,
    source_offline_control_prediction_paths: Mapping[str, Any],
) -> bool:
    handoff_root = _text_or_none(claim_scope_handoff_plan.get("handoff_root"))
    if handoff_root is None:
        return False
    root = Path(handoff_root)
    external_artifact_slots = _mapping_or_empty(
        claim_scope_handoff_plan.get("external_artifact_slots")
    )
    offline_control_paths = _mapping_or_empty(
        external_artifact_slots.get("offline_control_prediction_paths")
    )
    expected_candidate_path = str(
        root / "inputs/candidate/predicted-graph-tool.jsonl"
    )
    expected_detector_path = str(
        root / "inputs/predicted-dsg/detector-rgbd.jsonl"
    )
    if (
        _text_or_none(external_artifact_slots.get("candidate_prediction_path"))
        != expected_candidate_path
        or _text_or_none(external_artifact_slots.get("detector_jsonl_path"))
        != expected_detector_path
        or _string_sequence_or_empty(external_artifact_slots.get("track_order"))
        != ["real_controls", "predicted_dsg"]
        or not offline_control_paths
    ):
        return False
    control_kinds: list[str] = []
    for control_kind, path in offline_control_paths.items():
        control_kind_text = _text_or_none(control_kind)
        if control_kind_text is None or _text_or_none(path) is None:
            return False
        control_kinds.append(control_kind_text)
    source_control_kinds = sorted(
        str(control_kind) for control_kind in source_offline_control_prediction_paths
    )
    if sorted(control_kinds) != source_control_kinds:
        return False
    expected_offline_control_paths = {
        control_kind: str(root / "inputs/offline-controls" / f"{control_kind}.jsonl")
        for control_kind in sorted(control_kinds)
    }
    return bool(
        _json_value(offline_control_paths)
        == _json_value(expected_offline_control_paths)
    )


def _claim_scope_handoff_episode_plan_matches(
    claim_scope_handoff_plan: Mapping[str, Any],
    *,
    scale_deficits: Mapping[str, Any],
    target_thresholds: Mapping[str, int],
) -> bool:
    handoff_root = _text_or_none(claim_scope_handoff_plan.get("handoff_root"))
    dataset_name = _text_or_none(claim_scope_handoff_plan.get("dataset_name"))
    if handoff_root is None or dataset_name is None:
        return False
    episode_collection_plan = _mapping_or_empty(
        claim_scope_handoff_plan.get("episode_collection_plan")
    )
    existing_paths = _string_sequence_or_empty(
        episode_collection_plan.get("existing_episode_paths")
    )
    planned_paths = _string_sequence_or_empty(
        episode_collection_plan.get("planned_episode_paths")
    )
    current_episode_count = _summary_int(
        episode_collection_plan,
        "current_episode_count",
    )
    episode_deficit = _summary_int(episode_collection_plan, "episode_deficit")
    target_episode_count = _summary_int(
        episode_collection_plan,
        "target_episode_count",
    )
    if (
        current_episode_count != len(existing_paths)
        or target_episode_count != target_thresholds["min_episode_count"]
    ):
        return False
    expected_episode_deficit = max(
        0,
        target_thresholds["min_episode_count"] - current_episode_count,
    )
    episode_scale_deficit = _mapping_or_empty(scale_deficits.get("episode_count"))
    if episode_scale_deficit:
        if (
            _summary_int(episode_scale_deficit, "actual") != current_episode_count
            or _summary_int(episode_scale_deficit, "deficit")
            != expected_episode_deficit
            or _summary_int(episode_scale_deficit, "minimum")
            != target_thresholds["min_episode_count"]
        ):
            return False
    elif expected_episode_deficit != 0:
        return False
    expected_planned_paths = [
        str(_claim_planned_episode_path_from_name(dataset_name, handoff_root, index))
        for index in range(
            len(existing_paths) + 1,
            target_thresholds["min_episode_count"] + 1,
        )
    ]
    return (
        episode_deficit == expected_episode_deficit
        and len(planned_paths) == expected_episode_deficit
        and planned_paths == expected_planned_paths
    )


def _claim_scope_handoff_commands_match_plan(
    claim_scope_handoff_plan: Mapping[str, Any],
    *,
    target_thresholds: Mapping[str, int],
) -> bool:
    commands = _mapping_or_empty(claim_scope_handoff_plan.get("commands"))
    write_command = _text_or_none(commands.get("write_handoff_manifests"))
    handoff_root = _text_or_none(claim_scope_handoff_plan.get("handoff_root"))
    dataset_name = _text_or_none(claim_scope_handoff_plan.get("dataset_name"))
    if write_command is None or handoff_root is None or dataset_name is None:
        return False
    episode_collection_plan = _mapping_or_empty(
        claim_scope_handoff_plan.get("episode_collection_plan")
    )
    episode_paths = _string_sequence_or_empty(
        episode_collection_plan.get("existing_episode_paths")
    ) + _string_sequence_or_empty(
        episode_collection_plan.get("planned_episode_paths")
    )
    external_artifact_slots = _mapping_or_empty(
        claim_scope_handoff_plan.get("external_artifact_slots")
    )
    offline_control_paths = _mapping_or_empty(
        external_artifact_slots.get("offline_control_prediction_paths")
    )
    required_predicted_input_kinds = sorted(
        _string_sequence_or_empty(
            claim_scope_handoff_plan.get("required_predicted_input_kinds")
        )
    )
    if not required_predicted_input_kinds:
        return False
    write_fragments = [
        f"--handoff-root {handoff_root}",
        f"--dataset-name {dataset_name}",
        f"--candidate-prediction {_text_field(external_artifact_slots, 'candidate_prediction_path')}",
        f"--detector-jsonl {_text_field(external_artifact_slots, 'detector_jsonl_path')}",
        f"--min-episode-count {target_thresholds['min_episode_count']}",
        f"--min-scene-count {target_thresholds['min_scene_count']}",
        f"--min-qa-count {target_thresholds['min_qa_count']}",
    ]
    write_fragments += [
        f"--required-control-kind {control_kind}"
        for control_kind in sorted(
            str(control_kind) for control_kind in offline_control_paths
        )
    ]
    write_fragments += [
        f"--required-predicted-input-kind {input_kind}"
        for input_kind in required_predicted_input_kinds
    ]
    write_fragments += [f"--episode {path}" for path in episode_paths]
    if any(fragment not in write_command for fragment in write_fragments):
        return False
    next_run_review_plan = _mapping_or_empty(
        claim_scope_handoff_plan.get("next_run_review_plan")
    )
    next_run_commands = _mapping_or_empty(next_run_review_plan.get("commands"))
    claim_readiness_command = _text_or_none(
        next_run_commands.get("claim_readiness")
    )
    if claim_readiness_command is None:
        return False
    claim_fragments = [
        f"--claim-min-episode-count {target_thresholds['min_episode_count']}",
        f"--claim-min-scene-count {target_thresholds['min_scene_count']}",
        f"--claim-min-qa-count {target_thresholds['min_qa_count']}",
        (
            "--claim-min-dynamic-qa-count "
            f"{target_thresholds['min_dynamic_qa_count']}"
        ),
    ]
    if any(fragment not in claim_readiness_command for fragment in claim_fragments):
        return False
    operator_checklist = _mapping_or_empty(
        claim_scope_handoff_plan.get("operator_checklist")
    )
    operator_steps = _mapping_sequence(operator_checklist.get("steps"))
    return (
        len(operator_steps) > 0
        and _text_or_none(operator_steps[0].get("command")) == write_command
        and _text_or_none(operator_steps[-1].get("command"))
        == _text_or_none(next_run_commands.get("compare_claim_readiness"))
    )


def _claim_current_scale(scale_summary: Mapping[str, Any]) -> dict[str, int]:
    return {
        scale_field: _summary_int(scale_summary, scale_field) or 0
        for scale_field in CLAIM_SCALE_THRESHOLD_FIELDS
    }


def _claim_scale_deficits(
    scale_summary: Mapping[str, Any],
    thresholds: Mapping[str, int],
) -> dict[str, dict[str, int | str]]:
    deficits: dict[str, dict[str, int | str]] = {}
    for scale_field, threshold_field in CLAIM_SCALE_THRESHOLD_FIELDS.items():
        actual = _summary_int(scale_summary, scale_field) or 0
        minimum = thresholds[threshold_field]
        deficit = max(0, minimum - actual)
        if deficit <= 0:
            continue
        deficits[scale_field] = {
            "actual": actual,
            "deficit": deficit,
            "minimum": minimum,
            "threshold_field": threshold_field,
        }
    return deficits


def _claim_scope_label(
    *,
    claim_ready: bool,
    default_scale_ready: bool,
    below_default_threshold_fields: Sequence[str],
) -> str:
    if not claim_ready:
        return "pilot_only"
    if default_scale_ready:
        return "full_scale_claim_ready"
    if below_default_threshold_fields:
        return "smoke_threshold_ready"
    return "custom_threshold_ready"


def _claim_threshold_profile(
    *,
    below_default_fields: Sequence[str],
    above_default_fields: Sequence[str],
) -> str:
    if below_default_fields and above_default_fields:
        return "mixed_custom"
    if below_default_fields:
        return "below_default"
    if above_default_fields:
        return "above_default"
    return "default"


def _claim_next_actions(
    blockers: Sequence[Mapping[str, Any]],
    *,
    claim_gap_summary: Mapping[str, Any],
    scale_summary: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not blockers:
        return []
    blocker_names = [_text_field(blocker, "name") for blocker in blockers]
    actions: list[dict[str, Any]] = []
    research_question_gaps = _mapping_or_empty(
        claim_gap_summary.get("research_question_gaps")
    )
    research_question_targets = _claim_research_question_evidence_targets(
        research_question_gaps
    )
    scale_blockers = [
        name for name in blocker_names if name in CLAIM_SCALE_THRESHOLD_FIELDS
    ]
    if scale_blockers:
        actions.append(
            {
                "action": "expand_real_benchmark_scale",
                "blocker_names": scale_blockers,
                "current_scale": _json_value(scale_summary),
                "order": len(actions) + 1,
                "reason": "Real benchmark scale is below the saved claim policy.",
                "scale_deficits": _json_value(
                    _mapping_or_empty(claim_gap_summary.get("scale_deficits"))
                ),
                "target_thresholds": _claim_target_thresholds(thresholds),
                "track": "real_data",
            }
        )
    for track, action_name, reason, names in _claim_evidence_action_groups(
        blocker_names
    ):
        action: dict[str, Any] = {
            "action": action_name,
            "blocker_names": names,
            "order": len(actions) + 1,
            "reason": reason,
            "target_thresholds": _claim_target_thresholds(thresholds),
            "track": track,
        }
        if (
            "research_question_availability" in names
            and research_question_targets
        ):
            action["evidence_targets"] = research_question_targets
            action["research_question_gaps"] = {
                "inconclusive_keys": _string_sequence_or_empty(
                    research_question_gaps.get("inconclusive_keys")
                ),
                "missing_keys": _string_sequence_or_empty(
                    research_question_gaps.get("missing_keys")
                ),
                "verdicts": _json_value(
                    _mapping_or_empty(research_question_gaps.get("verdicts"))
                ),
            }
            action["tracks_to_expand"] = _claim_evidence_target_tracks(
                research_question_targets
            )
        actions.append(action)
    return actions


def _claim_next_handoff_plan(
    *,
    claim_ready: bool,
    claim_gap_summary: Mapping[str, Any],
    next_actions: Sequence[Mapping[str, Any]],
    run_manifest: Mapping[str, Any],
    run_manifest_path: str,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    target_thresholds = _claim_target_thresholds(thresholds)
    commands: dict[str, str] = {}
    tracks_to_expand = _claim_action_tracks(next_actions)
    handoff_root = str(Path(run_manifest_path).with_name("next-claim-ready-handoff"))
    episode_collection_plan = _claim_episode_collection_plan(
        run_manifest,
        handoff_root=handoff_root,
        target_episode_count=target_thresholds["min_episode_count"],
    )
    external_artifact_slots = _claim_external_artifact_slots(
        run_manifest,
        handoff_root=handoff_root,
    )
    required_predicted_input_kinds = sorted(
        _string_sequence_or_empty(run_manifest.get("required_predicted_input_kinds"))
    )
    after_write_intake_plan = _claim_after_write_intake_plan(
        run_manifest,
        episode_collection_plan=episode_collection_plan,
        handoff_required=not claim_ready,
        handoff_root=handoff_root,
        target_thresholds=target_thresholds,
    )
    next_run_review_plan = _claim_next_run_review_plan(
        handoff_required=not claim_ready,
        handoff_root=handoff_root,
        target_thresholds=target_thresholds,
    )
    plan: dict[str, Any] = {
        "after_write_intake_plan": after_write_intake_plan,
        "required": not claim_ready,
        "commands": commands,
        "current_handoff_thresholds": _claim_current_handoff_thresholds(
            run_manifest
        ),
        "dataset_name": _text_field(run_manifest, "dataset_name"),
        "episode_collection_plan": episode_collection_plan,
        "external_artifact_slots": external_artifact_slots,
        "handoff_root": handoff_root,
        "required_predicted_input_kinds": required_predicted_input_kinds,
        "scale_deficits": _json_value(
            _mapping_or_empty(claim_gap_summary.get("scale_deficits"))
        ),
        "source_run_manifest_digest": real_experiment_run_manifest_digest(
            cast(dict[str, Any], run_manifest)
        ),
        "source_run_manifest_path": run_manifest_path,
        "next_run_review_plan": next_run_review_plan,
        "target_thresholds": target_thresholds,
        "threshold_updates": _claim_threshold_updates(
            run_manifest,
            target_thresholds=target_thresholds,
        ),
        "tracks_to_expand": tracks_to_expand,
    }
    if not claim_ready:
        commands["write_handoff_manifests"] = _claim_write_handoff_command(
            run_manifest,
            episode_paths=_string_sequence_or_empty(
                episode_collection_plan.get("existing_episode_paths")
            )
            + _string_sequence_or_empty(
                episode_collection_plan.get("planned_episode_paths")
            ),
            handoff_root=handoff_root,
            target_thresholds=target_thresholds,
            candidate_prediction_path=_text_field(
                external_artifact_slots,
                "candidate_prediction_path",
            ),
            detector_jsonl_path=_text_field(
                external_artifact_slots,
                "detector_jsonl_path",
            ),
        )
    plan["operator_checklist"] = _claim_operator_checklist(
        after_write_intake_plan=after_write_intake_plan,
        handoff_commands=commands,
        handoff_required=not claim_ready,
        next_run_review_plan=next_run_review_plan,
    )
    return plan


def _claim_action_tracks(next_actions: Sequence[Mapping[str, Any]]) -> list[str]:
    tracks: list[str] = []
    for action in next_actions:
        for expanded_track in _string_sequence_or_empty(
            action.get("tracks_to_expand")
        ):
            if expanded_track not in tracks:
                tracks.append(expanded_track)
        primary_track = _text_or_none(action.get("track"))
        if primary_track is not None and primary_track not in tracks:
            tracks.append(primary_track)
    return tracks


def _claim_research_question_gap_action_matches(
    next_actions: Sequence[Mapping[str, Any]],
    *,
    research_question_gaps: Mapping[str, Any],
) -> bool:
    expected_targets = _claim_research_question_evidence_targets(
        research_question_gaps
    )
    expected_tracks = _claim_evidence_target_tracks(expected_targets)
    expected_gaps = {
        "inconclusive_keys": _string_sequence_or_empty(
            research_question_gaps.get("inconclusive_keys")
        ),
        "missing_keys": _string_sequence_or_empty(
            research_question_gaps.get("missing_keys")
        ),
        "verdicts": _json_value(
            _mapping_or_empty(research_question_gaps.get("verdicts"))
        ),
    }
    return any(
        "research_question_availability"
        in _string_sequence_or_empty(action.get("blocker_names"))
        and _json_value(_mapping_sequence(action.get("evidence_targets")))
        == _json_value(expected_targets)
        and _string_sequence_or_empty(action.get("tracks_to_expand"))
        == expected_tracks
        and _json_value(_mapping_or_empty(action.get("research_question_gaps")))
        == _json_value(expected_gaps)
        for action in next_actions
    )


def _claim_research_question_evidence_targets(
    research_question_gaps: Mapping[str, Any],
) -> list[dict[str, Any]]:
    verdicts = _mapping_or_empty(research_question_gaps.get("verdicts"))
    targets: list[dict[str, Any]] = []
    for gap_type, field_name in (
        ("missing", "missing_keys"),
        ("inconclusive", "inconclusive_keys"),
    ):
        for key in _string_sequence_or_empty(research_question_gaps.get(field_name)):
            targets.append(
                {
                    "gap_type": gap_type,
                    "research_question": key,
                    "source_artifact_type": (
                        _claim_research_question_source_artifact_type(key)
                    ),
                    "tracks_to_expand": _claim_research_question_tracks(key),
                    "verdict": _text_or_none(verdicts.get(key)),
                }
            )
    return targets


def _claim_research_question_source_artifact_type(key: str) -> str:
    return CLAIM_RESEARCH_QUESTION_SOURCE_ARTIFACT_TYPES.get(
        key,
        "qa_eval_delta_report",
    )


def _claim_research_question_tracks(key: str) -> list[str]:
    return list(
        CLAIM_RESEARCH_QUESTION_TRACKS.get(
            key,
            ("review_artifacts",),
        )
    )


def _claim_evidence_target_tracks(
    targets: Sequence[Mapping[str, Any]],
) -> list[str]:
    tracks: list[str] = []
    for target in targets:
        for track in _string_sequence_or_empty(target.get("tracks_to_expand")):
            if track not in tracks:
                tracks.append(track)
    return tracks


def _claim_current_handoff_thresholds(
    run_manifest: Mapping[str, Any],
) -> dict[str, int]:
    return {
        "min_episode_count": _summary_int(run_manifest, "min_episode_count") or 0,
        "min_frame_count": _summary_int(run_manifest, "min_frame_count") or 30,
        "min_qa_count": _summary_int(run_manifest, "min_qa_count") or 0,
        "min_scene_count": _summary_int(run_manifest, "min_scene_count") or 0,
    }


def _claim_threshold_updates(
    run_manifest: Mapping[str, Any],
    *,
    target_thresholds: Mapping[str, int],
) -> dict[str, dict[str, int]]:
    current = _claim_current_handoff_thresholds(run_manifest)
    updates: dict[str, dict[str, int]] = {}
    for field in ("min_episode_count", "min_scene_count", "min_qa_count"):
        current_value = current[field]
        target_value = target_thresholds[field]
        if target_value <= current_value:
            continue
        updates[field] = {
            "current": current_value,
            "increase": target_value - current_value,
            "target": target_value,
        }
    return updates


def _claim_episode_collection_plan(
    run_manifest: Mapping[str, Any],
    *,
    handoff_root: str,
    target_episode_count: int,
) -> dict[str, Any]:
    existing_paths = _string_sequence_or_empty(run_manifest.get("episode_paths"))
    episode_deficit = max(0, target_episode_count - len(existing_paths))
    planned_paths = [
        str(_claim_planned_episode_path(run_manifest, handoff_root, index))
        for index in range(len(existing_paths) + 1, target_episode_count + 1)
    ]
    return {
        "current_episode_count": len(existing_paths),
        "episode_deficit": episode_deficit,
        "existing_episode_paths": existing_paths,
        "planned_episode_paths": planned_paths,
        "target_episode_count": target_episode_count,
    }


def _claim_planned_episode_path(
    run_manifest: Mapping[str, Any],
    handoff_root: str,
    episode_index: int,
) -> Path:
    dataset_name = _text_field(run_manifest, "dataset_name")
    return _claim_planned_episode_path_from_name(
        dataset_name,
        handoff_root,
        episode_index,
    )


def _claim_planned_episode_path_from_name(
    dataset_name: str,
    handoff_root: str,
    episode_index: int,
) -> Path:
    return (
        Path(handoff_root)
        / "inputs"
        / "episodes"
        / f"{dataset_name}-episode-{episode_index:03d}.jsonl"
    )


def _claim_external_artifact_slots(
    run_manifest: Mapping[str, Any],
    *,
    handoff_root: str,
) -> dict[str, Any]:
    control_kinds = sorted(
        _string_sequence_or_empty(run_manifest.get("required_control_kinds"))
    )
    root = Path(handoff_root)
    return {
        "candidate_prediction_path": str(
            root / "inputs/candidate/predicted-graph-tool.jsonl"
        ),
        "detector_jsonl_path": str(root / "inputs/predicted-dsg/detector-rgbd.jsonl"),
        "offline_control_prediction_paths": {
            control_kind: str(root / "inputs/offline-controls" / f"{control_kind}.jsonl")
            for control_kind in control_kinds
        },
        "track_order": ["real_controls", "predicted_dsg"],
    }


def _claim_after_write_intake_plan(
    run_manifest: Mapping[str, Any],
    *,
    episode_collection_plan: Mapping[str, Any],
    handoff_required: bool,
    handoff_root: str,
    target_thresholds: Mapping[str, int],
) -> dict[str, Any]:
    root = Path(handoff_root)
    episode_paths = [
        Path(path)
        for path in (
            _string_sequence_or_empty(
                episode_collection_plan.get("existing_episode_paths")
            )
            + _string_sequence_or_empty(
                episode_collection_plan.get("planned_episode_paths")
            )
        )
    ]
    real_collection_report_path = root / "inputs/real-collection-report.json"
    real_collection_request_bundle_path = root / "real-collection-request-bundle.json"
    offline_manifest_path = root / "offline-control-import-manifest.json"
    offline_request_bundle_path = root / "offline-control-prediction-request-bundle.json"
    offline_receipt_bundle_path = root / "offline-control-prediction-receipt-bundle.json"
    predicted_manifest_path = root / "predicted-dsg-detector-run-manifest.json"
    predicted_request_bundle_path = root / "predicted-dsg-detector-request-bundle.json"
    predicted_receipt_bundle_path = root / "predicted-dsg-detector-receipt-bundle.json"
    run_manifest_path = root / "real-experiment-run-manifest.json"
    external_contracts_path = root / "real-experiment-external-artifact-contracts.json"
    launch_report_path = root / "real-experiment-external-artifact-launch-report.json"
    primary_status_path = root / "real-experiment-primary-evidence-status.json"
    primary_request_package_path = (
        root / "real-experiment-primary-evidence-request-package.json"
    )
    primary_return_checklist_path = (
        root / "real-experiment-primary-evidence-return-checklist.json"
    )
    primary_return_progress_path = (
        root / "real-experiment-primary-evidence-return-progress.json"
    )
    primary_acceptance_report_path = (
        root / "real-experiment-primary-evidence-acceptance-report.json"
    )
    artifact_paths = {
        "external_artifact_contracts_path": str(external_contracts_path),
        "external_artifact_launch_report_path": str(launch_report_path),
        "offline_control_import_manifest_path": str(offline_manifest_path),
        "offline_control_prediction_receipt_bundle_path": str(
            offline_receipt_bundle_path
        ),
        "offline_control_prediction_request_bundle_path": str(
            offline_request_bundle_path
        ),
        "predicted_dsg_detector_receipt_bundle_path": str(
            predicted_receipt_bundle_path
        ),
        "predicted_dsg_detector_request_bundle_path": str(
            predicted_request_bundle_path
        ),
        "predicted_dsg_detector_run_manifest_path": str(predicted_manifest_path),
        "primary_evidence_acceptance_report_path": str(
            primary_acceptance_report_path
        ),
        "primary_evidence_request_package_path": str(primary_request_package_path),
        "primary_evidence_return_checklist_path": str(primary_return_checklist_path),
        "primary_evidence_return_progress_path": str(primary_return_progress_path),
        "primary_evidence_status_path": str(primary_status_path),
        "real_collection_report_path": str(real_collection_report_path),
        "real_collection_request_bundle_path": str(real_collection_request_bundle_path),
        "real_experiment_run_manifest_path": str(run_manifest_path),
    }
    commands: dict[str, str] = {}
    if handoff_required:
        commands = {
            "compare_external_artifact_contracts": (
                "python scripts/run_real_experiment.py "
                f"--compare-external-artifact-contracts {external_contracts_path}"
            ),
            "compare_external_artifact_launch_report": (
                "python scripts/run_real_experiment.py "
                f"--compare-external-artifact-launch-report {launch_report_path}"
            ),
            "compare_primary_evidence_acceptance_report": (
                "python scripts/run_real_experiment.py "
                "--compare-primary-evidence-acceptance-report "
                f"{primary_acceptance_report_path}"
            ),
            "compare_primary_evidence_request_package": (
                "python scripts/run_real_experiment.py "
                "--compare-primary-evidence-request-package "
                f"{primary_request_package_path}"
            ),
            "compare_primary_evidence_return_checklist": (
                "python scripts/run_real_experiment.py "
                "--compare-primary-evidence-return-checklist "
                f"{primary_return_checklist_path}"
            ),
            "compare_primary_evidence_return_progress_report": (
                "python scripts/run_real_experiment.py "
                "--compare-primary-evidence-return-progress-report "
                f"{primary_return_progress_path}"
            ),
            "compare_primary_evidence_status": (
                "python scripts/run_real_experiment.py "
                f"--compare-primary-evidence-status {primary_status_path}"
            ),
            "external_artifact_launch_report": (
                "python scripts/run_real_experiment.py "
                f"--external-artifact-launch-report {external_contracts_path} "
                f"--launch-report-output {launch_report_path}"
            ),
            "offline_control_prediction_receipt_bundle": (
                "python scripts/run_offline_controls.py "
                f"--prediction-receipt-bundle {offline_manifest_path} "
                f"--receipt-bundle-output {offline_receipt_bundle_path}"
            ),
            "offline_control_prediction_request_bundle": (
                "python scripts/run_offline_controls.py "
                f"--prediction-request-bundle {offline_manifest_path} "
                f"--request-bundle-output {offline_request_bundle_path}"
            ),
            "predicted_dsg_detector_receipt_bundle": (
                "python scripts/run_predicted_dsg.py "
                f"--detector-receipt-bundle {predicted_manifest_path} "
                f"--receipt-bundle-output {predicted_receipt_bundle_path}"
            ),
            "predicted_dsg_detector_request_bundle": (
                "python scripts/run_predicted_dsg.py "
                f"--detector-request-bundle {predicted_manifest_path} "
                f"--request-bundle-output {predicted_request_bundle_path}"
            ),
            "preflight_run_manifest": (
                "python scripts/run_real_experiment.py "
                f"--preflight-run-manifest {run_manifest_path}"
            ),
            "primary_evidence_acceptance_report": (
                "python scripts/run_real_experiment.py "
                f"--primary-evidence-acceptance-report {primary_return_progress_path} "
                f"--primary-evidence-acceptance-output {primary_acceptance_report_path}"
            ),
            "primary_evidence_request_package": (
                "python scripts/run_real_experiment.py "
                f"--primary-evidence-request-package {launch_report_path} "
                f"--primary-evidence-request-package-output {primary_request_package_path}"
            ),
            "primary_evidence_return_checklist": (
                "python scripts/run_real_experiment.py "
                f"--primary-evidence-return-checklist {primary_request_package_path} "
                f"--primary-evidence-return-checklist-output {primary_return_checklist_path}"
            ),
            "primary_evidence_return_progress_report": (
                "python scripts/run_real_experiment.py "
                f"--primary-evidence-return-progress-report {primary_return_checklist_path} "
                f"--primary-evidence-return-progress-output {primary_return_progress_path}"
            ),
            "primary_evidence_status": (
                "python scripts/run_real_experiment.py "
                f"--primary-evidence-status {launch_report_path} "
                f"--primary-evidence-status-output {primary_status_path}"
            ),
            "real_collection_report": _real_collection_report_command(
                dataset_name=_text_field(run_manifest, "dataset_name"),
                source_kind=_text_field_or_default(
                    run_manifest,
                    "real_collection_source_kind",
                    "ai2thor",
                ),
                episode_paths=episode_paths,
                report_path=real_collection_report_path,
                min_episode_count=target_thresholds["min_episode_count"],
                min_scene_count=target_thresholds["min_scene_count"],
                min_frame_count=_summary_int(run_manifest, "min_frame_count") or 30,
            ),
            "real_collection_request_bundle": (
                _real_collection_request_bundle_command(
                    bundle_path=real_collection_request_bundle_path,
                    dataset_name=_text_field(run_manifest, "dataset_name"),
                    source_kind=_text_field_or_default(
                        run_manifest,
                        "real_collection_source_kind",
                        "ai2thor",
                    ),
                    episode_paths=episode_paths,
                    report_path=real_collection_report_path,
                    min_episode_count=target_thresholds["min_episode_count"],
                    min_scene_count=target_thresholds["min_scene_count"],
                    min_frame_count=_summary_int(run_manifest, "min_frame_count")
                    or 30,
                )
            ),
            "validate_external_artifact_contracts": (
                "python scripts/run_real_experiment.py "
                f"--validate-external-artifact-contracts {external_contracts_path}"
            ),
            "validate_external_artifact_launch_report": (
                "python scripts/run_real_experiment.py "
                f"--validate-external-artifact-launch-report {launch_report_path}"
            ),
            "validate_primary_evidence_acceptance_report": (
                "python scripts/run_real_experiment.py "
                "--validate-primary-evidence-acceptance-report "
                f"{primary_acceptance_report_path}"
            ),
            "validate_primary_evidence_request_package": (
                "python scripts/run_real_experiment.py "
                "--validate-primary-evidence-request-package "
                f"{primary_request_package_path}"
            ),
            "validate_primary_evidence_return_checklist": (
                "python scripts/run_real_experiment.py "
                "--validate-primary-evidence-return-checklist "
                f"{primary_return_checklist_path}"
            ),
            "validate_primary_evidence_return_progress_report": (
                "python scripts/run_real_experiment.py "
                "--validate-primary-evidence-return-progress-report "
                f"{primary_return_progress_path}"
            ),
            "validate_primary_evidence_status": (
                "python scripts/run_real_experiment.py "
                f"--validate-primary-evidence-status {primary_status_path}"
            ),
            "write_primary_evidence_request_bundles": (
                "python scripts/run_real_experiment.py "
                f"--write-primary-evidence-request-bundles {primary_request_package_path}"
            ),
        }
    return {
        "required": handoff_required,
        "artifact_paths": artifact_paths,
        "commands": commands,
        "track_order": [
            "real_data",
            "real_controls",
            "predicted_dsg",
            "primary_evidence",
            "launch_audit",
        ],
    }


def _claim_next_run_review_plan(
    *,
    handoff_required: bool,
    handoff_root: str,
    target_thresholds: Mapping[str, int],
) -> dict[str, Any]:
    root = Path(handoff_root)
    launch_report_path = root / "real-experiment-external-artifact-launch-report.json"
    primary_acceptance_report_path = (
        root / "real-experiment-primary-evidence-acceptance-report.json"
    )
    execution_packet_path = root / "real-experiment-execution-packet.json"
    smoke_checklist_path = root / "real-experiment-smoke-run-checklist.json"
    smoke_runbook_path = root / "real-experiment-smoke-run-runbook.json"
    execution_receipt_path = root / "real-experiment-execution-receipt.json"
    run_ledger_path = root / "outputs/real-experiment-run-ledger.json"
    research_review_path = root / "real-experiment-research-review.json"
    claim_readiness_path = root / "real-experiment-claim-readiness.json"
    artifact_paths = {
        "claim_readiness_path": str(claim_readiness_path),
        "execution_packet_path": str(execution_packet_path),
        "execution_receipt_path": str(execution_receipt_path),
        "external_artifact_launch_report_path": str(launch_report_path),
        "primary_evidence_acceptance_report_path": str(
            primary_acceptance_report_path
        ),
        "real_experiment_run_ledger_path": str(run_ledger_path),
        "research_review_path": str(research_review_path),
        "smoke_run_checklist_path": str(smoke_checklist_path),
        "smoke_run_runbook_path": str(smoke_runbook_path),
    }
    commands: dict[str, str] = {}
    if handoff_required:
        commands = {
            "claim_readiness": (
                "python scripts/run_real_experiment.py "
                f"--claim-readiness {research_review_path} "
                f"--claim-readiness-output {claim_readiness_path} "
                "--claim-min-episode-count "
                f"{target_thresholds['min_episode_count']} "
                "--claim-min-scene-count "
                f"{target_thresholds['min_scene_count']} "
                f"--claim-min-qa-count {target_thresholds['min_qa_count']} "
                "--claim-min-dynamic-qa-count "
                f"{target_thresholds['min_dynamic_qa_count']}"
            ),
            "compare_claim_readiness": (
                "python scripts/run_real_experiment.py "
                f"--compare-claim-readiness {claim_readiness_path}"
            ),
            "compare_execution_packet": (
                "python scripts/run_real_experiment.py "
                f"--compare-execution-packet {execution_packet_path}"
            ),
            "compare_execution_receipt": (
                "python scripts/run_real_experiment.py "
                f"--compare-execution-receipt {execution_receipt_path}"
            ),
            "compare_research_review": (
                "python scripts/run_real_experiment.py "
                f"--compare-research-review {research_review_path}"
            ),
            "compare_run_ledger": (
                "python scripts/run_real_experiment.py "
                f"--compare-run-ledger {run_ledger_path}"
            ),
            "compare_smoke_run_checklist": (
                "python scripts/run_real_experiment.py "
                f"--compare-smoke-run-checklist {smoke_checklist_path}"
            ),
            "compare_smoke_run_runbook": (
                "python scripts/run_real_experiment.py "
                f"--compare-smoke-run-runbook {smoke_runbook_path}"
            ),
            "execution_packet": (
                "python scripts/run_real_experiment.py "
                f"--execution-packet {launch_report_path} "
                "--execution-packet-primary-evidence-acceptance-report "
                f"{primary_acceptance_report_path} "
                f"--execution-packet-output {execution_packet_path}"
            ),
            "execution_receipt": (
                "python scripts/run_real_experiment.py "
                f"--execution-receipt {execution_packet_path} "
                f"--execution-receipt-output {execution_receipt_path}"
            ),
            "research_review": (
                "python scripts/run_real_experiment.py "
                f"--research-review {execution_receipt_path} "
                f"--research-review-output {research_review_path}"
            ),
            "smoke_run_checklist": (
                "python scripts/run_real_experiment.py "
                f"--smoke-run-checklist {execution_packet_path} "
                f"--smoke-run-checklist-output {smoke_checklist_path} "
                "--smoke-run-checklist-receipt-output "
                f"{execution_receipt_path}"
            ),
            "smoke_run_runbook": (
                "python scripts/run_real_experiment.py "
                f"--smoke-run-runbook {smoke_checklist_path} "
                f"--smoke-run-runbook-output {smoke_runbook_path}"
            ),
            "validate_claim_readiness": (
                "python scripts/run_real_experiment.py "
                f"--validate-claim-readiness {claim_readiness_path}"
            ),
            "validate_execution_packet": (
                "python scripts/run_real_experiment.py "
                f"--validate-execution-packet {execution_packet_path}"
            ),
            "validate_execution_receipt": (
                "python scripts/run_real_experiment.py "
                f"--validate-execution-receipt {execution_receipt_path}"
            ),
            "validate_research_review": (
                "python scripts/run_real_experiment.py "
                f"--validate-research-review {research_review_path}"
            ),
            "validate_run_ledger": (
                "python scripts/run_real_experiment.py "
                f"--validate-run-ledger {run_ledger_path}"
            ),
            "validate_smoke_run_checklist": (
                "python scripts/run_real_experiment.py "
                f"--validate-smoke-run-checklist {smoke_checklist_path}"
            ),
            "validate_smoke_run_runbook": (
                "python scripts/run_real_experiment.py "
                f"--validate-smoke-run-runbook {smoke_runbook_path}"
            ),
        }
    return {
        "required": handoff_required,
        "artifact_paths": artifact_paths,
        "claim_thresholds": _json_value(target_thresholds),
        "commands": commands,
        "phase_order": [
            "execution_packet",
            "smoke_run",
            "post_run_receipt",
            "research_review",
            "claim_recheck",
        ],
    }


def _claim_operator_checklist(
    *,
    after_write_intake_plan: Mapping[str, Any],
    handoff_commands: Mapping[str, str],
    handoff_required: bool,
    next_run_review_plan: Mapping[str, Any],
) -> dict[str, Any]:
    phase_order = [
        "handoff_manifests",
        "launch_audit",
        "primary_evidence_request",
        "external_receipts",
        "primary_evidence_return",
        "primary_evidence_acceptance",
        "execution_packet",
        "smoke_run",
        "post_run_receipt",
        "research_review",
        "claim_recheck",
    ]
    if not handoff_required:
        return {
            "required": False,
            "phase_order": phase_order,
            "step_count": 0,
            "steps": [],
        }

    after_write_commands = _mapping_or_empty(after_write_intake_plan.get("commands"))
    next_run_commands = _mapping_or_empty(next_run_review_plan.get("commands"))
    rows = [
        (
            "handoff_manifests",
            "handoff",
            "write_handoff_manifests",
            _text_or_none(handoff_commands.get("write_handoff_manifests")),
        ),
        (
            "handoff_manifests",
            "launch_audit",
            "validate_external_artifact_contracts",
            _text_or_none(after_write_commands.get("validate_external_artifact_contracts")),
        ),
        (
            "handoff_manifests",
            "launch_audit",
            "compare_external_artifact_contracts",
            _text_or_none(after_write_commands.get("compare_external_artifact_contracts")),
        ),
        (
            "launch_audit",
            "launch_audit",
            "external_artifact_launch_report",
            _text_or_none(after_write_commands.get("external_artifact_launch_report")),
        ),
        (
            "launch_audit",
            "launch_audit",
            "validate_external_artifact_launch_report",
            _text_or_none(
                after_write_commands.get("validate_external_artifact_launch_report")
            ),
        ),
        (
            "launch_audit",
            "launch_audit",
            "compare_external_artifact_launch_report",
            _text_or_none(
                after_write_commands.get("compare_external_artifact_launch_report")
            ),
        ),
        (
            "primary_evidence_request",
            "primary_evidence",
            "primary_evidence_status",
            _text_or_none(after_write_commands.get("primary_evidence_status")),
        ),
        (
            "primary_evidence_request",
            "primary_evidence",
            "validate_primary_evidence_status",
            _text_or_none(
                after_write_commands.get("validate_primary_evidence_status")
            ),
        ),
        (
            "primary_evidence_request",
            "primary_evidence",
            "compare_primary_evidence_status",
            _text_or_none(
                after_write_commands.get("compare_primary_evidence_status")
            ),
        ),
        (
            "primary_evidence_request",
            "primary_evidence",
            "primary_evidence_request_package",
            _text_or_none(
                after_write_commands.get("primary_evidence_request_package")
            ),
        ),
        (
            "primary_evidence_request",
            "primary_evidence",
            "validate_primary_evidence_request_package",
            _text_or_none(
                after_write_commands.get(
                    "validate_primary_evidence_request_package"
                )
            ),
        ),
        (
            "primary_evidence_request",
            "primary_evidence",
            "compare_primary_evidence_request_package",
            _text_or_none(
                after_write_commands.get(
                    "compare_primary_evidence_request_package"
                )
            ),
        ),
        (
            "primary_evidence_request",
            "primary_evidence",
            "write_primary_evidence_request_bundles",
            _text_or_none(
                after_write_commands.get("write_primary_evidence_request_bundles")
            ),
        ),
        (
            "primary_evidence_request",
            "primary_evidence",
            "primary_evidence_return_checklist",
            _text_or_none(
                after_write_commands.get("primary_evidence_return_checklist")
            ),
        ),
        (
            "primary_evidence_request",
            "primary_evidence",
            "validate_primary_evidence_return_checklist",
            _text_or_none(
                after_write_commands.get(
                    "validate_primary_evidence_return_checklist"
                )
            ),
        ),
        (
            "primary_evidence_request",
            "primary_evidence",
            "compare_primary_evidence_return_checklist",
            _text_or_none(
                after_write_commands.get(
                    "compare_primary_evidence_return_checklist"
                )
            ),
        ),
        (
            "external_receipts",
            "real_data",
            "real_collection_report",
            _text_or_none(after_write_commands.get("real_collection_report")),
        ),
        (
            "external_receipts",
            "real_controls",
            "offline_control_prediction_receipt_bundle",
            _text_or_none(
                after_write_commands.get("offline_control_prediction_receipt_bundle")
            ),
        ),
        (
            "external_receipts",
            "predicted_dsg",
            "predicted_dsg_detector_receipt_bundle",
            _text_or_none(
                after_write_commands.get("predicted_dsg_detector_receipt_bundle")
            ),
        ),
        (
            "primary_evidence_return",
            "primary_evidence",
            "primary_evidence_return_progress_report",
            _text_or_none(
                after_write_commands.get("primary_evidence_return_progress_report")
            ),
        ),
        (
            "primary_evidence_return",
            "primary_evidence",
            "validate_primary_evidence_return_progress_report",
            _text_or_none(
                after_write_commands.get(
                    "validate_primary_evidence_return_progress_report"
                )
            ),
        ),
        (
            "primary_evidence_return",
            "primary_evidence",
            "compare_primary_evidence_return_progress_report",
            _text_or_none(
                after_write_commands.get(
                    "compare_primary_evidence_return_progress_report"
                )
            ),
        ),
        (
            "primary_evidence_acceptance",
            "primary_evidence",
            "primary_evidence_acceptance_report",
            _text_or_none(
                after_write_commands.get("primary_evidence_acceptance_report")
            ),
        ),
        (
            "primary_evidence_acceptance",
            "primary_evidence",
            "validate_primary_evidence_acceptance_report",
            _text_or_none(
                after_write_commands.get(
                    "validate_primary_evidence_acceptance_report"
                )
            ),
        ),
        (
            "primary_evidence_acceptance",
            "primary_evidence",
            "compare_primary_evidence_acceptance_report",
            _text_or_none(
                after_write_commands.get(
                    "compare_primary_evidence_acceptance_report"
                )
            ),
        ),
        (
            "execution_packet",
            "run_review",
            "execution_packet",
            _text_or_none(next_run_commands.get("execution_packet")),
        ),
        (
            "execution_packet",
            "run_review",
            "validate_execution_packet",
            _text_or_none(next_run_commands.get("validate_execution_packet")),
        ),
        (
            "execution_packet",
            "run_review",
            "compare_execution_packet",
            _text_or_none(next_run_commands.get("compare_execution_packet")),
        ),
        (
            "smoke_run",
            "run_review",
            "smoke_run_checklist",
            _text_or_none(next_run_commands.get("smoke_run_checklist")),
        ),
        (
            "smoke_run",
            "run_review",
            "validate_smoke_run_checklist",
            _text_or_none(next_run_commands.get("validate_smoke_run_checklist")),
        ),
        (
            "smoke_run",
            "run_review",
            "compare_smoke_run_checklist",
            _text_or_none(next_run_commands.get("compare_smoke_run_checklist")),
        ),
        (
            "smoke_run",
            "run_review",
            "smoke_run_runbook",
            _text_or_none(next_run_commands.get("smoke_run_runbook")),
        ),
        (
            "smoke_run",
            "run_review",
            "validate_smoke_run_runbook",
            _text_or_none(next_run_commands.get("validate_smoke_run_runbook")),
        ),
        (
            "smoke_run",
            "run_review",
            "compare_smoke_run_runbook",
            _text_or_none(next_run_commands.get("compare_smoke_run_runbook")),
        ),
        (
            "post_run_receipt",
            "run_review",
            "validate_run_ledger",
            _text_or_none(next_run_commands.get("validate_run_ledger")),
        ),
        (
            "post_run_receipt",
            "run_review",
            "compare_run_ledger",
            _text_or_none(next_run_commands.get("compare_run_ledger")),
        ),
        (
            "post_run_receipt",
            "run_review",
            "execution_receipt",
            _text_or_none(next_run_commands.get("execution_receipt")),
        ),
        (
            "post_run_receipt",
            "run_review",
            "validate_execution_receipt",
            _text_or_none(next_run_commands.get("validate_execution_receipt")),
        ),
        (
            "post_run_receipt",
            "run_review",
            "compare_execution_receipt",
            _text_or_none(next_run_commands.get("compare_execution_receipt")),
        ),
        (
            "research_review",
            "run_review",
            "research_review",
            _text_or_none(next_run_commands.get("research_review")),
        ),
        (
            "research_review",
            "run_review",
            "validate_research_review",
            _text_or_none(next_run_commands.get("validate_research_review")),
        ),
        (
            "research_review",
            "run_review",
            "compare_research_review",
            _text_or_none(next_run_commands.get("compare_research_review")),
        ),
        (
            "claim_recheck",
            "run_review",
            "claim_readiness",
            _text_or_none(next_run_commands.get("claim_readiness")),
        ),
        (
            "claim_recheck",
            "run_review",
            "validate_claim_readiness",
            _text_or_none(next_run_commands.get("validate_claim_readiness")),
        ),
        (
            "claim_recheck",
            "run_review",
            "compare_claim_readiness",
            _text_or_none(next_run_commands.get("compare_claim_readiness")),
        ),
    ]
    step_rows = [
        (phase, track, key, command)
        for phase, track, key, command in rows
        if command is not None
    ]
    steps = [
        {
            "command": command,
            "key": key,
            "order": order,
            "phase": phase,
            "track": track,
        }
        for order, (phase, track, key, command) in enumerate(step_rows, start=1)
    ]
    return {
        "required": True,
        "phase_order": phase_order,
        "step_count": len(steps),
        "steps": steps,
    }


def _claim_write_handoff_command(
    run_manifest: Mapping[str, Any],
    *,
    candidate_prediction_path: str,
    detector_jsonl_path: str,
    episode_paths: Sequence[str],
    handoff_root: str,
    target_thresholds: Mapping[str, int],
) -> str:
    parts = [
        "python scripts/run_real_experiment.py",
        "--write-handoff-manifests",
        f"--handoff-root {handoff_root}",
        f"--dataset-name {_text_field(run_manifest, 'dataset_name')}",
    ]
    for episode_path in episode_paths:
        parts.append(f"--episode {episode_path}")
    parts.append(f"--candidate-prediction {candidate_prediction_path}")
    parts.append(f"--detector-jsonl {detector_jsonl_path}")
    max_qa_per_episode = _summary_int(run_manifest, "max_qa_per_episode")
    if max_qa_per_episode is not None:
        parts.append(f"--max-qa-per-episode {max_qa_per_episode}")
    for tag in _string_sequence_or_empty(run_manifest.get("tags")):
        parts.append(f"--tag {tag}")
    parts.extend(
        [
            "--data-source-kind "
            f"{_text_field_or_default(run_manifest, 'declared_data_source_kind', 'real')}",
            "--real-collection-source-kind "
            f"{_text_field_or_default(run_manifest, 'real_collection_source_kind', 'ai2thor')}",
            f"--min-episode-count {target_thresholds['min_episode_count']}",
            f"--min-scene-count {target_thresholds['min_scene_count']}",
            "--real-collection-min-frame-count "
            f"{_summary_int(run_manifest, 'min_frame_count') or 30}",
            f"--min-qa-count {target_thresholds['min_qa_count']}",
        ]
    )
    for control_kind in sorted(
        _string_sequence_or_empty(run_manifest.get("required_control_kinds"))
    ):
        parts.append(f"--required-control-kind {control_kind}")
    for input_kind in sorted(
        _string_sequence_or_empty(run_manifest.get("required_predicted_input_kinds"))
    ):
        parts.append(f"--required-predicted-input-kind {input_kind}")
    return " ".join(parts)


def _claim_evidence_action_groups(
    blocker_names: Sequence[str],
) -> list[tuple[str, str, str, list[str]]]:
    action_rows = {
        "benchmark_manifest_ready": (
            "run_outputs",
            "rerun_benchmark_manifest",
            "Benchmark manifest or its validation is not ready for claim review.",
        ),
        "failure_linkage_diagnostic_count": (
            "review_artifacts",
            "rebuild_failure_linkage_review",
            "Failure linkage diagnostics are missing from the research review.",
        ),
        "graph_construction_diagnostic_count": (
            "predicted_dsg",
            "rebuild_real_predicted_dsg_diagnostics",
            "Predicted DSG graph-construction diagnostics are missing.",
        ),
        "research_question_availability": (
            "review_artifacts",
            "rerun_research_review_artifacts",
            "One or more research questions is unavailable or inconclusive.",
        ),
        "research_review_ready": (
            "review_artifacts",
            "repair_research_review_receipt",
            "Saved receipt or research-review validation is not ready.",
        ),
        "source_profile_count": (
            "real_controls",
            "collect_real_offline_control_predictions",
            "Real offline-control source profiles are missing.",
        ),
    }
    grouped: dict[tuple[str, str, str], list[str]] = {}
    for name in blocker_names:
        if name in CLAIM_SCALE_THRESHOLD_FIELDS:
            continue
        row = action_rows.get(
            name,
            (
                "review_artifacts",
                "repair_claim_evidence",
                "Claim-readiness evidence is incomplete.",
            ),
        )
        grouped.setdefault(row, []).append(name)
    return [
        (track, action_name, reason, names)
        for (track, action_name, reason), names in grouped.items()
    ]


def _directory_execution_artifact_receipt(
    *,
    role: str,
    path: str,
) -> dict[str, Any]:
    artifact_path = Path(path)
    exists = artifact_path.exists()
    is_directory = artifact_path.is_dir()
    status = "ready" if is_directory else "missing" if not exists else "invalid"
    return {
        "role": role,
        "path": path,
        "kind": "directory",
        "exists": exists,
        "is_directory": is_directory,
        "status": status,
    }


def _json_execution_artifact_receipt(
    *,
    role: str,
    path: str,
    digest_field: str,
    digest_fn: Any,
    load_fn: Any,
    validate_fn: Any,
) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        return {
            "role": role,
            "path": path,
            "kind": "json",
            "exists": False,
            "status": "missing",
            "digest": None,
            "digest_valid": False,
            "valid": False,
            "validation": None,
        }
    try:
        payload = load_fn(artifact_path)
        digest = _text_or_none(payload.get(digest_field))
        expected_digest = digest_fn(payload)
        digest_valid = digest == expected_digest
        validation = validate_fn(payload)
        valid = validation.get("valid") is True
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        return {
            "role": role,
            "path": path,
            "kind": "json",
            "exists": True,
            "status": "invalid",
            "digest": None,
            "digest_valid": False,
            "valid": False,
            "validation": None,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
    return {
        "role": role,
        "path": path,
        "kind": "json",
        "exists": True,
        "status": "ready" if digest_valid and valid else "invalid",
        "digest": digest,
        "digest_valid": digest_valid,
        "expected_digest": expected_digest,
        "valid": valid,
        "validation": {
            "action": validation.get("action"),
            "valid": validation.get("valid"),
        },
    }


def _track_rows(
    rows: Sequence[Mapping[str, Any]],
    track: str,
) -> list[Mapping[str, Any]]:
    return sorted(
        (row for row in rows if _artifact_track(_text_field(row, "group")) == track),
        key=lambda row: (
            _text_field(row, "group"),
            _text_field(row, "role"),
            str(row.get("path", "")),
            json.dumps(_json_value(row.get("metadata", {})), sort_keys=True),
        ),
    )


def _launch_path_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "group": _text_field(row, "group"),
        "path": _text_field(row, "path"),
        "role": _text_field(row, "role"),
    }
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        result["metadata"] = _json_value(metadata)
    return result


def _launch_invalid_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = _launch_path_row(row)
    error = _text_or_none(row.get("error"))
    if error is not None:
        result["error"] = error
    return result


def _launch_missing_requirement_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "group": _text_field(row, "group"),
        "reason": _text_field(row, "reason"),
        "role": _text_field(row, "role"),
    }
    return result


def _external_artifact_contracts(
    *,
    root: Path,
    run_manifest: Mapping[str, Any],
    offline_manifest: Mapping[str, Any],
    predicted_manifest: Mapping[str, Any],
    artifact_checklist: Mapping[str, Any],
    run_manifest_path: Path,
    preflight_report_path: Path,
    artifact_checklist_path: Path,
) -> dict[str, Any]:
    checklist_summary = _mapping(artifact_checklist.get("summary"), "summary")
    track_summary = _mapping(artifact_checklist.get("track_summary"), "track_summary")
    sources = list(_mapping_sequence(offline_manifest.get("sources")))
    summary = {
        "missing_input_artifact_count": _int_field(
            checklist_summary,
            "missing_input_artifact_count",
        ),
        "planned_output_artifact_count": _int_field(
            checklist_summary,
            "planned_output_artifact_count",
        ),
        "real_control_source_count": len(sources),
        "required_input_artifact_count": _int_field(
            checklist_summary,
            "input_artifact_count",
        ),
        "track_count": len(track_summary),
    }
    contracts: dict[str, Any] = {
        "schema_version": (
            REAL_EXPERIMENT_EXTERNAL_ARTIFACT_CONTRACTS_SCHEMA_VERSION
        ),
        "action": "real_experiment_external_artifact_contracts",
        "root": str(root),
        "run_manifest_path": str(run_manifest_path),
        "preflight_report_path": str(preflight_report_path),
        "artifact_checklist_path": str(artifact_checklist_path),
        "summary": summary,
        "track_summary": _json_value(track_summary),
        "tracks": {
            "real_data": _real_data_contract(run_manifest),
            "real_controls": _real_controls_contract(
                run_manifest,
                offline_manifest,
            ),
            "predicted_dsg": _predicted_dsg_contract(
                run_manifest,
                predicted_manifest,
            ),
            "review_artifacts": _review_artifacts_contract(run_manifest),
            "run_outputs": _run_outputs_contract(run_manifest),
        },
    }
    contracts["contracts_digest"] = (
        real_experiment_external_artifact_contracts_digest(contracts)
    )
    return contracts


def _current_external_artifact_contracts(
    contracts: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(_text_field(contracts, "root"))
    run_manifest_path = Path(_text_field(contracts, "run_manifest_path"))
    preflight_report_path = Path(_text_field(contracts, "preflight_report_path"))
    artifact_checklist_path = Path(
        _text_field(contracts, "artifact_checklist_path")
    )
    run_manifest = _load_json_mapping(run_manifest_path)
    base_dir = run_manifest_path.parent
    offline_manifest_path = _anchored_existing_path(
        _text_field(run_manifest, "offline_control_import_manifest_path"),
        base_dir,
    )
    predicted_manifest_path = _anchored_existing_path(
        _text_field(run_manifest, "predicted_dsg_detector_run_manifest_path"),
        base_dir,
    )
    return _external_artifact_contracts(
        root=root,
        run_manifest=run_manifest,
        offline_manifest=_load_json_mapping(offline_manifest_path),
        predicted_manifest=_load_json_mapping(predicted_manifest_path),
        artifact_checklist=_load_json_mapping(artifact_checklist_path),
        run_manifest_path=run_manifest_path,
        preflight_report_path=preflight_report_path,
        artifact_checklist_path=artifact_checklist_path,
    )


def _current_handoff_operator_checklist(
    checklist: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(_text_field(checklist, "root"))
    run_manifest_path = Path(_text_field(checklist, "run_manifest_path"))
    external_contracts_path = Path(
        _text_field(checklist, "external_artifact_contracts_path")
    )
    run_manifest = load_real_experiment_run_manifest(run_manifest_path)
    return _handoff_operator_checklist(
        root=root,
        run_manifest=run_manifest,
        run_manifest_path=run_manifest_path,
        external_contracts_path=external_contracts_path,
        preflight_report={
            "ready_to_run": checklist.get("ready_to_run") is True,
        },
    )


def _operator_progress_target_paths(
    checklist: Mapping[str, Any],
) -> dict[str, Path]:
    root = Path(_text_field(checklist, "root"))
    run_manifest = load_real_experiment_run_manifest(
        _text_field(checklist, "run_manifest_path")
    )
    real_collection_reports = _string_sequence_or_empty(
        run_manifest.get("real_collection_report_paths")
    )
    real_collection_report_path = (
        Path(real_collection_reports[0])
        if real_collection_reports
        else root / "inputs/real-collection-report.json"
    )
    launch_report_path = root / "real-experiment-external-artifact-launch-report.json"
    primary_status_path = root / "real-experiment-primary-evidence-status.json"
    primary_request_package_path = (
        root / "real-experiment-primary-evidence-request-package.json"
    )
    primary_return_checklist_path = (
        root / "real-experiment-primary-evidence-return-checklist.json"
    )
    primary_return_progress_path = (
        root / "real-experiment-primary-evidence-return-progress.json"
    )
    primary_acceptance_report_path = (
        root / "real-experiment-primary-evidence-acceptance-report.json"
    )
    execution_packet_path = root / "real-experiment-execution-packet.json"
    smoke_checklist_path = root / "real-experiment-smoke-run-checklist.json"
    smoke_runbook_path = root / "real-experiment-smoke-run-runbook.json"
    execution_receipt_path = root / "real-experiment-execution-receipt.json"
    run_ledger_path = root / "outputs/real-experiment-run-ledger.json"
    research_review_path = root / "real-experiment-research-review.json"
    claim_readiness_path = root / "real-experiment-claim-readiness.json"
    paths = {
        "claim_readiness": claim_readiness_path,
        "compare_claim_readiness": claim_readiness_path,
        "compare_execution_packet": execution_packet_path,
        "compare_execution_receipt": execution_receipt_path,
        "compare_external_artifact_contracts": Path(
            _text_field(checklist, "external_artifact_contracts_path")
        ),
        "compare_external_artifact_launch_report": launch_report_path,
        "compare_primary_evidence_acceptance_report": primary_acceptance_report_path,
        "compare_primary_evidence_request_package": primary_request_package_path,
        "compare_primary_evidence_return_checklist": primary_return_checklist_path,
        "compare_primary_evidence_return_progress_report": (
            primary_return_progress_path
        ),
        "compare_primary_evidence_status": primary_status_path,
        "compare_research_review": research_review_path,
        "compare_run_ledger": run_ledger_path,
        "compare_smoke_run_checklist": smoke_checklist_path,
        "execution_packet": execution_packet_path,
        "execution_receipt": execution_receipt_path,
        "external_artifact_launch_report": launch_report_path,
        "offline_control_prediction_receipt_bundle": (
            root / "offline-control-prediction-receipt-bundle.json"
        ),
        "offline_control_prediction_request_bundle": (
            root / "offline-control-prediction-request-bundle.json"
        ),
        "predicted_dsg_detector_receipt_bundle": (
            root / "predicted-dsg-detector-receipt-bundle.json"
        ),
        "predicted_dsg_detector_request_bundle": (
            root / "predicted-dsg-detector-request-bundle.json"
        ),
        "primary_evidence_acceptance_report": primary_acceptance_report_path,
        "primary_evidence_request_package": primary_request_package_path,
        "primary_evidence_return_checklist": primary_return_checklist_path,
        "primary_evidence_return_progress_report": primary_return_progress_path,
        "primary_evidence_status": primary_status_path,
        "real_collection_report": real_collection_report_path,
        "real_collection_request_bundle": root / "real-collection-request-bundle.json",
        "research_review": research_review_path,
        "smoke_run_checklist": smoke_checklist_path,
        "smoke_run_runbook": smoke_runbook_path,
        "validate_claim_readiness": claim_readiness_path,
        "validate_execution_packet": execution_packet_path,
        "validate_execution_receipt": execution_receipt_path,
        "validate_external_artifact_contracts": Path(
            _text_field(checklist, "external_artifact_contracts_path")
        ),
        "validate_external_artifact_launch_report": launch_report_path,
        "validate_primary_evidence_acceptance_report": primary_acceptance_report_path,
        "validate_primary_evidence_request_package": primary_request_package_path,
        "validate_primary_evidence_return_checklist": primary_return_checklist_path,
        "validate_primary_evidence_return_progress_report": (
            primary_return_progress_path
        ),
        "validate_primary_evidence_status": primary_status_path,
        "validate_research_review": research_review_path,
        "validate_run_ledger": run_ledger_path,
        "validate_smoke_run_checklist": smoke_checklist_path,
        "validate_smoke_run_runbook": smoke_runbook_path,
        "write_primary_evidence_request_bundles": (
            root / "real-collection-request-bundle.json"
        ),
        "compare_smoke_run_runbook": smoke_runbook_path,
    }
    step_keys = {_text_field(step, "key") for step in _mapping_sequence(checklist.get("steps"))}
    missing_keys = sorted(step_keys - set(paths))
    if missing_keys:
        raise SpatialQAError(
            "Real experiment operator progress report cannot map step keys: "
            + ", ".join(missing_keys)
        )
    return paths


def _operator_progress_next_missing_step(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    for row in rows:
        if row.get("target_exists") is True:
            continue
        return {
            "key": _text_field(row, "key"),
            "order": _summary_int(row, "order") or 0,
            "phase": _text_field(row, "phase"),
            "target_path": _text_field(row, "target_path"),
            "track": _text_field(row, "track"),
        }
    return None


def _operator_progress_next_not_ready_step(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    for row in rows:
        if row.get("target_ready") is True:
            continue
        return {
            "key": _text_field(row, "key"),
            "order": _summary_int(row, "order") or 0,
            "phase": _text_field(row, "phase"),
            "target_path": _text_field(row, "target_path"),
            "target_status": _text_field(row, "target_status"),
            "track": _text_field(row, "track"),
        }
    return None


def _operator_progress_track_summary(
    rows: Sequence[Mapping[str, Any]],
    tracks: Sequence[str],
) -> dict[str, dict[str, int]]:
    summary = {
        track: {
            "missing_target_step_count": 0,
            "not_ready_target_step_count": 0,
            "present_target_step_count": 0,
            "ready_target_step_count": 0,
            "step_count": 0,
        }
        for track in tracks
    }
    for row in rows:
        track = _text_field(row, "track")
        track_summary = summary[track]
        track_summary["step_count"] += 1
        if row.get("target_exists") is True:
            track_summary["present_target_step_count"] += 1
        else:
            track_summary["missing_target_step_count"] += 1
        if row.get("target_ready") is True:
            track_summary["ready_target_step_count"] += 1
        else:
            track_summary["not_ready_target_step_count"] += 1
    return summary


def _real_data_contract(run_manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset_name": _text_field(run_manifest, "dataset_name"),
        "episode_paths": _string_list(run_manifest.get("episode_paths")),
        "min_episode_count": _int_field(run_manifest, "min_episode_count"),
        "min_frame_count": _int_field_or_default(
            run_manifest,
            "min_frame_count",
            30,
        ),
        "min_qa_count": _int_field(run_manifest, "min_qa_count"),
        "min_scene_count": _int_field(run_manifest, "min_scene_count"),
        "real_collection_report_paths": _string_list(
            run_manifest.get("real_collection_report_paths")
        ),
        "source_kind": _text_field_or_default(
            run_manifest,
            "real_collection_source_kind",
            "ai2thor",
        ),
    }


def _real_controls_contract(
    run_manifest: Mapping[str, Any],
    offline_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "offline_control_import_manifest_path": _text_field(
            run_manifest,
            "offline_control_import_manifest_path",
        ),
        "offline_control_import_run_ledger_path": _text_field(
            run_manifest,
            "offline_control_import_run_ledger_path",
        ),
        "qa_path": _text_field(offline_manifest, "qa_path"),
        "candidate_prediction_path": _text_field(
            offline_manifest,
            "candidate_prediction_path",
        ),
        "required_source_kinds": _string_list(
            offline_manifest.get("required_source_kinds")
        ),
        "planned_outputs": {
            "matrix_report_path": _text_field(
                offline_manifest,
                "matrix_report_path",
            ),
            "output_dir": _text_field(offline_manifest, "output_dir"),
            "qa_eval_output_dir": _text_field(
                offline_manifest,
                "qa_eval_output_dir",
            ),
            "result_report_path": _text_field(
                offline_manifest,
                "result_report_path",
            ),
        },
        "sources": [
            _real_control_source_contract(source)
            for source in _mapping_sequence(offline_manifest.get("sources"))
        ],
    }


def _real_control_source_contract(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "expected_input_format": _text_field(source, "input_format"),
        "input_path": _text_field(source, "input_path"),
        "metadata": _json_value(source.get("metadata", {})),
        "planned_import_report_path": _text_field(source, "import_report_path"),
        "planned_prediction_path": _text_field(source, "prediction_path"),
        "source_kind": _text_field(source, "source_kind"),
        "source_name": _text_field(source, "source_name"),
    }


def _predicted_dsg_contract(
    run_manifest: Mapping[str, Any],
    predicted_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "predicted_dsg_detector_run_manifest_path": _text_field(
            run_manifest,
            "predicted_dsg_detector_run_manifest_path",
        ),
        "predicted_dsg_detector_run_ledger_path": _text_field(
            run_manifest,
            "predicted_dsg_detector_run_ledger_path",
        ),
        "detector_jsonl_path": _text_field(predicted_manifest, "detector_jsonl_path"),
        "infer_relations": _string_list(predicted_manifest.get("infer_relations")),
        "min_object_observation_count": _int_field(
            predicted_manifest,
            "min_object_observation_count",
        ),
        "min_observation_count": _int_field(
            predicted_manifest,
            "min_observation_count",
        ),
        "reference_frames": _string_list(predicted_manifest.get("reference_frames")),
        "required_evidence_kinds": _string_list(
            predicted_manifest.get("required_evidence_kinds")
        ),
        "planned_outputs": {
            "detector_import_report_path": _text_field(
                predicted_manifest,
                "detector_import_report_path",
            ),
            "observation_sequence_path": _text_field(
                predicted_manifest,
                "output_sequence_path",
            ),
            "predicted_dsg_evidence_report_path": _text_field(
                predicted_manifest,
                "predicted_dsg_evidence_report_path",
            ),
            "predicted_graph_path": _text_field(
                predicted_manifest,
                "output_graph_path",
            ),
            "predicted_graph_report_path": _text_field(
                predicted_manifest,
                "predicted_graph_report_path",
            ),
        },
    }


def _review_artifacts_contract(run_manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "active_task_delta_report_paths": _string_list(
            run_manifest.get("active_task_delta_report_paths")
        ),
        "dashboard_bundle_paths": _string_list(
            run_manifest.get("dashboard_bundle_paths")
        ),
        "error_attribution_report_paths": _string_list(
            run_manifest.get("error_attribution_report_paths")
        ),
        "graph_eval_report_paths": _string_list(
            run_manifest.get("graph_eval_report_paths")
        ),
    }


def _run_outputs_contract(run_manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "benchmark_manifest_path": _text_field(run_manifest, "manifest_path"),
        "output_dir": _text_field(run_manifest, "output_dir"),
        "readiness_report_path": _text_field(
            run_manifest,
            "readiness_report_path",
        ),
        "record_path": _text_field(run_manifest, "record_path"),
        "real_experiment_run_ledger_path": _text_or_none(
            run_manifest.get("real_experiment_run_ledger_path")
        ),
        "summary_report_path": _text_field(run_manifest, "summary_report_path"),
    }


def _offline_control_manifest(
    *,
    root: Path,
    dataset_name: str,
    qa_path: str | Path,
    candidate_prediction_path: str | Path,
    required_control_kinds: Sequence[str],
    input_format: str,
) -> dict[str, Any]:
    return {
        "schema_version": OFFLINE_CONTROL_IMPORT_MANIFEST_SCHEMA_VERSION,
        "qa_path": _path_text(qa_path, root),
        "output_dir": "outputs/offline-controls/imports",
        "matrix_report_path": "outputs/offline-controls/offline-control-matrix.json",
        "result_report_path": "outputs/offline-controls/offline-control-result.json",
        "candidate_prediction_path": _path_text(candidate_prediction_path, root),
        "candidate_name": "predicted_graph_tool",
        "qa_eval_output_dir": "outputs/offline-controls/qa-eval",
        "required_source_kinds": sorted(required_control_kinds),
        "sources": [
            _offline_source(dataset_name, source_kind, input_format)
            for source_kind in (
                "vlm",
                "multi_frame_vlm",
                "caption_memory",
                "graph_text",
            )
        ],
    }


def _offline_source(
    dataset_name: str,
    source_kind: str,
    input_format: str,
) -> dict[str, Any]:
    return {
        "source_kind": source_kind,
        "source_name": f"{source_kind}_real_control",
        "input_path": f"inputs/offline-controls/{source_kind}.jsonl",
        "input_format": input_format,
        "import_report_path": (
            f"outputs/offline-controls/imports/{source_kind}/import-report.json"
        ),
        "metadata": _source_metadata(dataset_name, source_kind),
        "prediction_path": (
            f"outputs/offline-controls/imports/{source_kind}/predictions.jsonl"
        ),
    }


def _source_metadata(dataset_name: str, source_kind: str) -> dict[str, Any]:
    capabilities = {
        "caption_memory": ["spatial_qa", "dynamic_memory"],
        "graph_text": ["spatial_qa", "dynamic_memory", "graph_tool_query"],
        "multi_frame_vlm": ["spatial_qa", "dynamic_memory"],
        "vlm": ["spatial_qa"],
    }
    return {
        "capabilities": capabilities[source_kind],
        "dataset_id": dataset_name,
        "model_id": f"external-{source_kind}",
        "prompt_id": f"{source_kind}-spatial-qa",
    }


def _predicted_dsg_manifest(
    *,
    root: Path,
    detector_jsonl_path: str | Path,
) -> dict[str, Any]:
    return {
        "schema_version": PREDICTED_DSG_DETECTOR_RUN_MANIFEST_SCHEMA_VERSION,
        "detector_jsonl_path": _path_text(detector_jsonl_path, root),
        "output_sequence_path": "outputs/predicted-dsg/detector-observations.json",
        "output_graph_path": "outputs/predicted-dsg/predicted-graph.json",
        "predicted_graph_report_path": (
            "outputs/predicted-dsg/predicted-graph-report.json"
        ),
        "detector_import_report_path": (
            "outputs/predicted-dsg/detector-import-report.json"
        ),
        "predicted_dsg_evidence_report_path": (
            "outputs/predicted-dsg/predicted-dsg-evidence.json"
        ),
        "infer_relations": ["LEFT_OF", "RIGHT_OF", "NEAR"],
        "reference_frames": ["world"],
        "min_observation_count": 2,
        "min_object_observation_count": 2,
        "required_evidence_kinds": ["depth", "detector", "rgb"],
    }


def _run_manifest(
    *,
    root: Path,
    dataset_name: str,
    episode_paths: Sequence[Path],
    offline_manifest_path: Path,
    offline_control_run_ledger_path: Path,
    predicted_manifest_path: Path,
    predicted_dsg_run_ledger_path: Path,
    real_experiment_run_ledger_path: Path,
    real_collection_report_path: str | Path,
    active_task_delta_report_path: str | Path,
    dashboard_bundle_path: str | Path,
    error_attribution_report_path: str | Path,
    graph_eval_report_path: str | Path,
    max_qa_per_episode: int | None,
    tags: Sequence[str],
    declared_data_source_kind: str,
    real_collection_source_kind: str,
    min_episode_count: int,
    min_scene_count: int,
    min_frame_count: int,
    min_qa_count: int,
    required_control_kinds: Sequence[str],
    required_predicted_input_kinds: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": REAL_EXPERIMENT_RUN_MANIFEST_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "episode_paths": [_path_text(path, root) for path in episode_paths],
        "output_dir": "outputs/benchmark",
        "manifest_path": "outputs/benchmark-manifest.json",
        "readiness_report_path": "outputs/real-experiment-readiness.json",
        "summary_report_path": "outputs/experiment-summary.json",
        "record_path": "outputs/experiment-record.json",
        "real_experiment_run_ledger_path": _path_text(
            real_experiment_run_ledger_path,
            root,
        ),
        "max_qa_per_episode": max_qa_per_episode,
        "tags": list(tags),
        "declared_data_source_kind": declared_data_source_kind,
        "real_collection_source_kind": real_collection_source_kind,
        "min_episode_count": min_episode_count,
        "min_scene_count": min_scene_count,
        "min_frame_count": min_frame_count,
        "min_qa_count": min_qa_count,
        "required_control_kinds": sorted(required_control_kinds),
        "required_predicted_input_kinds": sorted(required_predicted_input_kinds),
        "active_task_delta_report_paths": [
            _path_text(active_task_delta_report_path, root)
        ],
        "dashboard_bundle_paths": [_path_text(dashboard_bundle_path, root)],
        "error_attribution_report_paths": [
            _path_text(error_attribution_report_path, root)
        ],
        "graph_eval_report_paths": [_path_text(graph_eval_report_path, root)],
        "offline_control_import_manifest_path": _path_text(
            offline_manifest_path,
            root,
        ),
        "offline_control_import_run_ledger_path": _path_text(
            offline_control_run_ledger_path,
            root,
        ),
        "predicted_dsg_detector_run_manifest_path": _path_text(
            predicted_manifest_path,
            root,
        ),
        "predicted_dsg_detector_run_ledger_path": _path_text(
            predicted_dsg_run_ledger_path,
            root,
        ),
        "real_collection_report_paths": [
            _path_text(real_collection_report_path, root)
        ],
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _path_text(path: str | Path, root: Path) -> str:
    local_path = Path(path)
    if not local_path.is_absolute():
        return local_path.as_posix()
    try:
        return local_path.relative_to(root).as_posix()
    except ValueError:
        return str(local_path)


def _non_empty_path_sequence(
    value: Sequence[str | Path],
    field: str,
) -> tuple[Path, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise SpatialQAError(f"Real handoff field must be a path sequence: {field}")
    paths = tuple(Path(path) for path in value)
    if not paths:
        raise SpatialQAError(f"Real handoff field requires at least one path: {field}")
    return paths


def _non_empty_string_sequence(value: Sequence[str], field: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise SpatialQAError(f"Real handoff field must be a string sequence: {field}")
    items = tuple(value)
    if not items:
        raise SpatialQAError(f"Real handoff field requires at least one value: {field}")
    for item in items:
        _validate_non_empty_str(item, field)
    return items


def _validate_non_empty_str(value: object, field: str) -> None:
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Real handoff field must be a non-empty string: {field}")


def _validate_real_collection_source_kind(value: str) -> None:
    _validate_non_empty_str(value, "real_collection_source_kind")
    if value not in SUPPORTED_REAL_COLLECTION_SOURCE_KINDS:
        raise SpatialQAError(
            "Real handoff field must be a supported real collection source kind: "
            "real_collection_source_kind"
        )


def _validate_positive_int(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SpatialQAError(
            f"Real handoff field must be a positive integer: {field}"
        )


def _mapping_sequence(value: object) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpatialQAError(f"Real handoff field must be an object: {field}")
    return value


def _mapping_or_empty(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _text_field(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Real handoff row field must be a non-empty string: {field}")
    return value


def _text_or_none(value: object) -> str | None:
    if isinstance(value, str) and value != "":
        return value
    return None


def _text_field_or_default(
    row: Mapping[str, Any],
    field: str,
    default: str,
) -> str:
    value = row.get(field)
    if value is None:
        return default
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Real handoff row field must be a non-empty string: {field}")
    return value


def _int_field(row: Mapping[str, Any], field: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpatialQAError(f"Real handoff row field must be an integer: {field}")
    return value


def _int_field_or_default(
    row: Mapping[str, Any],
    field: str,
    default: int,
) -> int:
    value = row.get(field)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpatialQAError(f"Real handoff row field must be an integer: {field}")
    return value


def _int_threshold(
    row: Mapping[str, Any],
    field: str,
    default: int,
) -> int:
    value = _summary_int(row, field)
    return default if value is None else value


def _validate_non_negative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SpatialQAError(f"{name} must be a non-negative integer")


def _string_list(value: object) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise SpatialQAError("Real handoff field must be a string sequence")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise SpatialQAError("Real handoff field sequence must contain strings")
        result.append(item)
    return result


def _string_sequence_or_empty(value: object) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return []
    return [item for item in value if isinstance(item, str)]


def _summary_int(summary: object, field: str) -> int | None:
    if not isinstance(summary, Mapping):
        return None
    value = summary.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _track_summary_count(
    track_summary: Mapping[str, Any],
    field: str,
) -> int:
    total = 0
    for row in track_summary.values():
        if not isinstance(row, Mapping):
            raise SpatialQAError("Real handoff track summary row must be an object")
        total += _int_field(row, field)
    return total


def _load_json_mapping(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Real handoff JSON artifact must be an object")
    return cast(dict[str, Any], payload)


def _anchored_path(path: str, base_dir: Path) -> Path:
    local_path = Path(path)
    if local_path.is_absolute():
        return local_path
    return base_dir / local_path


def _anchored_existing_path(path: str, base_dir: Path) -> Path:
    local_path = Path(path)
    if local_path.is_absolute() or local_path.exists():
        return local_path
    return base_dir / local_path


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_json_value(item) for item in value]
    return value
