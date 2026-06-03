from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab.benchmark.experiment_record import (
    experiment_record,
    save_experiment_record,
)
from dsg_spatialqa_lab.benchmark.experiment_summary import (
    experiment_summary_report,
    save_experiment_summary_report,
)
from dsg_spatialqa_lab.benchmark.manifest import load_benchmark_manifest
from dsg_spatialqa_lab.benchmark.readiness import (
    load_real_experiment_readiness_report,
)
from dsg_spatialqa_lab.benchmark.real_package import assemble_real_experiment_package
from dsg_spatialqa_lab.eval.offline_control_run import (
    load_offline_control_import_manifest,
    offline_control_import_run_ledger,
    run_offline_control_import_manifest,
    save_offline_control_import_run_ledger,
)
from dsg_spatialqa_lab.predicted_run import (
    load_predicted_dsg_detector_run_manifest,
    predicted_dsg_detector_run_ledger,
    run_predicted_dsg_detector_run_manifest,
    save_predicted_dsg_detector_run_ledger,
)
from dsg_spatialqa_lab.schema import SpatialQAError
from dsg_spatialqa_lab.visualization.dashboard_export import load_dashboard_bundle


REAL_EXPERIMENT_RUN_SCHEMA_VERSION = "dsg-spatialqa-lab.real-experiment-run.v1"
REAL_EXPERIMENT_RUN_MANIFEST_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.real-experiment-run-manifest.v1"
)
REAL_EXPERIMENT_RUN_LEDGER_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.real-experiment-run-ledger.v1"
)
REAL_EXPERIMENT_PREFLIGHT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.real-experiment-preflight.v1"
)


def load_real_experiment_run_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SpatialQAError("Real experiment run manifest JSON must be an object")
    return _real_experiment_run_manifest(payload, manifest_path.parent)


def real_experiment_run_manifest_digest(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_value(manifest),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def real_experiment_run_ledger_digest(ledger: Mapping[str, Any]) -> str:
    payload = {
        key: value for key, value in ledger.items() if key != "ledger_digest"
    }
    return hashlib.sha256(
        json.dumps(
            _json_value(payload),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def real_experiment_run_ledger(run_result: Mapping[str, Any]) -> dict[str, Any]:
    run_result_payload = _json_value(run_result)
    if not isinstance(run_result_payload, dict):
        raise SpatialQAError("Real experiment run result must be a JSON object")
    execution_approval = run_result_payload.get("execution_approval")
    if not isinstance(execution_approval, dict):
        execution_approval = None
    ledger: dict[str, Any] = {
        "schema_version": REAL_EXPERIMENT_RUN_LEDGER_SCHEMA_VERSION,
        "action": "real_experiment_run_ledger",
        "run_schema_version": run_result_payload.get("schema_version"),
        "run_action": run_result_payload.get("action"),
        "dataset_name": run_result_payload.get("dataset_name"),
        "run_manifest_path": run_result_payload.get("run_manifest_path"),
        "run_manifest_digest": run_result_payload.get("run_manifest_digest"),
        "ready": run_result_payload.get("ready") is True,
        "execution_approval": execution_approval,
        "run_result": run_result_payload,
        "summary": {
            "approved_execution_packet_path": (
                execution_approval.get("approved_execution_packet_path")
                if execution_approval is not None
                else None
            ),
            "execution_approval_ready": (
                execution_approval.get("ready") is True
                if execution_approval is not None
                else False
            ),
            "execution_approval_required": (
                execution_approval.get("required") is True
                if execution_approval is not None
                else False
            ),
            "ready": run_result_payload.get("ready") is True,
        },
    }
    ledger["ledger_digest"] = real_experiment_run_ledger_digest(ledger)
    return ledger


def real_experiment_run_ledger_json(ledger: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(ledger), indent=2, sort_keys=True) + "\n"


def save_real_experiment_run_ledger(
    ledger: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        real_experiment_run_ledger_json(ledger),
        encoding="utf-8",
    )
    return output_path


def load_real_experiment_run_ledger(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SpatialQAError("Real experiment run ledger JSON must be an object")
    schema_version = payload.get("schema_version")
    if schema_version != REAL_EXPERIMENT_RUN_LEDGER_SCHEMA_VERSION:
        raise SpatialQAError(
            "Unsupported real experiment run ledger schema version: "
            f"{schema_version}"
        )
    return payload


def validate_real_experiment_run_ledger(
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = ledger.get("schema_version")
    action = ledger.get("action")
    ledger_digest = ledger.get("ledger_digest")
    expected_digest = real_experiment_run_ledger_digest(ledger)
    run_result = ledger.get("run_result")
    execution_approval = ledger.get("execution_approval")
    run_result_mapping = run_result if isinstance(run_result, Mapping) else {}
    approval_mapping = (
        execution_approval if isinstance(execution_approval, Mapping) else {}
    )
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == REAL_EXPERIMENT_RUN_LEDGER_SCHEMA_VERSION,
            "expected": REAL_EXPERIMENT_RUN_LEDGER_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "action",
            "passed": action == "real_experiment_run_ledger",
            "expected": "real_experiment_run_ledger",
            "actual": action,
        },
        {
            "name": "ledger_digest",
            "passed": ledger_digest == expected_digest,
            "expected": expected_digest,
            "actual": ledger_digest,
        },
        {
            "name": "run_result_object",
            "passed": isinstance(run_result, Mapping),
            "expected": "object",
            "actual": type(run_result).__name__,
        },
        {
            "name": "run_result_schema_version",
            "passed": (
                run_result_mapping.get("schema_version")
                == REAL_EXPERIMENT_RUN_SCHEMA_VERSION
            ),
            "expected": REAL_EXPERIMENT_RUN_SCHEMA_VERSION,
            "actual": run_result_mapping.get("schema_version"),
        },
        {
            "name": "run_result_action",
            "passed": run_result_mapping.get("action")
            == "run_real_experiment_manifest",
            "expected": "run_real_experiment_manifest",
            "actual": run_result_mapping.get("action"),
        },
        {
            "name": "ready_matches_run_result",
            "passed": ledger.get("ready") is (run_result_mapping.get("ready") is True),
            "expected": run_result_mapping.get("ready") is True,
            "actual": ledger.get("ready"),
        },
        {
            "name": "run_manifest_path_matches",
            "passed": ledger.get("run_manifest_path")
            == run_result_mapping.get("run_manifest_path"),
            "expected": run_result_mapping.get("run_manifest_path"),
            "actual": ledger.get("run_manifest_path"),
        },
        {
            "name": "run_manifest_digest_matches",
            "passed": ledger.get("run_manifest_digest")
            == run_result_mapping.get("run_manifest_digest"),
            "expected": run_result_mapping.get("run_manifest_digest"),
            "actual": ledger.get("run_manifest_digest"),
        },
        {
            "name": "execution_approval_matches",
            "passed": _json_value(approval_mapping)
            == _json_value(run_result_mapping.get("execution_approval")),
            "expected": _json_value(run_result_mapping.get("execution_approval")),
            "actual": _json_value(approval_mapping),
        },
    ]
    return {
        "action": "validate_real_experiment_run_ledger",
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "ledger_digest": ledger_digest,
        "checks": checks,
    }


def compare_real_experiment_run_ledger(
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_real_experiment_run_ledger(ledger)
    run_manifest_path = ledger.get("run_manifest_path")
    saved_digest = ledger.get("ledger_digest")
    manifest_digest_matches = False
    current_run_manifest_digest = None
    if isinstance(run_manifest_path, str):
        try:
            current_manifest = load_real_experiment_run_manifest(run_manifest_path)
            current_run_manifest_digest = real_experiment_run_manifest_digest(
                current_manifest
            )
            manifest_digest_matches = (
                current_run_manifest_digest == ledger.get("run_manifest_digest")
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError):
            manifest_digest_matches = False
    checks = [
        {
            "name": "run_ledger_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        {
            "name": "run_manifest_digest_matches_current",
            "passed": manifest_digest_matches,
            "expected": ledger.get("run_manifest_digest"),
            "actual": current_run_manifest_digest,
        },
    ]
    return {
        "action": "compare_real_experiment_run_ledger",
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": saved_digest,
        "validation": validation,
        "checks": checks,
    }


def real_experiment_run_manifest_preflight(manifest_path: str | Path) -> dict[str, Any]:
    manifest = load_real_experiment_run_manifest(manifest_path)
    required_inputs: list[dict[str, Any]] = []
    planned_outputs: list[dict[str, Any]] = []
    invalid_inputs: list[dict[str, Any]] = []
    missing_requirements: list[dict[str, str]] = []

    _add_required_inputs(
        required_inputs,
        group="real_collection",
        role="episode",
        paths=manifest["episode_paths"],
    )
    _add_required_inputs(
        required_inputs,
        group="real_collection",
        role="real_collection_report",
        paths=manifest["real_collection_report_paths"],
    )
    if not manifest["real_collection_report_paths"]:
        _add_missing_requirement(
            missing_requirements,
            group="real_collection",
            role="real_collection_report",
            reason="at least one real collection report path is required",
        )

    offline_generates_deltas = _preflight_offline_controls(
        manifest,
        required_inputs=required_inputs,
        planned_outputs=planned_outputs,
        invalid_inputs=invalid_inputs,
        missing_requirements=missing_requirements,
    )
    _preflight_predicted_dsg(
        manifest,
        required_inputs=required_inputs,
        planned_outputs=planned_outputs,
        invalid_inputs=invalid_inputs,
        missing_requirements=missing_requirements,
    )
    _preflight_review_artifacts(
        manifest,
        required_inputs=required_inputs,
        missing_requirements=missing_requirements,
        offline_generates_deltas=offline_generates_deltas,
    )
    _preflight_run_outputs(manifest, planned_outputs)
    run_ledger_path = manifest["real_experiment_run_ledger_path"]
    if run_ledger_path is not None:
        _add_planned_output(
            planned_outputs,
            group="run_outputs",
            role="real_experiment_run_ledger",
            path=run_ledger_path,
        )

    required_inputs = _sorted_path_rows(required_inputs)
    planned_outputs = _sorted_path_rows(planned_outputs)
    invalid_inputs = _sorted_path_rows(invalid_inputs)
    missing_inputs = [item for item in required_inputs if item["exists"] is False]
    missing_requirements = sorted(
        missing_requirements,
        key=lambda item: (item["group"], item["role"], item["reason"]),
    )
    groups = _preflight_groups(
        required_inputs=required_inputs,
        planned_outputs=planned_outputs,
        invalid_inputs=invalid_inputs,
        missing_requirements=missing_requirements,
    )
    ready_to_run = (
        not missing_inputs
        and not invalid_inputs
        and not missing_requirements
    )
    return {
        "schema_version": REAL_EXPERIMENT_PREFLIGHT_SCHEMA_VERSION,
        "action": "real_experiment_run_manifest_preflight",
        "run_manifest_schema_version": manifest["schema_version"],
        "run_manifest_path": str(manifest_path),
        "run_manifest_digest": real_experiment_run_manifest_digest(manifest),
        "ready_to_run": ready_to_run,
        "summary": {
            "required_input_count": len(required_inputs),
            "present_input_count": len(required_inputs) - len(missing_inputs),
            "missing_input_count": len(missing_inputs),
            "invalid_input_count": len(invalid_inputs),
            "missing_requirement_count": len(missing_requirements),
            "planned_output_count": len(planned_outputs),
            "existing_planned_output_count": sum(
                1 for item in planned_outputs if item["exists"] is True
            ),
        },
        "groups": groups,
        "required_inputs": required_inputs,
        "missing_inputs": missing_inputs,
        "invalid_inputs": invalid_inputs,
        "missing_requirements": missing_requirements,
        "planned_outputs": planned_outputs,
    }


def _real_experiment_execution_approval(
    *,
    manifest_path: str | Path,
    approved_execution_packet_path: str | Path | None,
) -> dict[str, Any]:
    if approved_execution_packet_path is None:
        return {
            "approved_execution_packet_path": None,
            "approved_run_manifest_path": str(manifest_path),
            "execution_packet_digest": None,
            "manifest_matches": None,
            "ready": True,
            "ready_to_execute": None,
            "required": False,
            "status": "not_required",
            "validation_valid": None,
        }

    packet_path = Path(approved_execution_packet_path)
    base = {
        "approved_execution_packet_path": str(packet_path),
        "approved_run_manifest_path": str(manifest_path),
        "execution_packet_digest": None,
        "manifest_matches": False,
        "ready": False,
        "ready_to_execute": False,
        "required": True,
        "status": "missing",
        "validation_valid": False,
    }
    if not packet_path.exists():
        return base

    try:
        from dsg_spatialqa_lab.benchmark.real_handoff import (
            load_real_experiment_execution_packet,
            validate_real_experiment_execution_packet,
        )

        packet = load_real_experiment_execution_packet(packet_path)
        validation = validate_real_experiment_execution_packet(packet)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        return {
            **base,
            "error": str(exc),
            "status": "invalid",
        }

    validation_valid = validation["valid"] is True
    ready_to_execute = packet.get("ready_to_execute") is True
    manifest_matches = str(packet.get("run_manifest_path")) == str(manifest_path)
    ready = validation_valid and ready_to_execute and manifest_matches
    if ready:
        status = "approved"
    elif not validation_valid:
        status = "invalid"
    elif not manifest_matches:
        status = "mismatch"
    else:
        status = "not_ready"
    return {
        **base,
        "execution_packet_digest": packet.get("packet_digest"),
        "manifest_matches": manifest_matches,
        "ready": ready,
        "ready_to_execute": ready_to_execute,
        "status": status,
        "validation_valid": validation_valid,
    }


def _real_experiment_run_ledger_output_path(
    manifest: Mapping[str, Any],
    run_ledger_output_path: str | Path | None,
) -> Path | None:
    if run_ledger_output_path is not None:
        return Path(run_ledger_output_path)
    manifest_path = manifest.get("real_experiment_run_ledger_path")
    if isinstance(manifest_path, str):
        return Path(manifest_path)
    return None


def _real_experiment_run_result_with_ledger_metadata(
    run_result: Mapping[str, Any],
    *,
    ledger_path: Path | None,
    write_ledger: bool,
) -> dict[str, Any]:
    result = dict(run_result)
    result["real_experiment_run_ledger_path"] = (
        str(ledger_path) if ledger_path is not None else None
    )
    result["real_experiment_run_ledger_digest"] = None
    result["real_experiment_run_ledger_written"] = False
    if ledger_path is None or not write_ledger:
        return result

    ledger = real_experiment_run_ledger(run_result)
    save_real_experiment_run_ledger(ledger, ledger_path)
    result["real_experiment_run_ledger_digest"] = ledger["ledger_digest"]
    result["real_experiment_run_ledger_written"] = True
    return result


def run_real_experiment_manifest(
    manifest_path: str | Path,
    *,
    approved_execution_packet_path: str | Path | None = None,
    run_ledger_output_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_real_experiment_run_manifest(manifest_path)
    ledger_path = _real_experiment_run_ledger_output_path(
        manifest,
        run_ledger_output_path,
    )
    execution_approval = _real_experiment_execution_approval(
        manifest_path=manifest_path,
        approved_execution_packet_path=approved_execution_packet_path,
    )
    if execution_approval["ready"] is not True:
        blocked_result = {
            "schema_version": REAL_EXPERIMENT_RUN_SCHEMA_VERSION,
            "action": "run_real_experiment_manifest",
            "dataset_name": manifest["dataset_name"],
            "run_manifest_schema_version": manifest["schema_version"],
            "run_manifest_path": str(manifest_path),
            "run_manifest_digest": real_experiment_run_manifest_digest(manifest),
            "ready": False,
            "execution_approval": execution_approval,
        }
        return _real_experiment_run_result_with_ledger_metadata(
            blocked_result,
            ledger_path=ledger_path,
            write_ledger=False,
        )
    result = run_real_experiment_package(
        dataset_name=manifest["dataset_name"],
        episode_paths=manifest["episode_paths"],
        output_dir=manifest["output_dir"],
        manifest_path=manifest["manifest_path"],
        readiness_report_path=manifest["readiness_report_path"],
        summary_report_path=manifest["summary_report_path"],
        record_path=manifest["record_path"],
        max_qa_per_episode=manifest["max_qa_per_episode"],
        tags=manifest["tags"],
        declared_data_source_kind=manifest["declared_data_source_kind"],
        min_episode_count=manifest["min_episode_count"],
        min_scene_count=manifest["min_scene_count"],
        min_qa_count=manifest["min_qa_count"],
        required_control_kinds=manifest["required_control_kinds"],
        required_predicted_input_kinds=manifest["required_predicted_input_kinds"],
        qa_eval_report_paths=manifest["qa_eval_report_paths"],
        qa_eval_delta_report_paths=manifest["qa_eval_delta_report_paths"],
        active_task_report_paths=manifest["active_task_report_paths"],
        active_task_delta_report_paths=manifest["active_task_delta_report_paths"],
        dashboard_bundle_paths=manifest["dashboard_bundle_paths"],
        error_attribution_report_paths=manifest["error_attribution_report_paths"],
        graph_eval_report_paths=manifest["graph_eval_report_paths"],
        offline_control_import_manifest_path=manifest[
            "offline_control_import_manifest_path"
        ],
        offline_control_import_run_ledger_path=manifest[
            "offline_control_import_run_ledger_path"
        ],
        offline_control_matrix_report_paths=manifest[
            "offline_control_matrix_report_paths"
        ],
        offline_control_result_report_paths=manifest[
            "offline_control_result_report_paths"
        ],
        offline_prediction_import_report_paths=manifest[
            "offline_prediction_import_report_paths"
        ],
        predicted_dsg_detector_run_manifest_path=manifest[
            "predicted_dsg_detector_run_manifest_path"
        ],
        predicted_dsg_detector_run_ledger_path=manifest[
            "predicted_dsg_detector_run_ledger_path"
        ],
        predicted_dsg_evidence_report_paths=manifest[
            "predicted_dsg_evidence_report_paths"
        ],
        predicted_graph_report_paths=manifest["predicted_graph_report_paths"],
        real_collection_report_paths=manifest["real_collection_report_paths"],
    )
    run_result = {
        **result,
        "action": "run_real_experiment_manifest",
        "run_manifest_schema_version": manifest["schema_version"],
        "run_manifest_path": str(manifest_path),
        "run_manifest_digest": real_experiment_run_manifest_digest(manifest),
        "execution_approval": execution_approval,
    }
    return _real_experiment_run_result_with_ledger_metadata(
        run_result,
        ledger_path=ledger_path,
        write_ledger=run_result["ready"] is True,
    )


def run_real_experiment_package(
    *,
    dataset_name: str,
    episode_paths: Sequence[str | Path],
    output_dir: str | Path,
    manifest_path: str | Path,
    readiness_report_path: str | Path,
    summary_report_path: str | Path,
    record_path: str | Path,
    max_qa_per_episode: int | None = None,
    tags: Sequence[str] = ("benchmark", "real"),
    declared_data_source_kind: str = "real",
    min_episode_count: int = 3,
    min_scene_count: int = 1,
    min_qa_count: int = 30,
    required_control_kinds: Sequence[str] = (
        "caption_memory",
        "graph_text",
        "multi_frame_vlm",
        "vlm",
    ),
    required_predicted_input_kinds: Sequence[str] = ("observation_sequence",),
    qa_eval_report_paths: Sequence[str | Path] = (),
    qa_eval_delta_report_paths: Sequence[str | Path] = (),
    active_task_report_paths: Sequence[str | Path] = (),
    active_task_delta_report_paths: Sequence[str | Path] = (),
    dashboard_bundle_paths: Sequence[str | Path] = (),
    error_attribution_report_paths: Sequence[str | Path] = (),
    graph_eval_report_paths: Sequence[str | Path] = (),
    offline_control_import_manifest_path: str | Path | None = None,
    offline_control_import_run_ledger_path: str | Path | None = None,
    offline_control_matrix_report_paths: Sequence[str | Path] = (),
    offline_control_result_report_paths: Sequence[str | Path] = (),
    offline_prediction_import_report_paths: Sequence[str | Path] = (),
    predicted_dsg_detector_run_manifest_path: str | Path | None = None,
    predicted_dsg_detector_run_ledger_path: str | Path | None = None,
    predicted_dsg_evidence_report_paths: Sequence[str | Path] = (),
    predicted_graph_report_paths: Sequence[str | Path] = (),
    real_collection_report_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    offline_control_import = _offline_control_import_result(
        offline_control_import_manifest_path,
        offline_control_import_run_ledger_path,
    )
    generated_artifacts = _generated_offline_artifacts(offline_control_import)
    predicted_dsg_run = _predicted_dsg_run_result(
        predicted_dsg_detector_run_manifest_path,
        predicted_dsg_detector_run_ledger_path,
    )
    generated_predicted_artifacts = _generated_predicted_artifacts(
        predicted_dsg_run
    )
    package_result = assemble_real_experiment_package(
        dataset_name=dataset_name,
        episode_paths=episode_paths,
        output_dir=output_dir,
        manifest_path=manifest_path,
        readiness_report_path=readiness_report_path,
        max_qa_per_episode=max_qa_per_episode,
        tags=tags,
        declared_data_source_kind=declared_data_source_kind,
        min_episode_count=min_episode_count,
        min_scene_count=min_scene_count,
        min_qa_count=min_qa_count,
        required_control_kinds=required_control_kinds,
        required_predicted_input_kinds=required_predicted_input_kinds,
        qa_eval_report_paths=(
            tuple(qa_eval_report_paths)
            + generated_artifacts["qa_eval_report_paths"]
        ),
        qa_eval_delta_report_paths=(
            tuple(qa_eval_delta_report_paths)
            + generated_artifacts["qa_eval_delta_report_paths"]
        ),
        active_task_report_paths=active_task_report_paths,
        active_task_delta_report_paths=active_task_delta_report_paths,
        dashboard_bundle_paths=dashboard_bundle_paths,
        error_attribution_report_paths=error_attribution_report_paths,
        graph_eval_report_paths=graph_eval_report_paths,
        offline_control_matrix_report_paths=(
            tuple(offline_control_matrix_report_paths)
            + generated_artifacts["offline_control_matrix_report_paths"]
        ),
        offline_control_result_report_paths=(
            tuple(offline_control_result_report_paths)
            + generated_artifacts["offline_control_result_report_paths"]
        ),
        offline_prediction_import_report_paths=(
            tuple(offline_prediction_import_report_paths)
            + generated_artifacts["offline_prediction_import_report_paths"]
        ),
        predicted_dsg_evidence_report_paths=(
            tuple(predicted_dsg_evidence_report_paths)
            + generated_predicted_artifacts["predicted_dsg_evidence_report_paths"]
        ),
        predicted_graph_report_paths=(
            tuple(predicted_graph_report_paths)
            + generated_predicted_artifacts["predicted_graph_report_paths"]
        ),
        real_collection_report_paths=real_collection_report_paths,
    )
    if package_result["ready"] is not True:
        return _not_ready_result(
            dataset_name=dataset_name,
            package_result=package_result,
            offline_control_import=offline_control_import,
            generated_artifacts=generated_artifacts,
            predicted_dsg_run=predicted_dsg_run,
            generated_predicted_artifacts=generated_predicted_artifacts,
        )

    manifest = load_benchmark_manifest(manifest_path)
    readiness_report = load_real_experiment_readiness_report(readiness_report_path)
    summary_report = experiment_summary_report(manifest, manifest_path=manifest_path)
    save_experiment_summary_report(summary_report, summary_report_path)
    dashboard_bundle_path = _first_path(dashboard_bundle_paths)
    dashboard_bundle = (
        load_dashboard_bundle(dashboard_bundle_path)
        if dashboard_bundle_path is not None
        else None
    )
    record = experiment_record(
        summary_report,
        summary_report_path=summary_report_path,
        dashboard_bundle=dashboard_bundle,
        dashboard_bundle_path=dashboard_bundle_path,
        real_readiness_report=readiness_report,
        real_readiness_report_path=readiness_report_path,
    )
    save_experiment_record(record, record_path)
    return {
        "schema_version": REAL_EXPERIMENT_RUN_SCHEMA_VERSION,
        "action": "run_real_experiment_package",
        "dataset_name": dataset_name,
        "manifest_path": str(manifest_path),
        "readiness_report_path": str(readiness_report_path),
        "summary_report_path": str(summary_report_path),
        "record_path": str(record_path),
        "manifest_digest": package_result["manifest_digest"],
        "readiness_report_digest": readiness_report["report_digest"],
        "summary_report_digest": summary_report["report_digest"],
        "record_digest": record["record_digest"],
        "ready": True,
        "readiness": readiness_report["readiness"],
        "readiness_status": summary_report["readiness"]["status"],
        "real_package_status": record["real_package_status"],
        **_offline_control_import_payload(
            offline_control_import,
            generated_artifacts,
        ),
        **_predicted_dsg_run_payload(
            predicted_dsg_run,
            generated_predicted_artifacts,
        ),
        "summary": summary_report["summary"],
        "verdict_counts": record["verdict_counts"],
    }


def _preflight_offline_controls(
    manifest: dict[str, Any],
    *,
    required_inputs: list[dict[str, Any]],
    planned_outputs: list[dict[str, Any]],
    invalid_inputs: list[dict[str, Any]],
    missing_requirements: list[dict[str, str]],
) -> bool:
    _add_required_inputs(
        required_inputs,
        group="offline_controls",
        role="offline_prediction_import_report",
        paths=manifest["offline_prediction_import_report_paths"],
    )
    _add_required_inputs(
        required_inputs,
        group="offline_controls",
        role="offline_control_matrix_report",
        paths=manifest["offline_control_matrix_report_paths"],
    )
    _add_required_inputs(
        required_inputs,
        group="offline_controls",
        role="offline_control_result_report",
        paths=manifest["offline_control_result_report_paths"],
    )
    manifest_path = manifest["offline_control_import_manifest_path"]
    if manifest_path is None:
        if not manifest["offline_prediction_import_report_paths"]:
            _add_missing_requirement(
                missing_requirements,
                group="offline_controls",
                role="offline_control_import_manifest_or_import_reports",
                reason="offline controls need an import manifest or import reports",
            )
        if not manifest["offline_control_matrix_report_paths"]:
            _add_missing_requirement(
                missing_requirements,
                group="offline_controls",
                role="offline_control_matrix_report",
                reason="offline controls need a matrix report",
            )
        if not manifest["offline_control_result_report_paths"]:
            _add_missing_requirement(
                missing_requirements,
                group="offline_controls",
                role="offline_control_result_report",
                reason="offline controls need a result report",
            )
        return bool(manifest["qa_eval_delta_report_paths"])

    _add_required_input(
        required_inputs,
        group="offline_controls",
        role="offline_control_import_manifest",
        path=manifest_path,
    )
    if not Path(str(manifest_path)).exists():
        return False
    ledger_path = manifest["offline_control_import_run_ledger_path"]
    if ledger_path is not None:
        _add_planned_output(
            planned_outputs,
            group="offline_controls",
            role="offline_control_import_run_ledger",
            path=ledger_path,
        )
    try:
        offline_manifest = load_offline_control_import_manifest(manifest_path)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _add_invalid_input(
            invalid_inputs,
            group="offline_controls",
            role="offline_control_import_manifest",
            path=manifest_path,
            error=exc,
        )
        return False

    _add_required_input(
        required_inputs,
        group="offline_controls",
        role="gold_qa",
        path=offline_manifest["qa_path"],
    )
    _add_planned_output(
        planned_outputs,
        group="offline_controls",
        role="output_dir",
        path=offline_manifest["output_dir"],
    )
    _add_planned_output(
        planned_outputs,
        group="offline_controls",
        role="offline_control_matrix_report",
        path=offline_manifest["matrix_report_path"],
    )
    qa_eval_output_dir = offline_manifest.get("qa_eval_output_dir")
    if qa_eval_output_dir is not None:
        _add_planned_output(
            planned_outputs,
            group="offline_controls",
            role="qa_eval_output_dir",
            path=qa_eval_output_dir,
        )
    result_report_path = offline_manifest.get("result_report_path")
    if result_report_path is not None:
        _add_planned_output(
            planned_outputs,
            group="offline_controls",
            role="offline_control_result_report",
            path=result_report_path,
        )
    candidate_path = offline_manifest.get("candidate_prediction_path")
    if candidate_path is not None:
        _add_required_input(
            required_inputs,
            group="offline_controls",
            role="candidate_prediction",
            path=candidate_path,
        )
    sources = offline_manifest["sources"]
    for source in sources:
        metadata = {
            "source_kind": str(source["source_kind"]),
            "source_name": str(source["source_name"]),
        }
        _add_required_input(
            required_inputs,
            group="offline_controls",
            role="offline_control_source_input",
            path=source["input_path"],
            metadata=metadata,
        )
        prediction_path = source.get("prediction_path")
        if prediction_path is not None:
            _add_planned_output(
                planned_outputs,
                group="offline_controls",
                role="offline_control_prediction",
                path=prediction_path,
                metadata=metadata,
            )
        import_report_path = source.get("import_report_path")
        if import_report_path is not None:
            _add_planned_output(
                planned_outputs,
                group="offline_controls",
                role="offline_prediction_import_report",
                path=import_report_path,
                metadata=metadata,
            )
    return candidate_path is not None


def _preflight_predicted_dsg(
    manifest: dict[str, Any],
    *,
    required_inputs: list[dict[str, Any]],
    planned_outputs: list[dict[str, Any]],
    invalid_inputs: list[dict[str, Any]],
    missing_requirements: list[dict[str, str]],
) -> None:
    _add_required_inputs(
        required_inputs,
        group="predicted_dsg",
        role="predicted_graph_report",
        paths=manifest["predicted_graph_report_paths"],
    )
    _add_required_inputs(
        required_inputs,
        group="predicted_dsg",
        role="predicted_dsg_evidence_report",
        paths=manifest["predicted_dsg_evidence_report_paths"],
    )
    manifest_path = manifest["predicted_dsg_detector_run_manifest_path"]
    if manifest_path is None:
        if not (
            manifest["predicted_graph_report_paths"]
            and manifest["predicted_dsg_evidence_report_paths"]
        ):
            _add_missing_requirement(
                missing_requirements,
                group="predicted_dsg",
                role="predicted_dsg_detector_run_manifest_or_reports",
                reason="predicted DSG needs a detector-run manifest or graph/evidence reports",
            )
        return

    _add_required_input(
        required_inputs,
        group="predicted_dsg",
        role="predicted_dsg_detector_run_manifest",
        path=manifest_path,
    )
    if not Path(str(manifest_path)).exists():
        return
    ledger_path = manifest["predicted_dsg_detector_run_ledger_path"]
    if ledger_path is not None:
        _add_planned_output(
            planned_outputs,
            group="predicted_dsg",
            role="predicted_dsg_detector_run_ledger",
            path=ledger_path,
        )
    try:
        predicted_manifest = load_predicted_dsg_detector_run_manifest(manifest_path)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _add_invalid_input(
            invalid_inputs,
            group="predicted_dsg",
            role="predicted_dsg_detector_run_manifest",
            path=manifest_path,
            error=exc,
        )
        return

    _add_required_input(
        required_inputs,
        group="predicted_dsg",
        role="detector_jsonl",
        path=predicted_manifest["detector_jsonl_path"],
    )
    for role, key in (
        ("observation_sequence", "output_sequence_path"),
        ("predicted_graph", "output_graph_path"),
        ("predicted_graph_report", "predicted_graph_report_path"),
        ("detector_import_report", "detector_import_report_path"),
        ("predicted_dsg_evidence_report", "predicted_dsg_evidence_report_path"),
    ):
        _add_planned_output(
            planned_outputs,
            group="predicted_dsg",
            role=role,
            path=predicted_manifest[key],
        )


def _preflight_review_artifacts(
    manifest: dict[str, Any],
    *,
    required_inputs: list[dict[str, Any]],
    missing_requirements: list[dict[str, str]],
    offline_generates_deltas: bool,
) -> None:
    review_fields = (
        ("qa_eval_report", "qa_eval_report_paths"),
        ("qa_eval_delta_report", "qa_eval_delta_report_paths"),
        ("active_task_report", "active_task_report_paths"),
        ("active_task_delta_report", "active_task_delta_report_paths"),
        ("dashboard_bundle", "dashboard_bundle_paths"),
        ("error_attribution_report", "error_attribution_report_paths"),
        ("graph_eval_report", "graph_eval_report_paths"),
    )
    for role, field in review_fields:
        _add_required_inputs(
            required_inputs,
            group="review_artifacts",
            role=role,
            paths=manifest[field],
        )
    if not manifest["qa_eval_delta_report_paths"] and not offline_generates_deltas:
        _add_missing_requirement(
            missing_requirements,
            group="review_artifacts",
            role="qa_eval_delta_report",
            reason="DSG-vs-control QA delta evidence is required",
        )
    for role, field in (
        ("active_task_delta_report", "active_task_delta_report_paths"),
        ("dashboard_bundle", "dashboard_bundle_paths"),
        ("error_attribution_report", "error_attribution_report_paths"),
        ("graph_eval_report", "graph_eval_report_paths"),
    ):
        if not manifest[field]:
            _add_missing_requirement(
                missing_requirements,
                group="review_artifacts",
                role=role,
                reason=f"review artifact path is required: {role}",
            )


def _preflight_run_outputs(
    manifest: dict[str, Any],
    planned_outputs: list[dict[str, Any]],
) -> None:
    for role, key in (
        ("output_dir", "output_dir"),
        ("benchmark_manifest", "manifest_path"),
        ("real_readiness_report", "readiness_report_path"),
        ("experiment_summary", "summary_report_path"),
        ("experiment_record", "record_path"),
    ):
        _add_planned_output(
            planned_outputs,
            group="real_run",
            role=role,
            path=manifest[key],
        )


def _add_required_inputs(
    items: list[dict[str, Any]],
    *,
    group: str,
    role: str,
    paths: Sequence[str | Path],
) -> None:
    for path in paths:
        _add_required_input(items, group=group, role=role, path=path)


def _add_required_input(
    items: list[dict[str, Any]],
    *,
    group: str,
    role: str,
    path: str | Path,
    metadata: dict[str, str] | None = None,
) -> None:
    items.append(_path_row(group=group, role=role, path=path, metadata=metadata))


def _add_planned_output(
    items: list[dict[str, Any]],
    *,
    group: str,
    role: str,
    path: str | Path,
    metadata: dict[str, str] | None = None,
) -> None:
    items.append(_path_row(group=group, role=role, path=path, metadata=metadata))


def _add_invalid_input(
    items: list[dict[str, Any]],
    *,
    group: str,
    role: str,
    path: str | Path,
    error: Exception,
) -> None:
    row = _path_row(group=group, role=role, path=path)
    row["error"] = str(error)
    items.append(row)


def _add_missing_requirement(
    items: list[dict[str, str]],
    *,
    group: str,
    role: str,
    reason: str,
) -> None:
    items.append({"group": group, "role": role, "reason": reason})


def _path_row(
    *,
    group: str,
    role: str,
    path: str | Path,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    local_path = Path(str(path))
    row: dict[str, Any] = {
        "exists": local_path.exists(),
        "group": group,
        "path": str(path),
        "role": role,
    }
    if metadata is not None:
        row["metadata"] = {key: metadata[key] for key in sorted(metadata)}
    return row


def _sorted_path_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda item: (
            str(item["group"]),
            str(item["role"]),
            str(item["path"]),
            json.dumps(item.get("metadata", {}), sort_keys=True),
        ),
    )


def _preflight_groups(
    *,
    required_inputs: Sequence[dict[str, Any]],
    planned_outputs: Sequence[dict[str, Any]],
    invalid_inputs: Sequence[dict[str, Any]],
    missing_requirements: Sequence[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    group_names = (
        "real_collection",
        "offline_controls",
        "predicted_dsg",
        "review_artifacts",
        "real_run",
    )
    groups: dict[str, dict[str, Any]] = {}
    for group in group_names:
        group_inputs = [item for item in required_inputs if item["group"] == group]
        group_outputs = [item for item in planned_outputs if item["group"] == group]
        group_missing = [
            item for item in group_inputs if item["exists"] is False
        ]
        group_invalid = [
            item for item in invalid_inputs if item["group"] == group
        ]
        group_missing_requirements = [
            item for item in missing_requirements if item["group"] == group
        ]
        groups[group] = {
            "ready": (
                not group_missing
                and not group_invalid
                and not group_missing_requirements
            ),
            "required_input_count": len(group_inputs),
            "missing_input_count": len(group_missing),
            "invalid_input_count": len(group_invalid),
            "missing_requirement_count": len(group_missing_requirements),
            "planned_output_count": len(group_outputs),
        }
    return groups


def _real_experiment_run_manifest(
    payload: dict[str, Any],
    base_dir: Path,
) -> dict[str, Any]:
    schema_version = _required_str(payload, "schema_version")
    if schema_version != REAL_EXPERIMENT_RUN_MANIFEST_SCHEMA_VERSION:
        raise SpatialQAError(
            f"Unsupported real experiment run manifest schema version: {schema_version}"
        )
    return {
        "schema_version": schema_version,
        "dataset_name": _optional_str(
            payload.get("dataset_name"),
            "real_experiment",
            "dataset_name",
        ),
        "episode_paths": list(
            _path_sequence(payload.get("episode_paths"), base_dir, "episode_paths")
        ),
        "output_dir": str(_required_path(payload, "output_dir", base_dir)),
        "manifest_path": str(_required_path(payload, "manifest_path", base_dir)),
        "readiness_report_path": str(
            _required_path(payload, "readiness_report_path", base_dir)
        ),
        "summary_report_path": str(
            _required_path(payload, "summary_report_path", base_dir)
        ),
        "record_path": str(_required_path(payload, "record_path", base_dir)),
        "real_experiment_run_ledger_path": _optional_path(
            payload,
            "real_experiment_run_ledger_path",
            base_dir,
        ),
        "max_qa_per_episode": _optional_int_or_none(
            payload.get("max_qa_per_episode"),
            "max_qa_per_episode",
        ),
        "tags": list(
            _optional_string_sequence(
                payload.get("tags"),
                ("benchmark", "real"),
                "tags",
            )
        ),
        "declared_data_source_kind": _optional_str(
            payload.get("declared_data_source_kind"),
            "real",
            "declared_data_source_kind",
        ),
        "real_collection_source_kind": _optional_str(
            payload.get("real_collection_source_kind"),
            "ai2thor",
            "real_collection_source_kind",
        ),
        "min_episode_count": _optional_int(
            payload.get("min_episode_count"),
            3,
            "min_episode_count",
        ),
        "min_scene_count": _optional_int(
            payload.get("min_scene_count"),
            1,
            "min_scene_count",
        ),
        "min_frame_count": _optional_int(
            payload.get("min_frame_count"),
            30,
            "min_frame_count",
        ),
        "min_qa_count": _optional_int(
            payload.get("min_qa_count"),
            30,
            "min_qa_count",
        ),
        "required_control_kinds": list(
            _optional_string_sequence(
                payload.get("required_control_kinds"),
                ("caption_memory", "graph_text", "multi_frame_vlm", "vlm"),
                "required_control_kinds",
            )
        ),
        "required_predicted_input_kinds": list(
            _optional_string_sequence(
                payload.get("required_predicted_input_kinds"),
                ("observation_sequence",),
                "required_predicted_input_kinds",
            )
        ),
        "qa_eval_report_paths": list(
            _optional_path_sequence(payload, "qa_eval_report_paths", base_dir)
        ),
        "qa_eval_delta_report_paths": list(
            _optional_path_sequence(payload, "qa_eval_delta_report_paths", base_dir)
        ),
        "active_task_report_paths": list(
            _optional_path_sequence(payload, "active_task_report_paths", base_dir)
        ),
        "active_task_delta_report_paths": list(
            _optional_path_sequence(
                payload,
                "active_task_delta_report_paths",
                base_dir,
            )
        ),
        "dashboard_bundle_paths": list(
            _optional_path_sequence(payload, "dashboard_bundle_paths", base_dir)
        ),
        "error_attribution_report_paths": list(
            _optional_path_sequence(
                payload,
                "error_attribution_report_paths",
                base_dir,
            )
        ),
        "graph_eval_report_paths": list(
            _optional_path_sequence(payload, "graph_eval_report_paths", base_dir)
        ),
        "offline_control_import_manifest_path": _optional_path(
            payload,
            "offline_control_import_manifest_path",
            base_dir,
        ),
        "offline_control_import_run_ledger_path": _optional_path(
            payload,
            "offline_control_import_run_ledger_path",
            base_dir,
        ),
        "offline_control_matrix_report_paths": list(
            _optional_path_sequence(
                payload,
                "offline_control_matrix_report_paths",
                base_dir,
            )
        ),
        "offline_control_result_report_paths": list(
            _optional_path_sequence(
                payload,
                "offline_control_result_report_paths",
                base_dir,
            )
        ),
        "offline_prediction_import_report_paths": list(
            _optional_path_sequence(
                payload,
                "offline_prediction_import_report_paths",
                base_dir,
            )
        ),
        "predicted_dsg_detector_run_manifest_path": _optional_path(
            payload,
            "predicted_dsg_detector_run_manifest_path",
            base_dir,
        ),
        "predicted_dsg_detector_run_ledger_path": _optional_path(
            payload,
            "predicted_dsg_detector_run_ledger_path",
            base_dir,
        ),
        "predicted_dsg_evidence_report_paths": list(
            _optional_path_sequence(
                payload,
                "predicted_dsg_evidence_report_paths",
                base_dir,
            )
        ),
        "predicted_graph_report_paths": list(
            _optional_path_sequence(payload, "predicted_graph_report_paths", base_dir)
        ),
        "real_collection_report_paths": list(
            _optional_path_sequence(payload, "real_collection_report_paths", base_dir)
        ),
    }


def _not_ready_result(
    *,
    dataset_name: str,
    package_result: dict[str, Any],
    offline_control_import: dict[str, Any] | None = None,
    generated_artifacts: dict[str, tuple[str | Path, ...]] | None = None,
    predicted_dsg_run: dict[str, Any] | None = None,
    generated_predicted_artifacts: dict[str, tuple[str | Path, ...]] | None = None,
) -> dict[str, Any]:
    generated = generated_artifacts or _empty_generated_offline_artifacts()
    generated_predicted = (
        generated_predicted_artifacts or _empty_generated_predicted_artifacts()
    )
    return {
        "schema_version": REAL_EXPERIMENT_RUN_SCHEMA_VERSION,
        "action": "run_real_experiment_package",
        "dataset_name": dataset_name,
        "manifest_path": package_result["manifest_path"],
        "readiness_report_path": package_result["readiness_report_path"],
        "summary_report_path": None,
        "record_path": None,
        "manifest_digest": package_result["manifest_digest"],
        "readiness_report_digest": package_result["readiness_report_digest"],
        "summary_report_digest": None,
        "record_digest": None,
        "ready": False,
        "readiness": package_result["readiness"],
        "readiness_status": "not_ready",
        "real_package_status": "not_ready",
        **_offline_control_import_payload(
            offline_control_import,
            generated,
        ),
        **_predicted_dsg_run_payload(
            predicted_dsg_run,
            generated_predicted,
        ),
        "summary": package_result["summary"],
        "verdict_counts": None,
    }


def _offline_control_import_result(
    manifest_path: str | Path | None,
    ledger_path: str | Path | None,
) -> dict[str, Any] | None:
    if manifest_path is None:
        return None
    result = run_offline_control_import_manifest(manifest_path)
    if ledger_path is not None:
        ledger = offline_control_import_run_ledger(result)
        save_offline_control_import_run_ledger(ledger, ledger_path)
        result["run_ledger_path"] = str(ledger_path)
        result["run_ledger_digest"] = ledger["ledger_digest"]
    return result


def _predicted_dsg_run_result(
    manifest_path: str | Path | None,
    ledger_path: str | Path | None,
) -> dict[str, Any] | None:
    if manifest_path is None:
        return None
    result = run_predicted_dsg_detector_run_manifest(manifest_path)
    if ledger_path is not None:
        ledger = predicted_dsg_detector_run_ledger(result)
        save_predicted_dsg_detector_run_ledger(ledger, ledger_path)
        result["run_ledger_path"] = str(ledger_path)
        result["run_ledger_digest"] = ledger["ledger_digest"]
    return result


def _generated_offline_artifacts(
    offline_control_import: dict[str, Any] | None,
) -> dict[str, tuple[str | Path, ...]]:
    if offline_control_import is None:
        return _empty_generated_offline_artifacts()
    return {
        "offline_prediction_import_report_paths": tuple(
            str(source["import_report_path"])
            for source in offline_control_import["sources"]
        ),
        "offline_control_matrix_report_paths": (
            str(offline_control_import["matrix_report_path"]),
        ),
        "offline_control_result_report_paths": _generated_offline_result_report_paths(
            offline_control_import
        ),
        "qa_eval_report_paths": _generated_qa_eval_report_paths(
            offline_control_import
        ),
        "qa_eval_delta_report_paths": tuple(
            str(path)
            for _, path in sorted(
                offline_control_import["qa_eval_delta_report_paths"].items()
            )
        ),
    }


def _empty_generated_offline_artifacts() -> dict[str, tuple[str | Path, ...]]:
    return {
        "offline_prediction_import_report_paths": (),
        "offline_control_matrix_report_paths": (),
        "offline_control_result_report_paths": (),
        "qa_eval_report_paths": (),
        "qa_eval_delta_report_paths": (),
    }


def _generated_predicted_artifacts(
    predicted_dsg_run: dict[str, Any] | None,
) -> dict[str, tuple[str | Path, ...]]:
    if predicted_dsg_run is None:
        return _empty_generated_predicted_artifacts()
    return {
        "predicted_graph_report_paths": (
            str(predicted_dsg_run["predicted_graph_report_path"]),
        ),
        "predicted_dsg_evidence_report_paths": (
            str(predicted_dsg_run["predicted_dsg_evidence_report_path"]),
        ),
    }


def _empty_generated_predicted_artifacts() -> dict[str, tuple[str | Path, ...]]:
    return {
        "predicted_graph_report_paths": (),
        "predicted_dsg_evidence_report_paths": (),
    }


def _generated_qa_eval_report_paths(
    offline_control_import: dict[str, Any],
) -> tuple[str, ...]:
    handoff = offline_control_import["qa_eval_handoff"]
    paths: list[str] = []
    candidate_path = handoff.get("candidate_qa_eval_report_path")
    if isinstance(candidate_path, str):
        paths.append(candidate_path)
    baseline_paths = handoff.get("baseline_qa_eval_report_paths")
    if isinstance(baseline_paths, dict):
        paths.extend(str(path) for _, path in sorted(baseline_paths.items()))
    return tuple(paths)


def _generated_offline_result_report_paths(
    offline_control_import: dict[str, Any],
) -> tuple[str, ...]:
    path = offline_control_import.get("offline_control_result_report_path")
    if isinstance(path, str) and path != "":
        return (path,)
    return ()


def _offline_control_import_payload(
    offline_control_import: dict[str, Any] | None,
    generated_artifacts: dict[str, tuple[str | Path, ...]],
) -> dict[str, Any]:
    if offline_control_import is None:
        return {
            "offline_control_import": None,
            "offline_control_import_manifest_path": None,
            "offline_control_import_manifest_digest": None,
            "offline_control_import_ready": None,
            "generated_offline_control_import_run_ledger_path": None,
            "offline_control_import_run_ledger_digest": None,
            "generated_offline_control_source_kinds": [],
            "generated_offline_prediction_import_report_paths": [],
            "generated_offline_control_matrix_report_path": None,
            "generated_offline_control_result_report_path": None,
            "generated_qa_eval_report_paths": [],
            "generated_qa_eval_delta_report_paths": [],
        }
    return {
        "offline_control_import": offline_control_import,
        "offline_control_import_manifest_path": offline_control_import[
            "manifest_path"
        ],
        "offline_control_import_manifest_digest": offline_control_import[
            "manifest_digest"
        ],
        "offline_control_import_ready": offline_control_import["ready"],
        "generated_offline_control_import_run_ledger_path": (
            offline_control_import.get("run_ledger_path")
        ),
        "offline_control_import_run_ledger_digest": (
            offline_control_import.get("run_ledger_digest")
        ),
        "generated_offline_control_source_kinds": sorted(
            str(source["source_kind"])
            for source in offline_control_import["sources"]
        ),
        "generated_offline_prediction_import_report_paths": [
            str(path)
            for path in generated_artifacts["offline_prediction_import_report_paths"]
        ],
        "generated_offline_control_matrix_report_path": str(
            generated_artifacts["offline_control_matrix_report_paths"][0]
        )
        if generated_artifacts["offline_control_matrix_report_paths"]
        else None,
        "generated_offline_control_result_report_path": str(
            generated_artifacts["offline_control_result_report_paths"][0]
        )
        if generated_artifacts["offline_control_result_report_paths"]
        else None,
        "generated_qa_eval_report_paths": [
            str(path) for path in generated_artifacts["qa_eval_report_paths"]
        ],
        "generated_qa_eval_delta_report_paths": [
            str(path) for path in generated_artifacts["qa_eval_delta_report_paths"]
        ],
    }


def _predicted_dsg_run_payload(
    predicted_dsg_run: dict[str, Any] | None,
    generated_artifacts: dict[str, tuple[str | Path, ...]],
) -> dict[str, Any]:
    if predicted_dsg_run is None:
        return {
            "predicted_dsg_detector_run": None,
            "predicted_dsg_detector_run_manifest_path": None,
            "predicted_dsg_detector_run_manifest_digest": None,
            "predicted_dsg_detector_run_ready": None,
            "generated_predicted_dsg_detector_run_ledger_path": None,
            "predicted_dsg_detector_run_ledger_digest": None,
            "generated_observation_sequence_path": None,
            "generated_predicted_graph_path": None,
            "generated_predicted_graph_report_path": None,
            "generated_detector_import_report_path": None,
            "generated_predicted_dsg_evidence_report_path": None,
        }
    predicted_graph_report_path = generated_artifacts["predicted_graph_report_paths"][0]
    predicted_dsg_evidence_report_path = generated_artifacts[
        "predicted_dsg_evidence_report_paths"
    ][0]
    return {
        "predicted_dsg_detector_run": predicted_dsg_run,
        "predicted_dsg_detector_run_manifest_path": predicted_dsg_run[
            "manifest_path"
        ],
        "predicted_dsg_detector_run_manifest_digest": predicted_dsg_run[
            "manifest_digest"
        ],
        "predicted_dsg_detector_run_ready": predicted_dsg_run["ready"],
        "generated_predicted_dsg_detector_run_ledger_path": (
            predicted_dsg_run.get("run_ledger_path")
        ),
        "predicted_dsg_detector_run_ledger_digest": (
            predicted_dsg_run.get("run_ledger_digest")
        ),
        "generated_observation_sequence_path": predicted_dsg_run[
            "observation_sequence_path"
        ],
        "generated_predicted_graph_path": predicted_dsg_run["graph_path"],
        "generated_predicted_graph_report_path": str(predicted_graph_report_path),
        "generated_detector_import_report_path": predicted_dsg_run[
            "detector_import_report_path"
        ],
        "generated_predicted_dsg_evidence_report_path": str(
            predicted_dsg_evidence_report_path
        ),
    }


def _first_path(paths: Sequence[str | Path]) -> Path | None:
    if not paths:
        return None
    return Path(paths[0])


def _required_path(payload: dict[str, Any], key: str, base_dir: Path) -> Path:
    value = _required_str(payload, key)
    return _manifest_path(value, base_dir)


def _optional_path(payload: dict[str, Any], key: str, base_dir: Path) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Real experiment run manifest path must be a string: {key}")
    return str(_manifest_path(value, base_dir))


def _manifest_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    try:
        path.relative_to(base_dir)
        return path
    except ValueError:
        pass
    return base_dir / path


def _path_sequence(value: object, base_dir: Path, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise SpatialQAError(
            f"Real experiment run manifest field must be a path list: {field}"
        )
    paths: list[str] = []
    for item in value:
        if not isinstance(item, str) or item == "":
            raise SpatialQAError(
                f"Real experiment run manifest field must be a path list: {field}"
            )
        paths.append(str(_manifest_path(item, base_dir)))
    if not paths:
        raise SpatialQAError(
            f"Real experiment run manifest field must not be empty: {field}"
        )
    return tuple(paths)


def _optional_path_sequence(
    payload: dict[str, Any],
    field: str,
    base_dir: Path,
) -> tuple[str, ...]:
    value = payload.get(field)
    if value is None:
        return ()
    return _path_sequence(value, base_dir, field)


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Real experiment run manifest field is required: {key}")
    return value


def _optional_str(value: object, default: str, field: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(
            f"Real experiment run manifest field must be a string: {field}"
        )
    return value


def _optional_string_sequence(
    value: object,
    default: Sequence[str],
    field: str,
) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise SpatialQAError(
            f"Real experiment run manifest field must be a string list: {field}"
        )
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or item == "":
            raise SpatialQAError(
                f"Real experiment run manifest field must be a string list: {field}"
            )
        items.append(item)
    if not items:
        raise SpatialQAError(
            f"Real experiment run manifest field must not be empty: {field}"
        )
    return tuple(items)


def _optional_int(value: object, default: int, field: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpatialQAError(
            f"Real experiment run manifest field must be an integer: {field}"
        )
    return value


def _optional_int_or_none(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpatialQAError(
            f"Real experiment run manifest field must be an integer or null: {field}"
        )
    return value


def _json_value(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise SpatialQAError("Real experiment run manifest contains non-JSON value")
