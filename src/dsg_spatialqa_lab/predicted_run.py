from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.observations import (
    DETECTOR_OBSERVATION_RECORD_SCHEMA_VERSION,
    detector_observation_import_report,
    detector_observation_import_report_digest,
    detector_observation_records_digest,
    detector_observation_sequence_from_jsonl,
    load_detector_observation_import_report,
    load_scene_observation_sequence,
    save_detector_observation_import_report,
    save_scene_observation_sequence,
    scene_observation_sequence_digest,
)
from dsg_spatialqa_lab.predicted import (
    OBSERVATION_PREDICTED_REFERENCE_FRAMES,
    OBSERVATION_PREDICTED_RELATIONS,
    build_predicted_graph_from_observations,
    load_predicted_graph_report,
    predicted_graph_report_from_observations,
    predicted_graph_report_digest,
    save_predicted_graph_report,
    validate_predicted_graph_report,
)
from dsg_spatialqa_lab.predicted_evidence import (
    DEFAULT_REQUIRED_PREDICTED_DSG_EVIDENCE_KINDS,
    load_predicted_dsg_evidence_report,
    predicted_dsg_evidence_report,
    predicted_dsg_evidence_report_digest,
    save_predicted_dsg_evidence_report,
)
from dsg_spatialqa_lab.scene_io import (
    graph_json_digest,
    load_graph_json,
    save_graph_json,
)
from dsg_spatialqa_lab.schema import SpatialQAError


PREDICTED_DSG_DETECTOR_RUN_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.predicted-dsg-detector-run.v1"
)
PREDICTED_DSG_DETECTOR_RUN_MANIFEST_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.predicted-dsg-detector-run-manifest.v1"
)
PREDICTED_DSG_DETECTOR_PREFLIGHT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.predicted-dsg-detector-preflight.v1"
)
PREDICTED_DSG_DETECTOR_ARTIFACT_CONTRACT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.predicted-dsg-detector-artifact-contract.v1"
)
PREDICTED_DSG_DETECTOR_ARTIFACT_LAUNCH_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.predicted-dsg-detector-artifact-launch-report.v1"
)
PREDICTED_DSG_DETECTOR_RUN_LEDGER_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.predicted-dsg-detector-run-ledger.v1"
)
PREDICTED_DSG_DETECTOR_REQUEST_BUNDLE_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.predicted-dsg-detector-request-bundle.v1"
)
PREDICTED_DSG_DETECTOR_RECEIPT_BUNDLE_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.predicted-dsg-detector-receipt-bundle.v1"
)
NON_REAL_PREDICTED_DETECTOR_SOURCE_MARKERS = (
    "ai2thor",
    "dummy",
    "fake",
    "mock",
    "placeholder",
    "synthetic",
)


def load_predicted_dsg_detector_run_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SpatialQAError("Predicted DSG detector run manifest JSON must be an object")
    return _predicted_dsg_detector_run_manifest(payload, manifest_path.parent)


def predicted_dsg_detector_run_manifest_digest(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_value(manifest),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def predicted_dsg_detector_request_bundle_digest(bundle: Mapping[str, Any]) -> str:
    payload = {
        key: value for key, value in bundle.items() if key != "request_bundle_digest"
    }
    return hashlib.sha256(
        json.dumps(
            _json_value(payload),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def predicted_dsg_detector_receipt_bundle_digest(bundle: Mapping[str, Any]) -> str:
    payload = {
        key: value for key, value in bundle.items() if key != "receipt_bundle_digest"
    }
    return hashlib.sha256(
        json.dumps(
            _json_value(payload),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def predicted_dsg_detector_request_bundle_json(bundle: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(bundle), indent=2, sort_keys=True) + "\n"


def predicted_dsg_detector_receipt_bundle_json(bundle: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(bundle), indent=2, sort_keys=True) + "\n"


def save_predicted_dsg_detector_request_bundle(
    bundle: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        predicted_dsg_detector_request_bundle_json(bundle),
        encoding="utf-8",
    )
    return output_path


def save_predicted_dsg_detector_receipt_bundle(
    bundle: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        predicted_dsg_detector_receipt_bundle_json(bundle),
        encoding="utf-8",
    )
    return output_path


def load_predicted_dsg_detector_request_bundle(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError(
            "Predicted DSG detector request bundle JSON must be an object"
        )
    schema_version = payload.get("schema_version")
    if schema_version != PREDICTED_DSG_DETECTOR_REQUEST_BUNDLE_SCHEMA_VERSION:
        raise SpatialQAError(
            "Unsupported predicted DSG detector request bundle schema version: "
            f"{schema_version}"
        )
    return cast(dict[str, Any], payload)


def load_predicted_dsg_detector_receipt_bundle(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError(
            "Predicted DSG detector receipt bundle JSON must be an object"
        )
    schema_version = payload.get("schema_version")
    if schema_version != PREDICTED_DSG_DETECTOR_RECEIPT_BUNDLE_SCHEMA_VERSION:
        raise SpatialQAError(
            "Unsupported predicted DSG detector receipt bundle schema version: "
            f"{schema_version}"
        )
    return cast(dict[str, Any], payload)


def predicted_dsg_detector_request_bundle(manifest_path: str | Path) -> dict[str, Any]:
    manifest = load_predicted_dsg_detector_run_manifest(manifest_path)
    bundle: dict[str, Any] = {
        "schema_version": PREDICTED_DSG_DETECTOR_REQUEST_BUNDLE_SCHEMA_VERSION,
        "action": "predicted_dsg_detector_request_bundle",
        "manifest_path": str(manifest_path),
        "manifest_digest": predicted_dsg_detector_run_manifest_digest(manifest),
        "detector_jsonl": {
            "expected_schema_version": DETECTOR_OBSERVATION_RECORD_SCHEMA_VERSION,
            "input_format": "detector_observation_jsonl",
            "path": manifest["detector_jsonl_path"],
        },
        "frame_asset_fields": [
            "rgb_path",
            "depth_path",
            "segmentation_path",
        ],
        "build_requirements": _request_bundle_build_requirements(manifest),
        "planned_outputs": _request_bundle_planned_outputs(manifest),
        "commands": {
            "build": f"python scripts/run_predicted_dsg.py --manifest {manifest_path}",
            "preflight": (
                "python scripts/run_predicted_dsg.py --preflight-manifest "
                f"{manifest_path}"
            ),
        },
        "record_template": _detector_record_template(),
    }
    bundle["request_bundle_digest"] = predicted_dsg_detector_request_bundle_digest(
        bundle
    )
    return bundle


def predicted_dsg_detector_receipt_bundle(manifest_path: str | Path) -> dict[str, Any]:
    manifest = load_predicted_dsg_detector_run_manifest(manifest_path)
    preflight = predicted_dsg_detector_run_manifest_preflight(manifest_path)
    artifact_contract = _mapping(preflight.get("artifact_contract"), "artifact_contract")
    detector_input = _mapping(artifact_contract.get("detector_input"), "detector_input")
    commands = _artifact_launch_next_commands(
        contract_path=None,
        manifest_path=str(manifest_path),
    )
    readiness = _mapping(artifact_contract.get("readiness"), "readiness")
    summary = _mapping(preflight.get("summary"), "summary")
    bundle: dict[str, Any] = {
        "schema_version": PREDICTED_DSG_DETECTOR_RECEIPT_BUNDLE_SCHEMA_VERSION,
        "action": "predicted_dsg_detector_receipt_bundle",
        "manifest_path": str(manifest_path),
        "manifest_digest": predicted_dsg_detector_run_manifest_digest(manifest),
        "ready_to_build": preflight.get("ready_to_build") is True,
        "detector_jsonl": {
            "expected_schema_version": _required_text(
                detector_input,
                "expected_schema_version",
            ),
            "input_digest": detector_input.get("input_digest"),
            "input_format": "detector_observation_jsonl",
            "object_observation_count": _int_value(
                detector_input,
                "object_observation_count",
            ),
            "observation_count": _int_value(detector_input, "observation_count"),
            "observation_sequence_digest": detector_input.get(
                "observation_sequence_digest"
            ),
            "path": _required_text(detector_input, "path"),
            "status": _required_text(detector_input, "status"),
        },
        "asset_summary": _json_value(artifact_contract.get("asset_summary")),
        "build_requirements": _json_value(
            artifact_contract.get("build_requirements")
        ),
        "planned_outputs": _json_value(preflight.get("planned_outputs")),
        "readiness": {
            "failed_check_count": _int_value(readiness, "failed_check_count"),
            "failed_checks": _string_list(readiness.get("failed_checks")),
            "ready": readiness.get("ready") is True,
        },
        "summary": {
            "asset_summary": _json_value(summary.get("asset_summary")),
            "detector_import_summary": _json_value(
                summary.get("detector_import_summary")
            ),
            "evidence_summary": _json_value(summary.get("evidence_summary")),
            "object_observation_count": _int_value(
                summary,
                "object_observation_count",
            ),
            "observation_count": _int_value(summary, "observation_count"),
            "ready_to_build": preflight.get("ready_to_build") is True,
        },
        "commands": {
            "build": _required_text(commands, "build"),
            "detector_request_bundle": _required_text(
                commands,
                "detector_request_bundle",
            ),
            "preflight": _required_text(commands, "preflight"),
        },
    }
    bundle["receipt_bundle_digest"] = predicted_dsg_detector_receipt_bundle_digest(
        bundle
    )
    return bundle


def validate_predicted_dsg_detector_receipt_bundle(
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = bundle.get("schema_version")
    action = bundle.get("action")
    receipt_digest = _string_or_none(bundle.get("receipt_bundle_digest"))
    expected_digest = predicted_dsg_detector_receipt_bundle_digest(bundle)
    detector_jsonl = _mapping(bundle.get("detector_jsonl"), "detector_jsonl")
    readiness = _mapping(bundle.get("readiness"), "readiness")
    summary = _mapping(bundle.get("summary"), "summary")
    failed_checks = _string_list(readiness.get("failed_checks"))
    expected_summary = {
        "asset_summary": _json_value(bundle.get("asset_summary")),
        "detector_import_summary": _json_value(
            summary.get("detector_import_summary")
        ),
        "evidence_summary": _json_value(summary.get("evidence_summary")),
        "object_observation_count": _int_value(
            detector_jsonl,
            "object_observation_count",
        ),
        "observation_count": _int_value(detector_jsonl, "observation_count"),
        "ready_to_build": bundle.get("ready_to_build") is True,
    }
    detector_row_valid = (
        _string_or_none(detector_jsonl.get("path")) is not None
        and _string_or_none(detector_jsonl.get("status")) is not None
        and _string_or_none(detector_jsonl.get("input_format")) is not None
    )
    checks = [
        {
            "name": "schema_version",
            "passed": (
                schema_version
                == PREDICTED_DSG_DETECTOR_RECEIPT_BUNDLE_SCHEMA_VERSION
            ),
            "expected": PREDICTED_DSG_DETECTOR_RECEIPT_BUNDLE_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "action",
            "passed": action == "predicted_dsg_detector_receipt_bundle",
            "expected": "predicted_dsg_detector_receipt_bundle",
            "actual": action,
        },
        {
            "name": "receipt_bundle_digest",
            "passed": receipt_digest == expected_digest,
            "expected": expected_digest,
            "actual": receipt_digest,
        },
        {
            "name": "manifest_path",
            "passed": _string_or_none(bundle.get("manifest_path")) is not None,
            "expected": "present",
            "actual": _string_or_none(bundle.get("manifest_path")),
        },
        {
            "name": "manifest_digest",
            "passed": _string_or_none(bundle.get("manifest_digest")) is not None,
            "expected": "present",
            "actual": _string_or_none(bundle.get("manifest_digest")),
        },
        {
            "name": "summary",
            "passed": _json_value(summary) == _json_value(expected_summary),
            "expected": _json_value(expected_summary),
            "actual": _json_value(summary),
        },
        {
            "name": "readiness",
            "passed": (
                _int_value(readiness, "failed_check_count") == len(failed_checks)
                and isinstance(readiness.get("ready"), bool)
                and bundle.get("ready_to_build") is readiness.get("ready")
            ),
            "expected": {
                "failed_check_count": len(failed_checks),
                "ready_to_build": readiness.get("ready"),
            },
            "actual": {
                "failed_check_count": _int_value(readiness, "failed_check_count"),
                "ready_to_build": bundle.get("ready_to_build"),
            },
        },
        {
            "name": "detector_jsonl",
            "passed": detector_row_valid,
            "expected": True,
            "actual": detector_row_valid,
        },
    ]
    return {
        "action": "validate_predicted_dsg_detector_receipt_bundle",
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "receipt_bundle_digest": receipt_digest,
        "checks": checks,
    }


def predicted_dsg_detector_artifact_contract_digest(
    contract: Mapping[str, Any],
) -> str:
    payload = {key: value for key, value in contract.items() if key != "contract_digest"}
    return hashlib.sha256(
        json.dumps(
            _json_value(payload),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def predicted_dsg_detector_artifact_contract_json(
    contract: Mapping[str, Any],
) -> str:
    return json.dumps(_json_value(contract), indent=2, sort_keys=True) + "\n"


def save_predicted_dsg_detector_artifact_contract(
    contract: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        predicted_dsg_detector_artifact_contract_json(contract),
        encoding="utf-8",
    )
    return output_path


def load_predicted_dsg_detector_artifact_contract(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Predicted DSG detector artifact contract JSON must be an object")
    return cast(dict[str, Any], payload)


def predicted_dsg_detector_artifact_launch_report_digest(
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


def predicted_dsg_detector_artifact_launch_report_json(
    report: Mapping[str, Any],
) -> str:
    return json.dumps(_json_value(report), indent=2, sort_keys=True) + "\n"


def predicted_dsg_detector_artifact_launch_report(
    contract: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    contract_path: str | Path | None = None,
) -> dict[str, Any]:
    current_preflight: Mapping[str, Any] | None
    try:
        preflight_result = predicted_dsg_detector_run_manifest_preflight(manifest_path)
        current_preflight = preflight_result
        current_contract = _mapping(
            preflight_result.get("artifact_contract"),
            "artifact_contract",
        )
        preflight = _artifact_launch_preflight_status()
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        current_preflight = None
        current_contract = _artifact_contract_from_launch_error(
            contract,
            manifest_path=manifest_path,
            error=exc,
        )
        preflight = _artifact_launch_preflight_status(exc)
    contract_digest = _string_or_none(contract.get("contract_digest"))
    expected_digest = predicted_dsg_detector_artifact_contract_digest(contract)
    current_digest = _string_or_none(current_contract.get("contract_digest"))
    contract_valid = (
        contract.get("schema_version")
        == PREDICTED_DSG_DETECTOR_ARTIFACT_CONTRACT_SCHEMA_VERSION
        and contract_digest == expected_digest
    )
    contract_matches_current = _json_value(contract) == _json_value(current_contract)
    readiness = _mapping(current_contract.get("readiness"), "readiness")
    detector_input = _artifact_launch_detector_input(current_contract, readiness)
    summary = _artifact_launch_summary(current_contract, detector_input, readiness)
    ready_to_build = (
        contract_valid
        and contract_matches_current
        and current_preflight is not None
        and current_preflight.get("ready_to_build") is True
    )
    contract_path_text = str(contract_path) if contract_path is not None else None
    manifest_path_text = str(manifest_path)
    build_command = _artifact_launch_build_command(current_contract)
    next_commands = _artifact_launch_next_commands(
        contract_path=contract_path_text,
        manifest_path=manifest_path_text,
    )
    report: dict[str, Any] = {
        "schema_version": (
            PREDICTED_DSG_DETECTOR_ARTIFACT_LAUNCH_REPORT_SCHEMA_VERSION
        ),
        "action": "predicted_dsg_detector_artifact_launch_report",
        "contract_path": contract_path_text,
        "manifest_path": manifest_path_text,
        "contract_digest": contract_digest,
        "current_contract_digest": current_digest,
        "manifest_digest": _string_or_none(current_contract.get("manifest_digest")),
        "ready_to_build": ready_to_build,
        "preflight_ready_to_build": (
            current_preflight is not None
            and current_preflight.get("ready_to_build") is True
        ),
        "detector_input": detector_input,
        "actionable_blockers": _artifact_launch_actionable_blockers(
            detector_input,
            readiness=readiness,
            preflight=preflight,
            build_command=build_command,
            contract=current_contract,
        ),
        "build_command": build_command,
        "build_plan": _artifact_launch_build_plan(
            detector_input,
            contract=current_contract,
            build_command=build_command,
            next_commands=next_commands,
        ),
        "external_detector_intake_plan": _artifact_launch_external_detector_intake_plan(
            detector_input,
            contract=current_contract,
            preflight=preflight,
            build_command=build_command,
            next_commands=next_commands,
        ),
        "build_requirements": _json_value(current_contract.get("build_requirements")),
        "planned_outputs": _json_value(current_contract.get("planned_outputs")),
        "preflight": preflight,
        "readiness": _json_value(readiness),
        "summary": summary,
        "validation": {
            "valid": contract_valid,
            "contract_digest": contract_digest,
            "expected_contract_digest": expected_digest,
            "schema_version": contract.get("schema_version"),
        },
        "comparison": {
            "matches": contract_matches_current,
            "saved_digest": contract_digest,
            "current_digest": current_digest,
        },
        "next_commands": next_commands,
    }
    report["report_digest"] = predicted_dsg_detector_artifact_launch_report_digest(
        report
    )
    return report


def predicted_dsg_detector_run_ledger_digest(ledger: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in ledger.items() if key != "ledger_digest"}
    return hashlib.sha256(
        json.dumps(
            _json_value(payload),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def predicted_dsg_detector_run_ledger_json(ledger: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(ledger), indent=2, sort_keys=True) + "\n"


def save_predicted_dsg_detector_run_ledger(
    ledger: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        predicted_dsg_detector_run_ledger_json(ledger),
        encoding="utf-8",
    )
    return output_path


def load_predicted_dsg_detector_run_ledger(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Predicted DSG detector run ledger JSON must be an object")
    return cast(dict[str, Any], payload)


def predicted_dsg_detector_run_ledger(result: Mapping[str, Any]) -> dict[str, Any]:
    ledger: dict[str, Any] = {
        "schema_version": PREDICTED_DSG_DETECTOR_RUN_LEDGER_SCHEMA_VERSION,
        "run": {
            "detector_import_report_digest": result.get(
                "detector_import_report_digest"
            ),
            "detector_import_report_path": result.get("detector_import_report_path"),
            "detector_jsonl_path": result.get("detector_jsonl_path"),
            "graph_digest": result.get("graph_digest"),
            "graph_path": result.get("graph_path"),
            "input_digest": result.get("input_digest"),
            "manifest_digest": result.get("manifest_digest"),
            "manifest_path": result.get("manifest_path"),
            "observation_sequence_digest": result.get("observation_sequence_digest"),
            "observation_sequence_path": result.get("observation_sequence_path"),
            "predicted_dsg_evidence_report_digest": result.get(
                "predicted_dsg_evidence_report_digest"
            ),
            "predicted_dsg_evidence_report_path": result.get(
                "predicted_dsg_evidence_report_path"
            ),
            "predicted_graph_report_digest": result.get(
                "predicted_graph_report_digest"
            ),
            "predicted_graph_report_path": result.get(
                "predicted_graph_report_path"
            ),
            "ready": result.get("ready") is True,
            "schema_version": result.get("schema_version"),
        },
        "readiness": _json_value(result.get("readiness")),
        "summary": {
            "detector_import_summary": _json_value(
                _mapping(result.get("summary"), "summary").get(
                    "detector_import_summary"
                )
            ),
            "evidence_summary": _json_value(
                _mapping(result.get("summary"), "summary").get("evidence_summary")
            ),
            "predicted_graph_summary": _json_value(
                _mapping(result.get("summary"), "summary").get(
                    "predicted_graph_summary"
                )
            ),
        },
    }
    ledger["ledger_digest"] = predicted_dsg_detector_run_ledger_digest(ledger)
    return ledger


def validate_predicted_dsg_detector_run_ledger(
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = ledger.get("schema_version")
    ledger_digest = _string_or_none(ledger.get("ledger_digest"))
    expected_digest = predicted_dsg_detector_run_ledger_digest(ledger)
    run = ledger.get("run")
    readiness = ledger.get("readiness")
    required_paths = (
        "detector_import_report_path",
        "detector_jsonl_path",
        "graph_path",
        "observation_sequence_path",
        "predicted_dsg_evidence_report_path",
        "predicted_graph_report_path",
    )
    path_count = sum(
        1
        for key in required_paths
        if isinstance(run, Mapping) and isinstance(run.get(key), str)
    )
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == PREDICTED_DSG_DETECTOR_RUN_LEDGER_SCHEMA_VERSION,
            "expected": PREDICTED_DSG_DETECTOR_RUN_LEDGER_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "ledger_digest",
            "passed": ledger_digest == expected_digest,
            "expected": expected_digest,
            "actual": ledger_digest,
        },
        {
            "name": "run_paths_present",
            "passed": path_count == len(required_paths),
            "expected": len(required_paths),
            "actual": path_count,
        },
        {
            "name": "readiness_ready_matches_run",
            "passed": (
                isinstance(run, Mapping)
                and isinstance(readiness, Mapping)
                and readiness.get("ready") is run.get("ready")
            ),
            "expected": run.get("ready") if isinstance(run, Mapping) else None,
            "actual": (
                readiness.get("ready")
                if isinstance(readiness, Mapping)
                else None
            ),
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "ledger_digest": ledger_digest,
        "checks": checks,
    }


def compare_predicted_dsg_detector_run_ledger(
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_predicted_dsg_detector_run_ledger(ledger)
    run = _mapping(ledger.get("run"), "run")
    checks = [
        {
            "name": "ledger_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        _digest_check(
            "manifest_digest_matches_current",
            run.get("manifest_digest"),
            _manifest_digest_or_none(run.get("manifest_path")),
        ),
        _digest_check(
            "detector_input_digest_matches_current",
            run.get("input_digest"),
            _detector_jsonl_digest_or_none(run.get("detector_jsonl_path")),
        ),
        _digest_check(
            "observation_sequence_digest_matches_current",
            run.get("observation_sequence_digest"),
            _observation_sequence_digest_or_none(
                run.get("observation_sequence_path")
            ),
        ),
        _digest_check(
            "graph_digest_matches_current",
            run.get("graph_digest"),
            _graph_digest_or_none(run.get("graph_path")),
        ),
        _digest_check(
            "detector_import_report_digest_matches_current",
            run.get("detector_import_report_digest"),
            _detector_import_report_digest_or_none(
                run.get("detector_import_report_path")
            ),
        ),
        _digest_check(
            "predicted_graph_report_digest_matches_current",
            run.get("predicted_graph_report_digest"),
            _predicted_graph_report_digest_or_none(
                run.get("predicted_graph_report_path")
            ),
        ),
        _digest_check(
            "predicted_dsg_evidence_report_digest_matches_current",
            run.get("predicted_dsg_evidence_report_digest"),
            _predicted_dsg_evidence_report_digest_or_none(
                run.get("predicted_dsg_evidence_report_path")
            ),
        ),
    ]
    detector_import_report = load_detector_observation_import_report(
        _required_text(run, "detector_import_report_path")
    )
    predicted_report = load_predicted_graph_report(
        _required_text(run, "predicted_graph_report_path")
    )
    evidence_report = load_predicted_dsg_evidence_report(
        _required_text(run, "predicted_dsg_evidence_report_path")
    )
    checks.extend(
        [
            _digest_check(
                "detector_import_input_digest_matches_run",
                run.get("input_digest"),
                detector_import_report.get("input_digest"),
            ),
            _digest_check(
                "detector_import_output_sequence_digest_matches_run",
                run.get("observation_sequence_digest"),
                detector_import_report.get("sequence_digest"),
            ),
            _digest_check(
                "predicted_graph_report_graph_digest_matches_run",
                run.get("graph_digest"),
                _mapping(predicted_report.get("graph_report"), "graph_report").get(
                    "digest"
                ),
            ),
            _digest_check(
                "predicted_graph_report_observation_digest_matches_run",
                run.get("observation_sequence_digest"),
                predicted_report.get("observation_sequence_digest"),
            ),
            _digest_check(
                "evidence_report_predicted_graph_digest_matches_run",
                run.get("predicted_graph_report_digest"),
                evidence_report.get("predicted_graph_report_digest"),
            ),
        ]
    )
    return {
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": _string_or_none(ledger.get("ledger_digest")),
        "validation": validation,
        "checks": checks,
    }


def predicted_dsg_detector_run_manifest_preflight(
    manifest_path: str | Path,
) -> dict[str, Any]:
    manifest = load_predicted_dsg_detector_run_manifest(manifest_path)
    detector_payload = Path(manifest["detector_jsonl_path"]).read_text(encoding="utf-8")
    observations = detector_observation_sequence_from_jsonl(detector_payload)
    detector_import = detector_observation_import_report(
        input_path=manifest["detector_jsonl_path"],
        output_sequence_path=manifest["output_sequence_path"],
        observations=observations,
        input_payload=detector_payload,
    )
    graph = build_predicted_graph_from_observations(
        observations,
        source_path=manifest["output_sequence_path"],
        infer_relations=manifest["infer_relations"],
        reference_frames=manifest["reference_frames"],
        infer_containment=manifest["infer_containment"],
        containment_axis=manifest["containment_axis"],
        relation_top_k=manifest["relation_top_k"],
        require_detector_state_evidence=manifest["require_detector_state_evidence"],
    )
    predicted_report = predicted_graph_report_from_observations(
        input_path=manifest["output_sequence_path"],
        graph_path=manifest["output_graph_path"],
        graph=graph,
        observations=observations,
        infer_relations=manifest["infer_relations"],
        reference_frames=manifest["reference_frames"],
        infer_containment=manifest["infer_containment"],
        containment_axis=manifest["containment_axis"],
        relation_top_k=manifest["relation_top_k"],
        require_detector_state_evidence=manifest["require_detector_state_evidence"],
    )
    evidence_summary = _preflight_evidence_summary(predicted_report, observations)
    asset_summary = _preflight_asset_summary(
        detector_payload,
        detector_jsonl_path=manifest["detector_jsonl_path"],
    )
    checks = _preflight_evidence_checks(
        predicted_report,
        observations=observations,
        asset_summary=asset_summary,
        evidence_summary=evidence_summary,
        min_observation_count=manifest["min_observation_count"],
        min_object_observation_count=manifest["min_object_observation_count"],
        required_evidence_kinds=manifest["required_evidence_kinds"],
    )
    readiness = _readiness_from_checks(checks)
    planned_outputs = {
        "detector_import_report_path": manifest["detector_import_report_path"],
        "observation_sequence_path": manifest["output_sequence_path"],
        "predicted_dsg_evidence_report_path": manifest[
            "predicted_dsg_evidence_report_path"
        ],
        "predicted_graph_path": manifest["output_graph_path"],
        "predicted_graph_report_path": manifest["predicted_graph_report_path"],
    }
    artifact_contract = _artifact_contract(
        manifest=manifest,
        manifest_path=manifest_path,
        detector_import=detector_import,
        observations=observations,
        asset_summary=asset_summary,
        evidence_summary=evidence_summary,
        planned_outputs=planned_outputs,
        readiness=readiness,
    )
    return {
        "schema_version": PREDICTED_DSG_DETECTOR_PREFLIGHT_SCHEMA_VERSION,
        "action": "predicted_dsg_detector_run_manifest_preflight",
        "manifest_schema_version": manifest["schema_version"],
        "manifest_path": str(manifest_path),
        "manifest_digest": predicted_dsg_detector_run_manifest_digest(manifest),
        "ready_to_build": readiness["ready"],
        "detector_jsonl_path": manifest["detector_jsonl_path"],
        "input_digest": detector_import["input_digest"],
        "observation_sequence_digest": scene_observation_sequence_digest(
            observations
        ),
        "graph_digest": graph_json_digest(graph),
        "detector_import_report_digest": detector_import["digest"],
        "predicted_graph_report_digest": predicted_report["digest"],
        "artifact_contract": artifact_contract,
        "planned_outputs": planned_outputs,
        "readiness": {
            **readiness,
            "checks": checks,
        },
        "summary": {
            "asset_summary": asset_summary,
            "detector_import_summary": detector_import["summary"],
            "evidence_summary": evidence_summary,
            "object_observation_count": evidence_summary[
                "object_observation_count"
            ],
            "observation_count": evidence_summary["observation_count"],
            "predicted_graph_summary": predicted_report["summary"],
        },
    }


def run_predicted_dsg_detector_run_manifest(manifest_path: str | Path) -> dict[str, Any]:
    manifest = load_predicted_dsg_detector_run_manifest(manifest_path)
    result = run_predicted_dsg_from_detector_jsonl(
        detector_jsonl_path=manifest["detector_jsonl_path"],
        output_sequence_path=manifest["output_sequence_path"],
        output_graph_path=manifest["output_graph_path"],
        predicted_graph_report_path=manifest["predicted_graph_report_path"],
        detector_import_report_path=manifest["detector_import_report_path"],
        predicted_dsg_evidence_report_path=manifest[
            "predicted_dsg_evidence_report_path"
        ],
        infer_relations=manifest["infer_relations"],
        reference_frames=manifest["reference_frames"],
        infer_containment=manifest["infer_containment"],
        containment_axis=manifest["containment_axis"],
        relation_top_k=manifest["relation_top_k"],
        require_detector_state_evidence=manifest["require_detector_state_evidence"],
        min_observation_count=manifest["min_observation_count"],
        min_object_observation_count=manifest["min_object_observation_count"],
        required_evidence_kinds=manifest["required_evidence_kinds"],
    )
    return {
        **result,
        "action": "run_predicted_dsg_detector_run_manifest",
        "manifest_schema_version": manifest["schema_version"],
        "manifest_path": str(manifest_path),
        "manifest_digest": predicted_dsg_detector_run_manifest_digest(manifest),
    }


def run_predicted_dsg_from_detector_jsonl(
    *,
    detector_jsonl_path: str | Path,
    output_sequence_path: str | Path,
    output_graph_path: str | Path,
    predicted_graph_report_path: str | Path,
    detector_import_report_path: str | Path,
    predicted_dsg_evidence_report_path: str | Path,
    infer_relations: Sequence[str] = OBSERVATION_PREDICTED_RELATIONS,
    reference_frames: Sequence[str] = OBSERVATION_PREDICTED_REFERENCE_FRAMES,
    infer_containment: bool = False,
    containment_axis: str = "z",
    relation_top_k: int | None = None,
    require_detector_state_evidence: bool = False,
    min_observation_count: int = 2,
    min_object_observation_count: int = 2,
    required_evidence_kinds: Sequence[str] = (
        DEFAULT_REQUIRED_PREDICTED_DSG_EVIDENCE_KINDS
    ),
) -> dict[str, Any]:
    detector_payload = Path(detector_jsonl_path).read_text(encoding="utf-8")
    observations = detector_observation_sequence_from_jsonl(detector_payload)
    save_scene_observation_sequence(observations, output_sequence_path)
    detector_import = detector_observation_import_report(
        input_path=detector_jsonl_path,
        output_sequence_path=output_sequence_path,
        observations=observations,
        input_payload=detector_payload,
    )
    save_detector_observation_import_report(
        detector_import,
        detector_import_report_path,
    )

    graph = build_predicted_graph_from_observations(
        observations,
        source_path=output_sequence_path,
        infer_relations=infer_relations,
        reference_frames=reference_frames,
        infer_containment=infer_containment,
        containment_axis=containment_axis,
        relation_top_k=relation_top_k,
        require_detector_state_evidence=require_detector_state_evidence,
    )
    save_graph_json(graph, output_graph_path)
    predicted_report = predicted_graph_report_from_observations(
        input_path=output_sequence_path,
        graph_path=output_graph_path,
        graph=graph,
        observations=observations,
        infer_relations=infer_relations,
        reference_frames=reference_frames,
        infer_containment=infer_containment,
        containment_axis=containment_axis,
        relation_top_k=relation_top_k,
        require_detector_state_evidence=require_detector_state_evidence,
    )
    save_predicted_graph_report(predicted_report, predicted_graph_report_path)
    evidence_report = predicted_dsg_evidence_report(
        predicted_report,
        predicted_graph_report_path=predicted_graph_report_path,
        observation_sequence_path=output_sequence_path,
        min_observation_count=min_observation_count,
        min_object_observation_count=min_object_observation_count,
        required_evidence_kinds=required_evidence_kinds,
    )
    save_predicted_dsg_evidence_report(
        evidence_report,
        predicted_dsg_evidence_report_path,
    )

    return {
        "schema_version": PREDICTED_DSG_DETECTOR_RUN_SCHEMA_VERSION,
        "action": "run_predicted_dsg_from_detector_jsonl",
        "detector_jsonl_path": str(detector_jsonl_path),
        "observation_sequence_path": str(output_sequence_path),
        "graph_path": str(output_graph_path),
        "predicted_graph_report_path": str(predicted_graph_report_path),
        "detector_import_report_path": str(detector_import_report_path),
        "predicted_dsg_evidence_report_path": str(
            predicted_dsg_evidence_report_path
        ),
        "input_digest": detector_import["input_digest"],
        "observation_sequence_digest": scene_observation_sequence_digest(
            observations
        ),
        "graph_digest": graph_json_digest(graph),
        "detector_import_report_digest": detector_import["digest"],
        "predicted_graph_report_digest": predicted_report["digest"],
        "predicted_dsg_evidence_report_digest": evidence_report["report_digest"],
        "ready": evidence_report["readiness"]["ready"],
        "readiness": evidence_report["readiness"],
        "summary": {
            "detector_import_summary": detector_import["summary"],
            "predicted_graph_summary": predicted_report["summary"],
            "evidence_summary": evidence_report["evidence_summary"],
        },
    }


def _predicted_dsg_detector_run_manifest(
    payload: dict[str, Any],
    base_dir: Path,
) -> dict[str, Any]:
    schema_version = _required_str(payload, "schema_version")
    if schema_version != PREDICTED_DSG_DETECTOR_RUN_MANIFEST_SCHEMA_VERSION:
        raise SpatialQAError(
            "Unsupported predicted DSG detector run manifest schema version: "
            f"{schema_version}"
        )
    return {
        "schema_version": schema_version,
        "detector_jsonl_path": str(
            _manifest_required_path(payload, "detector_jsonl_path", base_dir)
        ),
        "output_sequence_path": str(
            _manifest_required_path(payload, "output_sequence_path", base_dir)
        ),
        "output_graph_path": str(
            _manifest_required_path(payload, "output_graph_path", base_dir)
        ),
        "predicted_graph_report_path": str(
            _manifest_required_path(
                payload,
                "predicted_graph_report_path",
                base_dir,
            )
        ),
        "detector_import_report_path": str(
            _manifest_required_path(
                payload,
                "detector_import_report_path",
                base_dir,
            )
        ),
        "predicted_dsg_evidence_report_path": str(
            _manifest_required_path(
                payload,
                "predicted_dsg_evidence_report_path",
                base_dir,
            )
        ),
        "infer_relations": list(
            _optional_string_sequence(
                payload.get("infer_relations"),
                OBSERVATION_PREDICTED_RELATIONS,
                "infer_relations",
            )
        ),
        "reference_frames": list(
            _optional_string_sequence(
                payload.get("reference_frames"),
                OBSERVATION_PREDICTED_REFERENCE_FRAMES,
                "reference_frames",
            )
        ),
        "infer_containment": _optional_bool(
            payload.get("infer_containment"),
            False,
            "infer_containment",
        ),
        "containment_axis": _optional_containment_axis(
            payload.get("containment_axis"),
            "containment_axis",
        ),
        "relation_top_k": _optional_int_or_none(
            payload.get("relation_top_k"),
            "relation_top_k",
        ),
        "require_detector_state_evidence": _optional_bool(
            payload.get("require_detector_state_evidence"),
            False,
            "require_detector_state_evidence",
        ),
        "min_observation_count": _optional_int(
            payload.get("min_observation_count"),
            2,
            "min_observation_count",
        ),
        "min_object_observation_count": _optional_int(
            payload.get("min_object_observation_count"),
            2,
            "min_object_observation_count",
        ),
        "required_evidence_kinds": list(
            _optional_string_sequence(
                payload.get("required_evidence_kinds"),
                DEFAULT_REQUIRED_PREDICTED_DSG_EVIDENCE_KINDS,
                "required_evidence_kinds",
            )
        ),
    }


def _preflight_evidence_summary(
    predicted_graph_report: Mapping[str, Any],
    observations: Sequence[Any],
) -> dict[str, Any]:
    objects = tuple(
        obj
        for observation in observations
        for obj in getattr(observation, "objects", ())
    )
    return {
        "evidence_kind_counts": _evidence_kind_counts(objects),
        "input_kind": predicted_graph_report.get("input_kind"),
        "object_observation_count": len(objects),
        "observation_count": len(observations),
        "observation_sequence_digest": scene_observation_sequence_digest(
            observations
        ),
        "source_counts": _source_counts(objects),
    }


def _request_bundle_planned_outputs(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "detector_import_report_path": manifest["detector_import_report_path"],
        "observation_sequence_path": manifest["output_sequence_path"],
        "predicted_dsg_evidence_report_path": manifest[
            "predicted_dsg_evidence_report_path"
        ],
        "predicted_graph_path": manifest["output_graph_path"],
        "predicted_graph_report_path": manifest["predicted_graph_report_path"],
    }


def _request_bundle_build_requirements(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "infer_relations": list(
            _string_sequence_from_mapping(manifest, "infer_relations")
        ),
        "min_object_observation_count": _int_value(
            manifest,
            "min_object_observation_count",
        ),
        "min_observation_count": _int_value(manifest, "min_observation_count"),
        "reference_frames": list(
            _string_sequence_from_mapping(manifest, "reference_frames")
        ),
        "infer_containment": manifest.get("infer_containment") is True,
        "containment_axis": _required_text(manifest, "containment_axis"),
        "relation_top_k": manifest.get("relation_top_k"),
        "require_detector_state_evidence": (
            manifest.get("require_detector_state_evidence") is True
        ),
        "required_evidence_kinds": list(
            _string_sequence_from_mapping(manifest, "required_evidence_kinds")
        ),
    }


def _detector_record_template() -> dict[str, Any]:
    pose = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
    return {
        "schema_version": DETECTOR_OBSERVATION_RECORD_SCHEMA_VERSION,
        "step": 0,
        "agent_id": "agent",
        "agent_pose": dict(pose),
        "rgb_path": "frames/000001.rgb.png",
        "depth_path": "frames/000001.depth.png",
        "segmentation_path": "frames/000001.seg.png",
        "metadata": {},
        "detections": [
            {
                "object_id": "track_object_1",
                "label": "object",
                "pose": dict(pose),
                "bbox": {
                    "center": dict(pose),
                    "size": [0.0, 0.0, 0.0],
                },
                "confidence": 0.0,
                "visible": True,
                "attributes": {},
            }
        ],
    }


def _artifact_contract(
    *,
    manifest: Mapping[str, Any],
    manifest_path: str | Path,
    detector_import: Mapping[str, Any],
    observations: Sequence[Any],
    asset_summary: Mapping[str, Any],
    evidence_summary: Mapping[str, Any],
    planned_outputs: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "schema_version": PREDICTED_DSG_DETECTOR_ARTIFACT_CONTRACT_SCHEMA_VERSION,
        "manifest_path": str(manifest_path),
        "manifest_digest": predicted_dsg_detector_run_manifest_digest(
            cast(dict[str, Any], dict(manifest))
        ),
        "detector_input": {
            "expected_schema_version": DETECTOR_OBSERVATION_RECORD_SCHEMA_VERSION,
            "input_digest": detector_import["input_digest"],
            "object_observation_count": _int_value(
                evidence_summary,
                "object_observation_count",
            ),
            "observation_count": len(observations),
            "observation_sequence_digest": scene_observation_sequence_digest(
                observations
            ),
            "path": manifest["detector_jsonl_path"],
            "status": "ready",
        },
        "asset_summary": _json_value(asset_summary),
        "build_requirements": {
            "infer_relations": list(
                _string_sequence_from_mapping(manifest, "infer_relations")
            ),
            "min_object_observation_count": _int_value(
                manifest,
                "min_object_observation_count",
            ),
            "min_observation_count": _int_value(manifest, "min_observation_count"),
            "reference_frames": list(
                _string_sequence_from_mapping(manifest, "reference_frames")
            ),
            "infer_containment": manifest.get("infer_containment") is True,
            "containment_axis": _required_text(manifest, "containment_axis"),
            "relation_top_k": manifest.get("relation_top_k"),
            "require_detector_state_evidence": (
                manifest.get("require_detector_state_evidence") is True
            ),
            "required_evidence_kinds": list(
                _string_sequence_from_mapping(manifest, "required_evidence_kinds")
            ),
        },
        "planned_outputs": _json_value(planned_outputs),
        "readiness": {
            "failed_check_count": _int_value(readiness, "failed_check_count"),
            "failed_checks": _string_list(readiness.get("failed_checks")),
            "ready": readiness.get("ready") is True,
        },
        "summary": {
            "asset_summary": _json_value(asset_summary),
            "evidence_kind_counts": _int_mapping(
                evidence_summary.get("evidence_kind_counts")
            ),
            "object_observation_count": _int_value(
                evidence_summary,
                "object_observation_count",
            ),
            "observation_count": _int_value(evidence_summary, "observation_count"),
        },
    }
    contract["contract_digest"] = predicted_dsg_detector_artifact_contract_digest(
        contract
    )
    return contract


def _artifact_contract_from_launch_error(
    contract: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    error: Exception,
) -> dict[str, Any]:
    manifest = _load_predicted_dsg_detector_run_manifest_or_none(manifest_path)
    saved_detector_input = _optional_mapping(contract.get("detector_input"))
    status = _artifact_launch_detector_input_status(error)
    failed_check = f"detector_input_{status}"
    detector_path = _artifact_launch_detector_input_path(
        manifest=manifest,
        saved_detector_input=saved_detector_input,
        manifest_path=manifest_path,
    )
    current_contract: dict[str, Any] = {
        "schema_version": PREDICTED_DSG_DETECTOR_ARTIFACT_CONTRACT_SCHEMA_VERSION,
        "manifest_path": str(manifest_path),
        "manifest_digest": _artifact_launch_manifest_digest(manifest, contract),
        "detector_input": {
            "expected_schema_version": DETECTOR_OBSERVATION_RECORD_SCHEMA_VERSION,
            "input_digest": None,
            "object_observation_count": 0,
            "observation_count": 0,
            "observation_sequence_digest": None,
            "path": detector_path,
            "status": status,
        },
        "build_requirements": _artifact_launch_build_requirements(
            manifest,
            contract,
        ),
        "planned_outputs": _artifact_launch_planned_outputs(manifest, contract),
        "readiness": {
            "failed_check_count": 1,
            "failed_checks": [failed_check],
            "ready": False,
        },
        "summary": {
            "evidence_kind_counts": {"depth": 0, "detector": 0, "rgb": 0},
            "object_observation_count": 0,
            "observation_count": 0,
        },
    }
    current_contract["contract_digest"] = (
        predicted_dsg_detector_artifact_contract_digest(current_contract)
    )
    return current_contract


def _artifact_launch_preflight_status(
    error: Exception | None = None,
) -> dict[str, Any]:
    if error is None:
        return {
            "available": True,
            "error": None,
            "error_type": None,
            "status": "ready",
        }
    return {
        "available": False,
        "error": str(error),
        "error_type": type(error).__name__,
        "status": "failed",
    }


def _artifact_launch_detector_input_status(error: Exception) -> str:
    if isinstance(error, FileNotFoundError):
        return "missing"
    return "invalid"


def _artifact_launch_detector_input_path(
    *,
    manifest: Mapping[str, Any] | None,
    saved_detector_input: Mapping[str, Any],
    manifest_path: str | Path,
) -> str:
    if manifest is not None:
        return _required_text(manifest, "detector_jsonl_path")
    saved_path = _string_or_none(saved_detector_input.get("path"))
    if saved_path is not None:
        return saved_path
    return str(manifest_path)


def _artifact_launch_manifest_digest(
    manifest: Mapping[str, Any] | None,
    contract: Mapping[str, Any],
) -> str | None:
    if manifest is not None:
        return predicted_dsg_detector_run_manifest_digest(
            cast(dict[str, Any], dict(manifest))
        )
    return _string_or_none(contract.get("manifest_digest"))


def _artifact_launch_build_requirements(
    manifest: Mapping[str, Any] | None,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if manifest is None:
        return dict(_optional_mapping(contract.get("build_requirements")))
    return {
        "infer_relations": list(_string_sequence_from_mapping(manifest, "infer_relations")),
        "min_object_observation_count": _int_value(
            manifest,
            "min_object_observation_count",
        ),
        "min_observation_count": _int_value(manifest, "min_observation_count"),
        "reference_frames": list(
            _string_sequence_from_mapping(manifest, "reference_frames")
        ),
        "infer_containment": manifest.get("infer_containment") is True,
        "containment_axis": _required_text(manifest, "containment_axis"),
        "relation_top_k": manifest.get("relation_top_k"),
        "require_detector_state_evidence": (
            manifest.get("require_detector_state_evidence") is True
        ),
        "required_evidence_kinds": list(
            _string_sequence_from_mapping(manifest, "required_evidence_kinds")
        ),
    }


def _artifact_launch_planned_outputs(
    manifest: Mapping[str, Any] | None,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if manifest is None:
        return dict(_optional_mapping(contract.get("planned_outputs")))
    return {
        "detector_import_report_path": manifest["detector_import_report_path"],
        "observation_sequence_path": manifest["output_sequence_path"],
        "predicted_dsg_evidence_report_path": manifest[
            "predicted_dsg_evidence_report_path"
        ],
        "predicted_graph_path": manifest["output_graph_path"],
        "predicted_graph_report_path": manifest["predicted_graph_report_path"],
    }


def _load_predicted_dsg_detector_run_manifest_or_none(
    manifest_path: str | Path,
) -> dict[str, Any] | None:
    try:
        return load_predicted_dsg_detector_run_manifest(manifest_path)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError):
        return None


def _artifact_launch_detector_input(
    contract: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    detector_input = _mapping(contract.get("detector_input"), "detector_input")
    blocking_reasons = _artifact_launch_blocking_reasons(detector_input, readiness)
    return {
        "blocking_reasons": blocking_reasons,
        "object_observation_count": _int_value(
            detector_input,
            "object_observation_count",
        ),
        "observation_count": _int_value(detector_input, "observation_count"),
        "path": _required_text(detector_input, "path"),
        "status": _required_text(detector_input, "status"),
    }


def _artifact_launch_blocking_reasons(
    detector_input: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    status = _string_or_none(detector_input.get("status"))
    if status == "missing":
        reasons.append("detector_input_missing")
    elif status == "invalid":
        reasons.append("detector_input_invalid")
    elif status != "ready":
        reasons.append("detector_input_not_ready")
    if readiness.get("ready") is not True:
        reasons.append("readiness_checks_failed")
    return sorted(reasons)


def _artifact_launch_summary(
    contract: Mapping[str, Any],
    detector_input: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    planned_outputs = _mapping(contract.get("planned_outputs"), "planned_outputs")
    return {
        "blocking_reason_count": len(_string_list(detector_input.get("blocking_reasons"))),
        "failed_check_count": _int_value(readiness, "failed_check_count"),
        "object_observation_count": _int_value(
            detector_input,
            "object_observation_count",
        ),
        "observation_count": _int_value(detector_input, "observation_count"),
        "planned_output_count": len(planned_outputs),
        "ready_to_build": readiness.get("ready") is True
        and not _string_list(detector_input.get("blocking_reasons")),
    }


def _artifact_launch_actionable_blockers(
    detector_input: Mapping[str, Any],
    *,
    readiness: Mapping[str, Any],
    preflight: Mapping[str, Any],
    build_command: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    blockers: dict[str, Any] = {}
    status = _required_text(detector_input, "status")
    if status != "ready":
        blockers["detector_input"] = {
            "blocking_reasons": _string_list(
                detector_input.get("blocking_reasons")
            ),
            "build_command": build_command,
            "path": _required_text(detector_input, "path"),
            "preflight": _json_value(preflight),
            "status": status,
        }
    failed_checks = _string_list(readiness.get("failed_checks"))
    if failed_checks:
        blockers["build_readiness"] = {
            "build_command": build_command,
            "detector_input": _json_value(detector_input),
            "failed_check_count": _int_value(readiness, "failed_check_count"),
            "failed_checks": failed_checks,
            "requirements": _json_value(contract.get("build_requirements")),
        }
    return blockers


def _artifact_launch_build_plan(
    detector_input: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    build_command: str,
    next_commands: Mapping[str, str],
) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "track": "predicted_dsg",
        "build_command": build_command,
        "detector_input": {
            "blocking_reasons": _string_list(
                detector_input.get("blocking_reasons")
            ),
            "object_observation_count": _int_value(
                detector_input,
                "object_observation_count",
            ),
            "observation_count": _int_value(detector_input, "observation_count"),
            "path": _required_text(detector_input, "path"),
            "ready_to_build": not _string_list(
                detector_input.get("blocking_reasons")
            ),
            "status": _required_text(detector_input, "status"),
        },
        "manifest_build_command": _required_text(next_commands, "build"),
        "planned_outputs": _json_value(contract.get("planned_outputs")),
        "preflight_command": _required_text(next_commands, "preflight"),
        "requirements": _json_value(contract.get("build_requirements")),
    }
    asset_summary = _optional_mapping(contract.get("asset_summary"))
    if asset_summary:
        plan["asset_summary"] = _json_value(asset_summary)
    return plan


def _artifact_launch_external_detector_intake_plan(
    detector_input: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    preflight: Mapping[str, Any],
    build_command: str,
    next_commands: Mapping[str, str],
) -> dict[str, Any]:
    detector_input_contract = _mapping(
        contract.get("detector_input"),
        "detector_input",
    )
    blocking_reasons = _string_list(detector_input.get("blocking_reasons"))
    plan: dict[str, Any] = {
        "track": "predicted_dsg",
        "blocked": bool(blocking_reasons),
        "blocking_reasons": blocking_reasons,
        "build_command": build_command,
        "detector_request_bundle_command": _required_text(
            next_commands,
            "detector_request_bundle",
        ),
        "detector_receipt_bundle_command": _required_text(
            next_commands,
            "detector_receipt_bundle",
        ),
        "detector_input": {
            "expected_schema_version": _required_text(
                detector_input_contract,
                "expected_schema_version",
            ),
            "object_observation_count": _int_value(
                detector_input,
                "object_observation_count",
            ),
            "observation_count": _int_value(detector_input, "observation_count"),
            "path": _required_text(detector_input, "path"),
            "ready_to_build": not blocking_reasons,
            "status": _required_text(detector_input, "status"),
        },
        "manifest_build_command": _required_text(next_commands, "build"),
        "planned_outputs": _json_value(contract.get("planned_outputs")),
        "preflight": _json_value(preflight),
        "preflight_command": _required_text(next_commands, "preflight"),
        "readiness": _json_value(contract.get("readiness")),
        "requirements": _json_value(contract.get("build_requirements")),
    }
    asset_summary = _optional_mapping(contract.get("asset_summary"))
    if asset_summary:
        plan["asset_summary"] = _json_value(asset_summary)
    return plan


def _artifact_launch_next_commands(
    *,
    contract_path: str | None,
    manifest_path: str,
) -> dict[str, str]:
    request_bundle_path = (
        Path(manifest_path).parent / "predicted-dsg-detector-request-bundle.json"
    )
    receipt_bundle_path = (
        Path(manifest_path).parent / "predicted-dsg-detector-receipt-bundle.json"
    )
    commands = {
        "build": f"python scripts/run_predicted_dsg.py --manifest {manifest_path}",
        "detector_request_bundle": (
            "python scripts/run_predicted_dsg.py "
            f"--detector-request-bundle {manifest_path} "
            f"--request-bundle-output {request_bundle_path}"
        ),
        "detector_receipt_bundle": (
            "python scripts/run_predicted_dsg.py "
            f"--detector-receipt-bundle {manifest_path} "
            f"--receipt-bundle-output {receipt_bundle_path}"
        ),
        "preflight": (
            "python scripts/run_predicted_dsg.py --preflight-manifest "
            f"{manifest_path}"
        ),
    }
    if contract_path is not None:
        commands["artifact_launch_report"] = (
            "python scripts/run_predicted_dsg.py "
            f"--artifact-launch-report {contract_path} --manifest {manifest_path}"
        )
    return {key: commands[key] for key in sorted(commands)}


def _artifact_launch_build_command(contract: Mapping[str, Any]) -> str:
    detector_input = _mapping(contract.get("detector_input"), "detector_input")
    planned_outputs = _mapping(contract.get("planned_outputs"), "planned_outputs")
    build_requirements = _mapping(
        contract.get("build_requirements"),
        "build_requirements",
    )
    parts = [
        "python scripts/run_predicted_dsg.py",
        f"--detector-jsonl {_required_text(detector_input, 'path')}",
        "--observation-sequence "
        f"{_required_text(planned_outputs, 'observation_sequence_path')}",
        f"--output-graph {_required_text(planned_outputs, 'predicted_graph_path')}",
        "--predicted-report "
        f"{_required_text(planned_outputs, 'predicted_graph_report_path')}",
        "--detector-import-report "
        f"{_required_text(planned_outputs, 'detector_import_report_path')}",
        "--predicted-dsg-evidence-report "
        f"{_required_text(planned_outputs, 'predicted_dsg_evidence_report_path')}",
    ]
    parts.extend(
        f"--infer-relation {relation}"
        for relation in _string_list(build_requirements.get("infer_relations"))
    )
    parts.extend(
        f"--reference-frame {frame}"
        for frame in _string_list(build_requirements.get("reference_frames"))
    )
    if build_requirements.get("infer_containment") is True:
        parts.append("--infer-containment")
        parts.append(
            f"--containment-axis {_required_text(build_requirements, 'containment_axis')}"
        )
    relation_top_k = build_requirements.get("relation_top_k")
    if isinstance(relation_top_k, int) and not isinstance(relation_top_k, bool):
        parts.append(f"--relation-top-k {relation_top_k}")
    if build_requirements.get("require_detector_state_evidence") is True:
        parts.append("--require-detector-state-evidence")
    parts.extend(
        [
            "--min-observation-count "
            f"{_int_value(build_requirements, 'min_observation_count')}",
            "--min-object-observation-count "
            f"{_int_value(build_requirements, 'min_object_observation_count')}",
        ]
    )
    parts.extend(
        f"--required-evidence-kind {kind}"
        for kind in _string_list(build_requirements.get("required_evidence_kinds"))
    )
    return " ".join(parts)


def _preflight_evidence_checks(
    predicted_graph_report: Mapping[str, Any],
    *,
    observations: Sequence[Any],
    asset_summary: Mapping[str, Any],
    evidence_summary: Mapping[str, Any],
    min_observation_count: int,
    min_object_observation_count: int,
    required_evidence_kinds: Sequence[str],
) -> list[dict[str, Any]]:
    input_kind = str(predicted_graph_report.get("input_kind"))
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
    return [
        {
            "name": "predicted_graph_report_valid",
            "passed": validate_predicted_graph_report(predicted_graph_report)[
                "valid"
            ]
            is True,
        },
        {
            "name": "input_kind_observation_sequence",
            "passed": input_kind == "observation_sequence",
            "expected": "observation_sequence",
            "actual": input_kind,
        },
        {
            "name": "observation_sequence_loads",
            "passed": True,
        },
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
            "name": "frame_assets_present",
            "passed": _int_value(asset_summary, "missing_asset_count") == 0,
            "asset_path_count": _int_value(asset_summary, "asset_path_count"),
            "missing": _json_value(asset_summary.get("missing_assets", [])),
            "missing_asset_count": _int_value(
                asset_summary,
                "missing_asset_count",
            ),
            "present_asset_count": _int_value(
                asset_summary,
                "present_asset_count",
            ),
        },
    ]


def _preflight_asset_summary(
    detector_payload: str,
    *,
    detector_jsonl_path: str | Path,
) -> dict[str, Any]:
    base_dir = Path(detector_jsonl_path).parent
    field_kinds = {
        "depth_path": "depth",
        "rgb_path": "rgb",
        "segmentation_path": "segmentation",
    }
    seen: set[tuple[str, str]] = set()
    assets: list[dict[str, Any]] = []
    for line in detector_payload.splitlines():
        if line.strip() == "":
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            continue
        for field_name, kind in field_kinds.items():
            value = payload.get(field_name)
            if not isinstance(value, str) or value == "":
                continue
            key = (kind, value)
            if key in seen:
                continue
            seen.add(key)
            local_path = Path(value)
            resolved_path = (
                local_path if local_path.is_absolute() else base_dir / local_path
            )
            assets.append(
                {
                    "kind": kind,
                    "path": value,
                    "present": resolved_path.exists(),
                    "resolved_path": str(resolved_path),
                }
            )
    asset_kind_counts = {"depth": 0, "rgb": 0, "segmentation": 0}
    missing_assets: list[dict[str, str]] = []
    for asset in assets:
        kind = str(asset["kind"])
        asset_kind_counts[kind] = asset_kind_counts.get(kind, 0) + 1
        if asset["present"] is not True:
            missing_assets.append(
                {
                    "kind": kind,
                    "path": str(asset["path"]),
                    "resolved_path": str(asset["resolved_path"]),
                }
            )
    return {
        "asset_kind_counts": {
            key: asset_kind_counts[key] for key in sorted(asset_kind_counts)
        },
        "asset_path_count": len(assets),
        "missing_asset_count": len(missing_assets),
        "missing_assets": missing_assets,
        "present_asset_count": len(assets) - len(missing_assets),
    }


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
    return any(
        marker in normalized for marker in NON_REAL_PREDICTED_DETECTOR_SOURCE_MARKERS
    )


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
    return any(
        isinstance(attributes.get(key), str) and bool(attributes.get(key))
        for key in keys
    )


def _readiness_from_checks(checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed = [
        _required_str(cast(dict[str, Any], check), "name")
        for check in checks
        if check.get("passed") is not True
    ]
    return {
        "ready": len(failed) == 0,
        "failed_check_count": len(failed),
        "failed_checks": failed,
    }


def _int_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): int(item)
        for key, item in value.items()
        if isinstance(item, int) and not isinstance(item, bool)
    }


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpatialQAError(
            f"Predicted DSG detector preflight field must be an object: {field}"
        )
    return cast(Mapping[str, Any], value)


def _string_sequence_from_mapping(
    value: Mapping[str, Any],
    key: str,
) -> tuple[str, ...]:
    item = value.get(key)
    if not isinstance(item, Sequence) or isinstance(item, str):
        return ()
    return tuple(str(entry) for entry in item if isinstance(entry, str))


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [item for item in value if isinstance(item, str)]


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    return {}


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Predicted DSG detector run ledger field is required: {key}")
    return value


def _manifest_digest_or_none(path: object) -> str | None:
    path_text = _string_or_none(path)
    if path_text is None:
        return None
    return predicted_dsg_detector_run_manifest_digest(
        load_predicted_dsg_detector_run_manifest(path_text)
    )


def _detector_jsonl_digest_or_none(path: object) -> str | None:
    path_text = _string_or_none(path)
    if path_text is None:
        return None
    return detector_observation_records_digest(Path(path_text).read_text(encoding="utf-8"))


def _observation_sequence_digest_or_none(path: object) -> str | None:
    path_text = _string_or_none(path)
    if path_text is None:
        return None
    return scene_observation_sequence_digest(load_scene_observation_sequence(path_text))


def _graph_digest_or_none(path: object) -> str | None:
    path_text = _string_or_none(path)
    if path_text is None:
        return None
    return graph_json_digest(load_graph_json(path_text))


def _detector_import_report_digest_or_none(path: object) -> str | None:
    path_text = _string_or_none(path)
    if path_text is None:
        return None
    return detector_observation_import_report_digest(
        load_detector_observation_import_report(path_text)
    )


def _predicted_graph_report_digest_or_none(path: object) -> str | None:
    path_text = _string_or_none(path)
    if path_text is None:
        return None
    return predicted_graph_report_digest(load_predicted_graph_report(path_text))


def _predicted_dsg_evidence_report_digest_or_none(path: object) -> str | None:
    path_text = _string_or_none(path)
    if path_text is None:
        return None
    return predicted_dsg_evidence_report_digest(
        load_predicted_dsg_evidence_report(path_text)
    )


def _digest_check(name: str, expected: object, actual: object) -> dict[str, Any]:
    return {
        "name": name,
        "passed": expected == actual,
        "expected": expected,
        "actual": actual,
    }


def _int_value(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        return 0
    return item


def _manifest_required_path(
    payload: dict[str, Any],
    key: str,
    base_dir: Path,
) -> Path:
    value = _required_str(payload, key)
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Predicted DSG detector run manifest field is required: {key}")
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
            f"Predicted DSG detector run manifest field must be a string list: {field}"
        )
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or item == "":
            raise SpatialQAError(
                "Predicted DSG detector run manifest field must be a string list: "
                f"{field}"
            )
        items.append(item)
    return tuple(items)


def _optional_int(value: object, default: int, field: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpatialQAError(
            f"Predicted DSG detector run manifest field must be an integer: {field}"
        )
    return value


def _optional_bool(value: object, default: bool, field: str) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise SpatialQAError(
            f"Predicted DSG detector run manifest field must be a boolean: {field}"
        )
    return value


def _optional_containment_axis(value: object, field: str) -> str:
    if value is None:
        return "z"
    if not isinstance(value, str) or value not in ("z", "y"):
        raise SpatialQAError(
            f"Predicted DSG detector run manifest field must be z or y: {field}"
        )
    return value


def _optional_int_or_none(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpatialQAError(
            f"Predicted DSG detector run manifest field must be an integer or null: {field}"
        )
    if value < 0:
        raise SpatialQAError(
            f"Predicted DSG detector run manifest field must be non-negative: {field}"
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
    raise SpatialQAError("Predicted DSG detector run manifest contains non-JSON value")
