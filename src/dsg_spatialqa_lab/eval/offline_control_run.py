from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.benchmark import (
    load_qa_dataset,
    qa_case_to_dict,
    qa_dataset_digest,
)
from dsg_spatialqa_lab.eval.offline_control_matrix import (
    DEFAULT_REQUIRED_OFFLINE_CONTROL_SOURCE_KINDS,
    load_offline_control_matrix_report,
    offline_control_matrix_report,
    offline_control_matrix_report_digest,
    save_offline_control_matrix_report,
)
from dsg_spatialqa_lab.eval.offline_control_result import (
    load_offline_control_result_report,
    offline_control_result_report,
    offline_control_result_report_digest,
    save_offline_control_result_report,
)
from dsg_spatialqa_lab.eval.offline_predictions import (
    OFFLINE_PREDICTION_RECORD_SCHEMA_VERSION,
    OFFLINE_PREDICTION_RECORD_INPUT_FORMAT,
    QA_PREDICTION_INPUT_FORMAT,
    import_offline_predictions,
    import_qa_prediction_inputs,
    load_offline_prediction_import_report,
    load_offline_prediction_records,
    offline_prediction_import_report_digest,
    offline_prediction_records_digest,
    offline_prediction_source_profile,
    save_offline_prediction_import_report,
)
from dsg_spatialqa_lab.eval.qa_metrics import (
    QA_PREDICTION_SCHEMA_VERSION,
    load_qa_eval_delta_report,
    load_qa_eval_report,
    load_qa_predictions,
    qa_eval_delta_report,
    qa_eval_delta_report_digest,
    qa_eval_report,
    qa_eval_report_digest,
    qa_predictions_digest,
    save_qa_eval_delta_report,
    save_qa_eval_report,
    save_qa_predictions,
)
from dsg_spatialqa_lab.schema import SpatialQAError


OFFLINE_CONTROL_IMPORT_RUN_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.offline-control-import-run.v1"
)
OFFLINE_CONTROL_IMPORT_MANIFEST_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.offline-control-import-manifest.v1"
)
OFFLINE_CONTROL_IMPORT_PREFLIGHT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.offline-control-import-preflight.v1"
)
OFFLINE_CONTROL_ARTIFACT_CONTRACTS_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.offline-control-artifact-contracts.v1"
)
OFFLINE_CONTROL_ARTIFACT_LAUNCH_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.offline-control-artifact-launch-report.v1"
)
OFFLINE_CONTROL_PREDICTION_REQUEST_BUNDLE_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.offline-control-prediction-request-bundle.v1"
)
OFFLINE_CONTROL_PREDICTION_RECEIPT_BUNDLE_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.offline-control-prediction-receipt-bundle.v1"
)
OFFLINE_CONTROL_IMPORT_RUN_LEDGER_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.offline-control-import-run-ledger.v1"
)
REAL_SOURCE_METADATA_FIELDS = ("model_id", "prompt_id", "dataset_id")
PLACEHOLDER_SOURCE_MARKERS = (
    "fixture",
    "mock",
    "placeholder",
    "synthetic",
    "unspecified",
)


def load_offline_control_import_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Offline control import manifest JSON must be an object")
    return _offline_control_import_manifest(
        cast(Mapping[str, object], payload),
        manifest_path.parent,
    )


def offline_control_import_manifest_digest(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_value(manifest),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def offline_control_artifact_contracts_digest(contracts: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in contracts.items() if key != "contracts_digest"}
    return hashlib.sha256(
        json.dumps(
            _json_value(payload),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def offline_control_artifact_contracts_json(contracts: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(contracts), indent=2, sort_keys=True) + "\n"


def save_offline_control_artifact_contracts(
    contracts: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        offline_control_artifact_contracts_json(contracts),
        encoding="utf-8",
    )
    return output_path


def load_offline_control_artifact_contracts(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Offline control artifact contracts JSON must be an object")
    return cast(dict[str, Any], payload)


def offline_control_prediction_request_bundle_digest(
    bundle: Mapping[str, Any],
) -> str:
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


def offline_control_prediction_request_bundle_json(
    bundle: Mapping[str, Any],
) -> str:
    return json.dumps(_json_value(bundle), indent=2, sort_keys=True) + "\n"


def save_offline_control_prediction_request_bundle(
    bundle: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        offline_control_prediction_request_bundle_json(bundle),
        encoding="utf-8",
    )
    return output_path


def load_offline_control_prediction_request_bundle(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError(
            "Offline control prediction request bundle JSON must be an object"
        )
    schema_version = payload.get("schema_version")
    if schema_version != OFFLINE_CONTROL_PREDICTION_REQUEST_BUNDLE_SCHEMA_VERSION:
        raise SpatialQAError(
            "Unsupported offline control prediction request bundle schema version: "
            f"{schema_version}"
        )
    return cast(dict[str, Any], payload)


def offline_control_prediction_receipt_bundle_digest(
    bundle: Mapping[str, Any],
) -> str:
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


def offline_control_prediction_receipt_bundle_json(
    bundle: Mapping[str, Any],
) -> str:
    return json.dumps(_json_value(bundle), indent=2, sort_keys=True) + "\n"


def save_offline_control_prediction_receipt_bundle(
    bundle: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        offline_control_prediction_receipt_bundle_json(bundle),
        encoding="utf-8",
    )
    return output_path


def load_offline_control_prediction_receipt_bundle(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError(
            "Offline control prediction receipt bundle JSON must be an object"
        )
    schema_version = payload.get("schema_version")
    if schema_version != OFFLINE_CONTROL_PREDICTION_RECEIPT_BUNDLE_SCHEMA_VERSION:
        raise SpatialQAError(
            "Unsupported offline control prediction receipt bundle schema version: "
            f"{schema_version}"
        )
    return cast(dict[str, Any], payload)


def offline_control_prediction_request_bundle(
    manifest_path: str | Path,
) -> dict[str, Any]:
    manifest = load_offline_control_import_manifest(manifest_path)
    cases = tuple(load_qa_dataset(_required_str(manifest, "qa_path")))
    bundle: dict[str, Any] = {
        "schema_version": OFFLINE_CONTROL_PREDICTION_REQUEST_BUNDLE_SCHEMA_VERSION,
        "action": "offline_control_prediction_request_bundle",
        "manifest_path": str(manifest_path),
        "manifest_digest": offline_control_import_manifest_digest(manifest),
        "qa": {
            "case_count": len(cases),
            "digest": qa_dataset_digest(cases),
            "path": _required_str(manifest, "qa_path"),
        },
        "case_count": len(cases),
        "case_inputs": [_prediction_request_case(case) for case in cases],
        "expected_input_formats": {
            OFFLINE_PREDICTION_RECORD_INPUT_FORMAT: (
                OFFLINE_PREDICTION_RECORD_SCHEMA_VERSION
            ),
            QA_PREDICTION_INPUT_FORMAT: QA_PREDICTION_SCHEMA_VERSION,
        },
        "prediction_templates": {
            OFFLINE_PREDICTION_RECORD_INPUT_FORMAT: [
                _offline_prediction_record_template(case.id) for case in cases
            ],
            QA_PREDICTION_INPUT_FORMAT: [
                _qa_prediction_template(case.id) for case in cases
            ],
        },
        "required_metadata_fields": list(REAL_SOURCE_METADATA_FIELDS),
        "source_count": len(_manifest_source_rows(manifest)),
        "sources": sorted(
            (
                _prediction_request_source(source)
                for source in _manifest_source_rows(manifest)
            ),
            key=lambda source: str(source["source_key"]),
        ),
    }
    bundle["request_bundle_digest"] = (
        offline_control_prediction_request_bundle_digest(bundle)
    )
    return bundle


def offline_control_prediction_receipt_bundle(
    manifest_path: str | Path,
) -> dict[str, Any]:
    manifest = load_offline_control_import_manifest(manifest_path)
    preflight = offline_control_import_manifest_preflight(manifest_path)
    contracts = _mapping(preflight.get("artifact_contracts"), "artifact_contracts")
    source_rows = {
        _required_str(source, "source_key"): source
        for source in _contract_source_rows(preflight.get("sources"))
    }
    launch_sources = [
        _artifact_launch_source_row(source, manifest=manifest)
        for source in _contract_source_rows(contracts.get("sources"))
    ]
    candidate = _prediction_receipt_candidate(
        _artifact_launch_candidate_row(
            _mapping(contracts.get("candidate"), "candidate")
        )
    )
    commands = _artifact_launch_next_commands(
        contracts_path=None,
        manifest_path=str(manifest_path),
    )
    sources = [
        _prediction_receipt_source(source, source_rows, order=index)
        for index, source in enumerate(launch_sources, start=1)
    ]
    bundle: dict[str, Any] = {
        "schema_version": OFFLINE_CONTROL_PREDICTION_RECEIPT_BUNDLE_SCHEMA_VERSION,
        "action": "offline_control_prediction_receipt_bundle",
        "manifest_path": str(manifest_path),
        "manifest_digest": preflight["manifest_digest"],
        "ready_to_import": preflight.get("ready_to_import") is True,
        "qa": _json_value(contracts.get("qa")),
        "readiness": _json_value(preflight.get("readiness")),
        "required_metadata_fields": list(REAL_SOURCE_METADATA_FIELDS),
        "commands": {
            "import": _required_str(commands, "import"),
            "prediction_request_bundle": _required_str(
                commands,
                "prediction_request_bundle",
            ),
            "preflight": _required_str(commands, "preflight"),
        },
        "candidate": candidate,
        "summary": {
            "blocked_source_count": sum(
                1 for source in sources if source.get("ready_to_import") is not True
            ),
            "candidate_ready": candidate["ready_to_import"] is True,
            "candidate_status": candidate["status"],
            "ready_source_count": sum(
                1 for source in sources if source.get("ready_to_import") is True
            ),
            "source_count": len(sources),
        },
        "sources": sources,
    }
    bundle["receipt_bundle_digest"] = (
        offline_control_prediction_receipt_bundle_digest(bundle)
    )
    return bundle


def validate_offline_control_prediction_receipt_bundle(
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = bundle.get("schema_version")
    action = bundle.get("action")
    receipt_digest = _text_or_none(bundle.get("receipt_bundle_digest"))
    expected_digest = offline_control_prediction_receipt_bundle_digest(bundle)
    candidate = _mapping(bundle.get("candidate"), "candidate")
    summary = _mapping(bundle.get("summary"), "summary")
    sources = _contract_source_rows(bundle.get("sources"))
    expected_summary = {
        "blocked_source_count": sum(
            1 for source in sources if source.get("ready_to_import") is not True
        ),
        "candidate_ready": candidate.get("ready_to_import") is True,
        "candidate_status": _text_or_none(candidate.get("status")),
        "ready_source_count": sum(
            1 for source in sources if source.get("ready_to_import") is True
        ),
        "source_count": len(sources),
    }
    expected_ready = (
        expected_summary["candidate_ready"] is True
        and expected_summary["blocked_source_count"] == 0
    )
    source_rows_valid = all(
        _text_or_none(source.get("source_key")) is not None
        and _text_or_none(source.get("source_kind")) is not None
        and isinstance(source.get("ready_to_import"), bool)
        for source in sources
    )
    checks = [
        {
            "name": "schema_version",
            "passed": (
                schema_version
                == OFFLINE_CONTROL_PREDICTION_RECEIPT_BUNDLE_SCHEMA_VERSION
            ),
            "expected": OFFLINE_CONTROL_PREDICTION_RECEIPT_BUNDLE_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "action",
            "passed": action == "offline_control_prediction_receipt_bundle",
            "expected": "offline_control_prediction_receipt_bundle",
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
            "passed": _text_or_none(bundle.get("manifest_path")) is not None,
            "expected": "present",
            "actual": _text_or_none(bundle.get("manifest_path")),
        },
        {
            "name": "manifest_digest",
            "passed": _text_or_none(bundle.get("manifest_digest")) is not None,
            "expected": "present",
            "actual": _text_or_none(bundle.get("manifest_digest")),
        },
        {
            "name": "summary",
            "passed": _json_value(summary) == _json_value(expected_summary),
            "expected": _json_value(expected_summary),
            "actual": _json_value(summary),
        },
        {
            "name": "ready_to_import",
            "passed": bundle.get("ready_to_import") is expected_ready,
            "expected": expected_ready,
            "actual": bundle.get("ready_to_import"),
        },
        {
            "name": "source_rows",
            "passed": source_rows_valid and len(sources) > 0,
            "expected": True,
            "actual": source_rows_valid and len(sources) > 0,
        },
    ]
    return {
        "action": "validate_offline_control_prediction_receipt_bundle",
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "receipt_bundle_digest": receipt_digest,
        "checks": checks,
    }


def offline_control_import_run_ledger_digest(ledger: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in ledger.items() if key != "ledger_digest"}
    return hashlib.sha256(
        json.dumps(
            _json_value(payload),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def offline_control_import_run_ledger_json(ledger: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(ledger), indent=2, sort_keys=True) + "\n"


def save_offline_control_import_run_ledger(
    ledger: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        offline_control_import_run_ledger_json(ledger),
        encoding="utf-8",
    )
    return output_path


def load_offline_control_import_run_ledger(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Offline control import run ledger JSON must be an object")
    return cast(dict[str, Any], payload)


def offline_control_import_run_ledger(result: Mapping[str, Any]) -> dict[str, Any]:
    qa_eval_handoff = _mapping(result.get("qa_eval_handoff"), "qa_eval_handoff")
    source_eval_rows = _source_eval_rows_by_key(qa_eval_handoff)
    sources = sorted(
        (
            _run_ledger_source_row(source_row, source_eval_rows)
            for source_row in _contract_source_rows(result.get("sources"))
        ),
        key=lambda source: str(source["source_key"]),
    )
    candidate = _run_ledger_candidate(qa_eval_handoff)
    ledger: dict[str, Any] = {
        "schema_version": OFFLINE_CONTROL_IMPORT_RUN_LEDGER_SCHEMA_VERSION,
        "run": {
            "candidate_qa_eval_report_path": result.get("candidate_qa_eval_report_path"),
            "manifest_digest": result.get("manifest_digest"),
            "manifest_path": result.get("manifest_path"),
            "matrix_report_digest": result.get("matrix_report_digest"),
            "matrix_report_path": result.get("matrix_report_path"),
            "offline_control_result_report_digest": result.get(
                "offline_control_result_report_digest"
            ),
            "offline_control_result_report_path": result.get(
                "offline_control_result_report_path"
            ),
            "qa_digest": result.get("qa_digest"),
            "qa_path": result.get("qa_path"),
            "ready": result.get("ready") is True,
            "schema_version": result.get("schema_version"),
        },
        "readiness": _json_value(result.get("readiness")),
        "matrix_readiness": _json_value(result.get("matrix_readiness")),
        "offline_control_result_readiness": _json_value(
            result.get("offline_control_result_readiness")
        ),
        "candidate": candidate,
        "summary": {
            "candidate_requested": qa_eval_handoff.get("requested") is True,
            "ready_source_count": sum(
                1
                for source in sources
                if not _source_missing_or_diagnostic(
                    _mapping(source.get("summary"), "summary")
                )
            ),
            "source_count": len(sources),
        },
        "sources": sources,
    }
    ledger["ledger_digest"] = offline_control_import_run_ledger_digest(ledger)
    return ledger


def validate_offline_control_import_run_ledger(
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = ledger.get("schema_version")
    ledger_digest = _text_or_none(ledger.get("ledger_digest"))
    expected_digest = offline_control_import_run_ledger_digest(ledger)
    sources = _contract_source_rows(ledger.get("sources"))
    source_keys = [
        source_key
        for source in sources
        if (source_key := _text_or_none(source.get("source_key"))) is not None
    ]
    ready_source_count = sum(
        1
        for source in sources
        if not _source_missing_or_diagnostic(_mapping(source.get("summary"), "summary"))
    )
    summary = ledger.get("summary")
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == OFFLINE_CONTROL_IMPORT_RUN_LEDGER_SCHEMA_VERSION,
            "expected": OFFLINE_CONTROL_IMPORT_RUN_LEDGER_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "ledger_digest",
            "passed": ledger_digest == expected_digest,
            "expected": expected_digest,
            "actual": ledger_digest,
        },
        {
            "name": "source_count",
            "passed": _summary_int(summary, "source_count") == len(sources),
            "expected": len(sources),
            "actual": _summary_int(summary, "source_count"),
        },
        {
            "name": "ready_source_count",
            "passed": _summary_int(summary, "ready_source_count") == ready_source_count,
            "expected": ready_source_count,
            "actual": _summary_int(summary, "ready_source_count"),
        },
        {
            "name": "source_keys_complete",
            "passed": len(source_keys) == len(sources),
            "expected": len(sources),
            "actual": len(source_keys),
        },
        {
            "name": "source_keys_unique",
            "passed": len(set(source_keys)) == len(source_keys),
            "expected": len(source_keys),
            "actual": len(set(source_keys)),
        },
        {
            "name": "source_keys_sorted",
            "passed": source_keys == sorted(source_keys),
            "expected": sorted(source_keys),
            "actual": source_keys,
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "ledger_digest": ledger_digest,
        "checks": checks,
    }


def compare_offline_control_import_run_ledger(
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_offline_control_import_run_ledger(ledger)
    run = _mapping(ledger.get("run"), "run")
    candidate = _mapping(ledger.get("candidate"), "candidate")
    checks = [
        {
            "name": "ledger_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        _ledger_digest_check(
            "manifest_digest_matches_current",
            run.get("manifest_digest"),
            offline_control_import_manifest_digest(
                load_offline_control_import_manifest(_required_str(run, "manifest_path"))
            ),
        ),
        _ledger_digest_check(
            "qa_digest_matches_current",
            run.get("qa_digest"),
            qa_dataset_digest(load_qa_dataset(_required_str(run, "qa_path"))),
        ),
        _ledger_digest_check(
            "matrix_report_digest_matches_current",
            run.get("matrix_report_digest"),
            offline_control_matrix_report_digest(
                load_offline_control_matrix_report(
                    _required_str(run, "matrix_report_path")
                )
            ),
        ),
        _ledger_digest_check(
            "offline_control_result_report_digest_matches_current",
            run.get("offline_control_result_report_digest"),
            offline_control_result_report_digest(
                load_offline_control_result_report(
                    _required_str(run, "offline_control_result_report_path")
                )
            ),
        ),
        _ledger_digest_check(
            "candidate_prediction_digest_matches_current",
            candidate.get("prediction_digest"),
            _qa_prediction_digest_or_none(candidate.get("prediction_path")),
        ),
        _ledger_digest_check(
            "candidate_qa_eval_report_digest_matches_current",
            candidate.get("qa_eval_report_digest"),
            _qa_eval_report_digest_or_none(candidate.get("qa_eval_report_path")),
        ),
    ]
    for source in _contract_source_rows(ledger.get("sources")):
        source_key = _required_str(source, "source_key")
        import_report = load_offline_prediction_import_report(
            _required_str(source, "import_report_path")
        )
        checks.extend(
            [
                _ledger_digest_check(
                    f"{source_key}:input_digest_matches_current",
                    source.get("input_digest"),
                    _source_input_digest(source),
                ),
                _ledger_digest_check(
                    f"{source_key}:normalized_prediction_digest_matches_current",
                    source.get("normalized_prediction_digest"),
                    _qa_prediction_digest_or_none(
                        source.get("normalized_prediction_path")
                    ),
                ),
                _ledger_digest_check(
                    f"{source_key}:import_report_digest_matches_current",
                    source.get("import_report_digest"),
                    offline_prediction_import_report_digest(import_report),
                ),
                _ledger_digest_check(
                    f"{source_key}:import_report_input_digest_matches_source",
                    source.get("input_digest"),
                    import_report.get("input_digest"),
                ),
                _ledger_digest_check(
                    f"{source_key}:import_report_prediction_digest_matches_source",
                    source.get("normalized_prediction_digest"),
                    import_report.get("prediction_digest"),
                ),
                _ledger_digest_check(
                    f"{source_key}:baseline_qa_eval_report_digest_matches_current",
                    source.get("baseline_qa_eval_report_digest"),
                    _qa_eval_report_digest_or_none(
                        source.get("baseline_qa_eval_report_path")
                    ),
                ),
                _ledger_digest_check(
                    f"{source_key}:qa_eval_delta_report_digest_matches_current",
                    source.get("qa_eval_delta_report_digest"),
                    _qa_eval_delta_report_digest_or_none(
                        source.get("qa_eval_delta_report_path")
                    ),
                ),
            ]
        )
    return {
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": _text_or_none(ledger.get("ledger_digest")),
        "validation": validation,
        "checks": checks,
    }


def validate_offline_control_artifact_contracts(
    contracts: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = contracts.get("schema_version")
    contracts_digest = _text_or_none(contracts.get("contracts_digest"))
    expected_digest = offline_control_artifact_contracts_digest(contracts)
    sources = _contract_source_rows(contracts.get("sources"))
    source_keys = [
        source_key
        for source in sources
        if (source_key := _text_or_none(source.get("source_key"))) is not None
    ]
    ready_source_count = sum(
        1 for source in sources if source.get("ready_to_import") is True
    )
    summary = contracts.get("summary")
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == OFFLINE_CONTROL_ARTIFACT_CONTRACTS_SCHEMA_VERSION,
            "expected": OFFLINE_CONTROL_ARTIFACT_CONTRACTS_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "contracts_digest",
            "passed": contracts_digest == expected_digest,
            "expected": expected_digest,
            "actual": contracts_digest,
        },
        {
            "name": "source_contract_count",
            "passed": _summary_int(summary, "source_contract_count") == len(sources),
            "expected": len(sources),
            "actual": _summary_int(summary, "source_contract_count"),
        },
        {
            "name": "ready_source_contract_count",
            "passed": (
                _summary_int(summary, "ready_source_contract_count")
                == ready_source_count
            ),
            "expected": ready_source_count,
            "actual": _summary_int(summary, "ready_source_contract_count"),
        },
        {
            "name": "source_keys_complete",
            "passed": len(source_keys) == len(sources),
            "expected": len(sources),
            "actual": len(source_keys),
        },
        {
            "name": "source_keys_unique",
            "passed": len(set(source_keys)) == len(source_keys),
            "expected": len(source_keys),
            "actual": len(set(source_keys)),
        },
        {
            "name": "source_keys_sorted",
            "passed": source_keys == sorted(source_keys),
            "expected": sorted(source_keys),
            "actual": source_keys,
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "contracts_digest": contracts_digest,
        "checks": checks,
    }


def compare_offline_control_artifact_contracts(
    contracts: Mapping[str, Any],
    manifest_path: str | Path,
) -> dict[str, Any]:
    validation = validate_offline_control_artifact_contracts(contracts)
    current = _mapping(
        offline_control_import_manifest_preflight(manifest_path).get("artifact_contracts"),
        "artifact_contracts",
    )
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
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def offline_control_artifact_launch_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return hashlib.sha256(
        json.dumps(
            _json_value(payload),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def offline_control_artifact_launch_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(report), indent=2, sort_keys=True) + "\n"


def offline_control_artifact_launch_report(
    contracts: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    contracts_path: str | Path | None = None,
) -> dict[str, Any]:
    validation = validate_offline_control_artifact_contracts(contracts)
    comparison = compare_offline_control_artifact_contracts(contracts, manifest_path)
    manifest = load_offline_control_import_manifest(manifest_path)
    preflight = offline_control_import_manifest_preflight(manifest_path)
    current_contracts = _mapping(
        preflight.get("artifact_contracts"),
        "artifact_contracts",
    )
    candidate = _artifact_launch_candidate_row(
        _mapping(current_contracts.get("candidate"), "candidate")
    )
    sources = [
        _artifact_launch_source_row(source, manifest=manifest)
        for source in _contract_source_rows(current_contracts.get("sources"))
    ]
    summary = _artifact_launch_summary(sources, candidate)
    ready_to_import = (
        validation["valid"] is True
        and comparison["matches"] is True
        and preflight.get("ready_to_import") is True
        and candidate["ready_to_import"] is True
    )
    contracts_path_text = str(contracts_path) if contracts_path is not None else None
    manifest_path_text = str(manifest_path)
    next_commands = _artifact_launch_next_commands(
        contracts_path=contracts_path_text,
        manifest_path=manifest_path_text,
    )
    report: dict[str, Any] = {
        "schema_version": OFFLINE_CONTROL_ARTIFACT_LAUNCH_REPORT_SCHEMA_VERSION,
        "action": "offline_control_artifact_launch_report",
        "contracts_path": contracts_path_text,
        "manifest_path": manifest_path_text,
        "contracts_digest": _text_or_none(contracts.get("contracts_digest")),
        "current_contracts_digest": _text_or_none(
            current_contracts.get("contracts_digest")
        ),
        "manifest_digest": preflight["manifest_digest"],
        "ready_to_import": ready_to_import,
        "preflight_ready_to_import": preflight.get("ready_to_import") is True,
        "candidate": candidate,
        "actionable_blockers": _artifact_launch_actionable_blockers(
            sources,
            candidate,
        ),
        "external_prediction_intake_plan": (
            _artifact_launch_external_prediction_intake_plan(
                sources,
                next_commands=next_commands,
            )
        ),
        "source_import_plan": _artifact_launch_source_import_plan(
            sources,
            candidate,
            next_commands=next_commands,
        ),
        "summary": summary,
        "sources": sources,
        "validation": validation,
        "comparison": {
            "matches": comparison["matches"],
            "saved_digest": comparison["saved_digest"],
            "current_digest": comparison["current_digest"],
        },
        "next_commands": next_commands,
    }
    report["report_digest"] = offline_control_artifact_launch_report_digest(report)
    return report


def offline_control_import_manifest_preflight(manifest_path: str | Path) -> dict[str, Any]:
    manifest = load_offline_control_import_manifest(manifest_path)
    cases = tuple(load_qa_dataset(_required_str(manifest, "qa_path")))
    root = Path(_required_str(manifest, "output_dir"))
    source_rows: list[dict[str, Any]] = []
    contract_sources: list[tuple[dict[str, Any], dict[str, Any]]] = []
    import_reports: list[dict[str, Any]] = []
    import_report_paths: list[Path] = []
    invalid_sources: list[dict[str, Any]] = []
    missing_sources: list[str] = []
    for source_spec in cast(Sequence[Mapping[str, object]], manifest["sources"]):
        source = _source_spec(source_spec, root)
        source_row = _preflight_source_row(cases, manifest, source)
        source_rows.append(source_row)
        contract_sources.append((source, source_row))
        if source_row["valid"] is True:
            import_reports.append(cast(dict[str, Any], source_row["import_report"]))
            import_report_paths.append(Path(str(source_row["import_report_path"])))
        else:
            invalid_sources.append(
                {
                    "source_kind": source_row["source_kind"],
                    "source_name": source_row["source_name"],
                    "error": source_row["error"],
                }
            )
        summary = _mapping(source_row.get("summary"), "summary")
        if _source_missing_or_diagnostic(summary):
            missing_sources.append(_required_str(source_row, "source_name"))

    matrix_report = offline_control_matrix_report(
        tuple(import_reports),
        report_paths=tuple(import_report_paths),
        required_source_kinds=_string_sequence(
            manifest.get("required_source_kinds"),
            "required_source_kinds",
        ),
    )
    matrix_readiness = _mapping(matrix_report["readiness"], "readiness")
    source_metadata_summary = _source_metadata_summary(import_reports)
    source_metadata_checks = _source_metadata_checks(source_metadata_summary)
    readiness = _run_readiness(matrix_readiness, source_metadata_checks)
    candidate = _candidate_preflight(manifest.get("candidate_prediction_path"))
    artifact_contracts = _artifact_contracts(
        manifest,
        cases,
        tuple(contract_sources),
        candidate,
    )
    ready_to_import = (
        readiness["ready"] is True
        and not invalid_sources
        and not missing_sources
        and candidate["valid"] is True
    )
    return {
        "schema_version": OFFLINE_CONTROL_IMPORT_PREFLIGHT_SCHEMA_VERSION,
        "action": "offline_control_import_manifest_preflight",
        "manifest_schema_version": manifest["schema_version"],
        "manifest_path": str(manifest_path),
        "manifest_digest": offline_control_import_manifest_digest(manifest),
        "ready_to_import": ready_to_import,
        "qa_path": manifest["qa_path"],
        "qa_digest": qa_dataset_digest(cases),
        "required_source_kinds": list(matrix_report["required_source_kinds"]),
        "artifact_contracts": artifact_contracts,
        "candidate_prediction": candidate,
        "planned_outputs": {
            "matrix_report_path": manifest["matrix_report_path"],
            "output_dir": manifest["output_dir"],
            "qa_eval_output_dir": manifest.get("qa_eval_output_dir"),
            "result_report_path": manifest.get("result_report_path"),
        },
        "matrix_readiness": matrix_readiness,
        "source_metadata_checks": source_metadata_checks,
        "source_metadata_summary": source_metadata_summary,
        "readiness": {
            **readiness,
            "ready": ready_to_import,
        },
        "summary": {
            "candidate_prediction_count": candidate["prediction_count"],
            "invalid_source_count": len(invalid_sources),
            "missing_source_count": len(missing_sources),
            "qa_case_count": len(cases),
            "source_count": len(source_rows),
        },
        "missing_sources": sorted(missing_sources),
        "invalid_sources": sorted(
            invalid_sources,
            key=lambda item: (str(item["source_kind"]), str(item["source_name"])),
        ),
        "sources": sorted(source_rows, key=lambda row: str(row["source_key"])),
    }


def run_offline_control_import_manifest(manifest_path: str | Path) -> dict[str, Any]:
    manifest = load_offline_control_import_manifest(manifest_path)
    result = run_offline_control_imports(
        qa_path=_required_str(manifest, "qa_path"),
        source_specs=cast(Sequence[Mapping[str, object]], manifest["sources"]),
        output_dir=_required_str(manifest, "output_dir"),
        matrix_report_path=_required_str(manifest, "matrix_report_path"),
        required_source_kinds=_string_sequence(
            manifest.get("required_source_kinds"),
            "required_source_kinds",
        ),
        candidate_prediction_path=manifest.get("candidate_prediction_path"),
        candidate_name=_required_str(manifest, "candidate_name"),
        qa_eval_output_dir=manifest.get("qa_eval_output_dir"),
        result_report_path=manifest.get("result_report_path"),
    )
    return {
        **result,
        "action": "run_offline_control_import_manifest",
        "manifest_schema_version": manifest["schema_version"],
        "manifest_path": str(manifest_path),
        "manifest_digest": offline_control_import_manifest_digest(manifest),
        "manifest_summary": {
            "has_candidate_prediction": manifest.get("candidate_prediction_path")
            is not None,
            "source_count": len(cast(Sequence[object], manifest["sources"])),
        },
    }


def run_offline_control_imports(
    *,
    qa_path: str | Path,
    source_specs: Sequence[Mapping[str, object]],
    output_dir: str | Path,
    matrix_report_path: str | Path,
    required_source_kinds: Sequence[str] = DEFAULT_REQUIRED_OFFLINE_CONTROL_SOURCE_KINDS,
    candidate_prediction_path: str | Path | None = None,
    candidate_name: str = "predicted_graph_tool",
    qa_eval_output_dir: str | Path | None = None,
    result_report_path: str | Path | None = None,
) -> dict[str, Any]:
    if not source_specs:
        raise SpatialQAError("Offline control import run requires at least one source")
    cases = tuple(load_qa_dataset(qa_path))
    root = Path(output_dir)
    import_reports: list[dict[str, Any]] = []
    import_report_paths: list[Path] = []
    imported_sources: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for source_spec in source_specs:
        source = _source_spec(source_spec, root)
        if source["input_format"] == QA_PREDICTION_INPUT_FORMAT:
            predictions, import_report = import_qa_prediction_inputs(
                cases,
                load_qa_predictions(source["input_path"]),
                source_name=source["source_name"],
                source_kind=source["source_kind"],
                source_metadata=source["metadata"],
                qa_path=qa_path,
                input_path=source["input_path"],
                prediction_path=source["prediction_path"],
            )
        else:
            records = tuple(load_offline_prediction_records(source["input_path"]))
            predictions, import_report = import_offline_predictions(
                cases,
                records,
                source_name=source["source_name"],
                source_kind=source["source_kind"],
                source_metadata=source["metadata"],
                qa_path=qa_path,
                input_path=source["input_path"],
                prediction_path=source["prediction_path"],
            )
        save_qa_predictions(predictions, source["prediction_path"])
        save_offline_prediction_import_report(
            import_report,
            source["import_report_path"],
        )
        import_reports.append(import_report)
        import_report_paths.append(source["import_report_path"])
        source_row = {
            "dataset_id": import_report["source_profile"]["dataset_id"],
            "input_format": source["input_format"],
            "import_report_digest": import_report["report_digest"],
            "import_report_path": str(source["import_report_path"]),
            "input_path": str(source["input_path"]),
            "model_id": import_report["source_profile"]["model_id"],
            "prediction_digest": qa_predictions_digest(predictions),
            "prediction_path": str(source["prediction_path"]),
            "prompt_id": import_report["source_profile"]["prompt_id"],
            "source_key": import_report["source_profile"]["source_key"],
            "source_kind": source["source_kind"],
            "source_name": source["source_name"],
            "summary": import_report["summary"],
        }
        source_rows.append(source_row)
        imported_sources.append(
            {
                "predictions": tuple(predictions),
                "source_row": source_row,
            }
        )
    matrix_report = offline_control_matrix_report(
        tuple(import_reports),
        report_paths=tuple(import_report_paths),
        required_source_kinds=required_source_kinds,
    )
    save_offline_control_matrix_report(matrix_report, matrix_report_path)
    source_metadata_summary = _source_metadata_summary(import_reports)
    source_metadata_checks = _source_metadata_checks(source_metadata_summary)
    matrix_readiness = _mapping(matrix_report["readiness"], "readiness")
    readiness = _run_readiness(matrix_readiness, source_metadata_checks)
    qa_eval_handoff = _qa_eval_handoff(
        cases=cases,
        qa_path=qa_path,
        candidate_prediction_path=candidate_prediction_path,
        candidate_name=candidate_name,
        qa_eval_output_dir=qa_eval_output_dir,
        fallback_output_dir=root,
        imported_sources=tuple(imported_sources),
    )
    result_handoff = _offline_control_result_handoff(
        matrix_report=matrix_report,
        matrix_report_path=matrix_report_path,
        qa_eval_handoff=qa_eval_handoff,
        qa_eval_output_dir=qa_eval_output_dir,
        fallback_output_dir=root,
        result_report_path=result_report_path,
    )
    return {
        "schema_version": OFFLINE_CONTROL_IMPORT_RUN_SCHEMA_VERSION,
        "action": "run_offline_control_imports",
        "candidate_qa_eval_report_path": qa_eval_handoff[
            "candidate_qa_eval_report_path"
        ],
        "qa_path": str(qa_path),
        "qa_digest": qa_dataset_digest(cases),
        "qa_eval_delta_report_paths": qa_eval_handoff["qa_eval_delta_report_paths"],
        "qa_eval_delta_reports": qa_eval_handoff["qa_eval_delta_reports"],
        "qa_eval_handoff": qa_eval_handoff,
        "output_dir": str(root),
        "matrix_report_path": str(matrix_report_path),
        "matrix_report_digest": matrix_report["report_digest"],
        "matrix_readiness": matrix_readiness,
        "offline_control_result_readiness": result_handoff["readiness"],
        "offline_control_result_report_digest": result_handoff["report_digest"],
        "offline_control_result_report_path": result_handoff["path"],
        "ready": readiness["ready"],
        "readiness": readiness,
        "required_source_kinds": list(matrix_report["required_source_kinds"]),
        "source_metadata_checks": source_metadata_checks,
        "source_metadata_summary": source_metadata_summary,
        "source_count": len(source_rows),
        "sources": sorted(source_rows, key=lambda row: str(row["source_key"])),
        "summary": matrix_report["summary"],
    }


def _offline_control_import_manifest(
    payload: Mapping[str, object],
    base_dir: Path,
) -> dict[str, Any]:
    schema_version = _required_str(payload, "schema_version")
    if schema_version != OFFLINE_CONTROL_IMPORT_MANIFEST_SCHEMA_VERSION:
        raise SpatialQAError(
            f"Unsupported offline control import manifest schema version: {schema_version}"
        )
    sources = _manifest_sources(payload.get("sources"), base_dir)
    return {
        "schema_version": schema_version,
        "qa_path": str(_manifest_required_path(payload, "qa_path", base_dir)),
        "output_dir": str(_manifest_required_path(payload, "output_dir", base_dir)),
        "matrix_report_path": str(
            _manifest_required_path(payload, "matrix_report_path", base_dir)
        ),
        "required_source_kinds": list(
            _optional_string_sequence(
                payload.get("required_source_kinds"),
                DEFAULT_REQUIRED_OFFLINE_CONTROL_SOURCE_KINDS,
                "required_source_kinds",
            )
        ),
        "candidate_prediction_path": _manifest_optional_path(
            payload,
            "candidate_prediction_path",
            base_dir,
        ),
        "candidate_name": _optional_text(
            payload.get("candidate_name"),
            "predicted_graph_tool",
            "candidate_name",
        ),
        "qa_eval_output_dir": _manifest_optional_path(
            payload,
            "qa_eval_output_dir",
            base_dir,
        ),
        "result_report_path": _manifest_optional_path(
            payload,
            "result_report_path",
            base_dir,
        ),
        "sources": list(sources),
    }


def _manifest_sources(value: object, base_dir: Path) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise SpatialQAError("Offline control import manifest sources must be a list")
    sources = tuple(_manifest_source(source, base_dir) for source in value)
    if not sources:
        raise SpatialQAError("Offline control import manifest requires at least one source")
    return sources


def _manifest_source(value: object, base_dir: Path) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SpatialQAError("Offline control import manifest source must be an object")
    source = cast(Mapping[str, object], value)
    result: dict[str, Any] = {
        "source_kind": _required_str(source, "source_kind"),
        "source_name": _required_str(source, "source_name"),
        "input_path": str(_manifest_required_path(source, "input_path", base_dir)),
        "input_format": _input_format(source.get("input_format")),
        "metadata": _metadata(source.get("metadata")),
    }
    prediction_path = _manifest_optional_path(source, "prediction_path", base_dir)
    if prediction_path is not None:
        result["prediction_path"] = prediction_path
    import_report_path = _manifest_optional_path(source, "import_report_path", base_dir)
    if import_report_path is not None:
        result["import_report_path"] = import_report_path
    return result


def _manifest_required_path(
    payload: Mapping[str, object],
    key: str,
    base_dir: Path,
) -> Path:
    return _manifest_path(_required_path(payload, key), base_dir)


def _manifest_optional_path(
    payload: Mapping[str, object],
    key: str,
    base_dir: Path,
) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, str) and value == "":
        return None
    if not isinstance(value, (str, Path)):
        raise SpatialQAError(f"Offline control import manifest path must be a string: {key}")
    return str(_manifest_path(value, base_dir))


def _manifest_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def _optional_text(value: object, default: str, field: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Offline control import manifest field must be a string: {field}")
    return value


def _optional_string_sequence(
    value: object,
    default: Sequence[str],
    field: str,
) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    return _string_sequence(value, field)


def _string_sequence(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise SpatialQAError(
            f"Offline control import manifest field must be a string list: {field}"
        )
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or item == "":
            raise SpatialQAError(
                f"Offline control import manifest field must be a string list: {field}"
            )
        items.append(item)
    return tuple(items)


def _manifest_source_rows(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    value = manifest.get("sources")
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise SpatialQAError("Offline control import manifest sources must be a list")
    return tuple(
        cast(Mapping[str, Any], item) for item in value if isinstance(item, Mapping)
    )


def _prediction_request_case(case: Any) -> dict[str, Any]:
    payload = qa_case_to_dict(case)
    return {
        "answer_type": payload["answer_type"],
        "case_id": payload["id"],
        "choices": payload["choices"],
        "difficulty": payload["difficulty"],
        "episode_id": payload["episode_id"],
        "question": payload["question"],
        "question_type": payload["question_type"],
        "reference_frame": payload["reference_frame"],
        "scene_id": payload["scene_id"],
        "step": payload["step"],
        "tags": payload["tags"],
    }


def _prediction_request_source(source: Mapping[str, Any]) -> dict[str, Any]:
    source_kind = _required_str(source, "source_kind")
    source_name = _required_str(source, "source_name")
    input_format = _required_str(source, "input_format")
    return {
        "expected_schema_version": _expected_input_schema_version(input_format),
        "input_format": input_format,
        "prediction_output_path": _required_str(source, "input_path"),
        "source_key": f"{source_kind}:{source_name}",
        "source_kind": source_kind,
        "source_metadata": _json_value(source.get("metadata", {})),
        "source_name": source_name,
    }


def _qa_prediction_template(case_id: str) -> dict[str, Any]:
    return {
        "schema_version": QA_PREDICTION_SCHEMA_VERSION,
        "id": case_id,
        "answer": {},
        "evidence_nodes": [],
        "evidence_edges": [],
        "confidence": 0.0,
        "error": None,
    }


def _offline_prediction_record_template(case_id: str) -> dict[str, Any]:
    return {
        "schema_version": OFFLINE_PREDICTION_RECORD_SCHEMA_VERSION,
        "case_id": case_id,
        "answer": {},
        "evidence_nodes": [],
        "evidence_edges": [],
        "confidence": 0.0,
        "error": None,
        "metadata": {},
    }


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
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
    raise SpatialQAError("Offline control import manifest contains non-JSON value")


def _qa_eval_handoff(
    *,
    cases: Sequence[Any],
    qa_path: str | Path,
    candidate_prediction_path: str | Path | None,
    candidate_name: str,
    qa_eval_output_dir: str | Path | None,
    fallback_output_dir: Path,
    imported_sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if candidate_prediction_path is None:
        return {
            "requested": False,
            "ready": False,
            "candidate_name": candidate_name,
            "candidate_prediction_path": None,
            "candidate_qa_eval_report_path": None,
            "candidate_qa_eval_report_digest": None,
            "baseline_qa_eval_report_paths": {},
            "qa_eval_delta_report_paths": {},
            "qa_eval_delta_reports": [],
            "summary": {
                "baseline_report_count": 0,
                "delta_report_count": 0,
            },
        }

    eval_root = (
        Path(qa_eval_output_dir)
        if qa_eval_output_dir is not None
        else fallback_output_dir / "qa-eval"
    )
    candidate_predictions = tuple(load_qa_predictions(candidate_prediction_path))
    candidate_report_path = (
        eval_root / "candidate" / _path_part(candidate_name) / "qa-eval.json"
    )
    candidate_report = qa_eval_report(
        cases,
        candidate_predictions,
        gold_path=qa_path,
        prediction_path=candidate_prediction_path,
    )
    save_qa_eval_report(candidate_report, candidate_report_path)

    baseline_report_paths: dict[str, str] = {}
    delta_report_paths: dict[str, str] = {}
    delta_report_rows: list[dict[str, Any]] = []
    for imported_source in sorted(
        imported_sources,
        key=lambda source: str(
            _mapping(source.get("source_row"), "source_row").get("source_key")
        ),
    ):
        source_row = _mapping(imported_source.get("source_row"), "source_row")
        source_key = _required_str(source_row, "source_key")
        source_kind = _required_str(source_row, "source_kind")
        source_name = _required_str(source_row, "source_name")
        prediction_path = _required_str(source_row, "prediction_path")
        source_path_part = _path_part(source_key)
        baseline_report_path = (
            eval_root / "baselines" / source_path_part / "qa-eval.json"
        )
        baseline_report = qa_eval_report(
            cases,
            cast(Sequence[Any], imported_source.get("predictions")),
            gold_path=qa_path,
            prediction_path=prediction_path,
        )
        save_qa_eval_report(baseline_report, baseline_report_path)
        delta_report_path = (
            eval_root
            / "deltas"
            / f"{_path_part(candidate_name)}-vs-{source_path_part}.json"
        )
        delta_report = qa_eval_delta_report(
            candidate_report,
            baseline_report,
            candidate_name=candidate_name,
            baseline_name=source_kind,
            candidate_report_path=candidate_report_path,
            baseline_report_path=baseline_report_path,
        )
        save_qa_eval_delta_report(delta_report, delta_report_path)
        baseline_report_paths[source_key] = str(baseline_report_path)
        delta_report_paths[source_key] = str(delta_report_path)
        delta_report_rows.append(
            {
                "source_key": source_key,
                "source_kind": source_kind,
                "source_name": source_name,
                "baseline_name": source_kind,
                "baseline_qa_eval_report_path": str(baseline_report_path),
                "baseline_qa_eval_report_digest": baseline_report["report_digest"],
                "qa_eval_delta_report_path": str(delta_report_path),
                "qa_eval_delta_report_digest": delta_report["report_digest"],
            }
        )

    return {
        "requested": True,
        "ready": len(delta_report_rows) == len(imported_sources),
        "candidate_name": candidate_name,
        "candidate_prediction_path": str(candidate_prediction_path),
        "candidate_qa_eval_report_path": str(candidate_report_path),
        "candidate_qa_eval_report_digest": candidate_report["report_digest"],
        "baseline_qa_eval_report_paths": baseline_report_paths,
        "qa_eval_delta_report_paths": delta_report_paths,
        "qa_eval_delta_reports": delta_report_rows,
        "summary": {
            "baseline_report_count": len(baseline_report_paths),
            "delta_report_count": len(delta_report_rows),
        },
    }


def _offline_control_result_handoff(
    *,
    matrix_report: Mapping[str, Any],
    matrix_report_path: str | Path,
    qa_eval_handoff: Mapping[str, Any],
    qa_eval_output_dir: str | Path | None,
    fallback_output_dir: Path,
    result_report_path: str | Path | None,
) -> dict[str, Any]:
    if qa_eval_handoff.get("requested") is not True:
        return {
            "path": None,
            "report_digest": None,
            "readiness": {
                "ready": False,
                "failed_check_count": 1,
                "failed_checks": ["qa_eval_handoff_not_requested"],
            },
        }
    candidate_report_path = qa_eval_handoff.get("candidate_qa_eval_report_path")
    delta_paths = qa_eval_handoff.get("qa_eval_delta_report_paths")
    if not isinstance(candidate_report_path, str) or not isinstance(delta_paths, Mapping):
        return {
            "path": None,
            "report_digest": None,
            "readiness": {
                "ready": False,
                "failed_check_count": 1,
                "failed_checks": ["qa_eval_handoff_incomplete"],
            },
        }
    output_path = (
        Path(result_report_path)
        if result_report_path is not None
        else _default_result_report_path(qa_eval_output_dir, fallback_output_dir)
    )
    report = offline_control_result_report(
        matrix_report,
        matrix_report_path=matrix_report_path,
        candidate_qa_eval_report_path=candidate_report_path,
        qa_eval_delta_report_paths=cast(Mapping[str, str | Path], delta_paths),
    )
    save_offline_control_result_report(report, output_path)
    return {
        "path": str(output_path),
        "report_digest": report["report_digest"],
        "readiness": report["readiness"],
    }


def _default_result_report_path(
    qa_eval_output_dir: str | Path | None,
    fallback_output_dir: Path,
) -> Path:
    eval_root = (
        Path(qa_eval_output_dir)
        if qa_eval_output_dir is not None
        else fallback_output_dir / "qa-eval"
    )
    return eval_root / "offline-control-result.json"


def _preflight_source_row(
    cases: Sequence[Any],
    manifest: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    base_row = {
        "source_kind": _required_str(source, "source_kind"),
        "source_name": _required_str(source, "source_name"),
        "input_format": _required_str(source, "input_format"),
        "input_path": str(source["input_path"]),
        "prediction_path": str(source["prediction_path"]),
        "import_report_path": str(source["import_report_path"]),
    }
    try:
        if source["input_format"] == QA_PREDICTION_INPUT_FORMAT:
            predictions, import_report = import_qa_prediction_inputs(
                cases,
                load_qa_predictions(source["input_path"]),
                source_name=_required_str(source, "source_name"),
                source_kind=_required_str(source, "source_kind"),
                source_metadata=_mapping(source.get("metadata"), "metadata"),
                qa_path=_required_str(manifest, "qa_path"),
                input_path=source["input_path"],
                prediction_path=source["prediction_path"],
            )
        else:
            records = tuple(load_offline_prediction_records(source["input_path"]))
            predictions, import_report = import_offline_predictions(
                cases,
                records,
                source_name=_required_str(source, "source_name"),
                source_kind=_required_str(source, "source_kind"),
                source_metadata=_mapping(source.get("metadata"), "metadata"),
                qa_path=_required_str(manifest, "qa_path"),
                input_path=source["input_path"],
                prediction_path=source["prediction_path"],
            )
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        return {
            **base_row,
            "error": str(exc),
            "import_report": None,
            "input_digest": None,
            "prediction_digest": None,
            "source_key": (
                f"{base_row['source_kind']}:{base_row['source_name']}"
            ),
            "summary": {
                "duplicate_case_count": 0,
                "error_prediction_count": 0,
                "gold_case_count": len(cases),
                "imported_prediction_count": 0,
                "missing_case_count": len(cases),
                "record_count": 0,
                "unknown_case_count": 0,
            },
            "valid": False,
        }
    return {
        **base_row,
        "error": None,
        "import_report": import_report,
        "import_report_digest": import_report["report_digest"],
        "input_digest": import_report["input_digest"],
        "prediction_digest": qa_predictions_digest(predictions),
        "source_key": import_report["source_profile"]["source_key"],
        "summary": import_report["summary"],
        "valid": True,
    }


def _artifact_contracts(
    manifest: Mapping[str, Any],
    cases: Sequence[Any],
    sources: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    source_contracts = sorted(
        (
            _source_artifact_contract(manifest, source, source_row)
            for source, source_row in sources
        ),
        key=lambda contract: str(contract["source_key"]),
    )
    contracts: dict[str, Any] = {
        "schema_version": OFFLINE_CONTROL_ARTIFACT_CONTRACTS_SCHEMA_VERSION,
        "qa": {
            "case_count": len(cases),
            "digest": qa_dataset_digest(cases),
            "path": _required_str(manifest, "qa_path"),
        },
        "candidate": {
            "error": candidate.get("error"),
            "path": candidate.get("path"),
            "prediction_count": candidate.get("prediction_count"),
            "qa_eval_report_path": _candidate_qa_eval_report_path(manifest),
            "status": _candidate_status(candidate),
            "valid": candidate.get("valid") is True,
        },
        "summary": {
            "ready_source_contract_count": sum(
                1 for contract in source_contracts if contract["ready_to_import"] is True
            ),
            "source_contract_count": len(source_contracts),
        },
        "sources": source_contracts,
    }
    contracts["contracts_digest"] = offline_control_artifact_contracts_digest(contracts)
    return contracts


def _artifact_launch_source_row(
    source: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    input_contract = _mapping(source.get("input"), "input")
    metadata = _mapping(source.get("metadata"), "metadata")
    source_metadata = _mapping(source.get("source_metadata"), "source_metadata")
    diagnostics = _mapping(source.get("diagnostics"), "diagnostics")
    planned_outputs = _mapping(source.get("planned_outputs"), "planned_outputs")
    input_status = _required_str(input_contract, "status")
    metadata_ready = metadata.get("ready") is True
    blocking_reasons = _artifact_launch_blocking_reasons(
        input_status=input_status,
        metadata_ready=metadata_ready,
        ready_to_import=source.get("ready_to_import") is True,
    )
    result = {
        "source_key": _required_str(source, "source_key"),
        "source_kind": _required_str(source, "source_kind"),
        "source_name": _required_str(source, "source_name"),
        "ready_to_import": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
        "input_path": _required_str(input_contract, "path"),
        "input_format": _required_str(input_contract, "format"),
        "input_status": input_status,
        "prediction_count": _int_value(input_contract.get("prediction_count")),
        "metadata_ready": metadata_ready,
        "metadata": _json_value(metadata),
        "source_metadata": _json_value(source_metadata),
        "diagnostics": _json_value(diagnostics),
        "planned_outputs": _json_value(planned_outputs),
        "source_import_command": _source_import_command(
            manifest=manifest,
            source=source,
            input_contract=input_contract,
            source_metadata=source_metadata,
            planned_outputs=planned_outputs,
        ),
    }
    error = _text_or_none(source.get("error"))
    if error is not None:
        result["error"] = error
    return result


def _artifact_launch_candidate_row(candidate: Mapping[str, Any]) -> dict[str, Any]:
    status = _required_str(candidate, "status")
    blocking_reasons = _artifact_launch_candidate_blocking_reasons(status)
    return {
        "blocking_reasons": blocking_reasons,
        "error": candidate.get("error"),
        "path": candidate.get("path"),
        "prediction_count": _int_value(candidate.get("prediction_count")),
        "qa_eval_report_path": candidate.get("qa_eval_report_path"),
        "ready_to_import": not blocking_reasons,
        "status": status,
        "valid": candidate.get("valid") is True,
    }


def _artifact_launch_candidate_blocking_reasons(status: str) -> list[str]:
    if status in ("ready", "not_requested"):
        return []
    if status == "missing":
        return ["candidate_prediction_missing"]
    if status == "invalid":
        return ["candidate_prediction_invalid"]
    return ["candidate_prediction_not_ready"]


def _artifact_launch_blocking_reasons(
    *,
    input_status: str,
    metadata_ready: bool,
    ready_to_import: bool,
) -> list[str]:
    reasons: list[str] = []
    if input_status == "missing":
        reasons.append("input_missing")
    elif input_status == "invalid":
        reasons.append("input_invalid")
    elif input_status == "diagnostic":
        reasons.append("input_diagnostics")
    if not metadata_ready:
        reasons.append("metadata_not_ready")
    if not ready_to_import and not reasons:
        reasons.append("source_not_ready")
    return sorted(reasons)


def _artifact_launch_summary(
    sources: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
) -> dict[str, int]:
    return {
        "blocked_source_count": sum(
            1 for source in sources if source.get("ready_to_import") is not True
        ),
        "candidate_blocked_count": (
            0 if candidate.get("ready_to_import") is True else 1
        ),
        "candidate_ready_count": (
            1 if candidate.get("ready_to_import") is True else 0
        ),
        "diagnostic_source_count": sum(
            1 for source in sources if source.get("input_status") == "diagnostic"
        ),
        "invalid_source_count": sum(
            1 for source in sources if source.get("input_status") == "invalid"
        ),
        "metadata_blocked_source_count": sum(
            1 for source in sources if source.get("metadata_ready") is not True
        ),
        "missing_source_count": sum(
            1 for source in sources if source.get("input_status") == "missing"
        ),
        "ready_source_count": sum(
            1 for source in sources if source.get("ready_to_import") is True
        ),
        "source_count": len(sources),
    }


def _artifact_launch_actionable_blockers(
    sources: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    blockers: dict[str, Any] = {}
    source_items = [
        {
            "blocking_reasons": list(
                _string_sequence(
                    source.get("blocking_reasons"),
                    "blocking_reasons",
                )
            ),
            "diagnostics": _json_value(source.get("diagnostics", {})),
            "input_path": _required_str(source, "input_path"),
            "input_status": _required_str(source, "input_status"),
            "metadata_ready": source.get("metadata_ready") is True,
            "source_import_command": _required_str(
                source,
                "source_import_command",
            ),
            "source_key": _required_str(source, "source_key"),
            "source_kind": _required_str(source, "source_kind"),
            "source_name": _required_str(source, "source_name"),
        }
        for source in sources
        if source.get("ready_to_import") is not True
    ]
    if source_items:
        blockers["sources"] = {
            "blocked_source_count": len(source_items),
            "items": source_items,
        }
    if candidate.get("ready_to_import") is not True:
        blockers["candidate_prediction"] = {
            "blocking_reasons": list(
                _string_sequence(
                    candidate.get("blocking_reasons"),
                    "blocking_reasons",
                )
            ),
            "error": candidate.get("error"),
            "path": candidate.get("path"),
            "qa_eval_report_path": candidate.get("qa_eval_report_path"),
            "status": _required_str(candidate, "status"),
        }
    return blockers


def _artifact_launch_external_prediction_intake_plan(
    sources: Sequence[Mapping[str, Any]],
    *,
    next_commands: Mapping[str, str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    blocked_sources: list[dict[str, Any]] = []
    source_kinds: list[str] = []
    for index, source in enumerate(sources, start=1):
        source_kind = _required_str(source, "source_kind")
        if source_kind not in source_kinds:
            source_kinds.append(source_kind)
        planned_outputs = _mapping(source.get("planned_outputs"), "planned_outputs")
        row = {
            "blocking_reasons": list(
                _string_sequence(
                    source.get("blocking_reasons"),
                    "blocking_reasons",
                )
            ),
            "import_report_path": _required_str(
                planned_outputs,
                "import_report_path",
            ),
            "input_format": _required_str(source, "input_format"),
            "input_status": _required_str(source, "input_status"),
            "normalized_prediction_path": _required_str(
                planned_outputs,
                "normalized_prediction_path",
            ),
            "order": index,
            "prediction_path": _required_str(source, "input_path"),
            "ready_to_import": source.get("ready_to_import") is True,
            "source_import_command": _required_str(source, "source_import_command"),
            "source_key": _required_str(source, "source_key"),
            "source_kind": source_kind,
            "source_metadata": _json_value(source.get("source_metadata", {})),
            "source_name": _required_str(source, "source_name"),
        }
        rows.append(row)
        if row["ready_to_import"] is not True:
            blocked_sources.append(
                {
                    "blocking_reasons": row["blocking_reasons"],
                    "input_status": row["input_status"],
                    "prediction_path": row["prediction_path"],
                    "source_import_command": row["source_import_command"],
                    "source_key": row["source_key"],
                    "source_kind": row["source_kind"],
                    "source_name": row["source_name"],
                }
            )
    return {
        "track": "real_controls",
        "atomic_import_command": _required_str(next_commands, "import"),
        "blocked_source_count": len(blocked_sources),
        "blocked_sources": blocked_sources,
        "preflight_command": _required_str(next_commands, "preflight"),
        "prediction_request_bundle_command": _required_str(
            next_commands,
            "prediction_request_bundle",
        ),
        "prediction_receipt_bundle_command": _required_str(
            next_commands,
            "prediction_receipt_bundle",
        ),
        "ready_source_count": sum(
            1 for source in sources if source.get("ready_to_import") is True
        ),
        "required_metadata_fields": list(REAL_SOURCE_METADATA_FIELDS),
        "source_count": len(sources),
        "source_kinds": source_kinds,
        "sources": rows,
    }


def _prediction_receipt_source(
    source: Mapping[str, Any],
    source_rows: Mapping[str, Mapping[str, Any]],
    *,
    order: int,
) -> dict[str, Any]:
    source_key = _required_str(source, "source_key")
    planned_outputs = _mapping(source.get("planned_outputs"), "planned_outputs")
    source_row = _mapping(source_rows.get(source_key, {}), source_key)
    row: dict[str, Any] = {
        "blocking_reasons": list(
            _string_sequence(source.get("blocking_reasons"), "blocking_reasons")
        ),
        "import_report_path": _required_str(planned_outputs, "import_report_path"),
        "input_digest": source_row.get("input_digest"),
        "input_format": _required_str(source, "input_format"),
        "input_status": _required_str(source, "input_status"),
        "normalized_prediction_path": _required_str(
            planned_outputs,
            "normalized_prediction_path",
        ),
        "order": order,
        "prediction_count": _int_value(source.get("prediction_count")),
        "prediction_path": _required_str(source, "input_path"),
        "ready_to_import": source.get("ready_to_import") is True,
        "source_import_command": _required_str(source, "source_import_command"),
        "source_key": source_key,
        "source_kind": _required_str(source, "source_kind"),
        "source_metadata": _json_value(source.get("source_metadata", {})),
        "source_name": _required_str(source, "source_name"),
    }
    error = _text_or_none(source.get("error"))
    if error is not None:
        row["error"] = error
    return row


def _prediction_receipt_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "blocking_reasons": list(
            _string_sequence(candidate.get("blocking_reasons"), "blocking_reasons")
        ),
        "path": candidate.get("path"),
        "prediction_count": _int_value(candidate.get("prediction_count")),
        "prediction_digest": _safe_qa_prediction_digest_or_none(candidate.get("path")),
        "qa_eval_report_path": candidate.get("qa_eval_report_path"),
        "ready_to_import": candidate.get("ready_to_import") is True,
        "status": _required_str(candidate, "status"),
    }


def _artifact_launch_source_import_plan(
    sources: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    *,
    next_commands: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "track": "real_controls",
        "atomic_import_command": _required_str(next_commands, "import"),
        "preflight_command": _required_str(next_commands, "preflight"),
        "candidate": {
            "blocking_reasons": list(
                _string_sequence(
                    candidate.get("blocking_reasons"),
                    "blocking_reasons",
                )
            ),
            "path": candidate.get("path"),
            "qa_eval_report_path": candidate.get("qa_eval_report_path"),
            "ready_to_import": candidate.get("ready_to_import") is True,
            "status": _required_str(candidate, "status"),
        },
        "ready_source_count": sum(
            1 for source in sources if source.get("ready_to_import") is True
        ),
        "source_commands": [
            {
                "blocking_reasons": list(
                    _string_sequence(
                        source.get("blocking_reasons"),
                        "blocking_reasons",
                    )
                ),
                "input_format": _required_str(source, "input_format"),
                "input_path": _required_str(source, "input_path"),
                "input_status": _required_str(source, "input_status"),
                "ready_to_import": source.get("ready_to_import") is True,
                "source_import_command": _required_str(
                    source,
                    "source_import_command",
                ),
                "source_key": _required_str(source, "source_key"),
                "source_kind": _required_str(source, "source_kind"),
                "source_name": _required_str(source, "source_name"),
            }
            for source in sources
        ],
        "source_count": len(sources),
    }


def _artifact_launch_next_commands(
    *,
    contracts_path: str | None,
    manifest_path: str,
) -> dict[str, str]:
    request_bundle_path = (
        Path(manifest_path).parent / "offline-control-prediction-request-bundle.json"
    )
    receipt_bundle_path = (
        Path(manifest_path).parent / "offline-control-prediction-receipt-bundle.json"
    )
    commands = {
        "import": f"python scripts/run_offline_controls.py --manifest {manifest_path}",
        "preflight": (
            "python scripts/run_offline_controls.py "
            f"--preflight-manifest {manifest_path}"
        ),
        "prediction_request_bundle": (
            "python scripts/run_offline_controls.py "
            f"--prediction-request-bundle {manifest_path} "
            f"--request-bundle-output {request_bundle_path}"
        ),
        "prediction_receipt_bundle": (
            "python scripts/run_offline_controls.py "
            f"--prediction-receipt-bundle {manifest_path} "
            f"--receipt-bundle-output {receipt_bundle_path}"
        ),
    }
    if contracts_path is not None:
        commands["compare_artifact_contracts"] = (
            "python scripts/check_offline_controls.py "
            f"--compare-artifact-contracts {contracts_path} --manifest {manifest_path}"
        )
        commands["validate_artifact_contracts"] = (
            "python scripts/check_offline_controls.py "
            f"--validate-artifact-contracts {contracts_path}"
        )
    return {key: commands[key] for key in sorted(commands)}


def _source_import_command(
    *,
    manifest: Mapping[str, Any],
    source: Mapping[str, Any],
    input_contract: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
    planned_outputs: Mapping[str, Any],
) -> str:
    parts = [
        "python scripts/import_predictions.py",
        f"--qa {_required_str(manifest, 'qa_path')}",
        f"--input {_required_str(input_contract, 'path')}",
        f"--input-format {_required_str(input_contract, 'format')}",
        f"--source-name {_required_str(source, 'source_name')}",
        f"--source-kind {_required_str(source, 'source_kind')}",
    ]
    parts.extend(
        f"--metadata {key}={value}"
        for key, value in _metadata_cli_items(source_metadata)
    )
    parts.extend(
        [
            f"--pred {_required_str(planned_outputs, 'normalized_prediction_path')}",
            f"--report {_required_str(planned_outputs, 'import_report_path')}",
        ]
    )
    return " ".join(parts)


def _metadata_cli_items(metadata: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(key), value)
        for key, value in sorted(metadata.items(), key=lambda item: str(item[0]))
        if isinstance(value, str) and value != ""
    )


def _source_eval_rows_by_key(
    qa_eval_handoff: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    rows = qa_eval_handoff.get("qa_eval_delta_reports")
    if not isinstance(rows, Sequence) or isinstance(rows, str):
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        source_key = _text_or_none(row.get("source_key"))
        if source_key is not None:
            result[source_key] = cast(Mapping[str, Any], row)
    return result


def _run_ledger_candidate(qa_eval_handoff: Mapping[str, Any]) -> dict[str, Any]:
    prediction_path = _text_or_none(qa_eval_handoff.get("candidate_prediction_path"))
    qa_eval_report_path = _text_or_none(
        qa_eval_handoff.get("candidate_qa_eval_report_path")
    )
    return {
        "prediction_digest": _qa_prediction_digest_or_none(prediction_path),
        "prediction_path": prediction_path,
        "qa_eval_report_digest": _qa_eval_report_digest_or_none(qa_eval_report_path),
        "qa_eval_report_path": qa_eval_report_path,
    }


def _run_ledger_source_row(
    source_row: Mapping[str, Any],
    source_eval_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    source_key = _required_str(source_row, "source_key")
    import_report = load_offline_prediction_import_report(
        _required_str(source_row, "import_report_path")
    )
    eval_row = source_eval_rows.get(source_key, {})
    baseline_report_path = _text_or_none(eval_row.get("baseline_qa_eval_report_path"))
    delta_report_path = _text_or_none(eval_row.get("qa_eval_delta_report_path"))
    return {
        "baseline_qa_eval_report_digest": _qa_eval_report_digest_or_none(
            baseline_report_path
        ),
        "baseline_qa_eval_report_path": baseline_report_path,
        "import_report_digest": offline_prediction_import_report_digest(import_report),
        "import_report_path": _required_str(source_row, "import_report_path"),
        "input_digest": import_report.get("input_digest"),
        "input_format": _required_str(source_row, "input_format"),
        "input_path": _required_str(source_row, "input_path"),
        "normalized_prediction_digest": _qa_prediction_digest_or_none(
            source_row.get("prediction_path")
        ),
        "normalized_prediction_path": _required_str(source_row, "prediction_path"),
        "qa_eval_delta_report_digest": _qa_eval_delta_report_digest_or_none(
            delta_report_path
        ),
        "qa_eval_delta_report_path": delta_report_path,
        "source_key": source_key,
        "source_kind": _required_str(source_row, "source_kind"),
        "source_name": _required_str(source_row, "source_name"),
        "summary": _json_value(source_row.get("summary")),
    }


def _source_artifact_contract(
    manifest: Mapping[str, Any],
    source: Mapping[str, Any],
    source_row: Mapping[str, Any],
) -> dict[str, Any]:
    source_kind = _required_str(source, "source_kind")
    source_name = _required_str(source, "source_name")
    metadata = _mapping(source.get("metadata"), "metadata")
    source_profile = offline_prediction_source_profile(
        source_name,
        source_kind,
        metadata,
    )
    source_key = _required_str(source_profile, "source_key")
    diagnostics = _mapping(source_row.get("summary"), "summary")
    input_status = _source_contract_input_status(source, source_row, diagnostics)
    metadata_contract = _source_metadata_contract(source_profile, source)
    return {
        "source_key": source_key,
        "source_kind": source_kind,
        "source_name": source_name,
        "diagnostics": diagnostics,
        "error": source_row.get("error"),
        "input": {
            "expected_schema_version": _expected_input_schema_version(
                _required_str(source, "input_format")
            ),
            "format": _required_str(source, "input_format"),
            "path": str(source["input_path"]),
            "prediction_count": _int_value(diagnostics.get("imported_prediction_count")),
            "status": input_status,
        },
        "metadata": metadata_contract,
        "source_metadata": _json_value(metadata),
        "planned_outputs": {
            "baseline_qa_eval_report_path": _baseline_qa_eval_report_path(
                manifest,
                source_key,
            ),
            "import_report_path": str(source["import_report_path"]),
            "normalized_prediction_path": str(source["prediction_path"]),
            "qa_eval_delta_report_path": _qa_eval_delta_report_path(
                manifest,
                source_key,
            ),
        },
        "ready_to_import": (
            input_status == "ready" and metadata_contract["ready"] is True
        ),
    }


def _source_contract_input_status(
    source: Mapping[str, Any],
    source_row: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> str:
    if source_row.get("valid") is True:
        if _source_missing_or_diagnostic(diagnostics):
            return "diagnostic"
        return "ready"
    if not Path(str(source["input_path"])).exists():
        return "missing"
    return "invalid"


def _candidate_status(candidate: Mapping[str, Any]) -> str:
    path = _text_or_none(candidate.get("path"))
    if path is None:
        return "not_requested"
    if candidate.get("valid") is True:
        return "ready"
    if not Path(path).exists():
        return "missing"
    return "invalid"


def _source_metadata_contract(
    source_profile: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    placeholder_identity = any(
        _contains_placeholder_marker(value)
        for value in _contract_identity_values(source_profile, source)
    )
    ready = (
        all(
            _real_profile_text(source_profile, field) is not None
            for field in REAL_SOURCE_METADATA_FIELDS
        )
        and not placeholder_identity
    )
    return {
        "dataset_id": source_profile.get("dataset_id"),
        "metadata_keys": source_profile.get("metadata_keys"),
        "model_id": source_profile.get("model_id"),
        "placeholder_identity": placeholder_identity,
        "prompt_id": source_profile.get("prompt_id"),
        "ready": ready,
    }


def _contract_identity_values(
    source_profile: Mapping[str, Any],
    source: Mapping[str, Any],
) -> tuple[str, ...]:
    values: list[str] = []
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
    for field in ("source_kind", "source_name"):
        value = source.get(field)
        if isinstance(value, str):
            values.append(value)
    return tuple(values)


def _expected_input_schema_version(input_format: str) -> str:
    if input_format == QA_PREDICTION_INPUT_FORMAT:
        return QA_PREDICTION_SCHEMA_VERSION
    return OFFLINE_PREDICTION_RECORD_SCHEMA_VERSION


def _candidate_qa_eval_report_path(manifest: Mapping[str, Any]) -> str | None:
    if manifest.get("candidate_prediction_path") is None:
        return None
    return str(
        _qa_eval_root(manifest)
        / "candidate"
        / _path_part(_required_str(manifest, "candidate_name"))
        / "qa-eval.json"
    )


def _baseline_qa_eval_report_path(
    manifest: Mapping[str, Any],
    source_key: str,
) -> str | None:
    if manifest.get("candidate_prediction_path") is None:
        return None
    return str(
        _qa_eval_root(manifest)
        / "baselines"
        / _path_part(source_key)
        / "qa-eval.json"
    )


def _qa_eval_delta_report_path(
    manifest: Mapping[str, Any],
    source_key: str,
) -> str | None:
    if manifest.get("candidate_prediction_path") is None:
        return None
    return str(
        _qa_eval_root(manifest)
        / "deltas"
        / f"{_path_part(_required_str(manifest, 'candidate_name'))}-vs-{_path_part(source_key)}.json"
    )


def _qa_eval_root(manifest: Mapping[str, Any]) -> Path:
    qa_eval_output_dir = manifest.get("qa_eval_output_dir")
    if qa_eval_output_dir is not None:
        return Path(str(qa_eval_output_dir))
    return Path(_required_str(manifest, "output_dir")) / "qa-eval"


def _candidate_preflight(path: object) -> dict[str, Any]:
    if path is None:
        return {
            "digest": None,
            "error": None,
            "path": None,
            "prediction_count": 0,
            "valid": True,
        }
    if not isinstance(path, (str, Path)):
        return {
            "digest": None,
            "error": "candidate_prediction_path must be a local path",
            "path": str(path),
            "prediction_count": 0,
            "valid": False,
        }
    try:
        predictions = tuple(load_qa_predictions(path))
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        return {
            "digest": None,
            "error": str(exc),
            "path": str(path),
            "prediction_count": 0,
            "valid": False,
        }
    return {
        "digest": qa_predictions_digest(predictions),
        "error": None,
        "path": str(path),
        "prediction_count": len(predictions),
        "valid": True,
    }


def _source_missing_or_diagnostic(summary: Mapping[str, Any]) -> bool:
    return any(
        _int_value(summary.get(field)) > 0
        for field in (
            "duplicate_case_count",
            "error_prediction_count",
            "missing_case_count",
            "unknown_case_count",
        )
    )


def _int_value(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _source_spec(
    source_spec: Mapping[str, object],
    root: Path,
) -> dict[str, Any]:
    source_kind = _required_str(source_spec, "source_kind")
    source_name = _required_str(source_spec, "source_name")
    input_path = Path(_required_path(source_spec, "input_path"))
    source_dir = root / _path_part(source_kind) / _path_part(source_name)
    prediction_path = Path(
        _optional_path(
            source_spec,
            "prediction_path",
            source_dir / "predictions.jsonl",
        )
    )
    import_report_path = Path(
        _optional_path(
            source_spec,
            "import_report_path",
            source_dir / "import-report.json",
        )
    )
    return {
        "source_kind": source_kind,
        "source_name": source_name,
        "input_format": _input_format(source_spec.get("input_format")),
        "input_path": input_path,
        "prediction_path": prediction_path,
        "import_report_path": import_report_path,
        "metadata": _metadata(source_spec.get("metadata")),
    }


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Offline control source field is required: {key}")
    return value


def _required_path(payload: Mapping[str, object], key: str) -> str | Path:
    value = payload.get(key)
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value != "":
        return value
    raise SpatialQAError(f"Offline control source path is required: {key}")


def _optional_path(
    payload: Mapping[str, object],
    key: str,
    default: Path,
) -> str | Path:
    value = payload.get(key)
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value != "":
        return value
    return default


def _metadata(value: object) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SpatialQAError("Offline control source metadata must be an object")
    return cast(Mapping[str, Any], value)


def _input_format(value: object) -> str:
    if value is None:
        return OFFLINE_PREDICTION_RECORD_INPUT_FORMAT
    if value in (
        OFFLINE_PREDICTION_RECORD_INPUT_FORMAT,
        QA_PREDICTION_INPUT_FORMAT,
    ):
        return str(value)
    raise SpatialQAError(f"Unsupported offline control source input format: {value}")


def _source_metadata_summary(
    import_reports: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    missing_metadata_source_keys: set[str] = set()
    placeholder_source_keys: set[str] = set()
    for import_report in import_reports:
        source_key = _source_key(import_report)
        source_profile = _mapping(import_report.get("source_profile"), "source_profile")
        if any(
            _real_profile_text(source_profile, field) is None
            for field in REAL_SOURCE_METADATA_FIELDS
        ):
            missing_metadata_source_keys.add(source_key)
        if any(
            _contains_placeholder_marker(value)
            for value in _source_identity_values(import_report, source_profile, source_key)
        ):
            placeholder_source_keys.add(source_key)
    return {
        "missing_metadata_source_keys": sorted(missing_metadata_source_keys),
        "placeholder_source_keys": sorted(placeholder_source_keys),
    }


def _source_metadata_checks(
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    missing = _string_list(summary.get("missing_metadata_source_keys"))
    placeholders = _string_list(summary.get("placeholder_source_keys"))
    return [
        {
            "name": "real_source_metadata_present",
            "passed": len(missing) == 0,
            "actual": missing,
        },
        {
            "name": "no_placeholder_source_identity",
            "passed": len(placeholders) == 0,
            "actual": placeholders,
        },
    ]


def _run_readiness(
    matrix_readiness: Mapping[str, Any],
    source_metadata_checks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failed_checks = _string_list(matrix_readiness.get("failed_checks"))
    failed_checks.extend(
        _required_str(check, "name")
        for check in source_metadata_checks
        if check.get("passed") is not True
    )
    return {
        "ready": matrix_readiness.get("ready") is True and len(failed_checks) == 0,
        "failed_check_count": len(failed_checks),
        "failed_checks": failed_checks,
    }


def _source_key(import_report: Mapping[str, Any]) -> str:
    source_profile = _mapping(import_report.get("source_profile"), "source_profile")
    return _required_str(source_profile, "source_key")


def _real_profile_text(
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


def _source_identity_values(
    import_report: Mapping[str, Any],
    source_profile: Mapping[str, Any],
    source_key: str,
) -> tuple[str, ...]:
    values: list[str] = [source_key]
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
    source = import_report.get("source")
    if isinstance(source, Mapping):
        for field in ("kind", "name"):
            value = source.get(field)
            if isinstance(value, str):
                values.append(value)
    return tuple(values)


def _contains_placeholder_marker(value: str) -> bool:
    normalized = value.lower()
    return any(marker in normalized for marker in PLACEHOLDER_SOURCE_MARKERS)


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpatialQAError(f"Offline control import run field must be an object: {field}")
    return cast(Mapping[str, Any], value)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [item for item in value if isinstance(item, str)]


def _contract_source_rows(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return ()
    return tuple(cast(Mapping[str, Any], item) for item in value if isinstance(item, Mapping))


def _summary_int(summary: object, key: str) -> int | None:
    if not isinstance(summary, Mapping):
        return None
    value = summary.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _text_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _source_input_digest(source: Mapping[str, Any]) -> str | None:
    input_path = _required_str(source, "input_path")
    input_format = _required_str(source, "input_format")
    if input_format == QA_PREDICTION_INPUT_FORMAT:
        return qa_predictions_digest(load_qa_predictions(input_path))
    if input_format == OFFLINE_PREDICTION_RECORD_INPUT_FORMAT:
        return offline_prediction_records_digest(
            load_offline_prediction_records(input_path)
        )
    raise SpatialQAError(f"Unsupported offline control source input format: {input_format}")


def _qa_prediction_digest_or_none(path: object) -> str | None:
    path_text = _text_or_none(path)
    if path_text is None:
        return None
    return qa_predictions_digest(load_qa_predictions(path_text))


def _safe_qa_prediction_digest_or_none(path: object) -> str | None:
    try:
        return _qa_prediction_digest_or_none(path)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError):
        return None


def _qa_eval_report_digest_or_none(path: object) -> str | None:
    path_text = _text_or_none(path)
    if path_text is None:
        return None
    return qa_eval_report_digest(load_qa_eval_report(path_text))


def _qa_eval_delta_report_digest_or_none(path: object) -> str | None:
    path_text = _text_or_none(path)
    if path_text is None:
        return None
    return qa_eval_delta_report_digest(load_qa_eval_delta_report(path_text))


def _ledger_digest_check(
    name: str,
    expected: object,
    actual: object,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": expected == actual,
        "expected": expected,
        "actual": actual,
    }


def _path_part(value: str) -> str:
    part = "".join(
        character if character.isalnum() or character in ("-", "_", ".") else "_"
        for character in value
    )
    if part == "":
        raise SpatialQAError("Offline control source path component is empty")
    return part
