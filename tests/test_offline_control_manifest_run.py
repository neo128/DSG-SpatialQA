from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from _pytest.capture import CaptureFixture

import dsg_spatialqa_lab as lab


ROOT = Path(__file__).resolve().parents[1]
RUN_OFFLINE_CONTROLS_SCRIPT = ROOT / "scripts" / "run_offline_controls.py"
CHECK_OFFLINE_CONTROLS_SCRIPT = ROOT / "scripts" / "check_offline_controls.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_run_offline_controls_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_offline_controls_manifest_script",
        RUN_OFFLINE_CONTROLS_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_check_offline_controls_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_offline_controls_contract_script",
        CHECK_OFFLINE_CONTROLS_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_offline_control_import_manifest_writes_ready_artifacts(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "OFFLINE_CONTROL_IMPORT_MANIFEST_SCHEMA_VERSION")
    assert hasattr(lab, "OFFLINE_CONTROL_IMPORT_RUN_LEDGER_SCHEMA_VERSION")
    assert hasattr(lab, "load_offline_control_import_manifest")
    assert hasattr(lab, "offline_control_import_manifest_digest")
    assert hasattr(lab, "offline_control_import_run_ledger")
    assert hasattr(lab, "offline_control_import_run_ledger_digest")
    assert hasattr(lab, "save_offline_control_import_run_ledger")
    assert hasattr(lab, "load_offline_control_import_run_ledger")
    assert hasattr(lab, "validate_offline_control_import_run_ledger")
    assert hasattr(lab, "compare_offline_control_import_run_ledger")
    assert hasattr(lab, "run_offline_control_import_manifest")
    manifest_path = _write_manifest_package(tmp_path)

    manifest = lab.load_offline_control_import_manifest(manifest_path)
    result = lab.run_offline_control_import_manifest(manifest_path)
    ledger = lab.offline_control_import_run_ledger(result)

    assert result["schema_version"] == (
        "dsg-spatialqa-lab.offline-control-import-run.v1"
    )
    assert result["action"] == "run_offline_control_import_manifest"
    assert result["manifest_schema_version"] == (
        "dsg-spatialqa-lab.offline-control-import-manifest.v1"
    )
    assert result["manifest_path"] == str(manifest_path)
    assert result["manifest_digest"] == lab.offline_control_import_manifest_digest(
        manifest
    )
    assert result["manifest_summary"] == {
        "has_candidate_prediction": True,
        "source_count": 4,
    }
    assert result["ready"] is True
    assert result["qa_eval_handoff"]["ready"] is True
    assert result["offline_control_result_report_path"] == str(
        manifest_path.parent / "offline-controls" / "offline-control-result.json"
    )
    assert result["offline_control_result_readiness"]["ready"] is True
    assert result["required_source_kinds"] == [
        "caption_memory",
        "graph_text",
        "multi_frame_vlm",
        "vlm",
    ]
    assert [source["source_kind"] for source in result["sources"]] == [
        "caption_memory",
        "graph_text",
        "multi_frame_vlm",
        "vlm",
    ]
    assert {source["input_format"] for source in result["sources"]} == {
        "qa_prediction"
    }
    matrix_report = lab.load_offline_control_matrix_report(
        result["matrix_report_path"]
    )
    assert lab.validate_offline_control_matrix_report(matrix_report)["valid"] is True
    assert lab.compare_offline_control_matrix_report(matrix_report)["matches"] is True
    assert len(result["qa_eval_delta_report_paths"]) == 4
    for source in result["sources"]:
        import_report = lab.load_offline_prediction_import_report(
            source["import_report_path"]
        )
        assert lab.validate_offline_prediction_import_report(import_report)[
            "valid"
        ] is True
        assert lab.compare_offline_prediction_import_report(import_report)[
            "matches"
        ] is True

    assert ledger["schema_version"] == (
        "dsg-spatialqa-lab.offline-control-import-run-ledger.v1"
    )
    assert ledger["ledger_digest"] == lab.offline_control_import_run_ledger_digest(
        ledger
    )
    assert ledger["run"] == {
        "candidate_qa_eval_report_path": result["candidate_qa_eval_report_path"],
        "manifest_digest": result["manifest_digest"],
        "manifest_path": str(manifest_path),
        "matrix_report_digest": result["matrix_report_digest"],
        "matrix_report_path": result["matrix_report_path"],
        "offline_control_result_report_digest": result[
            "offline_control_result_report_digest"
        ],
        "offline_control_result_report_path": result[
            "offline_control_result_report_path"
        ],
        "qa_digest": result["qa_digest"],
        "qa_path": result["qa_path"],
        "ready": True,
        "schema_version": result["schema_version"],
    }
    assert ledger["summary"] == {
        "candidate_requested": True,
        "ready_source_count": 4,
        "source_count": 4,
    }
    assert ledger["readiness"] == result["readiness"]
    assert ledger["matrix_readiness"] == result["matrix_readiness"]
    assert ledger["offline_control_result_readiness"] == (
        result["offline_control_result_readiness"]
    )
    assert ledger["candidate"] == {
        "prediction_digest": lab.qa_predictions_digest(
            lab.load_qa_predictions(
                manifest_path.parent / "candidate" / "predicted-graph-tool.jsonl"
            )
        ),
        "prediction_path": str(
            manifest_path.parent / "candidate" / "predicted-graph-tool.jsonl"
        ),
        "qa_eval_report_digest": result["qa_eval_handoff"][
            "candidate_qa_eval_report_digest"
        ],
        "qa_eval_report_path": result["candidate_qa_eval_report_path"],
    }
    vlm_ledger = next(
        source for source in ledger["sources"] if source["source_kind"] == "vlm"
    )
    vlm_result = next(
        source for source in result["sources"] if source["source_kind"] == "vlm"
    )
    vlm_import_report = lab.load_offline_prediction_import_report(
        vlm_result["import_report_path"]
    )
    assert vlm_ledger == {
        "baseline_qa_eval_report_digest": result["qa_eval_handoff"][
            "qa_eval_delta_reports"
        ][3]["baseline_qa_eval_report_digest"],
        "baseline_qa_eval_report_path": result["qa_eval_handoff"][
            "qa_eval_delta_reports"
        ][3]["baseline_qa_eval_report_path"],
        "import_report_digest": vlm_import_report["report_digest"],
        "import_report_path": vlm_result["import_report_path"],
        "input_digest": vlm_import_report["input_digest"],
        "input_format": "qa_prediction",
        "input_path": vlm_result["input_path"],
        "normalized_prediction_digest": vlm_result["prediction_digest"],
        "normalized_prediction_path": vlm_result["prediction_path"],
        "qa_eval_delta_report_digest": result["qa_eval_handoff"][
            "qa_eval_delta_reports"
        ][3]["qa_eval_delta_report_digest"],
        "qa_eval_delta_report_path": result["qa_eval_handoff"][
            "qa_eval_delta_reports"
        ][3]["qa_eval_delta_report_path"],
        "source_key": vlm_result["source_key"],
        "source_kind": "vlm",
        "source_name": vlm_result["source_name"],
        "summary": vlm_result["summary"],
    }
    ledger_path = tmp_path / "handoff" / "offline-control-import-run-ledger.json"
    saved_path = lab.save_offline_control_import_run_ledger(ledger, ledger_path)
    loaded_ledger = lab.load_offline_control_import_run_ledger(ledger_path)
    validation = lab.validate_offline_control_import_run_ledger(loaded_ledger)
    comparison = lab.compare_offline_control_import_run_ledger(loaded_ledger)
    assert saved_path == ledger_path
    assert loaded_ledger == ledger
    assert validation["valid"] is True
    assert comparison["matches"] is True
    assert comparison["saved_digest"] == ledger["ledger_digest"]


def test_offline_control_import_manifest_preflight_reports_ready_sources(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "OFFLINE_CONTROL_IMPORT_PREFLIGHT_SCHEMA_VERSION")
    assert hasattr(lab, "offline_control_import_manifest_preflight")
    assert hasattr(lab, "offline_control_artifact_contracts_digest")
    assert hasattr(lab, "save_offline_control_artifact_contracts")
    assert hasattr(lab, "load_offline_control_artifact_contracts")
    assert hasattr(lab, "validate_offline_control_artifact_contracts")
    assert hasattr(lab, "compare_offline_control_artifact_contracts")
    assert hasattr(lab, "OFFLINE_CONTROL_ARTIFACT_LAUNCH_REPORT_SCHEMA_VERSION")
    assert hasattr(lab, "offline_control_artifact_launch_report")
    assert hasattr(lab, "offline_control_artifact_launch_report_digest")
    manifest_path = _write_manifest_package(tmp_path)
    manifest = lab.load_offline_control_import_manifest(manifest_path)

    result = lab.offline_control_import_manifest_preflight(manifest_path)

    assert result["schema_version"] == (
        "dsg-spatialqa-lab.offline-control-import-preflight.v1"
    )
    assert result["action"] == "offline_control_import_manifest_preflight"
    assert result["manifest_path"] == str(manifest_path)
    assert result["manifest_digest"] == lab.offline_control_import_manifest_digest(
        manifest
    )
    assert result["ready_to_import"] is True
    assert result["summary"] == {
        "candidate_prediction_count": 3,
        "invalid_source_count": 0,
        "missing_source_count": 0,
        "qa_case_count": 3,
        "source_count": 4,
    }
    assert result["matrix_readiness"]["ready"] is True
    assert result["planned_outputs"]["result_report_path"] == str(
        manifest_path.parent / "offline-controls" / "offline-control-result.json"
    )
    contracts = result["artifact_contracts"]
    assert contracts["schema_version"] == (
        "dsg-spatialqa-lab.offline-control-artifact-contracts.v1"
    )
    assert contracts["contracts_digest"] == lab.offline_control_artifact_contracts_digest(
        contracts
    )
    assert contracts["qa"] == {
        "case_count": 3,
        "digest": result["qa_digest"],
        "path": str(manifest["qa_path"]),
    }
    assert contracts["candidate"] == {
        "error": None,
        "path": str(manifest_path.parent / "candidate" / "predicted-graph-tool.jsonl"),
        "prediction_count": 3,
        "qa_eval_report_path": str(
            manifest_path.parent
            / "offline-controls"
            / "qa-eval"
            / "candidate"
            / "predicted_graph_tool"
            / "qa-eval.json"
        ),
        "status": "ready",
        "valid": True,
    }
    assert contracts["summary"] == {
        "ready_source_contract_count": 4,
        "source_contract_count": 4,
    }
    assert [source["source_kind"] for source in contracts["sources"]] == [
        "caption_memory",
        "graph_text",
        "multi_frame_vlm",
        "vlm",
    ]
    vlm_contract = next(
        source for source in contracts["sources"] if source["source_kind"] == "vlm"
    )
    assert vlm_contract["input"] == {
        "expected_schema_version": "dsg-spatialqa-lab.qa-prediction.v1",
        "format": "qa_prediction",
        "path": str(manifest_path.parent / "inputs" / "llava16_ai2thor_trial.jsonl"),
        "prediction_count": 3,
        "status": "ready",
    }
    assert vlm_contract["metadata"] == {
        "dataset_id": "ai2thor-real-trial-v1",
        "metadata_keys": [
            "capabilities",
            "dataset_id",
            "model_id",
            "prompt_id",
        ],
        "model_id": "llava-v1.6-34b",
        "placeholder_identity": False,
        "prompt_id": "vlm-spatial-qa-v1",
        "ready": True,
    }
    assert vlm_contract["planned_outputs"] == {
        "baseline_qa_eval_report_path": str(
            manifest_path.parent
            / "offline-controls"
            / "qa-eval"
            / "baselines"
            / "vlm_llava16_ai2thor_trial"
            / "qa-eval.json"
        ),
        "import_report_path": str(
            manifest_path.parent
            / "offline-controls"
            / "vlm"
            / "llava16_ai2thor_trial"
            / "import-report.json"
        ),
        "normalized_prediction_path": str(
            manifest_path.parent
            / "offline-controls"
            / "vlm"
            / "llava16_ai2thor_trial"
            / "predictions.jsonl"
        ),
        "qa_eval_delta_report_path": str(
            manifest_path.parent
            / "offline-controls"
            / "qa-eval"
            / "deltas"
            / "predicted_graph_tool-vs-vlm_llava16_ai2thor_trial.json"
        ),
    }
    assert vlm_contract["ready_to_import"] is True
    assert not Path(
        vlm_contract["planned_outputs"]["normalized_prediction_path"]
    ).exists()
    assert not Path(
        vlm_contract["planned_outputs"]["baseline_qa_eval_report_path"]
    ).exists()
    contracts_path = tmp_path / "handoff" / "offline-control-artifact-contracts.json"
    saved_path = lab.save_offline_control_artifact_contracts(contracts, contracts_path)
    loaded_contracts = lab.load_offline_control_artifact_contracts(contracts_path)
    validation = lab.validate_offline_control_artifact_contracts(loaded_contracts)
    comparison = lab.compare_offline_control_artifact_contracts(
        loaded_contracts,
        manifest_path,
    )
    assert saved_path == contracts_path
    assert loaded_contracts == contracts
    assert loaded_contracts["contracts_digest"] == (
        lab.offline_control_artifact_contracts_digest(loaded_contracts)
    )
    assert validation["valid"] is True
    assert comparison["matches"] is True
    assert comparison["saved_digest"] == contracts["contracts_digest"]
    assert comparison["current_digest"] == contracts["contracts_digest"]
    launch_report = lab.offline_control_artifact_launch_report(
        loaded_contracts,
        manifest_path=manifest_path,
        contracts_path=contracts_path,
    )
    assert launch_report["schema_version"] == (
        "dsg-spatialqa-lab.offline-control-artifact-launch-report.v1"
    )
    assert launch_report["action"] == "offline_control_artifact_launch_report"
    assert launch_report["contracts_path"] == str(contracts_path)
    assert launch_report["manifest_path"] == str(manifest_path)
    assert launch_report["contracts_digest"] == contracts["contracts_digest"]
    assert launch_report["ready_to_import"] is True
    assert launch_report["summary"] == {
        "blocked_source_count": 0,
        "candidate_blocked_count": 0,
        "candidate_ready_count": 1,
        "diagnostic_source_count": 0,
        "invalid_source_count": 0,
        "metadata_blocked_source_count": 0,
        "missing_source_count": 0,
        "ready_source_count": 4,
        "source_count": 4,
    }
    assert launch_report["actionable_blockers"] == {}
    assert launch_report["candidate"] == {
        "blocking_reasons": [],
        "error": None,
        "path": str(manifest_path.parent / "candidate" / "predicted-graph-tool.jsonl"),
        "prediction_count": 3,
        "qa_eval_report_path": str(
            manifest_path.parent
            / "offline-controls"
            / "qa-eval"
            / "candidate"
            / "predicted_graph_tool"
            / "qa-eval.json"
        ),
        "ready_to_import": True,
        "status": "ready",
        "valid": True,
    }
    assert [source["source_kind"] for source in launch_report["sources"]] == [
        "caption_memory",
        "graph_text",
        "multi_frame_vlm",
        "vlm",
    ]
    assert all(source["blocking_reasons"] == [] for source in launch_report["sources"])
    vlm_launch_source = next(
        source for source in launch_report["sources"] if source["source_kind"] == "vlm"
    )
    assert vlm_launch_source["source_metadata"] == {
        "capabilities": ["spatial_qa", "dynamic_memory"],
        "dataset_id": "ai2thor-real-trial-v1",
        "model_id": "llava-v1.6-34b",
        "prompt_id": "vlm-spatial-qa-v1",
    }
    assert vlm_launch_source["source_import_command"] == (
        "python scripts/import_predictions.py "
        f"--qa {manifest['qa_path']} "
        f"--input {vlm_launch_source['input_path']} "
        "--input-format qa_prediction "
        "--source-name llava16_ai2thor_trial "
        "--source-kind vlm "
        "--metadata dataset_id=ai2thor-real-trial-v1 "
        "--metadata model_id=llava-v1.6-34b "
        "--metadata prompt_id=vlm-spatial-qa-v1 "
        f"--pred {vlm_launch_source['planned_outputs']['normalized_prediction_path']} "
        f"--report {vlm_launch_source['planned_outputs']['import_report_path']}"
    )
    assert launch_report["source_import_plan"]["track"] == "real_controls"
    assert launch_report["source_import_plan"]["source_count"] == 4
    assert launch_report["source_import_plan"]["ready_source_count"] == 4
    assert launch_report["source_import_plan"]["atomic_import_command"] == (
        f"python scripts/run_offline_controls.py --manifest {manifest_path}"
    )
    assert launch_report["source_import_plan"]["preflight_command"] == (
        "python scripts/run_offline_controls.py "
        f"--preflight-manifest {manifest_path}"
    )
    assert launch_report["source_import_plan"]["candidate"] == {
        "blocking_reasons": [],
        "path": str(manifest_path.parent / "candidate" / "predicted-graph-tool.jsonl"),
        "qa_eval_report_path": str(
            manifest_path.parent
            / "offline-controls"
            / "qa-eval"
            / "candidate"
            / "predicted_graph_tool"
            / "qa-eval.json"
        ),
        "ready_to_import": True,
        "status": "ready",
    }
    vlm_plan_row = next(
        row
        for row in launch_report["source_import_plan"]["source_commands"]
        if row["source_kind"] == "vlm"
    )
    assert vlm_plan_row == {
        "blocking_reasons": [],
        "input_format": "qa_prediction",
        "input_path": str(manifest_path.parent / "inputs" / "llava16_ai2thor_trial.jsonl"),
        "input_status": "ready",
        "ready_to_import": True,
        "source_import_command": vlm_launch_source["source_import_command"],
        "source_key": "vlm:llava16_ai2thor_trial",
        "source_kind": "vlm",
        "source_name": "llava16_ai2thor_trial",
    }
    prediction_intake_plan = launch_report["external_prediction_intake_plan"]
    assert prediction_intake_plan["track"] == "real_controls"
    assert prediction_intake_plan["source_count"] == 4
    assert prediction_intake_plan["ready_source_count"] == 4
    assert prediction_intake_plan["blocked_source_count"] == 0
    assert prediction_intake_plan["blocked_sources"] == []
    assert prediction_intake_plan["source_kinds"] == [
        "caption_memory",
        "graph_text",
        "multi_frame_vlm",
        "vlm",
    ]
    assert prediction_intake_plan["required_metadata_fields"] == [
        "model_id",
        "prompt_id",
        "dataset_id",
    ]
    assert prediction_intake_plan["atomic_import_command"] == (
        f"python scripts/run_offline_controls.py --manifest {manifest_path}"
    )
    assert prediction_intake_plan["preflight_command"] == (
        "python scripts/run_offline_controls.py "
        f"--preflight-manifest {manifest_path}"
    )
    assert prediction_intake_plan["prediction_request_bundle_command"] == (
        "python scripts/run_offline_controls.py "
        f"--prediction-request-bundle {manifest_path} "
        f"--request-bundle-output "
        f"{manifest_path.parent / 'offline-control-prediction-request-bundle.json'}"
    )
    assert prediction_intake_plan["prediction_receipt_bundle_command"] == (
        "python scripts/run_offline_controls.py "
        f"--prediction-receipt-bundle {manifest_path} "
        f"--receipt-bundle-output "
        f"{manifest_path.parent / 'offline-control-prediction-receipt-bundle.json'}"
    )
    vlm_prediction_intake_row = next(
        row
        for row in prediction_intake_plan["sources"]
        if row["source_kind"] == "vlm"
    )
    assert vlm_prediction_intake_row == {
        "blocking_reasons": [],
        "import_report_path": (
            vlm_launch_source["planned_outputs"]["import_report_path"]
        ),
        "input_format": "qa_prediction",
        "input_status": "ready",
        "normalized_prediction_path": (
            vlm_launch_source["planned_outputs"]["normalized_prediction_path"]
        ),
        "order": 4,
        "prediction_path": str(
            manifest_path.parent / "inputs" / "llava16_ai2thor_trial.jsonl"
        ),
        "ready_to_import": True,
        "source_import_command": vlm_launch_source["source_import_command"],
        "source_key": "vlm:llava16_ai2thor_trial",
        "source_kind": "vlm",
        "source_metadata": {
            "capabilities": ["spatial_qa", "dynamic_memory"],
            "dataset_id": "ai2thor-real-trial-v1",
            "model_id": "llava-v1.6-34b",
            "prompt_id": "vlm-spatial-qa-v1",
        },
        "source_name": "llava16_ai2thor_trial",
    }
    assert launch_report["next_commands"] == {
        "compare_artifact_contracts": (
            "python scripts/check_offline_controls.py "
            f"--compare-artifact-contracts {contracts_path} --manifest {manifest_path}"
        ),
        "import": f"python scripts/run_offline_controls.py --manifest {manifest_path}",
        "preflight": (
            "python scripts/run_offline_controls.py "
            f"--preflight-manifest {manifest_path}"
        ),
        "prediction_request_bundle": (
            "python scripts/run_offline_controls.py "
            f"--prediction-request-bundle {manifest_path} "
            f"--request-bundle-output "
            f"{manifest_path.parent / 'offline-control-prediction-request-bundle.json'}"
        ),
        "prediction_receipt_bundle": (
            "python scripts/run_offline_controls.py "
            f"--prediction-receipt-bundle {manifest_path} "
            f"--receipt-bundle-output "
            f"{manifest_path.parent / 'offline-control-prediction-receipt-bundle.json'}"
        ),
        "validate_artifact_contracts": (
            "python scripts/check_offline_controls.py "
            f"--validate-artifact-contracts {contracts_path}"
        ),
    }
    assert launch_report["report_digest"] == (
        lab.offline_control_artifact_launch_report_digest(launch_report)
    )
    assert result["source_metadata_summary"] == {
        "missing_metadata_source_keys": [],
        "placeholder_source_keys": [],
    }
    assert [source["source_kind"] for source in result["sources"]] == [
        "caption_memory",
        "graph_text",
        "multi_frame_vlm",
        "vlm",
    ]
    assert {source["summary"]["missing_case_count"] for source in result["sources"]} == {
        0
    }
    assert not Path(result["planned_outputs"]["matrix_report_path"]).exists()
    assert not Path(result["planned_outputs"]["output_dir"]).exists()


def test_offline_control_prediction_request_bundle_exports_external_baseline_templates(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    assert hasattr(lab, "OFFLINE_CONTROL_PREDICTION_REQUEST_BUNDLE_SCHEMA_VERSION")
    assert hasattr(lab, "offline_control_prediction_request_bundle")
    assert hasattr(lab, "offline_control_prediction_request_bundle_digest")
    assert hasattr(lab, "save_offline_control_prediction_request_bundle")
    assert hasattr(lab, "load_offline_control_prediction_request_bundle")
    manifest_path = _write_manifest_package(tmp_path)
    manifest = lab.load_offline_control_import_manifest(manifest_path)
    cases = tuple(lab.load_qa_dataset(manifest["qa_path"]))

    bundle = lab.offline_control_prediction_request_bundle(manifest_path)

    assert bundle["schema_version"] == (
        "dsg-spatialqa-lab.offline-control-prediction-request-bundle.v1"
    )
    assert bundle["action"] == "offline_control_prediction_request_bundle"
    assert bundle["manifest_path"] == str(manifest_path)
    assert bundle["manifest_digest"] == lab.offline_control_import_manifest_digest(
        manifest
    )
    assert bundle["qa"] == {
        "case_count": 3,
        "digest": lab.qa_dataset_digest(cases),
        "path": str(manifest["qa_path"]),
    }
    assert bundle["frame_index"] == {
        "available": True,
        "frame_count": len(
            {(case.episode_id, case.scene_id, case.step) for case in cases}
        ),
        "path": str(manifest["frame_index_path"]),
    }
    assert bundle["expected_input_formats"] == {
        "offline_prediction_record": (
            "dsg-spatialqa-lab.offline-prediction-record.v1"
        ),
        "qa_prediction": "dsg-spatialqa-lab.qa-prediction.v1",
    }
    assert bundle["required_metadata_fields"] == [
        "model_id",
        "prompt_id",
        "dataset_id",
    ]
    assert bundle["source_count"] == 4
    assert [source["source_kind"] for source in bundle["sources"]] == [
        "caption_memory",
        "graph_text",
        "multi_frame_vlm",
        "vlm",
    ]
    vlm_source = next(
        source for source in bundle["sources"] if source["source_kind"] == "vlm"
    )
    assert vlm_source == {
        "expected_schema_version": "dsg-spatialqa-lab.qa-prediction.v1",
        "input_format": "qa_prediction",
        "prediction_output_path": str(
            manifest_path.parent / "inputs" / "llava16_ai2thor_trial.jsonl"
        ),
        "source_key": "vlm:llava16_ai2thor_trial",
        "source_kind": "vlm",
        "source_metadata": {
            "capabilities": ["spatial_qa", "dynamic_memory"],
            "dataset_id": "ai2thor-real-trial-v1",
            "model_id": "llava-v1.6-34b",
            "prompt_id": "vlm-spatial-qa-v1",
        },
        "source_name": "llava16_ai2thor_trial",
    }
    assert bundle["case_count"] == 3
    first_case_request = bundle["case_inputs"][0]
    assert {
        key: first_case_request[key]
        for key in (
            "answer_type",
            "case_id",
            "choices",
            "difficulty",
            "episode_id",
            "question",
            "question_type",
            "reference_frame",
            "scene_id",
            "step",
            "tags",
        )
    } == {
        "answer_type": cases[0].answer_type,
        "case_id": cases[0].id,
        "choices": list(cases[0].choices),
        "difficulty": cases[0].difficulty,
        "episode_id": cases[0].episode_id,
        "question": cases[0].question,
        "question_type": cases[0].question_type,
        "reference_frame": cases[0].reference_frame,
        "scene_id": cases[0].scene_id,
        "step": cases[0].step,
        "tags": list(cases[0].tags),
    }
    assert first_case_request["question_text"].startswith("Where is the ")
    assert first_case_request["target"]["object_id"] == cases[0].question["object_id"]
    assert first_case_request["target"]["label"] in first_case_request["question_text"]
    assert first_case_request["primary_frame"] == {
        "depth_digest": f"depth-digest-{cases[0].step:04d}",
        "depth_path": f"frames/{cases[0].episode_id}/{cases[0].step:04d}.depth.npy",
        "episode_id": cases[0].episode_id,
        "frame_id": f"{cases[0].episode_id}:{cases[0].scene_id}:{cases[0].step:04d}",
        "rgb_digest": f"rgb-digest-{cases[0].step:04d}",
        "rgb_path": f"frames/{cases[0].episode_id}/{cases[0].step:04d}.rgb.ppm",
        "scene_id": cases[0].scene_id,
        "segmentation_digest": f"seg-digest-{cases[0].step:04d}",
        "segmentation_path": f"frames/{cases[0].episode_id}/{cases[0].step:04d}.seg.ppm",
        "step": cases[0].step,
    }
    assert first_case_request["frames"] == [first_case_request["primary_frame"]]
    assert "visible_object_ids" not in json.dumps(first_case_request, sort_keys=True)
    assert "visible_object_labels" not in json.dumps(first_case_request, sort_keys=True)
    assert first_case_request["answer_schema_hint"]["answer_type"] == (
        cases[0].answer_type
    )
    assert "current_location" in first_case_request["answer_schema_hint"][
        "required_answer_fields"
    ]
    assert bundle["control_prompt_guidance"]["vlm"]["question_text_required"] is True
    object_location_answer = cast(dict[str, object], cases[0].answer)
    current_location = cast(
        dict[str, object],
        object_location_answer["current_location"],
    )
    assert str(current_location["dst"]) not in json.dumps(
        first_case_request,
        sort_keys=True,
    )
    assert "answer" not in first_case_request
    assert "required_nodes" not in first_case_request
    assert "required_edges" not in first_case_request
    forbidden_gold_keys = {
        "gold_answer",
        "gold_evidence",
        "gold_evidence_edges",
        "gold_evidence_nodes",
        "required_edges",
        "required_nodes",
    }
    for case_request in bundle["case_inputs"]:
        assert forbidden_gold_keys.isdisjoint(case_request)
    qa_template = bundle["prediction_templates"]["qa_prediction"][0]
    assert qa_template == {
        "answer": {},
        "confidence": 0.0,
        "error": None,
        "evidence_edges": [],
        "evidence_nodes": [],
        "id": cases[0].id,
        "schema_version": "dsg-spatialqa-lab.qa-prediction.v1",
    }
    offline_template = bundle["prediction_templates"]["offline_prediction_record"][0]
    assert offline_template == {
        "answer": {},
        "case_id": cases[0].id,
        "confidence": 0.0,
        "error": None,
        "evidence_edges": [],
        "evidence_nodes": [],
        "metadata": {},
        "schema_version": "dsg-spatialqa-lab.offline-prediction-record.v1",
    }
    assert bundle["request_bundle_digest"] == (
        lab.offline_control_prediction_request_bundle_digest(bundle)
    )
    bundle_path = tmp_path / "handoff" / "offline-control-prediction-request-bundle.json"
    saved_path = lab.save_offline_control_prediction_request_bundle(
        bundle,
        bundle_path,
    )
    loaded_bundle = lab.load_offline_control_prediction_request_bundle(bundle_path)
    assert saved_path == bundle_path
    assert loaded_bundle == bundle

    script = load_run_offline_controls_script()
    main = cast(MainFn, script.main)
    cli_bundle_path = tmp_path / "handoff" / "cli-request-bundle.json"
    exit_code = main(
        [
            "--prediction-request-bundle",
            str(manifest_path),
            "--request-bundle-output",
            str(cli_bundle_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["action"] == "offline_control_prediction_request_bundle"
    assert payload["manifest_path"] == str(manifest_path)
    assert payload["request_bundle_path"] == str(cli_bundle_path)
    assert payload["bundle"] == bundle
    assert lab.load_offline_control_prediction_request_bundle(cli_bundle_path) == bundle


def test_prediction_request_bundle_selects_target_visible_primary_frame_without_leaking_visibility(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest_package(tmp_path)
    manifest = lab.load_offline_control_import_manifest(manifest_path)
    cases = tuple(lab.load_qa_dataset(manifest["qa_path"]))
    case = cases[0]
    target_id = cast(str, case.question["object_id"])
    frame_index_path = Path(cast(str, manifest["frame_index_path"]))
    _write_custom_frame_index(
        frame_index_path,
        rows=(
            _frame_index_row(case, visible_object_ids=("other_object",)),
            _frame_index_row(
                case,
                step=case.step + 1,
                visible_object_ids=(target_id,),
            ),
        ),
    )

    bundle = lab.offline_control_prediction_request_bundle(manifest_path)
    first_case_request = bundle["case_inputs"][0]

    assert first_case_request["primary_frame"]["step"] == case.step + 1
    assert first_case_request["primary_frame"]["rgb_path"] == (
        f"frames/{case.episode_id}/{case.step + 1:04d}.rgb.ppm"
    )
    serialized = json.dumps(first_case_request, sort_keys=True)
    assert "visible_object_ids" not in serialized
    assert "visible_object_labels" not in serialized
    assert "other_object" not in serialized


def test_prediction_request_bundle_can_use_detector_vlm_frame_index_without_leakage(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest_package(tmp_path)
    manifest = lab.load_offline_control_import_manifest(manifest_path)
    cases = tuple(lab.load_qa_dataset(manifest["qa_path"]))
    case = cases[0]
    target_id = cast(str, case.question["object_id"])
    frame_index_path = Path(cast(str, manifest["frame_index_path"]))
    base_rows = [
        _frame_index_row(case, visible_object_ids=("other_object",)),
    ]
    detector_rows = lab.vlm_frame_index_rows_from_detector_records(
        [
            {
                "episode_id": case.episode_id,
                "scene_id": case.scene_id,
                "step": case.step + 20,
                "rgb_path": f"coverage/{case.episode_id}/0020.rgb.ppm",
                "depth_path": f"coverage/{case.episode_id}/0020.depth.npy",
                "segmentation_path": f"coverage/{case.episode_id}/0020.seg.ppm",
                "metadata": {"dataset_id": "coverage-visible-p2"},
                "detections": [
                    {
                        "object_id": target_id,
                        "label": "target label should not leak",
                        "visible": True,
                    }
                ],
            }
        ]
    )
    merged_rows = lab.merge_vlm_frame_index_rows(base_rows, detector_rows)
    lab.save_vlm_frame_index_rows(merged_rows, frame_index_path)

    bundle = lab.offline_control_prediction_request_bundle(manifest_path)
    first_case_request = bundle["case_inputs"][0]

    assert first_case_request["primary_frame"]["step"] == case.step + 20
    assert first_case_request["primary_frame"]["rgb_path"] == (
        f"coverage/{case.episode_id}/0020.rgb.ppm"
    )
    serialized = json.dumps(first_case_request, sort_keys=True)
    assert "visible_object_ids" not in serialized
    assert "visible_object_labels" not in serialized
    assert "other_object" not in serialized
    assert "target label should not leak" not in serialized


def test_offline_control_prediction_receipt_bundle_exports_returned_file_receipts(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    assert hasattr(lab, "OFFLINE_CONTROL_PREDICTION_RECEIPT_BUNDLE_SCHEMA_VERSION")
    assert hasattr(lab, "offline_control_prediction_receipt_bundle")
    assert hasattr(lab, "offline_control_prediction_receipt_bundle_digest")
    assert hasattr(lab, "save_offline_control_prediction_receipt_bundle")
    assert hasattr(lab, "load_offline_control_prediction_receipt_bundle")
    assert hasattr(lab, "validate_offline_control_prediction_receipt_bundle")
    manifest_path = _write_manifest_package(tmp_path)
    manifest = lab.load_offline_control_import_manifest(manifest_path)
    cases = tuple(lab.load_qa_dataset(manifest["qa_path"]))
    candidate_predictions = tuple(
        lab.load_qa_predictions(manifest["candidate_prediction_path"])
    )

    bundle = lab.offline_control_prediction_receipt_bundle(manifest_path)

    assert bundle["schema_version"] == (
        "dsg-spatialqa-lab.offline-control-prediction-receipt-bundle.v1"
    )
    assert bundle["action"] == "offline_control_prediction_receipt_bundle"
    assert bundle["manifest_path"] == str(manifest_path)
    assert bundle["manifest_digest"] == lab.offline_control_import_manifest_digest(
        manifest
    )
    assert bundle["ready_to_import"] is True
    assert bundle["qa"] == {
        "case_count": 3,
        "digest": lab.qa_dataset_digest(cases),
        "path": str(manifest["qa_path"]),
    }
    assert bundle["summary"] == {
        "blocked_source_count": 0,
        "candidate_ready": True,
        "candidate_status": "ready",
        "ready_source_count": 4,
        "source_count": 4,
    }
    assert bundle["required_metadata_fields"] == [
        "model_id",
        "prompt_id",
        "dataset_id",
    ]
    assert bundle["commands"] == {
        "import": f"python scripts/run_offline_controls.py --manifest {manifest_path}",
        "prediction_request_bundle": (
            "python scripts/run_offline_controls.py "
            f"--prediction-request-bundle {manifest_path} "
            f"--request-bundle-output "
            f"{manifest_path.parent / 'offline-control-prediction-request-bundle.json'}"
        ),
        "preflight": (
            "python scripts/run_offline_controls.py "
            f"--preflight-manifest {manifest_path}"
        ),
    }
    assert bundle["candidate"] == {
        "blocking_reasons": [],
        "path": str(manifest_path.parent / "candidate" / "predicted-graph-tool.jsonl"),
        "prediction_count": 3,
        "prediction_digest": lab.qa_predictions_digest(candidate_predictions),
        "qa_eval_report_path": str(
            manifest_path.parent
            / "offline-controls"
            / "qa-eval"
            / "candidate"
            / "predicted_graph_tool"
            / "qa-eval.json"
        ),
        "ready_to_import": True,
        "status": "ready",
    }
    assert [source["source_kind"] for source in bundle["sources"]] == [
        "caption_memory",
        "graph_text",
        "multi_frame_vlm",
        "vlm",
    ]
    vlm_source = next(
        source for source in bundle["sources"] if source["source_kind"] == "vlm"
    )
    vlm_input_path = manifest_path.parent / "inputs" / "llava16_ai2thor_trial.jsonl"
    vlm_predictions = tuple(lab.load_qa_predictions(vlm_input_path))
    assert vlm_source == {
        "blocking_reasons": [],
        "import_report_path": str(
            manifest_path.parent
            / "offline-controls"
            / "vlm"
            / "llava16_ai2thor_trial"
            / "import-report.json"
        ),
        "input_digest": lab.qa_predictions_digest(vlm_predictions),
        "input_format": "qa_prediction",
        "input_status": "ready",
        "normalized_prediction_path": str(
            manifest_path.parent
            / "offline-controls"
            / "vlm"
            / "llava16_ai2thor_trial"
            / "predictions.jsonl"
        ),
        "order": 4,
        "prediction_count": 3,
        "prediction_path": str(vlm_input_path),
        "ready_to_import": True,
        "source_import_command": (
            "python scripts/import_predictions.py "
            f"--qa {manifest['qa_path']} "
            f"--input {vlm_input_path} "
            "--input-format qa_prediction "
            "--source-name llava16_ai2thor_trial "
            "--source-kind vlm "
            "--metadata dataset_id=ai2thor-real-trial-v1 "
            "--metadata model_id=llava-v1.6-34b "
            "--metadata prompt_id=vlm-spatial-qa-v1 "
            f"--pred {manifest_path.parent / 'offline-controls' / 'vlm' / 'llava16_ai2thor_trial' / 'predictions.jsonl'} "
            f"--report {manifest_path.parent / 'offline-controls' / 'vlm' / 'llava16_ai2thor_trial' / 'import-report.json'}"
        ),
        "source_key": "vlm:llava16_ai2thor_trial",
        "source_kind": "vlm",
        "source_metadata": {
            "capabilities": ["spatial_qa", "dynamic_memory"],
            "dataset_id": "ai2thor-real-trial-v1",
            "model_id": "llava-v1.6-34b",
            "prompt_id": "vlm-spatial-qa-v1",
        },
        "source_name": "llava16_ai2thor_trial",
    }
    assert bundle["receipt_bundle_digest"] == (
        lab.offline_control_prediction_receipt_bundle_digest(bundle)
    )
    bundle_path = tmp_path / "handoff" / "offline-control-prediction-receipt-bundle.json"
    saved_path = lab.save_offline_control_prediction_receipt_bundle(
        bundle,
        bundle_path,
    )
    loaded_bundle = lab.load_offline_control_prediction_receipt_bundle(bundle_path)
    assert saved_path == bundle_path
    assert loaded_bundle == bundle
    validation = lab.validate_offline_control_prediction_receipt_bundle(loaded_bundle)
    assert validation["action"] == "validate_offline_control_prediction_receipt_bundle"
    assert validation["valid"] is True
    assert validation["receipt_bundle_digest"] == bundle["receipt_bundle_digest"]

    tampered_bundle = json.loads(json.dumps(bundle))
    tampered_bundle["summary"]["ready_source_count"] = 3
    tampered_bundle["receipt_bundle_digest"] = (
        lab.offline_control_prediction_receipt_bundle_digest(tampered_bundle)
    )
    tampered_validation = lab.validate_offline_control_prediction_receipt_bundle(
        tampered_bundle
    )
    assert tampered_validation["valid"] is False
    assert {
        check["name"]
        for check in tampered_validation["checks"]
        if check["passed"] is False
    } == {"summary"}

    script = load_run_offline_controls_script()
    main = cast(MainFn, script.main)
    cli_bundle_path = tmp_path / "handoff" / "cli-receipt-bundle.json"
    exit_code = main(
        [
            "--prediction-receipt-bundle",
            str(manifest_path),
            "--receipt-bundle-output",
            str(cli_bundle_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["action"] == "offline_control_prediction_receipt_bundle"
    assert payload["manifest_path"] == str(manifest_path)
    assert payload["receipt_bundle_path"] == str(cli_bundle_path)
    assert payload["bundle"] == bundle
    assert lab.load_offline_control_prediction_receipt_bundle(cli_bundle_path) == bundle

    assert (
        main(["--validate-prediction-receipt-bundle", str(cli_bundle_path)])
        == 0
    )
    validation_payload = json.loads(capsys.readouterr().out)
    assert validation_payload["action"] == (
        "validate_offline_control_prediction_receipt_bundle"
    )
    assert validation_payload["path"] == str(cli_bundle_path)
    assert validation_payload["valid"] is True

    tampered_path = tmp_path / "handoff" / "tampered-receipt-bundle.json"
    lab.save_offline_control_prediction_receipt_bundle(
        tampered_bundle,
        tampered_path,
    )
    assert (
        main(["--validate-prediction-receipt-bundle", str(tampered_path)])
        == 1
    )
    tampered_payload = json.loads(capsys.readouterr().out)
    assert tampered_payload["action"] == (
        "validate_offline_control_prediction_receipt_bundle"
    )
    assert tampered_payload["path"] == str(tampered_path)
    assert tampered_payload["valid"] is False


def test_offline_control_import_manifest_preflight_reports_incomplete_source(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest_package(tmp_path)
    manifest = lab.load_offline_control_import_manifest(manifest_path)
    first_source = manifest["sources"][0]
    input_path = Path(str(first_source["input_path"]))
    predictions = tuple(lab.load_qa_predictions(input_path))
    lab.save_qa_predictions(predictions[:-1], input_path)

    result = lab.offline_control_import_manifest_preflight(manifest_path)

    assert result["ready_to_import"] is False
    assert result["summary"]["missing_source_count"] == 1
    assert result["matrix_readiness"]["ready"] is False
    missing_sources = result["missing_sources"]
    assert missing_sources == [first_source["source_name"]]
    source = next(
        item
        for item in result["sources"]
        if item["source_name"] == first_source["source_name"]
    )
    assert source["summary"]["missing_case_count"] == 1
    contract = next(
        item
        for item in result["artifact_contracts"]["sources"]
        if item["source_name"] == first_source["source_name"]
    )
    assert contract["input"]["status"] == "diagnostic"
    assert contract["input"]["prediction_count"] == 2
    assert contract["diagnostics"]["missing_case_count"] == 1
    assert contract["ready_to_import"] is False
    assert not Path(str(manifest["matrix_report_path"])).exists()
    launch_report = lab.offline_control_artifact_launch_report(
        result["artifact_contracts"],
        manifest_path=manifest_path,
    )
    assert launch_report["ready_to_import"] is False
    assert launch_report["summary"]["blocked_source_count"] == 1
    assert launch_report["summary"]["diagnostic_source_count"] == 1
    source_report = next(
        item
        for item in launch_report["sources"]
        if item["source_name"] == first_source["source_name"]
    )
    assert source_report["input_status"] == "diagnostic"
    assert source_report["blocking_reasons"] == ["input_diagnostics"]
    assert launch_report["actionable_blockers"] == {
        "sources": {
            "blocked_source_count": 1,
            "items": [
                {
                    "blocking_reasons": ["input_diagnostics"],
                    "diagnostics": source_report["diagnostics"],
                    "input_path": source_report["input_path"],
                    "input_status": "diagnostic",
                    "metadata_ready": True,
                    "source_import_command": source_report["source_import_command"],
                    "source_key": source_report["source_key"],
                    "source_kind": source_report["source_kind"],
                    "source_name": source_report["source_name"],
                },
            ],
        },
    }
    assert launch_report["external_prediction_intake_plan"]["blocked_source_count"] == 1
    assert launch_report["external_prediction_intake_plan"]["blocked_sources"] == [
        {
            "blocking_reasons": ["input_diagnostics"],
            "input_status": "diagnostic",
            "prediction_path": source_report["input_path"],
            "source_import_command": source_report["source_import_command"],
            "source_key": source_report["source_key"],
            "source_kind": source_report["source_kind"],
            "source_name": source_report["source_name"],
        },
    ]
    diagnostic_intake_row = next(
        row
        for row in launch_report["external_prediction_intake_plan"]["sources"]
        if row["source_name"] == first_source["source_name"]
    )
    assert diagnostic_intake_row["input_status"] == "diagnostic"
    assert diagnostic_intake_row["ready_to_import"] is False
    assert diagnostic_intake_row["blocking_reasons"] == ["input_diagnostics"]


def test_offline_control_artifact_launch_report_handles_missing_candidate_prediction(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest_package(tmp_path)
    preflight = lab.offline_control_import_manifest_preflight(manifest_path)
    contracts_path = tmp_path / "contracts" / "offline-control-artifact-contracts.json"
    lab.save_offline_control_artifact_contracts(
        preflight["artifact_contracts"],
        contracts_path,
    )
    manifest = lab.load_offline_control_import_manifest(manifest_path)
    Path(str(manifest["candidate_prediction_path"])).unlink()

    launch_report = lab.offline_control_artifact_launch_report(
        lab.load_offline_control_artifact_contracts(contracts_path),
        manifest_path=manifest_path,
        contracts_path=contracts_path,
    )

    assert launch_report["schema_version"] == (
        "dsg-spatialqa-lab.offline-control-artifact-launch-report.v1"
    )
    assert launch_report["action"] == "offline_control_artifact_launch_report"
    assert launch_report["ready_to_import"] is False
    assert launch_report["preflight_ready_to_import"] is False
    assert launch_report["contracts_digest"] == preflight["artifact_contracts"][
        "contracts_digest"
    ]
    assert launch_report["current_contracts_digest"] != launch_report[
        "contracts_digest"
    ]
    assert launch_report["summary"] == {
        "blocked_source_count": 0,
        "candidate_blocked_count": 1,
        "candidate_ready_count": 0,
        "diagnostic_source_count": 0,
        "invalid_source_count": 0,
        "metadata_blocked_source_count": 0,
        "missing_source_count": 0,
        "ready_source_count": 4,
        "source_count": 4,
    }
    assert launch_report["candidate"]["status"] == "missing"
    assert launch_report["candidate"]["ready_to_import"] is False
    assert launch_report["candidate"]["blocking_reasons"] == [
        "candidate_prediction_missing"
    ]
    assert launch_report["candidate"]["prediction_count"] == 0
    assert launch_report["candidate"]["valid"] is False
    assert "No such file or directory" in launch_report["candidate"]["error"]
    assert launch_report["actionable_blockers"] == {
        "candidate_prediction": {
            "blocking_reasons": ["candidate_prediction_missing"],
            "error": launch_report["candidate"]["error"],
            "path": launch_report["candidate"]["path"],
            "qa_eval_report_path": launch_report["candidate"]["qa_eval_report_path"],
            "status": "missing",
        },
    }
    assert all(source["ready_to_import"] is True for source in launch_report["sources"])
    assert launch_report["comparison"]["matches"] is False
    assert launch_report["report_digest"] == (
        lab.offline_control_artifact_launch_report_digest(launch_report)
    )


def test_run_offline_controls_cli_accepts_preflight_manifest(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_offline_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    manifest_path = _write_manifest_package(tmp_path)
    contracts_path = tmp_path / "contracts" / "offline-control-artifact-contracts.json"

    assert (
        main(
            [
                "--preflight-manifest",
                str(manifest_path),
                "--artifact-contracts",
                str(contracts_path),
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["action"] == "offline_control_import_manifest_preflight"
    assert output["manifest_path"] == str(manifest_path)
    assert output["ready_to_import"] is True
    assert output["matrix_readiness"]["ready"] is True
    assert output["artifact_contracts_path"] == str(contracts_path)
    assert output["artifact_contracts"]["summary"] == {
        "ready_source_contract_count": 4,
        "source_contract_count": 4,
    }
    assert contracts_path.exists()
    saved_contracts = lab.load_offline_control_artifact_contracts(contracts_path)
    assert saved_contracts == output["artifact_contracts"]
    assert saved_contracts["contracts_digest"] == (
        lab.offline_control_artifact_contracts_digest(saved_contracts)
    )

    check_module = load_check_offline_controls_script()
    check_main = cast(MainFn, getattr(check_module, "main"))
    assert check_main(["--validate-artifact-contracts", str(contracts_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_offline_control_artifact_contracts"
    assert validation["valid"] is True
    assert check_main(
        [
            "--compare-artifact-contracts",
            str(contracts_path),
            "--manifest",
            str(manifest_path),
        ]
    ) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["action"] == "compare_offline_control_artifact_contracts"
    assert comparison["matches"] is True

    assert check_main(
        [
            "--artifact-launch-report",
            str(contracts_path),
            "--manifest",
            str(manifest_path),
        ]
    ) == 0
    launch_report = json.loads(capsys.readouterr().out)
    assert launch_report["action"] == "offline_control_artifact_launch_report"
    assert launch_report["ready_to_import"] is True
    assert launch_report["summary"]["ready_source_count"] == 4


def test_check_offline_controls_cli_reports_missing_candidate_launch_readiness(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    run_module = load_run_offline_controls_script()
    run_main = cast(MainFn, getattr(run_module, "main"))
    check_module = load_check_offline_controls_script()
    check_main = cast(MainFn, getattr(check_module, "main"))
    manifest_path = _write_manifest_package(tmp_path)
    contracts_path = tmp_path / "contracts" / "offline-control-artifact-contracts.json"

    assert (
        run_main(
            [
                "--preflight-manifest",
                str(manifest_path),
                "--artifact-contracts",
                str(contracts_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    manifest = lab.load_offline_control_import_manifest(manifest_path)
    Path(str(manifest["candidate_prediction_path"])).unlink()

    assert (
        check_main(
            [
                "--artifact-launch-report",
                str(contracts_path),
                "--manifest",
                str(manifest_path),
            ]
        )
        == 1
    )

    launch_report = json.loads(capsys.readouterr().out)
    assert launch_report["action"] == "offline_control_artifact_launch_report"
    assert "error" not in launch_report
    assert launch_report["ready_to_import"] is False
    assert launch_report["candidate"]["status"] == "missing"
    assert launch_report["candidate"]["blocking_reasons"] == [
        "candidate_prediction_missing"
    ]
    assert launch_report["summary"]["candidate_blocked_count"] == 1


def test_run_offline_controls_cli_accepts_manifest(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_offline_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    manifest_path = _write_manifest_package(tmp_path)
    ledger_path = tmp_path / "run-ledger" / "offline-control-import-run-ledger.json"

    assert main(["--manifest", str(manifest_path), "--run-ledger", str(ledger_path)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["action"] == "run_offline_control_import_manifest"
    assert output["manifest_path"] == str(manifest_path)
    assert output["ready"] is True
    assert output["qa_eval_handoff"]["ready"] is True
    assert Path(output["matrix_report_path"]).exists()
    assert output["run_ledger_path"] == str(ledger_path)
    assert output["run_ledger_digest"] == (
        lab.load_offline_control_import_run_ledger(ledger_path)["ledger_digest"]
    )

    check_module = load_check_offline_controls_script()
    check_main = cast(MainFn, getattr(check_module, "main"))
    assert check_main(["--validate-run-ledger", str(ledger_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_offline_control_import_run_ledger"
    assert validation["valid"] is True
    assert check_main(["--compare-run-ledger", str(ledger_path)]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["action"] == "compare_offline_control_import_run_ledger"
    assert comparison["matches"] is True


def test_run_offline_controls_cli_returns_structured_json_for_missing_manifest(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_offline_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    manifest_path = tmp_path / "missing-manifest.json"

    assert main(["--manifest", str(manifest_path)]) == 1

    output = json.loads(capsys.readouterr().out)
    assert output["action"] == "run_offline_control_import_manifest"
    assert output["path"] == str(manifest_path)
    assert output["valid"] is False
    assert "missing-manifest.json" in output["error"]


def _write_manifest_package(tmp_path: Path) -> Path:
    package_dir = tmp_path / "ai2thor-real-trial"
    qa_path = package_dir / "qa.jsonl"
    candidate_path = package_dir / "candidate" / "predicted-graph-tool.jsonl"
    frame_index_path = package_dir / "inputs" / "traces" / "frame-index.jsonl"
    cases = _cases()
    lab.save_qa_dataset(cases, qa_path)
    lab.save_qa_predictions(_predictions(cases), candidate_path)
    _write_frame_index(cases, frame_index_path)

    sources = []
    for source_kind in ("vlm", "multi_frame_vlm", "caption_memory", "graph_text"):
        source_name = _source_name(source_kind)
        input_path = package_dir / "inputs" / f"{source_name}.jsonl"
        lab.save_qa_predictions(_predictions(cases), input_path)
        sources.append(
            {
                "source_kind": source_kind,
                "source_name": source_name,
                "input_path": _relative(package_dir, input_path),
                "input_format": "qa_prediction",
                "metadata": _source_metadata(source_kind),
            }
        )

    manifest = {
        "schema_version": "dsg-spatialqa-lab.offline-control-import-manifest.v1",
        "qa_path": _relative(package_dir, qa_path),
        "frame_index_path": _relative(package_dir, frame_index_path),
        "output_dir": "offline-controls",
        "matrix_report_path": "offline-controls/offline-control-matrix.json",
        "result_report_path": "offline-controls/offline-control-result.json",
        "candidate_prediction_path": _relative(package_dir, candidate_path),
        "candidate_name": "predicted_graph_tool",
        "qa_eval_output_dir": "offline-controls/qa-eval",
        "required_source_kinds": [
            "caption_memory",
            "graph_text",
            "multi_frame_vlm",
            "vlm",
        ],
        "sources": sources,
    }
    manifest_path = package_dir / "offline-control-import-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _write_frame_index(cases: tuple[lab.QACase, ...], path: Path) -> None:
    rows = []
    seen: set[tuple[str, str, int]] = set()
    for case in cases:
        key = (case.episode_id, case.scene_id, case.step)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "schema_version": "dsg-spatialqa-lab.real-experiment-frame-trace.v1",
                "dataset_id": "unit-test",
                "episode_id": case.episode_id,
                "scene_id": case.scene_id,
                "step": case.step,
                "asset_paths": {
                    "rgb": f"frames/{case.episode_id}/{case.step:04d}.rgb.ppm",
                    "depth": f"frames/{case.episode_id}/{case.step:04d}.depth.npy",
                    "segmentation": f"frames/{case.episode_id}/{case.step:04d}.seg.ppm",
                },
                "asset_digests": {
                    "rgb": f"rgb-digest-{case.step:04d}",
                    "depth": f"depth-digest-{case.step:04d}",
                    "segmentation": f"seg-digest-{case.step:04d}",
                },
                "asset_present": {
                    "rgb": True,
                    "depth": True,
                    "segmentation": True,
                },
                "visible_object_ids": ["hidden-from-vlm-request"],
                "visible_object_labels": ["hidden-label"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_custom_frame_index(path: Path, *, rows: tuple[dict[str, object], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _frame_index_row(
    case: lab.QACase,
    *,
    step: int | None = None,
    visible_object_ids: tuple[str, ...],
) -> dict[str, object]:
    frame_step = case.step if step is None else step
    return {
        "schema_version": "dsg-spatialqa-lab.real-experiment-frame-trace.v1",
        "dataset_id": "unit-test",
        "episode_id": case.episode_id,
        "scene_id": case.scene_id,
        "step": frame_step,
        "asset_paths": {
            "rgb": f"frames/{case.episode_id}/{frame_step:04d}.rgb.ppm",
            "depth": f"frames/{case.episode_id}/{frame_step:04d}.depth.npy",
            "segmentation": f"frames/{case.episode_id}/{frame_step:04d}.seg.ppm",
        },
        "asset_digests": {
            "rgb": f"rgb-digest-{frame_step:04d}",
            "depth": f"depth-digest-{frame_step:04d}",
            "segmentation": f"seg-digest-{frame_step:04d}",
        },
        "asset_present": {
            "rgb": True,
            "depth": True,
            "segmentation": True,
        },
        "visible_object_ids": list(visible_object_ids),
        "visible_object_labels": ["hidden-label"],
    }


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _cases() -> tuple[lab.QACase, ...]:
    graph = lab.load_scene_fixture("tabletop")
    generated = lab.generate_qa_cases(
        graph,
        scene_id="tabletop_scene",
        episode_id="episode_001",
    )
    location = next(case for case in generated if case.question_type == "object_location")
    relation = next(case for case in generated if case.question_type == "relative_relation")
    nearest = next(case for case in generated if case.question_type == "nearest_object")
    return (location, relation, nearest)


def _predictions(cases: tuple[lab.QACase, ...]) -> tuple[lab.QAPrediction, ...]:
    return tuple(
        lab.QAPrediction(
            id=case.id,
            answer=case.answer,
            evidence_nodes=case.required_nodes,
            evidence_edges=case.required_edges,
            confidence=0.83,
        )
        for case in cases
    )


def _source_name(source_kind: str) -> str:
    names = {
        "caption_memory": "caption_memory_ai2thor_trial",
        "graph_text": "graph_text_ai2thor_trial",
        "multi_frame_vlm": "llava16_multiframe_ai2thor_trial",
        "vlm": "llava16_ai2thor_trial",
    }
    return names[source_kind]


def _source_metadata(source_kind: str) -> dict[str, object]:
    model_ids = {
        "caption_memory": "blip2-flan-t5-xl",
        "graph_text": "gpt-4.1-mini",
        "multi_frame_vlm": "llava-v1.6-34b",
        "vlm": "llava-v1.6-34b",
    }
    prompt_ids = {
        "caption_memory": "caption-memory-spatial-v1",
        "graph_text": "graph-text-spatial-qa-v1",
        "multi_frame_vlm": "multi-frame-vlm-spatial-qa-v1",
        "vlm": "vlm-spatial-qa-v1",
    }
    return {
        "capabilities": ["spatial_qa", "dynamic_memory"],
        "dataset_id": "ai2thor-real-trial-v1",
        "model_id": model_ids[source_kind],
        "prompt_id": prompt_ids[source_kind],
    }
