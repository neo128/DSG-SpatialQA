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
RUN_PREDICTED_DSG_SCRIPT = ROOT / "scripts" / "run_predicted_dsg.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_run_predicted_dsg_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_predicted_dsg_script",
        RUN_PREDICTED_DSG_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_predicted_dsg_from_detector_jsonl_writes_ready_artifacts(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "PREDICTED_DSG_DETECTOR_RUN_SCHEMA_VERSION")
    assert hasattr(lab, "run_predicted_dsg_from_detector_jsonl")
    detector_jsonl_path = _write_detector_jsonl(tmp_path)
    sequence_path = tmp_path / "predicted" / "detector-observations.json"
    graph_path = tmp_path / "predicted" / "predicted-graph.json"
    predicted_report_path = tmp_path / "predicted" / "predicted-report.json"
    detector_import_report_path = (
        tmp_path / "predicted" / "detector-import-report.json"
    )
    evidence_report_path = tmp_path / "predicted" / "predicted-dsg-evidence.json"

    result = lab.run_predicted_dsg_from_detector_jsonl(
        detector_jsonl_path=detector_jsonl_path,
        output_sequence_path=sequence_path,
        output_graph_path=graph_path,
        predicted_graph_report_path=predicted_report_path,
        detector_import_report_path=detector_import_report_path,
        predicted_dsg_evidence_report_path=evidence_report_path,
    )

    observations = lab.load_scene_observation_sequence(sequence_path)
    detector_import_report = lab.load_detector_observation_import_report(
        detector_import_report_path
    )
    predicted_report = lab.load_predicted_graph_report(predicted_report_path)
    evidence_report = lab.load_predicted_dsg_evidence_report(evidence_report_path)
    graph = lab.load_graph_json(graph_path)

    assert result["schema_version"] == (
        "dsg-spatialqa-lab.predicted-dsg-detector-run.v1"
    )
    assert result["action"] == "run_predicted_dsg_from_detector_jsonl"
    assert result["ready"] is True
    assert result["detector_jsonl_path"] == str(detector_jsonl_path)
    assert result["observation_sequence_path"] == str(sequence_path)
    assert result["graph_path"] == str(graph_path)
    assert result["predicted_graph_report_path"] == str(predicted_report_path)
    assert result["detector_import_report_path"] == str(detector_import_report_path)
    assert result["predicted_dsg_evidence_report_path"] == str(evidence_report_path)
    assert result["input_digest"] == detector_import_report["input_digest"]
    assert result["observation_sequence_digest"] == (
        lab.scene_observation_sequence_digest(observations)
    )
    assert result["graph_digest"] == lab.graph_json_digest(graph)
    assert result["predicted_graph_report_digest"] == predicted_report["digest"]
    assert result["detector_import_report_digest"] == detector_import_report["digest"]
    assert result["predicted_dsg_evidence_report_digest"] == evidence_report[
        "report_digest"
    ]
    assert result["readiness"] == {
        "ready": True,
        "failed_check_count": 0,
        "failed_checks": [],
    }
    assert result["summary"]["evidence_summary"] == evidence_report[
        "evidence_summary"
    ]
    assert predicted_report["input_kind"] == "observation_sequence"
    assert predicted_report["path"] == str(sequence_path)
    assert evidence_report["predicted_graph_report_path"] == str(predicted_report_path)
    assert lab.validate_detector_observation_import_report(detector_import_report)[
        "valid"
    ] is True
    assert lab.compare_detector_observation_import_report(detector_import_report)[
        "matches"
    ] is True
    assert lab.validate_predicted_graph_report(predicted_report)["valid"] is True
    assert lab.compare_predicted_graph_report(predicted_report)["matches"] is True
    assert lab.validate_predicted_dsg_evidence_report(evidence_report)["valid"] is True
    assert lab.compare_predicted_dsg_evidence_report(evidence_report)["matches"] is True


def test_run_predicted_dsg_cli_writes_ready_detector_package(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_predicted_dsg_script()
    main = cast(MainFn, getattr(module, "main"))
    detector_jsonl_path = _write_detector_jsonl(tmp_path)
    sequence_path = tmp_path / "cli" / "detector-observations.json"
    graph_path = tmp_path / "cli" / "predicted-graph.json"
    predicted_report_path = tmp_path / "cli" / "predicted-report.json"
    detector_import_report_path = tmp_path / "cli" / "detector-import-report.json"
    evidence_report_path = tmp_path / "cli" / "predicted-dsg-evidence.json"

    assert (
        main(
            [
                "--detector-jsonl",
                str(detector_jsonl_path),
                "--observation-sequence",
                str(sequence_path),
                "--output-graph",
                str(graph_path),
                "--predicted-report",
                str(predicted_report_path),
                "--detector-import-report",
                str(detector_import_report_path),
                "--predicted-dsg-evidence-report",
                str(evidence_report_path),
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["action"] == "run_predicted_dsg_from_detector_jsonl"
    assert output["ready"] is True
    assert output["predicted_graph_report_digest"] == (
        lab.load_predicted_graph_report(predicted_report_path)["digest"]
    )
    assert output["predicted_dsg_evidence_report_digest"] == (
        lab.load_predicted_dsg_evidence_report(evidence_report_path)["report_digest"]
    )
    assert lab.compare_detector_observation_import_report(
        lab.load_detector_observation_import_report(detector_import_report_path)
    )["matches"] is True
    assert lab.compare_predicted_graph_report(
        lab.load_predicted_graph_report(predicted_report_path)
    )["matches"] is True
    assert lab.compare_predicted_dsg_evidence_report(
        lab.load_predicted_dsg_evidence_report(evidence_report_path)
    )["matches"] is True


def test_run_predicted_dsg_detector_run_manifest_writes_ready_artifacts(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "PREDICTED_DSG_DETECTOR_RUN_MANIFEST_SCHEMA_VERSION")
    assert hasattr(lab, "PREDICTED_DSG_DETECTOR_RUN_LEDGER_SCHEMA_VERSION")
    assert hasattr(lab, "load_predicted_dsg_detector_run_manifest")
    assert hasattr(lab, "predicted_dsg_detector_run_manifest_digest")
    assert hasattr(lab, "predicted_dsg_detector_run_ledger")
    assert hasattr(lab, "predicted_dsg_detector_run_ledger_digest")
    assert hasattr(lab, "save_predicted_dsg_detector_run_ledger")
    assert hasattr(lab, "load_predicted_dsg_detector_run_ledger")
    assert hasattr(lab, "validate_predicted_dsg_detector_run_ledger")
    assert hasattr(lab, "compare_predicted_dsg_detector_run_ledger")
    assert hasattr(lab, "run_predicted_dsg_detector_run_manifest")
    manifest_path = _write_detector_run_manifest(tmp_path)

    manifest = lab.load_predicted_dsg_detector_run_manifest(manifest_path)
    result = lab.run_predicted_dsg_detector_run_manifest(manifest_path)
    ledger = lab.predicted_dsg_detector_run_ledger(result)

    assert result["action"] == "run_predicted_dsg_detector_run_manifest"
    assert result["manifest_schema_version"] == (
        "dsg-spatialqa-lab.predicted-dsg-detector-run-manifest.v1"
    )
    assert result["manifest_path"] == str(manifest_path)
    assert result["manifest_digest"] == lab.predicted_dsg_detector_run_manifest_digest(
        manifest
    )
    assert result["ready"] is True
    assert Path(result["observation_sequence_path"]).exists()
    assert Path(result["graph_path"]).exists()
    assert Path(result["predicted_graph_report_path"]).exists()
    assert Path(result["detector_import_report_path"]).exists()
    assert Path(result["predicted_dsg_evidence_report_path"]).exists()
    assert lab.compare_predicted_graph_report(
        lab.load_predicted_graph_report(result["predicted_graph_report_path"])
    )["matches"] is True
    assert lab.compare_predicted_dsg_evidence_report(
        lab.load_predicted_dsg_evidence_report(
            result["predicted_dsg_evidence_report_path"]
        )
    )["matches"] is True
    assert ledger["schema_version"] == (
        "dsg-spatialqa-lab.predicted-dsg-detector-run-ledger.v1"
    )
    assert ledger["ledger_digest"] == (
        lab.predicted_dsg_detector_run_ledger_digest(ledger)
    )
    assert ledger["run"] == {
        "detector_import_report_digest": result["detector_import_report_digest"],
        "detector_import_report_path": result["detector_import_report_path"],
        "detector_jsonl_path": result["detector_jsonl_path"],
        "graph_digest": result["graph_digest"],
        "graph_path": result["graph_path"],
        "input_digest": result["input_digest"],
        "manifest_digest": result["manifest_digest"],
        "manifest_path": str(manifest_path),
        "observation_sequence_digest": result["observation_sequence_digest"],
        "observation_sequence_path": result["observation_sequence_path"],
        "predicted_dsg_evidence_report_digest": result[
            "predicted_dsg_evidence_report_digest"
        ],
        "predicted_dsg_evidence_report_path": result[
            "predicted_dsg_evidence_report_path"
        ],
        "predicted_graph_report_digest": result["predicted_graph_report_digest"],
        "predicted_graph_report_path": result["predicted_graph_report_path"],
        "ready": True,
        "schema_version": result["schema_version"],
    }
    assert ledger["readiness"] == result["readiness"]
    assert ledger["summary"] == {
        "detector_import_summary": result["summary"]["detector_import_summary"],
        "evidence_summary": result["summary"]["evidence_summary"],
        "predicted_graph_summary": result["summary"]["predicted_graph_summary"],
    }
    ledger_path = tmp_path / "ledgers" / "predicted-dsg-detector-run-ledger.json"
    saved_path = lab.save_predicted_dsg_detector_run_ledger(ledger, ledger_path)
    loaded_ledger = lab.load_predicted_dsg_detector_run_ledger(ledger_path)
    validation = lab.validate_predicted_dsg_detector_run_ledger(loaded_ledger)
    comparison = lab.compare_predicted_dsg_detector_run_ledger(loaded_ledger)
    assert saved_path == ledger_path
    assert loaded_ledger == ledger
    assert validation["valid"] is True
    assert comparison["matches"] is True
    assert comparison["saved_digest"] == ledger["ledger_digest"]


def test_predicted_dsg_detector_run_manifest_preflight_reports_ready_inputs(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "PREDICTED_DSG_DETECTOR_PREFLIGHT_SCHEMA_VERSION")
    assert hasattr(lab, "PREDICTED_DSG_DETECTOR_ARTIFACT_CONTRACT_SCHEMA_VERSION")
    assert hasattr(lab, "predicted_dsg_detector_run_manifest_preflight")
    assert hasattr(lab, "predicted_dsg_detector_artifact_contract_digest")
    assert hasattr(lab, "save_predicted_dsg_detector_artifact_contract")
    assert hasattr(lab, "load_predicted_dsg_detector_artifact_contract")
    assert hasattr(lab, "PREDICTED_DSG_DETECTOR_ARTIFACT_LAUNCH_REPORT_SCHEMA_VERSION")
    assert hasattr(lab, "predicted_dsg_detector_artifact_launch_report")
    assert hasattr(lab, "predicted_dsg_detector_artifact_launch_report_digest")
    manifest_path = _write_detector_run_manifest(tmp_path)
    manifest = lab.load_predicted_dsg_detector_run_manifest(manifest_path)

    result = lab.predicted_dsg_detector_run_manifest_preflight(manifest_path)

    assert result["schema_version"] == (
        "dsg-spatialqa-lab.predicted-dsg-detector-preflight.v1"
    )
    assert result["action"] == "predicted_dsg_detector_run_manifest_preflight"
    assert result["manifest_path"] == str(manifest_path)
    assert result["manifest_digest"] == lab.predicted_dsg_detector_run_manifest_digest(
        manifest
    )
    assert result["ready_to_build"] is True
    assert result["readiness"]["ready"] is True
    assert result["summary"]["observation_count"] == 2
    assert result["summary"]["object_observation_count"] == 3
    assert result["summary"]["evidence_summary"]["evidence_kind_counts"] == {
        "depth": 3,
        "detector": 3,
        "rgb": 3,
    }
    assert result["summary"]["asset_summary"] == {
        "asset_kind_counts": {"depth": 2, "rgb": 2, "segmentation": 2},
        "asset_path_count": 6,
        "missing_asset_count": 0,
        "missing_assets": [],
        "present_asset_count": 6,
    }
    assert not Path(result["planned_outputs"]["observation_sequence_path"]).exists()
    assert not Path(result["planned_outputs"]["predicted_graph_report_path"]).exists()
    assert not Path(result["planned_outputs"]["predicted_dsg_evidence_report_path"]).exists()
    contract = result["artifact_contract"]
    assert contract["schema_version"] == (
        "dsg-spatialqa-lab.predicted-dsg-detector-artifact-contract.v1"
    )
    assert contract["contract_digest"] == (
        lab.predicted_dsg_detector_artifact_contract_digest(contract)
    )
    assert contract["manifest_path"] == str(manifest_path)
    assert contract["manifest_digest"] == result["manifest_digest"]
    assert contract["detector_input"] == {
        "expected_schema_version": "dsg-spatialqa-lab.detector-observation-record.v1",
        "input_digest": result["input_digest"],
        "object_observation_count": 3,
        "observation_count": 2,
        "observation_sequence_digest": result["observation_sequence_digest"],
        "path": manifest["detector_jsonl_path"],
        "status": "ready",
    }
    assert contract["build_requirements"] == {
        "infer_relations": ["LEFT_OF", "RIGHT_OF", "NEAR"],
        "min_object_observation_count": 2,
        "min_observation_count": 2,
        "reference_frames": ["world"],
        "required_evidence_kinds": ["depth", "detector", "rgb"],
    }
    assert contract["planned_outputs"] == result["planned_outputs"]
    assert contract["readiness"] == {
        "failed_check_count": 0,
        "failed_checks": [],
        "ready": True,
    }
    assert contract["asset_summary"] == result["summary"]["asset_summary"]
    assert contract["summary"] == {
        "asset_summary": result["summary"]["asset_summary"],
        "evidence_kind_counts": {"depth": 3, "detector": 3, "rgb": 3},
        "object_observation_count": 3,
        "observation_count": 2,
    }
    contract_path = tmp_path / "contracts" / "predicted-dsg-detector-contract.json"
    saved_path = lab.save_predicted_dsg_detector_artifact_contract(
        contract,
        contract_path,
    )
    loaded_contract = lab.load_predicted_dsg_detector_artifact_contract(contract_path)
    assert saved_path == contract_path
    assert loaded_contract == contract
    assert loaded_contract["contract_digest"] == (
        lab.predicted_dsg_detector_artifact_contract_digest(loaded_contract)
    )
    launch_report = lab.predicted_dsg_detector_artifact_launch_report(
        loaded_contract,
        manifest_path=manifest_path,
        contract_path=contract_path,
    )
    assert launch_report["schema_version"] == (
        "dsg-spatialqa-lab.predicted-dsg-detector-artifact-launch-report.v1"
    )
    assert launch_report["action"] == (
        "predicted_dsg_detector_artifact_launch_report"
    )
    assert launch_report["contract_path"] == str(contract_path)
    assert launch_report["manifest_path"] == str(manifest_path)
    assert launch_report["contract_digest"] == contract["contract_digest"]
    assert launch_report["current_contract_digest"] == contract["contract_digest"]
    assert launch_report["ready_to_build"] is True
    assert launch_report["summary"] == {
        "blocking_reason_count": 0,
        "failed_check_count": 0,
        "object_observation_count": 3,
        "observation_count": 2,
        "planned_output_count": 5,
        "ready_to_build": True,
    }
    assert launch_report["detector_input"] == {
        "blocking_reasons": [],
        "object_observation_count": 3,
        "observation_count": 2,
        "path": manifest["detector_jsonl_path"],
        "status": "ready",
    }
    assert launch_report["actionable_blockers"] == {}
    assert launch_report["build_command"] == (
        "python scripts/run_predicted_dsg.py "
        f"--detector-jsonl {manifest['detector_jsonl_path']} "
        f"--observation-sequence {manifest['output_sequence_path']} "
        f"--output-graph {manifest['output_graph_path']} "
        f"--predicted-report {manifest['predicted_graph_report_path']} "
        f"--detector-import-report {manifest['detector_import_report_path']} "
        "--predicted-dsg-evidence-report "
        f"{manifest['predicted_dsg_evidence_report_path']} "
        "--infer-relation LEFT_OF "
        "--infer-relation RIGHT_OF "
        "--infer-relation NEAR "
        "--reference-frame world "
        "--min-observation-count 2 "
        "--min-object-observation-count 2 "
        "--required-evidence-kind depth "
        "--required-evidence-kind detector "
        "--required-evidence-kind rgb"
    )
    assert launch_report["build_plan"] == {
        "track": "predicted_dsg",
        "build_command": launch_report["build_command"],
        "detector_input": {
            "blocking_reasons": [],
            "object_observation_count": 3,
            "observation_count": 2,
            "path": manifest["detector_jsonl_path"],
            "ready_to_build": True,
            "status": "ready",
        },
        "asset_summary": result["summary"]["asset_summary"],
        "manifest_build_command": (
            f"python scripts/run_predicted_dsg.py --manifest {manifest_path}"
        ),
        "planned_outputs": launch_report["planned_outputs"],
        "preflight_command": (
            "python scripts/run_predicted_dsg.py --preflight-manifest "
            f"{manifest_path}"
        ),
        "requirements": launch_report["build_requirements"],
    }
    assert launch_report["external_detector_intake_plan"] == {
        "track": "predicted_dsg",
        "blocked": False,
        "blocking_reasons": [],
        "build_command": launch_report["build_command"],
        "asset_summary": result["summary"]["asset_summary"],
        "detector_request_bundle_command": (
            "python scripts/run_predicted_dsg.py "
            f"--detector-request-bundle {manifest_path} "
            f"--request-bundle-output "
            f"{manifest_path.parent / 'predicted-dsg-detector-request-bundle.json'}"
        ),
        "detector_receipt_bundle_command": (
            "python scripts/run_predicted_dsg.py "
            f"--detector-receipt-bundle {manifest_path} "
            f"--receipt-bundle-output "
            f"{manifest_path.parent / 'predicted-dsg-detector-receipt-bundle.json'}"
        ),
        "detector_input": {
            "expected_schema_version": "dsg-spatialqa-lab.detector-observation-record.v1",
            "object_observation_count": 3,
            "observation_count": 2,
            "path": manifest["detector_jsonl_path"],
            "ready_to_build": True,
            "status": "ready",
        },
        "manifest_build_command": (
            f"python scripts/run_predicted_dsg.py --manifest {manifest_path}"
        ),
        "planned_outputs": launch_report["planned_outputs"],
        "preflight": launch_report["preflight"],
        "preflight_command": (
            "python scripts/run_predicted_dsg.py --preflight-manifest "
            f"{manifest_path}"
        ),
        "readiness": {
            "failed_check_count": 0,
            "failed_checks": [],
            "ready": True,
        },
        "requirements": {
            "infer_relations": ["LEFT_OF", "RIGHT_OF", "NEAR"],
            "min_object_observation_count": 2,
            "min_observation_count": 2,
            "reference_frames": ["world"],
            "required_evidence_kinds": ["depth", "detector", "rgb"],
        },
    }
    assert launch_report["next_commands"] == {
        "artifact_launch_report": (
            "python scripts/run_predicted_dsg.py "
            f"--artifact-launch-report {contract_path} --manifest {manifest_path}"
        ),
        "build": (
            "python scripts/run_predicted_dsg.py --manifest "
            f"{manifest_path}"
        ),
        "detector_request_bundle": (
            "python scripts/run_predicted_dsg.py "
            f"--detector-request-bundle {manifest_path} "
            f"--request-bundle-output "
            f"{manifest_path.parent / 'predicted-dsg-detector-request-bundle.json'}"
        ),
        "detector_receipt_bundle": (
            "python scripts/run_predicted_dsg.py "
            f"--detector-receipt-bundle {manifest_path} "
            f"--receipt-bundle-output "
            f"{manifest_path.parent / 'predicted-dsg-detector-receipt-bundle.json'}"
        ),
        "preflight": (
            "python scripts/run_predicted_dsg.py --preflight-manifest "
            f"{manifest_path}"
        ),
    }
    assert launch_report["report_digest"] == (
        lab.predicted_dsg_detector_artifact_launch_report_digest(launch_report)
    )


def test_predicted_dsg_detector_run_manifest_preflight_rejects_synthetic_sources(
    tmp_path: Path,
) -> None:
    manifest_path = _write_detector_run_manifest(tmp_path)
    manifest = lab.load_predicted_dsg_detector_run_manifest(manifest_path)
    detector_jsonl_path = Path(manifest["detector_jsonl_path"])
    records = [
        json.loads(line)
        for line in detector_jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    for record in records:
        metadata = record["metadata"]
        assert isinstance(metadata, dict)
        metadata["source"] = "SyntheticRGBDDetector"
    detector_jsonl_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    result = lab.predicted_dsg_detector_run_manifest_preflight(manifest_path)
    checks = {check["name"]: check for check in result["readiness"]["checks"]}

    assert result["ready_to_build"] is False
    assert "non_real_sources_absent" in result["readiness"]["failed_checks"]
    assert checks["non_real_sources_absent"] == {
        "name": "non_real_sources_absent",
        "passed": False,
        "actual": ["SyntheticRGBDDetector"],
    }


def test_predicted_dsg_detector_request_bundle_exports_detector_templates(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    assert hasattr(lab, "PREDICTED_DSG_DETECTOR_REQUEST_BUNDLE_SCHEMA_VERSION")
    assert hasattr(lab, "predicted_dsg_detector_request_bundle")
    assert hasattr(lab, "predicted_dsg_detector_request_bundle_digest")
    assert hasattr(lab, "save_predicted_dsg_detector_request_bundle")
    assert hasattr(lab, "load_predicted_dsg_detector_request_bundle")
    manifest_path = _write_detector_run_manifest(tmp_path)
    manifest = lab.load_predicted_dsg_detector_run_manifest(manifest_path)
    Path(manifest["detector_jsonl_path"]).unlink()

    bundle = lab.predicted_dsg_detector_request_bundle(manifest_path)

    assert bundle["schema_version"] == (
        "dsg-spatialqa-lab.predicted-dsg-detector-request-bundle.v1"
    )
    assert bundle["action"] == "predicted_dsg_detector_request_bundle"
    assert bundle["manifest_path"] == str(manifest_path)
    assert bundle["manifest_digest"] == (
        lab.predicted_dsg_detector_run_manifest_digest(manifest)
    )
    assert bundle["detector_jsonl"] == {
        "expected_schema_version": "dsg-spatialqa-lab.detector-observation-record.v1",
        "input_format": "detector_observation_jsonl",
        "path": manifest["detector_jsonl_path"],
    }
    assert bundle["frame_asset_fields"] == [
        "rgb_path",
        "depth_path",
        "segmentation_path",
    ]
    assert bundle["build_requirements"] == {
        "infer_relations": ["LEFT_OF", "RIGHT_OF", "NEAR"],
        "min_object_observation_count": 2,
        "min_observation_count": 2,
        "reference_frames": ["world"],
        "required_evidence_kinds": ["depth", "detector", "rgb"],
    }
    assert bundle["planned_outputs"] == {
        "detector_import_report_path": manifest["detector_import_report_path"],
        "observation_sequence_path": manifest["output_sequence_path"],
        "predicted_dsg_evidence_report_path": (
            manifest["predicted_dsg_evidence_report_path"]
        ),
        "predicted_graph_path": manifest["output_graph_path"],
        "predicted_graph_report_path": manifest["predicted_graph_report_path"],
    }
    assert bundle["commands"] == {
        "build": f"python scripts/run_predicted_dsg.py --manifest {manifest_path}",
        "preflight": (
            "python scripts/run_predicted_dsg.py --preflight-manifest "
            f"{manifest_path}"
        ),
    }
    assert bundle["record_template"] == {
        "schema_version": "dsg-spatialqa-lab.detector-observation-record.v1",
        "step": 0,
        "agent_id": "agent",
        "agent_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        "rgb_path": "frames/000001.rgb.png",
        "depth_path": "frames/000001.depth.png",
        "segmentation_path": "frames/000001.seg.png",
        "metadata": {},
        "detections": [
            {
                "object_id": "track_object_1",
                "label": "object",
                "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                "bbox": {
                    "center": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                    "size": [0.0, 0.0, 0.0],
                },
                "confidence": 0.0,
                "visible": True,
                "attributes": {},
            }
        ],
    }
    assert bundle["request_bundle_digest"] == (
        lab.predicted_dsg_detector_request_bundle_digest(bundle)
    )
    bundle_path = tmp_path / "handoff" / "predicted-dsg-detector-request-bundle.json"
    saved_path = lab.save_predicted_dsg_detector_request_bundle(bundle, bundle_path)
    loaded_bundle = lab.load_predicted_dsg_detector_request_bundle(bundle_path)
    assert saved_path == bundle_path
    assert loaded_bundle == bundle

    module = load_run_predicted_dsg_script()
    main = cast(MainFn, getattr(module, "main"))
    cli_bundle_path = tmp_path / "handoff" / "cli-detector-request-bundle.json"
    exit_code = main(
        [
            "--detector-request-bundle",
            str(manifest_path),
            "--request-bundle-output",
            str(cli_bundle_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["action"] == "predicted_dsg_detector_request_bundle"
    assert payload["manifest_path"] == str(manifest_path)
    assert payload["request_bundle_path"] == str(cli_bundle_path)
    assert payload["bundle"] == bundle
    assert lab.load_predicted_dsg_detector_request_bundle(cli_bundle_path) == bundle


def test_predicted_dsg_detector_receipt_bundle_exports_returned_detector_receipts(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    assert hasattr(lab, "PREDICTED_DSG_DETECTOR_RECEIPT_BUNDLE_SCHEMA_VERSION")
    assert hasattr(lab, "predicted_dsg_detector_receipt_bundle")
    assert hasattr(lab, "predicted_dsg_detector_receipt_bundle_digest")
    assert hasattr(lab, "save_predicted_dsg_detector_receipt_bundle")
    assert hasattr(lab, "load_predicted_dsg_detector_receipt_bundle")
    assert hasattr(lab, "validate_predicted_dsg_detector_receipt_bundle")
    manifest_path = _write_detector_run_manifest(tmp_path)
    manifest = lab.load_predicted_dsg_detector_run_manifest(manifest_path)
    preflight = lab.predicted_dsg_detector_run_manifest_preflight(manifest_path)

    bundle = lab.predicted_dsg_detector_receipt_bundle(manifest_path)

    assert bundle["schema_version"] == (
        "dsg-spatialqa-lab.predicted-dsg-detector-receipt-bundle.v1"
    )
    assert bundle["action"] == "predicted_dsg_detector_receipt_bundle"
    assert bundle["manifest_path"] == str(manifest_path)
    assert bundle["manifest_digest"] == (
        lab.predicted_dsg_detector_run_manifest_digest(manifest)
    )
    assert bundle["ready_to_build"] is True
    assert bundle["detector_jsonl"] == {
        "expected_schema_version": "dsg-spatialqa-lab.detector-observation-record.v1",
        "input_digest": preflight["input_digest"],
        "input_format": "detector_observation_jsonl",
        "object_observation_count": 3,
        "observation_count": 2,
        "observation_sequence_digest": preflight["observation_sequence_digest"],
        "path": manifest["detector_jsonl_path"],
        "status": "ready",
    }
    assert bundle["asset_summary"] == preflight["summary"]["asset_summary"]
    assert bundle["build_requirements"] == {
        "infer_relations": ["LEFT_OF", "RIGHT_OF", "NEAR"],
        "min_object_observation_count": 2,
        "min_observation_count": 2,
        "reference_frames": ["world"],
        "required_evidence_kinds": ["depth", "detector", "rgb"],
    }
    assert bundle["planned_outputs"] == preflight["planned_outputs"]
    assert bundle["readiness"] == {
        "failed_check_count": 0,
        "failed_checks": [],
        "ready": True,
    }
    assert bundle["summary"] == {
        "asset_summary": preflight["summary"]["asset_summary"],
        "detector_import_summary": preflight["summary"]["detector_import_summary"],
        "evidence_summary": preflight["summary"]["evidence_summary"],
        "object_observation_count": 3,
        "observation_count": 2,
        "ready_to_build": True,
    }
    assert bundle["commands"] == {
        "build": f"python scripts/run_predicted_dsg.py --manifest {manifest_path}",
        "detector_request_bundle": (
            "python scripts/run_predicted_dsg.py "
            f"--detector-request-bundle {manifest_path} "
            f"--request-bundle-output "
            f"{manifest_path.parent / 'predicted-dsg-detector-request-bundle.json'}"
        ),
        "preflight": (
            "python scripts/run_predicted_dsg.py --preflight-manifest "
            f"{manifest_path}"
        ),
    }
    assert bundle["receipt_bundle_digest"] == (
        lab.predicted_dsg_detector_receipt_bundle_digest(bundle)
    )
    bundle_path = tmp_path / "handoff" / "predicted-dsg-detector-receipt-bundle.json"
    saved_path = lab.save_predicted_dsg_detector_receipt_bundle(bundle, bundle_path)
    loaded_bundle = lab.load_predicted_dsg_detector_receipt_bundle(bundle_path)
    assert saved_path == bundle_path
    assert loaded_bundle == bundle
    validation = lab.validate_predicted_dsg_detector_receipt_bundle(loaded_bundle)
    assert validation["action"] == "validate_predicted_dsg_detector_receipt_bundle"
    assert validation["valid"] is True
    assert validation["receipt_bundle_digest"] == bundle["receipt_bundle_digest"]

    tampered_bundle = json.loads(json.dumps(bundle))
    tampered_bundle["summary"]["observation_count"] = 999
    tampered_bundle["receipt_bundle_digest"] = (
        lab.predicted_dsg_detector_receipt_bundle_digest(tampered_bundle)
    )
    tampered_validation = lab.validate_predicted_dsg_detector_receipt_bundle(
        tampered_bundle
    )
    assert tampered_validation["valid"] is False
    assert {
        check["name"]
        for check in tampered_validation["checks"]
        if check["passed"] is False
    } == {"summary"}

    module = load_run_predicted_dsg_script()
    main = cast(MainFn, getattr(module, "main"))
    cli_bundle_path = tmp_path / "handoff" / "cli-detector-receipt-bundle.json"
    exit_code = main(
        [
            "--detector-receipt-bundle",
            str(manifest_path),
            "--receipt-bundle-output",
            str(cli_bundle_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["action"] == "predicted_dsg_detector_receipt_bundle"
    assert payload["manifest_path"] == str(manifest_path)
    assert payload["receipt_bundle_path"] == str(cli_bundle_path)
    assert payload["bundle"] == bundle
    assert lab.load_predicted_dsg_detector_receipt_bundle(cli_bundle_path) == bundle

    assert (
        main(["--validate-detector-receipt-bundle", str(cli_bundle_path)])
        == 0
    )
    validation_payload = json.loads(capsys.readouterr().out)
    assert validation_payload["action"] == (
        "validate_predicted_dsg_detector_receipt_bundle"
    )
    assert validation_payload["path"] == str(cli_bundle_path)
    assert validation_payload["valid"] is True

    tampered_path = tmp_path / "handoff" / "tampered-detector-receipt-bundle.json"
    lab.save_predicted_dsg_detector_receipt_bundle(
        tampered_bundle,
        tampered_path,
    )
    assert main(["--validate-detector-receipt-bundle", str(tampered_path)]) == 1
    tampered_payload = json.loads(capsys.readouterr().out)
    assert tampered_payload["action"] == (
        "validate_predicted_dsg_detector_receipt_bundle"
    )
    assert tampered_payload["path"] == str(tampered_path)
    assert tampered_payload["valid"] is False


def test_predicted_dsg_detector_run_manifest_preflight_reports_not_ready_threshold(
    tmp_path: Path,
) -> None:
    manifest_path = _write_detector_run_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["min_observation_count"] = 3
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = lab.predicted_dsg_detector_run_manifest_preflight(manifest_path)

    assert result["ready_to_build"] is False
    assert result["readiness"]["ready"] is False
    assert "observation_count_minimum" in result["readiness"]["failed_checks"]
    checks = {check["name"]: check for check in result["readiness"]["checks"]}
    assert checks["observation_count_minimum"]["actual"] == 2
    assert checks["observation_count_minimum"]["minimum"] == 3
    launch_report = lab.predicted_dsg_detector_artifact_launch_report(
        result["artifact_contract"],
        manifest_path=manifest_path,
    )
    assert launch_report["ready_to_build"] is False
    assert launch_report["summary"]["failed_check_count"] == 1
    assert launch_report["detector_input"]["status"] == "ready"
    assert launch_report["detector_input"]["blocking_reasons"] == [
        "readiness_checks_failed"
    ]
    assert launch_report["actionable_blockers"] == {
        "build_readiness": {
            "build_command": launch_report["build_command"],
            "detector_input": launch_report["detector_input"],
            "failed_check_count": 1,
            "failed_checks": ["observation_count_minimum"],
            "requirements": launch_report["build_requirements"],
        },
    }
    assert launch_report["external_detector_intake_plan"]["blocked"] is True
    assert launch_report["external_detector_intake_plan"]["blocking_reasons"] == [
        "readiness_checks_failed"
    ]
    assert launch_report["external_detector_intake_plan"]["detector_input"][
        "ready_to_build"
    ] is False
    assert launch_report["external_detector_intake_plan"]["readiness"] == {
        "failed_check_count": 1,
        "failed_checks": ["observation_count_minimum"],
        "ready": False,
    }


def test_predicted_dsg_detector_run_manifest_preflight_reports_missing_frame_asset(
    tmp_path: Path,
) -> None:
    manifest_path = _write_detector_run_manifest(tmp_path)
    manifest = lab.load_predicted_dsg_detector_run_manifest(manifest_path)
    missing_asset_path = (
        Path(manifest["detector_jsonl_path"]).parent / "frames/000001.rgb.png"
    )
    missing_asset_path.unlink()

    result = lab.predicted_dsg_detector_run_manifest_preflight(manifest_path)

    assert result["ready_to_build"] is False
    assert result["readiness"]["ready"] is False
    assert result["summary"]["asset_summary"] == {
        "asset_kind_counts": {"depth": 2, "rgb": 2, "segmentation": 2},
        "asset_path_count": 6,
        "missing_asset_count": 1,
        "missing_assets": [
            {
                "kind": "rgb",
                "path": "frames/000001.rgb.png",
                "resolved_path": str(missing_asset_path),
            },
        ],
        "present_asset_count": 5,
    }
    checks = {check["name"]: check for check in result["readiness"]["checks"]}
    assert checks["frame_assets_present"] == {
        "name": "frame_assets_present",
        "passed": False,
        "asset_path_count": 6,
        "missing": result["summary"]["asset_summary"]["missing_assets"],
        "missing_asset_count": 1,
        "present_asset_count": 5,
    }
    launch_report = lab.predicted_dsg_detector_artifact_launch_report(
        result["artifact_contract"],
        manifest_path=manifest_path,
    )
    assert launch_report["ready_to_build"] is False
    assert launch_report["summary"]["failed_check_count"] == 1
    assert launch_report["detector_input"]["blocking_reasons"] == [
        "readiness_checks_failed"
    ]
    assert launch_report["actionable_blockers"] == {
        "build_readiness": {
            "build_command": launch_report["build_command"],
            "detector_input": launch_report["detector_input"],
            "failed_check_count": 1,
            "failed_checks": ["frame_assets_present"],
            "requirements": launch_report["build_requirements"],
        },
    }
    assert launch_report["external_detector_intake_plan"]["asset_summary"] == (
        result["summary"]["asset_summary"]
    )
    assert launch_report["external_detector_intake_plan"]["readiness"] == {
        "failed_check_count": 1,
        "failed_checks": ["frame_assets_present"],
        "ready": False,
    }


def test_predicted_dsg_detector_artifact_launch_report_handles_missing_detector_input(
    tmp_path: Path,
) -> None:
    manifest_path = _write_detector_run_manifest(tmp_path)
    preflight = lab.predicted_dsg_detector_run_manifest_preflight(manifest_path)
    contract_path = tmp_path / "contracts" / "predicted-dsg-detector-contract.json"
    lab.save_predicted_dsg_detector_artifact_contract(
        preflight["artifact_contract"],
        contract_path,
    )
    manifest = lab.load_predicted_dsg_detector_run_manifest(manifest_path)
    Path(manifest["detector_jsonl_path"]).unlink()

    launch_report = lab.predicted_dsg_detector_artifact_launch_report(
        lab.load_predicted_dsg_detector_artifact_contract(contract_path),
        manifest_path=manifest_path,
        contract_path=contract_path,
    )

    assert launch_report["schema_version"] == (
        "dsg-spatialqa-lab.predicted-dsg-detector-artifact-launch-report.v1"
    )
    assert launch_report["action"] == (
        "predicted_dsg_detector_artifact_launch_report"
    )
    assert launch_report["ready_to_build"] is False
    assert launch_report["preflight_ready_to_build"] is False
    assert launch_report["contract_digest"] == preflight["artifact_contract"][
        "contract_digest"
    ]
    assert launch_report["current_contract_digest"] != launch_report[
        "contract_digest"
    ]
    assert launch_report["detector_input"] == {
        "blocking_reasons": [
            "detector_input_missing",
            "readiness_checks_failed",
        ],
        "object_observation_count": 0,
        "observation_count": 0,
        "path": manifest["detector_jsonl_path"],
        "status": "missing",
    }
    assert launch_report["actionable_blockers"] == {
        "build_readiness": {
            "build_command": launch_report["build_command"],
            "detector_input": launch_report["detector_input"],
            "failed_check_count": 1,
            "failed_checks": ["detector_input_missing"],
            "requirements": launch_report["build_requirements"],
        },
        "detector_input": {
            "blocking_reasons": [
                "detector_input_missing",
                "readiness_checks_failed",
            ],
            "build_command": launch_report["build_command"],
            "path": manifest["detector_jsonl_path"],
            "preflight": launch_report["preflight"],
            "status": "missing",
        },
    }
    assert launch_report["external_detector_intake_plan"]["blocked"] is True
    assert launch_report["external_detector_intake_plan"]["blocking_reasons"] == [
        "detector_input_missing",
        "readiness_checks_failed",
    ]
    assert launch_report["external_detector_intake_plan"]["detector_input"] == {
        "expected_schema_version": "dsg-spatialqa-lab.detector-observation-record.v1",
        "object_observation_count": 0,
        "observation_count": 0,
        "path": manifest["detector_jsonl_path"],
        "ready_to_build": False,
        "status": "missing",
    }
    assert launch_report["external_detector_intake_plan"]["preflight"] == (
        launch_report["preflight"]
    )
    assert launch_report["preflight"] == {
        "available": False,
        "error": launch_report["preflight"]["error"],
        "error_type": "FileNotFoundError",
        "status": "failed",
    }
    assert "No such file or directory" in launch_report["preflight"]["error"]
    assert launch_report["summary"] == {
        "blocking_reason_count": 2,
        "failed_check_count": 1,
        "object_observation_count": 0,
        "observation_count": 0,
        "planned_output_count": 5,
        "ready_to_build": False,
    }
    assert launch_report["readiness"] == {
        "failed_check_count": 1,
        "failed_checks": ["detector_input_missing"],
        "ready": False,
    }
    assert launch_report["comparison"]["matches"] is False
    assert launch_report["report_digest"] == (
        lab.predicted_dsg_detector_artifact_launch_report_digest(launch_report)
    )


def test_run_predicted_dsg_cli_accepts_preflight_manifest(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_predicted_dsg_script()
    main = cast(MainFn, getattr(module, "main"))
    manifest_path = _write_detector_run_manifest(tmp_path)
    contract_path = tmp_path / "contracts" / "predicted-dsg-detector-contract.json"

    assert (
        main(
            [
                "--preflight-manifest",
                str(manifest_path),
                "--artifact-contract",
                str(contract_path),
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["action"] == "predicted_dsg_detector_run_manifest_preflight"
    assert output["manifest_path"] == str(manifest_path)
    assert output["ready_to_build"] is True
    assert output["artifact_contract_path"] == str(contract_path)
    assert contract_path.exists()
    saved_contract = lab.load_predicted_dsg_detector_artifact_contract(contract_path)
    assert saved_contract == output["artifact_contract"]
    assert saved_contract["contract_digest"] == (
        lab.predicted_dsg_detector_artifact_contract_digest(saved_contract)
    )
    assert (
        main(
            [
                "--artifact-launch-report",
                str(contract_path),
                "--manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    launch_report = json.loads(capsys.readouterr().out)
    assert launch_report["action"] == (
        "predicted_dsg_detector_artifact_launch_report"
    )
    assert launch_report["ready_to_build"] is True
    assert launch_report["summary"]["planned_output_count"] == 5
    assert launch_report["build_command"].startswith(
        "python scripts/run_predicted_dsg.py --detector-jsonl "
    )
    assert "--predicted-dsg-evidence-report" in launch_report["build_command"]


def test_run_predicted_dsg_cli_reports_missing_detector_launch_readiness(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_predicted_dsg_script()
    main = cast(MainFn, getattr(module, "main"))
    manifest_path = _write_detector_run_manifest(tmp_path)
    contract_path = tmp_path / "contracts" / "predicted-dsg-detector-contract.json"

    assert (
        main(
            [
                "--preflight-manifest",
                str(manifest_path),
                "--artifact-contract",
                str(contract_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    manifest = lab.load_predicted_dsg_detector_run_manifest(manifest_path)
    Path(manifest["detector_jsonl_path"]).unlink()

    assert (
        main(
            [
                "--artifact-launch-report",
                str(contract_path),
                "--manifest",
                str(manifest_path),
            ]
        )
        == 1
    )

    launch_report = json.loads(capsys.readouterr().out)
    assert launch_report["action"] == (
        "predicted_dsg_detector_artifact_launch_report"
    )
    assert "error" not in launch_report
    assert launch_report["ready_to_build"] is False
    assert launch_report["detector_input"]["status"] == "missing"
    assert launch_report["detector_input"]["blocking_reasons"] == [
        "detector_input_missing",
        "readiness_checks_failed",
    ]


def test_run_predicted_dsg_cli_accepts_manifest(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_predicted_dsg_script()
    main = cast(MainFn, getattr(module, "main"))
    manifest_path = _write_detector_run_manifest(tmp_path)
    ledger_path = tmp_path / "ledgers" / "predicted-dsg-detector-run-ledger.json"

    assert main(["--manifest", str(manifest_path), "--run-ledger", str(ledger_path)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["action"] == "run_predicted_dsg_detector_run_manifest"
    assert output["manifest_path"] == str(manifest_path)
    assert output["ready"] is True
    assert Path(output["predicted_graph_report_path"]).exists()
    assert Path(output["predicted_dsg_evidence_report_path"]).exists()
    assert output["run_ledger_path"] == str(ledger_path)
    assert output["run_ledger_digest"] == (
        lab.load_predicted_dsg_detector_run_ledger(ledger_path)["ledger_digest"]
    )

    assert main(["--validate-run-ledger", str(ledger_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_predicted_dsg_detector_run_ledger"
    assert validation["valid"] is True
    assert main(["--compare-run-ledger", str(ledger_path)]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["action"] == "compare_predicted_dsg_detector_run_ledger"
    assert comparison["matches"] is True


def _write_detector_jsonl(tmp_path: Path) -> Path:
    path = tmp_path / "detector" / "real-rgbd-detections.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    records = _records()
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    for record in records:
        for field_name in ("rgb_path", "depth_path", "segmentation_path"):
            value = record[field_name]
            assert isinstance(value, str)
            asset_path = path.parent / value
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            asset_path.write_text(f"{field_name}\n", encoding="utf-8")
    return path


def _write_detector_run_manifest(tmp_path: Path) -> Path:
    root = tmp_path / "detector-run"
    detector_jsonl_path = _write_detector_jsonl(root)
    manifest = {
        "schema_version": (
            "dsg-spatialqa-lab.predicted-dsg-detector-run-manifest.v1"
        ),
        "detector_jsonl_path": str(detector_jsonl_path),
        "output_sequence_path": "predicted/detector-observations.json",
        "output_graph_path": "predicted/predicted-graph.json",
        "predicted_graph_report_path": "predicted/predicted-report.json",
        "detector_import_report_path": "predicted/detector-import-report.json",
        "predicted_dsg_evidence_report_path": (
            "predicted/predicted-dsg-evidence.json"
        ),
        "infer_relations": ["LEFT_OF", "RIGHT_OF", "NEAR"],
        "reference_frames": ["world"],
        "min_observation_count": 2,
        "min_object_observation_count": 2,
        "required_evidence_kinds": ["depth", "detector", "rgb"],
    }
    manifest_path = root / "predicted-dsg-detector-run-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _records() -> tuple[dict[str, object], ...]:
    return (
        {
            "schema_version": "dsg-spatialqa-lab.detector-observation-record.v1",
            "step": 1,
            "agent_id": "agent",
            "agent_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            "rgb_path": "frames/000001.rgb.png",
            "depth_path": "frames/000001.depth.png",
            "segmentation_path": "frames/000001.seg.png",
            "metadata": {
                "detector": "detic_real_trial",
                "detector_id": "detic-real-trial-v1",
                "source": "rgbd_detector",
            },
            "detections": [
                {
                    "object_id": "track_mug_1",
                    "label": "mug",
                    "pose": {"x": -0.4, "y": 1.0, "z": 0.78, "yaw": 0.0},
                    "bbox": {
                        "center": {"x": -0.4, "y": 1.0, "z": 0.78, "yaw": 0.0},
                        "size": [0.12, 0.12, 0.16],
                    },
                    "confidence": 0.93,
                    "visible": True,
                    "attributes": {"track_id": "track_mug_1"},
                },
                {
                    "object_id": "track_plate_1",
                    "label": "plate",
                    "pose": {"x": 0.2, "y": 1.0, "z": 0.72, "yaw": 0.0},
                    "bbox": {
                        "center": {"x": 0.2, "y": 1.0, "z": 0.72, "yaw": 0.0},
                        "size": [0.26, 0.26, 0.04],
                    },
                    "confidence": 0.88,
                    "visible": True,
                    "attributes": {"track_id": "track_plate_1"},
                },
            ],
        },
        {
            "schema_version": "dsg-spatialqa-lab.detector-observation-record.v1",
            "step": 2,
            "agent_id": "agent",
            "agent_pose": {"x": 0.1, "y": 0.0, "z": 0.0, "yaw": 0.0},
            "rgb_path": "frames/000002.rgb.png",
            "depth_path": "frames/000002.depth.png",
            "segmentation_path": "frames/000002.seg.png",
            "metadata": {
                "detector": "detic_real_trial",
                "detector_id": "detic-real-trial-v1",
                "source": "rgbd_detector",
            },
            "detections": [
                {
                    "object_id": "track_mug_1",
                    "label": "mug",
                    "pose": {"x": -0.2, "y": 1.1, "z": 0.78, "yaw": 0.0},
                    "bbox": {
                        "center": {"x": -0.2, "y": 1.1, "z": 0.78, "yaw": 0.0},
                        "size": [0.12, 0.12, 0.16],
                    },
                    "confidence": 0.52,
                    "visible": False,
                    "attributes": {
                        "hidden_reason": "not_detected_in_frame",
                        "track_id": "track_mug_1",
                    },
                }
            ],
        },
    )
