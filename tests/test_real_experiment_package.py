from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, TypedDict

from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch

import dsg_spatialqa_lab as lab


ROOT = Path(__file__).resolve().parents[1]
ASSEMBLER_SCRIPT = ROOT / "scripts" / "assemble_real_experiment.py"
RUN_REAL_SCRIPT = ROOT / "scripts" / "run_real_experiment.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


class ReadyPackageInputs(TypedDict):
    episode_paths: tuple[Path, ...]
    qa_delta_report_path: Path
    active_delta_report_path: Path
    dashboard_bundle_path: Path
    error_attribution_report_path: Path
    graph_eval_report_path: Path
    offline_control_matrix_report_path: Path
    offline_control_result_report_path: Path
    offline_import_report_path: Path
    predicted_dsg_evidence_report_path: Path
    predicted_graph_report_path: Path
    real_collection_report_path: Path


def load_assembler_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "assemble_real_experiment_script",
        ASSEMBLER_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_run_real_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_real_experiment_script",
        RUN_REAL_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_assemble_real_experiment_package_writes_manifest_and_readiness(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "assemble_real_experiment_package")
    inputs = _ready_package_inputs(tmp_path)
    manifest_path = tmp_path / "assembled" / "benchmark-manifest.json"
    readiness_path = tmp_path / "assembled" / "real-readiness.json"

    result = lab.assemble_real_experiment_package(
        dataset_name="ai2thor_real_smoke",
        episode_paths=inputs["episode_paths"],
        output_dir=tmp_path / "assembled" / "benchmark",
        manifest_path=manifest_path,
        readiness_report_path=readiness_path,
        max_qa_per_episode=20,
        tags=("benchmark", "real"),
        qa_eval_delta_report_paths=(inputs["qa_delta_report_path"],),
        active_task_delta_report_paths=(inputs["active_delta_report_path"],),
        dashboard_bundle_paths=(inputs["dashboard_bundle_path"],),
        error_attribution_report_paths=(inputs["error_attribution_report_path"],),
        graph_eval_report_paths=(inputs["graph_eval_report_path"],),
        offline_prediction_import_report_paths=(inputs["offline_import_report_path"],),
        offline_control_matrix_report_paths=(
            inputs["offline_control_matrix_report_path"],
        ),
        offline_control_result_report_paths=(
            inputs["offline_control_result_report_path"],
        ),
        predicted_dsg_evidence_report_paths=(
            inputs["predicted_dsg_evidence_report_path"],
        ),
        predicted_graph_report_paths=(inputs["predicted_graph_report_path"],),
        real_collection_report_paths=(inputs["real_collection_report_path"],),
        min_episode_count=1,
        min_scene_count=1,
        min_qa_count=8,
        required_control_kinds=("vlm",),
        required_predicted_input_kinds=("observation_sequence",),
    )
    manifest = lab.load_benchmark_manifest(manifest_path)
    readiness = lab.load_real_experiment_readiness_report(readiness_path)

    assert result["schema_version"] == "dsg-spatialqa-lab.real-experiment-package.v1"
    assert result["action"] == "assemble_real_experiment_package"
    assert result["manifest_path"] == str(manifest_path)
    assert result["readiness_report_path"] == str(readiness_path)
    assert result["manifest_digest"] == manifest["manifest_digest"]
    assert result["readiness_report_digest"] == readiness["report_digest"]
    assert result["ready"] is True
    assert readiness["readiness"]["ready"] is True
    assert readiness["declared_data_source_kind"] == "real"
    assert readiness["artifact_summary"]["offline_control_matrix_ready"] is True
    assert readiness["artifact_summary"]["offline_control_result_ready"] is True
    assert readiness["artifact_summary"]["offline_control_kinds"] == ["vlm"]
    assert readiness["artifact_summary"]["predicted_dsg_evidence_ready"] is True
    assert readiness["artifact_summary"]["predicted_input_kinds"] == [
        "observation_sequence"
    ]
    assert readiness["artifact_summary"]["real_collection_ready"] is True
    assert readiness["artifact_summary"]["real_collection_source_kinds"] == ["ai2thor"]


def test_assemble_real_experiment_cli_outputs_structured_readiness(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_assembler_script()
    main = getattr(module, "main")
    inputs = _ready_package_inputs(tmp_path)
    manifest_path = tmp_path / "cli" / "benchmark-manifest.json"
    readiness_path = tmp_path / "cli" / "real-readiness.json"

    assert main(
        [
            "--episode",
            str(inputs["episode_paths"][0]),
            "--dataset-name",
            "ai2thor_real_smoke",
            "--output-dir",
            str(tmp_path / "cli" / "benchmark"),
            "--manifest",
            str(manifest_path),
            "--readiness-report",
            str(readiness_path),
            "--max-qa-per-episode",
            "20",
            "--tag",
            "benchmark",
            "--tag",
            "real",
            "--qa-eval-delta-report",
            str(inputs["qa_delta_report_path"]),
            "--active-task-delta-report",
            str(inputs["active_delta_report_path"]),
            "--dashboard-bundle",
            str(inputs["dashboard_bundle_path"]),
            "--error-attribution-report",
            str(inputs["error_attribution_report_path"]),
            "--graph-eval-report",
            str(inputs["graph_eval_report_path"]),
                "--offline-prediction-import-report",
                str(inputs["offline_import_report_path"]),
                "--offline-control-matrix-report",
                str(inputs["offline_control_matrix_report_path"]),
                "--offline-control-result-report",
                str(inputs["offline_control_result_report_path"]),
                "--predicted-dsg-evidence-report",
            str(inputs["predicted_dsg_evidence_report_path"]),
            "--predicted-graph-report",
            str(inputs["predicted_graph_report_path"]),
            "--real-collection-report",
            str(inputs["real_collection_report_path"]),
            "--min-episode-count",
            "1",
            "--min-scene-count",
            "1",
            "--min-qa-count",
            "8",
            "--required-control-kind",
            "vlm",
            "--required-predicted-input-kind",
            "observation_sequence",
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    readiness = lab.load_real_experiment_readiness_report(readiness_path)
    assert output["action"] == "assemble_real_experiment_package"
    assert output["ready"] is True
    assert output["readiness_report_digest"] == readiness["report_digest"]
    assert manifest_path.exists()
    assert readiness_path.exists()


def test_run_real_experiment_package_writes_summary_and_linked_record(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "run_real_experiment_package")
    inputs = _ready_package_inputs(tmp_path)
    root = tmp_path / "run"
    manifest_path = root / "benchmark-manifest.json"
    readiness_path = root / "real-readiness.json"
    summary_path = root / "experiment-summary.json"
    record_path = root / "experiment-record.json"

    result = lab.run_real_experiment_package(
        dataset_name="ai2thor_real_smoke",
        episode_paths=inputs["episode_paths"],
        output_dir=root / "benchmark",
        manifest_path=manifest_path,
        readiness_report_path=readiness_path,
        summary_report_path=summary_path,
        record_path=record_path,
        max_qa_per_episode=20,
        tags=("benchmark", "real"),
        qa_eval_delta_report_paths=(inputs["qa_delta_report_path"],),
        active_task_delta_report_paths=(inputs["active_delta_report_path"],),
        dashboard_bundle_paths=(inputs["dashboard_bundle_path"],),
        error_attribution_report_paths=(inputs["error_attribution_report_path"],),
        graph_eval_report_paths=(inputs["graph_eval_report_path"],),
        offline_prediction_import_report_paths=(inputs["offline_import_report_path"],),
        offline_control_matrix_report_paths=(
            inputs["offline_control_matrix_report_path"],
        ),
        offline_control_result_report_paths=(
            inputs["offline_control_result_report_path"],
        ),
        predicted_dsg_evidence_report_paths=(
            inputs["predicted_dsg_evidence_report_path"],
        ),
        predicted_graph_report_paths=(inputs["predicted_graph_report_path"],),
        real_collection_report_paths=(inputs["real_collection_report_path"],),
        min_episode_count=1,
        min_scene_count=1,
        min_qa_count=8,
        required_control_kinds=("vlm",),
        required_predicted_input_kinds=("observation_sequence",),
    )
    readiness = lab.load_real_experiment_readiness_report(readiness_path)
    summary = lab.load_experiment_summary_report(summary_path)
    record = lab.load_experiment_record(record_path)

    assert result["schema_version"] == "dsg-spatialqa-lab.real-experiment-run.v1"
    assert result["action"] == "run_real_experiment_package"
    assert result["ready"] is True
    assert result["manifest_path"] == str(manifest_path)
    assert result["readiness_report_path"] == str(readiness_path)
    assert result["summary_report_path"] == str(summary_path)
    assert result["record_path"] == str(record_path)
    assert result["readiness_report_digest"] == readiness["report_digest"]
    assert result["summary_report_digest"] == summary["report_digest"]
    assert result["record_digest"] == record["record_digest"]
    assert result["real_package_status"] == "ready"
    assert readiness["artifact_summary"]["offline_control_result_ready"] is True
    assert record["real_readiness_report_digest"] == readiness["report_digest"]
    assert record["real_package_status"] == "ready"
    assert record["summary_report_digest"] == summary["report_digest"]
    assert lab.validate_experiment_record(record)["valid"] is True
    assert lab.compare_experiment_record(record)["matches"] is True


def test_run_real_experiment_package_runs_offline_control_import_manifest(
    tmp_path: Path,
) -> None:
    inputs = _ready_package_inputs(tmp_path)
    offline_manifest_path = _offline_control_import_manifest_for_real_run(
        tmp_path,
        inputs,
    )
    root = tmp_path / "manifest-run"
    manifest_path = root / "benchmark-manifest.json"
    readiness_path = root / "real-readiness.json"
    summary_path = root / "experiment-summary.json"
    record_path = root / "experiment-record.json"

    result = lab.run_real_experiment_package(
        dataset_name="ai2thor_real_smoke",
        episode_paths=inputs["episode_paths"],
        output_dir=root / "benchmark",
        manifest_path=manifest_path,
        readiness_report_path=readiness_path,
        summary_report_path=summary_path,
        record_path=record_path,
        max_qa_per_episode=20,
        tags=("benchmark", "real"),
        active_task_delta_report_paths=(inputs["active_delta_report_path"],),
        dashboard_bundle_paths=(inputs["dashboard_bundle_path"],),
        error_attribution_report_paths=(inputs["error_attribution_report_path"],),
        graph_eval_report_paths=(inputs["graph_eval_report_path"],),
        offline_control_import_manifest_path=offline_manifest_path,
        predicted_dsg_evidence_report_paths=(
            inputs["predicted_dsg_evidence_report_path"],
        ),
        predicted_graph_report_paths=(inputs["predicted_graph_report_path"],),
        real_collection_report_paths=(inputs["real_collection_report_path"],),
        min_episode_count=1,
        min_scene_count=1,
        min_qa_count=8,
        required_control_kinds=(
            "caption_memory",
            "graph_text",
            "multi_frame_vlm",
            "vlm",
        ),
        required_predicted_input_kinds=("observation_sequence",),
    )
    readiness = lab.load_real_experiment_readiness_report(readiness_path)
    summary = lab.load_experiment_summary_report(summary_path)
    record = lab.load_experiment_record(record_path)

    assert result["ready"] is True
    assert result["offline_control_import"]["ready"] is True
    assert result["offline_control_import_manifest_path"] == str(offline_manifest_path)
    assert result["offline_control_import_manifest_digest"] == result[
        "offline_control_import"
    ]["manifest_digest"]
    assert sorted(result["generated_offline_control_source_kinds"]) == [
        "caption_memory",
        "graph_text",
        "multi_frame_vlm",
        "vlm",
    ]
    assert len(result["generated_offline_prediction_import_report_paths"]) == 4
    assert len(result["generated_qa_eval_delta_report_paths"]) == 4
    assert Path(result["generated_offline_control_matrix_report_path"]).exists()
    assert Path(result["generated_offline_control_result_report_path"]).exists()
    assert readiness["artifact_summary"]["offline_control_result_ready"] is True
    assert readiness["artifact_summary"]["offline_control_kinds"] == [
        "caption_memory",
        "graph_text",
        "multi_frame_vlm",
        "vlm",
    ]
    assert readiness["artifact_summary"]["qa_delta_baseline_names"] == [
        "caption_memory",
        "graph_text",
        "multi_frame_vlm",
        "vlm",
    ]
    assert summary["summary"]["qa_eval_delta_report_count"] == 4
    assert record["real_package_status"] == "ready"
    assert lab.compare_experiment_record(record)["matches"] is True


def test_run_real_experiment_package_runs_predicted_dsg_detector_run_manifest(
    tmp_path: Path,
) -> None:
    inputs = _ready_package_inputs(tmp_path)
    (
        predicted_manifest_path,
        graph_eval_report_path,
        error_attribution_report_path,
    ) = _predicted_dsg_detector_run_manifest_for_real_run(tmp_path, inputs)
    root = tmp_path / "predicted-manifest-run"
    manifest_path = root / "benchmark-manifest.json"
    readiness_path = root / "real-readiness.json"
    summary_path = root / "experiment-summary.json"
    record_path = root / "experiment-record.json"

    result = lab.run_real_experiment_package(
        dataset_name="ai2thor_real_smoke",
        episode_paths=inputs["episode_paths"],
        output_dir=root / "benchmark",
        manifest_path=manifest_path,
        readiness_report_path=readiness_path,
        summary_report_path=summary_path,
        record_path=record_path,
        max_qa_per_episode=20,
        tags=("benchmark", "real"),
        qa_eval_delta_report_paths=(inputs["qa_delta_report_path"],),
        active_task_delta_report_paths=(inputs["active_delta_report_path"],),
        dashboard_bundle_paths=(inputs["dashboard_bundle_path"],),
        error_attribution_report_paths=(error_attribution_report_path,),
        graph_eval_report_paths=(graph_eval_report_path,),
        offline_prediction_import_report_paths=(inputs["offline_import_report_path"],),
        offline_control_matrix_report_paths=(
            inputs["offline_control_matrix_report_path"],
        ),
        offline_control_result_report_paths=(
            inputs["offline_control_result_report_path"],
        ),
        predicted_dsg_detector_run_manifest_path=predicted_manifest_path,
        real_collection_report_paths=(inputs["real_collection_report_path"],),
        min_episode_count=1,
        min_scene_count=1,
        min_qa_count=8,
        required_control_kinds=("vlm",),
        required_predicted_input_kinds=("observation_sequence",),
    )
    readiness = lab.load_real_experiment_readiness_report(readiness_path)
    summary = lab.load_experiment_summary_report(summary_path)
    record = lab.load_experiment_record(record_path)

    assert result["ready"] is True
    assert result["predicted_dsg_detector_run"]["ready"] is True
    assert result["predicted_dsg_detector_run_manifest_path"] == str(
        predicted_manifest_path
    )
    assert result["predicted_dsg_detector_run_manifest_digest"] == result[
        "predicted_dsg_detector_run"
    ]["manifest_digest"]
    assert Path(result["generated_predicted_graph_report_path"]).exists()
    assert Path(result["generated_predicted_dsg_evidence_report_path"]).exists()
    assert Path(result["generated_predicted_graph_path"]).exists()
    assert Path(result["generated_detector_import_report_path"]).exists()
    assert readiness["artifact_summary"]["predicted_graph_ready"] is True
    assert readiness["artifact_summary"]["predicted_dsg_evidence_ready"] is True
    assert readiness["artifact_summary"]["predicted_input_kinds"] == [
        "observation_sequence"
    ]
    assert summary["summary"]["graph_construction_diagnostic_count"] == 1
    assert record["real_package_status"] == "ready"
    assert lab.compare_experiment_record(record)["matches"] is True


def test_run_real_experiment_cli_accepts_offline_control_import_manifest(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_real_script()
    main = getattr(module, "main")
    inputs = _ready_package_inputs(tmp_path)
    offline_manifest_path = _offline_control_import_manifest_for_real_run(
        tmp_path,
        inputs,
    )
    root = tmp_path / "cli-manifest-run"
    manifest_path = root / "benchmark-manifest.json"
    readiness_path = root / "real-readiness.json"
    summary_path = root / "experiment-summary.json"
    record_path = root / "experiment-record.json"

    assert main(
        [
            "--episode",
            str(inputs["episode_paths"][0]),
            "--dataset-name",
            "ai2thor_real_smoke",
            "--output-dir",
            str(root / "benchmark"),
            "--manifest",
            str(manifest_path),
            "--readiness-report",
            str(readiness_path),
            "--summary-report",
            str(summary_path),
            "--record",
            str(record_path),
            "--max-qa-per-episode",
            "20",
            "--tag",
            "benchmark",
            "--tag",
            "real",
            "--active-task-delta-report",
            str(inputs["active_delta_report_path"]),
            "--dashboard-bundle",
            str(inputs["dashboard_bundle_path"]),
            "--error-attribution-report",
            str(inputs["error_attribution_report_path"]),
            "--graph-eval-report",
            str(inputs["graph_eval_report_path"]),
            "--offline-control-import-manifest",
            str(offline_manifest_path),
            "--predicted-dsg-evidence-report",
            str(inputs["predicted_dsg_evidence_report_path"]),
            "--predicted-graph-report",
            str(inputs["predicted_graph_report_path"]),
            "--real-collection-report",
            str(inputs["real_collection_report_path"]),
            "--min-episode-count",
            "1",
            "--min-scene-count",
            "1",
            "--min-qa-count",
            "8",
            "--required-predicted-input-kind",
            "observation_sequence",
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["action"] == "run_real_experiment_package"
    assert output["ready"] is True
    assert output["offline_control_import"]["ready"] is True
    assert output["offline_control_import_manifest_path"] == str(offline_manifest_path)
    assert len(output["generated_qa_eval_delta_report_paths"]) == 4
    assert Path(output["generated_offline_control_matrix_report_path"]).exists()
    assert Path(output["generated_offline_control_result_report_path"]).exists()


def test_run_real_experiment_cli_accepts_predicted_dsg_detector_run_manifest(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_real_script()
    main = getattr(module, "main")
    inputs = _ready_package_inputs(tmp_path)
    (
        predicted_manifest_path,
        graph_eval_report_path,
        error_attribution_report_path,
    ) = _predicted_dsg_detector_run_manifest_for_real_run(tmp_path, inputs)
    root = tmp_path / "cli-predicted-manifest-run"
    manifest_path = root / "benchmark-manifest.json"
    readiness_path = root / "real-readiness.json"
    summary_path = root / "experiment-summary.json"
    record_path = root / "experiment-record.json"

    assert main(
        [
            "--episode",
            str(inputs["episode_paths"][0]),
            "--dataset-name",
            "ai2thor_real_smoke",
            "--output-dir",
            str(root / "benchmark"),
            "--manifest",
            str(manifest_path),
            "--readiness-report",
            str(readiness_path),
            "--summary-report",
            str(summary_path),
            "--record",
            str(record_path),
            "--max-qa-per-episode",
            "20",
            "--tag",
            "benchmark",
            "--tag",
            "real",
            "--qa-eval-delta-report",
            str(inputs["qa_delta_report_path"]),
            "--active-task-delta-report",
            str(inputs["active_delta_report_path"]),
            "--dashboard-bundle",
            str(inputs["dashboard_bundle_path"]),
            "--error-attribution-report",
            str(error_attribution_report_path),
            "--graph-eval-report",
            str(graph_eval_report_path),
            "--offline-prediction-import-report",
            str(inputs["offline_import_report_path"]),
            "--offline-control-matrix-report",
            str(inputs["offline_control_matrix_report_path"]),
            "--offline-control-result-report",
            str(inputs["offline_control_result_report_path"]),
            "--predicted-dsg-detector-run-manifest",
            str(predicted_manifest_path),
            "--real-collection-report",
            str(inputs["real_collection_report_path"]),
            "--min-episode-count",
            "1",
            "--min-scene-count",
            "1",
            "--min-qa-count",
            "8",
            "--required-control-kind",
            "vlm",
            "--required-predicted-input-kind",
            "observation_sequence",
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["ready"] is True
    assert output["predicted_dsg_detector_run"]["ready"] is True
    assert output["predicted_dsg_detector_run_manifest_path"] == str(
        predicted_manifest_path
    )
    assert Path(output["generated_predicted_graph_report_path"]).exists()
    assert Path(output["generated_predicted_dsg_evidence_report_path"]).exists()


def test_run_real_experiment_manifest_runs_complete_local_handoff(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "REAL_EXPERIMENT_RUN_MANIFEST_SCHEMA_VERSION")
    assert hasattr(lab, "REAL_EXPERIMENT_RUN_LEDGER_SCHEMA_VERSION")
    assert hasattr(lab, "load_real_experiment_run_manifest")
    assert hasattr(lab, "load_real_experiment_run_ledger")
    assert hasattr(lab, "real_experiment_run_manifest_digest")
    assert hasattr(lab, "real_experiment_run_ledger_digest")
    assert hasattr(lab, "run_real_experiment_manifest")
    assert hasattr(lab, "validate_real_experiment_run_ledger")
    assert hasattr(lab, "compare_real_experiment_run_ledger")
    assert hasattr(lab, "REAL_EXPERIMENT_SMOKE_RUN_RUNBOOK_SCHEMA_VERSION")
    assert hasattr(lab, "real_experiment_smoke_run_runbook")
    assert hasattr(lab, "real_experiment_smoke_run_runbook_digest")
    assert hasattr(lab, "save_real_experiment_smoke_run_runbook")
    assert hasattr(lab, "load_real_experiment_smoke_run_runbook")
    assert hasattr(lab, "validate_real_experiment_smoke_run_runbook")
    assert hasattr(lab, "compare_real_experiment_smoke_run_runbook")
    inputs = _ready_package_inputs(tmp_path)
    run_manifest_path = _real_experiment_run_manifest(tmp_path, inputs)
    manifest = lab.load_real_experiment_run_manifest(run_manifest_path)
    run_ledger_path = Path(manifest["real_experiment_run_ledger_path"])
    execution_packet_path = tmp_path / "run-manifest" / "execution-packet.json"
    execution_receipt_path = tmp_path / "run-manifest" / "execution-receipt.json"
    execution_packet = _execution_packet_for_run_manifest(
        run_manifest_path,
        execution_packet_path,
    )
    smoke_checklist = lab.real_experiment_smoke_run_checklist(
        execution_packet,
        execution_packet_path=execution_packet_path,
        execution_receipt_output_path=execution_receipt_path,
    )
    smoke_checklist_path = tmp_path / "run-manifest" / "smoke-run-checklist.json"
    assert (
        lab.save_real_experiment_smoke_run_checklist(
            smoke_checklist,
            smoke_checklist_path,
        )
        == smoke_checklist_path
    )
    loaded_smoke_checklist = lab.load_real_experiment_smoke_run_checklist(
        smoke_checklist_path
    )
    assert loaded_smoke_checklist == smoke_checklist
    smoke_validation = lab.validate_real_experiment_smoke_run_checklist(
        loaded_smoke_checklist
    )
    assert smoke_validation["valid"] is True
    smoke_comparison = lab.compare_real_experiment_smoke_run_checklist(
        loaded_smoke_checklist
    )
    assert smoke_comparison["matches"] is True
    assert smoke_checklist["ready_to_start"] is True
    assert smoke_checklist["blocked"] is False
    assert smoke_checklist["execution_receipt_output_path"] == str(
        execution_receipt_path
    )
    assert [step["key"] for step in smoke_checklist["steps"]] == [
        "validate_execution_packet",
        "compare_execution_packet",
        "validate_launch_report",
        "compare_launch_report",
        "validate_primary_evidence_acceptance_report",
        "compare_primary_evidence_acceptance_report",
        "preflight_run_manifest",
        "run_real_experiment",
        "validate_run_ledger",
        "compare_run_ledger",
        "write_execution_receipt",
        "validate_execution_receipt",
        "compare_execution_receipt",
    ]
    assert smoke_checklist["summary"] == {
        "audit_step_count": 6,
        "execute_step_count": 2,
        "required_step_count": 12,
        "review_step_count": 5,
        "step_count": 13,
    }
    assert smoke_checklist["planned_outputs"]["execution_receipt"] == str(
        execution_receipt_path
    )
    assert smoke_checklist["planned_outputs"]["real_experiment_run_ledger"] == str(
        run_ledger_path
    )
    smoke_runbook = lab.real_experiment_smoke_run_runbook(
        loaded_smoke_checklist,
        smoke_run_checklist_path=smoke_checklist_path,
    )
    smoke_runbook_path = tmp_path / "run-manifest" / "smoke-run-runbook.json"
    assert lab.save_real_experiment_smoke_run_runbook(
        smoke_runbook,
        smoke_runbook_path,
    ) == smoke_runbook_path
    loaded_smoke_runbook = lab.load_real_experiment_smoke_run_runbook(
        smoke_runbook_path
    )
    assert loaded_smoke_runbook == smoke_runbook
    smoke_runbook_validation = lab.validate_real_experiment_smoke_run_runbook(
        loaded_smoke_runbook
    )
    assert smoke_runbook_validation["valid"] is True
    smoke_runbook_comparison = lab.compare_real_experiment_smoke_run_runbook(
        loaded_smoke_runbook
    )
    assert smoke_runbook_comparison["matches"] is True
    assert smoke_runbook["action"] == "real_experiment_smoke_run_runbook"
    assert smoke_runbook["smoke_run_checklist_path"] == str(smoke_checklist_path)
    assert smoke_runbook["ready_to_start"] is True
    assert smoke_runbook["blocked"] is False
    assert smoke_runbook["summary"] == {
        "audit_command_count": 6,
        "command_count": 13,
        "execute_command_count": 2,
        "required_command_count": 12,
        "review_command_count": 5,
    }
    assert [command["key"] for command in smoke_runbook["commands"]] == [
        step["key"] for step in smoke_checklist["steps"]
    ]
    assert smoke_runbook["commands"][7]["key"] == "run_real_experiment"
    assert "--approved-execution-packet" in smoke_runbook["commands"][7]["command"]

    blocked_packet = dict(execution_packet)
    blocked_packet["ready_to_execute"] = False
    blocked_packet["execution_blocked"] = True
    blocked_packet["execution_commands"] = []
    blocked_packet["packet_digest"] = lab.real_experiment_execution_packet_digest(
        blocked_packet
    )
    blocked_execution_packet_path = (
        tmp_path / "run-manifest" / "blocked-execution-packet.json"
    )
    lab.save_real_experiment_execution_packet(
        blocked_packet,
        blocked_execution_packet_path,
    )
    blocked_checklist = lab.real_experiment_smoke_run_checklist(
        blocked_packet,
        execution_packet_path=execution_packet_path,
    )
    assert blocked_checklist["ready_to_start"] is False
    assert blocked_checklist["blocked"] is True
    assert [step["phase"] for step in blocked_checklist["steps"]] == [
        "audit",
        "audit",
        "audit",
        "audit",
        "audit",
        "audit",
    ]

    missing_receipt = lab.real_experiment_execution_receipt(
        execution_packet,
        execution_packet_path=execution_packet_path,
    )
    missing_research_review = lab.real_experiment_research_review(
        missing_receipt,
        execution_receipt_path=execution_receipt_path,
    )
    assert missing_research_review["ready_for_research_review"] is False
    assert missing_research_review["blocked"] is True
    assert "execution_receipt_not_ready" in missing_research_review["blockers"]
    assert missing_receipt["ready_to_review"] is False
    assert missing_receipt["artifact_summary"]["missing_artifact_count"] == 8
    assert {
        artifact["role"]
        for artifact in missing_receipt["artifacts"]
        if artifact["status"] == "missing"
    } == {
        "benchmark_manifest",
        "experiment_record",
        "experiment_summary",
        "offline_control_import_run_ledger",
        "output_dir",
        "predicted_dsg_detector_run_ledger",
        "real_readiness_report",
        "real_experiment_run_ledger",
    }

    result = lab.run_real_experiment_manifest(run_manifest_path)
    direct_run_receipt = lab.real_experiment_execution_receipt(
        execution_packet,
        execution_packet_path=execution_packet_path,
    )
    assert direct_run_receipt["ready_to_review"] is False
    assert direct_run_receipt["run_ledger_approval"] == {
        "path": str(run_ledger_path),
        "present": True,
        "ready": False,
        "required": True,
        "status": "not_approved",
        "validation_valid": True,
    }
    approved_result = lab.run_real_experiment_manifest(
        run_manifest_path,
        approved_execution_packet_path=execution_packet_path,
    )
    assert approved_result["ready"] is True
    assert approved_result["execution_approval"] == {
        "approved_execution_packet_path": str(execution_packet_path),
        "approved_run_manifest_path": str(run_manifest_path),
        "execution_packet_digest": execution_packet["packet_digest"],
        "manifest_matches": True,
        "ready": True,
        "ready_to_execute": True,
        "required": True,
        "status": "approved",
        "validation_valid": True,
    }
    assert approved_result["real_experiment_run_ledger_path"] == str(
        run_ledger_path
    )
    assert Path(approved_result["real_experiment_run_ledger_path"]).exists()
    run_ledger = lab.load_real_experiment_run_ledger(run_ledger_path)
    assert run_ledger["action"] == "real_experiment_run_ledger"
    assert run_ledger["ready"] is True
    assert run_ledger["execution_approval"]["status"] == "approved"
    assert run_ledger["ledger_digest"] == (
        lab.real_experiment_run_ledger_digest(run_ledger)
    )
    run_ledger_validation = lab.validate_real_experiment_run_ledger(run_ledger)
    assert run_ledger_validation["valid"] is True
    run_ledger_comparison = lab.compare_real_experiment_run_ledger(run_ledger)
    assert run_ledger_comparison["matches"] is True
    blocked_result = lab.run_real_experiment_manifest(
        run_manifest_path,
        approved_execution_packet_path=blocked_execution_packet_path,
    )
    assert blocked_result["ready"] is False
    assert blocked_result["execution_approval"]["ready"] is False
    assert blocked_result["execution_approval"]["status"] == "not_ready"
    record = lab.load_experiment_record(result["record_path"])
    execution_receipt = lab.real_experiment_execution_receipt(
        execution_packet,
        execution_packet_path=execution_packet_path,
    )
    assert (
        lab.save_real_experiment_execution_receipt(
            execution_receipt,
            execution_receipt_path,
        )
        == execution_receipt_path
    )
    loaded_receipt = lab.load_real_experiment_execution_receipt(
        execution_receipt_path
    )
    assert loaded_receipt == execution_receipt
    receipt_validation = lab.validate_real_experiment_execution_receipt(
        loaded_receipt
    )
    assert receipt_validation["valid"] is True
    assert receipt_validation["receipt_digest"] == execution_receipt["receipt_digest"]
    receipt_comparison = lab.compare_real_experiment_execution_receipt(loaded_receipt)
    assert receipt_comparison["matches"] is True
    assert receipt_comparison["saved_digest"] == execution_receipt["receipt_digest"]
    assert receipt_comparison["current_digest"] == execution_receipt["receipt_digest"]
    research_review = lab.real_experiment_research_review(
        loaded_receipt,
        execution_receipt_path=execution_receipt_path,
    )
    research_review_path = tmp_path / "run-manifest" / "research-review.json"
    assert (
        lab.save_real_experiment_research_review(
            research_review,
            research_review_path,
        )
        == research_review_path
    )
    loaded_research_review = lab.load_real_experiment_research_review(
        research_review_path
    )
    assert loaded_research_review == research_review
    review_validation = lab.validate_real_experiment_research_review(
        loaded_research_review
    )
    assert review_validation["valid"] is True
    assert review_validation["review_digest"] == research_review["review_digest"]
    review_comparison = lab.compare_real_experiment_research_review(
        loaded_research_review
    )
    assert review_comparison["matches"] is True
    assert review_comparison["saved_digest"] == research_review["review_digest"]
    assert review_comparison["current_digest"] == research_review["review_digest"]
    assert research_review["ready_for_research_review"] is True
    assert research_review["blocked"] is False
    assert research_review["blockers"] == []
    assert research_review["research_question_summary"] == {
        "available_count": 4,
        "conclusive_count": 4,
        "inconclusive_count": 0,
        "required_count": 4,
    }
    assert set(research_review["research_questions"]) == {
        "dynamic_memory",
        "graph_tool_query",
        "interactive_task",
        "spatial_qa",
    }
    assert research_review["evidence_summary"][
        "graph_construction_diagnostic_count"
    ] >= 1
    assert research_review["evidence_summary"][
        "failure_linkage_diagnostic_count"
    ] >= 1
    assert research_review["evidence_summary"]["source_profile_count"] >= 1
    default_claim = lab.real_experiment_claim_readiness(
        loaded_research_review,
        research_review_path=research_review_path,
    )
    assert default_claim["claim_ready"] is False
    assert default_claim["status"] == "pilot_only"
    assert {
        blocker["name"] for blocker in default_claim["blockers"]
    } == {"episode_count", "qa_count"}
    assert default_claim["claim_gap_summary"] == {
        "evidence_gap_count": 0,
        "failed_check_count": 2,
        "research_question_gap_count": 0,
        "research_question_gaps": {
            "inconclusive_keys": [],
            "missing_keys": [],
            "verdicts": {
                key: row["verdict"]
                for key, row in loaded_research_review["research_questions"].items()
            },
        },
        "scale_deficit_count": 2,
        "scale_deficits": {
            "episode_count": {
                "actual": 1,
                "deficit": 2,
                "minimum": 3,
                "threshold_field": "min_episode_count",
            },
            "qa_count": {
                "actual": 9,
                "deficit": 21,
                "minimum": 30,
                "threshold_field": "min_qa_count",
            },
        },
        "target_thresholds": {
            "min_dynamic_qa_count": 1,
            "min_episode_count": 3,
            "min_qa_count": 30,
            "min_scene_count": 1,
        },
    }
    assert default_claim["claim_scope_assessment"] == {
        "active_scale_ready": False,
        "below_default_threshold_fields": [],
        "claim_scope": "pilot_only",
        "default_scale_deficits": {
            "episode_count": {
                "actual": 1,
                "deficit": 2,
                "minimum": 3,
                "threshold_field": "min_episode_count",
            },
            "qa_count": {
                "actual": 9,
                "deficit": 21,
                "minimum": 30,
                "threshold_field": "min_qa_count",
            },
        },
        "default_scale_ready": False,
        "default_thresholds": {
            "min_dynamic_qa_count": 1,
            "min_episode_count": 3,
            "min_qa_count": 30,
            "min_scene_count": 1,
        },
        "full_scale_claim_permitted": False,
        "threshold_profile": "default",
    }
    assert default_claim["claim_scope_next_actions"] == []
    assert default_claim["claim_scope_handoff_plan"]["required"] is False
    assert default_claim["claim_scope_handoff_plan"]["commands"] == {}
    assert default_claim["next_actions"] == [
        {
            "action": "expand_real_benchmark_scale",
            "blocker_names": ["episode_count", "qa_count"],
            "current_scale": {
                "dynamic_qa_count": 4,
                "episode_count": 1,
                "qa_count": 9,
                "scene_count": 1,
            },
            "order": 1,
            "reason": "Real benchmark scale is below the saved claim policy.",
            "scale_deficits": default_claim["claim_gap_summary"]["scale_deficits"],
            "target_thresholds": default_claim["claim_gap_summary"][
                "target_thresholds"
            ],
            "track": "real_data",
        }
    ]
    next_handoff_plan = default_claim["next_handoff_plan"]
    expected_handoff_root = run_manifest_path.with_name(
        "next-claim-ready-handoff"
    )
    assert next_handoff_plan["required"] is True
    assert next_handoff_plan["source_run_manifest_path"] == str(run_manifest_path)
    assert next_handoff_plan["handoff_root"] == str(expected_handoff_root)
    assert next_handoff_plan["tracks_to_expand"] == ["real_data"]
    assert next_handoff_plan["target_thresholds"] == {
        "min_dynamic_qa_count": 1,
        "min_episode_count": 3,
        "min_qa_count": 30,
        "min_scene_count": 1,
    }
    assert next_handoff_plan["threshold_updates"] == {
        "min_episode_count": {
            "current": 1,
            "increase": 2,
            "target": 3,
        },
        "min_qa_count": {
            "current": 8,
            "increase": 22,
            "target": 30,
        },
    }
    planned_episode_paths = [
        expected_handoff_root / "inputs/episodes/ai2thor_real_smoke-episode-002.jsonl",
        expected_handoff_root / "inputs/episodes/ai2thor_real_smoke-episode-003.jsonl",
    ]
    assert next_handoff_plan["episode_collection_plan"] == {
        "current_episode_count": 1,
        "episode_deficit": 2,
        "existing_episode_paths": [str(inputs["episode_paths"][0])],
        "planned_episode_paths": [str(path) for path in planned_episode_paths],
        "target_episode_count": 3,
    }
    expected_candidate_prediction_path = (
        expected_handoff_root / "inputs/candidate/predicted-graph-tool.jsonl"
    )
    expected_detector_jsonl_path = (
        expected_handoff_root / "inputs/predicted-dsg/detector-rgbd.jsonl"
    )
    expected_control_prediction_paths = {
        "caption_memory": (
            expected_handoff_root
            / "inputs/offline-controls/caption_memory.jsonl"
        ),
        "graph_text": (
            expected_handoff_root / "inputs/offline-controls/graph_text.jsonl"
        ),
        "multi_frame_vlm": (
            expected_handoff_root
            / "inputs/offline-controls/multi_frame_vlm.jsonl"
        ),
        "vlm": expected_handoff_root / "inputs/offline-controls/vlm.jsonl",
    }
    assert next_handoff_plan["external_artifact_slots"] == {
        "candidate_prediction_path": str(expected_candidate_prediction_path),
        "detector_jsonl_path": str(expected_detector_jsonl_path),
        "offline_control_prediction_paths": {
            key: str(path)
            for key, path in expected_control_prediction_paths.items()
        },
        "track_order": ["real_controls", "predicted_dsg"],
    }
    assert next_handoff_plan["required_predicted_input_kinds"] == [
        "observation_sequence"
    ]
    expected_handoff_run_manifest_path = (
        expected_handoff_root / "real-experiment-run-manifest.json"
    )
    expected_external_contracts_path = (
        expected_handoff_root / "real-experiment-external-artifact-contracts.json"
    )
    expected_launch_report_path = (
        expected_handoff_root / "real-experiment-external-artifact-launch-report.json"
    )
    expected_real_collection_report_path = (
        expected_handoff_root / "inputs/real-collection-report.json"
    )
    expected_real_collection_request_bundle_path = (
        expected_handoff_root / "real-collection-request-bundle.json"
    )
    expected_offline_manifest_path = (
        expected_handoff_root / "offline-control-import-manifest.json"
    )
    expected_offline_request_bundle_path = (
        expected_handoff_root / "offline-control-prediction-request-bundle.json"
    )
    expected_offline_receipt_bundle_path = (
        expected_handoff_root / "offline-control-prediction-receipt-bundle.json"
    )
    expected_predicted_manifest_path = (
        expected_handoff_root / "predicted-dsg-detector-run-manifest.json"
    )
    expected_predicted_request_bundle_path = (
        expected_handoff_root / "predicted-dsg-detector-request-bundle.json"
    )
    expected_predicted_receipt_bundle_path = (
        expected_handoff_root / "predicted-dsg-detector-receipt-bundle.json"
    )
    expected_primary_status_path = (
        expected_handoff_root / "real-experiment-primary-evidence-status.json"
    )
    expected_primary_request_package_path = (
        expected_handoff_root
        / "real-experiment-primary-evidence-request-package.json"
    )
    expected_primary_return_checklist_path = (
        expected_handoff_root
        / "real-experiment-primary-evidence-return-checklist.json"
    )
    expected_primary_return_progress_path = (
        expected_handoff_root / "real-experiment-primary-evidence-return-progress.json"
    )
    expected_primary_acceptance_report_path = (
        expected_handoff_root
        / "real-experiment-primary-evidence-acceptance-report.json"
    )
    after_write_intake_plan = next_handoff_plan["after_write_intake_plan"]
    assert after_write_intake_plan == {
        "required": True,
        "artifact_paths": {
            "external_artifact_contracts_path": str(expected_external_contracts_path),
            "external_artifact_launch_report_path": str(
                expected_launch_report_path
            ),
            "offline_control_import_manifest_path": str(expected_offline_manifest_path),
            "offline_control_prediction_receipt_bundle_path": str(
                expected_offline_receipt_bundle_path
            ),
            "offline_control_prediction_request_bundle_path": str(
                expected_offline_request_bundle_path
            ),
            "predicted_dsg_detector_receipt_bundle_path": str(
                expected_predicted_receipt_bundle_path
            ),
            "predicted_dsg_detector_request_bundle_path": str(
                expected_predicted_request_bundle_path
            ),
            "predicted_dsg_detector_run_manifest_path": str(
                expected_predicted_manifest_path
            ),
            "primary_evidence_acceptance_report_path": str(
                expected_primary_acceptance_report_path
            ),
            "primary_evidence_request_package_path": str(
                expected_primary_request_package_path
            ),
            "primary_evidence_return_checklist_path": str(
                expected_primary_return_checklist_path
            ),
            "primary_evidence_return_progress_path": str(
                expected_primary_return_progress_path
            ),
            "primary_evidence_status_path": str(expected_primary_status_path),
            "real_collection_report_path": str(
                expected_real_collection_report_path
            ),
            "real_collection_request_bundle_path": str(
                expected_real_collection_request_bundle_path
            ),
            "real_experiment_run_manifest_path": str(
                expected_handoff_run_manifest_path
            ),
        },
        "commands": {
            "compare_external_artifact_contracts": (
                "python scripts/run_real_experiment.py "
                f"--compare-external-artifact-contracts {expected_external_contracts_path}"
            ),
            "compare_external_artifact_launch_report": (
                "python scripts/run_real_experiment.py "
                f"--compare-external-artifact-launch-report {expected_launch_report_path}"
            ),
            "compare_primary_evidence_acceptance_report": (
                "python scripts/run_real_experiment.py "
                "--compare-primary-evidence-acceptance-report "
                f"{expected_primary_acceptance_report_path}"
            ),
            "compare_primary_evidence_request_package": (
                "python scripts/run_real_experiment.py "
                "--compare-primary-evidence-request-package "
                f"{expected_primary_request_package_path}"
            ),
            "compare_primary_evidence_return_checklist": (
                "python scripts/run_real_experiment.py "
                "--compare-primary-evidence-return-checklist "
                f"{expected_primary_return_checklist_path}"
            ),
            "compare_primary_evidence_return_progress_report": (
                "python scripts/run_real_experiment.py "
                "--compare-primary-evidence-return-progress-report "
                f"{expected_primary_return_progress_path}"
            ),
            "compare_primary_evidence_status": (
                "python scripts/run_real_experiment.py "
                f"--compare-primary-evidence-status {expected_primary_status_path}"
            ),
            "external_artifact_launch_report": (
                "python scripts/run_real_experiment.py "
                f"--external-artifact-launch-report {expected_external_contracts_path} "
                f"--launch-report-output {expected_launch_report_path}"
            ),
            "offline_control_prediction_receipt_bundle": (
                "python scripts/run_offline_controls.py "
                f"--prediction-receipt-bundle {expected_offline_manifest_path} "
                f"--receipt-bundle-output {expected_offline_receipt_bundle_path}"
            ),
            "offline_control_prediction_request_bundle": (
                "python scripts/run_offline_controls.py "
                f"--prediction-request-bundle {expected_offline_manifest_path} "
                f"--request-bundle-output {expected_offline_request_bundle_path}"
            ),
            "predicted_dsg_detector_receipt_bundle": (
                "python scripts/run_predicted_dsg.py "
                f"--detector-receipt-bundle {expected_predicted_manifest_path} "
                f"--receipt-bundle-output {expected_predicted_receipt_bundle_path}"
            ),
            "predicted_dsg_detector_request_bundle": (
                "python scripts/run_predicted_dsg.py "
                f"--detector-request-bundle {expected_predicted_manifest_path} "
                f"--request-bundle-output {expected_predicted_request_bundle_path}"
            ),
            "preflight_run_manifest": (
                "python scripts/run_real_experiment.py "
                f"--preflight-run-manifest {expected_handoff_run_manifest_path}"
            ),
            "primary_evidence_acceptance_report": (
                "python scripts/run_real_experiment.py "
                "--primary-evidence-acceptance-report "
                f"{expected_primary_return_progress_path} "
                "--primary-evidence-acceptance-output "
                f"{expected_primary_acceptance_report_path}"
            ),
            "primary_evidence_request_package": (
                "python scripts/run_real_experiment.py "
                f"--primary-evidence-request-package {expected_launch_report_path} "
                "--primary-evidence-request-package-output "
                f"{expected_primary_request_package_path}"
            ),
            "primary_evidence_return_checklist": (
                "python scripts/run_real_experiment.py "
                "--primary-evidence-return-checklist "
                f"{expected_primary_request_package_path} "
                "--primary-evidence-return-checklist-output "
                f"{expected_primary_return_checklist_path}"
            ),
            "primary_evidence_return_progress_report": (
                "python scripts/run_real_experiment.py "
                "--primary-evidence-return-progress-report "
                f"{expected_primary_return_checklist_path} "
                "--primary-evidence-return-progress-output "
                f"{expected_primary_return_progress_path}"
            ),
            "primary_evidence_status": (
                "python scripts/run_real_experiment.py "
                f"--primary-evidence-status {expected_launch_report_path} "
                f"--primary-evidence-status-output {expected_primary_status_path}"
            ),
            "real_collection_report": (
                "python scripts/check_real_collection.py "
                "--dataset-name ai2thor_real_smoke "
                "--source-kind ai2thor "
                f"--episode {inputs['episode_paths'][0]} "
                f"--episode {planned_episode_paths[0]} "
                f"--episode {planned_episode_paths[1]} "
                f"--report {expected_real_collection_report_path} "
                "--min-episode-count 3 --min-scene-count 1 --min-frame-count 30"
            ),
            "real_collection_request_bundle": (
                "python scripts/check_real_collection.py "
                f"--request-bundle {expected_real_collection_request_bundle_path} "
                "--dataset-name ai2thor_real_smoke "
                "--source-kind ai2thor "
                f"--episode {inputs['episode_paths'][0]} "
                f"--episode {planned_episode_paths[0]} "
                f"--episode {planned_episode_paths[1]} "
                f"--report {expected_real_collection_report_path} "
                "--min-episode-count 3 --min-scene-count 1 --min-frame-count 30"
            ),
            "validate_external_artifact_contracts": (
                "python scripts/run_real_experiment.py "
                f"--validate-external-artifact-contracts {expected_external_contracts_path}"
            ),
            "validate_external_artifact_launch_report": (
                "python scripts/run_real_experiment.py "
                f"--validate-external-artifact-launch-report {expected_launch_report_path}"
            ),
            "validate_primary_evidence_acceptance_report": (
                "python scripts/run_real_experiment.py "
                "--validate-primary-evidence-acceptance-report "
                f"{expected_primary_acceptance_report_path}"
            ),
            "validate_primary_evidence_request_package": (
                "python scripts/run_real_experiment.py "
                "--validate-primary-evidence-request-package "
                f"{expected_primary_request_package_path}"
            ),
            "validate_primary_evidence_return_checklist": (
                "python scripts/run_real_experiment.py "
                "--validate-primary-evidence-return-checklist "
                f"{expected_primary_return_checklist_path}"
            ),
            "validate_primary_evidence_return_progress_report": (
                "python scripts/run_real_experiment.py "
                "--validate-primary-evidence-return-progress-report "
                f"{expected_primary_return_progress_path}"
            ),
            "validate_primary_evidence_status": (
                "python scripts/run_real_experiment.py "
                f"--validate-primary-evidence-status {expected_primary_status_path}"
            ),
            "write_primary_evidence_request_bundles": (
                "python scripts/run_real_experiment.py "
                "--write-primary-evidence-request-bundles "
                f"{expected_primary_request_package_path}"
            ),
        },
        "track_order": [
            "real_data",
            "real_controls",
            "predicted_dsg",
            "primary_evidence",
            "launch_audit",
        ],
    }
    expected_next_execution_packet_path = (
        expected_handoff_root / "real-experiment-execution-packet.json"
    )
    expected_next_smoke_checklist_path = (
        expected_handoff_root / "real-experiment-smoke-run-checklist.json"
    )
    expected_next_smoke_runbook_path = (
        expected_handoff_root / "real-experiment-smoke-run-runbook.json"
    )
    expected_next_execution_receipt_path = (
        expected_handoff_root / "real-experiment-execution-receipt.json"
    )
    expected_next_run_ledger_path = (
        expected_handoff_root / "outputs/real-experiment-run-ledger.json"
    )
    expected_next_research_review_path = (
        expected_handoff_root / "real-experiment-research-review.json"
    )
    expected_next_claim_readiness_path = (
        expected_handoff_root / "real-experiment-claim-readiness.json"
    )
    next_run_review_plan = next_handoff_plan["next_run_review_plan"]
    assert next_run_review_plan == {
        "required": True,
        "artifact_paths": {
            "claim_readiness_path": str(expected_next_claim_readiness_path),
            "execution_packet_path": str(expected_next_execution_packet_path),
            "execution_receipt_path": str(expected_next_execution_receipt_path),
            "external_artifact_launch_report_path": str(
                expected_launch_report_path
            ),
            "primary_evidence_acceptance_report_path": str(
                expected_primary_acceptance_report_path
            ),
            "real_experiment_run_ledger_path": str(expected_next_run_ledger_path),
            "research_review_path": str(expected_next_research_review_path),
            "smoke_run_checklist_path": str(expected_next_smoke_checklist_path),
            "smoke_run_runbook_path": str(expected_next_smoke_runbook_path),
        },
        "claim_thresholds": {
            "min_dynamic_qa_count": 1,
            "min_episode_count": 3,
            "min_qa_count": 30,
            "min_scene_count": 1,
        },
        "commands": {
            "claim_readiness": (
                "python scripts/run_real_experiment.py "
                f"--claim-readiness {expected_next_research_review_path} "
                f"--claim-readiness-output {expected_next_claim_readiness_path} "
                "--claim-min-episode-count 3 --claim-min-scene-count 1 "
                "--claim-min-qa-count 30 --claim-min-dynamic-qa-count 1"
            ),
            "compare_claim_readiness": (
                "python scripts/run_real_experiment.py "
                f"--compare-claim-readiness {expected_next_claim_readiness_path}"
            ),
            "compare_execution_packet": (
                "python scripts/run_real_experiment.py "
                f"--compare-execution-packet {expected_next_execution_packet_path}"
            ),
            "compare_execution_receipt": (
                "python scripts/run_real_experiment.py "
                f"--compare-execution-receipt {expected_next_execution_receipt_path}"
            ),
            "compare_research_review": (
                "python scripts/run_real_experiment.py "
                f"--compare-research-review {expected_next_research_review_path}"
            ),
            "compare_run_ledger": (
                "python scripts/run_real_experiment.py "
                f"--compare-run-ledger {expected_next_run_ledger_path}"
            ),
            "compare_smoke_run_checklist": (
                "python scripts/run_real_experiment.py "
                f"--compare-smoke-run-checklist {expected_next_smoke_checklist_path}"
            ),
            "compare_smoke_run_runbook": (
                "python scripts/run_real_experiment.py "
                f"--compare-smoke-run-runbook {expected_next_smoke_runbook_path}"
            ),
            "execution_packet": (
                "python scripts/run_real_experiment.py "
                f"--execution-packet {expected_launch_report_path} "
                "--execution-packet-primary-evidence-acceptance-report "
                f"{expected_primary_acceptance_report_path} "
                f"--execution-packet-output {expected_next_execution_packet_path}"
            ),
            "execution_receipt": (
                "python scripts/run_real_experiment.py "
                f"--execution-receipt {expected_next_execution_packet_path} "
                f"--execution-receipt-output {expected_next_execution_receipt_path}"
            ),
            "research_review": (
                "python scripts/run_real_experiment.py "
                f"--research-review {expected_next_execution_receipt_path} "
                f"--research-review-output {expected_next_research_review_path}"
            ),
            "smoke_run_checklist": (
                "python scripts/run_real_experiment.py "
                f"--smoke-run-checklist {expected_next_execution_packet_path} "
                f"--smoke-run-checklist-output {expected_next_smoke_checklist_path} "
                "--smoke-run-checklist-receipt-output "
                f"{expected_next_execution_receipt_path}"
            ),
            "smoke_run_runbook": (
                "python scripts/run_real_experiment.py "
                f"--smoke-run-runbook {expected_next_smoke_checklist_path} "
                f"--smoke-run-runbook-output {expected_next_smoke_runbook_path}"
            ),
            "validate_claim_readiness": (
                "python scripts/run_real_experiment.py "
                f"--validate-claim-readiness {expected_next_claim_readiness_path}"
            ),
            "validate_execution_packet": (
                "python scripts/run_real_experiment.py "
                f"--validate-execution-packet {expected_next_execution_packet_path}"
            ),
            "validate_execution_receipt": (
                "python scripts/run_real_experiment.py "
                f"--validate-execution-receipt {expected_next_execution_receipt_path}"
            ),
            "validate_research_review": (
                "python scripts/run_real_experiment.py "
                f"--validate-research-review {expected_next_research_review_path}"
            ),
            "validate_run_ledger": (
                "python scripts/run_real_experiment.py "
                f"--validate-run-ledger {expected_next_run_ledger_path}"
            ),
            "validate_smoke_run_checklist": (
                "python scripts/run_real_experiment.py "
                f"--validate-smoke-run-checklist {expected_next_smoke_checklist_path}"
            ),
            "validate_smoke_run_runbook": (
                "python scripts/run_real_experiment.py "
                f"--validate-smoke-run-runbook {expected_next_smoke_runbook_path}"
            ),
        },
        "phase_order": [
            "execution_packet",
            "smoke_run",
            "post_run_receipt",
            "research_review",
            "claim_recheck",
        ],
    }
    write_handoff_command = next_handoff_plan["commands"][
        "write_handoff_manifests"
    ]
    assert write_handoff_command == (
        "python scripts/run_real_experiment.py --write-handoff-manifests "
        f"--handoff-root {expected_handoff_root} "
        "--dataset-name ai2thor_real_smoke "
        f"--episode {inputs['episode_paths'][0]} "
        f"--episode {planned_episode_paths[0]} "
        f"--episode {planned_episode_paths[1]} "
        f"--candidate-prediction {expected_candidate_prediction_path} "
        f"--detector-jsonl {expected_detector_jsonl_path} "
        "--max-qa-per-episode 20 "
        "--tag benchmark --tag real "
        "--data-source-kind real "
        "--real-collection-source-kind ai2thor "
        "--min-episode-count 3 --min-scene-count 1 "
        "--real-collection-min-frame-count 30 --min-qa-count 30 "
        "--required-control-kind caption_memory "
        "--required-control-kind graph_text "
        "--required-control-kind multi_frame_vlm "
        "--required-control-kind vlm "
        "--required-predicted-input-kind observation_sequence"
    )
    operator_checklist = next_handoff_plan["operator_checklist"]
    expected_operator_keys = [
        "write_handoff_manifests",
        "validate_external_artifact_contracts",
        "compare_external_artifact_contracts",
        "external_artifact_launch_report",
        "validate_external_artifact_launch_report",
        "compare_external_artifact_launch_report",
        "primary_evidence_status",
        "validate_primary_evidence_status",
        "compare_primary_evidence_status",
        "primary_evidence_request_package",
        "validate_primary_evidence_request_package",
        "compare_primary_evidence_request_package",
        "write_primary_evidence_request_bundles",
        "primary_evidence_return_checklist",
        "validate_primary_evidence_return_checklist",
        "compare_primary_evidence_return_checklist",
        "real_collection_report",
        "offline_control_prediction_receipt_bundle",
        "predicted_dsg_detector_receipt_bundle",
        "primary_evidence_return_progress_report",
        "validate_primary_evidence_return_progress_report",
        "compare_primary_evidence_return_progress_report",
        "primary_evidence_acceptance_report",
        "validate_primary_evidence_acceptance_report",
        "compare_primary_evidence_acceptance_report",
        "execution_packet",
        "validate_execution_packet",
        "compare_execution_packet",
        "smoke_run_checklist",
        "validate_smoke_run_checklist",
        "compare_smoke_run_checklist",
        "smoke_run_runbook",
        "validate_smoke_run_runbook",
        "compare_smoke_run_runbook",
        "validate_run_ledger",
        "compare_run_ledger",
        "execution_receipt",
        "validate_execution_receipt",
        "compare_execution_receipt",
        "research_review",
        "validate_research_review",
        "compare_research_review",
        "claim_readiness",
        "validate_claim_readiness",
        "compare_claim_readiness",
    ]
    assert operator_checklist["required"] is True
    assert operator_checklist["phase_order"] == [
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
    assert operator_checklist["step_count"] == len(expected_operator_keys)
    assert [step["key"] for step in operator_checklist["steps"]] == (
        expected_operator_keys
    )
    assert [step["order"] for step in operator_checklist["steps"]] == list(
        range(1, len(expected_operator_keys) + 1)
    )
    assert operator_checklist["steps"][0] == {
        "command": write_handoff_command,
        "key": "write_handoff_manifests",
        "order": 1,
        "phase": "handoff_manifests",
        "track": "handoff",
    }
    assert operator_checklist["steps"][-1] == {
        "command": next_run_review_plan["commands"]["compare_claim_readiness"],
        "key": "compare_claim_readiness",
        "order": len(expected_operator_keys),
        "phase": "claim_recheck",
        "track": "run_review",
    }

    claim_readiness = lab.real_experiment_claim_readiness(
        loaded_research_review,
        research_review_path=research_review_path,
        min_episode_count=1,
        min_scene_count=1,
        min_qa_count=8,
        min_dynamic_qa_count=0,
    )
    claim_readiness_path = tmp_path / "run-manifest" / "claim-readiness.json"
    assert (
        lab.save_real_experiment_claim_readiness(
            claim_readiness,
            claim_readiness_path,
        )
        == claim_readiness_path
    )
    loaded_claim_readiness = lab.load_real_experiment_claim_readiness(
        claim_readiness_path
    )
    assert loaded_claim_readiness == claim_readiness
    claim_validation = lab.validate_real_experiment_claim_readiness(
        loaded_claim_readiness
    )
    assert claim_validation["valid"] is True
    assert claim_validation["claim_readiness_digest"] == (
        claim_readiness["claim_readiness_digest"]
    )
    claim_comparison = lab.compare_real_experiment_claim_readiness(
        loaded_claim_readiness
    )
    assert claim_comparison["matches"] is True
    assert claim_comparison["saved_digest"] == (
        claim_readiness["claim_readiness_digest"]
    )
    assert claim_comparison["current_digest"] == (
        claim_readiness["claim_readiness_digest"]
    )
    assert claim_readiness["claim_ready"] is True
    assert claim_readiness["status"] == "claim_ready"
    assert claim_readiness["blockers"] == []
    assert claim_readiness["research_question_verdicts"] == {
        "dynamic_memory": "unchanged",
        "graph_tool_query": "regressed",
        "interactive_task": "improved",
        "spatial_qa": "regressed",
    }
    assert claim_readiness["claim_conclusion_summary"] == {
        "available_count": 4,
        "claim_conclusion": "regression",
        "conclusive_count": 4,
        "improved_keys": [
            "interactive_task",
        ],
        "inconclusive_keys": [],
        "missing_keys": [],
        "ready_to_conclude": True,
        "regressed_keys": [
            "graph_tool_query",
            "spatial_qa",
        ],
        "required_count": 4,
        "unchanged_keys": ["dynamic_memory"],
        "verdict_counts": {
            "improved": 1,
            "inconclusive": 0,
            "missing": 0,
            "regressed": 2,
            "unchanged": 1,
        },
    }
    assert claim_readiness["claim_conclusion_evidence"] == {
        key: {
            "available": row["available"],
            "measurement_count": row["measurement_count"],
            "primary_metric": row["primary_metric"],
            "source_artifact_type": row["source_artifact_type"],
            "verdict": row["verdict"],
        }
        for key, row in loaded_research_review["research_questions"].items()
    }
    assert claim_readiness["claim_effect_matrix"] == [
        {
            "available": row["available"],
            "measurement_count": row["measurement_count"],
            "metric_name": row["primary_metric"]["name"],
            "metric_value": row["primary_metric"]["value"],
            "research_question": key,
            "source_artifact_type": row["source_artifact_type"],
            "verdict": row["verdict"],
        }
        for key, row in loaded_research_review["research_questions"].items()
    ]
    assert claim_readiness["claim_effect_direction_summary"] == {
        "consistent": True,
        "direction_counts": {
            "missing_metric": 0,
            "negative": 2,
            "positive": 1,
            "verdict_mismatch": 0,
            "zero": 1,
        },
        "missing_metric_keys": [],
        "negative_keys": [
            "graph_tool_query",
            "spatial_qa",
        ],
        "positive_keys": ["interactive_task"],
        "verdict_mismatch_keys": [],
        "zero_keys": ["dynamic_memory"],
    }
    assert claim_readiness["claim_hypothesis_assessment"] == {
        "assessment": "contradicted_by_regression",
        "hypothesis": "dynamic_scene_graph_improves_all_target_capabilities",
        "missing_or_inconclusive_keys": [],
        "negative_evidence_keys": [
            "graph_tool_query",
            "spatial_qa",
        ],
        "neutral_evidence_keys": ["dynamic_memory"],
        "no_change_observed": True,
        "partial_improvement_observed": True,
        "positive_evidence_keys": ["interactive_task"],
        "ready_to_assess": True,
        "regression_observed": True,
        "required_count": 4,
        "supports_full_hypothesis": False,
    }
    assert claim_readiness["claim_scope_assessment"] == {
        "active_scale_ready": True,
        "below_default_threshold_fields": [
            "min_dynamic_qa_count",
            "min_episode_count",
            "min_qa_count",
        ],
        "claim_scope": "smoke_threshold_ready",
        "default_scale_deficits": {
            "episode_count": {
                "actual": 1,
                "deficit": 2,
                "minimum": 3,
                "threshold_field": "min_episode_count",
            },
            "qa_count": {
                "actual": 9,
                "deficit": 21,
                "minimum": 30,
                "threshold_field": "min_qa_count",
            },
        },
        "default_scale_ready": False,
        "default_thresholds": {
            "min_dynamic_qa_count": 1,
            "min_episode_count": 3,
            "min_qa_count": 30,
            "min_scene_count": 1,
        },
        "full_scale_claim_permitted": False,
        "threshold_profile": "below_default",
    }
    assert claim_readiness["claim_scope_next_actions"] == [
        {
            "action": "expand_to_default_benchmark_scale",
            "claim_scope": "smoke_threshold_ready",
            "current_scale": {
                "dynamic_qa_count": 4,
                "episode_count": 1,
                "qa_count": 9,
                "scene_count": 1,
            },
            "default_scale_deficits": claim_readiness["claim_scope_assessment"][
                "default_scale_deficits"
            ],
            "default_thresholds": claim_readiness["claim_scope_assessment"][
                "default_thresholds"
            ],
            "order": 1,
            "reason": "Saved claim thresholds are below the default benchmark policy.",
            "threshold_profile": "below_default",
            "track": "real_data",
        }
    ]
    expected_full_scale_handoff_root = run_manifest_path.with_name(
        "next-full-scale-claim-handoff"
    )
    full_scale_planned_episode_paths = [
        expected_full_scale_handoff_root
        / "inputs/episodes/ai2thor_real_smoke-episode-002.jsonl",
        expected_full_scale_handoff_root
        / "inputs/episodes/ai2thor_real_smoke-episode-003.jsonl",
    ]
    expected_full_scale_candidate_prediction_path = (
        expected_full_scale_handoff_root / "inputs/candidate/predicted-graph-tool.jsonl"
    )
    expected_full_scale_detector_jsonl_path = (
        expected_full_scale_handoff_root / "inputs/predicted-dsg/detector-rgbd.jsonl"
    )
    expected_full_scale_control_prediction_paths = {
        "caption_memory": (
            expected_full_scale_handoff_root
            / "inputs/offline-controls/caption_memory.jsonl"
        ),
        "graph_text": (
            expected_full_scale_handoff_root
            / "inputs/offline-controls/graph_text.jsonl"
        ),
        "multi_frame_vlm": (
            expected_full_scale_handoff_root
            / "inputs/offline-controls/multi_frame_vlm.jsonl"
        ),
        "vlm": expected_full_scale_handoff_root / "inputs/offline-controls/vlm.jsonl",
    }
    claim_scope_handoff_plan = claim_readiness["claim_scope_handoff_plan"]
    assert claim_scope_handoff_plan["required"] is True
    assert claim_scope_handoff_plan["dataset_name"] == "ai2thor_real_smoke"
    assert claim_scope_handoff_plan["handoff_kind"] == "full_scale_claim_scope"
    assert claim_scope_handoff_plan["source_claim_scope"] == "smoke_threshold_ready"
    assert claim_scope_handoff_plan["reason"] == (
        "Smoke-threshold-ready claims require default benchmark scale before "
        "full-scale DSG benefit claims."
    )
    assert claim_scope_handoff_plan["handoff_root"] == str(
        expected_full_scale_handoff_root
    )
    assert claim_scope_handoff_plan["source_run_manifest_path"] == str(
        run_manifest_path
    )
    assert claim_scope_handoff_plan["source_run_manifest_digest"] == (
        lab.real_experiment_run_manifest_digest(manifest)
    )
    assert claim_scope_handoff_plan["scale_deficits"] == (
        claim_readiness["claim_scope_assessment"]["default_scale_deficits"]
    )
    assert claim_scope_handoff_plan["target_thresholds"] == (
        claim_readiness["claim_scope_assessment"]["default_thresholds"]
    )
    assert claim_scope_handoff_plan["required_predicted_input_kinds"] == [
        "observation_sequence"
    ]
    assert claim_scope_handoff_plan["current_handoff_thresholds"] == {
        "min_episode_count": 1,
        "min_frame_count": 30,
        "min_qa_count": 8,
        "min_scene_count": 1,
    }
    assert claim_scope_handoff_plan["tracks_to_expand"] == ["real_data"]
    assert claim_scope_handoff_plan["threshold_updates"] == {
        "min_episode_count": {
            "current": 1,
            "increase": 2,
            "target": 3,
        },
        "min_qa_count": {
            "current": 8,
            "increase": 22,
            "target": 30,
        },
    }
    assert claim_scope_handoff_plan["episode_collection_plan"] == {
        "current_episode_count": 1,
        "episode_deficit": 2,
        "existing_episode_paths": [str(inputs["episode_paths"][0])],
        "planned_episode_paths": [
            str(path) for path in full_scale_planned_episode_paths
        ],
        "target_episode_count": 3,
    }
    assert claim_scope_handoff_plan["external_artifact_slots"] == {
        "candidate_prediction_path": str(expected_full_scale_candidate_prediction_path),
        "detector_jsonl_path": str(expected_full_scale_detector_jsonl_path),
        "offline_control_prediction_paths": {
            key: str(path)
            for key, path in expected_full_scale_control_prediction_paths.items()
        },
        "track_order": ["real_controls", "predicted_dsg"],
    }
    assert claim_scope_handoff_plan["commands"]["write_handoff_manifests"] == (
        "python scripts/run_real_experiment.py --write-handoff-manifests "
        f"--handoff-root {expected_full_scale_handoff_root} "
        "--dataset-name ai2thor_real_smoke "
        f"--episode {inputs['episode_paths'][0]} "
        f"--episode {full_scale_planned_episode_paths[0]} "
        f"--episode {full_scale_planned_episode_paths[1]} "
        f"--candidate-prediction {expected_full_scale_candidate_prediction_path} "
        f"--detector-jsonl {expected_full_scale_detector_jsonl_path} "
        "--max-qa-per-episode 20 "
        "--tag benchmark --tag real "
        "--data-source-kind real "
        "--real-collection-source-kind ai2thor "
        "--min-episode-count 3 --min-scene-count 1 "
        "--real-collection-min-frame-count 30 --min-qa-count 30 "
        "--required-control-kind caption_memory "
        "--required-control-kind graph_text "
        "--required-control-kind multi_frame_vlm "
        "--required-control-kind vlm "
        "--required-predicted-input-kind observation_sequence"
    )
    assert claim_scope_handoff_plan["operator_checklist"]["required"] is True
    assert claim_scope_handoff_plan["operator_checklist"]["step_count"] == len(
        expected_operator_keys
    )
    assert [
        step["key"]
        for step in claim_scope_handoff_plan["operator_checklist"]["steps"]
    ] == expected_operator_keys
    drifted_claim_readiness = json.loads(json.dumps(claim_readiness))
    drifted_claim_readiness["claim_effect_matrix"][0]["metric_value"] = 999.0
    drifted_claim_readiness["claim_readiness_digest"] = (
        lab.real_experiment_claim_readiness_digest(drifted_claim_readiness)
    )
    drifted_claim_validation = lab.validate_real_experiment_claim_readiness(
        drifted_claim_readiness
    )
    assert drifted_claim_validation["valid"] is False
    drifted_effect_check = next(
        check
        for check in drifted_claim_validation["checks"]
        if check["name"] == "claim_effect_matrix"
    )
    assert drifted_effect_check["passed"] is False
    inconsistent_claim_readiness = json.loads(json.dumps(claim_readiness))
    inconsistent_claim_readiness["claim_conclusion_evidence"]["dynamic_memory"][
        "primary_metric"
    ]["value"] = 0.25
    inconsistent_claim_readiness["claim_effect_matrix"][0]["metric_value"] = 0.25
    inconsistent_claim_readiness["claim_readiness_digest"] = (
        lab.real_experiment_claim_readiness_digest(inconsistent_claim_readiness)
    )
    inconsistent_claim_validation = lab.validate_real_experiment_claim_readiness(
        inconsistent_claim_readiness
    )
    assert inconsistent_claim_validation["valid"] is False
    inconsistent_direction_check = next(
        check
        for check in inconsistent_claim_validation["checks"]
        if check["name"] == "claim_effect_direction_summary"
    )
    assert inconsistent_direction_check["passed"] is False
    drifted_assessment_claim_readiness = json.loads(json.dumps(claim_readiness))
    drifted_assessment_claim_readiness["claim_hypothesis_assessment"][
        "assessment"
    ] = "supported_all_capabilities"
    drifted_assessment_claim_readiness["claim_hypothesis_assessment"][
        "supports_full_hypothesis"
    ] = True
    drifted_assessment_claim_readiness["claim_readiness_digest"] = (
        lab.real_experiment_claim_readiness_digest(
            drifted_assessment_claim_readiness
        )
    )
    drifted_assessment_validation = lab.validate_real_experiment_claim_readiness(
        drifted_assessment_claim_readiness
    )
    assert drifted_assessment_validation["valid"] is False
    drifted_assessment_check = next(
        check
        for check in drifted_assessment_validation["checks"]
        if check["name"] == "claim_hypothesis_assessment"
    )
    assert drifted_assessment_check["passed"] is False
    drifted_scope_claim_readiness = json.loads(json.dumps(claim_readiness))
    drifted_scope_claim_readiness["claim_scope_assessment"][
        "full_scale_claim_permitted"
    ] = True
    drifted_scope_claim_readiness["claim_scope_assessment"][
        "claim_scope"
    ] = "full_scale_claim_ready"
    drifted_scope_claim_readiness["claim_readiness_digest"] = (
        lab.real_experiment_claim_readiness_digest(drifted_scope_claim_readiness)
    )
    drifted_scope_validation = lab.validate_real_experiment_claim_readiness(
        drifted_scope_claim_readiness
    )
    assert drifted_scope_validation["valid"] is False
    drifted_scope_check = next(
        check
        for check in drifted_scope_validation["checks"]
        if check["name"] == "claim_scope_assessment"
    )
    assert drifted_scope_check["passed"] is False
    drifted_scope_action_claim_readiness = json.loads(json.dumps(claim_readiness))
    drifted_scope_action_claim_readiness["claim_scope_next_actions"] = []
    drifted_scope_action_claim_readiness["claim_readiness_digest"] = (
        lab.real_experiment_claim_readiness_digest(
            drifted_scope_action_claim_readiness
        )
    )
    drifted_scope_action_validation = lab.validate_real_experiment_claim_readiness(
        drifted_scope_action_claim_readiness
    )
    assert drifted_scope_action_validation["valid"] is False
    drifted_scope_action_check = next(
        check
        for check in drifted_scope_action_validation["checks"]
        if check["name"] == "claim_scope_next_actions"
    )
    assert drifted_scope_action_check["passed"] is False
    drifted_scope_handoff_claim_readiness = json.loads(json.dumps(claim_readiness))
    drifted_scope_handoff_claim_readiness["claim_scope_handoff_plan"][
        "commands"
    ] = {}
    drifted_scope_handoff_claim_readiness["claim_readiness_digest"] = (
        lab.real_experiment_claim_readiness_digest(
            drifted_scope_handoff_claim_readiness
        )
    )
    drifted_scope_handoff_validation = lab.validate_real_experiment_claim_readiness(
        drifted_scope_handoff_claim_readiness
    )
    assert drifted_scope_handoff_validation["valid"] is False
    drifted_scope_handoff_check = next(
        check
        for check in drifted_scope_handoff_validation["checks"]
        if check["name"] == "claim_scope_handoff_plan"
    )
    assert drifted_scope_handoff_check["passed"] is False
    drifted_scope_handoff_command_claim_readiness = json.loads(
        json.dumps(claim_readiness)
    )
    drifted_scope_handoff_command = drifted_scope_handoff_command_claim_readiness[
        "claim_scope_handoff_plan"
    ]["commands"]["write_handoff_manifests"].replace(
        "--min-episode-count 3",
        "--min-episode-count 1",
    )
    drifted_scope_handoff_command_claim_readiness["claim_scope_handoff_plan"][
        "commands"
    ]["write_handoff_manifests"] = drifted_scope_handoff_command
    drifted_scope_handoff_command_claim_readiness["claim_scope_handoff_plan"][
        "operator_checklist"
    ]["steps"][0]["command"] = drifted_scope_handoff_command
    drifted_scope_handoff_command_claim_readiness["claim_readiness_digest"] = (
        lab.real_experiment_claim_readiness_digest(
            drifted_scope_handoff_command_claim_readiness
        )
    )
    drifted_scope_handoff_command_validation = (
        lab.validate_real_experiment_claim_readiness(
            drifted_scope_handoff_command_claim_readiness
        )
    )
    assert drifted_scope_handoff_command_validation["valid"] is False
    drifted_scope_handoff_command_check = next(
        check
        for check in drifted_scope_handoff_command_validation["checks"]
        if check["name"] == "claim_scope_handoff_plan"
    )
    assert drifted_scope_handoff_command_check["passed"] is False
    drifted_scope_episode_plan_claim_readiness = json.loads(
        json.dumps(claim_readiness)
    )
    drifted_scope_episode_plan = drifted_scope_episode_plan_claim_readiness[
        "claim_scope_handoff_plan"
    ]["episode_collection_plan"]
    drifted_scope_episode_plan["episode_deficit"] = 0
    drifted_scope_episode_plan["planned_episode_paths"] = []
    drifted_scope_episode_plan_claim_readiness["claim_readiness_digest"] = (
        lab.real_experiment_claim_readiness_digest(
            drifted_scope_episode_plan_claim_readiness
        )
    )
    drifted_scope_episode_plan_validation = (
        lab.validate_real_experiment_claim_readiness(
            drifted_scope_episode_plan_claim_readiness
        )
    )
    assert drifted_scope_episode_plan_validation["valid"] is False
    drifted_scope_episode_plan_check = next(
        check
        for check in drifted_scope_episode_plan_validation["checks"]
        if check["name"] == "claim_scope_handoff_plan"
    )
    assert drifted_scope_episode_plan_check["passed"] is False
    drifted_scope_external_slots_claim_readiness = json.loads(
        json.dumps(claim_readiness)
    )
    drifted_scope_external_slots = drifted_scope_external_slots_claim_readiness[
        "claim_scope_handoff_plan"
    ]["external_artifact_slots"]
    drifted_scope_external_slots["offline_control_prediction_paths"] = {}
    drifted_scope_external_slots_claim_readiness["claim_readiness_digest"] = (
        lab.real_experiment_claim_readiness_digest(
            drifted_scope_external_slots_claim_readiness
        )
    )
    drifted_scope_external_slots_validation = (
        lab.validate_real_experiment_claim_readiness(
            drifted_scope_external_slots_claim_readiness
        )
    )
    assert drifted_scope_external_slots_validation["valid"] is False
    drifted_scope_external_slots_check = next(
        check
        for check in drifted_scope_external_slots_validation["checks"]
        if check["name"] == "claim_scope_handoff_plan"
    )
    assert drifted_scope_external_slots_check["passed"] is False
    drifted_scope_partial_controls_claim_readiness = json.loads(
        json.dumps(claim_readiness)
    )
    drifted_scope_partial_controls_plan = (
        drifted_scope_partial_controls_claim_readiness["claim_scope_handoff_plan"]
    )
    del drifted_scope_partial_controls_plan["external_artifact_slots"][
        "offline_control_prediction_paths"
    ]["graph_text"]
    drifted_scope_partial_controls_command = drifted_scope_partial_controls_plan[
        "commands"
    ]["write_handoff_manifests"].replace(
        " --required-control-kind graph_text",
        "",
    )
    drifted_scope_partial_controls_plan["commands"][
        "write_handoff_manifests"
    ] = drifted_scope_partial_controls_command
    drifted_scope_partial_controls_plan["operator_checklist"]["steps"][0][
        "command"
    ] = drifted_scope_partial_controls_command
    drifted_scope_partial_controls_claim_readiness["claim_readiness_digest"] = (
        lab.real_experiment_claim_readiness_digest(
            drifted_scope_partial_controls_claim_readiness
        )
    )
    drifted_scope_partial_controls_validation = (
        lab.validate_real_experiment_claim_readiness(
            drifted_scope_partial_controls_claim_readiness
        )
    )
    assert drifted_scope_partial_controls_validation["valid"] is False
    drifted_scope_partial_controls_check = next(
        check
        for check in drifted_scope_partial_controls_validation["checks"]
        if check["name"] == "claim_scope_handoff_plan"
    )
    assert drifted_scope_partial_controls_check["passed"] is False
    drifted_scope_predicted_input_claim_readiness = json.loads(
        json.dumps(claim_readiness)
    )
    drifted_scope_predicted_input_plan = (
        drifted_scope_predicted_input_claim_readiness["claim_scope_handoff_plan"]
    )
    drifted_scope_predicted_input_command = drifted_scope_predicted_input_plan[
        "commands"
    ]["write_handoff_manifests"].replace(
        "--required-predicted-input-kind observation_sequence",
        "",
    )
    drifted_scope_predicted_input_plan["commands"][
        "write_handoff_manifests"
    ] = drifted_scope_predicted_input_command
    drifted_scope_predicted_input_plan["operator_checklist"]["steps"][0][
        "command"
    ] = drifted_scope_predicted_input_command
    drifted_scope_predicted_input_claim_readiness["claim_readiness_digest"] = (
        lab.real_experiment_claim_readiness_digest(
            drifted_scope_predicted_input_claim_readiness
        )
    )
    drifted_scope_predicted_input_validation = (
        lab.validate_real_experiment_claim_readiness(
            drifted_scope_predicted_input_claim_readiness
        )
    )
    assert drifted_scope_predicted_input_validation["valid"] is False
    drifted_scope_predicted_input_check = next(
        check
        for check in drifted_scope_predicted_input_validation["checks"]
        if check["name"] == "claim_scope_handoff_plan"
    )
    assert drifted_scope_predicted_input_check["passed"] is False
    drifted_scope_intake_plan_claim_readiness = json.loads(
        json.dumps(claim_readiness)
    )
    drifted_scope_intake_plan_claim_readiness["claim_scope_handoff_plan"][
        "after_write_intake_plan"
    ]["commands"] = {}
    drifted_scope_intake_plan_claim_readiness["claim_readiness_digest"] = (
        lab.real_experiment_claim_readiness_digest(
            drifted_scope_intake_plan_claim_readiness
        )
    )
    drifted_scope_intake_plan_validation = (
        lab.validate_real_experiment_claim_readiness(
            drifted_scope_intake_plan_claim_readiness
        )
    )
    assert drifted_scope_intake_plan_validation["valid"] is False
    drifted_scope_intake_plan_check = next(
        check
        for check in drifted_scope_intake_plan_validation["checks"]
        if check["name"] == "claim_scope_handoff_plan"
    )
    assert drifted_scope_intake_plan_check["passed"] is False
    drifted_scope_threshold_claim_readiness = json.loads(
        json.dumps(claim_readiness)
    )
    drifted_scope_threshold_claim_readiness["claim_scope_handoff_plan"][
        "threshold_updates"
    ] = {}
    drifted_scope_threshold_claim_readiness["claim_readiness_digest"] = (
        lab.real_experiment_claim_readiness_digest(
            drifted_scope_threshold_claim_readiness
        )
    )
    drifted_scope_threshold_validation = (
        lab.validate_real_experiment_claim_readiness(
            drifted_scope_threshold_claim_readiness
        )
    )
    assert drifted_scope_threshold_validation["valid"] is False
    drifted_scope_threshold_check = next(
        check
        for check in drifted_scope_threshold_validation["checks"]
        if check["name"] == "claim_scope_handoff_plan"
    )
    assert drifted_scope_threshold_check["passed"] is False
    drifted_scope_current_threshold_claim_readiness = json.loads(
        json.dumps(claim_readiness)
    )
    drifted_scope_current_threshold_plan = (
        drifted_scope_current_threshold_claim_readiness["claim_scope_handoff_plan"]
    )
    drifted_scope_current_threshold_plan["current_handoff_thresholds"][
        "min_qa_count"
    ] = 30
    del drifted_scope_current_threshold_plan["threshold_updates"]["min_qa_count"]
    drifted_scope_current_threshold_claim_readiness["claim_readiness_digest"] = (
        lab.real_experiment_claim_readiness_digest(
            drifted_scope_current_threshold_claim_readiness
        )
    )
    drifted_scope_current_threshold_validation = (
        lab.validate_real_experiment_claim_readiness(
            drifted_scope_current_threshold_claim_readiness
        )
    )
    assert drifted_scope_current_threshold_validation["valid"] is False
    drifted_scope_current_threshold_check = next(
        check
        for check in drifted_scope_current_threshold_validation["checks"]
        if check["name"] == "claim_scope_handoff_plan"
    )
    assert drifted_scope_current_threshold_check["passed"] is False
    drifted_scope_provenance_claim_readiness = json.loads(
        json.dumps(claim_readiness)
    )
    drifted_scope_provenance_claim_readiness["claim_scope_handoff_plan"][
        "source_run_manifest_path"
    ] = str(run_manifest_path.with_name("other-real-experiment-run-manifest.json"))
    drifted_scope_provenance_claim_readiness["claim_readiness_digest"] = (
        lab.real_experiment_claim_readiness_digest(
            drifted_scope_provenance_claim_readiness
        )
    )
    drifted_scope_provenance_validation = (
        lab.validate_real_experiment_claim_readiness(
            drifted_scope_provenance_claim_readiness
        )
    )
    assert drifted_scope_provenance_validation["valid"] is False
    drifted_scope_provenance_check = next(
        check
        for check in drifted_scope_provenance_validation["checks"]
        if check["name"] == "claim_scope_handoff_plan"
    )
    assert drifted_scope_provenance_check["passed"] is False
    drifted_scope_dataset_claim_readiness = json.loads(json.dumps(claim_readiness))
    drifted_scope_dataset_plan_json = json.dumps(
        drifted_scope_dataset_claim_readiness["claim_scope_handoff_plan"]
    ).replace("ai2thor_real_smoke", "other_real_dataset")
    drifted_scope_dataset_claim_readiness["claim_scope_handoff_plan"] = json.loads(
        drifted_scope_dataset_plan_json
    )
    drifted_scope_dataset_claim_readiness["claim_readiness_digest"] = (
        lab.real_experiment_claim_readiness_digest(
            drifted_scope_dataset_claim_readiness
        )
    )
    drifted_scope_dataset_validation = lab.validate_real_experiment_claim_readiness(
        drifted_scope_dataset_claim_readiness
    )
    assert drifted_scope_dataset_validation["valid"] is False
    drifted_scope_dataset_check = next(
        check
        for check in drifted_scope_dataset_validation["checks"]
        if check["name"] == "claim_scope_handoff_plan"
    )
    assert drifted_scope_dataset_check["passed"] is False
    assert claim_readiness["scale_summary"] == {
        "dynamic_qa_count": 4,
        "episode_count": 1,
        "qa_count": 9,
        "scene_count": 1,
    }
    assert claim_readiness["thresholds"] == {
        "min_dynamic_qa_count": 0,
        "min_episode_count": 1,
        "min_qa_count": 8,
        "min_scene_count": 1,
    }
    assert claim_readiness["claim_gap_summary"]["scale_deficit_count"] == 0
    assert claim_readiness["next_actions"] == []
    assert claim_readiness["next_handoff_plan"]["required"] is False
    assert claim_readiness["next_handoff_plan"]["commands"] == {}
    assert claim_readiness["next_handoff_plan"]["episode_collection_plan"] == {
        "current_episode_count": 1,
        "episode_deficit": 0,
        "existing_episode_paths": [str(inputs["episode_paths"][0])],
        "planned_episode_paths": [],
        "target_episode_count": 1,
    }
    assert claim_readiness["next_handoff_plan"]["external_artifact_slots"] == {
        "candidate_prediction_path": str(expected_candidate_prediction_path),
        "detector_jsonl_path": str(expected_detector_jsonl_path),
        "offline_control_prediction_paths": {
            key: str(path)
            for key, path in expected_control_prediction_paths.items()
        },
        "track_order": ["real_controls", "predicted_dsg"],
    }
    assert (
        claim_readiness["next_handoff_plan"]["after_write_intake_plan"]["required"]
        is False
    )
    assert (
        claim_readiness["next_handoff_plan"]["after_write_intake_plan"]["commands"]
        == {}
    )
    assert (
        claim_readiness["next_handoff_plan"]["next_run_review_plan"]["required"]
        is False
    )
    assert (
        claim_readiness["next_handoff_plan"]["next_run_review_plan"]["commands"]
        == {}
    )
    assert claim_readiness["next_handoff_plan"]["operator_checklist"] == {
        "required": False,
        "phase_order": [
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
        ],
        "step_count": 0,
        "steps": [],
    }

    assert result["action"] == "run_real_experiment_manifest"
    assert result["run_manifest_schema_version"] == (
        "dsg-spatialqa-lab.real-experiment-run-manifest.v1"
    )
    assert result["run_manifest_path"] == str(run_manifest_path)
    assert result["run_manifest_digest"] == lab.real_experiment_run_manifest_digest(
        manifest
    )
    assert result["ready"] is True
    assert result["offline_control_import"]["ready"] is True
    assert result["predicted_dsg_detector_run"]["ready"] is True
    assert len(result["generated_qa_eval_delta_report_paths"]) == 4
    assert Path(result["generated_offline_control_matrix_report_path"]).exists()
    assert Path(result["generated_offline_control_result_report_path"]).exists()
    assert Path(result["generated_offline_control_import_run_ledger_path"]).exists()
    assert lab.compare_offline_control_import_run_ledger(
        lab.load_offline_control_import_run_ledger(
            result["generated_offline_control_import_run_ledger_path"]
        )
    )["matches"] is True
    assert Path(result["generated_predicted_dsg_evidence_report_path"]).exists()
    assert Path(result["generated_predicted_dsg_detector_run_ledger_path"]).exists()
    assert lab.compare_predicted_dsg_detector_run_ledger(
        lab.load_predicted_dsg_detector_run_ledger(
            result["generated_predicted_dsg_detector_run_ledger_path"]
        )
    )["matches"] is True
    assert record["real_package_status"] == "ready"
    assert lab.compare_experiment_record(record)["matches"] is True
    assert execution_receipt["ready_to_review"] is True
    assert execution_receipt["artifact_summary"] == {
        "artifact_count": 8,
        "invalid_artifact_count": 0,
        "missing_artifact_count": 0,
        "ready_artifact_count": 8,
    }
    assert {
        artifact["role"]: artifact["digest_valid"]
        for artifact in execution_receipt["artifacts"]
        if artifact["kind"] == "json"
    } == {
        "benchmark_manifest": True,
        "experiment_record": True,
        "experiment_summary": True,
            "offline_control_import_run_ledger": True,
            "predicted_dsg_detector_run_ledger": True,
            "real_experiment_run_ledger": True,
            "real_readiness_report": True,
        }


def test_claim_readiness_reports_inconclusive_research_question_gaps(
    tmp_path: Path,
) -> None:
    inputs = _ready_package_inputs(tmp_path)
    run_manifest_path = _real_experiment_run_manifest(tmp_path, inputs)
    execution_packet_path = tmp_path / "run-manifest" / "execution-packet.json"
    execution_receipt_path = tmp_path / "run-manifest" / "execution-receipt.json"
    execution_packet = _execution_packet_for_run_manifest(
        run_manifest_path,
        execution_packet_path,
    )
    lab.run_real_experiment_manifest(
        run_manifest_path,
        approved_execution_packet_path=execution_packet_path,
    )
    execution_receipt = lab.real_experiment_execution_receipt(
        execution_packet,
        execution_packet_path=execution_packet_path,
    )
    lab.save_real_experiment_execution_receipt(
        execution_receipt,
        execution_receipt_path,
    )
    research_review_path = tmp_path / "run-manifest" / "research-review.json"
    research_review = lab.real_experiment_research_review(
        execution_receipt,
        execution_receipt_path=execution_receipt_path,
    )
    inconclusive_review = json.loads(json.dumps(research_review))
    graph_tool_row = inconclusive_review["research_questions"]["graph_tool_query"]
    graph_tool_row["verdict"] = "inconclusive"
    inconclusive_review["research_question_summary"] = {
        "available_count": 4,
        "conclusive_count": 3,
        "inconclusive_count": 1,
        "required_count": 4,
    }
    inconclusive_review["review_digest"] = lab.real_experiment_research_review_digest(
        inconclusive_review
    )
    lab.save_real_experiment_research_review(
        inconclusive_review,
        research_review_path,
    )

    claim_readiness = lab.real_experiment_claim_readiness(
        inconclusive_review,
        research_review_path=research_review_path,
        min_episode_count=1,
        min_scene_count=1,
        min_qa_count=8,
        min_dynamic_qa_count=0,
    )

    assert claim_readiness["claim_ready"] is False
    assert claim_readiness["status"] == "pilot_only"
    assert [blocker["name"] for blocker in claim_readiness["blockers"]] == [
        "research_question_availability"
    ]
    research_question_check = next(
        check
        for check in claim_readiness["checks"]
        if check["name"] == "research_question_availability"
    )
    expected_verdicts = {
        key: row["verdict"]
        for key, row in inconclusive_review["research_questions"].items()
    }
    assert research_question_check["actual"]["inconclusive_keys"] == [
        "graph_tool_query"
    ]
    assert research_question_check["actual"]["missing_keys"] == []
    assert research_question_check["actual"]["verdicts"] == expected_verdicts
    assert claim_readiness["claim_gap_summary"]["research_question_gap_count"] == 1
    assert claim_readiness["claim_gap_summary"]["research_question_gaps"] == {
        "inconclusive_keys": ["graph_tool_query"],
        "missing_keys": [],
        "verdicts": expected_verdicts,
    }
    assert claim_readiness["next_actions"] == [
        {
            "action": "rerun_research_review_artifacts",
            "blocker_names": ["research_question_availability"],
            "evidence_targets": [
                {
                    "gap_type": "inconclusive",
                    "research_question": "graph_tool_query",
                    "source_artifact_type": "qa_eval_delta_report",
                    "tracks_to_expand": [
                        "real_controls",
                        "predicted_dsg",
                        "review_artifacts",
                    ],
                    "verdict": "inconclusive",
                }
            ],
            "order": 1,
            "reason": "One or more research questions is unavailable or inconclusive.",
            "research_question_gaps": claim_readiness["claim_gap_summary"][
                "research_question_gaps"
            ],
            "target_thresholds": claim_readiness["claim_gap_summary"][
                "target_thresholds"
            ],
            "track": "review_artifacts",
            "tracks_to_expand": [
                "real_controls",
                "predicted_dsg",
                "review_artifacts",
            ],
        }
    ]
    assert claim_readiness["next_handoff_plan"]["tracks_to_expand"] == [
        "real_controls",
        "predicted_dsg",
        "review_artifacts",
    ]
    assert (
        lab.validate_real_experiment_claim_readiness(claim_readiness)["valid"]
        is True
    )
    claim_readiness_path = tmp_path / "run-manifest" / "claim-readiness.json"
    lab.save_real_experiment_claim_readiness(
        claim_readiness,
        claim_readiness_path,
    )
    assert (
        lab.compare_real_experiment_claim_readiness(claim_readiness)["matches"]
        is True
    )


def test_real_experiment_run_manifest_preflight_reports_ready_inputs(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "REAL_EXPERIMENT_PREFLIGHT_SCHEMA_VERSION")
    assert hasattr(lab, "real_experiment_run_manifest_preflight")
    inputs = _ready_package_inputs(tmp_path)
    run_manifest_path = _real_experiment_run_manifest(tmp_path, inputs)
    manifest = lab.load_real_experiment_run_manifest(run_manifest_path)

    result = lab.real_experiment_run_manifest_preflight(run_manifest_path)

    assert result["schema_version"] == "dsg-spatialqa-lab.real-experiment-preflight.v1"
    assert result["action"] == "real_experiment_run_manifest_preflight"
    assert result["run_manifest_path"] == str(run_manifest_path)
    assert result["run_manifest_digest"] == lab.real_experiment_run_manifest_digest(
        manifest
    )
    assert result["ready_to_run"] is True
    assert result["summary"]["missing_input_count"] == 0
    assert result["summary"]["missing_requirement_count"] == 0
    assert result["groups"]["offline_controls"]["ready"] is True
    assert result["groups"]["predicted_dsg"]["ready"] is True
    assert result["groups"]["real_collection"]["ready"] is True
    roles = {(item["group"], item["role"]) for item in result["required_inputs"]}
    assert ("offline_controls", "offline_control_source_input") in roles
    assert ("offline_controls", "candidate_prediction") in roles
    assert ("predicted_dsg", "detector_jsonl") in roles
    assert ("real_collection", "episode") in roles
    planned_roles = {
        (item["group"], item["role"]) for item in result["planned_outputs"]
    }
    assert ("real_run", "experiment_record") in planned_roles
    assert ("offline_controls", "offline_control_import_run_ledger") in planned_roles
    assert ("offline_controls", "offline_control_result_report") in planned_roles
    assert ("predicted_dsg", "predicted_dsg_detector_run_ledger") in planned_roles
    assert ("predicted_dsg", "predicted_graph_report") in planned_roles


def test_real_experiment_run_manifest_preflight_reports_missing_real_inputs(
    tmp_path: Path,
) -> None:
    inputs = _ready_package_inputs(tmp_path)
    run_manifest_path = _real_experiment_run_manifest(tmp_path, inputs)
    manifest = lab.load_real_experiment_run_manifest(run_manifest_path)
    offline_manifest = lab.load_offline_control_import_manifest(
        manifest["offline_control_import_manifest_path"]
    )
    predicted_manifest = lab.load_predicted_dsg_detector_run_manifest(
        manifest["predicted_dsg_detector_run_manifest_path"]
    )
    Path(str(offline_manifest["sources"][0]["input_path"])).unlink()
    Path(str(predicted_manifest["detector_jsonl_path"])).unlink()

    result = lab.real_experiment_run_manifest_preflight(run_manifest_path)

    assert result["ready_to_run"] is False
    assert result["summary"]["missing_input_count"] == 5
    assert result["groups"]["offline_controls"]["ready"] is False
    assert result["groups"]["predicted_dsg"]["ready"] is False
    missing = {(item["group"], item["role"]) for item in result["missing_inputs"]}
    assert ("offline_controls", "offline_control_source_input") in missing
    assert ("predicted_dsg", "detector_jsonl") in missing


def test_write_real_experiment_handoff_manifests_creates_preflightable_package(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "REAL_EXPERIMENT_EXTERNAL_ARTIFACT_CONTRACTS_SCHEMA_VERSION")
    assert hasattr(lab, "REAL_EXPERIMENT_OPERATOR_CHECKLIST_SCHEMA_VERSION")
    assert hasattr(lab, "real_experiment_external_artifact_contracts_digest")
    assert hasattr(lab, "real_experiment_operator_checklist_digest")
    assert hasattr(lab, "load_real_experiment_operator_checklist")
    assert hasattr(lab, "validate_real_experiment_operator_checklist")
    assert hasattr(lab, "compare_real_experiment_operator_checklist")
    assert hasattr(lab, "real_experiment_operator_progress_report")
    assert hasattr(lab, "real_experiment_operator_progress_report_digest")
    assert hasattr(lab, "real_experiment_primary_evidence_status")
    assert hasattr(lab, "real_experiment_primary_evidence_status_digest")
    assert hasattr(lab, "save_real_experiment_primary_evidence_status")
    assert hasattr(lab, "load_real_experiment_primary_evidence_status")
    assert hasattr(lab, "validate_real_experiment_primary_evidence_status")
    assert hasattr(lab, "compare_real_experiment_primary_evidence_status")
    assert hasattr(lab, "real_experiment_primary_evidence_request_package")
    assert hasattr(lab, "real_experiment_primary_evidence_request_package_digest")
    assert hasattr(lab, "save_real_experiment_primary_evidence_request_package")
    assert hasattr(lab, "load_real_experiment_primary_evidence_request_package")
    assert hasattr(lab, "validate_real_experiment_primary_evidence_request_package")
    assert hasattr(lab, "compare_real_experiment_primary_evidence_request_package")
    assert hasattr(lab, "write_real_experiment_primary_evidence_request_bundles")
    assert hasattr(lab, "real_experiment_primary_evidence_return_checklist")
    assert hasattr(lab, "real_experiment_primary_evidence_return_checklist_digest")
    assert hasattr(lab, "save_real_experiment_primary_evidence_return_checklist")
    assert hasattr(lab, "load_real_experiment_primary_evidence_return_checklist")
    assert hasattr(lab, "validate_real_experiment_primary_evidence_return_checklist")
    assert hasattr(lab, "compare_real_experiment_primary_evidence_return_checklist")
    assert hasattr(lab, "real_experiment_primary_evidence_return_progress_report")
    assert hasattr(
        lab,
        "real_experiment_primary_evidence_return_progress_report_digest",
    )
    assert hasattr(
        lab,
        "save_real_experiment_primary_evidence_return_progress_report",
    )
    assert hasattr(
        lab,
        "load_real_experiment_primary_evidence_return_progress_report",
    )
    assert hasattr(
        lab,
        "validate_real_experiment_primary_evidence_return_progress_report",
    )
    assert hasattr(
        lab,
        "compare_real_experiment_primary_evidence_return_progress_report",
    )
    assert hasattr(lab, "load_real_experiment_external_artifact_contracts")
    assert hasattr(lab, "validate_real_experiment_external_artifact_contracts")
    assert hasattr(lab, "compare_real_experiment_external_artifact_contracts")
    assert hasattr(lab, "real_experiment_external_artifact_launch_report")
    assert hasattr(lab, "real_experiment_external_artifact_launch_report_digest")
    assert hasattr(lab, "save_real_experiment_external_artifact_launch_report")
    assert hasattr(lab, "load_real_experiment_external_artifact_launch_report")
    assert hasattr(lab, "validate_real_experiment_external_artifact_launch_report")
    assert hasattr(lab, "compare_real_experiment_external_artifact_launch_report")
    assert hasattr(lab, "real_experiment_execution_packet")
    assert hasattr(lab, "real_experiment_execution_packet_digest")
    assert hasattr(lab, "save_real_experiment_execution_packet")
    assert hasattr(lab, "load_real_experiment_execution_packet")
    assert hasattr(lab, "validate_real_experiment_execution_packet")
    assert hasattr(lab, "compare_real_experiment_execution_packet")
    assert hasattr(lab, "real_experiment_execution_receipt")
    assert hasattr(lab, "real_experiment_execution_receipt_digest")
    assert hasattr(lab, "save_real_experiment_execution_receipt")
    assert hasattr(lab, "load_real_experiment_execution_receipt")
    assert hasattr(lab, "validate_real_experiment_execution_receipt")
    assert hasattr(lab, "compare_real_experiment_execution_receipt")
    assert hasattr(lab, "real_experiment_smoke_run_checklist")
    assert hasattr(lab, "real_experiment_smoke_run_checklist_digest")
    assert hasattr(lab, "save_real_experiment_smoke_run_checklist")
    assert hasattr(lab, "load_real_experiment_smoke_run_checklist")
    assert hasattr(lab, "validate_real_experiment_smoke_run_checklist")
    assert hasattr(lab, "compare_real_experiment_smoke_run_checklist")
    assert hasattr(lab, "real_experiment_smoke_run_runbook")
    assert hasattr(lab, "real_experiment_smoke_run_runbook_digest")
    assert hasattr(lab, "save_real_experiment_smoke_run_runbook")
    assert hasattr(lab, "load_real_experiment_smoke_run_runbook")
    assert hasattr(lab, "validate_real_experiment_smoke_run_runbook")
    assert hasattr(lab, "compare_real_experiment_smoke_run_runbook")
    assert hasattr(lab, "real_experiment_research_review")
    assert hasattr(lab, "real_experiment_research_review_digest")
    assert hasattr(lab, "save_real_experiment_research_review")
    assert hasattr(lab, "load_real_experiment_research_review")
    assert hasattr(lab, "validate_real_experiment_research_review")
    assert hasattr(lab, "compare_real_experiment_research_review")
    assert hasattr(lab, "real_experiment_claim_readiness")
    assert hasattr(lab, "real_experiment_claim_readiness_digest")
    assert hasattr(lab, "save_real_experiment_claim_readiness")
    assert hasattr(lab, "load_real_experiment_claim_readiness")
    assert hasattr(lab, "validate_real_experiment_claim_readiness")
    assert hasattr(lab, "compare_real_experiment_claim_readiness")
    assert hasattr(lab, "write_real_experiment_handoff_manifests")
    root = tmp_path / "real-handoff"

    result = lab.write_real_experiment_handoff_manifests(
        root=root,
        dataset_name="ai2thor_real_smoke",
        episode_paths=(Path("inputs/episodes/FloorPlan1.jsonl"),),
        min_episode_count=1,
        min_qa_count=8,
    )

    assert result["schema_version"] == "dsg-spatialqa-lab.real-experiment-handoff.v1"
    assert result["action"] == "write_real_experiment_handoff_manifests"
    manifest_paths = result["manifest_paths"]
    run_manifest_path = Path(manifest_paths["real_experiment_run_manifest"])
    offline_manifest_path = Path(manifest_paths["offline_control_import_manifest"])
    predicted_manifest_path = Path(manifest_paths["predicted_dsg_detector_run_manifest"])
    contracts_path = Path(
        manifest_paths["real_experiment_external_artifact_contracts"]
    )
    primary_status_path = Path(
        manifest_paths["real_experiment_primary_evidence_status"]
    )
    request_package_path = Path(
        manifest_paths["real_experiment_primary_evidence_request_package"]
    )
    return_checklist_path = Path(
        manifest_paths["real_experiment_primary_evidence_return_checklist"]
    )
    return_progress_path = Path(
        manifest_paths["real_experiment_primary_evidence_return_progress"]
    )
    acceptance_report_path = Path(
        manifest_paths["real_experiment_primary_evidence_acceptance_report"]
    )
    preflight_report_path = Path(manifest_paths["real_experiment_preflight_report"])
    checklist_path = Path(manifest_paths["real_experiment_artifact_checklist"])
    operator_checklist_path = Path(
        manifest_paths["real_experiment_operator_checklist"]
    )
    smoke_runbook_path = Path(manifest_paths["real_experiment_smoke_run_runbook"])
    assert run_manifest_path.exists()
    assert offline_manifest_path.exists()
    assert predicted_manifest_path.exists()
    assert contracts_path.exists()
    assert preflight_report_path.exists()
    assert checklist_path.exists()
    assert operator_checklist_path.exists()
    assert result["preflight_report_path"] == str(preflight_report_path)
    assert result["preflight_ready_to_run"] is False
    assert result["preflight_summary"] == {
        "existing_planned_output_count": 0,
        "invalid_input_count": 0,
        "missing_input_count": 13,
        "missing_requirement_count": 0,
        "planned_output_count": 25,
        "present_input_count": 2,
        "required_input_count": 15,
    }
    assert result["artifact_checklist_path"] == str(checklist_path)
    assert result["artifact_checklist_summary"] == {
        "blocked_group_count": 4,
        "existing_planned_output_count": 0,
        "input_artifact_count": 15,
        "missing_input_artifact_count": 13,
        "planned_output_artifact_count": 25,
        "present_input_artifact_count": 2,
        "ready_to_run": False,
    }
    assert result["operator_checklist_path"] == str(operator_checklist_path)
    assert result["operator_checklist_summary"] == {
        "phase_count": 11,
        "ready_to_run": False,
        "step_count": 44,
        "track_count": 6,
    }
    assert primary_status_path == (
        root / "real-experiment-primary-evidence-status.json"
    )
    assert request_package_path == (
        root / "real-experiment-primary-evidence-request-package.json"
    )
    assert return_checklist_path == (
        root / "real-experiment-primary-evidence-return-checklist.json"
    )
    assert return_progress_path == (
        root / "real-experiment-primary-evidence-return-progress.json"
    )
    assert acceptance_report_path == (
        root / "real-experiment-primary-evidence-acceptance-report.json"
    )
    assert result["external_artifact_contracts_path"] == str(contracts_path)
    assert result["external_artifact_contracts_summary"] == {
        "missing_input_artifact_count": 13,
        "planned_output_artifact_count": 25,
        "real_control_source_count": 4,
        "required_input_artifact_count": 15,
        "track_count": 5,
    }
    assert result["artifact_track_summary"] == {
        "predicted_dsg": {
            "existing_planned_output_artifact_count": 0,
            "input_artifact_count": 2,
            "input_ready": False,
            "missing_input_artifact_count": 1,
            "planned_output_artifact_count": 6,
            "present_input_artifact_count": 1,
        },
        "real_controls": {
            "existing_planned_output_artifact_count": 0,
            "input_artifact_count": 7,
            "input_ready": False,
            "missing_input_artifact_count": 6,
            "planned_output_artifact_count": 13,
            "present_input_artifact_count": 1,
        },
        "real_data": {
            "existing_planned_output_artifact_count": 0,
            "input_artifact_count": 2,
            "input_ready": False,
            "missing_input_artifact_count": 2,
            "planned_output_artifact_count": 0,
            "present_input_artifact_count": 0,
        },
        "review_artifacts": {
            "existing_planned_output_artifact_count": 0,
            "input_artifact_count": 4,
            "input_ready": False,
            "missing_input_artifact_count": 4,
            "planned_output_artifact_count": 0,
            "present_input_artifact_count": 0,
        },
        "run_outputs": {
            "existing_planned_output_artifact_count": 0,
            "input_artifact_count": 0,
            "input_ready": True,
            "missing_input_artifact_count": 0,
            "planned_output_artifact_count": 6,
            "present_input_artifact_count": 0,
        },
    }
    assert result["next_commands"] == {
        "preflight": (
            "python scripts/run_real_experiment.py --preflight-run-manifest "
            f"{run_manifest_path}"
        ),
        "run": (
            "python scripts/run_real_experiment.py --run-manifest "
            f"{run_manifest_path} --run-ledger-output "
            f"{root / 'outputs/real-experiment-run-ledger.json'}"
        ),
    }

    run_payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    offline_payload = json.loads(offline_manifest_path.read_text(encoding="utf-8"))
    predicted_payload = json.loads(predicted_manifest_path.read_text(encoding="utf-8"))
    assert run_payload["episode_paths"] == ["inputs/episodes/FloorPlan1.jsonl"]
    assert run_payload["offline_control_import_manifest_path"] == (
        "offline-control-import-manifest.json"
    )
    assert run_payload["real_experiment_run_ledger_path"] == (
        "outputs/real-experiment-run-ledger.json"
    )
    assert run_payload["predicted_dsg_detector_run_manifest_path"] == (
        "predicted-dsg-detector-run-manifest.json"
    )
    assert run_payload["offline_control_import_run_ledger_path"] == (
        "outputs/offline-controls/offline-control-import-run-ledger.json"
    )
    assert run_payload["predicted_dsg_detector_run_ledger_path"] == (
        "outputs/predicted-dsg/predicted-dsg-detector-run-ledger.json"
    )
    assert offline_payload["qa_path"] == "inputs/qa.jsonl"
    assert offline_payload["candidate_prediction_path"] == (
        "inputs/candidate/predicted-graph-tool.jsonl"
    )
    assert [
        source["source_kind"] for source in offline_payload["sources"]
    ] == ["vlm", "multi_frame_vlm", "caption_memory", "graph_text"]
    assert {
        source["input_format"] for source in offline_payload["sources"]
    } == {"qa_prediction"}
    assert all("prediction_path" in source for source in offline_payload["sources"])
    assert all("import_report_path" in source for source in offline_payload["sources"])
    assert predicted_payload["detector_jsonl_path"] == (
        "inputs/predicted-dsg/detector-rgbd.jsonl"
    )
    assert run_payload["real_collection_source_kind"] == "ai2thor"
    assert run_payload["min_frame_count"] == 30

    run_manifest = lab.load_real_experiment_run_manifest(run_manifest_path)
    assert result["run_manifest_digest"] == lab.real_experiment_run_manifest_digest(
        run_manifest
    )
    assert result["offline_control_import_manifest_digest"] == (
        lab.offline_control_import_manifest_digest(
            lab.load_offline_control_import_manifest(offline_manifest_path)
        )
    )
    assert result["predicted_dsg_detector_run_manifest_digest"] == (
        lab.predicted_dsg_detector_run_manifest_digest(
            lab.load_predicted_dsg_detector_run_manifest(predicted_manifest_path)
        )
    )

    preflight = lab.real_experiment_run_manifest_preflight(run_manifest_path)
    saved_preflight = json.loads(preflight_report_path.read_text(encoding="utf-8"))
    assert saved_preflight == preflight
    checklist = json.loads(checklist_path.read_text(encoding="utf-8"))
    operator_checklist = json.loads(
        operator_checklist_path.read_text(encoding="utf-8")
    )
    contracts = json.loads(contracts_path.read_text(encoding="utf-8"))
    assert checklist["schema_version"] == (
        "dsg-spatialqa-lab.real-experiment-artifact-checklist.v1"
    )
    assert checklist["action"] == "real_experiment_artifact_checklist"
    assert checklist["run_manifest_path"] == str(run_manifest_path)
    assert checklist["preflight_report_path"] == str(preflight_report_path)
    assert checklist["summary"] == result["artifact_checklist_summary"]
    assert checklist["track_summary"] == result["artifact_track_summary"]
    assert operator_checklist["schema_version"] == (
        "dsg-spatialqa-lab.real-experiment-operator-checklist.v1"
    )
    assert operator_checklist["action"] == "real_experiment_operator_checklist"
    assert operator_checklist["root"] == str(root)
    assert operator_checklist["run_manifest_path"] == str(run_manifest_path)
    assert operator_checklist["external_artifact_contracts_path"] == (
        str(contracts_path)
    )
    assert operator_checklist["summary"] == result["operator_checklist_summary"]
    assert operator_checklist["operator_checklist_digest"] == (
        lab.real_experiment_operator_checklist_digest(operator_checklist)
    )
    assert operator_checklist["phase_order"] == [
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
    assert [step["order"] for step in operator_checklist["steps"]] == list(
        range(1, operator_checklist["step_count"] + 1)
    )
    assert operator_checklist["steps"][0] == {
        "command": (
            "python scripts/run_real_experiment.py "
            f"--validate-external-artifact-contracts {contracts_path}"
        ),
        "key": "validate_external_artifact_contracts",
        "order": 1,
        "phase": "handoff_manifests",
        "track": "launch_audit",
    }
    assert operator_checklist["steps"][1] == {
        "command": (
            "python scripts/run_real_experiment.py "
            f"--compare-external-artifact-contracts {contracts_path}"
        ),
        "key": "compare_external_artifact_contracts",
        "order": 2,
        "phase": "handoff_manifests",
        "track": "launch_audit",
    }
    assert [step["key"] for step in operator_checklist["steps"][2:15]] == [
        "external_artifact_launch_report",
        "validate_external_artifact_launch_report",
        "compare_external_artifact_launch_report",
        "primary_evidence_status",
        "validate_primary_evidence_status",
        "compare_primary_evidence_status",
        "primary_evidence_request_package",
        "validate_primary_evidence_request_package",
        "compare_primary_evidence_request_package",
        "write_primary_evidence_request_bundles",
        "primary_evidence_return_checklist",
        "validate_primary_evidence_return_checklist",
        "compare_primary_evidence_return_checklist",
    ]
    assert operator_checklist["steps"][5]["command"] == (
        "python scripts/run_real_experiment.py "
        f"--primary-evidence-status {root / 'real-experiment-external-artifact-launch-report.json'} "
        f"--primary-evidence-status-output {primary_status_path}"
    )
    assert operator_checklist["steps"][11]["command"] == (
        "python scripts/run_real_experiment.py "
        f"--write-primary-evidence-request-bundles {request_package_path}"
    )
    assert [step["key"] for step in operator_checklist["steps"][18:24]] == [
        "primary_evidence_return_progress_report",
        "validate_primary_evidence_return_progress_report",
        "compare_primary_evidence_return_progress_report",
        "primary_evidence_acceptance_report",
        "validate_primary_evidence_acceptance_report",
        "compare_primary_evidence_acceptance_report",
    ]
    assert [step["key"] for step in operator_checklist["steps"][29:38]] == [
        "compare_smoke_run_checklist",
        "smoke_run_runbook",
        "validate_smoke_run_runbook",
        "compare_smoke_run_runbook",
        "validate_run_ledger",
        "compare_run_ledger",
        "execution_receipt",
        "validate_execution_receipt",
        "compare_execution_receipt",
    ]
    assert operator_checklist["steps"][30]["command"] == (
        "python scripts/run_real_experiment.py "
        f"--smoke-run-runbook {root / 'real-experiment-smoke-run-checklist.json'} "
        f"--smoke-run-runbook-output {smoke_runbook_path}"
    )
    assert operator_checklist["steps"][-1] == {
        "command": (
            "python scripts/run_real_experiment.py "
            f"--compare-claim-readiness {root / 'real-experiment-claim-readiness.json'}"
        ),
        "key": "compare_claim_readiness",
        "order": 44,
        "phase": "claim_recheck",
        "track": "run_review",
    }
    loaded_operator_checklist = lab.load_real_experiment_operator_checklist(
        operator_checklist_path
    )
    assert loaded_operator_checklist == operator_checklist
    operator_validation = lab.validate_real_experiment_operator_checklist(
        loaded_operator_checklist
    )
    assert operator_validation["action"] == (
        "validate_real_experiment_operator_checklist"
    )
    assert operator_validation["valid"] is True
    assert operator_validation["operator_checklist_digest"] == (
        operator_checklist["operator_checklist_digest"]
    )
    operator_comparison = lab.compare_real_experiment_operator_checklist(
        loaded_operator_checklist
    )
    assert operator_comparison["action"] == (
        "compare_real_experiment_operator_checklist"
    )
    assert operator_comparison["matches"] is True
    assert operator_comparison["saved_digest"] == (
        operator_checklist["operator_checklist_digest"]
    )
    assert operator_comparison["current_digest"] == (
        operator_checklist["operator_checklist_digest"]
    )
    operator_progress = lab.real_experiment_operator_progress_report(
        loaded_operator_checklist,
        checklist_path=operator_checklist_path,
    )
    assert operator_progress["schema_version"] == (
        "dsg-spatialqa-lab.real-experiment-operator-progress-report.v1"
    )
    assert operator_progress["action"] == "real_experiment_operator_progress_report"
    assert operator_progress["operator_checklist_path"] == str(operator_checklist_path)
    assert operator_progress["operator_checklist_digest"] == (
        operator_checklist["operator_checklist_digest"]
    )
    assert operator_progress["summary"] == {
        "all_targets_present": False,
        "all_targets_ready": False,
        "missing_target_step_count": 42,
        "not_ready_target_step_count": 42,
        "phase_count": 11,
        "present_target_step_count": 2,
        "ready_target_step_count": 2,
        "step_count": 44,
        "track_count": 6,
    }
    assert operator_progress["next_missing_step"] == {
        "key": "external_artifact_launch_report",
        "order": 3,
        "phase": "launch_audit",
        "target_path": str(root / "real-experiment-external-artifact-launch-report.json"),
        "track": "launch_audit",
    }
    assert operator_progress["steps"][0]["key"] == (
        "validate_external_artifact_contracts"
    )
    assert operator_progress["steps"][0]["target_exists"] is True
    assert operator_progress["steps"][0]["target_status"] == "ready"
    assert operator_progress["steps"][0]["target_ready"] is True
    assert operator_progress["steps"][0]["target_audit"] == {
        "action": "validate_real_experiment_external_artifact_contracts",
        "checked": True,
        "valid": True,
    }
    assert operator_progress["steps"][1]["key"] == (
        "compare_external_artifact_contracts"
    )
    assert operator_progress["steps"][1]["target_exists"] is True
    assert operator_progress["steps"][1]["target_status"] == "ready"
    assert operator_progress["steps"][1]["target_ready"] is True
    assert operator_progress["steps"][1]["target_audit"] == {
        "action": "compare_real_experiment_external_artifact_contracts",
        "checked": True,
        "matches": True,
    }
    assert operator_progress["steps"][2]["key"] == "external_artifact_launch_report"
    assert operator_progress["steps"][2]["target_exists"] is False
    assert operator_progress["steps"][2]["target_status"] == "missing"
    assert operator_progress["track_summary"]["launch_audit"] == {
        "missing_target_step_count": 3,
        "not_ready_target_step_count": 3,
        "present_target_step_count": 2,
        "ready_target_step_count": 2,
        "step_count": 5,
    }
    assert operator_progress["report_digest"] == (
        lab.real_experiment_operator_progress_report_digest(operator_progress)
    )
    operator_progress_path = root / "real-experiment-operator-progress-report.json"
    lab.save_real_experiment_operator_progress_report(
        operator_progress,
        operator_progress_path,
    )
    loaded_operator_progress = lab.load_real_experiment_operator_progress_report(
        operator_progress_path
    )
    assert loaded_operator_progress == operator_progress
    operator_progress_validation = (
        lab.validate_real_experiment_operator_progress_report(
            loaded_operator_progress
        )
    )
    assert operator_progress_validation["action"] == (
        "validate_real_experiment_operator_progress_report"
    )
    assert operator_progress_validation["valid"] is True
    assert operator_progress_validation["report_digest"] == (
        operator_progress["report_digest"]
    )
    operator_progress_comparison = lab.compare_real_experiment_operator_progress_report(
        loaded_operator_progress
    )
    assert operator_progress_comparison["action"] == (
        "compare_real_experiment_operator_progress_report"
    )
    assert operator_progress_comparison["matches"] is True
    assert operator_progress_comparison["saved_digest"] == (
        operator_progress["report_digest"]
    )
    assert operator_progress_comparison["current_digest"] == (
        operator_progress["report_digest"]
    )
    assert contracts["schema_version"] == (
        "dsg-spatialqa-lab.real-experiment-external-artifact-contracts.v1"
    )
    assert contracts["action"] == "real_experiment_external_artifact_contracts"
    assert contracts["root"] == str(root)
    assert contracts["run_manifest_path"] == str(run_manifest_path)
    assert contracts["preflight_report_path"] == str(preflight_report_path)
    assert contracts["artifact_checklist_path"] == str(checklist_path)
    assert contracts["summary"] == result["external_artifact_contracts_summary"]
    assert contracts["track_summary"] == result["artifact_track_summary"]
    assert contracts["contracts_digest"] == (
        lab.real_experiment_external_artifact_contracts_digest(contracts)
    )
    loaded_contracts = lab.load_real_experiment_external_artifact_contracts(
        contracts_path
    )
    assert loaded_contracts == contracts
    validation = lab.validate_real_experiment_external_artifact_contracts(
        loaded_contracts
    )
    assert validation["action"] == "validate_real_experiment_external_artifact_contracts"
    assert validation["valid"] is True
    assert validation["contracts_digest"] == contracts["contracts_digest"]
    comparison = lab.compare_real_experiment_external_artifact_contracts(
        loaded_contracts
    )
    assert comparison["action"] == "compare_real_experiment_external_artifact_contracts"
    assert comparison["matches"] is True
    assert comparison["saved_digest"] == contracts["contracts_digest"]
    assert comparison["current_digest"] == contracts["contracts_digest"]
    launch_report = lab.real_experiment_external_artifact_launch_report(
        loaded_contracts,
        contracts_path=contracts_path,
    )
    assert launch_report["schema_version"] == (
        "dsg-spatialqa-lab.real-experiment-launch-report.v1"
    )
    assert launch_report["action"] == (
        "real_experiment_external_artifact_launch_report"
    )
    assert launch_report["contracts_path"] == str(contracts_path)

    assert launch_report["contracts_digest"] == contracts["contracts_digest"]
    assert launch_report["ready_to_run"] is False
    assert launch_report["summary"] == {
        "blocked_track_count": 4,
            "invalid_input_count": 0,
            "missing_input_count": 13,
            "missing_requirement_count": 0,
            "planned_output_count": 25,
            "ready_track_count": 1,
            "required_input_count": 15,
            "track_count": 5,
    }
    assert launch_report["tracks"]["real_data"]["ready"] is False
    assert launch_report["tracks"]["real_data"]["missing_input_count"] == 2
    assert launch_report["tracks"]["real_data"]["blocking_roles"] == [
        "episode",
        "real_collection_report",
    ]
    assert launch_report["tracks"]["real_controls"]["ready"] is False
    assert launch_report["tracks"]["real_controls"]["missing_input_count"] == 6
    assert launch_report["tracks"]["real_controls"]["blocking_roles"] == [
        "candidate_prediction",
        "gold_qa",
        "offline_control_source_input",
    ]
    assert launch_report["tracks"]["predicted_dsg"]["blocking_roles"] == [
        "detector_jsonl"
    ]
    assert launch_report["tracks"]["review_artifacts"]["blocking_roles"] == [
        "active_task_delta_report",
        "dashboard_bundle",
        "error_attribution_report",
        "graph_eval_report",
    ]
    assert launch_report["tracks"]["run_outputs"]["ready"] is True
    episode_path = root / "inputs/episodes/FloorPlan1.jsonl"
    real_collection_report_path = root / "inputs/real-collection-report.json"
    real_collection_request_bundle_path = root / "real-collection-request-bundle.json"
    offline_contracts_path = root / "offline-control-artifact-contracts.json"
    offline_request_bundle_path = root / "offline-control-prediction-request-bundle.json"
    offline_receipt_bundle_path = root / "offline-control-prediction-receipt-bundle.json"
    predicted_contract_path = root / "predicted-dsg-detector-artifact-contract.json"
    predicted_request_bundle_path = root / "predicted-dsg-detector-request-bundle.json"
    predicted_receipt_bundle_path = root / "predicted-dsg-detector-receipt-bundle.json"
    active_task_delta_report_path = root / "inputs/review/active-task-delta.json"
    dashboard_bundle_path = root / "inputs/review/dashboard.json"
    error_attribution_report_path = root / "inputs/review/error-attribution.json"
    graph_eval_report_path = root / "inputs/review/graph-eval.json"
    assert launch_report["child_launch_gates"] == {
        "real_data": {
            "collection_report_command": (
                "python scripts/check_real_collection.py "
                "--dataset-name ai2thor_real_smoke "
                "--source-kind ai2thor "
                f"--episode {episode_path} "
                f"--report {real_collection_report_path} "
                "--min-episode-count 1 "
                "--min-scene-count 1 "
                "--min-frame-count 30"
            ),
            "compare_report_command": (
                "python scripts/check_real_collection.py "
                f"--compare-report {real_collection_report_path}"
            ),
            "episode_paths": [str(episode_path)],
            "real_collection_report_path": str(real_collection_report_path),
            "request_bundle_command": (
                "python scripts/check_real_collection.py "
                f"--request-bundle {real_collection_request_bundle_path} "
                "--dataset-name ai2thor_real_smoke "
                "--source-kind ai2thor "
                f"--episode {episode_path} "
                f"--report {real_collection_report_path} "
                "--min-episode-count 1 "
                "--min-scene-count 1 "
                "--min-frame-count 30"
            ),
            "request_bundle_path": str(real_collection_request_bundle_path),
            "source_kind": "ai2thor",
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
            "active_task_delta_report_commands": [
                {
                    "compare_command": (
                        "python scripts/run_active_tasks.py "
                        f"--compare-delta-report {active_task_delta_report_path}"
                    ),
                    "path": str(active_task_delta_report_path),
                    "validate_command": (
                        "python scripts/run_active_tasks.py "
                        f"--validate-delta-report {active_task_delta_report_path}"
                    ),
                },
            ],
            "active_task_delta_report_paths": [str(active_task_delta_report_path)],
            "dashboard_bundle_commands": [
                {
                    "path": str(dashboard_bundle_path),
                    "validate_command": (
                        "python scripts/export_dashboard.py "
                        f"--validate-bundle {dashboard_bundle_path}"
                    ),
                },
            ],
            "dashboard_bundle_paths": [str(dashboard_bundle_path)],
            "error_attribution_report_commands": [
                {
                    "compare_command": (
                        "python scripts/attribute_errors.py "
                        f"--compare-report {error_attribution_report_path}"
                    ),
                    "path": str(error_attribution_report_path),
                    "validate_command": (
                        "python scripts/attribute_errors.py "
                        f"--validate-report {error_attribution_report_path}"
                    ),
                },
            ],
            "error_attribution_report_paths": [
                str(error_attribution_report_path),
            ],
            "graph_eval_report_commands": [
                {
                    "compare_command": (
                        "python scripts/evaluate_graphs.py "
                        f"--compare-report {graph_eval_report_path}"
                    ),
                    "path": str(graph_eval_report_path),
                    "validate_command": (
                        "python scripts/evaluate_graphs.py "
                        f"--validate-report {graph_eval_report_path}"
                    ),
                },
            ],
            "graph_eval_report_paths": [str(graph_eval_report_path)],
            "track": "review_artifacts",
        },
    }
    assert set(launch_report["actionable_blockers"]) == {
        "predicted_dsg",
        "real_controls",
        "real_data",
        "review_artifacts",
    }
    assert launch_report["actionable_blockers"]["real_data"][
        "blocking_roles"
    ] == ["episode", "real_collection_report"]
    assert launch_report["actionable_blockers"]["real_data"][
        "child_launch_gate"
    ] == launch_report["child_launch_gates"]["real_data"]
    assert launch_report["actionable_blockers"]["real_controls"][
        "child_launch_gate"
    ] == launch_report["child_launch_gates"]["offline_controls"]
    assert launch_report["actionable_blockers"]["predicted_dsg"][
        "child_launch_gate"
    ] == launch_report["child_launch_gates"]["predicted_dsg"]
    assert launch_report["actionable_blockers"]["review_artifacts"][
        "child_launch_gate"
    ] == launch_report["child_launch_gates"]["review_artifacts"]
    assert launch_report["actionable_blockers"]["review_artifacts"][
        "blocking_roles"
    ] == [
        "active_task_delta_report",
        "dashboard_bundle",
        "error_attribution_report",
        "graph_eval_report",
    ]
    assert launch_report["real_data_collection_intake_plan"] == {
        "track": "real_data",
        "blocked": True,
        "blocking_roles": ["episode", "real_collection_report"],
        "commands": {
            "collection_report": launch_report["child_launch_gates"]["real_data"][
                "collection_report_command"
            ],
            "compare_report": launch_report["child_launch_gates"]["real_data"][
                "compare_report_command"
            ],
            "request_bundle": launch_report["child_launch_gates"]["real_data"][
                "request_bundle_command"
            ],
            "validate_report": launch_report["child_launch_gates"]["real_data"][
                "validate_report_command"
            ],
        },
        "dataset_name": "ai2thor_real_smoke",
        "episode_paths": [str(episode_path)],
        "invalid_inputs": [],
        "missing_inputs": launch_report["tracks"]["real_data"]["missing_inputs"],
        "collection_report_receipt": {
            "asset_summary": None,
            "digest_valid": False,
            "failed_checks": ["real_collection_report_missing"],
            "path": str(real_collection_report_path),
            "readiness": None,
            "ready": False,
            "report_digest": None,
            "status": "missing",
            "validation_valid": False,
        },
        "real_collection_report_path": str(real_collection_report_path),
        "ready": False,
        "source_kind": "ai2thor",
        "thresholds": {
            "min_episode_count": 1,
            "min_frame_count": 30,
            "min_qa_count": 8,
            "min_scene_count": 1,
        },
    }
    assert launch_report["primary_evidence_intake_plan"] == {
        "track": "primary_evidence",
        "ready": False,
        "blocked_track_count": 3,
        "ready_track_count": 0,
        "final_commands": launch_report["next_commands"],
        "steps": [
            {
                "artifact_goal": "real_collection",
                "blocking_roles": ["episode", "real_collection_report"],
                "child_launch_gate": launch_report["child_launch_gates"][
                    "real_data"
                ],
                "order": 1,
                "ready": False,
                "recommended_command_keys": [
                    "collection_report_command",
                    "request_bundle_command",
                    "validate_report_command",
                    "compare_report_command",
                ],
                "track": "real_data",
            },
            {
                "artifact_goal": "offline_control_predictions",
                "blocking_roles": [
                    "candidate_prediction",
                    "gold_qa",
                    "offline_control_source_input",
                ],
                "child_launch_gate": launch_report["child_launch_gates"][
                    "offline_controls"
                ],
                "order": 2,
                "ready": False,
                "recommended_command_keys": [
                    "prediction_request_bundle_command",
                    "prediction_receipt_bundle_command",
                    "preflight_contract_command",
                    "artifact_launch_report_command",
                ],
                "track": "real_controls",
            },
            {
                "artifact_goal": "predicted_dsg_detector_inputs",
                "blocking_roles": ["detector_jsonl"],
                "child_launch_gate": launch_report["child_launch_gates"][
                    "predicted_dsg"
                ],
                "order": 3,
                "ready": False,
                "recommended_command_keys": [
                    "detector_request_bundle_command",
                    "detector_receipt_bundle_command",
                    "preflight_contract_command",
                    "artifact_launch_report_command",
                ],
                "track": "predicted_dsg",
            },
        ],
        "track_order": [
            "real_data",
            "real_controls",
            "predicted_dsg",
        ],
    }
    intake_plan = launch_report["external_artifact_intake_plan"]
    assert intake_plan == {
        "blocked_track_count": 4,
        "final_commands": launch_report["next_commands"],
        "ready_track_count": 1,
        "ready_tracks": ["run_outputs"],
        "steps": [
            {
                "blocking_roles": ["episode", "real_collection_report"],
                "child_launch_gate": launch_report["child_launch_gates"][
                    "real_data"
                ],
                "order": 1,
                "recommended_command_keys": [
                    "collection_report_command",
                    "request_bundle_command",
                    "validate_report_command",
                    "compare_report_command",
                ],
                "track": "real_data",
            },
            {
                "blocking_roles": [
                    "candidate_prediction",
                    "gold_qa",
                    "offline_control_source_input",
                ],
                "child_launch_gate": launch_report["child_launch_gates"][
                    "offline_controls"
                ],
                "order": 2,
                "recommended_command_keys": [
                    "prediction_request_bundle_command",
                    "prediction_receipt_bundle_command",
                    "preflight_contract_command",
                    "artifact_launch_report_command",
                ],
                "track": "real_controls",
            },
            {
                "blocking_roles": ["detector_jsonl"],
                "child_launch_gate": launch_report["child_launch_gates"][
                    "predicted_dsg"
                ],
                "order": 3,
                "recommended_command_keys": [
                    "detector_request_bundle_command",
                    "detector_receipt_bundle_command",
                    "preflight_contract_command",
                    "artifact_launch_report_command",
                ],
                "track": "predicted_dsg",
            },
            {
                "blocking_roles": [
                    "active_task_delta_report",
                    "dashboard_bundle",
                    "error_attribution_report",
                    "graph_eval_report",
                ],
                "child_launch_gate": launch_report["child_launch_gates"][
                    "review_artifacts"
                ],
                "order": 4,
                "recommended_command_keys": [
                    "active_task_delta_report_commands",
                    "dashboard_bundle_commands",
                    "error_attribution_report_commands",
                    "graph_eval_report_commands",
                ],
                "track": "review_artifacts",
            },
        ],
        "track_order": [
            "real_data",
            "real_controls",
            "predicted_dsg",
            "review_artifacts",
            "run_outputs",
        ],
    }
    assert launch_report["next_commands"] == {
        "compare_external_artifact_contracts": (
            "python scripts/run_real_experiment.py "
            f"--compare-external-artifact-contracts {contracts_path}"
        ),
        "preflight": (
            "python scripts/run_real_experiment.py --preflight-run-manifest "
            f"{run_manifest_path}"
        ),
        "run": (
            "python scripts/run_real_experiment.py --run-manifest "
            f"{run_manifest_path} --run-ledger-output "
            f"{root / 'outputs/real-experiment-run-ledger.json'}"
        ),
        "validate_external_artifact_contracts": (
            "python scripts/run_real_experiment.py "
            f"--validate-external-artifact-contracts {contracts_path}"
        ),
    }
    assert launch_report["report_digest"] == (
        lab.real_experiment_external_artifact_launch_report_digest(launch_report)
    )
    launch_report_path = root / "real-experiment-external-artifact-launch-report.json"
    saved_launch_report_path = (
        lab.save_real_experiment_external_artifact_launch_report(
            launch_report,
            launch_report_path,
        )
    )
    loaded_launch_report = lab.load_real_experiment_external_artifact_launch_report(
        launch_report_path
    )
    launch_report_validation = (
        lab.validate_real_experiment_external_artifact_launch_report(
            loaded_launch_report
        )
    )
    launch_report_comparison = (
        lab.compare_real_experiment_external_artifact_launch_report(
            loaded_launch_report
        )
    )
    assert saved_launch_report_path == launch_report_path
    assert loaded_launch_report == launch_report
    assert launch_report_validation["valid"] is True
    assert launch_report_validation["report_digest"] == launch_report["report_digest"]
    assert launch_report_comparison["matches"] is True
    assert launch_report_comparison["saved_digest"] == launch_report["report_digest"]
    assert launch_report_comparison["current_digest"] == launch_report["report_digest"]
    assert contracts["tracks"]["real_data"] == {
        "dataset_name": "ai2thor_real_smoke",
        "episode_paths": ["inputs/episodes/FloorPlan1.jsonl"],
        "min_episode_count": 1,
        "min_frame_count": 30,
        "min_qa_count": 8,
        "min_scene_count": 1,
        "real_collection_report_paths": ["inputs/real-collection-report.json"],
        "source_kind": "ai2thor",
    }
    assert contracts["tracks"]["real_controls"]["qa_path"] == "inputs/qa.jsonl"
    assert contracts["tracks"]["real_controls"]["candidate_prediction_path"] == (
        "inputs/candidate/predicted-graph-tool.jsonl"
    )
    assert [
        source["source_kind"]
        for source in contracts["tracks"]["real_controls"]["sources"]
    ] == ["vlm", "multi_frame_vlm", "caption_memory", "graph_text"]
    assert {
        source["expected_input_format"]
        for source in contracts["tracks"]["real_controls"]["sources"]
    } == {"qa_prediction"}
    assert contracts["tracks"]["predicted_dsg"]["detector_jsonl_path"] == (
        "inputs/predicted-dsg/detector-rgbd.jsonl"
    )
    assert contracts["tracks"]["predicted_dsg"]["required_evidence_kinds"] == [
        "depth",
        "detector",
        "rgb",
    ]
    assert contracts["tracks"]["review_artifacts"] == {
        "active_task_delta_report_paths": ["inputs/review/active-task-delta.json"],
        "dashboard_bundle_paths": ["inputs/review/dashboard.json"],
        "error_attribution_report_paths": ["inputs/review/error-attribution.json"],
        "graph_eval_report_paths": ["inputs/review/graph-eval.json"],
    }
    assert contracts["tracks"]["run_outputs"]["record_path"] == (
        "outputs/experiment-record.json"
    )
    assert contracts["tracks"]["run_outputs"][
        "real_experiment_run_ledger_path"
    ] == "outputs/real-experiment-run-ledger.json"
    input_rows = checklist["input_artifacts"]
    planned_rows = checklist["planned_output_artifacts"]
    assert len(input_rows) == 15
    assert len(planned_rows) == 25
    assert {
        (row["track"], row["group"], row["role"])
        for row in input_rows
        if row["status"] == "missing"
    } >= {
        ("real_data", "real_collection", "episode"),
        ("real_controls", "offline_controls", "gold_qa"),
        ("real_controls", "offline_controls", "offline_control_source_input"),
        ("predicted_dsg", "predicted_dsg", "detector_jsonl"),
        ("review_artifacts", "review_artifacts", "dashboard_bundle"),
    }
    assert {
        (row["track"], row["group"], row["role"]) for row in planned_rows
    } >= {
        ("real_controls", "offline_controls", "offline_control_prediction"),
        (
            "real_controls",
            "offline_controls",
            "offline_control_import_run_ledger",
        ),
        (
            "real_controls",
            "offline_controls",
            "offline_prediction_import_report",
        ),
        ("real_controls", "offline_controls", "offline_control_result_report"),
        (
            "predicted_dsg",
            "predicted_dsg",
            "predicted_dsg_detector_run_ledger",
        ),
        ("predicted_dsg", "predicted_dsg", "predicted_graph_report"),
        ("run_outputs", "real_run", "experiment_record"),
    }
    offline_missing = [
        row
        for row in input_rows
        if row["group"] == "offline_controls"
        and row["role"] == "offline_control_source_input"
    ]
    assert {row["metadata"]["source_kind"] for row in offline_missing} == {
        "caption_memory",
        "graph_text",
        "multi_frame_vlm",
        "vlm",
    }
    assert preflight["ready_to_run"] is False
    assert preflight["summary"]["missing_requirement_count"] == 0
    missing = {(item["group"], item["role"]) for item in preflight["missing_inputs"]}
    assert ("real_collection", "episode") in missing
    assert ("real_collection", "real_collection_report") in missing
    assert ("offline_controls", "gold_qa") in missing
    assert ("offline_controls", "offline_control_source_input") in missing
    assert ("offline_controls", "candidate_prediction") in missing
    assert ("predicted_dsg", "detector_jsonl") in missing
    assert ("review_artifacts", "active_task_delta_report") in missing
    assert ("review_artifacts", "dashboard_bundle") in missing
    assert ("review_artifacts", "error_attribution_report") in missing
    assert ("review_artifacts", "graph_eval_report") in missing
    tampered_contracts = {**loaded_contracts, "summary": {**contracts["summary"]}}
    tampered_contracts["summary"]["track_count"] = 4
    tampered_contracts["contracts_digest"] = (
        lab.real_experiment_external_artifact_contracts_digest(tampered_contracts)
    )
    tampered_validation = lab.validate_real_experiment_external_artifact_contracts(
        tampered_contracts
    )
    assert tampered_validation["valid"] is False
    assert {
        check["name"]
        for check in tampered_validation["checks"]
        if check["passed"] is False
    } == {"track_count"}


def test_external_artifact_contract_comparison_handles_relative_handoff_root(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    handoff = lab.write_real_experiment_handoff_manifests(
        root=Path("real-handoff"),
        dataset_name="ai2thor_real_smoke",
        episode_paths=(Path("inputs/episodes/FloorPlan1.jsonl"),),
        min_episode_count=1,
        min_scene_count=1,
        min_frame_count=1,
        min_qa_count=8,
    )
    contracts_path = Path(
        handoff["manifest_paths"]["real_experiment_external_artifact_contracts"]
    )
    contracts = lab.load_real_experiment_external_artifact_contracts(contracts_path)

    comparison = lab.compare_real_experiment_external_artifact_contracts(contracts)

    assert comparison["action"] == "compare_real_experiment_external_artifact_contracts"
    assert comparison["matches"] is True


def test_operator_progress_audits_saved_smoke_run_runbook_content(
    tmp_path: Path,
) -> None:
    root = tmp_path / "real-handoff"
    handoff = lab.write_real_experiment_handoff_manifests(
        root=root,
        dataset_name="ai2thor_real_smoke",
        episode_paths=(Path("inputs/episodes/FloorPlan1.jsonl"),),
        min_episode_count=1,
        min_qa_count=8,
    )
    operator_checklist_path = Path(
        handoff["manifest_paths"]["real_experiment_operator_checklist"]
    )
    run_manifest_path = Path(
        handoff["manifest_paths"]["real_experiment_run_manifest"]
    )
    operator_checklist = lab.load_real_experiment_operator_checklist(
        operator_checklist_path
    )

    inputs = _ready_package_inputs(tmp_path)
    run_manifest_path = _real_experiment_run_manifest(tmp_path, inputs)
    execution_packet_path = tmp_path / "run-manifest" / "execution-packet.json"
    execution_packet = _execution_packet_for_run_manifest(
        run_manifest_path,
        execution_packet_path,
    )
    smoke_checklist_path = root / "real-experiment-smoke-run-checklist.json"
    smoke_runbook_path = root / "real-experiment-smoke-run-runbook.json"
    execution_receipt_path = root / "real-experiment-execution-receipt.json"
    smoke_checklist = lab.real_experiment_smoke_run_checklist(
        execution_packet,
        execution_packet_path=execution_packet_path,
        execution_receipt_output_path=execution_receipt_path,
    )
    lab.save_real_experiment_smoke_run_checklist(
        smoke_checklist,
        smoke_checklist_path,
    )
    smoke_runbook = lab.real_experiment_smoke_run_runbook(
        smoke_checklist,
        smoke_run_checklist_path=smoke_checklist_path,
    )
    lab.save_real_experiment_smoke_run_runbook(smoke_runbook, smoke_runbook_path)

    ready_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    ready_rows = {step["key"]: step for step in ready_progress["steps"]}
    assert ready_rows["validate_smoke_run_runbook"]["target_status"] == "ready"
    assert ready_rows["validate_smoke_run_runbook"]["target_ready"] is True
    assert ready_rows["validate_smoke_run_runbook"]["target_audit"] == {
        "action": "validate_real_experiment_smoke_run_runbook",
        "checked": True,
        "valid": True,
    }
    assert ready_rows["compare_smoke_run_runbook"]["target_status"] == "ready"
    assert ready_rows["compare_smoke_run_runbook"]["target_ready"] is True
    assert ready_rows["compare_smoke_run_runbook"]["target_audit"] == {
        "action": "compare_real_experiment_smoke_run_runbook",
        "checked": True,
        "matches": True,
    }

    updated_checklist = lab.real_experiment_smoke_run_checklist(
        execution_packet,
        execution_packet_path=execution_packet_path,
        execution_receipt_output_path=root / "updated-execution-receipt.json",
    )
    lab.save_real_experiment_smoke_run_checklist(
        updated_checklist,
        smoke_checklist_path,
    )

    stale_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    stale_rows = {step["key"]: step for step in stale_progress["steps"]}
    assert stale_rows["validate_smoke_run_runbook"]["target_status"] == "ready"
    assert stale_rows["validate_smoke_run_runbook"]["target_ready"] is True
    assert stale_rows["compare_smoke_run_runbook"]["target_status"] == "stale"
    assert stale_rows["compare_smoke_run_runbook"]["target_ready"] is False
    assert stale_rows["compare_smoke_run_runbook"]["target_audit"] == {
        "action": "compare_real_experiment_smoke_run_runbook",
        "checked": True,
        "matches": False,
    }
    assert stale_progress["summary"]["all_targets_ready"] is False
    assert stale_progress["next_not_ready_step"]["key"] == (
        "external_artifact_launch_report"
    )
    validation = lab.validate_real_experiment_operator_progress_report(
        stale_progress
    )
    assert validation["valid"] is True
    comparison = lab.compare_real_experiment_operator_progress_report(stale_progress)
    assert comparison["matches"] is True


def test_operator_progress_audits_saved_run_ledger_content(
    tmp_path: Path,
) -> None:
    root = tmp_path / "real-handoff"
    handoff = lab.write_real_experiment_handoff_manifests(
        root=root,
        dataset_name="ai2thor_real_smoke",
        episode_paths=(Path("inputs/episodes/FloorPlan1.jsonl"),),
        min_episode_count=1,
        min_qa_count=8,
    )
    operator_checklist_path = Path(
        handoff["manifest_paths"]["real_experiment_operator_checklist"]
    )
    run_manifest_path = Path(
        handoff["manifest_paths"]["real_experiment_run_manifest"]
    )
    operator_checklist = lab.load_real_experiment_operator_checklist(
        operator_checklist_path
    )

    inputs = _ready_package_inputs(tmp_path)
    run_manifest_path = _real_experiment_run_manifest(tmp_path, inputs)
    execution_packet_path = tmp_path / "run-manifest" / "execution-packet.json"
    _execution_packet_for_run_manifest(run_manifest_path, execution_packet_path)
    run_result = lab.run_real_experiment_manifest(
        run_manifest_path,
        approved_execution_packet_path=execution_packet_path,
    )
    source_ledger = lab.load_real_experiment_run_ledger(
        run_result["real_experiment_run_ledger_path"]
    )
    target_ledger_path = root / "outputs/real-experiment-run-ledger.json"
    lab.save_real_experiment_run_ledger(source_ledger, target_ledger_path)

    ready_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    ready_rows = {step["key"]: step for step in ready_progress["steps"]}
    assert ready_rows["validate_run_ledger"]["target_status"] == "ready"
    assert ready_rows["validate_run_ledger"]["target_ready"] is True
    assert ready_rows["validate_run_ledger"]["target_audit"] == {
        "action": "validate_real_experiment_run_ledger",
        "checked": True,
        "valid": True,
    }
    assert ready_rows["compare_run_ledger"]["target_status"] == "ready"
    assert ready_rows["compare_run_ledger"]["target_ready"] is True
    assert ready_rows["compare_run_ledger"]["target_audit"] == {
        "action": "compare_real_experiment_run_ledger",
        "checked": True,
        "matches": True,
    }

    manifest_payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    manifest_payload["min_qa_count"] = 9
    run_manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    stale_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    stale_rows = {step["key"]: step for step in stale_progress["steps"]}
    assert stale_rows["validate_run_ledger"]["target_status"] == "ready"
    assert stale_rows["validate_run_ledger"]["target_ready"] is True
    assert stale_rows["compare_run_ledger"]["target_status"] == "stale"
    assert stale_rows["compare_run_ledger"]["target_ready"] is False
    assert stale_rows["compare_run_ledger"]["target_audit"] == {
        "action": "compare_real_experiment_run_ledger",
        "checked": True,
        "matches": False,
    }
    validation = lab.validate_real_experiment_operator_progress_report(
        stale_progress
    )
    assert validation["valid"] is True
    comparison = lab.compare_real_experiment_operator_progress_report(stale_progress)
    assert comparison["matches"] is True


def test_operator_progress_audits_saved_execution_receipt_content(
    tmp_path: Path,
) -> None:
    root = tmp_path / "real-handoff"
    handoff = lab.write_real_experiment_handoff_manifests(
        root=root,
        dataset_name="ai2thor_real_smoke",
        episode_paths=(Path("inputs/episodes/FloorPlan1.jsonl"),),
        min_episode_count=1,
        min_qa_count=8,
    )
    operator_checklist_path = Path(
        handoff["manifest_paths"]["real_experiment_operator_checklist"]
    )
    operator_checklist = lab.load_real_experiment_operator_checklist(
        operator_checklist_path
    )

    inputs = _ready_package_inputs(tmp_path)
    run_manifest_path = _real_experiment_run_manifest(tmp_path, inputs)
    execution_packet_path = tmp_path / "run-manifest" / "execution-packet.json"
    execution_packet = _execution_packet_for_run_manifest(
        run_manifest_path,
        execution_packet_path,
    )
    lab.run_real_experiment_manifest(
        run_manifest_path,
        approved_execution_packet_path=execution_packet_path,
    )
    execution_receipt = lab.real_experiment_execution_receipt(
        execution_packet,
        execution_packet_path=execution_packet_path,
    )
    target_receipt_path = root / "real-experiment-execution-receipt.json"
    lab.save_real_experiment_execution_receipt(
        execution_receipt,
        target_receipt_path,
    )

    ready_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    ready_rows = {step["key"]: step for step in ready_progress["steps"]}
    assert ready_rows["validate_execution_receipt"]["target_status"] == "ready"
    assert ready_rows["validate_execution_receipt"]["target_ready"] is True
    assert ready_rows["validate_execution_receipt"]["target_audit"] == {
        "action": "validate_real_experiment_execution_receipt",
        "checked": True,
        "valid": True,
    }
    assert ready_rows["compare_execution_receipt"]["target_status"] == "ready"
    assert ready_rows["compare_execution_receipt"]["target_ready"] is True
    assert ready_rows["compare_execution_receipt"]["target_audit"] == {
        "action": "compare_real_experiment_execution_receipt",
        "checked": True,
        "matches": True,
    }

    manifest_payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    manifest_payload["min_qa_count"] = 9
    run_manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    stale_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    stale_rows = {step["key"]: step for step in stale_progress["steps"]}
    assert stale_rows["validate_execution_receipt"]["target_status"] == "ready"
    assert stale_rows["validate_execution_receipt"]["target_ready"] is True
    assert stale_rows["compare_execution_receipt"]["target_status"] == "stale"
    assert stale_rows["compare_execution_receipt"]["target_ready"] is False
    assert stale_rows["compare_execution_receipt"]["target_audit"] == {
        "action": "compare_real_experiment_execution_receipt",
        "checked": True,
        "matches": False,
    }
    validation = lab.validate_real_experiment_operator_progress_report(
        stale_progress
    )
    assert validation["valid"] is True
    comparison = lab.compare_real_experiment_operator_progress_report(stale_progress)
    assert comparison["matches"] is True


def test_operator_progress_audits_saved_research_review_content(
    tmp_path: Path,
) -> None:
    root = tmp_path / "real-handoff"
    handoff = lab.write_real_experiment_handoff_manifests(
        root=root,
        dataset_name="ai2thor_real_smoke",
        episode_paths=(Path("inputs/episodes/FloorPlan1.jsonl"),),
        min_episode_count=1,
        min_qa_count=8,
    )
    operator_checklist_path = Path(
        handoff["manifest_paths"]["real_experiment_operator_checklist"]
    )
    operator_checklist = lab.load_real_experiment_operator_checklist(
        operator_checklist_path
    )

    inputs = _ready_package_inputs(tmp_path)
    run_manifest_path = _real_experiment_run_manifest(tmp_path, inputs)
    execution_packet_path = tmp_path / "run-manifest" / "execution-packet.json"
    execution_packet = _execution_packet_for_run_manifest(
        run_manifest_path,
        execution_packet_path,
    )
    lab.run_real_experiment_manifest(
        run_manifest_path,
        approved_execution_packet_path=execution_packet_path,
    )
    target_receipt_path = root / "real-experiment-execution-receipt.json"
    execution_receipt = lab.real_experiment_execution_receipt(
        execution_packet,
        execution_packet_path=execution_packet_path,
    )
    lab.save_real_experiment_execution_receipt(
        execution_receipt,
        target_receipt_path,
    )
    research_review = lab.real_experiment_research_review(
        execution_receipt,
        execution_receipt_path=target_receipt_path,
    )
    target_review_path = root / "real-experiment-research-review.json"
    lab.save_real_experiment_research_review(research_review, target_review_path)

    ready_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    ready_rows = {step["key"]: step for step in ready_progress["steps"]}
    assert ready_rows["validate_research_review"]["target_status"] == "ready"
    assert ready_rows["validate_research_review"]["target_ready"] is True
    assert ready_rows["validate_research_review"]["target_audit"] == {
        "action": "validate_real_experiment_research_review",
        "checked": True,
        "valid": True,
    }
    assert ready_rows["compare_research_review"]["target_status"] == "ready"
    assert ready_rows["compare_research_review"]["target_ready"] is True
    assert ready_rows["compare_research_review"]["target_audit"] == {
        "action": "compare_real_experiment_research_review",
        "checked": True,
        "matches": True,
    }

    stale_receipt = dict(execution_receipt)
    stale_receipt["run_manifest_digest"] = "f" * 64
    stale_receipt["receipt_digest"] = lab.real_experiment_execution_receipt_digest(
        stale_receipt
    )
    lab.save_real_experiment_execution_receipt(stale_receipt, target_receipt_path)

    stale_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    stale_rows = {step["key"]: step for step in stale_progress["steps"]}
    assert stale_rows["validate_research_review"]["target_status"] == "ready"
    assert stale_rows["validate_research_review"]["target_ready"] is True
    assert stale_rows["compare_research_review"]["target_status"] == "stale"
    assert stale_rows["compare_research_review"]["target_ready"] is False
    assert stale_rows["compare_research_review"]["target_audit"] == {
        "action": "compare_real_experiment_research_review",
        "checked": True,
        "matches": False,
    }
    validation = lab.validate_real_experiment_operator_progress_report(
        stale_progress
    )
    assert validation["valid"] is True
    comparison = lab.compare_real_experiment_operator_progress_report(stale_progress)
    assert comparison["matches"] is True


def test_operator_progress_audits_saved_claim_readiness_content(
    tmp_path: Path,
) -> None:
    root = tmp_path / "real-handoff"
    handoff = lab.write_real_experiment_handoff_manifests(
        root=root,
        dataset_name="ai2thor_real_smoke",
        episode_paths=(Path("inputs/episodes/FloorPlan1.jsonl"),),
        min_episode_count=1,
        min_qa_count=8,
    )
    operator_checklist_path = Path(
        handoff["manifest_paths"]["real_experiment_operator_checklist"]
    )
    operator_checklist = lab.load_real_experiment_operator_checklist(
        operator_checklist_path
    )

    inputs = _ready_package_inputs(tmp_path)
    run_manifest_path = _real_experiment_run_manifest(tmp_path, inputs)
    execution_packet_path = tmp_path / "run-manifest" / "execution-packet.json"
    execution_packet = _execution_packet_for_run_manifest(
        run_manifest_path,
        execution_packet_path,
    )
    lab.run_real_experiment_manifest(
        run_manifest_path,
        approved_execution_packet_path=execution_packet_path,
    )
    execution_receipt_path = root / "real-experiment-execution-receipt.json"
    execution_receipt = lab.real_experiment_execution_receipt(
        execution_packet,
        execution_packet_path=execution_packet_path,
    )
    lab.save_real_experiment_execution_receipt(
        execution_receipt,
        execution_receipt_path,
    )
    research_review_path = root / "real-experiment-research-review.json"
    research_review = lab.real_experiment_research_review(
        execution_receipt,
        execution_receipt_path=execution_receipt_path,
    )
    lab.save_real_experiment_research_review(research_review, research_review_path)
    claim_readiness_path = root / "real-experiment-claim-readiness.json"
    claim_readiness = lab.real_experiment_claim_readiness(
        research_review,
        research_review_path=research_review_path,
        min_dynamic_qa_count=0,
        min_episode_count=1,
        min_qa_count=8,
        min_scene_count=1,
    )
    lab.save_real_experiment_claim_readiness(
        claim_readiness,
        claim_readiness_path,
    )

    ready_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    ready_rows = {step["key"]: step for step in ready_progress["steps"]}
    assert ready_rows["validate_claim_readiness"]["target_status"] == "ready"
    assert ready_rows["validate_claim_readiness"]["target_ready"] is True
    assert ready_rows["validate_claim_readiness"]["target_audit"] == {
        "action": "validate_real_experiment_claim_readiness",
        "checked": True,
        "valid": True,
    }
    assert ready_rows["compare_claim_readiness"]["target_status"] == "ready"
    assert ready_rows["compare_claim_readiness"]["target_ready"] is True
    assert ready_rows["compare_claim_readiness"]["target_audit"] == {
        "action": "compare_real_experiment_claim_readiness",
        "checked": True,
        "matches": True,
    }

    stale_review = dict(research_review)
    stale_review["execution_receipt_digest"] = "e" * 64
    stale_review["review_digest"] = lab.real_experiment_research_review_digest(
        stale_review
    )
    lab.save_real_experiment_research_review(stale_review, research_review_path)

    stale_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    stale_rows = {step["key"]: step for step in stale_progress["steps"]}
    assert stale_rows["validate_claim_readiness"]["target_status"] == "ready"
    assert stale_rows["validate_claim_readiness"]["target_ready"] is True
    assert stale_rows["compare_claim_readiness"]["target_status"] == "stale"
    assert stale_rows["compare_claim_readiness"]["target_ready"] is False
    assert stale_rows["compare_claim_readiness"]["target_audit"] == {
        "action": "compare_real_experiment_claim_readiness",
        "checked": True,
        "matches": False,
    }
    validation = lab.validate_real_experiment_operator_progress_report(
        stale_progress
    )
    assert validation["valid"] is True
    comparison = lab.compare_real_experiment_operator_progress_report(stale_progress)
    assert comparison["matches"] is True


def test_operator_progress_audits_saved_external_artifact_contracts_content(
    tmp_path: Path,
) -> None:
    root = tmp_path / "real-handoff"
    handoff = lab.write_real_experiment_handoff_manifests(
        root=root,
        dataset_name="ai2thor_real_smoke",
        episode_paths=(Path("inputs/episodes/FloorPlan1.jsonl"),),
        min_episode_count=1,
        min_scene_count=1,
        min_frame_count=1,
        min_qa_count=8,
    )
    operator_checklist_path = Path(
        handoff["manifest_paths"]["real_experiment_operator_checklist"]
    )
    run_manifest_path = Path(
        handoff["manifest_paths"]["real_experiment_run_manifest"]
    )
    operator_checklist = lab.load_real_experiment_operator_checklist(
        operator_checklist_path
    )
    contracts_path = Path(
        handoff["manifest_paths"]["real_experiment_external_artifact_contracts"]
    )

    ready_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    ready_rows = {step["key"]: step for step in ready_progress["steps"]}
    assert ready_rows["validate_external_artifact_contracts"][
        "target_status"
    ] == "ready"
    assert ready_rows["validate_external_artifact_contracts"]["target_ready"] is True
    assert ready_rows["validate_external_artifact_contracts"]["target_audit"] == {
        "action": "validate_real_experiment_external_artifact_contracts",
        "checked": True,
        "valid": True,
    }
    assert ready_rows["compare_external_artifact_contracts"][
        "target_status"
    ] == "ready"
    assert ready_rows["compare_external_artifact_contracts"]["target_ready"] is True
    assert ready_rows["compare_external_artifact_contracts"]["target_audit"] == {
        "action": "compare_real_experiment_external_artifact_contracts",
        "checked": True,
        "matches": True,
    }

    original_run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    stale_run_manifest = dict(original_run_manifest)
    stale_run_manifest["episode_paths"] = [
        *stale_run_manifest["episode_paths"],
        "inputs/episodes/FloorPlan2.jsonl",
    ]
    run_manifest_path.write_text(
        json.dumps(stale_run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    stale_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    stale_rows = {step["key"]: step for step in stale_progress["steps"]}
    assert stale_rows["validate_external_artifact_contracts"][
        "target_status"
    ] == "ready"
    assert stale_rows["validate_external_artifact_contracts"]["target_ready"] is True
    assert stale_rows["compare_external_artifact_contracts"][
        "target_status"
    ] == "stale"
    assert stale_rows["compare_external_artifact_contracts"]["target_ready"] is False
    assert stale_rows["compare_external_artifact_contracts"]["target_audit"] == {
        "action": "compare_real_experiment_external_artifact_contracts",
        "checked": True,
        "matches": False,
    }

    run_manifest_path.write_text(
        json.dumps(original_run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    tampered_contracts = lab.load_real_experiment_external_artifact_contracts(
        contracts_path
    )
    tampered_contracts["contracts_digest"] = "0" * 64
    lab.save_real_experiment_external_artifact_contracts(
        tampered_contracts,
        contracts_path,
    )

    invalid_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    invalid_rows = {step["key"]: step for step in invalid_progress["steps"]}
    assert invalid_rows["validate_external_artifact_contracts"][
        "target_status"
    ] == "invalid"
    assert invalid_rows["validate_external_artifact_contracts"][
        "target_ready"
    ] is False
    assert invalid_rows["validate_external_artifact_contracts"]["target_audit"] == {
        "action": "validate_real_experiment_external_artifact_contracts",
        "checked": True,
        "valid": False,
    }
    validation = lab.validate_real_experiment_operator_progress_report(
        invalid_progress
    )
    assert validation["valid"] is True
    comparison = lab.compare_real_experiment_operator_progress_report(
        invalid_progress
    )
    assert comparison["matches"] is True


def test_operator_progress_audits_saved_smoke_run_checklist_content(
    tmp_path: Path,
) -> None:
    root = tmp_path / "real-handoff"
    handoff = lab.write_real_experiment_handoff_manifests(
        root=root,
        dataset_name="ai2thor_real_smoke",
        episode_paths=(Path("inputs/episodes/FloorPlan1.jsonl"),),
        min_episode_count=1,
        min_qa_count=8,
    )
    operator_checklist_path = Path(
        handoff["manifest_paths"]["real_experiment_operator_checklist"]
    )
    operator_checklist = lab.load_real_experiment_operator_checklist(
        operator_checklist_path
    )

    inputs = _ready_package_inputs(tmp_path)
    run_manifest_path = _real_experiment_run_manifest(tmp_path, inputs)
    execution_packet_path = tmp_path / "run-manifest" / "execution-packet.json"
    execution_packet = _execution_packet_for_run_manifest(
        run_manifest_path,
        execution_packet_path,
    )
    smoke_checklist_path = root / "real-experiment-smoke-run-checklist.json"
    execution_receipt_path = root / "real-experiment-execution-receipt.json"
    smoke_checklist = lab.real_experiment_smoke_run_checklist(
        execution_packet,
        execution_packet_path=execution_packet_path,
        execution_receipt_output_path=execution_receipt_path,
    )
    lab.save_real_experiment_smoke_run_checklist(
        smoke_checklist,
        smoke_checklist_path,
    )

    ready_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    ready_rows = {step["key"]: step for step in ready_progress["steps"]}
    assert ready_rows["validate_smoke_run_checklist"]["target_status"] == "ready"
    assert ready_rows["validate_smoke_run_checklist"]["target_ready"] is True
    assert ready_rows["validate_smoke_run_checklist"]["target_audit"] == {
        "action": "validate_real_experiment_smoke_run_checklist",
        "checked": True,
        "valid": True,
    }
    assert ready_rows["compare_smoke_run_checklist"]["target_status"] == "ready"
    assert ready_rows["compare_smoke_run_checklist"]["target_ready"] is True
    assert ready_rows["compare_smoke_run_checklist"]["target_audit"] == {
        "action": "compare_real_experiment_smoke_run_checklist",
        "checked": True,
        "matches": True,
    }

    updated_packet = json.loads(json.dumps(execution_packet))
    updated_packet["execution_commands"][0]["command"] = (
        "python scripts/run_real_experiment.py --preflight-run-manifest "
        f"{run_manifest_path} --strict"
    )
    updated_packet["packet_digest"] = lab.real_experiment_execution_packet_digest(
        updated_packet
    )
    lab.save_real_experiment_execution_packet(
        updated_packet,
        execution_packet_path,
    )

    stale_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    stale_rows = {step["key"]: step for step in stale_progress["steps"]}
    assert stale_rows["validate_smoke_run_checklist"]["target_status"] == "ready"
    assert stale_rows["validate_smoke_run_checklist"]["target_ready"] is True
    assert stale_rows["compare_smoke_run_checklist"]["target_status"] == "stale"
    assert stale_rows["compare_smoke_run_checklist"]["target_ready"] is False
    assert stale_rows["compare_smoke_run_checklist"]["target_audit"] == {
        "action": "compare_real_experiment_smoke_run_checklist",
        "checked": True,
        "matches": False,
    }
    validation = lab.validate_real_experiment_operator_progress_report(
        stale_progress
    )
    assert validation["valid"] is True
    comparison = lab.compare_real_experiment_operator_progress_report(stale_progress)
    assert comparison["matches"] is True


def test_operator_progress_audits_external_returned_evidence_content(
    tmp_path: Path,
) -> None:
    root = tmp_path / "real-handoff"
    handoff = lab.write_real_experiment_handoff_manifests(
        root=root,
        dataset_name="ai2thor_real_smoke",
        episode_paths=(Path("inputs/episodes/FloorPlan1.jsonl"),),
        min_episode_count=1,
        min_scene_count=1,
        min_frame_count=1,
        min_qa_count=8,
    )
    operator_checklist_path = Path(
        handoff["manifest_paths"]["real_experiment_operator_checklist"]
    )
    operator_checklist = lab.load_real_experiment_operator_checklist(
        operator_checklist_path
    )
    offline_manifest_path = Path(
        handoff["manifest_paths"]["offline_control_import_manifest"]
    )
    predicted_manifest_path = Path(
        handoff["manifest_paths"]["predicted_dsg_detector_run_manifest"]
    )
    real_collection_report_path = root / "inputs/real-collection-report.json"
    offline_receipt_path = root / "offline-control-prediction-receipt-bundle.json"
    predicted_receipt_path = root / "predicted-dsg-detector-receipt-bundle.json"

    _write_ready_real_collection_receipt(root)
    _write_ready_offline_control_receipt(root, offline_manifest_path)
    _write_ready_predicted_dsg_receipt(root, predicted_manifest_path)

    ready_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    ready_rows = {step["key"]: step for step in ready_progress["steps"]}
    assert ready_rows["real_collection_report"]["target_status"] == "ready"
    assert ready_rows["real_collection_report"]["target_ready"] is True
    assert ready_rows["real_collection_report"]["target_audit"] == {
        "action": "validate_real_collection_report",
        "checked": True,
        "valid": True,
    }
    assert ready_rows["offline_control_prediction_receipt_bundle"][
        "target_status"
    ] == "ready"
    assert ready_rows["offline_control_prediction_receipt_bundle"][
        "target_ready"
    ] is True
    assert ready_rows["offline_control_prediction_receipt_bundle"][
        "target_audit"
    ] == {
        "action": "validate_offline_control_prediction_receipt_bundle",
        "checked": True,
        "valid": True,
    }
    assert ready_rows["predicted_dsg_detector_receipt_bundle"][
        "target_status"
    ] == "ready"
    assert ready_rows["predicted_dsg_detector_receipt_bundle"][
        "target_ready"
    ] is True
    assert ready_rows["predicted_dsg_detector_receipt_bundle"][
        "target_audit"
    ] == {
        "action": "validate_predicted_dsg_detector_receipt_bundle",
        "checked": True,
        "valid": True,
    }

    invalid_real_report = lab.load_real_collection_report(
        real_collection_report_path
    )
    invalid_real_report["report_digest"] = "0" * 64
    lab.save_real_collection_report(
        invalid_real_report,
        real_collection_report_path,
    )
    invalid_offline_receipt = lab.load_offline_control_prediction_receipt_bundle(
        offline_receipt_path
    )
    invalid_offline_receipt["receipt_digest"] = "0" * 64
    lab.save_offline_control_prediction_receipt_bundle(
        invalid_offline_receipt,
        offline_receipt_path,
    )
    invalid_predicted_receipt = lab.load_predicted_dsg_detector_receipt_bundle(
        predicted_receipt_path
    )
    invalid_predicted_receipt["receipt_digest"] = "0" * 64
    lab.save_predicted_dsg_detector_receipt_bundle(
        invalid_predicted_receipt,
        predicted_receipt_path,
    )

    invalid_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    invalid_rows = {step["key"]: step for step in invalid_progress["steps"]}
    assert invalid_rows["real_collection_report"]["target_status"] == "invalid"
    assert invalid_rows["real_collection_report"]["target_ready"] is False
    assert invalid_rows["real_collection_report"]["target_audit"] == {
        "action": "validate_real_collection_report",
        "checked": True,
        "valid": False,
    }
    assert invalid_rows["offline_control_prediction_receipt_bundle"][
        "target_status"
    ] == "invalid"
    assert invalid_rows["offline_control_prediction_receipt_bundle"][
        "target_ready"
    ] is False
    assert invalid_rows["offline_control_prediction_receipt_bundle"][
        "target_audit"
    ] == {
        "action": "validate_offline_control_prediction_receipt_bundle",
        "checked": True,
        "valid": False,
    }
    assert invalid_rows["predicted_dsg_detector_receipt_bundle"][
        "target_status"
    ] == "invalid"
    assert invalid_rows["predicted_dsg_detector_receipt_bundle"][
        "target_ready"
    ] is False
    assert invalid_rows["predicted_dsg_detector_receipt_bundle"][
        "target_audit"
    ] == {
        "action": "validate_predicted_dsg_detector_receipt_bundle",
        "checked": True,
        "valid": False,
    }
    validation = lab.validate_real_experiment_operator_progress_report(
        invalid_progress
    )
    assert validation["valid"] is True
    comparison = lab.compare_real_experiment_operator_progress_report(
        invalid_progress
    )
    assert comparison["matches"] is True


def test_operator_progress_audits_external_request_bundle_content(
    tmp_path: Path,
) -> None:
    root = tmp_path / "real-handoff"
    handoff = lab.write_real_experiment_handoff_manifests(
        root=root,
        dataset_name="ai2thor_real_smoke",
        episode_paths=(Path("inputs/episodes/FloorPlan1.jsonl"),),
        min_episode_count=1,
        min_scene_count=1,
        min_frame_count=1,
        min_qa_count=8,
    )
    operator_checklist_path = Path(
        handoff["manifest_paths"]["real_experiment_operator_checklist"]
    )
    operator_checklist = lab.load_real_experiment_operator_checklist(
        operator_checklist_path
    )
    contracts_path = Path(
        handoff["manifest_paths"]["real_experiment_external_artifact_contracts"]
    )
    offline_manifest_path = Path(
        handoff["manifest_paths"]["offline_control_import_manifest"]
    )
    predicted_manifest_path = Path(
        handoff["manifest_paths"]["predicted_dsg_detector_run_manifest"]
    )
    launch_report_path = root / "real-experiment-external-artifact-launch-report.json"
    request_package_path = (
        root / "real-experiment-primary-evidence-request-package.json"
    )
    real_request_path = root / "real-collection-request-bundle.json"
    offline_request_path = root / "offline-control-prediction-request-bundle.json"
    predicted_request_path = root / "predicted-dsg-detector-request-bundle.json"

    _write_ready_real_collection_receipt(root)
    _write_ready_offline_control_receipt(root, offline_manifest_path)
    _write_ready_predicted_dsg_receipt(root, predicted_manifest_path)
    launch_report = lab.real_experiment_external_artifact_launch_report(
        lab.load_real_experiment_external_artifact_contracts(contracts_path),
        contracts_path=contracts_path,
    )
    lab.save_real_experiment_external_artifact_launch_report(
        launch_report,
        launch_report_path,
    )
    request_package = lab.real_experiment_primary_evidence_request_package(
        launch_report,
        launch_report_path=launch_report_path,
    )
    lab.save_real_experiment_primary_evidence_request_package(
        request_package,
        request_package_path,
    )
    write_result = lab.write_real_experiment_primary_evidence_request_bundles(
        request_package
    )
    assert write_result["summary"]["written_request_bundle_count"] == 3

    ready_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    ready_rows = {step["key"]: step for step in ready_progress["steps"]}
    assert ready_rows["write_primary_evidence_request_bundles"][
        "target_status"
    ] == "ready"
    assert ready_rows["write_primary_evidence_request_bundles"][
        "target_ready"
    ] is True
    assert ready_rows["write_primary_evidence_request_bundles"][
        "target_audit"
    ] == {
        "action": "validate_real_experiment_primary_evidence_request_bundles",
        "checked": True,
        "track_statuses": {
            "predicted_dsg": "ready",
            "real_controls": "ready",
            "real_data": "ready",
        },
        "valid": True,
    }

    invalid_real_request = lab.load_real_collection_request_bundle(real_request_path)
    invalid_real_request["request_bundle_digest"] = "0" * 64
    lab.save_real_collection_request_bundle(invalid_real_request, real_request_path)
    invalid_offline_request = lab.load_offline_control_prediction_request_bundle(
        offline_request_path
    )
    invalid_offline_request["request_bundle_digest"] = "0" * 64
    lab.save_offline_control_prediction_request_bundle(
        invalid_offline_request,
        offline_request_path,
    )
    invalid_predicted_request = lab.load_predicted_dsg_detector_request_bundle(
        predicted_request_path
    )
    invalid_predicted_request["request_bundle_digest"] = "0" * 64
    lab.save_predicted_dsg_detector_request_bundle(
        invalid_predicted_request,
        predicted_request_path,
    )

    invalid_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    invalid_rows = {step["key"]: step for step in invalid_progress["steps"]}
    assert invalid_rows["write_primary_evidence_request_bundles"][
        "target_status"
    ] == "invalid"
    assert invalid_rows["write_primary_evidence_request_bundles"][
        "target_ready"
    ] is False
    assert invalid_rows["write_primary_evidence_request_bundles"][
        "target_audit"
    ] == {
        "action": "validate_real_experiment_primary_evidence_request_bundles",
        "checked": True,
        "track_statuses": {
            "predicted_dsg": "invalid",
            "real_controls": "invalid",
            "real_data": "invalid",
        },
        "valid": False,
    }
    validation = lab.validate_real_experiment_operator_progress_report(
        invalid_progress
    )
    assert validation["valid"] is True
    comparison = lab.compare_real_experiment_operator_progress_report(
        invalid_progress
    )
    assert comparison["matches"] is True


def test_operator_progress_audits_saved_primary_evidence_status_content(
    tmp_path: Path,
) -> None:
    root = tmp_path / "real-handoff"
    handoff = lab.write_real_experiment_handoff_manifests(
        root=root,
        dataset_name="ai2thor_real_smoke",
        episode_paths=(Path("inputs/episodes/FloorPlan1.jsonl"),),
        min_episode_count=1,
        min_scene_count=1,
        min_frame_count=1,
        min_qa_count=8,
    )
    operator_checklist_path = Path(
        handoff["manifest_paths"]["real_experiment_operator_checklist"]
    )
    operator_checklist = lab.load_real_experiment_operator_checklist(
        operator_checklist_path
    )
    contracts_path = Path(
        handoff["manifest_paths"]["real_experiment_external_artifact_contracts"]
    )
    offline_manifest_path = Path(
        handoff["manifest_paths"]["offline_control_import_manifest"]
    )
    predicted_manifest_path = Path(
        handoff["manifest_paths"]["predicted_dsg_detector_run_manifest"]
    )
    launch_report_path = root / "real-experiment-external-artifact-launch-report.json"
    primary_status_path = root / "real-experiment-primary-evidence-status.json"
    _write_ready_real_collection_receipt(root)
    _write_ready_offline_control_receipt(root, offline_manifest_path)
    _write_ready_predicted_dsg_receipt(root, predicted_manifest_path)
    launch_report = lab.real_experiment_external_artifact_launch_report(
        lab.load_real_experiment_external_artifact_contracts(contracts_path),
        contracts_path=contracts_path,
    )
    lab.save_real_experiment_external_artifact_launch_report(
        launch_report,
        launch_report_path,
    )
    primary_status = lab.real_experiment_primary_evidence_status(
        launch_report,
        launch_report_path=launch_report_path,
    )
    lab.save_real_experiment_primary_evidence_status(
        primary_status,
        primary_status_path,
    )

    ready_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    ready_rows = {step["key"]: step for step in ready_progress["steps"]}
    assert ready_rows["validate_primary_evidence_status"][
        "target_status"
    ] == "ready"
    assert ready_rows["validate_primary_evidence_status"]["target_ready"] is True
    assert ready_rows["validate_primary_evidence_status"]["target_audit"] == {
        "action": "validate_real_experiment_primary_evidence_status",
        "checked": True,
        "valid": True,
    }
    assert ready_rows["compare_primary_evidence_status"][
        "target_status"
    ] == "ready"
    assert ready_rows["compare_primary_evidence_status"]["target_ready"] is True
    assert ready_rows["compare_primary_evidence_status"]["target_audit"] == {
        "action": "compare_real_experiment_primary_evidence_status",
        "checked": True,
        "matches": True,
    }

    stale_collection_report_path = root / "inputs/real-collection-report.json"
    stale_collection_report = lab.load_real_collection_report(
        stale_collection_report_path
    )
    stale_collection_report["report_digest"] = "0" * 64
    lab.save_real_collection_report(
        stale_collection_report,
        stale_collection_report_path,
    )

    stale_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    stale_rows = {step["key"]: step for step in stale_progress["steps"]}
    assert stale_rows["validate_primary_evidence_status"][
        "target_status"
    ] == "ready"
    assert stale_rows["validate_primary_evidence_status"]["target_ready"] is True
    assert stale_rows["compare_primary_evidence_status"][
        "target_status"
    ] == "stale"
    assert stale_rows["compare_primary_evidence_status"]["target_ready"] is False
    assert stale_rows["compare_primary_evidence_status"]["target_audit"] == {
        "action": "compare_real_experiment_primary_evidence_status",
        "checked": True,
        "matches": False,
    }
    validation = lab.validate_real_experiment_operator_progress_report(
        stale_progress
    )
    assert validation["valid"] is True
    comparison = lab.compare_real_experiment_operator_progress_report(stale_progress)
    assert comparison["matches"] is True


def test_operator_progress_audits_saved_primary_evidence_request_package_content(
    tmp_path: Path,
) -> None:
    root = tmp_path / "real-handoff"
    handoff = lab.write_real_experiment_handoff_manifests(
        root=root,
        dataset_name="ai2thor_real_smoke",
        episode_paths=(Path("inputs/episodes/FloorPlan1.jsonl"),),
        min_episode_count=1,
        min_scene_count=1,
        min_frame_count=1,
        min_qa_count=8,
    )
    operator_checklist_path = Path(
        handoff["manifest_paths"]["real_experiment_operator_checklist"]
    )
    operator_checklist = lab.load_real_experiment_operator_checklist(
        operator_checklist_path
    )
    contracts_path = Path(
        handoff["manifest_paths"]["real_experiment_external_artifact_contracts"]
    )
    offline_manifest_path = Path(
        handoff["manifest_paths"]["offline_control_import_manifest"]
    )
    predicted_manifest_path = Path(
        handoff["manifest_paths"]["predicted_dsg_detector_run_manifest"]
    )
    launch_report_path = root / "real-experiment-external-artifact-launch-report.json"
    request_package_path = (
        root / "real-experiment-primary-evidence-request-package.json"
    )
    _write_ready_real_collection_receipt(root)
    _write_ready_offline_control_receipt(root, offline_manifest_path)
    _write_ready_predicted_dsg_receipt(root, predicted_manifest_path)
    launch_report = lab.real_experiment_external_artifact_launch_report(
        lab.load_real_experiment_external_artifact_contracts(contracts_path),
        contracts_path=contracts_path,
    )
    lab.save_real_experiment_external_artifact_launch_report(
        launch_report,
        launch_report_path,
    )
    request_package = lab.real_experiment_primary_evidence_request_package(
        launch_report,
        launch_report_path=launch_report_path,
    )
    lab.save_real_experiment_primary_evidence_request_package(
        request_package,
        request_package_path,
    )

    ready_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    ready_rows = {step["key"]: step for step in ready_progress["steps"]}
    assert (
        ready_rows["validate_primary_evidence_request_package"]["target_status"]
        == "ready"
    )
    assert (
        ready_rows["validate_primary_evidence_request_package"]["target_ready"]
        is True
    )
    assert ready_rows["validate_primary_evidence_request_package"]["target_audit"] == {
        "action": "validate_real_experiment_primary_evidence_request_package",
        "checked": True,
        "valid": True,
    }
    assert (
        ready_rows["compare_primary_evidence_request_package"]["target_status"]
        == "ready"
    )
    assert (
        ready_rows["compare_primary_evidence_request_package"]["target_ready"]
        is True
    )
    assert ready_rows["compare_primary_evidence_request_package"]["target_audit"] == {
        "action": "compare_real_experiment_primary_evidence_request_package",
        "checked": True,
        "matches": True,
    }

    stale_collection_report_path = root / "inputs/real-collection-report.json"
    stale_collection_report = lab.load_real_collection_report(
        stale_collection_report_path
    )
    stale_collection_report["report_digest"] = "0" * 64
    lab.save_real_collection_report(
        stale_collection_report,
        stale_collection_report_path,
    )

    stale_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    stale_rows = {step["key"]: step for step in stale_progress["steps"]}
    assert (
        stale_rows["validate_primary_evidence_request_package"]["target_status"]
        == "ready"
    )
    assert (
        stale_rows["validate_primary_evidence_request_package"]["target_ready"]
        is True
    )
    assert (
        stale_rows["compare_primary_evidence_request_package"]["target_status"]
        == "stale"
    )
    assert (
        stale_rows["compare_primary_evidence_request_package"]["target_ready"]
        is False
    )
    assert stale_rows["compare_primary_evidence_request_package"]["target_audit"] == {
        "action": "compare_real_experiment_primary_evidence_request_package",
        "checked": True,
        "matches": False,
    }
    validation = lab.validate_real_experiment_operator_progress_report(
        stale_progress
    )
    assert validation["valid"] is True
    comparison = lab.compare_real_experiment_operator_progress_report(stale_progress)
    assert comparison["matches"] is True


def test_operator_progress_audits_saved_primary_evidence_return_content(
    tmp_path: Path,
) -> None:
    root = tmp_path / "real-handoff"
    handoff = lab.write_real_experiment_handoff_manifests(
        root=root,
        dataset_name="ai2thor_real_smoke",
        episode_paths=(Path("inputs/episodes/FloorPlan1.jsonl"),),
        min_episode_count=1,
        min_scene_count=1,
        min_frame_count=1,
        min_qa_count=8,
    )
    operator_checklist_path = Path(
        handoff["manifest_paths"]["real_experiment_operator_checklist"]
    )
    operator_checklist = lab.load_real_experiment_operator_checklist(
        operator_checklist_path
    )
    contracts_path = Path(
        handoff["manifest_paths"]["real_experiment_external_artifact_contracts"]
    )
    offline_manifest_path = Path(
        handoff["manifest_paths"]["offline_control_import_manifest"]
    )
    predicted_manifest_path = Path(
        handoff["manifest_paths"]["predicted_dsg_detector_run_manifest"]
    )
    launch_report_path = root / "real-experiment-external-artifact-launch-report.json"
    request_package_path = (
        root / "real-experiment-primary-evidence-request-package.json"
    )
    return_checklist_path = (
        root / "real-experiment-primary-evidence-return-checklist.json"
    )
    return_progress_path = (
        root / "real-experiment-primary-evidence-return-progress.json"
    )
    _write_ready_real_collection_receipt(root)
    _write_ready_offline_control_receipt(root, offline_manifest_path)
    _write_ready_predicted_dsg_receipt(root, predicted_manifest_path)
    contracts = lab.load_real_experiment_external_artifact_contracts(contracts_path)
    launch_report = lab.real_experiment_external_artifact_launch_report(
        contracts,
        contracts_path=contracts_path,
    )
    lab.save_real_experiment_external_artifact_launch_report(
        launch_report,
        launch_report_path,
    )
    request_package = lab.real_experiment_primary_evidence_request_package(
        launch_report,
        launch_report_path=launch_report_path,
    )
    lab.save_real_experiment_primary_evidence_request_package(
        request_package,
        request_package_path,
    )
    return_checklist = lab.real_experiment_primary_evidence_return_checklist(
        request_package,
        request_package_path=request_package_path,
    )
    lab.save_real_experiment_primary_evidence_return_checklist(
        return_checklist,
        return_checklist_path,
    )
    return_progress = lab.real_experiment_primary_evidence_return_progress_report(
        return_checklist,
        return_checklist_path=return_checklist_path,
    )
    lab.save_real_experiment_primary_evidence_return_progress_report(
        return_progress,
        return_progress_path,
    )

    ready_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    ready_rows = {step["key"]: step for step in ready_progress["steps"]}
    assert ready_rows["validate_primary_evidence_return_checklist"][
        "target_status"
    ] == "ready"
    assert ready_rows["validate_primary_evidence_return_checklist"][
        "target_ready"
    ] is True
    assert ready_rows["validate_primary_evidence_return_checklist"][
        "target_audit"
    ] == {
        "action": "validate_real_experiment_primary_evidence_return_checklist",
        "checked": True,
        "valid": True,
    }
    assert ready_rows["compare_primary_evidence_return_checklist"][
        "target_status"
    ] == "ready"
    assert ready_rows["compare_primary_evidence_return_checklist"][
        "target_ready"
    ] is True
    assert ready_rows["compare_primary_evidence_return_checklist"][
        "target_audit"
    ] == {
        "action": "compare_real_experiment_primary_evidence_return_checklist",
        "checked": True,
        "matches": True,
    }
    assert ready_rows["validate_primary_evidence_return_progress_report"][
        "target_status"
    ] == "ready"
    assert ready_rows["validate_primary_evidence_return_progress_report"][
        "target_ready"
    ] is True
    assert ready_rows["validate_primary_evidence_return_progress_report"][
        "target_audit"
    ] == {
        "action": "validate_real_experiment_primary_evidence_return_progress_report",
        "checked": True,
        "valid": True,
    }
    assert ready_rows["compare_primary_evidence_return_progress_report"][
        "target_status"
    ] == "ready"
    assert ready_rows["compare_primary_evidence_return_progress_report"][
        "target_ready"
    ] is True
    assert ready_rows["compare_primary_evidence_return_progress_report"][
        "target_audit"
    ] == {
        "action": "compare_real_experiment_primary_evidence_return_progress_report",
        "checked": True,
        "matches": True,
    }

    stale_collection_report_path = root / "inputs/real-collection-report.json"
    stale_collection_report_path.unlink()
    refreshed_launch_report = lab.real_experiment_external_artifact_launch_report(
        contracts,
        contracts_path=contracts_path,
    )
    lab.save_real_experiment_external_artifact_launch_report(
        refreshed_launch_report,
        launch_report_path,
    )
    refreshed_request_package = lab.real_experiment_primary_evidence_request_package(
        refreshed_launch_report,
        launch_report_path=launch_report_path,
    )
    lab.save_real_experiment_primary_evidence_request_package(
        refreshed_request_package,
        request_package_path,
    )

    stale_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    stale_rows = {step["key"]: step for step in stale_progress["steps"]}
    assert stale_rows["validate_primary_evidence_return_checklist"][
        "target_status"
    ] == "ready"
    assert stale_rows["validate_primary_evidence_return_checklist"][
        "target_ready"
    ] is True
    assert stale_rows["compare_primary_evidence_return_checklist"][
        "target_status"
    ] == "stale"
    assert stale_rows["compare_primary_evidence_return_checklist"][
        "target_ready"
    ] is False
    assert stale_rows["compare_primary_evidence_return_checklist"][
        "target_audit"
    ] == {
        "action": "compare_real_experiment_primary_evidence_return_checklist",
        "checked": True,
        "matches": False,
    }
    assert stale_rows["validate_primary_evidence_return_progress_report"][
        "target_status"
    ] == "ready"
    assert stale_rows["validate_primary_evidence_return_progress_report"][
        "target_ready"
    ] is True
    assert stale_rows["compare_primary_evidence_return_progress_report"][
        "target_status"
    ] == "stale"
    assert stale_rows["compare_primary_evidence_return_progress_report"][
        "target_ready"
    ] is False
    assert stale_rows["compare_primary_evidence_return_progress_report"][
        "target_audit"
    ] == {
        "action": "compare_real_experiment_primary_evidence_return_progress_report",
        "checked": True,
        "matches": False,
    }
    validation = lab.validate_real_experiment_operator_progress_report(
        stale_progress
    )
    assert validation["valid"] is True
    comparison = lab.compare_real_experiment_operator_progress_report(stale_progress)
    assert comparison["matches"] is True


def test_operator_progress_audits_saved_external_artifact_launch_report_content(
    tmp_path: Path,
) -> None:
    root = tmp_path / "real-handoff"
    handoff = lab.write_real_experiment_handoff_manifests(
        root=root,
        dataset_name="ai2thor_real_smoke",
        episode_paths=(Path("inputs/episodes/FloorPlan1.jsonl"),),
        min_episode_count=1,
        min_scene_count=1,
        min_frame_count=1,
        min_qa_count=8,
    )
    operator_checklist_path = Path(
        handoff["manifest_paths"]["real_experiment_operator_checklist"]
    )
    operator_checklist = lab.load_real_experiment_operator_checklist(
        operator_checklist_path
    )
    contracts_path = Path(
        handoff["manifest_paths"]["real_experiment_external_artifact_contracts"]
    )
    offline_manifest_path = Path(
        handoff["manifest_paths"]["offline_control_import_manifest"]
    )
    predicted_manifest_path = Path(
        handoff["manifest_paths"]["predicted_dsg_detector_run_manifest"]
    )
    launch_report_path = root / "real-experiment-external-artifact-launch-report.json"
    _write_ready_real_collection_receipt(root)
    _write_ready_offline_control_receipt(root, offline_manifest_path)
    _write_ready_predicted_dsg_receipt(root, predicted_manifest_path)
    launch_report = lab.real_experiment_external_artifact_launch_report(
        lab.load_real_experiment_external_artifact_contracts(contracts_path),
        contracts_path=contracts_path,
    )
    lab.save_real_experiment_external_artifact_launch_report(
        launch_report,
        launch_report_path,
    )

    ready_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    ready_rows = {step["key"]: step for step in ready_progress["steps"]}
    assert ready_rows["validate_external_artifact_launch_report"][
        "target_status"
    ] == "ready"
    assert ready_rows["validate_external_artifact_launch_report"][
        "target_ready"
    ] is True
    assert ready_rows["validate_external_artifact_launch_report"]["target_audit"] == {
        "action": "validate_real_experiment_external_artifact_launch_report",
        "checked": True,
        "valid": True,
    }
    assert ready_rows["compare_external_artifact_launch_report"][
        "target_status"
    ] == "ready"
    assert ready_rows["compare_external_artifact_launch_report"][
        "target_ready"
    ] is True
    assert ready_rows["compare_external_artifact_launch_report"]["target_audit"] == {
        "action": "compare_real_experiment_external_artifact_launch_report",
        "checked": True,
        "matches": True,
    }

    stale_collection_report_path = root / "inputs/real-collection-report.json"
    stale_collection_report = lab.load_real_collection_report(
        stale_collection_report_path
    )
    stale_collection_report["report_digest"] = "0" * 64
    lab.save_real_collection_report(
        stale_collection_report,
        stale_collection_report_path,
    )

    stale_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    stale_rows = {step["key"]: step for step in stale_progress["steps"]}
    assert stale_rows["validate_external_artifact_launch_report"][
        "target_status"
    ] == "ready"
    assert stale_rows["validate_external_artifact_launch_report"][
        "target_ready"
    ] is True
    assert stale_rows["compare_external_artifact_launch_report"][
        "target_status"
    ] == "stale"
    assert stale_rows["compare_external_artifact_launch_report"][
        "target_ready"
    ] is False
    assert stale_rows["compare_external_artifact_launch_report"]["target_audit"] == {
        "action": "compare_real_experiment_external_artifact_launch_report",
        "checked": True,
        "matches": False,
    }
    validation = lab.validate_real_experiment_operator_progress_report(
        stale_progress
    )
    assert validation["valid"] is True
    comparison = lab.compare_real_experiment_operator_progress_report(stale_progress)
    assert comparison["matches"] is True


def test_operator_progress_audits_saved_primary_evidence_acceptance_report_content(
    tmp_path: Path,
) -> None:
    root = tmp_path / "real-handoff"
    handoff = lab.write_real_experiment_handoff_manifests(
        root=root,
        dataset_name="ai2thor_real_smoke",
        episode_paths=(Path("inputs/episodes/FloorPlan1.jsonl"),),
        min_episode_count=1,
        min_scene_count=1,
        min_frame_count=1,
        min_qa_count=8,
    )
    operator_checklist_path = Path(
        handoff["manifest_paths"]["real_experiment_operator_checklist"]
    )
    operator_checklist = lab.load_real_experiment_operator_checklist(
        operator_checklist_path
    )
    contracts_path = Path(
        handoff["manifest_paths"]["real_experiment_external_artifact_contracts"]
    )
    offline_manifest_path = Path(
        handoff["manifest_paths"]["offline_control_import_manifest"]
    )
    predicted_manifest_path = Path(
        handoff["manifest_paths"]["predicted_dsg_detector_run_manifest"]
    )
    launch_report_path = root / "real-experiment-external-artifact-launch-report.json"
    request_package_path = (
        root / "real-experiment-primary-evidence-request-package.json"
    )
    return_checklist_path = (
        root / "real-experiment-primary-evidence-return-checklist.json"
    )
    return_progress_path = (
        root / "real-experiment-primary-evidence-return-progress.json"
    )
    acceptance_report_path = (
        root / "real-experiment-primary-evidence-acceptance-report.json"
    )
    _write_ready_real_collection_receipt(root)
    _write_ready_offline_control_receipt(root, offline_manifest_path)
    _write_ready_predicted_dsg_receipt(root, predicted_manifest_path)
    launch_report = lab.real_experiment_external_artifact_launch_report(
        lab.load_real_experiment_external_artifact_contracts(contracts_path),
        contracts_path=contracts_path,
    )
    lab.save_real_experiment_external_artifact_launch_report(
        launch_report,
        launch_report_path,
    )
    request_package = lab.real_experiment_primary_evidence_request_package(
        launch_report,
        launch_report_path=launch_report_path,
    )
    lab.save_real_experiment_primary_evidence_request_package(
        request_package,
        request_package_path,
    )
    return_checklist = lab.real_experiment_primary_evidence_return_checklist(
        request_package,
        request_package_path=request_package_path,
    )
    lab.save_real_experiment_primary_evidence_return_checklist(
        return_checklist,
        return_checklist_path,
    )
    return_progress = lab.real_experiment_primary_evidence_return_progress_report(
        return_checklist,
        return_checklist_path=return_checklist_path,
    )
    lab.save_real_experiment_primary_evidence_return_progress_report(
        return_progress,
        return_progress_path,
    )
    acceptance_report = lab.real_experiment_primary_evidence_acceptance_report(
        return_progress,
        return_progress_path=return_progress_path,
    )
    lab.save_real_experiment_primary_evidence_acceptance_report(
        acceptance_report,
        acceptance_report_path,
    )

    ready_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    ready_rows = {step["key"]: step for step in ready_progress["steps"]}
    assert ready_rows["validate_primary_evidence_acceptance_report"][
        "target_status"
    ] == "ready"
    assert ready_rows["validate_primary_evidence_acceptance_report"][
        "target_ready"
    ] is True
    assert ready_rows["validate_primary_evidence_acceptance_report"][
        "target_audit"
    ] == {
        "action": "validate_real_experiment_primary_evidence_acceptance_report",
        "checked": True,
        "valid": True,
    }
    assert ready_rows["compare_primary_evidence_acceptance_report"][
        "target_status"
    ] == "ready"
    assert ready_rows["compare_primary_evidence_acceptance_report"][
        "target_ready"
    ] is True
    assert ready_rows["compare_primary_evidence_acceptance_report"][
        "target_audit"
    ] == {
        "action": "compare_real_experiment_primary_evidence_acceptance_report",
        "checked": True,
        "matches": True,
    }

    stale_collection_report_path = root / "inputs/real-collection-report.json"
    stale_collection_report = lab.load_real_collection_report(
        stale_collection_report_path
    )
    stale_collection_report["report_digest"] = "0" * 64
    lab.save_real_collection_report(
        stale_collection_report,
        stale_collection_report_path,
    )

    stale_progress = lab.real_experiment_operator_progress_report(
        operator_checklist,
        checklist_path=operator_checklist_path,
    )
    stale_rows = {step["key"]: step for step in stale_progress["steps"]}
    assert stale_rows["validate_primary_evidence_acceptance_report"][
        "target_status"
    ] == "ready"
    assert stale_rows["validate_primary_evidence_acceptance_report"][
        "target_ready"
    ] is True
    assert stale_rows["compare_primary_evidence_acceptance_report"][
        "target_status"
    ] == "stale"
    assert stale_rows["compare_primary_evidence_acceptance_report"][
        "target_ready"
    ] is False
    assert stale_rows["compare_primary_evidence_acceptance_report"][
        "target_audit"
    ] == {
        "action": "compare_real_experiment_primary_evidence_acceptance_report",
        "checked": True,
        "matches": False,
    }
    validation = lab.validate_real_experiment_operator_progress_report(
        stale_progress
    )
    assert validation["valid"] is True
    comparison = lab.compare_real_experiment_operator_progress_report(stale_progress)
    assert comparison["matches"] is True


def test_real_experiment_launch_report_projects_real_collection_receipt_failures(
    tmp_path: Path,
) -> None:
    root = tmp_path / "receipt-handoff"
    result = lab.write_real_experiment_handoff_manifests(
        root=root,
        dataset_name="ai2thor_real_smoke",
        episode_paths=(Path("inputs/episodes/FloorPlan1.jsonl"),),
        min_episode_count=1,
        min_scene_count=1,
        min_frame_count=1,
        min_qa_count=8,
    )
    contracts_path = Path(
        result["manifest_paths"]["real_experiment_external_artifact_contracts"]
    )
    episode_path = root / "inputs/episodes/FloorPlan1.jsonl"
    real_collection_report_path = root / "inputs/real-collection-report.json"
    frame = lab.EpisodeFrame(
        episode_id="ai2thor_real_receipt_001",
        scene_id="FloorPlan1",
        step=1,
        rgb_path="frames/000001.rgb.png",
        depth_path="frames/000001.depth.png",
        segmentation_path="frames/000001.segmentation.png",
        agent_id="agent",
        agent_pose=lab.Pose3D(0.0, 0.0, 0.0),
        action="Initialize",
        visible_object_ids=("mug_1",),
        metadata={
            "adapter": "ai2thor",
            "collection_kind": "real",
            "source_kind": "ai2thor",
        },
    )
    lab.save_episode_sequence((frame,), episode_path)
    for asset_path_text in (frame.depth_path, frame.segmentation_path):
        assert asset_path_text is not None
        asset_path = episode_path.parent / asset_path_text
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        asset_path.write_text(f"{frame.step}\n", encoding="utf-8")
    report = lab.real_collection_report(
        dataset_name="ai2thor_real_smoke",
        episode_paths=(episode_path,),
        source_kind="ai2thor",
        min_episode_count=1,
        min_scene_count=1,
        min_frame_count=1,
    )
    lab.save_real_collection_report(report, real_collection_report_path)
    launch_report = lab.real_experiment_external_artifact_launch_report(
        lab.load_real_experiment_external_artifact_contracts(contracts_path),
        contracts_path=contracts_path,
    )
    plan = launch_report["real_data_collection_intake_plan"]

    assert launch_report["tracks"]["real_data"]["ready"] is True
    assert plan["blocked"] is True
    assert plan["ready"] is False
    assert plan["collection_report_receipt"] == {
        "asset_summary": report["collection_summary"]["asset_summary"],
        "digest_valid": True,
        "failed_checks": ["frame_assets_present"],
        "path": str(real_collection_report_path),
        "readiness": {
            "failed_check_count": 1,
            "failed_checks": ["frame_assets_present"],
            "ready": False,
        },
        "ready": False,
        "report_digest": report["report_digest"],
        "status": "not_ready",
        "validation_valid": True,
    }


def test_real_experiment_launch_report_rejects_tampered_real_collection_receipt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tampered-real-collection-receipt"
    result = lab.write_real_experiment_handoff_manifests(
        root=root,
        dataset_name="ai2thor_real_smoke",
        episode_paths=(Path("inputs/episodes/FloorPlan1.jsonl"),),
        min_episode_count=1,
        min_scene_count=1,
        min_frame_count=1,
        min_qa_count=8,
    )
    contracts_path = Path(
        result["manifest_paths"]["real_experiment_external_artifact_contracts"]
    )
    real_collection_report_path = root / "inputs/real-collection-report.json"
    _write_ready_real_collection_receipt(root)
    report = lab.load_real_collection_report(real_collection_report_path)
    report["report_digest"] = "0" * 64
    lab.save_real_collection_report(report, real_collection_report_path)

    launch_report = lab.real_experiment_external_artifact_launch_report(
        lab.load_real_experiment_external_artifact_contracts(contracts_path),
        contracts_path=contracts_path,
    )
    receipt = launch_report["real_data_collection_intake_plan"][
        "collection_report_receipt"
    ]

    assert launch_report["tracks"]["real_data"]["ready"] is True
    assert launch_report["real_data_collection_intake_plan"]["ready"] is False
    assert launch_report["real_data_collection_intake_plan"]["blocked"] is True
    assert "real_collection_report" in launch_report["real_data_collection_intake_plan"][
        "blocking_roles"
    ]
    assert receipt["status"] == "invalid"
    assert receipt["ready"] is False
    assert receipt["digest_valid"] is False
    assert receipt["validation_valid"] is False
    assert receipt["failed_checks"] == ["real_collection_report_digest_invalid"]
    assert receipt["report_digest"] == "0" * 64
    assert launch_report["primary_evidence_receipt_gate"]["ready"] is False
    assert {
        row["track"]: row["receipt_status"]
        for row in launch_report["primary_evidence_receipt_gate"]["blocked_tracks"]
    }["real_data"] == "invalid"


def test_real_experiment_launch_report_rejects_tampered_returned_receipt_bundles(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tampered-returned-receipt-bundles"
    result = lab.write_real_experiment_handoff_manifests(
        root=root,
        dataset_name="ai2thor_real_smoke",
        episode_paths=(Path("inputs/episodes/FloorPlan1.jsonl"),),
        min_episode_count=1,
        min_scene_count=1,
        min_frame_count=1,
        min_qa_count=8,
    )
    contracts_path = Path(
        result["manifest_paths"]["real_experiment_external_artifact_contracts"]
    )
    offline_manifest_path = Path(
        result["manifest_paths"]["offline_control_import_manifest"]
    )
    predicted_manifest_path = Path(
        result["manifest_paths"]["predicted_dsg_detector_run_manifest"]
    )
    _write_ready_real_collection_receipt(root)
    _write_ready_offline_control_contract(root, offline_manifest_path)
    _write_ready_predicted_dsg_contract(root, predicted_manifest_path)
    _write_ready_offline_control_receipt_bundle(root, offline_manifest_path)
    _write_ready_predicted_dsg_receipt_bundle(root, predicted_manifest_path)
    _write_review_artifact_placeholders(root)

    offline_receipt_path = root / "offline-control-prediction-receipt-bundle.json"
    offline_bundle = lab.load_offline_control_prediction_receipt_bundle(
        offline_receipt_path
    )
    offline_bundle["summary"]["ready_source_count"] = 3
    offline_bundle["receipt_bundle_digest"] = (
        lab.offline_control_prediction_receipt_bundle_digest(offline_bundle)
    )
    lab.save_offline_control_prediction_receipt_bundle(
        offline_bundle,
        offline_receipt_path,
    )

    predicted_receipt_path = root / "predicted-dsg-detector-receipt-bundle.json"
    predicted_bundle = lab.load_predicted_dsg_detector_receipt_bundle(
        predicted_receipt_path
    )
    predicted_bundle["summary"]["observation_count"] = 999
    predicted_bundle["receipt_bundle_digest"] = (
        lab.predicted_dsg_detector_receipt_bundle_digest(predicted_bundle)
    )
    lab.save_predicted_dsg_detector_receipt_bundle(
        predicted_bundle,
        predicted_receipt_path,
    )

    launch_report = lab.real_experiment_external_artifact_launch_report(
        lab.load_real_experiment_external_artifact_contracts(contracts_path),
        contracts_path=contracts_path,
    )
    control_receipt = launch_report["real_controls_prediction_intake_plan"][
        "prediction_receipt_bundle"
    ]
    detector_receipt = launch_report["predicted_dsg_detector_intake_plan"][
        "detector_receipt_bundle"
    ]

    assert control_receipt["digest_valid"] is True
    assert control_receipt["manifest_matches"] is True
    assert control_receipt["validation_valid"] is False
    assert control_receipt["ready_to_import"] is False
    assert control_receipt["status"] == "invalid"
    assert detector_receipt["digest_valid"] is True
    assert detector_receipt["manifest_matches"] is True
    assert detector_receipt["validation_valid"] is False
    assert detector_receipt["ready_to_build"] is False
    assert detector_receipt["status"] == "invalid"
    assert launch_report["ready_to_run"] is False
    assert launch_report["primary_evidence_receipt_gate"]["ready"] is False
    assert {
        row["track"]: row["receipt_status"]
        for row in launch_report["primary_evidence_receipt_gate"]["blocked_tracks"]
    } == {
        "predicted_dsg": "invalid",
        "real_controls": "invalid",
    }


def test_real_experiment_launch_report_projects_offline_control_receipt_failures(
    tmp_path: Path,
) -> None:
    root = tmp_path / "control-receipt-handoff"
    result = lab.write_real_experiment_handoff_manifests(
        root=root,
        dataset_name="ai2thor_real_smoke",
        episode_paths=(Path("inputs/episodes/FloorPlan1.jsonl"),),
        min_episode_count=1,
        min_scene_count=1,
        min_frame_count=1,
        min_qa_count=8,
    )
    contracts_path = Path(
        result["manifest_paths"]["real_experiment_external_artifact_contracts"]
    )
    offline_manifest_path = Path(
        result["manifest_paths"]["offline_control_import_manifest"]
    )
    offline_contracts_path = root / "offline-control-artifact-contracts.json"
    manifest = lab.load_offline_control_import_manifest(offline_manifest_path)
    cases = _offline_control_cases()
    predictions = _offline_control_predictions(cases)

    qa_path = _anchor(offline_manifest_path.parent, str(manifest["qa_path"]))
    candidate_path = _anchor(
        offline_manifest_path.parent,
        str(manifest["candidate_prediction_path"]),
    )
    lab.save_qa_dataset(cases, qa_path)
    lab.save_qa_predictions(predictions, candidate_path)
    for source in manifest["sources"]:
        source_kind = str(source["source_kind"])
        source_path = _anchor(offline_manifest_path.parent, str(source["input_path"]))
        if source_kind == "vlm":
            lab.save_qa_predictions(predictions[:-1], source_path)
        else:
            lab.save_qa_predictions(predictions, source_path)

    child_preflight = lab.offline_control_import_manifest_preflight(
        offline_manifest_path
    )
    lab.save_offline_control_artifact_contracts(
        child_preflight["artifact_contracts"],
        offline_contracts_path,
    )
    child_launch_report = lab.offline_control_artifact_launch_report(
        lab.load_offline_control_artifact_contracts(offline_contracts_path),
        manifest_path=offline_manifest_path,
        contracts_path=offline_contracts_path,
    )
    launch_report = lab.real_experiment_external_artifact_launch_report(
        lab.load_real_experiment_external_artifact_contracts(contracts_path),
        contracts_path=contracts_path,
    )
    plan = launch_report["real_controls_prediction_intake_plan"]

    assert launch_report["tracks"]["real_controls"]["ready"] is True
    assert child_launch_report["ready_to_import"] is False
    assert plan["blocked"] is True
    assert plan["ready"] is False
    assert plan["artifact_contract_receipt"] == {
        "actionable_blockers": child_launch_report["actionable_blockers"],
        "external_prediction_intake_plan": child_launch_report[
            "external_prediction_intake_plan"
        ],
        "manifest_path": str(offline_manifest_path),
        "path": str(offline_contracts_path),
        "ready_to_import": False,
        "report_digest": child_launch_report["report_digest"],
        "status": "not_ready",
        "summary": child_launch_report["summary"],
    }


def test_real_experiment_launch_report_projects_predicted_dsg_receipt_failures(
    tmp_path: Path,
) -> None:
    root = tmp_path / "detector-receipt-handoff"
    result = lab.write_real_experiment_handoff_manifests(
        root=root,
        dataset_name="ai2thor_real_smoke",
        episode_paths=(Path("inputs/episodes/FloorPlan1.jsonl"),),
        min_episode_count=1,
        min_scene_count=1,
        min_frame_count=1,
        min_qa_count=8,
    )
    contracts_path = Path(
        result["manifest_paths"]["real_experiment_external_artifact_contracts"]
    )
    predicted_manifest_path = Path(
        result["manifest_paths"]["predicted_dsg_detector_run_manifest"]
    )
    predicted_contract_path = (
        root / "predicted-dsg-detector-artifact-contract.json"
    )
    manifest = lab.load_predicted_dsg_detector_run_manifest(predicted_manifest_path)
    detector_jsonl_path = Path(str(manifest["detector_jsonl_path"]))
    _write_detector_records_and_assets(detector_jsonl_path)
    missing_asset_path = detector_jsonl_path.parent / "frames/000001.rgb.png"
    missing_asset_path.unlink()

    child_preflight = lab.predicted_dsg_detector_run_manifest_preflight(
        predicted_manifest_path
    )
    lab.save_predicted_dsg_detector_artifact_contract(
        child_preflight["artifact_contract"],
        predicted_contract_path,
    )
    child_launch_report = lab.predicted_dsg_detector_artifact_launch_report(
        lab.load_predicted_dsg_detector_artifact_contract(predicted_contract_path),
        manifest_path=predicted_manifest_path,
        contract_path=predicted_contract_path,
    )
    launch_report = lab.real_experiment_external_artifact_launch_report(
        lab.load_real_experiment_external_artifact_contracts(contracts_path),
        contracts_path=contracts_path,
    )
    plan = launch_report["predicted_dsg_detector_intake_plan"]

    assert launch_report["tracks"]["predicted_dsg"]["ready"] is True
    assert child_launch_report["ready_to_build"] is False
    assert child_launch_report["external_detector_intake_plan"]["asset_summary"][
        "missing_assets"
    ] == [
        {
            "kind": "rgb",
            "path": "frames/000001.rgb.png",
            "resolved_path": str(missing_asset_path),
        },
    ]
    assert plan["blocked"] is True
    assert plan["ready"] is False
    assert plan["artifact_contract_receipt"] == {
        "actionable_blockers": child_launch_report["actionable_blockers"],
        "external_detector_intake_plan": child_launch_report[
            "external_detector_intake_plan"
        ],
        "manifest_path": str(predicted_manifest_path),
        "path": str(predicted_contract_path),
        "ready_to_build": False,
        "report_digest": child_launch_report["report_digest"],
        "status": "not_ready",
        "summary": child_launch_report["summary"],
    }


def test_real_experiment_launch_report_gates_ready_to_run_on_primary_receipts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "receipt-gated-handoff"
    result = lab.write_real_experiment_handoff_manifests(
        root=root,
        dataset_name="ai2thor_real_smoke",
        episode_paths=(Path("inputs/episodes/FloorPlan1.jsonl"),),
        min_episode_count=1,
        min_scene_count=1,
        min_frame_count=1,
        min_qa_count=8,
    )
    contracts_path = Path(
        result["manifest_paths"]["real_experiment_external_artifact_contracts"]
    )
    offline_manifest_path = Path(
        result["manifest_paths"]["offline_control_import_manifest"]
    )
    predicted_manifest_path = Path(
        result["manifest_paths"]["predicted_dsg_detector_run_manifest"]
    )
    _write_not_ready_real_collection_receipt(root)
    _write_ready_offline_control_receipt(root, offline_manifest_path)
    _write_ready_predicted_dsg_receipt(root, predicted_manifest_path)
    _write_review_artifact_placeholders(root)

    launch_report = lab.real_experiment_external_artifact_launch_report(
        lab.load_real_experiment_external_artifact_contracts(contracts_path),
        contracts_path=contracts_path,
    )

    assert launch_report["preflight_ready_to_run"] is True
    assert launch_report["tracks"]["real_data"]["ready"] is True
    assert launch_report["tracks"]["real_controls"]["ready"] is True
    assert launch_report["tracks"]["predicted_dsg"]["ready"] is True
    assert launch_report["ready_to_run"] is False
    assert launch_report["real_data_collection_intake_plan"]["ready"] is False
    assert launch_report["real_controls_prediction_intake_plan"]["ready"] is True
    assert launch_report["predicted_dsg_detector_intake_plan"]["ready"] is True
    assert launch_report["primary_evidence_receipt_gate"] == {
        "track": "primary_evidence",
        "ready": False,
        "blocked_track_count": 1,
        "blocked_tracks": [
            {
                "blocking_roles": ["real_collection_report"],
                "intake_plan_key": "real_data_collection_intake_plan",
                "receipt_status": "not_ready",
                "track": "real_data",
            },
        ],
        "ready_track_count": 2,
        "ready_tracks": ["real_controls", "predicted_dsg"],
        "track_order": ["real_data", "real_controls", "predicted_dsg"],
    }
    assert launch_report["primary_evidence_intake_plan"]["ready"] is False
    assert launch_report["primary_evidence_intake_plan"]["blocked_track_count"] == 1


def test_real_experiment_launch_report_requires_returned_receipt_bundles(
    tmp_path: Path,
) -> None:
    root = tmp_path / "returned-receipt-gated-handoff"
    result = lab.write_real_experiment_handoff_manifests(
        root=root,
        dataset_name="ai2thor_real_smoke",
        episode_paths=(Path("inputs/episodes/FloorPlan1.jsonl"),),
        min_episode_count=1,
        min_scene_count=1,
        min_frame_count=1,
        min_qa_count=8,
    )
    contracts_path = Path(
        result["manifest_paths"]["real_experiment_external_artifact_contracts"]
    )
    offline_manifest_path = Path(
        result["manifest_paths"]["offline_control_import_manifest"]
    )
    predicted_manifest_path = Path(
        result["manifest_paths"]["predicted_dsg_detector_run_manifest"]
    )
    _write_ready_real_collection_receipt(root)
    _write_ready_offline_control_contract(root, offline_manifest_path)
    _write_ready_predicted_dsg_contract(root, predicted_manifest_path)
    _write_review_artifact_placeholders(root)

    launch_report = lab.real_experiment_external_artifact_launch_report(
        lab.load_real_experiment_external_artifact_contracts(contracts_path),
        contracts_path=contracts_path,
    )

    assert launch_report["preflight_ready_to_run"] is True
    assert launch_report["tracks"]["real_data"]["ready"] is True
    assert launch_report["tracks"]["real_controls"]["ready"] is True
    assert launch_report["tracks"]["predicted_dsg"]["ready"] is True
    assert launch_report["real_data_collection_intake_plan"]["ready"] is True
    assert launch_report["real_controls_prediction_intake_plan"]["artifact_contract_receipt"][
        "ready_to_import"
    ] is True
    assert launch_report["predicted_dsg_detector_intake_plan"]["artifact_contract_receipt"][
        "ready_to_build"
    ] is True
    assert launch_report["real_controls_prediction_intake_plan"][
        "prediction_receipt_bundle"
    ] == {
        "digest_valid": False,
        "manifest_matches": False,
        "manifest_path": str(offline_manifest_path),
        "path": str(root / "offline-control-prediction-receipt-bundle.json"),
        "ready_to_import": False,
        "receipt_bundle_digest": None,
        "status": "missing",
        "summary": None,
        "validation_valid": False,
    }
    assert launch_report["predicted_dsg_detector_intake_plan"][
        "detector_receipt_bundle"
    ] == {
        "digest_valid": False,
        "manifest_matches": False,
        "manifest_path": str(predicted_manifest_path),
        "path": str(root / "predicted-dsg-detector-receipt-bundle.json"),
        "ready_to_build": False,
        "receipt_bundle_digest": None,
        "status": "missing",
        "summary": None,
        "validation_valid": False,
    }
    assert launch_report["ready_to_run"] is False
    assert launch_report["primary_evidence_receipt_gate"] == {
        "track": "primary_evidence",
        "ready": False,
        "blocked_track_count": 2,
        "blocked_tracks": [
            {
                "blocking_roles": ["offline_control_source_input"],
                "intake_plan_key": "real_controls_prediction_intake_plan",
                "receipt_status": "missing",
                "track": "real_controls",
            },
            {
                "blocking_roles": ["detector_jsonl"],
                "intake_plan_key": "predicted_dsg_detector_intake_plan",
                "receipt_status": "missing",
                "track": "predicted_dsg",
            },
        ],
        "ready_track_count": 1,
        "ready_tracks": ["real_data"],
        "track_order": ["real_data", "real_controls", "predicted_dsg"],
    }
    launch_report_path = root / "real-experiment-external-artifact-launch-report.json"
    lab.save_real_experiment_external_artifact_launch_report(
        launch_report,
        launch_report_path,
    )
    primary_status = lab.real_experiment_primary_evidence_status(
        launch_report,
        launch_report_path=launch_report_path,
    )
    assert primary_status["schema_version"] == (
        "dsg-spatialqa-lab.real-experiment-primary-evidence-status.v1"
    )
    assert primary_status["action"] == "real_experiment_primary_evidence_status"
    assert primary_status["summary"] == {
        "blocked_track_count": 2,
        "preflight_ready_to_run": True,
        "ready": False,
        "ready_to_run": False,
        "ready_track_count": 1,
        "track_count": 3,
    }
    assert [row["track"] for row in primary_status["tracks"]] == [
        "real_data",
        "real_controls",
        "predicted_dsg",
    ]
    assert primary_status["tracks"][0]["ready"] is True
    assert primary_status["tracks"][1]["receipt_status"] == "missing"
    assert primary_status["tracks"][1]["next_command_key"] == (
        "prediction_request_bundle_command"
    )
    assert primary_status["next_blocked_track"] == {
        "blocking_roles": ["offline_control_source_input"],
        "next_command": primary_status["tracks"][1]["next_command"],
        "next_command_key": "prediction_request_bundle_command",
        "receipt_status": "missing",
        "track": "real_controls",
    }
    assert primary_status["status_digest"] == (
        lab.real_experiment_primary_evidence_status_digest(primary_status)
    )

    _write_ready_offline_control_receipt_bundle(root, offline_manifest_path)
    _write_ready_predicted_dsg_receipt_bundle(root, predicted_manifest_path)
    launch_report = lab.real_experiment_external_artifact_launch_report(
        lab.load_real_experiment_external_artifact_contracts(contracts_path),
        contracts_path=contracts_path,
    )
    assert (
        lab.save_real_experiment_external_artifact_launch_report(
            launch_report,
            launch_report_path,
        )
        == launch_report_path
    )
    execution_packet_without_acceptance = lab.real_experiment_execution_packet(
        launch_report,
        launch_report_path=launch_report_path,
    )
    assert launch_report["ready_to_run"] is True
    assert execution_packet_without_acceptance["ready_to_execute"] is False
    assert execution_packet_without_acceptance["execution_blocked"] is True
    assert execution_packet_without_acceptance["execution_commands"] == []
    assert execution_packet_without_acceptance["readiness"][
        "primary_evidence_acceptance_report_present"
    ] is False
    assert execution_packet_without_acceptance["readiness"][
        "primary_evidence_acceptance_report_ready"
    ] is False
    assert execution_packet_without_acceptance["primary_evidence_acceptance"] == {
        "acceptance_digest": None,
        "all_tracks_accepted": False,
        "matches_current": False,
        "path": str(root / "real-experiment-primary-evidence-acceptance-report.json"),
        "present": False,
        "ready_for_launch_refresh": False,
        "summary": None,
        "valid": False,
    }
    assert execution_packet_without_acceptance["blocker_summary"][
        "blocked_track_count"
    ] == 0

    execution_packet_path = root / "real-experiment-execution-packet.json"

    assert launch_report["real_controls_prediction_intake_plan"][
        "prediction_receipt_bundle"
    ]["status"] == "ready"
    assert launch_report["real_controls_prediction_intake_plan"][
        "prediction_receipt_bundle"
    ]["digest_valid"] is True
    assert launch_report["real_controls_prediction_intake_plan"][
        "prediction_receipt_bundle"
    ]["validation_valid"] is True
    assert launch_report["real_controls_prediction_intake_plan"][
        "prediction_receipt_bundle"
    ]["manifest_matches"] is True
    assert launch_report["predicted_dsg_detector_intake_plan"][
        "detector_receipt_bundle"
    ]["status"] == "ready"
    assert launch_report["predicted_dsg_detector_intake_plan"][
        "detector_receipt_bundle"
    ]["digest_valid"] is True
    assert launch_report["predicted_dsg_detector_intake_plan"][
        "detector_receipt_bundle"
    ]["validation_valid"] is True
    assert launch_report["predicted_dsg_detector_intake_plan"][
        "detector_receipt_bundle"
    ]["manifest_matches"] is True
    assert launch_report["primary_evidence_receipt_gate"]["ready"] is True
    primary_status = lab.real_experiment_primary_evidence_status(
        launch_report,
        launch_report_path=launch_report_path,
    )
    primary_status_path = root / "real-experiment-primary-evidence-status.json"
    assert (
        lab.save_real_experiment_primary_evidence_status(
            primary_status,
            primary_status_path,
        )
        == primary_status_path
    )
    loaded_primary_status = lab.load_real_experiment_primary_evidence_status(
        primary_status_path
    )
    assert loaded_primary_status == primary_status
    primary_status_validation = lab.validate_real_experiment_primary_evidence_status(
        loaded_primary_status
    )
    assert primary_status_validation["action"] == (
        "validate_real_experiment_primary_evidence_status"
    )
    assert primary_status_validation["valid"] is True
    primary_status_comparison = lab.compare_real_experiment_primary_evidence_status(
        loaded_primary_status
    )
    assert primary_status_comparison["action"] == (
        "compare_real_experiment_primary_evidence_status"
    )
    assert primary_status_comparison["matches"] is True
    request_package = lab.real_experiment_primary_evidence_request_package(
        launch_report,
        launch_report_path=launch_report_path,
    )
    assert request_package["schema_version"] == (
        "dsg-spatialqa-lab.real-experiment-primary-evidence-request-package.v1"
    )
    assert request_package["action"] == (
        "real_experiment_primary_evidence_request_package"
    )
    assert request_package["summary"] == {
        "all_request_tracks_ready": True,
        "blocked_request_track_count": 0,
        "ready_request_track_count": 3,
        "track_count": 3,
    }
    assert [row["track"] for row in request_package["tracks"]] == [
        "real_data",
        "real_controls",
        "predicted_dsg",
    ]
    assert [
        row["request_bundle"]["action"] for row in request_package["tracks"]
    ] == [
        "real_collection_request_bundle",
        "offline_control_prediction_request_bundle",
        "predicted_dsg_detector_request_bundle",
    ]
    assert all(row["status"] == "ready" for row in request_package["tracks"])
    assert [
        row["request_bundle_validation"]["valid"] for row in request_package["tracks"]
    ] == [True, True, True]
    assert request_package["package_digest"] == (
        lab.real_experiment_primary_evidence_request_package_digest(
            request_package
        )
    )
    request_package_path = root / "real-experiment-primary-evidence-request-package.json"
    assert (
        lab.save_real_experiment_primary_evidence_request_package(
            request_package,
            request_package_path,
        )
        == request_package_path
    )
    loaded_request_package = lab.load_real_experiment_primary_evidence_request_package(
        request_package_path
    )
    assert loaded_request_package == request_package
    request_package_validation = (
        lab.validate_real_experiment_primary_evidence_request_package(
            loaded_request_package
        )
    )
    assert request_package_validation["action"] == (
        "validate_real_experiment_primary_evidence_request_package"
    )
    assert request_package_validation["valid"] is True
    assert not Path(request_package["tracks"][0]["request_bundle_path"]).exists()
    assert not Path(request_package["tracks"][1]["request_bundle_path"]).exists()
    assert not Path(request_package["tracks"][2]["request_bundle_path"]).exists()
    request_bundle_write = lab.write_real_experiment_primary_evidence_request_bundles(
        loaded_request_package
    )
    assert request_bundle_write["action"] == (
        "write_real_experiment_primary_evidence_request_bundles"
    )
    assert request_bundle_write["summary"] == {
        "all_request_bundles_written": True,
        "blocked_request_track_count": 0,
        "ready_request_track_count": 3,
        "skipped_request_track_count": 0,
        "track_count": 3,
        "written_request_bundle_count": 3,
    }
    assert [row["status"] for row in request_bundle_write["tracks"]] == [
        "written",
        "written",
        "written",
    ]
    assert lab.load_real_collection_request_bundle(
        request_package["tracks"][0]["request_bundle_path"]
    ) == request_package["tracks"][0]["request_bundle"]
    assert lab.load_offline_control_prediction_request_bundle(
        request_package["tracks"][1]["request_bundle_path"]
    ) == request_package["tracks"][1]["request_bundle"]
    assert lab.load_predicted_dsg_detector_request_bundle(
        request_package["tracks"][2]["request_bundle_path"]
    ) == request_package["tracks"][2]["request_bundle"]
    tampered_request_package = json.loads(json.dumps(request_package))
    real_data_row = tampered_request_package["tracks"][0]
    real_data_bundle = real_data_row["request_bundle"]
    real_data_bundle["commands"] = {
        **real_data_bundle["commands"],
        "collection_report": "python scripts/check_real_collection.py --report drifted.json",
    }
    real_data_bundle["request_bundle_digest"] = (
        lab.real_collection_request_bundle_digest(real_data_bundle)
    )
    real_data_row["request_bundle_digest"] = real_data_bundle["request_bundle_digest"]
    real_data_row["request_bundle_validation"] = {
        **real_data_row["request_bundle_validation"],
        "request_bundle_digest": real_data_bundle["request_bundle_digest"],
    }
    tampered_request_package["package_digest"] = (
        lab.real_experiment_primary_evidence_request_package_digest(
            tampered_request_package
        )
    )
    tampered_request_package_validation = (
        lab.validate_real_experiment_primary_evidence_request_package(
            tampered_request_package
        )
    )
    assert tampered_request_package_validation["valid"] is False
    assert {
        check["name"]: check["passed"]
        for check in tampered_request_package_validation["checks"]
    }["request_bundle_validations"] is False
    request_package_comparison = (
        lab.compare_real_experiment_primary_evidence_request_package(
            loaded_request_package
        )
    )
    assert request_package_comparison["action"] == (
        "compare_real_experiment_primary_evidence_request_package"
    )
    assert request_package_comparison["matches"] is True
    return_checklist = lab.real_experiment_primary_evidence_return_checklist(
        request_package,
        request_package_path=request_package_path,
    )
    assert return_checklist["schema_version"] == (
        "dsg-spatialqa-lab.real-experiment-primary-evidence-return-checklist.v1"
    )
    assert return_checklist["action"] == (
        "real_experiment_primary_evidence_return_checklist"
    )
    assert return_checklist["summary"] == {
        "actionable_return_track_count": 3,
        "all_return_tracks_actionable": True,
        "blocked_return_track_count": 0,
        "track_count": 3,
    }
    assert [step["track"] for step in return_checklist["steps"]] == [
        "real_data",
        "real_controls",
        "predicted_dsg",
    ]
    assert [step["status"] for step in return_checklist["steps"]] == [
        "actionable",
        "actionable",
        "actionable",
    ]
    assert [
        step["next_return_command_key"] for step in return_checklist["steps"]
    ] == [
        "collection_report",
        "prediction_receipt_bundle",
        "detector_receipt_bundle",
    ]
    assert "scripts/check_real_collection.py" in (
        return_checklist["steps"][0]["next_return_command"]
    )
    assert "scripts/run_offline_controls.py" in (
        return_checklist["steps"][1]["next_return_command"]
    )
    assert "scripts/run_predicted_dsg.py" in (
        return_checklist["steps"][2]["next_return_command"]
    )
    assert return_checklist["checklist_digest"] == (
        lab.real_experiment_primary_evidence_return_checklist_digest(
            return_checklist
        )
    )
    return_checklist_path = root / "real-experiment-primary-evidence-return-checklist.json"
    assert (
        lab.save_real_experiment_primary_evidence_return_checklist(
            return_checklist,
            return_checklist_path,
        )
        == return_checklist_path
    )
    loaded_return_checklist = (
        lab.load_real_experiment_primary_evidence_return_checklist(
            return_checklist_path
        )
    )
    assert loaded_return_checklist == return_checklist
    return_checklist_validation = (
        lab.validate_real_experiment_primary_evidence_return_checklist(
            loaded_return_checklist
        )
    )
    assert return_checklist_validation["action"] == (
        "validate_real_experiment_primary_evidence_return_checklist"
    )
    assert return_checklist_validation["valid"] is True
    return_checklist_comparison = (
        lab.compare_real_experiment_primary_evidence_return_checklist(
            loaded_return_checklist
        )
    )
    assert return_checklist_comparison["action"] == (
        "compare_real_experiment_primary_evidence_return_checklist"
    )
    assert return_checklist_comparison["matches"] is True
    return_progress = lab.real_experiment_primary_evidence_return_progress_report(
        return_checklist,
        return_checklist_path=return_checklist_path,
    )
    assert return_progress["schema_version"] == (
        "dsg-spatialqa-lab.real-experiment-primary-evidence-return-progress-report.v1"
    )
    assert return_progress["action"] == (
        "real_experiment_primary_evidence_return_progress_report"
    )
    assert return_progress["summary"] == {
        "actionable_return_track_count": 3,
        "all_return_artifacts_present": True,
        "blocked_return_track_count": 0,
        "complete_return_track_count": 3,
        "missing_return_artifact_count": 0,
        "present_return_artifact_count": 9,
        "ready_for_launch_refresh": True,
        "return_artifact_count": 9,
        "track_count": 3,
    }
    assert return_progress["next_missing_return"] is None
    assert [row["return_status"] for row in return_progress["tracks"]] == [
        "complete",
        "complete",
        "complete",
    ]
    assert return_progress["report_digest"] == (
        lab.real_experiment_primary_evidence_return_progress_report_digest(
            return_progress
        )
    )
    return_progress_path = root / "real-experiment-primary-evidence-return-progress.json"
    assert (
        lab.save_real_experiment_primary_evidence_return_progress_report(
            return_progress,
            return_progress_path,
        )
        == return_progress_path
    )
    loaded_return_progress = (
        lab.load_real_experiment_primary_evidence_return_progress_report(
            return_progress_path
        )
    )
    assert loaded_return_progress == return_progress
    return_progress_validation = (
        lab.validate_real_experiment_primary_evidence_return_progress_report(
            loaded_return_progress
        )
    )
    assert return_progress_validation["action"] == (
        "validate_real_experiment_primary_evidence_return_progress_report"
    )
    assert return_progress_validation["valid"] is True
    return_progress_comparison = (
        lab.compare_real_experiment_primary_evidence_return_progress_report(
            loaded_return_progress
        )
    )
    assert return_progress_comparison["action"] == (
        "compare_real_experiment_primary_evidence_return_progress_report"
    )
    assert return_progress_comparison["matches"] is True
    assert hasattr(lab, "real_experiment_primary_evidence_acceptance_report")
    acceptance_report = lab.real_experiment_primary_evidence_acceptance_report(
        loaded_return_progress,
        return_progress_path=return_progress_path,
    )
    assert acceptance_report["schema_version"] == (
        "dsg-spatialqa-lab.real-experiment-primary-evidence-acceptance-report.v1"
    )
    assert acceptance_report["action"] == (
        "real_experiment_primary_evidence_acceptance_report"
    )
    assert acceptance_report["summary"] == {
        "accepted_track_count": 3,
        "all_tracks_accepted": True,
        "blocked_track_count": 0,
        "invalid_track_count": 0,
        "missing_track_count": 0,
        "not_ready_track_count": 0,
        "ready_for_launch_refresh": True,
        "track_count": 3,
    }
    assert [row["track"] for row in acceptance_report["tracks"]] == [
        "real_data",
        "real_controls",
        "predicted_dsg",
    ]
    assert [row["status"] for row in acceptance_report["tracks"]] == [
        "accepted",
        "accepted",
        "accepted",
    ]
    assert [
        row["receipt_kind"] for row in acceptance_report["tracks"]
    ] == [
        "real_collection_report",
        "offline_control_prediction_receipt_bundle",
        "predicted_dsg_detector_receipt_bundle",
    ]
    assert [
        row["receipt_digest"] for row in acceptance_report["tracks"]
    ] == [
        launch_report["real_data_collection_intake_plan"][
            "collection_report_receipt"
        ]["report_digest"],
        launch_report["real_controls_prediction_intake_plan"][
            "prediction_receipt_bundle"
        ]["receipt_bundle_digest"],
        launch_report["predicted_dsg_detector_intake_plan"][
            "detector_receipt_bundle"
        ]["receipt_bundle_digest"],
    ]
    assert [
        row["validation_valid"] for row in acceptance_report["tracks"]
    ] == [True, True, True]
    assert [
        row["manifest_matches"] for row in acceptance_report["tracks"]
    ] == [None, True, True]
    assert acceptance_report["next_unaccepted_track"] is None
    assert acceptance_report["acceptance_digest"] == (
        lab.real_experiment_primary_evidence_acceptance_report_digest(
            acceptance_report
        )
    )
    acceptance_report_path = (
        root / "real-experiment-primary-evidence-acceptance-report.json"
    )
    assert (
        lab.save_real_experiment_primary_evidence_acceptance_report(
            acceptance_report,
            acceptance_report_path,
        )
        == acceptance_report_path
    )
    loaded_acceptance_report = (
        lab.load_real_experiment_primary_evidence_acceptance_report(
            acceptance_report_path
        )
    )
    assert loaded_acceptance_report == acceptance_report
    acceptance_validation = (
        lab.validate_real_experiment_primary_evidence_acceptance_report(
            loaded_acceptance_report
        )
    )
    assert acceptance_validation["action"] == (
        "validate_real_experiment_primary_evidence_acceptance_report"
    )
    assert acceptance_validation["valid"] is True
    acceptance_comparison = (
        lab.compare_real_experiment_primary_evidence_acceptance_report(
            loaded_acceptance_report
        )
    )
    assert acceptance_comparison["action"] == (
        "compare_real_experiment_primary_evidence_acceptance_report"
    )
    assert acceptance_comparison["matches"] is True
    execution_packet = lab.real_experiment_execution_packet(
        launch_report,
        launch_report_path=launch_report_path,
    )
    assert (
        lab.save_real_experiment_execution_packet(
            execution_packet,
            execution_packet_path,
        )
        == execution_packet_path
    )
    loaded_execution_packet = lab.load_real_experiment_execution_packet(
        execution_packet_path
    )
    assert loaded_execution_packet == execution_packet
    packet_validation = lab.validate_real_experiment_execution_packet(
        loaded_execution_packet
    )
    assert packet_validation["valid"] is True
    assert packet_validation["packet_digest"] == execution_packet["packet_digest"]
    packet_comparison = lab.compare_real_experiment_execution_packet(
        loaded_execution_packet
    )
    assert packet_comparison["matches"] is True
    assert packet_comparison["saved_digest"] == execution_packet["packet_digest"]
    assert packet_comparison["current_digest"] == execution_packet["packet_digest"]
    assert execution_packet["action"] == "real_experiment_execution_packet"
    assert execution_packet["ready_to_execute"] is True
    assert execution_packet["execution_blocked"] is False
    assert execution_packet["launch_report_path"] == str(launch_report_path)
    assert execution_packet["launch_report_digest"] == launch_report["report_digest"]
    assert execution_packet["readiness"][
        "primary_evidence_acceptance_report_ready"
    ] is True
    assert execution_packet["primary_evidence_acceptance"] == {
        "acceptance_digest": acceptance_report["acceptance_digest"],
        "all_tracks_accepted": True,
        "matches_current": True,
        "path": str(acceptance_report_path),
        "present": True,
        "ready_for_launch_refresh": True,
        "summary": acceptance_report["summary"],
        "valid": True,
    }
    assert [command["key"] for command in execution_packet["execution_commands"]] == [
        "preflight_run_manifest",
        "run_real_experiment",
    ]
    assert execution_packet["execution_commands"][1]["command"] == (
        "python scripts/run_real_experiment.py --run-manifest "
        f"{launch_report['run_manifest_path']} --run-ledger-output "
        f"{root / 'outputs/real-experiment-run-ledger.json'} "
        f"--approved-execution-packet {execution_packet_path}"
    )
    assert [command["key"] for command in execution_packet["audit_commands"]] == [
        "validate_launch_report",
        "compare_launch_report",
        "refresh_launch_report",
        "validate_primary_evidence_acceptance_report",
        "compare_primary_evidence_acceptance_report",
    ]
    tampered_execution_packet = json.loads(json.dumps(execution_packet))
    tampered_execution_packet["audit_commands"] = [
        command
        for command in tampered_execution_packet["audit_commands"]
        if not command["key"].endswith("primary_evidence_acceptance_report")
    ]
    tampered_execution_packet["packet_digest"] = (
        lab.real_experiment_execution_packet_digest(tampered_execution_packet)
    )
    tampered_packet_validation = lab.validate_real_experiment_execution_packet(
        tampered_execution_packet
    )
    assert tampered_packet_validation["valid"] is False
    assert {
        check["name"]: check["passed"]
        for check in tampered_packet_validation["checks"]
    }["audit_commands_include_primary_evidence_acceptance"] is False
    assert primary_status["summary"] == {
        "blocked_track_count": 0,
        "preflight_ready_to_run": True,
        "ready": True,
        "ready_to_run": True,
        "ready_track_count": 3,
        "track_count": 3,
    }
    assert primary_status["next_blocked_track"] is None


def test_run_real_experiment_cli_accepts_preflight_run_manifest(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_real_script()
    main = getattr(module, "main")
    inputs = _ready_package_inputs(tmp_path)
    run_manifest_path = _real_experiment_run_manifest(tmp_path, inputs)

    assert main(["--preflight-run-manifest", str(run_manifest_path)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["action"] == "real_experiment_run_manifest_preflight"
    assert output["run_manifest_path"] == str(run_manifest_path)
    assert output["ready_to_run"] is True


def test_run_real_experiment_cli_writes_handoff_manifests(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_real_script()
    main = getattr(module, "main")
    root = tmp_path / "cli-handoff"

    assert main(
        [
            "--write-handoff-manifests",
            "--handoff-root",
            str(root),
            "--dataset-name",
            "habitat_real_smoke",
            "--episode",
            "inputs/episodes/FloorPlan1.jsonl",
            "--real-collection-source-kind",
            "habitat",
            "--real-collection-min-frame-count",
            "12",
            "--min-episode-count",
            "1",
            "--min-qa-count",
            "8",
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["action"] == "write_real_experiment_handoff_manifests"
    run_manifest_path = Path(
        output["manifest_paths"]["real_experiment_run_manifest"]
    )
    preflight_report_path = Path(
        output["manifest_paths"]["real_experiment_preflight_report"]
    )
    checklist_path = Path(
        output["manifest_paths"]["real_experiment_artifact_checklist"]
    )
    contracts_path = Path(
        output["manifest_paths"]["real_experiment_external_artifact_contracts"]
    )
    primary_status_path = Path(
        output["manifest_paths"]["real_experiment_primary_evidence_status"]
    )
    request_package_path = Path(
        output["manifest_paths"]["real_experiment_primary_evidence_request_package"]
    )
    return_checklist_path = Path(
        output["manifest_paths"]["real_experiment_primary_evidence_return_checklist"]
    )
    return_progress_path = Path(
        output["manifest_paths"]["real_experiment_primary_evidence_return_progress"]
    )
    acceptance_report_path = Path(
        output["manifest_paths"]["real_experiment_primary_evidence_acceptance_report"]
    )
    operator_checklist_path = Path(
        output["manifest_paths"]["real_experiment_operator_checklist"]
    )
    assert run_manifest_path.exists()
    assert preflight_report_path.exists()
    assert checklist_path.exists()
    assert contracts_path.exists()
    assert operator_checklist_path.exists()
    assert output["preflight_ready_to_run"] is False
    assert output["preflight_summary"]["missing_input_count"] == 13
    assert output["preflight_summary"]["planned_output_count"] == 25
    assert output["artifact_checklist_summary"]["missing_input_artifact_count"] == 13
    assert output["external_artifact_contracts_summary"]["track_count"] == 5
    assert output["external_artifact_contracts_summary"][
        "real_control_source_count"
    ] == 4
    assert output["artifact_track_summary"]["real_data"]["missing_input_artifact_count"] == 2
    assert output["artifact_track_summary"]["real_controls"]["missing_input_artifact_count"] == 6
    assert output["artifact_track_summary"]["predicted_dsg"]["planned_output_artifact_count"] == 6
    assert output["operator_checklist_summary"] == {
        "phase_count": 11,
        "ready_to_run": False,
        "step_count": 44,
        "track_count": 6,
    }
    assert primary_status_path == (
        root / "real-experiment-primary-evidence-status.json"
    )
    assert request_package_path == (
        root / "real-experiment-primary-evidence-request-package.json"
    )
    assert return_checklist_path == (
        root / "real-experiment-primary-evidence-return-checklist.json"
    )
    assert return_progress_path == (
        root / "real-experiment-primary-evidence-return-progress.json"
    )
    assert acceptance_report_path == (
        root / "real-experiment-primary-evidence-acceptance-report.json"
    )
    run_manifest_payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    assert run_manifest_payload["real_collection_source_kind"] == "habitat"
    assert run_manifest_payload["min_frame_count"] == 12

    assert main(["--preflight-run-manifest", str(run_manifest_path)]) == 1
    preflight = json.loads(capsys.readouterr().out)
    saved_preflight = json.loads(preflight_report_path.read_text(encoding="utf-8"))
    assert saved_preflight == preflight
    checklist = json.loads(checklist_path.read_text(encoding="utf-8"))
    contracts = json.loads(contracts_path.read_text(encoding="utf-8"))
    operator_checklist = json.loads(
        operator_checklist_path.read_text(encoding="utf-8")
    )
    assert checklist["summary"] == output["artifact_checklist_summary"]
    assert contracts["summary"] == output["external_artifact_contracts_summary"]
    assert operator_checklist["summary"] == output["operator_checklist_summary"]
    assert operator_checklist["operator_checklist_digest"] == (
        lab.real_experiment_operator_checklist_digest(operator_checklist)
    )
    assert operator_checklist["steps"][0]["key"] == (
        "validate_external_artifact_contracts"
    )
    assert operator_checklist["steps"][1]["key"] == (
        "compare_external_artifact_contracts"
    )
    assert [step["key"] for step in operator_checklist["steps"][2:15]] == [
        "external_artifact_launch_report",
        "validate_external_artifact_launch_report",
        "compare_external_artifact_launch_report",
        "primary_evidence_status",
        "validate_primary_evidence_status",
        "compare_primary_evidence_status",
        "primary_evidence_request_package",
        "validate_primary_evidence_request_package",
        "compare_primary_evidence_request_package",
        "write_primary_evidence_request_bundles",
        "primary_evidence_return_checklist",
        "validate_primary_evidence_return_checklist",
        "compare_primary_evidence_return_checklist",
    ]
    assert [step["key"] for step in operator_checklist["steps"][18:24]] == [
        "primary_evidence_return_progress_report",
        "validate_primary_evidence_return_progress_report",
        "compare_primary_evidence_return_progress_report",
        "primary_evidence_acceptance_report",
        "validate_primary_evidence_acceptance_report",
        "compare_primary_evidence_acceptance_report",
    ]
    assert [step["key"] for step in operator_checklist["steps"][29:38]] == [
        "compare_smoke_run_checklist",
        "smoke_run_runbook",
        "validate_smoke_run_runbook",
        "compare_smoke_run_runbook",
        "validate_run_ledger",
        "compare_run_ledger",
        "execution_receipt",
        "validate_execution_receipt",
        "compare_execution_receipt",
    ]
    assert operator_checklist["steps"][-1]["key"] == "compare_claim_readiness"
    assert contracts["tracks"]["real_data"]["source_kind"] == "habitat"
    assert contracts["tracks"]["real_data"]["min_frame_count"] == 12
    assert preflight["action"] == "real_experiment_run_manifest_preflight"
    assert preflight["ready_to_run"] is False
    assert preflight["summary"]["missing_requirement_count"] == 0

    assert main(["--validate-external-artifact-contracts", str(contracts_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_real_experiment_external_artifact_contracts"
    assert validation["valid"] is True

    assert main(["--compare-external-artifact-contracts", str(contracts_path)]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["action"] == "compare_real_experiment_external_artifact_contracts"
    assert comparison["matches"] is True

    assert main(["--validate-operator-checklist", str(operator_checklist_path)]) == 0
    operator_validation = json.loads(capsys.readouterr().out)
    assert operator_validation["action"] == (
        "validate_real_experiment_operator_checklist"
    )
    assert operator_validation["valid"] is True

    assert main(["--compare-operator-checklist", str(operator_checklist_path)]) == 0
    operator_comparison = json.loads(capsys.readouterr().out)
    assert operator_comparison["action"] == (
        "compare_real_experiment_operator_checklist"
    )
    assert operator_comparison["matches"] is True

    operator_progress_path = root / "real-experiment-operator-progress-report.json"
    assert (
        main(
            [
                "--operator-progress-report",
                str(operator_checklist_path),
                "--operator-progress-output",
                str(operator_progress_path),
            ]
        )
        == 1
    )
    operator_progress = json.loads(capsys.readouterr().out)
    assert operator_progress_path.exists()
    assert json.loads(operator_progress_path.read_text(encoding="utf-8")) == (
        operator_progress
    )
    assert operator_progress["action"] == "real_experiment_operator_progress_report"
    assert operator_progress["summary"]["present_target_step_count"] == 2
    assert operator_progress["summary"]["missing_target_step_count"] == 42
    assert operator_progress["next_missing_step"]["key"] == (
        "external_artifact_launch_report"
    )

    assert (
        main(["--validate-operator-progress-report", str(operator_progress_path)])
        == 0
    )
    operator_progress_validation = json.loads(capsys.readouterr().out)
    assert operator_progress_validation["action"] == (
        "validate_real_experiment_operator_progress_report"
    )
    assert operator_progress_validation["valid"] is True

    assert (
        main(["--compare-operator-progress-report", str(operator_progress_path)])
        == 0
    )
    operator_progress_comparison = json.loads(capsys.readouterr().out)
    assert operator_progress_comparison["action"] == (
        "compare_real_experiment_operator_progress_report"
    )
    assert operator_progress_comparison["matches"] is True

    launch_report_path = root / "real-experiment-external-artifact-launch-report.json"
    assert (
        main(
            [
                "--external-artifact-launch-report",
                str(contracts_path),
                "--launch-report-output",
                str(launch_report_path),
            ]
        )
        == 1
    )
    launch_report = json.loads(capsys.readouterr().out)
    assert launch_report_path.exists()
    assert json.loads(launch_report_path.read_text(encoding="utf-8")) == launch_report
    assert launch_report["action"] == (
        "real_experiment_external_artifact_launch_report"
    )
    assert launch_report["ready_to_run"] is False
    assert launch_report["summary"]["blocked_track_count"] == 4
    assert launch_report["tracks"]["real_controls"]["blocking_roles"] == [
        "candidate_prediction",
        "gold_qa",
        "offline_control_source_input",
    ]
    assert set(launch_report["actionable_blockers"]) == {
        "predicted_dsg",
        "real_controls",
        "real_data",
        "review_artifacts",
    }
    assert launch_report["actionable_blockers"]["real_controls"][
        "child_launch_gate"
    ]["track"] == "real_controls"
    assert launch_report["external_artifact_intake_plan"]["blocked_track_count"] == 4
    assert [
        step["track"] for step in launch_report["external_artifact_intake_plan"]["steps"]
    ] == [
        "real_data",
        "real_controls",
        "predicted_dsg",
        "review_artifacts",
    ]
    assert launch_report["external_artifact_intake_plan"]["steps"][0][
        "recommended_command_keys"
    ] == [
        "collection_report_command",
        "request_bundle_command",
        "validate_report_command",
        "compare_report_command",
    ]
    assert "--source-kind habitat" in (
        launch_report["child_launch_gates"]["real_data"][
            "collection_report_command"
        ]
    )
    assert "--min-frame-count 12" in (
        launch_report["child_launch_gates"]["real_data"][
            "collection_report_command"
        ]
    )
    assert launch_report["real_data_collection_intake_plan"]["source_kind"] == "habitat"
    assert launch_report["real_data_collection_intake_plan"]["thresholds"][
        "min_frame_count"
    ] == 12
    assert [
        step["track"] for step in launch_report["primary_evidence_intake_plan"]["steps"]
    ] == ["real_data", "real_controls", "predicted_dsg"]
    assert launch_report["primary_evidence_intake_plan"]["blocked_track_count"] == 3

    assert (
        main(["--validate-external-artifact-launch-report", str(launch_report_path)])
        == 0
    )
    launch_validation = json.loads(capsys.readouterr().out)
    assert launch_validation["action"] == (
        "validate_real_experiment_external_artifact_launch_report"
    )
    assert launch_validation["valid"] is True

    assert (
        main(["--compare-external-artifact-launch-report", str(launch_report_path)])
        == 0
    )
    launch_comparison = json.loads(capsys.readouterr().out)
    assert launch_comparison["action"] == (
        "compare_real_experiment_external_artifact_launch_report"
    )
    assert launch_comparison["matches"] is True

    primary_status_path = root / "real-experiment-primary-evidence-status.json"
    assert (
        main(
            [
                "--primary-evidence-status",
                str(launch_report_path),
                "--primary-evidence-status-output",
                str(primary_status_path),
            ]
        )
        == 1
    )
    primary_status = json.loads(capsys.readouterr().out)
    assert primary_status_path.exists()
    assert json.loads(primary_status_path.read_text(encoding="utf-8")) == (
        primary_status
    )
    assert primary_status["action"] == "real_experiment_primary_evidence_status"
    assert primary_status["summary"] == {
        "blocked_track_count": 3,
        "preflight_ready_to_run": False,
        "ready": False,
        "ready_to_run": False,
        "ready_track_count": 0,
        "track_count": 3,
    }
    assert [row["track"] for row in primary_status["tracks"]] == [
        "real_data",
        "real_controls",
        "predicted_dsg",
    ]
    assert primary_status["next_blocked_track"]["track"] == "real_data"
    assert primary_status["next_blocked_track"]["receipt_status"] == "missing"
    assert primary_status["next_blocked_track"]["next_command_key"] == (
        "collection_report_command"
    )
    assert "scripts/check_real_collection.py" in (
        primary_status["next_blocked_track"]["next_command"]
    )
    assert " --report " in primary_status["next_blocked_track"]["next_command"]

    assert main(["--validate-primary-evidence-status", str(primary_status_path)]) == 0
    primary_status_validation = json.loads(capsys.readouterr().out)
    assert primary_status_validation["action"] == (
        "validate_real_experiment_primary_evidence_status"
    )
    assert primary_status_validation["valid"] is True

    assert main(["--compare-primary-evidence-status", str(primary_status_path)]) == 0
    primary_status_comparison = json.loads(capsys.readouterr().out)
    assert primary_status_comparison["action"] == (
        "compare_real_experiment_primary_evidence_status"
    )
    assert primary_status_comparison["matches"] is True

    request_package_path = (
        root / "real-experiment-primary-evidence-request-package.json"
    )
    assert (
        main(
            [
                "--primary-evidence-request-package",
                str(launch_report_path),
                "--primary-evidence-request-package-output",
                str(request_package_path),
            ]
        )
        == 1
    )
    request_package = json.loads(capsys.readouterr().out)
    assert request_package_path.exists()
    assert json.loads(request_package_path.read_text(encoding="utf-8")) == (
        request_package
    )
    assert request_package["action"] == (
        "real_experiment_primary_evidence_request_package"
    )
    assert request_package["summary"] == {
        "all_request_tracks_ready": False,
        "blocked_request_track_count": 1,
        "ready_request_track_count": 2,
        "track_count": 3,
    }
    assert {
        row["track"]: row["status"] for row in request_package["tracks"]
    } == {
        "predicted_dsg": "ready",
        "real_controls": "blocked",
        "real_data": "ready",
    }
    assert request_package["tracks"][0]["request_bundle"]["action"] == (
        "real_collection_request_bundle"
    )
    assert request_package["tracks"][1]["request_bundle"] is None
    assert request_package["tracks"][1]["error_type"] in {
        "FileNotFoundError",
        "SpatialQAError",
    }
    assert request_package["tracks"][2]["request_bundle"]["action"] == (
        "predicted_dsg_detector_request_bundle"
    )

    assert (
        main(["--validate-primary-evidence-request-package", str(request_package_path)])
        == 0
    )
    request_package_validation = json.loads(capsys.readouterr().out)
    assert request_package_validation["action"] == (
        "validate_real_experiment_primary_evidence_request_package"
    )
    assert request_package_validation["valid"] is True

    assert (
        main(["--compare-primary-evidence-request-package", str(request_package_path)])
        == 0
    )
    request_package_comparison = json.loads(capsys.readouterr().out)
    assert request_package_comparison["action"] == (
        "compare_real_experiment_primary_evidence_request_package"
    )
    assert request_package_comparison["matches"] is True

    assert (
        main(["--write-primary-evidence-request-bundles", str(request_package_path)])
        == 1
    )
    request_bundle_write = json.loads(capsys.readouterr().out)
    assert request_bundle_write["action"] == (
        "write_real_experiment_primary_evidence_request_bundles"
    )
    assert request_bundle_write["summary"] == {
        "all_request_bundles_written": False,
        "blocked_request_track_count": 1,
        "ready_request_track_count": 2,
        "skipped_request_track_count": 1,
        "track_count": 3,
        "written_request_bundle_count": 2,
    }
    assert {
        row["track"]: row["status"] for row in request_bundle_write["tracks"]
    } == {
        "predicted_dsg": "written",
        "real_controls": "skipped_blocked",
        "real_data": "written",
    }
    assert lab.load_real_collection_request_bundle(
        request_package["tracks"][0]["request_bundle_path"]
    ) == request_package["tracks"][0]["request_bundle"]
    assert not Path(request_package["tracks"][1]["request_bundle_path"]).exists()
    assert lab.load_predicted_dsg_detector_request_bundle(
        request_package["tracks"][2]["request_bundle_path"]
    ) == request_package["tracks"][2]["request_bundle"]

    return_checklist_path = (
        root / "real-experiment-primary-evidence-return-checklist.json"
    )
    assert (
        main(
            [
                "--primary-evidence-return-checklist",
                str(request_package_path),
                "--primary-evidence-return-checklist-output",
                str(return_checklist_path),
            ]
        )
        == 1
    )
    return_checklist = json.loads(capsys.readouterr().out)
    assert return_checklist_path.exists()
    assert json.loads(return_checklist_path.read_text(encoding="utf-8")) == (
        return_checklist
    )
    assert return_checklist["action"] == (
        "real_experiment_primary_evidence_return_checklist"
    )
    assert return_checklist["summary"] == {
        "actionable_return_track_count": 2,
        "all_return_tracks_actionable": False,
        "blocked_return_track_count": 1,
        "track_count": 3,
    }
    assert {
        step["track"]: step["status"] for step in return_checklist["steps"]
    } == {
        "predicted_dsg": "actionable",
        "real_controls": "blocked",
        "real_data": "actionable",
    }
    assert return_checklist["steps"][0]["next_return_command_key"] == (
        "collection_report"
    )
    assert return_checklist["steps"][1]["next_return_command_key"] == (
        "request_bundle"
    )
    assert return_checklist["steps"][1]["return_commands"] == []
    assert return_checklist["steps"][1]["error_type"] in {
        "FileNotFoundError",
        "SpatialQAError",
    }
    assert return_checklist["steps"][2]["next_return_command_key"] == (
        "detector_receipt_bundle"
    )

    assert (
        main(["--validate-primary-evidence-return-checklist", str(return_checklist_path)])
        == 0
    )
    return_checklist_validation = json.loads(capsys.readouterr().out)
    assert return_checklist_validation["action"] == (
        "validate_real_experiment_primary_evidence_return_checklist"
    )
    assert return_checklist_validation["valid"] is True

    assert (
        main(["--compare-primary-evidence-return-checklist", str(return_checklist_path)])
        == 0
    )
    return_checklist_comparison = json.loads(capsys.readouterr().out)
    assert return_checklist_comparison["action"] == (
        "compare_real_experiment_primary_evidence_return_checklist"
    )
    assert return_checklist_comparison["matches"] is True

    return_progress_path = (
        root / "real-experiment-primary-evidence-return-progress.json"
    )
    assert (
        main(
            [
                "--primary-evidence-return-progress-report",
                str(return_checklist_path),
                "--primary-evidence-return-progress-output",
                str(return_progress_path),
            ]
        )
        == 1
    )
    return_progress = json.loads(capsys.readouterr().out)
    assert return_progress_path.exists()
    assert json.loads(return_progress_path.read_text(encoding="utf-8")) == (
        return_progress
    )
    assert return_progress["action"] == (
        "real_experiment_primary_evidence_return_progress_report"
    )
    assert return_progress["summary"] == {
        "actionable_return_track_count": 2,
        "all_return_artifacts_present": False,
        "blocked_return_track_count": 1,
        "complete_return_track_count": 0,
        "missing_return_artifact_count": 4,
        "present_return_artifact_count": 0,
        "ready_for_launch_refresh": False,
        "return_artifact_count": 4,
        "track_count": 3,
    }
    assert {
        row["track"]: row["return_status"] for row in return_progress["tracks"]
    } == {
        "predicted_dsg": "missing",
        "real_controls": "blocked",
        "real_data": "missing",
    }
    assert return_progress["next_missing_return"]["track"] == "real_data"
    assert return_progress["next_missing_return"]["return_status"] == "missing"
    assert return_progress["next_missing_return"]["missing_return_artifact_path"] == (
        str(root / "inputs/episodes/FloorPlan1.jsonl")
    )
    assert return_progress["next_missing_return"]["next_return_command_key"] == (
        "collection_report"
    )

    assert (
        main(["--validate-primary-evidence-return-progress-report", str(return_progress_path)])
        == 0
    )
    return_progress_validation = json.loads(capsys.readouterr().out)
    assert return_progress_validation["action"] == (
        "validate_real_experiment_primary_evidence_return_progress_report"
    )
    assert return_progress_validation["valid"] is True

    assert (
        main(["--compare-primary-evidence-return-progress-report", str(return_progress_path)])
        == 0
    )
    return_progress_comparison = json.loads(capsys.readouterr().out)
    assert return_progress_comparison["action"] == (
        "compare_real_experiment_primary_evidence_return_progress_report"
    )
    assert return_progress_comparison["matches"] is True

    acceptance_report_path = (
        root / "real-experiment-primary-evidence-acceptance-report.json"
    )
    assert (
        main(
            [
                "--primary-evidence-acceptance-report",
                str(return_progress_path),
                "--primary-evidence-acceptance-output",
                str(acceptance_report_path),
            ]
        )
        == 1
    )
    acceptance_report = json.loads(capsys.readouterr().out)
    assert acceptance_report_path.exists()
    assert json.loads(acceptance_report_path.read_text(encoding="utf-8")) == (
        acceptance_report
    )
    assert acceptance_report["action"] == (
        "real_experiment_primary_evidence_acceptance_report"
    )
    assert acceptance_report["summary"] == {
        "accepted_track_count": 0,
        "all_tracks_accepted": False,
        "blocked_track_count": 1,
        "invalid_track_count": 0,
        "missing_track_count": 2,
        "not_ready_track_count": 0,
        "ready_for_launch_refresh": False,
        "track_count": 3,
    }
    assert {
        row["track"]: row["status"] for row in acceptance_report["tracks"]
    } == {
        "predicted_dsg": "missing",
        "real_controls": "blocked",
        "real_data": "missing",
    }
    assert acceptance_report["next_unaccepted_track"]["track"] == "real_data"
    assert acceptance_report["next_unaccepted_track"]["status"] == "missing"
    assert acceptance_report["next_unaccepted_track"]["next_return_command_key"] == (
        "collection_report"
    )

    assert (
        main(["--validate-primary-evidence-acceptance-report", str(acceptance_report_path)])
        == 0
    )
    acceptance_validation = json.loads(capsys.readouterr().out)
    assert acceptance_validation["action"] == (
        "validate_real_experiment_primary_evidence_acceptance_report"
    )
    assert acceptance_validation["valid"] is True

    assert (
        main(["--compare-primary-evidence-acceptance-report", str(acceptance_report_path)])
        == 0
    )
    acceptance_comparison = json.loads(capsys.readouterr().out)
    assert acceptance_comparison["action"] == (
        "compare_real_experiment_primary_evidence_acceptance_report"
    )
    assert acceptance_comparison["matches"] is True

    execution_packet_path = root / "real-experiment-execution-packet.json"
    assert (
        main(
            [
                "--execution-packet",
                str(launch_report_path),
                "--execution-packet-output",
                str(execution_packet_path),
            ]
        )
        == 1
    )
    execution_packet = json.loads(capsys.readouterr().out)
    assert execution_packet_path.exists()
    assert json.loads(execution_packet_path.read_text(encoding="utf-8")) == (
        execution_packet
    )
    assert execution_packet["action"] == "real_experiment_execution_packet"
    assert execution_packet["ready_to_execute"] is False
    assert execution_packet["execution_blocked"] is True
    assert execution_packet["primary_evidence_acceptance_report_path"] == str(
        acceptance_report_path
    )
    assert execution_packet["primary_evidence_acceptance"]["present"] is True
    assert execution_packet["primary_evidence_acceptance"]["ready_for_launch_refresh"] is (
        False
    )
    assert execution_packet["execution_commands"] == []
    assert [command["key"] for command in execution_packet["audit_commands"]] == [
        "validate_launch_report",
        "compare_launch_report",
        "refresh_launch_report",
        "validate_primary_evidence_acceptance_report",
        "compare_primary_evidence_acceptance_report",
    ]
    assert execution_packet["blocker_summary"]["blocked_track_count"] == 3
    assert execution_packet["blocker_summary"]["ready_tracks"] == []

    assert main(["--validate-execution-packet", str(execution_packet_path)]) == 0
    packet_validation = json.loads(capsys.readouterr().out)
    assert packet_validation["action"] == "validate_real_experiment_execution_packet"
    assert packet_validation["valid"] is True

    assert main(["--compare-execution-packet", str(execution_packet_path)]) == 0
    packet_comparison = json.loads(capsys.readouterr().out)
    assert packet_comparison["action"] == "compare_real_experiment_execution_packet"
    assert packet_comparison["matches"] is True


def test_run_real_experiment_cli_accepts_run_manifest(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_real_script()
    main = getattr(module, "main")
    inputs = _ready_package_inputs(tmp_path)
    run_manifest_path = _real_experiment_run_manifest(tmp_path, inputs)
    manifest = lab.load_real_experiment_run_manifest(run_manifest_path)
    run_ledger_path = Path(manifest["real_experiment_run_ledger_path"])
    execution_packet_path = tmp_path / "run-manifest" / "execution-packet.json"
    _execution_packet_for_run_manifest(run_manifest_path, execution_packet_path)
    execution_receipt_path = tmp_path / "run-manifest" / "execution-receipt.json"
    smoke_checklist_path = tmp_path / "run-manifest" / "smoke-run-checklist.json"
    smoke_runbook_path = tmp_path / "run-manifest" / "smoke-run-runbook.json"
    research_review_path = tmp_path / "run-manifest" / "research-review.json"
    claim_readiness_path = tmp_path / "run-manifest" / "claim-readiness.json"

    assert (
        main(
            [
                "--smoke-run-checklist",
                str(execution_packet_path),
                "--smoke-run-checklist-output",
                str(smoke_checklist_path),
                "--smoke-run-checklist-receipt-output",
                str(execution_receipt_path),
            ]
        )
        == 0
    )
    smoke_checklist = json.loads(capsys.readouterr().out)
    assert smoke_checklist["action"] == "real_experiment_smoke_run_checklist"
    assert smoke_checklist["ready_to_start"] is True
    assert smoke_checklist["execution_receipt_output_path"] == str(
        execution_receipt_path
    )
    assert smoke_checklist_path.exists()
    assert json.loads(smoke_checklist_path.read_text(encoding="utf-8")) == (
        smoke_checklist
    )

    assert main(["--validate-smoke-run-checklist", str(smoke_checklist_path)]) == 0
    smoke_validation = json.loads(capsys.readouterr().out)
    assert smoke_validation["action"] == "validate_real_experiment_smoke_run_checklist"
    assert smoke_validation["valid"] is True

    assert main(["--compare-smoke-run-checklist", str(smoke_checklist_path)]) == 0
    smoke_comparison = json.loads(capsys.readouterr().out)
    assert smoke_comparison["action"] == "compare_real_experiment_smoke_run_checklist"
    assert smoke_comparison["matches"] is True

    assert (
        main(
            [
                "--smoke-run-runbook",
                str(smoke_checklist_path),
                "--smoke-run-runbook-output",
                str(smoke_runbook_path),
            ]
        )
        == 0
    )
    smoke_runbook = json.loads(capsys.readouterr().out)
    assert smoke_runbook["action"] == "real_experiment_smoke_run_runbook"
    assert smoke_runbook["smoke_run_checklist_path"] == str(smoke_checklist_path)
    assert smoke_runbook_path.exists()
    assert json.loads(smoke_runbook_path.read_text(encoding="utf-8")) == (
        smoke_runbook
    )

    assert main(["--validate-smoke-run-runbook", str(smoke_runbook_path)]) == 0
    smoke_runbook_validation = json.loads(capsys.readouterr().out)
    assert smoke_runbook_validation["action"] == (
        "validate_real_experiment_smoke_run_runbook"
    )
    assert smoke_runbook_validation["valid"] is True

    assert main(["--compare-smoke-run-runbook", str(smoke_runbook_path)]) == 0
    smoke_runbook_comparison = json.loads(capsys.readouterr().out)
    assert smoke_runbook_comparison["action"] == (
        "compare_real_experiment_smoke_run_runbook"
    )
    assert smoke_runbook_comparison["matches"] is True

    assert (
        main(
            [
                "--run-manifest",
                str(run_manifest_path),
                "--approved-execution-packet",
                str(execution_packet_path),
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["action"] == "run_real_experiment_manifest"
    assert output["run_manifest_path"] == str(run_manifest_path)
    assert output["ready"] is True
    assert output["execution_approval"]["status"] == "approved"
    assert output["real_experiment_run_ledger_path"] == str(run_ledger_path)
    assert run_ledger_path.exists()
    assert output["offline_control_import"]["ready"] is True
    assert output["predicted_dsg_detector_run"]["ready"] is True
    assert Path(output["record_path"]).exists()

    assert main(["--validate-run-ledger", str(run_ledger_path)]) == 0
    run_ledger_validation = json.loads(capsys.readouterr().out)
    assert run_ledger_validation["action"] == "validate_real_experiment_run_ledger"
    assert run_ledger_validation["valid"] is True

    assert main(["--compare-run-ledger", str(run_ledger_path)]) == 0
    run_ledger_comparison = json.loads(capsys.readouterr().out)
    assert run_ledger_comparison["action"] == "compare_real_experiment_run_ledger"
    assert run_ledger_comparison["matches"] is True

    assert (
        main(
            [
                "--execution-receipt",
                str(execution_packet_path),
                "--execution-receipt-output",
                str(execution_receipt_path),
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["action"] == "real_experiment_execution_receipt"
    assert receipt["ready_to_review"] is True
    assert execution_receipt_path.exists()
    assert json.loads(execution_receipt_path.read_text(encoding="utf-8")) == receipt

    assert main(["--validate-execution-receipt", str(execution_receipt_path)]) == 0
    receipt_validation = json.loads(capsys.readouterr().out)
    assert receipt_validation["action"] == "validate_real_experiment_execution_receipt"
    assert receipt_validation["valid"] is True

    assert main(["--compare-execution-receipt", str(execution_receipt_path)]) == 0
    receipt_comparison = json.loads(capsys.readouterr().out)
    assert receipt_comparison["action"] == "compare_real_experiment_execution_receipt"
    assert receipt_comparison["matches"] is True

    assert (
        main(
            [
                "--research-review",
                str(execution_receipt_path),
                "--research-review-output",
                str(research_review_path),
            ]
        )
        == 0
    )
    research_review = json.loads(capsys.readouterr().out)
    assert research_review["action"] == "real_experiment_research_review"
    assert research_review["ready_for_research_review"] is True
    assert research_review_path.exists()
    assert json.loads(research_review_path.read_text(encoding="utf-8")) == (
        research_review
    )

    assert main(["--validate-research-review", str(research_review_path)]) == 0
    review_validation = json.loads(capsys.readouterr().out)
    assert review_validation["action"] == "validate_real_experiment_research_review"
    assert review_validation["valid"] is True

    assert main(["--compare-research-review", str(research_review_path)]) == 0
    review_comparison = json.loads(capsys.readouterr().out)
    assert review_comparison["action"] == "compare_real_experiment_research_review"
    assert review_comparison["matches"] is True

    assert (
        main(
            [
                "--claim-readiness",
                str(research_review_path),
                "--claim-readiness-output",
                str(claim_readiness_path),
                "--claim-min-episode-count",
                "1",
                "--claim-min-scene-count",
                "1",
                "--claim-min-qa-count",
                "8",
                "--claim-min-dynamic-qa-count",
                "0",
            ]
        )
        == 0
    )
    claim_readiness = json.loads(capsys.readouterr().out)
    assert claim_readiness["action"] == "real_experiment_claim_readiness"
    assert claim_readiness["claim_ready"] is True
    assert claim_readiness["next_handoff_plan"]["required"] is False
    assert claim_readiness["next_handoff_plan"]["commands"] == {}
    assert claim_readiness_path.exists()
    assert json.loads(claim_readiness_path.read_text(encoding="utf-8")) == (
        claim_readiness
    )

    assert main(["--validate-claim-readiness", str(claim_readiness_path)]) == 0
    claim_validation = json.loads(capsys.readouterr().out)
    assert claim_validation["action"] == "validate_real_experiment_claim_readiness"
    assert claim_validation["valid"] is True

    assert main(["--compare-claim-readiness", str(claim_readiness_path)]) == 0
    claim_comparison = json.loads(capsys.readouterr().out)
    assert claim_comparison["action"] == "compare_real_experiment_claim_readiness"
    assert claim_comparison["matches"] is True


def test_run_real_experiment_package_does_not_write_record_when_not_ready(
    tmp_path: Path,
) -> None:
    inputs = _ready_package_inputs(tmp_path)
    root = tmp_path / "not-ready"
    summary_path = root / "experiment-summary.json"
    record_path = root / "experiment-record.json"

    result = lab.run_real_experiment_package(
        dataset_name="ai2thor_real_smoke",
        episode_paths=inputs["episode_paths"],
        output_dir=root / "benchmark",
        manifest_path=root / "benchmark-manifest.json",
        readiness_report_path=root / "real-readiness.json",
        summary_report_path=summary_path,
        record_path=record_path,
        max_qa_per_episode=20,
        tags=("benchmark", "real"),
        qa_eval_delta_report_paths=(inputs["qa_delta_report_path"],),
        active_task_delta_report_paths=(inputs["active_delta_report_path"],),
        dashboard_bundle_paths=(inputs["dashboard_bundle_path"],),
        error_attribution_report_paths=(inputs["error_attribution_report_path"],),
        graph_eval_report_paths=(inputs["graph_eval_report_path"],),
        offline_prediction_import_report_paths=(inputs["offline_import_report_path"],),
        offline_control_matrix_report_paths=(
            inputs["offline_control_matrix_report_path"],
        ),
        offline_control_result_report_paths=(
            inputs["offline_control_result_report_path"],
        ),
        predicted_dsg_evidence_report_paths=(
            inputs["predicted_dsg_evidence_report_path"],
        ),
        predicted_graph_report_paths=(inputs["predicted_graph_report_path"],),
        real_collection_report_paths=(),
        min_episode_count=1,
        min_scene_count=1,
        min_qa_count=8,
        required_control_kinds=("vlm",),
        required_predicted_input_kinds=("observation_sequence",),
    )

    assert result["ready"] is False
    assert result["record_path"] is None
    assert result["summary_report_path"] is None
    assert result["real_package_status"] == "not_ready"
    assert "real_collection_report_present" in result["readiness"]["failed_checks"]
    assert not summary_path.exists()
    assert not record_path.exists()


def test_run_real_experiment_cli_outputs_real_run_record(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_real_script()
    main = getattr(module, "main")
    inputs = _ready_package_inputs(tmp_path)
    root = tmp_path / "cli-run"
    manifest_path = root / "benchmark-manifest.json"
    readiness_path = root / "real-readiness.json"
    summary_path = root / "experiment-summary.json"
    record_path = root / "experiment-record.json"

    assert main(
        [
            "--episode",
            str(inputs["episode_paths"][0]),
            "--dataset-name",
            "ai2thor_real_smoke",
            "--output-dir",
            str(root / "benchmark"),
            "--manifest",
            str(manifest_path),
            "--readiness-report",
            str(readiness_path),
            "--summary-report",
            str(summary_path),
            "--record",
            str(record_path),
            "--max-qa-per-episode",
            "20",
            "--tag",
            "benchmark",
            "--tag",
            "real",
            "--qa-eval-delta-report",
            str(inputs["qa_delta_report_path"]),
            "--active-task-delta-report",
            str(inputs["active_delta_report_path"]),
            "--dashboard-bundle",
            str(inputs["dashboard_bundle_path"]),
            "--error-attribution-report",
            str(inputs["error_attribution_report_path"]),
            "--graph-eval-report",
            str(inputs["graph_eval_report_path"]),
                "--offline-prediction-import-report",
                str(inputs["offline_import_report_path"]),
                "--offline-control-matrix-report",
                str(inputs["offline_control_matrix_report_path"]),
                "--offline-control-result-report",
                str(inputs["offline_control_result_report_path"]),
                "--predicted-dsg-evidence-report",
            str(inputs["predicted_dsg_evidence_report_path"]),
            "--predicted-graph-report",
            str(inputs["predicted_graph_report_path"]),
            "--real-collection-report",
            str(inputs["real_collection_report_path"]),
            "--min-episode-count",
            "1",
            "--min-scene-count",
            "1",
            "--min-qa-count",
            "8",
            "--required-control-kind",
            "vlm",
            "--required-predicted-input-kind",
            "observation_sequence",
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    record = lab.load_experiment_record(record_path)
    assert output["action"] == "run_real_experiment_package"
    assert output["ready"] is True
    assert output["record_digest"] == record["record_digest"]
    assert output["real_package_status"] == "ready"
    assert record["real_readiness_report_path"] == str(readiness_path)


def _ready_package_inputs(tmp_path: Path) -> ReadyPackageInputs:
    episode_path = tmp_path / "source" / "episode.jsonl"
    mock_frames = lab.build_mock_ai2thor_episode(
        lab.AI2ThorAdapterConfig(
            scene_id="FloorPlan1",
            episode_id="ai2thor_real_smoke_001",
            steps=(1, 2),
            actions=("Initialize", "MoveAhead"),
        )
    )
    frames = tuple(
        lab.EpisodeFrame(
            episode_id=frame.episode_id,
            scene_id=frame.scene_id,
            step=frame.step,
            rgb_path=f"real-ai2thor/{frame.episode_id}/{frame.step:06d}.rgb.png",
            depth_path=f"real-ai2thor/{frame.episode_id}/{frame.step:06d}.depth.png",
            segmentation_path=(
                f"real-ai2thor/{frame.episode_id}/{frame.step:06d}.segmentation.png"
            ),
            agent_id=frame.agent_id,
            agent_pose=frame.agent_pose,
            action=frame.action,
            visible_object_ids=frame.visible_object_ids,
            metadata={
                **frame.metadata,
                "collection_kind": "real",
                "source_kind": "ai2thor",
            },
        )
        for frame in mock_frames
    )
    lab.save_episode_sequence(frames, episode_path)
    _write_real_frame_assets(episode_path, frames)
    real_collection_report = lab.real_collection_report(
        dataset_name="ai2thor_real_smoke",
        episode_paths=(episode_path,),
        source_kind="ai2thor",
        min_episode_count=1,
        min_scene_count=1,
        min_frame_count=2,
    )
    real_collection_report_path = tmp_path / "reports" / "real-collection.json"
    lab.save_real_collection_report(real_collection_report, real_collection_report_path)
    seed_manifest = lab.build_benchmark_artifacts(
        dataset_name="seed",
        episode_paths=(episode_path,),
        output_dir=tmp_path / "seed-benchmark",
        max_qa_per_episode=20,
        tags=("benchmark", "real"),
    )
    artifact = seed_manifest["artifacts"][0]
    graph_path = Path(str(artifact["graph_path"]))
    qa_path = Path(str(artifact["qa_path"]))
    graph = lab.load_graph_json(graph_path)
    cases = tuple(lab.load_qa_dataset(qa_path))
    prediction = lab.QAPrediction(
        id=cases[0].id,
        answer=cases[0].answer,
        evidence_nodes=cases[0].required_nodes,
        evidence_edges=cases[0].required_edges,
        confidence=1.0,
    )
    prediction_path = tmp_path / "reports" / "oracle-predictions.jsonl"
    lab.save_qa_predictions((prediction,), prediction_path)
    candidate_report = lab.qa_eval_report(
        cases,
        (prediction,),
        gold_path=qa_path,
        prediction_path=prediction_path,
    )
    candidate_report_path = tmp_path / "reports" / "candidate-qa-eval.json"
    lab.save_qa_eval_report(candidate_report, candidate_report_path)
    baseline_report = lab.qa_eval_report(cases, ())
    baseline_report_path = tmp_path / "reports" / "vlm-qa-eval.json"
    lab.save_qa_eval_report(baseline_report, baseline_report_path)
    qa_delta_report = lab.qa_eval_delta_report(
        candidate_report,
        baseline_report,
        candidate_name="graph_tool",
        baseline_name="vlm",
        candidate_report_path=candidate_report_path,
        baseline_report_path=baseline_report_path,
    )
    qa_delta_report_path = tmp_path / "reports" / "qa-delta.json"
    lab.save_qa_eval_delta_report(qa_delta_report, qa_delta_report_path)

    active_task = lab.ActiveEQATask(
        id=f"active:{cases[0].id}",
        scene_id=cases[0].scene_id,
        episode_id=cases[0].episode_id,
        initial_step=cases[0].step,
        question=cases[0].question,
        gold_answer=cases[0].answer,
        success_conditions={"answer_exact": True},
        max_actions=1,
        required_evidence={
            "nodes": cases[0].required_nodes,
            "edges": cases[0].required_edges,
        },
    )
    active_candidate = lab.ActiveTaskResult(
        task_id=active_task.id,
        policy="oracle_evidence",
        answer=cases[0].answer,
        success=True,
        action_count=1,
        evidence_nodes=cases[0].required_nodes,
        evidence_edges=cases[0].required_edges,
        final_step=cases[0].step + 1,
        confidence=1.0,
    )
    active_baseline = lab.ActiveTaskResult(
        task_id=active_task.id,
        policy="direct_answer",
        answer={},
        success=False,
        action_count=0,
        final_step=cases[0].step,
        confidence=0.0,
        error="missing_required_evidence",
    )
    active_candidate_report = lab.active_task_report(
        (active_task,),
        (active_candidate,),
    )
    active_baseline_report = lab.active_task_report(
        (active_task,),
        (active_baseline,),
    )
    active_candidate_report_path = tmp_path / "reports" / "active-candidate.json"
    active_baseline_report_path = tmp_path / "reports" / "active-baseline.json"
    lab.save_active_task_report(active_candidate_report, active_candidate_report_path)
    lab.save_active_task_report(active_baseline_report, active_baseline_report_path)
    active_delta_report = lab.active_task_delta_report(
        active_candidate_report,
        active_baseline_report,
        candidate_name="oracle_evidence",
        baseline_name="direct_answer",
        candidate_report_path=active_candidate_report_path,
        baseline_report_path=active_baseline_report_path,
    )
    active_delta_report_path = tmp_path / "reports" / "active-delta.json"
    lab.save_active_task_delta_report(active_delta_report, active_delta_report_path)

    observations = (
        lab.SceneObservation(
            step=1,
            agent_pose=lab.Pose3D(0.0, 0.0, 0.0),
            objects=(
                lab.ObjectObservation(
                    "mug_1",
                    "mug",
                    lab.Pose3D(0.1, 1.0, 0.78),
                    lab.BBox3D(
                        center=lab.Pose3D(0.1, 1.0, 0.78),
                        size=(0.12, 0.12, 0.16),
                    ),
                    confidence=0.91,
                    visible=True,
                    attributes={
                        "depth_path": "frames/000001.depth.png",
                        "detector": "detic_fixture",
                        "rgb_path": "frames/000001.rgb.png",
                        "source": "rgbd_detector",
                    },
                ),
                lab.ObjectObservation(
                    "plate_1",
                    "plate",
                    lab.Pose3D(0.3, 1.0, 0.78),
                    lab.BBox3D(
                        center=lab.Pose3D(0.3, 1.0, 0.78),
                        size=(0.12, 0.12, 0.16),
                    ),
                    confidence=0.88,
                    visible=True,
                    attributes={
                        "depth_path": "frames/000001.depth.png",
                        "detector": "detic_fixture",
                        "rgb_path": "frames/000001.rgb.png",
                        "source": "rgbd_detector",
                    },
                ),
            ),
        ),
        lab.SceneObservation(
            step=2,
            agent_pose=lab.Pose3D(0.1, 0.0, 0.0),
            objects=(
                lab.ObjectObservation(
                    "mug_1",
                    "mug",
                    lab.Pose3D(0.1, 1.0, 0.78),
                    lab.BBox3D(
                        center=lab.Pose3D(0.1, 1.0, 0.78),
                        size=(0.12, 0.12, 0.16),
                    ),
                    confidence=0.35,
                    visible=False,
                    attributes={
                        "depth_path": "frames/000002.depth.png",
                        "detector": "detic_fixture",
                        "hidden_reason": "not_detected_in_frame",
                        "source": "rgbd_tracker",
                    },
                ),
            ),
        ),
    )
    observation_path = tmp_path / "source" / "detector-observations.json"
    predicted_graph_path = tmp_path / "reports" / "predicted-graph.json"
    predicted_graph_report_path = tmp_path / "reports" / "predicted-report.json"
    predicted_dsg_evidence_report_path = (
        tmp_path / "reports" / "predicted-dsg-evidence.json"
    )
    lab.save_scene_observation_sequence(observations, observation_path)
    predicted_graph = lab.build_predicted_graph_from_observations(
        observations,
        source_path=observation_path,
    )
    lab.save_graph_json(predicted_graph, predicted_graph_path)
    predicted_report = lab.predicted_graph_report_from_observations(
        input_path=observation_path,
        graph_path=predicted_graph_path,
        graph=predicted_graph,
        observations=observations,
    )
    lab.save_predicted_graph_report(predicted_report, predicted_graph_report_path)
    predicted_evidence = lab.predicted_dsg_evidence_report(
        predicted_report,
        predicted_graph_report_path=predicted_graph_report_path,
    )
    lab.save_predicted_dsg_evidence_report(
        predicted_evidence,
        predicted_dsg_evidence_report_path,
    )

    graph_eval_report = lab.graph_eval_report(
        graph,
        predicted_graph,
        oracle_path=graph_path,
        predicted_path=predicted_graph_path,
    )
    graph_eval_report_path = tmp_path / "reports" / "graph-eval.json"
    lab.save_graph_eval_report(graph_eval_report, graph_eval_report_path)
    error_attribution = lab.error_attribution_report(
        cases,
        oracle_graph=graph,
        predicted_graph=predicted_graph,
        predictions=(prediction,),
        gold_path=qa_path,
        oracle_graph_path=graph_path,
        predicted_graph_path=predicted_graph_path,
        prediction_path=prediction_path,
    )
    error_attribution_report_path = tmp_path / "reports" / "error-attribution.json"
    lab.save_error_attribution_report(error_attribution, error_attribution_report_path)

    offline_records = tuple(
        lab.OfflinePredictionRecord(
            case_id=case.id,
            answer=case.answer,
            confidence=0.8,
        )
        for case in cases
    )
    offline_input_path = tmp_path / "reports" / "vlm-input.jsonl"
    offline_prediction_path = tmp_path / "reports" / "vlm-predictions.jsonl"
    offline_import_report_path = tmp_path / "reports" / "vlm-import.json"
    lab.save_offline_prediction_records(offline_records, offline_input_path)
    imported_predictions, offline_import_report = lab.import_offline_predictions(
        cases,
        offline_records,
        source_name="llava16_ai2thor_trial",
        source_kind="vlm",
        source_metadata={
            "capabilities": ("spatial_qa",),
            "dataset_id": "ai2thor-real-trial-v1",
            "model_id": "llava-v1.6-34b",
            "prompt_id": "vlm-spatial-qa-v1",
        },
        qa_path=qa_path,
        input_path=offline_input_path,
        prediction_path=offline_prediction_path,
    )
    lab.save_qa_predictions(imported_predictions, offline_prediction_path)
    lab.save_offline_prediction_import_report(
        offline_import_report,
        offline_import_report_path,
    )
    offline_control_matrix = lab.offline_control_matrix_report(
        (offline_import_report,),
        report_paths=(offline_import_report_path,),
        required_source_kinds=("vlm",),
    )
    offline_control_matrix_report_path = tmp_path / "reports" / "offline-matrix.json"
    lab.save_offline_control_matrix_report(
        offline_control_matrix,
        offline_control_matrix_report_path,
    )
    offline_control_result = lab.offline_control_result_report(
        offline_control_matrix,
        matrix_report_path=offline_control_matrix_report_path,
        candidate_qa_eval_report_path=candidate_report_path,
        qa_eval_delta_report_paths={
            "vlm:llava16_ai2thor_trial": qa_delta_report_path,
        },
    )
    offline_control_result_report_path = (
        tmp_path / "reports" / "offline-control-result.json"
    )
    lab.save_offline_control_result_report(
        offline_control_result,
        offline_control_result_report_path,
    )

    dashboard = lab.dashboard_bundle(
        cases,
        predictions=(prediction,),
        qa_eval_report=candidate_report,
        graph=graph,
        active_task_report=active_candidate_report,
        active_task_delta_report=active_delta_report,
        qa_path=qa_path,
        prediction_path=prediction_path,
        qa_eval_report_path=candidate_report_path,
        graph_path=graph_path,
        active_task_report_path=active_candidate_report_path,
        active_task_delta_report_path=active_delta_report_path,
    )
    dashboard_bundle_path = tmp_path / "reports" / "dashboard.json"
    lab.save_dashboard_bundle(dashboard, dashboard_bundle_path)
    return {
        "episode_paths": (episode_path,),
        "qa_delta_report_path": qa_delta_report_path,
        "active_delta_report_path": active_delta_report_path,
        "dashboard_bundle_path": dashboard_bundle_path,
        "error_attribution_report_path": error_attribution_report_path,
        "graph_eval_report_path": graph_eval_report_path,
        "offline_control_matrix_report_path": offline_control_matrix_report_path,
        "offline_control_result_report_path": offline_control_result_report_path,
        "offline_import_report_path": offline_import_report_path,
        "predicted_dsg_evidence_report_path": predicted_dsg_evidence_report_path,
        "predicted_graph_report_path": predicted_graph_report_path,
        "real_collection_report_path": real_collection_report_path,
    }


def _write_real_frame_assets(
    episode_path: Path,
    frames: tuple[lab.EpisodeFrame, ...],
) -> None:
    for frame in frames:
        for asset_path_text in (
            frame.depth_path,
            frame.rgb_path,
            frame.segmentation_path,
        ):
            assert asset_path_text is not None
            asset_path = episode_path.parent / asset_path_text
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            asset_path.write_text(f"{frame.step}\n", encoding="utf-8")


def _anchor(root: Path, path: str) -> Path:
    local_path = Path(path)
    if local_path.is_absolute():
        return local_path
    return root / local_path


def _offline_control_cases() -> tuple[lab.QACase, ...]:
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


def _offline_control_predictions(
    cases: tuple[lab.QACase, ...],
) -> tuple[lab.QAPrediction, ...]:
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


def _offline_control_import_manifest_for_real_run(
    tmp_path: Path,
    inputs: ReadyPackageInputs,
) -> Path:
    offline_import_report = lab.load_offline_prediction_import_report(
        inputs["offline_import_report_path"]
    )
    error_attribution_report = lab.load_error_attribution_report(
        inputs["error_attribution_report_path"]
    )
    source_metadata = {
        "caption_memory": {
            "capabilities": ["spatial_qa", "dynamic_memory"],
            "dataset_id": "ai2thor-real-trial-v1",
            "model_id": "blip2-flan-t5-xl",
            "prompt_id": "caption-memory-spatial-v1",
        },
        "graph_text": {
            "capabilities": ["spatial_qa", "dynamic_memory", "graph_tool_query"],
            "dataset_id": "ai2thor-real-trial-v1",
            "model_id": "gpt-4.1-mini",
            "prompt_id": "graph-text-spatial-qa-v1",
        },
        "multi_frame_vlm": {
            "capabilities": ["spatial_qa", "dynamic_memory"],
            "dataset_id": "ai2thor-real-trial-v1",
            "model_id": "llava-v1.6-34b",
            "prompt_id": "multi-frame-vlm-spatial-qa-v1",
        },
        "vlm": {
            "capabilities": ["spatial_qa"],
            "dataset_id": "ai2thor-real-trial-v1",
            "model_id": "llava-v1.6-34b",
            "prompt_id": "vlm-spatial-qa-v1",
        },
    }
    source_names = {
        "caption_memory": "caption_memory_ai2thor_trial",
        "graph_text": "graph_text_ai2thor_trial",
        "multi_frame_vlm": "llava16_multiframe_ai2thor_trial",
        "vlm": "llava16_ai2thor_trial",
    }
    sources = [
        {
            "source_kind": source_kind,
            "source_name": source_names[source_kind],
            "input_path": offline_import_report["input_path"],
            "input_format": "offline_prediction_record",
            "metadata": source_metadata[source_kind],
        }
        for source_kind in ("vlm", "multi_frame_vlm", "caption_memory", "graph_text")
    ]
    root = tmp_path / "offline-manifest"
    manifest = {
        "schema_version": "dsg-spatialqa-lab.offline-control-import-manifest.v1",
        "qa_path": offline_import_report["qa_path"],
        "output_dir": str(root / "imports"),
        "matrix_report_path": str(root / "offline-control-matrix.json"),
        "result_report_path": str(root / "offline-control-result.json"),
        "candidate_prediction_path": error_attribution_report["prediction_path"],
        "candidate_name": "predicted_graph_tool",
        "qa_eval_output_dir": str(root / "qa-eval"),
        "required_source_kinds": [
            "caption_memory",
            "graph_text",
            "multi_frame_vlm",
            "vlm",
        ],
        "sources": sources,
    }
    manifest_path = root / "offline-control-import-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _predicted_dsg_detector_run_manifest_for_real_run(
    tmp_path: Path,
    inputs: ReadyPackageInputs,
) -> tuple[Path, Path, Path]:
    source_error_report = lab.load_error_attribution_report(
        inputs["error_attribution_report_path"]
    )
    gold_path = Path(str(source_error_report["gold_path"]))
    oracle_graph_path = Path(str(source_error_report["oracle_graph_path"]))
    prediction_path = Path(str(source_error_report["prediction_path"]))
    cases = tuple(lab.load_qa_dataset(gold_path))
    oracle_graph = lab.load_graph_json(oracle_graph_path)
    predictions = tuple(lab.load_qa_predictions(prediction_path))
    root = tmp_path / "predicted-detector-manifest"
    detector_jsonl_path = _write_real_run_detector_jsonl(root)
    sequence_path = root / "predicted" / "detector-observations.json"
    graph_path = root / "predicted" / "predicted-graph.json"
    predicted_report_path = root / "predicted" / "predicted-report.json"
    detector_import_report_path = root / "predicted" / "detector-import-report.json"
    evidence_report_path = root / "predicted" / "predicted-dsg-evidence.json"
    lab.run_predicted_dsg_from_detector_jsonl(
        detector_jsonl_path=detector_jsonl_path,
        output_sequence_path=sequence_path,
        output_graph_path=graph_path,
        predicted_graph_report_path=predicted_report_path,
        detector_import_report_path=detector_import_report_path,
        predicted_dsg_evidence_report_path=evidence_report_path,
    )
    predicted_graph = lab.load_graph_json(graph_path)
    graph_eval_report = lab.graph_eval_report(
        oracle_graph,
        predicted_graph,
        oracle_path=oracle_graph_path,
        predicted_path=graph_path,
    )
    graph_eval_report_path = root / "reports" / "graph-eval.json"
    lab.save_graph_eval_report(graph_eval_report, graph_eval_report_path)
    error_attribution_report = lab.error_attribution_report(
        cases,
        oracle_graph=oracle_graph,
        predicted_graph=predicted_graph,
        predictions=predictions,
        gold_path=gold_path,
        oracle_graph_path=oracle_graph_path,
        predicted_graph_path=graph_path,
        prediction_path=prediction_path,
    )
    error_attribution_report_path = root / "reports" / "error-attribution.json"
    lab.save_error_attribution_report(
        error_attribution_report,
        error_attribution_report_path,
    )
    manifest = {
        "schema_version": (
            "dsg-spatialqa-lab.predicted-dsg-detector-run-manifest.v1"
        ),
        "detector_jsonl_path": str(detector_jsonl_path),
        "output_sequence_path": str(sequence_path),
        "output_graph_path": str(graph_path),
        "predicted_graph_report_path": str(predicted_report_path),
        "detector_import_report_path": str(detector_import_report_path),
        "predicted_dsg_evidence_report_path": str(evidence_report_path),
        "infer_relations": ["LEFT_OF", "RIGHT_OF", "NEAR"],
        "reference_frames": ["world"],
        "min_observation_count": 2,
        "min_object_observation_count": 2,
        "required_evidence_kinds": ["depth", "detector", "rgb"],
    }
    manifest_path = root / "predicted-dsg-detector-run-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, graph_eval_report_path, error_attribution_report_path


def _write_real_run_detector_jsonl(root: Path) -> Path:
    path = root / "source" / "real-rgbd-detections.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n"
            for record in _real_run_detector_records()
        ),
        encoding="utf-8",
    )
    return path


def _write_detector_records_and_assets(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = _real_run_detector_records()
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


def _write_not_ready_real_collection_receipt(root: Path) -> None:
    episode_path = root / "inputs/episodes/FloorPlan1.jsonl"
    frame = lab.EpisodeFrame(
        episode_id="ai2thor_real_receipt_gate_001",
        scene_id="FloorPlan1",
        step=1,
        rgb_path="frames/000001.rgb.png",
        depth_path="frames/000001.depth.png",
        segmentation_path="frames/000001.segmentation.png",
        agent_id="agent",
        agent_pose=lab.Pose3D(0.0, 0.0, 0.0),
        action="Initialize",
        visible_object_ids=("mug_1",),
        metadata={
            "adapter": "ai2thor",
            "collection_kind": "real",
            "source_kind": "ai2thor",
        },
    )
    lab.save_episode_sequence((frame,), episode_path)
    for asset_path_text in (frame.depth_path, frame.segmentation_path):
        assert asset_path_text is not None
        asset_path = episode_path.parent / asset_path_text
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        asset_path.write_text(f"{frame.step}\n", encoding="utf-8")
    report = lab.real_collection_report(
        dataset_name="ai2thor_real_smoke",
        episode_paths=(episode_path,),
        source_kind="ai2thor",
        min_episode_count=1,
        min_scene_count=1,
        min_frame_count=1,
    )
    lab.save_real_collection_report(
        report,
        root / "inputs/real-collection-report.json",
    )


def _write_ready_real_collection_receipt(root: Path) -> None:
    episode_path = root / "inputs/episodes/FloorPlan1.jsonl"
    frame = lab.EpisodeFrame(
        episode_id="ai2thor_real_receipt_gate_001",
        scene_id="FloorPlan1",
        step=1,
        rgb_path="frames/000001.rgb.png",
        depth_path="frames/000001.depth.png",
        segmentation_path="frames/000001.segmentation.png",
        agent_id="agent",
        agent_pose=lab.Pose3D(0.0, 0.0, 0.0),
        action="Initialize",
        visible_object_ids=("mug_1",),
        metadata={
            "adapter": "ai2thor",
            "collection_kind": "real",
            "source_kind": "ai2thor",
        },
    )
    lab.save_episode_sequence((frame,), episode_path)
    for asset_path_text in (frame.rgb_path, frame.depth_path, frame.segmentation_path):
        assert asset_path_text is not None
        asset_path = episode_path.parent / asset_path_text
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        asset_path.write_text(f"{frame.step}\n", encoding="utf-8")
    report = lab.real_collection_report(
        dataset_name="ai2thor_real_smoke",
        episode_paths=(episode_path,),
        source_kind="ai2thor",
        min_episode_count=1,
        min_scene_count=1,
        min_frame_count=1,
    )
    lab.save_real_collection_report(
        report,
        root / "inputs/real-collection-report.json",
    )


def _write_ready_offline_control_receipt(
    root: Path,
    offline_manifest_path: Path,
) -> None:
    _write_ready_offline_control_contract(root, offline_manifest_path)
    _write_ready_offline_control_receipt_bundle(root, offline_manifest_path)


def _write_ready_offline_control_contract(
    root: Path,
    offline_manifest_path: Path,
) -> None:
    manifest = lab.load_offline_control_import_manifest(offline_manifest_path)
    cases = _offline_control_cases()
    predictions = _offline_control_predictions(cases)
    lab.save_qa_dataset(
        cases,
        _anchor(offline_manifest_path.parent, str(manifest["qa_path"])),
    )
    lab.save_qa_predictions(
        predictions,
        _anchor(
            offline_manifest_path.parent,
            str(manifest["candidate_prediction_path"]),
        ),
    )
    for source in manifest["sources"]:
        lab.save_qa_predictions(
            predictions,
            _anchor(offline_manifest_path.parent, str(source["input_path"])),
        )
    preflight = lab.offline_control_import_manifest_preflight(offline_manifest_path)
    lab.save_offline_control_artifact_contracts(
        preflight["artifact_contracts"],
        root / "offline-control-artifact-contracts.json",
    )


def _write_ready_offline_control_receipt_bundle(
    root: Path,
    offline_manifest_path: Path,
) -> None:
    bundle = lab.offline_control_prediction_receipt_bundle(offline_manifest_path)
    lab.save_offline_control_prediction_receipt_bundle(
        bundle,
        root / "offline-control-prediction-receipt-bundle.json",
    )


def _write_ready_predicted_dsg_receipt(
    root: Path,
    predicted_manifest_path: Path,
) -> None:
    _write_ready_predicted_dsg_contract(root, predicted_manifest_path)
    _write_ready_predicted_dsg_receipt_bundle(root, predicted_manifest_path)


def _write_ready_predicted_dsg_contract(
    root: Path,
    predicted_manifest_path: Path,
) -> None:
    manifest = lab.load_predicted_dsg_detector_run_manifest(predicted_manifest_path)
    _write_detector_records_and_assets(Path(str(manifest["detector_jsonl_path"])))
    preflight = lab.predicted_dsg_detector_run_manifest_preflight(
        predicted_manifest_path
    )
    lab.save_predicted_dsg_detector_artifact_contract(
        preflight["artifact_contract"],
        root / "predicted-dsg-detector-artifact-contract.json",
    )


def _write_ready_predicted_dsg_receipt_bundle(
    root: Path,
    predicted_manifest_path: Path,
) -> None:
    bundle = lab.predicted_dsg_detector_receipt_bundle(predicted_manifest_path)
    lab.save_predicted_dsg_detector_receipt_bundle(
        bundle,
        root / "predicted-dsg-detector-receipt-bundle.json",
    )


def _write_review_artifact_placeholders(root: Path) -> None:
    for path in (
        root / "inputs/review/active-task-delta.json",
        root / "inputs/review/dashboard.json",
        root / "inputs/review/error-attribution.json",
        root / "inputs/review/graph-eval.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")


def _real_experiment_run_manifest(
    tmp_path: Path,
    inputs: ReadyPackageInputs,
) -> Path:
    offline_manifest_path = _offline_control_import_manifest_for_real_run(
        tmp_path,
        inputs,
    )
    (
        predicted_manifest_path,
        graph_eval_report_path,
        error_attribution_report_path,
    ) = _predicted_dsg_detector_run_manifest_for_real_run(tmp_path, inputs)
    root = tmp_path / "run-manifest"
    manifest = {
        "schema_version": "dsg-spatialqa-lab.real-experiment-run-manifest.v1",
        "dataset_name": "ai2thor_real_smoke",
        "episode_paths": [str(path) for path in inputs["episode_paths"]],
        "output_dir": str(root / "benchmark"),
        "manifest_path": str(root / "benchmark-manifest.json"),
        "readiness_report_path": str(root / "real-readiness.json"),
        "summary_report_path": str(root / "experiment-summary.json"),
        "record_path": str(root / "experiment-record.json"),
        "real_experiment_run_ledger_path": str(
            root / "real-experiment-run-ledger.json"
        ),
        "max_qa_per_episode": 20,
        "tags": ["benchmark", "real"],
        "declared_data_source_kind": "real",
        "min_episode_count": 1,
        "min_scene_count": 1,
        "min_qa_count": 8,
        "required_control_kinds": [
            "caption_memory",
            "graph_text",
            "multi_frame_vlm",
            "vlm",
        ],
        "required_predicted_input_kinds": ["observation_sequence"],
        "active_task_delta_report_paths": [
            str(inputs["active_delta_report_path"])
        ],
        "dashboard_bundle_paths": [str(inputs["dashboard_bundle_path"])],
        "error_attribution_report_paths": [str(error_attribution_report_path)],
        "graph_eval_report_paths": [str(graph_eval_report_path)],
        "offline_control_import_manifest_path": str(offline_manifest_path),
        "offline_control_import_run_ledger_path": str(
            root / "offline-control-import-run-ledger.json"
        ),
        "predicted_dsg_detector_run_manifest_path": str(predicted_manifest_path),
        "predicted_dsg_detector_run_ledger_path": str(
            root / "predicted-dsg-detector-run-ledger.json"
        ),
        "real_collection_report_paths": [str(inputs["real_collection_report_path"])],
    }
    manifest_path = root / "real-experiment-run-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _execution_packet_for_run_manifest(
    run_manifest_path: Path,
    execution_packet_path: Path,
) -> dict[str, object]:
    packet: dict[str, object] = {
        "schema_version": "dsg-spatialqa-lab.real-experiment-execution-packet.v1",
        "action": "real_experiment_execution_packet",
        "launch_report_path": str(execution_packet_path.parent / "launch-report.json"),
        "launch_report_digest": "0" * 64,
        "contracts_path": None,
        "contracts_digest": None,
        "run_manifest_path": str(run_manifest_path),
        "execution_packet_path": str(execution_packet_path),
        "primary_evidence_acceptance_report_path": str(
            execution_packet_path.parent
            / "real-experiment-primary-evidence-acceptance-report.json"
        ),
        "primary_evidence_acceptance": {
            "acceptance_digest": "1" * 64,
            "all_tracks_accepted": True,
            "matches_current": True,
            "path": str(
                execution_packet_path.parent
                / "real-experiment-primary-evidence-acceptance-report.json"
            ),
            "present": True,
            "ready_for_launch_refresh": True,
            "summary": {
                "accepted_track_count": 3,
                "all_tracks_accepted": True,
                "blocked_track_count": 0,
                "invalid_track_count": 0,
                "missing_track_count": 0,
                "not_ready_track_count": 0,
                "ready_for_launch_refresh": True,
                "track_count": 3,
            },
            "valid": True,
        },
        "ready_to_execute": True,
        "execution_blocked": False,
        "readiness": {
            "launch_report_valid": True,
            "launch_report_matches_current": True,
            "preflight_ready_to_run": True,
            "primary_evidence_acceptance_report_matches_current": True,
            "primary_evidence_acceptance_report_present": True,
            "primary_evidence_acceptance_report_ready": True,
            "primary_evidence_acceptance_report_valid": True,
            "primary_evidence_receipt_gate_ready": True,
            "ready_to_run": True,
        },
        "blocker_summary": {
            "track": "primary_evidence",
            "ready": True,
            "blocked_track_count": 0,
            "blocked_tracks": [],
            "ready_track_count": 3,
            "ready_tracks": ["real_data", "real_controls", "predicted_dsg"],
            "track_order": ["real_data", "real_controls", "predicted_dsg"],
        },
        "audit_commands": [
            {
                "key": "validate_launch_report",
                "command": "python scripts/run_real_experiment.py --validate-external-artifact-launch-report launch-report.json",
                "order": 1,
                "phase": "audit",
                "required": True,
            },
            {
                "key": "compare_launch_report",
                "command": "python scripts/run_real_experiment.py --compare-external-artifact-launch-report launch-report.json",
                "order": 2,
                "phase": "audit",
                "required": True,
            },
            {
                "key": "validate_primary_evidence_acceptance_report",
                "command": "python scripts/run_real_experiment.py --validate-primary-evidence-acceptance-report real-experiment-primary-evidence-acceptance-report.json",
                "order": 3,
                "phase": "audit",
                "required": True,
            },
            {
                "key": "compare_primary_evidence_acceptance_report",
                "command": "python scripts/run_real_experiment.py --compare-primary-evidence-acceptance-report real-experiment-primary-evidence-acceptance-report.json",
                "order": 4,
                "phase": "audit",
                "required": True,
            },
        ],
        "execution_commands": [
            {
                "key": "preflight_run_manifest",
                "command": (
                    "python scripts/run_real_experiment.py --preflight-run-manifest "
                    f"{run_manifest_path}"
                ),
                "order": 1,
                "phase": "execute",
                "required": True,
            },
            {
                "key": "run_real_experiment",
                "command": (
                    "python scripts/run_real_experiment.py --run-manifest "
                    f"{run_manifest_path} --approved-execution-packet "
                    f"{execution_packet_path}"
                ),
                "order": 2,
                "phase": "execute",
                "required": True,
            },
        ],
        "validation": {"valid": True, "report_digest": "0" * 64},
        "comparison": {
            "matches": True,
            "saved_digest": "0" * 64,
            "current_digest": "0" * 64,
        },
    }
    packet["packet_digest"] = lab.real_experiment_execution_packet_digest(packet)
    lab.save_real_experiment_execution_packet(packet, execution_packet_path)
    return packet


def _real_run_detector_records() -> tuple[dict[str, object], ...]:
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
