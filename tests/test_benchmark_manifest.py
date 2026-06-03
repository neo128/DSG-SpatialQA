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
BUILD_BENCHMARK_SCRIPT = ROOT / "scripts" / "build_benchmark.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_build_benchmark_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_benchmark_script",
        BUILD_BENCHMARK_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_benchmark_artifacts_build_small_manifest_and_coverage(tmp_path: Path) -> None:
    assert hasattr(lab, "build_benchmark_artifacts")
    assert hasattr(lab, "benchmark_manifest_digest")
    assert hasattr(lab, "validate_benchmark_manifest")
    episode_paths = _write_mock_episodes(tmp_path)
    output_dir = tmp_path / "benchmark"

    manifest = lab.build_benchmark_artifacts(
        dataset_name="mock_benchmark",
        episode_paths=episode_paths,
        output_dir=output_dir,
        max_qa_per_episode=4,
        tags=("benchmark", "oracle"),
    )
    validation = lab.validate_benchmark_manifest(manifest)

    assert manifest["schema_version"] == "dsg-spatialqa-lab.benchmark-manifest.v1"
    assert manifest["dataset_name"] == "mock_benchmark"
    assert manifest["summary"] == {
        "dataset_name": "mock_benchmark",
        "episode_count": 2,
        "qa_count": 8,
        "scene_count": 2,
        "task_count": 0,
    }
    assert manifest["scene_count"] == 2
    assert manifest["episode_count"] == 2
    assert manifest["qa_count"] == 8
    assert manifest["task_count"] == 0
    assert manifest["filters"] == {
        "max_qa_per_episode": 4,
        "source": "oracle",
        "tags": ["benchmark", "oracle"],
    }
    assert manifest["coverage"]["by_episode"] == {
        "ai2thor_mock_001": 4,
        "habitat_mock_001": 4,
    }
    assert manifest["coverage"]["by_scene"] == {"FloorPlan1": 4, "apartment_0": 4}
    assert manifest["coverage"]["by_question_type"] == {
        "object_location": 4,
        "object_room": 2,
        "relative_relation": 2,
    }
    assert manifest["coverage"]["by_reference_frame"] == {"none": 6, "world": 2}
    assert manifest["coverage"]["by_tag"]["benchmark"] == 8
    assert manifest["coverage"]["dynamic_static"] == {"dynamic": 0, "static": 8}
    assert manifest["coverage"]["oracle_predicted"] == {"oracle": 2, "predicted": 0}
    assert sorted(manifest["graph_digests"]) == ["ai2thor_mock_001", "habitat_mock_001"]
    assert sorted(manifest["qa_dataset_digests"]) == [
        "ai2thor_mock_001",
        "habitat_mock_001",
    ]
    assert validation["valid"] is True
    assert manifest["manifest_digest"] == lab.benchmark_manifest_digest(manifest)
    for artifact in manifest["artifacts"]:
        assert Path(artifact["episode_path"]).exists()
        assert Path(artifact["graph_path"]).exists()
        assert Path(artifact["qa_path"]).exists()


def test_benchmark_manifest_json_save_load_and_compare_detects_qa_drift(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "benchmark_manifest_json")
    assert hasattr(lab, "save_benchmark_manifest")
    assert hasattr(lab, "load_benchmark_manifest")
    assert hasattr(lab, "compare_benchmark_manifest")
    manifest_path = tmp_path / "benchmark-manifest.json"
    manifest = lab.build_benchmark_artifacts(
        dataset_name="mock_benchmark",
        episode_paths=_write_mock_episodes(tmp_path),
        output_dir=tmp_path / "benchmark",
        max_qa_per_episode=3,
    )

    lab.save_benchmark_manifest(manifest, manifest_path)
    loaded = lab.load_benchmark_manifest(manifest_path)
    comparison = lab.compare_benchmark_manifest(loaded)

    assert json.loads(lab.benchmark_manifest_json(manifest)) == manifest
    assert loaded == manifest
    assert comparison["matches"] is True

    first_artifact = cast(dict[str, object], loaded["artifacts"][0])
    qa_path = Path(cast(str, first_artifact["qa_path"]))
    cases = lab.load_qa_dataset(qa_path)
    cases[0].answer["object_id"] = "changed"
    lab.save_qa_dataset(cases, qa_path)

    drift = lab.compare_benchmark_manifest(loaded)
    checks = {check["name"]: check for check in drift["checks"]}
    assert drift["matches"] is False
    assert checks["qa_dataset_digests_match_current"]["passed"] is False
    assert checks["coverage_matches_current"]["passed"] is True


def test_benchmark_manifest_records_experiment_artifacts_and_detects_drift(
    tmp_path: Path,
) -> None:
    manifest = lab.build_benchmark_artifacts(
        dataset_name="mock_benchmark",
        episode_paths=_write_mock_episodes(tmp_path),
        output_dir=tmp_path / "benchmark",
        max_qa_per_episode=3,
    )
    case = lab.load_qa_dataset(cast(str, manifest["artifacts"][0]["qa_path"]))[0]
    graph = lab.load_graph_json(cast(str, manifest["artifacts"][0]["graph_path"]))
    episode_path = Path(cast(str, manifest["artifacts"][0]["episode_path"]))
    frames = lab.load_episode_sequence(episode_path)
    predicted_graph = lab.build_predicted_graph_from_episode(frames)
    prediction = lab.QAPrediction(id=case.id, answer=case.answer, confidence=1.0)
    candidate_report = lab.qa_eval_report((case,), (prediction,))
    baseline_report = lab.qa_eval_report((case,), ())
    qa_delta_report = lab.qa_eval_delta_report(
        candidate_report,
        baseline_report,
        candidate_name="graph_tool",
        baseline_name="majority",
    )
    active_task = _task_for_case(case, max_actions=1)
    active_candidate_result = lab.ActiveTaskResult(
        task_id=active_task.id,
        policy="oracle_evidence",
        answer=case.answer,
        success=True,
        action_count=1,
        evidence_nodes=case.required_nodes,
        evidence_edges=case.required_edges,
        final_step=active_task.initial_step + 1,
        confidence=1.0,
    )
    active_baseline_result = lab.ActiveTaskResult(
        task_id=active_task.id,
        policy="direct_answer",
        answer={},
        success=False,
        action_count=0,
        final_step=active_task.initial_step,
        confidence=0.0,
        error="missing_required_evidence",
    )
    active_report = lab.active_task_report((active_task,), (active_candidate_result,))
    active_delta_report = lab.active_task_delta_report(
        active_report,
        lab.active_task_report((active_task,), (active_baseline_result,)),
        candidate_name="oracle_evidence",
        baseline_name="direct_answer",
    )
    dashboard = lab.dashboard_bundle(
        (case,),
        predictions=(prediction,),
        qa_eval_report=candidate_report,
        graph=graph,
        active_task_report=active_report,
        active_task_delta_report=active_delta_report,
    )
    predicted_graph_path = tmp_path / "predicted-graph.json"
    predicted_report_path = tmp_path / "predicted-report.json"
    predicted_evidence_path = tmp_path / "predicted-dsg-evidence.json"
    graph_eval_path = tmp_path / "graph-eval-report.json"
    error_attribution_path = tmp_path / "error-attribution-report.json"
    prediction_path = tmp_path / "predictions.jsonl"
    offline_input_path = tmp_path / "offline-input.jsonl"
    offline_prediction_path = tmp_path / "offline-predictions.jsonl"
    offline_import_report_path = tmp_path / "offline-import-report.json"
    offline_control_matrix_path = tmp_path / "offline-control-matrix.json"
    offline_control_matrix_path = tmp_path / "offline-control-matrix.json"
    offline_control_matrix_path = tmp_path / "offline-control-matrix.json"
    qa_report_path = tmp_path / "qa-eval-report.json"
    qa_delta_path = tmp_path / "qa-delta-report.json"
    active_report_path = tmp_path / "active-report.json"
    active_delta_path = tmp_path / "active-delta-report.json"
    dashboard_path = tmp_path / "dashboard.json"
    lab.save_graph_json(predicted_graph, predicted_graph_path)
    lab.save_qa_predictions((prediction,), prediction_path)
    predicted_report = lab.predicted_graph_report(
        input_path=episode_path,
        graph_path=predicted_graph_path,
        graph=predicted_graph,
        frames=frames,
    )
    graph_eval = lab.graph_eval_report(
        graph,
        predicted_graph,
        oracle_path=cast(str, manifest["artifacts"][0]["graph_path"]),
        predicted_path=predicted_graph_path,
    )
    error_attribution_report = lab.error_attribution_report(
        (case,),
        oracle_graph=graph,
        predicted_graph=predicted_graph,
        predictions=(prediction,),
        gold_path=cast(str, manifest["artifacts"][0]["qa_path"]),
        oracle_graph_path=cast(str, manifest["artifacts"][0]["graph_path"]),
        predicted_graph_path=predicted_graph_path,
        prediction_path=prediction_path,
    )
    lab.save_predicted_graph_report(predicted_report, predicted_report_path)
    predicted_evidence = lab.predicted_dsg_evidence_report(
        predicted_report,
        predicted_graph_report_path=predicted_report_path,
    )
    lab.save_predicted_dsg_evidence_report(
        predicted_evidence,
        predicted_evidence_path,
    )
    lab.save_offline_prediction_records(
        (
            lab.OfflinePredictionRecord(
                case_id=case.id,
                answer=case.answer,
                evidence_nodes=case.required_nodes,
                evidence_edges=case.required_edges,
                confidence=0.88,
            ),
        ),
        offline_input_path,
    )
    imported_predictions, offline_import_report = lab.import_offline_predictions(
        (case,),
        lab.load_offline_prediction_records(offline_input_path),
        source_name="vlm_fixture",
        source_kind="vlm",
        source_metadata={
            "capabilities": ("spatial_qa", "graph_tool_query"),
            "model_id": "mock-vlm",
            "prompt_id": "spatial-qa-v1",
        },
        qa_path=cast(str, manifest["artifacts"][0]["qa_path"]),
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
        required_source_kinds=("caption_memory",),
    )
    lab.save_offline_control_matrix_report(
        offline_control_matrix,
        offline_control_matrix_path,
    )
    offline_control_matrix = lab.offline_control_matrix_report(
        (offline_import_report,),
        report_paths=(offline_import_report_path,),
        required_source_kinds=("vlm",),
    )
    lab.save_offline_control_matrix_report(
        offline_control_matrix,
        offline_control_matrix_path,
    )
    lab.save_graph_eval_report(graph_eval, graph_eval_path)
    lab.save_error_attribution_report(
        error_attribution_report,
        error_attribution_path,
    )
    lab.save_qa_eval_report(candidate_report, qa_report_path)
    lab.save_qa_eval_delta_report(qa_delta_report, qa_delta_path)
    lab.save_active_task_report(active_report, active_report_path)
    lab.save_active_task_delta_report(active_delta_report, active_delta_path)
    lab.save_dashboard_bundle(dashboard, dashboard_path)

    manifest_with_artifacts = lab.build_benchmark_artifacts(
        dataset_name="mock_benchmark",
        episode_paths=_write_mock_episodes(tmp_path / "with-artifacts"),
        output_dir=tmp_path / "benchmark-with-artifacts",
        max_qa_per_episode=3,
        qa_eval_report_paths=(qa_report_path,),
        qa_eval_delta_report_paths=(qa_delta_path,),
        active_task_report_paths=(active_report_path,),
        active_task_delta_report_paths=(active_delta_path,),
        dashboard_bundle_paths=(dashboard_path,),
        error_attribution_report_paths=(error_attribution_path,),
        graph_eval_report_paths=(graph_eval_path,),
        offline_control_matrix_report_paths=(offline_control_matrix_path,),
        offline_prediction_import_report_paths=(offline_import_report_path,),
        predicted_dsg_evidence_report_paths=(predicted_evidence_path,),
        predicted_graph_report_paths=(predicted_report_path,),
    )
    validation = lab.validate_benchmark_manifest(manifest_with_artifacts)
    comparison = lab.compare_benchmark_manifest(manifest_with_artifacts)

    assert manifest_with_artifacts["summary"]["experiment_artifact_count"] == 11
    assert manifest_with_artifacts["experiment_artifact_digests"] == {
        "active_task_delta_report:active-delta-report.json": active_delta_report[
            "report_digest"
        ],
        "active_task_report:active-report.json": active_report["report_digest"],
        "dashboard_bundle:dashboard.json": dashboard["bundle_digest"],
        "error_attribution_report:error-attribution-report.json": (
            error_attribution_report["report_digest"]
        ),
        "graph_eval_report:graph-eval-report.json": graph_eval["report_digest"],
        "offline_control_matrix_report:offline-control-matrix.json": (
            offline_control_matrix["report_digest"]
        ),
        "offline_prediction_import_report:offline-import-report.json": (
            offline_import_report["report_digest"]
        ),
        "predicted_dsg_evidence_report:predicted-dsg-evidence.json": (
            predicted_evidence["report_digest"]
        ),
        "predicted_graph_report:predicted-report.json": predicted_report["digest"],
        "qa_eval_delta_report:qa-delta-report.json": qa_delta_report["report_digest"],
        "qa_eval_report:qa-eval-report.json": candidate_report["report_digest"],
    }
    assert [item["artifact_type"] for item in manifest_with_artifacts["experiment_artifacts"]] == [
        "active_task_delta_report",
        "active_task_report",
        "dashboard_bundle",
        "error_attribution_report",
        "graph_eval_report",
        "offline_control_matrix_report",
        "offline_prediction_import_report",
        "predicted_dsg_evidence_report",
        "predicted_graph_report",
        "qa_eval_delta_report",
        "qa_eval_report",
    ]
    assert validation["valid"] is True
    assert comparison["matches"] is True

    drifted_active_delta = dict(active_delta_report)
    drifted_metrics = dict(active_delta_report["metrics_delta"])
    drifted_metrics["task_success"] = dict(drifted_metrics["task_success"])
    drifted_metrics["task_success"]["rate_delta"] = 0.0
    drifted_active_delta["metrics_delta"] = drifted_metrics
    drifted_active_delta["report_digest"] = lab.active_task_delta_report_digest(
        drifted_active_delta
    )
    lab.save_active_task_delta_report(drifted_active_delta, active_delta_path)

    drift = lab.compare_benchmark_manifest(manifest_with_artifacts)
    checks = {check["name"]: check for check in drift["checks"]}
    assert drift["matches"] is False
    assert checks["experiment_artifacts_match_current"]["passed"] is False


def test_build_benchmark_cli_accepts_experiment_artifact_paths(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_build_benchmark_script()
    main = cast(MainFn, getattr(module, "main"))
    episode_paths = _write_mock_episodes(tmp_path)
    graph = lab.load_scene_fixture("tabletop")
    frames = lab.load_episode_sequence(episode_paths[0])
    predicted_graph = lab.build_predicted_graph_from_episode(frames)
    case = lab.generate_qa_cases(graph, scene_id="tabletop_scene", episode_id="episode_001")[0]
    prediction = lab.QAPrediction(id=case.id, answer=case.answer, confidence=1.0)
    report = lab.qa_eval_report((case,), (prediction,))
    qa_report_path = tmp_path / "qa-eval-report.json"
    predicted_graph_path = tmp_path / "predicted-graph.json"
    predicted_report_path = tmp_path / "predicted-report.json"
    graph_eval_path = tmp_path / "graph-eval-report.json"
    error_attribution_path = tmp_path / "error-attribution-report.json"
    prediction_path = tmp_path / "predictions.jsonl"
    offline_input_path = tmp_path / "offline-input.jsonl"
    offline_prediction_path = tmp_path / "offline-predictions.jsonl"
    offline_import_report_path = tmp_path / "offline-import-report.json"
    offline_control_matrix_path = tmp_path / "offline-control-matrix.json"
    lab.save_graph_json(predicted_graph, predicted_graph_path)
    lab.save_qa_predictions((prediction,), prediction_path)
    qa_path = tmp_path / "qa.jsonl"
    lab.save_qa_dataset((case,), qa_path)
    lab.save_offline_prediction_records(
        (
            lab.OfflinePredictionRecord(
                case_id=case.id,
                answer=case.answer,
                confidence=0.88,
            ),
        ),
        offline_input_path,
    )
    imported_predictions, offline_import_report = lab.import_offline_predictions(
        (case,),
        lab.load_offline_prediction_records(offline_input_path),
        source_name="caption_fixture",
        source_kind="caption_memory",
        source_metadata={
            "capabilities": "spatial_qa,dynamic_memory",
            "model_id": "caption-baseline",
            "prompt_id": "caption-v1",
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
        required_source_kinds=("caption_memory",),
    )
    lab.save_offline_control_matrix_report(
        offline_control_matrix,
        offline_control_matrix_path,
    )
    lab.save_predicted_graph_report(
        lab.predicted_graph_report(
            input_path=episode_paths[0],
            graph_path=predicted_graph_path,
            graph=predicted_graph,
            frames=frames,
        ),
        predicted_report_path,
    )
    lab.save_graph_eval_report(
        lab.graph_eval_report(
            graph,
            predicted_graph,
            predicted_path=predicted_graph_path,
        ),
        graph_eval_path,
    )
    lab.save_error_attribution_report(
        lab.error_attribution_report(
            (case,),
            oracle_graph=graph,
            predicted_graph=predicted_graph,
            predictions=(prediction,),
            predicted_graph_path=predicted_graph_path,
            prediction_path=prediction_path,
        ),
        error_attribution_path,
    )
    lab.save_qa_eval_report(report, qa_report_path)
    output_dir = tmp_path / "benchmark"
    manifest_path = tmp_path / "benchmark-manifest.json"

    assert main(
        [
            "--episodes",
            str(episode_paths[0]),
            "--dataset-name",
            "mock_benchmark",
            "--output-dir",
            str(output_dir),
            "--max-qa-per-episode",
            "2",
            "--qa-eval-report",
            str(qa_report_path),
            "--offline-prediction-import-report",
            str(offline_import_report_path),
            "--offline-control-matrix-report",
            str(offline_control_matrix_path),
            "--graph-eval-report",
            str(graph_eval_path),
            "--error-attribution-report",
            str(error_attribution_path),
            "--predicted-graph-report",
            str(predicted_report_path),
            "--manifest",
            str(manifest_path),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    manifest = lab.load_benchmark_manifest(manifest_path)
    assert output["valid"] is True
    assert output["digest"] == manifest["manifest_digest"]
    assert manifest["summary"]["experiment_artifact_count"] == 6
    assert [item["artifact_type"] for item in manifest["experiment_artifacts"]] == [
        "error_attribution_report",
        "graph_eval_report",
        "offline_control_matrix_report",
        "offline_prediction_import_report",
        "predicted_graph_report",
        "qa_eval_report",
    ]


def test_build_benchmark_cli_outputs_validates_compares_and_reports_invalid_json(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_build_benchmark_script()
    main = cast(MainFn, getattr(module, "main"))
    episode_paths = _write_mock_episodes(tmp_path)
    output_dir = tmp_path / "benchmark"
    manifest_path = tmp_path / "benchmark-manifest.json"

    assert main(
        [
            "--episodes",
            str(episode_paths[0]),
            "--episodes",
            str(episode_paths[1]),
            "--dataset-name",
            "mock_benchmark",
            "--output-dir",
            str(output_dir),
            "--max-qa-per-episode",
            "3",
            "--manifest",
            str(manifest_path),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    manifest = lab.load_benchmark_manifest(manifest_path)
    assert output == {
        "action": "build_benchmark",
        "path": str(manifest_path),
        "valid": True,
        "digest": manifest["manifest_digest"],
        "summary": manifest["summary"],
    }

    assert main(["--validate-manifest", str(manifest_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_benchmark_manifest"
    assert validation["valid"] is True

    assert main(["--compare-manifest", str(manifest_path)]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["action"] == "compare_benchmark_manifest"
    assert comparison["matches"] is True

    invalid_path = tmp_path / "invalid-manifest.json"
    invalid_path.write_text("[]\n", encoding="utf-8")
    assert main(["--validate-manifest", str(invalid_path)]) == 1
    invalid = json.loads(capsys.readouterr().out)
    assert invalid == {
        "action": "validate_benchmark_manifest",
        "path": str(invalid_path),
        "valid": False,
        "error": "Benchmark manifest JSON must be an object",
    }


def _write_mock_episodes(tmp_path: Path) -> tuple[Path, Path]:
    episode_dir = tmp_path / "episodes"
    ai2thor_path = episode_dir / "ai2thor.jsonl"
    habitat_path = episode_dir / "habitat.jsonl"
    lab.save_episode_sequence(
        lab.build_mock_ai2thor_episode(
            lab.AI2ThorAdapterConfig(
                scene_id="FloorPlan1",
                episode_id="ai2thor_mock_001",
                steps=(1, 2),
                actions=("Initialize", "MoveAhead"),
            )
        ),
        ai2thor_path,
    )
    lab.save_episode_sequence(
        lab.build_mock_habitat_episode(
            lab.HabitatAdapterConfig(
                scene_id="apartment_0",
                episode_id="habitat_mock_001",
                steps=(1, 2),
                actions=("reset", "turn_left"),
            )
        ),
        habitat_path,
    )
    return ai2thor_path, habitat_path


def _task_for_case(case: lab.QACase, *, max_actions: int) -> lab.ActiveEQATask:
    return lab.ActiveEQATask(
        id=f"active:{case.id}",
        scene_id=case.scene_id,
        episode_id=case.episode_id,
        initial_step=case.step,
        question=case.question,
        gold_answer=case.answer,
        success_conditions={"answer_exact": True},
        max_actions=max_actions,
        required_evidence={
            "nodes": case.required_nodes,
            "edges": case.required_edges,
        },
    )
