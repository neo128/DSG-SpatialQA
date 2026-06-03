from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast

from _pytest.capture import CaptureFixture

import dsg_spatialqa_lab as lab


ROOT = Path(__file__).resolve().parents[1]
READINESS_SCRIPT = ROOT / "scripts" / "check_real_experiment.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_readiness_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_real_experiment_script",
        READINESS_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_real_experiment_readiness_report_accepts_manifest_with_required_evidence(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "real_experiment_readiness_report")
    assert hasattr(lab, "real_experiment_readiness_report_digest")
    assert hasattr(lab, "validate_real_experiment_readiness_report")
    assert hasattr(lab, "compare_real_experiment_readiness_report")
    manifest_path, manifest = _write_ready_manifest(tmp_path)

    report = lab.real_experiment_readiness_report(
        manifest,
        manifest_path=manifest_path,
        declared_data_source_kind="real",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    validation = lab.validate_real_experiment_readiness_report(report)
    comparison = lab.compare_real_experiment_readiness_report(report)
    checks = {check["name"]: check for check in report["checks"]}
    graph_eval_path = _artifact_paths(manifest, "graph_eval_report")[0]
    graph_eval = lab.load_graph_eval_report(graph_eval_path)
    error_attribution_path = _artifact_paths(manifest, "error_attribution_report")[0]
    error_attribution = lab.load_error_attribution_report(error_attribution_path)
    predicted_report_path = _artifact_paths(manifest, "predicted_graph_report")[0]
    predicted_report = lab.load_predicted_graph_report(predicted_report_path)
    predicted_graph_report = predicted_report["graph_report"]
    assert isinstance(predicted_graph_report, dict)
    dashboard_path = _artifact_paths(manifest, "dashboard_bundle")[0]
    dashboard = lab.load_dashboard_bundle(dashboard_path)

    assert report["schema_version"] == "dsg-spatialqa-lab.real-experiment-readiness.v1"
    assert report["manifest_path"] == str(manifest_path)
    assert report["manifest_digest"] == manifest["manifest_digest"]
    assert report["declared_data_source_kind"] == "real"
    assert report["artifact_summary"] == {
        "artifact_counts": {
            "active_task_delta_report": 1,
            "dashboard_bundle": 1,
            "error_attribution_report": 1,
            "graph_eval_report": 1,
            "offline_control_matrix_report": 1,
            "offline_control_result_report": 1,
            "offline_prediction_import_report": 2,
            "predicted_dsg_evidence_report": 1,
            "predicted_graph_report": 1,
            "qa_eval_delta_report": 2,
            "real_collection_report": 1,
        },
        "active_delta_baseline_names": ["direct_answer"],
        "active_delta_candidate_names": ["oracle_evidence"],
        "active_delta_not_ready_paths": [],
        "active_delta_placeholder_name_paths": [],
        "active_delta_ready": True,
        "active_delta_stale_paths": [],
        "active_delta_task_mismatch_paths": [],
        "benchmark_manifest_artifact_digest_invalid_paths": [],
        "benchmark_manifest_artifact_digest_mismatch_paths": [],
        "benchmark_manifest_artifact_digests_current": True,
        "dashboard_bundle_digests": [dashboard["bundle_digest"]],
        "dashboard_bundle_not_ready_paths": [],
        "dashboard_bundle_ready": True,
        "dashboard_bundle_stale_paths": [],
        "dashboard_case_counts": [dashboard["summary"]["case_count"]],
        "error_attribution_gold_digests": [error_attribution["gold_digest"]],
        "error_attribution_not_ready_paths": [],
        "error_attribution_oracle_graph_digests": [
            error_attribution["oracle_graph_digest"]
        ],
        "error_attribution_predicted_graph_digests": [
            error_attribution["predicted_graph_digest"]
        ],
        "error_attribution_prediction_digests": [
            error_attribution["prediction_digest"]
        ],
        "error_attribution_ready": True,
        "error_attribution_stale_paths": [],
        "graph_eval_not_ready_paths": [],
        "graph_eval_oracle_digests": [graph_eval["oracle_digest"]],
        "graph_eval_predicted_digests": [graph_eval["predicted_digest"]],
        "graph_eval_ready": True,
        "graph_eval_stale_paths": [],
        "manifest_qa_digest": manifest["qa_digest"],
        "offline_control_matrix_not_ready_paths": [],
        "offline_control_matrix_ready": True,
        "offline_control_matrix_required_kinds": ["caption_memory", "vlm"],
        "offline_control_matrix_stale_paths": [],
        "offline_control_result_candidate_names": ["graph_tool"],
        "offline_control_result_not_ready_paths": [],
        "offline_control_result_ready": True,
        "offline_control_result_source_kinds": ["caption_memory", "vlm"],
        "offline_control_result_stale_paths": [],
        "offline_control_diagnostic_source_keys": [],
        "offline_control_incomplete_source_keys": [],
        "offline_control_invalid_source_keys": [],
        "offline_control_kinds": ["caption_memory", "vlm"],
        "offline_control_missing_metadata_source_keys": [],
        "offline_control_placeholder_source_keys": [],
        "offline_control_qa_digests": [manifest["qa_digest"]],
        "offline_control_stale_source_keys": [],
        "predicted_dsg_evidence_not_ready_paths": [],
        "predicted_dsg_evidence_predicted_report_digests": [
            predicted_report["digest"]
        ],
        "predicted_dsg_evidence_ready": True,
        "predicted_graph_graph_digests": [predicted_graph_report["digest"]],
        "predicted_graph_not_ready_paths": [],
        "predicted_graph_ready": True,
        "predicted_graph_report_digests": [predicted_report["digest"]],
        "predicted_graph_stale_paths": [],
        "predicted_input_kinds": ["observation_sequence"],
        "qa_delta_baseline_names": ["caption_memory", "vlm"],
        "qa_delta_candidate_names": ["graph_tool"],
        "qa_delta_case_mismatch_paths": [],
        "qa_delta_not_ready_paths": [],
        "qa_delta_placeholder_name_paths": [],
        "qa_delta_ready": True,
        "qa_delta_stale_paths": [],
        "question_type_counts": {
            "object_location": 8,
            "relative_relation": 6,
            "relation_timeline": 4,
            "scene_delta": 4,
        },
        "real_collection_not_ready_paths": [],
        "real_collection_ready": True,
        "real_collection_source_kinds": ["ai2thor"],
        "real_collection_stale_paths": [],
    }
    assert checks["data_source_kind_real"]["passed"] is True
    assert checks["benchmark_manifest_artifact_digests_current"]["passed"] is True
    assert checks["real_collection_report_present"]["passed"] is True
    assert checks["real_collection_ready"]["passed"] is True
    assert checks["offline_control_matrix_report_present"]["passed"] is True
    assert checks["offline_control_matrix_ready"]["passed"] is True
    assert checks["offline_control_result_report_present"]["passed"] is True
    assert checks["offline_control_result_ready"]["passed"] is True
    assert checks["offline_control_result_deltas_cover_controls"]["passed"] is True
    assert checks["offline_control_matrix_required_kinds_cover_controls"][
        "passed"
    ] is True
    assert checks["offline_controls_present"]["actual"] == ["caption_memory", "vlm"]
    assert checks["offline_control_complete_prediction_coverage"]["passed"] is True
    assert checks["offline_control_clean_import_diagnostics"]["passed"] is True
    assert checks["offline_control_metadata_present"]["passed"] is True
    assert checks["offline_control_no_placeholder_sources"]["passed"] is True
    assert checks["offline_control_qa_digest_matches_manifest"]["passed"] is True
    assert checks["offline_control_qa_digest_matches_manifest"]["actual"] == [
        manifest["qa_digest"]
    ]
    assert checks["predicted_observation_graph_present"]["actual"] == [
        "observation_sequence"
    ]
    assert checks["qa_delta_reports_ready"]["passed"] is True
    assert checks["qa_delta_baselines_cover_controls"]["passed"] is True
    assert checks["qa_delta_case_counts_match"]["passed"] is True
    assert checks["qa_delta_non_placeholder_names"]["passed"] is True
    assert checks["active_delta_reports_ready"]["passed"] is True
    assert checks["active_delta_task_counts_match"]["passed"] is True
    assert checks["active_delta_non_placeholder_names"]["passed"] is True
    assert checks["predicted_dsg_evidence_report_present"]["passed"] is True
    assert checks["predicted_dsg_evidence_ready"]["passed"] is True
    assert checks["predicted_dsg_evidence_report_digest_alignment"]["passed"] is True
    assert checks["predicted_graph_reports_ready"]["passed"] is True
    assert checks["graph_eval_reports_ready"]["passed"] is True
    assert checks["error_attribution_reports_ready"]["passed"] is True
    assert checks["graph_error_predicted_digest_alignment"]["passed"] is True
    assert checks["predicted_graph_eval_digest_alignment"]["passed"] is True
    assert checks["predicted_graph_error_digest_alignment"]["passed"] is True
    assert checks["dashboard_bundle_ready"]["passed"] is True
    assert report["readiness"] == {
        "ready": True,
        "passed_count": len(report["checks"]),
        "failed_count": 0,
        "missing_groups": [],
        "failed_checks": [],
    }
    assert report["report_digest"] == lab.real_experiment_readiness_report_digest(report)
    assert validation["valid"] is True
    assert comparison["matches"] is True


def test_real_experiment_readiness_report_rejects_mock_or_incomplete_manifest(
    tmp_path: Path,
) -> None:
    manifest = _incomplete_manifest()

    report = lab.real_experiment_readiness_report(
        manifest,
        declared_data_source_kind="mock",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert report["readiness"]["missing_groups"] == [
        "real_controls",
        "real_data",
        "real_predicted_dsg",
        "review_artifacts",
    ]
    assert checks["data_source_kind_real"]["passed"] is False
    assert checks["qa_count_minimum"]["actual"] == 3
    assert checks["dynamic_memory_coverage"]["actual"] == 0
    assert checks["offline_controls_present"]["missing"] == [
        "caption_memory",
        "vlm",
    ]
    assert checks["predicted_observation_graph_present"]["missing"] == [
        "observation_sequence"
    ]
    assert checks["real_collection_report_present"]["passed"] is False


def test_real_experiment_readiness_report_rejects_offline_controls_on_wrong_qa(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _write_ready_manifest(tmp_path)
    mismatched = dict(manifest)
    mismatched["qa_digest"] = "f" * 64
    mismatched["manifest_digest"] = lab.benchmark_manifest_digest(mismatched)

    report = lab.real_experiment_readiness_report(
        mismatched,
        manifest_path=manifest_path,
        declared_data_source_kind="real",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert "offline_control_qa_digest_matches_manifest" in report["readiness"][
        "failed_checks"
    ]
    assert checks["offline_control_qa_digest_matches_manifest"]["passed"] is False
    assert checks["offline_control_qa_digest_matches_manifest"]["expected"] == "f" * 64
    assert checks["offline_control_qa_digest_matches_manifest"]["actual"] != [
        "f" * 64
    ]


def test_real_experiment_readiness_report_rejects_invalid_offline_prediction_import_report(
    tmp_path: Path,
) -> None:
    _, manifest = _write_ready_manifest(tmp_path)
    import_path = _artifact_paths(manifest, "offline_prediction_import_report")[0]
    import_report = lab.load_offline_prediction_import_report(import_path)
    source = import_report["source"]
    assert isinstance(source, dict)
    source_key = f"{source['kind']}:{source['name']}"
    import_report["report_digest"] = "f" * 64
    import_path.write_text(
        lab.offline_prediction_import_report_json(import_report),
        encoding="utf-8",
    )

    report = lab.real_experiment_readiness_report(
        manifest,
        declared_data_source_kind="real",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert report["artifact_summary"]["offline_control_invalid_source_keys"] == [
        source_key
    ]
    assert "offline_control_import_reports_valid" in report["readiness"][
        "failed_checks"
    ]
    assert checks["offline_control_import_reports_valid"] == {
        "name": "offline_control_import_reports_valid",
        "group": "real_controls",
        "passed": False,
        "expected": [],
        "actual": [source_key],
    }


def test_real_experiment_readiness_report_rejects_stale_offline_prediction_import_report(
    tmp_path: Path,
) -> None:
    _, manifest = _write_ready_manifest(tmp_path)
    import_path = _artifact_paths(manifest, "offline_prediction_import_report")[0]
    import_report = lab.load_offline_prediction_import_report(import_path)
    source_profile = import_report["source_profile"]
    assert isinstance(source_profile, dict)
    source_key = source_profile["source_key"]
    input_path = Path(str(import_report["input_path"]))
    records = tuple(lab.load_offline_prediction_records(input_path))
    assert len(records) > 0
    stale_records = (
        lab.OfflinePredictionRecord(
            case_id=records[0].case_id,
            answer={"stale_answer": True},
        ),
        *records[1:],
    )
    lab.save_offline_prediction_records(stale_records, input_path)

    report = lab.real_experiment_readiness_report(
        manifest,
        declared_data_source_kind="real",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert report["artifact_summary"]["offline_control_stale_source_keys"] == [
        source_key
    ]
    assert report["artifact_summary"]["offline_control_invalid_source_keys"] == [
        source_key
    ]
    assert "offline_control_import_reports_valid" in report["readiness"][
        "failed_checks"
    ]
    assert checks["offline_control_import_reports_valid"]["actual"] == [source_key]


def test_real_experiment_readiness_report_rejects_offline_control_import_gaps(
    tmp_path: Path,
) -> None:
    cases = _offline_import_cases()
    offline_caption_path = _offline_import_report_path_for_cases(
        tmp_path,
        name="caption_fixture",
        kind="caption_memory",
        cases=cases,
        record_case_ids=("case_001",),
    )
    offline_vlm_path = _offline_import_report_path_for_cases(
        tmp_path,
        name="vlm_fixture",
        kind="vlm",
        cases=cases,
        record_case_ids=("case_001", "case_002", "unknown_case"),
    )
    offline_matrix_path = _offline_control_matrix_path(
        tmp_path,
        (offline_caption_path, offline_vlm_path),
        required_source_kinds=("caption_memory", "vlm"),
    )
    predicted_report_path, predicted_evidence_path = _predicted_observation_report_paths(
        tmp_path
    )
    real_collection_report_path = _real_collection_report_path(tmp_path)
    graph_eval_report_path, error_attribution_report_path = _graph_error_report_paths(
        tmp_path,
        predicted_report_path,
    )
    manifest = _ready_manifest(
        offline_vlm_path=offline_vlm_path,
        offline_caption_path=offline_caption_path,
        offline_matrix_path=offline_matrix_path,
        predicted_report_path=predicted_report_path,
        predicted_evidence_path=predicted_evidence_path,
        real_collection_report_path=real_collection_report_path,
        graph_eval_report_path=graph_eval_report_path,
        error_attribution_report_path=error_attribution_report_path,
        qa_delta_paths=_qa_delta_report_paths(tmp_path),
        active_delta_paths=_active_delta_report_paths(tmp_path),
    )

    report = lab.real_experiment_readiness_report(
        manifest,
        declared_data_source_kind="real",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert report["artifact_summary"]["offline_control_incomplete_source_keys"] == [
        "caption_memory:caption_fixture"
    ]
    assert report["artifact_summary"]["offline_control_diagnostic_source_keys"] == [
        "vlm:vlm_fixture"
    ]
    assert "offline_control_complete_prediction_coverage" in report["readiness"][
        "failed_checks"
    ]
    assert "offline_control_clean_import_diagnostics" in report["readiness"][
        "failed_checks"
    ]
    assert checks["offline_control_complete_prediction_coverage"]["actual"] == [
        "caption_memory:caption_fixture"
    ]
    assert checks["offline_control_clean_import_diagnostics"]["actual"] == [
        "vlm:vlm_fixture"
    ]


def test_real_experiment_readiness_report_rejects_matrix_missing_required_control(
    tmp_path: Path,
) -> None:
    offline_vlm_path = _offline_import_report_path(tmp_path, "vlm_fixture", "vlm")
    offline_caption_path = _offline_import_report_path(
        tmp_path,
        "caption_fixture",
        "caption_memory",
    )
    offline_matrix_path = _offline_control_matrix_path(
        tmp_path,
        (offline_caption_path, offline_vlm_path),
        required_source_kinds=("vlm",),
    )
    predicted_report_path, predicted_evidence_path = _predicted_observation_report_paths(
        tmp_path
    )
    real_collection_report_path = _real_collection_report_path(tmp_path)
    manifest = _ready_manifest(
        offline_vlm_path=offline_vlm_path,
        offline_caption_path=offline_caption_path,
        offline_matrix_path=offline_matrix_path,
        predicted_report_path=predicted_report_path,
        predicted_evidence_path=predicted_evidence_path,
        real_collection_report_path=real_collection_report_path,
        qa_delta_paths=_qa_delta_report_paths(tmp_path),
        active_delta_paths=_active_delta_report_paths(tmp_path),
    )

    report = lab.real_experiment_readiness_report(
        manifest,
        declared_data_source_kind="real",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert "offline_control_matrix_required_kinds_cover_controls" in report[
        "readiness"
    ]["failed_checks"]
    assert checks["offline_control_matrix_required_kinds_cover_controls"][
        "missing"
    ] == ["caption_memory"]


def test_real_experiment_readiness_report_rejects_qa_delta_missing_control_baseline(
    tmp_path: Path,
) -> None:
    offline_vlm_path = _offline_import_report_path(
        tmp_path,
        "llava16_ai2thor_trial",
        "vlm",
    )
    offline_caption_path = _offline_import_report_path(
        tmp_path,
        "caption_memory_ai2thor_trial",
        "caption_memory",
    )
    offline_matrix_path = _offline_control_matrix_path(
        tmp_path,
        (offline_caption_path, offline_vlm_path),
        required_source_kinds=("caption_memory", "vlm"),
    )
    predicted_report_path, predicted_evidence_path = _predicted_observation_report_paths(
        tmp_path
    )
    real_collection_report_path = _real_collection_report_path(tmp_path)
    manifest = _ready_manifest(
        offline_vlm_path=offline_vlm_path,
        offline_caption_path=offline_caption_path,
        offline_matrix_path=offline_matrix_path,
        predicted_report_path=predicted_report_path,
        predicted_evidence_path=predicted_evidence_path,
        real_collection_report_path=real_collection_report_path,
        qa_delta_paths=_qa_delta_report_paths(tmp_path, baseline_names=("vlm",)),
        active_delta_paths=_active_delta_report_paths(tmp_path),
    )

    report = lab.real_experiment_readiness_report(
        manifest,
        declared_data_source_kind="real",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert report["artifact_summary"]["qa_delta_baseline_names"] == ["vlm"]
    assert "qa_delta_baselines_cover_controls" in report["readiness"][
        "failed_checks"
    ]
    assert checks["qa_delta_baselines_cover_controls"]["missing"] == [
        "caption_memory"
    ]


def test_real_experiment_readiness_report_rejects_stale_qa_delta_report(
    tmp_path: Path,
) -> None:
    _, manifest = _write_ready_manifest(tmp_path)
    qa_delta_path = _artifact_paths(manifest, "qa_eval_delta_report")[0]
    qa_delta = lab.load_qa_eval_delta_report(qa_delta_path)
    baseline_report_path = Path(str(qa_delta["baseline_report_path"]))
    cases = _offline_import_cases()
    improved_baseline_predictions = tuple(
        lab.QAPrediction(
            id=case.id,
            answer=case.answer,
            evidence_nodes=case.required_nodes,
            evidence_edges=case.required_edges,
            confidence=1.0,
        )
        for case in cases
    )
    improved_baseline_report = lab.qa_eval_report(
        cases,
        improved_baseline_predictions,
    )
    lab.save_qa_eval_report(improved_baseline_report, baseline_report_path)

    report = lab.real_experiment_readiness_report(
        manifest,
        declared_data_source_kind="real",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert report["artifact_summary"]["qa_delta_stale_paths"] == [str(qa_delta_path)]
    assert report["artifact_summary"]["qa_delta_not_ready_paths"] == [
        str(qa_delta_path)
    ]
    assert "qa_delta_reports_ready" in report["readiness"]["failed_checks"]
    assert checks["qa_delta_reports_ready"]["not_ready_paths"] == [str(qa_delta_path)]


def test_real_experiment_readiness_report_rejects_stale_offline_control_matrix_report(
    tmp_path: Path,
) -> None:
    _, manifest = _write_ready_manifest(tmp_path)
    matrix_path = _artifact_paths(manifest, "offline_control_matrix_report")[0]
    import_path = _artifact_paths(manifest, "offline_prediction_import_report")[0]
    import_report = lab.load_offline_prediction_import_report(import_path)
    source_profile = import_report["source_profile"]
    assert isinstance(source_profile, dict)
    source_profile["model_id"] = "updated-real-control-model"
    import_report["report_digest"] = lab.offline_prediction_import_report_digest(
        import_report
    )
    import_path.write_text(
        lab.offline_prediction_import_report_json(import_report),
        encoding="utf-8",
    )

    report = lab.real_experiment_readiness_report(
        manifest,
        declared_data_source_kind="real",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert report["artifact_summary"]["offline_control_matrix_stale_paths"] == [
        str(matrix_path)
    ]
    assert report["artifact_summary"]["offline_control_matrix_not_ready_paths"] == [
        str(matrix_path)
    ]
    assert "offline_control_matrix_ready" in report["readiness"]["failed_checks"]
    assert checks["offline_control_matrix_ready"]["not_ready_paths"] == [
        str(matrix_path)
    ]


def test_real_experiment_readiness_report_rejects_stale_offline_control_result_report(
    tmp_path: Path,
) -> None:
    _, manifest = _write_ready_manifest(tmp_path)
    matrix_path = _artifact_paths(manifest, "offline_control_matrix_report")[0]
    result_path = _artifact_paths(manifest, "offline_control_result_report")[0]
    matrix = lab.load_offline_control_matrix_report(matrix_path)
    source_rows = matrix["source_profile_matrix"]
    assert isinstance(source_rows, list)
    assert isinstance(source_rows[0], dict)
    source_rows[0]["model_id"] = "updated-real-control-model"
    matrix["report_digest"] = lab.offline_control_matrix_report_digest(matrix)
    matrix_path.write_text(
        lab.offline_control_matrix_report_json(matrix),
        encoding="utf-8",
    )

    report = lab.real_experiment_readiness_report(
        manifest,
        declared_data_source_kind="real",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert report["artifact_summary"]["offline_control_result_stale_paths"] == [
        str(result_path)
    ]
    assert report["artifact_summary"]["offline_control_result_not_ready_paths"] == [
        str(result_path)
    ]
    assert "offline_control_result_ready" in report["readiness"]["failed_checks"]
    assert checks["offline_control_result_ready"]["not_ready_paths"] == [
        str(result_path)
    ]


def test_real_experiment_readiness_report_rejects_stale_active_delta_report(
    tmp_path: Path,
) -> None:
    _, manifest = _write_ready_manifest(tmp_path)
    active_delta_path = _artifact_paths(manifest, "active_task_delta_report")[0]
    active_delta = lab.load_active_task_delta_report(active_delta_path)
    baseline_report_path = Path(str(active_delta["baseline_report_path"]))
    case = _offline_import_cases()[0]
    task = lab.ActiveEQATask(
        id=f"active:{case.id}",
        scene_id=case.scene_id,
        episode_id=case.episode_id,
        initial_step=case.step,
        question=case.question,
        gold_answer=case.answer,
        success_conditions={"answer_exact": True},
        max_actions=1,
        required_evidence={
            "nodes": case.required_nodes,
            "edges": case.required_edges,
        },
    )
    improved_baseline_result = lab.ActiveTaskResult(
        task_id=task.id,
        policy=str(active_delta["baseline_name"]),
        answer=case.answer,
        success=True,
        action_count=1,
        evidence_nodes=case.required_nodes,
        evidence_edges=case.required_edges,
        final_step=case.step + 1,
        confidence=1.0,
    )
    improved_baseline_report = lab.active_task_report(
        (task,),
        (improved_baseline_result,),
    )
    lab.save_active_task_report(improved_baseline_report, baseline_report_path)

    report = lab.real_experiment_readiness_report(
        manifest,
        declared_data_source_kind="real",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert report["artifact_summary"]["active_delta_stale_paths"] == [
        str(active_delta_path)
    ]
    assert report["artifact_summary"]["active_delta_not_ready_paths"] == [
        str(active_delta_path)
    ]
    assert "active_delta_reports_ready" in report["readiness"]["failed_checks"]
    assert checks["active_delta_reports_ready"]["not_ready_paths"] == [
        str(active_delta_path)
    ]


def test_real_experiment_readiness_report_rejects_placeholder_active_delta_names(
    tmp_path: Path,
) -> None:
    offline_vlm_path = _offline_import_report_path(
        tmp_path,
        "llava16_ai2thor_trial",
        "vlm",
    )
    offline_caption_path = _offline_import_report_path(
        tmp_path,
        "caption_memory_ai2thor_trial",
        "caption_memory",
    )
    offline_matrix_path = _offline_control_matrix_path(
        tmp_path,
        (offline_caption_path, offline_vlm_path),
        required_source_kinds=("caption_memory", "vlm"),
    )
    predicted_report_path, predicted_evidence_path = _predicted_observation_report_paths(
        tmp_path
    )
    real_collection_report_path = _real_collection_report_path(tmp_path)
    manifest = _ready_manifest(
        offline_vlm_path=offline_vlm_path,
        offline_caption_path=offline_caption_path,
        offline_matrix_path=offline_matrix_path,
        predicted_report_path=predicted_report_path,
        predicted_evidence_path=predicted_evidence_path,
        real_collection_report_path=real_collection_report_path,
        qa_delta_paths=_qa_delta_report_paths(tmp_path),
        active_delta_paths=_active_delta_report_paths(
            tmp_path,
            candidate_name="candidate",
            baseline_name="baseline",
        ),
    )

    report = lab.real_experiment_readiness_report(
        manifest,
        declared_data_source_kind="real",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert report["artifact_summary"]["active_delta_candidate_names"] == ["candidate"]
    assert report["artifact_summary"]["active_delta_baseline_names"] == ["baseline"]
    assert "active_delta_non_placeholder_names" in report["readiness"][
        "failed_checks"
    ]
    assert checks["active_delta_non_placeholder_names"]["actual"] == [
        str(tmp_path / "active-delta-baseline.json")
    ]


def test_real_experiment_readiness_report_rejects_placeholder_offline_control_metadata(
    tmp_path: Path,
) -> None:
    offline_vlm_path = _offline_import_report_path(
        tmp_path,
        "vlm_fixture",
        "vlm",
        source_metadata={"capabilities": ("spatial_qa",), "model_id": "mock-vlm"},
    )
    offline_caption_path = _offline_import_report_path(
        tmp_path,
        "caption_fixture",
        "caption_memory",
        source_metadata={
            "capabilities": ("spatial_qa",),
            "model_id": "fixture-caption-memory",
        },
    )
    offline_matrix_path = _offline_control_matrix_path(
        tmp_path,
        (offline_caption_path, offline_vlm_path),
        required_source_kinds=("caption_memory", "vlm"),
    )
    predicted_report_path, predicted_evidence_path = _predicted_observation_report_paths(
        tmp_path
    )
    real_collection_report_path = _real_collection_report_path(tmp_path)
    manifest = _ready_manifest(
        offline_vlm_path=offline_vlm_path,
        offline_caption_path=offline_caption_path,
        offline_matrix_path=offline_matrix_path,
        predicted_report_path=predicted_report_path,
        predicted_evidence_path=predicted_evidence_path,
        real_collection_report_path=real_collection_report_path,
        qa_delta_paths=_qa_delta_report_paths(tmp_path),
        active_delta_paths=_active_delta_report_paths(tmp_path),
    )

    report = lab.real_experiment_readiness_report(
        manifest,
        declared_data_source_kind="real",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    checks = {check["name"]: check for check in report["checks"]}

    expected_source_keys = ["caption_memory:caption_fixture", "vlm:vlm_fixture"]
    assert report["readiness"]["ready"] is False
    assert report["artifact_summary"][
        "offline_control_missing_metadata_source_keys"
    ] == expected_source_keys
    assert report["artifact_summary"][
        "offline_control_placeholder_source_keys"
    ] == expected_source_keys
    assert "offline_control_metadata_present" in report["readiness"][
        "failed_checks"
    ]
    assert "offline_control_no_placeholder_sources" in report["readiness"][
        "failed_checks"
    ]
    assert checks["offline_control_metadata_present"]["actual"] == expected_source_keys
    assert checks["offline_control_no_placeholder_sources"][
        "actual"
    ] == expected_source_keys


def test_real_experiment_readiness_report_rejects_invalid_graph_eval_report(
    tmp_path: Path,
) -> None:
    _, manifest = _write_ready_manifest(tmp_path)
    graph_eval_path = _artifact_paths(manifest, "graph_eval_report")[0]
    graph_eval = lab.load_graph_eval_report(graph_eval_path)
    graph_eval["summary"]["matched_object_count"] = 999
    graph_eval_path.write_text(
        lab.graph_eval_report_json(graph_eval),
        encoding="utf-8",
    )

    report = lab.real_experiment_readiness_report(
        manifest,
        declared_data_source_kind="real",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert report["artifact_summary"]["graph_eval_not_ready_paths"] == [
        str(graph_eval_path)
    ]
    assert "graph_eval_reports_ready" in report["readiness"]["failed_checks"]
    assert checks["graph_eval_reports_ready"]["not_ready_paths"] == [
        str(graph_eval_path)
    ]


def test_real_experiment_readiness_report_rejects_stale_graph_eval_report(
    tmp_path: Path,
) -> None:
    _, manifest = _write_ready_manifest(tmp_path)
    graph_eval_path = _artifact_paths(manifest, "graph_eval_report")[0]
    graph_eval = lab.load_graph_eval_report(graph_eval_path)
    oracle_graph_path = Path(str(graph_eval["oracle_path"]))
    oracle_graph = lab.load_graph_json(oracle_graph_path)
    oracle_graph.upsert_object(
        "late_real_object",
        "spoon",
        lab.Pose3D(1.8, 0.4, 0.8),
        lab.BBox3D(
            center=lab.Pose3D(1.8, 0.4, 0.8),
            size=(0.1, 0.05, 0.03),
        ),
        confidence=0.88,
        visible=True,
        step=1,
    )
    lab.save_graph_json(oracle_graph, oracle_graph_path)

    report = lab.real_experiment_readiness_report(
        manifest,
        declared_data_source_kind="real",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert report["artifact_summary"]["graph_eval_stale_paths"] == [
        str(graph_eval_path)
    ]
    assert report["artifact_summary"]["graph_eval_not_ready_paths"] == [
        str(graph_eval_path)
    ]
    assert "graph_eval_reports_ready" in report["readiness"]["failed_checks"]
    assert checks["graph_eval_reports_ready"]["not_ready_paths"] == [
        str(graph_eval_path)
    ]


def test_real_experiment_readiness_report_rejects_invalid_error_attribution_report(
    tmp_path: Path,
) -> None:
    _, manifest = _write_ready_manifest(tmp_path)
    error_attribution_path = _artifact_paths(manifest, "error_attribution_report")[0]
    error_attribution = lab.load_error_attribution_report(error_attribution_path)
    error_attribution["summary"]["case_count"] = 999
    error_attribution_path.write_text(
        lab.error_attribution_report_json(error_attribution),
        encoding="utf-8",
    )

    report = lab.real_experiment_readiness_report(
        manifest,
        declared_data_source_kind="real",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert report["artifact_summary"]["error_attribution_not_ready_paths"] == [
        str(error_attribution_path)
    ]
    assert "error_attribution_reports_ready" in report["readiness"]["failed_checks"]
    assert checks["error_attribution_reports_ready"]["not_ready_paths"] == [
        str(error_attribution_path)
    ]


def test_real_experiment_readiness_report_rejects_stale_error_attribution_report(
    tmp_path: Path,
) -> None:
    _, manifest = _write_ready_manifest(tmp_path)
    error_attribution_path = _artifact_paths(manifest, "error_attribution_report")[0]
    error_attribution = lab.load_error_attribution_report(error_attribution_path)
    prediction_path = Path(str(error_attribution["prediction_path"]))
    predictions = lab.load_qa_predictions(prediction_path)
    assert predictions
    stale_predictions = list(predictions)
    first_prediction = stale_predictions[0]
    stale_predictions[0] = lab.QAPrediction(
        id=first_prediction.id,
        answer={"stale_answer": True},
        confidence=0.0,
    )
    lab.save_qa_predictions(tuple(stale_predictions), prediction_path)

    report = lab.real_experiment_readiness_report(
        manifest,
        declared_data_source_kind="real",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert report["artifact_summary"]["error_attribution_stale_paths"] == [
        str(error_attribution_path)
    ]
    assert report["artifact_summary"]["error_attribution_not_ready_paths"] == [
        str(error_attribution_path)
    ]
    assert "error_attribution_reports_ready" in report["readiness"]["failed_checks"]
    assert checks["error_attribution_reports_ready"]["not_ready_paths"] == [
        str(error_attribution_path)
    ]


def test_real_experiment_readiness_report_rejects_invalid_predicted_graph_report(
    tmp_path: Path,
) -> None:
    _, manifest = _write_ready_manifest(tmp_path)
    predicted_report_path = _artifact_paths(manifest, "predicted_graph_report")[0]
    predicted_report = lab.load_predicted_graph_report(predicted_report_path)
    predicted_report["digest"] = "f" * 64
    predicted_report_path.write_text(
        lab.predicted_graph_report_json(predicted_report),
        encoding="utf-8",
    )

    report = lab.real_experiment_readiness_report(
        manifest,
        declared_data_source_kind="real",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert report["artifact_summary"]["predicted_graph_not_ready_paths"] == [
        str(predicted_report_path)
    ]
    assert "predicted_graph_reports_ready" in report["readiness"]["failed_checks"]
    assert checks["predicted_graph_reports_ready"]["not_ready_paths"] == [
        str(predicted_report_path)
    ]


def test_real_experiment_readiness_report_rejects_stale_predicted_graph_report(
    tmp_path: Path,
) -> None:
    _, manifest = _write_ready_manifest(tmp_path)
    predicted_report_path = _artifact_paths(manifest, "predicted_graph_report")[0]
    predicted_report = lab.load_predicted_graph_report(predicted_report_path)
    observation_path = Path(str(predicted_report["path"]))
    observations = tuple(lab.load_scene_observation_sequence(observation_path))
    assert observations
    first_observation = observations[0]
    assert first_observation.objects
    first_object = first_observation.objects[0]
    stale_object = lab.ObjectObservation(
        object_id=first_object.object_id,
        label=first_object.label,
        pose=lab.Pose3D(9.0, 9.0, 9.0),
        bbox=lab.BBox3D(
            center=lab.Pose3D(9.0, 9.0, 9.0),
            size=first_object.bbox.size,
        ),
        confidence=first_object.confidence,
        visible=first_object.visible,
        attributes=first_object.attributes,
    )
    stale_observations = (
        lab.SceneObservation(
            step=first_observation.step,
            agent_pose=first_observation.agent_pose,
            agent_id=first_observation.agent_id,
            rooms=first_observation.rooms,
            regions=first_observation.regions,
            objects=(stale_object, *first_observation.objects[1:]),
        ),
        *observations[1:],
    )
    lab.save_scene_observation_sequence(stale_observations, observation_path)

    report = lab.real_experiment_readiness_report(
        manifest,
        declared_data_source_kind="real",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert report["artifact_summary"]["predicted_graph_stale_paths"] == [
        str(predicted_report_path)
    ]
    assert report["artifact_summary"]["predicted_graph_not_ready_paths"] == [
        str(predicted_report_path)
    ]
    assert "predicted_graph_reports_ready" in report["readiness"]["failed_checks"]
    assert checks["predicted_graph_reports_ready"]["not_ready_paths"] == [
        str(predicted_report_path)
    ]


def test_real_experiment_readiness_report_rejects_invalid_real_collection_report(
    tmp_path: Path,
) -> None:
    _, manifest = _write_ready_manifest(tmp_path)
    real_collection_path = _artifact_paths(manifest, "real_collection_report")[0]
    real_collection = lab.load_real_collection_report(real_collection_path)
    real_collection["report_digest"] = "f" * 64
    real_collection_path.write_text(
        lab.real_collection_report_json(real_collection),
        encoding="utf-8",
    )

    report = lab.real_experiment_readiness_report(
        manifest,
        declared_data_source_kind="real",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert report["artifact_summary"]["real_collection_not_ready_paths"] == [
        str(real_collection_path)
    ]
    assert "real_collection_ready" in report["readiness"]["failed_checks"]
    assert checks["real_collection_ready"]["not_ready_paths"] == [
        str(real_collection_path)
    ]


def test_real_experiment_readiness_report_rejects_stale_real_collection_report(
    tmp_path: Path,
) -> None:
    _, manifest = _write_ready_manifest(tmp_path)
    real_collection_path = _artifact_paths(manifest, "real_collection_report")[0]
    real_collection = lab.load_real_collection_report(real_collection_path)
    episode_path = Path(str(real_collection["episode_paths"][0]))
    frames = lab.load_episode_sequence(episode_path)
    assert len(frames) == 2
    lab.save_episode_sequence(frames[:1], episode_path)

    report = lab.real_experiment_readiness_report(
        manifest,
        declared_data_source_kind="real",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert report["artifact_summary"]["real_collection_stale_paths"] == [
        str(real_collection_path)
    ]
    assert report["artifact_summary"]["real_collection_not_ready_paths"] == [
        str(real_collection_path)
    ]
    assert "real_collection_ready" in report["readiness"]["failed_checks"]
    assert checks["real_collection_ready"]["not_ready_paths"] == [
        str(real_collection_path)
    ]


def test_real_experiment_readiness_report_rejects_invalid_predicted_dsg_evidence_report(
    tmp_path: Path,
) -> None:
    _, manifest = _write_ready_manifest(tmp_path)
    predicted_evidence_path = _artifact_paths(
        manifest,
        "predicted_dsg_evidence_report",
    )[0]
    predicted_evidence = lab.load_predicted_dsg_evidence_report(
        predicted_evidence_path
    )
    predicted_evidence["report_digest"] = "f" * 64
    predicted_evidence_path.write_text(
        lab.predicted_dsg_evidence_report_json(predicted_evidence),
        encoding="utf-8",
    )

    report = lab.real_experiment_readiness_report(
        manifest,
        declared_data_source_kind="real",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert report["artifact_summary"]["predicted_dsg_evidence_not_ready_paths"] == [
        str(predicted_evidence_path)
    ]
    assert "predicted_dsg_evidence_ready" in report["readiness"]["failed_checks"]
    assert checks["predicted_dsg_evidence_ready"]["not_ready_paths"] == [
        str(predicted_evidence_path)
    ]


def test_real_experiment_readiness_report_rejects_mismatched_predicted_dsg_evidence_digest(
    tmp_path: Path,
) -> None:
    _, manifest = _write_ready_manifest(tmp_path)
    predicted_evidence_path = _artifact_paths(
        manifest,
        "predicted_dsg_evidence_report",
    )[0]
    predicted_report_path = _artifact_paths(manifest, "predicted_graph_report")[0]
    predicted_report = lab.load_predicted_graph_report(predicted_report_path)
    predicted_report_digest = lab.predicted_graph_report_digest(predicted_report)
    predicted_evidence = lab.load_predicted_dsg_evidence_report(
        predicted_evidence_path
    )
    predicted_evidence["predicted_graph_report_digest"] = "f" * 64
    predicted_evidence["report_digest"] = lab.predicted_dsg_evidence_report_digest(
        predicted_evidence
    )
    predicted_evidence_path.write_text(
        lab.predicted_dsg_evidence_report_json(predicted_evidence),
        encoding="utf-8",
    )

    report = lab.real_experiment_readiness_report(
        manifest,
        declared_data_source_kind="real",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert report["artifact_summary"][
        "predicted_dsg_evidence_predicted_report_digests"
    ] == ["f" * 64]
    assert report["artifact_summary"]["predicted_dsg_evidence_not_ready_paths"] == [
        str(predicted_evidence_path)
    ]
    assert "predicted_dsg_evidence_report_digest_alignment" in report["readiness"][
        "failed_checks"
    ]
    alignment_check = checks["predicted_dsg_evidence_report_digest_alignment"]
    assert alignment_check["expected"] == [predicted_report_digest]
    assert alignment_check["actual"] == ["f" * 64]
    assert alignment_check["passed"] is False
    assert alignment_check["group"] == "real_predicted_dsg"
    assert alignment_check["differences"] == [
        {
            "path": "[0]",
            "expected": predicted_report_digest,
            "actual": "f" * 64,
        }
    ]


def test_real_experiment_readiness_report_rejects_diagnostics_on_wrong_predicted_graph(
    tmp_path: Path,
) -> None:
    _, manifest = _write_ready_manifest(tmp_path)
    wrong_predicted_digest = "f" * 64
    graph_eval_path = _artifact_paths(manifest, "graph_eval_report")[0]
    graph_eval = lab.load_graph_eval_report(graph_eval_path)
    graph_eval["predicted_digest"] = wrong_predicted_digest
    graph_eval["report_digest"] = lab.graph_eval_report_digest(graph_eval)
    graph_eval_path.write_text(
        lab.graph_eval_report_json(graph_eval),
        encoding="utf-8",
    )
    error_attribution_path = _artifact_paths(manifest, "error_attribution_report")[0]
    error_attribution = lab.load_error_attribution_report(error_attribution_path)
    error_attribution["predicted_graph_digest"] = wrong_predicted_digest
    error_attribution["report_digest"] = lab.error_attribution_report_digest(
        error_attribution
    )
    error_attribution_path.write_text(
        lab.error_attribution_report_json(error_attribution),
        encoding="utf-8",
    )

    report = lab.real_experiment_readiness_report(
        manifest,
        declared_data_source_kind="real",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert checks["graph_error_predicted_digest_alignment"]["passed"] is True
    assert "predicted_graph_eval_digest_alignment" in report["readiness"][
        "failed_checks"
    ]
    assert "predicted_graph_error_digest_alignment" in report["readiness"][
        "failed_checks"
    ]


def test_real_experiment_readiness_report_rejects_invalid_dashboard_bundle(
    tmp_path: Path,
) -> None:
    _, manifest = _write_ready_manifest(tmp_path)
    dashboard_path = _artifact_paths(manifest, "dashboard_bundle")[0]
    dashboard = lab.load_dashboard_bundle(dashboard_path)
    dashboard["summary"]["case_count"] = 999
    dashboard_path.write_text(
        lab.dashboard_bundle_json(dashboard),
        encoding="utf-8",
    )

    report = lab.real_experiment_readiness_report(
        manifest,
        declared_data_source_kind="real",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert report["artifact_summary"]["dashboard_bundle_not_ready_paths"] == [
        str(dashboard_path)
    ]
    assert "dashboard_bundle_ready" in report["readiness"]["failed_checks"]
    assert checks["dashboard_bundle_ready"]["not_ready_paths"] == [str(dashboard_path)]


def test_real_experiment_readiness_report_rejects_stale_dashboard_bundle(
    tmp_path: Path,
) -> None:
    _, manifest = _write_ready_manifest(tmp_path)
    dashboard_path = _artifact_paths(manifest, "dashboard_bundle")[0]
    dashboard = lab.load_dashboard_bundle(dashboard_path)
    source_paths = dashboard["source_paths"]
    prediction_path = Path(str(source_paths["prediction_path"]))
    case = _offline_import_cases()[0]
    lab.save_qa_predictions(
        (
            lab.QAPrediction(
                id=case.id,
                answer={"changed": True},
                confidence=0.1,
            ),
        ),
        prediction_path,
    )

    report = lab.real_experiment_readiness_report(
        manifest,
        declared_data_source_kind="real",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert report["artifact_summary"]["dashboard_bundle_stale_paths"] == [
        str(dashboard_path)
    ]
    assert report["artifact_summary"]["dashboard_bundle_not_ready_paths"] == [
        str(dashboard_path)
    ]
    assert "dashboard_bundle_ready" in report["readiness"]["failed_checks"]
    assert checks["dashboard_bundle_ready"]["not_ready_paths"] == [str(dashboard_path)]


def test_real_experiment_readiness_report_rejects_stale_manifest_artifact_digest(
    tmp_path: Path,
) -> None:
    _, manifest = _write_ready_manifest(tmp_path)
    dashboard_path = _artifact_paths(manifest, "dashboard_bundle")[0]
    experiment_artifacts = cast(list[dict[str, Any]], manifest["experiment_artifacts"])
    for artifact in experiment_artifacts:
        if artifact["artifact_type"] == "dashboard_bundle":
            artifact["digest"] = "0" * 64
    experiment_artifact_digests = cast(
        dict[str, str],
        manifest["experiment_artifact_digests"],
    )
    experiment_artifact_digests[f"dashboard_bundle:{dashboard_path.name}"] = "0" * 64
    manifest["manifest_digest"] = lab.benchmark_manifest_digest(manifest)

    report = lab.real_experiment_readiness_report(
        manifest,
        declared_data_source_kind="real",
        min_episode_count=2,
        min_scene_count=1,
        min_qa_count=20,
        required_control_kinds=("caption_memory", "vlm"),
        required_predicted_input_kinds=("observation_sequence",),
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert report["artifact_summary"]["dashboard_bundle_ready"] is True
    assert report["artifact_summary"][
        "benchmark_manifest_artifact_digest_mismatch_paths"
    ] == [str(dashboard_path)]
    assert "benchmark_manifest_artifact_digests_current" in report["readiness"][
        "failed_checks"
    ]
    assert checks["benchmark_manifest_artifact_digests_current"]["actual"] == [
        str(dashboard_path)
    ]


def test_real_experiment_readiness_cli_writes_validates_and_compares_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_readiness_script()
    main = getattr(module, "main")
    manifest_path, _ = _write_ready_manifest(tmp_path)
    report_path = tmp_path / "real-readiness.json"

    assert main(
        [
            "--manifest",
            str(manifest_path),
            "--report",
            str(report_path),
            "--data-source-kind",
            "real",
            "--min-episode-count",
            "2",
            "--min-scene-count",
            "1",
            "--min-qa-count",
            "20",
            "--required-control-kind",
            "caption_memory",
            "--required-control-kind",
            "vlm",
            "--required-predicted-input-kind",
            "observation_sequence",
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["action"] == "real_experiment_readiness"
    assert output["path"] == str(report_path)
    assert output["ready"] is True
    assert report_path.exists()

    assert main(["--validate-report", str(report_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_real_experiment_readiness_report"
    assert validation["valid"] is True

    assert main(["--compare-report", str(report_path)]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["action"] == "compare_real_experiment_readiness_report"
    assert comparison["matches"] is True


def _write_ready_manifest(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    offline_vlm_path = _offline_import_report_path(
        tmp_path,
        "llava16_ai2thor_trial",
        "vlm",
    )
    offline_caption_path = _offline_import_report_path(
        tmp_path,
        "caption_memory_ai2thor_trial",
        "caption_memory",
    )
    offline_matrix_path = _offline_control_matrix_path(
        tmp_path,
        (offline_caption_path, offline_vlm_path),
        required_source_kinds=("caption_memory", "vlm"),
    )
    predicted_report_path, predicted_evidence_path = _predicted_observation_report_paths(
        tmp_path
    )
    real_collection_report_path = _real_collection_report_path(tmp_path)
    manifest = _ready_manifest(
        offline_vlm_path=offline_vlm_path,
        offline_caption_path=offline_caption_path,
        offline_matrix_path=offline_matrix_path,
        predicted_report_path=predicted_report_path,
        predicted_evidence_path=predicted_evidence_path,
        real_collection_report_path=real_collection_report_path,
        qa_delta_paths=_qa_delta_report_paths(tmp_path),
        active_delta_paths=_active_delta_report_paths(tmp_path),
    )
    manifest_path = tmp_path / "benchmark-manifest.json"
    lab.save_benchmark_manifest(manifest, manifest_path)
    return manifest_path, manifest


def _ready_manifest(
    *,
    offline_vlm_path: Path,
    offline_caption_path: Path,
    offline_matrix_path: Path,
    predicted_report_path: Path,
    predicted_evidence_path: Path,
    real_collection_report_path: Path,
    qa_delta_paths: tuple[Path, ...],
    active_delta_paths: tuple[Path, ...],
    graph_eval_report_path: Path | None = None,
    error_attribution_report_path: Path | None = None,
    dashboard_bundle_path: Path | None = None,
) -> dict[str, object]:
    if graph_eval_report_path is None or error_attribution_report_path is None:
        generated_graph_eval_path, generated_error_attribution_path = (
            _graph_error_report_paths(predicted_report_path.parent, predicted_report_path)
        )
        graph_eval_report_path = graph_eval_report_path or generated_graph_eval_path
        error_attribution_report_path = (
            error_attribution_report_path or generated_error_attribution_path
        )
    dashboard_bundle_path = dashboard_bundle_path or _dashboard_bundle_path(
        predicted_report_path.parent,
        predicted_report_path,
        error_attribution_report_path,
    )
    offline_result_path = _offline_control_result_report_path(
        predicted_report_path.parent,
        offline_matrix_path,
        qa_delta_paths,
    )
    artifacts = [
        *(
            _experiment_artifact(
                "active_task_delta_report",
                str(active_delta_path),
                digest=lab.active_task_delta_report_digest(
                    lab.load_active_task_delta_report(active_delta_path)
                ),
            )
            for active_delta_path in active_delta_paths
        ),
        _experiment_artifact(
            "dashboard_bundle",
            str(dashboard_bundle_path),
            digest=lab.dashboard_bundle_digest(
                lab.load_dashboard_bundle(dashboard_bundle_path)
            ),
        ),
        _experiment_artifact(
            "error_attribution_report",
            str(error_attribution_report_path),
            digest=lab.error_attribution_report_digest(
                lab.load_error_attribution_report(error_attribution_report_path)
            ),
        ),
        _experiment_artifact(
            "graph_eval_report",
            str(graph_eval_report_path),
            digest=lab.graph_eval_report_digest(
                lab.load_graph_eval_report(graph_eval_report_path)
            ),
        ),
        _experiment_artifact(
            "offline_control_matrix_report",
            str(offline_matrix_path),
            digest=lab.offline_control_matrix_report_digest(
                lab.load_offline_control_matrix_report(offline_matrix_path)
            ),
        ),
        _experiment_artifact(
            "offline_control_result_report",
            str(offline_result_path),
            digest=lab.offline_control_result_report_digest(
                lab.load_offline_control_result_report(offline_result_path)
            ),
        ),
        _experiment_artifact(
            "offline_prediction_import_report",
            str(offline_caption_path),
            digest=lab.offline_prediction_import_report_digest(
                lab.load_offline_prediction_import_report(offline_caption_path)
            ),
        ),
        _experiment_artifact(
            "offline_prediction_import_report",
            str(offline_vlm_path),
            digest=lab.offline_prediction_import_report_digest(
                lab.load_offline_prediction_import_report(offline_vlm_path)
            ),
        ),
        _experiment_artifact(
            "predicted_dsg_evidence_report",
            str(predicted_evidence_path),
            digest=lab.predicted_dsg_evidence_report_digest(
                lab.load_predicted_dsg_evidence_report(predicted_evidence_path)
            ),
        ),
        _experiment_artifact(
            "predicted_graph_report",
            str(predicted_report_path),
            digest=lab.predicted_graph_report_digest(
                lab.load_predicted_graph_report(predicted_report_path)
            ),
        ),
        *(
            _experiment_artifact(
                "qa_eval_delta_report",
                str(qa_delta_path),
                digest=lab.qa_eval_delta_report_digest(
                    lab.load_qa_eval_delta_report(qa_delta_path)
                ),
            )
            for qa_delta_path in qa_delta_paths
        ),
        _experiment_artifact(
            "real_collection_report",
            str(real_collection_report_path),
            digest=lab.real_collection_report_digest(
                lab.load_real_collection_report(real_collection_report_path)
            ),
        ),
    ]
    qa_digest = _offline_import_qa_digest(offline_vlm_path)
    manifest: dict[str, object] = {
        "schema_version": "dsg-spatialqa-lab.benchmark-manifest.v1",
        "dataset_name": "ai2thor_small_real",
        "scene_count": 2,
        "episode_count": 3,
        "qa_count": 22,
        "task_count": 1,
        "graph_digests": {},
        "qa_digest": qa_digest,
        "qa_dataset_digests": {"ai2thor_real_001": qa_digest},
        "filters": {
            "max_qa_per_episode": None,
            "source": "oracle",
            "tags": ["benchmark", "real"],
        },
        "coverage": {
            "by_question_type": {
                "object_location": 8,
                "relative_relation": 6,
                "relation_timeline": 4,
                "scene_delta": 4,
            },
            "by_scene": {"FloorPlan1": 12, "FloorPlan2": 10},
            "by_episode": {
                "ai2thor_real_001": 8,
                "ai2thor_real_002": 7,
                "ai2thor_real_003": 7,
            },
            "by_reference_frame": {"none": 12, "world": 10},
            "by_tag": {"benchmark": 22, "real": 22},
            "dynamic_static": {"dynamic": 8, "static": 14},
            "oracle_predicted": {"oracle": 3, "predicted": 1},
        },
        "summary": {
            "dataset_name": "ai2thor_small_real",
            "episode_count": 3,
            "qa_count": 22,
            "scene_count": 2,
            "task_count": 1,
            "experiment_artifact_count": len(artifacts),
        },
        "artifacts": [],
        "experiment_artifacts": artifacts,
        "experiment_artifact_digests": {
            f"{artifact['artifact_type']}:{Path(str(artifact['path'])).name}": artifact[
                "digest"
            ]
            for artifact in artifacts
        },
    }
    manifest["manifest_digest"] = lab.benchmark_manifest_digest(manifest)
    return manifest


def _offline_import_qa_digest(path: Path) -> str:
    report = lab.load_offline_prediction_import_report(path)
    qa_digest = report["qa_digest"]
    assert isinstance(qa_digest, str)
    return qa_digest


def _offline_control_matrix_path(
    tmp_path: Path,
    import_report_paths: tuple[Path, ...],
    *,
    required_source_kinds: tuple[str, ...],
) -> Path:
    reports = tuple(
        lab.load_offline_prediction_import_report(path) for path in import_report_paths
    )
    matrix = lab.offline_control_matrix_report(
        reports,
        report_paths=import_report_paths,
        required_source_kinds=required_source_kinds,
    )
    matrix_path = tmp_path / "offline-control-matrix.json"
    lab.save_offline_control_matrix_report(matrix, matrix_path)
    return matrix_path


def _graph_error_report_paths(
    tmp_path: Path,
    predicted_report_path: Path,
) -> tuple[Path, Path]:
    report_dir = tmp_path / "readiness-reports"
    predicted_report = lab.load_predicted_graph_report(predicted_report_path)
    predicted_graph_path = Path(str(predicted_report["graph_path"]))
    predicted_graph = lab.load_graph_json(predicted_graph_path)
    oracle_graph_path = report_dir / "oracle-graph.json"
    qa_path = report_dir / "qa.jsonl"
    prediction_path = report_dir / "graph-tool-predictions.jsonl"
    graph_eval_path = report_dir / "graph-eval-report.json"
    error_attribution_path = report_dir / "error-attribution-report.json"
    cases = _offline_import_cases()
    predictions = tuple(
        lab.QAPrediction(
            id=case.id,
            answer=case.answer,
            evidence_nodes=case.required_nodes,
            evidence_edges=case.required_edges,
            confidence=1.0,
        )
        for case in cases
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    lab.save_graph_json(predicted_graph, oracle_graph_path)
    lab.save_qa_dataset(cases, qa_path)
    lab.save_qa_predictions(predictions, prediction_path)
    graph_eval = lab.graph_eval_report(
        predicted_graph,
        predicted_graph,
        oracle_path=oracle_graph_path,
        predicted_path=predicted_graph_path,
    )
    error_attribution = lab.error_attribution_report(
        cases,
        oracle_graph=predicted_graph,
        predicted_graph=predicted_graph,
        predictions=predictions,
        gold_path=qa_path,
        oracle_graph_path=oracle_graph_path,
        predicted_graph_path=predicted_graph_path,
        prediction_path=prediction_path,
    )
    lab.save_graph_eval_report(graph_eval, graph_eval_path)
    lab.save_error_attribution_report(error_attribution, error_attribution_path)
    return graph_eval_path, error_attribution_path


def _dashboard_bundle_path(
    tmp_path: Path,
    predicted_report_path: Path,
    error_attribution_report_path: Path,
) -> Path:
    predicted_report = lab.load_predicted_graph_report(predicted_report_path)
    predicted_graph = lab.load_graph_json(Path(str(predicted_report["graph_path"])))
    error_attribution = lab.load_error_attribution_report(error_attribution_report_path)
    cases = _offline_import_cases()
    predictions = tuple(
        lab.QAPrediction(
            id=case.id,
            answer=case.answer,
            evidence_nodes=case.required_nodes,
            evidence_edges=case.required_edges,
            confidence=1.0,
        )
        for case in cases
    )
    qa_path = Path(str(error_attribution["gold_path"]))
    prediction_path = Path(str(error_attribution["prediction_path"]))
    qa_eval_path = tmp_path / "readiness-reports" / "dashboard-qa-eval-report.json"
    graph_path = Path(str(predicted_report["graph_path"]))
    qa_eval = lab.qa_eval_report(
        cases,
        predictions,
        gold_path=qa_path,
        prediction_path=prediction_path,
    )
    lab.save_qa_eval_report(qa_eval, qa_eval_path)
    dashboard = lab.dashboard_bundle(
        cases,
        predictions=predictions,
        qa_eval_report=qa_eval,
        graph=predicted_graph,
        error_attribution_report=error_attribution,
        qa_path=qa_path,
        prediction_path=prediction_path,
        qa_eval_report_path=qa_eval_path,
        graph_path=graph_path,
        error_attribution_report_path=error_attribution_report_path,
    )
    dashboard_path = tmp_path / "readiness-reports" / "dashboard.json"
    lab.save_dashboard_bundle(dashboard, dashboard_path)
    return dashboard_path


def _qa_delta_report_paths(
    tmp_path: Path,
    *,
    baseline_names: tuple[str, ...] = ("caption_memory", "vlm"),
) -> tuple[Path, ...]:
    cases = _offline_import_cases()
    predictions = tuple(
        lab.QAPrediction(
            id=case.id,
            answer=case.answer,
            evidence_nodes=case.required_nodes,
            evidence_edges=case.required_edges,
            confidence=1.0,
        )
        for case in cases
    )
    candidate_report = lab.qa_eval_report(cases, predictions)
    candidate_report_path = tmp_path / "candidate-qa-eval.json"
    lab.save_qa_eval_report(candidate_report, candidate_report_path)
    baseline_report = lab.qa_eval_report(cases, ())
    paths: list[Path] = []
    for baseline_name in baseline_names:
        baseline_report_path = tmp_path / f"baseline-qa-eval-{baseline_name}.json"
        lab.save_qa_eval_report(baseline_report, baseline_report_path)
        delta = lab.qa_eval_delta_report(
            candidate_report,
            baseline_report,
            candidate_name="graph_tool",
            baseline_name=baseline_name,
            candidate_report_path=candidate_report_path,
            baseline_report_path=baseline_report_path,
        )
        path = tmp_path / f"qa-delta-{baseline_name}.json"
        lab.save_qa_eval_delta_report(delta, path)
        paths.append(path)
    return tuple(paths)


def _offline_control_result_report_path(
    tmp_path: Path,
    offline_matrix_path: Path,
    qa_delta_paths: tuple[Path, ...],
) -> Path:
    matrix_report = lab.load_offline_control_matrix_report(offline_matrix_path)
    delta_paths_by_source: dict[str, Path] = {}
    candidate_report_path: str | None = None
    for delta_path in qa_delta_paths:
        delta_report = lab.load_qa_eval_delta_report(delta_path)
        baseline_name = str(delta_report["baseline_name"])
        source_key = next(
            row["source_key"]
            for row in matrix_report["source_profile_matrix"]
            if row["source_kind"] == baseline_name
        )
        delta_paths_by_source[str(source_key)] = delta_path
        candidate_report_path = str(delta_report["candidate_report_path"])
    assert candidate_report_path is not None
    result = lab.offline_control_result_report(
        matrix_report,
        matrix_report_path=offline_matrix_path,
        candidate_qa_eval_report_path=candidate_report_path,
        qa_eval_delta_report_paths=delta_paths_by_source,
    )
    path = tmp_path / "offline-control-result.json"
    lab.save_offline_control_result_report(result, path)
    return path


def _active_delta_report_paths(
    tmp_path: Path,
    *,
    candidate_name: str = "oracle_evidence",
    baseline_name: str = "direct_answer",
) -> tuple[Path, ...]:
    case = _offline_import_cases()[0]
    task = lab.ActiveEQATask(
        id=f"active:{case.id}",
        scene_id=case.scene_id,
        episode_id=case.episode_id,
        initial_step=case.step,
        question=case.question,
        gold_answer=case.answer,
        success_conditions={"answer_exact": True},
        max_actions=1,
        required_evidence={
            "nodes": case.required_nodes,
            "edges": case.required_edges,
        },
    )
    candidate_result = lab.ActiveTaskResult(
        task_id=task.id,
        policy=candidate_name,
        answer=case.answer,
        success=True,
        action_count=1,
        evidence_nodes=case.required_nodes,
        evidence_edges=case.required_edges,
        final_step=case.step + 1,
        confidence=1.0,
    )
    baseline_result = lab.ActiveTaskResult(
        task_id=task.id,
        policy=baseline_name,
        answer={},
        success=False,
        action_count=0,
        final_step=case.step,
        confidence=0.0,
        error="missing_required_evidence",
    )
    candidate_report = lab.active_task_report((task,), (candidate_result,))
    baseline_report = lab.active_task_report((task,), (baseline_result,))
    candidate_report_path = tmp_path / f"active-candidate-{candidate_name}.json"
    baseline_report_path = tmp_path / f"active-baseline-{baseline_name}.json"
    lab.save_active_task_report(candidate_report, candidate_report_path)
    lab.save_active_task_report(baseline_report, baseline_report_path)
    delta = lab.active_task_delta_report(
        candidate_report,
        baseline_report,
        candidate_name=candidate_name,
        baseline_name=baseline_name,
        candidate_report_path=candidate_report_path,
        baseline_report_path=baseline_report_path,
    )
    path = tmp_path / f"active-delta-{baseline_name}.json"
    lab.save_active_task_delta_report(delta, path)
    return (path,)


def _incomplete_manifest() -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": "dsg-spatialqa-lab.benchmark-manifest.v1",
        "dataset_name": "mock_benchmark",
        "scene_count": 1,
        "episode_count": 1,
        "qa_count": 3,
        "task_count": 0,
        "graph_digests": {},
        "qa_dataset_digests": {},
        "filters": {"max_qa_per_episode": 3, "source": "oracle", "tags": ["mock"]},
        "coverage": {
            "by_question_type": {"object_location": 3},
            "by_scene": {"FloorPlan1": 3},
            "by_episode": {"ai2thor_mock_001": 3},
            "by_reference_frame": {"none": 3},
            "by_tag": {"mock": 3},
            "dynamic_static": {"dynamic": 0, "static": 3},
            "oracle_predicted": {"oracle": 1, "predicted": 0},
        },
        "summary": {
            "dataset_name": "mock_benchmark",
            "episode_count": 1,
            "qa_count": 3,
            "scene_count": 1,
            "task_count": 0,
        },
        "artifacts": [],
    }
    manifest["manifest_digest"] = lab.benchmark_manifest_digest(manifest)
    return manifest


def _offline_import_report_path(
    tmp_path: Path,
    name: str,
    kind: str,
    *,
    source_metadata: dict[str, object] | None = None,
) -> Path:
    return _offline_import_report_path_for_cases(
        tmp_path,
        name=name,
        kind=kind,
        cases=_offline_import_cases()[:1],
        record_case_ids=("case_001",),
        source_metadata=source_metadata,
    )


def _offline_import_report_path_for_cases(
    tmp_path: Path,
    *,
    name: str,
    kind: str,
    cases: tuple[lab.QACase, ...],
    record_case_ids: tuple[str, ...],
    source_metadata: dict[str, object] | None = None,
) -> Path:
    answer_by_case_id = {case.id: case.answer for case in cases}
    records = tuple(
        lab.OfflinePredictionRecord(
            case_id=case_id,
            answer=answer_by_case_id.get(case_id, {}),
        )
        for case_id in record_case_ids
    )
    qa_path = tmp_path / f"{kind}-{name}-qa.jsonl"
    input_path = tmp_path / f"{kind}-{name}-offline-input.jsonl"
    prediction_path = tmp_path / f"{kind}-{name}-predictions.jsonl"
    lab.save_qa_dataset(cases, qa_path)
    lab.save_offline_prediction_records(records, input_path)
    predictions, report = lab.import_offline_predictions(
        cases,
        records,
        source_name=name,
        source_kind=kind,
        source_metadata=source_metadata
        if source_metadata is not None
        else _real_offline_source_metadata(kind),
        qa_path=qa_path,
        input_path=input_path,
        prediction_path=prediction_path,
    )
    lab.save_qa_predictions(predictions, prediction_path)
    report_path = tmp_path / f"{kind}-import-report.json"
    lab.save_offline_prediction_import_report(report, report_path)
    return report_path


def _real_offline_source_metadata(kind: str) -> dict[str, object]:
    model_ids = {
        "caption_memory": "blip2-flan-t5-xl",
        "vlm": "llava-v1.6-34b",
    }
    prompt_ids = {
        "caption_memory": "caption-memory-spatial-v1",
        "vlm": "vlm-spatial-qa-v1",
    }
    return {
        "capabilities": ("spatial_qa",),
        "dataset_id": "ai2thor-real-trial-v1",
        "model_id": model_ids.get(kind, f"{kind}-offline-model"),
        "prompt_id": prompt_ids.get(kind, f"{kind}-spatial-qa-v1"),
    }


def _offline_import_cases() -> tuple[lab.QACase, ...]:
    case = lab.QACase(
        id="case_001",
        scene_id="FloorPlan1",
        episode_id="ai2thor_real_001",
        graph_digest="0" * 64,
        step=1,
        question={"type": "object_location", "object_id": "mug_1"},
        question_type="object_location",
        answer={"object_id": "mug_1"},
        answer_type="object_location",
        tags=("real",),
    )
    second_case = lab.QACase(
        id="case_002",
        scene_id="FloorPlan1",
        episode_id="ai2thor_real_001",
        graph_digest="0" * 64,
        step=2,
        question={"type": "object_location", "object_id": "plate_1"},
        question_type="object_location",
        answer={"object_id": "plate_1"},
        answer_type="object_location",
        tags=("real",),
    )
    return (case, second_case)


def _predicted_observation_report_paths(tmp_path: Path) -> tuple[Path, Path]:
    observations = (
        lab.SceneObservation(
            step=1,
            objects=(
                lab.ObjectObservation(
                    "mug_1",
                    "mug",
                    lab.Pose3D(0.0, 1.0, 0.8),
                    lab.BBox3D(
                        center=lab.Pose3D(0.0, 1.0, 0.8),
                        size=(0.2, 0.2, 0.3),
                    ),
                    confidence=0.9,
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
                    lab.Pose3D(0.2, 1.0, 0.8),
                    lab.BBox3D(
                        center=lab.Pose3D(0.2, 1.0, 0.8),
                        size=(0.2, 0.2, 0.3),
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
            objects=(
                lab.ObjectObservation(
                    "mug_1",
                    "mug",
                    lab.Pose3D(0.0, 1.0, 0.8),
                    lab.BBox3D(
                        center=lab.Pose3D(0.0, 1.0, 0.8),
                        size=(0.2, 0.2, 0.3),
                    ),
                    confidence=0.4,
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
    observation_path = tmp_path / "detector-observations.json"
    graph_path = tmp_path / "predicted-graph.json"
    report_path = tmp_path / "predicted-report.json"
    evidence_path = tmp_path / "predicted-dsg-evidence.json"
    lab.save_scene_observation_sequence(observations, observation_path)
    graph = lab.build_predicted_graph_from_observations(
        observations,
        source_path=observation_path,
    )
    lab.save_graph_json(graph, graph_path)
    report = lab.predicted_graph_report_from_observations(
        input_path=observation_path,
        graph_path=graph_path,
        graph=graph,
        observations=observations,
    )
    lab.save_predicted_graph_report(report, report_path)
    evidence_report = lab.predicted_dsg_evidence_report(
        report,
        predicted_graph_report_path=report_path,
    )
    lab.save_predicted_dsg_evidence_report(evidence_report, evidence_path)
    return report_path, evidence_path


def _real_collection_report_path(tmp_path: Path) -> Path:
    episode_path = tmp_path / "real-collection-episode.jsonl"
    frames = (
        lab.EpisodeFrame(
            episode_id="ai2thor_real_001",
            scene_id="FloorPlan1",
            step=1,
            rgb_path="real-ai2thor/ai2thor_real_001/000001.rgb.png",
            depth_path="real-ai2thor/ai2thor_real_001/000001.depth.png",
            segmentation_path="real-ai2thor/ai2thor_real_001/000001.segmentation.png",
            agent_id="agent",
            agent_pose=lab.Pose3D(0.0, 0.0, 0.0),
            action="Initialize",
            visible_object_ids=("mug_1",),
            metadata={
                "adapter": "ai2thor",
                "collection_kind": "real",
                "source_kind": "ai2thor",
            },
        ),
        lab.EpisodeFrame(
            episode_id="ai2thor_real_001",
            scene_id="FloorPlan1",
            step=2,
            rgb_path="real-ai2thor/ai2thor_real_001/000002.rgb.png",
            depth_path="real-ai2thor/ai2thor_real_001/000002.depth.png",
            segmentation_path="real-ai2thor/ai2thor_real_001/000002.segmentation.png",
            agent_id="agent",
            agent_pose=lab.Pose3D(0.1, 0.0, 0.0),
            action="MoveAhead",
            visible_object_ids=("mug_1",),
            metadata={
                "adapter": "ai2thor",
                "collection_kind": "real",
                "source_kind": "ai2thor",
            },
        ),
    )
    lab.save_episode_sequence(frames, episode_path)
    _write_real_frame_assets(episode_path, frames)
    report = lab.real_collection_report(
        dataset_name="ai2thor_small_real",
        episode_paths=(episode_path,),
        source_kind="ai2thor",
        min_episode_count=1,
        min_scene_count=1,
        min_frame_count=2,
    )
    report_path = tmp_path / "real-collection-report.json"
    lab.save_real_collection_report(report, report_path)
    return report_path


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


def _experiment_artifact(
    artifact_type: str,
    path: str,
    *,
    digest: str | None = None,
) -> dict[str, object]:
    return {
        "artifact_type": artifact_type,
        "path": path,
        "schema_version": f"fixture.{artifact_type}.v1",
        "digest": digest or ("1" * 64),
    }


def _artifact_paths(manifest: dict[str, object], artifact_type: str) -> tuple[Path, ...]:
    artifacts = manifest["experiment_artifacts"]
    assert isinstance(artifacts, list)
    return tuple(
        Path(str(artifact["path"]))
        for artifact in artifacts
        if isinstance(artifact, dict) and artifact.get("artifact_type") == artifact_type
    )
