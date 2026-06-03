from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Protocol, cast

from _pytest.capture import CaptureFixture

import dsg_spatialqa_lab as lab


ROOT = Path(__file__).resolve().parents[1]
SUMMARIZE_EXPERIMENT_SCRIPT = ROOT / "scripts" / "summarize_experiment.py"
RECORD_EXPERIMENT_SCRIPT = ROOT / "scripts" / "record_experiment.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_summarize_experiment_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "summarize_experiment_script",
        SUMMARIZE_EXPERIMENT_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_record_experiment_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "record_experiment_script",
        RECORD_EXPERIMENT_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_experiment_summary_report_summarizes_research_question_lift_and_drift(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "experiment_summary_report")
    assert hasattr(lab, "experiment_summary_report_digest")
    assert hasattr(lab, "experiment_summary_report_json")
    assert hasattr(lab, "save_experiment_summary_report")
    assert hasattr(lab, "load_experiment_summary_report")
    assert hasattr(lab, "validate_experiment_summary_report")
    assert hasattr(lab, "compare_experiment_summary_report")
    artifacts = _write_experiment_artifacts(tmp_path)
    manifest = lab.build_benchmark_artifacts(
        dataset_name="mock_benchmark",
        episode_paths=_write_mock_episodes(tmp_path / "episodes"),
        output_dir=tmp_path / "benchmark",
        max_qa_per_episode=3,
        qa_eval_delta_report_paths=(artifacts.qa_delta_path,),
        active_task_delta_report_paths=(artifacts.active_delta_path,),
        error_attribution_report_paths=(artifacts.error_attribution_path,),
        graph_eval_report_paths=(artifacts.graph_eval_path,),
        offline_prediction_import_report_paths=(artifacts.offline_import_path,),
    )
    manifest_path = tmp_path / "benchmark-manifest.json"
    report_path = tmp_path / "experiment-summary.json"
    lab.save_benchmark_manifest(manifest, manifest_path)

    report = lab.experiment_summary_report(manifest, manifest_path=manifest_path)
    saved_path = lab.save_experiment_summary_report(report, report_path)
    loaded = lab.load_experiment_summary_report(report_path)
    validation = lab.validate_experiment_summary_report(loaded)
    comparison = lab.compare_experiment_summary_report(loaded)

    assert report["schema_version"] == (
        "dsg-spatialqa-lab.experiment-summary-report.v1"
    )
    assert report["manifest_path"] == str(manifest_path)
    assert report["manifest_digest"] == manifest["manifest_digest"]
    assert report["source_artifact_digests"] == {
        "active_task_delta_report:active-delta-report.json": artifacts.active_delta[
            "report_digest"
        ],
        "error_attribution_report:error-attribution-report.json": (
            artifacts.error_attribution["report_digest"]
        ),
        "graph_eval_report:graph-eval-report.json": artifacts.graph_eval[
            "report_digest"
        ],
        "offline_prediction_import_report:offline-import-report.json": (
            artifacts.offline_import["report_digest"]
        ),
        "qa_eval_delta_report:qa-delta-report.json": artifacts.qa_delta[
            "report_digest"
        ],
    }
    assert report["source_profile_matrix"] == [
        {
            "adapter": "vlm",
            "artifact_key": (
                "offline_prediction_import_report:offline-import-report.json"
            ),
            "capability_axes": ["graph_tool_query", "spatial_qa"],
            "dataset_id": "mock_eval",
            "digest": artifacts.offline_import["report_digest"],
            "duplicate_case_count": 0,
            "imported_prediction_count": 3,
            "metadata_keys": [
                "capabilities",
                "dataset_id",
                "model_id",
                "prompt_id",
            ],
            "missing_case_count": 0,
            "model_id": "mock-vlm",
            "path": str(artifacts.offline_import_path),
            "prediction_digest": artifacts.offline_import["prediction_digest"],
            "prompt_id": "spatial-qa-v1",
            "qa_digest": artifacts.offline_import["qa_digest"],
            "source_key": "vlm:vlm_fixture",
            "source_kind": "vlm",
            "source_name": "vlm_fixture",
            "unknown_case_count": 0,
        }
    ]
    graph_key = "graph_eval_report:graph-eval-report.json"
    assert report["graph_construction_diagnostics"][graph_key][
        "primary_metrics"
    ] == {
        "object_recall_rate": 1.0,
        "relation_f1_rate": 1.0,
        "state_accuracy_rate": 1.0,
    }
    assert report["graph_construction_diagnostics"][graph_key]["diagnostics"] == {
        "duplicate_track_count": 0,
        "id_fragmentation_count": 0,
    }
    assert report["graph_construction_diagnostics"][graph_key]["source_breakdown"][
        "objects"
    ]["mock_segmenter"]["precision"] == 1.0
    assert report["graph_construction_diagnostics"][graph_key]["source_breakdown"][
        "relations"
    ]["geometry"]["precision"] == 1.0
    attribution_key = "error_attribution_report:error-attribution-report.json"
    assert report["error_attribution_diagnostics"][attribution_key][
        "summary"
    ] == artifacts.error_attribution["summary"]
    assert report["error_attribution_diagnostics"][attribution_key]["summary"][
        "by_error_category"
    ] == {
        "correct": 1,
        "reasoning_or_tool_use": 2,
    }
    attribution_axis_summary = report["error_attribution_diagnostics"][attribution_key][
        "summary"
    ]["by_research_axis"]
    assert attribution_axis_summary["dynamic_memory"]["by_error_category"] == {
        "reasoning_or_tool_use": 1,
    }
    assert attribution_axis_summary["graph_tool_query"]["case_count"] == 3
    assert report["failure_linkage_diagnostics"][attribution_key] == {
        "attribution_summary": artifacts.error_attribution["summary"],
        "error_attribution_artifact_key": attribution_key,
        "graph_diagnostics": {
            "duplicate_track_count": 0,
            "id_fragmentation_count": 0,
        },
        "graph_eval_artifact_key": graph_key,
        "graph_primary_metrics": {
            "object_recall_rate": 1.0,
            "relation_f1_rate": 1.0,
            "state_accuracy_rate": 1.0,
        },
        "linked_by": "oracle_and_predicted_graph_digest",
        "oracle_graph_digest": artifacts.error_attribution["oracle_graph_digest"],
        "predicted_graph_digest": artifacts.error_attribution[
            "predicted_graph_digest"
        ],
    }
    assert report["failure_linkage_diagnostics"][attribution_key][
        "attribution_summary"
    ]["by_research_axis"] == attribution_axis_summary
    qa_slice_key = "qa_eval_delta_report:qa-delta-report.json"
    assert report["qa_diagnostic_slices"][qa_slice_key]["by_question_type"][
        "object_location"
    ] == {
        "baseline_case_count": 1,
        "baseline_exact_match_count": 0,
        "baseline_exact_match_rate": 0.0,
        "baseline_mean_evidence_edge_recall": 0.0,
        "baseline_mean_evidence_node_recall": 0.0,
        "candidate_case_count": 1,
        "candidate_exact_match_count": 1,
        "candidate_exact_match_rate": 1.0,
        "candidate_mean_evidence_edge_recall": 1.0,
        "candidate_mean_evidence_node_recall": 1.0,
        "case_count_delta": 0,
        "case_count_match": True,
        "exact_match_count_delta": 1,
        "exact_match_rate_delta": 1.0,
        "mean_evidence_edge_recall_delta": 1.0,
        "mean_evidence_node_recall_delta": 1.0,
    }
    assert report["qa_diagnostic_slices"][qa_slice_key]["by_tag"]["dynamic"][
        "mean_evidence_node_recall_delta"
    ] == 0.5
    assert report["qa_diagnostic_slices"][qa_slice_key]["by_reference_frame"]["none"][
        "exact_match_rate_delta"
    ] == 0.5
    assert report["qa_diagnostic_slices"][qa_slice_key]["by_scene_id"][
        "tabletop_scene"
    ]["exact_match_rate_delta"] == 0.333333
    assert report["qa_diagnostic_slices"][qa_slice_key]["by_episode_id"][
        "episode_001"
    ]["exact_match_rate_delta"] == 0.333333
    assert report["research_questions"]["spatial_qa"]["primary_metric"] == {
        "name": "exact_match_rate_delta",
        "value": 0.333333,
    }
    assert report["research_questions"]["dynamic_memory"]["primary_metric"] == {
        "name": "exact_match_rate_delta",
        "value": 0.0,
    }
    assert report["research_questions"]["dynamic_memory"]["supporting_metrics"][
        "mean_evidence_node_recall_delta"
    ] == 0.5
    assert report["research_questions"]["graph_tool_query"]["primary_metric"] == {
        "name": "exact_match_rate_delta",
        "value": 0.333333,
    }
    assert report["research_questions"]["interactive_task"]["primary_metric"] == {
        "name": "task_success_rate_delta",
        "value": 1.0,
    }
    assert report["research_questions"]["spatial_qa"]["verdict"] == "improved"
    assert report["research_questions"]["dynamic_memory"]["verdict"] == "unchanged"
    assert report["research_questions"]["graph_tool_query"]["verdict"] == "improved"
    assert report["research_questions"]["interactive_task"]["verdict"] == "improved"
    assert report["summary"]["available_research_question_count"] == 4
    assert report["summary"]["error_attribution_diagnostic_count"] == 1
    assert report["summary"]["failure_linkage_diagnostic_count"] == 1
    assert report["summary"]["graph_construction_diagnostic_count"] == 1
    assert report["summary"]["qa_diagnostic_slice_count"] == 14
    assert report["summary"]["source_profile_count"] == 1
    assert report["summary"]["verdict_counts"] == {
        "improved": 3,
        "inconclusive": 0,
        "regressed": 0,
        "unchanged": 1,
    }
    assert saved_path == report_path
    assert json.loads(lab.experiment_summary_report_json(report)) == report
    assert loaded == report
    assert validation["valid"] is True
    assert comparison["matches"] is True

    tampered_report = dict(report)
    tampered_report["source_profile_matrix"] = [
        {
            **dict(report["source_profile_matrix"][0]),
            "source_key": "vlm:changed",
        }
    ]
    tampered_report["report_digest"] = lab.experiment_summary_report_digest(
        tampered_report
    )
    tampered_validation = lab.validate_experiment_summary_report(tampered_report)
    tampered_checks = {check["name"]: check for check in tampered_validation["checks"]}
    assert tampered_validation["valid"] is False
    assert tampered_checks["source_profile_matrix"]["passed"] is False

    drifted_qa_delta = dict(artifacts.qa_delta)
    drifted_breakdown = dict(artifacts.qa_delta["breakdown_delta"])
    drifted_axes = dict(drifted_breakdown["by_research_axis"])
    drifted_spatial = dict(drifted_axes["spatial_qa"])
    drifted_spatial["exact_match_rate_delta"] = 0.0
    drifted_axes["spatial_qa"] = drifted_spatial
    drifted_breakdown["by_research_axis"] = drifted_axes
    drifted_qa_delta["breakdown_delta"] = drifted_breakdown
    drifted_qa_delta["report_digest"] = lab.qa_eval_delta_report_digest(drifted_qa_delta)
    lab.save_qa_eval_delta_report(drifted_qa_delta, artifacts.qa_delta_path)

    drift = lab.compare_experiment_summary_report(loaded)
    checks = {check["name"]: check for check in drift["checks"]}
    assert drift["matches"] is False
    assert checks["research_questions_match_current"]["passed"] is False


def test_experiment_record_projects_final_readiness_and_verdicts(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "experiment_record")
    assert hasattr(lab, "experiment_record_digest")
    assert hasattr(lab, "experiment_record_json")
    assert hasattr(lab, "save_experiment_record")
    assert hasattr(lab, "load_experiment_record")
    assert hasattr(lab, "validate_experiment_record")
    assert hasattr(lab, "compare_experiment_record")
    artifacts = _write_experiment_artifacts(tmp_path)
    manifest = lab.build_benchmark_artifacts(
        dataset_name="mock_benchmark",
        episode_paths=_write_mock_episodes(tmp_path / "episodes"),
        output_dir=tmp_path / "benchmark",
        max_qa_per_episode=3,
        qa_eval_delta_report_paths=(artifacts.qa_delta_path,),
        active_task_delta_report_paths=(artifacts.active_delta_path,),
        error_attribution_report_paths=(artifacts.error_attribution_path,),
        graph_eval_report_paths=(artifacts.graph_eval_path,),
        offline_prediction_import_report_paths=(artifacts.offline_import_path,),
    )
    manifest_path = tmp_path / "benchmark-manifest.json"
    summary_path = tmp_path / "experiment-summary.json"
    record_path = tmp_path / "experiment-record.json"
    lab.save_benchmark_manifest(manifest, manifest_path)
    summary_report = lab.experiment_summary_report(manifest, manifest_path=manifest_path)
    lab.save_experiment_summary_report(summary_report, summary_path)
    real_readiness = _ready_real_readiness_report(manifest, manifest_path)
    real_readiness_path = tmp_path / "real-readiness.json"
    lab.save_real_experiment_readiness_report(real_readiness, real_readiness_path)

    record = lab.experiment_record(
        summary_report,
        summary_report_path=summary_path,
        real_readiness_report=real_readiness,
        real_readiness_report_path=real_readiness_path,
    )
    saved_path = lab.save_experiment_record(record, record_path)
    loaded = lab.load_experiment_record(record_path)
    validation = lab.validate_experiment_record(loaded)
    comparison = lab.compare_experiment_record(loaded)

    assert record["schema_version"] == "dsg-spatialqa-lab.experiment-record.v1"
    assert record["summary_report_path"] == str(summary_path)
    assert record["summary_report_digest"] == summary_report["report_digest"]
    assert record["manifest_path"] == str(manifest_path)
    assert record["manifest_digest"] == manifest["manifest_digest"]
    assert record["real_readiness_report_path"] == str(real_readiness_path)
    assert record["real_readiness_report_digest"] == real_readiness["report_digest"]
    assert record["real_package_status"] == "ready"
    assert record["real_package_readiness"] == {
        "declared_data_source_kind": "real",
        "failed_checks": [],
        "failed_count": 0,
        "manifest_digest": manifest["manifest_digest"],
        "missing_groups": [],
        "ready": True,
        "report_digest": real_readiness["report_digest"],
        "valid": True,
    }
    assert record["readiness_status"] == "ready"
    assert record["verdict_counts"] == {
        "improved": 3,
        "inconclusive": 0,
        "regressed": 0,
        "unchanged": 1,
    }
    assert record["research_question_verdicts"] == {
        "dynamic_memory": {
            "measurement_count": 1,
            "primary_metric": {
                "name": "exact_match_rate_delta",
                "value": 0.0,
            },
            "source_artifact_type": "qa_eval_delta_report",
            "status": "available",
            "verdict": "unchanged",
        },
        "graph_tool_query": {
            "measurement_count": 1,
            "primary_metric": {
                "name": "exact_match_rate_delta",
                "value": 0.333333,
            },
            "source_artifact_type": "qa_eval_delta_report",
            "status": "available",
            "verdict": "improved",
        },
        "interactive_task": {
            "measurement_count": 1,
            "primary_metric": {
                "name": "task_success_rate_delta",
                "value": 1.0,
            },
            "source_artifact_type": "active_task_delta_report",
            "status": "available",
            "verdict": "improved",
        },
        "spatial_qa": {
            "measurement_count": 1,
            "primary_metric": {
                "name": "exact_match_rate_delta",
                "value": 0.333333,
            },
            "source_artifact_type": "qa_eval_delta_report",
            "status": "available",
            "verdict": "improved",
        },
    }
    assert [
        row["research_question"] for row in record["research_question_matrix"]
    ] == [
        "dynamic_memory",
        "graph_tool_query",
        "interactive_task",
        "spatial_qa",
    ]
    assert record["research_question_matrix"][0] == {
        "artifact_key": "qa_eval_delta_report:qa-delta-report.json",
        "baseline_name": "majority",
        "candidate_name": "graph_tool",
        "case_count_match": True,
        "label": "Does the Dynamic Scene Graph improve dynamic memory?",
        "measurement_verdict": "unchanged",
        "primary_metric": {
            "name": "exact_match_rate_delta",
            "value": 0.0,
        },
        "question_verdict": "unchanged",
        "research_question": "dynamic_memory",
        "source_artifact_type": "qa_eval_delta_report",
        "status": "available",
        "supporting_metrics": {
            "baseline_case_count": 1,
            "candidate_case_count": 1,
            "exact_match_count_delta": 0,
            "mean_evidence_edge_recall_delta": 0.0,
            "mean_evidence_node_recall_delta": 0.5,
        },
    }
    assert record["source_profile_count"] == 1
    assert record["source_profile_matrix"] == summary_report["source_profile_matrix"]
    assert record["diagnostic_ledger"] == {
        "error_attribution_artifact_keys": [
            "error_attribution_report:error-attribution-report.json",
        ],
        "error_attribution_diagnostic_count": 1,
        "failure_linkage_pair_count": 1,
        "failure_linkage_pairs": [
            {
                "error_attribution_artifact_key": (
                    "error_attribution_report:error-attribution-report.json"
                ),
                "graph_eval_artifact_key": "graph_eval_report:graph-eval-report.json",
                "linked_by": "oracle_and_predicted_graph_digest",
            },
        ],
        "graph_construction_artifact_keys": [
            "graph_eval_report:graph-eval-report.json",
        ],
        "graph_construction_diagnostic_count": 1,
        "qa_diagnostic_slice_count": 14,
        "qa_diagnostic_slice_keys": [
            "qa_eval_delta_report:qa-delta-report.json",
        ],
    }
    assert saved_path == record_path
    assert json.loads(lab.experiment_record_json(record)) == record
    assert loaded == record
    assert validation["valid"] is True
    assert comparison["matches"] is True

    tampered_record = json.loads(lab.experiment_record_json(record))
    tampered_record["diagnostic_ledger"]["failure_linkage_pair_count"] = 0
    tampered_record["record_digest"] = lab.experiment_record_digest(tampered_record)
    tampered_validation = lab.validate_experiment_record(tampered_record)
    tampered_checks = {
        check["name"]: check for check in tampered_validation["checks"]
    }
    assert tampered_validation["valid"] is False
    assert tampered_checks["diagnostic_ledger"]["passed"] is False

    drifted_summary = json.loads(lab.experiment_summary_report_json(summary_report))
    drifted_summary["research_questions"]["spatial_qa"]["verdict"] = "regressed"
    drifted_summary["summary"]["verdict_counts"] = {
        "improved": 2,
        "inconclusive": 0,
        "regressed": 1,
        "unchanged": 1,
    }
    drifted_summary["failure_linkage_diagnostics"][
        "error_attribution_report:error-attribution-report.json"
    ]["linked_by"] = "unmatched"
    drifted_summary["source_profile_matrix"][0]["source_key"] = "vlm:changed"
    drifted_summary["report_digest"] = lab.experiment_summary_report_digest(
        drifted_summary
    )
    lab.save_experiment_summary_report(drifted_summary, summary_path)

    drift = lab.compare_experiment_record(loaded)
    checks = {check["name"]: check for check in drift["checks"]}
    assert drift["matches"] is False
    assert checks["record_digest_matches_current"]["passed"] is False
    assert checks["research_question_verdicts_match_current"]["passed"] is False
    assert checks["diagnostic_ledger_match_current"]["passed"] is False
    assert checks["source_profile_matrix_match_current"]["passed"] is False


def test_record_experiment_cli_links_ready_real_package_readiness(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_record_experiment_script()
    main = cast(MainFn, getattr(module, "main"))
    artifacts = _write_experiment_artifacts(tmp_path)
    manifest = lab.build_benchmark_artifacts(
        dataset_name="mock_benchmark",
        episode_paths=_write_mock_episodes(tmp_path / "episodes"),
        output_dir=tmp_path / "benchmark",
        max_qa_per_episode=3,
        qa_eval_delta_report_paths=(artifacts.qa_delta_path,),
        active_task_delta_report_paths=(artifacts.active_delta_path,),
        error_attribution_report_paths=(artifacts.error_attribution_path,),
        graph_eval_report_paths=(artifacts.graph_eval_path,),
    )
    manifest_path = tmp_path / "benchmark-manifest.json"
    summary_path = tmp_path / "experiment-summary.json"
    readiness_path = tmp_path / "real-readiness.json"
    record_path = tmp_path / "experiment-record.json"
    lab.save_benchmark_manifest(manifest, manifest_path)
    lab.save_experiment_summary_report(
        lab.experiment_summary_report(manifest, manifest_path=manifest_path),
        summary_path,
    )
    real_readiness = _ready_real_readiness_report(manifest, manifest_path)
    lab.save_real_experiment_readiness_report(real_readiness, readiness_path)

    assert main(
        [
            "--summary-report",
            str(summary_path),
            "--real-readiness-report",
            str(readiness_path),
            "--record",
            str(record_path),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    record = lab.load_experiment_record(record_path)
    assert output["action"] == "experiment_record"
    assert output["valid"] is True
    assert output["real_package_status"] == "ready"
    assert output["digest"] == record["record_digest"]
    assert record["real_readiness_report_digest"] == real_readiness["report_digest"]


def test_record_experiment_cli_writes_validates_compares_and_reports_invalid_json(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_record_experiment_script()
    main = cast(MainFn, getattr(module, "main"))
    artifacts = _write_experiment_artifacts(tmp_path)
    manifest = lab.build_benchmark_artifacts(
        dataset_name="mock_benchmark",
        episode_paths=_write_mock_episodes(tmp_path / "episodes"),
        output_dir=tmp_path / "benchmark",
        max_qa_per_episode=3,
        qa_eval_delta_report_paths=(artifacts.qa_delta_path,),
        active_task_delta_report_paths=(artifacts.active_delta_path,),
        error_attribution_report_paths=(artifacts.error_attribution_path,),
        graph_eval_report_paths=(artifacts.graph_eval_path,),
    )
    manifest_path = tmp_path / "benchmark-manifest.json"
    summary_path = tmp_path / "experiment-summary.json"
    record_path = tmp_path / "experiment-record.json"
    lab.save_benchmark_manifest(manifest, manifest_path)
    lab.save_experiment_summary_report(
        lab.experiment_summary_report(manifest, manifest_path=manifest_path),
        summary_path,
    )

    assert main(
        [
            "--summary-report",
            str(summary_path),
            "--record",
            str(record_path),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    record = lab.load_experiment_record(record_path)
    assert output == {
        "action": "experiment_record",
        "path": str(record_path),
        "valid": True,
        "digest": record["record_digest"],
        "readiness_status": "ready",
        "verdict_counts": record["verdict_counts"],
    }

    assert main(["--validate-record", str(record_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_experiment_record"
    assert validation["valid"] is True

    assert main(["--compare-record", str(record_path)]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["action"] == "compare_experiment_record"
    assert comparison["matches"] is True

    invalid_path = tmp_path / "invalid-record.json"
    invalid_path.write_text("[]\n", encoding="utf-8")
    assert main(["--validate-record", str(invalid_path)]) == 1
    invalid = json.loads(capsys.readouterr().out)
    assert invalid == {
        "action": "validate_experiment_record",
        "path": str(invalid_path),
        "valid": False,
        "error": "Experiment record JSON must be an object",
    }


def _ready_real_readiness_report(
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    checks = [
        {
            "name": "real_package_import_ready",
            "group": "real_data",
            "passed": True,
        }
    ]
    report: dict[str, Any] = {
        "schema_version": "dsg-spatialqa-lab.real-experiment-readiness.v1",
        "manifest_path": str(manifest_path),
        "manifest_digest": manifest["manifest_digest"],
        "declared_data_source_kind": "real",
        "thresholds": {
            "min_episode_count": 1,
            "min_qa_count": 1,
            "min_scene_count": 1,
        },
        "required_control_kinds": [],
        "required_predicted_input_kinds": [],
        "artifact_summary": {},
        "checks": checks,
        "readiness": {
            "ready": True,
            "passed_count": 1,
            "failed_count": 0,
            "missing_groups": [],
            "failed_checks": [],
        },
    }
    report["report_digest"] = lab.real_experiment_readiness_report_digest(report)
    return report


def test_experiment_summary_validation_detects_verdict_drift_after_digest_recompute(
    tmp_path: Path,
) -> None:
    artifacts = _write_experiment_artifacts(tmp_path)
    manifest = lab.build_benchmark_artifacts(
        dataset_name="mock_benchmark",
        episode_paths=_write_mock_episodes(tmp_path / "episodes"),
        output_dir=tmp_path / "benchmark",
        max_qa_per_episode=3,
        qa_eval_delta_report_paths=(artifacts.qa_delta_path,),
        active_task_delta_report_paths=(artifacts.active_delta_path,),
    )
    report = lab.experiment_summary_report(manifest)
    drifted = json.loads(lab.experiment_summary_report_json(report))
    drifted["research_questions"]["spatial_qa"]["verdict"] = "regressed"
    drifted["summary"]["verdict_counts"]["improved"] = 0
    drifted["report_digest"] = lab.experiment_summary_report_digest(drifted)

    validation = lab.validate_experiment_summary_report(drifted)

    checks = {check["name"]: check for check in validation["checks"]}
    assert validation["valid"] is False
    assert checks["research_questions"]["passed"] is False
    assert checks["research_questions"]["expected"] == report["research_questions"]
    assert checks["summary"]["passed"] is False
    assert checks["summary"]["expected"] == report["summary"]


def test_summarize_experiment_cli_writes_validates_compares_and_reports_invalid_json(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_summarize_experiment_script()
    main = cast(MainFn, getattr(module, "main"))
    artifacts = _write_experiment_artifacts(tmp_path)
    manifest = lab.build_benchmark_artifacts(
        dataset_name="mock_benchmark",
        episode_paths=_write_mock_episodes(tmp_path / "episodes"),
        output_dir=tmp_path / "benchmark",
        max_qa_per_episode=3,
        qa_eval_delta_report_paths=(artifacts.qa_delta_path,),
        active_task_delta_report_paths=(artifacts.active_delta_path,),
    )
    manifest_path = tmp_path / "benchmark-manifest.json"
    report_path = tmp_path / "experiment-summary.json"
    lab.save_benchmark_manifest(manifest, manifest_path)

    assert main(
        [
            "--manifest",
            str(manifest_path),
            "--report",
            str(report_path),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    report = lab.load_experiment_summary_report(report_path)
    assert output == {
        "action": "experiment_summary_report",
        "path": str(report_path),
        "valid": True,
        "digest": report["report_digest"],
        "readiness": report["readiness"],
        "summary": report["summary"],
    }

    assert main(["--validate-report", str(report_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_experiment_summary_report"
    assert validation["valid"] is True

    assert main(["--compare-report", str(report_path)]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["action"] == "compare_experiment_summary_report"
    assert comparison["matches"] is True

    invalid_path = tmp_path / "invalid-summary.json"
    invalid_path.write_text("[]\n", encoding="utf-8")
    assert main(["--validate-report", str(invalid_path)]) == 1
    invalid = json.loads(capsys.readouterr().out)
    assert invalid == {
        "action": "validate_experiment_summary_report",
        "path": str(invalid_path),
        "valid": False,
        "error": "Experiment summary report JSON must be an object",
    }


def test_experiment_summary_readiness_identifies_missing_research_evidence(
    tmp_path: Path,
) -> None:
    artifacts = _write_experiment_artifacts(tmp_path)
    manifest = lab.build_benchmark_artifacts(
        dataset_name="mock_benchmark",
        episode_paths=_write_mock_episodes(tmp_path / "episodes"),
        output_dir=tmp_path / "benchmark",
        max_qa_per_episode=3,
        qa_eval_delta_report_paths=(artifacts.qa_delta_path,),
    )

    report = lab.experiment_summary_report(manifest)
    validation = lab.validate_experiment_summary_report(report)

    assert report["readiness"] == {
        "status": "incomplete",
        "required_research_questions": [
            "dynamic_memory",
            "graph_tool_query",
            "interactive_task",
            "spatial_qa",
        ],
        "available_research_questions": [
            "dynamic_memory",
            "graph_tool_query",
            "spatial_qa",
        ],
        "missing_research_questions": ["interactive_task"],
        "required_source_artifact_types": [
            "active_task_delta_report",
            "qa_eval_delta_report",
        ],
        "available_source_artifact_types": ["qa_eval_delta_report"],
        "missing_source_artifact_types": ["active_task_delta_report"],
        "checks": [
            {
                "name": "dynamic_memory",
                "passed": True,
                "source_artifact_type": "qa_eval_delta_report",
                "measurement_count": 1,
                "missing_reason": None,
            },
            {
                "name": "graph_tool_query",
                "passed": True,
                "source_artifact_type": "qa_eval_delta_report",
                "measurement_count": 1,
                "missing_reason": None,
            },
            {
                "name": "interactive_task",
                "passed": False,
                "source_artifact_type": "active_task_delta_report",
                "measurement_count": 0,
                "missing_reason": "missing_active_task_delta_report",
            },
            {
                "name": "spatial_qa",
                "passed": True,
                "source_artifact_type": "qa_eval_delta_report",
                "measurement_count": 1,
                "missing_reason": None,
            },
        ],
    }
    assert report["summary"]["readiness_status"] == "incomplete"
    assert validation["valid"] is True


def test_experiment_summary_validation_detects_metric_drift_after_digest_recompute(
    tmp_path: Path,
) -> None:
    artifacts = _write_experiment_artifacts(tmp_path)
    manifest = lab.build_benchmark_artifacts(
        dataset_name="mock_benchmark",
        episode_paths=_write_mock_episodes(tmp_path / "episodes"),
        output_dir=tmp_path / "benchmark",
        max_qa_per_episode=3,
        qa_eval_delta_report_paths=(artifacts.qa_delta_path,),
        active_task_delta_report_paths=(artifacts.active_delta_path,),
        error_attribution_report_paths=(artifacts.error_attribution_path,),
        graph_eval_report_paths=(artifacts.graph_eval_path,),
    )
    report = lab.experiment_summary_report(manifest)
    drifted = json.loads(lab.experiment_summary_report_json(report))
    drifted["research_questions"]["spatial_qa"]["primary_metric"]["value"] = 0.0
    drifted["qa_diagnostic_slices"]["qa_eval_delta_report:qa-delta-report.json"][
        "by_question_type"
    ]["object_location"]["exact_match_rate_delta"] = 0.0
    drifted["qa_diagnostic_slices"]["qa_eval_delta_report:qa-delta-report.json"][
        "by_scene_id"
    ]["tabletop_scene"]["exact_match_rate_delta"] = 0.0
    drifted["graph_construction_diagnostics"][
        "graph_eval_report:graph-eval-report.json"
    ]["primary_metrics"]["object_recall_rate"] = 0.0
    drifted["error_attribution_diagnostics"][
        "error_attribution_report:error-attribution-report.json"
    ]["summary"]["by_error_category"]["correct"] = 0
    drifted["failure_linkage_diagnostics"][
        "error_attribution_report:error-attribution-report.json"
    ]["graph_primary_metrics"]["object_recall_rate"] = 0.0
    drifted["summary"]["qa_eval_delta_report_count"] = 0
    drifted["summary"]["qa_diagnostic_slice_count"] = 0
    drifted["summary"]["error_attribution_diagnostic_count"] = 0
    drifted["summary"]["failure_linkage_diagnostic_count"] = 0
    drifted["summary"]["graph_construction_diagnostic_count"] = 0
    drifted["report_digest"] = lab.experiment_summary_report_digest(drifted)

    validation = lab.validate_experiment_summary_report(drifted)

    checks = {check["name"]: check for check in validation["checks"]}
    assert validation["valid"] is False
    assert checks["qa_diagnostic_slices"]["passed"] is False
    assert checks["qa_diagnostic_slices"]["expected"] == report["qa_diagnostic_slices"]
    assert checks["graph_construction_diagnostics"]["passed"] is False
    assert checks["graph_construction_diagnostics"]["expected"] == report[
        "graph_construction_diagnostics"
    ]
    assert checks["error_attribution_diagnostics"]["passed"] is False
    assert checks["error_attribution_diagnostics"]["expected"] == report[
        "error_attribution_diagnostics"
    ]
    assert checks["failure_linkage_diagnostics"]["passed"] is False
    assert checks["failure_linkage_diagnostics"]["expected"] == report[
        "failure_linkage_diagnostics"
    ]
    assert checks["research_questions"]["passed"] is False
    assert checks["research_questions"]["expected"] == report["research_questions"]
    assert checks["summary"]["passed"] is False
    assert checks["summary"]["expected"] == report["summary"]


def test_experiment_summary_validation_detects_comparison_source_mismatch_after_digest_recompute(
    tmp_path: Path,
) -> None:
    artifacts = _write_experiment_artifacts(tmp_path)
    manifest = lab.build_benchmark_artifacts(
        dataset_name="mock_benchmark",
        episode_paths=_write_mock_episodes(tmp_path / "episodes"),
        output_dir=tmp_path / "benchmark",
        max_qa_per_episode=3,
        qa_eval_delta_report_paths=(artifacts.qa_delta_path,),
        active_task_delta_report_paths=(artifacts.active_delta_path,),
    )
    report = lab.experiment_summary_report(manifest)
    drifted = json.loads(lab.experiment_summary_report_json(report))
    drifted["qa_delta_comparisons"][0]["digest"] = "0" * 64
    drifted["report_digest"] = lab.experiment_summary_report_digest(drifted)

    validation = lab.validate_experiment_summary_report(drifted)

    checks = {check["name"]: check for check in validation["checks"]}
    assert validation["valid"] is False
    assert checks["source_artifact_comparisons"]["passed"] is False
    assert checks["source_artifact_comparisons"]["expected"] == {
        "active_task_delta_report:active-delta-report.json": {
            "artifact_type": "active_task_delta_report",
            "digest": artifacts.active_delta["report_digest"],
            "path": str(artifacts.active_delta_path),
        },
        "qa_eval_delta_report:qa-delta-report.json": {
            "artifact_type": "qa_eval_delta_report",
            "digest": artifacts.qa_delta["report_digest"],
            "path": str(artifacts.qa_delta_path),
        },
    }
    assert checks["source_artifact_comparisons"]["actual"][
        "qa_eval_delta_report:qa-delta-report.json"
    ]["digest"] == "0" * 64


class ExperimentArtifacts(Protocol):
    qa_delta_path: Path
    qa_delta: dict[str, Any]
    active_delta_path: Path
    active_delta: dict[str, Any]
    graph_eval_path: Path
    graph_eval: dict[str, Any]
    error_attribution_path: Path
    error_attribution: dict[str, Any]
    offline_import_path: Path
    offline_import: dict[str, Any]


def _write_experiment_artifacts(tmp_path: Path) -> ExperimentArtifacts:
    class Artifacts:
        qa_delta_path: Path
        qa_delta: dict[str, Any]
        active_delta_path: Path
        active_delta: dict[str, Any]
        graph_eval_path: Path
        graph_eval: dict[str, Any]
        error_attribution_path: Path
        error_attribution: dict[str, Any]
        offline_import_path: Path
        offline_import: dict[str, Any]

    cases = list(_qa_metric_cases())
    relation_payload = lab.qa_case_to_dict(cases[1])
    relation_payload["tags"] = [*relation_payload["tags"], "dynamic", "memory"]
    cases[1] = lab.qa_case_from_dict(relation_payload)
    candidate_report = lab.qa_eval_report(cases, _qa_metric_predictions(tuple(cases)))
    baseline_report = lab.qa_eval_report(cases, ())
    candidate_report_path = tmp_path / "qa-candidate-report.json"
    baseline_report_path = tmp_path / "qa-baseline-report.json"
    qa_delta_path = tmp_path / "qa-delta-report.json"
    lab.save_qa_eval_report(candidate_report, candidate_report_path)
    lab.save_qa_eval_report(baseline_report, baseline_report_path)
    qa_delta = lab.qa_eval_delta_report(
        candidate_report,
        baseline_report,
        candidate_name="graph_tool",
        baseline_name="majority",
        candidate_report_path=candidate_report_path,
        baseline_report_path=baseline_report_path,
    )
    lab.save_qa_eval_delta_report(qa_delta, qa_delta_path)

    active_case = _case_by_id_suffix("object_location:plate_1")
    task = _task_for_case(active_case, max_actions=1)
    active_candidate_report_path = tmp_path / "active-candidate-report.json"
    active_baseline_report_path = tmp_path / "active-baseline-report.json"
    active_delta_path = tmp_path / "active-delta-report.json"
    candidate_result = lab.ActiveTaskResult(
        task_id=task.id,
        policy="oracle_evidence",
        answer=active_case.answer,
        success=True,
        action_count=1,
        evidence_nodes=active_case.required_nodes,
        evidence_edges=active_case.required_edges,
        final_step=task.initial_step + 1,
        confidence=1.0,
    )
    baseline_result = lab.ActiveTaskResult(
        task_id=task.id,
        policy="direct_answer",
        answer={},
        success=False,
        action_count=0,
        final_step=task.initial_step,
        confidence=0.0,
        error="missing_required_evidence",
    )
    active_candidate_report = lab.active_task_report((task,), (candidate_result,))
    active_baseline_report = lab.active_task_report((task,), (baseline_result,))
    lab.save_active_task_report(active_candidate_report, active_candidate_report_path)
    lab.save_active_task_report(active_baseline_report, active_baseline_report_path)
    active_delta = lab.active_task_delta_report(
        active_candidate_report,
        active_baseline_report,
        candidate_name="oracle_evidence",
        baseline_name="direct_answer",
        candidate_report_path=active_candidate_report_path,
        baseline_report_path=active_baseline_report_path,
    )
    lab.save_active_task_delta_report(active_delta, active_delta_path)

    oracle_graph = lab.load_scene_fixture("tabletop")
    predicted_graph = lab.graph_from_json(lab.graph_to_json(oracle_graph))
    for node in predicted_graph.nodes.values():
        if node.type == "object":
            node.attributes["source"] = "mock_segmenter"
    for edge in predicted_graph.edges:
        edge.attributes["source"] = "geometry"
    oracle_graph_path = tmp_path / "oracle-graph.json"
    predicted_graph_path = tmp_path / "predicted-graph.json"
    graph_eval_path = tmp_path / "graph-eval-report.json"
    qa_dataset_path = tmp_path / "qa.jsonl"
    prediction_path = tmp_path / "qa-predictions.jsonl"
    error_attribution_path = tmp_path / "error-attribution-report.json"
    offline_input_path = tmp_path / "offline-input.jsonl"
    offline_prediction_path = tmp_path / "offline-predictions.jsonl"
    offline_import_path = tmp_path / "offline-import-report.json"
    lab.save_graph_json(oracle_graph, oracle_graph_path)
    lab.save_graph_json(predicted_graph, predicted_graph_path)
    predictions = _qa_metric_predictions(tuple(cases))
    lab.save_qa_dataset(cases, qa_dataset_path)
    lab.save_qa_predictions(predictions, prediction_path)
    graph_eval = lab.graph_eval_report(
        oracle_graph,
        predicted_graph,
        oracle_path=oracle_graph_path,
        predicted_path=predicted_graph_path,
    )
    error_attribution = lab.error_attribution_report(
        cases,
        oracle_graph=oracle_graph,
        predicted_graph=predicted_graph,
        predictions=predictions,
        gold_path=qa_dataset_path,
        oracle_graph_path=oracle_graph_path,
        predicted_graph_path=predicted_graph_path,
        prediction_path=prediction_path,
    )
    lab.save_graph_eval_report(graph_eval, graph_eval_path)
    lab.save_error_attribution_report(error_attribution, error_attribution_path)
    lab.save_offline_prediction_records(
        tuple(
            lab.OfflinePredictionRecord(
                case_id=case.id,
                answer=case.answer,
                evidence_nodes=case.required_nodes,
                evidence_edges=case.required_edges,
                confidence=0.88,
            )
            for case in cases
        ),
        offline_input_path,
    )
    imported_predictions, offline_import = lab.import_offline_predictions(
        cases,
        lab.load_offline_prediction_records(offline_input_path),
        source_name="vlm_fixture",
        source_kind="vlm",
        source_metadata={
            "capabilities": ("spatial_qa", "graph_tool_query"),
            "dataset_id": "mock_eval",
            "model_id": "mock-vlm",
            "prompt_id": "spatial-qa-v1",
        },
        qa_path=qa_dataset_path,
        input_path=offline_input_path,
        prediction_path=offline_prediction_path,
    )
    lab.save_qa_predictions(imported_predictions, offline_prediction_path)
    lab.save_offline_prediction_import_report(offline_import, offline_import_path)

    artifacts = Artifacts()
    artifacts.qa_delta_path = qa_delta_path
    artifacts.qa_delta = qa_delta
    artifacts.active_delta_path = active_delta_path
    artifacts.active_delta = active_delta
    artifacts.graph_eval_path = graph_eval_path
    artifacts.graph_eval = graph_eval
    artifacts.error_attribution_path = error_attribution_path
    artifacts.error_attribution = error_attribution
    artifacts.offline_import_path = offline_import_path
    artifacts.offline_import = offline_import
    return artifacts


def _qa_metric_cases() -> tuple[lab.QACase, ...]:
    graph = lab.load_scene_fixture("tabletop")
    generated = lab.generate_qa_cases(
        graph,
        scene_id="tabletop_scene",
        episode_id="episode_001",
    )
    relation = next(case for case in generated if case.question_type == "relative_relation")
    nearest = next(case for case in generated if case.question_type == "nearest_object")
    return (generated[0], relation, nearest)


def _qa_metric_predictions(cases: tuple[lab.QACase, ...]) -> tuple[lab.QAPrediction, ...]:
    location, relation, _nearest = cases
    wrong_relation_answer = lab.qa_case_from_dict(lab.qa_case_to_dict(relation)).answer
    wrong_relation_answer["holds"] = not bool(relation.answer["holds"])
    return (
        lab.QAPrediction(
            id=location.id,
            answer=location.answer,
            evidence_nodes=location.required_nodes,
            evidence_edges=location.required_edges,
            confidence=0.99,
        ),
        lab.QAPrediction(
            id=relation.id,
            answer=wrong_relation_answer,
            evidence_nodes=relation.required_nodes[:1],
            evidence_edges=(),
            confidence=0.4,
        ),
    )


def _tabletop_cases() -> list[lab.QACase]:
    return lab.generate_qa_cases(
        lab.load_scene_fixture("tabletop"),
        scene_id="tabletop_scene",
        episode_id="episode_001",
    )


def _case_by_id_suffix(suffix: str) -> lab.QACase:
    return next(case for case in _tabletop_cases() if case.id.endswith(suffix))


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


def _write_mock_episodes(tmp_path: Path) -> tuple[Path, Path]:
    episode_dir = tmp_path / "episode-files"
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
