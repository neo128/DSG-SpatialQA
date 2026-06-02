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
RUN_MOCK_EXPERIMENT_SCRIPT = ROOT / "scripts" / "run_mock_experiment.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_run_mock_experiment_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_mock_experiment_script",
        RUN_MOCK_EXPERIMENT_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_mock_experiment_pipeline_builds_ready_final_record(tmp_path: Path) -> None:
    assert hasattr(lab, "run_mock_experiment")
    output_dir = tmp_path / "mock-experiment"

    result = lab.run_mock_experiment(
        output_dir=output_dir,
        dataset_name="mock_pipeline",
        max_qa_per_episode=3,
    )

    record = lab.load_experiment_record(result["record_path"])
    summary = lab.load_experiment_summary_report(result["summary_report_path"])
    manifest = lab.load_benchmark_manifest(result["manifest_path"])
    dashboard = lab.load_dashboard_bundle(result["dashboard_bundle_path"])
    predicted_report = lab.load_predicted_graph_report(
        result["predicted_graph_report_path"]
    )
    graph_eval_report = lab.load_graph_eval_report(result["graph_eval_report_path"])
    error_attribution_report = lab.load_error_attribution_report(
        result["error_attribution_report_path"]
    )
    graph_construction_delta = lab.load_qa_eval_delta_report(
        result["qa_graph_construction_delta_report_path"]
    )
    active_graph_construction_delta = lab.load_active_task_delta_report(
        result["active_graph_construction_delta_report_path"]
    )
    record_validation = lab.validate_experiment_record(record)
    record_comparison = lab.compare_experiment_record(record)

    assert result == {
        "schema_version": "dsg-spatialqa-lab.mock-experiment-result.v1",
        "dataset_name": "mock_pipeline",
        "output_dir": str(output_dir),
        "episode_path": str(output_dir / "episodes" / "mock-episode.jsonl"),
        "episode_paths": [str(output_dir / "episodes" / "mock-episode.jsonl")],
        "graph_path": str(output_dir / "benchmark" / "graphs" / "ai2thor_mock_001-oracle-graph.json"),
        "graph_paths": [
            str(output_dir / "benchmark" / "graphs" / "ai2thor_mock_001-oracle-graph.json")
        ],
        "predicted_graph_path": str(output_dir / "predicted" / "ai2thor_mock_001-predicted-graph.json"),
        "predicted_graph_paths": [
            str(output_dir / "predicted" / "ai2thor_mock_001-predicted-graph.json")
        ],
        "predicted_graph_report_path": str(output_dir / "reports" / "predicted-ai2thor_mock_001-report.json"),
        "predicted_graph_report_paths": [
            str(output_dir / "reports" / "predicted-ai2thor_mock_001-report.json")
        ],
        "graph_eval_report_path": str(output_dir / "reports" / "graph-eval-ai2thor_mock_001.json"),
        "graph_eval_report_paths": [
            str(output_dir / "reports" / "graph-eval-ai2thor_mock_001.json")
        ],
        "error_attribution_report_path": str(
            output_dir / "reports" / "error-attribution-ai2thor_mock_001.json"
        ),
        "error_attribution_report_paths": [
            str(output_dir / "reports" / "error-attribution-ai2thor_mock_001.json")
        ],
        "qa_path": str(output_dir / "benchmark" / "qa" / "ai2thor_mock_001-qa.jsonl"),
        "qa_paths": [str(output_dir / "benchmark" / "qa" / "ai2thor_mock_001-qa.jsonl")],
        "combined_qa_path": str(output_dir / "benchmark" / "qa" / "mock_pipeline-qa.jsonl"),
        "qa_candidate_name": "graph_tool",
        "qa_baseline_names": ["majority"],
        "qa_graph_construction_baseline_name": "predicted_graph_tool",
        "qa_prediction_paths": {
            "graph_tool": str(output_dir / "predictions" / "graph-tool-predictions.jsonl"),
            "majority": str(output_dir / "predictions" / "majority-predictions.jsonl"),
            "predicted_graph_tool": str(
                output_dir / "predictions" / "predicted-graph-tool-predictions.jsonl"
            ),
        },
        "qa_report_paths": {
            "graph_tool": str(output_dir / "reports" / "qa-graph-tool-report.json"),
            "majority": str(output_dir / "reports" / "qa-majority-report.json"),
            "predicted_graph_tool": str(
                output_dir / "reports" / "qa-predicted-graph-tool-report.json"
            ),
        },
        "qa_delta_report_paths": {
            "majority": str(output_dir / "reports" / "qa-delta-report.json"),
            "predicted_graph_tool": str(
                output_dir
                / "reports"
                / "zz-qa-delta-graph-tool-vs-predicted-graph-tool.json"
            ),
        },
        "graph_tool_prediction_path": str(output_dir / "predictions" / "graph-tool-predictions.jsonl"),
        "predicted_graph_tool_prediction_path": str(
            output_dir / "predictions" / "predicted-graph-tool-predictions.jsonl"
        ),
        "majority_prediction_path": str(output_dir / "predictions" / "majority-predictions.jsonl"),
        "qa_candidate_report_path": str(output_dir / "reports" / "qa-graph-tool-report.json"),
        "qa_baseline_report_path": str(output_dir / "reports" / "qa-majority-report.json"),
        "predicted_graph_tool_report_path": str(
            output_dir / "reports" / "qa-predicted-graph-tool-report.json"
        ),
        "qa_delta_report_path": str(output_dir / "reports" / "qa-delta-report.json"),
        "qa_graph_construction_delta_report_path": str(
            output_dir
            / "reports"
            / "zz-qa-delta-graph-tool-vs-predicted-graph-tool.json"
        ),
        "active_task_path": str(output_dir / "active" / "active-tasks.jsonl"),
        "active_candidate_report_path": str(output_dir / "reports" / "active-oracle-report.json"),
        "active_baseline_report_path": str(output_dir / "reports" / "active-direct-report.json"),
        "active_predicted_graph_report_path": str(
            output_dir / "reports" / "active-predicted-graph-report.json"
        ),
        "active_delta_report_path": str(output_dir / "reports" / "active-delta-report.json"),
        "active_graph_construction_delta_report_path": str(
            output_dir
            / "reports"
            / "zz-active-delta-oracle-vs-predicted-graph.json"
        ),
        "manifest_path": str(output_dir / "benchmark-manifest.json"),
        "summary_report_path": str(output_dir / "experiment-summary.json"),
        "dashboard_bundle_path": str(output_dir / "dashboard" / "dashboard.json"),
        "dashboard_index_path": str(output_dir / "dashboard" / "index.html"),
        "record_path": str(output_dir / "experiment-record.json"),
        "readiness_status": "ready",
        "verdict_counts": {
            "improved": 3,
            "inconclusive": 0,
            "regressed": 0,
            "unchanged": 1,
        },
        "record_digest": record["record_digest"],
    }
    assert manifest["summary"]["experiment_artifact_count"] == 7
    assert summary["summary"]["qa_eval_delta_report_count"] == 2
    assert summary["summary"]["active_task_delta_report_count"] == 2
    assert summary["summary"]["error_attribution_diagnostic_count"] == 1
    assert summary["summary"]["failure_linkage_diagnostic_count"] == 1
    assert summary["summary"]["graph_construction_diagnostic_count"] == 1
    assert summary["summary"]["qa_diagnostic_slice_count"] > 0
    assert summary["qa_diagnostic_slices"][
        "qa_eval_delta_report:qa-delta-report.json"
    ]["by_question_type"]
    assert summary["qa_diagnostic_slices"][
        "qa_eval_delta_report:qa-delta-report.json"
    ]["by_tag"]
    assert summary["qa_diagnostic_slices"][
        "qa_eval_delta_report:qa-delta-report.json"
    ]["by_reference_frame"]
    assert summary["qa_diagnostic_slices"][
        "qa_eval_delta_report:qa-delta-report.json"
    ]["by_scene_id"]
    assert summary["qa_diagnostic_slices"][
        "qa_eval_delta_report:qa-delta-report.json"
    ]["by_episode_id"]
    assert summary["source_artifact_digests"][
        "graph_eval_report:graph-eval-ai2thor_mock_001.json"
    ] == graph_eval_report["report_digest"]
    attribution_key = "error_attribution_report:error-attribution-ai2thor_mock_001.json"
    assert summary["source_artifact_digests"][
        attribution_key
    ] == error_attribution_report["report_digest"]
    assert summary["error_attribution_diagnostics"][attribution_key][
        "summary"
    ] == error_attribution_report["summary"]
    assert summary["error_attribution_diagnostics"][attribution_key]["summary"][
        "by_error_category"
    ]
    graph_key = "graph_eval_report:graph-eval-ai2thor_mock_001.json"
    assert summary["failure_linkage_diagnostics"][attribution_key][
        "graph_eval_artifact_key"
    ] == graph_key
    assert summary["failure_linkage_diagnostics"][attribution_key][
        "graph_primary_metrics"
    ] == summary["graph_construction_diagnostics"][graph_key]["primary_metrics"]
    assert summary["failure_linkage_diagnostics"][attribution_key][
        "attribution_summary"
    ] == error_attribution_report["summary"]
    assert summary["graph_construction_diagnostics"][graph_key]["primary_metrics"][
        "object_recall_rate"
    ] == graph_eval_report["metrics"]["object_recall"]["rate"]
    assert summary["graph_construction_diagnostics"][graph_key]["primary_metrics"][
        "relation_f1_rate"
    ] == graph_eval_report["metrics"]["relation_f1"]["rate"]
    assert summary["graph_construction_diagnostics"][graph_key][
        "source_breakdown"
    ] == graph_eval_report["breakdown"]["by_prediction_source"]
    assert summary["source_artifact_digests"][
        "predicted_graph_report:predicted-ai2thor_mock_001-report.json"
    ] == predicted_report["digest"]
    assert summary["readiness"]["status"] == "ready"
    assert dashboard["experiment_summary_review"]["readiness"]["status"] == "ready"
    assert dashboard["experiment_summary_review"]["failure_linkage_review"][0][
        "graph_eval_artifact_key"
    ] == graph_key
    assert dashboard["experiment_summary_review"]["failure_linkage_review"][0][
        "attribution_summary"
    ] == error_attribution_report["summary"]
    assert record["readiness_status"] == "ready"
    assert record["verdict_counts"]["improved"] == 3
    assert [
        row["baseline_name"]
        for row in record["research_question_matrix"]
        if row["research_question"] == "spatial_qa"
    ] == ["majority", "predicted_graph_tool"]
    assert [
        row["baseline_name"]
        for row in record["research_question_matrix"]
        if row["research_question"] == "interactive_task"
    ] == ["direct_answer", "predicted_graph_evidence"]
    assert record["research_question_verdicts"]["interactive_task"][
        "measurement_count"
    ] == 2
    assert record["dashboard_bundle"]["has_experiment_summary_review"] is True
    assert record["diagnostic_ledger"]["graph_construction_artifact_keys"] == [
        graph_key,
    ]
    assert record["diagnostic_ledger"]["error_attribution_artifact_keys"] == [
        attribution_key,
    ]
    assert record["diagnostic_ledger"]["failure_linkage_pairs"] == [
        {
            "error_attribution_artifact_key": attribution_key,
            "graph_eval_artifact_key": graph_key,
            "linked_by": "oracle_and_predicted_graph_digest",
        },
    ]
    assert record["source_artifact_digests"][
        "graph_eval_report:graph-eval-ai2thor_mock_001.json"
    ] == graph_eval_report["report_digest"]
    assert record["source_artifact_digests"][
        attribution_key
    ] == error_attribution_report["report_digest"]
    assert record["source_artifact_digests"][
        "predicted_graph_report:predicted-ai2thor_mock_001-report.json"
    ] == predicted_report["digest"]
    assert lab.validate_predicted_graph_report(predicted_report)["valid"] is True
    assert lab.compare_predicted_graph_report(predicted_report)["matches"] is True
    assert lab.validate_graph_eval_report(graph_eval_report)["valid"] is True
    assert lab.compare_graph_eval_report(graph_eval_report)["matches"] is True
    assert lab.validate_error_attribution_report(error_attribution_report)["valid"] is True
    assert (
        lab.compare_error_attribution_report(error_attribution_report)["matches"]
        is True
    )
    assert graph_construction_delta["candidate_name"] == "graph_tool"
    assert graph_construction_delta["baseline_name"] == "predicted_graph_tool"
    assert lab.compare_qa_eval_delta_report(graph_construction_delta)["matches"] is True
    assert active_graph_construction_delta["candidate_name"] == "oracle_evidence"
    assert active_graph_construction_delta["baseline_name"] == "predicted_graph_evidence"
    assert (
        lab.compare_active_task_delta_report(active_graph_construction_delta)["matches"]
        is True
    )
    assert record_validation["valid"] is True
    assert record_comparison["matches"] is True


def test_mock_experiment_pipeline_aggregates_explicit_episode_configs(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "multi-mock-experiment"
    episode_configs = (
        lab.AI2ThorAdapterConfig(
            scene_id="FloorPlan1",
            episode_id="ai2thor_mock_001",
            steps=(1, 2),
            actions=("Initialize", "MoveAhead"),
        ),
        lab.AI2ThorAdapterConfig(
            scene_id="FloorPlan2",
            episode_id="ai2thor_mock_002",
            steps=(1, 2),
            actions=("Initialize", "RotateRight"),
        ),
    )

    result = lab.run_mock_experiment(
        output_dir=output_dir,
        dataset_name="multi_mock_pipeline",
        max_qa_per_episode=3,
        episode_configs=episode_configs,
    )

    manifest = lab.load_benchmark_manifest(result["manifest_path"])
    candidate_report = lab.load_qa_eval_report(result["qa_candidate_report_path"])
    baseline_report = lab.load_qa_eval_report(result["qa_baseline_report_path"])
    active_report = lab.load_active_task_report(result["active_candidate_report_path"])
    active_tasks = lab.load_active_eqa_tasks(result["active_task_path"])
    record = lab.load_experiment_record(result["record_path"])

    assert result["episode_paths"] == [
        str(output_dir / "episodes" / "ai2thor_mock_001.jsonl"),
        str(output_dir / "episodes" / "ai2thor_mock_002.jsonl"),
    ]
    assert result["graph_paths"] == [
        str(output_dir / "benchmark" / "graphs" / "ai2thor_mock_001-oracle-graph.json"),
        str(output_dir / "benchmark" / "graphs" / "ai2thor_mock_002-oracle-graph.json"),
    ]
    assert result["predicted_graph_paths"] == [
        str(output_dir / "predicted" / "ai2thor_mock_001-predicted-graph.json"),
        str(output_dir / "predicted" / "ai2thor_mock_002-predicted-graph.json"),
    ]
    assert result["predicted_graph_report_paths"] == [
        str(output_dir / "reports" / "predicted-ai2thor_mock_001-report.json"),
        str(output_dir / "reports" / "predicted-ai2thor_mock_002-report.json"),
    ]
    assert result["graph_eval_report_paths"] == [
        str(output_dir / "reports" / "graph-eval-ai2thor_mock_001.json"),
        str(output_dir / "reports" / "graph-eval-ai2thor_mock_002.json"),
    ]
    assert result["error_attribution_report_paths"] == [
        str(output_dir / "reports" / "error-attribution-ai2thor_mock_001.json"),
        str(output_dir / "reports" / "error-attribution-ai2thor_mock_002.json"),
    ]
    assert result["qa_paths"] == [
        str(output_dir / "benchmark" / "qa" / "ai2thor_mock_001-qa.jsonl"),
        str(output_dir / "benchmark" / "qa" / "ai2thor_mock_002-qa.jsonl"),
    ]
    assert result["combined_qa_path"] == str(
        output_dir / "benchmark" / "qa" / "multi_mock_pipeline-qa.jsonl"
    )
    assert manifest["summary"]["episode_count"] == 2
    assert manifest["summary"]["qa_count"] == 6
    assert manifest["summary"]["experiment_artifact_count"] == 10
    summary = lab.load_experiment_summary_report(result["summary_report_path"])
    assert candidate_report["summary"]["case_count"] == 6
    assert baseline_report["summary"]["case_count"] == 6
    assert active_report["summary"]["task_count"] == 2
    assert summary["summary"]["error_attribution_diagnostic_count"] == 2
    assert summary["summary"]["failure_linkage_diagnostic_count"] == 2
    assert summary["summary"]["graph_construction_diagnostic_count"] == 2
    assert [task.episode_id for task in active_tasks] == [
        "ai2thor_mock_001",
        "ai2thor_mock_002",
    ]
    assert record["readiness_status"] == "ready"
    assert lab.compare_experiment_record(record)["matches"] is True


def test_mock_experiment_pipeline_writes_qa_agent_matrix(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "matrix-mock-experiment"

    result = lab.run_mock_experiment(
        output_dir=output_dir,
        dataset_name="matrix_mock_pipeline",
        max_qa_per_episode=3,
        qa_baseline_names=("majority", "graph_text", "caption_memory"),
    )

    manifest = lab.load_benchmark_manifest(result["manifest_path"])
    summary = lab.load_experiment_summary_report(result["summary_report_path"])
    dashboard = lab.load_dashboard_bundle(result["dashboard_bundle_path"])
    record = lab.load_experiment_record(result["record_path"])
    html = Path(result["dashboard_index_path"]).read_text(encoding="utf-8")
    delta_reports = {
        name: lab.load_qa_eval_delta_report(path)
        for name, path in result["qa_delta_report_paths"].items()
    }

    assert result["qa_candidate_name"] == "graph_tool"
    assert result["qa_baseline_names"] == [
        "majority",
        "graph_text",
        "caption_memory",
    ]
    assert result["qa_graph_construction_baseline_name"] == "predicted_graph_tool"
    assert result["qa_prediction_paths"] == {
        "graph_tool": str(output_dir / "predictions" / "graph-tool-predictions.jsonl"),
        "majority": str(output_dir / "predictions" / "majority-predictions.jsonl"),
        "graph_text": str(output_dir / "predictions" / "graph-text-predictions.jsonl"),
        "caption_memory": str(
            output_dir / "predictions" / "caption-memory-predictions.jsonl"
        ),
        "predicted_graph_tool": str(
            output_dir / "predictions" / "predicted-graph-tool-predictions.jsonl"
        ),
    }
    assert result["qa_report_paths"] == {
        "graph_tool": str(output_dir / "reports" / "qa-graph-tool-report.json"),
        "majority": str(output_dir / "reports" / "qa-majority-report.json"),
        "graph_text": str(output_dir / "reports" / "qa-graph-text-report.json"),
        "caption_memory": str(output_dir / "reports" / "qa-caption-memory-report.json"),
        "predicted_graph_tool": str(
            output_dir / "reports" / "qa-predicted-graph-tool-report.json"
        ),
    }
    assert result["qa_delta_report_paths"] == {
        "majority": str(
            output_dir / "reports" / "qa-delta-00-graph-tool-vs-majority.json"
        ),
        "graph_text": str(
            output_dir / "reports" / "qa-delta-01-graph-tool-vs-graph-text.json"
        ),
        "caption_memory": str(
            output_dir / "reports" / "qa-delta-02-graph-tool-vs-caption-memory.json"
        ),
        "predicted_graph_tool": str(
            output_dir
            / "reports"
            / "zz-qa-delta-graph-tool-vs-predicted-graph-tool.json"
        ),
    }
    assert result["qa_delta_report_path"] == result["qa_delta_report_paths"]["majority"]
    assert result["qa_graph_construction_delta_report_path"] == (
        result["qa_delta_report_paths"]["predicted_graph_tool"]
    )
    assert manifest["summary"]["experiment_artifact_count"] == 9
    assert summary["summary"]["qa_eval_delta_report_count"] == 4
    assert summary["summary"]["active_task_delta_report_count"] == 2
    assert [
        row["baseline_name"]
        for row in dashboard["experiment_summary_review"]["research_question_matrix"]
        if row["research_question"] == "spatial_qa"
    ] == ["majority", "graph_text", "caption_memory", "predicted_graph_tool"]
    assert [
        row["baseline_name"]
        for row in record["research_question_matrix"]
        if row["research_question"] == "spatial_qa"
    ] == ["majority", "graph_text", "caption_memory", "predicted_graph_tool"]
    assert record["research_question_verdicts"]["spatial_qa"][
        "measurement_count"
    ] == 4
    assert record["research_question_verdicts"]["interactive_task"][
        "measurement_count"
    ] == 2
    assert "Measurement Matrix" in html
    assert "graph_tool" in html
    assert "caption_memory" in html
    assert {
        measurement["baseline_name"]
        for measurement in summary["research_questions"]["spatial_qa"]["measurements"]
    } == {"majority", "graph_text", "caption_memory", "predicted_graph_tool"}
    assert {
        name: report["baseline_name"] for name, report in delta_reports.items()
    } == {
        "majority": "majority",
        "graph_text": "graph_text",
        "caption_memory": "caption_memory",
        "predicted_graph_tool": "predicted_graph_tool",
    }
    assert all(
        lab.compare_qa_eval_delta_report(report)["matches"] is True
        for report in delta_reports.values()
    )


def test_run_mock_experiment_cli_writes_pipeline_result(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_mock_experiment_script()
    main = cast(MainFn, getattr(module, "main"))
    output_dir = tmp_path / "cli-mock-experiment"

    assert main(
        [
            "--output-dir",
            str(output_dir),
            "--dataset-name",
            "cli_mock_pipeline",
            "--max-qa-per-episode",
            "3",
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    record = lab.load_experiment_record(output["record_path"])
    assert output["action"] == "run_mock_experiment"
    assert output["record_digest"] == record["record_digest"]


def test_run_mock_experiment_cli_accepts_repeated_qa_baselines(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_mock_experiment_script()
    main = cast(MainFn, getattr(module, "main"))
    output_dir = tmp_path / "cli-matrix-mock-experiment"

    assert main(
        [
            "--output-dir",
            str(output_dir),
            "--dataset-name",
            "cli_matrix_mock_pipeline",
            "--max-qa-per-episode",
            "3",
            "--qa-baseline",
            "majority",
            "--qa-baseline",
            "graph_text",
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    summary = lab.load_experiment_summary_report(output["summary_report_path"])
    assert output["action"] == "run_mock_experiment"
    assert output["valid"] is True
    assert output["qa_baseline_names"] == ["majority", "graph_text"]
    assert output["qa_delta_report_paths"] == {
        "majority": str(
            output_dir / "reports" / "qa-delta-00-graph-tool-vs-majority.json"
        ),
        "graph_text": str(
            output_dir / "reports" / "qa-delta-01-graph-tool-vs-graph-text.json"
        ),
        "predicted_graph_tool": str(
            output_dir
            / "reports"
            / "zz-qa-delta-graph-tool-vs-predicted-graph-tool.json"
        ),
    }
    assert output["qa_graph_construction_baseline_name"] == "predicted_graph_tool"
    assert summary["summary"]["qa_eval_delta_report_count"] == 3
    assert output["readiness_status"] == "ready"
    assert output["verdict_counts"] == {
        "improved": 3,
        "inconclusive": 0,
        "regressed": 0,
        "unchanged": 1,
    }


def test_run_mock_experiment_cli_can_generate_multiple_mock_episodes(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_mock_experiment_script()
    main = cast(MainFn, getattr(module, "main"))
    output_dir = tmp_path / "cli-multi-mock-experiment"

    assert main(
        [
            "--output-dir",
            str(output_dir),
            "--dataset-name",
            "cli_multi_mock_pipeline",
            "--max-qa-per-episode",
            "3",
            "--episode-count",
            "2",
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    manifest = lab.load_benchmark_manifest(output["manifest_path"])
    record = lab.load_experiment_record(output["record_path"])
    assert output["action"] == "run_mock_experiment"
    assert output["valid"] is True
    assert output["episode_paths"] == [
        str(output_dir / "episodes" / "ai2thor_mock_001.jsonl"),
        str(output_dir / "episodes" / "ai2thor_mock_002.jsonl"),
    ]
    assert manifest["summary"]["episode_count"] == 2
    assert manifest["summary"]["qa_count"] == 6
    assert output["readiness_status"] == "ready"
    assert output["record_digest"] == record["record_digest"]
