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
EXPORT_DASHBOARD_SCRIPT = ROOT / "scripts" / "export_dashboard.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_export_dashboard_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "export_dashboard_script",
        EXPORT_DASHBOARD_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dashboard_bundle_contains_qa_prediction_eval_attribution_and_subgraph() -> None:
    assert hasattr(lab, "dashboard_bundle")
    assert hasattr(lab, "dashboard_bundle_digest")
    graph = lab.load_scene_fixture("tabletop")
    case = _case_by_type("relative_relation")
    prediction = lab.QAPrediction(
        id=case.id,
        answer={"wrong": True},
        evidence_nodes=(case.required_nodes[0],),
        confidence=0.2,
    )
    eval_report = lab.qa_eval_report((case,), (prediction,))
    attribution = lab.error_attribution_report(
        (case,),
        oracle_graph=graph,
        predicted_graph=_graph_without_edge(case.required_edges[0]),
        predictions=(prediction,),
    )

    bundle = lab.dashboard_bundle(
        (case,),
        predictions=(prediction,),
        qa_eval_report=eval_report,
        graph=graph,
        error_attribution_report=attribution,
    )

    assert bundle["schema_version"] == "dsg-spatialqa-lab.dashboard-bundle.v1"
    assert bundle["bundle_digest"] == lab.dashboard_bundle_digest(bundle)
    empty_axis_summary = {
        "answer_correct_count": 0,
        "by_error_category": {},
        "by_evidence_error_category": {},
        "by_predicted_evidence_source": {},
        "case_count": 0,
        "error_count": 0,
        "oracle_graph_tool_correct_count": 0,
        "predicted_graph_tool_correct_count": 0,
    }
    evidence_missing_axis_summary = {
        "answer_correct_count": 0,
        "by_error_category": {"evidence_missing": 1},
        "by_evidence_error_category": {"missing_relation": 1},
        "by_predicted_evidence_source": {
            "unknown": {
                "by_error_category": {"evidence_missing": 1},
                "by_evidence_error_category": {"missing_relation": 1},
                "case_count": 1,
                "error_count": 1,
            },
        },
        "case_count": 1,
        "error_count": 1,
        "oracle_graph_tool_correct_count": 1,
        "predicted_graph_tool_correct_count": 1,
    }
    assert bundle["summary"] == {
        "case_count": 1,
        "prediction_count": 1,
        "eval_result_count": 1,
        "attribution_count": 1,
        "exact_match_count": 0,
        "by_question_type": {"relative_relation": 1},
        "by_error_category": {"evidence_missing": 1},
        "by_predicted_evidence_source": {
            "unknown": {
                "by_error_category": {"evidence_missing": 1},
                "by_evidence_error_category": {"missing_relation": 1},
                "case_count": 1,
                "error_count": 1,
            },
        },
        "by_research_axis": {
            "dynamic_memory": empty_axis_summary,
            "graph_tool_query": evidence_missing_axis_summary,
            "spatial_qa": evidence_missing_axis_summary,
        },
    }
    assert bundle["graph_summary"] == lab.graph_summary(graph)
    item = bundle["cases"][0]
    assert item["case_id"] == case.id
    assert item["qa_case"] == lab.qa_case_to_dict(case)
    assert item["prediction"] == lab.qa_prediction_to_dict(prediction)
    assert item["eval_result"]["case_id"] == case.id
    assert item["error_attribution"]["error_category"] == "evidence_missing"
    assert item["frame_paths"] == {}
    assert [node["id"] for node in item["evidence_subgraph"]["nodes"]] == [
        "mug_1",
        "plate_1",
    ]
    assert [edge["id"] for edge in item["evidence_subgraph"]["edges"]] == [
        case.required_edges[0],
    ]


def test_dashboard_bundle_exposes_predicted_evidence_source_review() -> None:
    graph = lab.load_scene_fixture("tabletop")
    case = _case_by_type("object_location")
    predicted_graph = _graph_with_moved_mug_source("vlm_detector")
    prediction = lab.QAPrediction(id=case.id, answer={"wrong": True}, confidence=0.2)
    eval_report = lab.qa_eval_report((case,), (prediction,))
    attribution = lab.error_attribution_report(
        (case,),
        oracle_graph=graph,
        predicted_graph=predicted_graph,
        predictions=(prediction,),
    )

    bundle = lab.dashboard_bundle(
        (case,),
        predictions=(prediction,),
        qa_eval_report=eval_report,
        graph=predicted_graph,
        error_attribution_report=attribution,
    )
    html = lab.dashboard_html(bundle)

    assert bundle["summary"]["by_predicted_evidence_source"] == {
        "vlm_detector": {
            "by_error_category": {"graph_construction": 1},
            "by_evidence_error_category": {"none": 1},
            "case_count": 1,
            "error_count": 1,
        },
    }
    assert bundle["cases"][0]["predicted_evidence_sources"] == ["vlm_detector"]
    assert "Evidence Source" in html
    assert "vlm_detector" in html


def test_dashboard_bundle_exposes_research_axis_attribution_review() -> None:
    graph = lab.load_scene_fixture("tabletop")
    dynamic_case = _case_with_tags(
        _case_by_type("relative_relation"),
        ("dynamic", "memory"),
    )
    graph_tool_case = _case_by_type("object_location")
    predicted_graph = _graph_with_moved_mug_source("vlm_detector")
    predictions = (
        lab.QAPrediction(id=dynamic_case.id, answer={"wrong": True}, confidence=0.2),
        lab.QAPrediction(id=graph_tool_case.id, answer={"wrong": True}, confidence=0.2),
    )
    eval_report = lab.qa_eval_report((dynamic_case, graph_tool_case), predictions)
    attribution = lab.error_attribution_report(
        (dynamic_case, graph_tool_case),
        oracle_graph=graph,
        predicted_graph=predicted_graph,
        predictions=predictions,
    )

    bundle = lab.dashboard_bundle(
        (dynamic_case, graph_tool_case),
        predictions=predictions,
        qa_eval_report=eval_report,
        graph=predicted_graph,
        error_attribution_report=attribution,
    )
    html = lab.dashboard_html(bundle)

    assert bundle["summary"]["by_research_axis"]["dynamic_memory"][
        "by_error_category"
    ] == {"reasoning_or_tool_use": 1}
    assert bundle["summary"]["by_research_axis"]["graph_tool_query"][
        "by_error_category"
    ] == {
        "graph_construction": 1,
        "reasoning_or_tool_use": 1,
    }
    assert bundle["summary"]["by_research_axis"]["spatial_qa"]["case_count"] == 2
    assert bundle["cases"][0]["research_axes"] == [
        "dynamic_memory",
        "graph_tool_query",
        "spatial_qa",
    ]
    assert bundle["cases"][1]["research_axes"] == [
        "graph_tool_query",
        "spatial_qa",
    ]
    assert 'id="research-axis-filter"' in html
    assert '<option value="dynamic_memory">dynamic_memory</option>' in html
    assert 'data-research-axes="dynamic_memory|graph_tool_query|spatial_qa"' in html


def test_dashboard_html_exposes_predicted_evidence_source_filter_metadata() -> None:
    graph = lab.load_scene_fixture("tabletop")
    case = _case_by_type("object_location")
    predicted_graph = _graph_with_moved_mug_source("vlm_detector")
    prediction = lab.QAPrediction(id=case.id, answer={"wrong": True}, confidence=0.2)
    eval_report = lab.qa_eval_report((case,), (prediction,))
    attribution = lab.error_attribution_report(
        (case,),
        oracle_graph=graph,
        predicted_graph=predicted_graph,
        predictions=(prediction,),
    )
    bundle = lab.dashboard_bundle(
        (case,),
        predictions=(prediction,),
        qa_eval_report=eval_report,
        graph=predicted_graph,
        error_attribution_report=attribution,
    )

    html = lab.dashboard_html(bundle)

    assert 'id="evidence-source-filter"' in html
    assert '<option value="vlm_detector">vlm_detector</option>' in html
    assert 'data-evidence-sources="vlm_detector"' in html


def test_dashboard_validation_detects_predicted_evidence_source_summary_drift() -> None:
    graph = lab.load_scene_fixture("tabletop")
    case = _case_by_type("object_location")
    predicted_graph = _graph_with_moved_mug_source("vlm_detector")
    prediction = lab.QAPrediction(id=case.id, answer={"wrong": True}, confidence=0.2)
    eval_report = lab.qa_eval_report((case,), (prediction,))
    attribution = lab.error_attribution_report(
        (case,),
        oracle_graph=graph,
        predicted_graph=predicted_graph,
        predictions=(prediction,),
    )
    bundle = lab.dashboard_bundle(
        (case,),
        predictions=(prediction,),
        qa_eval_report=eval_report,
        graph=predicted_graph,
        error_attribution_report=attribution,
    )
    bundle["summary"]["by_predicted_evidence_source"] = {}
    bundle["bundle_digest"] = lab.dashboard_bundle_digest(bundle)

    validation = lab.validate_dashboard_bundle(bundle)
    checks = {check["name"]: check for check in validation["checks"]}

    assert validation["valid"] is False
    assert checks["predicted_evidence_source_summary"] == {
        "name": "predicted_evidence_source_summary",
        "passed": False,
        "expected": {
            "vlm_detector": {
                "by_error_category": {"graph_construction": 1},
                "by_evidence_error_category": {"none": 1},
                "case_count": 1,
                "error_count": 1,
            },
        },
        "actual": {},
    }


def test_dashboard_validation_detects_research_axis_summary_drift() -> None:
    graph = lab.load_scene_fixture("tabletop")
    dynamic_case = _case_with_tags(
        _case_by_type("relative_relation"),
        ("dynamic", "memory"),
    )
    predicted_graph = _graph_with_moved_mug_source("vlm_detector")
    prediction = lab.QAPrediction(
        id=dynamic_case.id,
        answer={"wrong": True},
        confidence=0.2,
    )
    eval_report = lab.qa_eval_report((dynamic_case,), (prediction,))
    attribution = lab.error_attribution_report(
        (dynamic_case,),
        oracle_graph=graph,
        predicted_graph=predicted_graph,
        predictions=(prediction,),
    )
    bundle = lab.dashboard_bundle(
        (dynamic_case,),
        predictions=(prediction,),
        qa_eval_report=eval_report,
        graph=predicted_graph,
        error_attribution_report=attribution,
    )
    expected_axis_summary = bundle["summary"]["by_research_axis"]
    bundle["summary"]["by_research_axis"] = {}
    bundle["bundle_digest"] = lab.dashboard_bundle_digest(bundle)

    validation = lab.validate_dashboard_bundle(bundle)
    checks = {check["name"]: check for check in validation["checks"]}

    assert validation["valid"] is False
    assert checks["research_axis_summary"] == {
        "name": "research_axis_summary",
        "passed": False,
        "expected": expected_axis_summary,
        "actual": {},
    }


def test_dashboard_compare_detects_stale_prediction_input(tmp_path: Path) -> None:
    assert hasattr(lab, "compare_dashboard_bundle")
    graph = lab.load_scene_fixture("tabletop")
    case = _case_by_type("object_location")
    prediction = lab.QAPrediction(id=case.id, answer=case.answer, confidence=0.9)
    qa_path = tmp_path / "qa.jsonl"
    pred_path = tmp_path / "predictions.jsonl"
    eval_path = tmp_path / "qa-eval-report.json"
    graph_path = tmp_path / "graph.json"
    eval_report = lab.qa_eval_report(
        (case,),
        (prediction,),
        gold_path=qa_path,
        prediction_path=pred_path,
    )
    lab.save_qa_dataset((case,), qa_path)
    lab.save_qa_predictions((prediction,), pred_path)
    lab.save_qa_eval_report(eval_report, eval_path)
    lab.save_graph_json(graph, graph_path)
    bundle = lab.dashboard_bundle(
        (case,),
        predictions=(prediction,),
        qa_eval_report=eval_report,
        graph=graph,
        qa_path=qa_path,
        prediction_path=pred_path,
        qa_eval_report_path=eval_path,
        graph_path=graph_path,
    )

    comparison = lab.compare_dashboard_bundle(bundle)
    changed_prediction = lab.QAPrediction(
        id=case.id,
        answer={"changed": True},
        confidence=0.1,
    )
    lab.save_qa_predictions((changed_prediction,), pred_path)
    stale_comparison = lab.compare_dashboard_bundle(bundle)
    stale_checks = {check["name"]: check for check in stale_comparison["checks"]}

    assert bundle["source_paths"] == {
        "graph_path": str(graph_path),
        "prediction_path": str(pred_path),
        "qa_eval_report_path": str(eval_path),
        "qa_path": str(qa_path),
    }
    assert comparison["matches"] is True
    assert stale_comparison["matches"] is False
    assert stale_comparison["saved_bundle_digest"] == bundle["bundle_digest"]
    assert stale_comparison["current_bundle_digest"] != bundle["bundle_digest"]
    assert stale_checks["bundle_digest_matches_current"]["passed"] is False
    assert stale_checks["cases_match_current"]["passed"] is False


def test_dashboard_bundle_handles_missing_optional_attribution_and_writes_html(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "dashboard_bundle_json")
    assert hasattr(lab, "dashboard_html")
    assert hasattr(lab, "save_dashboard_bundle")
    assert hasattr(lab, "export_dashboard")
    graph = lab.load_scene_fixture("tabletop")
    case = _case_by_type("object_location")
    prediction = lab.QAPrediction(id=case.id, answer=case.answer, confidence=0.9)
    eval_report = lab.qa_eval_report((case,), (prediction,))

    bundle = lab.dashboard_bundle(
        (case,),
        predictions=(prediction,),
        qa_eval_report=eval_report,
        graph=graph,
    )
    bundle_path = tmp_path / "dashboard.json"
    output_dir = tmp_path / "dashboard"
    saved_path = lab.save_dashboard_bundle(bundle, bundle_path)
    export_result = lab.export_dashboard(bundle, output_dir)
    html = (output_dir / "index.html").read_text(encoding="utf-8")

    assert saved_path == bundle_path
    assert json.loads(lab.dashboard_bundle_json(bundle)) == bundle
    assert bundle["summary"]["by_error_category"] == {}
    assert bundle["cases"][0]["error_attribution"] is None
    assert export_result == {
        "bundle_path": str(output_dir / "dashboard.json"),
        "index_path": str(output_dir / "index.html"),
        "summary": bundle["summary"],
        "digest": bundle["bundle_digest"],
    }
    assert json.loads((output_dir / "dashboard.json").read_text(encoding="utf-8")) == bundle
    assert "DSG-SpatialQA Dashboard" in html
    assert 'id="question-type-filter"' in html
    assert 'id="error-category-filter"' in html
    assert 'id="dashboard-data"' in html


def test_dashboard_bundle_includes_active_task_review_panels() -> None:
    graph = lab.load_scene_fixture("tabletop")
    qa_case = _case_by_id_suffix("object_location:plate_1")
    prediction = lab.QAPrediction(id=qa_case.id, answer=qa_case.answer, confidence=0.9)
    eval_report = lab.qa_eval_report((qa_case,), (prediction,))
    active_task = _task_for_case(qa_case, max_actions=1)
    active_result = lab.ActiveGraphAgent(policy="oracle_evidence").run(
        active_task,
        lab.MockActiveEnvironment(
            {
                active_task.initial_step: _graph_without_object("plate_1"),
                active_task.initial_step + 1: graph,
            }
        ),
    )
    active_report = lab.active_task_report(
        (active_task,),
        (active_result,),
        task_path="active-tasks.jsonl",
        graph_path="graph.json",
        policy="oracle_evidence",
    )

    bundle = lab.dashboard_bundle(
        (qa_case,),
        predictions=(prediction,),
        qa_eval_report=eval_report,
        graph=graph,
        active_task_report=active_report,
    )
    review = bundle["active_task_review"]
    panel = review["panels"][0]
    html = lab.dashboard_html(bundle)

    assert bundle["bundle_digest"] == lab.dashboard_bundle_digest(bundle)
    assert review["report_digest"] == active_report["report_digest"]
    assert review["summary"] == active_report["summary"]
    assert review["metrics"] == active_report["metrics"]
    assert review["budget_analysis"] == active_report["budget_analysis"]
    assert panel["task_id"] == active_task.id
    assert panel["task"] == lab.active_eqa_task_to_dict(active_task)
    assert panel["result"] == lab.active_task_result_to_dict(active_result)
    assert panel["action_evidence_snapshots"] == list(
        active_result.action_evidence_snapshots
    )
    assert panel["case_result"]["evidence_coverage"] == 1.0
    assert panel["evidence"] == {
        "observed_edges": list(active_result.evidence_edges),
        "observed_nodes": list(active_result.evidence_nodes),
        "required_edges": list(active_task.required_evidence["edges"]),
        "required_nodes": list(active_task.required_evidence["nodes"]),
        "missing_edges": [],
        "missing_nodes": [],
    }
    assert panel["transcript"] == [
        {
            "action": "observe_required_evidence",
            "from_step": 1,
            "to_step": 2,
        }
    ]
    assert "Active Tasks" in html
    assert active_task.id in html
    assert "observe_required_evidence" in html


def test_dashboard_bundle_includes_active_task_delta_review() -> None:
    graph = lab.load_scene_fixture("tabletop")
    qa_case = _case_by_id_suffix("object_location:plate_1")
    prediction = lab.QAPrediction(id=qa_case.id, answer=qa_case.answer, confidence=0.9)
    eval_report = lab.qa_eval_report((qa_case,), (prediction,))
    active_task = _task_for_case(qa_case, max_actions=1)
    baseline_result = lab.ActiveTaskResult(
        task_id=active_task.id,
        policy="direct_answer",
        answer={},
        success=False,
        action_count=0,
        final_step=active_task.initial_step,
        confidence=0.0,
        error="missing_required_evidence",
    )
    candidate_result = lab.ActiveTaskResult(
        task_id=active_task.id,
        policy="oracle_evidence",
        answer=qa_case.answer,
        success=True,
        action_count=1,
        evidence_nodes=qa_case.required_nodes,
        evidence_edges=qa_case.required_edges,
        final_step=active_task.initial_step + 1,
        confidence=1.0,
    )
    candidate_report = lab.active_task_report((active_task,), (candidate_result,))
    baseline_report = lab.active_task_report((active_task,), (baseline_result,))
    delta_report = lab.active_task_delta_report(
        candidate_report,
        baseline_report,
        candidate_name="oracle_evidence",
        baseline_name="direct_answer",
    )

    bundle = lab.dashboard_bundle(
        (qa_case,),
        predictions=(prediction,),
        qa_eval_report=eval_report,
        graph=graph,
        active_task_delta_report=delta_report,
    )
    review = bundle["active_task_delta_review"]
    html = lab.dashboard_html(bundle)
    validation = lab.validate_dashboard_bundle(bundle)

    assert review == {
        "schema_version": delta_report["schema_version"],
        "report_digest": delta_report["report_digest"],
        "candidate_name": "oracle_evidence",
        "baseline_name": "direct_answer",
        "candidate_report_digest": candidate_report["report_digest"],
        "baseline_report_digest": baseline_report["report_digest"],
        "paths": {
            "candidate_report_path": None,
            "baseline_report_path": None,
        },
        "summary_delta": delta_report["summary_delta"],
        "metrics_delta": delta_report["metrics_delta"],
        "budget_delta": delta_report["budget_delta"],
    }
    assert validation["valid"] is True
    assert "Active Task Delta" in html
    assert "oracle_evidence" in html
    assert "direct_answer" in html
    assert "task_success" in html


def test_dashboard_bundle_includes_experiment_summary_review() -> None:
    graph = lab.load_scene_fixture("tabletop")
    qa_case = _case_by_type("object_location")
    prediction = lab.QAPrediction(id=qa_case.id, answer=qa_case.answer, confidence=0.9)
    eval_report = lab.qa_eval_report((qa_case,), (prediction,))
    summary_report = _experiment_summary_report_for_dashboard()

    bundle = lab.dashboard_bundle(
        (qa_case,),
        predictions=(prediction,),
        qa_eval_report=eval_report,
        graph=graph,
        experiment_summary_report=summary_report,
    )
    review = bundle["experiment_summary_review"]
    html = lab.dashboard_html(bundle)
    validation = lab.validate_dashboard_bundle(bundle)

    assert review == {
        "schema_version": summary_report["schema_version"],
        "report_digest": summary_report["report_digest"],
        "manifest_path": "benchmark-manifest.json",
        "manifest_digest": "manifest-digest",
        "summary": summary_report["summary"],
        "readiness": summary_report["readiness"],
        "source_artifact_digests": summary_report["source_artifact_digests"],
        "research_questions": summary_report["research_questions"],
        "source_profile_matrix": summary_report["source_profile_matrix"],
        "failure_linkage_review": [
            {
                "error_attribution_artifact_key": (
                    "error_attribution_report:error-attribution-report.json"
                ),
                "graph_eval_artifact_key": "graph_eval_report:graph-eval-report.json",
                "linked_by": "oracle_and_predicted_graph_digest",
                "graph_primary_metrics": {
                    "object_recall_rate": 0.75,
                    "relation_f1_rate": 0.5,
                    "state_accuracy_rate": 1.0,
                },
                "graph_diagnostics": {
                    "duplicate_track_count": 0,
                    "id_fragmentation_count": 1,
                },
                "attribution_summary": {
                    "answer_correct_count": 0,
                    "by_error_category": {
                        "graph_construction": 2,
                        "reasoning_or_tool_use": 1,
                    },
                    "case_count": 3,
                    "error_count": 3,
                },
            },
        ],
        "research_question_matrix": [
            _matrix_row(
                research_question="dynamic_memory",
                metric_value=0.0,
                source_artifact_type="qa_eval_delta_report",
                supporting_metrics={"mean_evidence_node_recall_delta": 0.5},
            ),
            _matrix_row(
                research_question="graph_tool_query",
                metric_value=0.333333,
                source_artifact_type="qa_eval_delta_report",
                supporting_metrics={"mean_evidence_node_recall_delta": 0.5},
            ),
            _matrix_row(
                research_question="interactive_task",
                metric_name="task_success_rate_delta",
                metric_value=1.0,
                source_artifact_type="active_task_delta_report",
                supporting_metrics={"success_count_delta": 1},
            ),
            _matrix_row(
                research_question="spatial_qa",
                metric_value=0.333333,
                source_artifact_type="qa_eval_delta_report",
                supporting_metrics={"mean_evidence_node_recall_delta": 0.5},
            ),
        ],
    }
    assert validation["valid"] is True
    assert "Experiment Summary" in html
    assert "Failure Linkage" in html
    assert "Measurement Matrix" in html
    assert "Source Profiles" in html
    assert 'id="source-profile-filter"' in html
    assert '<option value="vlm:vlm_fixture">vlm:vlm_fixture</option>' in html
    assert 'data-source-profile-key="vlm:vlm_fixture"' in html
    assert 'data-source-profile-capabilities="graph_tool_query|spatial_qa"' in html
    assert "Readiness: ready" in html
    assert "<th>Verdict</th>" in html
    assert "<th>Baseline</th>" in html
    assert "object_recall_rate" in html
    assert "graph_construction" in html
    assert "spatial_qa" in html
    assert "exact_match_rate_delta" in html
    assert "baseline" in html
    assert "interactive_task" in html
    assert "task_success_rate_delta" in html


def test_export_dashboard_cli_writes_bundle_and_html(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_export_dashboard_script()
    main = cast(MainFn, getattr(module, "main"))
    graph = lab.load_scene_fixture("tabletop")
    case = _case_by_type("object_location")
    prediction = lab.QAPrediction(id=case.id, answer=case.answer, confidence=0.9)
    eval_report = lab.qa_eval_report((case,), (prediction,))
    attribution = lab.error_attribution_report(
        (case,),
        oracle_graph=graph,
        predicted_graph=graph,
        predictions=(prediction,),
    )
    qa_path = tmp_path / "qa.jsonl"
    pred_path = tmp_path / "predictions.jsonl"
    eval_path = tmp_path / "qa-eval-report.json"
    graph_path = tmp_path / "graph.json"
    attribution_path = tmp_path / "error-attribution.json"
    output_dir = tmp_path / "dashboard"
    lab.save_qa_dataset((case,), qa_path)
    lab.save_qa_predictions((prediction,), pred_path)
    lab.save_qa_eval_report(eval_report, eval_path)
    lab.save_graph_json(graph, graph_path)
    lab.save_error_attribution_report(attribution, attribution_path)

    assert main(
        [
            "--qa",
            str(qa_path),
            "--pred",
            str(pred_path),
            "--eval-report",
            str(eval_path),
            "--graph",
            str(graph_path),
            "--error-attribution",
            str(attribution_path),
            "--output",
            str(output_dir),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    bundle = lab.load_dashboard_bundle(output_dir / "dashboard.json")
    assert output == {
        "action": "export_dashboard",
        "path": str(output_dir),
        "valid": True,
        "digest": bundle["bundle_digest"],
        "summary": bundle["summary"],
        "bundle_path": str(output_dir / "dashboard.json"),
        "index_path": str(output_dir / "index.html"),
    }
    assert (output_dir / "dashboard.json").exists()
    assert (output_dir / "index.html").exists()


def test_export_dashboard_cli_validates_explicit_bundle(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_export_dashboard_script()
    main = cast(MainFn, getattr(module, "main"))
    graph = lab.load_scene_fixture("tabletop")
    case = _case_by_type("object_location")
    prediction = lab.QAPrediction(id=case.id, answer=case.answer, confidence=0.9)
    bundle = lab.dashboard_bundle(
        (case,),
        predictions=(prediction,),
        qa_eval_report=lab.qa_eval_report((case,), (prediction,)),
        graph=graph,
    )
    bundle_path = tmp_path / "dashboard.json"
    lab.save_dashboard_bundle(bundle, bundle_path)

    assert main(["--validate-bundle", str(bundle_path)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "action": "validate_dashboard_bundle",
        "path": str(bundle_path),
        "valid": True,
        "digest": bundle["bundle_digest"],
    }


def test_export_dashboard_cli_accepts_active_task_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_export_dashboard_script()
    main = cast(MainFn, getattr(module, "main"))
    graph = lab.load_scene_fixture("tabletop")
    qa_case = _case_by_id_suffix("object_location:plate_1")
    prediction = lab.QAPrediction(id=qa_case.id, answer=qa_case.answer, confidence=0.9)
    eval_report = lab.qa_eval_report((qa_case,), (prediction,))
    active_task = _task_for_case(qa_case, max_actions=1)
    active_result = lab.ActiveGraphAgent(policy="oracle_evidence").run(
        active_task,
        lab.MockActiveEnvironment(
            {
                active_task.initial_step: _graph_without_object("plate_1"),
                active_task.initial_step + 1: graph,
            }
        ),
    )
    active_report = lab.active_task_report((active_task,), (active_result,))
    qa_path = tmp_path / "qa.jsonl"
    pred_path = tmp_path / "predictions.jsonl"
    eval_path = tmp_path / "qa-eval-report.json"
    graph_path = tmp_path / "graph.json"
    active_report_path = tmp_path / "active-report.json"
    output_dir = tmp_path / "dashboard"
    lab.save_qa_dataset((qa_case,), qa_path)
    lab.save_qa_predictions((prediction,), pred_path)
    lab.save_qa_eval_report(eval_report, eval_path)
    lab.save_graph_json(graph, graph_path)
    lab.save_active_task_report(active_report, active_report_path)

    assert main(
        [
            "--qa",
            str(qa_path),
            "--pred",
            str(pred_path),
            "--eval-report",
            str(eval_path),
            "--graph",
            str(graph_path),
            "--active-task-report",
            str(active_report_path),
            "--output",
            str(output_dir),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    bundle = lab.load_dashboard_bundle(output_dir / "dashboard.json")
    assert output["valid"] is True
    assert output["digest"] == bundle["bundle_digest"]
    assert bundle["active_task_review"]["report_digest"] == active_report["report_digest"]
    assert bundle["active_task_review"]["panels"][0]["task_id"] == active_task.id


def test_export_dashboard_cli_accepts_active_task_delta_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_export_dashboard_script()
    main = cast(MainFn, getattr(module, "main"))
    graph = lab.load_scene_fixture("tabletop")
    qa_case = _case_by_id_suffix("object_location:plate_1")
    prediction = lab.QAPrediction(id=qa_case.id, answer=qa_case.answer, confidence=0.9)
    eval_report = lab.qa_eval_report((qa_case,), (prediction,))
    active_task = _task_for_case(qa_case, max_actions=1)
    baseline_result = lab.ActiveTaskResult(
        task_id=active_task.id,
        policy="direct_answer",
        answer={},
        success=False,
        action_count=0,
        final_step=active_task.initial_step,
        confidence=0.0,
        error="missing_required_evidence",
    )
    candidate_result = lab.ActiveTaskResult(
        task_id=active_task.id,
        policy="oracle_evidence",
        answer=qa_case.answer,
        success=True,
        action_count=1,
        evidence_nodes=qa_case.required_nodes,
        evidence_edges=qa_case.required_edges,
        final_step=active_task.initial_step + 1,
        confidence=1.0,
    )
    candidate_report = lab.active_task_report((active_task,), (candidate_result,))
    baseline_report = lab.active_task_report((active_task,), (baseline_result,))
    delta_report = lab.active_task_delta_report(
        candidate_report,
        baseline_report,
        candidate_name="oracle_evidence",
        baseline_name="direct_answer",
    )
    qa_path = tmp_path / "qa.jsonl"
    pred_path = tmp_path / "predictions.jsonl"
    eval_path = tmp_path / "qa-eval-report.json"
    graph_path = tmp_path / "graph.json"
    active_delta_path = tmp_path / "active-delta-report.json"
    output_dir = tmp_path / "dashboard"
    lab.save_qa_dataset((qa_case,), qa_path)
    lab.save_qa_predictions((prediction,), pred_path)
    lab.save_qa_eval_report(eval_report, eval_path)
    lab.save_graph_json(graph, graph_path)
    lab.save_active_task_delta_report(delta_report, active_delta_path)

    assert main(
        [
            "--qa",
            str(qa_path),
            "--pred",
            str(pred_path),
            "--eval-report",
            str(eval_path),
            "--graph",
            str(graph_path),
            "--active-task-delta-report",
            str(active_delta_path),
            "--output",
            str(output_dir),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    bundle = lab.load_dashboard_bundle(output_dir / "dashboard.json")
    assert output["valid"] is True
    assert output["digest"] == bundle["bundle_digest"]
    assert bundle["active_task_delta_review"]["report_digest"] == delta_report["report_digest"]
    assert bundle["active_task_delta_review"]["metrics_delta"] == delta_report["metrics_delta"]


def test_export_dashboard_cli_accepts_experiment_summary_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_export_dashboard_script()
    main = cast(MainFn, getattr(module, "main"))
    graph = lab.load_scene_fixture("tabletop")
    qa_case = _case_by_type("object_location")
    prediction = lab.QAPrediction(id=qa_case.id, answer=qa_case.answer, confidence=0.9)
    eval_report = lab.qa_eval_report((qa_case,), (prediction,))
    summary_report = _experiment_summary_report_for_dashboard()
    qa_path = tmp_path / "qa.jsonl"
    pred_path = tmp_path / "predictions.jsonl"
    eval_path = tmp_path / "qa-eval-report.json"
    graph_path = tmp_path / "graph.json"
    summary_path = tmp_path / "experiment-summary.json"
    output_dir = tmp_path / "dashboard"
    lab.save_qa_dataset((qa_case,), qa_path)
    lab.save_qa_predictions((prediction,), pred_path)
    lab.save_qa_eval_report(eval_report, eval_path)
    lab.save_graph_json(graph, graph_path)
    lab.save_experiment_summary_report(summary_report, summary_path)

    assert main(
        [
            "--qa",
            str(qa_path),
            "--pred",
            str(pred_path),
            "--eval-report",
            str(eval_path),
            "--graph",
            str(graph_path),
            "--experiment-summary-report",
            str(summary_path),
            "--output",
            str(output_dir),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    bundle = lab.load_dashboard_bundle(output_dir / "dashboard.json")
    assert output["valid"] is True
    assert output["digest"] == bundle["bundle_digest"]
    assert bundle["experiment_summary_review"]["report_digest"] == summary_report[
        "report_digest"
    ]
    assert "Experiment Summary" in (output_dir / "index.html").read_text(
        encoding="utf-8"
    )


def test_export_dashboard_cli_returns_structured_json_for_invalid_eval_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_export_dashboard_script()
    main = cast(MainFn, getattr(module, "main"))
    graph = lab.load_scene_fixture("tabletop")
    case = _case_by_type("object_location")
    qa_path = tmp_path / "qa.jsonl"
    pred_path = tmp_path / "predictions.jsonl"
    eval_path = tmp_path / "qa-eval-report.json"
    graph_path = tmp_path / "graph.json"
    output_dir = tmp_path / "dashboard"
    lab.save_qa_dataset((case,), qa_path)
    lab.save_qa_predictions((lab.QAPrediction(id=case.id, answer=case.answer),), pred_path)
    lab.save_graph_json(graph, graph_path)
    eval_path.write_text("[]\n", encoding="utf-8")

    assert main(
        [
            "--qa",
            str(qa_path),
            "--pred",
            str(pred_path),
            "--eval-report",
            str(eval_path),
            "--graph",
            str(graph_path),
            "--output",
            str(output_dir),
        ]
    ) == 1

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "action": "export_dashboard",
        "path": str(output_dir),
        "valid": False,
        "error": "QA eval report JSON must be an object",
    }


def _tabletop_cases() -> list[lab.QACase]:
    return lab.generate_qa_cases(
        lab.load_scene_fixture("tabletop"),
        scene_id="tabletop_scene",
        episode_id="episode_001",
    )


def _case_by_type(question_type: str) -> lab.QACase:
    return next(case for case in _tabletop_cases() if case.question_type == question_type)


def _case_by_id_suffix(suffix: str) -> lab.QACase:
    return next(case for case in _tabletop_cases() if case.id.endswith(suffix))


def _case_with_tags(case: lab.QACase, tags: tuple[str, ...]) -> lab.QACase:
    payload = lab.qa_case_to_dict(case)
    payload["tags"] = [*payload["tags"], *tags]
    return lab.qa_case_from_dict(payload)


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


def _clone_graph(graph: lab.DynamicSceneGraph) -> lab.DynamicSceneGraph:
    return lab.graph_from_json(lab.graph_to_json(graph))


def _graph_without_edge(edge_id: str) -> lab.DynamicSceneGraph:
    graph = _clone_graph(lab.load_scene_fixture("tabletop"))
    graph.edges = [edge for edge in graph.edges if edge.id != edge_id]
    return graph


def _graph_without_object(object_id: str) -> lab.DynamicSceneGraph:
    graph = _clone_graph(lab.load_scene_fixture("tabletop"))
    graph.object_states.pop(object_id)
    graph.object_state_history.pop(object_id)
    for node_id in list(graph.nodes):
        if node_id == object_id or node_id.startswith(f"state:{object_id}:"):
            graph.nodes.pop(node_id)
    graph.edges = [
        edge
        for edge in graph.edges
        if edge.src != object_id
        and edge.dst != object_id
        and not edge.dst.startswith(f"state:{object_id}:")
    ]
    return graph


def _graph_with_moved_mug_source(source: str) -> lab.DynamicSceneGraph:
    graph = _clone_graph(lab.load_scene_fixture("tabletop"))
    mug = graph.get_object_state("mug_1")
    graph.upsert_object(
        "mug_1",
        mug.label,
        lab.Pose3D(9.0, 9.0, 9.0),
        mug.bbox,
        confidence=mug.confidence,
        visible=mug.visible,
        step=mug.step,
    )
    graph.nodes["mug_1"].attributes["source"] = source
    graph.nodes["state:mug_1:1"].attributes["source"] = source
    for edge in graph.edges:
        if edge.src == "mug_1" or edge.dst == "mug_1":
            edge.attributes["source"] = source
    return graph


def _experiment_summary_report_for_dashboard() -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "dsg-spatialqa-lab.experiment-summary-report.v1",
        "manifest_path": "benchmark-manifest.json",
        "manifest_digest": "manifest-digest",
        "source_artifact_digests": {
            "active_task_delta_report:active-delta-report.json": "active-digest",
            "error_attribution_report:error-attribution-report.json": (
                "attribution-digest"
            ),
            "graph_eval_report:graph-eval-report.json": "graph-digest",
            "qa_eval_delta_report:qa-delta-report.json": "qa-digest",
            "offline_prediction_import_report:offline-import-report.json": (
                "offline-import-digest"
            ),
        },
        "source_artifacts": [],
        "qa_delta_comparisons": [],
        "active_task_delta_comparisons": [],
        "offline_prediction_import_summaries": [],
        "source_profile_matrix": [
            {
                "adapter": "vlm",
                "artifact_key": (
                    "offline_prediction_import_report:offline-import-report.json"
                ),
                "capability_axes": ["graph_tool_query", "spatial_qa"],
                "dataset_id": "mock_eval",
                "digest": "offline-import-digest",
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
                "path": "vlm-import-report.json",
                "prediction_digest": "prediction-digest",
                "prompt_id": "spatial-qa-v1",
                "qa_digest": "qa-digest",
                "source_key": "vlm:vlm_fixture",
                "source_kind": "vlm",
                "source_name": "vlm_fixture",
                "unknown_case_count": 0,
            },
        ],
        "research_questions": {
            "dynamic_memory": _research_question(
                primary_name="exact_match_rate_delta",
                primary_value=0.0,
                source_artifact_type="qa_eval_delta_report",
                supporting_metrics={"mean_evidence_node_recall_delta": 0.5},
            ),
            "graph_tool_query": _research_question(
                primary_name="exact_match_rate_delta",
                primary_value=0.333333,
                source_artifact_type="qa_eval_delta_report",
                supporting_metrics={"mean_evidence_node_recall_delta": 0.5},
            ),
            "interactive_task": _research_question(
                primary_name="task_success_rate_delta",
                primary_value=1.0,
                source_artifact_type="active_task_delta_report",
                supporting_metrics={"success_count_delta": 1},
            ),
            "spatial_qa": _research_question(
                primary_name="exact_match_rate_delta",
                primary_value=0.333333,
                source_artifact_type="qa_eval_delta_report",
                supporting_metrics={"mean_evidence_node_recall_delta": 0.5},
            ),
        },
        "summary": {
            "active_task_delta_report_count": 1,
            "available_research_question_count": 4,
            "error_attribution_diagnostic_count": 1,
            "failure_linkage_diagnostic_count": 1,
            "graph_construction_diagnostic_count": 1,
            "qa_eval_delta_report_count": 1,
            "source_profile_count": 1,
            "readiness_status": "ready",
            "research_question_count": 4,
            "source_artifact_count": 5,
            "verdict_counts": {
                "improved": 3,
                "inconclusive": 0,
                "regressed": 0,
                "unchanged": 1,
            },
        },
        "readiness": {
            "status": "ready",
            "required_research_questions": [
                "dynamic_memory",
                "graph_tool_query",
                "interactive_task",
                "spatial_qa",
            ],
            "available_research_questions": [
                "dynamic_memory",
                "graph_tool_query",
                "interactive_task",
                "spatial_qa",
            ],
            "missing_research_questions": [],
            "required_source_artifact_types": [
                "active_task_delta_report",
                "qa_eval_delta_report",
            ],
            "available_source_artifact_types": [
                "active_task_delta_report",
                "qa_eval_delta_report",
            ],
            "missing_source_artifact_types": [],
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
                    "passed": True,
                    "source_artifact_type": "active_task_delta_report",
                    "measurement_count": 1,
                    "missing_reason": None,
                },
                {
                    "name": "spatial_qa",
                    "passed": True,
                    "source_artifact_type": "qa_eval_delta_report",
                    "measurement_count": 1,
                    "missing_reason": None,
                },
            ],
        },
        "graph_construction_diagnostics": {
            "graph_eval_report:graph-eval-report.json": {
                "diagnostics": {
                    "duplicate_track_count": 0,
                    "id_fragmentation_count": 1,
                },
                "graph_summary": {
                    "edge_count": 12,
                    "node_count": 9,
                    "object_count": 4,
                },
                "oracle_graph_digest": "oracle-graph-digest",
                "predicted_graph_digest": "predicted-graph-digest",
                "primary_metrics": {
                    "object_recall_rate": 0.75,
                    "relation_f1_rate": 0.5,
                    "state_accuracy_rate": 1.0,
                },
                "source_breakdown": {},
            },
        },
        "error_attribution_diagnostics": {
            "error_attribution_report:error-attribution-report.json": {
                "gold_digest": "qa-digest",
                "oracle_graph_digest": "oracle-graph-digest",
                "predicted_graph_digest": "predicted-graph-digest",
                "prediction_digest": "prediction-digest",
                "summary": {
                    "answer_correct_count": 0,
                    "by_error_category": {
                        "graph_construction": 2,
                        "reasoning_or_tool_use": 1,
                    },
                    "by_evidence_error_category": {"none": 3},
                    "by_predicted_evidence_source": {},
                    "by_research_axis": {
                        "dynamic_memory": {
                            "case_count": 1,
                            "error_count": 1,
                            "by_error_category": {"reasoning_or_tool_use": 1},
                        },
                        "graph_tool_query": {
                            "case_count": 3,
                            "error_count": 3,
                            "by_error_category": {
                                "graph_construction": 2,
                                "reasoning_or_tool_use": 1,
                            },
                        },
                        "spatial_qa": {
                            "case_count": 3,
                            "error_count": 3,
                            "by_error_category": {
                                "graph_construction": 2,
                                "reasoning_or_tool_use": 1,
                            },
                        },
                    },
                    "case_count": 3,
                    "error_count": 3,
                    "oracle_graph_tool_correct_count": 3,
                    "predicted_graph_tool_correct_count": 1,
                },
            },
        },
        "failure_linkage_diagnostics": {
            "error_attribution_report:error-attribution-report.json": {
                "attribution_summary": {
                    "answer_correct_count": 0,
                    "by_error_category": {
                        "graph_construction": 2,
                        "reasoning_or_tool_use": 1,
                    },
                    "case_count": 3,
                    "error_count": 3,
                },
                "error_attribution_artifact_key": (
                    "error_attribution_report:error-attribution-report.json"
                ),
                "graph_diagnostics": {
                    "duplicate_track_count": 0,
                    "id_fragmentation_count": 1,
                },
                "graph_eval_artifact_key": "graph_eval_report:graph-eval-report.json",
                "graph_primary_metrics": {
                    "object_recall_rate": 0.75,
                    "relation_f1_rate": 0.5,
                    "state_accuracy_rate": 1.0,
                },
                "linked_by": "oracle_and_predicted_graph_digest",
                "oracle_graph_digest": "oracle-graph-digest",
                "predicted_graph_digest": "predicted-graph-digest",
            },
        },
    }
    report["report_digest"] = lab.experiment_summary_report_digest(report)
    return report


def _research_question(
    *,
    primary_name: str,
    primary_value: float,
    source_artifact_type: str,
    supporting_metrics: dict[str, Any],
) -> dict[str, Any]:
    primary_metric = {"name": primary_name, "value": primary_value}
    return {
        "label": primary_name,
        "measurements": [
            {
                "artifact_key": f"{source_artifact_type}:artifact.json",
                "baseline_name": "baseline",
                "candidate_name": "candidate",
                "case_count_match": True,
                "primary_metric": primary_metric,
                "supporting_metrics": supporting_metrics,
            }
        ],
        "primary_metric": primary_metric,
        "source_artifact_type": source_artifact_type,
        "status": "available",
        "supporting_metrics": supporting_metrics,
        "verdict": _verdict(primary_value),
    }


def _matrix_row(
    *,
    research_question: str,
    metric_value: float,
    source_artifact_type: str,
    supporting_metrics: dict[str, Any],
    metric_name: str = "exact_match_rate_delta",
) -> dict[str, Any]:
    primary_metric = {"name": metric_name, "value": metric_value}
    return {
        "artifact_key": f"{source_artifact_type}:artifact.json",
        "baseline_name": "baseline",
        "candidate_name": "candidate",
        "case_count_match": True,
        "label": metric_name,
        "measurement_verdict": _verdict(metric_value),
        "primary_metric": primary_metric,
        "question_verdict": _verdict(metric_value),
        "research_question": research_question,
        "source_artifact_type": source_artifact_type,
        "status": "available",
        "supporting_metrics": supporting_metrics,
    }


def _verdict(value: float) -> str:
    if value > 0:
        return "improved"
    if value < 0:
        return "regressed"
    return "unchanged"
