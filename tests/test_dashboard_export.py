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
