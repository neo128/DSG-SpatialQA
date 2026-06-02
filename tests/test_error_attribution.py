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
ATTRIBUTE_ERRORS_SCRIPT = ROOT / "scripts" / "attribute_errors.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_attribute_errors_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "attribute_errors_script",
        ATTRIBUTE_ERRORS_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_error_attribution_distinguishes_graph_construction_and_reasoning() -> None:
    assert hasattr(lab, "attribute_qa_errors")
    oracle_graph = lab.load_scene_fixture("tabletop")
    predicted_graph = _graph_with_moved_mug()
    mug_case = _case_by_type("object_location")
    plate_case = _case_by_id_suffix("object_location:plate_1")
    predictions = (
        lab.QAPrediction(
            id=mug_case.id,
            answer={"object_id": "mug_1", "label": "mug", "pose": {"x": 9.0}},
            confidence=0.3,
        ),
        lab.QAPrediction(
            id=plate_case.id,
            answer={"wrong": True},
            evidence_nodes=plate_case.required_nodes,
            evidence_edges=plate_case.required_edges,
            confidence=0.4,
        ),
    )

    rows = lab.attribute_qa_errors(
        (mug_case, plate_case),
        oracle_graph=oracle_graph,
        predicted_graph=predicted_graph,
        predictions=predictions,
    )

    assert [row["case_id"] for row in rows] == [mug_case.id, plate_case.id]
    assert rows[0]["answer_correct"] is False
    assert rows[0]["oracle_graph_tool_correct"] is True
    assert rows[0]["predicted_graph_tool_correct"] is False
    assert rows[0]["required_nodes_present"] is True
    assert rows[0]["required_edges_present"] is True
    assert rows[0]["error_category"] == "graph_construction"
    assert rows[0]["evidence_error_category"] == "none"
    assert rows[1]["answer_correct"] is False
    assert rows[1]["predicted_graph_tool_correct"] is True
    assert rows[1]["error_category"] == "reasoning_or_tool_use"


def test_error_attribution_reports_missing_object_and_missing_relation_evidence() -> None:
    oracle_graph = lab.load_scene_fixture("tabletop")
    plate_case = _case_by_id_suffix("object_location:plate_1")
    relation_case = _case_by_type("relative_relation")

    missing_object_rows = lab.attribute_qa_errors(
        (plate_case,),
        oracle_graph=oracle_graph,
        predicted_graph=_graph_without_object("plate_1"),
        predictions=(lab.QAPrediction(id=plate_case.id, answer={"wrong": True}),),
    )
    missing_relation_rows = lab.attribute_qa_errors(
        (relation_case,),
        oracle_graph=oracle_graph,
        predicted_graph=_graph_without_edge(relation_case.required_edges[0]),
        predictions=(lab.QAPrediction(id=relation_case.id, answer={"wrong": True}),),
    )

    assert missing_object_rows[0]["error_category"] == "evidence_missing"
    assert missing_object_rows[0]["evidence_error_category"] == "missing_object"
    assert missing_object_rows[0]["missing_required_nodes"] == [
        "plate_1",
        "state:plate_1:1",
    ]
    assert missing_relation_rows[0]["error_category"] == "evidence_missing"
    assert missing_relation_rows[0]["evidence_error_category"] == "missing_relation"
    assert missing_relation_rows[0]["missing_required_edges"] == [
        relation_case.required_edges[0],
    ]


def test_error_attribution_summarizes_predicted_evidence_sources() -> None:
    oracle_graph = lab.load_scene_fixture("tabletop")
    predicted_graph = _graph_with_moved_mug_source("vlm_detector")
    case = _case_by_type("object_location")
    predictions = (lab.QAPrediction(id=case.id, answer={"wrong": True}),)

    report = lab.error_attribution_report(
        (case,),
        oracle_graph=oracle_graph,
        predicted_graph=predicted_graph,
        predictions=predictions,
    )

    assert report["cases"][0]["error_category"] == "graph_construction"
    assert report["cases"][0]["predicted_evidence_sources"] == ["vlm_detector"]
    assert report["summary"]["by_predicted_evidence_source"] == {
        "vlm_detector": {
            "by_error_category": {"graph_construction": 1},
            "by_evidence_error_category": {"none": 1},
            "case_count": 1,
            "error_count": 1,
        },
    }
    assert lab.validate_error_attribution_report(report)["valid"] is True


def test_error_attribution_reports_benchmark_or_engine_error() -> None:
    oracle_graph = lab.load_scene_fixture("tabletop")
    case = lab.qa_case_from_dict(lab.qa_case_to_dict(_case_by_type("object_location")))
    case.answer["object_id"] = "wrong_gold"

    rows = lab.attribute_qa_errors(
        (case,),
        oracle_graph=oracle_graph,
        predicted_graph=oracle_graph,
        predictions=(lab.QAPrediction(id=case.id, answer=case.answer),),
    )

    assert rows[0]["answer_correct"] is True
    assert rows[0]["oracle_graph_tool_correct"] is False
    assert rows[0]["error_category"] == "benchmark_or_engine_error"


def test_error_attribution_report_digest_validation_and_comparison(tmp_path: Path) -> None:
    assert hasattr(lab, "error_attribution_report")
    assert hasattr(lab, "error_attribution_report_digest")
    assert hasattr(lab, "error_attribution_report_json")
    assert hasattr(lab, "save_error_attribution_report")
    assert hasattr(lab, "load_error_attribution_report")
    assert hasattr(lab, "validate_error_attribution_report")
    assert hasattr(lab, "compare_error_attribution_report")
    oracle_graph = lab.load_scene_fixture("tabletop")
    predicted_graph = _graph_with_moved_mug()
    case = _case_by_type("object_location")
    predictions = (lab.QAPrediction(id=case.id, answer={"wrong": True}),)
    gold_path = tmp_path / "qa.jsonl"
    oracle_graph_path = tmp_path / "oracle.json"
    predicted_graph_path = tmp_path / "predicted.json"
    prediction_path = tmp_path / "predictions.jsonl"
    report_path = tmp_path / "error-attribution.json"
    lab.save_qa_dataset((case,), gold_path)
    lab.save_graph_json(oracle_graph, oracle_graph_path)
    lab.save_graph_json(predicted_graph, predicted_graph_path)
    lab.save_qa_predictions(predictions, prediction_path)

    report = lab.error_attribution_report(
        (case,),
        oracle_graph=oracle_graph,
        predicted_graph=predicted_graph,
        predictions=predictions,
        gold_path=gold_path,
        oracle_graph_path=oracle_graph_path,
        predicted_graph_path=predicted_graph_path,
        prediction_path=prediction_path,
    )
    saved_path = lab.save_error_attribution_report(report, report_path)
    loaded_report = lab.load_error_attribution_report(report_path)
    validation = lab.validate_error_attribution_report(loaded_report)
    comparison = lab.compare_error_attribution_report(loaded_report)

    assert saved_path == report_path
    assert json.loads(lab.error_attribution_report_json(report)) == report
    assert report["report_digest"] == lab.error_attribution_report_digest(report)
    assert report["summary"] == {
        "answer_correct_count": 0,
        "by_error_category": {"graph_construction": 1},
        "by_evidence_error_category": {"none": 1},
        "by_predicted_evidence_source": {
            "unknown": {
                "by_error_category": {"graph_construction": 1},
                "by_evidence_error_category": {"none": 1},
                "case_count": 1,
                "error_count": 1,
            },
        },
        "case_count": 1,
        "error_count": 1,
        "oracle_graph_tool_correct_count": 1,
        "predicted_graph_tool_correct_count": 0,
    }
    assert validation["valid"] is True
    assert comparison["matches"] is True

    tampered_report = dict(loaded_report)
    tampered_report["report_digest"] = "0" * 64
    tampered_validation = lab.validate_error_attribution_report(tampered_report)
    checks = {check["name"]: check for check in tampered_validation["checks"]}
    assert tampered_validation["valid"] is False
    assert checks["report_digest"]["passed"] is False


def test_attribute_errors_cli_writes_validates_compares_and_handles_invalid_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_attribute_errors_script()
    main = cast(MainFn, getattr(module, "main"))
    oracle_graph = lab.load_scene_fixture("tabletop")
    predicted_graph = _graph_with_moved_mug()
    case = _case_by_type("object_location")
    predictions = (lab.QAPrediction(id=case.id, answer={"wrong": True}),)
    gold_path = tmp_path / "qa.jsonl"
    oracle_graph_path = tmp_path / "oracle.json"
    predicted_graph_path = tmp_path / "predicted.json"
    prediction_path = tmp_path / "predictions.jsonl"
    report_path = tmp_path / "error-attribution.json"
    invalid_path = tmp_path / "invalid-report.json"
    lab.save_qa_dataset((case,), gold_path)
    lab.save_graph_json(oracle_graph, oracle_graph_path)
    lab.save_graph_json(predicted_graph, predicted_graph_path)
    lab.save_qa_predictions(predictions, prediction_path)

    assert main(
        [
            "--gold",
            str(gold_path),
            "--oracle-graph",
            str(oracle_graph_path),
            "--predicted-graph",
            str(predicted_graph_path),
            "--predictions",
            str(prediction_path),
            "--report",
            str(report_path),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    report = lab.load_error_attribution_report(report_path)
    assert output == {
        "action": "error_attribution_report",
        "path": str(report_path),
        "valid": True,
        "digest": report["report_digest"],
        "summary": report["summary"],
    }

    assert main(["--validate-report", str(report_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_error_attribution_report"
    assert validation["path"] == str(report_path)
    assert validation["valid"] is True

    assert main(["--compare-report", str(report_path)]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["action"] == "compare_error_attribution_report"
    assert comparison["path"] == str(report_path)
    assert comparison["matches"] is True

    invalid_path.write_text("[]\n", encoding="utf-8")
    assert main(["--validate-report", str(invalid_path)]) == 1
    invalid = json.loads(capsys.readouterr().out)
    assert invalid == {
        "action": "validate_error_attribution_report",
        "path": str(invalid_path),
        "valid": False,
        "error": "Error attribution report JSON must be an object",
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


def _clone_graph(graph: lab.DynamicSceneGraph) -> lab.DynamicSceneGraph:
    return lab.graph_from_json(lab.graph_to_json(graph))


def _graph_with_moved_mug() -> lab.DynamicSceneGraph:
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
    return graph


def _graph_with_moved_mug_source(source: str) -> lab.DynamicSceneGraph:
    graph = _graph_with_moved_mug()
    graph.nodes["mug_1"].attributes["source"] = source
    graph.nodes["state:mug_1:1"].attributes["source"] = source
    for edge in graph.edges:
        if edge.src == "mug_1" or edge.dst == "mug_1":
            edge.attributes["source"] = source
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


def _graph_without_edge(edge_id: str) -> lab.DynamicSceneGraph:
    graph = _clone_graph(lab.load_scene_fixture("tabletop"))
    graph.edges = [edge for edge in graph.edges if edge.id != edge_id]
    return graph
