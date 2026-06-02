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
RUN_QA_EVAL_SCRIPT = ROOT / "scripts" / "run_qa_eval.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_run_qa_eval_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_qa_eval_script", RUN_QA_EVAL_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def qa_metric_cases() -> tuple[lab.QACase, ...]:
    graph = lab.load_scene_fixture("tabletop")
    generated = lab.generate_qa_cases(
        graph,
        scene_id="tabletop_scene",
        episode_id="episode_001",
    )
    relation = next(case for case in generated if case.question_type == "relative_relation")
    nearest = next(case for case in generated if case.question_type == "nearest_object")
    return (generated[0], relation, nearest)


def qa_metric_predictions(cases: tuple[lab.QACase, ...]) -> tuple[lab.QAPrediction, ...]:
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


def test_evaluate_qa_predictions_reports_accuracy_evidence_and_breakdowns() -> None:
    assert hasattr(lab, "QAPrediction")
    assert hasattr(lab, "evaluate_qa_predictions")
    cases = qa_metric_cases()
    predictions = qa_metric_predictions(cases)

    report = lab.evaluate_qa_predictions(cases, predictions)

    assert report["summary"] == {
        "case_count": 3,
        "prediction_count": 2,
        "matched_prediction_count": 2,
        "missing_prediction_count": 1,
        "exact_match_count": 1,
        "exact_match_rate": 0.333333,
    }
    assert report["metrics"] == {
        "answer_graph_consistency": {"count": 1, "rate": 0.333333, "total": 3},
        "evidence_edge_recall": {"average": 0.333333, "total": 3},
        "evidence_node_recall": {"average": 0.5, "total": 3},
        "exact_match": {"count": 1, "rate": 0.333333, "total": 3},
        "multiple_choice_accuracy": {"count": 0, "rate": 0.0, "total": 1},
        "numeric_mae": {
            "count": 0,
            "mean_absolute_error": None,
            "total_absolute_error": 0.0,
        },
    }
    assert report["breakdown"]["by_question_type"] == {
        "nearest_object": {
            "case_count": 1,
            "exact_match_count": 0,
            "exact_match_rate": 0.0,
            "mean_evidence_edge_recall": 0.0,
            "mean_evidence_node_recall": 0.0,
        },
        "object_location": {
            "case_count": 1,
            "exact_match_count": 1,
            "exact_match_rate": 1.0,
            "mean_evidence_edge_recall": 1.0,
            "mean_evidence_node_recall": 1.0,
        },
        "relative_relation": {
            "case_count": 1,
            "exact_match_count": 0,
            "exact_match_rate": 0.0,
            "mean_evidence_edge_recall": 0.0,
            "mean_evidence_node_recall": 0.5,
        },
    }
    assert report["breakdown"]["by_reference_frame"] == {
        "agent": {
            "case_count": 1,
            "exact_match_count": 0,
            "exact_match_rate": 0.0,
            "mean_evidence_edge_recall": 0.0,
            "mean_evidence_node_recall": 0.5,
        },
        "none": {
            "case_count": 2,
            "exact_match_count": 1,
            "exact_match_rate": 0.5,
            "mean_evidence_edge_recall": 0.5,
            "mean_evidence_node_recall": 0.5,
        },
    }
    assert report["cases"][1]["error"] is None
    assert report["cases"][1]["exact_match"] is False
    assert report["cases"][1]["evidence_node_recall"] == 0.5
    assert report["cases"][2]["error"] == "missing_prediction"


def test_qa_eval_report_includes_research_axis_breakdowns() -> None:
    cases = list(qa_metric_cases())
    relation_payload = lab.qa_case_to_dict(cases[1])
    relation_payload["tags"] = [*relation_payload["tags"], "dynamic", "memory"]
    cases[1] = lab.qa_case_from_dict(relation_payload)
    predictions = qa_metric_predictions(tuple(cases))

    report = lab.qa_eval_report(cases, predictions)

    assert report["breakdown"]["by_research_axis"] == {
        "dynamic_memory": {
            "case_count": 1,
            "exact_match_count": 0,
            "exact_match_rate": 0.0,
            "mean_evidence_edge_recall": 0.0,
            "mean_evidence_node_recall": 0.5,
        },
        "graph_tool_query": {
            "case_count": 3,
            "exact_match_count": 1,
            "exact_match_rate": 0.333333,
            "mean_evidence_edge_recall": 0.333333,
            "mean_evidence_node_recall": 0.5,
        },
        "spatial_qa": {
            "case_count": 3,
            "exact_match_count": 1,
            "exact_match_rate": 0.333333,
            "mean_evidence_edge_recall": 0.333333,
            "mean_evidence_node_recall": 0.5,
        },
    }


def test_qa_eval_report_and_delta_include_scene_and_episode_breakdowns() -> None:
    cases = list(qa_metric_cases())
    moved_payload = lab.qa_case_to_dict(cases[2])
    moved_payload["id"] = "episode_002:kitchen_scene:0008:nearest_object:mug_1"
    moved_payload["scene_id"] = "kitchen_scene"
    moved_payload["episode_id"] = "episode_002"
    cases[2] = lab.qa_case_from_dict(moved_payload)
    candidate_report = lab.qa_eval_report(cases, qa_metric_predictions(tuple(cases)))
    baseline_report = lab.qa_eval_report(cases, ())

    assert candidate_report["cases"][0]["scene_id"] == "tabletop_scene"
    assert candidate_report["cases"][0]["episode_id"] == "episode_001"
    assert candidate_report["cases"][2]["scene_id"] == "kitchen_scene"
    assert candidate_report["cases"][2]["episode_id"] == "episode_002"
    assert candidate_report["breakdown"]["by_scene_id"] == {
        "kitchen_scene": {
            "case_count": 1,
            "exact_match_count": 0,
            "exact_match_rate": 0.0,
            "mean_evidence_edge_recall": 0.0,
            "mean_evidence_node_recall": 0.0,
        },
        "tabletop_scene": {
            "case_count": 2,
            "exact_match_count": 1,
            "exact_match_rate": 0.5,
            "mean_evidence_edge_recall": 0.5,
            "mean_evidence_node_recall": 0.75,
        },
    }
    assert candidate_report["breakdown"]["by_episode_id"] == {
        "episode_001": {
            "case_count": 2,
            "exact_match_count": 1,
            "exact_match_rate": 0.5,
            "mean_evidence_edge_recall": 0.5,
            "mean_evidence_node_recall": 0.75,
        },
        "episode_002": {
            "case_count": 1,
            "exact_match_count": 0,
            "exact_match_rate": 0.0,
            "mean_evidence_edge_recall": 0.0,
            "mean_evidence_node_recall": 0.0,
        },
    }

    delta = lab.qa_eval_delta_report(candidate_report, baseline_report)

    assert delta["breakdown_delta"]["by_scene_id"]["tabletop_scene"] == {
        "baseline_case_count": 2,
        "baseline_exact_match_count": 0,
        "baseline_exact_match_rate": 0.0,
        "baseline_mean_evidence_edge_recall": 0.0,
        "baseline_mean_evidence_node_recall": 0.0,
        "candidate_case_count": 2,
        "candidate_exact_match_count": 1,
        "candidate_exact_match_rate": 0.5,
        "candidate_mean_evidence_edge_recall": 0.5,
        "candidate_mean_evidence_node_recall": 0.75,
        "case_count_delta": 0,
        "case_count_match": True,
        "exact_match_count_delta": 1,
        "exact_match_rate_delta": 0.5,
        "mean_evidence_edge_recall_delta": 0.5,
        "mean_evidence_node_recall_delta": 0.75,
    }
    assert delta["breakdown_delta"]["by_episode_id"]["episode_002"][
        "case_count_match"
    ] is True
    assert delta["breakdown_delta"]["by_episode_id"]["episode_002"][
        "exact_match_rate_delta"
    ] == 0.0


def test_qa_eval_validation_detects_research_axis_breakdown_drift() -> None:
    cases = qa_metric_cases()
    predictions = qa_metric_predictions(cases)
    report = lab.qa_eval_report(cases, predictions)
    report["breakdown"]["by_research_axis"] = {}
    report["report_digest"] = lab.qa_eval_report_digest(report)

    validation = lab.validate_qa_eval_report(report)

    checks = {check["name"]: check for check in validation["checks"]}
    assert validation["valid"] is False
    assert checks["research_axis_breakdown"]["passed"] is False
    assert checks["research_axis_breakdown"]["expected"] == {
        "dynamic_memory": {
            "case_count": 0,
            "exact_match_count": 0,
            "exact_match_rate": 0.0,
            "mean_evidence_edge_recall": 0.0,
            "mean_evidence_node_recall": 0.0,
        },
        "graph_tool_query": {
            "case_count": 3,
            "exact_match_count": 1,
            "exact_match_rate": 0.333333,
            "mean_evidence_edge_recall": 0.333333,
            "mean_evidence_node_recall": 0.5,
        },
        "spatial_qa": {
            "case_count": 3,
            "exact_match_count": 1,
            "exact_match_rate": 0.333333,
            "mean_evidence_edge_recall": 0.333333,
            "mean_evidence_node_recall": 0.5,
        },
    }
    assert checks["research_axis_breakdown"]["actual"] == {}


def test_qa_eval_delta_report_compares_candidate_against_baseline() -> None:
    assert hasattr(lab, "qa_eval_delta_report")
    assert hasattr(lab, "validate_qa_eval_delta_report")
    cases = list(qa_metric_cases())
    relation_payload = lab.qa_case_to_dict(cases[1])
    relation_payload["tags"] = [*relation_payload["tags"], "dynamic", "memory"]
    cases[1] = lab.qa_case_from_dict(relation_payload)
    candidate_report = lab.qa_eval_report(cases, qa_metric_predictions(tuple(cases)))
    baseline_report = lab.qa_eval_report(cases, ())

    delta = lab.qa_eval_delta_report(
        candidate_report,
        baseline_report,
        candidate_name="graph_tool",
        baseline_name="majority",
    )
    validation = lab.validate_qa_eval_delta_report(delta)

    assert delta["schema_version"] == "dsg-spatialqa-lab.qa-eval-delta-report.v1"
    assert delta["candidate_name"] == "graph_tool"
    assert delta["baseline_name"] == "majority"
    assert delta["summary_delta"] == {
        "baseline_case_count": 3,
        "baseline_exact_match_count": 0,
        "baseline_exact_match_rate": 0.0,
        "candidate_case_count": 3,
        "candidate_exact_match_count": 1,
        "candidate_exact_match_rate": 0.333333,
        "case_count_delta": 0,
        "case_count_match": True,
        "exact_match_count_delta": 1,
        "exact_match_rate_delta": 0.333333,
    }
    assert delta["metrics_delta"]["exact_match"] == {
        "baseline_count": 0,
        "baseline_rate": 0.0,
        "candidate_count": 1,
        "candidate_rate": 0.333333,
        "count_delta": 1,
        "rate_delta": 0.333333,
    }
    assert delta["metrics_delta"]["evidence_node_recall"] == {
        "average_delta": 0.5,
        "baseline_average": 0.0,
        "candidate_average": 0.5,
    }
    assert delta["breakdown_delta"]["by_research_axis"]["dynamic_memory"] == {
        "baseline_case_count": 1,
        "baseline_exact_match_count": 0,
        "baseline_exact_match_rate": 0.0,
        "baseline_mean_evidence_edge_recall": 0.0,
        "baseline_mean_evidence_node_recall": 0.0,
        "candidate_case_count": 1,
        "candidate_exact_match_count": 0,
        "candidate_exact_match_rate": 0.0,
        "candidate_mean_evidence_edge_recall": 0.0,
        "candidate_mean_evidence_node_recall": 0.5,
        "case_count_delta": 0,
        "case_count_match": True,
        "exact_match_count_delta": 0,
        "exact_match_rate_delta": 0.0,
        "mean_evidence_edge_recall_delta": 0.0,
        "mean_evidence_node_recall_delta": 0.5,
    }
    assert delta["breakdown_delta"]["by_question_type"]["object_location"] == {
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
    assert delta["breakdown_delta"]["by_tag"]["dynamic"] == {
        "baseline_case_count": 1,
        "baseline_exact_match_count": 0,
        "baseline_exact_match_rate": 0.0,
        "baseline_mean_evidence_edge_recall": 0.0,
        "baseline_mean_evidence_node_recall": 0.0,
        "candidate_case_count": 1,
        "candidate_exact_match_count": 0,
        "candidate_exact_match_rate": 0.0,
        "candidate_mean_evidence_edge_recall": 0.0,
        "candidate_mean_evidence_node_recall": 0.5,
        "case_count_delta": 0,
        "case_count_match": True,
        "exact_match_count_delta": 0,
        "exact_match_rate_delta": 0.0,
        "mean_evidence_edge_recall_delta": 0.0,
        "mean_evidence_node_recall_delta": 0.5,
    }
    assert delta["breakdown_delta"]["by_reference_frame"]["none"] == {
        "baseline_case_count": 2,
        "baseline_exact_match_count": 0,
        "baseline_exact_match_rate": 0.0,
        "baseline_mean_evidence_edge_recall": 0.0,
        "baseline_mean_evidence_node_recall": 0.0,
        "candidate_case_count": 2,
        "candidate_exact_match_count": 1,
        "candidate_exact_match_rate": 0.5,
        "candidate_mean_evidence_edge_recall": 0.5,
        "candidate_mean_evidence_node_recall": 0.5,
        "case_count_delta": 0,
        "case_count_match": True,
        "exact_match_count_delta": 1,
        "exact_match_rate_delta": 0.5,
        "mean_evidence_edge_recall_delta": 0.5,
        "mean_evidence_node_recall_delta": 0.5,
    }
    assert validation["valid"] is True

    drifted_delta = dict(delta)
    drifted_summary = dict(delta["summary_delta"])
    drifted_summary["exact_match_rate_delta"] = 1.0
    drifted_delta["summary_delta"] = drifted_summary
    drifted_delta["report_digest"] = lab.qa_eval_delta_report_digest(drifted_delta)
    drifted_validation = lab.validate_qa_eval_delta_report(drifted_delta)
    checks = {check["name"]: check for check in drifted_validation["checks"]}
    assert drifted_validation["valid"] is False
    assert checks["summary_delta"]["passed"] is False


def test_qa_prediction_jsonl_digest_report_validation_and_comparison(tmp_path: Path) -> None:
    assert hasattr(lab, "qa_prediction_to_dict")
    assert hasattr(lab, "qa_prediction_from_dict")
    assert hasattr(lab, "qa_predictions_jsonl")
    assert hasattr(lab, "qa_predictions_digest")
    assert hasattr(lab, "save_qa_predictions")
    assert hasattr(lab, "load_qa_predictions")
    assert hasattr(lab, "qa_eval_report")
    assert hasattr(lab, "qa_eval_report_digest")
    assert hasattr(lab, "qa_eval_report_json")
    assert hasattr(lab, "save_qa_eval_report")
    assert hasattr(lab, "load_qa_eval_report")
    assert hasattr(lab, "validate_qa_eval_report")
    assert hasattr(lab, "compare_qa_eval_report")
    cases = qa_metric_cases()
    predictions = qa_metric_predictions(cases)
    gold_path = tmp_path / "gold.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    report_path = tmp_path / "report.json"

    lab.save_qa_dataset(cases, gold_path)
    saved_prediction_path = lab.save_qa_predictions(predictions, prediction_path)
    loaded_predictions = lab.load_qa_predictions(prediction_path)
    report = lab.qa_eval_report(
        cases,
        loaded_predictions,
        gold_path=gold_path,
        prediction_path=prediction_path,
    )
    saved_report_path = lab.save_qa_eval_report(report, report_path)
    loaded_report = lab.load_qa_eval_report(report_path)
    validation = lab.validate_qa_eval_report(loaded_report)
    comparison = lab.compare_qa_eval_report(loaded_report)

    assert [lab.qa_prediction_from_dict(lab.qa_prediction_to_dict(item)) for item in predictions] == list(
        predictions
    )
    assert lab.qa_predictions_jsonl(predictions).endswith("\n")
    assert lab.qa_predictions_digest(predictions) == lab.qa_predictions_digest(loaded_predictions)
    assert saved_prediction_path == prediction_path
    assert loaded_predictions == list(predictions)
    assert lab.qa_eval_report_digest(report) == report["report_digest"]
    assert lab.qa_eval_report_json(report) == lab.qa_eval_report_json(loaded_report)
    assert saved_report_path == report_path
    assert validation["valid"] is True
    assert comparison["matches"] is True

    tampered_report = dict(loaded_report)
    tampered_report["report_digest"] = "0" * 64
    tampered_validation = lab.validate_qa_eval_report(tampered_report)
    checks = {check["name"]: check for check in tampered_validation["checks"]}
    assert tampered_validation["valid"] is False
    assert checks["report_digest"]["passed"] is False


def test_run_qa_eval_cli_writes_validates_and_compares_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_qa_eval_script()
    main = cast(MainFn, getattr(module, "main"))
    cases = qa_metric_cases()
    predictions = qa_metric_predictions(cases)
    gold_path = tmp_path / "qa.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    report_path = tmp_path / "qa-eval-report.json"
    lab.save_qa_dataset(cases, gold_path)
    lab.save_qa_predictions(predictions, prediction_path)

    assert main(
        [
            "--gold",
            str(gold_path),
            "--pred",
            str(prediction_path),
            "--report",
            str(report_path),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    report = lab.load_qa_eval_report(report_path)
    assert output == {
        "action": "qa_eval_report",
        "path": str(report_path),
        "valid": True,
        "digest": report["report_digest"],
        "summary": report["summary"],
        "metrics": report["metrics"],
    }

    assert main(["--validate-report", str(report_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_qa_eval_report"
    assert validation["path"] == str(report_path)
    assert validation["valid"] is True

    assert main(["--compare-report", str(report_path)]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["action"] == "compare_qa_eval_report"
    assert comparison["path"] == str(report_path)
    assert comparison["matches"] is True


def test_run_qa_eval_cli_writes_validates_and_compares_delta_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_qa_eval_script()
    main = cast(MainFn, getattr(module, "main"))
    cases = qa_metric_cases()
    candidate_report_path = tmp_path / "graph-tool-report.json"
    baseline_report_path = tmp_path / "majority-report.json"
    delta_report_path = tmp_path / "qa-delta-report.json"
    candidate_report = lab.qa_eval_report(cases, qa_metric_predictions(cases))
    baseline_report = lab.qa_eval_report(cases, ())
    lab.save_qa_eval_report(candidate_report, candidate_report_path)
    lab.save_qa_eval_report(baseline_report, baseline_report_path)

    assert main(
        [
            "--candidate-report",
            str(candidate_report_path),
            "--baseline-report",
            str(baseline_report_path),
            "--candidate-name",
            "graph_tool",
            "--baseline-name",
            "majority",
            "--delta-report",
            str(delta_report_path),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    delta = lab.load_qa_eval_delta_report(delta_report_path)
    assert output == {
        "action": "qa_eval_delta_report",
        "path": str(delta_report_path),
        "valid": True,
        "digest": delta["report_digest"],
        "summary_delta": delta["summary_delta"],
    }

    assert main(["--validate-delta-report", str(delta_report_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_qa_eval_delta_report"
    assert validation["path"] == str(delta_report_path)
    assert validation["valid"] is True

    assert main(["--compare-delta-report", str(delta_report_path)]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["action"] == "compare_qa_eval_delta_report"
    assert comparison["path"] == str(delta_report_path)
    assert comparison["matches"] is True


def test_run_qa_eval_cli_returns_structured_json_for_invalid_predictions(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_qa_eval_script()
    main = cast(MainFn, getattr(module, "main"))
    gold_path = tmp_path / "qa.jsonl"
    prediction_path = tmp_path / "invalid-predictions.jsonl"
    report_path = tmp_path / "qa-eval-report.json"
    lab.save_qa_dataset(qa_metric_cases(), gold_path)
    prediction_path.write_text("[]\n", encoding="utf-8")

    assert main(
        [
            "--gold",
            str(gold_path),
            "--pred",
            str(prediction_path),
            "--report",
            str(report_path),
        ]
    ) == 1

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "action": "qa_eval_report",
        "path": str(report_path),
        "valid": False,
        "error": "QA prediction line 1 must be an object",
    }
