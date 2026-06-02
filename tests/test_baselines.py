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
RUN_BASELINES_SCRIPT = ROOT / "scripts" / "run_baselines.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_run_baselines_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_baselines_script", RUN_BASELINES_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def baseline_cases() -> list[lab.QACase]:
    graph = lab.load_scene_fixture("tabletop")
    cases = lab.generate_qa_cases(
        graph,
        scene_id="tabletop_scene",
        episode_id="episode_001",
    )
    return [
        next(case for case in cases if case.question_type == "object_location"),
        next(case for case in cases if case.question_type == "relative_relation"),
        next(case for case in cases if case.question_type == "nearest_object"),
    ]


def test_graph_tool_baseline_answers_generated_qa_with_full_evidence() -> None:
    assert hasattr(lab, "BaselineSpec")
    assert hasattr(lab, "list_baselines")
    assert hasattr(lab, "run_baseline_predictions")
    graph = lab.load_scene_fixture("tabletop")
    cases = baseline_cases()

    predictions = lab.run_baseline_predictions("graph_tool", graph=graph, cases=cases)
    report = lab.evaluate_qa_predictions(cases, predictions)

    assert [prediction.id for prediction in predictions] == [case.id for case in cases]
    assert [prediction.answer for prediction in predictions] == [case.answer for case in cases]
    assert [prediction.evidence_nodes for prediction in predictions] == [
        case.required_nodes for case in cases
    ]
    assert [prediction.evidence_edges for prediction in predictions] == [
        case.required_edges for case in cases
    ]
    assert report["summary"]["exact_match_rate"] == 1.0
    assert report["metrics"]["evidence_node_recall"]["average"] == 1.0
    assert report["metrics"]["evidence_edge_recall"]["average"] == 1.0


def test_majority_baseline_is_stable_and_uses_first_choice() -> None:
    graph = lab.load_scene_fixture("tabletop")
    cases = baseline_cases()
    nearest_case = next(case for case in cases if case.question_type == "nearest_object")

    predictions = lab.run_baseline_predictions("majority", graph=graph, cases=cases)
    repeated_predictions = lab.run_baseline_predictions("majority", graph=graph, cases=cases)
    nearest_prediction = next(prediction for prediction in predictions if prediction.id == nearest_case.id)

    assert predictions == repeated_predictions
    assert nearest_prediction.answer == {
        "choice": nearest_case.choices[0],
        "strategy": "first_choice",
    }
    assert nearest_prediction.evidence_nodes == ()
    assert nearest_prediction.confidence == 0.0


def test_disabled_caption_memory_baseline_returns_structured_errors() -> None:
    graph = lab.load_scene_fixture("tabletop")
    cases = baseline_cases()[:2]

    predictions = lab.run_baseline_predictions("caption_memory", graph=graph, cases=cases)

    assert [prediction.id for prediction in predictions] == [case.id for case in cases]
    assert [prediction.error for prediction in predictions] == [
        "baseline_disabled",
        "baseline_disabled",
    ]
    assert all(prediction.answer == {} for prediction in predictions)


def test_run_baselines_cli_lists_and_writes_predictions(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_baselines_script()
    main = cast(MainFn, getattr(module, "main"))
    graph = lab.load_scene_fixture("tabletop")
    cases = baseline_cases()
    graph_path = tmp_path / "graph.json"
    qa_path = tmp_path / "qa.jsonl"
    pred_path = tmp_path / "predictions.jsonl"
    lab.save_graph_json(graph, graph_path)
    lab.save_qa_dataset(cases, qa_path)

    assert main(["--list-baselines"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing == {
        "action": "list_baselines",
        "baselines": [
            {"enabled": False, "kind": "interface", "name": "caption_memory"},
            {"enabled": True, "kind": "local", "name": "graph_text"},
            {"enabled": True, "kind": "local", "name": "graph_tool"},
            {"enabled": True, "kind": "local", "name": "majority"},
        ],
    }

    assert main(
        [
            "--baseline",
            "graph_tool",
            "--graph",
            str(graph_path),
            "--qa",
            str(qa_path),
            "--pred",
            str(pred_path),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    predictions = lab.load_qa_predictions(pred_path)
    assert output == {
        "action": "run_baseline",
        "baseline": "graph_tool",
        "path": str(pred_path),
        "prediction_count": len(cases),
        "digest": lab.qa_predictions_digest(predictions),
    }
    assert lab.evaluate_qa_predictions(cases, predictions)["summary"]["exact_match_rate"] == 1.0


def test_run_baselines_cli_returns_structured_json_for_unknown_baseline(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_baselines_script()
    main = cast(MainFn, getattr(module, "main"))
    graph = lab.load_scene_fixture("tabletop")
    graph_path = tmp_path / "graph.json"
    qa_path = tmp_path / "qa.jsonl"
    pred_path = tmp_path / "predictions.jsonl"
    lab.save_graph_json(graph, graph_path)
    lab.save_qa_dataset(baseline_cases(), qa_path)

    assert main(
        [
            "--baseline",
            "missing",
            "--graph",
            str(graph_path),
            "--qa",
            str(qa_path),
            "--pred",
            str(pred_path),
        ]
    ) == 1

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "action": "run_baseline",
        "baseline": "missing",
        "path": str(pred_path),
        "valid": False,
        "error": "Unsupported baseline: missing",
    }
