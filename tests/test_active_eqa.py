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
RUN_ACTIVE_TASKS_SCRIPT = ROOT / "scripts" / "run_active_tasks.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_run_active_tasks_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_active_tasks_script",
        RUN_ACTIVE_TASKS_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_active_task_jsonl_round_trip_and_direct_answer_policy() -> None:
    assert hasattr(lab, "ActiveEQATask")
    assert hasattr(lab, "active_eqa_task_to_dict")
    assert hasattr(lab, "active_eqa_task_from_dict")
    assert hasattr(lab, "active_eqa_tasks_jsonl")
    assert hasattr(lab, "active_eqa_tasks_digest")
    assert hasattr(lab, "ActiveGraphAgent")
    graph = lab.load_scene_fixture("tabletop")
    task = _task_for_case(_case_by_type("object_location"), max_actions=1)

    payload = lab.active_eqa_tasks_jsonl((task,))
    agent = lab.ActiveGraphAgent(policy="direct_answer")
    result = agent.run(task, lab.MockActiveEnvironment({task.initial_step: graph}))

    assert lab.active_eqa_task_from_dict(lab.active_eqa_task_to_dict(task)) == task
    assert payload.endswith("\n")
    assert lab.active_eqa_tasks_digest((task,)) == lab.active_eqa_tasks_digest(
        lab.active_eqa_tasks_from_jsonl(payload)
    )
    assert result.task_id == task.id
    assert result.policy == "direct_answer"
    assert result.success is True
    assert result.action_count == 0
    assert result.answer == task.gold_answer
    assert set(result.evidence_nodes) >= set(task.required_evidence["nodes"])
    assert result.error is None


def test_oracle_evidence_policy_collects_evidence_in_mock_environment() -> None:
    full_graph = lab.load_scene_fixture("tabletop")
    task = _task_for_case(_case_by_id_suffix("object_location:plate_1"), max_actions=1)
    initial_graph = _graph_without_object("plate_1")

    result = lab.ActiveGraphAgent(policy="oracle_evidence").run(
        task,
        lab.MockActiveEnvironment(
            {
                task.initial_step: initial_graph,
                task.initial_step + 1: full_graph,
            }
        ),
    )
    report = lab.active_task_report((task,), (result,))

    assert result.success is True
    assert result.action_count == 1
    assert result.transcript == (
        {
            "action": "observe_required_evidence",
            "from_step": 1,
            "to_step": 2,
        },
    )
    assert report["summary"] == {
        "task_count": 1,
        "success_count": 1,
        "failure_count": 0,
        "total_action_count": 1,
        "by_policy": {"oracle_evidence": 1},
    }
    assert report["metrics"] == {
        "answer_accuracy": {"count": 1, "rate": 1.0, "total": 1},
        "answer_graph_consistency": {"count": 1, "rate": 1.0, "total": 1},
        "evidence_coverage": {"average": 1.0, "total": 1},
        "task_success": {"count": 1, "rate": 1.0, "total": 1},
        "action_count": {"average": 1.0, "total": 1},
    }


def test_active_policy_records_per_action_evidence_snapshots() -> None:
    full_graph = lab.load_scene_fixture("tabletop")
    task = _task_for_case(_case_by_id_suffix("object_location:plate_1"), max_actions=1)
    initial_graph = _graph_without_object("plate_1")

    result = lab.ActiveGraphAgent(policy="oracle_evidence").run(
        task,
        lab.MockActiveEnvironment(
            {
                task.initial_step: initial_graph,
                task.initial_step + 1: full_graph,
            }
        ),
    )
    report = lab.active_task_report((task,), (result,))

    assert len(result.action_evidence_snapshots) == 1
    snapshot = result.action_evidence_snapshots[0]
    assert snapshot["action"] == "observe_required_evidence"
    assert snapshot["action_index"] == 1
    assert snapshot["from_step"] == 1
    assert snapshot["to_step"] == 2
    assert snapshot["graph_digest"] == lab.graph_json_digest(full_graph)
    assert snapshot["evidence_coverage"] == 1.0
    assert snapshot["missing_required_edges"] == []
    assert snapshot["missing_required_nodes"] == []
    assert set(snapshot["evidence_edges"]) >= set(task.required_evidence["edges"])
    assert set(snapshot["evidence_nodes"]) >= set(task.required_evidence["nodes"])
    assert set(snapshot["new_evidence_edges"]) >= set(task.required_evidence["edges"])
    assert set(snapshot["new_evidence_nodes"]) >= set(task.required_evidence["nodes"])
    assert report["cases"][0]["action_evidence_snapshots"] == [snapshot]
    assert lab.active_task_result_from_dict(lab.active_task_result_to_dict(result)) == result


def test_next_best_view_policy_targets_missing_required_evidence() -> None:
    full_graph = lab.load_scene_fixture("tabletop")
    task = _task_for_case(_case_by_id_suffix("object_location:plate_1"), max_actions=1)
    initial_graph = _graph_without_object("plate_1")

    result = lab.ActiveGraphAgent(policy="next_best_view").run(
        task,
        lab.MockActiveEnvironment(
            {
                task.initial_step: initial_graph,
                task.initial_step + 1: full_graph,
            }
        ),
    )

    expected_target = {
        "missing_required_edges": list(task.required_evidence["edges"]),
        "missing_required_nodes": list(task.required_evidence["nodes"]),
        "selection_rule": "missing_required_evidence_then_sorted_id",
    }
    assert "next_best_view" in lab.list_active_policies()
    assert result.success is True
    assert result.transcript == (
        {
            "action": "next_best_view",
            "from_step": 1,
            "target": expected_target,
            "to_step": 2,
        },
    )
    assert result.action_evidence_snapshots[0]["action"] == "next_best_view"
    assert result.action_evidence_snapshots[0]["action_target"] == expected_target


def test_active_policy_enforces_max_actions() -> None:
    task = _task_for_case(_case_by_id_suffix("object_location:plate_1"), max_actions=0)
    initial_graph = _graph_without_object("plate_1")

    result = lab.ActiveGraphAgent(policy="oracle_evidence").run(
        task,
        lab.MockActiveEnvironment({task.initial_step: initial_graph}),
    )
    report = lab.active_task_report((task,), (result,))

    assert result.success is False
    assert result.action_count == 0
    assert result.error == "max_actions_exceeded"
    assert report["metrics"]["task_success"]["rate"] == 0.0
    assert report["metrics"]["evidence_coverage"]["average"] == 0.0


def test_active_task_report_includes_budget_success_analysis() -> None:
    case = _case_by_id_suffix("object_location:plate_1")
    failed_task = _task_for_case_with_id(case, task_id="active:budget0", max_actions=0)
    success_task = _task_for_case_with_id(case, task_id="active:budget1", max_actions=1)
    failed_result = lab.ActiveTaskResult(
        task_id=failed_task.id,
        policy="oracle_evidence",
        answer={},
        success=False,
        action_count=0,
        final_step=1,
        confidence=0.0,
        error="max_actions_exceeded",
    )
    success_result = lab.ActiveTaskResult(
        task_id=success_task.id,
        policy="oracle_evidence",
        answer=case.answer,
        success=True,
        action_count=1,
        evidence_nodes=case.required_nodes,
        evidence_edges=case.required_edges,
        final_step=2,
        confidence=1.0,
        transcript=(
            {
                "action": "observe_required_evidence",
                "from_step": 1,
                "to_step": 2,
            },
        ),
    )

    report = lab.active_task_report(
        (failed_task, success_task),
        (failed_result, success_result),
    )
    validation = lab.validate_active_task_report(report)

    assert report["cases"][0]["max_actions"] == 0
    assert report["cases"][1]["max_actions"] == 1
    assert report["budget_analysis"] == {
        "budget_curve": [
            {
                "average_action_count": 0.0,
                "average_evidence_coverage": 0.0,
                "max_actions": 0,
                "success_count": 0,
                "success_rate": 0.0,
                "task_count": 1,
            },
            {
                "average_action_count": 1.0,
                "average_evidence_coverage": 1.0,
                "max_actions": 1,
                "success_count": 1,
                "success_rate": 1.0,
                "task_count": 1,
            },
        ],
        "by_max_actions": {
            "0": {
                "average_action_count": 0.0,
                "average_evidence_coverage": 0.0,
                "success_count": 0,
                "success_rate": 0.0,
                "task_count": 1,
            },
            "1": {
                "average_action_count": 1.0,
                "average_evidence_coverage": 1.0,
                "success_count": 1,
                "success_rate": 1.0,
                "task_count": 1,
            },
        },
    }
    assert validation["valid"] is True

    tampered_report = dict(report)
    tampered_budget = dict(report["budget_analysis"])
    tampered_budget["budget_curve"] = []
    tampered_report["budget_analysis"] = tampered_budget
    tampered_report["report_digest"] = lab.active_task_report_digest(tampered_report)
    tampered_validation = lab.validate_active_task_report(tampered_report)
    checks = {check["name"]: check for check in tampered_validation["checks"]}
    assert tampered_validation["valid"] is False
    assert checks["budget_curve_count"]["passed"] is False


def test_active_task_report_digest_validation_and_cli(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    assert hasattr(lab, "active_task_report_digest")
    assert hasattr(lab, "active_task_report_json")
    assert hasattr(lab, "save_active_task_report")
    assert hasattr(lab, "load_active_task_report")
    assert hasattr(lab, "validate_active_task_report")
    module = load_run_active_tasks_script()
    main = cast(MainFn, getattr(module, "main"))
    graph = lab.load_scene_fixture("tabletop")
    task = _task_for_case(_case_by_type("object_location"), max_actions=1)
    task_path = tmp_path / "active-tasks.jsonl"
    graph_path = tmp_path / "graph.json"
    report_path = tmp_path / "active-report.json"
    lab.save_active_eqa_tasks((task,), task_path)
    lab.save_graph_json(graph, graph_path)

    assert main(
        [
            "--tasks",
            str(task_path),
            "--graph",
            str(graph_path),
            "--policy",
            "direct_answer",
            "--report",
            str(report_path),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    report = lab.load_active_task_report(report_path)
    assert output == {
        "action": "active_task_report",
        "path": str(report_path),
        "valid": True,
        "digest": report["report_digest"],
        "summary": report["summary"],
        "metrics": report["metrics"],
    }
    assert report["report_digest"] == lab.active_task_report_digest(report)
    assert json.loads(lab.active_task_report_json(report)) == report
    assert lab.validate_active_task_report(report)["valid"] is True

    assert main(["--validate-report", str(report_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_active_task_report"
    assert validation["path"] == str(report_path)
    assert validation["valid"] is True


def test_run_active_tasks_cli_returns_structured_json_for_invalid_tasks(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_active_tasks_script()
    main = cast(MainFn, getattr(module, "main"))
    graph_path = tmp_path / "graph.json"
    task_path = tmp_path / "invalid-tasks.jsonl"
    report_path = tmp_path / "active-report.json"
    lab.save_graph_json(lab.load_scene_fixture("tabletop"), graph_path)
    task_path.write_text("[]\n", encoding="utf-8")

    assert main(
        [
            "--tasks",
            str(task_path),
            "--graph",
            str(graph_path),
            "--policy",
            "direct_answer",
            "--report",
            str(report_path),
        ]
    ) == 1

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "action": "active_task_report",
        "path": str(report_path),
        "valid": False,
        "error": "Active EQA task line 1 must be an object",
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
    return _task_for_case_with_id(case, task_id=f"active:{case.id}", max_actions=max_actions)


def _task_for_case_with_id(
    case: lab.QACase,
    *,
    task_id: str,
    max_actions: int,
) -> lab.ActiveEQATask:
    return lab.ActiveEQATask(
        id=task_id,
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
