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
OBSERVABILITY_SCRIPT = ROOT / "scripts" / "analyze_qa_observability.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_observability_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "analyze_qa_observability_script",
        OBSERVABILITY_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_qa_observability_report_splits_cases_by_predicted_evidence() -> None:
    graph = _predicted_graph()
    cases = _qa_cases()

    report = lab.qa_observability_report(cases, graph)
    validation = lab.validate_qa_observability_report(report)

    assert report["summary"] == {
        "by_observability_status": {
            "evidence_observable": 1,
            "target_missing": 1,
            "target_observable_relation_missing": 2,
        },
        "by_question_type": {
            "object_location": {
                "case_count": 3,
                "evidence_observable_count": 1,
                "missing_evidence_count": 2,
                "target_missing_count": 1,
                "target_observable_count": 2,
                "target_observable_relation_missing_count": 1,
            },
            "reobserve_targets": {
                "case_count": 1,
                "evidence_observable_count": 0,
                "missing_evidence_count": 1,
                "target_missing_count": 0,
                "target_observable_count": 1,
                "target_observable_relation_missing_count": 1,
            },
        },
        "case_count": 4,
        "evidence_observable_count": 1,
        "missing_required_edge_count": 2,
        "missing_required_edge_relations": {"IN_ROOM": 3},
        "missing_evidence_count": 3,
        "missing_target_nodes": ["plate_1"],
        "target_node_count": 3,
        "target_node_missing_count": 1,
        "target_node_observed_count": 2,
        "target_node_recall": 0.666667,
        "target_observable_count": 3,
        "target_observable_relation_missing_count": 2,
        "target_missing_count": 1,
    }
    assert report["splits"]["evidence_observable"] == ["case:cup"]
    assert report["split_qa_digests"] == {
        "evidence_observable": lab.qa_dataset_digest((cases[0],)),
        "missing_evidence": lab.qa_dataset_digest((cases[1], cases[2], cases[3])),
        "target_missing": lab.qa_dataset_digest((cases[2],)),
        "target_observable": lab.qa_dataset_digest((cases[0], cases[1], cases[3])),
        "target_observable_relation_missing": lab.qa_dataset_digest(
            (cases[1], cases[3])
        ),
    }
    assert report["splits"]["target_observable"] == [
        "case:cup",
        "case:mug",
        "case:scene",
    ]
    assert report["splits"]["missing_evidence"] == [
        "case:mug",
        "case:plate",
        "case:scene",
    ]
    assert {
        case["case_id"]: case["observability_status"]
        for case in report["cases"]
    } == {
        "case:cup": "evidence_observable",
        "case:mug": "target_observable_relation_missing",
        "case:plate": "target_missing",
        "case:scene": "target_observable_relation_missing",
    }
    assert {
        case["case_id"]: case["missing_required_edge_relations"]
        for case in report["cases"]
    } == {
        "case:cup": [],
        "case:mug": ["IN_ROOM"],
        "case:plate": ["IN_ROOM"],
        "case:scene": ["IN_ROOM"],
    }
    assert validation["valid"] is True


def test_qa_observability_cli_writes_report_and_split_datasets(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_observability_script()
    main = cast(MainFn, getattr(module, "main"))
    qa_path = tmp_path / "qa.jsonl"
    graph_path = tmp_path / "predicted-graph.json"
    report_path = tmp_path / "qa-observability.json"
    evidence_path = tmp_path / "evidence-observable.jsonl"
    target_path = tmp_path / "target-observable.jsonl"
    missing_path = tmp_path / "missing-evidence.jsonl"
    relation_missing_path = tmp_path / "target-observable-relation-missing.jsonl"
    lab.save_qa_dataset(_qa_cases(), qa_path)
    lab.save_graph_json(_predicted_graph(), graph_path)

    assert main(
        [
            "--qa",
            str(qa_path),
            "--graph",
            str(graph_path),
            "--report",
            str(report_path),
            "--evidence-observable-qa",
            str(evidence_path),
            "--target-observable-qa",
            str(target_path),
            "--missing-evidence-qa",
            str(missing_path),
            "--target-observable-relation-missing-qa",
            str(relation_missing_path),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["action"] == "qa_observability_report"
    assert output["valid"] is True
    assert output["summary"]["evidence_observable_count"] == 1
    assert len(lab.load_qa_dataset(evidence_path)) == 1
    assert len(lab.load_qa_dataset(target_path)) == 3
    assert len(lab.load_qa_dataset(missing_path)) == 3
    assert len(lab.load_qa_dataset(relation_missing_path)) == 2

    assert main(["--validate-report", str(report_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_qa_observability_report"
    assert validation["valid"] is True

    assert main(["--compare-report", str(report_path)]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["action"] == "compare_qa_observability_report"
    assert comparison["matches"] is True


def _predicted_graph() -> lab.DynamicSceneGraph:
    graph = lab.DynamicSceneGraph()
    graph.add_room("room_1", "Room", step=1)
    graph.upsert_object(
        "cup_1",
        "cup",
        lab.Pose3D(0.0, 0.0, 1.0),
        lab.BBox3D(center=lab.Pose3D(0.0, 0.0, 1.0), size=(0.2, 0.2, 0.2)),
        confidence=0.9,
        visible=True,
        step=1,
    )
    graph.upsert_object(
        "mug_1",
        "mug",
        lab.Pose3D(0.2, 0.0, 1.0),
        lab.BBox3D(center=lab.Pose3D(0.2, 0.0, 1.0), size=(0.2, 0.2, 0.2)),
        confidence=0.9,
        visible=True,
        step=1,
    )
    graph.add_edge(
        "cup_1",
        "IN_ROOM",
        "room_1",
        "world",
        0.9,
        step=1,
        evidence=("fixture",),
        attributes={"source": "fixture"},
    )
    return graph


def _qa_cases() -> tuple[lab.QACase, ...]:
    digest = "0" * 64
    return (
        _case(
            "case:cup",
            {"type": "object_location", "object_id": "cup_1"},
            required_nodes=("cup_1", "room_1"),
            required_edges=("cup_1-IN_ROOM-room_1-1",),
            graph_digest=digest,
        ),
        _case(
            "case:mug",
            {"type": "object_location", "object_id": "mug_1"},
            required_nodes=("mug_1", "room_1"),
            required_edges=("mug_1-IN_ROOM-room_1-1",),
            graph_digest=digest,
        ),
        _case(
            "case:plate",
            {"type": "object_location", "object_id": "plate_1"},
            required_nodes=("plate_1", "room_1"),
            required_edges=("plate_1-IN_ROOM-room_1-1",),
            graph_digest=digest,
        ),
        _case(
            "case:scene",
            {"type": "reobserve_targets"},
            required_nodes=("cup_1", "mug_1"),
            required_edges=("mug_1-IN_ROOM-room_1-1",),
            graph_digest=digest,
        ),
    )


def _case(
    case_id: str,
    question: dict[str, object],
    *,
    required_nodes: tuple[str, ...],
    required_edges: tuple[str, ...],
    graph_digest: str,
) -> lab.QACase:
    return lab.QACase(
        id=case_id,
        scene_id="scene",
        episode_id="episode",
        graph_digest=graph_digest,
        step=1,
        question=question,
        question_type=str(question["type"]),
        answer={},
        answer_type=str(question["type"]),
        required_nodes=required_nodes,
        required_edges=required_edges,
        tags=("generated", "qa", str(question["type"])),
    )
