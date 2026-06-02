from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from _pytest.capture import CaptureFixture

import dsg_spatialqa_lab as lab
from dsg_spatialqa_lab import GraphTool, SpatialQAEngine


ROOT = Path(__file__).resolve().parents[1]
GENERATE_QA_SCRIPT = ROOT / "scripts" / "generate_qa.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_generate_qa_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generate_qa_script", GENERATE_QA_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_qa_cases_are_stable_answerable_and_evidence_backed() -> None:
    assert hasattr(lab, "QACase")
    assert hasattr(lab, "generate_qa_cases")
    graph = lab.load_scene_fixture("moved_mug")

    cases = lab.generate_qa_cases(
        graph,
        scene_id="moved_mug_scene",
        episode_id="episode_001",
        tags=("oracle",),
    )
    repeated_cases = lab.generate_qa_cases(
        graph,
        scene_id="moved_mug_scene",
        episode_id="episode_001",
        tags=("oracle",),
    )
    by_type = {case.question_type: case for case in cases}
    qa = SpatialQAEngine(GraphTool(graph))

    assert cases == repeated_cases
    assert [case.question_type for case in cases[:6]] == [
        "object_location",
        "object_location",
        "object_location",
        "object_room",
        "object_room",
        "relative_relation",
    ]
    assert {
        "nearest_object",
        "next_action_validity",
        "relation_timeline",
        "reobserve_targets",
        "scene_delta",
    }.issubset(by_type)
    assert cases[0].id == "episode_001:moved_mug_scene:0001:object_location:mug_1"
    assert cases[0].graph_digest == lab.graph_json_digest(graph)
    assert cases[0].tags == ("generated", "oracle", "qa", "object_location")
    assert cases[0].required_nodes == ("mug_1", "state:mug_1:2")
    assert "mug_1-STATE_CHANGED-state:mug_1:2-2" in cases[0].required_edges

    for case in cases:
        response = qa.answer(case.question)
        assert response.error is None
        assert response.answer == case.answer
        assert tuple(response.evidence_nodes) == case.required_nodes
        assert tuple(response.evidence_edges) == case.required_edges


def test_generate_qa_cases_respects_max_cases_and_nearest_margin() -> None:
    graph = lab.load_scene_fixture("tabletop")

    cases = lab.generate_qa_cases(
        graph,
        scene_id="tabletop_scene",
        episode_id="episode_001",
        max_cases=4,
    )
    nearest_cases = [
        case
        for case in lab.generate_qa_cases(
            graph,
            scene_id="tabletop_scene",
            episode_id="episode_001",
        )
        if case.question_type == "nearest_object"
    ]

    assert len(cases) == 4
    assert [case.id for case in cases] == [
        "episode_001:tabletop_scene:0001:object_location:mug_1",
        "episode_001:tabletop_scene:0002:object_location:plate_1",
        "episode_001:tabletop_scene:0003:object_location:table_1",
        "episode_001:tabletop_scene:0004:relative_relation:mug_1_LEFT_OF_plate_1",
    ]
    assert [case.question["src"] for case in nearest_cases] == ["mug_1"]
    assert nearest_cases[0].choices == ("plate_1", "table_1")


def test_qa_dataset_jsonl_digest_validate_and_compare_current_graph(tmp_path: Path) -> None:
    assert hasattr(lab, "qa_case_to_dict")
    assert hasattr(lab, "qa_case_from_dict")
    assert hasattr(lab, "qa_dataset_jsonl")
    assert hasattr(lab, "qa_dataset_digest")
    assert hasattr(lab, "save_qa_dataset")
    assert hasattr(lab, "load_qa_dataset")
    assert hasattr(lab, "validate_qa_dataset")
    assert hasattr(lab, "compare_qa_dataset")
    graph = lab.load_scene_fixture("moved_mug")
    cases = lab.generate_qa_cases(
        graph,
        scene_id="moved_mug_scene",
        episode_id="episode_001",
        max_cases=6,
    )
    dataset_path = tmp_path / "qa" / "dataset.jsonl"

    payload = lab.qa_dataset_jsonl(cases)
    repeated_payload = lab.qa_dataset_jsonl(cases)
    saved_path = lab.save_qa_dataset(cases, dataset_path)
    loaded_cases = lab.load_qa_dataset(dataset_path)
    validation = lab.validate_qa_dataset(loaded_cases)
    comparison = lab.compare_qa_dataset(loaded_cases, graph)

    assert payload == repeated_payload
    assert payload.endswith("\n")
    assert [lab.qa_case_from_dict(lab.qa_case_to_dict(case)) for case in cases] == cases
    assert lab.qa_dataset_digest(cases) == lab.qa_dataset_digest(loaded_cases)
    assert saved_path == dataset_path
    assert loaded_cases == cases
    assert validation["valid"] is True
    assert validation["summary"] == {
        "schema_version": "dsg-spatialqa-lab.qa-dataset-summary.v1",
        "case_count": 6,
        "scene_ids": ["moved_mug_scene"],
        "episode_ids": ["episode_001"],
        "question_types": {
            "object_location": 3,
            "object_room": 2,
            "relative_relation": 1,
        },
        "tags": {
            "generated": 6,
            "object_location": 3,
            "object_room": 2,
            "qa": 6,
            "relative_relation": 1,
        },
    }
    assert comparison["matches"] is True

    tampered_case = lab.qa_case_from_dict(lab.qa_case_to_dict(cases[0]))
    tampered_case.answer["object_id"] = "wrong"
    drift = lab.compare_qa_dataset((tampered_case, *cases[1:]), graph)
    assert drift["matches"] is False
    checks = {check["name"]: check for check in drift["checks"]}
    assert checks["answers_match_graph"]["passed"] is False
    assert checks["answers_match_graph"]["differences"][0]["case_id"] == cases[0].id


def test_validate_qa_dataset_reports_shape_errors() -> None:
    graph = lab.load_scene_fixture("tabletop")
    case = lab.generate_qa_cases(
        graph,
        scene_id="tabletop_scene",
        episode_id="episode_001",
        max_cases=1,
    )[0]

    validation = lab.validate_qa_dataset((case, case))

    assert validation["valid"] is False
    checks = {check["name"]: check for check in validation["checks"]}
    assert checks["unique_case_ids"] == {
        "name": "unique_case_ids",
        "passed": False,
        "expected": 2,
        "actual": 1,
    }


def test_generate_qa_cli_writes_validates_and_compares_explicit_dataset(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_generate_qa_script()
    main = cast(MainFn, getattr(module, "main"))
    graph = lab.load_scene_fixture("moved_mug")
    graph_path = tmp_path / "oracle-graph.json"
    dataset_path = tmp_path / "qa.jsonl"
    lab.save_graph_json(graph, graph_path)

    assert main(
        [
            "--graph",
            str(graph_path),
            "--scene-id",
            "moved_mug_scene",
            "--episode-id",
            "episode_001",
            "--max-cases",
            "5",
            "--output",
            str(dataset_path),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    cases = lab.load_qa_dataset(dataset_path)
    assert output == {
        "action": "generate_qa_dataset",
        "path": str(dataset_path),
        "valid": True,
        "digest": lab.qa_dataset_digest(cases),
        "summary": lab.qa_dataset_summary(cases),
    }

    assert main(["--validate", str(dataset_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_qa_dataset"
    assert validation["path"] == str(dataset_path)
    assert validation["valid"] is True

    assert main(["--compare", str(dataset_path), "--graph", str(graph_path)]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["action"] == "compare_qa_dataset"
    assert comparison["path"] == str(dataset_path)
    assert comparison["matches"] is True


def test_generate_qa_cli_returns_structured_json_for_invalid_dataset(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_generate_qa_script()
    main = cast(MainFn, getattr(module, "main"))
    dataset_path = tmp_path / "invalid.jsonl"
    dataset_path.write_text("[]\n", encoding="utf-8")

    assert main(["--validate", str(dataset_path)]) == 1

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "action": "validate_qa_dataset",
        "path": str(dataset_path),
        "valid": False,
        "error": "QA dataset line 1 must be an object",
    }
