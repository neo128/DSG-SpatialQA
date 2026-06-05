from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Protocol, cast

from _pytest.capture import CaptureFixture

import dsg_spatialqa_lab as lab


ROOT = Path(__file__).resolve().parents[1]
BUILD_OBSERVATION_AWARE_QA_SCRIPT = ROOT / "scripts" / "build_observation_aware_qa.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_build_observation_aware_qa_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_observation_aware_qa_script",
        BUILD_OBSERVATION_AWARE_QA_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_observation_aware_qa_cases_use_visible_detector_current_location() -> None:
    original = lab.QACase(
        id="case:mug",
        scene_id="scene",
        episode_id="episode",
        graph_digest="0" * 64,
        step=1,
        question={"type": "object_location", "object_id": "mug_1"},
        question_type="object_location",
        answer={
            "object_id": "mug_1",
            "label": "mug",
            "current_location": {"relation": "IN_ROOM", "dst": "room_1", "step": 1},
            "visible": False,
        },
        answer_type="object_location",
        required_nodes=("mug_1", "state:mug_1:1"),
        required_edges=(
            "mug_1-IN_ROOM-room_1-1",
            "mug_1-STATE_CHANGED-state:mug_1:1-1",
        ),
        tags=("real", "qa"),
    )
    observations = (
        lab.SceneObservation(
            step=7,
            agent_pose=lab.Pose3D(0.0, 0.0, 0.0),
            objects=(
                lab.ObjectObservation(
                    "mug_1",
                    "mug",
                    lab.Pose3D(1.0, 0.8, 2.0),
                    lab.BBox3D(
                        center=lab.Pose3D(1.0, 0.8, 2.0),
                        size=(0.1, 0.1, 0.2),
                    ),
                    confidence=0.91,
                    visible=True,
                    attributes={
                        "current_location_id": "table_1",
                        "current_location_relation": "ON",
                        "depth_path": "depth/0007.npy",
                        "evidence_kinds": ["rgb", "depth", "detector"],
                        "rgb_path": "rgb/0007.png",
                        "source_kind": "detector",
                        "source_name": "grounded_sam2",
                        "states": {"isDirty": False},
                    },
                ),
            ),
        ),
    )

    cases, report = lab.observation_aware_qa_cases(
        (original,),
        observations,
        qa_path="qa.jsonl",
        observation_sequence_path="observations.json",
    )

    assert report["summary"] == {
        "base_case_count": 1,
        "generated_case_count": 1,
        "object_location_case_count": 1,
        "skipped_case_count": 0,
        "skipped_reasons": {},
    }
    assert report["base_qa_path"] == "qa.jsonl"
    assert report["observation_sequence_path"] == "observations.json"
    assert report["observation_sequence_digest"] == lab.scene_observation_sequence_digest(
        observations
    )
    assert len(cases) == 1
    case = cases[0]
    assert case.id == "case:mug:observation_aware:7"
    assert case.step == 7
    assert case.graph_digest == report["observation_sequence_digest"]
    assert case.answer == {
        "confidence": 0.91,
        "current_location": {"dst": "table_1", "relation": "ON", "step": 7},
        "label": "mug",
        "last_seen_step": 7,
        "object_id": "mug_1",
        "pose": {"x": 1.0, "y": 0.8, "yaw": 0.0, "z": 2.0},
        "state_step": 7,
        "visible": True,
    }
    assert case.required_nodes == ("mug_1", "state:mug_1:7", "table_1")
    assert case.required_edges == (
        "mug_1-ON-table_1-7",
        "mug_1-STATE_CHANGED-state:mug_1:7-7",
    )
    assert "observation_aware" in case.tags


def test_observation_aware_qa_cases_use_latest_visible_detector_observation() -> None:
    original = lab.QACase(
        id="case:mug",
        scene_id="scene",
        episode_id="episode",
        graph_digest="0" * 64,
        step=1,
        question={"type": "object_location", "object_id": "mug_1"},
        question_type="object_location",
        answer={},
        answer_type="object_location",
    )
    observations = (
        _detector_observation(
            step=3,
            object_id="mug_1",
            location_id="counter_1",
            relation="ON",
        ),
        _detector_observation(
            step=8,
            object_id="mug_1",
            location_id="sink_1",
            relation="INSIDE",
        ),
    )

    cases, report = lab.observation_aware_qa_cases((original,), observations)

    assert report["cases"][0]["step"] == 8
    assert len(cases) == 1
    assert cases[0].id == "case:mug:observation_aware:8"
    assert cases[0].answer["current_location"] == {
        "dst": "sink_1",
        "relation": "INSIDE",
        "step": 8,
    }
    assert cases[0].answer["last_seen_step"] == 8
    assert cases[0].answer["state_step"] == 8
    assert cases[0].required_edges == (
        "mug_1-INSIDE-sink_1-8",
        "mug_1-STATE_CHANGED-state:mug_1:8-8",
    )


def test_observation_aware_qa_cases_supplement_to_target_count_with_specific_relations() -> None:
    original = lab.QACase(
        id="case:mug",
        scene_id="scene",
        episode_id="episode",
        graph_digest="0" * 64,
        step=1,
        question={"type": "object_location", "object_id": "mug_1"},
        question_type="object_location",
        answer={},
        answer_type="object_location",
    )
    observations = (
        _detector_observation(
            step=3,
            object_id="mug_1",
            location_id="counter_1",
            relation="ON",
        ),
        _detector_observation(
            step=4,
            object_id="chair_1",
            location_id="ai2thor_room",
            relation="IN_ROOM",
        ),
        _detector_observation(
            step=5,
            object_id="book_1",
            location_id="desk_1",
            relation="ON",
        ),
    )

    cases, report = lab.observation_aware_qa_cases(
        (original,),
        observations,
        target_case_count=3,
    )

    assert report["summary"]["generated_case_count"] == 3
    assert report["summary"]["supplemental_case_count"] == 2
    assert report["summary"]["target_case_count"] == 3
    assert [case.answer["object_id"] for case in cases] == ["mug_1", "book_1", "chair_1"]
    assert cases[1].id == (
        "episode:scene:supplemental_object_location:0001:book_1:observation_aware:5"
    )
    assert cases[1].question == {"type": "object_location", "object_id": "book_1"}
    assert cases[1].required_edges == (
        "book_1-ON-desk_1-5",
        "book_1-STATE_CHANGED-state:book_1:5-5",
    )
    assert "observation_aware_supplemental" in cases[1].tags
    assert cases[2].answer["current_location"]["relation"] == "IN_ROOM"


def test_observation_aware_qa_cli_writes_dataset_and_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_build_observation_aware_qa_script()
    main = cast(MainFn, getattr(module, "main"))

    qa_path = tmp_path / "qa.jsonl"
    sequence_path = tmp_path / "observations.json"
    output_qa = tmp_path / "observation-aware.jsonl"
    report_path = tmp_path / "observation-aware-report.json"
    lab.save_qa_dataset(
        (
            lab.QACase(
                id="case:mug",
                scene_id="scene",
                episode_id="episode",
                graph_digest="0" * 64,
                step=1,
                question={"type": "object_location", "object_id": "mug_1"},
                question_type="object_location",
                answer={},
                answer_type="object_location",
            ),
        ),
        qa_path,
    )
    lab.save_scene_observation_sequence(
        (
            lab.SceneObservation(
                step=3,
                objects=(
                    lab.ObjectObservation(
                        "mug_1",
                        "mug",
                        lab.Pose3D(0.0, 0.0, 0.0),
                        lab.BBox3D(
                            center=lab.Pose3D(0.0, 0.0, 0.0),
                            size=(0.1, 0.1, 0.1),
                        ),
                        confidence=1.0,
                        visible=True,
                        attributes={
                            "current_location_id": "counter_1",
                            "current_location_relation": "ON",
                            "evidence_kinds": ["detector", "depth", "rgb"],
                            "episode_id": "episode",
                            "scene_id": "scene",
                            "source_kind": "detector",
                            "states": {"isOpen": False},
                        },
                    ),
                ),
            ),
        ),
        sequence_path,
    )

    assert main(
        [
            "--qa",
            str(qa_path),
            "--observation-sequence",
            str(sequence_path),
            "--output-qa",
            str(output_qa),
            "--report",
            str(report_path),
        ]
    ) == 0

    output = capsys.readouterr().out
    assert '"generated_case_count": 1' in output
    assert len(lab.load_qa_dataset(output_qa)) == 1
    report = lab.load_observation_aware_qa_report(report_path)
    assert lab.validate_observation_aware_qa_report(report)["valid"] is True


def _detector_observation(
    *,
    step: int,
    object_id: str,
    location_id: str,
    relation: str,
) -> lab.SceneObservation:
    return lab.SceneObservation(
        step=step,
        objects=(
            lab.ObjectObservation(
                object_id,
                object_id.split("_", 1)[0],
                lab.Pose3D(float(step), 0.0, 0.0),
                lab.BBox3D(
                    center=lab.Pose3D(float(step), 0.0, 0.0),
                    size=(0.1, 0.1, 0.1),
                ),
                confidence=1.0,
                visible=True,
                attributes={
                    "current_location_id": location_id,
                    "current_location_relation": relation,
                    "evidence_kinds": ["detector", "depth", "rgb"],
                    "episode_id": "episode",
                    "scene_id": "scene",
                    "source_kind": "detector",
                    "states": {"isVisible": True},
                },
            ),
        ),
    )
