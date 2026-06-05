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
QA_V2_SCRIPT = ROOT / "scripts" / "build_qa_v2.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_qa_v2_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_qa_v2_script", QA_V2_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_qa_v2_splits_add_quality_fields_and_episode_situation() -> None:
    cases = (
        _case("case:apple", "apple_1", "countertop_1", relation="ON"),
        _case("case:book", "book_1", "room_1", relation="IN_ROOM"),
        _case(
            "case:left",
            "mug_1",
            "plate_1",
            relation="LEFT_OF",
            question_type="relative_relation",
        ),
        _case(
            "case:timeline",
            "cup_1",
            "table_1",
            relation="ON",
            question_type="relation_timeline",
        ),
    )
    observability_report = _observability_report()
    frames = (
        lab.EpisodeFrame(
            episode_id="episode-001",
            scene_id="FloorPlan1",
            step=1,
            rgb_path="rgb/000001.png",
            depth_path="depth/000001.npy",
            segmentation_path=None,
            agent_id="agent",
            agent_pose=lab.Pose3D(1.0, 0.9, 2.0, yaw=90.0),
            action="Initialize",
            visible_object_ids=("apple_1", "countertop_1"),
        ),
    )

    splits = lab.qa_v2_splits(
        cases,
        observability_report=observability_report,
        episode_frames=frames,
    )
    validation = lab.validate_qa_v2_splits(splits)

    assert validation["valid"] is True
    assert set(splits) == {
        "full_oracle",
        "observation_aware",
        "situated",
        "temporal",
        "anti_shortcut",
    }
    assert len(splits["full_oracle"]) == 4
    assert [row["source_case_id"] for row in splits["observation_aware"]] == ["case:apple"]
    assert [row["source_case_id"] for row in splits["situated"]] == ["case:left"]
    assert [row["source_case_id"] for row in splits["temporal"]] == ["case:timeline"]
    assert [row["source_case_id"] for row in splits["anti_shortcut"]] == [
        "case:apple",
        "case:left",
        "case:timeline",
    ]

    apple = splits["observation_aware"][0]
    assert apple["schema_version"] == lab.QA_V2_CASE_SCHEMA_VERSION
    assert apple["split"] == "observation_aware"
    assert apple["question_text"] == "Where is the apple?"
    assert apple["situation"] == {
        "step": 1,
        "reference_frame": "world",
        "agent_pose": {"x": 1.0, "y": 0.9, "z": 2.0, "yaw": 90.0},
        "view_frame": "rgb/000001.png",
        "source": "episode_frame",
    }
    assert apple["target"] == {"object_id": "apple_1", "label": "apple"}
    assert apple["answer"] == {
        "relation": "ON",
        "dst": "countertop_1",
        "dst_label": "countertop",
        "step": 1,
    }
    assert apple["answer_options"][0] == {
        "relation": "ON",
        "dst_label": "countertop",
    }
    assert apple["required_evidence"]["nodes"] == ["apple_1", "countertop_1"]
    assert apple["required_evidence"]["edges"] == ["apple_1-ON-countertop_1-1"]
    assert apple["observability"]["evidence_observable"] is True
    assert apple["anti_shortcut"]["requires_3d_evidence"] is True
    assert apple["anti_shortcut"]["language_prior_risk"] == "medium"


def test_qa_v2_cli_writes_split_files_and_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_qa_v2_script()
    main = cast(MainFn, getattr(module, "main"))
    qa_path = tmp_path / "qa.jsonl"
    episode_path = tmp_path / "episode.jsonl"
    observability_path = tmp_path / "observability.json"
    output_dir = tmp_path / "qa-v2"
    report_path = tmp_path / "qa-v2-report.json"
    lab.save_qa_dataset(
        (
            _case("case:apple", "apple_1", "countertop_1", relation="ON"),
            _case("case:book", "book_1", "room_1", relation="IN_ROOM"),
        ),
        qa_path,
    )
    lab.save_episode_sequence(
        (
            lab.EpisodeFrame(
                episode_id="episode-001",
                scene_id="FloorPlan1",
                step=1,
                rgb_path="rgb/000001.png",
                depth_path=None,
                segmentation_path=None,
                agent_id="agent",
                agent_pose=lab.Pose3D(0.0, 0.9, 0.0, yaw=0.0),
                action="Initialize",
            ),
        ),
        episode_path,
    )
    observability_path.write_text(
        json.dumps(_observability_report(), sort_keys=True),
        encoding="utf-8",
    )

    assert main(
        [
            "--qa",
            str(qa_path),
            "--episode",
            str(episode_path),
            "--observability-report",
            str(observability_path),
            "--output-dir",
            str(output_dir),
            "--report",
            str(report_path),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["action"] == "build_qa_v2_splits"
    assert output["valid"] is True
    assert output["summary"]["split_counts"]["full_oracle"] == 2
    assert (output_dir / "qa-full-oracle.jsonl").exists()
    assert (output_dir / "qa-observation-aware.jsonl").exists()
    assert (output_dir / "qa-situated.jsonl").exists()
    assert (output_dir / "qa-temporal.jsonl").exists()
    assert (output_dir / "qa-anti-shortcut.jsonl").exists()
    assert json.loads((output_dir / "qa-full-oracle.jsonl").read_text().splitlines()[0])[
        "question_text"
    ] == "Where is the apple?"

    assert main(["--validate-report", str(report_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_qa_v2_split_report"
    assert validation["valid"] is True


def _observability_report() -> dict[str, object]:
    return {
        "splits": {
            "evidence_observable": ["case:apple"],
            "target_observable": ["case:apple", "case:book"],
            "target_observable_relation_missing": ["case:book"],
            "target_missing": [],
            "missing_evidence": ["case:book"],
        },
        "cases": [
            {"case_id": "case:apple", "observability_status": "evidence_observable"},
            {
                "case_id": "case:book",
                "observability_status": "target_observable_relation_missing",
            },
        ],
    }


def _case(
    case_id: str,
    object_id: str,
    dst: str,
    *,
    relation: str = "ON",
    question_type: str = "object_location",
) -> lab.QACase:
    question = {"type": question_type, "object_id": object_id}
    if question_type == "relative_relation":
        question = {
            "type": question_type,
            "src": object_id,
            "relation": relation,
            "dst": dst,
            "reference_frame": "agent_egocentric",
        }
    if question_type == "relation_timeline":
        question = {
            "type": question_type,
            "src": object_id,
            "relation": relation,
            "dst": dst,
            "reference_frame": "world",
        }
    return lab.QACase(
        id=case_id,
        scene_id="FloorPlan1",
        episode_id="episode-001",
        graph_digest="0" * 64,
        step=1,
        question=question,
        question_type=question_type,
        answer={
            "object_id": object_id,
            "label": object_id.split("_")[0],
            "current_location": {"relation": relation, "dst": dst, "step": 1},
        },
        answer_type=question_type,
        reference_frame=question.get("reference_frame"),
        required_nodes=(object_id, dst),
        required_edges=(f"{object_id}-{relation}-{dst}-1",),
        tags=("generated", "qa", question_type),
    )
