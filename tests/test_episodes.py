from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import pytest
from _pytest.capture import CaptureFixture

import dsg_spatialqa_lab as lab
from dsg_spatialqa_lab import EpisodeFrame, Pose3D, SpatialQAError


ROOT = Path(__file__).resolve().parents[1]
EPISODES_SCRIPT = ROOT / "scripts" / "episodes.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_episodes_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "episodes_script",
        EPISODES_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_episode_sequence_jsonl_round_trip_digest_and_save_loads_explicit_file(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "EpisodeFrame")
    assert hasattr(lab, "episode_frame_to_dict")
    assert hasattr(lab, "episode_frame_from_dict")
    assert hasattr(lab, "episode_sequence_to_jsonl")
    assert hasattr(lab, "episode_sequence_from_jsonl")
    assert hasattr(lab, "save_episode_sequence")
    assert hasattr(lab, "load_episode_sequence")
    assert hasattr(lab, "episode_sequence_digest")
    frames = (
        EpisodeFrame(
            episode_id="episode_001",
            scene_id="FloorPlan1",
            step=1,
            rgb_path="rgb/0001.png",
            depth_path="depth/0001.npy",
            segmentation_path=None,
            agent_id="agent",
            agent_pose=Pose3D(0.0, 0.0, 0.0, yaw=0.0),
            action="MoveAhead",
            visible_object_ids=("mug_1", "plate_1"),
            metadata={"split": "mock", "weather": "none"},
        ),
        EpisodeFrame(
            episode_id="episode_001",
            scene_id="FloorPlan1",
            step=2,
            rgb_path="rgb/0002.png",
            depth_path="depth/0002.npy",
            segmentation_path="seg/0002.png",
            agent_id="agent",
            agent_pose=Pose3D(0.25, 0.0, 0.0, yaw=0.0),
            action=None,
            visible_object_ids=("mug_1",),
            metadata={"split": "mock"},
        ),
    )

    payload = lab.episode_sequence_to_jsonl(frames)
    repeated_payload = lab.episode_sequence_to_jsonl(frames)
    episode_path = tmp_path / "episodes" / "mock-episode.jsonl"
    saved_path = lab.save_episode_sequence(frames, episode_path)
    loaded_frames = lab.load_episode_sequence(episode_path)

    assert payload == repeated_payload
    assert payload.endswith("\n")
    assert [json.loads(line) for line in payload.splitlines()] == [
        {
            "schema_version": "dsg-spatialqa-lab.episode-frame.v1",
            "episode_id": "episode_001",
            "scene_id": "FloorPlan1",
            "step": 1,
            "rgb_path": "rgb/0001.png",
            "depth_path": "depth/0001.npy",
            "segmentation_path": None,
            "agent_id": "agent",
            "agent_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            "action": "MoveAhead",
            "visible_object_ids": ["mug_1", "plate_1"],
            "metadata": {"split": "mock", "weather": "none"},
        },
        {
            "schema_version": "dsg-spatialqa-lab.episode-frame.v1",
            "episode_id": "episode_001",
            "scene_id": "FloorPlan1",
            "step": 2,
            "rgb_path": "rgb/0002.png",
            "depth_path": "depth/0002.npy",
            "segmentation_path": "seg/0002.png",
            "agent_id": "agent",
            "agent_pose": {"x": 0.25, "y": 0.0, "z": 0.0, "yaw": 0.0},
            "action": None,
            "visible_object_ids": ["mug_1"],
            "metadata": {"split": "mock"},
        },
    ]
    assert lab.episode_sequence_digest(frames) == hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()
    assert lab.episode_sequence_from_jsonl(payload) == frames
    assert saved_path == episode_path
    assert loaded_frames == frames
    assert episode_path.read_text(encoding="utf-8") == payload


def test_episode_sequence_validation_reports_ordering_and_duplicate_steps() -> None:
    first = EpisodeFrame(
        episode_id="episode_001",
        scene_id="FloorPlan1",
        step=1,
        rgb_path=None,
        depth_path=None,
        segmentation_path=None,
        agent_id="agent",
        agent_pose=Pose3D(0.0, 0.0, 0.0),
        action="MoveAhead",
        visible_object_ids=("mug_1",),
        metadata={},
    )
    second = EpisodeFrame(
        episode_id="episode_001",
        scene_id="FloorPlan1",
        step=2,
        rgb_path=None,
        depth_path=None,
        segmentation_path=None,
        agent_id="agent",
        agent_pose=Pose3D(0.25, 0.0, 0.0),
        action="RotateRight",
        visible_object_ids=("mug_1", "plate_1"),
        metadata={},
    )

    validation = lab.validate_episode_sequence((first, second))
    unsorted_validation = lab.validate_episode_sequence((second, first))
    duplicate_validation = lab.validate_episode_sequence((first, first))
    checks = {check["name"]: check for check in validation["checks"]}
    unsorted_checks = {check["name"]: check for check in unsorted_validation["checks"]}
    duplicate_checks = {check["name"]: check for check in duplicate_validation["checks"]}

    assert validation["valid"] is True
    assert validation["digest"] == lab.episode_sequence_digest((first, second))
    assert validation["summary"] == {
        "schema_version": "dsg-spatialqa-lab.episode-sequence-summary.v1",
        "frame_count": 2,
        "episode_count": 1,
        "scene_count": 1,
        "first_step": 1,
        "last_step": 2,
        "episode_ids": ["episode_001"],
        "scene_ids": ["FloorPlan1"],
        "action_count": 2,
        "visible_object_observation_count": 3,
        "unique_visible_object_ids": ["mug_1", "plate_1"],
        "by_episode": {"episode_001": 2},
        "by_scene": {"FloorPlan1": 2},
        "by_action": {"MoveAhead": 1, "RotateRight": 1},
    }
    assert checks["frame_count_positive"] == {
        "name": "frame_count_positive",
        "passed": True,
        "expected": "at least one frame",
        "actual": 2,
    }
    assert checks["ordered_by_episode_and_step"]["passed"] is True
    assert checks["unique_episode_steps"]["passed"] is True
    assert unsorted_validation["valid"] is False
    assert unsorted_checks["ordered_by_episode_and_step"] == {
        "name": "ordered_by_episode_and_step",
        "passed": False,
        "expected": [("episode_001", 1), ("episode_001", 2)],
        "actual": [("episode_001", 2), ("episode_001", 1)],
    }
    assert duplicate_validation["valid"] is False
    assert duplicate_checks["unique_episode_steps"] == {
        "name": "unique_episode_steps",
        "passed": False,
        "expected": 2,
        "actual": 1,
    }


def test_episode_frame_from_dict_rejects_missing_required_field() -> None:
    payload = {
        "schema_version": "dsg-spatialqa-lab.episode-frame.v1",
        "episode_id": "episode_001",
        "step": 1,
        "rgb_path": None,
        "depth_path": None,
        "segmentation_path": None,
        "agent_id": "agent",
        "agent_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        "action": None,
        "visible_object_ids": [],
        "metadata": {},
    }

    with pytest.raises(SpatialQAError, match="scene_id must be a string"):
        lab.episode_frame_from_dict(payload)


def test_episodes_cli_summarizes_validates_and_compares_explicit_jsonl(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_episodes_script()
    main = cast(MainFn, getattr(module, "main"))
    episode_path = tmp_path / "episodes" / "mock-episode.jsonl"
    frames = (
        EpisodeFrame(
            episode_id="episode_001",
            scene_id="FloorPlan1",
            step=1,
            rgb_path=None,
            depth_path=None,
            segmentation_path=None,
            agent_id="agent",
            agent_pose=Pose3D(0.0, 0.0, 0.0),
            action="MoveAhead",
            visible_object_ids=("mug_1",),
            metadata={},
        ),
    )
    lab.save_episode_sequence(frames, episode_path)

    assert main(["--summary", str(episode_path)]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "action": "summary_episode_sequence",
        "path": str(episode_path),
        "valid": True,
        "digest": lab.episode_sequence_digest(frames),
        "summary": lab.episode_sequence_summary(frames),
    }

    assert main(["--validate", str(episode_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_episode_sequence"
    assert validation["path"] == str(episode_path)
    assert validation["valid"] is True
    assert validation["digest"] == lab.episode_sequence_digest(frames)
    assert [check["passed"] for check in validation["checks"]] == [True, True, True]

    assert main(["--compare", str(episode_path)]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison == {
        "action": "compare_episode_sequence",
        "path": str(episode_path),
        "matches": True,
        "saved_digest": lab.episode_sequence_digest(frames),
        "current_digest": lab.episode_sequence_digest(frames),
        "checks": [
            {"name": "sequence_valid", "passed": True},
            {
                "name": "jsonl_round_trip_matches",
                "passed": True,
                "expected": lab.episode_sequence_digest(frames),
                "actual": lab.episode_sequence_digest(frames),
            },
        ],
    }


def test_episodes_cli_returns_nonzero_structured_json_for_invalid_jsonl(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_episodes_script()
    main = cast(MainFn, getattr(module, "main"))
    episode_path = tmp_path / "invalid-episode.jsonl"
    episode_path.write_text(
        json.dumps(
            {
                "schema_version": "dsg-spatialqa-lab.episode-frame.v1",
                "episode_id": "episode_001",
                "step": 1,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["--validate", str(episode_path)]) == 1

    validation = json.loads(capsys.readouterr().out)
    assert validation == {
        "action": "validate_episode_sequence",
        "path": str(episode_path),
        "valid": False,
        "error": "scene_id must be a string",
    }
