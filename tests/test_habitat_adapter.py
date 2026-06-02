from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from _pytest.capture import CaptureFixture
import pytest

import dsg_spatialqa_lab as lab
from dsg_spatialqa_lab import Pose3D, SpatialQAError


ROOT = Path(__file__).resolve().parents[1]
COLLECT_SCRIPT = ROOT / "scripts" / "collect_habitat.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_collect_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("collect_habitat_script", COLLECT_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_habitat_adapter_imports_without_optional_dependency_and_reports_gap() -> None:
    assert hasattr(lab, "HabitatAdapterConfig")
    assert hasattr(lab, "HabitatEpisodeCollector")
    assert hasattr(lab, "HABITAT_MISSING_DEPENDENCY_MESSAGE")
    config = lab.HabitatAdapterConfig(
        scene_id="apartment_0",
        episode_id="habitat_mock_001",
        steps=(1,),
    )
    collector = lab.HabitatEpisodeCollector(config)

    with pytest.raises(SpatialQAError, match="Habitat optional dependency is not installed"):
        collector.collect_episode()

    assert lab.HABITAT_MISSING_DEPENDENCY_MESSAGE == (
        "Habitat optional dependency is not installed. Install with .[habitat]."
    )


def test_mock_habitat_episode_is_deterministic_and_oracle_compatible() -> None:
    assert hasattr(lab, "build_mock_habitat_episode")
    assert hasattr(lab, "convert_habitat_observation_to_episode_frame")
    config = lab.HabitatAdapterConfig(
        scene_id="apartment_0",
        episode_id="habitat_mock_001",
        steps=(1, 2),
        actions=("reset", "turn_left"),
        artifact_root="artifacts/habitat",
    )

    frames = lab.build_mock_habitat_episode(config)
    repeated_frames = lab.build_mock_habitat_episode(config)
    graph = lab.build_oracle_graph_from_episode(frames)
    validation = lab.validate_episode_sequence(frames)

    assert frames == repeated_frames
    assert validation["valid"] is True
    assert [frame.step for frame in frames] == [1, 2]
    assert [frame.action for frame in frames] == ["reset", "turn_left"]
    assert frames[0].rgb_path == "artifacts/habitat/habitat_mock_001/rgb/0001.png"
    assert frames[0].metadata["adapter"] == "habitat"
    assert frames[0].metadata["rooms"] == [
        {"label": "Living room", "room_id": "living_room"}
    ]
    assert frames[1].visible_object_ids == ("chair_1",)
    assert graph.get_object_state("chair_1").last_seen_step == 2
    assert graph.find_edges(src="chair_1", relation="NEAR")[0].dst == "table_1"

    frame = lab.convert_habitat_observation_to_episode_frame(
        {
            "agent_pose": {"x": 0.2, "y": 0.0, "z": 0.4, "yaw": 45.0},
            "visible_object_ids": ["chair_1"],
            "metadata": {"objects": []},
        },
        config=config,
        step=5,
        action="move_forward",
    )

    assert frame.agent_pose == Pose3D(0.2, 0.0, 0.4, yaw=45.0)
    assert frame.metadata == {"adapter": "habitat", "objects": []}


def test_mock_habitat_episode_rejects_implicit_steps_and_action_mismatch() -> None:
    empty_steps_config = lab.HabitatAdapterConfig(
        scene_id="apartment_0",
        episode_id="habitat_mock_001",
        steps=(),
    )
    mismatched_actions_config = lab.HabitatAdapterConfig(
        scene_id="apartment_0",
        episode_id="habitat_mock_001",
        steps=(1, 2),
        actions=("reset",),
    )

    with pytest.raises(SpatialQAError, match="Habitat adapter steps must be explicit"):
        lab.build_mock_habitat_episode(empty_steps_config)

    with pytest.raises(SpatialQAError, match="actions length must match steps length"):
        lab.build_mock_habitat_episode(mismatched_actions_config)


def test_collect_habitat_cli_writes_mock_episode_and_validates_output(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_collect_script()
    main = cast(MainFn, getattr(module, "main"))
    output_path = tmp_path / "episodes" / "mock-habitat.jsonl"

    assert main(
        [
            "--mock",
            "--scene",
            "apartment_0",
            "--episode-id",
            "habitat_mock_001",
            "--step",
            "1",
            "--step",
            "2",
            "--action",
            "reset",
            "--action",
            "turn_left",
            "--output",
            str(output_path),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    frames = lab.load_episode_sequence(output_path)
    assert output == {
        "action": "collect_habitat_mock",
        "path": str(output_path),
        "valid": True,
        "digest": lab.episode_sequence_digest(frames),
        "summary": lab.episode_sequence_summary(frames),
    }
    assert lab.validate_episode_sequence(frames)["valid"] is True
    assert lab.build_oracle_graph_from_episode(frames).get_object_state("table_1").label == "table"


def test_collect_habitat_cli_returns_structured_json_when_real_collector_is_unavailable(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_collect_script()
    main = cast(MainFn, getattr(module, "main"))
    output_path = tmp_path / "episodes" / "real-habitat.jsonl"

    assert main(
        [
            "--scene",
            "apartment_0",
            "--episode-id",
            "habitat_real_001",
            "--step",
            "1",
            "--output",
            str(output_path),
        ]
    ) == 1

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "action": "collect_habitat_episode",
        "path": str(output_path),
        "valid": False,
        "error": lab.HABITAT_MISSING_DEPENDENCY_MESSAGE,
    }
