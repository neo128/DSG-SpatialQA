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
COLLECT_SCRIPT = ROOT / "scripts" / "collect_ai2thor.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_collect_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("collect_ai2thor_script", COLLECT_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ai2thor_adapter_imports_without_optional_dependency_and_reports_real_collection_gap() -> None:
    assert hasattr(lab, "AI2ThorAdapterConfig")
    assert hasattr(lab, "AI2ThorEpisodeCollector")
    assert hasattr(lab, "AI2THOR_MISSING_DEPENDENCY_MESSAGE")
    config = lab.AI2ThorAdapterConfig(
        scene_id="FloorPlan1",
        episode_id="ai2thor_mock_001",
        steps=(1,),
    )
    collector = lab.AI2ThorEpisodeCollector(config)

    with pytest.raises(SpatialQAError, match="AI2-THOR optional dependency is not installed"):
        collector.collect_episode()

    assert lab.AI2THOR_MISSING_DEPENDENCY_MESSAGE == (
        "AI2-THOR optional dependency is not installed. Install with .[ai2thor]."
    )


def test_mock_ai2thor_episode_is_deterministic_and_oracle_compatible() -> None:
    assert hasattr(lab, "build_mock_ai2thor_episode")
    assert hasattr(lab, "convert_ai2thor_event_to_episode_frame")
    config = lab.AI2ThorAdapterConfig(
        scene_id="FloorPlan1",
        episode_id="ai2thor_mock_001",
        steps=(1, 2),
        actions=("Initialize", "MoveAhead"),
        artifact_root="artifacts/ai2thor",
    )

    frames = lab.build_mock_ai2thor_episode(config)
    repeated_frames = lab.build_mock_ai2thor_episode(config)
    graph = lab.build_oracle_graph_from_episode(frames)
    validation = lab.validate_episode_sequence(frames)

    assert frames == repeated_frames
    assert validation["valid"] is True
    assert [frame.step for frame in frames] == [1, 2]
    assert [frame.action for frame in frames] == ["Initialize", "MoveAhead"]
    assert frames[0].rgb_path == "artifacts/ai2thor/ai2thor_mock_001/rgb/0001.png"
    assert frames[0].metadata["adapter"] == "ai2thor"
    assert frames[0].metadata["rooms"] == [{"label": "Kitchen", "room_id": "kitchen"}]
    assert frames[1].visible_object_ids == ("mug_1",)
    assert graph.get_object_state("mug_1").last_seen_step == 2
    assert graph.find_edges(src="mug_1", relation="ON")[0].dst == "table_1"

    frame = lab.convert_ai2thor_event_to_episode_frame(
        {
            "agent_pose": {"x": 0.5, "y": 0.0, "z": 0.25, "yaw": 90.0},
            "visible_object_ids": ["mug_1"],
            "metadata": {"objects": []},
        },
        config=config,
        step=7,
        action="LookDown",
    )

    assert frame.agent_pose == Pose3D(0.5, 0.0, 0.25, yaw=90.0)
    assert frame.metadata == {"adapter": "ai2thor", "objects": []}


def test_mock_ai2thor_episode_rejects_implicit_steps_and_action_mismatch() -> None:
    empty_steps_config = lab.AI2ThorAdapterConfig(
        scene_id="FloorPlan1",
        episode_id="ai2thor_mock_001",
        steps=(),
    )
    mismatched_actions_config = lab.AI2ThorAdapterConfig(
        scene_id="FloorPlan1",
        episode_id="ai2thor_mock_001",
        steps=(1, 2),
        actions=("Initialize",),
    )

    with pytest.raises(SpatialQAError, match="AI2-THOR adapter steps must be explicit"):
        lab.build_mock_ai2thor_episode(empty_steps_config)

    with pytest.raises(SpatialQAError, match="actions length must match steps length"):
        lab.build_mock_ai2thor_episode(mismatched_actions_config)


def test_collect_ai2thor_cli_writes_mock_episode_and_validates_output(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_collect_script()
    main = cast(MainFn, getattr(module, "main"))
    output_path = tmp_path / "episodes" / "mock-ai2thor.jsonl"

    assert main(
        [
            "--mock",
            "--scene",
            "FloorPlan1",
            "--episode-id",
            "ai2thor_mock_001",
            "--step",
            "1",
            "--step",
            "2",
            "--action",
            "Initialize",
            "--action",
            "MoveAhead",
            "--output",
            str(output_path),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    frames = lab.load_episode_sequence(output_path)
    assert output == {
        "action": "collect_ai2thor_mock",
        "path": str(output_path),
        "valid": True,
        "digest": lab.episode_sequence_digest(frames),
        "summary": lab.episode_sequence_summary(frames),
    }
    assert lab.validate_episode_sequence(frames)["valid"] is True
    assert lab.build_oracle_graph_from_episode(frames).get_object_state("table_1").label == "table"


def test_collect_ai2thor_cli_returns_structured_json_when_real_collector_is_unavailable(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_collect_script()
    main = cast(MainFn, getattr(module, "main"))
    output_path = tmp_path / "episodes" / "real-ai2thor.jsonl"

    assert main(
        [
            "--scene",
            "FloorPlan1",
            "--episode-id",
            "ai2thor_real_001",
            "--step",
            "1",
            "--output",
            str(output_path),
        ]
    ) == 1

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "action": "collect_ai2thor_episode",
        "path": str(output_path),
        "valid": False,
        "error": lab.AI2THOR_MISSING_DEPENDENCY_MESSAGE,
    }
