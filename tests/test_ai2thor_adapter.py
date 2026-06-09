from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from collections.abc import Mapping
from typing import Any, Protocol, cast

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


def test_ai2thor_real_collector_uses_fake_controller_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    config = lab.AI2ThorAdapterConfig(
        scene_id="FloorPlan1",
        episode_id="ai2thor_real_smoke_001",
        steps=(1, 2),
        actions=("Initialize", "MoveAhead"),
        artifact_root=str(tmp_path / "raw"),
    )
    controller_factory = FakeControllerFactory()

    frames = lab.AI2ThorEpisodeCollector(
        config,
        ai2thor_available=True,
        controller_factory=controller_factory,
    ).collect_episode()

    controller = controller_factory.instances[0]
    assert controller.scene == "FloorPlan1"
    assert controller_factory.calls == [
        {
            "renderDepthImage": True,
            "renderInstanceSegmentation": True,
            "scene": "FloorPlan1",
        }
    ]
    assert controller.actions == ["Initialize", "MoveAhead"]
    assert [frame.step for frame in frames] == [1, 2]
    assert [frame.action for frame in frames] == ["Initialize", "MoveAhead"]
    assert frames[0].visible_object_ids == ("Mug|1",)
    assert frames[0].agent_pose == Pose3D(1.0, 0.0, 2.0, yaw=90.0)
    assert frames[0].metadata["adapter"] == "ai2thor"
    assert frames[0].metadata["source_kind"] == "real_simulator"
    assert frames[0].metadata["simulator"] == "ai2thor"
    assert frames[0].metadata["collection_kind"] == "real"
    assert frames[0].metadata["segmentation_color_map"] == [
        {"object_id": "Mug|1", "rgb": [0, 0, 255]},
        {"object_id": "Floor|1", "rgb": [255, 255, 0]},
    ]
    assert frames[0].metadata["segmentation_source"] == (
        "ai2thor_instance_segmentation_frame"
    )
    assert frames[0].metadata["objects"] == [
        {
            "bbox": {
                "center": {"x": 1.25, "y": 0.8, "z": 2.25, "yaw": 0.0},
                "size": [0.1, 0.1, 0.2],
            },
            "confidence": 1.0,
            "label": "Mug",
            "object_id": "Mug|1",
            "pose": {"x": 1.25, "y": 0.8, "z": 2.25, "yaw": 0.0},
            "states": {"pickupable": True},
            "visible": True,
        }
    ]
    assert frames[0].rgb_path == str(
        tmp_path / "raw" / "ai2thor_real_smoke_001" / "rgb" / "0001.ppm"
    )
    assert frames[0].depth_path == str(
        tmp_path / "raw" / "ai2thor_real_smoke_001" / "depth" / "0001.npy"
    )
    assert frames[0].segmentation_path == str(
        tmp_path / "raw" / "ai2thor_real_smoke_001" / "segmentation" / "0001.ppm"
    )
    assert Path(frames[0].rgb_path).read_text(encoding="utf-8").startswith(
        "P3\n2 1\n255\n"
    )
    assert json.loads(Path(frames[0].depth_path).read_text(encoding="utf-8")) == [
        [1.0, 1.1]
    ]
    assert Path(frames[0].segmentation_path).exists()
    assert controller.stopped is True


def test_ai2thor_real_collector_marks_missing_segmentation_color_map(
    tmp_path: Path,
) -> None:
    config = lab.AI2ThorAdapterConfig(
        scene_id="FloorPlan1",
        episode_id="ai2thor_real_smoke_001",
        steps=(1,),
        actions=("Initialize",),
        artifact_root=str(tmp_path / "raw"),
    )

    frames = lab.AI2ThorEpisodeCollector(
        config,
        ai2thor_available=True,
        controller_factory=NoColorMapControllerFactory(),
    ).collect_episode()

    assert frames[0].metadata["segmentation_color_map"] == []
    assert frames[0].metadata["segmentation_color_map_available"] is False
    assert frames[0].metadata["segmentation_color_map_unavailable_reason"] == (
        "ai2thor_event_color_maps_empty"
    )
    assert frames[0].segmentation_path is not None
    assert Path(frames[0].segmentation_path).exists()


def test_ai2thor_real_collector_writes_array_like_segmentation_frame(
    tmp_path: Path,
) -> None:
    config = lab.AI2ThorAdapterConfig(
        scene_id="FloorPlan1",
        episode_id="ai2thor_real_smoke_001",
        steps=(1,),
        actions=("Initialize",),
        artifact_root=str(tmp_path / "raw"),
    )

    frames = lab.AI2ThorEpisodeCollector(
        config,
        ai2thor_available=True,
        controller_factory=ArrayLikeControllerFactory(),
    ).collect_episode()

    assert frames[0].segmentation_path is not None
    assert Path(frames[0].segmentation_path).read_text(encoding="utf-8").startswith(
        "P3\n2 1\n255\n"
    )


def test_ai2thor_real_collector_rejects_missing_artifact_root() -> None:
    config = lab.AI2ThorAdapterConfig(
        scene_id="FloorPlan1",
        episode_id="ai2thor_real_smoke_001",
        steps=(1,),
        actions=("Initialize",),
    )

    with pytest.raises(
        SpatialQAError,
        match="AI2-THOR real collection requires artifact_root",
    ):
        lab.AI2ThorEpisodeCollector(
            config,
            ai2thor_available=True,
            controller_factory=FakeControllerFactory(),
        ).collect_episode()


def test_ai2thor_real_collector_requires_explicit_action_per_step(
    tmp_path: Path,
) -> None:
    config = lab.AI2ThorAdapterConfig(
        scene_id="FloorPlan1",
        episode_id="ai2thor_real_smoke_001",
        steps=(1,),
        artifact_root=str(tmp_path / "raw"),
    )

    with pytest.raises(
        SpatialQAError,
        match="AI2-THOR real collection requires one explicit action per step",
    ):
        lab.AI2ThorEpisodeCollector(
            config,
            ai2thor_available=True,
            controller_factory=FakeControllerFactory(),
        ).collect_episode()


def test_ai2thor_real_collector_rejects_missing_event_fields(tmp_path: Path) -> None:
    config = lab.AI2ThorAdapterConfig(
        scene_id="FloorPlan1",
        episode_id="ai2thor_real_smoke_001",
        steps=(1,),
        actions=("Initialize",),
        artifact_root=str(tmp_path / "raw"),
    )

    with pytest.raises(SpatialQAError, match="AI2-THOR event metadata.agent"):
        lab.AI2ThorEpisodeCollector(
            config,
            ai2thor_available=True,
            controller_factory=BrokenControllerFactory(),
        ).collect_episode()


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


class FakeControllerFactory:
    def __init__(self) -> None:
        self.instances: list[FakeController] = []
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, scene: str, **kwargs: Any) -> "FakeController":
        self.calls.append({"scene": scene, **kwargs})
        controller = FakeController(scene)
        self.instances.append(controller)
        return controller


class FakeController:
    def __init__(self, scene: str) -> None:
        self.scene = scene
        self.actions: list[str] = []
        self.stopped = False

    def step(self, *, action: str) -> "FakeEvent":
        self.actions.append(action)
        return FakeEvent(action)

    def stop(self) -> None:
        self.stopped = True


class FakeEvent:
    def __init__(self, action: str) -> None:
        offset = 0.0 if action == "Initialize" else 0.5
        self.metadata: dict[str, Any] = {
            "agent": {
                "position": {"x": 1.0 + offset, "y": 0.0, "z": 2.0},
                "rotation": {"y": 90.0},
            },
            "objects": [
                {
                    "objectId": "Mug|1",
                    "objectType": "Mug",
                    "visible": True,
                    "position": {"x": 1.25 + offset, "y": 0.8, "z": 2.25},
                    "axisAlignedBoundingBox": {
                        "center": {"x": 1.25 + offset, "y": 0.8, "z": 2.25},
                        "size": {"x": 0.1, "y": 0.1, "z": 0.2},
                    },
                    "pickupable": True,
                }
            ],
        }
        self.frame: Any = [
            [[255, 0, 0], [0, 255, 0]],
        ]
        self.depth_frame: Any = [[1.0 + offset, 1.1 + offset]]
        self.instance_segmentation_frame: Any = [
            [[0, 0, 255], [255, 255, 0]],
        ]
        self.color_to_object_id = {
            (0, 0, 255): "Mug|1",
            (255, 255, 0): "Floor|1",
        }


class NoColorMapControllerFactory:
    def __call__(self, *, scene: str, **_kwargs: Any) -> "NoColorMapController":
        return NoColorMapController(scene)


class NoColorMapController:
    def __init__(self, scene: str) -> None:
        self.scene = scene

    def step(self, *, action: str) -> "NoColorMapEvent":
        return NoColorMapEvent()

    def stop(self) -> None:
        return None


class NoColorMapEvent(FakeEvent):
    def __init__(self) -> None:
        super().__init__("Initialize")
        del self.color_to_object_id


class BrokenControllerFactory:
    def __call__(self, *, scene: str, **_kwargs: Any) -> "BrokenController":
        return BrokenController(scene)


class BrokenController:
    def __init__(self, scene: str) -> None:
        self.scene = scene

    def step(self, *, action: str) -> "BrokenEvent":
        return BrokenEvent()


class BrokenEvent:
    metadata: Mapping[str, Any] = {"objects": []}
    frame = [[[0, 0, 0]]]
    depth_frame = [[0.0]]
    instance_segmentation_frame = [[[0, 0, 0]]]


class ArrayLikeControllerFactory:
    def __call__(self, *, scene: str, **_kwargs: Any) -> "ArrayLikeController":
        return ArrayLikeController(scene)


class ArrayLikeController:
    def __init__(self, scene: str) -> None:
        self.scene = scene

    def step(self, *, action: str) -> "ArrayLikeEvent":
        return ArrayLikeEvent(action)

    def stop(self) -> None:
        return None


class ArrayLikeEvent(FakeEvent):
    def __init__(self, action: str) -> None:
        super().__init__(action)
        self.frame = ArrayLike([[[255, 0, 0], [0, 255, 0]]])
        self.instance_segmentation_frame = ArrayLike(
            [[[0, 0, 255], [255, 255, 0]]]
        )


class ArrayLike:
    def __init__(self, value: list[list[list[int]]]) -> None:
        self._value = value

    def tolist(self) -> list[list[list[int]]]:
        return self._value
