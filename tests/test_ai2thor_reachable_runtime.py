from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from dsg_spatialqa_lab.navigation.ai2thor_runtime import (
    ai2thor_event_to_observation,
    execute_ai2thor_actions,
    get_ai2thor_reachable_positions,
    real_ai2thor_candidate_priors,
)
from dsg_spatialqa_lab.navigation.reachable_nbv import (
    ExecutedViewpoint,
    reachable_relation_centric_nbv,
)
from dsg_spatialqa_lab.schema import Pose3D, SpatialQAError


def test_get_ai2thor_reachable_positions_reads_action_return() -> None:
    controller = FakeRuntimeController()

    positions, event = get_ai2thor_reachable_positions(controller)

    assert [position.to_dict() for position in positions] == [
        {"x": 0.0, "y": 0.9, "z": 0.0},
        {"x": 0.25, "y": 0.9, "z": 0.0},
    ]
    assert cast(FakeRuntimeEvent, event).metadata["lastActionSuccess"] is True
    assert controller.actions == ["GetReachablePositions"]


def test_execute_ai2thor_actions_records_last_action_success() -> None:
    controller = FakeRuntimeController(fail_action="MoveAhead")

    records, events = execute_ai2thor_actions(
        controller,
        ("RotateRight", "MoveAhead", "RotateLeft"),
    )

    assert records == (
        {"action": "RotateRight", "lastActionSuccess": True, "success": True},
        {
            "action": "MoveAhead",
            "errorMessage": "blocked",
            "lastActionSuccess": False,
            "success": False,
        },
    )
    assert len(events) == 2
    assert controller.actions == ["RotateRight", "MoveAhead"]


def test_real_ai2thor_candidate_priors_do_not_use_gold_or_hidden_objects() -> None:
    positions, _ = get_ai2thor_reachable_positions(FakeRuntimeController())

    priors = real_ai2thor_candidate_priors(positions)

    assert sorted(priors) == ["0.00:0.00", "0.25:0.00"]
    for prior in priors.values():
        payload = prior.expected_payload()
        assert payload["new_object_ids"] == []
        assert payload["support_object_ids"] == []
        assert "required_edges" not in payload
        assert "required_nodes" not in payload


def test_real_ai2thor_executor_branch_marks_navigation_failed_on_action_failure(
    tmp_path: Path,
) -> None:
    positions, _ = get_ai2thor_reachable_positions(FakeRuntimeController())

    def execute_viewpoint(
        planned_actions: tuple[str, ...],
        *_args: Any,
        **_kwargs: Any,
    ) -> ExecutedViewpoint:
        assert "TeleportFull" not in planned_actions
        return ExecutedViewpoint(
            executed_actions=(
                {
                    "action": "MoveAhead",
                    "lastActionSuccess": False,
                    "success": False,
                },
            ),
            executed_capture_actions=(),
            observations=(
                ai2thor_event_to_observation(
                    FakeRuntimeEvent("Pass"),
                    scene_id="FloorPlan1",
                    episode_id="episode001",
                    step=1,
                    artifact_root=tmp_path,
                ),
            ),
            observed_candidate=real_ai2thor_candidate_priors(positions)["0.25:0.00"],
            agent_pose_after=Pose3D(0.25, 0.9, 0.0, yaw=90.0),
        )

    result = reachable_relation_centric_nbv(
        scene_id="FloorPlan1",
        episode_id="episode001",
        reachable_positions=positions,
        start_pose=Pose3D(0.0, 0.9, 0.0, yaw=0.0),
        candidate_observations=real_ai2thor_candidate_priors(positions),
        max_iterations=1,
        yaw_sweep_degrees=(0.0, 90.0),
        pitch_sweep_degrees=(0.0,),
        runtime_kind="real_ai2thor",
        real_ai2thor_runtime=True,
        execute_viewpoint=execute_viewpoint,
    )

    assert result.trajectory["runtime_kind"] == "real_ai2thor"
    assert result.trajectory["real_ai2thor_runtime"] is True
    assert result.trajectory["navigation_validated"] is False
    assert result.trajectory["steps"][0]["executed_actions"][0]["lastActionSuccess"] is False


def test_ai2thor_event_to_observation_preserves_stable_ids_and_support_relation(
    tmp_path: Path,
) -> None:
    observation = ai2thor_event_to_observation(
        FakeRuntimeEvent("Pass"),
        scene_id="FloorPlan1",
        episode_id="episode001",
        step=7,
        artifact_root=tmp_path,
    )

    assert observation.step == 7
    ids = [obj.object_id for obj in observation.objects]
    assert ids == [
        "apple_00_47_01_15_00_48",
        "countertop_00_08_01_15_00_00",
    ]
    apple = observation.objects[0]
    assert apple.attributes["ai2thor_object_id"] == "Apple|-00.47|+01.15|+00.48"
    assert apple.attributes["current_location_id"] == "countertop_00_08_01_15_00_00"
    assert apple.attributes["current_location_relation"] == "ON"
    assert Path(str(apple.attributes["rgb_path"])).exists()


def test_ai2thor_event_to_observation_preserves_segmentation_color_evidence(
    tmp_path: Path,
) -> None:
    observation = ai2thor_event_to_observation(
        FakeRuntimeEvent("Pass"),
        scene_id="FloorPlan1",
        episode_id="episode001",
        step=7,
        artifact_root=tmp_path,
    )

    apple = observation.objects[0]
    countertop = observation.objects[1]
    assert apple.attributes["segmentation_color_rgb"] == [0, 0, 255]
    assert apple.attributes["segmentation_source"] == "ai2thor_instance_segmentation_frame"
    assert countertop.attributes["segmentation_color_rgb"] == [255, 255, 0]
    assert countertop.attributes["segmentation_source"] == (
        "ai2thor_instance_segmentation_frame"
    )


def test_ai2thor_event_to_observation_deduplicates_stable_object_ids(
    tmp_path: Path,
) -> None:
    event = FakeRuntimeEvent("Pass")
    duplicate = dict(event.metadata["objects"][0])
    event.metadata["objects"].append(duplicate)

    observation = ai2thor_event_to_observation(
        event,
        scene_id="FloorPlan1",
        episode_id="episode001",
        step=8,
        artifact_root=tmp_path,
    )

    ids = [obj.object_id for obj in observation.objects]
    assert ids.count("apple_00_47_01_15_00_48") == 1


def test_get_ai2thor_reachable_positions_requires_success() -> None:
    with pytest.raises(SpatialQAError, match="GetReachablePositions failed"):
        get_ai2thor_reachable_positions(FakeRuntimeController(fail_action="GetReachablePositions"))


class FakeRuntimeController:
    def __init__(self, *, fail_action: str | None = None) -> None:
        self.fail_action = fail_action
        self.actions: list[str] = []

    def step(self, *, action: str, **_kwargs: Any) -> "FakeRuntimeEvent":
        self.actions.append(action)
        return FakeRuntimeEvent(action, success=action != self.fail_action)


class FakeRuntimeEvent:
    def __init__(self, action: str, *, success: bool = True) -> None:
        self.metadata: dict[str, Any] = {
            "actionReturn": [
                {"x": 0.0, "y": 0.9, "z": 0.0},
                {"x": 0.25, "y": 0.9, "z": 0.0},
            ]
            if action == "GetReachablePositions"
            else None,
            "agent": {
                "position": {"x": 0.25 if action == "MoveAhead" else 0.0, "y": 0.9, "z": 0.0},
                "rotation": {"y": 90.0 if action == "RotateRight" else 0.0},
            },
            "errorMessage": "blocked" if not success else "",
            "lastAction": action,
            "lastActionSuccess": success,
            "objects": [
                {
                    "axisAlignedBoundingBox": {
                        "center": {"x": -0.47, "y": 1.15, "z": 0.48},
                        "size": {"x": 0.1, "y": 0.2, "z": 0.1},
                    },
                    "objectId": "Apple|-00.47|+01.15|+00.48",
                    "objectType": "Apple",
                    "parentReceptacles": ["CounterTop|-00.08|+01.15|00.00"],
                    "pickupable": True,
                    "position": {"x": -0.47, "y": 1.15, "z": 0.48},
                    "visible": True,
                },
                {
                    "axisAlignedBoundingBox": {
                        "center": {"x": -0.08, "y": 1.15, "z": 0.0},
                        "size": {"x": 1.0, "y": 0.1, "z": 1.0},
                    },
                    "objectId": "CounterTop|-00.08|+01.15|00.00",
                    "objectType": "CounterTop",
                    "parentReceptacles": [],
                    "pickupable": False,
                    "position": {"x": -0.08, "y": 1.15, "z": 0.0},
                    "visible": True,
                },
            ],
        }
        self.frame = [[[255, 0, 0]]]
        self.depth_frame = [[1.0]]
        self.instance_segmentation_frame = [[[0, 0, 255]]]
        self.color_to_object_id = {
            (0, 0, 255): "Apple|-00.47|+01.15|+00.48",
            (255, 255, 0): "CounterTop|-00.08|+01.15|00.00",
        }
