from dsg_spatialqa_lab.navigation.action_planner import (
    ReachablePosition,
    plan_reachable_path,
    reject_unreachable_candidates,
)
from dsg_spatialqa_lab.navigation.reachable_nbv import (
    CandidateObservation,
    reachable_relation_centric_nbv,
)
from dsg_spatialqa_lab.navigation.viewpoint_scoring import ViewpointCandidate
from dsg_spatialqa_lab.schema import Pose3D


def test_action_planner_outputs_continuous_actions_and_rejects_unreachable() -> None:
    reachable = (
        ReachablePosition(0.0, 0.9, 0.0),
        ReachablePosition(0.25, 0.9, 0.0),
        ReachablePosition(0.5, 0.9, 0.0),
    )
    path = plan_reachable_path(
        reachable,
        Pose3D(0.0, 0.9, 0.0, yaw=0.0),
        Pose3D(0.5, 0.9, 0.0, yaw=90.0),
        grid_step=0.25,
    )

    assert path.reachable is True
    assert path.path_positions == reachable
    assert path.actions.count("MoveAhead") == 2
    assert "RotateRight" in path.actions
    assert "TeleportFull" not in path.actions

    candidates = (
        ViewpointCandidate("ok", Pose3D(0.25, 0.9, 0.0, yaw=0.0), pitch=0.0),
        ViewpointCandidate("bad", Pose3D(5.0, 0.9, 5.0, yaw=0.0), pitch=0.0),
    )
    accepted, rejected = reject_unreachable_candidates(candidates, reachable)
    assert [candidate.candidate_id for candidate in accepted] == ["ok"]
    assert rejected == [{"candidate_id": "bad", "reason": "not_in_reachable_positions"}]


def test_reachable_nbv_produces_navigation_validated_closed_loop_trajectory() -> None:
    reachable = (
        ReachablePosition(0.0, 0.9, 0.0),
        ReachablePosition(0.25, 0.9, 0.0),
        ReachablePosition(0.5, 0.9, 0.0),
    )
    observations = {
        "0.25:0.00": CandidateObservation(
            object_ids=frozenset({"apple_1", "countertop_1"}),
            support_ids=frozenset({"countertop_1"}),
            same_frame_relations=frozenset({"apple_1-ON-countertop_1"}),
            current_location_edges=frozenset({"apple_1-ON-countertop_1"}),
            state_object_ids=frozenset({"apple_1"}),
        ),
        "0.50:0.00": CandidateObservation(
            object_ids=frozenset({"mug_1", "table_1"}),
            support_ids=frozenset({"table_1"}),
            same_frame_relations=frozenset({"mug_1-ON-table_1"}),
            current_location_edges=frozenset({"mug_1-ON-table_1"}),
            state_object_ids=frozenset({"mug_1"}),
        ),
    }

    result = reachable_relation_centric_nbv(
        scene_id="FloorPlan1",
        episode_id="episode001",
        reachable_positions=reachable,
        start_pose=Pose3D(0.0, 0.9, 0.0, yaw=0.0),
        candidate_observations=observations,
        max_iterations=2,
        yaw_sweep_degrees=(0.0, 90.0),
        pitch_sweep_degrees=(-30.0, 0.0),
    )

    assert result.trajectory["collection_kind"] == "reachable_relation_centric_nbv"
    assert result.trajectory["autonomous_exploration"] is True
    assert result.trajectory["navigation_validated"] is True
    assert result.trajectory["closed_loop_memory_update"] is True
    assert result.trajectory["teleport_used"] is False
    assert result.trajectory["uses_gold_answer"] is False
    assert result.trajectory["uses_gold_evidence"] is False
    assert result.trajectory["uses_required_edges"] is False
    assert result.trajectory["uses_required_nodes"] is False
    assert result.trajectory["all_actions_last_action_success_checked"] is True
    assert len(result.trajectory["stations"]) == 2
    assert result.trajectory["runtime_kind"] == "fake_controller"
    assert result.trajectory["real_ai2thor_runtime"] is False
    assert len(result.trajectory["steps"]) == 2
    assert all(
        action["action"] != "TeleportFull"
        for step in result.trajectory["steps"]
        for action in step["executed_actions"]
    )
    assert all("memory_before" in item and "memory_after" in item for item in result.decisions)
    assert result.observation_count > 0
