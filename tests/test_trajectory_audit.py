from dsg_spatialqa_lab.navigation.trajectory_audit import (
    compare_trajectory_protocols,
    diagnostic_protocol_metadata,
    filter_cases_for_trajectory,
    observed_node_ids_from_observations,
    trajectory_coverage_audit,
)
from dsg_spatialqa_lab.observations import NodeObservation, ObjectObservation, SceneObservation
from dsg_spatialqa_lab.schema import BBox3D, Pose3D


def test_trajectory_audit_flags_unvalidated_fixed_and_diagnostic_protocol() -> None:
    fixed = {
        "trajectory_id": "fixed",
        "collection_kind": "fixed_trajectory",
        "navigation_validated": False,
        "steps": [
            {
                "selected_viewpoint": {"x": 0.0, "z": 0.0, "yaw": 0.0, "pitch": 0.0},
                "executed_actions": [{"action": "Initialize", "success": True}],
                "capture_actions": [],
            }
        ],
    }
    audit = trajectory_coverage_audit(
        fixed,
        qa_case_count=10,
        graph_tool_semantic_match_count=1,
        target_object_ids={"apple_1", "mug_1"},
        support_object_ids={"countertop_1"},
    )

    assert audit["navigation_validated"] is False
    assert audit["trajectory_length"] == 1

    diagnostic = diagnostic_protocol_metadata()
    assert diagnostic["collection_kind"] == "coverage_diagnostic"
    assert diagnostic["not_autonomous_exploration"] is True
    assert diagnostic["not_predicted_graph_evidence"] is True


def test_comparison_report_distinguishes_fixed_diagnostic_and_reachable_nbv() -> None:
    fixed = {
        "protocol": "fixed_trajectory",
        "navigation_validated": False,
        "visited_grid_ratio": 0.1,
        "target_support_same_frame_rate": 0.1,
        "current_location_edge_acceptance_rate": 0.1,
        "evidence_observable_qa_count": 4,
        "missing_support_count": 8,
        "missing_relation_count": 9,
        "GraphTool_semantic_match": 2,
    }
    diagnostic = {
        "protocol": "coverage_diagnostic",
        "navigation_validated": False,
        "visited_grid_ratio": 0.2,
        "target_support_same_frame_rate": 0.6,
        "current_location_edge_acceptance_rate": 0.5,
        "evidence_observable_qa_count": 12,
        "missing_support_count": 3,
        "missing_relation_count": 4,
        "GraphTool_semantic_match": 8,
    }
    nbv = {
        "protocol": "reachable_relation_centric_nbv",
        "navigation_validated": True,
        "visited_grid_ratio": 0.4,
        "target_support_same_frame_rate": 0.5,
        "current_location_edge_acceptance_rate": 0.6,
        "evidence_observable_qa_count": 10,
        "missing_support_count": 4,
        "missing_relation_count": 5,
        "GraphTool_semantic_match": 7,
    }

    report = compare_trajectory_protocols(fixed, diagnostic, nbv)

    assert report["judgement"]["fixed_coverage_insufficient"] is True
    assert report["judgement"]["diagnostic_improves_evidence_coverage"] is True
    assert report["judgement"]["reachable_nbv_navigation_validated"] is True
    assert report["judgement"]["reachable_nbv_can_be_formal_protocol"] is True


def test_filter_cases_for_trajectory_uses_episode_id() -> None:
    class Case:
        def __init__(self, episode_id: str) -> None:
            self.episode_id = episode_id

    cases = [Case("episode-001"), Case("episode-002")]
    trajectory = {"episode_id": "episode-001"}

    filtered = filter_cases_for_trajectory(cases, trajectory)

    assert [case.episode_id for case in filtered] == ["episode-001"]


def test_observed_node_ids_include_rooms_regions_and_objects() -> None:
    pose = Pose3D(0.0, 0.9, 0.0, yaw=0.0)
    observations = (
        SceneObservation(
            step=1,
            rooms=(NodeObservation("ai2thor_room", "FloorPlan1"),),
            regions=(NodeObservation("visible_region", "visible"),),
            objects=(
                ObjectObservation(
                    "apple_1",
                    "apple",
                    pose,
                    BBox3D(pose, (0.1, 0.1, 0.1)),
                    confidence=1.0,
                    visible=True,
                ),
            ),
        ),
    )

    assert observed_node_ids_from_observations(observations) == {
        "ai2thor_room",
        "visible_region",
        "apple_1",
    }
