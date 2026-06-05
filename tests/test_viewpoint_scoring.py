import pytest

from dsg_spatialqa_lab.navigation.viewpoint_scoring import (
    CoverageMemory,
    ViewpointCandidate,
    viewpoint_score,
)
from dsg_spatialqa_lab.schema import Pose3D, SpatialQAError


def test_relation_centric_score_prioritizes_same_frame_support_evidence() -> None:
    memory = CoverageMemory(
        observed_object_ids=frozenset({"apple_1"}),
        observed_support_ids=frozenset(),
        same_frame_relation_keys=frozenset(),
        current_location_edge_keys=frozenset(),
        state_evidence_object_ids=frozenset(),
        visited_view_keys=frozenset(),
    )
    target_only = viewpoint_score(
        ViewpointCandidate("target-only", Pose3D(0.0, 0.9, 0.0, yaw=0.0), pitch=0.0),
        memory,
        expected={
            "new_object_ids": ["apple_1"],
            "target_object_ids": ["apple_1"],
        },
    )
    relation_view = viewpoint_score(
        ViewpointCandidate("relation", Pose3D(0.25, 0.9, 0.0, yaw=90.0), pitch=0.0),
        memory,
        expected={
            "new_object_ids": ["countertop_1"],
            "support_object_ids": ["countertop_1"],
            "same_frame_relations": ["apple_1-ON-countertop_1"],
            "current_location_edges": ["apple_1-ON-countertop_1"],
            "bbox_depth_quality": 1.0,
        },
    )

    assert relation_view.terms["target_support_same_frame_gain"] > 0.0
    assert relation_view.score > target_only.score
    assert relation_view.why_selected == "highest relation-centric gain per travel cost"


def test_predicted_memory_only_rejects_forbidden_gold_fields() -> None:
    with pytest.raises(SpatialQAError, match="forbidden gold field"):
        viewpoint_score(
            ViewpointCandidate("leaky", Pose3D(0.0, 0.9, 0.0, yaw=0.0), pitch=0.0),
            CoverageMemory(),
            expected={"required_edges": ["apple_1-ON-countertop_1"]},
            scoring_mode="predicted_memory_only",
        )


def test_repeated_view_penalty_lowers_score() -> None:
    candidate = ViewpointCandidate("view", Pose3D(0.0, 0.9, 0.0, yaw=90.0), pitch=30.0)
    expected = {
        "new_object_ids": ["mug_1"],
        "support_object_ids": ["table_1"],
        "same_frame_relations": ["mug_1-ON-table_1"],
    }
    fresh = viewpoint_score(candidate, CoverageMemory(), expected=expected)
    repeated = viewpoint_score(
        candidate,
        CoverageMemory(visited_view_keys=frozenset({candidate.view_key})),
        expected=expected,
    )

    assert repeated.terms["repeated_view_penalty"] > fresh.terms["repeated_view_penalty"]
    assert repeated.score < fresh.score


def test_support_surface_gap_gain_and_position_revisit_penalty_drive_selection() -> None:
    seen_position = ViewpointCandidate(
        "seen-target",
        Pose3D(0.0, 0.9, 0.0, yaw=90.0),
        pitch=30.0,
    )
    unseen_support = ViewpointCandidate(
        "unseen-support",
        Pose3D(1.0, 0.9, 0.0, yaw=0.0),
        pitch=-30.0,
    )
    memory = CoverageMemory(
        observed_object_ids=frozenset({"apple_1"}),
        observed_support_ids=frozenset(),
        visited_position_keys=frozenset({seen_position.position_key}),
    )

    repeated_target = viewpoint_score(
        seen_position,
        memory,
        expected={"new_object_ids": ["apple_1"], "target_object_ids": ["apple_1"]},
    )
    support_gap = viewpoint_score(
        unseen_support,
        memory,
        expected={
            "support_object_ids": ["countertop_1"],
            "support_surface_gap_ids": ["countertop_1"],
            "same_frame_relations": ["apple_1-ON-countertop_1"],
        },
    )

    assert support_gap.terms["support_surface_gap_gain"] > 0.0
    assert repeated_target.terms["position_revisit_penalty"] > 0.0
    assert support_gap.score > repeated_target.score
