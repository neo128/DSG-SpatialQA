import hashlib
import json
from pathlib import Path

import pytest

import dsg_spatialqa_lab as lab
from dsg_spatialqa_lab import (
    BBox3D,
    DynamicSceneGraph,
    EvaluationCase,
    Pose3D,
    SpatialQAError,
    compare_evaluation_bundle,
    compare_evaluation_case_listing,
    compare_evaluation_manifest,
    compare_evaluation_report,
    evaluation_bundle,
    evaluation_bundle_json,
    evaluation_case_listing,
    evaluation_case_listing_json,
    evaluation_case_metadata,
    evaluation_manifest,
    evaluation_manifest_json,
    evaluation_report,
    evaluation_report_json,
    evaluation_cases_metadata,
    get_evaluation_case,
    list_evaluation_case_metadata,
    list_evaluation_cases,
    load_evaluation_bundle,
    load_evaluation_case_listing,
    load_evaluation_manifest,
    load_evaluation_report,
    run_evaluation_case_definition,
    run_evaluation_cases,
    run_evaluation_case,
    run_evaluation_suite,
    save_evaluation_bundle,
    save_evaluation_case_listing,
    save_evaluation_manifest,
    save_evaluation_report,
    validate_evaluation_bundle,
    validate_evaluation_case_listing,
    validate_evaluation_manifest,
)


def test_evaluation_case_registry_is_deterministic() -> None:
    assert list_evaluation_cases() == (
        "ambiguous_mug_label_candidates",
        "ambiguous_mug_pick_by_label",
        "ambiguous_plate_place_reference_by_label",
        "moved_mug_next_action_validity",
        "moved_mug_object_timeline",
        "moved_mug_recent_events",
        "moved_mug_scene_delta",
        "moved_mug_scene_delta_reversed_window_error",
        "moved_mug_stale_pick",
        "moved_mug_stale_place_plate_right_of_mug",
        "moved_mug_world_state",
        "multi_room_rearrangement_object_room_cereal_box",
        "multi_room_rearrangement_object_room_milk",
        "multi_room_rearrangement_recent_events",
        "multi_room_rearrangement_reobserve_targets",
        "multi_room_rearrangement_scene_delta",
        "needs_reobserve_bowl_pick_target_not_visible",
        "needs_reobserve_bowl_place_reference_target_not_visible",
        "needs_reobserve_bowl_place_target_not_visible",
        "needs_reobserve_cup_pick_low_confidence",
        "needs_reobserve_cup_place_reference_low_confidence",
        "needs_reobserve_cup_place_target_low_confidence",
        "needs_reobserve_spoon_label_candidates",
        "needs_reobserve_spoon_pick",
        "needs_reobserve_spoon_place_reference",
        "needs_reobserve_spoon_place_target",
        "needs_reobserve_targets",
        "relation_shift_relation_timeline",
        "tabletop_agent_history",
        "tabletop_agent_location",
        "tabletop_agent_timeline",
        "tabletop_graph_query_mug_plate",
        "tabletop_missing_label_pick_error",
        "tabletop_missing_object_location_error",
        "tabletop_missing_object_pick_error",
        "tabletop_missing_reference_input_place_error",
        "tabletop_missing_reference_place_error",
        "tabletop_missing_target_input_place_error",
        "tabletop_missing_target_pick_error",
        "tabletop_mug_pick",
        "tabletop_nearest_candidate_plate",
        "tabletop_object_history_mug",
        "tabletop_object_location",
        "tabletop_object_status_plate",
        "tabletop_place_mug_right_of_plate",
        "tabletop_relation_timeline",
        "tabletop_relative_relation_mug_left_of_plate",
        "tabletop_retrieve_subgraph_mug",
        "tabletop_scene_snapshot_invalid_step_error",
        "tabletop_scene_snapshot_step1",
        "tabletop_unsupported_place_relation_error",
        "tabletop_unsupported_question_type_error",
    )
    assert list_evaluation_cases(tags=("vla", "anchor")) == (
        "tabletop_mug_pick",
        "tabletop_place_mug_right_of_plate",
    )
    assert list_evaluation_cases(tags=("qa", "dynamic")) == (
        "moved_mug_next_action_validity",
        "moved_mug_object_timeline",
        "moved_mug_recent_events",
        "moved_mug_scene_delta",
        "moved_mug_scene_delta_reversed_window_error",
        "moved_mug_world_state",
        "multi_room_rearrangement_object_room_cereal_box",
        "multi_room_rearrangement_object_room_milk",
        "multi_room_rearrangement_recent_events",
        "multi_room_rearrangement_scene_delta",
        "relation_shift_relation_timeline",
    )
    assert list_evaluation_cases(tags=("qa", "action_validity")) == (
        "moved_mug_next_action_validity",
    )
    assert list_evaluation_cases(tags=("qa", "temporal")) == (
        "moved_mug_object_timeline",
        "moved_mug_scene_delta",
        "moved_mug_scene_delta_reversed_window_error",
        "multi_room_rearrangement_recent_events",
        "multi_room_rearrangement_scene_delta",
        "relation_shift_relation_timeline",
        "tabletop_agent_history",
        "tabletop_agent_timeline",
        "tabletop_relation_timeline",
        "tabletop_scene_snapshot_step1",
    )
    assert list_evaluation_cases(tags=("qa", "relations")) == (
        "relation_shift_relation_timeline",
        "tabletop_relation_timeline",
        "tabletop_relative_relation_mug_left_of_plate",
    )
    assert list_evaluation_cases(tags=("qa", "foundation")) == (
        "tabletop_agent_history",
        "tabletop_agent_location",
        "tabletop_missing_object_location_error",
        "tabletop_object_history_mug",
        "tabletop_object_status_plate",
        "tabletop_relative_relation_mug_left_of_plate",
    )
    assert list_evaluation_cases(tags=("qa", "error")) == (
        "moved_mug_scene_delta_reversed_window_error",
        "tabletop_missing_object_location_error",
        "tabletop_scene_snapshot_invalid_step_error",
        "tabletop_unsupported_question_type_error",
    )
    assert list_evaluation_cases(tags=("vla", "error")) == (
        "needs_reobserve_bowl_pick_target_not_visible",
        "needs_reobserve_bowl_place_reference_target_not_visible",
        "needs_reobserve_bowl_place_target_not_visible",
        "needs_reobserve_cup_pick_low_confidence",
        "needs_reobserve_cup_place_reference_low_confidence",
        "needs_reobserve_cup_place_target_low_confidence",
        "tabletop_missing_label_pick_error",
        "tabletop_missing_object_pick_error",
        "tabletop_missing_reference_input_place_error",
        "tabletop_missing_reference_place_error",
        "tabletop_missing_target_input_place_error",
        "tabletop_missing_target_pick_error",
        "tabletop_unsupported_place_relation_error",
    )
    assert list_evaluation_cases(tags=("qa", "retrieval")) == (
        "tabletop_graph_query_mug_plate",
        "tabletop_nearest_candidate_plate",
        "tabletop_retrieve_subgraph_mug",
    )
    assert list_evaluation_cases(tags=("qa", "nearest")) == (
        "tabletop_nearest_candidate_plate",
    )
    assert list_evaluation_cases(question_types=("nearest_object",)) == (
        "tabletop_nearest_candidate_plate",
    )
    assert list_evaluation_cases(question_types=("label_candidates",)) == (
        "ambiguous_mug_label_candidates",
        "needs_reobserve_spoon_label_candidates",
    )
    assert list_evaluation_cases(question_types=("agent_history",)) == (
        "tabletop_agent_history",
    )
    assert list_evaluation_cases(question_types=("agent_location",)) == (
        "tabletop_agent_location",
    )
    assert list_evaluation_cases(question_types=("object_history",)) == (
        "tabletop_object_history_mug",
    )
    assert list_evaluation_cases(question_types=("object_status",)) == (
        "tabletop_object_status_plate",
    )
    assert list_evaluation_cases(question_types=("object_location",)) == (
        "tabletop_missing_object_location_error",
        "tabletop_object_location",
    )
    assert list_evaluation_cases(question_types=("object_room",)) == (
        "multi_room_rearrangement_object_room_cereal_box",
        "multi_room_rearrangement_object_room_milk",
    )
    assert list_evaluation_cases(question_types=("relative_relation",)) == (
        "tabletop_relative_relation_mug_left_of_plate",
    )
    assert list_evaluation_cases(question_types=("retrieve_subgraph",)) == (
        "tabletop_retrieve_subgraph_mug",
    )
    assert list_evaluation_cases(question_types=("unsupported_intent",)) == (
        "tabletop_unsupported_question_type_error",
    )
    assert list_evaluation_cases(question_types=("next_action_validity",)) == (
        "moved_mug_next_action_validity",
    )
    assert list_evaluation_cases(question_types=("relation_timeline",)) == (
        "relation_shift_relation_timeline",
        "tabletop_relation_timeline",
    )
    assert list_evaluation_cases(question_types=("scene_delta",)) == (
        "moved_mug_scene_delta",
        "moved_mug_scene_delta_reversed_window_error",
        "multi_room_rearrangement_scene_delta",
    )
    assert list_evaluation_cases(question_types=("scene_snapshot",)) == (
        "tabletop_scene_snapshot_invalid_step_error",
        "tabletop_scene_snapshot_step1",
    )
    assert list_evaluation_cases(tags=("qa", "snapshot")) == (
        "tabletop_scene_snapshot_invalid_step_error",
        "tabletop_scene_snapshot_step1",
    )
    assert list_evaluation_cases(tags=("qa", "reobserve")) == (
        "multi_room_rearrangement_reobserve_targets",
        "needs_reobserve_spoon_label_candidates",
        "needs_reobserve_targets",
    )
    assert list_evaluation_cases(tags=("qa", "label", "ambiguity")) == (
        "ambiguous_mug_label_candidates",
    )
    assert list_evaluation_cases(kinds=("vla_pick", "vla_place_relative")) == (
        "ambiguous_mug_pick_by_label",
        "ambiguous_plate_place_reference_by_label",
        "needs_reobserve_bowl_pick_target_not_visible",
        "needs_reobserve_bowl_place_reference_target_not_visible",
        "needs_reobserve_bowl_place_target_not_visible",
        "needs_reobserve_cup_pick_low_confidence",
        "needs_reobserve_cup_place_reference_low_confidence",
        "needs_reobserve_cup_place_target_low_confidence",
        "needs_reobserve_spoon_pick",
        "needs_reobserve_spoon_place_reference",
        "needs_reobserve_spoon_place_target",
        "tabletop_missing_label_pick_error",
        "tabletop_missing_object_pick_error",
        "tabletop_missing_reference_input_place_error",
        "tabletop_missing_reference_place_error",
        "tabletop_missing_target_input_place_error",
        "tabletop_missing_target_pick_error",
        "tabletop_mug_pick",
        "tabletop_place_mug_right_of_plate",
        "tabletop_unsupported_place_relation_error",
    )
    assert list_evaluation_cases(tags=("vla",), kinds=("vla_stale_pick",)) == (
        "moved_mug_stale_pick",
    )
    assert list_evaluation_cases(
        tags=("vla",),
        kinds=("vla_stale_pick", "vla_stale_place_relative"),
    ) == (
        "moved_mug_stale_pick",
        "moved_mug_stale_place_plate_right_of_mug",
    )
    assert list_evaluation_cases(tags=("vla", "dynamic")) == (
        "moved_mug_stale_pick",
        "moved_mug_stale_place_plate_right_of_mug",
    )
    assert list_evaluation_cases(tags=("vla", "reobserve")) == (
        "needs_reobserve_spoon_pick",
        "needs_reobserve_spoon_place_reference",
        "needs_reobserve_spoon_place_target",
    )
    assert list_evaluation_cases(tags=("vla", "label", "ambiguity")) == (
        "ambiguous_mug_pick_by_label",
        "ambiguous_plate_place_reference_by_label",
    )
    assert list_evaluation_cases(tags=("vla", "label", "ambiguity", "place")) == (
        "ambiguous_plate_place_reference_by_label",
    )

    case = get_evaluation_case("tabletop_object_location")

    assert case.name == "tabletop_object_location"
    assert case.scene_fixture == "tabletop"
    assert case.kind == "qa"
    assert case.tags == ("qa", "memory")


def test_evaluation_case_registry_filters_by_explicit_names_in_call_order() -> None:
    names = (
        "tabletop_object_location",
        "moved_mug_recent_events",
        "tabletop_mug_pick",
    )

    assert list_evaluation_cases(names=names) == names
    assert list_evaluation_cases(names=names, tags=("qa", "dynamic")) == (
        "moved_mug_recent_events",
    )
    assert list_evaluation_cases(names=names, kinds=("vla_pick",)) == (
        "tabletop_mug_pick",
    )
    assert list_evaluation_cases(names=names, question_types=("object_location",)) == (
        "tabletop_object_location",
    )


def test_evaluation_case_registry_rejects_unknown_explicit_name() -> None:
    with pytest.raises(SpatialQAError, match="Unknown evaluation case: missing_case"):
        list_evaluation_cases(names=("missing_case",))


def test_list_evaluation_case_metadata_returns_deterministic_manifest() -> None:
    manifest = list_evaluation_case_metadata(kinds=("vla_pick", "vla_place_relative"))

    assert manifest == (
        {
            "name": "ambiguous_mug_pick_by_label",
            "scene_fixture": "ambiguous_mugs",
            "scene_description": "Static tabletop scene with two visible mugs sharing one label.",
            "scene_tags": ["static", "tabletop", "ambiguity"],
            "kind": "vla_pick",
            "tags": ["vla", "label", "ambiguity"],
            "question": {},
            "question_type": None,
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": None,
            "target_label": "mug",
            "reference_object": None,
            "reference_label": None,
            "relation": None,
            "expected_keys": ["ambiguous_ids", "details", "error", "error_category", "status"],
        },
        {
            "name": "ambiguous_plate_place_reference_by_label",
            "scene_fixture": "ambiguous_plates",
            "scene_description": "Static tabletop scene with two visible plates sharing one label.",
            "scene_tags": ["static", "tabletop", "ambiguity"],
            "kind": "vla_place_relative",
            "tags": ["vla", "label", "ambiguity", "place"],
            "question": {},
            "question_type": None,
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": "mug_1",
            "target_label": None,
            "reference_object": None,
            "reference_label": "plate",
            "relation": "RIGHT_OF",
            "expected_keys": ["ambiguous_ids", "details", "error", "error_category", "status"],
        },
        {
            "name": "needs_reobserve_bowl_pick_target_not_visible",
            "scene_fixture": "needs_reobserve",
            "scene_description": (
                "Tabletop scene with invisible and low-confidence objects for re-observation checks."
            ),
            "scene_tags": ["static", "tabletop", "reobserve"],
            "kind": "vla_pick",
            "tags": ["vla", "pick", "error", "occlusion", "visibility"],
            "question": {},
            "question_type": None,
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": "bowl_1",
            "target_label": None,
            "reference_object": None,
            "reference_label": None,
            "relation": None,
            "expected_keys": [
                "command",
                "details",
                "error",
                "error_category",
                "needs_reobserve",
                "needs_replan",
                "status",
            ],
        },
        {
            "name": "needs_reobserve_bowl_place_reference_target_not_visible",
            "scene_fixture": "needs_reobserve",
            "scene_description": (
                "Tabletop scene with invisible and low-confidence objects for re-observation checks."
            ),
            "scene_tags": ["static", "tabletop", "reobserve"],
            "kind": "vla_place_relative",
            "tags": ["vla", "place", "error", "occlusion", "visibility"],
            "question": {},
            "question_type": None,
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": "mug_1",
            "target_label": None,
            "reference_object": "bowl_1",
            "reference_label": None,
            "relation": "RIGHT_OF",
            "expected_keys": [
                "command",
                "details",
                "error",
                "error_category",
                "needs_reobserve",
                "needs_replan",
                "status",
            ],
        },
        {
            "name": "needs_reobserve_bowl_place_target_not_visible",
            "scene_fixture": "needs_reobserve",
            "scene_description": (
                "Tabletop scene with invisible and low-confidence objects for re-observation checks."
            ),
            "scene_tags": ["static", "tabletop", "reobserve"],
            "kind": "vla_place_relative",
            "tags": ["vla", "place", "error", "occlusion", "visibility"],
            "question": {},
            "question_type": None,
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": "bowl_1",
            "target_label": None,
            "reference_object": "plate_1",
            "reference_label": None,
            "relation": "RIGHT_OF",
            "expected_keys": [
                "command",
                "details",
                "error",
                "error_category",
                "needs_reobserve",
                "needs_replan",
                "status",
            ],
        },
        {
            "name": "needs_reobserve_cup_pick_low_confidence",
            "scene_fixture": "needs_reobserve",
            "scene_description": (
                "Tabletop scene with invisible and low-confidence objects for re-observation checks."
            ),
            "scene_tags": ["static", "tabletop", "reobserve"],
            "kind": "vla_pick",
            "tags": ["vla", "pick", "error", "confidence"],
            "question": {},
            "question_type": None,
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": "cup_1",
            "target_label": None,
            "reference_object": None,
            "reference_label": None,
            "relation": None,
            "expected_keys": [
                "command",
                "details",
                "error",
                "error_category",
                "needs_reobserve",
                "needs_replan",
                "status",
            ],
        },
        {
            "name": "needs_reobserve_cup_place_reference_low_confidence",
            "scene_fixture": "needs_reobserve",
            "scene_description": (
                "Tabletop scene with invisible and low-confidence objects for re-observation checks."
            ),
            "scene_tags": ["static", "tabletop", "reobserve"],
            "kind": "vla_place_relative",
            "tags": ["vla", "place", "error", "confidence"],
            "question": {},
            "question_type": None,
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": "mug_1",
            "target_label": None,
            "reference_object": "cup_1",
            "reference_label": None,
            "relation": "RIGHT_OF",
            "expected_keys": [
                "command",
                "details",
                "error",
                "error_category",
                "needs_reobserve",
                "needs_replan",
                "status",
            ],
        },
        {
            "name": "needs_reobserve_cup_place_target_low_confidence",
            "scene_fixture": "needs_reobserve",
            "scene_description": (
                "Tabletop scene with invisible and low-confidence objects for re-observation checks."
            ),
            "scene_tags": ["static", "tabletop", "reobserve"],
            "kind": "vla_place_relative",
            "tags": ["vla", "place", "error", "confidence"],
            "question": {},
            "question_type": None,
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": "cup_1",
            "target_label": None,
            "reference_object": "plate_1",
            "reference_label": None,
            "relation": "RIGHT_OF",
            "expected_keys": [
                "command",
                "details",
                "error",
                "error_category",
                "needs_reobserve",
                "needs_replan",
                "status",
            ],
        },
        {
            "name": "needs_reobserve_spoon_pick",
            "scene_fixture": "needs_reobserve",
            "scene_description": (
                "Tabletop scene with invisible and low-confidence objects for re-observation checks."
            ),
            "scene_tags": ["static", "tabletop", "reobserve"],
            "kind": "vla_pick",
            "tags": ["vla", "pick", "reobserve"],
            "question": {},
            "question_type": None,
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": "spoon_1",
            "target_label": None,
            "reference_object": None,
            "reference_label": None,
            "relation": None,
            "expected_keys": ["error", "error_category", "needs_reobserve", "status"],
        },
        {
            "name": "needs_reobserve_spoon_place_reference",
            "scene_fixture": "needs_reobserve",
            "scene_description": (
                "Tabletop scene with invisible and low-confidence objects for re-observation checks."
            ),
            "scene_tags": ["static", "tabletop", "reobserve"],
            "kind": "vla_place_relative",
            "tags": ["vla", "place", "reobserve", "occlusion"],
            "question": {},
            "question_type": None,
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": "mug_1",
            "target_label": None,
            "reference_object": "spoon_1",
            "reference_label": None,
            "relation": "RIGHT_OF",
            "expected_keys": ["error", "error_category", "needs_reobserve", "status"],
        },
        {
            "name": "needs_reobserve_spoon_place_target",
            "scene_fixture": "needs_reobserve",
            "scene_description": (
                "Tabletop scene with invisible and low-confidence objects for re-observation checks."
            ),
            "scene_tags": ["static", "tabletop", "reobserve"],
            "kind": "vla_place_relative",
            "tags": ["vla", "place", "reobserve", "occlusion"],
            "question": {},
            "question_type": None,
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": "spoon_1",
            "target_label": None,
            "reference_object": "plate_1",
            "reference_label": None,
            "relation": "RIGHT_OF",
            "expected_keys": ["error", "error_category", "needs_reobserve", "status"],
        },
        {
            "name": "tabletop_missing_label_pick_error",
            "scene_fixture": "tabletop",
            "scene_description": "Static tabletop scene with mug, plate, table, room, and agent.",
            "scene_tags": ["static", "tabletop"],
            "kind": "vla_pick",
            "tags": ["vla", "pick", "error", "label"],
            "question": {},
            "question_type": None,
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": None,
            "target_label": "fork",
            "reference_object": None,
            "reference_label": None,
            "relation": None,
            "expected_keys": [
                "ambiguous_ids",
                "command",
                "details",
                "error",
                "error_category",
                "needs_reobserve",
                "needs_replan",
                "status",
            ],
        },
        {
            "name": "tabletop_missing_object_pick_error",
            "scene_fixture": "tabletop",
            "scene_description": "Static tabletop scene with mug, plate, table, room, and agent.",
            "scene_tags": ["static", "tabletop"],
            "kind": "vla_pick",
            "tags": ["vla", "pick", "error"],
            "question": {},
            "question_type": None,
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": "missing_object",
            "target_label": None,
            "reference_object": None,
            "reference_label": None,
            "relation": None,
            "expected_keys": [
                "ambiguous_ids",
                "command",
                "details",
                "error",
                "error_category",
                "needs_reobserve",
                "needs_replan",
                "status",
            ],
        },
        {
            "name": "tabletop_missing_reference_input_place_error",
            "scene_fixture": "tabletop",
            "scene_description": "Static tabletop scene with mug, plate, table, room, and agent.",
            "scene_tags": ["static", "tabletop"],
            "kind": "vla_place_relative",
            "tags": ["vla", "place", "error"],
            "question": {},
            "question_type": None,
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": "mug_1",
            "target_label": None,
            "reference_object": None,
            "reference_label": None,
            "relation": "RIGHT_OF",
            "expected_keys": [
                "ambiguous_ids",
                "command",
                "details",
                "error",
                "error_category",
                "needs_reobserve",
                "needs_replan",
                "status",
            ],
        },
        {
            "name": "tabletop_missing_reference_place_error",
            "scene_fixture": "tabletop",
            "scene_description": "Static tabletop scene with mug, plate, table, room, and agent.",
            "scene_tags": ["static", "tabletop"],
            "kind": "vla_place_relative",
            "tags": ["vla", "place", "error"],
            "question": {},
            "question_type": None,
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": "mug_1",
            "target_label": None,
            "reference_object": "missing_object",
            "reference_label": None,
            "relation": "RIGHT_OF",
            "expected_keys": [
                "ambiguous_ids",
                "command",
                "details",
                "error",
                "error_category",
                "needs_reobserve",
                "needs_replan",
                "status",
            ],
        },
        {
            "name": "tabletop_missing_target_input_place_error",
            "scene_fixture": "tabletop",
            "scene_description": "Static tabletop scene with mug, plate, table, room, and agent.",
            "scene_tags": ["static", "tabletop"],
            "kind": "vla_place_relative",
            "tags": ["vla", "place", "error"],
            "question": {},
            "question_type": None,
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": None,
            "target_label": None,
            "reference_object": "plate_1",
            "reference_label": None,
            "relation": "RIGHT_OF",
            "expected_keys": [
                "ambiguous_ids",
                "command",
                "details",
                "error",
                "error_category",
                "needs_reobserve",
                "needs_replan",
                "status",
            ],
        },
        {
            "name": "tabletop_missing_target_pick_error",
            "scene_fixture": "tabletop",
            "scene_description": "Static tabletop scene with mug, plate, table, room, and agent.",
            "scene_tags": ["static", "tabletop"],
            "kind": "vla_pick",
            "tags": ["vla", "pick", "error"],
            "question": {},
            "question_type": None,
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": None,
            "target_label": None,
            "reference_object": None,
            "reference_label": None,
            "relation": None,
            "expected_keys": [
                "ambiguous_ids",
                "command",
                "details",
                "error",
                "error_category",
                "needs_reobserve",
                "needs_replan",
                "status",
            ],
        },
        {
            "name": "tabletop_mug_pick",
            "scene_fixture": "tabletop",
            "scene_description": "Static tabletop scene with mug, plate, table, room, and agent.",
            "scene_tags": ["static", "tabletop"],
            "kind": "vla_pick",
            "tags": ["vla", "anchor", "pick"],
            "question": {},
            "question_type": None,
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": "mug_1",
            "target_label": None,
            "reference_object": None,
            "reference_label": None,
            "relation": None,
            "expected_keys": ["command", "status"],
        },
        {
            "name": "tabletop_place_mug_right_of_plate",
            "scene_fixture": "tabletop",
            "scene_description": "Static tabletop scene with mug, plate, table, room, and agent.",
            "scene_tags": ["static", "tabletop"],
            "kind": "vla_place_relative",
            "tags": ["vla", "anchor", "place"],
            "question": {},
            "question_type": None,
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": "mug_1",
            "target_label": None,
            "reference_object": "plate_1",
            "reference_label": None,
            "relation": "RIGHT_OF",
            "expected_keys": ["command", "status"],
        },
        {
            "name": "tabletop_unsupported_place_relation_error",
            "scene_fixture": "tabletop",
            "scene_description": "Static tabletop scene with mug, plate, table, room, and agent.",
            "scene_tags": ["static", "tabletop"],
            "kind": "vla_place_relative",
            "tags": ["vla", "place", "error"],
            "question": {},
            "question_type": None,
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": "mug_1",
            "target_label": None,
            "reference_object": "plate_1",
            "reference_label": None,
            "relation": "ABOVE",
            "expected_keys": [
                "ambiguous_ids",
                "command",
                "details",
                "error",
                "error_category",
                "needs_reobserve",
                "needs_replan",
                "status",
            ],
        },
    )


def test_evaluation_case_metadata_supports_custom_cases() -> None:
    case = EvaluationCase(
        name="custom_plate_status",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "custom"),
        question={"type": "object_status", "object_id": "plate_1"},
        expected={"answer": {"object_id": "plate_1"}, "error": None},
    )

    assert evaluation_case_metadata(case) == {
        "name": "custom_plate_status",
        "scene_fixture": "tabletop",
        "scene_description": "Static tabletop scene with mug, plate, table, room, and agent.",
        "scene_tags": ["static", "tabletop"],
        "kind": "qa",
        "tags": ["qa", "custom"],
        "question": {"type": "object_status", "object_id": "plate_1"},
        "question_type": "object_status",
        "baseline_scene_fixture": None,
        "baseline_scene_description": None,
        "baseline_scene_tags": [],
        "target_object": None,
        "target_label": None,
        "reference_object": None,
        "reference_label": None,
        "relation": None,
        "expected_keys": ["answer", "error"],
    }


def test_evaluation_case_metadata_allows_unknown_custom_scene_fixture() -> None:
    case = EvaluationCase(
        name="custom_cube_status",
        scene_fixture="custom_counter",
        kind="qa",
        tags=("qa", "custom"),
        question={"type": "object_status", "object_id": "cube_1"},
        expected={"answer": {"object_id": "cube_1"}, "error": None},
    )

    assert evaluation_case_metadata(case)["scene_description"] is None
    assert evaluation_case_metadata(case)["scene_tags"] == []


def test_evaluation_case_metadata_question_is_a_copy() -> None:
    case = EvaluationCase(
        name="custom_plate_status",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "custom"),
        question={"type": "object_status", "object_id": "plate_1"},
        expected={"answer": {"object_id": "plate_1"}, "error": None},
    )

    metadata = evaluation_case_metadata(case)
    metadata["question"]["object_id"] = "mutated"

    assert case.question == {"type": "object_status", "object_id": "plate_1"}
    assert evaluation_case_metadata(case)["question"] == {
        "type": "object_status",
        "object_id": "plate_1",
    }


def test_evaluation_case_metadata_includes_baseline_scene_fixture_metadata() -> None:
    metadata = evaluation_case_metadata(get_evaluation_case("moved_mug_stale_pick"))

    assert metadata["scene_fixture"] == "moved_mug"
    assert metadata["scene_description"] == (
        "Dynamic tabletop scene where mug_1 moves from table_1 to sink_region."
    )
    assert metadata["scene_tags"] == ["dynamic", "tabletop", "move"]
    assert metadata["baseline_scene_fixture"] == "tabletop"
    assert metadata["baseline_scene_description"] == (
        "Static tabletop scene with mug, plate, table, room, and agent."
    )
    assert metadata["baseline_scene_tags"] == ["static", "tabletop"]


def test_stale_place_evaluation_case_metadata_is_discoverable() -> None:
    metadata = evaluation_case_metadata(
        get_evaluation_case("moved_mug_stale_place_plate_right_of_mug")
    )

    assert metadata == {
        "name": "moved_mug_stale_place_plate_right_of_mug",
        "scene_fixture": "moved_mug",
        "scene_description": (
            "Dynamic tabletop scene where mug_1 moves from table_1 to sink_region."
        ),
        "scene_tags": ["dynamic", "tabletop", "move"],
        "kind": "vla_stale_place_relative",
        "tags": ["vla", "dynamic", "place"],
        "question": {},
        "question_type": None,
        "baseline_scene_fixture": "tabletop",
        "baseline_scene_description": (
            "Static tabletop scene with mug, plate, table, room, and agent."
        ),
        "baseline_scene_tags": ["static", "tabletop"],
        "target_object": "plate_1",
        "target_label": None,
        "reference_object": "mug_1",
        "reference_label": None,
        "relation": "RIGHT_OF",
        "expected_keys": ["error", "error_category", "status"],
    }


def test_next_action_validity_evaluation_case_metadata_is_discoverable() -> None:
    manifest = list_evaluation_case_metadata(tags=("qa", "action_validity"))

    assert manifest == (
        {
            "name": "moved_mug_next_action_validity",
            "scene_fixture": "moved_mug",
            "scene_description": (
                "Dynamic tabletop scene where mug_1 moves from table_1 to sink_region."
            ),
            "scene_tags": ["dynamic", "tabletop", "move"],
            "kind": "qa",
            "tags": ["qa", "dynamic", "action_validity"],
            "question": {"type": "next_action_validity"},
            "question_type": "next_action_validity",
            "baseline_scene_fixture": "tabletop",
            "baseline_scene_description": (
                "Static tabletop scene with mug, plate, table, room, and agent."
            ),
            "baseline_scene_tags": ["static", "tabletop"],
            "target_object": "mug_1",
            "target_label": None,
            "reference_object": None,
            "reference_label": None,
            "relation": None,
            "expected_keys": ["answer", "error", "evidence_edges"],
        },
    )


def test_timeline_evaluation_case_metadata_is_discoverable() -> None:
    manifest = list_evaluation_case_metadata(tags=("qa", "temporal"))

    assert [item["name"] for item in manifest] == [
        "moved_mug_object_timeline",
        "moved_mug_scene_delta",
        "moved_mug_scene_delta_reversed_window_error",
        "multi_room_rearrangement_recent_events",
        "multi_room_rearrangement_scene_delta",
        "relation_shift_relation_timeline",
        "tabletop_agent_history",
        "tabletop_agent_timeline",
        "tabletop_relation_timeline",
        "tabletop_scene_snapshot_step1",
    ]
    assert [item["question_type"] for item in manifest] == [
        "object_timeline",
        "scene_delta",
        "scene_delta",
        "recent_events",
        "scene_delta",
        "relation_timeline",
        "agent_history",
        "agent_timeline",
        "relation_timeline",
        "scene_snapshot",
    ]
    assert manifest[0]["question"] == {"type": "object_timeline", "object_id": "mug_1"}
    assert manifest[2]["question"] == {
        "type": "scene_delta",
        "from_step": 2,
        "to_step": 1,
    }
    assert manifest[5]["question"] == {
        "type": "relation_timeline",
        "src": "mug_1",
        "dst": "plate_1",
        "reference_frame": "agent",
    }
    assert manifest[6]["question"] == {"type": "agent_history"}
    assert manifest[7]["question"] == {"type": "agent_timeline"}
    assert manifest[8]["question"] == {
        "type": "relation_timeline",
        "src": "mug_1",
        "dst": "plate_1",
        "reference_frame": "agent",
    }
    assert manifest[9]["question"] == {
        "type": "scene_snapshot",
        "step": 1,
        "visible": True,
    }


def test_reobserve_evaluation_case_metadata_is_discoverable() -> None:
    manifest = list_evaluation_case_metadata(tags=("qa", "reobserve"))

    assert manifest == (
        {
            "name": "multi_room_rearrangement_reobserve_targets",
            "scene_fixture": "multi_room_rearrangement",
            "scene_description": (
                "Dynamic kitchen-to-pantry scene with relocated cereal and an occluded fork."
            ),
            "scene_tags": ["dynamic", "multi_room", "move", "occlusion", "reobserve"],
            "kind": "qa",
            "tags": ["qa", "memory", "multi_room", "occlusion", "reobserve"],
            "question": {"type": "reobserve_targets", "label": "fork"},
            "question_type": "reobserve_targets",
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": None,
            "target_label": None,
            "reference_object": None,
            "reference_label": None,
            "relation": None,
            "expected_keys": ["answer", "error", "evidence_edges", "needs_reobserve"],
        },
        {
            "name": "needs_reobserve_spoon_label_candidates",
            "scene_fixture": "needs_reobserve",
            "scene_description": (
                "Tabletop scene with invisible and low-confidence objects for re-observation checks."
            ),
            "scene_tags": ["static", "tabletop", "reobserve"],
            "kind": "qa",
            "tags": ["qa", "label", "reobserve", "occlusion"],
            "question": {"type": "label_candidates", "label": "spoon", "visible": False},
            "question_type": "label_candidates",
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": None,
            "target_label": None,
            "reference_object": None,
            "reference_label": None,
            "relation": None,
            "expected_keys": [
                "answer",
                "confidence",
                "error",
                "evidence_edges",
                "evidence_nodes",
                "needs_reobserve",
            ],
        },
        {
            "name": "needs_reobserve_targets",
            "scene_fixture": "needs_reobserve",
            "scene_description": (
                "Tabletop scene with invisible and low-confidence objects for re-observation checks."
            ),
            "scene_tags": ["static", "tabletop", "reobserve"],
            "kind": "qa",
            "tags": ["qa", "memory", "reobserve"],
            "question": {"type": "reobserve_targets"},
            "question_type": "reobserve_targets",
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": None,
            "target_label": None,
            "reference_object": None,
            "reference_label": None,
            "relation": None,
            "expected_keys": ["answer", "error", "evidence_edges", "needs_reobserve"],
        },
    )


def test_vla_reobserve_evaluation_case_metadata_is_discoverable() -> None:
    manifest = list_evaluation_case_metadata(tags=("vla", "reobserve"))

    assert manifest == (
        {
            "name": "needs_reobserve_spoon_pick",
            "scene_fixture": "needs_reobserve",
            "scene_description": (
                "Tabletop scene with invisible and low-confidence objects for re-observation checks."
            ),
            "scene_tags": ["static", "tabletop", "reobserve"],
            "kind": "vla_pick",
            "tags": ["vla", "pick", "reobserve"],
            "question": {},
            "question_type": None,
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": "spoon_1",
            "target_label": None,
            "reference_object": None,
            "reference_label": None,
            "relation": None,
            "expected_keys": ["error", "error_category", "needs_reobserve", "status"],
        },
        {
            "name": "needs_reobserve_spoon_place_reference",
            "scene_fixture": "needs_reobserve",
            "scene_description": (
                "Tabletop scene with invisible and low-confidence objects for re-observation checks."
            ),
            "scene_tags": ["static", "tabletop", "reobserve"],
            "kind": "vla_place_relative",
            "tags": ["vla", "place", "reobserve", "occlusion"],
            "question": {},
            "question_type": None,
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": "mug_1",
            "target_label": None,
            "reference_object": "spoon_1",
            "reference_label": None,
            "relation": "RIGHT_OF",
            "expected_keys": ["error", "error_category", "needs_reobserve", "status"],
        },
        {
            "name": "needs_reobserve_spoon_place_target",
            "scene_fixture": "needs_reobserve",
            "scene_description": (
                "Tabletop scene with invisible and low-confidence objects for re-observation checks."
            ),
            "scene_tags": ["static", "tabletop", "reobserve"],
            "kind": "vla_place_relative",
            "tags": ["vla", "place", "reobserve", "occlusion"],
            "question": {},
            "question_type": None,
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": "spoon_1",
            "target_label": None,
            "reference_object": "plate_1",
            "reference_label": None,
            "relation": "RIGHT_OF",
            "expected_keys": ["error", "error_category", "needs_reobserve", "status"],
        },
    )


def test_vla_ambiguity_evaluation_case_metadata_is_discoverable() -> None:
    manifest = list_evaluation_case_metadata(tags=("vla", "label", "ambiguity"))

    assert manifest == (
        {
            "name": "ambiguous_mug_pick_by_label",
            "scene_fixture": "ambiguous_mugs",
            "scene_description": "Static tabletop scene with two visible mugs sharing one label.",
            "scene_tags": ["static", "tabletop", "ambiguity"],
            "kind": "vla_pick",
            "tags": ["vla", "label", "ambiguity"],
            "question": {},
            "question_type": None,
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": None,
            "target_label": "mug",
            "reference_object": None,
            "reference_label": None,
            "relation": None,
            "expected_keys": ["ambiguous_ids", "details", "error", "error_category", "status"],
        },
        {
            "name": "ambiguous_plate_place_reference_by_label",
            "scene_fixture": "ambiguous_plates",
            "scene_description": "Static tabletop scene with two visible plates sharing one label.",
            "scene_tags": ["static", "tabletop", "ambiguity"],
            "kind": "vla_place_relative",
            "tags": ["vla", "label", "ambiguity", "place"],
            "question": {},
            "question_type": None,
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": "mug_1",
            "target_label": None,
            "reference_object": None,
            "reference_label": "plate",
            "relation": "RIGHT_OF",
            "expected_keys": ["ambiguous_ids", "details", "error", "error_category", "status"],
        },
    )


def test_qa_label_ambiguity_evaluation_case_metadata_is_discoverable() -> None:
    manifest = list_evaluation_case_metadata(tags=("qa", "label", "ambiguity"))

    assert manifest == (
        {
            "name": "ambiguous_mug_label_candidates",
            "scene_fixture": "ambiguous_mugs",
            "scene_description": "Static tabletop scene with two visible mugs sharing one label.",
            "scene_tags": ["static", "tabletop", "ambiguity"],
            "kind": "qa",
            "tags": ["qa", "label", "ambiguity"],
            "question": {"type": "label_candidates", "label": "mug", "visible": True},
            "question_type": "label_candidates",
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": None,
            "target_label": None,
            "reference_object": None,
            "reference_label": None,
            "relation": None,
            "expected_keys": [
                "answer",
                "confidence",
                "error",
                "evidence_edges",
                "evidence_nodes",
                "needs_reobserve",
            ],
        },
    )


def test_world_state_evaluation_case_metadata_is_discoverable() -> None:
    manifest = list_evaluation_case_metadata(tags=("qa", "world_state"))

    assert manifest == (
        {
            "name": "moved_mug_world_state",
            "scene_fixture": "moved_mug",
            "scene_description": (
                "Dynamic tabletop scene where mug_1 moves from table_1 to sink_region."
            ),
            "scene_tags": ["dynamic", "tabletop", "move"],
            "kind": "qa",
            "tags": ["qa", "dynamic", "world_state"],
            "question": {"type": "world_state", "visible": True},
            "question_type": "world_state",
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": None,
            "target_label": None,
            "reference_object": None,
            "reference_label": None,
            "relation": None,
            "expected_keys": ["answer", "error", "evidence_edges", "evidence_nodes"],
        },
    )


def test_foundation_evaluation_case_metadata_is_discoverable() -> None:
    manifest = list_evaluation_case_metadata(tags=("qa", "foundation"))

    assert [item["name"] for item in manifest] == [
        "tabletop_agent_history",
        "tabletop_agent_location",
        "tabletop_missing_object_location_error",
        "tabletop_object_history_mug",
        "tabletop_object_status_plate",
        "tabletop_relative_relation_mug_left_of_plate",
    ]
    assert [item["question_type"] for item in manifest] == [
        "agent_history",
        "agent_location",
        "object_location",
        "object_history",
        "object_status",
        "relative_relation",
    ]
    assert [item["expected_keys"] for item in manifest] == [
        ["answer", "error", "evidence_edges", "evidence_nodes"],
        ["answer", "error", "evidence_nodes"],
        [
            "answer",
            "confidence",
            "error",
            "error_category",
            "evidence_edges",
            "evidence_nodes",
            "needs_reobserve",
        ],
        ["answer", "error", "evidence_edges"],
        ["answer", "error", "evidence_edges"],
        ["answer", "error", "evidence_edges"],
    ]
    assert manifest[2]["question"] == {
        "type": "object_location",
        "object_id": "missing_object",
    }
    assert manifest[5]["question"] == {
        "type": "relative_relation",
        "src": "mug_1",
        "dst": "plate_1",
        "relation": "LEFT_OF",
        "reference_frame": "agent",
    }


def test_graph_query_evaluation_case_metadata_is_discoverable() -> None:
    manifest = list_evaluation_case_metadata(tags=("qa", "retrieval"))

    assert manifest == (
        {
            "name": "tabletop_graph_query_mug_plate",
            "scene_fixture": "tabletop",
            "scene_description": "Static tabletop scene with mug, plate, table, room, and agent.",
            "scene_tags": ["static", "tabletop"],
            "kind": "qa",
            "tags": ["qa", "retrieval", "graph_query"],
            "question": {
                "type": "graph_query",
                "query": {
                    "node_types": ["object"],
                    "labels": ["mug", "plate"],
                    "visible": True,
                    "relations": ["LEFT_OF"],
                    "reference_frame": "agent",
                    "max_nodes": 5,
                    "max_edges": 5,
                },
            },
            "question_type": "graph_query",
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": None,
            "target_label": None,
            "reference_object": None,
            "reference_label": None,
            "relation": None,
            "expected_keys": ["answer", "error", "evidence_edges", "evidence_nodes"],
        },
        {
            "name": "tabletop_nearest_candidate_plate",
            "scene_fixture": "tabletop",
            "scene_description": "Static tabletop scene with mug, plate, table, room, and agent.",
            "scene_tags": ["static", "tabletop"],
            "kind": "qa",
            "tags": ["qa", "retrieval", "nearest"],
            "question": {
                "type": "nearest_object",
                "src": "mug_1",
                "candidates": ["plate_1"],
            },
            "question_type": "nearest_object",
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": None,
            "target_label": None,
            "reference_object": None,
            "reference_label": None,
            "relation": None,
            "expected_keys": ["answer", "error", "evidence_nodes"],
        },
        {
            "name": "tabletop_retrieve_subgraph_mug",
            "scene_fixture": "tabletop",
            "scene_description": "Static tabletop scene with mug, plate, table, room, and agent.",
            "scene_tags": ["static", "tabletop"],
            "kind": "qa",
            "tags": ["qa", "retrieval", "retrieve_subgraph"],
            "question": {"type": "retrieve_subgraph", "query": "mug", "max_nodes": 3, "hops": 1},
            "question_type": "retrieve_subgraph",
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": None,
            "target_label": None,
            "reference_object": None,
            "reference_label": None,
            "relation": None,
            "expected_keys": ["answer", "error", "evidence_edges", "evidence_nodes"],
        },
    )


def test_scene_snapshot_evaluation_case_metadata_is_discoverable() -> None:
    manifest = list_evaluation_case_metadata(tags=("qa", "snapshot"))

    assert manifest == (
        {
            "name": "tabletop_scene_snapshot_invalid_step_error",
            "scene_fixture": "tabletop",
            "scene_description": "Static tabletop scene with mug, plate, table, room, and agent.",
            "scene_tags": ["static", "tabletop"],
            "kind": "qa",
            "tags": ["qa", "snapshot", "error"],
            "question": {"type": "scene_snapshot", "step": True},
            "question_type": "scene_snapshot",
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": None,
            "target_label": None,
            "reference_object": None,
            "reference_label": None,
            "relation": None,
            "expected_keys": [
                "answer",
                "confidence",
                "error",
                "error_category",
                "evidence_edges",
                "evidence_nodes",
                "needs_reobserve",
            ],
        },
        {
            "name": "tabletop_scene_snapshot_step1",
            "scene_fixture": "tabletop",
            "scene_description": "Static tabletop scene with mug, plate, table, room, and agent.",
            "scene_tags": ["static", "tabletop"],
            "kind": "qa",
            "tags": ["qa", "snapshot", "memory", "temporal"],
            "question": {"type": "scene_snapshot", "step": 1, "visible": True},
            "question_type": "scene_snapshot",
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": None,
            "target_label": None,
            "reference_object": None,
            "reference_label": None,
            "relation": None,
            "expected_keys": ["answer", "error", "evidence_edges", "evidence_nodes"],
        },
    )


def test_evaluation_cases_metadata_filters_custom_cases_deterministically() -> None:
    qa_case = EvaluationCase(
        name="custom_plate_status",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "custom"),
        question={"type": "object_status", "object_id": "plate_1"},
        expected={"answer": {"object_id": "plate_1"}, "error": None},
    )
    vla_case = EvaluationCase(
        name="custom_mug_pick",
        scene_fixture="tabletop",
        kind="vla_pick",
        tags=("vla", "custom"),
        target_object="mug_1",
        expected={"status": "ok"},
    )

    manifest = evaluation_cases_metadata(
        (qa_case, vla_case),
        tags=("custom",),
        kinds=("vla_pick",),
    )

    assert manifest == (
        {
            "name": "custom_mug_pick",
            "scene_fixture": "tabletop",
            "scene_description": "Static tabletop scene with mug, plate, table, room, and agent.",
            "scene_tags": ["static", "tabletop"],
            "kind": "vla_pick",
            "tags": ["vla", "custom"],
            "question": {},
            "question_type": None,
            "baseline_scene_fixture": None,
            "baseline_scene_description": None,
            "baseline_scene_tags": [],
            "target_object": "mug_1",
            "target_label": None,
            "reference_object": None,
            "reference_label": None,
            "relation": None,
            "expected_keys": ["status"],
        },
    )


def test_evaluation_case_metadata_filters_by_question_type() -> None:
    manifest = list_evaluation_case_metadata(question_types=("nearest_object",))

    assert [item["name"] for item in manifest] == ["tabletop_nearest_candidate_plate"]
    assert manifest[0]["question_type"] == "nearest_object"

    label_manifest = list_evaluation_case_metadata(question_types=("label_candidates",))

    assert [item["name"] for item in label_manifest] == [
        "ambiguous_mug_label_candidates",
        "needs_reobserve_spoon_label_candidates",
    ]
    assert label_manifest[0]["question"] == {
        "type": "label_candidates",
        "label": "mug",
        "visible": True,
    }
    assert label_manifest[1]["question"] == {
        "type": "label_candidates",
        "label": "spoon",
        "visible": False,
    }

    action_manifest = list_evaluation_case_metadata(question_types=("next_action_validity",))

    assert [item["name"] for item in action_manifest] == [
        "moved_mug_next_action_validity"
    ]
    assert action_manifest[0]["question"] == {"type": "next_action_validity"}

    room_manifest = list_evaluation_case_metadata(question_types=("object_room",))

    assert [item["name"] for item in room_manifest] == [
        "multi_room_rearrangement_object_room_cereal_box",
        "multi_room_rearrangement_object_room_milk",
    ]
    assert room_manifest[0]["question"] == {
        "type": "object_room",
        "object_id": "cereal_box_1",
    }
    assert room_manifest[1]["question"] == {
        "type": "object_room",
        "object_id": "milk_1",
    }


def test_evaluation_case_metadata_filters_by_explicit_names_in_call_order() -> None:
    manifest = list_evaluation_case_metadata(
        names=("tabletop_mug_pick", "tabletop_object_location"),
    )

    assert [item["name"] for item in manifest] == [
        "tabletop_mug_pick",
        "tabletop_object_location",
    ]
    assert [item["kind"] for item in manifest] == ["vla_pick", "qa"]


def test_evaluation_case_listing_includes_stable_digest_without_running_suite() -> None:
    assert hasattr(lab, "evaluation_case_listing")
    assert hasattr(lab, "evaluation_case_listing_digest")

    listing = evaluation_case_listing(tags=("qa",), question_types=("object_room",))
    expected_payload = {
        "schema_version": "dsg-spatialqa-lab.evaluation-case-listing.v1",
        "filters": {
            "names": [],
            "tags": ["qa"],
            "kinds": [],
            "question_types": ["object_room"],
        },
        "case_count": 2,
        "evaluation_cases": list(
            list_evaluation_case_metadata(
                tags=("qa",),
                question_types=("object_room",),
            )
        ),
    }
    expected_digest = hashlib.sha256(
        json.dumps(expected_payload, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()

    assert listing == {**expected_payload, "digest": expected_digest}
    assert lab.evaluation_case_listing_digest(listing) == expected_digest
    assert "suite" not in listing
    assert "report" not in listing

    tampered_listing = json.loads(json.dumps(listing))
    tampered_listing["filters"]["question_types"] = ["scene_delta"]
    assert lab.evaluation_case_listing_digest(tampered_listing) != expected_digest


def test_evaluation_case_listing_loads_from_explicit_file_and_validates(
    tmp_path: Path,
) -> None:
    listing = evaluation_case_listing(tags=("qa",), question_types=("object_room",))
    payload = evaluation_case_listing_json(listing)
    repeated = evaluation_case_listing_json(listing)
    listing_path = tmp_path / "listings" / "object-room.json"
    saved_path = save_evaluation_case_listing(
        listing_path,
        tags=("qa",),
        question_types=("object_room",),
    )

    loaded_listing = load_evaluation_case_listing(saved_path)
    validation = validate_evaluation_case_listing(loaded_listing)

    assert payload == repeated
    assert payload.endswith("\n")
    assert json.loads(payload) == listing
    assert saved_path == listing_path
    assert loaded_listing == listing
    assert validation == {
        "valid": True,
        "schema_version": "dsg-spatialqa-lab.evaluation-case-listing.v1",
        "digest": listing["digest"],
        "checks": [
            {
                "name": "schema_version",
                "passed": True,
                "expected": "dsg-spatialqa-lab.evaluation-case-listing.v1",
                "actual": "dsg-spatialqa-lab.evaluation-case-listing.v1",
            },
            {
                "name": "listing_digest",
                "passed": True,
                "expected": listing["digest"],
                "actual": listing["digest"],
            },
                {
                    "name": "case_count_matches_listing",
                    "passed": True,
                    "expected": 2,
                    "actual": 2,
                },
            {"name": "case_metadata_entries_valid", "passed": True},
        ],
    }


def test_evaluation_case_listing_validation_reports_tampered_digest_and_count() -> None:
    listing = evaluation_case_listing(tags=("qa",), question_types=("object_room",))
    tampered_listing = json.loads(evaluation_case_listing_json(listing))
    tampered_listing["digest"] = "0" * 64
    tampered_listing["case_count"] = 999

    validation = validate_evaluation_case_listing(tampered_listing)

    assert validation["valid"] is False
    assert validation["schema_version"] == "dsg-spatialqa-lab.evaluation-case-listing.v1"
    assert validation["digest"] == "0" * 64
    assert validation["checks"] == [
        {
            "name": "schema_version",
            "passed": True,
            "expected": "dsg-spatialqa-lab.evaluation-case-listing.v1",
            "actual": "dsg-spatialqa-lab.evaluation-case-listing.v1",
        },
        {
            "name": "listing_digest",
            "passed": False,
            "expected": lab.evaluation_case_listing_digest(tampered_listing),
            "actual": "0" * 64,
        },
            {
                "name": "case_count_matches_listing",
                "passed": False,
                "expected": 2,
                "actual": 999,
            },
        {"name": "case_metadata_entries_valid", "passed": True},
    ]


def test_evaluation_case_listing_validation_reports_tampered_schema_version() -> None:
    listing = evaluation_case_listing(tags=("qa",), question_types=("object_room",))
    tampered_listing = json.loads(evaluation_case_listing_json(listing))
    tampered_listing["schema_version"] = "legacy-listing"
    tampered_listing["digest"] = lab.evaluation_case_listing_digest(tampered_listing)

    validation = validate_evaluation_case_listing(tampered_listing)

    schema_check = next(
        check for check in validation["checks"] if check["name"] == "schema_version"
    )
    assert validation["valid"] is False
    assert validation["schema_version"] == "legacy-listing"
    assert schema_check == {
        "name": "schema_version",
        "passed": False,
        "expected": "dsg-spatialqa-lab.evaluation-case-listing.v1",
        "actual": "legacy-listing",
    }


def test_evaluation_case_listing_validation_reports_invalid_case_metadata() -> None:
    listing = evaluation_case_listing(tags=("qa",), question_types=("object_room",))
    tampered_listing = json.loads(evaluation_case_listing_json(listing))
    tampered_listing["evaluation_cases"][0]["name"] = 123
    tampered_listing["digest"] = lab.evaluation_case_listing_digest(tampered_listing)

    validation = validate_evaluation_case_listing(tampered_listing)

    checks = {check["name"]: check for check in validation["checks"]}
    assert validation["valid"] is False
    assert checks["listing_digest"]["passed"] is True
    assert checks["case_count_matches_listing"]["passed"] is True
    assert checks["case_metadata_entries_valid"] == {
        "name": "case_metadata_entries_valid",
        "passed": False,
        "differences": [
            {
                "path": "evaluation_cases[0].name",
                "expected": "non-empty string",
                "actual": 123,
            },
        ],
    }


def test_evaluation_case_listing_compare_matches_current_metadata() -> None:
    listing = evaluation_case_listing(tags=("qa",), question_types=("object_room",))

    comparison = compare_evaluation_case_listing(listing)

    assert comparison == {
        "matches": True,
        "filters": listing["filters"],
        "saved_digest": listing["digest"],
        "current_digest": listing["digest"],
        "checks": [
            {"name": "saved_listing_valid", "passed": True},
            {
                "name": "listing_digest_matches_current",
                "passed": True,
                "expected": listing["digest"],
                "actual": listing["digest"],
            },
                {
                    "name": "case_count_matches_current",
                    "passed": True,
                    "expected": 2,
                    "actual": 2,
                },
                {
                    "name": "case_metadata_matches_current",
                    "passed": True,
                    "expected": [
                        "multi_room_rearrangement_object_room_cereal_box",
                        "multi_room_rearrangement_object_room_milk",
                    ],
                    "actual": [
                        "multi_room_rearrangement_object_room_cereal_box",
                        "multi_room_rearrangement_object_room_milk",
                    ],
                },
        ],
    }


def test_evaluation_case_listing_compare_reports_current_metadata_drift() -> None:
    listing = evaluation_case_listing(tags=("qa",), question_types=("object_room",))
    drifted_listing = json.loads(evaluation_case_listing_json(listing))
    drifted_listing["filters"]["question_types"] = ["scene_delta"]
    current_listing = evaluation_case_listing(tags=("qa",), question_types=("scene_delta",))

    comparison = compare_evaluation_case_listing(drifted_listing)

    assert comparison["matches"] is False
    assert comparison["saved_digest"] == listing["digest"]
    assert comparison["current_digest"] == current_listing["digest"]
    assert comparison["checks"][0] == {"name": "saved_listing_valid", "passed": False}
    assert comparison["checks"][1] == {
        "name": "listing_digest_matches_current",
        "passed": False,
        "expected": listing["digest"],
        "actual": current_listing["digest"],
    }
    metadata_check = next(
        check
        for check in comparison["checks"]
        if check["name"] == "case_metadata_matches_current"
    )
    assert metadata_check["passed"] is False
    assert metadata_check["differences"][0]["path"] == "moved_mug_scene_delta"


def test_run_qa_evaluation_case_returns_comparable_result_dict() -> None:
    result = run_evaluation_case("tabletop_object_location")

    assert result == {
        "case": "tabletop_object_location",
        "scene_fixture": "tabletop",
        "kind": "qa",
        "question_type": "object_location",
        "tags": ["qa", "memory"],
        "passed": True,
        "actual": {
            "answer": {
                "object_id": "mug_1",
                "label": "mug",
                "pose": {"x": -0.4, "y": 1.0, "z": 0.78, "yaw": 0.0},
                "visible": True,
                "confidence": 0.95,
                "last_seen_step": 1,
                "state_step": 1,
                "current_location": {"relation": "ON", "dst": "table_1", "step": 1},
            },
            "evidence_nodes": ["mug_1", "state:mug_1:1"],
            "evidence_edges": [
                "mug_1-ON-table_1-1",
                "mug_1-STATE_CHANGED-state:mug_1:1-1",
            ],
            "confidence": 0.95,
            "needs_reobserve": False,
            "error": None,
        },
        "expected": {
            "answer": {
                "object_id": "mug_1",
                "label": "mug",
                "pose": {"x": -0.4, "y": 1.0, "z": 0.78, "yaw": 0.0},
                "visible": True,
                "confidence": 0.95,
                "last_seen_step": 1,
                "state_step": 1,
                "current_location": {"relation": "ON", "dst": "table_1", "step": 1},
            },
            "evidence_nodes": ["mug_1", "state:mug_1:1"],
            "evidence_edges": [
                "mug_1-ON-table_1-1",
                "mug_1-STATE_CHANGED-state:mug_1:1-1",
            ],
            "error": None,
        },
        "mismatches": [],
    }


def test_low_confidence_label_candidate_evaluation_case_marks_reobserve() -> None:
    result = run_evaluation_case("needs_reobserve_spoon_label_candidates")

    assert result["passed"] is True
    assert result["scene_fixture"] == "needs_reobserve"
    assert result["question_type"] == "label_candidates"
    assert result["tags"] == ["qa", "label", "reobserve", "occlusion"]
    assert result["actual"] == {
        "answer": {
            "label": "spoon",
            "visible": False,
            "count": 1,
            "ambiguous": False,
            "objects": [
                {
                    "object_id": "spoon_1",
                    "label": "spoon",
                    "pose": {"x": 0.2, "y": 0.8, "z": 0.75, "yaw": 0.0},
                    "visible": False,
                    "confidence": 0.25,
                    "last_seen_step": None,
                    "last_seen_pose": None,
                    "state_step": 2,
                    "needs_reobserve": True,
                }
            ],
        },
        "evidence_nodes": ["spoon_1", "state:spoon_1:2"],
        "evidence_edges": ["spoon_1-STATE_CHANGED-state:spoon_1:2-2"],
        "confidence": 0.25,
        "needs_reobserve": True,
        "error": None,
    }


def test_run_graph_query_evaluation_case_returns_retrieved_subgraph() -> None:
    result = run_evaluation_case("tabletop_graph_query_mug_plate")

    assert result["passed"] is True
    assert result["actual"]["answer"]["nodes"] == [
        {
            "id": "mug_1",
            "type": "object",
            "label": "mug",
            "attributes": {
                "pose": {"x": -0.4, "y": 1.0, "z": 0.78, "yaw": 0.0},
                "confidence": 0.95,
                "visible": True,
                "step": 1,
            },
        },
        {
            "id": "plate_1",
            "type": "object",
            "label": "plate",
            "attributes": {
                "pose": {"x": 0.35, "y": 1.0, "z": 0.72, "yaw": 0.0},
                "confidence": 0.9,
                "visible": True,
                "step": 1,
            },
        },
    ]
    assert result["actual"]["answer"]["edges"] == [
        {
            "id": "mug_1-LEFT_OF-plate_1-1",
            "src": "mug_1",
            "relation": "LEFT_OF",
            "dst": "plate_1",
            "reference_frame": "agent",
            "confidence": 1.0,
            "step": 1,
            "evidence": [],
            "attributes": {},
        }
    ]
    assert result["actual"]["evidence_nodes"] == ["mug_1", "plate_1"]
    assert result["actual"]["evidence_edges"] == ["mug_1-LEFT_OF-plate_1-1"]


def test_run_nearest_candidate_evaluation_case_returns_candidate_limited_result() -> None:
    result = run_evaluation_case("tabletop_nearest_candidate_plate")

    assert result["passed"] is True
    assert result["actual"]["answer"] == {
        "src": "mug_1",
        "nearest_object": "plate_1",
        "distance": 0.752396,
        "candidates": ["plate_1"],
        "candidate_distances": [
            {
                "object_id": "plate_1",
                "label": "plate",
                "distance": 0.752396,
                "visible": True,
                "confidence": 0.9,
                "needs_reobserve": False,
            }
        ],
    }
    assert result["actual"]["evidence_nodes"] == ["mug_1", "plate_1"]
    assert result["actual"]["confidence"] == 0.9


def test_run_missing_object_location_evaluation_case_returns_structured_error() -> None:
    result = run_evaluation_case("tabletop_missing_object_location_error")

    assert result["passed"] is True
    assert result["actual"] == {
        "answer": {},
        "evidence_nodes": [],
        "evidence_edges": [],
        "confidence": 0.0,
        "needs_reobserve": False,
        "error": "Object not found: missing_object",
        "error_category": "missing_object",
    }
    assert result["expected"] == result["actual"]
    assert result["mismatches"] == []


def test_run_unsupported_question_type_evaluation_case_returns_structured_error() -> None:
    result = run_evaluation_case("tabletop_unsupported_question_type_error")

    assert result["passed"] is True
    assert result["actual"] == {
        "answer": {},
        "evidence_nodes": [],
        "evidence_edges": [],
        "confidence": 0.0,
        "needs_reobserve": False,
        "error": "Unsupported question type: unsupported_intent",
        "error_category": "unsupported_question",
    }
    assert result["expected"] == result["actual"]
    assert result["mismatches"] == []


def test_run_next_action_validity_evaluation_case_detects_stale_command() -> None:
    result = run_evaluation_case("moved_mug_next_action_validity")

    assert result["passed"] is True
    assert result["actual"]["answer"] == {
        "valid": False,
        "needs_replan": True,
        "reason": "stale_object_state",
    }
    assert result["actual"]["evidence_edges"] == [
        "mug_1-LEFT_OF-plate_1-1",
        "mug_1-NEAR-plate_1-1",
        "mug_1-ON-table_1-1",
        "mug_1-STATE_CHANGED-state:mug_1:1-1",
        "plate_1-RIGHT_OF-mug_1-1",
        "mug_1-IN_REGION-sink_region-2",
        "mug_1-MOVED_FROM-table_1-2",
        "mug_1-MOVED_TO-sink_region-2",
        "mug_1-STATE_CHANGED-state:mug_1:2-2",
    ]
    assert result["actual"]["needs_reobserve"] is False


def test_run_multi_room_object_room_evaluation_case_covers_static_kitchen_object() -> None:
    result = run_evaluation_case("multi_room_rearrangement_object_room_milk")

    assert result["passed"] is True
    assert result["actual"] == {
        "answer": {
            "object_id": "milk_1",
            "room_id": "kitchen",
            "room_label": "Kitchen",
            "path": [
                {
                    "src": "milk_1",
                    "relation": "IN_REGION",
                    "dst": "prep_counter",
                    "step": 1,
                },
                {
                    "src": "prep_counter",
                    "relation": "IN_ROOM",
                    "dst": "kitchen",
                    "step": 1,
                },
            ],
        },
        "evidence_nodes": ["milk_1", "prep_counter", "kitchen"],
        "evidence_edges": [
            "milk_1-IN_REGION-prep_counter-1",
            "prep_counter-IN_ROOM-kitchen-1",
        ],
        "confidence": 0.9,
        "needs_reobserve": False,
        "error": None,
    }


def test_run_retrieve_subgraph_evaluation_case_returns_text_seeded_subgraph() -> None:
    result = run_evaluation_case("tabletop_retrieve_subgraph_mug")

    assert result["passed"] is True
    assert [node["id"] for node in result["actual"]["answer"]["nodes"]] == [
        "mug_1",
        "plate_1",
        "table_1",
    ]
    assert [edge["id"] for edge in result["actual"]["answer"]["edges"]] == [
        "mug_1-LEFT_OF-plate_1-1",
        "mug_1-NEAR-plate_1-1",
        "mug_1-ON-table_1-1",
    ]
    assert result["actual"]["evidence_nodes"] == ["mug_1", "plate_1", "table_1"]
    assert result["actual"]["evidence_edges"] == [
        "mug_1-LEFT_OF-plate_1-1",
        "mug_1-NEAR-plate_1-1",
        "mug_1-ON-table_1-1",
    ]


def test_run_scene_snapshot_evaluation_case_reconstructs_step_state() -> None:
    result = run_evaluation_case("tabletop_scene_snapshot_step1")

    assert result["passed"] is True
    assert result["actual"]["answer"]["step"] == 1
    assert result["actual"]["answer"]["agent"] == {
        "agent_id": "agent",
        "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        "state_step": 1,
    }
    assert result["actual"]["answer"]["objects"] == [
        {
            "object_id": "mug_1",
            "label": "mug",
            "pose": {"x": -0.4, "y": 1.0, "z": 0.78, "yaw": 0.0},
            "visible": True,
            "confidence": 0.95,
            "last_seen_step": 1,
            "state_step": 1,
            "current_location": {"relation": "ON", "dst": "table_1", "step": 1},
        },
        {
            "object_id": "plate_1",
            "label": "plate",
            "pose": {"x": 0.35, "y": 1.0, "z": 0.72, "yaw": 0.0},
            "visible": True,
            "confidence": 0.9,
            "last_seen_step": 1,
            "state_step": 1,
            "current_location": None,
        },
        {
            "object_id": "table_1",
            "label": "table",
            "pose": {"x": 0.0, "y": 1.0, "z": 0.35, "yaw": 0.0},
            "visible": True,
            "confidence": 1.0,
            "last_seen_step": 1,
            "state_step": 1,
            "current_location": None,
        },
    ]
    assert result["actual"]["evidence_nodes"] == [
        "agent",
        "state:agent:1",
        "mug_1",
        "state:mug_1:1",
        "plate_1",
        "state:plate_1:1",
        "table_1",
        "state:table_1:1",
    ]
    assert result["actual"]["evidence_edges"] == [
        "agent-STATE_CHANGED-state:agent:1-1",
        "mug_1-ON-table_1-1",
        "mug_1-STATE_CHANGED-state:mug_1:1-1",
        "plate_1-STATE_CHANGED-state:plate_1:1-1",
        "table_1-STATE_CHANGED-state:table_1:1-1",
    ]


def test_run_scene_snapshot_invalid_step_evaluation_case_returns_structured_error() -> None:
    result = run_evaluation_case("tabletop_scene_snapshot_invalid_step_error")

    assert result["passed"] is True
    assert result["actual"] == {
        "answer": {},
        "evidence_nodes": [],
        "evidence_edges": [],
        "confidence": 0.0,
        "needs_reobserve": False,
        "error": "Question field must be integer: step",
        "error_category": "invalid_question",
    }
    assert result["expected"] == result["actual"]
    assert result["mismatches"] == []


def test_run_vla_evaluation_case_detects_stale_pick() -> None:
    result = run_evaluation_case("moved_mug_stale_pick")

    assert result["passed"] is True
    assert result["actual"]["status"] == "needs_replan"
    assert result["actual"]["error"] == "stale_object_state"
    assert result["actual"]["error_category"] == "stale_object_state"
    assert result["actual"]["details"]["current_location"] == {
        "relation": "IN_REGION",
        "dst": "sink_region",
        "step": 2,
    }
    assert result["expected"] == {
        "status": "needs_replan",
        "error": "stale_object_state",
        "error_category": "stale_object_state",
    }


def test_run_vla_evaluation_case_detects_stale_place_reference() -> None:
    result = run_evaluation_case("moved_mug_stale_place_plate_right_of_mug")

    assert result["passed"] is True
    assert result["actual"]["status"] == "needs_replan"
    assert result["actual"]["error"] == "stale_reference_state"
    assert result["actual"]["error_category"] == "stale_reference_state"
    assert result["actual"]["details"] == {
        "target_object": "plate_1",
        "reference_object": "mug_1",
        "relation": "RIGHT_OF",
        "expected_anchor_pose": {"x": -0.11, "y": 1.0, "z": 0.88, "yaw": 0.0},
        "current_anchor_pose": {"x": 1.49, "y": 0.2, "z": 0.6, "yaw": 0.0},
        "expected_reference_last_seen_step": 1,
        "current_reference_last_seen_step": 2,
    }
    assert result["expected"] == {
        "status": "needs_replan",
        "error": "stale_reference_state",
        "error_category": "stale_reference_state",
    }


def test_run_vla_pick_evaluation_case_returns_skill_command() -> None:
    result = run_evaluation_case("tabletop_mug_pick")

    assert result["passed"] is True
    assert result["actual"]["status"] == "ok"
    assert result["actual"]["command"]["skill"] == "pick"
    assert result["actual"]["command"]["target_object"] == "mug_1"
    assert result["actual"]["command"]["target_pose"] == {
        "x": -0.4,
        "y": 1.0,
        "z": 0.78,
        "yaw": 0.0,
    }
    assert result["actual"]["command"]["evidence"] == ["mug_1"]


def test_run_missing_object_pick_evaluation_case_returns_structured_error() -> None:
    result = run_evaluation_case("tabletop_missing_object_pick_error")

    assert result["passed"] is True
    assert result["actual"] == {
        "status": "error",
        "command": None,
        "error": "Object not found: missing_object",
        "error_category": "missing_object",
        "needs_reobserve": False,
        "needs_replan": False,
        "ambiguous_ids": [],
        "details": {},
    }
    assert result["expected"] == result["actual"]
    assert result["mismatches"] == []


def test_run_missing_label_pick_evaluation_case_returns_structured_error() -> None:
    result = run_evaluation_case("tabletop_missing_label_pick_error")

    assert result["passed"] is True
    assert result["actual"] == {
        "status": "error",
        "command": None,
        "error": "Object label not found: fork",
        "error_category": "missing_label",
        "needs_reobserve": False,
        "needs_replan": False,
        "ambiguous_ids": [],
        "details": {},
    }
    assert result["expected"] == result["actual"]
    assert result["mismatches"] == []


def test_run_pick_case_without_target_returns_structured_error() -> None:
    result = run_evaluation_case_definition(
        EvaluationCase(
            name="custom_missing_pick_target",
            scene_fixture="tabletop",
            kind="vla_pick",
            tags=("vla", "pick", "error"),
            expected={
                "status": "error",
                "command": None,
                "error": "target_object or label is required",
                "error_category": "missing_target",
                "needs_reobserve": False,
                "needs_replan": False,
                "ambiguous_ids": [],
                "details": {},
            },
        )
    )

    assert result["passed"] is True
    assert result["actual"] == result["expected"]
    assert result["mismatches"] == []


def test_run_missing_target_pick_evaluation_case_returns_structured_error() -> None:
    result = run_evaluation_case("tabletop_missing_target_pick_error")

    assert result["passed"] is True
    assert result["actual"] == {
        "status": "error",
        "command": None,
        "error": "target_object or label is required",
        "error_category": "missing_target",
        "needs_reobserve": False,
        "needs_replan": False,
        "ambiguous_ids": [],
        "details": {},
    }
    assert result["expected"] == result["actual"]
    assert result["mismatches"] == []


def test_run_place_relative_case_without_target_returns_structured_error() -> None:
    result = run_evaluation_case_definition(
        EvaluationCase(
            name="custom_missing_place_target",
            scene_fixture="tabletop",
            kind="vla_place_relative",
            tags=("vla", "place", "error"),
            reference_object="plate_1",
            relation="RIGHT_OF",
            expected={
                "status": "error",
                "command": None,
                "error": "target_object or target_label is required",
                "error_category": "missing_target",
                "needs_reobserve": False,
                "needs_replan": False,
                "ambiguous_ids": [],
                "details": {},
            },
        )
    )

    assert result["passed"] is True
    assert result["actual"] == result["expected"]
    assert result["mismatches"] == []


def test_run_missing_target_input_place_evaluation_case_returns_structured_error() -> None:
    result = run_evaluation_case("tabletop_missing_target_input_place_error")

    assert result["passed"] is True
    assert result["actual"] == {
        "status": "error",
        "command": None,
        "error": "target_object or target_label is required",
        "error_category": "missing_target",
        "needs_reobserve": False,
        "needs_replan": False,
        "ambiguous_ids": [],
        "details": {},
    }
    assert result["expected"] == result["actual"]
    assert result["mismatches"] == []


def test_run_missing_reference_place_evaluation_case_returns_structured_error() -> None:
    result = run_evaluation_case("tabletop_missing_reference_place_error")

    assert result["passed"] is True
    assert result["actual"] == {
        "status": "error",
        "command": None,
        "error": "Object not found: missing_object",
        "error_category": "missing_object",
        "needs_reobserve": False,
        "needs_replan": False,
        "ambiguous_ids": [],
        "details": {},
    }
    assert result["expected"] == result["actual"]
    assert result["mismatches"] == []


def test_run_missing_reference_input_place_evaluation_case_returns_structured_error() -> None:
    result = run_evaluation_case("tabletop_missing_reference_input_place_error")

    assert result["passed"] is True
    assert result["actual"] == {
        "status": "error",
        "command": None,
        "error": "reference_object or reference_label is required",
        "error_category": "missing_reference",
        "needs_reobserve": False,
        "needs_replan": False,
        "ambiguous_ids": [],
        "details": {},
    }
    assert result["expected"] == result["actual"]
    assert result["mismatches"] == []


def test_run_place_relative_case_without_reference_returns_structured_error() -> None:
    result = run_evaluation_case_definition(
        EvaluationCase(
            name="custom_missing_reference_input",
            scene_fixture="tabletop",
            kind="vla_place_relative",
            tags=("vla", "place", "error"),
            target_object="mug_1",
            relation="RIGHT_OF",
            expected={
                "status": "error",
                "command": None,
                "error": "reference_object or reference_label is required",
                "error_category": "missing_reference",
                "needs_reobserve": False,
                "needs_replan": False,
                "ambiguous_ids": [],
                "details": {},
            },
        )
    )

    assert result["passed"] is True
    assert result["actual"] == result["expected"]
    assert result["mismatches"] == []


def test_run_unsupported_place_relation_evaluation_case_returns_structured_error() -> None:
    result = run_evaluation_case("tabletop_unsupported_place_relation_error")

    assert result["passed"] is True
    assert result["actual"] == {
        "status": "error",
        "command": None,
        "error": "Unsupported place relation: ABOVE",
        "error_category": "unsupported_relation",
        "needs_reobserve": False,
        "needs_replan": False,
        "ambiguous_ids": [],
        "details": {},
    }
    assert result["expected"] == result["actual"]
    assert result["mismatches"] == []


def test_run_vla_pick_evaluation_case_can_resolve_unique_label() -> None:
    case = EvaluationCase(
        name="custom_pick_plate_by_label",
        scene_fixture="tabletop",
        kind="vla_pick",
        tags=("vla", "label"),
        target_label="plate",
        expected={
            "status": "ok",
            "command": {
                "skill": "pick",
                "target_object": "plate_1",
            },
        },
    )

    result = run_evaluation_case_definition(case)

    assert result["passed"] is True
    assert result["actual"]["command"]["target_object"] == "plate_1"


def test_run_vla_pick_evaluation_case_reports_ambiguous_label() -> None:
    case = EvaluationCase(
        name="custom_ambiguous_mug_pick",
        scene_fixture="ambiguous_mugs",
        kind="vla_pick",
        tags=("vla", "label", "ambiguity"),
        target_label="mug",
        expected={
            "status": "ambiguous",
            "error": "Ambiguous label: mug",
            "ambiguous_ids": ["mug_1", "mug_2"],
        },
    )

    result = run_evaluation_case_definition(
        case,
        scene_loaders={"ambiguous_mugs": _build_ambiguous_mugs_scene},
    )

    assert result["passed"] is True
    assert result["actual"]["ambiguous_ids"] == ["mug_1", "mug_2"]


def test_run_vla_place_relative_evaluation_case_returns_anchor() -> None:
    result = run_evaluation_case("tabletop_place_mug_right_of_plate")

    assert result["passed"] is True
    assert result["actual"]["status"] == "ok"
    assert result["actual"]["command"]["skill"] == "place_relative"
    assert result["actual"]["command"]["target_object"] == "mug_1"
    assert result["actual"]["command"]["reference_object"] == "plate_1"
    assert result["actual"]["command"]["parameters"] == {"relation": "RIGHT_OF"}
    assert result["actual"]["command"]["target_pose"] == {
        "x": 0.64,
        "y": 1.0,
        "z": 0.82,
        "yaw": 0.0,
    }


def test_run_vla_place_relative_evaluation_case_can_resolve_labels() -> None:
    case = EvaluationCase(
        name="custom_place_mug_right_of_plate_by_label",
        scene_fixture="tabletop",
        kind="vla_place_relative",
        tags=("vla", "label", "place"),
        target_label="mug",
        reference_label="plate",
        relation="RIGHT_OF",
        expected={
            "status": "ok",
            "command": {
                "skill": "place_relative",
                "target_object": "mug_1",
                "reference_object": "plate_1",
                "parameters": {"relation": "RIGHT_OF"},
            },
        },
    )

    result = run_evaluation_case_definition(case)

    assert result["passed"] is True
    assert result["actual"]["command"]["target_pose"] == {
        "x": 0.64,
        "y": 1.0,
        "z": 0.82,
        "yaw": 0.0,
    }


def test_run_vla_place_relative_evaluation_case_reports_ambiguous_reference_label() -> None:
    case = EvaluationCase(
        name="custom_ambiguous_plate_place",
        scene_fixture="ambiguous_plates",
        kind="vla_place_relative",
        tags=("vla", "label", "ambiguity"),
        target_object="mug_1",
        reference_label="plate",
        relation="RIGHT_OF",
        expected={
            "status": "ambiguous",
            "error": "Ambiguous label: plate",
            "ambiguous_ids": ["plate_1", "plate_2"],
        },
    )

    result = run_evaluation_case_definition(
        case,
        scene_loaders={"ambiguous_plates": _build_ambiguous_plates_scene},
    )

    assert result["passed"] is True
    assert result["actual"]["ambiguous_ids"] == ["plate_1", "plate_2"]


def test_run_scene_delta_evaluation_case_reports_temporal_changes() -> None:
    result = run_evaluation_case("moved_mug_scene_delta")

    assert result["passed"] is True
    assert result["actual"]["answer"]["from_step"] == 1
    assert result["actual"]["answer"]["to_step"] == 2
    assert result["actual"]["answer"]["objects"] == [
        {
            "object_id": "mug_1",
            "label": "mug",
            "changes": ["pose", "last_seen_step", "location"],
            "from_pose": {"x": -0.4, "y": 1.0, "z": 0.78, "yaw": 0.0},
            "to_pose": {"x": 1.2, "y": 0.2, "z": 0.5, "yaw": 0.0},
            "from_visible": True,
            "to_visible": True,
            "from_confidence": 0.95,
            "to_confidence": 0.95,
            "from_last_seen_step": 1,
            "to_last_seen_step": 2,
            "from_location": {"relation": "ON", "dst": "table_1", "step": 1},
            "to_location": {"relation": "IN_REGION", "dst": "sink_region", "step": 2},
            "from_state_step": 1,
            "to_state_step": 2,
        }
    ]
    assert result["expected"]["answer"]["objects"][0]["changes"] == [
        "pose",
        "last_seen_step",
        "location",
    ]


def test_run_scene_delta_reversed_window_case_returns_structured_error() -> None:
    result = run_evaluation_case("moved_mug_scene_delta_reversed_window_error")

    assert result["passed"] is True
    assert result["actual"] == {
        "answer": {},
        "evidence_nodes": [],
        "evidence_edges": [],
        "confidence": 0.0,
        "needs_reobserve": False,
        "error": "from_step cannot be greater than to_step",
        "error_category": "invalid_time_window",
    }
    assert result["expected"] == result["actual"]
    assert result["mismatches"] == []


def test_run_world_state_evaluation_case_reports_current_dynamic_state() -> None:
    result = run_evaluation_case("moved_mug_world_state")

    assert result["passed"] is True
    assert result["actual"]["answer"]["agent_pose"] == {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "yaw": 0.0,
    }
    assert result["actual"]["answer"]["objects"] == [
        {
            "object_id": "mug_1",
            "label": "mug",
            "pose": {"x": 1.2, "y": 0.2, "z": 0.5, "yaw": 0.0},
            "visible": True,
            "confidence": 0.95,
            "last_seen_step": 2,
            "current_location": {
                "relation": "IN_REGION",
                "dst": "sink_region",
                "step": 2,
            },
        },
        {
            "object_id": "plate_1",
            "label": "plate",
            "pose": {"x": 0.35, "y": 1.0, "z": 0.72, "yaw": 0.0},
            "visible": True,
            "confidence": 0.9,
            "last_seen_step": 1,
            "current_location": None,
        },
        {
            "object_id": "table_1",
            "label": "table",
            "pose": {"x": 0.0, "y": 1.0, "z": 0.35, "yaw": 0.0},
            "visible": True,
            "confidence": 1.0,
            "last_seen_step": 1,
            "current_location": None,
        },
    ]
    assert result["actual"]["evidence_nodes"] == ["agent", "mug_1", "plate_1", "table_1"]
    assert result["actual"]["evidence_edges"] == [
        "mug_1-IN_REGION-sink_region-2",
        "mug_1-STATE_CHANGED-state:mug_1:2-2",
    ]


def test_run_multi_room_rearrangement_evaluation_cases_cover_move_reobserve_and_events() -> None:
    delta_result = run_evaluation_case("multi_room_rearrangement_scene_delta")
    reobserve_result = run_evaluation_case("multi_room_rearrangement_reobserve_targets")
    events_result = run_evaluation_case("multi_room_rearrangement_recent_events")

    assert delta_result["passed"] is True
    assert delta_result["actual"]["answer"]["agent"]["changed"] is True
    assert delta_result["actual"]["answer"]["objects"][0]["object_id"] == "cereal_box_1"
    assert delta_result["actual"]["answer"]["objects"][0]["to_location"] == {
        "relation": "IN_REGION",
        "dst": "pantry_shelf",
        "step": 2,
    }
    assert reobserve_result["passed"] is True
    assert reobserve_result["actual"]["answer"]["objects"][0]["object_id"] == "fork_1"
    assert reobserve_result["actual"]["answer"]["objects"][0]["confidence"] == 0.2
    assert events_result["passed"] is True
    assert events_result["actual"]["answer"]["events"] == [
        {"id": "action_move_cereal_box", "type": "action", "label": "move", "step": 2},
        {
            "id": "event_move_cereal_box",
            "type": "event",
            "label": "move_object",
            "step": 2,
        },
    ]


def test_run_timeline_evaluation_cases_return_state_evidence() -> None:
    agent_result = run_evaluation_case("tabletop_agent_timeline")
    object_result = run_evaluation_case("moved_mug_object_timeline")
    relation_shift_result = run_evaluation_case("relation_shift_relation_timeline")
    relation_result = run_evaluation_case("tabletop_relation_timeline")

    assert agent_result["passed"] is True
    assert agent_result["actual"]["answer"] == {
        "agent_id": "agent",
        "timeline": [
            {
                "agent_id": "agent",
                "step": 1,
                "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                "evidence_edges": ["agent-STATE_CHANGED-state:agent:1-1"],
            }
        ],
    }
    assert agent_result["actual"]["evidence_edges"] == [
        "agent-STATE_CHANGED-state:agent:1-1"
    ]
    assert object_result["passed"] is True
    assert [entry["step"] for entry in object_result["actual"]["answer"]["timeline"]] == [1, 2]
    assert object_result["actual"]["answer"]["timeline"][1]["current_location"] == {
        "relation": "IN_REGION",
        "dst": "sink_region",
        "step": 2,
    }
    assert object_result["actual"]["evidence_edges"] == [
        "mug_1-ON-table_1-1",
        "mug_1-STATE_CHANGED-state:mug_1:1-1",
        "mug_1-IN_REGION-sink_region-2",
        "mug_1-STATE_CHANGED-state:mug_1:2-2",
    ]
    assert relation_shift_result["passed"] is True
    assert [
        (entry["relation"], entry["step"])
        for entry in relation_shift_result["actual"]["answer"]["timeline"]
    ] == [
        ("LEFT_OF", 1),
        ("NEAR", 1),
        ("NEAR", 2),
        ("RIGHT_OF", 2),
    ]
    assert relation_shift_result["actual"]["evidence_edges"] == [
        "mug_1-LEFT_OF-plate_1-1",
        "mug_1-NEAR-plate_1-1",
        "mug_1-NEAR-plate_1-2",
        "mug_1-RIGHT_OF-plate_1-2",
    ]
    assert relation_result["passed"] is True
    assert relation_result["actual"]["answer"]["timeline"] == [
        {
            "id": "mug_1-LEFT_OF-plate_1-1",
            "src": "mug_1",
            "relation": "LEFT_OF",
            "dst": "plate_1",
            "reference_frame": "agent",
            "confidence": 1.0,
            "step": 1,
            "evidence": [],
            "attributes": {},
        },
        {
            "id": "mug_1-NEAR-plate_1-1",
            "src": "mug_1",
            "relation": "NEAR",
            "dst": "plate_1",
            "reference_frame": "agent",
            "confidence": 1.0,
            "step": 1,
            "evidence": [],
            "attributes": {},
        },
    ]
    assert relation_result["actual"]["evidence_edges"] == [
        "mug_1-LEFT_OF-plate_1-1",
        "mug_1-NEAR-plate_1-1",
    ]


def test_run_reobserve_targets_evaluation_case_returns_targets() -> None:
    result = run_evaluation_case("needs_reobserve_targets")

    assert result["passed"] is True
    assert result["actual"]["answer"] == {
        "count": 1,
        "objects": [
            {
                "object_id": "spoon_1",
                "label": "spoon",
                "pose": {"x": 0.2, "y": 0.8, "z": 0.75, "yaw": 0.0},
                "visible": False,
                "confidence": 0.25,
                "last_seen_step": None,
                "last_seen_pose": None,
                "state_step": 2,
            }
        ],
    }
    assert result["actual"]["evidence_edges"] == [
        "spoon_1-STATE_CHANGED-state:spoon_1:2-2"
    ]
    assert result["actual"]["needs_reobserve"] is True


def test_run_vla_reobserve_evaluation_case_does_not_emit_command() -> None:
    result = run_evaluation_case("needs_reobserve_spoon_pick")

    assert result["passed"] is True
    assert result["actual"]["status"] == "needs_reobserve"
    assert result["actual"]["command"] is None
    assert result["actual"]["error"] == "needs_reobserve"
    assert result["actual"]["error_category"] == "needs_reobserve"
    assert result["actual"]["needs_reobserve"] is True
    assert result["actual"]["needs_replan"] is False
    assert result["actual"]["details"] == {
        "target_object": "spoon_1",
        "visible": False,
        "confidence": 0.25,
        "min_confidence": 0.5,
        "last_seen_step": None,
        "current_step": 2,
    }


def test_run_vla_place_reference_reobserve_case_does_not_emit_command() -> None:
    result = run_evaluation_case("needs_reobserve_spoon_place_reference")

    assert result["passed"] is True
    assert result["actual"]["status"] == "needs_reobserve"
    assert result["actual"]["command"] is None
    assert result["actual"]["error"] == "needs_reobserve"
    assert result["actual"]["error_category"] == "needs_reobserve"
    assert result["actual"]["needs_reobserve"] is True
    assert result["actual"]["needs_replan"] is False
    assert result["actual"]["details"] == {
        "reference_object": "spoon_1",
        "visible": False,
        "confidence": 0.25,
        "min_confidence": 0.5,
        "last_seen_step": None,
        "current_step": 2,
    }


def test_run_vla_place_target_reobserve_case_does_not_emit_command() -> None:
    result = run_evaluation_case("needs_reobserve_spoon_place_target")

    assert result["passed"] is True
    assert result["actual"]["status"] == "needs_reobserve"
    assert result["actual"]["command"] is None
    assert result["actual"]["error"] == "needs_reobserve"
    assert result["actual"]["error_category"] == "needs_reobserve"
    assert result["actual"]["needs_reobserve"] is True
    assert result["actual"]["needs_replan"] is False
    assert result["actual"]["details"] == {
        "target_object": "spoon_1",
        "visible": False,
        "confidence": 0.25,
        "min_confidence": 0.5,
        "last_seen_step": None,
        "current_step": 2,
    }


def test_run_vla_target_not_visible_evaluation_case_is_categorized() -> None:
    result = run_evaluation_case("needs_reobserve_bowl_pick_target_not_visible")

    assert result["passed"] is True
    assert result["actual"]["status"] == "needs_replan"
    assert result["actual"]["command"] is None
    assert result["actual"]["error"] == "target_not_visible"
    assert result["actual"]["error_category"] == "target_not_visible"
    assert result["actual"]["needs_reobserve"] is False
    assert result["actual"]["needs_replan"] is True
    assert result["actual"]["details"] == {
        "target_object": "bowl_1",
        "visible": False,
        "confidence": 0.75,
        "min_confidence": 0.5,
        "last_seen_step": None,
        "current_step": 2,
    }


def test_run_vla_place_reference_not_visible_evaluation_case_is_categorized() -> None:
    result = run_evaluation_case("needs_reobserve_bowl_place_reference_target_not_visible")

    assert result["passed"] is True
    assert result["actual"]["status"] == "needs_replan"
    assert result["actual"]["command"] is None
    assert result["actual"]["error"] == "target_not_visible"
    assert result["actual"]["error_category"] == "target_not_visible"
    assert result["actual"]["needs_reobserve"] is False
    assert result["actual"]["needs_replan"] is True
    assert result["actual"]["details"] == {
        "reference_object": "bowl_1",
        "visible": False,
        "confidence": 0.75,
        "min_confidence": 0.5,
        "last_seen_step": None,
        "current_step": 2,
    }


def test_run_vla_place_target_not_visible_evaluation_case_is_categorized() -> None:
    result = run_evaluation_case("needs_reobserve_bowl_place_target_not_visible")

    assert result["passed"] is True
    assert result["actual"]["status"] == "needs_replan"
    assert result["actual"]["command"] is None
    assert result["actual"]["error"] == "target_not_visible"
    assert result["actual"]["error_category"] == "target_not_visible"
    assert result["actual"]["needs_reobserve"] is False
    assert result["actual"]["needs_replan"] is True
    assert result["actual"]["details"] == {
        "target_object": "bowl_1",
        "visible": False,
        "confidence": 0.75,
        "min_confidence": 0.5,
        "last_seen_step": None,
        "current_step": 2,
    }


def test_run_vla_low_confidence_evaluation_case_is_categorized() -> None:
    result = run_evaluation_case("needs_reobserve_cup_pick_low_confidence")

    assert result["passed"] is True
    assert result["actual"]["status"] == "needs_replan"
    assert result["actual"]["command"] is None
    assert result["actual"]["error"] == "low_confidence"
    assert result["actual"]["error_category"] == "low_confidence"
    assert result["actual"]["needs_reobserve"] is False
    assert result["actual"]["needs_replan"] is True
    assert result["actual"]["details"] == {
        "target_object": "cup_1",
        "visible": True,
        "confidence": 0.2,
        "min_confidence": 0.5,
        "last_seen_step": 2,
        "current_step": 2,
    }


def test_run_vla_place_target_low_confidence_evaluation_case_is_categorized() -> None:
    result = run_evaluation_case("needs_reobserve_cup_place_target_low_confidence")

    assert result["passed"] is True
    assert result["actual"]["status"] == "needs_replan"
    assert result["actual"]["command"] is None
    assert result["actual"]["error"] == "low_confidence"
    assert result["actual"]["error_category"] == "low_confidence"
    assert result["actual"]["needs_reobserve"] is False
    assert result["actual"]["needs_replan"] is True
    assert result["actual"]["details"] == {
        "target_object": "cup_1",
        "visible": True,
        "confidence": 0.2,
        "min_confidence": 0.5,
        "last_seen_step": 2,
        "current_step": 2,
    }


def test_run_vla_place_reference_low_confidence_evaluation_case_is_categorized() -> None:
    result = run_evaluation_case("needs_reobserve_cup_place_reference_low_confidence")

    assert result["passed"] is True
    assert result["actual"]["status"] == "needs_replan"
    assert result["actual"]["command"] is None
    assert result["actual"]["error"] == "low_confidence"
    assert result["actual"]["error_category"] == "low_confidence"
    assert result["actual"]["needs_reobserve"] is False
    assert result["actual"]["needs_replan"] is True
    assert result["actual"]["details"] == {
        "reference_object": "cup_1",
        "visible": True,
        "confidence": 0.2,
        "min_confidence": 0.5,
        "last_seen_step": 2,
        "current_step": 2,
    }


def test_visible_low_confidence_vla_pick_error_is_categorized() -> None:
    case = EvaluationCase(
        name="custom_visible_low_confidence_pick",
        scene_fixture="custom_low_confidence",
        kind="vla_pick",
        tags=("vla", "custom", "error"),
        target_object="cup_1",
        expected={
            "status": "needs_replan",
            "error": "low_confidence",
            "error_category": "low_confidence",
            "needs_reobserve": False,
            "needs_replan": True,
            "details": {
                "target_object": "cup_1",
                "visible": True,
                "confidence": 0.2,
                "min_confidence": 0.5,
                "last_seen_step": 4,
                "current_step": 4,
            },
        },
    )

    result = run_evaluation_case_definition(
        case,
        scene_loaders={"custom_low_confidence": _build_custom_low_confidence_scene},
    )

    assert result["passed"] is True
    assert result["actual"]["error_category"] == "low_confidence"


def test_run_qa_label_ambiguity_evaluation_case_lists_candidates() -> None:
    result = run_evaluation_case("ambiguous_mug_label_candidates")

    assert result["passed"] is True
    assert result["question_type"] == "label_candidates"
    assert result["actual"]["answer"] == {
        "label": "mug",
        "visible": True,
        "count": 2,
        "ambiguous": True,
        "objects": [
            {
                "object_id": "mug_1",
                "label": "mug",
                "pose": {"x": 0.0, "y": 1.0, "z": 0.7, "yaw": 0.0},
                "visible": True,
                "confidence": 0.9,
                "last_seen_step": 1,
                "last_seen_pose": {"x": 0.0, "y": 1.0, "z": 0.7, "yaw": 0.0},
                "state_step": 1,
                "needs_reobserve": False,
            },
            {
                "object_id": "mug_2",
                "label": "mug",
                "pose": {"x": 0.3, "y": 1.0, "z": 0.7, "yaw": 0.0},
                "visible": True,
                "confidence": 0.88,
                "last_seen_step": 1,
                "last_seen_pose": {"x": 0.3, "y": 1.0, "z": 0.7, "yaw": 0.0},
                "state_step": 1,
                "needs_reobserve": False,
            },
        ],
    }
    assert result["actual"]["evidence_nodes"] == [
        "mug_1",
        "state:mug_1:1",
        "mug_2",
        "state:mug_2:1",
    ]
    assert result["actual"]["evidence_edges"] == [
        "mug_1-STATE_CHANGED-state:mug_1:1-1",
        "mug_2-STATE_CHANGED-state:mug_2:1-1",
    ]
    assert result["actual"]["confidence"] == 0.88
    assert result["actual"]["needs_reobserve"] is False


def test_run_vla_ambiguity_evaluation_case_does_not_choose_target() -> None:
    result = run_evaluation_case("ambiguous_mug_pick_by_label")

    assert result["passed"] is True
    assert result["actual"]["status"] == "ambiguous"
    assert result["actual"]["command"] is None
    assert result["actual"]["error"] == "Ambiguous label: mug"
    assert result["actual"]["error_category"] == "ambiguous_label"
    assert result["actual"]["ambiguous_ids"] == ["mug_1", "mug_2"]
    assert result["actual"]["details"] == {
        "label": "mug",
        "candidate_count": 2,
        "candidates": [
            {
                "object_id": "mug_1",
                "label": "mug",
                "pose": {"x": 0.0, "y": 1.0, "z": 0.7, "yaw": 0.0},
                "visible": True,
                "confidence": 0.9,
                "last_seen_step": 1,
                "needs_reobserve": False,
            },
            {
                "object_id": "mug_2",
                "label": "mug",
                "pose": {"x": 0.3, "y": 1.0, "z": 0.7, "yaw": 0.0},
                "visible": True,
                "confidence": 0.88,
                "last_seen_step": 1,
                "needs_reobserve": False,
            },
        ],
    }


def test_run_vla_reference_ambiguity_evaluation_case_does_not_choose_reference() -> None:
    result = run_evaluation_case("ambiguous_plate_place_reference_by_label")

    assert result["passed"] is True
    assert result["actual"]["status"] == "ambiguous"
    assert result["actual"]["command"] is None
    assert result["actual"]["error"] == "Ambiguous label: plate"
    assert result["actual"]["error_category"] == "ambiguous_label"
    assert result["actual"]["ambiguous_ids"] == ["plate_1", "plate_2"]
    assert result["actual"]["details"] == {
        "label": "plate",
        "candidate_count": 2,
        "candidates": [
            {
                "object_id": "plate_1",
                "label": "plate",
                "pose": {"x": 0.35, "y": 1.0, "z": 0.72, "yaw": 0.0},
                "visible": True,
                "confidence": 0.9,
                "last_seen_step": 1,
                "needs_reobserve": False,
            },
            {
                "object_id": "plate_2",
                "label": "plate",
                "pose": {"x": 0.7, "y": 1.0, "z": 0.72, "yaw": 0.0},
                "visible": True,
                "confidence": 0.88,
                "last_seen_step": 1,
                "needs_reobserve": False,
            },
        ],
    }


def test_run_foundation_qa_evaluation_cases_cover_basic_intents() -> None:
    agent_location = run_evaluation_case("tabletop_agent_location")
    agent_history = run_evaluation_case("tabletop_agent_history")
    object_status = run_evaluation_case("tabletop_object_status_plate")
    object_history = run_evaluation_case("tabletop_object_history_mug")
    relative_relation = run_evaluation_case("tabletop_relative_relation_mug_left_of_plate")

    assert agent_location["passed"] is True
    assert agent_location["actual"]["answer"] == {
        "agent_id": "agent",
        "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
    }
    assert agent_location["actual"]["evidence_nodes"] == ["agent"]
    assert agent_history["passed"] is True
    assert agent_history["actual"]["answer"] == {
        "agent_id": "agent",
        "poses": [{"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}],
        "steps": [1],
    }
    assert agent_history["actual"]["evidence_edges"] == [
        "agent-STATE_CHANGED-state:agent:1-1"
    ]
    assert object_status["passed"] is True
    assert object_status["actual"]["answer"] == {
        "object_id": "plate_1",
        "label": "plate",
        "visible": True,
        "confidence": 0.9,
        "last_seen_step": 1,
        "last_seen_pose": {"x": 0.35, "y": 1.0, "z": 0.72, "yaw": 0.0},
        "needs_reobserve": False,
    }
    assert object_status["actual"]["evidence_edges"] == [
        "plate_1-STATE_CHANGED-state:plate_1:1-1"
    ]
    assert object_history["passed"] is True
    assert object_history["actual"]["answer"] == {
        "object_id": "mug_1",
        "relations": ["LEFT_OF", "NEAR", "ON", "STATE_CHANGED", "RIGHT_OF"],
        "steps": [1, 1, 1, 1, 1],
    }
    assert object_history["actual"]["evidence_edges"] == [
        "mug_1-LEFT_OF-plate_1-1",
        "mug_1-NEAR-plate_1-1",
        "mug_1-ON-table_1-1",
        "mug_1-STATE_CHANGED-state:mug_1:1-1",
        "plate_1-RIGHT_OF-mug_1-1",
    ]
    assert relative_relation["passed"] is True
    assert relative_relation["actual"]["answer"] == {
        "holds": True,
        "relation": "LEFT_OF",
        "src": "mug_1",
        "dst": "plate_1",
    }
    assert relative_relation["actual"]["evidence_edges"] == [
        "mug_1-LEFT_OF-plate_1-1"
    ]


def test_run_evaluation_suite_summarizes_selected_cases() -> None:
    suite = run_evaluation_suite(["moved_mug_recent_events", "tabletop_object_location"])

    assert suite["summary"] == {
        "total": 2,
        "passed": 2,
        "failed": 0,
        "failed_cases": [],
        "selected_cases": ["moved_mug_recent_events", "tabletop_object_location"],
    }
    assert suite["breakdown"]["by_kind"] == {
        "qa": {
            "total": 2,
            "passed": 2,
            "failed": 0,
            "failed_cases": [],
            "selected_cases": ["moved_mug_recent_events", "tabletop_object_location"],
        }
    }
    assert suite["breakdown"]["by_question_type"] == {
        "object_location": {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "failed_cases": [],
            "selected_cases": ["tabletop_object_location"],
        },
        "recent_events": {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "failed_cases": [],
            "selected_cases": ["moved_mug_recent_events"],
        },
    }
    assert suite["breakdown"]["by_tag"] == {
        "dynamic": {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "failed_cases": [],
            "selected_cases": ["moved_mug_recent_events"],
        },
        "memory": {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "failed_cases": [],
            "selected_cases": ["tabletop_object_location"],
        },
        "qa": {
            "total": 2,
            "passed": 2,
            "failed": 0,
            "failed_cases": [],
            "selected_cases": ["moved_mug_recent_events", "tabletop_object_location"],
        },
    }
    assert suite["breakdown"]["by_scene_fixture"] == {
        "moved_mug": {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "failed_cases": [],
            "selected_cases": ["moved_mug_recent_events"],
        },
        "tabletop": {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "failed_cases": [],
            "selected_cases": ["tabletop_object_location"],
        },
    }
    assert [result["case"] for result in suite["results"]] == [
        "moved_mug_recent_events",
        "tabletop_object_location",
    ]
    assert suite["results"][0]["passed"] is True
    assert suite["results"][1]["passed"] is True


def test_run_evaluation_suite_filters_by_tags() -> None:
    suite = run_evaluation_suite(tags=("qa", "dynamic"))

    assert suite["summary"] == {
        "total": 11,
        "passed": 11,
        "failed": 0,
        "failed_cases": [],
        "selected_cases": [
            "moved_mug_next_action_validity",
            "moved_mug_object_timeline",
            "moved_mug_recent_events",
            "moved_mug_scene_delta",
            "moved_mug_scene_delta_reversed_window_error",
            "moved_mug_world_state",
            "multi_room_rearrangement_object_room_cereal_box",
            "multi_room_rearrangement_object_room_milk",
            "multi_room_rearrangement_recent_events",
            "multi_room_rearrangement_scene_delta",
            "relation_shift_relation_timeline",
        ],
    }
    assert [result["case"] for result in suite["results"]] == [
        "moved_mug_next_action_validity",
        "moved_mug_object_timeline",
        "moved_mug_recent_events",
        "moved_mug_scene_delta",
        "moved_mug_scene_delta_reversed_window_error",
        "moved_mug_world_state",
        "multi_room_rearrangement_object_room_cereal_box",
        "multi_room_rearrangement_object_room_milk",
        "multi_room_rearrangement_recent_events",
        "multi_room_rearrangement_scene_delta",
        "relation_shift_relation_timeline",
    ]


def test_run_evaluation_suite_filters_by_kinds() -> None:
    suite = run_evaluation_suite(kinds=("vla_pick", "vla_place_relative"))

    assert suite["summary"] == {
        "total": 20,
        "passed": 20,
        "failed": 0,
        "failed_cases": [],
        "selected_cases": [
            "ambiguous_mug_pick_by_label",
            "ambiguous_plate_place_reference_by_label",
            "needs_reobserve_bowl_pick_target_not_visible",
            "needs_reobserve_bowl_place_reference_target_not_visible",
            "needs_reobserve_bowl_place_target_not_visible",
            "needs_reobserve_cup_pick_low_confidence",
            "needs_reobserve_cup_place_reference_low_confidence",
            "needs_reobserve_cup_place_target_low_confidence",
            "needs_reobserve_spoon_pick",
            "needs_reobserve_spoon_place_reference",
            "needs_reobserve_spoon_place_target",
            "tabletop_missing_label_pick_error",
            "tabletop_missing_object_pick_error",
            "tabletop_missing_reference_input_place_error",
            "tabletop_missing_reference_place_error",
            "tabletop_missing_target_input_place_error",
            "tabletop_missing_target_pick_error",
            "tabletop_mug_pick",
            "tabletop_place_mug_right_of_plate",
            "tabletop_unsupported_place_relation_error",
        ],
    }
    assert suite["breakdown"]["by_kind"] == {
        "vla_pick": {
            "total": 8,
            "passed": 8,
            "failed": 0,
            "failed_cases": [],
            "selected_cases": [
                "ambiguous_mug_pick_by_label",
                "needs_reobserve_bowl_pick_target_not_visible",
                "needs_reobserve_cup_pick_low_confidence",
                "needs_reobserve_spoon_pick",
                "tabletop_missing_label_pick_error",
                "tabletop_missing_object_pick_error",
                "tabletop_missing_target_pick_error",
                "tabletop_mug_pick",
            ],
        },
        "vla_place_relative": {
            "total": 12,
            "passed": 12,
            "failed": 0,
            "failed_cases": [],
            "selected_cases": [
                "ambiguous_plate_place_reference_by_label",
                "needs_reobserve_bowl_place_reference_target_not_visible",
                "needs_reobserve_bowl_place_target_not_visible",
                "needs_reobserve_cup_place_reference_low_confidence",
                "needs_reobserve_cup_place_target_low_confidence",
                "needs_reobserve_spoon_place_reference",
                "needs_reobserve_spoon_place_target",
                "tabletop_missing_reference_input_place_error",
                "tabletop_missing_reference_place_error",
                "tabletop_missing_target_input_place_error",
                "tabletop_place_mug_right_of_plate",
                "tabletop_unsupported_place_relation_error",
            ],
        },
    }


def test_run_evaluation_suite_summarizes_runtime_error_categories() -> None:
    suite = run_evaluation_suite(tags=("vla", "error"))

    assert suite["runtime_error_categories"] == [
        {
            "category": "low_confidence",
            "count": 3,
            "cases": [
                "needs_reobserve_cup_pick_low_confidence",
                "needs_reobserve_cup_place_reference_low_confidence",
                "needs_reobserve_cup_place_target_low_confidence",
            ],
        },
        {
            "category": "missing_label",
            "count": 1,
            "cases": ["tabletop_missing_label_pick_error"],
        },
        {
            "category": "missing_object",
            "count": 2,
            "cases": [
                "tabletop_missing_object_pick_error",
                "tabletop_missing_reference_place_error",
            ],
        },
        {
            "category": "missing_reference",
            "count": 1,
            "cases": ["tabletop_missing_reference_input_place_error"],
        },
        {
            "category": "missing_target",
            "count": 2,
            "cases": [
                "tabletop_missing_target_input_place_error",
                "tabletop_missing_target_pick_error",
            ],
        },
        {
            "category": "target_not_visible",
            "count": 3,
            "cases": [
                "needs_reobserve_bowl_pick_target_not_visible",
                "needs_reobserve_bowl_place_reference_target_not_visible",
                "needs_reobserve_bowl_place_target_not_visible",
            ],
        },
        {
            "category": "unsupported_relation",
            "count": 1,
            "cases": ["tabletop_unsupported_place_relation_error"],
        },
    ]
    assert evaluation_report(suite)["runtime_error_categories"] == suite[
        "runtime_error_categories"
    ]


def test_run_evaluation_suite_filters_by_question_type() -> None:
    suite = run_evaluation_suite(question_types=("nearest_object",))

    assert suite["summary"] == {
        "total": 1,
        "passed": 1,
        "failed": 0,
        "failed_cases": [],
        "selected_cases": ["tabletop_nearest_candidate_plate"],
    }
    assert [result["actual"]["answer"]["nearest_object"] for result in suite["results"]] == [
        "plate_1"
    ]

    room_suite = run_evaluation_suite(question_types=("object_room",))

    assert room_suite["summary"] == {
        "total": 2,
        "passed": 2,
        "failed": 0,
        "failed_cases": [],
        "selected_cases": [
            "multi_room_rearrangement_object_room_cereal_box",
            "multi_room_rearrangement_object_room_milk",
        ],
    }
    assert room_suite["results"][0]["actual"]["answer"]["room_id"] == "pantry"
    assert room_suite["results"][1]["actual"]["answer"]["room_id"] == "kitchen"


def test_run_evaluation_suite_filters_case_names_by_tags_and_kinds() -> None:
    suite = run_evaluation_suite(
        [
            "moved_mug_recent_events",
            "moved_mug_stale_pick",
            "tabletop_mug_pick",
            "tabletop_object_location",
        ],
        tags=("vla",),
        kinds=("vla_pick",),
    )

    assert suite["summary"] == {
        "total": 1,
        "passed": 1,
        "failed": 0,
        "failed_cases": [],
        "selected_cases": ["tabletop_mug_pick"],
    }
    assert [result["kind"] for result in suite["results"]] == ["vla_pick"]


def test_run_evaluation_suite_filters_by_names_keyword_in_call_order() -> None:
    suite = run_evaluation_suite(
        names=("tabletop_object_location", "moved_mug_recent_events"),
    )

    assert suite["summary"]["selected_cases"] == [
        "tabletop_object_location",
        "moved_mug_recent_events",
    ]
    assert [result["case"] for result in suite["results"]] == [
        "tabletop_object_location",
        "moved_mug_recent_events",
    ]


def test_run_evaluation_suite_includes_stable_digest_for_experiment_records() -> None:
    suite = run_evaluation_suite(
        names=("tabletop_object_location", "moved_mug_recent_events"),
    )
    repeated = run_evaluation_suite(
        names=("tabletop_object_location", "moved_mug_recent_events"),
    )
    reordered = run_evaluation_suite(
        names=("moved_mug_recent_events", "tabletop_object_location"),
    )

    digest_payload = {
        "summary": suite["summary"],
        "breakdown": suite["breakdown"],
        "runtime_error_categories": suite["runtime_error_categories"],
        "results": suite["results"],
    }
    expected_digest = hashlib.sha256(
        json.dumps(
            digest_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert suite["digest"] == expected_digest
    assert len(suite["digest"]) == 64
    assert repeated["digest"] == suite["digest"]
    assert reordered["digest"] != suite["digest"]


def test_run_custom_evaluation_case_without_registry() -> None:
    case = EvaluationCase(
        name="custom_plate_status",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "custom"),
        question={"type": "object_status", "object_id": "plate_1"},
        expected={
            "answer": {
                "object_id": "plate_1",
                "label": "plate",
                "visible": True,
                "needs_reobserve": False,
            },
            "error": None,
        },
    )

    result = run_evaluation_case_definition(case)

    assert result["case"] == "custom_plate_status"
    assert result["passed"] is True
    assert result["actual"]["answer"]["confidence"] == 0.9
    assert "custom_plate_status" not in list_evaluation_cases()


def test_run_custom_evaluation_cases_filters_by_tags() -> None:
    qa_case = EvaluationCase(
        name="custom_plate_status",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "custom"),
        question={"type": "object_status", "object_id": "plate_1"},
        expected={"answer": {"object_id": "plate_1"}, "error": None},
    )
    vla_case = EvaluationCase(
        name="custom_moved_mug_stale_pick",
        scene_fixture="moved_mug",
        baseline_scene_fixture="tabletop",
        kind="vla_stale_pick",
        tags=("vla", "custom"),
        target_object="mug_1",
        expected={"status": "needs_replan"},
    )

    suite = run_evaluation_cases((vla_case, qa_case), tags=("qa", "custom"))

    assert suite["summary"] == {
        "total": 1,
        "passed": 1,
        "failed": 0,
        "failed_cases": [],
        "selected_cases": ["custom_plate_status"],
    }
    assert [result["case"] for result in suite["results"]] == ["custom_plate_status"]
    assert "custom_moved_mug_stale_pick" not in [
        result["case"] for result in suite["results"]
    ]


def test_run_custom_evaluation_cases_filters_by_kinds() -> None:
    qa_case = EvaluationCase(
        name="custom_plate_status",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "custom"),
        question={"type": "object_status", "object_id": "plate_1"},
        expected={"answer": {"object_id": "plate_1"}, "error": None},
    )
    vla_case = EvaluationCase(
        name="custom_mug_pick",
        scene_fixture="tabletop",
        kind="vla_pick",
        tags=("vla", "custom"),
        target_object="mug_1",
        expected={"status": "ok"},
    )

    suite = run_evaluation_cases((vla_case, qa_case), kinds=("vla_pick",))

    assert suite["summary"] == {
        "total": 1,
        "passed": 1,
        "failed": 0,
        "failed_cases": [],
        "selected_cases": ["custom_mug_pick"],
    }
    assert suite["breakdown"]["by_kind"] == {
        "vla_pick": {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "failed_cases": [],
            "selected_cases": ["custom_mug_pick"],
        }
    }


def test_run_custom_evaluation_cases_filters_by_question_type() -> None:
    qa_case = EvaluationCase(
        name="custom_plate_status",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "custom"),
        question={"type": "object_status", "object_id": "plate_1"},
        expected={"answer": {"object_id": "plate_1"}, "error": None},
    )
    nearest_case = EvaluationCase(
        name="custom_nearest_plate",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "custom"),
        question={"type": "nearest_object", "src": "mug_1", "candidates": ["plate_1"]},
        expected={"answer": {"nearest_object": "plate_1"}, "error": None},
    )

    suite = run_evaluation_cases(
        (qa_case, nearest_case),
        question_types=("nearest_object",),
    )

    assert suite["summary"]["selected_cases"] == ["custom_nearest_plate"]
    assert suite["summary"]["total"] == 1


def test_run_custom_evaluation_cases_filters_by_explicit_names_in_call_order() -> None:
    qa_case = EvaluationCase(
        name="custom_plate_status",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "custom"),
        question={"type": "object_status", "object_id": "plate_1"},
        expected={"answer": {"object_id": "plate_1"}, "error": None},
    )
    vla_case = EvaluationCase(
        name="custom_mug_pick",
        scene_fixture="tabletop",
        kind="vla_pick",
        tags=("vla", "custom"),
        target_object="mug_1",
        expected={"status": "ok"},
    )

    suite = run_evaluation_cases(
        (vla_case, qa_case),
        names=("custom_plate_status", "custom_mug_pick"),
    )

    assert suite["summary"]["selected_cases"] == [
        "custom_plate_status",
        "custom_mug_pick",
    ]
    assert [result["case"] for result in suite["results"]] == [
        "custom_plate_status",
        "custom_mug_pick",
    ]


def test_custom_evaluation_case_metadata_filters_by_explicit_names_in_call_order() -> None:
    qa_case = EvaluationCase(
        name="custom_plate_status",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "custom"),
        question={"type": "object_status", "object_id": "plate_1"},
        expected={"answer": {"object_id": "plate_1"}, "error": None},
    )
    vla_case = EvaluationCase(
        name="custom_mug_pick",
        scene_fixture="tabletop",
        kind="vla_pick",
        tags=("vla", "custom"),
        target_object="mug_1",
        expected={"status": "ok"},
    )

    manifest = evaluation_cases_metadata(
        (vla_case, qa_case),
        names=("custom_plate_status", "custom_mug_pick"),
    )

    assert [item["name"] for item in manifest] == [
        "custom_plate_status",
        "custom_mug_pick",
    ]


def test_run_custom_evaluation_case_with_scene_loader() -> None:
    case = EvaluationCase(
        name="custom_cube_location",
        scene_fixture="custom_counter",
        kind="qa",
        tags=("qa", "custom-scene"),
        question={"type": "object_location", "object_id": "cube_1"},
        expected={
            "answer": {
                "object_id": "cube_1",
                "pose": {"x": 0.25, "y": -0.5, "z": 0.4, "yaw": 0.0},
                "visible": True,
                "last_seen_step": 4,
            },
            "error": None,
        },
    )

    result = run_evaluation_case_definition(
        case,
        scene_loaders={"custom_counter": _build_custom_counter_scene},
    )

    assert result["passed"] is True
    assert result["scene_fixture"] == "custom_counter"
    assert result["actual"]["confidence"] == 0.82


def test_run_custom_evaluation_cases_with_scene_loaders_and_tags() -> None:
    qa_case = EvaluationCase(
        name="custom_cube_status",
        scene_fixture="custom_counter",
        kind="qa",
        tags=("qa", "custom-scene"),
        question={"type": "object_status", "object_id": "cube_1"},
        expected={"answer": {"object_id": "cube_1", "needs_reobserve": False}},
    )
    skipped_case = EvaluationCase(
        name="custom_cube_location_skipped",
        scene_fixture="custom_counter",
        kind="qa",
        tags=("qa", "other"),
        question={"type": "object_location", "object_id": "cube_1"},
        expected={"error": None},
    )

    suite = run_evaluation_cases(
        (skipped_case, qa_case),
        tags=("qa", "custom-scene"),
        scene_loaders={"custom_counter": _build_custom_counter_scene},
    )

    assert suite["summary"] == {
        "total": 1,
        "passed": 1,
        "failed": 0,
        "failed_cases": [],
        "selected_cases": ["custom_cube_status"],
    }
    assert suite["breakdown"]["by_scene_fixture"] == {
        "custom_counter": {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "failed_cases": [],
            "selected_cases": ["custom_cube_status"],
        }
    }
    assert [result["case"] for result in suite["results"]] == ["custom_cube_status"]


def test_failed_evaluation_case_reports_deterministic_mismatches() -> None:
    case = EvaluationCase(
        name="custom_bad_expectation",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "diagnostics"),
        question={"type": "object_location", "object_id": "mug_1"},
        expected={
            "answer": {
                "object_id": "mug_1",
                "visible": False,
                "missing_field": "expected",
            },
            "evidence_edges": {},
            "evidence_nodes": ["mug_1", "plate_1", "table_1"],
            "error": None,
        },
    )

    result = run_evaluation_case_definition(case)

    assert result["passed"] is False
    assert result["mismatches"] == [
        {
            "path": "answer.missing_field",
            "reason": "missing_actual_key",
            "category": "missing_output",
            "expected": "expected",
            "actual": None,
        },
        {
            "path": "answer.visible",
            "reason": "value_mismatch",
            "category": "value_mismatch",
            "expected": False,
            "actual": True,
        },
        {
            "path": "evidence_edges",
            "reason": "type_mismatch",
            "category": "schema_mismatch",
            "expected": "mapping",
            "actual": "list",
        },
        {
                "path": "evidence_nodes",
                "reason": "sequence_length_mismatch",
                "category": "cardinality_mismatch",
                "expected": 3,
                "actual": 2,
            },
        ]


def test_evaluation_suite_summary_lists_failed_cases_deterministically() -> None:
    passing_case = EvaluationCase(
        name="custom_plate_status",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "summary"),
        question={"type": "object_status", "object_id": "plate_1"},
        expected={"answer": {"object_id": "plate_1"}, "error": None},
    )
    failing_case = EvaluationCase(
        name="custom_bad_expectation",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "summary"),
        question={"type": "object_location", "object_id": "mug_1"},
        expected={"answer": {"visible": False}, "error": None},
    )

    suite = run_evaluation_cases((passing_case, failing_case), tags=("qa", "summary"))

    assert suite["summary"] == {
        "total": 2,
        "passed": 1,
        "failed": 1,
        "failed_cases": ["custom_bad_expectation"],
        "selected_cases": ["custom_bad_expectation", "custom_plate_status"],
    }
    assert suite["breakdown"]["by_kind"] == {
        "qa": {
            "total": 2,
            "passed": 1,
            "failed": 1,
            "failed_cases": ["custom_bad_expectation"],
            "selected_cases": ["custom_bad_expectation", "custom_plate_status"],
        }
    }
    assert suite["breakdown"]["by_question_type"] == {
        "object_location": {
            "total": 1,
            "passed": 0,
            "failed": 1,
            "failed_cases": ["custom_bad_expectation"],
            "selected_cases": ["custom_bad_expectation"],
        },
        "object_status": {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "failed_cases": [],
            "selected_cases": ["custom_plate_status"],
        },
    }
    assert suite["breakdown"]["by_tag"] == {
        "qa": {
            "total": 2,
            "passed": 1,
            "failed": 1,
            "failed_cases": ["custom_bad_expectation"],
            "selected_cases": ["custom_bad_expectation", "custom_plate_status"],
        },
        "summary": {
            "total": 2,
            "passed": 1,
            "failed": 1,
            "failed_cases": ["custom_bad_expectation"],
            "selected_cases": ["custom_bad_expectation", "custom_plate_status"],
        },
    }
    assert [result["case"] for result in suite["results"]] == [
        "custom_bad_expectation",
        "custom_plate_status",
    ]


def test_evaluation_report_summarizes_metrics_and_failure_reasons() -> None:
    passing_case = EvaluationCase(
        name="custom_plate_status",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "report"),
        question={"type": "object_status", "object_id": "plate_1"},
        expected={"answer": {"object_id": "plate_1"}, "error": None},
    )
    failing_case = EvaluationCase(
        name="custom_bad_expectation",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "report"),
        question={"type": "object_location", "object_id": "mug_1"},
        expected={"answer": {"visible": False}, "error": None},
    )
    suite = run_evaluation_cases((passing_case, failing_case), tags=("qa", "report"))
    expected_case_digests = [
        {
            "case": result["case"],
            "kind": result["kind"],
            "question_type": result["question_type"],
            "scene_fixture": result["scene_fixture"],
            "passed": result["passed"],
            "digest": hashlib.sha256(
                json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }
        for result in suite["results"]
    ]
    expected_case_selection = [
        {
            "case": result["case"],
            "kind": result["kind"],
            "question_type": result["question_type"],
            "scene_fixture": result["scene_fixture"],
            "tags": result["tags"],
        }
        for result in suite["results"]
    ]
    expected_case_selection_digest = hashlib.sha256(
        json.dumps(
            {
                "case_count": len(expected_case_selection),
                "case_selection": expected_case_selection,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    report = evaluation_report(suite)

    expected_report = {
        "schema_version": "dsg-spatialqa-lab.evaluation-report.v1",
        "digest": suite["digest"],
        "summary": suite["summary"],
        "metrics": {
            "case_count": 2,
            "passed_case_count": 1,
            "failed_case_count": 1,
            "pass_rate": 0.5,
            "failure_rate": 0.5,
            "by_kind": {
                "qa": {
                    "case_count": 2,
                    "passed_case_count": 1,
                    "failed_case_count": 1,
                    "pass_rate": 0.5,
                    "failure_rate": 0.5,
                }
            },
            "by_question_type": {
                "object_location": {
                    "case_count": 1,
                    "passed_case_count": 0,
                    "failed_case_count": 1,
                    "pass_rate": 0.0,
                    "failure_rate": 1.0,
                },
                "object_status": {
                    "case_count": 1,
                    "passed_case_count": 1,
                    "failed_case_count": 0,
                    "pass_rate": 1.0,
                    "failure_rate": 0.0,
                },
            },
            "by_scene_fixture": {
                "tabletop": {
                    "case_count": 2,
                    "passed_case_count": 1,
                    "failed_case_count": 1,
                    "pass_rate": 0.5,
                    "failure_rate": 0.5,
                }
            },
            "by_tag": {
                "qa": {
                    "case_count": 2,
                    "passed_case_count": 1,
                    "failed_case_count": 1,
                    "pass_rate": 0.5,
                    "failure_rate": 0.5,
                },
                "report": {
                    "case_count": 2,
                    "passed_case_count": 1,
                    "failed_case_count": 1,
                    "pass_rate": 0.5,
                    "failure_rate": 0.5,
                },
            },
        },
        "evidence_metrics": {
            "case_count": 2,
            "cases_with_evidence_count": 2,
            "cases_without_evidence_count": 0,
            "evidence_node_count": 3,
            "evidence_edge_count": 3,
            "command_evidence_count": 0,
            "total_evidence_item_count": 6,
            "average_evidence_item_count": 3.0,
            "by_kind": {
                "qa": {
                    "case_count": 2,
                    "cases_with_evidence_count": 2,
                    "cases_without_evidence_count": 0,
                    "evidence_node_count": 3,
                    "evidence_edge_count": 3,
                    "command_evidence_count": 0,
                    "total_evidence_item_count": 6,
                    "average_evidence_item_count": 3.0,
                }
            },
            "by_question_type": {
                "object_location": {
                    "case_count": 1,
                    "cases_with_evidence_count": 1,
                    "cases_without_evidence_count": 0,
                    "evidence_node_count": 2,
                    "evidence_edge_count": 2,
                    "command_evidence_count": 0,
                    "total_evidence_item_count": 4,
                    "average_evidence_item_count": 4.0,
                },
                "object_status": {
                    "case_count": 1,
                    "cases_with_evidence_count": 1,
                    "cases_without_evidence_count": 0,
                    "evidence_node_count": 1,
                    "evidence_edge_count": 1,
                    "command_evidence_count": 0,
                    "total_evidence_item_count": 2,
                    "average_evidence_item_count": 2.0,
                },
            },
            "by_scene_fixture": {
                "tabletop": {
                    "case_count": 2,
                    "cases_with_evidence_count": 2,
                    "cases_without_evidence_count": 0,
                    "evidence_node_count": 3,
                    "evidence_edge_count": 3,
                    "command_evidence_count": 0,
                    "total_evidence_item_count": 6,
                    "average_evidence_item_count": 3.0,
                }
            },
            "by_tag": {
                "qa": {
                    "case_count": 2,
                    "cases_with_evidence_count": 2,
                    "cases_without_evidence_count": 0,
                    "evidence_node_count": 3,
                    "evidence_edge_count": 3,
                    "command_evidence_count": 0,
                    "total_evidence_item_count": 6,
                    "average_evidence_item_count": 3.0,
                },
                "report": {
                    "case_count": 2,
                    "cases_with_evidence_count": 2,
                    "cases_without_evidence_count": 0,
                    "evidence_node_count": 3,
                    "evidence_edge_count": 3,
                    "command_evidence_count": 0,
                    "total_evidence_item_count": 6,
                    "average_evidence_item_count": 3.0,
                },
            },
        },
        "case_selection": expected_case_selection,
        "case_selection_digest": expected_case_selection_digest,
        "case_digests": expected_case_digests,
        "failed_cases": [
            {
                "case": "custom_bad_expectation",
                "kind": "qa",
                "scene_fixture": "tabletop",
                "tags": ["qa", "report"],
                "mismatch_count": 1,
                "mismatch_paths": ["answer.visible"],
                "mismatch_reasons": ["value_mismatch"],
                "mismatch_categories": ["value_mismatch"],
                "error": None,
            }
        ],
        "runtime_error_categories": [],
        "runtime_error_metrics": {
            "case_count": 2,
            "cases_with_runtime_error_count": 0,
            "cases_without_runtime_error_count": 2,
            "runtime_error_rate": 0.0,
            "by_category": {},
        },
        "failure_reasons": [
            {
                "reason": "value_mismatch",
                "count": 1,
                "cases": ["custom_bad_expectation"],
            }
        ],
        "failure_categories": [
            {
                "category": "value_mismatch",
                "count": 1,
                "cases": ["custom_bad_expectation"],
            }
        ],
        "failure_paths": [
            {
                "path": "answer.visible",
                "count": 1,
                "cases": ["custom_bad_expectation"],
            }
        ],
        "breakdown": suite["breakdown"],
    }
    assert report == {
        **expected_report,
        "report_digest": lab.evaluation_report_digest(expected_report),
    }


def test_evaluation_report_groups_failure_categories_for_structural_mismatches() -> None:
    failing_case = EvaluationCase(
        name="custom_structural_bad_expectation",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "report"),
            question={"type": "object_location", "object_id": "mug_1"},
            expected={
                "answer": {"missing_field": "expected"},
                "evidence_nodes": ["mug_1", "plate_1", "table_1"],
                "error": None,
            },
        )
    suite = run_evaluation_cases((failing_case,), tags=("qa", "report"))

    report = evaluation_report(suite)

    assert report["failed_cases"] == [
        {
            "case": "custom_structural_bad_expectation",
            "kind": "qa",
            "scene_fixture": "tabletop",
            "tags": ["qa", "report"],
            "mismatch_count": 2,
            "mismatch_paths": ["answer.missing_field", "evidence_nodes"],
            "mismatch_reasons": [
                "missing_actual_key",
                "sequence_length_mismatch",
            ],
            "mismatch_categories": [
                "cardinality_mismatch",
                "missing_output",
            ],
            "error": None,
        }
    ]
    assert report["failure_categories"] == [
        {
            "category": "cardinality_mismatch",
            "count": 1,
            "cases": ["custom_structural_bad_expectation"],
        },
        {
            "category": "missing_output",
            "count": 1,
            "cases": ["custom_structural_bad_expectation"],
        },
    ]
    assert report["failure_paths"] == [
        {
            "path": "answer.missing_field",
            "count": 1,
            "cases": ["custom_structural_bad_expectation"],
        },
        {
            "path": "evidence_nodes",
            "count": 1,
            "cases": ["custom_structural_bad_expectation"],
        },
    ]


def test_evaluation_report_summarizes_runtime_error_metrics() -> None:
    suite = run_evaluation_suite(tags=("vla", "error"))

    report = evaluation_report(suite)

    assert report["runtime_error_metrics"] == {
        "case_count": 13,
        "cases_with_runtime_error_count": 13,
        "cases_without_runtime_error_count": 0,
        "runtime_error_rate": 1.0,
        "by_category": {
            "low_confidence": {
                "case_count": 3,
                "case_rate": 3 / 13,
            },
            "missing_label": {
                "case_count": 1,
                "case_rate": 1 / 13,
            },
            "missing_object": {
                "case_count": 2,
                "case_rate": 2 / 13,
            },
            "missing_reference": {
                "case_count": 1,
                "case_rate": 1 / 13,
            },
            "missing_target": {
                "case_count": 2,
                "case_rate": 2 / 13,
            },
            "target_not_visible": {
                "case_count": 3,
                "case_rate": 3 / 13,
            },
            "unsupported_relation": {
                "case_count": 1,
                "case_rate": 1 / 13,
            },
        },
    }


def test_evaluation_report_json_is_stable_and_savable(tmp_path: Path) -> None:
    suite = run_evaluation_suite(names=("tabletop_object_location",))
    report = evaluation_report(suite)

    payload = evaluation_report_json(report)
    repeated = evaluation_report_json(report)
    report_path = tmp_path / "report.json"
    saved_path = save_evaluation_report(report_path, suite)

    assert payload == repeated
    assert payload.endswith("\n")
    assert json.loads(payload) == report
    assert saved_path == report_path
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_evaluation_report_includes_stable_report_digest_and_validates_tampering() -> None:
    assert hasattr(lab, "evaluation_report_digest")
    assert hasattr(lab, "validate_evaluation_report")
    suite = run_evaluation_suite(names=("tabletop_object_location",))
    report = evaluation_report(suite)
    report_without_digest = {
        key: value for key, value in report.items() if key != "report_digest"
    }
    expected_case_selection_digest = hashlib.sha256(
        json.dumps(
            {
                "case_count": len(report["case_selection"]),
                "case_selection": report["case_selection"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    expected_report_digest = hashlib.sha256(
        json.dumps(report_without_digest, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()

    assert report["case_selection_digest"] == expected_case_selection_digest
    assert report["report_digest"] == expected_report_digest
    assert lab.evaluation_report_digest(report) == expected_report_digest
    validation = lab.validate_evaluation_report(report)
    checks = {check["name"]: check for check in validation["checks"]}
    assert validation["digest"] == report["digest"]
    assert validation["schema_version"] == "dsg-spatialqa-lab.evaluation-report.v1"
    assert validation["report_digest"] == expected_report_digest
    assert checks["schema_version"] == {
        "name": "schema_version",
        "passed": True,
        "expected": "dsg-spatialqa-lab.evaluation-report.v1",
        "actual": "dsg-spatialqa-lab.evaluation-report.v1",
    }
    assert checks["case_selection_digest"] == {
        "name": "case_selection_digest",
        "passed": True,
        "expected": expected_case_selection_digest,
        "actual": expected_case_selection_digest,
    }
    assert checks["report_digest"] == {
        "name": "report_digest",
        "passed": True,
        "expected": expected_report_digest,
        "actual": expected_report_digest,
    }

    tampered_report = json.loads(evaluation_report_json(report))
    tampered_report["metrics"]["pass_rate"] = 0.5

    tampered_validation = lab.validate_evaluation_report(tampered_report)
    tampered_checks = {check["name"]: check for check in tampered_validation["checks"]}
    assert tampered_validation["valid"] is False
    assert tampered_checks["report_digest"] == {
        "name": "report_digest",
        "passed": False,
        "expected": lab.evaluation_report_digest(tampered_report),
        "actual": expected_report_digest,
    }


def test_evaluation_report_validation_checks_metrics_against_summary() -> None:
    suite = run_evaluation_suite(names=("tabletop_object_location",))
    report = evaluation_report(suite)
    tampered_report = json.loads(evaluation_report_json(report))
    tampered_report["metrics"]["case_count"] = 99
    tampered_report["report_digest"] = lab.evaluation_report_digest(tampered_report)

    validation = lab.validate_evaluation_report(tampered_report)
    checks = {check["name"]: check for check in validation["checks"]}

    assert validation["valid"] is False
    assert checks["report_digest"]["passed"] is True
    assert checks["metrics_match_summary"] == {
        "name": "metrics_match_summary",
        "passed": False,
        "expected": report["metrics"],
        "actual": tampered_report["metrics"],
    }


def test_evaluation_report_validation_checks_evidence_metrics_consistency() -> None:
    suite = run_evaluation_suite(names=("tabletop_object_location",))
    report = evaluation_report(suite)
    tampered_report = json.loads(evaluation_report_json(report))
    tampered_report["evidence_metrics"]["evidence_node_count"] = 99
    tampered_report["report_digest"] = lab.evaluation_report_digest(tampered_report)

    validation = lab.validate_evaluation_report(tampered_report)
    checks = {check["name"]: check for check in validation["checks"]}

    assert validation["valid"] is False
    assert checks["report_digest"]["passed"] is True
    assert checks["evidence_metrics_consistent"] == {
        "name": "evidence_metrics_consistent",
        "passed": False,
        "expected": {
            "average_evidence_item_count": 101.0,
            "case_count": 1,
            "cases_with_evidence_count": 1,
            "cases_without_evidence_count": 0,
            "command_evidence_count": 0,
            "evidence_edge_count": 2,
            "evidence_node_count": 99,
            "total_evidence_item_count": 101,
        },
        "actual": {
            "average_evidence_item_count": 4.0,
            "case_count": 1,
            "cases_with_evidence_count": 1,
            "cases_without_evidence_count": 0,
            "command_evidence_count": 0,
            "evidence_edge_count": 2,
            "evidence_node_count": 99,
            "total_evidence_item_count": 4,
        },
    }


def test_evaluation_report_validation_checks_evidence_metric_value_ranges() -> None:
    suite = run_evaluation_suite(names=("tabletop_object_location",))
    report = evaluation_report(suite)
    tampered_report = json.loads(evaluation_report_json(report))
    tampered_report["evidence_metrics"]["evidence_node_count"] = -1
    tampered_report["evidence_metrics"]["total_evidence_item_count"] = 1
    tampered_report["evidence_metrics"]["average_evidence_item_count"] = 1.0
    tampered_report["report_digest"] = lab.evaluation_report_digest(tampered_report)

    validation = lab.validate_evaluation_report(tampered_report)
    checks = {check["name"]: check for check in validation["checks"]}

    assert validation["valid"] is False
    assert checks["report_digest"]["passed"] is True
    assert checks["evidence_metrics_consistent"]["passed"] is True
    assert checks["evidence_metric_groups_consistent"]["passed"] is True
    assert checks["evidence_metric_values_valid"] == {
        "name": "evidence_metric_values_valid",
        "passed": False,
        "differences": [
            {
                "path": "evidence_metrics.evidence_node_count",
                "expected": "non-negative integer",
                "actual": -1,
            },
        ],
    }


def test_evaluation_report_validation_checks_grouped_evidence_metrics_consistency() -> None:
    suite = run_evaluation_suite(names=("tabletop_object_location",))
    report = evaluation_report(suite)
    tampered_report = json.loads(evaluation_report_json(report))
    tampered_report["evidence_metrics"]["by_tag"]["qa"]["evidence_node_count"] = 99
    tampered_report["report_digest"] = lab.evaluation_report_digest(tampered_report)

    validation = lab.validate_evaluation_report(tampered_report)
    checks = {check["name"]: check for check in validation["checks"]}

    assert validation["valid"] is False
    assert checks["report_digest"]["passed"] is True
    grouped_check = checks["evidence_metric_groups_consistent"]
    assert grouped_check["name"] == "evidence_metric_groups_consistent"
    assert grouped_check["passed"] is False
    assert grouped_check["differences"] == [
        {
            "path": "by_tag.qa.average_evidence_item_count",
            "expected": 101.0,
            "actual": 4.0,
        },
        {
            "path": "by_tag.qa.total_evidence_item_count",
            "expected": 101,
            "actual": 4,
        },
    ]


def test_evaluation_report_validation_checks_runtime_error_category_counts() -> None:
    suite = run_evaluation_suite(tags=("vla", "error"))
    report = evaluation_report(suite)
    tampered_report = json.loads(evaluation_report_json(report))
    tampered_report["runtime_error_categories"][0]["count"] = 99
    tampered_report["report_digest"] = lab.evaluation_report_digest(tampered_report)

    validation = lab.validate_evaluation_report(tampered_report)
    checks = {check["name"]: check for check in validation["checks"]}

    assert validation["valid"] is False
    assert checks["report_digest"]["passed"] is True
    category_check = checks["runtime_error_categories_consistent"]
    assert category_check["name"] == "runtime_error_categories_consistent"
    assert category_check["passed"] is False
    assert category_check["differences"] == [
        {
            "path": "low_confidence.count",
            "expected": 3,
            "actual": 99,
        },
    ]


def test_evaluation_report_validation_checks_runtime_error_category_entry_shape() -> None:
    suite = run_evaluation_suite(names=("tabletop_object_location",))
    report = evaluation_report(suite)
    tampered_report = json.loads(evaluation_report_json(report))
    tampered_report["runtime_error_categories"] = [
        {
            "category": "ghost_runtime_error",
            "count": 0,
            "cases": [],
        }
    ]
    tampered_report["report_digest"] = lab.evaluation_report_digest(tampered_report)

    validation = lab.validate_evaluation_report(tampered_report)
    checks = {check["name"]: check for check in validation["checks"]}

    assert validation["valid"] is False
    assert checks["report_digest"]["passed"] is True
    assert checks["runtime_error_categories_consistent"]["passed"] is True
    assert checks["runtime_error_category_entries_valid"] == {
        "name": "runtime_error_category_entries_valid",
        "passed": False,
        "differences": [
            {
                "path": "runtime_error_categories[0].count",
                "expected": "positive integer",
                "actual": 0,
            },
        ],
    }


def test_evaluation_report_validation_checks_runtime_error_metrics() -> None:
    suite = run_evaluation_suite(tags=("vla", "error"))
    report = evaluation_report(suite)
    tampered_report = json.loads(evaluation_report_json(report))
    tampered_report["runtime_error_metrics"]["cases_with_runtime_error_count"] = 0
    tampered_report["runtime_error_metrics"]["runtime_error_rate"] = 0.0
    tampered_report["report_digest"] = lab.evaluation_report_digest(tampered_report)

    validation = lab.validate_evaluation_report(tampered_report)
    checks = {check["name"]: check for check in validation["checks"]}

    assert validation["valid"] is False
    assert checks["report_digest"]["passed"] is True
    metrics_check = checks["runtime_error_metrics_consistent"]
    assert metrics_check["name"] == "runtime_error_metrics_consistent"
    assert metrics_check["passed"] is False
    assert metrics_check["differences"] == [
        {
            "path": "cases_with_runtime_error_count",
            "expected": 13,
            "actual": 0,
        },
        {
            "path": "runtime_error_rate",
            "expected": 1.0,
            "actual": 0.0,
        },
    ]


def test_evaluation_report_validation_checks_failure_diagnostic_aggregates() -> None:
    failing_case = EvaluationCase(
        name="custom_bad_expectation",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "report"),
        question={"type": "object_location", "object_id": "mug_1"},
        expected={"answer": {"visible": False}, "error": None},
    )
    suite = run_evaluation_cases((failing_case,), tags=("qa", "report"))
    report = evaluation_report(suite)
    tampered_report = json.loads(evaluation_report_json(report))
    tampered_report["failure_categories"][0]["count"] = 99
    tampered_report["report_digest"] = lab.evaluation_report_digest(tampered_report)

    validation = lab.validate_evaluation_report(tampered_report)
    checks = {check["name"]: check for check in validation["checks"]}

    assert validation["valid"] is False
    assert checks["report_digest"]["passed"] is True
    diagnostics_check = checks["failure_diagnostics_consistent"]
    assert diagnostics_check["name"] == "failure_diagnostics_consistent"
    assert diagnostics_check["passed"] is False
    assert diagnostics_check["differences"] == [
        {
            "path": "failure_categories.value_mismatch.count",
            "expected": 1,
            "actual": 99,
        },
    ]


def test_evaluation_report_validation_checks_failed_cases_against_summary() -> None:
    failing_case = EvaluationCase(
        name="custom_bad_expectation",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "report"),
        question={"type": "object_location", "object_id": "mug_1"},
        expected={"answer": {"visible": False}, "error": None},
    )
    suite = run_evaluation_cases((failing_case,), tags=("qa", "report"))
    report = evaluation_report(suite)
    tampered_report = json.loads(evaluation_report_json(report))
    tampered_report["summary"]["failed_cases"] = []
    tampered_report["report_digest"] = lab.evaluation_report_digest(tampered_report)

    validation = lab.validate_evaluation_report(tampered_report)
    checks = {check["name"]: check for check in validation["checks"]}

    assert validation["valid"] is False
    assert checks["report_digest"]["passed"] is True
    assert checks["failed_cases_match_summary"] == {
        "name": "failed_cases_match_summary",
        "passed": False,
        "expected": [],
        "actual": ["custom_bad_expectation"],
    }


def test_evaluation_report_validation_checks_failed_case_metadata_shape() -> None:
    failing_case = EvaluationCase(
        name="custom_bad_expectation",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "report"),
        question={"type": "object_location", "object_id": "mug_1"},
        expected={"answer": {"visible": False}, "error": None},
    )
    suite = run_evaluation_cases((failing_case,), tags=("qa", "report"))
    report = evaluation_report(suite)
    tampered_report = json.loads(evaluation_report_json(report))
    tampered_report["failed_cases"][0]["kind"] = ""
    tampered_report["report_digest"] = lab.evaluation_report_digest(tampered_report)

    validation = lab.validate_evaluation_report(tampered_report)
    checks = {check["name"]: check for check in validation["checks"]}

    assert validation["valid"] is False
    assert checks["report_digest"]["passed"] is True
    assert checks["failed_cases_match_summary"]["passed"] is True
    assert checks["failure_diagnostics_consistent"]["passed"] is True
    assert checks["failed_case_entries_valid"] == {
        "name": "failed_case_entries_valid",
        "passed": False,
        "differences": [
            {
                "path": "failed_cases[0].kind",
                "expected": "non-empty string",
                "actual": "",
            },
        ],
    }


def test_evaluation_report_validation_checks_summary_counts_against_case_lists() -> None:
    suite = run_evaluation_suite(names=("tabletop_object_location",))
    report = evaluation_report(suite)
    tampered_report = json.loads(evaluation_report_json(report))
    tampered_report["summary"]["total"] = 2
    tampered_report["summary"]["passed"] = 2
    tampered_report["metrics"]["case_count"] = 2
    tampered_report["metrics"]["passed_case_count"] = 2
    tampered_report["evidence_metrics"]["case_count"] = 2
    tampered_report["evidence_metrics"]["cases_without_evidence_count"] = 1
    tampered_report["evidence_metrics"]["average_evidence_item_count"] = 2.0
    tampered_report["report_digest"] = lab.evaluation_report_digest(tampered_report)

    validation = lab.validate_evaluation_report(tampered_report)
    checks = {check["name"]: check for check in validation["checks"]}

    assert validation["valid"] is False
    assert checks["report_digest"]["passed"] is True
    summary_check = checks["summary_counts_consistent"]
    assert summary_check["name"] == "summary_counts_consistent"
    assert summary_check["passed"] is False
    assert summary_check["expected"] == {
        "failed": 0,
        "passed": 1,
        "total": 1,
    }
    assert summary_check["actual"] == {
        "failed": 0,
        "passed": 2,
        "total": 2,
    }
    assert summary_check["differences"] == [
        {
            "path": "passed",
            "expected": 1,
            "actual": 2,
        },
        {
            "path": "total",
            "expected": 1,
            "actual": 2,
        },
    ]


def test_evaluation_report_validation_checks_summary_failed_cases_are_selected() -> None:
    suite = run_evaluation_suite(names=("tabletop_object_location",))
    report = evaluation_report(suite)
    tampered_report = json.loads(evaluation_report_json(report))
    tampered_report["summary"]["failed_cases"] = ["ghost_failed_case"]
    tampered_report["summary"]["failed"] = 1
    tampered_report["summary"]["passed"] = 0
    tampered_report["metrics"]["failed_case_count"] = 1
    tampered_report["metrics"]["failure_rate"] = 1.0
    tampered_report["metrics"]["passed_case_count"] = 0
    tampered_report["metrics"]["pass_rate"] = 0.0
    tampered_report["failed_cases"] = [
        {
            "case": "ghost_failed_case",
            "kind": "qa",
            "scene_fixture": "tabletop",
            "tags": ["qa", "report"],
            "mismatch_count": 1,
            "mismatch_paths": ["answer.visible"],
            "mismatch_reasons": ["value mismatch"],
            "mismatch_categories": ["value_mismatch"],
            "error": None,
        }
    ]
    tampered_report["failure_reasons"] = [
        {
            "reason": "value mismatch",
            "count": 1,
            "cases": ["ghost_failed_case"],
        }
    ]
    tampered_report["failure_categories"] = [
        {
            "category": "value_mismatch",
            "count": 1,
            "cases": ["ghost_failed_case"],
        }
    ]
    tampered_report["failure_paths"] = [
        {
            "path": "answer.visible",
            "count": 1,
            "cases": ["ghost_failed_case"],
        }
    ]
    tampered_report["report_digest"] = lab.evaluation_report_digest(tampered_report)

    validation = lab.validate_evaluation_report(tampered_report)
    checks = {check["name"]: check for check in validation["checks"]}

    assert validation["valid"] is False
    assert checks["report_digest"]["passed"] is True
    assert checks["failed_cases_match_summary"]["passed"] is True
    assert checks["failed_case_entries_valid"]["passed"] is True
    assert checks["summary_counts_consistent"]["passed"] is True
    assert checks["failure_diagnostics_consistent"]["passed"] is True
    assert checks["summary_case_lists_valid"] == {
        "name": "summary_case_lists_valid",
        "passed": False,
        "differences": [
            {
                "path": "summary.failed_cases[0]",
                "expected": "case from summary.selected_cases",
                "actual": "ghost_failed_case",
            },
        ],
    }


def test_evaluation_report_validation_checks_breakdown_counts_against_case_lists() -> None:
    suite = run_evaluation_suite(names=("tabletop_object_location",))
    report = evaluation_report(suite)
    tampered_report = json.loads(evaluation_report_json(report))
    tampered_report["breakdown"]["by_tag"]["qa"]["selected_cases"].append("ghost_case")
    tampered_report["report_digest"] = lab.evaluation_report_digest(tampered_report)

    validation = lab.validate_evaluation_report(tampered_report)
    checks = {check["name"]: check for check in validation["checks"]}

    assert validation["valid"] is False
    assert checks["report_digest"]["passed"] is True
    breakdown_check = checks["breakdown_counts_consistent"]
    assert breakdown_check["name"] == "breakdown_counts_consistent"
    assert breakdown_check["passed"] is False
    assert breakdown_check["differences"] == [
        {
            "path": "by_tag.qa.passed",
            "expected": 2,
            "actual": 1,
        },
        {
            "path": "by_tag.qa.total",
            "expected": 2,
            "actual": 1,
        },
    ]


def test_evaluation_report_validation_checks_breakdown_case_lists_against_selection() -> None:
    suite = run_evaluation_suite(names=("tabletop_object_location",))
    report = evaluation_report(suite)
    tampered_report = json.loads(evaluation_report_json(report))
    tampered_report["breakdown"]["by_tag"]["qa"]["selected_cases"].append("ghost_case")
    tampered_report["breakdown"]["by_tag"]["qa"]["total"] = 2
    tampered_report["breakdown"]["by_tag"]["qa"]["passed"] = 2
    tampered_report["metrics"]["by_tag"]["qa"]["case_count"] = 2
    tampered_report["metrics"]["by_tag"]["qa"]["passed_case_count"] = 2
    tampered_report["evidence_metrics"]["by_tag"]["qa"]["case_count"] = 2
    tampered_report["evidence_metrics"]["by_tag"]["qa"]["cases_without_evidence_count"] = 1
    tampered_report["evidence_metrics"]["by_tag"]["qa"][
        "average_evidence_item_count"
    ] = 2.0
    tampered_report["report_digest"] = lab.evaluation_report_digest(tampered_report)

    validation = lab.validate_evaluation_report(tampered_report)
    checks = {check["name"]: check for check in validation["checks"]}

    assert validation["valid"] is False
    assert checks["report_digest"]["passed"] is True
    assert checks["breakdown_counts_consistent"]["passed"] is True
    assert checks["metrics_match_summary"]["passed"] is True
    assert checks["evidence_metric_groups_consistent"]["passed"] is True
    breakdown_check = checks["breakdown_case_lists_match_selection"]
    assert breakdown_check["name"] == "breakdown_case_lists_match_selection"
    assert breakdown_check["passed"] is False
    assert breakdown_check["differences"] == [
        {
            "path": "by_tag.qa.selected_cases",
            "expected": ["tabletop_object_location"],
            "actual": ["tabletop_object_location", "ghost_case"],
        },
    ]


def test_evaluation_report_validation_reports_tampered_schema_version() -> None:
    suite = run_evaluation_suite(names=("tabletop_object_location",))
    report = evaluation_report(suite)
    tampered_report = json.loads(evaluation_report_json(report))
    tampered_report["schema_version"] = "legacy-report"
    tampered_report["report_digest"] = lab.evaluation_report_digest(tampered_report)

    validation = lab.validate_evaluation_report(tampered_report)
    checks = {check["name"]: check for check in validation["checks"]}

    assert validation["valid"] is False
    assert validation["schema_version"] == "legacy-report"
    assert checks["schema_version"] == {
        "name": "schema_version",
        "passed": False,
        "expected": "dsg-spatialqa-lab.evaluation-report.v1",
        "actual": "legacy-report",
    }
    assert checks["report_digest"]["passed"] is True


def test_evaluation_report_validation_reports_tampered_case_selection_digest() -> None:
    suite = run_evaluation_suite(names=("tabletop_object_location",))
    report = evaluation_report(suite)
    tampered_report = json.loads(evaluation_report_json(report))
    tampered_report["case_selection_digest"] = "0" * 64

    validation = lab.validate_evaluation_report(tampered_report)
    checks = {check["name"]: check for check in validation["checks"]}

    assert validation["valid"] is False
    assert checks["case_selection_digest"] == {
        "name": "case_selection_digest",
        "passed": False,
        "expected": report["case_selection_digest"],
        "actual": "0" * 64,
    }


def test_evaluation_report_validation_checks_case_selection_against_summary() -> None:
    suite = run_evaluation_suite(names=("tabletop_object_location",))
    report = evaluation_report(suite)
    tampered_report = json.loads(evaluation_report_json(report))
    tampered_report["case_selection"][0]["case"] = "tabletop_mug_pick"
    tampered_report["case_selection_digest"] = hashlib.sha256(
        json.dumps(
            {
                "case_count": len(tampered_report["case_selection"]),
                "case_selection": tampered_report["case_selection"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    tampered_report["report_digest"] = lab.evaluation_report_digest(tampered_report)

    validation = lab.validate_evaluation_report(tampered_report)
    checks = {check["name"]: check for check in validation["checks"]}

    assert validation["valid"] is False
    assert checks["case_selection_digest"]["passed"] is True
    assert checks["report_digest"]["passed"] is True
    assert checks["case_selection_matches_summary"] == {
        "name": "case_selection_matches_summary",
        "passed": False,
        "expected": ["tabletop_object_location"],
        "actual": ["tabletop_mug_pick"],
    }


def test_evaluation_report_validation_checks_case_selection_metadata_shape() -> None:
    suite = run_evaluation_suite(names=("tabletop_object_location",))
    report = evaluation_report(suite)
    tampered_report = json.loads(evaluation_report_json(report))
    tampered_report["summary"]["selected_cases"] = [""]
    tampered_report["case_selection"][0]["case"] = ""
    tampered_report["case_digests"][0]["case"] = ""
    for group in tampered_report["breakdown"].values():
        for entry in group.values():
            entry["selected_cases"] = [
                "" if case_name == "tabletop_object_location" else case_name
                for case_name in entry["selected_cases"]
            ]
    tampered_report["case_selection_digest"] = hashlib.sha256(
        json.dumps(
            {
                "case_count": len(tampered_report["case_selection"]),
                "case_selection": tampered_report["case_selection"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    tampered_report["report_digest"] = lab.evaluation_report_digest(tampered_report)

    validation = lab.validate_evaluation_report(tampered_report)
    checks = {check["name"]: check for check in validation["checks"]}

    assert validation["valid"] is False
    assert checks["case_selection_digest"]["passed"] is True
    assert checks["report_digest"]["passed"] is True
    assert checks["case_selection_entries_valid"] == {
        "name": "case_selection_entries_valid",
        "passed": False,
        "differences": [
            {
                "path": "case_selection[0].case",
                "expected": "non-empty string",
                "actual": "",
            },
        ],
    }


def test_evaluation_report_validation_checks_case_digests_against_summary() -> None:
    suite = run_evaluation_suite(names=("tabletop_object_location",))
    report = evaluation_report(suite)
    tampered_report = json.loads(evaluation_report_json(report))
    tampered_report["case_digests"][0]["case"] = "tabletop_mug_pick"
    tampered_report["report_digest"] = lab.evaluation_report_digest(tampered_report)

    validation = lab.validate_evaluation_report(tampered_report)
    checks = {check["name"]: check for check in validation["checks"]}

    assert validation["valid"] is False
    assert checks["report_digest"]["passed"] is True
    assert checks["case_digests_match_summary"] == {
        "name": "case_digests_match_summary",
        "passed": False,
        "expected": ["tabletop_object_location"],
        "actual": ["tabletop_mug_pick"],
    }


def test_evaluation_report_validation_checks_case_digests_against_selection() -> None:
    suite = run_evaluation_suite(names=("tabletop_object_location",))
    report = evaluation_report(suite)
    tampered_report = json.loads(evaluation_report_json(report))
    tampered_report["case_digests"][0]["kind"] = "vla_pick"
    tampered_report["report_digest"] = lab.evaluation_report_digest(tampered_report)

    validation = lab.validate_evaluation_report(tampered_report)
    checks = {check["name"]: check for check in validation["checks"]}

    assert validation["valid"] is False
    assert checks["case_digests_match_summary"]["passed"] is True
    assert checks["report_digest"]["passed"] is True
    assert checks["case_digests_match_selection"] == {
        "name": "case_digests_match_selection",
        "passed": False,
        "expected": [
            {
                "case": "tabletop_object_location",
                "kind": "qa",
                "question_type": "object_location",
                "scene_fixture": "tabletop",
            }
        ],
        "actual": [
            {
                "case": "tabletop_object_location",
                "kind": "vla_pick",
                "question_type": "object_location",
                "scene_fixture": "tabletop",
            }
        ],
    }


def test_evaluation_report_validation_checks_case_digest_status_against_summary() -> None:
    suite = run_evaluation_suite(names=("tabletop_object_location",))
    report = evaluation_report(suite)
    tampered_report = json.loads(evaluation_report_json(report))
    tampered_report["case_digests"][0]["passed"] = False
    tampered_report["report_digest"] = lab.evaluation_report_digest(tampered_report)

    validation = lab.validate_evaluation_report(tampered_report)
    checks = {check["name"]: check for check in validation["checks"]}

    assert validation["valid"] is False
    assert checks["report_digest"]["passed"] is True
    status_check = checks["case_digest_status_matches_summary"]
    assert status_check["name"] == "case_digest_status_matches_summary"
    assert status_check["passed"] is False
    assert status_check["differences"] == [
        {
            "path": "tabletop_object_location.passed",
            "expected": True,
            "actual": False,
        },
    ]


def test_evaluation_report_validation_checks_case_digest_formats() -> None:
    suite = run_evaluation_suite(names=("tabletop_object_location",))
    report = evaluation_report(suite)
    tampered_report = json.loads(evaluation_report_json(report))
    tampered_report["case_digests"][0]["digest"] = "not-a-sha256-digest"
    tampered_report["report_digest"] = lab.evaluation_report_digest(tampered_report)

    validation = lab.validate_evaluation_report(tampered_report)
    checks = {check["name"]: check for check in validation["checks"]}

    assert validation["valid"] is False
    assert checks["report_digest"]["passed"] is True
    format_check = checks["case_digest_formats"]
    assert format_check["name"] == "case_digest_formats"
    assert format_check["passed"] is False
    assert format_check["differences"] == [
        {
            "path": "tabletop_object_location.digest",
            "expected": "64 lowercase sha256 hex characters",
            "actual": "not-a-sha256-digest",
        },
    ]


def test_evaluation_report_loads_from_explicit_file_and_compares_current_run(
    tmp_path: Path,
) -> None:
    suite = run_evaluation_suite(names=("tabletop_object_location",))
    report = evaluation_report(suite)
    report_path = tmp_path / "report.json"
    report_path.write_text(evaluation_report_json(report), encoding="utf-8")

    loaded_report = load_evaluation_report(report_path)
    comparison = compare_evaluation_report(loaded_report)

    assert loaded_report == report
    assert comparison == {
        "matches": True,
        "filters": {"names": ["tabletop_object_location"]},
        "saved_digest": report["digest"],
        "current_digest": report["digest"],
        "checks": [
            {
                "name": "saved_report_valid",
                "passed": True,
            },
            {
                "name": "report_digest_matches_current",
                "passed": True,
                "expected": report["digest"],
                "actual": report["digest"],
            },
            {
                "name": "summary_matches_current",
                "passed": True,
                "expected": report["summary"],
                "actual": report["summary"],
            },
            {
                "name": "metrics_match_current",
                "passed": True,
                "expected": report["metrics"],
                "actual": report["metrics"],
            },
            {
                "name": "evidence_metrics_match_current",
                "passed": True,
                "expected": report["evidence_metrics"],
                "actual": report["evidence_metrics"],
            },
            {
                "name": "case_selection_matches_current",
                "passed": True,
                "expected": report["case_selection"],
                "actual": report["case_selection"],
            },
            {
                "name": "case_digests_match_current",
                "passed": True,
                "expected": report["case_digests"],
                "actual": report["case_digests"],
            },
            {
                "name": "failed_cases_match_current",
                "passed": True,
                "expected": report["failed_cases"],
                "actual": report["failed_cases"],
            },
            {
                "name": "failure_reasons_match_current",
                "passed": True,
                "expected": report["failure_reasons"],
                "actual": report["failure_reasons"],
            },
            {
                "name": "runtime_error_categories_match_current",
                "passed": True,
                "expected": report["runtime_error_categories"],
                "actual": report["runtime_error_categories"],
            },
            {
                "name": "runtime_error_metrics_match_current",
                "passed": True,
                "expected": report["runtime_error_metrics"],
                "actual": report["runtime_error_metrics"],
            },
            {
                "name": "failure_categories_match_current",
                "passed": True,
                "expected": report["failure_categories"],
                "actual": report["failure_categories"],
            },
            {
                "name": "failure_paths_match_current",
                "passed": True,
                "expected": report["failure_paths"],
                "actual": report["failure_paths"],
            },
            {
                "name": "breakdown_matches_current",
                "passed": True,
                "expected": report["breakdown"],
                "actual": report["breakdown"],
            },
        ],
    }


def test_evaluation_report_compare_reports_current_run_drift() -> None:
    report = evaluation_report(run_evaluation_suite(names=("tabletop_object_location",)))
    drifted_report = json.loads(evaluation_report_json(report))
    drifted_report["summary"]["selected_cases"] = ["tabletop_mug_pick"]
    current_report = evaluation_report(run_evaluation_suite(names=("tabletop_mug_pick",)))

    comparison = compare_evaluation_report(drifted_report)

    assert comparison["matches"] is False
    assert comparison["filters"] == {"names": ["tabletop_mug_pick"]}
    assert comparison["saved_digest"] == report["digest"]
    assert comparison["current_digest"] == current_report["digest"]
    assert comparison["checks"][0] == {
        "name": "saved_report_valid",
        "passed": False,
    }
    assert comparison["checks"][1] == {
        "name": "report_digest_matches_current",
        "passed": False,
        "expected": report["digest"],
        "actual": current_report["digest"],
    }
    breakdown_check = comparison["checks"][-1]
    assert breakdown_check["name"] == "breakdown_matches_current"
    assert breakdown_check["passed"] is False
    assert breakdown_check["expected"] == report["breakdown"]
    assert breakdown_check["actual"] == current_report["breakdown"]
    assert [difference["path"] for difference in breakdown_check["differences"]] == [
        "by_kind.qa",
        "by_kind.vla_pick",
        "by_question_type.object_location",
        "by_scene_fixture.tabletop.selected_cases",
        "by_tag.anchor",
        "by_tag.memory",
        "by_tag.pick",
        "by_tag.qa",
        "by_tag.vla",
    ]


def test_evaluation_report_compare_reports_summary_path_drift() -> None:
    report = evaluation_report(run_evaluation_suite(names=("tabletop_object_location",)))
    drifted_report = json.loads(evaluation_report_json(report))
    drifted_report["summary"]["failed"] = 1

    comparison = compare_evaluation_report(drifted_report)

    summary_check = next(
        check for check in comparison["checks"] if check["name"] == "summary_matches_current"
    )
    assert comparison["matches"] is False
    assert [
        check["name"] for check in comparison["checks"] if check["passed"] is False
    ] == ["saved_report_valid", "summary_matches_current"]
    assert summary_check["differences"] == [
        {
            "path": "failed",
            "expected": 1,
            "actual": 0,
        },
    ]


def test_evaluation_report_compare_reports_failed_case_drift() -> None:
    report = evaluation_report(run_evaluation_suite(names=("tabletop_object_location",)))
    drifted_report = json.loads(evaluation_report_json(report))
    drifted_report["failed_cases"] = [
        {
            "case": "tabletop_object_location",
            "kind": "qa",
            "scene_fixture": "tabletop",
            "tags": ["memory", "qa"],
            "mismatch_count": 1,
            "mismatch_paths": ["answer.pose.x"],
            "mismatch_reasons": ["value_mismatch"],
            "mismatch_categories": ["value_mismatch"],
            "error": None,
        },
    ]

    comparison = compare_evaluation_report(drifted_report)

    failed_cases_check = next(
        check
        for check in comparison["checks"]
        if check["name"] == "failed_cases_match_current"
    )
    assert comparison["matches"] is False
    assert [
        check["name"] for check in comparison["checks"] if check["passed"] is False
    ] == ["saved_report_valid", "failed_cases_match_current"]
    assert failed_cases_check["differences"] == [
        {
            "path": "tabletop_object_location",
            "expected": {
                "kind": "qa",
                "scene_fixture": "tabletop",
                "tags": ["memory", "qa"],
                "mismatch_count": 1,
                "mismatch_paths": ["answer.pose.x"],
                "mismatch_reasons": ["value_mismatch"],
                "mismatch_categories": ["value_mismatch"],
                "error": None,
            },
            "actual": None,
        },
    ]


def test_evaluation_report_compare_reports_metric_path_drift() -> None:
    report = evaluation_report(run_evaluation_suite(names=("tabletop_object_location",)))
    drifted_report = json.loads(evaluation_report_json(report))
    drifted_report["metrics"]["by_question_type"]["object_location"][
        "failed_case_count"
    ] = 1
    drifted_report["metrics"]["by_tag"]["qa"]["pass_rate"] = 0.5

    comparison = compare_evaluation_report(drifted_report)

    metric_check = next(
        check for check in comparison["checks"] if check["name"] == "metrics_match_current"
    )
    assert comparison["matches"] is False
    assert [
        check["name"] for check in comparison["checks"] if check["passed"] is False
    ] == ["saved_report_valid", "metrics_match_current"]
    assert metric_check["differences"] == [
        {
            "path": "by_question_type.object_location.failed_case_count",
            "expected": 1,
            "actual": 0,
        },
        {
            "path": "by_tag.qa.pass_rate",
            "expected": 0.5,
            "actual": 1.0,
        },
    ]


def test_evaluation_report_compare_reports_evidence_metric_path_drift() -> None:
    report = evaluation_report(run_evaluation_suite(names=("tabletop_object_location",)))
    drifted_report = json.loads(evaluation_report_json(report))
    drifted_report["evidence_metrics"]["by_question_type"]["object_location"][
        "evidence_edge_count"
    ] = 99

    comparison = compare_evaluation_report(drifted_report)

    evidence_check = next(
        check
        for check in comparison["checks"]
        if check["name"] == "evidence_metrics_match_current"
    )
    assert comparison["matches"] is False
    assert [
        check["name"] for check in comparison["checks"] if check["passed"] is False
    ] == ["saved_report_valid", "evidence_metrics_match_current"]
    assert evidence_check["differences"] == [
        {
            "path": "by_question_type.object_location.evidence_edge_count",
            "expected": 99,
            "actual": 2,
        }
    ]


def test_evaluation_report_compare_reports_case_selection_drift() -> None:
    report = evaluation_report(run_evaluation_suite(names=("tabletop_object_location",)))
    drifted_report = json.loads(evaluation_report_json(report))
    drifted_report["case_selection"][0]["tags"] = ["qa", "memory", "tampered"]
    drifted_report["case_selection_digest"] = hashlib.sha256(
        json.dumps(
            {
                "case_count": len(drifted_report["case_selection"]),
                "case_selection": drifted_report["case_selection"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    drifted_report["report_digest"] = lab.evaluation_report_digest(drifted_report)

    comparison = compare_evaluation_report(drifted_report)

    selection_check = next(
        check
        for check in comparison["checks"]
        if check["name"] == "case_selection_matches_current"
    )
    assert comparison["matches"] is False
    assert [
        check["name"] for check in comparison["checks"] if check["passed"] is False
    ] == ["saved_report_valid", "case_selection_matches_current"]
    assert selection_check["differences"] == [
        {
            "path": "tabletop_object_location.tags",
            "expected": ["qa", "memory", "tampered"],
            "actual": ["qa", "memory"],
        }
    ]


def test_evaluation_report_compare_reports_case_digest_drift() -> None:
    report = evaluation_report(run_evaluation_suite(names=("tabletop_object_location",)))
    drifted_report = json.loads(evaluation_report_json(report))
    drifted_report["case_digests"][0]["digest"] = "0" * 64

    comparison = compare_evaluation_report(drifted_report)

    digest_check = next(
        check for check in comparison["checks"] if check["name"] == "case_digests_match_current"
    )
    assert comparison["matches"] is False
    assert [
        check["name"] for check in comparison["checks"] if check["passed"] is False
    ] == ["saved_report_valid", "case_digests_match_current"]
    assert digest_check["differences"] == [
        {
            "path": "tabletop_object_location.digest",
            "expected": "0" * 64,
            "actual": report["case_digests"][0]["digest"],
        }
    ]


def test_evaluation_report_compare_reports_breakdown_path_drift() -> None:
    report = evaluation_report(run_evaluation_suite(names=("tabletop_object_location",)))
    drifted_report = json.loads(evaluation_report_json(report))
    drifted_report["breakdown"]["by_tag"]["qa"]["failed"] = 1

    comparison = compare_evaluation_report(drifted_report)

    breakdown_check = next(
        check
        for check in comparison["checks"]
        if check["name"] == "breakdown_matches_current"
    )
    assert comparison["matches"] is False
    assert [
        check["name"] for check in comparison["checks"] if check["passed"] is False
    ] == ["saved_report_valid", "breakdown_matches_current"]
    assert breakdown_check["differences"] == [
        {
            "path": "by_tag.qa.failed",
            "expected": 1,
            "actual": 0,
        },
    ]


def test_evaluation_report_compare_reports_runtime_error_category_drift() -> None:
    report = evaluation_report(run_evaluation_suite(tags=("vla", "error")))
    drifted_report = json.loads(evaluation_report_json(report))
    drifted_report["runtime_error_categories"][0]["count"] = 99

    comparison = compare_evaluation_report(drifted_report)

    category_check = next(
        check
        for check in comparison["checks"]
        if check["name"] == "runtime_error_categories_match_current"
    )
    assert comparison["matches"] is False
    assert [
        check["name"] for check in comparison["checks"] if check["passed"] is False
    ] == ["saved_report_valid", "runtime_error_categories_match_current"]
    assert category_check["differences"] == [
        {
            "path": "low_confidence.count",
            "expected": 99,
            "actual": 3,
        },
    ]


def test_evaluation_report_compare_reports_runtime_error_metric_drift() -> None:
    report = evaluation_report(run_evaluation_suite(tags=("vla", "error")))
    drifted_report = json.loads(evaluation_report_json(report))
    drifted_report["runtime_error_metrics"]["by_category"]["missing_target"][
        "case_rate"
    ] = 0.0

    comparison = compare_evaluation_report(drifted_report)

    metrics_check = next(
        check
        for check in comparison["checks"]
        if check["name"] == "runtime_error_metrics_match_current"
    )
    assert comparison["matches"] is False
    assert [
        check["name"] for check in comparison["checks"] if check["passed"] is False
    ] == ["saved_report_valid", "runtime_error_metrics_match_current"]
    assert metrics_check["differences"] == [
        {
            "path": "by_category.missing_target.case_rate",
            "expected": 0.0,
            "actual": 2 / 13,
        },
    ]


def test_evaluation_report_compare_reports_failure_category_drift() -> None:
    report = evaluation_report(run_evaluation_suite(names=("tabletop_object_location",)))
    drifted_report = json.loads(evaluation_report_json(report))
    drifted_report["failure_categories"] = [
        {
            "category": "value_mismatch",
            "count": 1,
            "cases": ["tabletop_object_location"],
        },
    ]

    comparison = compare_evaluation_report(drifted_report)

    category_check = next(
        check
        for check in comparison["checks"]
        if check["name"] == "failure_categories_match_current"
    )
    assert comparison["matches"] is False
    assert [
        check["name"] for check in comparison["checks"] if check["passed"] is False
    ] == ["saved_report_valid", "failure_categories_match_current"]
    assert category_check["differences"] == [
        {
            "path": "value_mismatch",
            "expected": {
                "count": 1,
                "cases": ["tabletop_object_location"],
            },
            "actual": None,
        },
    ]


def test_evaluation_report_compare_reports_failure_reason_drift() -> None:
    report = evaluation_report(run_evaluation_suite(names=("tabletop_object_location",)))
    drifted_report = json.loads(evaluation_report_json(report))
    drifted_report["failure_reasons"] = [
        {
            "reason": "value_mismatch",
            "count": 1,
            "cases": ["tabletop_object_location"],
        },
    ]

    comparison = compare_evaluation_report(drifted_report)

    reason_check = next(
        check
        for check in comparison["checks"]
        if check["name"] == "failure_reasons_match_current"
    )
    assert comparison["matches"] is False
    assert [
        check["name"] for check in comparison["checks"] if check["passed"] is False
    ] == ["saved_report_valid", "failure_reasons_match_current"]
    assert reason_check["differences"] == [
        {
            "path": "value_mismatch",
            "expected": {
                "count": 1,
                "cases": ["tabletop_object_location"],
            },
            "actual": None,
        },
    ]


def test_evaluation_report_compare_reports_failure_path_drift() -> None:
    report = evaluation_report(run_evaluation_suite(names=("tabletop_object_location",)))
    drifted_report = json.loads(evaluation_report_json(report))
    drifted_report["failure_paths"] = [
        {
            "path": "answer.visible",
            "count": 1,
            "cases": ["tabletop_object_location"],
        },
    ]

    comparison = compare_evaluation_report(drifted_report)

    path_check = next(
        check
        for check in comparison["checks"]
        if check["name"] == "failure_paths_match_current"
    )
    assert comparison["matches"] is False
    assert [
        check["name"] for check in comparison["checks"] if check["passed"] is False
    ] == ["saved_report_valid", "failure_paths_match_current"]
    assert path_check["differences"] == [
        {
            "path": "answer.visible",
            "expected": {
                "count": 1,
                "cases": ["tabletop_object_location"],
            },
            "actual": None,
        },
    ]


def test_evaluation_manifest_collects_filtered_case_and_fixture_metadata() -> None:
    manifest = evaluation_manifest(tags=("qa", "relations"))
    digest = manifest["digest"]

    assert manifest == {
        "schema_version": "dsg-spatialqa-lab.evaluation-manifest.v1",
        "digest": digest,
        "filters": {
            "names": [],
            "tags": ["qa", "relations"],
            "kinds": [],
            "question_types": [],
        },
        "scene_fixtures": [
            {
                "name": "relation_shift",
                "description": (
                    "Dynamic tabletop scene where mug_1 moves from left of plate_1 "
                    "to right of it."
                ),
                "tags": ["dynamic", "tabletop", "relations", "move"],
            },
            {
                "name": "tabletop",
                "description": "Static tabletop scene with mug, plate, table, room, and agent.",
                "tags": ["static", "tabletop"],
            },
        ],
        "evaluation_cases": [
            {
                "name": "relation_shift_relation_timeline",
                "scene_fixture": "relation_shift",
                "scene_description": (
                    "Dynamic tabletop scene where mug_1 moves from left of plate_1 "
                    "to right of it."
                ),
                "scene_tags": ["dynamic", "tabletop", "relations", "move"],
                "kind": "qa",
                "tags": ["qa", "dynamic", "temporal", "relations", "move"],
                "question": {
                    "type": "relation_timeline",
                    "src": "mug_1",
                    "dst": "plate_1",
                    "reference_frame": "agent",
                },
                "question_type": "relation_timeline",
                "baseline_scene_fixture": None,
                "baseline_scene_description": None,
                "baseline_scene_tags": [],
                "target_object": None,
                "target_label": None,
                "reference_object": None,
                "reference_label": None,
                "relation": None,
                "expected_keys": ["answer", "error", "evidence_edges", "needs_reobserve"],
            },
            {
                "name": "tabletop_relation_timeline",
                "scene_fixture": "tabletop",
                "scene_description": "Static tabletop scene with mug, plate, table, room, and agent.",
                "scene_tags": ["static", "tabletop"],
                "kind": "qa",
                "tags": ["qa", "memory", "temporal", "relations"],
                "question": {
                    "type": "relation_timeline",
                    "src": "mug_1",
                    "dst": "plate_1",
                    "reference_frame": "agent",
                },
                "question_type": "relation_timeline",
                "baseline_scene_fixture": None,
                "baseline_scene_description": None,
                "baseline_scene_tags": [],
                "target_object": None,
                "target_label": None,
                "reference_object": None,
                "reference_label": None,
                "relation": None,
                "expected_keys": ["answer", "error", "evidence_edges"],
            },
            {
                "name": "tabletop_relative_relation_mug_left_of_plate",
                "scene_fixture": "tabletop",
                "scene_description": "Static tabletop scene with mug, plate, table, room, and agent.",
                "scene_tags": ["static", "tabletop"],
                "kind": "qa",
                "tags": ["qa", "foundation", "relations"],
                "question": {
                    "type": "relative_relation",
                    "src": "mug_1",
                    "dst": "plate_1",
                    "relation": "LEFT_OF",
                    "reference_frame": "agent",
                },
                "question_type": "relative_relation",
                "baseline_scene_fixture": None,
                "baseline_scene_description": None,
                "baseline_scene_tags": [],
                "target_object": None,
                "target_label": None,
                "reference_object": None,
                "reference_label": None,
                "relation": None,
                "expected_keys": ["answer", "error", "evidence_edges"],
            },
        ],
        "coverage": {
            "case_count": 3,
            "scene_fixture_count": 2,
            "by_kind": {"qa": 3},
            "by_question_type": {"relation_timeline": 2, "relative_relation": 1},
            "by_tag": {
                "dynamic": 1,
                "foundation": 1,
                "memory": 1,
                "move": 1,
                "qa": 3,
                "relations": 3,
                "temporal": 2,
            },
            "by_scene_fixture": {
                "relation_shift": 1,
                "tabletop": 2,
            },
            "by_scene_tag": {
                "dynamic": 1,
                "move": 1,
                "relations": 1,
                "static": 1,
                "tabletop": 2,
            },
        },
    }
    assert len(digest) == 64
    assert "suite" not in manifest
    assert "report" not in manifest


def test_evaluation_manifest_json_is_stable_and_savable(tmp_path: Path) -> None:
    manifest = evaluation_manifest(question_types=("scene_delta",))

    payload = evaluation_manifest_json(manifest)
    repeated = evaluation_manifest_json(manifest)
    manifest_path = tmp_path / "manifest.json"
    saved_path = save_evaluation_manifest(manifest_path, question_types=("scene_delta",))

    assert payload == repeated
    assert payload.endswith("\n")
    assert json.loads(payload) == manifest
    assert saved_path == manifest_path
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest


def test_evaluation_manifest_loads_from_explicit_file_and_validates(tmp_path: Path) -> None:
    manifest = evaluation_manifest(tags=("qa", "relations"))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(evaluation_manifest_json(manifest), encoding="utf-8")

    loaded_manifest = load_evaluation_manifest(manifest_path)
    validation = validate_evaluation_manifest(loaded_manifest)

    assert loaded_manifest == manifest
    assert validation == {
        "valid": True,
        "schema_version": "dsg-spatialqa-lab.evaluation-manifest.v1",
        "digest": manifest["digest"],
        "checks": [
            {
                "name": "schema_version",
                "passed": True,
                "expected": "dsg-spatialqa-lab.evaluation-manifest.v1",
                "actual": "dsg-spatialqa-lab.evaluation-manifest.v1",
            },
            {
                "name": "manifest_digest",
                "passed": True,
                "expected": manifest["digest"],
                "actual": manifest["digest"],
            },
            {"name": "case_metadata_entries_valid", "passed": True},
            {
                "name": "scene_fixture_manifest_covers_cases",
                "passed": True,
                "expected": ["relation_shift", "tabletop"],
                "actual": ["relation_shift", "tabletop"],
            },
            {
                "name": "coverage_matches_manifest",
                "passed": True,
                "expected": manifest["coverage"],
                "actual": manifest["coverage"],
            },
        ],
    }


def test_evaluation_manifest_digest_helper_recomputes_saved_digest_and_saves_nested_paths(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "evaluation_manifest_digest")
    manifest = evaluation_manifest(tags=("qa", "relations"))
    expected_payload = {
        "schema_version": manifest["schema_version"],
        "filters": manifest["filters"],
        "scene_fixtures": manifest["scene_fixtures"],
        "evaluation_cases": manifest["evaluation_cases"],
        "coverage": manifest["coverage"],
    }
    expected_digest = hashlib.sha256(
        json.dumps(expected_payload, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()

    assert manifest["digest"] == expected_digest
    assert lab.evaluation_manifest_digest(manifest) == expected_digest

    tampered_manifest = json.loads(evaluation_manifest_json(manifest))
    tampered_manifest["filters"]["tags"] = ["qa", "dynamic"]
    assert lab.evaluation_manifest_digest(tampered_manifest) != manifest["digest"]

    manifest_path = tmp_path / "manifests" / "relations.json"
    saved_path = save_evaluation_manifest(manifest_path, tags=("qa", "relations"))

    assert saved_path == manifest_path
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest


def test_evaluation_manifest_validation_reports_tampered_digest() -> None:
    manifest = evaluation_manifest(tags=("qa", "relations"))
    tampered_manifest = json.loads(evaluation_manifest_json(manifest))
    tampered_manifest["digest"] = "0" * 64

    validation = validate_evaluation_manifest(tampered_manifest)

    assert validation["valid"] is False
    assert validation["digest"] == "0" * 64
    assert validation["checks"][1] == {
        "name": "manifest_digest",
        "passed": False,
        "expected": manifest["digest"],
        "actual": "0" * 64,
    }


def test_evaluation_manifest_validation_reports_tampered_coverage_paths() -> None:
    manifest = evaluation_manifest(tags=("qa", "relations"))
    tampered_manifest = json.loads(evaluation_manifest_json(manifest))
    tampered_manifest["coverage"]["by_scene_fixture"]["tabletop"] = 1
    tampered_manifest["coverage"]["by_tag"]["relations"] = 2

    validation = validate_evaluation_manifest(tampered_manifest)

    assert validation["valid"] is False
    assert validation["checks"][-1]["name"] == "coverage_matches_manifest"
    assert validation["checks"][-1]["passed"] is False
    assert validation["checks"][-1]["differences"] == [
        {"path": "by_scene_fixture.tabletop", "expected": 2, "actual": 1},
        {"path": "by_tag.relations", "expected": 3, "actual": 2},
    ]


def test_evaluation_manifest_validation_reports_invalid_case_metadata() -> None:
    manifest = evaluation_manifest(tags=("qa", "relations"))
    tampered_manifest = json.loads(evaluation_manifest_json(manifest))
    tampered_manifest["evaluation_cases"][0]["name"] = 123
    tampered_manifest["digest"] = lab.evaluation_manifest_digest(tampered_manifest)

    validation = validate_evaluation_manifest(tampered_manifest)

    checks = {check["name"]: check for check in validation["checks"]}
    assert validation["valid"] is False
    assert checks["manifest_digest"]["passed"] is True
    assert checks["coverage_matches_manifest"]["passed"] is True
    assert checks["case_metadata_entries_valid"] == {
        "name": "case_metadata_entries_valid",
        "passed": False,
        "differences": [
            {
                "path": "evaluation_cases[0].name",
                "expected": "non-empty string",
                "actual": 123,
            },
        ],
    }


def test_evaluation_manifest_validation_reports_scene_fixture_metadata_drift() -> None:
    manifest = evaluation_manifest(tags=("qa", "relations"))
    tampered_manifest = json.loads(evaluation_manifest_json(manifest))
    fixture_name = "tabletop"
    fixture = next(
        scene_fixture
        for scene_fixture in tampered_manifest["scene_fixtures"]
        if scene_fixture["name"] == fixture_name
    )
    original_tags = next(
        scene_fixture["tags"]
        for scene_fixture in manifest["scene_fixtures"]
        if scene_fixture["name"] == fixture_name
    )
    tampered_tags = [
        tag
        for tag in original_tags
        if tag != "static"
    ]
    fixture["tags"] = tampered_tags
    del tampered_manifest["coverage"]["by_scene_tag"]["static"]
    tampered_manifest["digest"] = lab.evaluation_manifest_digest(tampered_manifest)

    validation = validate_evaluation_manifest(tampered_manifest)

    checks = {check["name"]: check for check in validation["checks"]}
    assert validation["valid"] is False
    assert checks["manifest_digest"]["passed"] is True
    assert checks["coverage_matches_manifest"]["passed"] is True
    assert checks["scene_fixture_manifest_covers_cases"]["passed"] is False
    assert checks["scene_fixture_manifest_covers_cases"]["differences"] == [
        {
            "path": f"{fixture_name}.tags",
            "expected": original_tags,
            "actual": tampered_tags,
        },
    ]


def test_evaluation_manifest_compare_matches_current_metadata() -> None:
    manifest = evaluation_manifest(tags=("qa", "relations"))

    comparison = compare_evaluation_manifest(manifest)

    assert comparison == {
        "matches": True,
        "filters": {
            "names": [],
            "tags": ["qa", "relations"],
            "kinds": [],
            "question_types": [],
        },
        "saved_digest": manifest["digest"],
        "current_digest": manifest["digest"],
        "checks": [
            {"name": "saved_manifest_valid", "passed": True},
            {
                "name": "manifest_digest_matches_current",
                "passed": True,
                "expected": manifest["digest"],
                "actual": manifest["digest"],
            },
            {
                "name": "coverage_matches_current",
                "passed": True,
                "expected": manifest["coverage"],
                "actual": manifest["coverage"],
            },
            {
                "name": "case_manifest_matches_current",
                "passed": True,
                "expected": [
                    "relation_shift_relation_timeline",
                    "tabletop_relation_timeline",
                    "tabletop_relative_relation_mug_left_of_plate",
                ],
                "actual": [
                    "relation_shift_relation_timeline",
                    "tabletop_relation_timeline",
                    "tabletop_relative_relation_mug_left_of_plate",
                ],
            },
            {
                "name": "scene_fixture_manifest_matches_current",
                "passed": True,
                "expected": ["relation_shift", "tabletop"],
                "actual": ["relation_shift", "tabletop"],
            },
        ],
    }


def test_evaluation_manifest_compare_reports_current_metadata_drift() -> None:
    manifest = evaluation_manifest(tags=("qa", "relations"))
    drifted_manifest = json.loads(evaluation_manifest_json(manifest))
    drifted_manifest["filters"]["tags"] = ["qa", "dynamic"]
    current_manifest = evaluation_manifest(tags=("qa", "dynamic"))

    comparison = compare_evaluation_manifest(drifted_manifest)

    assert comparison["matches"] is False
    assert comparison["filters"] == {
        "names": [],
        "tags": ["qa", "dynamic"],
        "kinds": [],
        "question_types": [],
    }
    assert comparison["saved_digest"] == manifest["digest"]
    assert comparison["current_digest"] == current_manifest["digest"]
    assert comparison["checks"][0] == {"name": "saved_manifest_valid", "passed": False}
    assert comparison["checks"][1] == {
        "name": "manifest_digest_matches_current",
        "passed": False,
        "expected": manifest["digest"],
        "actual": current_manifest["digest"],
    }
    assert comparison["checks"][2]["name"] == "coverage_matches_current"
    assert comparison["checks"][2]["passed"] is False


def test_evaluation_manifest_compare_reports_coverage_path_drift() -> None:
    manifest = evaluation_manifest(tags=("qa", "relations"))
    drifted_manifest = json.loads(evaluation_manifest_json(manifest))
    drifted_manifest["coverage"]["by_scene_fixture"]["tabletop"] = 1
    drifted_manifest["coverage"]["by_tag"]["qa"] = 4

    comparison = compare_evaluation_manifest(drifted_manifest)

    coverage_check = next(
        check
        for check in comparison["checks"]
        if check["name"] == "coverage_matches_current"
    )
    assert comparison["matches"] is False
    assert coverage_check["differences"] == [
        {"path": "by_scene_fixture.tabletop", "expected": 1, "actual": 2},
        {"path": "by_tag.qa", "expected": 4, "actual": 3},
    ]


def test_evaluation_manifest_compare_reports_case_manifest_path_drift() -> None:
    manifest = evaluation_manifest(tags=("qa", "relations"))
    drifted_manifest = json.loads(evaluation_manifest_json(manifest))
    case_name = drifted_manifest["evaluation_cases"][0]["name"]
    original_tags = manifest["evaluation_cases"][0]["tags"]
    drifted_manifest["evaluation_cases"][0]["tags"] = [*original_tags, "tampered"]

    comparison = compare_evaluation_manifest(drifted_manifest)

    case_check = next(
        check
        for check in comparison["checks"]
        if check["name"] == "case_manifest_matches_current"
    )
    assert comparison["matches"] is False
    assert case_check["passed"] is False
    assert case_check["differences"] == [
        {
            "path": f"{case_name}.tags",
            "expected": [*original_tags, "tampered"],
            "actual": original_tags,
        },
    ]


def test_evaluation_bundle_collects_report_manifests_and_suite() -> None:
    bundle = evaluation_bundle(tags=("qa", "reobserve"))

    assert bundle["schema_version"] == "dsg-spatialqa-lab.evaluation-bundle.v1"
    assert bundle["filters"] == {
        "names": [],
        "tags": ["qa", "reobserve"],
        "kinds": [],
        "question_types": [],
    }
    assert [fixture["name"] for fixture in bundle["scene_fixtures"]] == [
        "multi_room_rearrangement",
        "needs_reobserve",
    ]
    assert [case["name"] for case in bundle["evaluation_cases"]] == [
        "multi_room_rearrangement_reobserve_targets",
        "needs_reobserve_spoon_label_candidates",
        "needs_reobserve_targets",
    ]
    assert bundle["coverage"] == {
        "case_count": 3,
        "scene_fixture_count": 2,
        "by_kind": {"qa": 3},
        "by_question_type": {"label_candidates": 1, "reobserve_targets": 2},
        "by_tag": {
            "label": 1,
            "memory": 2,
            "multi_room": 1,
            "occlusion": 2,
            "qa": 3,
            "reobserve": 3,
        },
        "by_scene_fixture": {
            "multi_room_rearrangement": 1,
            "needs_reobserve": 2,
        },
        "by_scene_tag": {
            "dynamic": 1,
            "multi_room": 1,
            "move": 1,
            "occlusion": 1,
            "reobserve": 2,
            "static": 1,
            "tabletop": 1,
        },
    }
    assert bundle["suite"]["summary"] == {
        "total": 3,
        "passed": 3,
        "failed": 0,
        "failed_cases": [],
        "selected_cases": [
            "multi_room_rearrangement_reobserve_targets",
            "needs_reobserve_spoon_label_candidates",
            "needs_reobserve_targets",
        ],
    }
    assert bundle["report"]["digest"] == bundle["suite"]["digest"]
    assert bundle["report"]["metrics"] == {
        "case_count": 3,
        "passed_case_count": 3,
        "failed_case_count": 0,
        "pass_rate": 1.0,
        "failure_rate": 0.0,
        "by_kind": {
            "qa": {
                "case_count": 3,
                "passed_case_count": 3,
                "failed_case_count": 0,
                "pass_rate": 1.0,
                "failure_rate": 0.0,
            }
        },
        "by_question_type": {
            "label_candidates": {
                "case_count": 1,
                "passed_case_count": 1,
                "failed_case_count": 0,
                "pass_rate": 1.0,
                "failure_rate": 0.0,
            },
            "reobserve_targets": {
                "case_count": 2,
                "passed_case_count": 2,
                "failed_case_count": 0,
                "pass_rate": 1.0,
                "failure_rate": 0.0,
            }
        },
        "by_scene_fixture": {
            "multi_room_rearrangement": {
                "case_count": 1,
                "passed_case_count": 1,
                "failed_case_count": 0,
                "pass_rate": 1.0,
                "failure_rate": 0.0,
            },
            "needs_reobserve": {
                "case_count": 2,
                "passed_case_count": 2,
                "failed_case_count": 0,
                "pass_rate": 1.0,
                "failure_rate": 0.0,
            },
        },
        "by_tag": {
            "label": {
                "case_count": 1,
                "passed_case_count": 1,
                "failed_case_count": 0,
                "pass_rate": 1.0,
                "failure_rate": 0.0,
            },
            "memory": {
                "case_count": 2,
                "passed_case_count": 2,
                "failed_case_count": 0,
                "pass_rate": 1.0,
                "failure_rate": 0.0,
            },
            "multi_room": {
                "case_count": 1,
                "passed_case_count": 1,
                "failed_case_count": 0,
                "pass_rate": 1.0,
                "failure_rate": 0.0,
            },
            "occlusion": {
                "case_count": 2,
                "passed_case_count": 2,
                "failed_case_count": 0,
                "pass_rate": 1.0,
                "failure_rate": 0.0,
            },
            "qa": {
                "case_count": 3,
                "passed_case_count": 3,
                "failed_case_count": 0,
                "pass_rate": 1.0,
                "failure_rate": 0.0,
            },
            "reobserve": {
                "case_count": 3,
                "passed_case_count": 3,
                "failed_case_count": 0,
                "pass_rate": 1.0,
                "failure_rate": 0.0,
            },
        },
    }


def test_evaluation_bundle_json_is_stable_and_savable(tmp_path: Path) -> None:
    bundle = evaluation_bundle(
        names=("tabletop_mug_pick", "tabletop_object_location"),
    )

    payload = evaluation_bundle_json(bundle)
    repeated = evaluation_bundle_json(bundle)
    bundle_path = tmp_path / "bundle.json"
    saved_path = save_evaluation_bundle(
        bundle_path,
        names=("tabletop_mug_pick", "tabletop_object_location"),
    )

    assert payload == repeated
    assert payload.endswith("\n")
    assert json.loads(payload) == bundle
    assert saved_path == bundle_path
    assert json.loads(bundle_path.read_text(encoding="utf-8")) == bundle


def test_evaluation_bundle_loads_from_explicit_file_and_validates(tmp_path: Path) -> None:
    bundle = evaluation_bundle(tags=("qa", "reobserve"))
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(evaluation_bundle_json(bundle), encoding="utf-8")

    loaded_bundle = load_evaluation_bundle(bundle_path)
    validation = validate_evaluation_bundle(loaded_bundle)

    assert loaded_bundle == bundle
    assert validation == {
        "valid": True,
        "schema_version": "dsg-spatialqa-lab.evaluation-bundle.v1",
        "digest": bundle["suite"]["digest"],
        "bundle_digest": bundle["bundle_digest"],
        "checks": [
            {
                "name": "schema_version",
                "passed": True,
                "expected": "dsg-spatialqa-lab.evaluation-bundle.v1",
                "actual": "dsg-spatialqa-lab.evaluation-bundle.v1",
            },
            {
                "name": "suite_digest",
                "passed": True,
                "expected": bundle["suite"]["digest"],
                "actual": bundle["suite"]["digest"],
            },
            {"name": "report_matches_suite", "passed": True},
            {
                "name": "bundle_digest",
                "passed": True,
                "expected": bundle["bundle_digest"],
                "actual": bundle["bundle_digest"],
            },
            {
                "name": "case_manifest_matches_suite",
                "passed": True,
                "expected": [
                    "multi_room_rearrangement_reobserve_targets",
                    "needs_reobserve_spoon_label_candidates",
                    "needs_reobserve_targets",
                ],
                "actual": [
                    "multi_room_rearrangement_reobserve_targets",
                    "needs_reobserve_spoon_label_candidates",
                    "needs_reobserve_targets",
                ],
            },
            {"name": "case_metadata_entries_valid", "passed": True},
            {
                "name": "scene_fixture_manifest_covers_cases",
                "passed": True,
                "expected": ["multi_room_rearrangement", "needs_reobserve"],
                "actual": ["multi_room_rearrangement", "needs_reobserve"],
            },
            {
                "name": "coverage_matches_manifest",
                "passed": True,
                "expected": bundle["coverage"],
                "actual": bundle["coverage"],
            },
        ],
    }


def test_evaluation_bundle_includes_stable_bundle_digest_and_validates_filter_tampering(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "evaluation_bundle_digest")
    bundle = evaluation_bundle(tags=("qa", "reobserve"))
    expected_payload = {
        key: value for key, value in bundle.items() if key != "bundle_digest"
    }
    expected_digest = hashlib.sha256(
        json.dumps(expected_payload, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()

    assert bundle["bundle_digest"] == expected_digest
    assert lab.evaluation_bundle_digest(bundle) == expected_digest
    validation = validate_evaluation_bundle(bundle)
    checks = {check["name"]: check for check in validation["checks"]}
    assert validation["bundle_digest"] == expected_digest
    assert checks["bundle_digest"] == {
        "name": "bundle_digest",
        "passed": True,
        "expected": expected_digest,
        "actual": expected_digest,
    }

    tampered_bundle = json.loads(evaluation_bundle_json(bundle))
    tampered_bundle["filters"]["tags"] = ["qa", "dynamic"]
    tampered_validation = validate_evaluation_bundle(tampered_bundle)
    tampered_checks = {check["name"]: check for check in tampered_validation["checks"]}

    assert tampered_validation["valid"] is False
    assert tampered_checks["bundle_digest"] == {
        "name": "bundle_digest",
        "passed": False,
        "expected": lab.evaluation_bundle_digest(tampered_bundle),
        "actual": expected_digest,
    }

    bundle_path = tmp_path / "bundles" / "reobserve.json"
    saved_path = save_evaluation_bundle(bundle_path, tags=("qa", "reobserve"))

    assert saved_path == bundle_path
    assert json.loads(bundle_path.read_text(encoding="utf-8")) == bundle


def test_evaluation_bundle_validation_reports_tampered_digest() -> None:
    bundle = evaluation_bundle(tags=("qa", "reobserve"))
    tampered_bundle = json.loads(evaluation_bundle_json(bundle))
    tampered_bundle["suite"]["digest"] = "0" * 64

    validation = validate_evaluation_bundle(tampered_bundle)

    assert validation["valid"] is False
    assert validation["digest"] == "0" * 64
    assert validation["checks"][1] == {
        "name": "suite_digest",
        "passed": False,
        "expected": bundle["suite"]["digest"],
        "actual": "0" * 64,
    }


def test_evaluation_bundle_validation_reports_tampered_report_paths() -> None:
    bundle = evaluation_bundle(tags=("qa", "reobserve"))
    tampered_bundle = json.loads(evaluation_bundle_json(bundle))
    tampered_bundle["report"]["metrics"]["case_count"] = 999
    tampered_bundle["report"]["metrics"]["by_tag"]["qa"]["pass_rate"] = 0.5

    validation = validate_evaluation_bundle(tampered_bundle)

    report_check = next(
        check
        for check in validation["checks"]
        if check["name"] == "report_matches_suite"
    )
    assert validation["valid"] is False
    assert report_check["passed"] is False
    assert report_check["differences"] == [
        {
            "path": "metrics.by_tag.qa.pass_rate",
            "expected": 1.0,
            "actual": 0.5,
        },
        {"path": "metrics.case_count", "expected": 3, "actual": 999},
    ]


def test_evaluation_bundle_validation_reports_tampered_report_failed_case_paths() -> None:
    bundle = evaluation_bundle(names=("tabletop_object_location",))
    tampered_bundle = json.loads(evaluation_bundle_json(bundle))
    tampered_bundle["report"]["failed_cases"] = [
        {
            "case": "tabletop_object_location",
            "kind": "qa",
            "scene_fixture": "tabletop",
            "tags": ["memory", "qa"],
            "mismatch_count": 1,
            "mismatch_paths": ["answer.pose.x"],
            "mismatch_reasons": ["value_mismatch"],
            "mismatch_categories": ["value_mismatch"],
            "error": None,
        },
    ]

    validation = validate_evaluation_bundle(tampered_bundle)

    report_check = next(
        check
        for check in validation["checks"]
        if check["name"] == "report_matches_suite"
    )
    assert validation["valid"] is False
    assert report_check["passed"] is False
    assert report_check["differences"] == [
        {
            "path": "failed_cases.tabletop_object_location",
            "expected": None,
            "actual": {
                "kind": "qa",
                "scene_fixture": "tabletop",
                "tags": ["memory", "qa"],
                "mismatch_count": 1,
                "mismatch_paths": ["answer.pose.x"],
                "mismatch_reasons": ["value_mismatch"],
                "mismatch_categories": ["value_mismatch"],
                "error": None,
            },
        },
    ]


def test_evaluation_bundle_validation_reports_case_manifest_metadata_drift() -> None:
    bundle = evaluation_bundle(tags=("qa", "reobserve"))
    tampered_bundle = json.loads(evaluation_bundle_json(bundle))
    case_name = tampered_bundle["evaluation_cases"][0]["name"]
    original_tags = bundle["evaluation_cases"][0]["tags"]
    tampered_tags = [
        tag
        for tag in original_tags
        if tag != "occlusion"
    ]
    tampered_bundle["evaluation_cases"][0]["tags"] = tampered_tags
    tampered_bundle["coverage"]["by_tag"] = {
        "label": 1,
        "memory": 2,
        "multi_room": 1,
        "occlusion": 1,
        "qa": 3,
        "reobserve": 3,
    }

    validation = validate_evaluation_bundle(tampered_bundle)

    case_check = next(
        check
        for check in validation["checks"]
        if check["name"] == "case_manifest_matches_suite"
    )
    coverage_check = next(
        check
        for check in validation["checks"]
        if check["name"] == "coverage_matches_manifest"
    )
    assert validation["valid"] is False
    assert coverage_check["passed"] is True
    assert case_check["passed"] is False
    assert case_check["differences"] == [
        {
            "path": f"{case_name}.tags",
            "expected": original_tags,
            "actual": tampered_tags,
        },
    ]


def test_evaluation_bundle_validation_reports_invalid_case_metadata_shape() -> None:
    bundle = evaluation_bundle(tags=("qa", "reobserve"))
    tampered_bundle = json.loads(evaluation_bundle_json(bundle))
    tampered_bundle["evaluation_cases"][0]["question"] = ["not", "an", "object"]
    tampered_bundle["bundle_digest"] = lab.evaluation_bundle_digest(tampered_bundle)

    validation = validate_evaluation_bundle(tampered_bundle)

    checks = {check["name"]: check for check in validation["checks"]}
    assert validation["valid"] is False
    assert checks["bundle_digest"]["passed"] is True
    assert checks["case_manifest_matches_suite"]["passed"] is True
    assert checks["coverage_matches_manifest"]["passed"] is True
    assert checks["case_metadata_entries_valid"] == {
        "name": "case_metadata_entries_valid",
        "passed": False,
        "differences": [
            {
                "path": "evaluation_cases[0].question",
                "expected": "object",
                "actual": ["not", "an", "object"],
            },
        ],
    }


def test_evaluation_bundle_validation_reports_scene_fixture_metadata_drift() -> None:
    bundle = evaluation_bundle(tags=("qa", "reobserve"))
    tampered_bundle = json.loads(evaluation_bundle_json(bundle))
    fixture_name = "needs_reobserve"
    fixture = next(
        scene_fixture
        for scene_fixture in tampered_bundle["scene_fixtures"]
        if scene_fixture["name"] == fixture_name
    )
    original_tags = next(
        scene_fixture["tags"]
        for scene_fixture in bundle["scene_fixtures"]
        if scene_fixture["name"] == fixture_name
    )
    tampered_tags = [
        tag
        for tag in original_tags
        if tag != "tabletop"
    ]
    fixture["tags"] = tampered_tags
    del tampered_bundle["coverage"]["by_scene_tag"]["tabletop"]

    validation = validate_evaluation_bundle(tampered_bundle)

    fixture_check = next(
        check
        for check in validation["checks"]
        if check["name"] == "scene_fixture_manifest_covers_cases"
    )
    coverage_check = next(
        check
        for check in validation["checks"]
        if check["name"] == "coverage_matches_manifest"
    )
    assert validation["valid"] is False
    assert coverage_check["passed"] is True
    assert fixture_check["passed"] is False
    assert fixture_check["differences"] == [
        {
            "path": f"{fixture_name}.tags",
            "expected": original_tags,
            "actual": tampered_tags,
        },
    ]


def test_evaluation_bundle_validation_reports_tampered_coverage() -> None:
    bundle = evaluation_bundle(tags=("qa", "reobserve"))
    tampered_bundle = json.loads(evaluation_bundle_json(bundle))
    tampered_bundle["coverage"]["case_count"] = 999
    tampered_bundle["coverage"]["by_tag"]["memory"] = 1

    validation = validate_evaluation_bundle(tampered_bundle)

    assert validation["valid"] is False
    assert validation["checks"][-1]["name"] == "coverage_matches_manifest"
    assert validation["checks"][-1]["passed"] is False
    assert validation["checks"][-1]["expected"] == bundle["coverage"]
    assert validation["checks"][-1]["actual"] == tampered_bundle["coverage"]
    assert validation["checks"][-1]["differences"] == [
        {"path": "by_tag.memory", "expected": 2, "actual": 1},
        {"path": "case_count", "expected": 3, "actual": 999},
    ]


def test_evaluation_bundle_compare_matches_current_run() -> None:
    bundle = evaluation_bundle(tags=("qa", "reobserve"))

    comparison = compare_evaluation_bundle(bundle)

    assert comparison == {
        "matches": True,
        "filters": bundle["filters"],
        "saved_digest": bundle["suite"]["digest"],
        "current_digest": bundle["suite"]["digest"],
        "saved_bundle_digest": bundle["bundle_digest"],
        "current_bundle_digest": bundle["bundle_digest"],
        "checks": [
            {"name": "saved_bundle_valid", "passed": True},
            {
                "name": "suite_digest_matches_current",
                "passed": True,
                "expected": bundle["suite"]["digest"],
                "actual": bundle["suite"]["digest"],
            },
            {
                "name": "bundle_digest_matches_current",
                "passed": True,
                "expected": bundle["bundle_digest"],
                "actual": bundle["bundle_digest"],
            },
            {
                "name": "report_matches_current",
                "passed": True,
                "expected": bundle["report"],
                "actual": bundle["report"],
            },
            {
                "name": "coverage_matches_current",
                "passed": True,
                "expected": bundle["coverage"],
                "actual": bundle["coverage"],
            },
            {
                "name": "case_manifest_matches_current",
                "passed": True,
                "expected": [
                    "multi_room_rearrangement_reobserve_targets",
                    "needs_reobserve_spoon_label_candidates",
                    "needs_reobserve_targets",
                ],
                "actual": [
                    "multi_room_rearrangement_reobserve_targets",
                    "needs_reobserve_spoon_label_candidates",
                    "needs_reobserve_targets",
                ],
            },
            {
                "name": "scene_fixture_manifest_matches_current",
                "passed": True,
                "expected": ["multi_room_rearrangement", "needs_reobserve"],
                "actual": ["multi_room_rearrangement", "needs_reobserve"],
            },
        ],
    }


def test_evaluation_bundle_compare_reports_current_run_drift() -> None:
    bundle = evaluation_bundle(tags=("qa", "reobserve"))
    drifted_bundle = json.loads(evaluation_bundle_json(bundle))
    drifted_bundle["filters"]["tags"] = ["qa", "dynamic"]
    current_bundle = evaluation_bundle(tags=("qa", "dynamic"))

    comparison = compare_evaluation_bundle(drifted_bundle)

    assert comparison["matches"] is False
    assert comparison["saved_digest"] == bundle["suite"]["digest"]
    assert comparison["current_digest"] == current_bundle["suite"]["digest"]
    assert comparison["saved_bundle_digest"] == bundle["bundle_digest"]
    assert comparison["current_bundle_digest"] == current_bundle["bundle_digest"]
    assert comparison["checks"][1] == {
        "name": "suite_digest_matches_current",
        "passed": False,
        "expected": bundle["suite"]["digest"],
        "actual": current_bundle["suite"]["digest"],
    }
    assert comparison["checks"][2] == {
        "name": "bundle_digest_matches_current",
        "passed": False,
        "expected": bundle["bundle_digest"],
        "actual": current_bundle["bundle_digest"],
    }
    coverage_check = next(
        check
        for check in comparison["checks"]
        if check["name"] == "coverage_matches_current"
    )
    assert coverage_check["passed"] is False
    assert coverage_check["expected"] == bundle["coverage"]
    assert coverage_check["actual"] == current_bundle["coverage"]
    assert coverage_check["differences"][0]["path"] == "by_kind.qa"


def test_evaluation_bundle_compare_reports_report_path_drift() -> None:
    bundle = evaluation_bundle(tags=("qa", "reobserve"))
    drifted_bundle = json.loads(evaluation_bundle_json(bundle))
    drifted_bundle["report"]["metrics"]["by_tag"]["qa"]["pass_rate"] = 0.5

    comparison = compare_evaluation_bundle(drifted_bundle)

    report_check = next(
        check
        for check in comparison["checks"]
        if check["name"] == "report_matches_current"
    )
    assert comparison["matches"] is False
    assert report_check["passed"] is False
    assert report_check["differences"] == [
        {
            "path": "metrics.by_tag.qa.pass_rate",
            "expected": 0.5,
            "actual": 1.0,
        },
    ]


def test_evaluation_bundle_compare_reports_coverage_path_drift() -> None:
    bundle = evaluation_bundle(tags=("qa", "reobserve"))
    drifted_bundle = json.loads(evaluation_bundle_json(bundle))
    drifted_bundle["coverage"]["by_scene_fixture"]["needs_reobserve"] = 1
    drifted_bundle["coverage"]["by_tag"]["memory"] = 1

    comparison = compare_evaluation_bundle(drifted_bundle)

    coverage_check = next(
        check
        for check in comparison["checks"]
        if check["name"] == "coverage_matches_current"
    )
    assert comparison["matches"] is False
    assert coverage_check["differences"] == [
        {"path": "by_scene_fixture.needs_reobserve", "expected": 1, "actual": 2},
        {"path": "by_tag.memory", "expected": 1, "actual": 2},
    ]


def test_vla_place_relative_evaluation_requires_relation() -> None:
    missing_relation = EvaluationCase(
        name="custom_missing_relation",
        scene_fixture="tabletop",
        kind="vla_place_relative",
        tags=("vla",),
        target_object="mug_1",
        reference_object="plate_1",
        expected={},
    )

    with pytest.raises(SpatialQAError, match="Evaluation case missing relation: custom_missing_relation"):
        run_evaluation_case_definition(missing_relation)


def test_unknown_evaluation_case_returns_clear_error() -> None:
    with pytest.raises(SpatialQAError, match="Unknown evaluation case: missing"):
        run_evaluation_case("missing")


def _build_custom_counter_scene() -> DynamicSceneGraph:
    graph = DynamicSceneGraph()
    graph.set_agent_pose("agent", Pose3D(0.0, 0.0, 0.0), step=4)
    graph.upsert_object(
        "cube_1",
        "cube",
        Pose3D(0.25, -0.5, 0.4),
        BBox3D(center=Pose3D(0.25, -0.5, 0.4), size=(0.2, 0.2, 0.2)),
        confidence=0.82,
        visible=True,
        step=4,
    )
    return graph


def _build_custom_low_confidence_scene() -> DynamicSceneGraph:
    graph = DynamicSceneGraph()
    graph.set_agent_pose("agent", Pose3D(0.0, 0.0, 0.0), step=4)
    graph.upsert_object(
        "cup_1",
        "cup",
        Pose3D(0.4, -0.2, 0.5),
        BBox3D(center=Pose3D(0.4, -0.2, 0.5), size=(0.1, 0.1, 0.14)),
        confidence=0.2,
        visible=True,
        step=4,
    )
    return graph


def _build_ambiguous_mugs_scene() -> DynamicSceneGraph:
    graph = DynamicSceneGraph()
    graph.set_agent_pose("agent", Pose3D(0.0, 0.0, 0.0), step=1)
    graph.upsert_object(
        "mug_1",
        "mug",
        Pose3D(0.0, 1.0, 0.7),
        BBox3D(center=Pose3D(0.0, 1.0, 0.7), size=(0.12, 0.12, 0.16)),
        confidence=0.9,
        visible=True,
        step=1,
    )
    graph.upsert_object(
        "mug_2",
        "mug",
        Pose3D(0.3, 1.0, 0.7),
        BBox3D(center=Pose3D(0.3, 1.0, 0.7), size=(0.12, 0.12, 0.16)),
        confidence=0.88,
        visible=True,
        step=1,
    )
    return graph


def _build_ambiguous_plates_scene() -> DynamicSceneGraph:
    graph = _build_ambiguous_mugs_scene()
    graph.upsert_object(
        "mug_2",
        "bottle",
        Pose3D(0.3, 1.0, 0.7),
        BBox3D(center=Pose3D(0.3, 1.0, 0.7), size=(0.12, 0.12, 0.16)),
        confidence=0.88,
        visible=True,
        step=1,
    )
    graph.upsert_object(
        "plate_1",
        "plate",
        Pose3D(0.35, 1.0, 0.72),
        BBox3D(center=Pose3D(0.35, 1.0, 0.72), size=(0.26, 0.26, 0.04)),
        confidence=0.9,
        visible=True,
        step=1,
    )
    graph.upsert_object(
        "plate_2",
        "plate",
        Pose3D(0.7, 1.0, 0.72),
        BBox3D(center=Pose3D(0.7, 1.0, 0.72), size=(0.26, 0.26, 0.04)),
        confidence=0.88,
        visible=True,
        step=1,
    )
    return graph
