from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from dsg_spatialqa_lab.graph_tool import GraphTool
from dsg_spatialqa_lab.memory import DynamicSceneGraph
from dsg_spatialqa_lab.qa import SpatialQAEngine
from dsg_spatialqa_lab.scenes import (
    get_scene_fixture,
    list_scene_fixture_metadata,
    load_scene_fixture,
)
from dsg_spatialqa_lab.schema import PlannerResult, QAResponse, SkillCommand, SpatialQAError
from dsg_spatialqa_lab.vla import VLAAnchorPlanner


EvaluationKind = Literal[
    "qa",
    "vla_pick",
    "vla_place_relative",
    "vla_stale_pick",
    "vla_stale_place_relative",
]
SceneLoader = Callable[[], DynamicSceneGraph]
EVALUATION_MANIFEST_SCHEMA_VERSION = "dsg-spatialqa-lab.evaluation-manifest.v1"
EVALUATION_BUNDLE_SCHEMA_VERSION = "dsg-spatialqa-lab.evaluation-bundle.v1"


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    scene_fixture: str
    kind: EvaluationKind
    expected: dict[str, Any]
    tags: tuple[str, ...] = field(default_factory=tuple)
    question: dict[str, Any] = field(default_factory=dict)
    baseline_scene_fixture: str | None = None
    target_object: str | None = None
    target_label: str | None = None
    reference_object: str | None = None
    reference_label: str | None = None
    relation: str | None = None


_EVALUATION_CASES: dict[str, EvaluationCase] = {
    "ambiguous_mug_label_candidates": EvaluationCase(
        name="ambiguous_mug_label_candidates",
        scene_fixture="ambiguous_mugs",
        kind="qa",
        tags=("qa", "label", "ambiguity"),
        question={"type": "label_candidates", "label": "mug", "visible": True},
        expected={
            "answer": {
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
            },
            "evidence_nodes": [
                "mug_1",
                "state:mug_1:1",
                "mug_2",
                "state:mug_2:1",
            ],
            "evidence_edges": [
                "mug_1-STATE_CHANGED-state:mug_1:1-1",
                "mug_2-STATE_CHANGED-state:mug_2:1-1",
            ],
            "confidence": 0.88,
            "needs_reobserve": False,
            "error": None,
        },
    ),
    "ambiguous_mug_pick_by_label": EvaluationCase(
        name="ambiguous_mug_pick_by_label",
        scene_fixture="ambiguous_mugs",
        kind="vla_pick",
        tags=("vla", "label", "ambiguity"),
        target_label="mug",
        expected={
            "status": "ambiguous",
            "error": "Ambiguous label: mug",
            "error_category": "ambiguous_label",
            "ambiguous_ids": ["mug_1", "mug_2"],
            "details": {
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
            },
        },
    ),
    "ambiguous_plate_place_reference_by_label": EvaluationCase(
        name="ambiguous_plate_place_reference_by_label",
        scene_fixture="ambiguous_plates",
        kind="vla_place_relative",
        tags=("vla", "label", "ambiguity", "place"),
        target_object="mug_1",
        reference_label="plate",
        relation="RIGHT_OF",
        expected={
            "status": "ambiguous",
            "error": "Ambiguous label: plate",
            "error_category": "ambiguous_label",
            "ambiguous_ids": ["plate_1", "plate_2"],
            "details": {
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
            },
        },
    ),
    "moved_mug_next_action_validity": EvaluationCase(
        name="moved_mug_next_action_validity",
        scene_fixture="moved_mug",
        baseline_scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "dynamic", "action_validity"),
        target_object="mug_1",
        question={"type": "next_action_validity"},
        expected={
            "answer": {
                "valid": False,
                "needs_replan": True,
                "reason": "stale_object_state",
            },
            "evidence_edges": [
                "mug_1-LEFT_OF-plate_1-1",
                "mug_1-NEAR-plate_1-1",
                "mug_1-ON-table_1-1",
                "mug_1-STATE_CHANGED-state:mug_1:1-1",
                "plate_1-RIGHT_OF-mug_1-1",
                "mug_1-IN_REGION-sink_region-2",
                "mug_1-MOVED_FROM-table_1-2",
                "mug_1-MOVED_TO-sink_region-2",
                "mug_1-STATE_CHANGED-state:mug_1:2-2",
            ],
            "error": None,
        },
    ),
    "moved_mug_object_timeline": EvaluationCase(
        name="moved_mug_object_timeline",
        scene_fixture="moved_mug",
        kind="qa",
        tags=("qa", "memory", "temporal", "dynamic"),
        question={"type": "object_timeline", "object_id": "mug_1"},
        expected={
            "answer": {
                "object_id": "mug_1",
                "timeline": [
                    {
                        "step": 1,
                        "pose": {"x": -0.4, "y": 1.0, "z": 0.78, "yaw": 0.0},
                        "current_location": {"relation": "ON", "dst": "table_1", "step": 1},
                    },
                    {
                        "step": 2,
                        "pose": {"x": 1.2, "y": 0.2, "z": 0.5, "yaw": 0.0},
                        "current_location": {
                            "relation": "IN_REGION",
                            "dst": "sink_region",
                            "step": 2,
                        },
                    },
                ],
            },
            "evidence_edges": [
                "mug_1-ON-table_1-1",
                "mug_1-STATE_CHANGED-state:mug_1:1-1",
                "mug_1-IN_REGION-sink_region-2",
                "mug_1-STATE_CHANGED-state:mug_1:2-2",
            ],
            "error": None,
        },
    ),
    "moved_mug_recent_events": EvaluationCase(
        name="moved_mug_recent_events",
        scene_fixture="moved_mug",
        kind="qa",
        tags=("qa", "dynamic"),
        question={"type": "recent_events", "since_step": 2, "until_step": 2},
        expected={
            "answer": {
                "events": [
                    {"id": "action_move_mug", "type": "action", "label": "move", "step": 2},
                    {
                        "id": "event_move_mug",
                        "type": "event",
                        "label": "move_object",
                        "step": 2,
                    },
                ]
            },
            "error": None,
        },
    ),
    "moved_mug_scene_delta": EvaluationCase(
        name="moved_mug_scene_delta",
        scene_fixture="moved_mug",
        kind="qa",
        tags=("qa", "dynamic", "temporal"),
        question={"type": "scene_delta", "from_step": 1, "to_step": 2},
        expected={
            "answer": {
                "from_step": 1,
                "to_step": 2,
                "objects": [
                    {
                        "object_id": "mug_1",
                        "label": "mug",
                        "changes": ["pose", "last_seen_step", "location"],
                        "from_pose": {"x": -0.4, "y": 1.0, "z": 0.78, "yaw": 0.0},
                        "to_pose": {"x": 1.2, "y": 0.2, "z": 0.5, "yaw": 0.0},
                        "from_location": {"relation": "ON", "dst": "table_1", "step": 1},
                        "to_location": {
                            "relation": "IN_REGION",
                            "dst": "sink_region",
                            "step": 2,
                        },
                    }
                ],
            },
            "error": None,
        },
    ),
    "moved_mug_scene_delta_reversed_window_error": EvaluationCase(
        name="moved_mug_scene_delta_reversed_window_error",
        scene_fixture="moved_mug",
        kind="qa",
        tags=("qa", "dynamic", "temporal", "error"),
        question={"type": "scene_delta", "from_step": 2, "to_step": 1},
        expected={
            "answer": {},
            "evidence_nodes": [],
            "evidence_edges": [],
            "confidence": 0.0,
            "needs_reobserve": False,
            "error": "from_step cannot be greater than to_step",
            "error_category": "invalid_time_window",
        },
    ),
    "moved_mug_stale_pick": EvaluationCase(
        name="moved_mug_stale_pick",
        scene_fixture="moved_mug",
        baseline_scene_fixture="tabletop",
        kind="vla_stale_pick",
        tags=("vla", "dynamic"),
        target_object="mug_1",
        expected={
            "status": "needs_replan",
            "error": "stale_object_state",
            "error_category": "stale_object_state",
        },
    ),
    "moved_mug_stale_place_plate_right_of_mug": EvaluationCase(
        name="moved_mug_stale_place_plate_right_of_mug",
        scene_fixture="moved_mug",
        baseline_scene_fixture="tabletop",
        kind="vla_stale_place_relative",
        tags=("vla", "dynamic", "place"),
        target_object="plate_1",
        reference_object="mug_1",
        relation="RIGHT_OF",
        expected={
            "status": "needs_replan",
            "error": "stale_reference_state",
            "error_category": "stale_reference_state",
        },
    ),
    "moved_mug_world_state": EvaluationCase(
        name="moved_mug_world_state",
        scene_fixture="moved_mug",
        kind="qa",
        tags=("qa", "dynamic", "world_state"),
        question={"type": "world_state", "visible": True},
        expected={
            "answer": {
                "agent_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                "objects": [
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
                ],
            },
            "evidence_nodes": ["agent", "mug_1", "plate_1", "table_1"],
            "evidence_edges": [
                "mug_1-IN_REGION-sink_region-2",
                "mug_1-STATE_CHANGED-state:mug_1:2-2",
            ],
            "error": None,
        },
    ),
    "multi_room_rearrangement_object_room_cereal_box": EvaluationCase(
        name="multi_room_rearrangement_object_room_cereal_box",
        scene_fixture="multi_room_rearrangement",
        kind="qa",
        tags=("qa", "dynamic", "multi_room", "room"),
        question={"type": "object_room", "object_id": "cereal_box_1"},
        expected={
            "answer": {
                "object_id": "cereal_box_1",
                "room_id": "pantry",
                "room_label": "Pantry",
                "path": [
                    {
                        "src": "cereal_box_1",
                        "relation": "IN_REGION",
                        "dst": "pantry_shelf",
                        "step": 2,
                    },
                    {
                        "src": "pantry_shelf",
                        "relation": "IN_ROOM",
                        "dst": "pantry",
                        "step": 1,
                    },
                ],
            },
            "evidence_nodes": ["cereal_box_1", "pantry_shelf", "pantry"],
            "evidence_edges": [
                "cereal_box_1-IN_REGION-pantry_shelf-2",
                "pantry_shelf-IN_ROOM-pantry-1",
            ],
            "confidence": 0.92,
            "needs_reobserve": False,
            "error": None,
        },
    ),
    "multi_room_rearrangement_recent_events": EvaluationCase(
        name="multi_room_rearrangement_recent_events",
        scene_fixture="multi_room_rearrangement",
        kind="qa",
        tags=("qa", "dynamic", "multi_room", "temporal"),
        question={"type": "recent_events", "since_step": 2, "until_step": 3},
        expected={
            "answer": {
                "events": [
                    {
                        "id": "action_move_cereal_box",
                        "type": "action",
                        "label": "move",
                        "step": 2,
                    },
                    {
                        "id": "event_move_cereal_box",
                        "type": "event",
                        "label": "move_object",
                        "step": 2,
                    },
                ],
            },
            "error": None,
        },
    ),
    "multi_room_rearrangement_reobserve_targets": EvaluationCase(
        name="multi_room_rearrangement_reobserve_targets",
        scene_fixture="multi_room_rearrangement",
        kind="qa",
        tags=("qa", "memory", "multi_room", "occlusion", "reobserve"),
        question={"type": "reobserve_targets", "label": "fork"},
        expected={
            "answer": {
                "count": 1,
                "objects": [
                    {
                        "object_id": "fork_1",
                        "label": "fork",
                        "pose": {"x": -0.25, "y": 0.95, "z": 0.82, "yaw": 0.0},
                        "visible": False,
                        "confidence": 0.2,
                        "last_seen_step": 1,
                        "last_seen_pose": {"x": -0.25, "y": 0.95, "z": 0.82, "yaw": 0.0},
                        "state_step": 3,
                    }
                ],
            },
            "evidence_edges": ["fork_1-STATE_CHANGED-state:fork_1:3-3"],
            "needs_reobserve": True,
            "error": None,
        },
    ),
    "multi_room_rearrangement_scene_delta": EvaluationCase(
        name="multi_room_rearrangement_scene_delta",
        scene_fixture="multi_room_rearrangement",
        kind="qa",
        tags=("qa", "dynamic", "multi_room", "move", "temporal"),
        question={"type": "scene_delta", "from_step": 1, "to_step": 2},
        expected={
            "answer": {
                "from_step": 1,
                "to_step": 2,
                "agent": {"changed": True},
                "objects": [
                    {
                        "object_id": "cereal_box_1",
                        "label": "cereal_box",
                        "changes": ["pose", "last_seen_step", "location"],
                        "from_location": {
                            "relation": "IN_REGION",
                            "dst": "prep_counter",
                            "step": 1,
                        },
                        "to_location": {
                            "relation": "IN_REGION",
                            "dst": "pantry_shelf",
                            "step": 2,
                        },
                    }
                ],
            },
            "error": None,
        },
    ),
    "needs_reobserve_spoon_pick": EvaluationCase(
        name="needs_reobserve_spoon_pick",
        scene_fixture="needs_reobserve",
        kind="vla_pick",
        tags=("vla", "pick", "reobserve"),
        target_object="spoon_1",
        expected={
            "status": "needs_reobserve",
            "error": "needs_reobserve",
            "error_category": "needs_reobserve",
            "needs_reobserve": True,
        },
    ),
    "needs_reobserve_targets": EvaluationCase(
        name="needs_reobserve_targets",
        scene_fixture="needs_reobserve",
        kind="qa",
        tags=("qa", "memory", "reobserve"),
        question={"type": "reobserve_targets"},
        expected={
            "answer": {
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
            },
            "evidence_edges": ["spoon_1-STATE_CHANGED-state:spoon_1:2-2"],
            "needs_reobserve": True,
            "error": None,
        },
    ),
    "relation_shift_relation_timeline": EvaluationCase(
        name="relation_shift_relation_timeline",
        scene_fixture="relation_shift",
        kind="qa",
        tags=("qa", "dynamic", "temporal", "relations", "move"),
        question={
            "type": "relation_timeline",
            "src": "mug_1",
            "dst": "plate_1",
            "reference_frame": "agent",
        },
        expected={
            "answer": {
                "timeline": [
                    {
                        "id": "mug_1-LEFT_OF-plate_1-1",
                        "relation": "LEFT_OF",
                        "step": 1,
                    },
                    {
                        "id": "mug_1-NEAR-plate_1-1",
                        "relation": "NEAR",
                        "step": 1,
                    },
                    {
                        "id": "mug_1-NEAR-plate_1-2",
                        "relation": "NEAR",
                        "step": 2,
                    },
                    {
                        "id": "mug_1-RIGHT_OF-plate_1-2",
                        "relation": "RIGHT_OF",
                        "step": 2,
                    },
                ],
            },
            "evidence_edges": [
                "mug_1-LEFT_OF-plate_1-1",
                "mug_1-NEAR-plate_1-1",
                "mug_1-NEAR-plate_1-2",
                "mug_1-RIGHT_OF-plate_1-2",
            ],
            "needs_reobserve": False,
            "error": None,
        },
    ),
    "tabletop_mug_pick": EvaluationCase(
        name="tabletop_mug_pick",
        scene_fixture="tabletop",
        kind="vla_pick",
        tags=("vla", "anchor", "pick"),
        target_object="mug_1",
        expected={
            "status": "ok",
            "command": {
                "skill": "pick",
                "target_object": "mug_1",
                "target_pose": {"x": -0.4, "y": 1.0, "z": 0.78, "yaw": 0.0},
                "evidence": ["mug_1"],
            },
        },
    ),
    "tabletop_agent_location": EvaluationCase(
        name="tabletop_agent_location",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "foundation"),
        question={"type": "agent_location"},
        expected={
            "answer": {
                "agent_id": "agent",
                "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            },
            "evidence_nodes": ["agent"],
            "error": None,
        },
    ),
    "tabletop_agent_timeline": EvaluationCase(
        name="tabletop_agent_timeline",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "memory", "temporal"),
        question={"type": "agent_timeline"},
        expected={
            "answer": {
                "agent_id": "agent",
                "timeline": [
                    {
                        "step": 1,
                        "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                    }
                ],
            },
            "evidence_edges": ["agent-STATE_CHANGED-state:agent:1-1"],
            "error": None,
        },
    ),
    "tabletop_graph_query_mug_plate": EvaluationCase(
        name="tabletop_graph_query_mug_plate",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "retrieval", "graph_query"),
        question={
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
        expected={
            "answer": {
                "nodes": [
                    {"id": "mug_1", "type": "object", "label": "mug"},
                    {"id": "plate_1", "type": "object", "label": "plate"},
                ],
                "edges": [
                    {
                        "id": "mug_1-LEFT_OF-plate_1-1",
                        "relation": "LEFT_OF",
                        "step": 1,
                    }
                ],
            },
            "evidence_nodes": ["mug_1", "plate_1"],
            "evidence_edges": ["mug_1-LEFT_OF-plate_1-1"],
            "error": None,
        },
    ),
    "tabletop_missing_object_location_error": EvaluationCase(
        name="tabletop_missing_object_location_error",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "foundation", "error"),
        question={"type": "object_location", "object_id": "missing_object"},
        expected={
            "answer": {},
            "evidence_nodes": [],
            "evidence_edges": [],
            "confidence": 0.0,
            "needs_reobserve": False,
            "error": "Object not found: missing_object",
            "error_category": "missing_object",
        },
    ),
    "tabletop_missing_object_pick_error": EvaluationCase(
        name="tabletop_missing_object_pick_error",
        scene_fixture="tabletop",
        kind="vla_pick",
        tags=("vla", "pick", "error"),
        target_object="missing_object",
        expected={
            "status": "error",
            "command": None,
            "error": "Object not found: missing_object",
            "error_category": "missing_object",
            "needs_reobserve": False,
            "needs_replan": False,
            "ambiguous_ids": [],
            "details": {},
        },
    ),
    "tabletop_missing_reference_place_error": EvaluationCase(
        name="tabletop_missing_reference_place_error",
        scene_fixture="tabletop",
        kind="vla_place_relative",
        tags=("vla", "place", "error"),
        target_object="mug_1",
        reference_object="missing_object",
        relation="RIGHT_OF",
        expected={
            "status": "error",
            "command": None,
            "error": "Object not found: missing_object",
            "error_category": "missing_object",
            "needs_reobserve": False,
            "needs_replan": False,
            "ambiguous_ids": [],
            "details": {},
        },
    ),
    "tabletop_unsupported_place_relation_error": EvaluationCase(
        name="tabletop_unsupported_place_relation_error",
        scene_fixture="tabletop",
        kind="vla_place_relative",
        tags=("vla", "place", "error"),
        target_object="mug_1",
        reference_object="plate_1",
        relation="ABOVE",
        expected={
            "status": "error",
            "command": None,
            "error": "Unsupported place relation: ABOVE",
            "error_category": "unsupported_relation",
            "needs_reobserve": False,
            "needs_replan": False,
            "ambiguous_ids": [],
            "details": {},
        },
    ),
    "tabletop_nearest_candidate_plate": EvaluationCase(
        name="tabletop_nearest_candidate_plate",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "retrieval", "nearest"),
        question={"type": "nearest_object", "src": "mug_1", "candidates": ["plate_1"]},
        expected={
            "answer": {
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
            },
            "evidence_nodes": ["mug_1", "plate_1"],
            "error": None,
        },
    ),
    "tabletop_retrieve_subgraph_mug": EvaluationCase(
        name="tabletop_retrieve_subgraph_mug",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "retrieval", "retrieve_subgraph"),
        question={"type": "retrieve_subgraph", "query": "mug", "max_nodes": 3, "hops": 1},
        expected={
            "answer": {
                "nodes": [
                    {"id": "mug_1", "type": "object", "label": "mug"},
                    {"id": "plate_1", "type": "object", "label": "plate"},
                    {"id": "table_1", "type": "object", "label": "table"},
                ],
                "edges": [
                    {"id": "mug_1-LEFT_OF-plate_1-1", "relation": "LEFT_OF", "step": 1},
                    {"id": "mug_1-NEAR-plate_1-1", "relation": "NEAR", "step": 1},
                    {"id": "mug_1-ON-table_1-1", "relation": "ON", "step": 1},
                ],
            },
            "evidence_nodes": ["mug_1", "plate_1", "table_1"],
            "evidence_edges": [
                "mug_1-LEFT_OF-plate_1-1",
                "mug_1-NEAR-plate_1-1",
                "mug_1-ON-table_1-1",
            ],
            "error": None,
        },
    ),
    "tabletop_object_history_mug": EvaluationCase(
        name="tabletop_object_history_mug",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "foundation", "memory"),
        question={"type": "object_history", "object_id": "mug_1"},
        expected={
            "answer": {
                "object_id": "mug_1",
                "relations": ["LEFT_OF", "NEAR", "ON", "STATE_CHANGED", "RIGHT_OF"],
                "steps": [1, 1, 1, 1, 1],
            },
            "evidence_edges": [
                "mug_1-LEFT_OF-plate_1-1",
                "mug_1-NEAR-plate_1-1",
                "mug_1-ON-table_1-1",
                "mug_1-STATE_CHANGED-state:mug_1:1-1",
                "plate_1-RIGHT_OF-mug_1-1",
            ],
            "error": None,
        },
    ),
    "tabletop_object_location": EvaluationCase(
        name="tabletop_object_location",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "memory"),
        question={"type": "object_location", "object_id": "mug_1"},
        expected={
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
    ),
    "tabletop_object_status_plate": EvaluationCase(
        name="tabletop_object_status_plate",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "foundation"),
        question={"type": "object_status", "object_id": "plate_1"},
        expected={
            "answer": {
                "object_id": "plate_1",
                "label": "plate",
                "visible": True,
                "confidence": 0.9,
                "last_seen_step": 1,
                "last_seen_pose": {"x": 0.35, "y": 1.0, "z": 0.72, "yaw": 0.0},
                "needs_reobserve": False,
            },
            "evidence_edges": ["plate_1-STATE_CHANGED-state:plate_1:1-1"],
            "error": None,
        },
    ),
    "tabletop_relation_timeline": EvaluationCase(
        name="tabletop_relation_timeline",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "memory", "temporal", "relations"),
        question={
            "type": "relation_timeline",
            "src": "mug_1",
            "dst": "plate_1",
            "reference_frame": "agent",
        },
        expected={
            "answer": {
                "timeline": [
                    {
                        "id": "mug_1-LEFT_OF-plate_1-1",
                        "relation": "LEFT_OF",
                        "step": 1,
                    },
                    {
                        "id": "mug_1-NEAR-plate_1-1",
                        "relation": "NEAR",
                        "step": 1,
                    },
                ],
            },
            "evidence_edges": [
                "mug_1-LEFT_OF-plate_1-1",
                "mug_1-NEAR-plate_1-1",
            ],
            "error": None,
        },
    ),
    "tabletop_relative_relation_mug_left_of_plate": EvaluationCase(
        name="tabletop_relative_relation_mug_left_of_plate",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "foundation", "relations"),
        question={
            "type": "relative_relation",
            "src": "mug_1",
            "dst": "plate_1",
            "relation": "LEFT_OF",
            "reference_frame": "agent",
        },
        expected={
            "answer": {
                "holds": True,
                "relation": "LEFT_OF",
                "src": "mug_1",
                "dst": "plate_1",
            },
            "evidence_edges": ["mug_1-LEFT_OF-plate_1-1"],
            "error": None,
        },
    ),
    "tabletop_scene_snapshot_step1": EvaluationCase(
        name="tabletop_scene_snapshot_step1",
        scene_fixture="tabletop",
        kind="qa",
        tags=("qa", "snapshot", "memory", "temporal"),
        question={"type": "scene_snapshot", "step": 1, "visible": True},
        expected={
            "answer": {
                "step": 1,
                "agent": {
                    "agent_id": "agent",
                    "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                    "state_step": 1,
                },
                "objects": [
                    {
                        "object_id": "mug_1",
                        "label": "mug",
                        "state_step": 1,
                        "current_location": {"relation": "ON", "dst": "table_1", "step": 1},
                    },
                    {"object_id": "plate_1", "label": "plate", "state_step": 1},
                    {"object_id": "table_1", "label": "table", "state_step": 1},
                ],
            },
            "evidence_nodes": [
                "agent",
                "state:agent:1",
                "mug_1",
                "state:mug_1:1",
                "plate_1",
                "state:plate_1:1",
                "table_1",
                "state:table_1:1",
            ],
            "evidence_edges": [
                "agent-STATE_CHANGED-state:agent:1-1",
                "mug_1-ON-table_1-1",
                "mug_1-STATE_CHANGED-state:mug_1:1-1",
                "plate_1-STATE_CHANGED-state:plate_1:1-1",
                "table_1-STATE_CHANGED-state:table_1:1-1",
            ],
            "error": None,
        },
    ),
    "tabletop_place_mug_right_of_plate": EvaluationCase(
        name="tabletop_place_mug_right_of_plate",
        scene_fixture="tabletop",
        kind="vla_place_relative",
        tags=("vla", "anchor", "place"),
        target_object="mug_1",
        reference_object="plate_1",
        relation="RIGHT_OF",
        expected={
            "status": "ok",
            "command": {
                "skill": "place_relative",
                "target_object": "mug_1",
                "reference_object": "plate_1",
                "target_pose": {"x": 0.64, "y": 1.0, "z": 0.82, "yaw": 0.0},
                "parameters": {"relation": "RIGHT_OF"},
            },
        },
    ),
}


def list_evaluation_cases(
    tags: Sequence[str] | None = None,
    kinds: Sequence[EvaluationKind] | None = None,
    question_types: Sequence[str] | None = None,
    names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    return tuple(
        case.name
        for case in _select_cases(
            tuple(_EVALUATION_CASES.values()),
            names=names,
            tags=tags,
            kinds=kinds,
            question_types=question_types,
        )
    )


def list_evaluation_case_metadata(
    tags: Sequence[str] | None = None,
    kinds: Sequence[EvaluationKind] | None = None,
    question_types: Sequence[str] | None = None,
    names: Sequence[str] | None = None,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        evaluation_case_metadata(_EVALUATION_CASES[name])
        for name in list_evaluation_cases(
            tags=tags,
            kinds=kinds,
            question_types=question_types,
            names=names,
        )
    )


def evaluation_case_listing(
    *,
    tags: Sequence[str] | None = None,
    kinds: Sequence[EvaluationKind] | None = None,
    question_types: Sequence[str] | None = None,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    cases = list_evaluation_case_metadata(
        tags=tags,
        kinds=kinds,
        question_types=question_types,
        names=names,
    )
    listing = {
        "filters": {
            "names": list(names or ()),
            "tags": list(tags or ()),
            "kinds": list(kinds or ()),
            "question_types": list(question_types or ()),
        },
        "case_count": len(cases),
        "evaluation_cases": list(cases),
    }
    listing["digest"] = evaluation_case_listing_digest(listing)
    return listing


def evaluation_case_listing_digest(listing: Mapping[str, Any]) -> str:
    payload = {
        "filters": listing.get("filters"),
        "case_count": listing.get("case_count"),
        "evaluation_cases": listing.get("evaluation_cases"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluation_case_listing_json(listing: Mapping[str, Any]) -> str:
    return json.dumps(listing, indent=2, sort_keys=True) + "\n"


def save_evaluation_case_listing(
    path: str | Path,
    *,
    names: Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
    kinds: Sequence[EvaluationKind] | None = None,
    question_types: Sequence[str] | None = None,
) -> Path:
    target = Path(path)
    listing = evaluation_case_listing(
        names=names,
        tags=tags,
        kinds=kinds,
        question_types=question_types,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(evaluation_case_listing_json(listing), encoding="utf-8")
    return target


def load_evaluation_case_listing(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SpatialQAError("Evaluation case listing JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_evaluation_case_listing(listing: Mapping[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    expected_digest = evaluation_case_listing_digest(listing)
    listing_digest = _string_or_none(listing.get("digest"))
    _append_validation_check(
        checks,
        name="listing_digest",
        passed=listing_digest == expected_digest,
        expected=expected_digest,
        actual=listing_digest,
    )

    expected_case_count = len(_case_manifest_from_bundle(listing))
    actual_case_count = listing.get("case_count")
    _append_validation_check(
        checks,
        name="case_count_matches_listing",
        passed=actual_case_count == expected_case_count,
        expected=expected_case_count,
        actual=actual_case_count,
    )

    return {
        "valid": all(check["passed"] is True for check in checks),
        "digest": listing_digest,
        "checks": checks,
    }


def compare_evaluation_case_listing(listing: Mapping[str, Any]) -> dict[str, Any]:
    filters = _filters_from_bundle(listing)
    current_listing = evaluation_case_listing(
        names=_optional_filter_values(filters, "names"),
        tags=_optional_filter_values(filters, "tags"),
        kinds=cast(Sequence[EvaluationKind] | None, _optional_filter_values(filters, "kinds")),
        question_types=_optional_filter_values(filters, "question_types"),
    )
    saved_digest = _string_or_none(listing.get("digest"))
    current_digest = _string_or_none(current_listing.get("digest"))
    checks: list[dict[str, Any]] = []

    validation = validate_evaluation_case_listing(listing)
    _append_validation_check(
        checks,
        name="saved_listing_valid",
        passed=validation["valid"] is True,
    )
    _append_validation_check(
        checks,
        name="listing_digest_matches_current",
        passed=saved_digest == current_digest,
        expected=saved_digest,
        actual=current_digest,
    )
    _append_validation_check(
        checks,
        name="case_count_matches_current",
        passed=listing.get("case_count") == current_listing["case_count"],
        expected=listing.get("case_count"),
        actual=current_listing["case_count"],
    )
    case_metadata_differences = _manifest_entry_differences(
        _case_manifest_from_bundle(listing),
        _case_manifest_from_bundle(current_listing),
    )
    _append_validation_check(
        checks,
        name="case_metadata_matches_current",
        passed=case_metadata_differences == [],
        expected=_case_names_from_bundle(listing),
        actual=_case_names_from_bundle(current_listing),
    )
    if case_metadata_differences:
        checks[-1]["differences"] = case_metadata_differences

    return {
        "matches": all(check["passed"] is True for check in checks),
        "filters": {
            "names": list(_optional_filter_values(filters, "names") or ()),
            "tags": list(_optional_filter_values(filters, "tags") or ()),
            "kinds": list(_optional_filter_values(filters, "kinds") or ()),
            "question_types": list(_optional_filter_values(filters, "question_types") or ()),
        },
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "checks": checks,
    }


def evaluation_cases_metadata(
    cases: Sequence[EvaluationCase],
    *,
    tags: Sequence[str] | None = None,
    kinds: Sequence[EvaluationKind] | None = None,
    question_types: Sequence[str] | None = None,
    names: Sequence[str] | None = None,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        evaluation_case_metadata(case)
        for case in _select_cases(
            cases,
            names=names,
            tags=tags,
            kinds=kinds,
            question_types=question_types,
        )
    )


def evaluation_case_metadata(case: EvaluationCase) -> dict[str, Any]:
    question_type = case.question.get("type")
    scene_description, scene_tags = _scene_fixture_metadata(case.scene_fixture)
    baseline_scene_description, baseline_scene_tags = _optional_scene_fixture_metadata(
        case.baseline_scene_fixture
    )
    return {
        "name": case.name,
        "scene_fixture": case.scene_fixture,
        "scene_description": scene_description,
        "scene_tags": scene_tags,
        "kind": case.kind,
        "tags": list(case.tags),
        "question": dict(case.question),
        "question_type": question_type if isinstance(question_type, str) else None,
        "baseline_scene_fixture": case.baseline_scene_fixture,
        "baseline_scene_description": baseline_scene_description,
        "baseline_scene_tags": baseline_scene_tags,
        "target_object": case.target_object,
        "target_label": case.target_label,
        "reference_object": case.reference_object,
        "reference_label": case.reference_label,
        "relation": case.relation,
        "expected_keys": sorted(case.expected),
    }


def _optional_scene_fixture_metadata(scene_fixture: str | None) -> tuple[str | None, list[str]]:
    if scene_fixture is None:
        return None, []
    return _scene_fixture_metadata(scene_fixture)


def _scene_fixture_metadata(scene_fixture: str) -> tuple[str | None, list[str]]:
    try:
        fixture = get_scene_fixture(scene_fixture)
    except SpatialQAError:
        return None, []
    return fixture.description, list(fixture.tags)


def get_evaluation_case(name: str) -> EvaluationCase:
    case = _EVALUATION_CASES.get(name)
    if case is None:
        raise SpatialQAError(f"Unknown evaluation case: {name}")
    return case


def run_evaluation_case(
    name: str,
    *,
    scene_loaders: Mapping[str, SceneLoader] | None = None,
) -> dict[str, Any]:
    return run_evaluation_case_definition(get_evaluation_case(name), scene_loaders=scene_loaders)


def run_evaluation_case_definition(
    case: EvaluationCase,
    *,
    scene_loaders: Mapping[str, SceneLoader] | None = None,
) -> dict[str, Any]:
    actual = _run_case(case, scene_loaders)
    mismatches = _expected_mismatches(actual, case.expected)
    return {
        "case": case.name,
        "scene_fixture": case.scene_fixture,
        "kind": case.kind,
        "question_type": _case_question_type(case),
        "tags": list(case.tags),
        "passed": not mismatches,
        "actual": actual,
        "expected": case.expected,
        "mismatches": mismatches,
    }


def run_evaluation_cases(
    cases: Sequence[EvaluationCase],
    *,
    tags: Sequence[str] | None = None,
    kinds: Sequence[EvaluationKind] | None = None,
    question_types: Sequence[str] | None = None,
    names: Sequence[str] | None = None,
    scene_loaders: Mapping[str, SceneLoader] | None = None,
) -> dict[str, Any]:
    selected_cases = _select_cases(
        cases,
        names=names,
        tags=tags,
        kinds=kinds,
        question_types=question_types,
    )
    results = [
        run_evaluation_case_definition(case, scene_loaders=scene_loaders)
        for case in selected_cases
    ]
    return _suite_result(results)


def run_evaluation_suite(
    case_names: Sequence[str] | None = None,
    *,
    names: Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
    kinds: Sequence[EvaluationKind] | None = None,
    question_types: Sequence[str] | None = None,
    scene_loaders: Mapping[str, SceneLoader] | None = None,
) -> dict[str, Any]:
    if case_names is not None and names is not None:
        raise SpatialQAError("Use either case_names or names, not both")

    selected_names = names if names is not None else case_names
    selected_cases = list_evaluation_cases(
        tags=tags,
        kinds=kinds,
        question_types=question_types,
        names=selected_names,
    )
    results = [
        run_evaluation_case(name, scene_loaders=scene_loaders)
        for name in selected_cases
    ]
    return _suite_result(results)


def evaluation_report(suite: Mapping[str, Any]) -> dict[str, Any]:
    summary = cast(Mapping[str, Any], suite["summary"])
    breakdown = cast(Mapping[str, Any], suite["breakdown"])
    results = cast(Sequence[Mapping[str, Any]], suite["results"])
    total = int(summary["total"])
    passed = int(summary["passed"])
    failure_path_index: dict[str, dict[str, Any]] = {}
    failure_reason_index: dict[str, dict[str, Any]] = {}
    failure_category_index: dict[str, dict[str, Any]] = {}
    failed_cases: list[dict[str, Any]] = []

    for result in results:
        if result["passed"] is True:
            continue
        case_name = str(result["case"])
        mismatches = cast(Sequence[Mapping[str, Any]], result.get("mismatches", ()))
        mismatch_paths = sorted({str(mismatch["path"]) for mismatch in mismatches})
        mismatch_reasons = sorted({str(mismatch["reason"]) for mismatch in mismatches})
        mismatch_categories = sorted({_mismatch_category(mismatch) for mismatch in mismatches})
        for mismatch in mismatches:
            path = str(mismatch["path"])
            path_entry = failure_path_index.setdefault(path, {"count": 0, "cases": set()})
            path_entry["count"] = int(path_entry["count"]) + 1
            cast(set[str], path_entry["cases"]).add(case_name)
            reason = str(mismatch["reason"])
            entry = failure_reason_index.setdefault(reason, {"count": 0, "cases": set()})
            entry["count"] = int(entry["count"]) + 1
            cast(set[str], entry["cases"]).add(case_name)
            category = _mismatch_category(mismatch)
            category_entry = failure_category_index.setdefault(
                category,
                {"count": 0, "cases": set()},
            )
            category_entry["count"] = int(category_entry["count"]) + 1
            cast(set[str], category_entry["cases"]).add(case_name)
        actual = result.get("actual")
        error = actual.get("error") if isinstance(actual, Mapping) else None
        failed_cases.append(
            {
                "case": case_name,
                "kind": str(result["kind"]),
                "scene_fixture": str(result["scene_fixture"]),
                "tags": list(cast(Sequence[str], result["tags"])),
                "mismatch_count": len(mismatches),
                "mismatch_paths": mismatch_paths,
                "mismatch_reasons": mismatch_reasons,
                "mismatch_categories": mismatch_categories,
                "error": error,
            }
        )

    failure_reasons = [
        {
            "reason": reason,
            "count": int(entry["count"]),
            "cases": sorted(cast(set[str], entry["cases"])),
        }
        for reason, entry in sorted(failure_reason_index.items())
    ]
    failure_categories = [
        {
            "category": category,
            "count": int(entry["count"]),
            "cases": sorted(cast(set[str], entry["cases"])),
        }
        for category, entry in sorted(failure_category_index.items())
    ]
    failure_paths = [
        {
            "path": path,
            "count": int(entry["count"]),
            "cases": sorted(cast(set[str], entry["cases"])),
        }
        for path, entry in sorted(failure_path_index.items())
    ]
    report = {
        "digest": str(suite["digest"]),
        "summary": dict(summary),
        "metrics": _evaluation_report_metrics(
            total=total,
            passed=passed,
            breakdown=breakdown,
        ),
        "evidence_metrics": _evaluation_evidence_metrics(results),
        "case_digests": _evaluation_case_digests(results),
        "failed_cases": failed_cases,
        "runtime_error_categories": _runtime_error_category_summary(results),
        "failure_reasons": failure_reasons,
        "failure_categories": failure_categories,
        "failure_paths": failure_paths,
        "breakdown": dict(breakdown),
    }
    report["report_digest"] = evaluation_report_digest(report)
    return report


def _evaluation_report_metrics(
    *,
    total: int,
    passed: int,
    breakdown: Mapping[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, Any] = _evaluation_summary_metrics(total=total, passed=passed)
    for group_name in ("by_kind", "by_question_type", "by_scene_fixture", "by_tag"):
        metrics[group_name] = _evaluation_breakdown_group_metrics(
            breakdown.get(group_name, {}),
        )
    return metrics


def _evaluation_summary_metrics(*, total: int, passed: int) -> dict[str, float | int]:
    failed = total - passed
    return {
        "case_count": total,
        "passed_case_count": passed,
        "failed_case_count": failed,
        "pass_rate": 0.0 if total == 0 else passed / total,
        "failure_rate": 0.0 if total == 0 else failed / total,
    }


def _evaluation_breakdown_group_metrics(
    group: object,
) -> dict[str, dict[str, float | int]]:
    if not isinstance(group, Mapping):
        return {}
    return {
        str(name): _evaluation_summary_metrics(
            total=int(cast(Mapping[str, Any], summary)["total"]),
            passed=int(cast(Mapping[str, Any], summary)["passed"]),
        )
        for name, summary in sorted(group.items())
        if isinstance(summary, Mapping)
    }


def _evaluation_evidence_metrics(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = _evaluation_evidence_summary_metrics(results)
    metrics["by_kind"] = _evaluation_evidence_breakdown_by_kind(results)
    metrics["by_question_type"] = _evaluation_evidence_breakdown_by_question_type(results)
    metrics["by_scene_fixture"] = _evaluation_evidence_breakdown_by_scene_fixture(results)
    metrics["by_tag"] = _evaluation_evidence_breakdown_by_tag(results)
    return metrics


def _evaluation_case_digests(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "case": str(result["case"]),
            "kind": str(result["kind"]),
            "question_type": (
                str(result["question_type"])
                if isinstance(result.get("question_type"), str)
                else None
            ),
            "scene_fixture": str(result["scene_fixture"]),
            "passed": result["passed"] is True,
            "digest": _stable_digest(result),
        }
        for result in results
    ]


def _stable_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256_hexdigest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _evaluation_evidence_summary_metrics(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    case_count = len(results)
    evidence_node_count = 0
    evidence_edge_count = 0
    command_evidence_count = 0
    cases_with_evidence_count = 0
    for result in results:
        counts = _result_evidence_counts(result)
        evidence_node_count += counts["evidence_node_count"]
        evidence_edge_count += counts["evidence_edge_count"]
        command_evidence_count += counts["command_evidence_count"]
        if counts["total_evidence_item_count"] > 0:
            cases_with_evidence_count += 1
    total_evidence_item_count = (
        evidence_node_count + evidence_edge_count + command_evidence_count
    )
    return {
        "case_count": case_count,
        "cases_with_evidence_count": cases_with_evidence_count,
        "cases_without_evidence_count": case_count - cases_with_evidence_count,
        "evidence_node_count": evidence_node_count,
        "evidence_edge_count": evidence_edge_count,
        "command_evidence_count": command_evidence_count,
        "total_evidence_item_count": total_evidence_item_count,
        "average_evidence_item_count": (
            0.0 if case_count == 0 else total_evidence_item_count / case_count
        ),
    }


def _result_evidence_counts(result: Mapping[str, Any]) -> dict[str, int]:
    actual = result.get("actual")
    evidence_node_count = 0
    evidence_edge_count = 0
    command_evidence_count = 0
    if isinstance(actual, Mapping):
        evidence_node_count = _sequence_count(actual.get("evidence_nodes"))
        evidence_edge_count = _sequence_count(actual.get("evidence_edges"))
        command = actual.get("command")
        if isinstance(command, Mapping):
            command_evidence_count = _sequence_count(command.get("evidence"))
    return {
        "evidence_node_count": evidence_node_count,
        "evidence_edge_count": evidence_edge_count,
        "command_evidence_count": command_evidence_count,
        "total_evidence_item_count": (
            evidence_node_count + evidence_edge_count + command_evidence_count
        ),
    }


def _sequence_count(value: object) -> int:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return len(value)
    return 0


def _evaluation_evidence_breakdown_by_kind(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | int]]:
    by_kind: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        by_kind.setdefault(str(result["kind"]), []).append(result)
    return {
        kind: _evaluation_evidence_summary_metrics(by_kind[kind])
        for kind in sorted(by_kind)
    }


def _evaluation_evidence_breakdown_by_question_type(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | int]]:
    by_question_type: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        question_type = result.get("question_type")
        if isinstance(question_type, str):
            by_question_type.setdefault(question_type, []).append(result)
    return {
        question_type: _evaluation_evidence_summary_metrics(
            by_question_type[question_type]
        )
        for question_type in sorted(by_question_type)
    }


def _evaluation_evidence_breakdown_by_scene_fixture(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | int]]:
    by_scene_fixture: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        by_scene_fixture.setdefault(str(result["scene_fixture"]), []).append(result)
    return {
        scene_fixture: _evaluation_evidence_summary_metrics(
            by_scene_fixture[scene_fixture]
        )
        for scene_fixture in sorted(by_scene_fixture)
    }


def _evaluation_evidence_breakdown_by_tag(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | int]]:
    by_tag: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        for tag in result["tags"]:
            by_tag.setdefault(str(tag), []).append(result)
    return {
        tag: _evaluation_evidence_summary_metrics(by_tag[tag])
        for tag in sorted(by_tag)
    }


def evaluation_bundle(
    *,
    names: Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
    kinds: Sequence[EvaluationKind] | None = None,
    question_types: Sequence[str] | None = None,
    scene_loaders: Mapping[str, SceneLoader] | None = None,
) -> dict[str, Any]:
    case_manifest = list_evaluation_case_metadata(
        names=names,
        tags=tags,
        kinds=kinds,
        question_types=question_types,
    )
    suite = run_evaluation_suite(
        names=names,
        tags=tags,
        kinds=kinds,
        question_types=question_types,
        scene_loaders=scene_loaders,
    )
    scene_fixture_manifest = _scene_fixture_manifest_for_cases(case_manifest)
    bundle = {
        "schema_version": EVALUATION_BUNDLE_SCHEMA_VERSION,
        "filters": {
            "names": list(names or ()),
            "tags": list(tags or ()),
            "kinds": list(kinds or ()),
            "question_types": list(question_types or ()),
        },
        "scene_fixtures": scene_fixture_manifest,
        "evaluation_cases": list(case_manifest),
        "coverage": _evaluation_bundle_coverage(case_manifest, scene_fixture_manifest),
        "suite": suite,
        "report": evaluation_report(suite),
    }
    bundle["bundle_digest"] = evaluation_bundle_digest(bundle)
    return bundle


def evaluation_manifest(
    *,
    names: Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
    kinds: Sequence[EvaluationKind] | None = None,
    question_types: Sequence[str] | None = None,
) -> dict[str, Any]:
    case_manifest = list_evaluation_case_metadata(
        names=names,
        tags=tags,
        kinds=kinds,
        question_types=question_types,
    )
    scene_fixture_manifest = _scene_fixture_manifest_for_cases(case_manifest)
    manifest = {
        "schema_version": EVALUATION_MANIFEST_SCHEMA_VERSION,
        "filters": {
            "names": list(names or ()),
            "tags": list(tags or ()),
            "kinds": list(kinds or ()),
            "question_types": list(question_types or ()),
        },
        "scene_fixtures": scene_fixture_manifest,
        "evaluation_cases": list(case_manifest),
        "coverage": _evaluation_bundle_coverage(case_manifest, scene_fixture_manifest),
    }
    manifest["digest"] = evaluation_manifest_digest(manifest)
    return manifest


def evaluation_manifest_digest(manifest: Mapping[str, Any]) -> str:
    return _manifest_digest(manifest)


def evaluation_manifest_json(manifest: Mapping[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def save_evaluation_manifest(
    path: str | Path,
    *,
    names: Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
    kinds: Sequence[EvaluationKind] | None = None,
    question_types: Sequence[str] | None = None,
) -> Path:
    target = Path(path)
    manifest = evaluation_manifest(
        names=names,
        tags=tags,
        kinds=kinds,
        question_types=question_types,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(evaluation_manifest_json(manifest), encoding="utf-8")
    return target


def load_evaluation_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SpatialQAError("Evaluation manifest JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_evaluation_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    schema_version = manifest.get("schema_version")
    _append_validation_check(
        checks,
        name="schema_version",
        passed=schema_version == EVALUATION_MANIFEST_SCHEMA_VERSION,
        expected=EVALUATION_MANIFEST_SCHEMA_VERSION,
        actual=schema_version,
    )

    expected_digest = evaluation_manifest_digest(manifest)
    manifest_digest = _string_or_none(manifest.get("digest"))
    _append_validation_check(
        checks,
        name="manifest_digest",
        passed=manifest_digest == expected_digest,
        expected=expected_digest,
        actual=manifest_digest,
    )

    expected_scene_fixtures = _scene_fixture_names_from_case_manifest(manifest)
    actual_scene_fixtures = _scene_fixture_names_from_bundle(manifest)
    _append_validation_check(
        checks,
        name="scene_fixture_manifest_covers_cases",
        passed=set(expected_scene_fixtures).issubset(set(actual_scene_fixtures)),
        expected=expected_scene_fixtures,
        actual=actual_scene_fixtures,
    )

    expected_coverage = _evaluation_bundle_coverage(
        _case_manifest_from_bundle(manifest),
        _scene_fixture_manifest_from_bundle(manifest),
    )
    actual_coverage = manifest.get("coverage")
    coverage_differences = _nested_differences(expected_coverage, actual_coverage)
    _append_validation_check(
        checks,
        name="coverage_matches_manifest",
        passed=actual_coverage == expected_coverage,
        expected=expected_coverage,
        actual=actual_coverage,
    )
    if coverage_differences:
        checks[-1]["differences"] = coverage_differences

    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": _string_or_none(schema_version),
        "digest": manifest_digest,
        "checks": checks,
    }


def compare_evaluation_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    filters = _filters_from_bundle(manifest)
    current_manifest = evaluation_manifest(
        names=_optional_filter_values(filters, "names"),
        tags=_optional_filter_values(filters, "tags"),
        kinds=cast(Sequence[EvaluationKind] | None, _optional_filter_values(filters, "kinds")),
        question_types=_optional_filter_values(filters, "question_types"),
    )
    saved_digest = _string_or_none(manifest.get("digest"))
    current_digest = _string_or_none(current_manifest.get("digest"))
    checks: list[dict[str, Any]] = []
    validation = validate_evaluation_manifest(manifest)
    _append_validation_check(
        checks,
        name="saved_manifest_valid",
        passed=validation["valid"] is True,
    )
    _append_validation_check(
        checks,
        name="manifest_digest_matches_current",
        passed=saved_digest == current_digest,
        expected=saved_digest,
        actual=current_digest,
    )
    coverage_differences = _nested_differences(
        manifest.get("coverage"),
        current_manifest["coverage"],
    )
    _append_validation_check(
        checks,
        name="coverage_matches_current",
        passed=manifest.get("coverage") == current_manifest["coverage"],
        expected=manifest.get("coverage"),
        actual=current_manifest["coverage"],
    )
    if coverage_differences:
        checks[-1]["differences"] = coverage_differences
    case_manifest_differences = _manifest_entry_differences(
        _case_manifest_from_bundle(manifest),
        _case_manifest_from_bundle(current_manifest),
    )
    _append_validation_check(
        checks,
        name="case_manifest_matches_current",
        passed=case_manifest_differences == [],
        expected=_case_names_from_bundle(manifest),
        actual=_case_names_from_bundle(current_manifest),
    )
    if case_manifest_differences:
        checks[-1]["differences"] = case_manifest_differences
    scene_fixture_manifest_differences = _manifest_entry_differences(
        _scene_fixture_manifest_from_bundle(manifest),
        _scene_fixture_manifest_from_bundle(current_manifest),
    )
    _append_validation_check(
        checks,
        name="scene_fixture_manifest_matches_current",
        passed=scene_fixture_manifest_differences == [],
        expected=_scene_fixture_names_from_bundle(manifest),
        actual=_scene_fixture_names_from_bundle(current_manifest),
    )
    if scene_fixture_manifest_differences:
        checks[-1]["differences"] = scene_fixture_manifest_differences
    return {
        "matches": all(check["passed"] is True for check in checks),
        "filters": {
            "names": list(_optional_filter_values(filters, "names") or ()),
            "tags": list(_optional_filter_values(filters, "tags") or ()),
            "kinds": list(_optional_filter_values(filters, "kinds") or ()),
            "question_types": list(_optional_filter_values(filters, "question_types") or ()),
        },
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "checks": checks,
    }


def evaluation_bundle_json(bundle: Mapping[str, Any]) -> str:
    return json.dumps(bundle, indent=2, sort_keys=True) + "\n"


def evaluation_bundle_digest(bundle: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in bundle.items() if key != "bundle_digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def save_evaluation_bundle(
    path: str | Path,
    *,
    names: Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
    kinds: Sequence[EvaluationKind] | None = None,
    question_types: Sequence[str] | None = None,
    scene_loaders: Mapping[str, SceneLoader] | None = None,
) -> Path:
    target = Path(path)
    bundle = evaluation_bundle(
        names=names,
        tags=tags,
        kinds=kinds,
        question_types=question_types,
        scene_loaders=scene_loaders,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(evaluation_bundle_json(bundle), encoding="utf-8")
    return target


def load_evaluation_bundle(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SpatialQAError("Evaluation bundle JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_evaluation_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    schema_version = bundle.get("schema_version")
    _append_validation_check(
        checks,
        name="schema_version",
        passed=schema_version == EVALUATION_BUNDLE_SCHEMA_VERSION,
        expected=EVALUATION_BUNDLE_SCHEMA_VERSION,
        actual=schema_version,
    )

    suite = bundle.get("suite")
    suite_digest: str | None = None
    if isinstance(suite, Mapping):
        expected_digest = _suite_digest(suite)
        suite_digest = _string_or_none(suite.get("digest"))
        _append_validation_check(
            checks,
            name="suite_digest",
            passed=suite_digest == expected_digest,
            expected=expected_digest,
            actual=suite_digest,
        )
        expected_report = evaluation_report(suite)
        actual_report = bundle.get("report")
        report_differences = _evaluation_report_differences(
            expected_report,
            actual_report,
        )
        _append_validation_check(
            checks,
            name="report_matches_suite",
            passed=actual_report == expected_report,
        )
        if report_differences:
            checks[-1]["differences"] = report_differences
        selected_cases = _selected_cases_from_suite(suite)
        expected_case_manifest = _case_manifest_from_suite_results(suite)
    else:
        _append_validation_check(
            checks,
            name="suite_digest",
            passed=False,
            expected=None,
            actual=None,
        )
        _append_validation_check(checks, name="report_matches_suite", passed=False)
        selected_cases = []
        expected_case_manifest = []

    bundle_digest = _string_or_none(bundle.get("bundle_digest"))
    expected_bundle_digest = evaluation_bundle_digest(bundle)
    _append_validation_check(
        checks,
        name="bundle_digest",
        passed=bundle_digest == expected_bundle_digest,
        expected=expected_bundle_digest,
        actual=bundle_digest,
    )

    case_names = _case_names_from_bundle(bundle)
    case_manifest_differences = _manifest_entry_differences(
        expected_case_manifest,
        _case_manifest_suite_projection(_case_manifest_from_bundle(bundle)),
    )
    _append_validation_check(
        checks,
        name="case_manifest_matches_suite",
        passed=case_names == selected_cases and case_manifest_differences == [],
        expected=selected_cases,
        actual=case_names,
    )
    if case_manifest_differences:
        checks[-1]["differences"] = case_manifest_differences

    expected_scene_fixtures = _scene_fixture_names_from_case_manifest(bundle)
    actual_scene_fixtures = _scene_fixture_names_from_bundle(bundle)
    scene_fixture_manifest_differences = _manifest_entry_differences(
        _scene_fixture_manifest_from_case_manifest(bundle),
        _scene_fixture_manifest_case_projection(_scene_fixture_manifest_from_bundle(bundle)),
    )
    _append_validation_check(
        checks,
        name="scene_fixture_manifest_covers_cases",
        passed=(
            set(expected_scene_fixtures).issubset(set(actual_scene_fixtures))
            and scene_fixture_manifest_differences == []
        ),
        expected=expected_scene_fixtures,
        actual=actual_scene_fixtures,
    )
    if scene_fixture_manifest_differences:
        checks[-1]["differences"] = scene_fixture_manifest_differences
    expected_coverage = _evaluation_bundle_coverage(
        _case_manifest_from_bundle(bundle),
        _scene_fixture_manifest_from_bundle(bundle),
    )
    actual_coverage = bundle.get("coverage")
    coverage_differences = _nested_differences(expected_coverage, actual_coverage)
    _append_validation_check(
        checks,
        name="coverage_matches_manifest",
        passed=actual_coverage == expected_coverage,
        expected=expected_coverage,
        actual=actual_coverage,
    )
    if coverage_differences:
        checks[-1]["differences"] = coverage_differences

    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": _string_or_none(schema_version),
        "digest": suite_digest,
        "bundle_digest": bundle_digest,
        "checks": checks,
    }


def compare_evaluation_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    filters = _filters_from_bundle(bundle)
    current_bundle = evaluation_bundle(
        names=_optional_filter_values(filters, "names"),
        tags=_optional_filter_values(filters, "tags"),
        kinds=cast(Sequence[EvaluationKind] | None, _optional_filter_values(filters, "kinds")),
        question_types=_optional_filter_values(filters, "question_types"),
    )
    saved_digest = _bundle_suite_digest(bundle)
    current_digest = _bundle_suite_digest(current_bundle)
    saved_bundle_digest = _string_or_none(bundle.get("bundle_digest"))
    current_bundle_digest = _string_or_none(current_bundle.get("bundle_digest"))
    checks: list[dict[str, Any]] = []
    validation = validate_evaluation_bundle(bundle)
    _append_validation_check(
        checks,
        name="saved_bundle_valid",
        passed=validation["valid"] is True,
    )
    _append_validation_check(
        checks,
        name="suite_digest_matches_current",
        passed=saved_digest == current_digest,
        expected=saved_digest,
        actual=current_digest,
    )
    _append_validation_check(
        checks,
        name="bundle_digest_matches_current",
        passed=saved_bundle_digest == current_bundle_digest,
        expected=saved_bundle_digest,
        actual=current_bundle_digest,
    )
    report_differences = _evaluation_report_differences(
        bundle.get("report"),
        current_bundle["report"],
    )
    _append_validation_check(
        checks,
        name="report_matches_current",
        passed=bundle.get("report") == current_bundle["report"],
        expected=bundle.get("report"),
        actual=current_bundle["report"],
    )
    if report_differences:
        checks[-1]["differences"] = report_differences
    coverage_differences = _nested_differences(
        bundle.get("coverage"),
        current_bundle["coverage"],
    )
    _append_validation_check(
        checks,
        name="coverage_matches_current",
        passed=bundle.get("coverage") == current_bundle["coverage"],
        expected=bundle.get("coverage"),
        actual=current_bundle["coverage"],
    )
    if coverage_differences:
        checks[-1]["differences"] = coverage_differences
    case_manifest_differences = _manifest_entry_differences(
        _case_manifest_from_bundle(bundle),
        _case_manifest_from_bundle(current_bundle),
    )
    _append_validation_check(
        checks,
        name="case_manifest_matches_current",
        passed=case_manifest_differences == [],
        expected=_case_names_from_bundle(bundle),
        actual=_case_names_from_bundle(current_bundle),
    )
    if case_manifest_differences:
        checks[-1]["differences"] = case_manifest_differences
    scene_fixture_manifest_differences = _manifest_entry_differences(
        _scene_fixture_manifest_from_bundle(bundle),
        _scene_fixture_manifest_from_bundle(current_bundle),
    )
    _append_validation_check(
        checks,
        name="scene_fixture_manifest_matches_current",
        passed=scene_fixture_manifest_differences == [],
        expected=_scene_fixture_names_from_bundle(bundle),
        actual=_scene_fixture_names_from_bundle(current_bundle),
    )
    if scene_fixture_manifest_differences:
        checks[-1]["differences"] = scene_fixture_manifest_differences
    return {
        "matches": all(check["passed"] is True for check in checks),
        "filters": {
            "names": list(_optional_filter_values(filters, "names") or ()),
            "tags": list(_optional_filter_values(filters, "tags") or ()),
            "kinds": list(_optional_filter_values(filters, "kinds") or ()),
            "question_types": list(_optional_filter_values(filters, "question_types") or ()),
        },
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "saved_bundle_digest": saved_bundle_digest,
        "current_bundle_digest": current_bundle_digest,
        "checks": checks,
    }


def _filters_from_bundle(bundle: Mapping[str, Any]) -> Mapping[str, Any]:
    filters = bundle.get("filters", {})
    if not isinstance(filters, Mapping):
        return {}
    return filters


def _optional_filter_values(filters: Mapping[str, Any], key: str) -> tuple[str, ...] | None:
    values = filters.get(key, ())
    if not isinstance(values, Sequence) or isinstance(values, str):
        return None
    selected = tuple(str(value) for value in values)
    return selected if selected else None


def _bundle_suite_digest(bundle: Mapping[str, Any]) -> str | None:
    suite = bundle.get("suite")
    if not isinstance(suite, Mapping):
        return None
    return _string_or_none(suite.get("digest"))


def _append_validation_check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    passed: bool,
    expected: Any | None = None,
    actual: Any | None = None,
) -> None:
    check: dict[str, Any] = {"name": name, "passed": passed}
    if expected is not None or actual is not None:
        check["expected"] = expected
        check["actual"] = actual
    checks.append(check)


def _selected_cases_from_suite(suite: Mapping[str, Any]) -> list[str]:
    summary = suite.get("summary")
    if not isinstance(summary, Mapping):
        return []
    selected_cases = summary.get("selected_cases", ())
    if not isinstance(selected_cases, Sequence) or isinstance(selected_cases, str):
        return []
    return [str(name) for name in selected_cases]


def _case_manifest_from_suite_results(suite: Mapping[str, Any]) -> list[dict[str, Any]]:
    results = suite.get("results", ())
    if not isinstance(results, Sequence) or isinstance(results, str):
        return []
    entries: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, Mapping) or "case" not in result:
            continue
        entries.append(
            {
                "name": str(result["case"]),
                "scene_fixture": result.get("scene_fixture"),
                "kind": result.get("kind"),
                "question_type": result.get("question_type"),
                "tags": _string_list(result.get("tags", ())),
            }
        )
    return entries


def _case_manifest_suite_projection(
    case_manifest: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for case_metadata in case_manifest:
        if "name" not in case_metadata:
            continue
        entries.append(
            {
                "name": str(case_metadata["name"]),
                "scene_fixture": case_metadata.get("scene_fixture"),
                "kind": case_metadata.get("kind"),
                "question_type": case_metadata.get("question_type"),
                "tags": _string_list(case_metadata.get("tags", ())),
            }
        )
    return entries


def _string_list(values: object) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, str):
        return []
    return [str(value) for value in values]


def _case_names_from_bundle(bundle: Mapping[str, Any]) -> list[str]:
    return [
        str(case_metadata["name"])
        for case_metadata in _case_manifest_from_bundle(bundle)
        if "name" in case_metadata
    ]


def _scene_fixture_names_from_case_manifest(bundle: Mapping[str, Any]) -> list[str]:
    names: set[str] = set()
    for case_metadata in _case_manifest_from_bundle(bundle):
        scene_fixture = case_metadata.get("scene_fixture")
        if isinstance(scene_fixture, str):
            names.add(scene_fixture)
        baseline_scene_fixture = case_metadata.get("baseline_scene_fixture")
        if isinstance(baseline_scene_fixture, str):
            names.add(baseline_scene_fixture)
    return sorted(names)


def _scene_fixture_manifest_from_case_manifest(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    fixtures: dict[str, dict[str, Any]] = {}
    for case_metadata in _case_manifest_from_bundle(bundle):
        scene_fixture = case_metadata.get("scene_fixture")
        if isinstance(scene_fixture, str):
            fixtures[scene_fixture] = {
                "name": scene_fixture,
                "description": case_metadata.get("scene_description"),
                "tags": _string_list(case_metadata.get("scene_tags", ())),
            }
        baseline_scene_fixture = case_metadata.get("baseline_scene_fixture")
        if isinstance(baseline_scene_fixture, str):
            fixtures[baseline_scene_fixture] = {
                "name": baseline_scene_fixture,
                "description": case_metadata.get("baseline_scene_description"),
                "tags": _string_list(case_metadata.get("baseline_scene_tags", ())),
            }
    return [
        fixtures[name]
        for name in sorted(fixtures)
    ]


def _scene_fixture_manifest_case_projection(
    scene_fixture_manifest: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for scene_fixture in scene_fixture_manifest:
        if "name" not in scene_fixture:
            continue
        entries.append(
            {
                "name": str(scene_fixture["name"]),
                "description": scene_fixture.get("description"),
                "tags": _string_list(scene_fixture.get("tags", ())),
            }
        )
    return entries


def _scene_fixture_names_from_bundle(bundle: Mapping[str, Any]) -> list[str]:
    return sorted(
        str(fixture["name"])
        for fixture in _scene_fixture_manifest_from_bundle(bundle)
        if "name" in fixture
    )


def _case_manifest_from_bundle(bundle: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    case_manifest = bundle.get("evaluation_cases", ())
    if not isinstance(case_manifest, Sequence) or isinstance(case_manifest, str):
        return []
    return [
        cast(Mapping[str, Any], case_metadata)
        for case_metadata in case_manifest
        if isinstance(case_metadata, Mapping)
    ]


def _scene_fixture_manifest_from_bundle(bundle: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    scene_fixtures = bundle.get("scene_fixtures", ())
    if not isinstance(scene_fixtures, Sequence) or isinstance(scene_fixtures, str):
        return []
    return [
        cast(Mapping[str, Any], scene_fixture)
        for scene_fixture in scene_fixtures
        if isinstance(scene_fixture, Mapping)
    ]


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _scene_fixture_manifest_for_cases(
    case_manifest: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fixture_names: set[str] = set()
    for case_metadata in case_manifest:
        fixture_names.add(str(case_metadata["scene_fixture"]))
        baseline_fixture = case_metadata.get("baseline_scene_fixture")
        if isinstance(baseline_fixture, str):
            fixture_names.add(baseline_fixture)

    fixture_metadata_by_name = {
        str(fixture["name"]): fixture
        for fixture in list_scene_fixture_metadata()
    }
    return [
        dict(fixture_metadata_by_name[name])
        for name in sorted(fixture_names)
        if name in fixture_metadata_by_name
    ]


def _evaluation_bundle_coverage(
    case_manifest: Sequence[Mapping[str, Any]],
    scene_fixture_manifest: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_kind: dict[str, int] = {}
    by_question_type: dict[str, int] = {}
    by_tag: dict[str, int] = {}
    by_scene_fixture: dict[str, int] = {}
    by_scene_tag: dict[str, int] = {}

    for case_metadata in case_manifest:
        _increment_count(by_kind, str(case_metadata["kind"]))
        question_type = case_metadata.get("question_type")
        if isinstance(question_type, str):
            _increment_count(by_question_type, question_type)
        _increment_counts(by_tag, case_metadata.get("tags", ()))
        _increment_count(by_scene_fixture, str(case_metadata["scene_fixture"]))

    for scene_fixture in scene_fixture_manifest:
        _increment_counts(by_scene_tag, scene_fixture.get("tags", ()))

    return {
        "case_count": len(case_manifest),
        "scene_fixture_count": len(scene_fixture_manifest),
        "by_kind": _sorted_counts(by_kind),
        "by_question_type": _sorted_counts(by_question_type),
        "by_tag": _sorted_counts(by_tag),
        "by_scene_fixture": _sorted_counts(by_scene_fixture),
        "by_scene_tag": _sorted_counts(by_scene_tag),
    }


def _increment_counts(counts: dict[str, int], values: object) -> None:
    if not isinstance(values, Sequence) or isinstance(values, str):
        return
    for value in values:
        _increment_count(counts, str(value))


def _increment_count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _sorted_counts(counts: Mapping[str, int]) -> dict[str, int]:
    return {key: counts[key] for key in sorted(counts)}


def _failure_category_for_reason(reason: str) -> str:
    if reason == "missing_actual_key":
        return "missing_output"
    if reason == "sequence_length_mismatch":
        return "cardinality_mismatch"
    if reason == "type_mismatch":
        return "schema_mismatch"
    if reason == "value_mismatch":
        return "value_mismatch"
    return "other_mismatch"


def _mismatch_category(mismatch: Mapping[str, Any]) -> str:
    category = mismatch.get("category")
    if isinstance(category, str):
        return category
    return _failure_category_for_reason(str(mismatch.get("reason")))


def evaluation_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def evaluation_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def save_evaluation_report(path: str | Path, suite: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(evaluation_report_json(evaluation_report(suite)), encoding="utf-8")
    return target


def load_evaluation_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SpatialQAError("Evaluation report JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_evaluation_report(report: Mapping[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    suite_digest = _string_or_none(report.get("digest"))
    _append_validation_check(
        checks,
        name="suite_digest_format",
        passed=_is_sha256_hexdigest(suite_digest),
        expected="64 lowercase sha256 hex characters",
        actual=suite_digest,
    )
    report_digest = _string_or_none(report.get("report_digest"))
    expected_report_digest = evaluation_report_digest(report)
    _append_validation_check(
        checks,
        name="report_digest",
        passed=report_digest == expected_report_digest,
        expected=expected_report_digest,
        actual=report_digest,
    )
    return {
        "valid": all(check["passed"] is True for check in checks),
        "digest": suite_digest,
        "report_digest": report_digest,
        "checks": checks,
    }


def compare_evaluation_report(report: Mapping[str, Any]) -> dict[str, Any]:
    selected_cases = _selected_cases_from_suite(report)
    current_report = evaluation_report(run_evaluation_suite(names=selected_cases))
    saved_digest = _string_or_none(report.get("digest"))
    current_digest = _string_or_none(current_report.get("digest"))
    checks: list[dict[str, Any]] = []
    validation = validate_evaluation_report(report)
    _append_validation_check(
        checks,
        name="saved_report_valid",
        passed=validation["valid"] is True,
    )
    _append_validation_check(
        checks,
        name="report_digest_matches_current",
        passed=saved_digest == current_digest,
        expected=saved_digest,
        actual=current_digest,
    )
    summary_differences = _nested_differences(
        report.get("summary"),
        current_report["summary"],
    )
    _append_validation_check(
        checks,
        name="summary_matches_current",
        passed=report.get("summary") == current_report["summary"],
        expected=report.get("summary"),
        actual=current_report["summary"],
    )
    if summary_differences:
        checks[-1]["differences"] = summary_differences
    metric_differences = _nested_differences(report.get("metrics"), current_report["metrics"])
    _append_validation_check(
        checks,
        name="metrics_match_current",
        passed=report.get("metrics") == current_report["metrics"],
        expected=report.get("metrics"),
        actual=current_report["metrics"],
    )
    if metric_differences:
        checks[-1]["differences"] = metric_differences
    evidence_metric_differences = _nested_differences(
        report.get("evidence_metrics"),
        current_report["evidence_metrics"],
    )
    _append_validation_check(
        checks,
        name="evidence_metrics_match_current",
        passed=report.get("evidence_metrics") == current_report["evidence_metrics"],
        expected=report.get("evidence_metrics"),
        actual=current_report["evidence_metrics"],
    )
    if evidence_metric_differences:
        checks[-1]["differences"] = evidence_metric_differences
    case_digest_differences = _report_summary_entry_differences(
        report.get("case_digests"),
        current_report["case_digests"],
        key_name="case",
    )
    _append_validation_check(
        checks,
        name="case_digests_match_current",
        passed=report.get("case_digests") == current_report["case_digests"],
        expected=report.get("case_digests"),
        actual=current_report["case_digests"],
    )
    if case_digest_differences:
        checks[-1]["differences"] = case_digest_differences
    failed_case_differences = _report_summary_entry_differences(
        report.get("failed_cases"),
        current_report["failed_cases"],
        key_name="case",
    )
    _append_validation_check(
        checks,
        name="failed_cases_match_current",
        passed=report.get("failed_cases") == current_report["failed_cases"],
        expected=report.get("failed_cases"),
        actual=current_report["failed_cases"],
    )
    if failed_case_differences:
        checks[-1]["differences"] = failed_case_differences
    failure_reason_differences = _report_summary_entry_differences(
        report.get("failure_reasons"),
        current_report["failure_reasons"],
        key_name="reason",
    )
    _append_validation_check(
        checks,
        name="failure_reasons_match_current",
        passed=report.get("failure_reasons") == current_report["failure_reasons"],
        expected=report.get("failure_reasons"),
        actual=current_report["failure_reasons"],
    )
    if failure_reason_differences:
        checks[-1]["differences"] = failure_reason_differences
    runtime_error_category_differences = _runtime_error_category_differences(
        report.get("runtime_error_categories"),
        current_report["runtime_error_categories"],
    )
    _append_validation_check(
        checks,
        name="runtime_error_categories_match_current",
        passed=report.get("runtime_error_categories")
        == current_report["runtime_error_categories"],
        expected=report.get("runtime_error_categories"),
        actual=current_report["runtime_error_categories"],
    )
    if runtime_error_category_differences:
        checks[-1]["differences"] = runtime_error_category_differences
    failure_category_differences = _report_summary_entry_differences(
        report.get("failure_categories"),
        current_report["failure_categories"],
        key_name="category",
    )
    _append_validation_check(
        checks,
        name="failure_categories_match_current",
        passed=report.get("failure_categories") == current_report["failure_categories"],
        expected=report.get("failure_categories"),
        actual=current_report["failure_categories"],
    )
    if failure_category_differences:
        checks[-1]["differences"] = failure_category_differences
    failure_path_differences = _report_summary_entry_differences(
        report.get("failure_paths"),
        current_report["failure_paths"],
        key_name="path",
    )
    _append_validation_check(
        checks,
        name="failure_paths_match_current",
        passed=report.get("failure_paths") == current_report["failure_paths"],
        expected=report.get("failure_paths"),
        actual=current_report["failure_paths"],
    )
    if failure_path_differences:
        checks[-1]["differences"] = failure_path_differences
    breakdown_differences = _nested_differences(
        report.get("breakdown"),
        current_report["breakdown"],
    )
    _append_validation_check(
        checks,
        name="breakdown_matches_current",
        passed=report.get("breakdown") == current_report["breakdown"],
        expected=report.get("breakdown"),
        actual=current_report["breakdown"],
    )
    if breakdown_differences:
        checks[-1]["differences"] = breakdown_differences
    return {
        "matches": all(check["passed"] is True for check in checks),
        "filters": {"names": selected_cases},
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "checks": checks,
    }


def _nested_differences(
    expected: object,
    actual: object,
    path: str = "",
) -> list[dict[str, Any]]:
    if expected == actual:
        return []
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        differences: list[dict[str, Any]] = []
        keys = sorted(set(expected) | set(actual), key=str)
        for key in keys:
            child_path = str(key) if path == "" else f"{path}.{key}"
            if key not in expected:
                differences.append(
                    {"path": child_path, "expected": None, "actual": actual[key]},
                )
                continue
            if key not in actual:
                differences.append(
                    {"path": child_path, "expected": expected[key], "actual": None},
                )
                continue
            differences.extend(_nested_differences(expected[key], actual[key], child_path))
        return differences
    return [{"path": path if path else "$", "expected": expected, "actual": actual}]


def _evaluation_report_differences(
    expected: object,
    actual: object,
) -> list[dict[str, Any]]:
    if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
        return _nested_differences(expected, actual)

    fields = (
        "digest",
        "report_digest",
        "summary",
        "metrics",
        "evidence_metrics",
        "case_digests",
        "failed_cases",
        "runtime_error_categories",
        "failure_reasons",
        "failure_categories",
        "failure_paths",
        "breakdown",
    )
    differences: list[dict[str, Any]] = []
    for field_name in fields:
        differences.extend(
            _prefix_differences(
                field_name,
                _evaluation_report_field_differences(
                    field_name,
                    expected.get(field_name),
                    actual.get(field_name),
                ),
            )
        )
    for field_name in sorted((set(expected) | set(actual)) - set(fields), key=str):
        differences.extend(
            _prefix_differences(
                str(field_name),
                _nested_differences(expected.get(field_name), actual.get(field_name)),
            )
        )
    return differences


def _evaluation_report_field_differences(
    field: str,
    expected: object,
    actual: object,
) -> list[dict[str, Any]]:
    if field == "failed_cases":
        return _report_summary_entry_differences(expected, actual, key_name="case")
    if field == "case_digests":
        return _report_summary_entry_differences(expected, actual, key_name="case")
    if field == "runtime_error_categories":
        return _runtime_error_category_differences(expected, actual)
    if field == "failure_reasons":
        return _report_summary_entry_differences(expected, actual, key_name="reason")
    if field == "failure_categories":
        return _report_summary_entry_differences(expected, actual, key_name="category")
    if field == "failure_paths":
        return _report_summary_entry_differences(expected, actual, key_name="path")
    return _nested_differences(expected, actual)


def _manifest_entry_differences(
    expected: object,
    actual: object,
) -> list[dict[str, Any]]:
    return _report_summary_entry_differences(expected, actual, key_name="name")


def _prefix_differences(
    prefix: str,
    differences: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    prefixed: list[dict[str, Any]] = []
    for difference in differences:
        path = difference.get("path")
        path_text = str(path) if isinstance(path, str) else "$"
        prefixed.append(
            {
                **dict(difference),
                "path": prefix if path_text == "$" else f"{prefix}.{path_text}",
            }
        )
    return prefixed


def _runtime_error_category_differences(
    expected: object,
    actual: object,
) -> list[dict[str, Any]]:
    return _report_summary_entry_differences(expected, actual, key_name="category")


def _report_summary_entry_differences(
    expected: object,
    actual: object,
    *,
    key_name: str,
) -> list[dict[str, Any]]:
    return _nested_differences(
        _report_summary_entries_by_key(expected, key_name=key_name),
        _report_summary_entries_by_key(actual, key_name=key_name),
    )


def _report_summary_entries_by_key(value: object, *, key_name: str) -> object:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return value
    entries: dict[str, Any] = {}
    for item in value:
        if not isinstance(item, Mapping):
            return value
        entry_key = item.get(key_name)
        if not isinstance(entry_key, str):
            return value
        entries[entry_key] = {
            str(item_key): item_value
            for item_key, item_value in item.items()
            if item_key != key_name
        }
    return entries


def _runtime_error_category_summary(
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    category_index: dict[str, dict[str, Any]] = {}
    for result in results:
        actual = result.get("actual")
        if not isinstance(actual, Mapping):
            continue
        category = actual.get("error_category")
        if not isinstance(category, str):
            continue
        entry = category_index.setdefault(category, {"count": 0, "cases": set()})
        entry["count"] = int(entry["count"]) + 1
        cast(set[str], entry["cases"]).add(str(result["case"]))
    return [
        {
            "category": category,
            "count": int(entry["count"]),
            "cases": sorted(cast(set[str], entry["cases"])),
        }
        for category, entry in sorted(category_index.items())
    ]


def _suite_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    suite = {
        "summary": _result_group_summary(results),
        "breakdown": {
            "by_kind": _result_breakdown_by_kind(results),
            "by_question_type": _result_breakdown_by_question_type(results),
            "by_scene_fixture": _result_breakdown_by_scene_fixture(results),
            "by_tag": _result_breakdown_by_tag(results),
        },
        "runtime_error_categories": _runtime_error_category_summary(results),
        "results": results,
    }
    suite["digest"] = _suite_digest(suite)
    return suite


def _suite_digest(suite: Mapping[str, Any]) -> str:
    payload = {
        "summary": suite["summary"],
        "breakdown": suite["breakdown"],
        "runtime_error_categories": suite.get("runtime_error_categories", []),
        "results": suite["results"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    payload = {
        "schema_version": manifest.get("schema_version"),
        "filters": manifest.get("filters"),
        "scene_fixtures": manifest.get("scene_fixtures"),
        "evaluation_cases": manifest.get("evaluation_cases"),
        "coverage": manifest.get("coverage"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _result_group_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    passed = sum(1 for result in results if result["passed"] is True)
    selected_cases = [str(result["case"]) for result in results]
    failed_cases = [str(result["case"]) for result in results if result["passed"] is False]
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "failed_cases": failed_cases,
        "selected_cases": selected_cases,
    }


def _result_breakdown_by_kind(results: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    by_kind: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        by_kind.setdefault(str(result["kind"]), []).append(result)
    return {
        kind: _result_group_summary(by_kind[kind])
        for kind in sorted(by_kind)
    }


def _result_breakdown_by_question_type(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_question_type: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        question_type = result.get("question_type")
        if isinstance(question_type, str):
            by_question_type.setdefault(question_type, []).append(result)
    return {
        question_type: _result_group_summary(by_question_type[question_type])
        for question_type in sorted(by_question_type)
    }


def _result_breakdown_by_scene_fixture(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_scene_fixture: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        by_scene_fixture.setdefault(str(result["scene_fixture"]), []).append(result)
    return {
        scene_fixture: _result_group_summary(by_scene_fixture[scene_fixture])
        for scene_fixture in sorted(by_scene_fixture)
    }


def _result_breakdown_by_tag(results: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    by_tag: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        for tag in result["tags"]:
            by_tag.setdefault(str(tag), []).append(result)
    return {
        tag: _result_group_summary(by_tag[tag])
        for tag in sorted(by_tag)
    }


def _case_question_type(case: EvaluationCase) -> str | None:
    question_type = case.question.get("type")
    return question_type if isinstance(question_type, str) else None


def _run_case(
    case: EvaluationCase,
    scene_loaders: Mapping[str, SceneLoader] | None,
) -> dict[str, Any]:
    if case.kind == "qa":
        graph = _load_scene(case.scene_fixture, scene_loaders)
        response = SpatialQAEngine(GraphTool(graph)).answer(
            _qa_question_for_case(case, scene_loaders)
        )
        return _qa_response_to_dict(response)
    if case.kind == "vla_pick":
        graph = _load_scene(case.scene_fixture, scene_loaders)
        target_object, target_label = _required_pick_target(case)
        result = VLAAnchorPlanner(GraphTool(graph)).plan_pick(
            target_object=target_object,
            label=target_label,
        )
        return _planner_result_to_dict(result)
    if case.kind == "vla_place_relative":
        graph = _load_scene(case.scene_fixture, scene_loaders)
        target_object, target_label = _required_pick_target(case)
        reference_object, reference_label = _required_place_reference(case)
        result = VLAAnchorPlanner(GraphTool(graph)).plan_place_relative(
            target_object,
            reference_object,
            _required_relation(case),
            target_label=target_label,
            reference_label=reference_label,
        )
        return _planner_result_to_dict(result)
    if case.kind == "vla_stale_pick":
        baseline = _load_scene(_required_baseline(case), scene_loaders)
        target = _required_target(case)
        command = VLAAnchorPlanner(GraphTool(baseline)).plan_pick(target_object=target).command
        if command is None:
            raise SpatialQAError(f"Evaluation case could not create baseline command: {case.name}")
        graph = _load_scene(case.scene_fixture, scene_loaders)
        result = VLAAnchorPlanner(GraphTool(graph)).validate(command)
        return _planner_result_to_dict(result)
    if case.kind == "vla_stale_place_relative":
        baseline = _load_scene(_required_baseline(case), scene_loaders)
        target_object, target_label = _required_pick_target(case)
        reference_object, reference_label = _required_place_reference(case)
        command = VLAAnchorPlanner(GraphTool(baseline)).plan_place_relative(
            target_object,
            reference_object,
            _required_relation(case),
            target_label=target_label,
            reference_label=reference_label,
        ).command
        if command is None:
            raise SpatialQAError(f"Evaluation case could not create baseline command: {case.name}")
        graph = _load_scene(case.scene_fixture, scene_loaders)
        result = VLAAnchorPlanner(GraphTool(graph)).validate(command)
        return _planner_result_to_dict(result)
    raise SpatialQAError(f"Unsupported evaluation kind: {case.kind}")


def _qa_question_for_case(
    case: EvaluationCase,
    scene_loaders: Mapping[str, SceneLoader] | None,
) -> dict[str, Any]:
    question = dict(case.question)
    if question.get("type") == "next_action_validity" and "action" not in question:
        baseline = _load_scene(_required_baseline(case), scene_loaders)
        command = VLAAnchorPlanner(GraphTool(baseline)).plan_pick(
            target_object=_required_target(case)
        ).command
        if command is None:
            raise SpatialQAError(f"Evaluation case could not create baseline action: {case.name}")
        question["action"] = command
    return question


def _qa_response_to_dict(response: QAResponse) -> dict[str, Any]:
    payload = {
        "answer": response.answer,
        "evidence_nodes": response.evidence_nodes,
        "evidence_edges": response.evidence_edges,
        "confidence": response.confidence,
        "needs_reobserve": response.needs_reobserve,
        "error": response.error,
    }
    category = _runtime_error_category(response.error)
    if category is not None:
        payload["error_category"] = category
    return payload


def _planner_result_to_dict(result: PlannerResult) -> dict[str, Any]:
    payload = {
        "status": result.status,
        "command": _skill_command_to_dict(result.command) if result.command else None,
        "error": result.error,
        "needs_reobserve": result.needs_reobserve,
        "needs_replan": result.needs_replan,
        "ambiguous_ids": result.ambiguous_ids,
        "details": result.details,
    }
    category = _runtime_error_category(result.error)
    if category is not None:
        payload["error_category"] = category
    return payload


def _runtime_error_category(error: str | None) -> str | None:
    if error is None:
        return None
    if error.startswith("Object not found:"):
        return "missing_object"
    if error.startswith("Object label not found:"):
        return "missing_label"
    if error.startswith("Unsupported place relation:"):
        return "unsupported_relation"
    if error.startswith("Ambiguous label:"):
        return "ambiguous_label"
    if error in {
        "from_step cannot be greater than to_step",
        "since_step cannot be greater than until_step",
    }:
        return "invalid_time_window"
    if error in {
        "target_object or label is required",
        "target_object or target_label is required",
    }:
        return "missing_target"
    if error in {
        "reference_object or reference_label is required",
        "place_relative missing reference_object",
    }:
        return "missing_reference"
    if error in {
        "needs_reobserve",
        "stale_object_state",
        "stale_reference_state",
    }:
        return error
    return "runtime_error"


def _skill_command_to_dict(command: SkillCommand) -> dict[str, Any]:
    return {
        "skill": command.skill,
        "target_object": command.target_object,
        "target_pose": command.target_pose.to_dict(),
        "reference_object": command.reference_object,
        "preconditions": command.preconditions,
        "evidence": command.evidence,
        "parameters": command.parameters,
    }


def _matches_expected(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return False
        return all(key in actual and _matches_expected(actual[key], value) for key, value in expected.items())
    if isinstance(expected, Sequence) and not isinstance(expected, str):
        if not isinstance(actual, Sequence) or isinstance(actual, str):
            return False
        if len(actual) != len(expected):
            return False
        return all(_matches_expected(actual_item, expected_item) for actual_item, expected_item in zip(actual, expected))
    return bool(actual == expected)


def _expected_mismatches(actual: Any, expected: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return [
                {
                    "path": path,
                    "reason": "type_mismatch",
                    "category": _failure_category_for_reason("type_mismatch"),
                    "expected": "mapping",
                    "actual": type(actual).__name__,
                }
            ]
        mismatches: list[dict[str, Any]] = []
        for key in sorted(expected):
            child_path = _join_path(path, str(key))
            if key not in actual:
                mismatches.append(
                    {
                        "path": child_path,
                        "reason": "missing_actual_key",
                        "category": _failure_category_for_reason("missing_actual_key"),
                        "expected": expected[key],
                        "actual": None,
                    }
                )
                continue
            mismatches.extend(_expected_mismatches(actual[key], expected[key], child_path))
        return mismatches
    if isinstance(expected, Sequence) and not isinstance(expected, str):
        if not isinstance(actual, Sequence) or isinstance(actual, str):
            return [
                {
                    "path": path,
                    "reason": "type_mismatch",
                    "category": _failure_category_for_reason("type_mismatch"),
                    "expected": "sequence",
                    "actual": type(actual).__name__,
                }
            ]
        if len(actual) != len(expected):
            return [
                {
                    "path": path,
                    "reason": "sequence_length_mismatch",
                    "category": _failure_category_for_reason("sequence_length_mismatch"),
                    "expected": len(expected),
                    "actual": len(actual),
                }
            ]
        mismatches = []
        for index, expected_item in enumerate(expected):
            mismatches.extend(_expected_mismatches(actual[index], expected_item, f"{path}[{index}]"))
        return mismatches
    if actual != expected:
        return [
            {
                "path": path,
                "reason": "value_mismatch",
                "category": _failure_category_for_reason("value_mismatch"),
                "expected": expected,
                "actual": actual,
            }
        ]
    return []


def _join_path(prefix: str, key: str) -> str:
    return key if not prefix else f"{prefix}.{key}"


def _case_names_matching_filters(
    tags: Sequence[str] | None,
    kinds: Sequence[EvaluationKind] | None,
    question_types: Sequence[str] | None,
) -> tuple[str, ...]:
    return tuple(
        case.name
        for case in _select_cases(
            tuple(_EVALUATION_CASES.values()),
            names=None,
            tags=tags,
            kinds=kinds,
            question_types=question_types,
        )
    )


def _select_cases(
    cases: Sequence[EvaluationCase],
    *,
    names: Sequence[str] | None,
    tags: Sequence[str] | None,
    kinds: Sequence[EvaluationKind] | None,
    question_types: Sequence[str] | None,
) -> list[EvaluationCase]:
    return [
        case
        for case in _cases_in_requested_order(cases, names)
        if _case_matches_filters(case, tags, kinds, question_types)
    ]


def _cases_in_requested_order(
    cases: Sequence[EvaluationCase],
    names: Sequence[str] | None,
) -> list[EvaluationCase]:
    if names is None:
        return sorted(cases, key=lambda item: item.name)

    cases_by_name: dict[str, EvaluationCase] = {}
    for case in cases:
        if case.name in cases_by_name:
            raise SpatialQAError(f"Duplicate evaluation case name: {case.name}")
        cases_by_name[case.name] = case

    selected_cases = []
    for name in names:
        matched_case = cases_by_name.get(name)
        if matched_case is None:
            raise SpatialQAError(f"Unknown evaluation case: {name}")
        selected_cases.append(matched_case)
    return selected_cases


def _case_matches_filters(
    case: EvaluationCase,
    tags: Sequence[str] | None,
    kinds: Sequence[EvaluationKind] | None,
    question_types: Sequence[str] | None,
) -> bool:
    if tags is not None and not set(tags).issubset(set(case.tags)):
        return False
    if kinds is not None and case.kind not in set(kinds):
        return False
    if question_types is not None:
        question_type = case.question.get("type")
        if not isinstance(question_type, str) or question_type not in set(question_types):
            return False
    return True


def _load_scene(
    scene_fixture: str,
    scene_loaders: Mapping[str, SceneLoader] | None,
) -> DynamicSceneGraph:
    if scene_loaders is not None and scene_fixture in scene_loaders:
        return scene_loaders[scene_fixture]()
    return load_scene_fixture(scene_fixture)


def _required_baseline(case: EvaluationCase) -> str:
    if case.baseline_scene_fixture is None:
        raise SpatialQAError(f"Evaluation case missing baseline scene: {case.name}")
    return case.baseline_scene_fixture


def _required_target(case: EvaluationCase) -> str:
    if case.target_object is None:
        raise SpatialQAError(f"Evaluation case missing target object: {case.name}")
    return case.target_object


def _required_pick_target(case: EvaluationCase) -> tuple[str | None, str | None]:
    if case.target_object is None and case.target_label is None:
        raise SpatialQAError(f"Evaluation case missing target object or label: {case.name}")
    return case.target_object, case.target_label


def _required_place_reference(case: EvaluationCase) -> tuple[str | None, str | None]:
    if case.reference_object is None and case.reference_label is None:
        raise SpatialQAError(f"Evaluation case missing reference object or label: {case.name}")
    return case.reference_object, case.reference_label


def _required_relation(case: EvaluationCase) -> str:
    if case.relation is None:
        raise SpatialQAError(f"Evaluation case missing relation: {case.name}")
    return case.relation
