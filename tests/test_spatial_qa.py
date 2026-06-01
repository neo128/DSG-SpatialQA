from dsg_spatialqa_lab import (
    BBox3D,
    DynamicSceneGraph,
    GraphTool,
    Pose3D,
    SpatialQAEngine,
    VLAAnchorPlanner,
    load_scene_fixture,
)


def build_scene_with_history() -> DynamicSceneGraph:
    graph = DynamicSceneGraph()
    graph.set_agent_pose("agent", Pose3D(0.0, 0.0, 0.0, yaw=0.0), step=1)
    graph.add_region("sink_region", "Sink region", step=1)
    graph.upsert_object(
        "table_1",
        "table",
        Pose3D(0.0, 1.0, 0.35),
        BBox3D(center=Pose3D(0.0, 1.0, 0.35), size=(1.2, 0.8, 0.7)),
        confidence=1.0,
        visible=True,
        step=1,
    )
    graph.upsert_object(
        "mug_1",
        "mug",
        Pose3D(-0.4, 1.0, 0.78),
        BBox3D(center=Pose3D(-0.4, 1.0, 0.78), size=(0.12, 0.12, 0.16)),
        confidence=0.95,
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
    graph.add_edge("mug_1", "LEFT_OF", "plate_1", "agent", 1.0, step=1)
    graph.add_edge("mug_1", "ON", "table_1", "world", 1.0, step=1)
    return graph


def test_qa_answers_location_relation_nearest_and_history() -> None:
    graph = build_scene_with_history()
    graph.set_agent_pose("agent", Pose3D(0.5, 0.2, 0.0, yaw=0.25), step=2)
    qa = SpatialQAEngine(GraphTool(graph))

    agent_location = qa.answer({"type": "agent_location"})
    assert agent_location.answer == {
        "agent_id": "agent",
        "pose": {"x": 0.5, "y": 0.2, "z": 0.0, "yaw": 0.25},
    }
    assert agent_location.evidence_nodes == ["agent"]
    assert agent_location.confidence == 1.0

    agent_history = qa.answer({"type": "agent_history"})
    assert agent_history.answer == {
        "agent_id": "agent",
        "poses": [
            {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            {"x": 0.5, "y": 0.2, "z": 0.0, "yaw": 0.25},
        ],
        "steps": [1, 2],
    }
    assert agent_history.evidence_nodes == ["agent", "state:agent:1", "state:agent:2"]
    assert agent_history.evidence_edges == [
        "agent-STATE_CHANGED-state:agent:1-1",
        "agent-STATE_CHANGED-state:agent:2-2",
    ]

    location = qa.answer({"type": "object_location", "object_id": "mug_1"})
    assert location.answer == {
        "object_id": "mug_1",
        "label": "mug",
        "pose": {"x": -0.4, "y": 1.0, "z": 0.78, "yaw": 0.0},
        "visible": True,
        "confidence": 0.95,
        "last_seen_step": 1,
        "state_step": 1,
        "current_location": {"relation": "ON", "dst": "table_1", "step": 1},
    }
    assert location.evidence_nodes == ["mug_1", "state:mug_1:1"]
    assert location.evidence_edges == [
        "mug_1-ON-table_1-1",
        "mug_1-STATE_CHANGED-state:mug_1:1-1",
    ]
    assert location.needs_reobserve is False

    relation = qa.answer(
        {
            "type": "relative_relation",
            "src": "mug_1",
            "relation": "LEFT_OF",
            "dst": "plate_1",
            "reference_frame": "agent",
        }
    )
    assert relation.answer == {"holds": True, "relation": "LEFT_OF", "src": "mug_1", "dst": "plate_1"}
    assert relation.evidence_nodes == ["mug_1", "plate_1"]
    assert relation.evidence_edges == ["mug_1-LEFT_OF-plate_1-1"]

    nearest = qa.answer({"type": "nearest_object", "src": "mug_1"})
    assert nearest.answer == {
        "src": "mug_1",
        "nearest_object": "table_1",
        "distance": 0.587282,
        "candidate_distances": [
            {
                "object_id": "table_1",
                "label": "table",
                "distance": 0.587282,
                "visible": True,
                "confidence": 1.0,
                "needs_reobserve": False,
            },
            {
                "object_id": "plate_1",
                "label": "plate",
                "distance": 0.752396,
                "visible": True,
                "confidence": 0.9,
                "needs_reobserve": False,
            },
        ],
    }
    assert nearest.evidence_nodes == ["mug_1", "table_1", "plate_1"]

    nearest_from_candidates = qa.answer(
        {"type": "nearest_object", "src": "mug_1", "candidates": ["plate_1"]}
    )
    assert nearest_from_candidates.answer == {
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
    assert nearest_from_candidates.evidence_nodes == ["mug_1", "plate_1"]

    graph.move_object(
        "mug_1",
        new_pose=Pose3D(1.2, 0.2, 0.5),
        new_bbox=BBox3D(center=Pose3D(1.2, 0.2, 0.5), size=(0.12, 0.12, 0.16)),
        destination_id="sink_region",
        destination_relation="IN_REGION",
        step=2,
    )
    history = qa.answer({"type": "object_history", "object_id": "mug_1"})
    assert "MOVED_FROM" in history.answer["relations"]
    assert "MOVED_TO" in history.answer["relations"]


def test_qa_answers_object_status_with_reobserve_signal() -> None:
    graph = build_scene_with_history()
    last_seen_pose = graph.get_object_state("mug_1").pose
    graph.upsert_object(
        "mug_1",
        "mug",
        Pose3D(1.0, 1.2, 0.78),
        BBox3D(center=Pose3D(1.0, 1.2, 0.78), size=(0.12, 0.12, 0.16)),
        confidence=0.2,
        visible=False,
        step=3,
    )
    qa = SpatialQAEngine(GraphTool(graph))

    response = qa.answer({"type": "object_status", "object_id": "mug_1"})

    assert response.answer == {
        "object_id": "mug_1",
        "label": "mug",
        "visible": False,
        "confidence": 0.2,
        "last_seen_step": 1,
        "last_seen_pose": last_seen_pose.to_dict(),
        "needs_reobserve": True,
    }
    assert response.evidence_nodes == ["mug_1"]
    assert response.evidence_edges == ["mug_1-STATE_CHANGED-state:mug_1:3-3"]
    assert response.confidence == 0.2
    assert response.needs_reobserve is True


def test_qa_answers_object_room_with_containment_evidence() -> None:
    qa = SpatialQAEngine(GraphTool(load_scene_fixture("multi_room_rearrangement")))

    response = qa.answer({"type": "object_room", "object_id": "cereal_box_1"})

    assert response.error is None
    assert response.answer == {
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
    }
    assert response.evidence_nodes == ["cereal_box_1", "pantry_shelf", "pantry"]
    assert response.evidence_edges == [
        "cereal_box_1-IN_REGION-pantry_shelf-2",
        "pantry_shelf-IN_ROOM-pantry-1",
    ]
    assert response.confidence == 0.92
    assert response.needs_reobserve is False


def test_qa_object_room_returns_none_when_room_cannot_be_resolved() -> None:
    qa = SpatialQAEngine(GraphTool(build_scene_with_history()))

    response = qa.answer({"type": "object_room", "object_id": "mug_1"})

    assert response.error is None
    assert response.answer == {"object_id": "mug_1", "room": None}
    assert response.evidence_nodes == ["mug_1"]
    assert response.evidence_edges == []
    assert response.confidence == 0.0
    assert response.needs_reobserve is False


def test_qa_nearest_object_rejects_invalid_candidates() -> None:
    qa = SpatialQAEngine(GraphTool(build_scene_with_history()))

    string_candidates = qa.answer(
        {"type": "nearest_object", "src": "mug_1", "candidates": "plate_1"}
    )
    mixed_candidates = qa.answer(
        {"type": "nearest_object", "src": "mug_1", "candidates": ["plate_1", 3]}
    )

    assert string_candidates.error == "Question field must be list of strings: candidates"
    assert mixed_candidates.error == "Question field must be list of strings: candidates"


def test_qa_answers_reobserve_targets_with_state_evidence() -> None:
    graph = build_scene_with_history()
    graph.upsert_object(
        "spoon_1",
        "spoon",
        Pose3D(0.2, 0.8, 0.75),
        BBox3D(center=Pose3D(0.2, 0.8, 0.75), size=(0.2, 0.04, 0.02)),
        confidence=0.25,
        visible=False,
        step=3,
    )
    graph.upsert_object(
        "bowl_1",
        "bowl",
        Pose3D(0.6, 0.8, 0.75),
        BBox3D(center=Pose3D(0.6, 0.8, 0.75), size=(0.3, 0.3, 0.12)),
        confidence=0.7,
        visible=False,
        step=3,
    )
    qa = SpatialQAEngine(GraphTool(graph, reobserve_confidence_threshold=0.5))

    response = qa.answer({"type": "reobserve_targets"})
    spoon_response = qa.answer({"type": "reobserve_targets", "label": "spoon"})
    mug_response = qa.answer({"type": "reobserve_targets", "label": "mug"})

    expected_object = {
        "object_id": "spoon_1",
        "label": "spoon",
        "pose": {"x": 0.2, "y": 0.8, "z": 0.75, "yaw": 0.0},
        "visible": False,
        "confidence": 0.25,
        "last_seen_step": None,
        "last_seen_pose": None,
        "state_step": 3,
    }
    assert response.answer == {"count": 1, "objects": [expected_object]}
    assert response.evidence_nodes == ["spoon_1", "state:spoon_1:3"]
    assert response.evidence_edges == ["spoon_1-STATE_CHANGED-state:spoon_1:3-3"]
    assert response.confidence == 0.25
    assert response.needs_reobserve is True
    assert spoon_response.answer == {"count": 1, "objects": [expected_object]}
    assert mug_response.answer == {"count": 0, "objects": []}
    assert mug_response.evidence_nodes == []
    assert mug_response.evidence_edges == []
    assert mug_response.confidence == 1.0
    assert mug_response.needs_reobserve is False


def test_qa_label_candidates_reports_same_label_objects_with_state_evidence() -> None:
    graph = DynamicSceneGraph()
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
    qa = SpatialQAEngine(GraphTool(graph))

    response = qa.answer({"type": "label_candidates", "label": "mug", "visible": True})

    assert response.answer == {
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
    assert response.evidence_nodes == [
        "mug_1",
        "state:mug_1:1",
        "mug_2",
        "state:mug_2:1",
    ]
    assert response.evidence_edges == [
        "mug_1-STATE_CHANGED-state:mug_1:1-1",
        "mug_2-STATE_CHANGED-state:mug_2:1-1",
    ]
    assert response.confidence == 0.88
    assert response.needs_reobserve is False
    assert response.error is None


def test_qa_label_candidates_reports_unique_and_missing_labels() -> None:
    graph = build_scene_with_history()
    qa = SpatialQAEngine(GraphTool(graph))

    unique = qa.answer({"type": "label_candidates", "label": "plate", "visible": True})
    missing = qa.answer({"type": "label_candidates", "label": "fork", "visible": True})
    invalid_visible = qa.answer({"type": "label_candidates", "label": "plate", "visible": "yes"})

    assert unique.answer["count"] == 1
    assert unique.answer["ambiguous"] is False
    assert unique.answer["objects"][0]["object_id"] == "plate_1"
    assert unique.evidence_nodes == ["plate_1", "state:plate_1:1"]
    assert unique.evidence_edges == ["plate_1-STATE_CHANGED-state:plate_1:1-1"]
    assert unique.confidence == 0.9
    assert missing.answer == {
        "label": "fork",
        "visible": True,
        "count": 0,
        "ambiguous": False,
        "objects": [],
    }
    assert missing.evidence_nodes == []
    assert missing.evidence_edges == []
    assert missing.confidence == 0.0
    assert missing.needs_reobserve is False
    assert invalid_visible.error == "Question field must be boolean: visible"


def test_qa_reobserve_targets_on_empty_graph_is_empty() -> None:
    response = SpatialQAEngine(GraphTool(DynamicSceneGraph())).answer(
        {"type": "reobserve_targets"}
    )

    assert response.error is None
    assert response.answer == {"count": 0, "objects": []}
    assert response.evidence_nodes == []
    assert response.evidence_edges == []
    assert response.needs_reobserve is False


def test_qa_agent_location_reports_missing_pose_error() -> None:
    response = SpatialQAEngine(GraphTool(DynamicSceneGraph())).answer({"type": "agent_location"})
    history = SpatialQAEngine(GraphTool(DynamicSceneGraph())).answer({"type": "agent_history"})
    timeline = SpatialQAEngine(GraphTool(DynamicSceneGraph())).answer({"type": "agent_timeline"})

    assert response.error == "Agent pose not found: agent"
    assert response.answer == {}
    assert history.error == "Agent pose history not found: agent"
    assert history.answer == {}
    assert timeline.error == "Agent pose history not found: agent"
    assert timeline.answer == {}


def test_qa_answers_agent_timeline_with_state_evidence() -> None:
    graph = build_scene_with_history()
    graph.set_agent_pose("agent", Pose3D(0.5, 0.2, 0.0, yaw=0.25), step=2)
    qa = SpatialQAEngine(GraphTool(graph))

    response = qa.answer({"type": "agent_timeline"})

    assert response.error is None
    assert response.answer == {
        "agent_id": "agent",
        "timeline": [
            {
                "agent_id": "agent",
                "step": 1,
                "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                "evidence_edges": ["agent-STATE_CHANGED-state:agent:1-1"],
            },
            {
                "agent_id": "agent",
                "step": 2,
                "pose": {"x": 0.5, "y": 0.2, "z": 0.0, "yaw": 0.25},
                "evidence_edges": ["agent-STATE_CHANGED-state:agent:2-2"],
            },
        ],
    }
    assert response.evidence_nodes == ["agent", "state:agent:1", "state:agent:2"]
    assert response.evidence_edges == [
        "agent-STATE_CHANGED-state:agent:1-1",
        "agent-STATE_CHANGED-state:agent:2-2",
    ]
    assert response.confidence == 1.0
    assert response.needs_reobserve is False


def test_qa_reports_stale_action_validity() -> None:
    graph = build_scene_with_history()
    planner = VLAAnchorPlanner(GraphTool(graph))
    command_result = planner.plan_pick(target_object="mug_1")
    assert command_result.status == "ok"
    command = command_result.command
    assert command is not None

    graph.move_object(
        "mug_1",
        new_pose=Pose3D(1.2, 0.2, 0.5),
        new_bbox=BBox3D(center=Pose3D(1.2, 0.2, 0.5), size=(0.12, 0.12, 0.16)),
        destination_id="sink_region",
        destination_relation="IN_REGION",
        step=2,
    )

    qa = SpatialQAEngine(GraphTool(graph))
    response = qa.answer({"type": "next_action_validity", "action": command})

    assert response.answer["valid"] is False
    assert response.answer["needs_replan"] is True
    assert response.answer["reason"] == "stale_object_state"


def test_qa_answers_world_state_after_dynamic_change() -> None:
    graph = build_scene_with_history()
    graph.move_object(
        "mug_1",
        new_pose=Pose3D(1.2, 0.2, 0.5),
        new_bbox=BBox3D(center=Pose3D(1.2, 0.2, 0.5), size=(0.12, 0.12, 0.16)),
        destination_id="sink_region",
        destination_relation="IN_REGION",
        step=2,
        action_id="action_move_mug",
        event_id="event_move_mug",
    )
    qa = SpatialQAEngine(GraphTool(graph))

    response = qa.answer({"type": "world_state", "visible": True})

    assert response.error is None
    assert response.answer["agent_pose"] == {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
    assert response.answer["objects"] == [
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
    assert "agent" in response.evidence_nodes
    assert "mug_1" in response.evidence_nodes
    assert response.evidence_edges == [
        "mug_1-IN_REGION-sink_region-2",
        "mug_1-STATE_CHANGED-state:mug_1:2-2",
    ]


def test_qa_answers_scene_snapshot_at_explicit_step() -> None:
    graph = build_scene_with_history()
    graph.set_agent_pose("agent", Pose3D(0.5, 0.2, 0.0, yaw=0.25), step=2)
    graph.move_object(
        "mug_1",
        new_pose=Pose3D(1.2, 0.2, 0.5),
        new_bbox=BBox3D(center=Pose3D(1.2, 0.2, 0.5), size=(0.12, 0.12, 0.16)),
        destination_id="sink_region",
        destination_relation="IN_REGION",
        step=2,
    )
    qa = SpatialQAEngine(GraphTool(graph))

    response = qa.answer({"type": "scene_snapshot", "step": 1, "visible": True})

    assert response.error is None
    assert response.answer["step"] == 1
    assert response.answer["agent"] == {
        "agent_id": "agent",
        "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        "state_step": 1,
    }
    assert response.answer["objects"][0]["object_id"] == "mug_1"
    assert response.answer["objects"][0]["pose"] == {
        "x": -0.4,
        "y": 1.0,
        "z": 0.78,
        "yaw": 0.0,
    }
    assert response.evidence_nodes[:4] == ["agent", "state:agent:1", "mug_1", "state:mug_1:1"]
    assert response.evidence_edges[:2] == [
        "agent-STATE_CHANGED-state:agent:1-1",
        "mug_1-ON-table_1-1",
    ]
    assert response.confidence == 0.9
    assert response.needs_reobserve is False


def test_qa_scene_snapshot_requires_integer_step() -> None:
    response = SpatialQAEngine(GraphTool(build_scene_with_history())).answer(
        {"type": "scene_snapshot", "step": True}
    )

    assert response.error == "Question field must be integer: step"


def test_qa_answers_scene_delta_between_explicit_steps() -> None:
    graph = build_scene_with_history()
    graph.set_agent_pose("agent", Pose3D(0.5, 0.2, 0.0, yaw=0.25), step=2)
    graph.move_object(
        "mug_1",
        new_pose=Pose3D(1.2, 0.2, 0.5),
        new_bbox=BBox3D(center=Pose3D(1.2, 0.2, 0.5), size=(0.12, 0.12, 0.16)),
        destination_id="sink_region",
        destination_relation="IN_REGION",
        step=2,
    )
    qa = SpatialQAEngine(GraphTool(graph))

    response = qa.answer({"type": "scene_delta", "from_step": 1, "to_step": 2})

    assert response.error is None
    assert response.answer["from_step"] == 1
    assert response.answer["to_step"] == 2
    assert response.answer["agent"]["changed"] is True
    assert response.answer["objects"][0]["object_id"] == "mug_1"
    assert response.answer["objects"][0]["changes"] == ["pose", "last_seen_step", "location"]
    assert "mug_1-MOVED_TO-sink_region-2" in response.evidence_edges
    assert response.confidence == 0.95
    assert response.needs_reobserve is False


def test_qa_scene_delta_requires_ordered_integer_steps() -> None:
    qa = SpatialQAEngine(GraphTool(build_scene_with_history()))

    non_integer = qa.answer({"type": "scene_delta", "from_step": 1, "to_step": True})
    reversed_window = qa.answer({"type": "scene_delta", "from_step": 3, "to_step": 2})

    assert non_integer.error == "Question field must be integer: to_step"
    assert reversed_window.error == "from_step cannot be greater than to_step"


def test_qa_answers_object_timeline_with_state_evidence() -> None:
    graph = build_scene_with_history()
    graph.move_object(
        "mug_1",
        new_pose=Pose3D(1.2, 0.2, 0.5),
        new_bbox=BBox3D(center=Pose3D(1.2, 0.2, 0.5), size=(0.12, 0.12, 0.16)),
        destination_id="sink_region",
        destination_relation="IN_REGION",
        step=2,
    )
    qa = SpatialQAEngine(GraphTool(graph))

    response = qa.answer({"type": "object_timeline", "object_id": "mug_1"})

    assert response.error is None
    assert response.answer["object_id"] == "mug_1"
    assert [entry["step"] for entry in response.answer["timeline"]] == [1, 2]
    assert response.answer["timeline"][0]["current_location"] == {
        "relation": "ON",
        "dst": "table_1",
        "step": 1,
    }
    assert response.answer["timeline"][1]["pose"] == {
        "x": 1.2,
        "y": 0.2,
        "z": 0.5,
        "yaw": 0.0,
    }
    assert response.answer["timeline"][1]["current_location"] == {
        "relation": "IN_REGION",
        "dst": "sink_region",
        "step": 2,
    }
    assert response.evidence_nodes == ["mug_1", "state:mug_1:1", "state:mug_1:2"]
    assert response.evidence_edges == [
        "mug_1-ON-table_1-1",
        "mug_1-STATE_CHANGED-state:mug_1:1-1",
        "mug_1-IN_REGION-sink_region-2",
        "mug_1-STATE_CHANGED-state:mug_1:2-2",
    ]
    assert response.confidence == 0.95
    assert response.needs_reobserve is False


def test_qa_object_timeline_reports_missing_object_error() -> None:
    response = SpatialQAEngine(GraphTool(build_scene_with_history())).answer(
        {"type": "object_timeline", "object_id": "missing"}
    )

    assert response.error == "Object not found: missing"
    assert response.answer == {}


def test_qa_answers_relation_timeline_with_edge_evidence() -> None:
    graph = build_scene_with_history()
    graph.add_edge(
        "mug_1",
        "RIGHT_OF",
        "plate_1",
        "agent",
        0.8,
        step=3,
        evidence=("frame_3",),
        attributes={"inferred": True},
    )
    qa = SpatialQAEngine(GraphTool(graph))

    response = qa.answer(
        {
            "type": "relation_timeline",
            "src": "mug_1",
            "relation": "right_of",
            "dst": "plate_1",
            "reference_frame": "agent",
            "step_min": 2,
            "step_max": 3,
        }
    )

    assert response.error is None
    assert response.answer == {
        "timeline": [
            {
                "id": "mug_1-RIGHT_OF-plate_1-3",
                "src": "mug_1",
                "relation": "RIGHT_OF",
                "dst": "plate_1",
                "reference_frame": "agent",
                "confidence": 0.8,
                "step": 3,
                "evidence": ["frame_3"],
                "attributes": {"inferred": True},
            }
        ]
    }
    assert response.evidence_nodes == ["mug_1", "plate_1"]
    assert response.evidence_edges == ["mug_1-RIGHT_OF-plate_1-3"]
    assert response.confidence == 0.8
    assert response.needs_reobserve is False


def test_qa_answers_structured_graph_query() -> None:
    qa = SpatialQAEngine(GraphTool(build_scene_with_history()))

    response = qa.answer(
        {
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
        }
    )

    assert response.error is None
    assert response.answer == {
        "nodes": [
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
        ],
        "edges": [
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
        ],
    }
    assert response.evidence_nodes == ["mug_1", "plate_1"]
    assert response.evidence_edges == ["mug_1-LEFT_OF-plate_1-1"]
    assert response.confidence == 1.0
    assert response.needs_reobserve is False


def test_qa_graph_query_requires_mapping_query() -> None:
    qa = SpatialQAEngine(GraphTool(build_scene_with_history()))

    missing = qa.answer({"type": "graph_query"})
    invalid = qa.answer({"type": "graph_query", "query": "mug"})

    assert missing.error == "Question field must be mapping: query"
    assert invalid.error == "Question field must be mapping: query"


def test_qa_answers_text_retrieve_subgraph() -> None:
    qa = SpatialQAEngine(GraphTool(build_scene_with_history()))

    response = qa.answer(
        {
            "type": "retrieve_subgraph",
            "query": "mug",
            "max_nodes": 3,
            "hops": 1,
        }
    )

    assert response.error is None
    assert response.answer["nodes"] == [
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
        {
            "id": "table_1",
            "type": "object",
            "label": "table",
            "attributes": {
                "pose": {"x": 0.0, "y": 1.0, "z": 0.35, "yaw": 0.0},
                "confidence": 1.0,
                "visible": True,
                "step": 1,
            },
        },
    ]
    assert response.answer["edges"] == [
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
            "id": "mug_1-ON-table_1-1",
            "src": "mug_1",
            "relation": "ON",
            "dst": "table_1",
            "reference_frame": "world",
            "confidence": 1.0,
            "step": 1,
            "evidence": [],
            "attributes": {},
        },
    ]
    assert response.evidence_nodes == ["mug_1", "plate_1", "table_1"]
    assert response.evidence_edges == [
        "mug_1-LEFT_OF-plate_1-1",
        "mug_1-ON-table_1-1",
    ]
    assert response.confidence == 1.0
    assert response.needs_reobserve is False


def test_qa_retrieve_subgraph_requires_string_query_and_integer_limits() -> None:
    qa = SpatialQAEngine(GraphTool(build_scene_with_history()))

    missing = qa.answer({"type": "retrieve_subgraph"})
    invalid_query = qa.answer({"type": "retrieve_subgraph", "query": 3})
    invalid_max_nodes = qa.answer(
        {"type": "retrieve_subgraph", "query": "mug", "max_nodes": True}
    )
    invalid_hops = qa.answer({"type": "retrieve_subgraph", "query": "mug", "hops": "1"})

    assert missing.error == "Question missing string field: query"
    assert invalid_query.error == "Question missing string field: query"
    assert invalid_max_nodes.error == "Question field must be integer: max_nodes"
    assert invalid_hops.error == "Question field must be integer: hops"


def test_qa_answers_recent_events_with_step_window() -> None:
    graph = build_scene_with_history()
    graph.move_object(
        "mug_1",
        new_pose=Pose3D(1.2, 0.2, 0.5),
        new_bbox=BBox3D(center=Pose3D(1.2, 0.2, 0.5), size=(0.12, 0.12, 0.16)),
        destination_id="sink_region",
        destination_relation="IN_REGION",
        step=2,
        action_id="action_move_mug",
        event_id="event_move_mug",
    )
    qa = SpatialQAEngine(GraphTool(graph))

    response = qa.answer({"type": "recent_events", "since_step": 2, "until_step": 2})

    assert response.error is None
    assert response.answer == {
        "events": [
            {"id": "action_move_mug", "type": "action", "label": "move", "step": 2},
            {"id": "event_move_mug", "type": "event", "label": "move_object", "step": 2},
        ],
        "changes": [
            {
                "src": "action_move_mug",
                "relation": "ACTION_CAUSED",
                "dst": "event_move_mug",
                "step": 2,
            },
            {"src": "mug_1", "relation": "IN_REGION", "dst": "sink_region", "step": 2},
            {"src": "mug_1", "relation": "MOVED_FROM", "dst": "table_1", "step": 2},
            {"src": "mug_1", "relation": "MOVED_TO", "dst": "sink_region", "step": 2},
            {
                "src": "mug_1",
                "relation": "STATE_CHANGED",
                "dst": "state:mug_1:2",
                "step": 2,
            },
        ],
    }
    assert response.evidence_nodes == ["action_move_mug", "event_move_mug"]
    assert response.evidence_edges == [
        "action_move_mug-ACTION_CAUSED-event_move_mug-2",
        "mug_1-IN_REGION-sink_region-2",
        "mug_1-MOVED_FROM-table_1-2",
        "mug_1-MOVED_TO-sink_region-2",
        "mug_1-STATE_CHANGED-state:mug_1:2-2",
    ]
