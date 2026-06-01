import pytest

from dsg_spatialqa_lab import (
    BBox3D,
    DynamicSceneGraph,
    GraphQuery,
    GraphTool,
    load_scene_fixture,
    Pose3D,
    RelationConfig,
    RelationEngine,
    SpatialQAError,
)


def build_scene() -> DynamicSceneGraph:
    graph = DynamicSceneGraph()
    graph.set_agent_pose("agent", Pose3D(0.0, 0.0, 0.0, yaw=0.0), step=1)
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
    graph.upsert_object(
        "table_1",
        "table",
        Pose3D(0.0, 1.0, 0.35),
        BBox3D(center=Pose3D(0.0, 1.0, 0.35), size=(1.2, 0.8, 0.7)),
        confidence=1.0,
        visible=True,
        step=1,
    )
    engine = RelationEngine(RelationConfig(near_threshold=0.9, margin=0.05))
    graph.add_edge("mug_1", "LEFT_OF", "plate_1", "agent", 1.0, step=1)
    graph.add_edge("plate_1", "RIGHT_OF", "mug_1", "agent", 1.0, step=1)
    assert engine.evaluate(
        graph.get_object_state("mug_1").bbox,
        graph.get_object_state("plate_1").bbox,
        "NEAR",
        reference_frame="agent",
        agent_pose=graph.get_agent_pose("agent"),
    )
    graph.add_edge("mug_1", "NEAR", "plate_1", "agent", 1.0, step=1)
    assert engine.evaluate(
        graph.get_object_state("mug_1").bbox,
        graph.get_object_state("table_1").bbox,
        "ON",
        reference_frame="world",
    )
    graph.add_edge("mug_1", "ON", "table_1", "world", 1.0, step=1)
    return graph


def test_find_objects_nearest_relations_and_subgraph_are_deterministic() -> None:
    graph = build_scene()
    tool = GraphTool(graph)

    assert [obj.object_id for obj in tool.find_objects(label="mug")] == ["mug_1"]
    assert [obj.object_id for obj in tool.find_objects(visible=True)] == [
        "mug_1",
        "plate_1",
        "table_1",
    ]
    assert tool.nearest("mug_1").object_id == "table_1"
    assert tool.nearest("mug_1", candidates=["plate_1"]).object_id == "plate_1"
    assert tool.nearest_distances("mug_1") == [
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
    ]

    relations = tool.get_relation("mug_1", "LEFT_OF", "plate_1", reference_frame="agent")
    assert [(edge.src, edge.relation, edge.dst, edge.reference_frame) for edge in relations] == [
        ("mug_1", "LEFT_OF", "plate_1", "agent")
    ]

    subgraph = tool.retrieve_subgraph("mug", max_nodes=3, hops=1)
    assert [node.id for node in subgraph["nodes"]] == ["mug_1", "plate_1", "table_1"]
    assert len(subgraph["edges"]) <= 3
    assert tool.retrieve_subgraph("unknown", max_nodes=3, hops=1) == {"nodes": [], "edges": []}


def test_missing_object_error_is_explicit() -> None:
    tool = GraphTool(build_scene())

    with pytest.raises(SpatialQAError, match="Object not found: missing"):
        tool.get_object("missing")


def test_structured_graph_query_filters_nodes_edges_and_steps() -> None:
    tool = GraphTool(build_scene())

    result = tool.query_graph(
        GraphQuery(
            node_types=("object",),
            labels=("mug", "plate"),
            visible=True,
            relations=("LEFT_OF", "NEAR"),
            reference_frame="agent",
            step_min=1,
            step_max=1,
            max_nodes=5,
            max_edges=5,
        )
    )

    assert [node.id for node in result["nodes"]] == ["mug_1", "plate_1"]
    assert [edge.id for edge in result["edges"]] == [
        "mug_1-LEFT_OF-plate_1-1",
        "mug_1-NEAR-plate_1-1",
    ]


def test_structured_graph_query_accepts_mapping_and_is_deterministically_limited() -> None:
    graph = build_scene()
    graph.add_edge("agent", "VISIBLE_FROM", "mug_1", "agent", 1.0, step=2)
    graph.add_edge("agent", "VISIBLE_FROM", "plate_1", "agent", 1.0, step=2)
    tool = GraphTool(graph)

    result = tool.query_graph(
        {
            "node_ids": ["agent", "mug_1", "plate_1"],
            "relations": ["VISIBLE_FROM"],
            "src": "agent",
            "step_min": 2,
            "step_max": 2,
            "max_nodes": 2,
            "max_edges": 1,
        }
    )

    assert [node.id for node in result["nodes"]] == ["mug_1", "agent"]
    assert [edge.id for edge in result["edges"]] == ["agent-VISIBLE_FROM-mug_1-2"]


def test_structured_graph_query_rejects_invalid_limits() -> None:
    tool = GraphTool(build_scene())

    with pytest.raises(SpatialQAError, match="max_nodes must be positive"):
        tool.query_graph(GraphQuery(max_nodes=0))

    with pytest.raises(SpatialQAError, match="step_min cannot be greater than step_max"):
        tool.query_graph({"step_min": 3, "step_max": 2})


def test_scene_snapshot_reconstructs_state_at_explicit_step() -> None:
    graph = build_scene()
    graph.add_region("sink_region", "Sink region", step=1)
    graph.set_agent_pose("agent", Pose3D(0.5, 0.2, 0.0, yaw=0.25), step=2)
    graph.move_object(
        "mug_1",
        new_pose=Pose3D(1.2, 0.2, 0.5),
        new_bbox=BBox3D(center=Pose3D(1.2, 0.2, 0.5), size=(0.12, 0.12, 0.16)),
        destination_id="sink_region",
        destination_relation="IN_REGION",
        step=2,
    )
    tool = GraphTool(graph)

    step_one = tool.scene_snapshot(step=1)
    step_two = tool.scene_snapshot(step=2)

    assert step_one["answer"]["agent"] == {
        "agent_id": "agent",
        "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        "state_step": 1,
    }
    assert step_one["answer"]["objects"][0] == {
        "object_id": "mug_1",
        "label": "mug",
        "pose": {"x": -0.4, "y": 1.0, "z": 0.78, "yaw": 0.0},
        "visible": True,
        "confidence": 0.95,
        "last_seen_step": 1,
        "state_step": 1,
        "current_location": {"relation": "ON", "dst": "table_1", "step": 1},
    }
    assert step_two["answer"]["agent"] == {
        "agent_id": "agent",
        "pose": {"x": 0.5, "y": 0.2, "z": 0.0, "yaw": 0.25},
        "state_step": 2,
    }
    assert step_two["answer"]["objects"][0]["pose"] == {
        "x": 1.2,
        "y": 0.2,
        "z": 0.5,
        "yaw": 0.0,
    }
    assert step_two["answer"]["objects"][0]["current_location"] == {
        "relation": "IN_REGION",
        "dst": "sink_region",
        "step": 2,
    }
    assert step_two["evidence_nodes"][:4] == [
        "agent",
        "state:agent:2",
        "mug_1",
        "state:mug_1:2",
    ]
    assert "mug_1-IN_REGION-sink_region-2" in step_two["evidence_edges"]


def test_scene_snapshot_rejects_missing_or_invalid_step_context() -> None:
    graph = build_scene()
    tool = GraphTool(graph)

    with pytest.raises(SpatialQAError, match="step must be an integer"):
        tool.scene_snapshot(step=True)

    with pytest.raises(SpatialQAError, match="Agent pose history not found at or before step: agent@0"):
        tool.scene_snapshot(step=0)


def test_scene_delta_reports_changes_between_explicit_steps() -> None:
    graph = build_scene()
    graph.add_region("sink_region", "Sink region", step=1)
    graph.set_agent_pose("agent", Pose3D(0.5, 0.2, 0.0, yaw=0.25), step=2)
    graph.move_object(
        "mug_1",
        new_pose=Pose3D(1.2, 0.2, 0.5),
        new_bbox=BBox3D(center=Pose3D(1.2, 0.2, 0.5), size=(0.12, 0.12, 0.16)),
        destination_id="sink_region",
        destination_relation="IN_REGION",
        step=2,
    )

    delta = GraphTool(graph).scene_delta(from_step=1, to_step=2)

    assert delta["answer"]["agent"] == {
        "agent_id": "agent",
        "changed": True,
        "from_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        "to_pose": {"x": 0.5, "y": 0.2, "z": 0.0, "yaw": 0.25},
        "from_state_step": 1,
        "to_state_step": 2,
    }
    assert delta["answer"]["objects"] == [
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
    assert delta["evidence_nodes"][:6] == [
        "agent",
        "state:agent:1",
        "mug_1",
        "state:mug_1:1",
        "plate_1",
        "state:plate_1:1",
    ]
    assert "mug_1-MOVED_FROM-table_1-2" in delta["evidence_edges"]
    assert "mug_1-MOVED_TO-sink_region-2" in delta["evidence_edges"]


def test_scene_delta_rejects_invalid_step_window() -> None:
    tool = GraphTool(build_scene())

    with pytest.raises(SpatialQAError, match="from_step must be an integer"):
        tool.scene_delta(from_step=True, to_step=2)

    with pytest.raises(SpatialQAError, match="from_step cannot be greater than to_step"):
        tool.scene_delta(from_step=3, to_step=2)


def test_world_state_reports_current_objects_and_evidence() -> None:
    graph = build_scene()
    graph.add_region("sink_region", "Sink region", step=1)
    graph.move_object(
        "mug_1",
        new_pose=Pose3D(1.2, 0.2, 0.5),
        new_bbox=BBox3D(center=Pose3D(1.2, 0.2, 0.5), size=(0.12, 0.12, 0.16)),
        destination_id="sink_region",
        destination_relation="IN_REGION",
        step=2,
    )

    state = GraphTool(graph).world_state(visible=True)

    assert state["answer"]["agent_pose"] == {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
    assert state["answer"]["objects"] == [
        {
            "object_id": "mug_1",
            "label": "mug",
            "pose": {"x": 1.2, "y": 0.2, "z": 0.5, "yaw": 0.0},
            "visible": True,
            "confidence": 0.95,
            "last_seen_step": 2,
            "current_location": {"relation": "IN_REGION", "dst": "sink_region", "step": 2},
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
    assert state["evidence_nodes"] == ["agent", "mug_1", "plate_1", "table_1"]
    assert state["evidence_edges"] == [
        "mug_1-IN_REGION-sink_region-2",
        "mug_1-STATE_CHANGED-state:mug_1:2-2",
    ]


def test_world_state_reports_missing_agent_pose_error() -> None:
    with pytest.raises(SpatialQAError, match="Agent pose not found: agent"):
        GraphTool(DynamicSceneGraph()).world_state()


def test_current_room_reports_containment_path_and_evidence() -> None:
    tool = GraphTool(load_scene_fixture("multi_room_rearrangement"))

    assert tool.current_room("cereal_box_1") == {
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
        "evidence_edges": [
            "cereal_box_1-IN_REGION-pantry_shelf-2",
            "pantry_shelf-IN_ROOM-pantry-1",
        ],
    }
    assert tool.current_room("milk_1") == {
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
        "evidence_edges": [
            "milk_1-IN_REGION-prep_counter-1",
            "prep_counter-IN_ROOM-kitchen-1",
        ],
    }


def test_current_room_returns_none_when_room_cannot_be_resolved() -> None:
    assert GraphTool(build_scene()).current_room("mug_1") is None


def test_recent_events_reports_action_event_and_step_window_changes() -> None:
    graph = build_scene()
    graph.add_region("sink_region", "Sink region", step=1)
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

    recent = GraphTool(graph).recent_events(since_step=2, until_step=2)

    assert recent["answer"]["events"] == [
        {"id": "action_move_mug", "type": "action", "label": "move", "step": 2},
        {"id": "event_move_mug", "type": "event", "label": "move_object", "step": 2},
    ]
    assert recent["answer"]["changes"] == [
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
    ]
    assert recent["evidence_nodes"] == ["action_move_mug", "event_move_mug"]
    assert recent["evidence_edges"] == [
        "action_move_mug-ACTION_CAUSED-event_move_mug-2",
        "mug_1-IN_REGION-sink_region-2",
        "mug_1-MOVED_FROM-table_1-2",
        "mug_1-MOVED_TO-sink_region-2",
        "mug_1-STATE_CHANGED-state:mug_1:2-2",
    ]


def test_recent_events_rejects_invalid_step_window() -> None:
    tool = GraphTool(build_scene())

    with pytest.raises(SpatialQAError, match="since_step must be an integer"):
        tool.recent_events(since_step=True)

    with pytest.raises(SpatialQAError, match="since_step cannot be greater than until_step"):
        tool.recent_events(since_step=3, until_step=2)


def test_agent_timeline_reports_explicit_pose_sequence() -> None:
    graph = build_scene()
    graph.set_agent_pose("agent", Pose3D(0.5, 0.2, 0.0, yaw=0.25), step=2)

    timeline = GraphTool(graph).agent_timeline("agent")

    assert timeline == [
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
    ]


def test_agent_timeline_reports_missing_agent_error() -> None:
    with pytest.raises(SpatialQAError, match="Agent pose history not found: missing_agent"):
        GraphTool(build_scene()).agent_timeline("missing_agent")


def test_object_timeline_reports_explicit_state_sequence() -> None:
    graph = build_scene()
    graph.add_region("sink_region", "Sink region", step=1)
    graph.move_object(
        "mug_1",
        new_pose=Pose3D(1.2, 0.2, 0.5),
        new_bbox=BBox3D(center=Pose3D(1.2, 0.2, 0.5), size=(0.12, 0.12, 0.16)),
        destination_id="sink_region",
        destination_relation="IN_REGION",
        step=2,
    )

    timeline = GraphTool(graph).object_timeline("mug_1")

    assert timeline == [
        {
            "object_id": "mug_1",
            "label": "mug",
            "step": 1,
            "pose": {"x": -0.4, "y": 1.0, "z": 0.78, "yaw": 0.0},
            "visible": True,
            "confidence": 0.95,
            "last_seen_step": 1,
            "last_seen_pose": {"x": -0.4, "y": 1.0, "z": 0.78, "yaw": 0.0},
            "current_location": {"relation": "ON", "dst": "table_1", "step": 1},
            "evidence_edges": [
                "mug_1-ON-table_1-1",
                "mug_1-STATE_CHANGED-state:mug_1:1-1",
            ],
        },
        {
            "object_id": "mug_1",
            "label": "mug",
            "step": 2,
            "pose": {"x": 1.2, "y": 0.2, "z": 0.5, "yaw": 0.0},
            "visible": True,
            "confidence": 0.95,
            "last_seen_step": 2,
            "last_seen_pose": {"x": 1.2, "y": 0.2, "z": 0.5, "yaw": 0.0},
            "current_location": {"relation": "IN_REGION", "dst": "sink_region", "step": 2},
            "evidence_edges": [
                "mug_1-IN_REGION-sink_region-2",
                "mug_1-STATE_CHANGED-state:mug_1:2-2",
            ],
        },
    ]


def test_object_timeline_reports_missing_object_error() -> None:
    with pytest.raises(SpatialQAError, match="Object not found: missing"):
        GraphTool(build_scene()).object_timeline("missing")


def test_relation_timeline_filters_edges_deterministically() -> None:
    graph = build_scene()
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
    tool = GraphTool(graph)

    timeline = tool.relation_timeline(
        src="mug_1",
        dst="plate_1",
        reference_frame="agent",
    )
    filtered = tool.relation_timeline(
        src="mug_1",
        relation="right_of",
        dst="plate_1",
        reference_frame="agent",
        step_min=2,
        step_max=3,
    )

    assert [entry["id"] for entry in timeline] == [
        "mug_1-LEFT_OF-plate_1-1",
        "mug_1-NEAR-plate_1-1",
        "mug_1-RIGHT_OF-plate_1-3",
    ]
    assert filtered == [
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

    with pytest.raises(SpatialQAError, match="step_min cannot be greater than step_max"):
        tool.relation_timeline(step_min=4, step_max=3)


def test_reobserve_targets_returns_invisible_low_confidence_objects() -> None:
    graph = build_scene()
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
    graph.upsert_object(
        "mug_1",
        "mug",
        Pose3D(-0.4, 1.0, 0.78),
        BBox3D(center=Pose3D(-0.4, 1.0, 0.78), size=(0.12, 0.12, 0.16)),
        confidence=0.2,
        visible=True,
        step=4,
    )
    tool = GraphTool(graph, reobserve_confidence_threshold=0.5)

    assert [state.object_id for state in tool.reobserve_targets()] == ["spoon_1"]
    assert [state.object_id for state in tool.reobserve_targets(label="spoon")] == ["spoon_1"]
    assert tool.reobserve_targets(label="mug") == []


def test_reobserve_targets_on_empty_graph_is_empty() -> None:
    assert GraphTool(DynamicSceneGraph()).reobserve_targets() == []


def test_update_spatial_relations_infers_edges_and_preserves_history() -> None:
    graph = DynamicSceneGraph()
    graph.set_agent_pose("agent", Pose3D(0.0, 0.0, 0.0, yaw=0.0), step=1)
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
    tool = GraphTool(graph)

    edges = tool.update_spatial_relations(
        step=5,
        object_ids=("mug_1", "plate_1"),
        relations=("LEFT_OF", "RIGHT_OF", "NEAR"),
        reference_frames=("agent",),
        confidence=0.77,
        evidence=("unit_test",),
    )
    repeated_edges = tool.update_spatial_relations(
        step=5,
        object_ids=("mug_1", "plate_1"),
        relations=("LEFT_OF", "RIGHT_OF", "NEAR"),
        reference_frames=("agent",),
        confidence=0.77,
        evidence=("unit_test",),
    )

    assert [edge.id for edge in edges] == [
        "mug_1-LEFT_OF-plate_1-5",
        "mug_1-NEAR-plate_1-5",
        "plate_1-NEAR-mug_1-5",
        "plate_1-RIGHT_OF-mug_1-5",
    ]
    assert repeated_edges == []
    assert [(edge.reference_frame, edge.confidence, edge.evidence, edge.attributes) for edge in edges] == [
        ("agent", 0.77, ["unit_test"], {"inferred": True}),
        ("agent", 0.77, ["unit_test"], {"inferred": True}),
        ("agent", 0.77, ["unit_test"], {"inferred": True}),
        ("agent", 0.77, ["unit_test"], {"inferred": True}),
    ]
    assert [edge.id for edge in graph.history("mug_1")] == [
        "mug_1-STATE_CHANGED-state:mug_1:1-1",
        "mug_1-LEFT_OF-plate_1-5",
        "mug_1-NEAR-plate_1-5",
        "plate_1-NEAR-mug_1-5",
        "plate_1-RIGHT_OF-mug_1-5",
    ]


def test_update_spatial_relations_can_infer_world_on_relation() -> None:
    graph = DynamicSceneGraph()
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
        "table_1",
        "table",
        Pose3D(0.0, 1.0, 0.35),
        BBox3D(center=Pose3D(0.0, 1.0, 0.35), size=(1.2, 0.8, 0.7)),
        confidence=1.0,
        visible=True,
        step=1,
    )

    edges = GraphTool(graph).update_spatial_relations(
        step=4,
        object_ids=("mug_1", "table_1"),
        relations=("ON",),
        reference_frames=("world",),
    )

    assert [edge.id for edge in edges] == ["mug_1-ON-table_1-4"]


def test_update_spatial_relations_requires_agent_pose_for_agent_frame() -> None:
    graph = DynamicSceneGraph()
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

    with pytest.raises(SpatialQAError, match="Agent pose not found: agent"):
        GraphTool(graph).update_spatial_relations(
            step=2,
            object_ids=("mug_1", "plate_1"),
            relations=("LEFT_OF",),
            reference_frames=("agent",),
        )


def test_agent_frame_relation_requires_agent_pose() -> None:
    graph = DynamicSceneGraph()
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

    with pytest.raises(SpatialQAError, match="Agent pose required"):
        RelationEngine().evaluate(
            graph.get_object_state("mug_1").bbox,
            graph.get_object_state("plate_1").bbox,
            "LEFT_OF",
            reference_frame="agent",
            agent_pose=None,
        )
