from dsg_spatialqa_lab import BBox3D, DynamicSceneGraph, Pose3D


def test_dynamic_updates_preserve_move_history_and_current_state() -> None:
    graph = DynamicSceneGraph()
    graph.add_region("sink_region", "Sink region", step=1)
    graph.upsert_object(
        object_id="table_1",
        label="table",
        pose=Pose3D(0.0, 1.0, 0.35),
        bbox=BBox3D(center=Pose3D(0.0, 1.0, 0.35), size=(1.0, 0.7, 0.7)),
        confidence=1.0,
        visible=True,
        step=1,
    )
    graph.upsert_object(
        object_id="mug_1",
        label="mug",
        pose=Pose3D(-0.3, 1.0, 0.78),
        bbox=BBox3D(center=Pose3D(-0.3, 1.0, 0.78), size=(0.12, 0.12, 0.16)),
        confidence=0.94,
        visible=True,
        step=1,
    )
    graph.add_edge("mug_1", "ON", "table_1", "world", confidence=0.92, step=1)

    graph.move_object(
        object_id="mug_1",
        new_pose=Pose3D(1.2, 0.2, 0.5),
        new_bbox=BBox3D(center=Pose3D(1.2, 0.2, 0.5), size=(0.12, 0.12, 0.16)),
        destination_id="sink_region",
        destination_relation="IN_REGION",
        step=2,
        action_id="action_pick_mug_1",
        event_id="event_move_mug_1",
    )

    state = graph.get_object_state("mug_1")
    assert state.pose == Pose3D(1.2, 0.2, 0.5)
    assert state.last_seen_step == 2

    history = graph.history("mug_1")
    relations = [(edge.relation, edge.dst, edge.step) for edge in history]
    assert ("ON", "table_1", 1) in relations
    assert ("MOVED_FROM", "table_1", 2) in relations
    assert ("MOVED_TO", "sink_region", 2) in relations
    assert ("IN_REGION", "sink_region", 2) in relations
    assert any(edge.relation == "STATE_CHANGED" and edge.step == 2 for edge in history)

    assert graph.nodes["action_pick_mug_1"].type == "action"
    assert graph.nodes["action_pick_mug_1"].attributes["step"] == 2
    assert graph.nodes["event_move_mug_1"].type == "event"
    assert graph.nodes["event_move_mug_1"].attributes["step"] == 2
