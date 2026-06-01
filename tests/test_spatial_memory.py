import pytest

from dsg_spatialqa_lab import BBox3D, DynamicSceneGraph, GraphTool, Pose3D, SpatialQAError


def test_spatial_memory_tracks_agent_object_and_last_seen_pose() -> None:
    graph = DynamicSceneGraph()
    graph.set_agent_pose("agent", Pose3D(0.0, 0.0, 0.0, yaw=0.0), step=1)
    graph.set_agent_pose("agent", Pose3D(0.5, 0.2, 0.0, yaw=0.25), step=2)
    graph.add_room("kitchen", "Kitchen", step=1)
    graph.upsert_object(
        object_id="mug_1",
        label="mug",
        pose=Pose3D(-0.4, 1.0, 0.8),
        bbox=BBox3D(center=Pose3D(-0.4, 1.0, 0.8), size=(0.12, 0.12, 0.16)),
        confidence=0.95,
        visible=True,
        step=1,
    )

    tool = GraphTool(graph)

    assert tool.get_agent_pose() == Pose3D(0.5, 0.2, 0.0, yaw=0.25)
    assert [(state.agent_id, state.pose, state.step) for state in tool.get_agent_pose_history()] == [
        ("agent", Pose3D(0.0, 0.0, 0.0, yaw=0.0), 1),
        ("agent", Pose3D(0.5, 0.2, 0.0, yaw=0.25), 2),
    ]
    assert [edge.id for edge in graph.find_edges(src="agent", relation="STATE_CHANGED")] == [
        "agent-STATE_CHANGED-state:agent:1-1",
        "agent-STATE_CHANGED-state:agent:2-2",
    ]
    assert tool.get_object_pose("mug_1") == Pose3D(-0.4, 1.0, 0.8)
    assert graph.get_object_state("mug_1").last_seen_pose == Pose3D(-0.4, 1.0, 0.8)
    assert graph.get_object_state("mug_1").last_seen_step == 1
    assert graph.get_object_state("mug_1").visible is True
    assert tool.needs_reobserve("mug_1") is False


def test_invisible_low_confidence_object_needs_reobserve() -> None:
    graph = DynamicSceneGraph()
    graph.upsert_object(
        object_id="spoon_1",
        label="spoon",
        pose=Pose3D(0.2, 0.8, 0.75),
        bbox=BBox3D(center=Pose3D(0.2, 0.8, 0.75), size=(0.2, 0.04, 0.02)),
        confidence=0.25,
        visible=False,
        step=3,
    )

    tool = GraphTool(graph, reobserve_confidence_threshold=0.5)

    assert tool.needs_reobserve("spoon_1") is True
    assert graph.get_object_state("spoon_1").last_seen_step is None
    assert graph.get_object_state("spoon_1").last_seen_pose is None


def test_missing_agent_pose_and_object_return_clear_errors() -> None:
    tool = GraphTool(DynamicSceneGraph())

    with pytest.raises(SpatialQAError, match="Agent pose not found"):
        tool.get_agent_pose()

    with pytest.raises(SpatialQAError, match="Agent pose history not found"):
        tool.get_agent_pose_history()

    with pytest.raises(SpatialQAError, match="Object not found"):
        tool.get_object("missing_object")
