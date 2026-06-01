import pytest

from dsg_spatialqa_lab import (
    BBox3D,
    DynamicSceneGraph,
    GraphTool,
    NodeObservation,
    ObjectObservation,
    ObservationIngestor,
    Pose3D,
    SceneObservation,
    SpatialQAError,
)


def test_scene_observation_ingestion_updates_graph_and_infers_relations() -> None:
    graph = DynamicSceneGraph()
    ingestor = ObservationIngestor(graph)

    result = ingestor.ingest(
        SceneObservation(
            step=7,
            agent_pose=Pose3D(0.0, 0.0, 0.0, yaw=0.0),
            rooms=(NodeObservation("kitchen", "Kitchen"),),
            regions=(NodeObservation("counter_region", "Counter region"),),
            objects=(
                ObjectObservation(
                    "mug_1",
                    "mug",
                    Pose3D(-0.4, 1.0, 0.78),
                    BBox3D(center=Pose3D(-0.4, 1.0, 0.78), size=(0.12, 0.12, 0.16)),
                    confidence=0.95,
                    visible=True,
                ),
                ObjectObservation(
                    "plate_1",
                    "plate",
                    Pose3D(0.35, 1.0, 0.72),
                    BBox3D(center=Pose3D(0.35, 1.0, 0.72), size=(0.26, 0.26, 0.04)),
                    confidence=0.9,
                    visible=True,
                ),
                ObjectObservation(
                    "table_1",
                    "table",
                    Pose3D(0.0, 1.0, 0.35),
                    BBox3D(center=Pose3D(0.0, 1.0, 0.35), size=(1.2, 0.8, 0.7)),
                    confidence=1.0,
                    visible=True,
                ),
            ),
        ),
        infer_relations=("LEFT_OF", "RIGHT_OF", "NEAR", "ON"),
        reference_frames=("agent", "world"),
        relation_evidence=("mock_observation:7",),
    )

    tool = GraphTool(graph)
    assert result.step == 7
    assert result.node_ids == (
        "agent",
        "counter_region",
        "kitchen",
        "mug_1",
        "plate_1",
        "table_1",
    )
    assert result.object_ids == ("mug_1", "plate_1", "table_1")
    assert result.state_edge_ids == (
        "mug_1-STATE_CHANGED-state:mug_1:7-7",
        "plate_1-STATE_CHANGED-state:plate_1:7-7",
        "table_1-STATE_CHANGED-state:table_1:7-7",
    )
    assert graph.get_agent_pose("agent") == Pose3D(0.0, 0.0, 0.0, yaw=0.0)
    assert graph.nodes["kitchen"].type == "room"
    assert graph.nodes["counter_region"].type == "region"
    assert [edge.id for edge in tool.get_relation("mug_1", "LEFT_OF", "plate_1", "agent")] == [
        "mug_1-LEFT_OF-plate_1-7"
    ]
    assert [edge.evidence for edge in tool.get_relation("mug_1", "ON", "table_1", "world")] == [
        ["mock_observation:7"]
    ]
    assert "mug_1-ON-table_1-7" in result.inferred_edge_ids


def test_scene_observation_ingestion_preserves_last_seen_for_invisible_object() -> None:
    graph = DynamicSceneGraph()
    ingestor = ObservationIngestor(graph)
    visible_pose = Pose3D(0.2, 1.0, 0.4)

    ingestor.ingest(
        SceneObservation(
            step=1,
            objects=(
                ObjectObservation(
                    "spoon_1",
                    "spoon",
                    visible_pose,
                    BBox3D(center=visible_pose, size=(0.2, 0.04, 0.02)),
                    confidence=0.82,
                    visible=True,
                ),
            ),
        )
    )
    ingestor.ingest(
        SceneObservation(
            step=2,
            objects=(
                ObjectObservation(
                    "spoon_1",
                    "spoon",
                    Pose3D(0.6, 1.2, 0.4),
                    BBox3D(center=Pose3D(0.6, 1.2, 0.4), size=(0.2, 0.04, 0.02)),
                    confidence=0.2,
                    visible=False,
                ),
            ),
        )
    )

    state = graph.get_object_state("spoon_1")
    assert state.visible is False
    assert state.last_seen_step == 1
    assert state.last_seen_pose == visible_pose
    assert GraphTool(graph).needs_reobserve("spoon_1") is True


def test_scene_observation_ingestion_rejects_invalid_step() -> None:
    with pytest.raises(SpatialQAError, match="observation step must be an integer"):
        ObservationIngestor(DynamicSceneGraph()).ingest(SceneObservation(step=True))


def test_scene_observation_ingestion_requires_agent_pose_for_agent_relations() -> None:
    graph = DynamicSceneGraph()
    observation = SceneObservation(
        step=3,
        objects=(
            ObjectObservation(
                "mug_1",
                "mug",
                Pose3D(-0.4, 1.0, 0.78),
                BBox3D(center=Pose3D(-0.4, 1.0, 0.78), size=(0.12, 0.12, 0.16)),
                confidence=0.95,
                visible=True,
            ),
            ObjectObservation(
                "plate_1",
                "plate",
                Pose3D(0.35, 1.0, 0.72),
                BBox3D(center=Pose3D(0.35, 1.0, 0.72), size=(0.26, 0.26, 0.04)),
                confidence=0.9,
                visible=True,
            ),
        ),
    )

    with pytest.raises(SpatialQAError, match="Agent pose not found: agent"):
        ObservationIngestor(graph).ingest(
            observation,
            infer_relations=("LEFT_OF",),
            reference_frames=("agent",),
        )
