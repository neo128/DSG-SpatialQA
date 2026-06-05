from dsg_spatialqa_lab.benchmark.active_qa_v2 import (
    active_qa_v2_quality_report,
    build_active_qa_v2_splits,
    build_active_qa_v2_vlm_request_bundle,
)
from dsg_spatialqa_lab.memory import DynamicSceneGraph
from dsg_spatialqa_lab.observations import ObjectObservation, SceneObservation
from dsg_spatialqa_lab.schema import BBox3D, Pose3D


def test_active_qa_v2_generates_relation_situated_temporal_and_leak_free_bundle() -> None:
    pose = Pose3D(0.0, 0.9, 0.0, yaw=90.0)
    object_pose = Pose3D(0.1, 0.9, 0.2, yaw=0.0)
    support_pose = Pose3D(0.2, 0.8, 0.2, yaw=0.0)
    observations = (
        SceneObservation(
            step=10,
            agent_pose=pose,
            objects=(
                ObjectObservation(
                    "apple_1",
                    "apple",
                    object_pose,
                    BBox3D(object_pose, (0.1, 0.1, 0.1)),
                    confidence=1.0,
                    visible=True,
                    attributes={
                        "rgb_path": "rgb/000010.ppm",
                        "depth_path": "depth/000010.npy",
                        "evidence_kinds": ["rgb", "depth", "detector"],
                    },
                ),
                ObjectObservation(
                    "countertop_1",
                    "countertop",
                    support_pose,
                    BBox3D(support_pose, (1.0, 0.1, 1.0)),
                    confidence=1.0,
                    visible=True,
                    attributes={
                        "rgb_path": "rgb/000010.ppm",
                        "depth_path": "depth/000010.npy",
                        "evidence_kinds": ["rgb", "depth", "detector"],
                    },
                ),
            ),
        ),
    )
    graph = DynamicSceneGraph()
    graph.upsert_object(
        "apple_1",
        "apple",
        object_pose,
        BBox3D(object_pose, (0.1, 0.1, 0.1)),
        confidence=1.0,
        visible=True,
        step=10,
    )
    graph.upsert_object(
        "countertop_1",
        "countertop",
        support_pose,
        BBox3D(support_pose, (1.0, 0.1, 1.0)),
        confidence=1.0,
        visible=True,
        step=10,
    )
    graph.add_edge(
        "apple_1",
        "ON",
        "countertop_1",
        "world",
        1.0,
        step=10,
        evidence=["rgb/000010.ppm"],
    )
    trajectory = {
        "episode_id": "episode-001",
        "scene_id": "FloorPlan1",
        "collection_kind": "reachable_relation_centric_nbv",
        "real_ai2thor_runtime": True,
        "navigation_validated": True,
        "steps": [{"step_index": 0, "selected_viewpoint": {"x": 0.0, "z": 0.0}}],
    }

    splits = build_active_qa_v2_splits(
        episode_id="episode-001",
        scene_id="FloorPlan1",
        trajectory=trajectory,
        observations=observations,
        graph=graph,
    )
    report = active_qa_v2_quality_report(
        episode_id="episode-001",
        splits=splits,
    )
    bundle = build_active_qa_v2_vlm_request_bundle(
        episode_id="episode-001",
        records=(
            splits["observation_aware"]
            + splits["relation_centric"]
            + splits["situated"]
            + splits["temporal"]
        ),
    )
    request_case = next(
        case
        for case in bundle["prediction_cases"]
        if case["question_type"] == "support_relation"
    )

    assert splits["observation_aware"]
    assert splits["relation_centric"]
    assert splits["situated"]
    assert splits["temporal"]
    relation = splits["relation_centric"][0]
    assert relation["question_type"] == "support_relation"
    assert relation["required_evidence"]["edges"] == ["apple_1-ON-countertop_1"]
    assert relation["observability"]["evidence_observable"] is True
    assert report["valid"] is True
    assert report["summary"]["question_type_count"] >= 3
    assert bundle["request_count"] == len(
        {case["case_id"] for case in bundle["prediction_cases"]}
    )
    assert bundle["leak_free"] is True
    assert request_case["question_text"]
    assert request_case["primary_frame"]["rgb_path"] == "rgb/000010.ppm"
    assert request_case["answer_options"][0]["option_id"] == "option_1"
    assert request_case["answer_options"][0]["destination_label"] == "countertop"
    assert "gold_answer" not in str(bundle)
    assert "required_nodes" not in str(bundle)
    assert "visible_object_ids" not in str(bundle)
