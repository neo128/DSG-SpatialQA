from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from _pytest.capture import CaptureFixture
import pytest

import dsg_spatialqa_lab as lab
from dsg_spatialqa_lab import EpisodeFrame, Pose3D


ROOT = Path(__file__).resolve().parents[1]
PREDICTED_SCRIPT = ROOT / "scripts" / "build_predicted_graph.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_predicted_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_predicted_graph_script",
        PREDICTED_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_mock_perception_projects_detection_to_instance() -> None:
    assert hasattr(lab, "Detection2D")
    assert hasattr(lab, "Instance3D")
    assert hasattr(lab, "MockSegmenter")
    assert hasattr(lab, "MockDepthProjector")
    frame = _frame(
        1,
        detections=[
            _detection_payload(
                "mug_det",
                "mug_1",
                "mug",
                pose={"x": 0.2, "y": 1.0, "z": 0.8, "yaw": 0.0},
                bbox_size=(0.2, 0.2, 0.3),
            ),
        ],
    )

    detections = lab.MockSegmenter().detect(frame)
    instance = lab.MockDepthProjector().project(frame, detections[0])

    assert detections == (
        lab.Detection2D(
            detection_id="mug_det",
            label="mug",
            bbox_xyxy=(10.0, 20.0, 30.0, 50.0),
            confidence=0.9,
            depth=1.0,
            visible=True,
            attributes={
                "bbox_size": (0.2, 0.2, 0.3),
                "object_id": "mug_1",
                "pose": {"x": 0.2, "y": 1.0, "z": 0.8, "yaw": 0.0},
            },
        ),
    )
    assert instance == lab.Instance3D(
        instance_id="mug_1",
        label="mug",
        pose=Pose3D(0.2, 1.0, 0.8),
        bbox=lab.BBox3D(center=Pose3D(0.2, 1.0, 0.8), size=(0.2, 0.2, 0.3)),
        confidence=0.9,
        visible=True,
        step=1,
        source_detection_id="mug_det",
        attributes={
            "bbox_size": (0.2, 0.2, 0.3),
            "object_id": "mug_1",
            "pose": {"x": 0.2, "y": 1.0, "z": 0.8, "yaw": 0.0},
        },
    )


def test_predicted_graph_tracks_stable_ids_missing_detections_and_relations() -> None:
    assert hasattr(lab, "SimpleObjectTracker")
    assert hasattr(lab, "SimpleObjectFusion")
    assert hasattr(lab, "build_predicted_graph_from_episode")
    frames = _predicted_frames()

    graph = lab.build_predicted_graph_from_episode(frames)
    mug = graph.get_object_state("mug_1")
    plate = graph.get_object_state("plate_1")
    metrics = lab.compare_graphs(_oracle_graph_for_predicted_frames(), graph)

    assert sorted(graph.object_states) == ["mug_1", "plate_1"]
    assert [state.step for state in graph.object_state_history["mug_1"]] == [1, 2]
    assert mug.pose == Pose3D(0.35, 1.0, 0.82)
    assert mug.visible is True
    assert plate.visible is False
    assert plate.confidence == 0.2
    assert plate.last_seen_step == 1
    assert plate.last_seen_pose == Pose3D(0.0, 1.0, 0.74)
    assert [
        (edge.src, edge.relation, edge.dst, edge.reference_frame, edge.step)
        for edge in graph.find_edges(src="plate_1", relation="LEFT_OF")
    ] == [("plate_1", "LEFT_OF", "mug_1", "agent", 1)]
    assert metrics["summary"] == {
        "oracle_object_count": 2,
        "predicted_object_count": 2,
        "matched_object_count": 2,
        "predicted_unlocated_object_count": 2,
        "oracle_relation_count": 10,
        "predicted_relation_count": 10,
        "matched_relation_count": 10,
    }
    assert metrics["metrics"]["object_precision"]["rate"] == 1.0
    assert metrics["metrics"]["relation_recall"]["rate"] == 1.0


def test_predicted_graph_propagates_detection_source_metadata() -> None:
    frames = (
        _frame(
            1,
            detections=[
                _detection_payload(
                    "mug_det_1",
                    "mug_1",
                    "mug",
                    pose={"x": 0.3, "y": 1.0, "z": 0.82, "yaw": 0.0},
                    bbox_size=(0.2, 0.2, 0.3),
                    attributes={"source": "vlm_detector", "source_kind": "vlm"},
                ),
                _detection_payload(
                    "plate_det_1",
                    "plate_1",
                    "plate",
                    pose={"x": 0.0, "y": 1.0, "z": 0.74, "yaw": 0.0},
                    bbox_size=(0.3, 0.3, 0.04),
                    attributes={
                        "source_name": "caption_memory_import",
                        "source_kind": "caption_memory",
                    },
                ),
            ],
        ),
    )

    graph = lab.build_predicted_graph_from_episode(frames)
    comparison = lab.compare_graphs(_oracle_graph_for_source_frames(), graph)

    assert graph.nodes["mug_1"].attributes["source"] == "vlm_detector"
    assert graph.nodes["mug_1"].attributes["source_kind"] == "vlm"
    assert graph.nodes["mug_1"].attributes["source_detection_id"] == "mug_det_1"
    assert graph.nodes["plate_1"].attributes["source"] == "caption_memory_import"
    assert graph.nodes["plate_1"].attributes["source_kind"] == "caption_memory"
    assert {
        edge.attributes["source"]
        for edge in graph.edges
        if edge.attributes.get("inferred") is True
    } == {"geometry_inference"}
    assert comparison["breakdown"]["by_prediction_source"]["objects"] == {
        "caption_memory_import": {
            "confidence_weighted_precision": 1.0,
            "matched_count": 1,
            "matched_weight": 0.9,
            "precision": 1.0,
            "predicted_count": 1,
            "total_weight": 0.9,
        },
        "vlm_detector": {
            "confidence_weighted_precision": 1.0,
            "matched_count": 1,
            "matched_weight": 0.9,
            "precision": 1.0,
            "predicted_count": 1,
            "total_weight": 0.9,
        },
    }
    assert comparison["breakdown"]["by_prediction_source"]["relations"][
        "geometry_inference"
    ]["predicted_count"] == 4


def test_predicted_graph_builds_from_explicit_observation_sequence_artifact(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "build_predicted_graph_from_observations")
    assert hasattr(lab, "predicted_graph_observation_summary")
    assert hasattr(lab, "predicted_graph_report_from_observations")
    observations = _detector_observations()
    input_path = tmp_path / "detector-observations.json"
    graph_path = tmp_path / "observation-predicted-graph.json"
    lab.save_scene_observation_sequence(observations, input_path)

    graph = lab.build_predicted_graph_from_observations(
        observations,
        source_path=input_path,
        infer_relations=("LEFT_OF", "RIGHT_OF", "NEAR"),
        reference_frames=("world",),
    )
    lab.save_graph_json(graph, graph_path)
    report = lab.predicted_graph_report_from_observations(
        input_path=input_path,
        graph_path=graph_path,
        graph=graph,
        observations=observations,
        infer_relations=("LEFT_OF", "RIGHT_OF", "NEAR"),
        reference_frames=("world",),
    )
    validation = lab.validate_predicted_graph_report(report)
    comparison = lab.compare_predicted_graph_report(report)

    assert sorted(graph.object_states) == ["mug_1", "plate_1"]
    assert graph.nodes["mug_1"].attributes["source"] == "rgbd_tracker"
    assert graph.nodes["mug_1"].attributes["hidden_reason"] == "not_detected_in_frame"
    assert graph.nodes["plate_1"].attributes["source"] == "rgbd_detector"
    assert graph.get_object_state("mug_1").visible is False
    assert graph.get_object_state("mug_1").confidence == 0.35
    assert [
        (edge.src, edge.relation, edge.dst, edge.reference_frame, edge.step)
        for edge in graph.find_edges(src="plate_1", relation="NEAR", dst="mug_1")
    ] == [("plate_1", "NEAR", "mug_1", "world", 1)]
    assert report["input_kind"] == "observation_sequence"
    assert report["observation_sequence_digest"] == lab.scene_observation_sequence_digest(
        observations
    )
    assert report["summary"]["predicted_summary"] == {
        "input_kind": "observation_sequence",
        "observation_count": 2,
        "object_observation_count": 3,
        "visible_object_observation_count": 2,
        "hidden_object_observation_count": 1,
        "inferred_relation_count": 2,
        "by_observation_source": {
            "rgbd_detector": 2,
            "rgbd_tracker": 1,
        },
        "by_object_label": {"mug": 2, "plate": 1},
    }
    assert validation["valid"] is True
    assert comparison["matches"] is True


def test_predicted_graph_can_infer_observation_containment_relations(
    tmp_path: Path,
) -> None:
    observations = _containment_observations()
    input_path = tmp_path / "detector-observations.json"
    graph_path = tmp_path / "observation-predicted-graph.json"
    lab.save_scene_observation_sequence(observations, input_path)

    graph = lab.build_predicted_graph_from_observations(
        observations,
        source_path=input_path,
        infer_relations=("NEAR",),
        reference_frames=("world",),
        infer_containment=True,
        containment_axis="z",
    )
    lab.save_graph_json(graph, graph_path)
    report = lab.predicted_graph_report_from_observations(
        input_path=input_path,
        graph_path=graph_path,
        graph=graph,
        observations=observations,
        infer_relations=("NEAR",),
        reference_frames=("world",),
        infer_containment=True,
        containment_axis="z",
    )
    validation = lab.validate_predicted_graph_report(report)
    comparison = lab.compare_predicted_graph_report(report)

    assert [
        (edge.src, edge.relation, edge.dst, edge.reference_frame, edge.step)
        for edge in graph.find_edges(src="mug_1")
        if edge.relation in {"IN_REGION", "IN_ROOM", "ON"}
    ] == [
        ("mug_1", "IN_REGION", "visible_region", "world", 1),
        ("mug_1", "IN_ROOM", "kitchen", "world", 1),
        ("mug_1", "ON", "table_1", "world", 1),
    ]
    assert {
        edge.attributes["source"]
        for edge in graph.find_edges(src="mug_1")
        if edge.relation in {"IN_REGION", "IN_ROOM", "ON"}
    } == {"containment_inference"}
    assert report["options"]["infer_containment"] is True
    assert report["options"]["containment_axis"] == "z"
    assert validation["valid"] is True
    assert comparison["matches"] is True


def test_predicted_graph_containment_keeps_single_best_on_support() -> None:
    observations = (
        lab.SceneObservation(
            step=1,
            objects=(
                lab.ObjectObservation(
                    "counter_1",
                    "countertop",
                    Pose3D(0.18, 0.0, 0.5),
                    lab.BBox3D(
                        center=Pose3D(0.18, 0.0, 0.5),
                        size=(0.3, 1.0, 1.0),
                    ),
                    confidence=0.9,
                    visible=True,
                    attributes={"source": "rgbd_detector"},
                ),
                lab.ObjectObservation(
                    "mug_1",
                    "mug",
                    Pose3D(0.0, 0.0, 1.15),
                    lab.BBox3D(
                        center=Pose3D(0.0, 0.0, 1.15),
                        size=(0.2, 0.2, 0.3),
                    ),
                    confidence=0.88,
                    visible=True,
                    attributes={"source": "rgbd_detector"},
                ),
                lab.ObjectObservation(
                    "table_1",
                    "table",
                    Pose3D(0.0, 0.0, 0.5),
                    lab.BBox3D(
                        center=Pose3D(0.0, 0.0, 0.5),
                        size=(1.0, 1.0, 1.0),
                    ),
                    confidence=0.9,
                    visible=True,
                    attributes={"source": "rgbd_detector"},
                ),
            ),
        ),
    )

    graph = lab.build_predicted_graph_from_observations(
        observations,
        infer_relations=(),
        infer_containment=True,
        containment_axis="z",
    )

    assert [
        (edge.src, edge.relation, edge.dst)
        for edge in graph.find_edges(src="mug_1", relation="ON")
    ] == [("mug_1", "ON", "table_1")]
    assert graph.nodes["mug_1"].attributes["current_location_id"] == "table_1"
    assert graph.nodes["mug_1"].attributes["current_location_relation"] == "ON"


def test_predicted_graph_uses_explicit_detector_current_location() -> None:
    observations = (
        lab.SceneObservation(
            step=1,
            rooms=(lab.NodeObservation("kitchen", "Kitchen"),),
            regions=(
                lab.NodeObservation(
                    "counter_region",
                    "Counter region",
                    attributes={"room_id": "kitchen"},
                ),
            ),
            objects=(
                lab.ObjectObservation(
                    "mug_1",
                    "mug",
                    Pose3D(0.2, 0.0, 0.8),
                    lab.BBox3D(
                        center=Pose3D(0.2, 0.0, 0.8),
                        size=(0.2, 0.2, 0.3),
                    ),
                    confidence=0.86,
                    visible=True,
                    attributes={
                        "source": "rgbd_detector",
                        "source_kind": "detector",
                        "evidence_kinds": ("rgb", "depth", "detector"),
                        "current_location_id": "counter_region",
                        "current_location_relation": "IN_REGION",
                    },
                ),
            ),
        ),
    )

    graph = lab.build_predicted_graph_from_observations(
        observations,
        infer_relations=(),
        infer_containment=False,
    )

    assert [
        (edge.src, edge.relation, edge.dst, edge.reference_frame, edge.step)
        for edge in graph.find_edges(src="mug_1", relation="IN_REGION")
    ] == [("mug_1", "IN_REGION", "counter_region", "world", 1)]
    assert graph.nodes["mug_1"].attributes["current_location_id"] == "counter_region"
    assert graph.nodes["mug_1"].attributes["current_location_relation"] == "IN_REGION"
    assert graph.nodes["mug_1"].attributes["current_room_id"] == "kitchen"


def test_predicted_graph_creates_detector_current_region_when_missing() -> None:
    observations = (
        lab.SceneObservation(
            step=1,
            objects=(
                lab.ObjectObservation(
                    "mug_1",
                    "mug",
                    Pose3D(0.2, 0.0, 0.8),
                    lab.BBox3D(
                        center=Pose3D(0.2, 0.0, 0.8),
                        size=(0.2, 0.2, 0.3),
                    ),
                    confidence=0.86,
                    visible=True,
                    attributes={
                        "source": "rgbd_detector",
                        "source_kind": "detector",
                        "evidence_kinds": ("rgb", "depth", "detector"),
                        "current_location_id": "visible_frame_region:episode-1:0001",
                        "current_location_label": "Visible frame region",
                        "current_location_relation": "IN_REGION",
                    },
                ),
            ),
        ),
    )

    graph = lab.build_predicted_graph_from_observations(
        observations,
        infer_relations=(),
        infer_containment=False,
    )

    region = graph.nodes["visible_frame_region:episode-1:0001"]
    assert region.type == "region"
    assert region.label == "Visible frame region"
    assert region.attributes["source"] == "detector_current_location"
    assert region.attributes["source_kind"] == "detector"
    assert [
        (edge.src, edge.relation, edge.dst, edge.reference_frame, edge.step)
        for edge in graph.find_edges(src="mug_1", relation="IN_REGION")
    ] == [("mug_1", "IN_REGION", "visible_frame_region:episode-1:0001", "world", 1)]
    assert graph.nodes["mug_1"].attributes["current_location_id"] == (
        "visible_frame_region:episode-1:0001"
    )
    assert graph.nodes["mug_1"].attributes["current_location_relation"] == "IN_REGION"


def test_predicted_graph_requires_detector_state_evidence_when_enabled() -> None:
    observations = (
        lab.SceneObservation(
            step=1,
            objects=(
                lab.ObjectObservation(
                    "mug_1",
                    "mug",
                    Pose3D(0.2, 0.0, 0.8),
                    lab.BBox3D(
                        center=Pose3D(0.2, 0.0, 0.8),
                        size=(0.2, 0.2, 0.3),
                    ),
                    confidence=0.86,
                    visible=True,
                    attributes={
                        "source_kind": "ai2thor_metadata_coverage",
                        "states": {"isOpen": False},
                    },
                ),
            ),
        ),
    )

    with pytest.raises(
        lab.SpatialQAError,
        match="state evidence requires detector source_kind",
    ):
        lab.build_predicted_graph_from_observations(
            observations,
            infer_relations=(),
            infer_containment=False,
            require_detector_state_evidence=True,
        )


def test_predicted_graph_rejects_non_detector_current_location() -> None:
    observations = (
        lab.SceneObservation(
            step=1,
            regions=(lab.NodeObservation("counter_region", "Counter region"),),
            objects=(
                lab.ObjectObservation(
                    "mug_1",
                    "mug",
                    Pose3D(0.2, 0.0, 0.8),
                    lab.BBox3D(
                        center=Pose3D(0.2, 0.0, 0.8),
                        size=(0.2, 0.2, 0.3),
                    ),
                    confidence=0.86,
                    visible=True,
                    attributes={
                        "source_kind": "ai2thor_metadata_coverage",
                        "current_location_id": "counter_region",
                        "current_location_relation": "IN_REGION",
                    },
                ),
            ),
        ),
    )

    with pytest.raises(
        lab.SpatialQAError,
        match="current_location requires detector source_kind",
    ):
        lab.build_predicted_graph_from_observations(
            observations,
            infer_relations=(),
            infer_containment=False,
        )


def test_predicted_graph_observation_report_records_relation_top_k(
    tmp_path: Path,
) -> None:
    observations = (
        lab.SceneObservation(
            step=1,
            objects=tuple(
                lab.ObjectObservation(
                    f"obj_{index}",
                    "block",
                    Pose3D(index * 0.1, 0.0, 0.0),
                    lab.BBox3D(
                        center=Pose3D(index * 0.1, 0.0, 0.0),
                        size=(0.05, 0.05, 0.05),
                    ),
                    confidence=0.9,
                    visible=True,
                    attributes={"source": "rgbd_detector"},
                )
                for index in range(4)
            ),
        ),
    )
    input_path = tmp_path / "detector-observations.json"
    graph_path = tmp_path / "observation-predicted-graph.json"
    lab.save_scene_observation_sequence(observations, input_path)

    graph = lab.build_predicted_graph_from_observations(
        observations,
        source_path=input_path,
        infer_relations=("NEAR",),
        reference_frames=("world",),
        relation_top_k=1,
    )
    lab.save_graph_json(graph, graph_path)
    report = lab.predicted_graph_report_from_observations(
        input_path=input_path,
        graph_path=graph_path,
        graph=graph,
        observations=observations,
        infer_relations=("NEAR",),
        reference_frames=("world",),
        relation_top_k=1,
    )
    validation = lab.validate_predicted_graph_report(report)
    comparison = lab.compare_predicted_graph_report(report)

    assert len([edge for edge in graph.edges if edge.relation == "NEAR"]) == 4
    assert report["options"]["relation_top_k"] == 1
    assert validation["valid"] is True
    assert comparison["matches"] is True


def test_predicted_graph_observation_report_records_state_evidence_requirement(
    tmp_path: Path,
) -> None:
    observations = (
        lab.SceneObservation(
            step=1,
            objects=(
                lab.ObjectObservation(
                    "mug_1",
                    "mug",
                    Pose3D(0.2, 0.0, 0.8),
                    lab.BBox3D(
                        center=Pose3D(0.2, 0.0, 0.8),
                        size=(0.2, 0.2, 0.3),
                    ),
                    confidence=0.86,
                    visible=True,
                    attributes={
                        "source_kind": "detector",
                        "evidence_kinds": ("rgb", "depth", "detector"),
                        "rgb_path": "rgb/0001.png",
                        "depth_path": "depth/0001.npy",
                        "states": {"isOpen": False},
                    },
                ),
            ),
        ),
    )
    input_path = tmp_path / "detector-observations.json"
    graph_path = tmp_path / "observation-predicted-graph.json"
    lab.save_scene_observation_sequence(observations, input_path)

    graph = lab.build_predicted_graph_from_observations(
        observations,
        source_path=input_path,
        infer_relations=(),
        require_detector_state_evidence=True,
    )
    lab.save_graph_json(graph, graph_path)
    report = lab.predicted_graph_report_from_observations(
        input_path=input_path,
        graph_path=graph_path,
        graph=graph,
        observations=observations,
        infer_relations=(),
        require_detector_state_evidence=True,
    )

    assert report["options"]["require_detector_state_evidence"] is True
    assert lab.validate_predicted_graph_report(report)["valid"] is True
    assert lab.compare_predicted_graph_report(report)["matches"] is True


def test_predicted_graph_report_save_load_validate_and_compare_explicit_artifacts(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "predicted_graph_report")
    assert hasattr(lab, "predicted_graph_report_digest")
    assert hasattr(lab, "predicted_graph_report_json")
    assert hasattr(lab, "save_predicted_graph_report")
    assert hasattr(lab, "load_predicted_graph_report")
    assert hasattr(lab, "validate_predicted_graph_report")
    assert hasattr(lab, "compare_predicted_graph_report")
    frames = _predicted_frames()
    input_path = tmp_path / "episode.jsonl"
    graph_path = tmp_path / "predicted-graph.json"
    report_path = tmp_path / "predicted-report.json"
    lab.save_episode_sequence(frames, input_path)
    graph = lab.build_predicted_graph_from_episode(frames)
    lab.save_graph_json(graph, graph_path)

    report = lab.predicted_graph_report(
        input_path=input_path,
        graph_path=graph_path,
        graph=graph,
        frames=frames,
    )
    saved_path = lab.save_predicted_graph_report(report, report_path)
    loaded_report = lab.load_predicted_graph_report(report_path)
    validation = lab.validate_predicted_graph_report(loaded_report)
    comparison = lab.compare_predicted_graph_report(loaded_report)

    assert saved_path == report_path
    assert json.loads(lab.predicted_graph_report_json(report)) == report
    assert report["schema_version"] == "dsg-spatialqa-lab.predicted-graph-report.v1"
    assert report["digest"] == lab.predicted_graph_report_digest(report)
    assert report["summary"]["predicted_summary"] == {
        "detection_count": 3,
        "frame_count": 2,
        "hidden_update_count": 1,
        "instance_count": 3,
        "inferred_relation_count": 4,
        "unique_instance_count": 2,
        "by_detection_source": {"mock_perception": 3},
        "by_detection_label": {"mug": 2, "plate": 1},
    }
    assert validation["valid"] is True
    assert comparison["matches"] is True

    tampered_report = json.loads(lab.predicted_graph_report_json(report))
    tampered_report["summary"]["predicted_summary"]["detection_count"] = 99
    tampered_report["digest"] = lab.predicted_graph_report_digest(tampered_report)
    tampered_comparison = lab.compare_predicted_graph_report(tampered_report)
    checks = {check["name"]: check for check in tampered_comparison["checks"]}
    assert tampered_comparison["matches"] is False
    assert checks["predicted_summary_matches_current"]["passed"] is False


def test_predicted_graph_cli_builds_validates_and_compares_artifacts(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_predicted_script()
    main = cast(MainFn, getattr(module, "main"))
    frames = _predicted_frames()
    input_path = tmp_path / "episode.jsonl"
    graph_path = tmp_path / "predicted-graph.json"
    report_path = tmp_path / "predicted-report.json"
    lab.save_episode_sequence(frames, input_path)

    assert main(
        [
            "--mock",
            "--input",
            str(input_path),
            "--output-graph",
            str(graph_path),
            "--report",
            str(report_path),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["action"] == "build_predicted_graph"
    assert output["path"] == str(input_path)
    assert output["graph_path"] == str(graph_path)
    assert output["valid"] is True
    assert graph_path.exists()
    assert report_path.exists()

    assert main(["--validate-report", str(report_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_predicted_graph_report"
    assert validation["path"] == str(report_path)
    assert validation["valid"] is True

    assert main(["--compare-report", str(report_path)]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["action"] == "compare_predicted_graph_report"
    assert comparison["path"] == str(report_path)
    assert comparison["matches"] is True


def test_predicted_graph_cli_builds_from_observation_sequence(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_predicted_script()
    main = cast(MainFn, getattr(module, "main"))
    observations = _detector_observations()
    input_path = tmp_path / "detector-observations.json"
    graph_path = tmp_path / "observation-predicted-graph.json"
    report_path = tmp_path / "observation-predicted-report.json"
    lab.save_scene_observation_sequence(observations, input_path)

    assert main(
        [
            "--input-kind",
            "observation_sequence",
            "--input",
            str(input_path),
            "--output-graph",
            str(graph_path),
            "--report",
            str(report_path),
            "--infer-relation",
            "LEFT_OF",
            "--infer-relation",
            "RIGHT_OF",
            "--infer-relation",
            "NEAR",
            "--reference-frame",
            "world",
            "--infer-containment",
            "--containment-axis",
            "z",
            "--relation-top-k",
            "1",
            "--require-detector-state-evidence",
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["action"] == "build_predicted_graph"
    assert output["input_kind"] == "observation_sequence"
    assert output["path"] == str(input_path)
    assert output["graph_path"] == str(graph_path)
    assert output["valid"] is True
    assert output["summary"]["predicted_summary"]["by_observation_source"] == {
        "rgbd_detector": 2,
        "rgbd_tracker": 1,
    }
    assert output["options"]["infer_containment"] is True
    assert output["options"]["containment_axis"] == "z"
    assert output["options"]["relation_top_k"] == 1
    assert output["options"]["require_detector_state_evidence"] is True
    assert graph_path.exists()
    assert report_path.exists()

    assert main(["--validate-report", str(report_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_predicted_graph_report"
    assert validation["valid"] is True

    assert main(["--compare-report", str(report_path)]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["action"] == "compare_predicted_graph_report"
    assert comparison["matches"] is True


def test_predicted_graph_cli_returns_structured_json_for_invalid_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_predicted_script()
    main = cast(MainFn, getattr(module, "main"))
    report_path = tmp_path / "invalid-report.json"
    report_path.write_text("[]\n", encoding="utf-8")

    assert main(["--validate-report", str(report_path)]) == 1

    validation = json.loads(capsys.readouterr().out)
    assert validation == {
        "action": "validate_predicted_graph_report",
        "path": str(report_path),
        "valid": False,
        "error": "Predicted graph report JSON must be an object",
    }


def _predicted_frames() -> tuple[EpisodeFrame, ...]:
    return (
        _frame(
            1,
            detections=[
                _detection_payload(
                    "mug_det_1",
                    "mug_1",
                    "mug",
                    pose={"x": 0.3, "y": 1.0, "z": 0.82, "yaw": 0.0},
                    bbox_size=(0.2, 0.2, 0.3),
                ),
                _detection_payload(
                    "plate_det_1",
                    "plate_1",
                    "plate",
                    pose={"x": 0.0, "y": 1.0, "z": 0.74, "yaw": 0.0},
                    bbox_size=(0.3, 0.3, 0.04),
                    confidence=0.88,
                ),
            ],
        ),
        _frame(
            2,
            agent_pose=Pose3D(0.1, 0.0, 0.0),
            detections=[
                _detection_payload(
                    "mug_det_2",
                    "mug_1",
                    "mug",
                    pose={"x": 0.35, "y": 1.0, "z": 0.82, "yaw": 0.0},
                    bbox_size=(0.2, 0.2, 0.3),
                ),
            ],
        ),
    )


def _detector_observations() -> tuple[lab.SceneObservation, ...]:
    return (
        lab.SceneObservation(
            step=1,
            agent_pose=Pose3D(0.0, 0.0, 0.0),
            objects=(
                lab.ObjectObservation(
                    "mug_1",
                    "mug",
                    Pose3D(0.3, 1.0, 0.82),
                    lab.BBox3D(
                        center=Pose3D(0.3, 1.0, 0.82),
                        size=(0.2, 0.2, 0.3),
                    ),
                    confidence=0.91,
                    visible=True,
                    attributes={
                        "source": "rgbd_detector",
                        "detector": "detic_fixture",
                        "rgb_path": "frames/000001.rgb.png",
                        "depth_path": "frames/000001.depth.png",
                    },
                ),
                lab.ObjectObservation(
                    "plate_1",
                    "plate",
                    Pose3D(0.0, 1.0, 0.74),
                    lab.BBox3D(
                        center=Pose3D(0.0, 1.0, 0.74),
                        size=(0.3, 0.3, 0.04),
                    ),
                    confidence=0.87,
                    visible=True,
                    attributes={
                        "source": "rgbd_detector",
                        "detector": "detic_fixture",
                        "rgb_path": "frames/000001.rgb.png",
                        "depth_path": "frames/000001.depth.png",
                    },
                ),
            ),
        ),
        lab.SceneObservation(
            step=2,
            agent_pose=Pose3D(0.1, 0.0, 0.0),
            objects=(
                lab.ObjectObservation(
                    "mug_1",
                    "mug",
                    Pose3D(0.32, 1.0, 0.82),
                    lab.BBox3D(
                        center=Pose3D(0.32, 1.0, 0.82),
                        size=(0.2, 0.2, 0.3),
                    ),
                    confidence=0.35,
                    visible=False,
                    attributes={
                        "source": "rgbd_tracker",
                        "detector": "detic_fixture",
                        "hidden_reason": "not_detected_in_frame",
                    },
                ),
            ),
        ),
    )


def _containment_observations() -> tuple[lab.SceneObservation, ...]:
    return (
        lab.SceneObservation(
            step=1,
            rooms=(
                lab.NodeObservation(
                    "kitchen",
                    "Kitchen",
                    attributes={"source": "test_fixture"},
                ),
            ),
            regions=(
                lab.NodeObservation(
                    "visible_region",
                    "Visible region",
                    attributes={"room_id": "kitchen", "source": "test_fixture"},
                ),
            ),
            objects=(
                lab.ObjectObservation(
                    "table_1",
                    "table",
                    Pose3D(0.0, 0.0, 0.5),
                    lab.BBox3D(
                        center=Pose3D(0.0, 0.0, 0.5),
                        size=(1.0, 1.0, 1.0),
                    ),
                    confidence=0.9,
                    visible=True,
                    attributes={"source": "rgbd_detector"},
                ),
                lab.ObjectObservation(
                    "mug_1",
                    "mug",
                    Pose3D(0.0, 0.0, 1.15),
                    lab.BBox3D(
                        center=Pose3D(0.0, 0.0, 1.15),
                        size=(0.2, 0.2, 0.3),
                    ),
                    confidence=0.88,
                    visible=True,
                    attributes={"source": "rgbd_detector", "pickupable": True},
                ),
            ),
        ),
    )


def _oracle_graph_for_predicted_frames() -> lab.DynamicSceneGraph:
    graph = lab.DynamicSceneGraph()
    graph.set_agent_pose("agent", Pose3D(0.0, 0.0, 0.0), step=1)
    graph.upsert_object(
        "mug_1",
        "mug",
        Pose3D(0.3, 1.0, 0.82),
        lab.BBox3D(center=Pose3D(0.3, 1.0, 0.82), size=(0.2, 0.2, 0.3)),
        confidence=0.9,
        visible=True,
        step=1,
    )
    graph.upsert_object(
        "plate_1",
        "plate",
        Pose3D(0.0, 1.0, 0.74),
        lab.BBox3D(center=Pose3D(0.0, 1.0, 0.74), size=(0.3, 0.3, 0.04)),
        confidence=0.88,
        visible=True,
        step=1,
    )
    lab.GraphTool(graph).update_spatial_relations(
        step=1,
        object_ids=("mug_1", "plate_1"),
        relations=("LEFT_OF", "RIGHT_OF", "NEAR"),
        reference_frames=("agent",),
        evidence=("episode:episode_001:1",),
    )
    graph.set_agent_pose("agent", Pose3D(0.1, 0.0, 0.0), step=2)
    graph.upsert_object(
        "mug_1",
        "mug",
        Pose3D(0.35, 1.0, 0.82),
        lab.BBox3D(center=Pose3D(0.35, 1.0, 0.82), size=(0.2, 0.2, 0.3)),
        confidence=0.9,
        visible=True,
        step=2,
    )
    graph.upsert_object(
        "plate_1",
        "plate",
        Pose3D(0.0, 1.0, 0.74),
        lab.BBox3D(center=Pose3D(0.0, 1.0, 0.74), size=(0.3, 0.3, 0.04)),
        confidence=0.2,
        visible=False,
        step=2,
    )
    return graph


def _oracle_graph_for_source_frames() -> lab.DynamicSceneGraph:
    graph = lab.DynamicSceneGraph()
    graph.set_agent_pose("agent", Pose3D(0.0, 0.0, 0.0), step=1)
    graph.upsert_object(
        "mug_1",
        "mug",
        Pose3D(0.3, 1.0, 0.82),
        lab.BBox3D(center=Pose3D(0.3, 1.0, 0.82), size=(0.2, 0.2, 0.3)),
        confidence=0.9,
        visible=True,
        step=1,
    )
    graph.upsert_object(
        "plate_1",
        "plate",
        Pose3D(0.0, 1.0, 0.74),
        lab.BBox3D(center=Pose3D(0.0, 1.0, 0.74), size=(0.3, 0.3, 0.04)),
        confidence=0.9,
        visible=True,
        step=1,
    )
    lab.GraphTool(graph).update_spatial_relations(
        step=1,
        object_ids=("mug_1", "plate_1"),
        relations=("LEFT_OF", "RIGHT_OF", "NEAR"),
        reference_frames=("agent",),
        evidence=("episode:episode_001:1",),
    )
    return graph


def _frame(
    step: int,
    *,
    detections: list[dict[str, object]],
    agent_pose: Pose3D | None = None,
) -> EpisodeFrame:
    return EpisodeFrame(
        episode_id="episode_001",
        scene_id="mock_apartment",
        step=step,
        rgb_path=None,
        depth_path=None,
        segmentation_path=None,
        agent_id="agent",
        agent_pose=agent_pose or Pose3D(0.0, 0.0, 0.0),
        action="Look",
        visible_object_ids=tuple(
            str(payload["object_id"]) for payload in detections if payload.get("visible", True) is True
        ),
        metadata={"mock_detections": detections},
    )


def _detection_payload(
    detection_id: str,
    object_id: str,
    label: str,
    *,
    pose: dict[str, float],
    bbox_size: tuple[float, float, float],
    confidence: float = 0.9,
    attributes: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "detection_id": detection_id,
        "object_id": object_id,
        "label": label,
        "bbox_xyxy": [10.0, 20.0, 30.0, 50.0],
        "confidence": confidence,
        "depth": 1.0,
        "visible": True,
        "pose": pose,
        "bbox_size": list(bbox_size),
    }
    if attributes is not None:
        payload["attributes"] = attributes
    return payload
