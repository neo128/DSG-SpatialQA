from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from _pytest.capture import CaptureFixture

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
