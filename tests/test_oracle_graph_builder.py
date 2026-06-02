from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import dsg_spatialqa_lab as lab
from dsg_spatialqa_lab import EpisodeFrame, Pose3D
from _pytest.capture import CaptureFixture


ROOT = Path(__file__).resolve().parents[1]
ORACLE_SCRIPT = ROOT / "scripts" / "build_oracle_graph.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_oracle_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_oracle_graph_script",
        ORACLE_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_oracle_graph_builder_creates_rooms_regions_objects_and_relations() -> None:
    assert hasattr(lab, "OracleObjectRecord")
    assert hasattr(lab, "build_oracle_graph_from_episode")
    frames = (
        EpisodeFrame(
            episode_id="episode_001",
            scene_id="mock_apartment",
            step=1,
            rgb_path=None,
            depth_path=None,
            segmentation_path=None,
            agent_id="agent",
            agent_pose=Pose3D(0.0, 0.0, 0.0),
            action="Look",
            visible_object_ids=("mug_1", "table_1"),
            metadata={
                "rooms": [
                    {"room_id": "kitchen", "label": "Kitchen"},
                ],
                "regions": [
                    {
                        "region_id": "counter_region",
                        "label": "Counter region",
                        "room_id": "kitchen",
                    },
                ],
                "objects": [
                    _object_payload(
                        "mug_1",
                        "mug",
                        x=0.25,
                        y=1.0,
                        z=0.75,
                        room_id="kitchen",
                        region_id="counter_region",
                        states={"filled": False},
                    ),
                    _object_payload(
                        "table_1",
                        "table",
                        x=0.0,
                        y=1.0,
                        z=0.7,
                        size=(1.0, 0.8, 0.1),
                        room_id="kitchen",
                    ),
                ],
                "relations": [
                    {
                        "src": "mug_1",
                        "relation": "ON",
                        "dst": "table_1",
                        "confidence": 0.97,
                        "reference_frame": "world",
                    },
                ],
            },
        ),
    )

    graph = lab.build_oracle_graph_from_episode(frames)

    assert graph.nodes["kitchen"].type == "room"
    assert graph.nodes["counter_region"].type == "region"
    assert graph.get_agent_pose("agent").to_dict() == {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "yaw": 0.0,
    }
    assert graph.get_object_state("mug_1").label == "mug"
    assert graph.nodes["mug_1"].attributes["states"] == {"filled": False}
    assert [
        (edge.src, edge.relation, edge.dst)
        for edge in graph.find_edges(src="mug_1")
    ] == [
        ("mug_1", "IN_REGION", "counter_region"),
        ("mug_1", "IN_ROOM", "kitchen"),
        ("mug_1", "ON", "table_1"),
        ("mug_1", "STATE_CHANGED", "state:mug_1:1"),
    ]
    assert [
        (edge.src, edge.relation, edge.dst)
        for edge in graph.find_edges(src="counter_region")
    ] == [("counter_region", "IN_ROOM", "kitchen")]


def test_oracle_graph_builder_tracks_moved_and_hidden_low_confidence_objects() -> None:
    frames = (
        EpisodeFrame(
            episode_id="episode_001",
            scene_id="mock_apartment",
            step=1,
            rgb_path=None,
            depth_path=None,
            segmentation_path=None,
            agent_id="agent",
            agent_pose=Pose3D(0.0, 0.0, 0.0),
            action="Look",
            visible_object_ids=("mug_1",),
            metadata={
                "rooms": [
                    {"room_id": "kitchen", "label": "Kitchen"},
                    {"room_id": "pantry", "label": "Pantry"},
                ],
                "objects": [
                    _object_payload(
                        "mug_1",
                        "mug",
                        x=0.0,
                        y=1.0,
                        z=0.75,
                        room_id="kitchen",
                    ),
                ],
            },
        ),
        EpisodeFrame(
            episode_id="episode_001",
            scene_id="mock_apartment",
            step=2,
            rgb_path=None,
            depth_path=None,
            segmentation_path=None,
            agent_id="agent",
            agent_pose=Pose3D(0.2, 0.0, 0.0),
            action="MoveMug",
            visible_object_ids=(),
            metadata={
                "objects": [
                    _object_payload(
                        "mug_1",
                        "mug",
                        x=1.0,
                        y=1.5,
                        z=0.75,
                        confidence=0.25,
                        visible=False,
                        room_id="pantry",
                    ),
                ],
            },
        ),
    )

    graph = lab.build_oracle_graph_from_episode(frames)
    state = graph.get_object_state("mug_1")

    assert state.visible is False
    assert state.confidence == 0.25
    assert state.last_seen_step == 1
    assert state.last_seen_pose == Pose3D(0.0, 1.0, 0.75)
    assert [edge.id for edge in graph.find_edges(src="mug_1", relation="MOVED_FROM")] == [
        "mug_1-MOVED_FROM-kitchen-2",
    ]
    assert [edge.id for edge in graph.find_edges(src="mug_1", relation="MOVED_TO")] == [
        "mug_1-MOVED_TO-pantry-2",
    ]
    assert graph.nodes["event:mug_1:move:2"].type == "event"
    assert graph.nodes["action:episode_001:2"].type == "action"


def test_oracle_graph_report_save_load_validate_and_compare_explicit_artifacts(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "oracle_graph_report")
    assert hasattr(lab, "oracle_graph_report_digest")
    assert hasattr(lab, "oracle_graph_report_json")
    assert hasattr(lab, "save_oracle_graph_report")
    assert hasattr(lab, "load_oracle_graph_report")
    assert hasattr(lab, "validate_oracle_graph_report")
    assert hasattr(lab, "compare_oracle_graph_report")
    frames = _report_frames()
    input_path = tmp_path / "episode.jsonl"
    graph_path = tmp_path / "oracle-graph.json"
    report_path = tmp_path / "oracle-report.json"
    lab.save_episode_sequence(frames, input_path)
    graph = lab.build_oracle_graph_from_episode(frames)
    lab.save_graph_json(graph, graph_path)
    report = lab.oracle_graph_report(
        input_path=input_path,
        graph_path=graph_path,
        graph=graph,
        frames=frames,
    )

    saved_path = lab.save_oracle_graph_report(report, report_path)
    loaded_report = lab.load_oracle_graph_report(report_path)
    validation = lab.validate_oracle_graph_report(loaded_report)
    comparison = lab.compare_oracle_graph_report(loaded_report)

    assert saved_path == report_path
    assert json.loads(lab.oracle_graph_report_json(report)) == report
    assert report["schema_version"] == "dsg-spatialqa-lab.oracle-graph-report.v1"
    assert report["digest"] == lab.oracle_graph_report_digest(report)
    assert report["summary"]["oracle_summary"] == {
        "action_node_count": 1,
        "event_node_count": 0,
        "explicit_relation_count": 1,
        "frame_count": 1,
        "moved_edge_count": 0,
        "object_record_count": 2,
        "region_record_count": 1,
        "room_record_count": 1,
        "state_node_count": 3,
        "unique_object_count": 2,
        "by_object_state_key": {"clean": 1},
    }
    assert validation["valid"] is True
    assert comparison["matches"] is True

    tampered_report = json.loads(lab.oracle_graph_report_json(report))
    tampered_report["summary"]["oracle_summary"]["object_record_count"] = 99
    tampered_report["digest"] = lab.oracle_graph_report_digest(tampered_report)
    tampered_comparison = lab.compare_oracle_graph_report(tampered_report)
    checks = {check["name"]: check for check in tampered_comparison["checks"]}
    assert tampered_comparison["matches"] is False
    assert checks["oracle_summary_matches_current"]["passed"] is False
    assert checks["oracle_summary_matches_current"]["differences"] == [
        {
            "path": "oracle_summary.object_record_count",
            "expected": 99,
            "actual": 2,
        },
    ]


def test_oracle_graph_cli_builds_validates_and_compares_artifacts(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_oracle_script()
    main = cast(MainFn, getattr(module, "main"))
    frames = _report_frames()
    input_path = tmp_path / "episode.jsonl"
    graph_path = tmp_path / "oracle-graph.json"
    report_path = tmp_path / "oracle-report.json"
    lab.save_episode_sequence(frames, input_path)

    assert main(
        [
            "--input",
            str(input_path),
            "--output-graph",
            str(graph_path),
            "--report",
            str(report_path),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["action"] == "build_oracle_graph"
    assert output["path"] == str(input_path)
    assert output["graph_path"] == str(graph_path)
    assert output["valid"] is True
    assert graph_path.exists()
    assert report_path.exists()

    assert main(["--validate-report", str(report_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_oracle_graph_report"
    assert validation["path"] == str(report_path)
    assert validation["valid"] is True

    assert main(["--compare-report", str(report_path)]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["action"] == "compare_oracle_graph_report"
    assert comparison["path"] == str(report_path)
    assert comparison["matches"] is True


def test_oracle_graph_cli_returns_structured_json_for_invalid_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_oracle_script()
    main = cast(MainFn, getattr(module, "main"))
    report_path = tmp_path / "invalid-report.json"
    report_path.write_text("[]\n", encoding="utf-8")

    assert main(["--validate-report", str(report_path)]) == 1

    validation = json.loads(capsys.readouterr().out)
    assert validation == {
        "action": "validate_oracle_graph_report",
        "path": str(report_path),
        "valid": False,
        "error": "Oracle graph report JSON must be an object",
    }


def _report_frames() -> tuple[EpisodeFrame, ...]:
    return (
        EpisodeFrame(
            episode_id="episode_001",
            scene_id="mock_apartment",
            step=1,
            rgb_path=None,
            depth_path=None,
            segmentation_path=None,
            agent_id="agent",
            agent_pose=Pose3D(0.0, 0.0, 0.0),
            action="Look",
            visible_object_ids=("mug_1",),
            metadata={
                "rooms": [{"room_id": "kitchen", "label": "Kitchen"}],
                "regions": [
                    {
                        "region_id": "counter_region",
                        "label": "Counter region",
                        "room_id": "kitchen",
                    }
                ],
                "objects": [
                    _object_payload(
                        "mug_1",
                        "mug",
                        x=0.0,
                        y=1.0,
                        z=0.75,
                        room_id="kitchen",
                        region_id="counter_region",
                        states={"clean": True},
                    ),
                    _object_payload(
                        "plate_1",
                        "plate",
                        x=0.35,
                        y=1.0,
                        z=0.72,
                        room_id="kitchen",
                    ),
                ],
                "relations": [
                    {"src": "mug_1", "relation": "LEFT_OF", "dst": "plate_1"},
                ],
            },
        ),
    )


def _object_payload(
    object_id: str,
    label: str,
    *,
    x: float,
    y: float,
    z: float,
    size: tuple[float, float, float] = (0.1, 0.1, 0.1),
    confidence: float = 0.9,
    visible: bool = True,
    room_id: str | None = None,
    region_id: str | None = None,
    states: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "object_id": object_id,
        "label": label,
        "pose": {"x": x, "y": y, "z": z, "yaw": 0.0},
        "bbox": {
            "center": {"x": x, "y": y, "z": z, "yaw": 0.0},
            "size": list(size),
        },
        "confidence": confidence,
        "visible": visible,
        "states": states or {},
    }
    if room_id is not None:
        payload["room_id"] = room_id
    if region_id is not None:
        payload["region_id"] = region_id
    return payload
