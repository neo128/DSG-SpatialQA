from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Protocol, cast

from _pytest.capture import CaptureFixture
import dsg_spatialqa_lab as lab


ROOT = Path(__file__).resolve().parents[1]
SERVE_DSG_VIEWER_SCRIPT = ROOT / "scripts" / "serve_dsg_viewer.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_serve_dsg_viewer_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "serve_dsg_viewer_script",
        SERVE_DSG_VIEWER_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _single_object_graph() -> lab.DynamicSceneGraph:
    graph = lab.DynamicSceneGraph()
    graph.set_agent_pose("agent", lab.Pose3D(0.0, 0.0, 0.0), step=1)
    graph.add_room("room_1", "kitchen", step=1)
    graph.upsert_object(
        "mug_1",
        "mug",
        lab.Pose3D(1.0, 0.8, 2.0),
        lab.BBox3D(center=lab.Pose3D(1.0, 0.8, 2.0), size=(0.2, 0.2, 0.2)),
        confidence=0.9,
        visible=True,
        step=1,
        attributes={
            "source_kind": "detector",
            "evidence_kinds": ["rgb", "depth", "detector"],
        },
    )
    graph.add_edge(
        "mug_1",
        "IN_ROOM",
        "room_1",
        "world",
        0.9,
        step=1,
        evidence=["frame:1"],
        attributes={"source": "test"},
    )
    return graph


def _active_qa_case_v2() -> dict[str, object]:
    return {
        "schema_version": "dsg-spatialqa-lab.active-qa-case.v2",
        "id": "episode-1:FloorPlan1:7:object_location:mug_1:observation_aware",
        "scene_id": "FloorPlan1",
        "episode_id": "episode-1",
        "source_graph_digest": "graph-digest",
        "question_text": "Where is the mug?",
        "question_type": "object_location",
        "split": "observation_aware",
        "target": {"object_id": "mug_1", "label": "mug"},
        "answer": {"dst": "room_1", "dst_label": "kitchen", "relation": "IN_ROOM", "step": 7},
        "answer_options": [
            {"dst_label": "kitchen", "relation": "IN_ROOM"},
            {"dst_label": "table", "relation": "ON"},
        ],
        "evidence_frames": [7],
        "required_evidence": {
            "edges": ["mug_1-IN_ROOM-room_1"],
            "frames": [7],
            "nodes": ["mug_1", "room_1"],
            "states": [],
        },
        "situation": {
            "agent_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            "reference_frame": "agent_egocentric",
            "source": "reachable_nbv_observation",
            "step": 7,
            "view_frame": None,
        },
        "observability": {
            "answerable_from_dsg": True,
            "answerable_from_single_frame_vlm": True,
            "evidence_observable": True,
            "support_visible": True,
            "target_visible": True,
        },
        "anti_shortcut": {
            "has_distractor_same_category": False,
            "language_prior_risk": "high",
            "requires_3d_evidence": True,
        },
        "trajectory_context": {
            "collection_kind": "reachable_relation_centric_nbv",
            "navigation_validated": True,
            "real_ai2thor_runtime": True,
        },
    }


def _reachable_nbv_trajectory_report(predicted_graph_path: str) -> dict[str, object]:
    return {
        "schema_version": "dsg-spatialqa-lab.reachable-nbv-trajectory.v1",
        "trajectory_id": "reachable-nbv-episode-1",
        "episode_id": "episode-1",
        "scene_id": "FloorPlan1",
        "collection_kind": "reachable_relation_centric_nbv",
        "runtime_kind": "real_ai2thor",
        "navigation_validated": True,
        "real_ai2thor_runtime": True,
        "predicted_graph_path": predicted_graph_path,
        "decision_trace_path": "outputs/navigation/decision-trace.jsonl",
        "topdown_path_png": "inputs/episodes/topdown-path.png",
        "fixed_vs_nbv_overlay_png": "inputs/episodes/overlay.png",
        "steps": [
            {
                "step_index": 0,
                "navigation_success": True,
                "selected_viewpoint": {"candidate_id": "vp_0", "x": 0.0, "z": 0.0},
                "agent_pose_after": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            },
            {
                "step_index": 1,
                "navigation_success": True,
                "selected_viewpoint": {"candidate_id": "vp_1", "x": 1.0, "z": 0.0},
                "agent_pose_after": {"x": 1.0, "y": 0.0, "z": 0.0, "yaw": 90.0},
            },
        ],
        "stations": [
            {"step_index": 0, "navigation_success": True},
            {"step_index": 1, "navigation_success": True},
        ],
        "rejected_candidates": [],
    }


def test_dsg_viewer_payload_exposes_graph_metrics_and_selection_indexes() -> None:
    assert hasattr(lab, "dsg_viewer_payload")
    graph = _single_object_graph()
    evidence_report = {
        "readiness": {
            "ready": False,
            "failed_check_count": 1,
            "failed_checks": ["non_real_sources_absent"],
        },
        "evidence_summary": {
            "source_counts": {"ai2thor": 1},
            "evidence_kind_counts": {"rgb": 1, "depth": 1, "detector": 1},
        },
        "report_digest": "e" * 64,
    }

    payload = lab.dsg_viewer_payload(
        predicted_graph=graph,
        predicted_graph_path="predicted-graph.json",
        evidence_report=evidence_report,
        evidence_report_path="evidence.json",
    )

    assert payload["schema_version"] == "dsg-spatialqa-lab.dsg-viewer-payload.v1"
    assert payload["artifacts"]["predicted_graph_path"] == "predicted-graph.json"
    assert payload["graph"]["summary"]["object_count"] == 1
    assert payload["graph"]["nodes"][0]["id"] == "agent"
    assert payload["graph"]["edges"][0]["relation"] == "IN_ROOM"
    assert payload["metrics"]["object_count"] == 1
    assert payload["metrics"]["edge_count"] == 3
    assert payload["metrics"]["evidence_ready"] is False
    assert payload["diagnostics"]["failed_checks"] == ["non_real_sources_absent"]
    assert payload["indexes"]["qa_case_ids_by_object_id"] == {}


def test_dsg_viewer_payload_links_qa_cases_and_oracle_delta() -> None:
    graph = _single_object_graph()
    case = lab.QACase(
        id="case-1",
        scene_id="FloorPlan1",
        episode_id="episode-1",
        graph_digest="graph-digest",
        step=1,
        question={"type": "object_location", "object_id": "mug_1"},
        question_type="object_location",
        answer={"object_id": "mug_1"},
        answer_type="object_location",
    )
    qa_eval_report = {
        "cases": [
            {
                "case_id": "case-1",
                "exact_match": False,
                "prediction": {"object_id": "mug_1"},
                "gold": {"object_id": "mug_1"},
            }
        ]
    }
    graph_eval_report = {
        "comparison": {
            "object_matches": [
                {"oracle_object_id": "mug_1", "predicted_object_id": "mug_1"}
            ],
            "missing_relations": [{"src": "mug_1", "relation": "ON", "dst": "table_1"}],
            "extra_relations": [],
        },
        "summary": {
            "object_recall": {"rate": 1.0},
            "relation_precision": {"rate": 0.5},
            "relation_recall": {"rate": 0.25},
            "relation_f1": 0.333333,
        },
    }

    payload = lab.dsg_viewer_payload(
        predicted_graph=graph,
        qa_cases=(case,),
        qa_eval_report=qa_eval_report,
        graph_eval_report=graph_eval_report,
    )

    assert payload["qa"]["cases"][0]["case_id"] == "case-1"
    assert payload["qa"]["cases"][0]["target_object_ids"] == ["mug_1"]
    assert payload["indexes"]["qa_case_ids_by_object_id"] == {"mug_1": ["case-1"]}
    assert payload["oracle"]["object_matches_by_predicted_id"] == {
        "mug_1": {"oracle_object_id": "mug_1", "predicted_object_id": "mug_1"}
    }
    assert payload["oracle"]["missing_relation_count"] == 1
    assert payload["metrics"]["relation_f1"] == 0.333333


def test_dsg_viewer_payload_accepts_active_qa_v2_and_trajectory_report() -> None:
    payload = lab.dsg_viewer_payload(
        predicted_graph=_single_object_graph(),
        qa_cases=(_active_qa_case_v2(),),
        trajectory_report=_reachable_nbv_trajectory_report("predicted-graph.json"),
        trajectory_report_path="trajectory.json",
    )

    case = payload["qa"]["cases"][0]
    assert case["case_id"] == "episode-1:FloorPlan1:7:object_location:mug_1:observation_aware"
    assert case["question_text"] == "Where is the mug?"
    assert case["target_object_ids"] == ["mug_1"]
    assert case["evidence_frames"] == [7]
    assert payload["indexes"]["qa_case_ids_by_object_id"] == {"mug_1": [case["case_id"]]}
    assert payload["artifacts"]["trajectory_report_path"] == "trajectory.json"
    assert payload["trajectory"]["summary"]["step_count"] == 2
    assert payload["trajectory"]["summary"]["station_count"] == 2
    assert payload["metrics"]["qa_case_count"] == 1
    assert payload["metrics"]["trajectory_step_count"] == 2
    assert payload["metrics"]["navigation_validated"] is True


def test_dsg_viewer_workspace_preset_resolves_known_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "ai2thor-real-small"
    (workspace / "outputs" / "predicted-dsg").mkdir(parents=True)
    (workspace / "outputs" / "benchmark" / "graphs").mkdir(parents=True)
    (workspace / "inputs").mkdir()
    (
        workspace
        / "outputs"
        / "offline-controls"
        / "qa-eval-observation-aware-p4-target60"
    ).mkdir(parents=True)
    expected_graph = workspace / "outputs" / "predicted-dsg" / "predicted-graph.json"
    expected_graph.write_text("{}", encoding="utf-8")

    preset = lab.dsg_viewer_workspace_preset(workspace)

    assert preset["workspace_path"] == str(workspace)
    assert preset["paths"]["predicted_graph_path"] == str(expected_graph)


def test_dsg_viewer_workspace_preset_prefers_active_reachable_nbv_artifacts(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ai2thor-real-small"
    episode_id = "ai2thor-real-small-episode-005"
    graph_path = (
        workspace
        / "outputs"
        / "predicted-dsg"
        / "predicted-graph-real-ai2thor-reachable-nbv-episode005.json"
    )
    graph_path.parent.mkdir(parents=True)
    graph_path.write_text("{}", encoding="utf-8")
    trajectory_path = (
        workspace
        / "outputs"
        / "navigation"
        / "reachable-nbv-real-ai2thor-trajectory-episode005.json"
    )
    trajectory_path.parent.mkdir(parents=True)
    trajectory_path.write_text(
        json.dumps(
            _reachable_nbv_trajectory_report(
                "outputs/predicted-dsg/predicted-graph-real-ai2thor-reachable-nbv-episode005.json"
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    qa_dir = workspace / "inputs" / "qa-v2-active" / episode_id
    qa_dir.mkdir(parents=True)
    (qa_dir / "qa-observation-aware.jsonl").write_text(
        json.dumps(_active_qa_case_v2(), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    preset = lab.dsg_viewer_workspace_preset(workspace)

    assert preset["paths"]["predicted_graph_path"] == str(graph_path)
    assert preset["paths"]["trajectory_report_path"] == str(trajectory_path)
    assert preset["paths"]["qa_path"] == str(qa_dir)


def test_dsg_viewer_path_guard_rejects_paths_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    try:
        lab.dsg_viewer_resolve_workspace_path(workspace, outside)
    except lab.SpatialQAError as exc:
        assert "outside workspace" in str(exc)
    else:
        raise AssertionError("expected outside workspace path to be rejected")


def test_serve_dsg_viewer_cli_writes_payload_without_starting_server(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_serve_dsg_viewer_script()
    main = cast(MainFn, getattr(module, "main"))
    graph_path = tmp_path / "predicted-graph.json"
    lab.save_graph_json(_single_object_graph(), graph_path)
    output_path = tmp_path / "payload.json"

    assert main(["--graph", str(graph_path), "--write-payload", str(output_path)]) == 0

    output = json.loads(capsys.readouterr().out)
    payload = lab.load_dsg_viewer_payload(output_path)
    assert output["action"] == "write_dsg_viewer_payload"
    assert output["payload_path"] == str(output_path)
    assert payload["metrics"]["object_count"] == 1


def test_serve_dsg_viewer_cli_writes_latest_active_payload(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_serve_dsg_viewer_script()
    main = cast(MainFn, getattr(module, "main"))
    graph_path = tmp_path / "predicted-graph.json"
    lab.save_graph_json(_single_object_graph(), graph_path)
    qa_dir = tmp_path / "qa-v2-active" / "episode-1"
    qa_dir.mkdir(parents=True)
    (qa_dir / "qa-observation-aware.jsonl").write_text(
        json.dumps(_active_qa_case_v2(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(
        json.dumps(_reachable_nbv_trajectory_report(str(graph_path)), sort_keys=True),
        encoding="utf-8",
    )
    output_path = tmp_path / "payload.json"

    assert (
        main(
            [
                "--graph",
                str(graph_path),
                "--qa",
                str(qa_dir),
                "--trajectory",
                str(trajectory_path),
                "--write-payload",
                str(output_path),
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    payload = lab.load_dsg_viewer_payload(output_path)
    assert output["action"] == "write_dsg_viewer_payload"
    assert payload["qa"]["cases"][0]["question_text"] == "Where is the mug?"
    assert payload["trajectory"]["summary"]["step_count"] == 2


def test_dsg_viewer_static_html_contains_workbench_regions() -> None:
    html = lab.dsg_viewer_html()

    assert "DSG Viewer" in html
    assert 'id="data-rail"' in html
    assert 'id="graph-canvas"' in html
    assert 'id="detail-rail"' in html
    assert "Debug" in html
    assert "Demo" in html
    assert "Analysis" in html
    assert "fetch('payload.json')" in html
