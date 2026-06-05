from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from _pytest.capture import CaptureFixture

import dsg_spatialqa_lab as lab
from dsg_spatialqa_lab import BBox3D, Pose3D


ROOT = Path(__file__).resolve().parents[1]
EVALUATE_GRAPHS_SCRIPT = ROOT / "scripts" / "evaluate_graphs.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_evaluate_graphs_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_graphs_script", EVALUATE_GRAPHS_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def clone_graph(graph: lab.DynamicSceneGraph) -> lab.DynamicSceneGraph:
    return lab.graph_from_json(lab.graph_to_json(graph))


def degraded_tabletop_graph() -> lab.DynamicSceneGraph:
    graph = clone_graph(lab.load_scene_fixture("tabletop"))
    graph.object_states.pop("plate_1")
    graph.object_state_history.pop("plate_1")
    graph.nodes.pop("plate_1")
    graph.nodes.pop("state:plate_1:1")
    graph.edges = [
        edge
        for edge in graph.edges
        if edge.src != "plate_1"
        and edge.dst != "plate_1"
        and not edge.dst.startswith("state:plate_1:")
    ]
    graph.upsert_object(
        "fork_1",
        "fork",
        Pose3D(0.8, 1.0, 0.75),
        BBox3D(center=Pose3D(0.8, 1.0, 0.75), size=(0.2, 0.05, 0.04)),
        confidence=0.7,
        visible=True,
        step=1,
    )
    graph.add_edge("mug_1", "RIGHT_OF", "table_1", "agent", 0.6, step=1)
    return graph


def renamed_tabletop_graph() -> lab.DynamicSceneGraph:
    oracle = lab.load_scene_fixture("tabletop")
    renamed = lab.DynamicSceneGraph()
    id_map = {
        "mug_1": "track_mug_a",
        "plate_1": "track_plate_a",
        "table_1": "track_table_a",
    }
    renamed.set_agent_pose("agent", oracle.get_agent_pose("agent"), step=1)
    for oracle_id, predicted_id in id_map.items():
        state = oracle.get_object_state(oracle_id)
        renamed.upsert_object(
            predicted_id,
            state.label,
            state.pose,
            state.bbox,
            confidence=state.confidence,
            visible=state.visible,
            step=state.step,
        )
    for edge in oracle.edges:
        if edge.src == "agent" and edge.dst == "state:agent:1":
            continue
        renamed.add_edge(
            _renamed_node_id(edge.src, id_map),
            edge.relation,
            _renamed_node_id(edge.dst, id_map),
            edge.reference_frame,
            edge.confidence,
            step=edge.step,
            evidence=edge.evidence,
            attributes=edge.attributes,
        )
    return renamed


def duplicated_track_tabletop_graph() -> lab.DynamicSceneGraph:
    oracle = lab.load_scene_fixture("tabletop")
    duplicated = renamed_tabletop_graph()
    mug_state = oracle.get_object_state("mug_1")
    duplicated.upsert_object(
        "track_mug_b",
        mug_state.label,
        mug_state.pose,
        mug_state.bbox,
        confidence=mug_state.confidence,
        visible=mug_state.visible,
        step=mug_state.step,
    )
    return duplicated


def sourced_degraded_tabletop_graph() -> lab.DynamicSceneGraph:
    graph = degraded_tabletop_graph()
    graph.nodes["mug_1"].attributes["source"] = "mock_segmenter"
    graph.nodes["table_1"].attributes["source"] = "mock_segmenter"
    graph.nodes["fork_1"].attributes["source"] = "vlm_import"
    for edge in graph.edges:
        if edge.src == "fork_1" or edge.relation == "RIGHT_OF":
            edge.attributes["source"] = "vlm_import"
        else:
            edge.attributes["source"] = "geometry"
    return graph


def _renamed_node_id(node_id: str, id_map: dict[str, str]) -> str:
    if node_id in id_map:
        return id_map[node_id]
    for oracle_id, predicted_id in id_map.items():
        prefix = f"state:{oracle_id}:"
        if node_id.startswith(prefix):
            return f"state:{predicted_id}:{node_id.removeprefix(prefix)}"
    return node_id


def test_compare_graphs_scores_perfect_and_degraded_graphs() -> None:
    assert hasattr(lab, "compare_graphs")
    oracle = lab.load_scene_fixture("tabletop")
    perfect = clone_graph(oracle)
    degraded = degraded_tabletop_graph()

    perfect_comparison = lab.compare_graphs(oracle, perfect)
    degraded_comparison = lab.compare_graphs(oracle, degraded)

    assert perfect_comparison["summary"] == {
        "oracle_object_count": 3,
        "predicted_object_count": 3,
        "matched_object_count": 3,
        "predicted_unlocated_object_count": 2,
        "oracle_relation_count": 8,
        "predicted_relation_count": 8,
        "matched_relation_count": 8,
    }
    assert perfect_comparison["metrics"] == {
        "bbox_center_error": {"average": 0.0, "count": 3, "total": 0.0},
        "object_confidence_weighted_f1": {"rate": 1.0},
        "object_confidence_weighted_precision": {
            "matched_weight": 2.85,
            "rate": 1.0,
            "total_weight": 2.85,
        },
        "object_confidence_weighted_recall": {
            "matched_weight": 2.85,
            "rate": 1.0,
            "total_weight": 2.85,
        },
        "object_label_accuracy": {"count": 3, "rate": 1.0, "total": 3},
        "object_precision": {"count": 3, "rate": 1.0, "total": 3},
        "object_recall": {"count": 3, "rate": 1.0, "total": 3},
        "unlocated_object_count": {"count": 2, "total": 3},
        "relation_confidence_weighted_f1": {"rate": 1.0},
        "relation_confidence_weighted_precision": {
            "matched_weight": 7.85,
            "rate": 1.0,
            "total_weight": 7.85,
        },
        "relation_confidence_weighted_recall": {
            "matched_weight": 7.85,
            "rate": 1.0,
            "total_weight": 7.85,
        },
        "relation_f1": {"rate": 1.0},
        "relation_precision": {"count": 8, "rate": 1.0, "total": 8},
        "relation_recall": {"count": 8, "rate": 1.0, "total": 8},
        "state_accuracy": {"count": 3, "mode": "matched_object_state", "rate": 1.0, "total": 3},
    }

    assert degraded_comparison["summary"] == {
        "oracle_object_count": 3,
        "predicted_object_count": 3,
        "matched_object_count": 2,
        "predicted_unlocated_object_count": 2,
        "oracle_relation_count": 8,
        "predicted_relation_count": 6,
        "matched_relation_count": 4,
    }
    assert degraded_comparison["metrics"]["object_precision"]["rate"] == 0.666667
    assert degraded_comparison["metrics"]["object_recall"]["rate"] == 0.666667
    assert degraded_comparison["metrics"]["unlocated_object_count"] == {
        "count": 2,
        "total": 3,
    }
    assert degraded_comparison["metrics"]["relation_precision"]["rate"] == 0.666667
    assert degraded_comparison["metrics"]["relation_recall"]["rate"] == 0.5
    assert degraded_comparison["metrics"]["relation_f1"]["rate"] == 0.571429
    assert degraded_comparison["differences"]["missing_object_ids"] == ["plate_1"]
    assert degraded_comparison["differences"]["extra_object_ids"] == ["fork_1"]
    assert degraded_comparison["breakdown"]["by_object_label"]["plate"] == {
        "matched_count": 0,
        "oracle_count": 1,
        "predicted_count": 0,
        "precision": 0.0,
        "recall": 0.0,
    }
    assert degraded_comparison["breakdown"]["by_object_label"]["fork"] == {
        "matched_count": 0,
        "oracle_count": 0,
        "predicted_count": 1,
        "precision": 0.0,
        "recall": 0.0,
    }
    assert degraded_comparison["breakdown"]["by_relation"]["RIGHT_OF"] == {
        "f1": 0.0,
        "matched_count": 0,
        "oracle_count": 1,
        "predicted_count": 1,
        "precision": 0.0,
        "recall": 0.0,
    }


def test_compare_graphs_reports_confidence_weighted_metrics() -> None:
    comparison = lab.compare_graphs(
        lab.load_scene_fixture("tabletop"),
        degraded_tabletop_graph(),
    )

    assert comparison["metrics"]["object_confidence_weighted_precision"] == {
        "matched_weight": 1.95,
        "rate": 0.735849,
        "total_weight": 2.65,
    }
    assert comparison["metrics"]["object_confidence_weighted_recall"] == {
        "matched_weight": 1.95,
        "rate": 0.684211,
        "total_weight": 2.85,
    }
    assert comparison["metrics"]["object_confidence_weighted_f1"] == {
        "rate": 0.709091,
    }
    assert comparison["metrics"]["relation_confidence_weighted_precision"] == {
        "matched_weight": 3.95,
        "rate": 0.752381,
        "total_weight": 5.25,
    }
    assert comparison["metrics"]["relation_confidence_weighted_recall"] == {
        "matched_weight": 3.95,
        "rate": 0.503185,
        "total_weight": 7.85,
    }
    assert comparison["metrics"]["relation_confidence_weighted_f1"] == {
        "rate": 0.603054,
    }


def test_compare_graphs_reports_prediction_source_confidence_slices() -> None:
    comparison = lab.compare_graphs(
        lab.load_scene_fixture("tabletop"),
        sourced_degraded_tabletop_graph(),
    )

    assert comparison["breakdown"]["by_prediction_source"] == {
        "objects": {
            "mock_segmenter": {
                "confidence_weighted_precision": 1.0,
                "matched_count": 2,
                "matched_weight": 1.95,
                "precision": 1.0,
                "predicted_count": 2,
                "total_weight": 1.95,
            },
            "vlm_import": {
                "confidence_weighted_precision": 0.0,
                "matched_count": 0,
                "matched_weight": 0.0,
                "precision": 0.0,
                "predicted_count": 1,
                "total_weight": 0.7,
            },
        },
        "relations": {
            "geometry": {
                "confidence_weighted_precision": 1.0,
                "matched_count": 4,
                "matched_weight": 3.95,
                "precision": 1.0,
                "predicted_count": 4,
                "total_weight": 3.95,
            },
            "vlm_import": {
                "confidence_weighted_precision": 0.0,
                "matched_count": 0,
                "matched_weight": 0.0,
                "precision": 0.0,
                "predicted_count": 2,
                "total_weight": 1.3,
            },
        },
    }


def test_compare_graphs_label_center_matching_handles_changed_object_ids() -> None:
    oracle = lab.load_scene_fixture("tabletop")
    predicted = renamed_tabletop_graph()

    exact_comparison = lab.compare_graphs(oracle, predicted)
    label_center_comparison = lab.compare_graphs(
        oracle,
        predicted,
        matching="label_center",
        center_distance_threshold=0.05,
    )

    assert exact_comparison["summary"]["matched_object_count"] == 0
    assert exact_comparison["summary"]["matched_relation_count"] == 1
    assert label_center_comparison["matching"] == {
        "center_distance_threshold": 0.05,
        "strategy": "label_center",
    }
    assert label_center_comparison["summary"] == {
        "oracle_object_count": 3,
        "predicted_object_count": 3,
        "matched_object_count": 3,
        "predicted_unlocated_object_count": 2,
        "oracle_relation_count": 8,
        "predicted_relation_count": 8,
        "matched_relation_count": 8,
    }
    assert label_center_comparison["metrics"]["object_precision"]["rate"] == 1.0
    assert label_center_comparison["metrics"]["object_recall"]["rate"] == 1.0
    assert label_center_comparison["metrics"]["relation_f1"]["rate"] == 1.0
    assert label_center_comparison["metrics"]["bbox_center_error"] == {
        "average": 0.0,
        "count": 3,
        "total": 0.0,
    }
    assert label_center_comparison["differences"]["object_matches"] == [
        {
            "center_distance": 0.0,
            "label": "mug",
            "oracle_object_id": "mug_1",
            "predicted_object_id": "track_mug_a",
        },
        {
            "center_distance": 0.0,
            "label": "plate",
            "oracle_object_id": "plate_1",
            "predicted_object_id": "track_plate_a",
        },
        {
            "center_distance": 0.0,
            "label": "table",
            "oracle_object_id": "table_1",
            "predicted_object_id": "track_table_a",
        },
    ]


def test_compare_graphs_label_center_room_requires_matching_current_room() -> None:
    oracle = lab.load_scene_fixture("multi_room_rearrangement")
    predicted = clone_graph(oracle)
    cereal_state = oracle.get_object_state("cereal_box_1")
    predicted.object_states.pop("cereal_box_1")
    predicted.object_state_history.pop("cereal_box_1")
    for node_id in list(predicted.nodes):
        if node_id == "cereal_box_1" or node_id.startswith("state:cereal_box_1:"):
            predicted.nodes.pop(node_id)
    predicted.edges = [
        edge
        for edge in predicted.edges
        if edge.src != "cereal_box_1"
        and edge.dst != "cereal_box_1"
        and not edge.src.startswith("state:cereal_box_1:")
        and not edge.dst.startswith("state:cereal_box_1:")
    ]
    predicted.upsert_object(
        "track_cereal_wrong_room",
        cereal_state.label,
        cereal_state.pose,
        cereal_state.bbox,
        confidence=cereal_state.confidence,
        visible=cereal_state.visible,
        step=cereal_state.step,
    )
    predicted.add_edge(
        "track_cereal_wrong_room",
        "IN_REGION",
        "prep_counter",
        "world",
        0.92,
        step=2,
    )

    label_center_comparison = lab.compare_graphs(
        oracle,
        predicted,
        matching="label_center",
        center_distance_threshold=0.05,
    )
    room_comparison = lab.compare_graphs(
        oracle,
        predicted,
        matching="label_center_room",
        center_distance_threshold=0.05,
    )

    assert label_center_comparison["summary"]["matched_object_count"] == 3
    assert room_comparison["matching"] == {
        "center_distance_threshold": 0.05,
        "strategy": "label_center_room",
    }
    assert room_comparison["summary"]["matched_object_count"] == 2
    assert room_comparison["differences"]["missing_object_ids"] == ["cereal_box_1"]
    assert room_comparison["differences"]["extra_object_ids"] == [
        "track_cereal_wrong_room"
    ]


def test_compare_graphs_label_center_reports_duplicate_track_diagnostics() -> None:
    oracle = lab.load_scene_fixture("tabletop")
    predicted = duplicated_track_tabletop_graph()

    comparison = lab.compare_graphs(
        oracle,
        predicted,
        matching="label_center",
        center_distance_threshold=0.05,
    )

    assert comparison["summary"]["predicted_object_count"] == 4
    assert comparison["summary"]["matched_object_count"] == 3
    assert comparison["metrics"]["object_precision"] == {
        "count": 3,
        "rate": 0.75,
        "total": 4,
    }
    assert comparison["metrics"]["object_recall"]["rate"] == 1.0
    assert comparison["diagnostics"] == {
        "duplicate_track_count": 1,
        "duplicate_tracks": [
            {
                "center_distance": 0.0,
                "label": "mug",
                "matched_predicted_object_id": "track_mug_a",
                "oracle_object_id": "mug_1",
                "predicted_object_id": "track_mug_b",
            },
        ],
        "id_fragmentation": [
            {
                "candidate_count": 2,
                "candidate_predicted_object_ids": ["track_mug_a", "track_mug_b"],
                "extra_predicted_object_ids": ["track_mug_b"],
                "label": "mug",
                "matched_predicted_object_id": "track_mug_a",
                "oracle_object_id": "mug_1",
            },
        ],
        "id_fragmentation_count": 1,
    }
    assert comparison["differences"]["extra_object_ids"] == ["track_mug_b"]


def test_graph_eval_report_digest_validation_and_comparison(tmp_path: Path) -> None:
    assert hasattr(lab, "graph_eval_report")
    assert hasattr(lab, "graph_eval_report_digest")
    assert hasattr(lab, "graph_eval_report_json")
    assert hasattr(lab, "save_graph_eval_report")
    assert hasattr(lab, "load_graph_eval_report")
    assert hasattr(lab, "validate_graph_eval_report")
    assert hasattr(lab, "compare_graph_eval_report")
    oracle = lab.load_scene_fixture("tabletop")
    predicted = degraded_tabletop_graph()
    oracle_path = tmp_path / "oracle.json"
    predicted_path = tmp_path / "predicted.json"
    report_path = tmp_path / "graph-eval-report.json"
    lab.save_graph_json(oracle, oracle_path)
    lab.save_graph_json(predicted, predicted_path)

    report = lab.graph_eval_report(
        oracle,
        predicted,
        oracle_path=oracle_path,
        predicted_path=predicted_path,
    )
    saved_path = lab.save_graph_eval_report(report, report_path)
    loaded_report = lab.load_graph_eval_report(report_path)
    validation = lab.validate_graph_eval_report(loaded_report)
    comparison = lab.compare_graph_eval_report(loaded_report)

    assert report["report_digest"] == lab.graph_eval_report_digest(report)
    assert lab.graph_eval_report_json(report) == lab.graph_eval_report_json(loaded_report)
    assert saved_path == report_path
    assert validation["valid"] is True
    assert comparison["matches"] is True

    tampered_report = dict(loaded_report)
    tampered_report["report_digest"] = "0" * 64
    tampered_validation = lab.validate_graph_eval_report(tampered_report)
    checks = {check["name"]: check for check in tampered_validation["checks"]}
    assert tampered_validation["valid"] is False
    assert checks["report_digest"]["passed"] is False


def test_graph_eval_report_label_center_matching_round_trips(tmp_path: Path) -> None:
    oracle = lab.load_scene_fixture("tabletop")
    predicted = renamed_tabletop_graph()
    oracle_path = tmp_path / "oracle.json"
    predicted_path = tmp_path / "predicted.json"
    lab.save_graph_json(oracle, oracle_path)
    lab.save_graph_json(predicted, predicted_path)

    report = lab.graph_eval_report(
        oracle,
        predicted,
        oracle_path=oracle_path,
        predicted_path=predicted_path,
        matching="label_center",
        center_distance_threshold=0.05,
    )
    validation = lab.validate_graph_eval_report(report)
    comparison = lab.compare_graph_eval_report(report)

    assert report["matching"] == {
        "center_distance_threshold": 0.05,
        "strategy": "label_center",
    }
    assert report["summary"]["matched_object_count"] == 3
    assert report["summary"]["matched_relation_count"] == 8
    assert validation["valid"] is True
    assert comparison["matches"] is True


def test_graph_eval_report_duplicate_track_diagnostics_round_trip(tmp_path: Path) -> None:
    oracle = lab.load_scene_fixture("tabletop")
    predicted = duplicated_track_tabletop_graph()
    oracle_path = tmp_path / "oracle.json"
    predicted_path = tmp_path / "predicted.json"
    lab.save_graph_json(oracle, oracle_path)
    lab.save_graph_json(predicted, predicted_path)

    report = lab.graph_eval_report(
        oracle,
        predicted,
        oracle_path=oracle_path,
        predicted_path=predicted_path,
        matching="label_center",
        center_distance_threshold=0.05,
    )
    validation = lab.validate_graph_eval_report(report)
    comparison = lab.compare_graph_eval_report(report)

    assert report["diagnostics"]["duplicate_track_count"] == 1
    assert report["diagnostics"]["id_fragmentation_count"] == 1
    assert validation["valid"] is True
    assert comparison["matches"] is True

    tampered_report = dict(report)
    tampered_diagnostics = dict(report["diagnostics"])
    tampered_diagnostics["duplicate_track_count"] = 0
    tampered_report["diagnostics"] = tampered_diagnostics
    tampered_report["report_digest"] = lab.graph_eval_report_digest(tampered_report)
    tampered_validation = lab.validate_graph_eval_report(tampered_report)
    checks = {check["name"]: check for check in tampered_validation["checks"]}
    assert tampered_validation["valid"] is False
    assert checks["duplicate_track_count"]["passed"] is False


def test_graph_eval_report_validates_confidence_weighted_f1() -> None:
    report = lab.graph_eval_report(
        lab.load_scene_fixture("tabletop"),
        degraded_tabletop_graph(),
    )

    tampered_report = dict(report)
    tampered_metrics = dict(report["metrics"])
    tampered_metrics["object_confidence_weighted_f1"] = {"rate": 0.0}
    tampered_report["metrics"] = tampered_metrics
    tampered_report["report_digest"] = lab.graph_eval_report_digest(tampered_report)
    tampered_validation = lab.validate_graph_eval_report(tampered_report)
    checks = {check["name"]: check for check in tampered_validation["checks"]}

    assert tampered_validation["valid"] is False
    assert checks["object_confidence_weighted_f1"]["passed"] is False


def test_graph_eval_report_validates_prediction_source_breakdown_counts() -> None:
    report = lab.graph_eval_report(
        lab.load_scene_fixture("tabletop"),
        sourced_degraded_tabletop_graph(),
    )

    tampered_report = dict(report)
    tampered_breakdown = dict(report["breakdown"])
    tampered_by_source = dict(report["breakdown"]["by_prediction_source"])
    tampered_objects = dict(tampered_by_source["objects"])
    tampered_mock_segmenter = dict(tampered_objects["mock_segmenter"])
    tampered_mock_segmenter["predicted_count"] = 1
    tampered_objects["mock_segmenter"] = tampered_mock_segmenter
    tampered_by_source["objects"] = tampered_objects
    tampered_breakdown["by_prediction_source"] = tampered_by_source
    tampered_report["breakdown"] = tampered_breakdown
    tampered_report["report_digest"] = lab.graph_eval_report_digest(tampered_report)
    tampered_validation = lab.validate_graph_eval_report(tampered_report)
    checks = {check["name"]: check for check in tampered_validation["checks"]}

    assert tampered_validation["valid"] is False
    assert checks["prediction_source_object_count"]["passed"] is False


def test_evaluate_graphs_cli_writes_validates_and_compares_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_graphs_script()
    main = cast(MainFn, getattr(module, "main"))
    oracle_path = tmp_path / "oracle.json"
    predicted_path = tmp_path / "predicted.json"
    report_path = tmp_path / "graph-eval-report.json"
    lab.save_graph_json(lab.load_scene_fixture("tabletop"), oracle_path)
    lab.save_graph_json(degraded_tabletop_graph(), predicted_path)

    assert main(
        [
            "--oracle",
            str(oracle_path),
            "--predicted",
            str(predicted_path),
            "--report",
            str(report_path),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    report = lab.load_graph_eval_report(report_path)
    assert output == {
        "action": "graph_eval_report",
        "path": str(report_path),
        "valid": True,
        "digest": report["report_digest"],
        "summary": report["summary"],
        "metrics": report["metrics"],
    }

    assert main(["--validate-report", str(report_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_graph_eval_report"
    assert validation["path"] == str(report_path)
    assert validation["valid"] is True

    assert main(["--compare-report", str(report_path)]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["action"] == "compare_graph_eval_report"
    assert comparison["path"] == str(report_path)
    assert comparison["matches"] is True


def test_evaluate_graphs_cli_accepts_label_center_matching(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_graphs_script()
    main = cast(MainFn, getattr(module, "main"))
    oracle_path = tmp_path / "oracle.json"
    predicted_path = tmp_path / "predicted.json"
    report_path = tmp_path / "graph-eval-report.json"
    lab.save_graph_json(lab.load_scene_fixture("tabletop"), oracle_path)
    lab.save_graph_json(renamed_tabletop_graph(), predicted_path)

    assert main(
        [
            "--oracle",
            str(oracle_path),
            "--predicted",
            str(predicted_path),
            "--matching",
            "label_center",
            "--center-distance-threshold",
            "0.05",
            "--report",
            str(report_path),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    report = lab.load_graph_eval_report(report_path)
    assert output["valid"] is True
    assert output["summary"]["matched_object_count"] == 3
    assert output["summary"]["matched_relation_count"] == 8
    assert report["matching"] == {
        "center_distance_threshold": 0.05,
        "strategy": "label_center",
    }


def test_evaluate_graphs_cli_accepts_label_center_room_matching(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_graphs_script()
    main = cast(MainFn, getattr(module, "main"))
    oracle_path = tmp_path / "oracle.json"
    predicted_path = tmp_path / "predicted.json"
    report_path = tmp_path / "graph-eval-report.json"
    lab.save_graph_json(lab.load_scene_fixture("tabletop"), oracle_path)
    lab.save_graph_json(renamed_tabletop_graph(), predicted_path)

    assert main(
        [
            "--oracle",
            str(oracle_path),
            "--predicted",
            str(predicted_path),
            "--matching",
            "label_center_room",
            "--center-distance-threshold",
            "0.05",
            "--report",
            str(report_path),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    report = lab.load_graph_eval_report(report_path)
    assert output["valid"] is True
    assert report["matching"] == {
        "center_distance_threshold": 0.05,
        "strategy": "label_center_room",
    }
    assert lab.validate_graph_eval_report(report)["valid"] is True


def test_evaluate_graphs_cli_returns_structured_json_for_invalid_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_graphs_script()
    main = cast(MainFn, getattr(module, "main"))
    report_path = tmp_path / "invalid-report.json"
    report_path.write_text("[]\n", encoding="utf-8")

    assert main(["--validate-report", str(report_path)]) == 1

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "action": "validate_graph_eval_report",
        "path": str(report_path),
        "valid": False,
        "error": "Graph eval report JSON must be an object",
    }
