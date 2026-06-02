import hashlib
import json
from pathlib import Path

import pytest

import dsg_spatialqa_lab as lab
from dsg_spatialqa_lab import (
    GraphTool,
    SpatialQAEngine,
    SpatialQAError,
    VLAAnchorPlanner,
    build_multi_room_rearrangement_scene,
    build_relation_shift_scene,
    get_scene_fixture,
    list_scene_fixtures,
    build_tabletop_scene,
    compare_graph_to_fixture,
    load_scene_fixture,
    graph_from_json,
    graph_json_digest,
    graph_summary,
    graph_to_dict,
    graph_to_json,
    load_graph_json,
    save_graph_json,
)


def test_tabletop_scene_fixture_is_deterministic_and_queryable() -> None:
    graph = build_tabletop_scene()
    tool = GraphTool(graph)
    qa = SpatialQAEngine(tool)

    assert sorted(graph.nodes) == [
        "agent",
        "kitchen",
        "mug_1",
        "plate_1",
        "state:agent:1",
        "state:mug_1:1",
        "state:plate_1:1",
        "state:table_1:1",
        "table_1",
    ]
    assert [obj.object_id for obj in tool.find_objects(visible=True)] == [
        "mug_1",
        "plate_1",
        "table_1",
    ]

    response = qa.answer(
        {
            "type": "relative_relation",
            "src": "mug_1",
            "relation": "LEFT_OF",
            "dst": "plate_1",
            "reference_frame": "agent",
        }
    )

    assert response.error is None
    assert response.answer["holds"] is True
    assert response.evidence_edges == ["mug_1-LEFT_OF-plate_1-1"]


def test_graph_json_round_trip_preserves_state_history_and_planner_behavior() -> None:
    original = build_tabletop_scene()
    original_json = graph_to_json(original)

    restored = graph_from_json(original_json)
    restored_json = graph_to_json(restored)
    planner = VLAAnchorPlanner(GraphTool(restored))
    pick_result = planner.plan_pick(target_object="mug_1")

    assert restored_json == original_json
    assert graph_to_dict(restored)["schema_version"] == 1
    assert restored.get_agent_pose("agent").to_dict() == {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "yaw": 0.0,
    }
    assert [state.step for state in restored.get_agent_pose_history("agent")] == [1]
    assert [state.step for state in restored.object_state_history["mug_1"]] == [1]
    assert [edge.id for edge in restored.history("mug_1")] == [
        "mug_1-LEFT_OF-plate_1-1",
        "mug_1-NEAR-plate_1-1",
        "mug_1-ON-table_1-1",
        "mug_1-STATE_CHANGED-state:mug_1:1-1",
        "plate_1-RIGHT_OF-mug_1-1",
    ]
    assert pick_result.status == "ok"
    assert pick_result.command is not None
    assert pick_result.command.target_object == "mug_1"


def test_graph_json_file_round_trip_uses_explicit_path(tmp_path: Path) -> None:
    path = tmp_path / "tabletop_scene.json"

    save_graph_json(build_tabletop_scene(), path)
    restored = load_graph_json(path)

    assert restored.get_object_state("plate_1").pose.to_dict() == {
        "x": 0.35,
        "y": 1.0,
        "z": 0.72,
        "yaw": 0.0,
    }


def test_graph_digest_and_summary_are_deterministic() -> None:
    graph = build_tabletop_scene()
    restored = graph_from_json(graph_to_json(graph))

    assert graph_json_digest(graph) == graph_json_digest(restored)
    assert len(graph_json_digest(graph)) == 64
    assert graph_summary(graph) == {
        "schema_version": 1,
        "node_count": 9,
        "edge_count": 8,
        "object_count": 3,
        "agent_count": 1,
        "object_history_count": 3,
        "agent_history_count": 1,
        "visible_object_count": 3,
        "hidden_object_count": 0,
        "low_confidence_object_count": 0,
        "reobserve_candidate_count": 0,
        "unlocated_object_count": 2,
        "unroomed_object_count": 3,
        "by_node_type": {
            "agent": 1,
            "object": 3,
            "room": 1,
            "state": 4,
        },
        "by_edge_relation": {
            "LEFT_OF": 1,
            "NEAR": 1,
            "ON": 1,
            "RIGHT_OF": 1,
            "STATE_CHANGED": 4,
        },
        "by_object_label": {
            "mug": 1,
            "plate": 1,
            "table": 1,
        },
        "by_current_location": {
            "table_1": 1,
        },
        "by_current_room": {},
    }


def test_graph_report_json_and_save_use_explicit_paths(tmp_path: Path) -> None:
    assert hasattr(lab, "graph_report")
    assert hasattr(lab, "graph_report_json")
    assert hasattr(lab, "graph_report_digest")
    assert hasattr(lab, "save_graph_report")
    graph = build_tabletop_scene()
    graph_path = Path("tabletop-scene.json")

    report = lab.graph_report(
        graph,
        action="export_fixture",
        graph_path=graph_path,
        fixture="tabletop",
    )
    payload = lab.graph_report_json(report)
    repeated_payload = lab.graph_report_json(report)
    report_path = tmp_path / "reports" / "tabletop-report.json"
    saved_path = lab.save_graph_report(
        graph,
        report_path,
        action="export_fixture",
        graph_path=graph_path,
        fixture="tabletop",
    )

    expected_report = {
        "schema_version": "dsg-spatialqa-lab.graph-report.v1",
        "action": "export_fixture",
        "path": str(graph_path),
        "fixture": "tabletop",
        "digest": graph_json_digest(graph),
        "summary": graph_summary(graph),
    }
    assert report == {
        **expected_report,
        "report_digest": lab.graph_report_digest(expected_report),
    }
    assert payload == repeated_payload
    assert payload.endswith("\n")
    assert json.loads(payload) == report
    assert saved_path == report_path
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_graph_report_includes_stable_report_digest_and_validates_tampering() -> None:
    graph = load_scene_fixture("tabletop")
    report = lab.graph_report(
        graph,
        action="export_fixture",
        graph_path="tabletop-scene.json",
        fixture="tabletop",
    )
    report_without_digest = {
        key: value for key, value in report.items() if key != "report_digest"
    }
    expected_report_digest = hashlib.sha256(
        json.dumps(report_without_digest, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()

    assert report["report_digest"] == expected_report_digest
    assert lab.graph_report_digest(report) == expected_report_digest
    validation = lab.validate_graph_report(report)
    checks = {check["name"]: check for check in validation["checks"]}
    assert validation["report_digest"] == expected_report_digest
    assert checks["report_digest"] == {
        "name": "report_digest",
        "passed": True,
        "expected": expected_report_digest,
        "actual": expected_report_digest,
    }

    tampered_report = json.loads(lab.graph_report_json(report))
    tampered_report["summary"]["node_count"] = 10

    tampered_validation = lab.validate_graph_report(tampered_report)
    tampered_checks = {
        check["name"]: check for check in tampered_validation["checks"]
    }
    assert tampered_validation["valid"] is False
    assert tampered_checks["report_digest"] == {
        "name": "report_digest",
        "passed": False,
        "expected": lab.graph_report_digest(tampered_report),
        "actual": expected_report_digest,
    }


def test_graph_report_loads_validates_and_compares_current_fixture(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "load_graph_report")
    assert hasattr(lab, "validate_graph_report")
    assert hasattr(lab, "compare_graph_report")
    graph = load_scene_fixture("tabletop")
    report = lab.graph_report(graph, graph_path="tabletop-scene.json", fixture="tabletop")
    report_path = tmp_path / "tabletop-report.json"
    report_path.write_text(lab.graph_report_json(report), encoding="utf-8")

    loaded_report = lab.load_graph_report(report_path)
    validation = lab.validate_graph_report(loaded_report)
    comparison = lab.compare_graph_report(loaded_report)

    assert loaded_report == report
    assert validation == {
        "valid": True,
        "schema_version": "dsg-spatialqa-lab.graph-report.v1",
        "digest": report["digest"],
        "report_digest": report["report_digest"],
        "checks": [
            {
                "name": "schema_version",
                "passed": True,
                "expected": "dsg-spatialqa-lab.graph-report.v1",
                "actual": "dsg-spatialqa-lab.graph-report.v1",
            },
            {
                "name": "digest_format",
                "passed": True,
                "expected": "64 lowercase sha256 hex characters",
                "actual": report["digest"],
            },
            {
                "name": "report_digest",
                "passed": True,
                "expected": report["report_digest"],
                "actual": report["report_digest"],
            },
            {
                "name": "summary_schema_version",
                "passed": True,
                "expected": 1,
                "actual": 1,
            },
        ],
    }
    assert comparison == {
        "matches": True,
        "fixture": "tabletop",
        "saved_digest": report["digest"],
        "current_digest": report["digest"],
        "checks": [
            {"name": "saved_report_valid", "passed": True},
            {
                "name": "graph_digest_matches_current",
                "passed": True,
                "expected": report["digest"],
                "actual": report["digest"],
            },
            {
                "name": "summary_matches_current",
                "passed": True,
                "expected": graph_summary(graph),
                "actual": graph_summary(graph),
            },
        ],
    }


def test_graph_report_compare_reports_summary_drift() -> None:
    assert hasattr(lab, "compare_graph_report")
    graph = load_scene_fixture("tabletop")
    report = lab.graph_report(graph, graph_path="tabletop-scene.json", fixture="tabletop")
    drifted_report = json.loads(lab.graph_report_json(report))
    drifted_report["summary"]["node_count"] = 10

    comparison = lab.compare_graph_report(drifted_report)

    summary_check = next(
        check for check in comparison["checks"] if check["name"] == "summary_matches_current"
    )
    assert comparison["matches"] is False
    assert comparison["saved_digest"] == report["digest"]
    assert comparison["current_digest"] == report["digest"]
    assert summary_check["passed"] is False
    assert summary_check["differences"] == [
        {"path": "node_count", "expected": 10, "actual": 9},
    ]


def test_graph_report_compares_against_explicit_graph_file(tmp_path: Path) -> None:
    assert hasattr(lab, "compare_graph_report_to_file")
    graph = load_scene_fixture("tabletop")
    graph_path = tmp_path / "tabletop-scene.json"
    save_graph_json(graph, graph_path)
    report = lab.graph_report(graph, graph_path=graph_path, fixture="tabletop")

    comparison = lab.compare_graph_report_to_file(report, graph_path)

    assert comparison == {
        "matches": True,
        "path": str(graph_path),
        "saved_digest": report["digest"],
        "graph_digest": report["digest"],
        "checks": [
            {"name": "saved_report_valid", "passed": True},
            {
                "name": "graph_digest_matches_report",
                "passed": True,
                "expected": report["digest"],
                "actual": report["digest"],
            },
            {
                "name": "summary_matches_report",
                "passed": True,
                "expected": graph_summary(graph),
                "actual": graph_summary(graph),
            },
        ],
    }


def test_graph_report_to_file_compare_reports_graph_json_drift(tmp_path: Path) -> None:
    assert hasattr(lab, "compare_graph_report_to_file")
    graph = load_scene_fixture("tabletop")
    report = lab.graph_report(graph, graph_path="tabletop-scene.json", fixture="tabletop")
    drifted_graph = load_scene_fixture("tabletop")
    drifted_graph.add_region("drift_region", "Drift Region", step=9)
    graph_path = tmp_path / "tabletop-drift.json"
    save_graph_json(drifted_graph, graph_path)

    comparison = lab.compare_graph_report_to_file(report, graph_path)

    summary_check = next(
        check for check in comparison["checks"] if check["name"] == "summary_matches_report"
    )
    assert comparison["matches"] is False
    assert comparison["path"] == str(graph_path)
    assert comparison["saved_digest"] == report["digest"]
    assert comparison["graph_digest"] == graph_json_digest(drifted_graph)
    assert comparison["checks"][1]["passed"] is False
    assert summary_check["passed"] is False
    assert summary_check["differences"] == [
        {"path": "by_node_type.region", "expected": None, "actual": 1},
        {"path": "node_count", "expected": 9, "actual": 10},
    ]


def test_graph_summary_reports_visibility_and_reobserve_counts() -> None:
    summary = graph_summary(load_scene_fixture("needs_reobserve"))

    assert summary["object_count"] == 6
    assert summary["visible_object_count"] == 4
    assert summary["hidden_object_count"] == 2
    assert summary["low_confidence_object_count"] == 2
    assert summary["reobserve_candidate_count"] == 1
    assert summary["by_object_label"] == {
        "bowl": 1,
        "cup": 1,
        "mug": 1,
        "plate": 1,
        "spoon": 1,
        "table": 1,
    }


def test_graph_summary_reports_current_location_counts() -> None:
    summary = graph_summary(load_scene_fixture("multi_room_rearrangement"))

    assert summary["by_current_location"] == {
        "pantry_shelf": 1,
        "prep_counter": 2,
    }
    assert summary["unlocated_object_count"] == 0


def test_graph_summary_reports_current_room_counts() -> None:
    summary = graph_summary(load_scene_fixture("multi_room_rearrangement"))

    assert summary["by_current_room"] == {
        "kitchen": 2,
        "pantry": 1,
    }
    assert summary["unroomed_object_count"] == 0


def test_compare_graph_to_fixture_accepts_current_fixture_graph() -> None:
    graph = load_scene_fixture("tabletop")
    fixture_graph = load_scene_fixture("tabletop")
    fixture_digest = graph_json_digest(fixture_graph)
    fixture_summary = graph_summary(fixture_graph)

    comparison = compare_graph_to_fixture(graph, "tabletop")

    assert comparison == {
        "matches": True,
        "fixture": "tabletop",
        "graph_digest": fixture_digest,
        "fixture_digest": fixture_digest,
        "checks": [
            {
                "name": "graph_digest_matches_fixture",
                "passed": True,
                "expected": fixture_digest,
                "actual": fixture_digest,
            },
            {
                "name": "summary_matches_fixture",
                "passed": True,
                "expected": fixture_summary,
                "actual": fixture_summary,
            },
        ],
    }


def test_compare_graph_to_fixture_reports_drift_from_current_fixture() -> None:
    graph = load_scene_fixture("tabletop")
    graph.add_region("drift_region", "Drift Region", step=9)

    comparison = compare_graph_to_fixture(graph, "tabletop")
    checks = {check["name"]: check for check in comparison["checks"]}

    assert comparison["matches"] is False
    assert comparison["fixture"] == "tabletop"
    assert comparison["graph_digest"] != comparison["fixture_digest"]
    assert checks["graph_digest_matches_fixture"]["passed"] is False
    assert checks["summary_matches_fixture"]["passed"] is False
    assert checks["summary_matches_fixture"]["actual"]["node_count"] == 10
    assert checks["summary_matches_fixture"]["differences"] == [
        {"path": "by_node_type.region", "expected": None, "actual": 1},
        {"path": "node_count", "expected": 9, "actual": 10},
    ]


def test_scene_fixture_registry_lists_metadata_and_returns_fresh_graphs() -> None:
    assert list_scene_fixtures() == (
        "ambiguous_mugs",
        "ambiguous_plates",
        "moved_mug",
        "multi_room_rearrangement",
        "needs_reobserve",
        "relation_shift",
        "tabletop",
    )

    fixture = get_scene_fixture("tabletop")
    first = load_scene_fixture("tabletop")
    second = load_scene_fixture("tabletop")

    assert fixture.name == "tabletop"
    assert fixture.tags == ("static", "tabletop")
    assert fixture.description == "Static tabletop scene with mug, plate, table, room, and agent."
    assert first is not second
    assert first.get_object_state("mug_1").pose.to_dict() == {
        "x": -0.4,
        "y": 1.0,
        "z": 0.78,
        "yaw": 0.0,
    }


def test_scene_fixture_metadata_manifest_is_deterministic_and_filterable() -> None:
    from dsg_spatialqa_lab import list_scene_fixture_metadata

    manifest = list_scene_fixture_metadata()
    reobserve_manifest = list_scene_fixture_metadata(tags=("reobserve",))
    static_tabletop_manifest = list_scene_fixture_metadata(tags=("static", "tabletop"))

    assert manifest == (
        {
            "name": "ambiguous_mugs",
            "description": "Static tabletop scene with two visible mugs sharing one label.",
            "tags": ["static", "tabletop", "ambiguity"],
        },
        {
            "name": "ambiguous_plates",
            "description": "Static tabletop scene with two visible plates sharing one label.",
            "tags": ["static", "tabletop", "ambiguity"],
        },
        {
            "name": "moved_mug",
            "description": "Dynamic tabletop scene where mug_1 moves from table_1 to sink_region.",
            "tags": ["dynamic", "tabletop", "move"],
        },
        {
            "name": "multi_room_rearrangement",
            "description": (
                "Dynamic kitchen-to-pantry scene with relocated cereal and an occluded fork."
            ),
            "tags": ["dynamic", "multi_room", "move", "occlusion", "reobserve"],
        },
        {
            "name": "needs_reobserve",
            "description": (
                "Tabletop scene with invisible and low-confidence objects for re-observation checks."
            ),
            "tags": ["static", "tabletop", "reobserve"],
        },
        {
            "name": "relation_shift",
            "description": "Dynamic tabletop scene where mug_1 moves from left of plate_1 to right of it.",
            "tags": ["dynamic", "tabletop", "relations", "move"],
        },
        {
            "name": "tabletop",
            "description": "Static tabletop scene with mug, plate, table, room, and agent.",
            "tags": ["static", "tabletop"],
        },
    )
    assert reobserve_manifest == (manifest[3], manifest[4])
    assert [item["name"] for item in static_tabletop_manifest] == [
        "ambiguous_mugs",
        "ambiguous_plates",
        "needs_reobserve",
        "tabletop",
    ]


def test_scene_fixture_metadata_manifest_returns_tag_copies() -> None:
    from dsg_spatialqa_lab import list_scene_fixture_metadata

    metadata = [
        item
        for item in list_scene_fixture_metadata(tags=("reobserve",))
        if item["name"] == "needs_reobserve"
    ][0]

    metadata["tags"].append("mutated")

    repeated_metadata = [
        item
        for item in list_scene_fixture_metadata(tags=("reobserve",))
        if item["name"] == "needs_reobserve"
    ][0]

    assert repeated_metadata["tags"] == [
        "static",
        "tabletop",
        "reobserve",
    ]


def test_scene_fixture_manifest_includes_schema_filters_and_digest() -> None:
    assert hasattr(lab, "scene_fixture_manifest")

    payload_without_digest = {
        "schema_version": "dsg-spatialqa-lab.scene-fixture-manifest.v1",
        "filters": {
            "tags": ["reobserve"],
        },
        "fixture_count": 2,
        "scene_fixtures": [
            {
                "name": "multi_room_rearrangement",
                "description": (
                    "Dynamic kitchen-to-pantry scene with relocated cereal and an occluded fork."
                ),
                "tags": ["dynamic", "multi_room", "move", "occlusion", "reobserve"],
            },
            {
                "name": "needs_reobserve",
                "description": (
                    "Tabletop scene with invisible and low-confidence objects for re-observation checks."
                ),
                "tags": ["static", "tabletop", "reobserve"],
            },
        ],
    }
    expected_digest = hashlib.sha256(
        json.dumps(
            payload_without_digest,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    assert lab.scene_fixture_manifest(tags=("reobserve",)) == {
        **payload_without_digest,
        "digest": expected_digest,
    }


def test_scene_fixture_manifest_json_is_stable_and_savable(tmp_path: Path) -> None:
    assert hasattr(lab, "scene_fixture_manifest_json")
    assert hasattr(lab, "save_scene_fixture_manifest")
    manifest = lab.scene_fixture_manifest(tags=("multi_room",))

    payload = lab.scene_fixture_manifest_json(manifest)
    repeated_payload = lab.scene_fixture_manifest_json(manifest)
    manifest_path = tmp_path / "manifests" / "multi-room-fixtures.json"
    saved_path = lab.save_scene_fixture_manifest(manifest_path, tags=("multi_room",))

    assert payload == repeated_payload
    assert payload.endswith("\n")
    assert json.loads(payload) == manifest
    assert saved_path == manifest_path
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest


def test_scene_fixture_manifest_loads_from_explicit_file_and_validates(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "load_scene_fixture_manifest")
    assert hasattr(lab, "validate_scene_fixture_manifest")
    manifest = lab.scene_fixture_manifest(tags=("multi_room",))
    manifest_path = tmp_path / "multi-room-fixtures.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    loaded_manifest = lab.load_scene_fixture_manifest(manifest_path)
    validation = lab.validate_scene_fixture_manifest(loaded_manifest)

    assert loaded_manifest == manifest
    assert validation == {
        "valid": True,
        "schema_version": "dsg-spatialqa-lab.scene-fixture-manifest.v1",
        "digest": manifest["digest"],
        "checks": [
            {
                "name": "schema_version",
                "passed": True,
                "expected": "dsg-spatialqa-lab.scene-fixture-manifest.v1",
                "actual": "dsg-spatialqa-lab.scene-fixture-manifest.v1",
            },
            {
                "name": "manifest_digest",
                "passed": True,
                "expected": manifest["digest"],
                "actual": manifest["digest"],
            },
            {
                "name": "fixture_count_matches_manifest",
                "passed": True,
                "expected": 1,
                "actual": 1,
            },
        ],
    }


def test_scene_fixture_manifest_validation_reports_tampered_digest() -> None:
    assert hasattr(lab, "validate_scene_fixture_manifest")
    manifest = lab.scene_fixture_manifest(tags=("multi_room",))
    tampered_manifest = json.loads(json.dumps(manifest))
    tampered_manifest["digest"] = "0" * 64

    validation = lab.validate_scene_fixture_manifest(tampered_manifest)

    assert validation["valid"] is False
    assert validation["digest"] == "0" * 64
    assert validation["checks"][1] == {
        "name": "manifest_digest",
        "passed": False,
        "expected": manifest["digest"],
        "actual": "0" * 64,
    }


def test_scene_fixture_manifest_compare_matches_current_metadata() -> None:
    assert hasattr(lab, "compare_scene_fixture_manifest")
    manifest = lab.scene_fixture_manifest(tags=("multi_room",))

    comparison = lab.compare_scene_fixture_manifest(manifest)

    assert comparison == {
        "matches": True,
        "filters": {
            "tags": ["multi_room"],
        },
        "saved_digest": manifest["digest"],
        "current_digest": manifest["digest"],
        "checks": [
            {"name": "saved_manifest_valid", "passed": True},
            {
                "name": "manifest_digest_matches_current",
                "passed": True,
                "expected": manifest["digest"],
                "actual": manifest["digest"],
            },
            {
                "name": "fixture_manifest_matches_current",
                "passed": True,
                "expected": ["multi_room_rearrangement"],
                "actual": ["multi_room_rearrangement"],
            },
        ],
    }


def test_scene_fixture_manifest_compare_reports_metadata_drift() -> None:
    assert hasattr(lab, "compare_scene_fixture_manifest")
    manifest = lab.scene_fixture_manifest(tags=("multi_room",))
    drifted_manifest = json.loads(json.dumps(manifest))
    drifted_manifest["scene_fixtures"][0]["tags"] = [
        *manifest["scene_fixtures"][0]["tags"],
        "tampered",
    ]

    comparison = lab.compare_scene_fixture_manifest(drifted_manifest)

    fixture_check = next(
        check
        for check in comparison["checks"]
        if check["name"] == "fixture_manifest_matches_current"
    )
    assert comparison["matches"] is False
    assert comparison["filters"] == {"tags": ["multi_room"]}
    assert comparison["saved_digest"] == manifest["digest"]
    assert comparison["current_digest"] == manifest["digest"]
    assert comparison["checks"][0] == {"name": "saved_manifest_valid", "passed": False}
    assert fixture_check["passed"] is False
    assert fixture_check["differences"] == [
        {
            "path": "multi_room_rearrangement.tags",
            "expected": [
                "dynamic",
                "multi_room",
                "move",
                "occlusion",
                "reobserve",
                "tampered",
            ],
            "actual": ["dynamic", "multi_room", "move", "occlusion", "reobserve"],
        }
    ]


def test_needs_reobserve_fixture_supports_qa_regression() -> None:
    from dsg_spatialqa_lab import build_needs_reobserve_scene

    fixture = get_scene_fixture("needs_reobserve")
    graph = build_needs_reobserve_scene()
    loaded = load_scene_fixture("needs_reobserve")
    qa = SpatialQAEngine(GraphTool(graph, reobserve_confidence_threshold=0.5))

    response = qa.answer({"type": "reobserve_targets"})

    assert fixture.name == "needs_reobserve"
    assert fixture.tags == ("static", "tabletop", "reobserve")
    assert fixture.description == (
        "Tabletop scene with invisible and low-confidence objects for re-observation checks."
    )
    assert [state.object_id for state in GraphTool(graph).reobserve_targets()] == ["spoon_1"]
    assert graph.get_object_state("cup_1").visible is True
    assert graph.get_object_state("cup_1").confidence == 0.2
    assert loaded is not graph
    assert response.error is None
    assert response.answer["count"] == 1
    assert response.answer["objects"][0]["object_id"] == "spoon_1"
    assert response.answer["objects"][0]["confidence"] == 0.25
    assert response.evidence_edges == ["spoon_1-STATE_CHANGED-state:spoon_1:2-2"]


def test_ambiguous_mugs_fixture_supports_vla_label_ambiguity_regression() -> None:
    from dsg_spatialqa_lab import build_ambiguous_mugs_scene

    fixture = get_scene_fixture("ambiguous_mugs")
    graph = build_ambiguous_mugs_scene()
    loaded = load_scene_fixture("ambiguous_mugs")
    planner = VLAAnchorPlanner(GraphTool(graph))

    result = planner.plan_pick(label="mug")

    assert fixture.name == "ambiguous_mugs"
    assert fixture.tags == ("static", "tabletop", "ambiguity")
    assert fixture.description == "Static tabletop scene with two visible mugs sharing one label."
    assert [state.object_id for state in GraphTool(graph).find_objects(label="mug")] == [
        "mug_1",
        "mug_2",
    ]
    assert loaded is not graph
    assert result.status == "ambiguous"
    assert result.command is None
    assert result.ambiguous_ids == ["mug_1", "mug_2"]


def test_ambiguous_plates_fixture_supports_vla_reference_label_ambiguity_regression() -> None:
    from dsg_spatialqa_lab import build_ambiguous_plates_scene

    fixture = get_scene_fixture("ambiguous_plates")
    graph = build_ambiguous_plates_scene()
    loaded = load_scene_fixture("ambiguous_plates")
    planner = VLAAnchorPlanner(GraphTool(graph))

    result = planner.plan_place_relative(
        "mug_1",
        None,
        "RIGHT_OF",
        reference_label="plate",
    )

    assert fixture.name == "ambiguous_plates"
    assert fixture.tags == ("static", "tabletop", "ambiguity")
    assert fixture.description == "Static tabletop scene with two visible plates sharing one label."
    assert [state.object_id for state in GraphTool(graph).find_objects(label="plate")] == [
        "plate_1",
        "plate_2",
    ]
    assert loaded is not graph
    assert result.status == "ambiguous"
    assert result.command is None
    assert result.ambiguous_ids == ["plate_1", "plate_2"]


def test_moved_mug_fixture_supports_qa_and_vla_regression() -> None:
    graph = load_scene_fixture("moved_mug")
    qa = SpatialQAEngine(GraphTool(graph))
    planner = VLAAnchorPlanner(GraphTool(build_tabletop_scene()))
    pick = planner.plan_pick(target_object="mug_1")
    assert pick.command is not None

    moved_planner = VLAAnchorPlanner(GraphTool(graph))
    validation = moved_planner.validate(pick.command)
    recent_events = qa.answer({"type": "recent_events", "since_step": 2, "until_step": 2})
    world_state = qa.answer({"type": "world_state", "visible": True})

    assert graph.get_object_state("mug_1").pose.to_dict() == {
        "x": 1.2,
        "y": 0.2,
        "z": 0.5,
        "yaw": 0.0,
    }
    assert validation.status == "needs_replan"
    assert validation.details["current_location"] == {
        "relation": "IN_REGION",
        "dst": "sink_region",
        "step": 2,
    }
    assert recent_events.answer["events"] == [
        {"id": "action_move_mug", "type": "action", "label": "move", "step": 2},
        {"id": "event_move_mug", "type": "event", "label": "move_object", "step": 2},
    ]
    assert world_state.answer["objects"][0]["current_location"] == {
        "relation": "IN_REGION",
        "dst": "sink_region",
        "step": 2,
    }


def test_relation_shift_fixture_supports_dynamic_relation_regression() -> None:
    fixture = get_scene_fixture("relation_shift")
    graph = build_relation_shift_scene()
    loaded = load_scene_fixture("relation_shift")
    qa = SpatialQAEngine(GraphTool(graph))

    response = qa.answer(
        {
            "type": "relation_timeline",
            "src": "mug_1",
            "dst": "plate_1",
            "reference_frame": "agent",
        }
    )

    assert fixture.tags == ("dynamic", "tabletop", "relations", "move")
    assert fixture.description == (
        "Dynamic tabletop scene where mug_1 moves from left of plate_1 to right of it."
    )
    assert loaded is not graph
    assert graph.get_object_state("mug_1").pose.to_dict() == {
        "x": 0.64,
        "y": 1.0,
        "z": 0.78,
        "yaw": 0.0,
    }
    assert response.error is None
    assert [(entry["relation"], entry["step"]) for entry in response.answer["timeline"]] == [
        ("LEFT_OF", 1),
        ("NEAR", 1),
        ("NEAR", 2),
        ("RIGHT_OF", 2),
    ]
    assert response.evidence_edges == [
        "mug_1-LEFT_OF-plate_1-1",
        "mug_1-NEAR-plate_1-1",
        "mug_1-NEAR-plate_1-2",
        "mug_1-RIGHT_OF-plate_1-2",
    ]
    assert response.needs_reobserve is False


def test_multi_room_rearrangement_fixture_supports_temporal_reobserve_regressions() -> None:
    fixture = get_scene_fixture("multi_room_rearrangement")
    graph = build_multi_room_rearrangement_scene()
    loaded = load_scene_fixture("multi_room_rearrangement")
    qa = SpatialQAEngine(GraphTool(graph))

    delta = qa.answer({"type": "scene_delta", "from_step": 1, "to_step": 2})
    reobserve = qa.answer({"type": "reobserve_targets", "label": "fork"})
    recent_events = qa.answer({"type": "recent_events", "since_step": 2, "until_step": 3})

    assert fixture.tags == ("dynamic", "multi_room", "move", "occlusion", "reobserve")
    assert graph.nodes["kitchen"].label == "Kitchen"
    assert graph.nodes["pantry"].label == "Pantry"
    assert loaded is not graph
    assert graph.get_object_state("cereal_box_1").pose.to_dict() == {
        "x": 3.2,
        "y": 0.4,
        "z": 1.2,
        "yaw": 0.0,
    }
    assert delta.answer["agent"]["changed"] is True
    assert delta.answer["objects"] == [
        {
            "object_id": "cereal_box_1",
            "label": "cereal_box",
            "changes": ["pose", "last_seen_step", "location"],
            "from_pose": {"x": 0.2, "y": 1.0, "z": 0.9, "yaw": 0.0},
            "to_pose": {"x": 3.2, "y": 0.4, "z": 1.2, "yaw": 0.0},
            "from_visible": True,
            "to_visible": True,
            "from_confidence": 0.92,
            "to_confidence": 0.92,
            "from_last_seen_step": 1,
            "to_last_seen_step": 2,
            "from_location": {"relation": "IN_REGION", "dst": "prep_counter", "step": 1},
            "to_location": {"relation": "IN_REGION", "dst": "pantry_shelf", "step": 2},
            "from_state_step": 1,
            "to_state_step": 2,
        }
    ]
    assert reobserve.answer["count"] == 1
    assert reobserve.answer["objects"][0]["object_id"] == "fork_1"
    assert reobserve.answer["objects"][0]["last_seen_step"] == 1
    assert recent_events.answer["events"] == [
        {"id": "action_move_cereal_box", "type": "action", "label": "move", "step": 2},
        {
            "id": "event_move_cereal_box",
            "type": "event",
            "label": "move_object",
            "step": 2,
        },
    ]


def test_unknown_scene_fixture_returns_clear_error() -> None:
    with pytest.raises(SpatialQAError, match="Unknown scene fixture: missing"):
        load_scene_fixture("missing")
