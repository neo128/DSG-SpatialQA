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


def test_graph_summary_reports_visibility_and_reobserve_counts() -> None:
    summary = graph_summary(load_scene_fixture("needs_reobserve"))

    assert summary["object_count"] == 5
    assert summary["visible_object_count"] == 3
    assert summary["hidden_object_count"] == 2
    assert summary["low_confidence_object_count"] == 1
    assert summary["reobserve_candidate_count"] == 1
    assert summary["by_object_label"] == {
        "bowl": 1,
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
                "Tabletop scene with one invisible low-confidence spoon requiring re-observation."
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
                    "Tabletop scene with one invisible low-confidence spoon requiring re-observation."
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
        "Tabletop scene with one invisible low-confidence spoon requiring re-observation."
    )
    assert [state.object_id for state in GraphTool(graph).reobserve_targets()] == ["spoon_1"]
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
