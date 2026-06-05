from __future__ import annotations

import dsg_spatialqa_lab as lab


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
