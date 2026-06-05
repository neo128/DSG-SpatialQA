from __future__ import annotations

import dsg_spatialqa_lab as lab


def _detector_attributes() -> dict[str, object]:
    return {
        "evidence_kinds": ["rgb", "depth", "detector"],
        "scene_id": "FloorPlan1",
        "source_kind": "detector",
        "visible": True,
    }


def _qa_case(case_id: str, object_id: str) -> lab.QACase:
    return lab.QACase(
        id=case_id,
        scene_id="FloorPlan1",
        episode_id="episode-1",
        graph_digest="graph-digest",
        step=5,
        question={"type": "object_location", "object_id": object_id},
        question_type="object_location",
        answer={},
        answer_type="object_location",
    )


def test_object_location_query_diagnostic_report_splits_missing_and_ambiguous_support() -> None:
    assert hasattr(lab, "object_location_query_diagnostic_report")
    assert hasattr(lab, "object_location_query_diagnostic_report_digest")
    graph = lab.DynamicSceneGraph()
    attributes = _detector_attributes()
    graph.upsert_object(
        "chair_1",
        "chair",
        lab.Pose3D(0.0, 0.9, 0.0),
        lab.BBox3D(center=lab.Pose3D(0.0, 0.9, 0.0), size=(0.7, 0.6, 0.7)),
        confidence=0.9,
        visible=True,
        step=5,
        attributes=attributes,
    )
    graph.upsert_object(
        "diningtable_1",
        "diningtable",
        lab.Pose3D(0.2, 0.9, 0.0),
        lab.BBox3D(center=lab.Pose3D(0.2, 0.9, 0.0), size=(1.2, 0.8, 1.2)),
        confidence=0.9,
        visible=True,
        step=5,
        attributes=attributes,
    )
    graph.upsert_object(
        "book_1",
        "book",
        lab.Pose3D(0.1, 0.9, 0.0),
        lab.BBox3D(center=lab.Pose3D(0.1, 0.9, 0.0), size=(0.2, 0.1, 0.2)),
        confidence=0.86,
        visible=True,
        step=5,
        attributes=attributes,
    )
    graph.upsert_object(
        "apple_1",
        "apple",
        lab.Pose3D(3.0, 0.9, 0.0),
        lab.BBox3D(center=lab.Pose3D(3.0, 0.9, 0.0), size=(0.1, 0.1, 0.1)),
        confidence=0.84,
        visible=True,
        step=5,
        attributes=attributes,
    )
    cases = [
        _qa_case("case-ambiguous", "book_1"),
        _qa_case("case-missing-support", "apple_1"),
        _qa_case("case-missing-target", "missing_1"),
    ]
    semantic_report = {
        "cases": [
            {
                "case_id": "case-ambiguous",
                "failure_reason": "relation_mismatch",
                "semantic_match": False,
            },
            {
                "case_id": "case-missing-support",
                "failure_reason": "relation_mismatch",
                "semantic_match": False,
            },
            {
                "case_id": "case-missing-target",
                "failure_reason": "Object not found: missing_1",
                "semantic_match": False,
            },
        ]
    }

    report = lab.object_location_query_diagnostic_report(
        graph,
        cases,
        semantic_eval_report=semantic_report,
        graph_path="predicted-graph.json",
        qa_path="qa.jsonl",
        semantic_eval_path="semantic-eval.json",
    )

    assert report["schema_version"] == lab.OBJECT_LOCATION_QUERY_DIAGNOSTIC_REPORT_SCHEMA_VERSION
    assert report["summary"] == {
        "object_location_case_count": 3,
        "query_error_count": 1,
        "room_fallback_count": 2,
        "semantic_mismatch_count": 3,
        "semantic_mismatch_status_counts": {
            "query_error": 1,
            "support_fallback_ambiguous": 1,
            "support_fallback_missing": 1,
        },
        "semantic_match_status_counts": {},
        "status_counts": {
            "query_error": 1,
            "support_fallback_ambiguous": 1,
            "support_fallback_missing": 1,
        },
        "support_candidate_count_histogram": {"0": 1, "2": 1},
    }
    assert report["cases"] == [
        {
            "case_id": "case-ambiguous",
            "current_location": {"dst": "ai2thor_room", "relation": "IN_ROOM", "step": 5},
            "failure_reason": "relation_mismatch",
            "location_evidence_status": "support_fallback_ambiguous",
            "missing_evidence": ["unambiguous_detector_support"],
            "object_id": "book_1",
            "prediction_error": None,
            "question_type": "object_location",
            "room_fallback_applied": True,
            "semantic_match": False,
            "support_candidate_count": 2,
            "support_fallback_applied": False,
        },
        {
            "case_id": "case-missing-support",
            "current_location": {"dst": "ai2thor_room", "relation": "IN_ROOM", "step": 5},
            "failure_reason": "relation_mismatch",
            "location_evidence_status": "support_fallback_missing",
            "missing_evidence": ["detector_support_candidate"],
            "object_id": "apple_1",
            "prediction_error": None,
            "question_type": "object_location",
            "room_fallback_applied": True,
            "semantic_match": False,
            "support_candidate_count": 0,
            "support_fallback_applied": False,
        },
        {
            "case_id": "case-missing-target",
            "current_location": None,
            "failure_reason": "Object not found: missing_1",
            "location_evidence_status": "query_error",
            "missing_evidence": ["target_object"],
            "object_id": "missing_1",
            "prediction_error": "Object not found: missing_1",
            "question_type": "object_location",
            "room_fallback_applied": False,
            "semantic_match": False,
            "support_candidate_count": 0,
            "support_fallback_applied": False,
        },
    ]
    assert report["artifacts"] == {
        "graph_path": "predicted-graph.json",
        "qa_path": "qa.jsonl",
        "semantic_eval_path": "semantic-eval.json",
    }
    assert report["report_digest"] == lab.object_location_query_diagnostic_report_digest(report)
