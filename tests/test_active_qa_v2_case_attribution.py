from __future__ import annotations

from dsg_spatialqa_lab.eval.active_qa_v2_analysis import (
    active_qa_v2_case_attribution_report,
    adjudication_derived_fusion_predictions,
)


def test_active_qa_v2_case_attribution_counts_wins_failures_and_ties() -> None:
    records = [
        _record("case-support-win", "support_relation", "ON", "countertop"),
        _record("case-temporal-failure", "temporal_last_seen", "VISIBLE_FROM", "frame"),
        _record("case-situated-tie", "situated_egocentric", "VISIBLE_FROM", "frame"),
    ]
    vlm_predictions = {
        "case-support-win": _prediction("case-support-win", "ON", "table", confidence=0.4),
        "case-temporal-failure": _prediction(
            "case-temporal-failure", "VISIBLE_FROM", "wrong_frame", confidence=0.0
        ),
        "case-situated-tie": _prediction(
            "case-situated-tie", "VISIBLE_FROM", "frame", confidence=0.9
        ),
    }
    graph_predictions = {
        "case-support-win": _prediction("case-support-win", "ON", "countertop"),
        "case-temporal-failure": _prediction(
            "case-temporal-failure", "VISIBLE_FROM", "frame"
        ),
        "case-situated-tie": _prediction("case-situated-tie", "VISIBLE_FROM", "frame"),
    }
    trusted_predictions = {
        "case-support-win": _prediction("case-support-win", "ON", "table"),
        "case-temporal-failure": _prediction(
            "case-temporal-failure", "VISIBLE_FROM", "wrong_frame"
        ),
        "case-situated-tie": _prediction("case-situated-tie", "VISIBLE_FROM", "frame"),
    }
    adjudicated_predictions = {
        "case-support-win": _adjudicated("case-support-win", "accept_dsg", "ON", "countertop"),
        "case-temporal-failure": _adjudicated(
            "case-temporal-failure", "reject_both", "VISIBLE_FROM", "wrong_frame"
        ),
        "case-situated-tie": _adjudicated(
            "case-situated-tie", "accept_vlm", "VISIBLE_FROM", "frame"
        ),
    }

    report = active_qa_v2_case_attribution_report(
        records,
        vlm_predictions,
        graph_predictions,
        trusted_predictions,
        adjudicated_predictions,
    )

    assert report["summary"]["adjudicated_win_count"] == 1
    assert report["summary"]["adjudicated_failure_count"] == 1
    assert report["summary"]["tie_correct_count"] == 1
    assert report["summary"]["adjudicated_loss_count"] == 0
    rows = {row["case_id"]: row for row in report["cases"]}
    assert rows["case-support-win"]["outcome"] == "adjudicated_win"
    assert rows["case-support-win"]["primary_attribution"] == "dsg_support_relation_correction"
    assert rows["case-temporal-failure"]["primary_attribution"] == "adjudicator_rejected_both"
    assert rows["case-situated-tie"]["outcome"] == "tie_correct"
    assert len(report["report_digest"]) == 64


def test_adjudication_derived_fusion_marks_calibrated_graph_overrides() -> None:
    records = [
        _record("case-object-room", "object_location", "IN_ROOM", "FloorPlan1"),
        _record("case-temporal-unknown", "temporal_last_seen", "VISIBLE_FROM", "frame_3"),
        _record("case-situated-high-conf", "situated_egocentric", "VISIBLE_FROM", "frame_1"),
    ]
    vlm_predictions = {
        "case-object-room": _prediction("case-object-room", "IN_ROOM", "wrong_room", confidence=0.9),
        "case-temporal-unknown": {
            "id": "case-temporal-unknown",
            "answer": {
                "answer_text": "unknown",
                "current_location": {"relation": "UNKNOWN", "dst_label": None},
            },
            "confidence": 0.0,
        },
        "case-situated-high-conf": _prediction(
            "case-situated-high-conf", "VISIBLE_FROM", "frame_1", confidence=0.95
        ),
    }
    graph_predictions = {
        "case-object-room": _prediction("case-object-room", "IN_ROOM", "FloorPlan1"),
        "case-temporal-unknown": _prediction(
            "case-temporal-unknown", "VISIBLE_FROM", "frame_3"
        ),
        "case-situated-high-conf": _prediction(
            "case-situated-high-conf", "VISIBLE_FROM", "frame_2"
        ),
    }

    predictions, report = adjudication_derived_fusion_predictions(
        records,
        vlm_predictions,
        graph_predictions,
    )

    by_id = {row["id"]: row for row in predictions}
    assert by_id["case-object-room"]["answer"]["fusion"]["fusion_source"] == "graph_tool"
    assert by_id["case-object-room"]["answer"]["fusion"]["selection_reason"] == (
        "relation_or_object_location_graph_prior"
    )
    assert by_id["case-temporal-unknown"]["answer"]["fusion"]["fusion_source"] == "graph_tool"
    assert by_id["case-temporal-unknown"]["answer"]["fusion"]["selection_reason"] == (
        "vlm_unknown_or_low_confidence"
    )
    assert by_id["case-situated-high-conf"]["answer"]["fusion"]["fusion_source"] == "vlm"
    assert report["not_final_research_claim"] is True
    assert report["summary"]["graph_source_count"] == 2
    assert report["summary"]["vlm_source_count"] == 1


def _record(case_id: str, question_type: str, relation: str, dst_label: str) -> dict[str, object]:
    return {
        "id": case_id,
        "episode_id": "episode",
        "question_type": question_type,
        "answer": {
            "current_location": {
                "dst": dst_label,
                "dst_label": dst_label,
                "relation": relation,
            }
        },
    }


def _prediction(
    case_id: str,
    relation: str,
    dst_label: str,
    *,
    confidence: float = 1.0,
) -> dict[str, object]:
    return {
        "id": case_id,
        "answer": {
            "current_location": {
                "dst": dst_label,
                "relation": relation,
                "dst_label": dst_label,
            }
        },
        "confidence": confidence,
    }


def _adjudicated(
    case_id: str,
    decision: str,
    relation: str,
    dst_label: str,
) -> dict[str, object]:
    selected = "graph_tool_dsg" if decision == "accept_dsg" else "vlm" if decision == "accept_vlm" else "none"
    return {
        "id": case_id,
        "answer": {
            "current_location": {
                "dst": dst_label,
                "dst_label": dst_label,
                "relation": relation,
            },
            "decision": decision,
            "selected_candidate": selected,
            "evidence_summary": "structured evidence",
        },
        "confidence": 0.8,
    }
