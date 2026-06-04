from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dsg_spatialqa_lab as lab


def test_research_conclusion_rejects_tiny_current_like_lift(tmp_path: Path) -> None:
    bundle = _write_conclusion_bundle(
        tmp_path,
        candidate_correct={0},
        baseline_correct_by_kind={
            "vlm": set(),
            "multi_frame_vlm": set(),
            "caption_memory": set(),
            "graph_text": set(),
        },
        object_recall=0.153409,
    )

    report = lab.research_conclusion_report(**bundle)

    assert report["conclusion"]["verdict"] == "dsg_not_superior"
    assert report["conclusion"]["dsg_superiority_claim_allowed"] is False
    assert report["conclusion"]["ready_to_claim_not_superior"] is True
    assert report["candidate_summary"] == {
        "case_count": 60,
        "exact_match_count": 1,
        "exact_match_rate": 0.016667,
    }
    assert report["aggregate_comparison"] == {
        "control_count": 4,
        "passed_control_count": 0,
        "failed_control_count": 4,
        "minimum_exact_match_rate_delta": 0.016667,
        "minimum_sign_test_p_value": 0.5,
    }
    assert {reason["code"] for reason in report["conclusion"]["reasons"]} == {
        "candidate_exact_match_rate_below_floor",
        "graph_object_recall_below_floor",
        "no_control_passed_superiority",
    }
    assert lab.validate_research_conclusion_report(report)["valid"] is True


def test_research_conclusion_accepts_strong_paired_lift(tmp_path: Path) -> None:
    bundle = _write_conclusion_bundle(
        tmp_path,
        candidate_correct=set(range(45)),
        baseline_correct_by_kind={
            "vlm": set(range(20)),
            "multi_frame_vlm": set(range(20)),
            "caption_memory": set(range(20)),
            "graph_text": set(range(20)),
        },
        object_recall=0.75,
    )

    report = lab.research_conclusion_report(**bundle)

    assert report["conclusion"]["verdict"] == "dsg_superior"
    assert report["conclusion"]["dsg_superiority_claim_allowed"] is True
    assert report["aggregate_comparison"]["passed_control_count"] == 4
    assert all(row["decision"] == "candidate_superior" for row in report["control_comparisons"])
    assert lab.validate_research_conclusion_report(report)["valid"] is True


def test_research_conclusion_blocks_superiority_when_real_readiness_is_false(
    tmp_path: Path,
) -> None:
    bundle = _write_conclusion_bundle(
        tmp_path,
        candidate_correct=set(range(45)),
        baseline_correct_by_kind={
            "vlm": set(range(20)),
            "multi_frame_vlm": set(range(20)),
            "caption_memory": set(range(20)),
            "graph_text": set(range(20)),
        },
        object_recall=0.75,
        real_ready=False,
    )

    report = lab.research_conclusion_report(**bundle)

    assert report["conclusion"]["verdict"] == "inconclusive_not_ready"
    assert report["conclusion"]["dsg_superiority_claim_allowed"] is False
    assert report["conclusion"]["ready_to_claim_not_superior"] is False
    assert {reason["code"] for reason in report["conclusion"]["reasons"]} == {
        "real_experiment_readiness_not_ready"
    }


def test_research_conclusion_digest_validation_detects_tampering(tmp_path: Path) -> None:
    bundle = _write_conclusion_bundle(
        tmp_path,
        candidate_correct=set(range(45)),
        baseline_correct_by_kind={
            "vlm": set(range(20)),
            "multi_frame_vlm": set(range(20)),
            "caption_memory": set(range(20)),
            "graph_text": set(range(20)),
        },
        object_recall=0.75,
    )
    report = lab.research_conclusion_report(**bundle)
    saved_path = lab.save_research_conclusion_report(report, tmp_path / "conclusion.json")

    loaded = lab.load_research_conclusion_report(saved_path)
    assert lab.compare_research_conclusion_report(loaded)["matches"] is True

    loaded["conclusion"]["verdict"] = "dsg_not_superior"
    assert lab.validate_research_conclusion_report(loaded)["valid"] is False


def _write_conclusion_bundle(
    tmp_path: Path,
    *,
    candidate_correct: set[int],
    baseline_correct_by_kind: dict[str, set[int]],
    object_recall: float,
    real_ready: bool = True,
) -> dict[str, Any]:
    cases = _qa_cases(60)
    candidate_report = lab.qa_eval_report(
        cases,
        _predictions(cases, candidate_correct),
        prediction_path=tmp_path / "candidate.jsonl",
    )
    candidate_path = tmp_path / "candidate-qa-eval.json"
    lab.save_qa_eval_report(candidate_report, candidate_path)
    rows: list[dict[str, object]] = []
    for source_kind, correct in sorted(baseline_correct_by_kind.items()):
        baseline_report = lab.qa_eval_report(
            cases,
            _predictions(cases, correct),
            prediction_path=tmp_path / f"{source_kind}.jsonl",
        )
        baseline_path = tmp_path / f"{source_kind}-qa-eval.json"
        delta_path = tmp_path / f"candidate-vs-{source_kind}.json"
        lab.save_qa_eval_report(baseline_report, baseline_path)
        delta = lab.qa_eval_delta_report(
            candidate_report,
            baseline_report,
            candidate_name="predicted_graph_tool",
            baseline_name=source_kind,
            candidate_report_path=candidate_path,
            baseline_report_path=baseline_path,
        )
        lab.save_qa_eval_delta_report(delta, delta_path)
        rows.append(
            {
                "source_key": f"{source_kind}_{source_kind}_control",
                "source_kind": source_kind,
                "source_name": f"{source_kind}_control",
                "qa_eval_delta_report_path": str(delta_path),
            }
        )
    return {
        "real_readiness_report": {
            "readiness": {"ready": real_ready},
            "report_digest": "real-readiness-digest",
        },
        "offline_control_result_report": {
            "readiness": {"ready": True},
            "report_digest": "offline-result-digest",
            "source_result_matrix": rows,
        },
        "predicted_dsg_evidence_report": {
            "readiness": {"ready": True},
            "report_digest": "predicted-evidence-digest",
        },
        "graph_eval_report": {
            "metrics": {
                "object_recall": {"rate": object_recall},
                "relation_f1": {"rate": 0.2},
            },
            "report_digest": "graph-eval-digest",
        },
        "error_attribution_report": {
            "summary": {
                "case_count": 60,
                "answer_correct_count": len(candidate_correct),
                "by_error_category": {
                    "correct": len(candidate_correct),
                    "evidence_missing": 60 - len(candidate_correct),
                },
            },
            "report_digest": "error-attribution-digest",
        },
    }


def _qa_cases(count: int) -> tuple[lab.QACase, ...]:
    return tuple(
        lab.QACase(
            id=f"case-{index:03d}",
            scene_id="scene",
            episode_id="episode",
            graph_digest="graph-digest",
            step=index,
            question={"type": "object_location", "object_id": f"object-{index:03d}"},
            question_type="object_location",
            answer={"object_id": f"object-{index:03d}", "location": "counter"},
            answer_type="object_location",
            required_nodes=(f"object-{index:03d}",),
            required_edges=(f"edge-{index:03d}",),
            tags=("qa", "spatial_qa", "graph_query"),
        )
        for index in range(count)
    )


def _predictions(
    cases: tuple[lab.QACase, ...],
    correct_indexes: set[int],
) -> tuple[lab.QAPrediction, ...]:
    predictions: list[lab.QAPrediction] = []
    for index, case in enumerate(cases):
        answer = case.answer if index in correct_indexes else {"object_id": case.id}
        predictions.append(
            lab.QAPrediction(
                id=case.id,
                answer=json.loads(json.dumps(answer, sort_keys=True)),
                evidence_nodes=case.required_nodes,
                evidence_edges=case.required_edges,
                confidence=0.9,
            )
        )
    return tuple(predictions)
